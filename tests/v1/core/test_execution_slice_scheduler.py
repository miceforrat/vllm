import pytest

from vllm.experimental.execution_slice import ExecutionSliceDescriptor
from vllm.v1.core.sched.execution_slice_scheduler import (
    ExecutionSliceScheduler,
)
from vllm.v1.engine.model_request_control import (
    ModelEmitter,
    RequestControlPlane,
    RequestOutcome,
)


def test_engine_slice_scheduler_selects_planned_variants():
    scheduler = ExecutionSliceScheduler()
    scheduler.configure_model(
        "reasoner",
        full_variant="reasoner_full",
        shared_variant="shared",
        primary=True,
    )
    scheduler.configure_model(
        "generator",
        full_variant="full",
        shared_variant="shared",
    )

    alone = scheduler.schedule_slice(
        "generator", "g-1", ExecutionSliceDescriptor(0)
    )
    assert alone.resource_variant == "full"

    scheduler.set_model_demand("reasoner", True)
    scheduler.set_model_demand("generator", True)
    colocated = scheduler.schedule_slice(
        "generator", "g-1", ExecutionSliceDescriptor(1)
    )
    reasoner = scheduler.schedule_model_batch("reasoner", ("r-1",))
    assert colocated.resource_variant == "shared"
    assert reasoner.resource_variant == "shared"
    assert (
        alone.execution_epoch
        < colocated.execution_epoch
        < reasoner.execution_epoch
    )

    with pytest.raises(RuntimeError, match="I5 violation"):
        scheduler.schedule_slice(
            "generator",
            "g-2",
            ExecutionSliceDescriptor(
                2, {"requested_variant": "full"}
            ),
        )
    scheduler.set_model_demand("reasoner", False)
    expanded = scheduler.schedule_slice(
        "generator", "g-1", ExecutionSliceDescriptor(3)
    )
    assert expanded.resource_variant == "full"
    assert [
        transition["kind"]
        for transition in scheduler.snapshot()["variant_transitions"]
        if transition["model_id"] == "generator"
    ] == ["initial", "shrink", "expand"]
    with pytest.raises(ValueError, match="Unknown resource variant"):
        scheduler.validate_requested_variant("generator", "generator_full")


def test_request_control_and_model_emitter_own_request_execution():
    demand: list[tuple[str, bool]] = []
    worker_calls: list[tuple[str, tuple]] = []

    slices = ExecutionSliceScheduler()
    slices.configure_model(
        "reasoner",
        full_variant="reasoner_full",
        shared_variant="shared",
        primary=True,
    )
    slices.configure_model(
        "generator", full_variant="generator_full", shared_variant="shared"
    )

    class _ExecutionSlices:
        def set_model_demand(self, model_id, active):
            demand.append((model_id, active))
            return slices.set_model_demand(model_id, active)

        def schedule_slice(self, model_id, request_id, spec):
            return slices.schedule_slice(model_id, request_id, spec)

        def validate_requested_variant(self, model_id, requested):
            return slices.validate_requested_variant(model_id, requested)

    def collective_rpc(method, timeout=None, args=(), kwargs=None):
        del timeout, kwargs
        worker_calls.append((method, args))
        if method == "start_model_request":
            return [
                {
                    "rank": 0,
                    "next_slice": {
                        "sequence_no": 0,
                        "tags": ["component:preparation"],
                        "metadata": {"phase": "preparing"},
                    },
                }
            ]
        if method == "execute_model_request_slice":
            return [{"rank": 0, "request_done": True}]
        raise AssertionError(method)

    control = RequestControlPlane(_ExecutionSlices())
    emitter = ModelEmitter("generator", collective_rpc, control)
    control.open("r-1", "reasoner", "vllm")
    control.open("g-1", "generator", "vllm_omni")
    future = control.prepare_submit("g-1", "generator", "auto")
    assert future is not None
    emitter.submit("g-1", {"prompt": "video"}, "auto", 4096)
    assert future.result() == [
        {"rank": 0, "request_done": True}
    ]
    assert demand == [
        ("reasoner", True),
        ("generator", True),
        ("generator", False),
    ]
    assert worker_calls == [
        (
            "start_model_request",
            ("generator", {"prompt": "video"}, "auto", 4096, "g-1"),
        ),
        (
            "execute_model_request_slice",
            (
                "g-1",
                {
                    "model_id": "generator",
                    "request_id": "g-1",
                    "descriptor": {
                        "sequence_no": 0,
                        "tags": ["component:preparation"],
                        "metadata": {
                            "phase": "preparing",
                            "requested_variant": "auto",
                        },
                    },
                    "resource_variant": "shared",
                    "execution_epoch": 1,
                },
            ),
        ),
    ]
    emitter.shutdown()
    control.shutdown()


def test_request_control_propagates_terminal_error():
    scheduler = ExecutionSliceScheduler()
    scheduler.configure_model(
        "generator", full_variant="full", shared_variant="shared"
    )
    control = RequestControlPlane(scheduler)
    control.open("g-1", "generator", "vllm_omni")
    future = control.prepare_submit("g-1", "generator", "full")
    assert future is not None
    control.report_workers([
        {
            "request_done": True,
            "finished_request_ids": ("g-1",),
            "metadata": {"terminal_requests": {"g-1": {
                "status": "FINISHED_ERROR",
                "error": "injected failure",
            }}},
        }
    ], "g-1")
    with pytest.raises(RuntimeError, match="injected failure"):
        future.result()
    control.shutdown()


def test_request_control_selects_variant_and_splits_wave_results():
    scheduler = ExecutionSliceScheduler()
    scheduler.configure_model(
        "generator", full_variant="full", shared_variant="shared"
    )
    control = RequestControlPlane(scheduler)
    control.open("A", "generator", "vllm_omni")
    control.open("B", "generator", "vllm_omni")
    future_a = control.prepare_submit("A", "generator", "full")
    future_b = control.prepare_submit("B", "generator", "auto")
    assert future_a is not None and future_b is not None

    first = control.plan_slice("generator", "A", "auto", {
        "sequence_no": 0,
        "tags": ["diffusion"],
        "metadata": {"request_ids": ("A", "B")},
    })
    assert first["descriptor"]["metadata"]["requested_variant"] == "full"
    control.report_workers([{
        "request_done": False,
        "finished_request_ids": ("A",),
        "finished_seconds": {"A": 1.25},
        "per_request_step_counts": {"A": 4, "B": 2},
        "finished_media": {"A": {"handle": "media://A", "shape": (1, 3)}},
        "metadata": {},
    }], "A")
    second = control.plan_slice("generator", "A", "auto", {
        "sequence_no": 1,
        "tags": ["diffusion"],
        "metadata": {"request_ids": ("B",)},
    })
    assert second["descriptor"]["metadata"]["requested_variant"] == "auto"
    control.report_workers([{
        "request_done": True,
        "finished_request_ids": ("B",),
        "finished_seconds": {"B": 2.5},
        "per_request_step_counts": {"B": 4},
        "metadata": {},
    }], "A")

    assert future_a.result()[0]["seconds"] == 1.25
    assert future_a.result()[0]["step_count"] == 4
    assert future_a.result()[0]["media"] == {
        "handle": "media://A",
        "shape": (1, 3),
    }
    assert future_b.result()[0]["seconds"] == 2.5
    assert future_b.result()[0]["step_count"] == 4
    control.shutdown()


def test_request_finish_is_idempotent_and_releases_demand_once():
    demand: list[tuple[str, bool]] = []

    class _Slices:
        def set_model_demand(self, model_id, active):
            demand.append((model_id, active))

    control = RequestControlPlane(_Slices())
    control.open("g-1", "generator", "vllm_omni")
    future = control.prepare_submit("g-1", "generator", "auto")
    assert future is not None
    assert control.finish("g-1", RequestOutcome.CANCELLED)
    assert not control.finish("g-1", RequestOutcome.FAILED)
    assert future.cancelled()
    assert demand == [("generator", True), ("generator", False)]
    control.close("g-1")
    control.shutdown()
