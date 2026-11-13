from types import SimpleNamespace

from vllm.experimental.execution_slice import ExecutionSliceDescriptor
from vllm.experimental.model_instance_registry import ModelRegistry, ModelSpec
from vllm.v1.core.sched.execution_slice_scheduler import (
    ExecutionSliceScheduler,
)
from vllm.v1.engine.core import EngineCore


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

    fixed = scheduler.schedule_slice(
        "generator",
        "g-2",
        ExecutionSliceDescriptor(
            2, {"requested_variant": "full"}
        ),
    )
    assert fixed.resource_variant == "full"


def test_engine_core_owns_model_request_routing():
    registry = ModelRegistry(default_model_id="reasoner")
    registry.register(ModelSpec("reasoner", "model", "vllm"))
    registry.register(ModelSpec("generator", "model", "vllm_omni"))
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

    core = EngineCore.__new__(EngineCore)
    core.model_registry = registry
    core._model_request_sessions = {}
    core.scheduler = SimpleNamespace(
        execution_slices=_ExecutionSlices()
    )
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

    core.model_executor = SimpleNamespace(collective_rpc=collective_rpc)
    core._model_request_completion_executor = None
    core._model_request_futures = {}

    native = core.open_model_request("reasoner", "r-1")
    auxiliary_session = core.open_model_request("generator", "g-1")
    auxiliary = core.submit_model_request(
        "generator", "g-1", {"prompt": "video"}, {"variant": "auto"}
    )

    assert native == {"model_id": "reasoner", "request_id": "r-1"}
    assert auxiliary_session == {
        "model_id": "generator",
        "request_id": "g-1",
    }
    assert auxiliary["accepted"] is True
    assert demand == [("reasoner", True), ("generator", True)]
    assert core.wait_model_request_completion("g-1").result() == [
        {"rank": 0, "request_done": True}
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
    core._model_request_completion_executor.shutdown()
