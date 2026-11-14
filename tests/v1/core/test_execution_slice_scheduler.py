import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

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
    with pytest.raises(ValueError, match="Unknown resource variant"):
        scheduler.validate_requested_variant("generator", "generator_full")


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

        def validate_requested_variant(self, model_id, requested):
            return slices.validate_requested_variant(model_id, requested)

    core = EngineCore.__new__(EngineCore)
    core.model_registry = registry
    core._model_request_sessions = {}
    core._model_request_demand = set()
    core._model_request_demand_lock = threading.Lock()
    core._model_request_driver_futures = {}
    core._model_request_driver_lock = threading.Lock()
    core._aborted_model_requests = set()
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
    core._model_request_variants = {}

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
    # Container driver 在 completion executor 中异步推进；先等待最终结果，
    # 再断言最终 demand，避免把线程调度时序当成同步 API 契约。
    assert core.wait_model_request_completion("g-1").result() == [
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
    core._model_request_completion_executor.shutdown()


def test_engine_core_propagates_model_request_terminal_error():
    core = EngineCore.__new__(EngineCore)
    core._model_request_futures = {"g-1": Future()}
    core._model_request_variants = {"g-1": "full"}
    core._model_request_sessions = {"g-1": "generator"}
    core._model_request_demand = {"g-1"}
    core._model_request_demand_lock = threading.Lock()
    core._model_request_driver_lock = threading.Lock()
    core._aborted_model_requests = set()
    core.scheduler = SimpleNamespace(
        execution_slices=SimpleNamespace(
            schedule_slice=lambda *_args: SimpleNamespace(
                to_payload=lambda: {"scheduled": True}
            ),
            set_model_demand=lambda *_args: None,
        )
    )
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda *_args, **_kwargs: [
            {
                "request_done": True,
                "finished_request_ids": ("g-1",),
                "metadata": {
                    "terminal_requests": {
                        "g-1": {
                            "status": "FINISHED_ERROR",
                            "error": "injected failure",
                        }
                    }
                },
            }
        ]
    )

    first_workers = [
        {
            "next_slice": {
                "sequence_no": 0,
                "tags": ["vae"],
                "metadata": {},
            }
        }
    ]
    core._drive_model_request("generator", "g-1", "full", first_workers)

    with pytest.raises(RuntimeError, match="injected failure"):
        core._model_request_futures["g-1"].result()


def test_engine_core_selects_variant_from_each_wave_requests():
    core = EngineCore.__new__(EngineCore)
    core._model_request_futures = {"A": Future(), "B": Future()}
    core._model_request_variants = {"A": "full", "B": "auto"}
    core._model_request_sessions = {"A": "generator", "B": "generator"}
    core._model_request_demand = {"A", "B"}
    core._model_request_demand_lock = threading.Lock()
    core._model_request_driver_lock = threading.Lock()
    core._aborted_model_requests = set()
    requested_variants: list[str] = []

    def schedule_slice(_model_id, _request_id, descriptor):
        requested_variants.append(descriptor.metadata["requested_variant"])
        return SimpleNamespace(to_payload=lambda: {"scheduled": True})

    core.scheduler = SimpleNamespace(
        execution_slices=SimpleNamespace(
            schedule_slice=schedule_slice,
            set_model_demand=lambda *_args: None,
        )
    )
    responses = iter(
        (
            [
                {
                    "request_done": False,
                    "finished_request_ids": ("A",),
                    "finished_seconds": {"A": 1.25},
                    "per_request_step_counts": {"A": 4, "B": 2},
                    "finished_media": {
                        "A": {"handle": "media://A", "shape": (1, 3)}
                    },
                    "metadata": {},
                    "next_slice": {
                        "sequence_no": 1,
                        "tags": ["diffusion"],
                        "metadata": {"request_ids": ("B",)},
                    },
                }
            ],
            [
                {
                    "request_done": True,
                    "finished_request_ids": ("B",),
                    "finished_seconds": {"B": 2.5},
                    "per_request_step_counts": {"B": 4},
                    "metadata": {},
                }
            ],
        )
    )
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda *_args, **_kwargs: next(responses)
    )
    first_workers = [
        {
            "next_slice": {
                "sequence_no": 0,
                "tags": ["diffusion"],
                "metadata": {"request_ids": ("A", "B")},
            }
        }
    ]

    core._drive_model_request("generator", "A", "full", first_workers)

    assert requested_variants == ["full", "auto"]
    assert core._model_request_futures["A"].done()
    assert core._model_request_futures["B"].done()
    assert core._model_request_futures["A"].result()[0]["seconds"] == 1.25
    assert core._model_request_futures["A"].result()[0]["step_count"] == 4
    assert core._model_request_futures["A"].result()[0]["media"] == {
        "handle": "media://A",
        "shape": (1, 3),
    }
    assert core._model_request_futures["B"].result()[0]["seconds"] == 2.5
    assert core._model_request_futures["B"].result()[0]["step_count"] == 4
