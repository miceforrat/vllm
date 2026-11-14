# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import threading
import time
from typing import Any

import numpy as np
import pytest

from vllm.experimental.communication import CommunicationGroupDescription
from vllm.experimental.execution_slice import ExecutionSliceDescriptor
from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.worker.multi_model_worker import MultiModelWorker
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)
from vllm.v1.core.sched.execution_slice_scheduler import ScheduledExecutionSlice


def _bare_worker(
    spec: ModelSpec | None = None,
    *,
    specs: tuple[ModelSpec, ...] | None = None,
    resources: dict[str, Any] | None = None,
) -> MultiModelWorker:
    worker = MultiModelWorker.__new__(MultiModelWorker)
    worker._last_execution_epoch = 0
    if specs is None:
        specs = () if spec is None else (spec,)
    payloads = [item.to_rpc_payload() for item in specs]
    worker.vllm_config = SimpleNamespace(
        additional_config={
            "multi_model_runtime": {
                "startup_models": payloads,
                "startup_resources": resources or {},
            }
        }
    )
    return worker


def test_worker_media_handle_fetch_and_release() -> None:
    worker = _bare_worker()
    worker._init_media_store()
    media = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)

    descriptor = worker._put_media("request-a", media)

    assert descriptor == {
        "handle": "media://request-a",
        "shape": (1, 3, 2, 2),
        "dtype": "float32",
        "device": "cpu",
    }
    fetched = worker.fetch_media(descriptor["handle"], release=True)
    assert fetched is not None
    np.testing.assert_array_equal(fetched["tensor"], media)
    assert fetched["bytes"] == media.nbytes
    assert worker.fetch_media(descriptor["handle"]) is None


def test_startup_model_and_compatibility_resources() -> None:
    group = CommunicationGroupDescription("reasoner-tp", "tp", (0, 1))
    planned = ModelSpec(
        "reasoner",
        "model",
        "vllm",
        worker_ranks=(0, 1),
        communication_groups=(group,),
    )
    worker = _bare_worker(planned)

    assert worker._planned_model_specs() == (planned,)
    description = worker._default_resource_description(planned)
    assert description.communication_groups == (
        CommunicationGroupDescription.from_rpc_payload(group),
    )
    assert description.variant().stream.kind == "default"
    assert description.memory.kind == "default"


def test_planned_models_reject_missing_or_non_vllm_model() -> None:
    with pytest.raises(RuntimeError, match="at least one"):
        _bare_worker()._planned_model_specs()

    worker = _bare_worker(ModelSpec("generator", "model", "vllm_omni"))
    with pytest.raises(RuntimeError, match="exactly one"):
        worker._planned_model_specs()


def test_native_worker_operation_delegates_to_container() -> None:
    events: list[str] = []

    class _Container:
        def load_model(self, *, load_dummy_weights: bool = False) -> None:
            events.append(f"container-load:{load_dummy_weights}")

    worker = _bare_worker()

    class _Registry:
        def require_scheduler_managed(self) -> Any:
            return _Container()

    worker._container_registry = _Registry()

    worker.load_model(load_dummy_weights=True)
    assert events == ["container-load:True"]


def test_worker_executes_engine_scheduled_slice() -> None:
    events: list[Any] = []

    class _Container:
        def execute_slice(
            self,
            *,
            descriptor: ExecutionSliceDescriptor,
            work: Any,
            variant_name: str,
            request_id: str,
        ) -> str:
            events.append(
                (descriptor.sequence_no, work, variant_name, request_id)
            )
            return "done"

    class _Registry:
        def get(self, model_id: str) -> Any:
            assert model_id == "reasoner"
            return _Container()

    worker = _bare_worker()
    worker._container_registry = _Registry()
    worker._last_execution_epoch = 0
    scheduled = ScheduledExecutionSlice(
        model_id="reasoner",
        request_id="batch-1",
        descriptor=ExecutionSliceDescriptor(sequence_no=0),
        resource_variant="shared",
        execution_epoch=7,
    )

    assert worker.execute_slice(scheduled, work="batch") == "done"
    assert events == [(0, "batch", "shared", "batch-1")]
    assert worker._last_execution_epoch == 7


def test_native_warmup_without_scheduled_slice_uses_default_variant() -> None:
    events: list[Any] = []

    class _Container:
        def execute_slice(self, *, work: Any, variant_name: str | None) -> str:
            events.append((work, variant_name))
            return "warmup"

    worker = _bare_worker()
    worker._container_registry = SimpleNamespace(
        require_scheduler_managed=lambda: _Container()
    )
    scheduler_output = SimpleNamespace()

    assert worker.execute_model(scheduler_output) == "warmup"
    assert events == [(scheduler_output, None)]


def test_shutdown_delegates_runner_and_resources_to_container() -> None:
    events: list[str] = []

    class _Registry:
        def shutdown(self) -> None:
            events.append("containers")

    class _GraphManager:
        def clear(self) -> None:
            events.append("graphs")

    worker = _bare_worker()
    worker.model_runner = object()
    worker._container_registry = _Registry()
    worker._graph_manager = _GraphManager()

    worker._shutdown_model_runner()

    assert events == ["containers", "graphs"]
    assert worker.model_runner is None


def test_resource_description_rpc_round_trip() -> None:
    group = CommunicationGroupDescription("generator-tp", "tp", (0, 1))
    description = ResourceDescription(
        communication_groups=(group,),
        memory=MemoryResourceDescription("tagged_pool", 1024),
        execution_variants=(
            ExecutionVariantDescription(
                "shared", StreamResourceDescription("dedicated")
            ),
        ),
        default_variant="shared",
    )

    assert ResourceDescription.from_rpc_payload(
        description.to_rpc_payload()
    ) == description


def test_registry_selects_scheduler_managed_container() -> None:
    from vllm.experimental.worker.mini_container import MiniContainerRegistry

    class _Container:
        def __init__(self, model_id: str, scheduler_managed: bool) -> None:
            self.model_id = model_id
            self.scheduler_managed = scheduler_managed

    registry = MiniContainerRegistry()
    auxiliary = _Container("generator", False)
    scheduler = _Container("reasoner", True)
    registry.register(auxiliary)  # type: ignore[arg-type]
    registry.register(scheduler)  # type: ignore[arg-type]

    assert registry.scheduler_managed() == (scheduler,)
    assert registry.require_scheduler_managed() is scheduler


def test_atomic_startup_plan_restores_both_model_resources() -> None:
    reasoner_group = CommunicationGroupDescription("reasoner-tp", "tp", (0, 1))
    generator_group = CommunicationGroupDescription(
        "generator-tp", "tp", (0, 1)
    )
    reasoner_resource_group = CommunicationGroupDescription(
        "reasoner-tp", "tp", (0, 1)
    )
    generator_resource_group = CommunicationGroupDescription(
        "generator-tp", "tp", (0, 1)
    )
    reasoner = ModelSpec(
        "reasoner",
        "llm",
        "vllm",
        communication_groups=(reasoner_group,),
    )
    generator = ModelSpec(
        "generator",
        "diffusion",
        "vllm_omni",
        communication_groups=(generator_group,),
    )
    descriptions = {
        "reasoner": ResourceDescription(
            communication_groups=(reasoner_resource_group,),
            memory=MemoryResourceDescription("default"),
            execution_variants=(
                ExecutionVariantDescription(
                    "shared",
                    StreamResourceDescription(
                        "green_context", 34, "llm-generator"
                    ),
                ),
            ),
            default_variant="shared",
        ),
        "generator": ResourceDescription(
            communication_groups=(generator_resource_group,),
            memory=MemoryResourceDescription("tagged_pool"),
            execution_variants=(
                ExecutionVariantDescription(
                    "shared",
                    StreamResourceDescription(
                        "green_context", 136, "llm-generator"
                    ),
                ),
            ),
            default_variant="shared",
        ),
    }
    worker = _bare_worker(
        specs=(reasoner, generator),
        resources={
            model_id: description.to_rpc_payload()
            for model_id, description in descriptions.items()
        },
    )

    assert worker._planned_model_specs() == (reasoner, generator)
    assert worker._planned_resource_descriptions(
        (reasoner, generator)
    ) == descriptions


def test_runtime_omni_registration_uses_container_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    spec = ModelSpec("generator", "model", "vllm_omni", worker_ranks=(0,))
    description = ResourceDescription(
        memory=MemoryResourceDescription(kind="default"),
        execution_variants=(
            ExecutionVariantDescription(
                "default", StreamResourceDescription("default")
            ),
        ),
        default_variant="default",
    )

    class _Unit:
        model_id = "generator"

        def __init__(self, actual_spec: Any, config: Any, device: Any) -> None:
            self.spec = actual_spec
            events.append(("unit", config, device))

    class _Container:
        model_id = "generator"

        def load_model(self) -> None:
            events.append("load")

        def shutdown(self) -> None:
            events.append("shutdown")

    container = _Container()

    def fake_acquire(unit: Any, actual: Any, **kwargs: Any) -> Any:
        events.append(("acquire", unit.model_id, actual))
        return container

    class _Registry:
        def register(self, actual: Any) -> None:
            events.append(("register", actual.model_id))

    monkeypatch.setattr(
        "vllm.experimental.worker.multi_model_worker.OmniModelExecUnit",
        _Unit,
    )
    monkeypatch.setattr(
        "vllm.experimental.worker.multi_model_worker.MiniContainer.acquire",
        fake_acquire,
    )
    worker = _bare_worker()
    worker.rank = 0
    worker.device = "cuda:0"
    worker._stream_manager = object()
    worker._comm_manager = object()
    worker._memory_manager = object()
    worker._graph_manager = object()
    worker._container_registry = _Registry()

    result = worker.register_model(
        spec.to_rpc_payload(), description.to_rpc_payload()
    )

    assert result == {"model_id": "generator", "rank": 0, "active": True}
    assert events[-1] == ("register", "generator")


def test_runtime_registration_rejects_partial_green_partition() -> None:
    worker = _bare_worker()
    worker.rank = 0
    worker._stream_manager = object()
    spec = ModelSpec("generator", "model", "vllm_omni")
    description = ResourceDescription(
        execution_variants=(
            ExecutionVariantDescription(
                "shared",
                StreamResourceDescription("green_context", 136, "pair"),
            ),
        ),
        default_variant="shared",
    )

    with pytest.raises(ValueError, match="atomic startup"):
        worker.register_model(
            spec.to_rpc_payload(), description.to_rpc_payload()
        )


def test_model_request_slice_counts_each_wave_request_independently() -> None:
    class _Unit:
        pending_work_count = 2

        def next_slice_spec(self) -> ExecutionSliceDescriptor:
            return ExecutionSliceDescriptor(
                1, metadata={"request_ids": ("B",)}
            )

    class _Container:
        exec_unit = _Unit()

        @staticmethod
        def execute_slice(**_kwargs: Any) -> Any:
            return SimpleNamespace(
                request_done=False,
                sequence_no=0,
                tags=frozenset(("diffusion",)),
                metadata={"finished_request_ids": ("A",)},
            )

        @staticmethod
        def _result_field(result: Any, name: str) -> Any:
            return getattr(result, name)

    worker = _bare_worker()
    worker.rank = 0
    worker._model_jobs_lock = threading.Lock()
    started = time.perf_counter()
    worker._model_jobs = {
        "A": {
            "model_id": "generator",
            "variant_name": "full",
            "max_steps": 2,
            "step_count": 0,
            "started": started,
        },
        "B": {
            "model_id": "generator",
            "variant_name": "full",
            "max_steps": 2,
            "step_count": 0,
            "started": started,
        },
    }
    worker._container_registry = SimpleNamespace(
        get=lambda _model_id: _Container()
    )
    scheduled = ScheduledExecutionSlice(
        model_id="generator",
        request_id="A",
        descriptor=ExecutionSliceDescriptor(
            0, metadata={"request_ids": ("A", "B")}
        ),
        resource_variant="full",
        execution_epoch=1,
    )

    result = worker.execute_model_request_slice("A", scheduled)

    assert result["per_request_step_counts"] == {"A": 1, "B": 1}
    assert result["finished_seconds"]["A"] >= 0
    assert worker._model_jobs["A"]["step_count"] == 1
    assert worker._model_jobs["B"]["step_count"] == 1


def test_over_budget_request_does_not_fail_other_wave_requests() -> None:
    class _Unit:
        pending_work_count = 1
        aborted: tuple[str, ...] = ()

        def abort_requests(self, request_ids: tuple[str, ...]) -> tuple[str, ...]:
            self.aborted = request_ids
            return request_ids

        def next_slice_spec(self) -> ExecutionSliceDescriptor:
            return ExecutionSliceDescriptor(
                0, metadata={"request_ids": ("B",)}
            )

    unit = _Unit()
    container = SimpleNamespace(exec_unit=unit)
    worker = _bare_worker()
    worker.rank = 0
    worker._model_jobs_lock = threading.Lock()
    started = time.perf_counter()
    worker._model_jobs = {
        "A": {
            "model_id": "generator",
            "variant_name": "full",
            "max_steps": 0,
            "step_count": 0,
            "started": started,
        },
        "B": {
            "model_id": "generator",
            "variant_name": "full",
            "max_steps": 2,
            "step_count": 0,
            "started": started,
        },
    }
    worker._container_registry = SimpleNamespace(
        get=lambda _model_id: container
    )
    scheduled = ScheduledExecutionSlice(
        model_id="generator",
        request_id="A",
        descriptor=ExecutionSliceDescriptor(
            0, metadata={"request_ids": ("A", "B")}
        ),
        resource_variant="full",
        execution_epoch=1,
    )

    result = worker.execute_model_request_slice("A", scheduled)

    assert unit.aborted == ("A",)
    assert result["request_done"] is False
    assert result["finished_request_ids"] == ("A",)
    assert result["next_slice"]["metadata"]["request_ids"] == ("B",)
    assert result["metadata"]["terminal_requests"]["A"][
        "status"
    ] == "FINISHED_ERROR"
    # A 导致旧 wave 被撤销；B 尚未执行，将在 next_slice 中重新调度。
    assert worker._model_jobs["B"]["step_count"] == 0
