# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Any

from vllm.experimental.archived.communicator_registry import CommunicatorRegistry
from vllm.experimental.worker.binding import (
    CudaGraphBinding,
    CudaStreamBinding,
    ParallelGroupBinding,
    ResourceAssignment,
    TaggedMemoryPoolBinding,
)
from vllm.experimental.worker.resource_binding import (
    MiniContainer,
    RunnerResourceBinderFactory,
    VllmRunnerResourceBinder,
)
from vllm.experimental.worker.mini_container import MiniContainerRegistry
from vllm.experimental.worker.resource_desc import (
    CommunicationGroupDescription,
    ExecutionVariantDescription,
    ResourceDescription,
    StreamResourceDescription,
)


class _ContextBinding:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    @contextmanager
    def activate(
        self,
        runner: Any | None = None,
        *,
        synchronize_transition: bool = True,
    ):
        del runner, synchronize_transition
        self.events.append(f"enter:{self.name}")
        try:
            yield
        finally:
            self.events.append(f"exit:{self.name}")

    def allocation_context(self, tag: str):
        del tag
        return self.activate()

    def communication_context(self):
        return self.activate()


class _GraphBinding:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def activate(self, runner: Any) -> None:
        runner.graph_was_bound = True
        self.events.append("activate:graph")

    def validate_stream(self, streams: Any) -> None:
        del streams


class _ExecUnit:
    def __init__(self, runner: Any, events: list[str]) -> None:
        self.runner = runner
        self.events = events
        self.jobs: list[Any] = []

    def recv_job(self, work: Any) -> None:
        self.jobs.append(work)

    def step(self) -> Any:
        self.events.append("exec-unit:step")
        return self.jobs.pop(0)


def test_parallel_group_binding_references_materialized_registry_groups() -> None:
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = object()
    spec = CommunicationGroupDescription(
        group_id="model-a-tp",
        parallel_dimension="tp",
        ranks=(0, 1),
    )
    registry = CommunicatorRegistry(group_factory=lambda _: tp_group)
    registry.acquire(spec, "model-a")

    binding = ParallelGroupBinding.from_registry(registry, (spec,))

    assert binding.groups == {"tp": tp_group}
    assert binding.get("tp") is tp_group
    assert binding.require("tp") is tp_group
    assert registry.owners(spec.group_id) == frozenset({"model-a"})
    with binding.activate() as active:
        assert active is binding
        assert get_tp_group() is tp_group
    with binding.communication_context() as active:
        assert active is binding
        assert get_tp_group() is tp_group


def test_parallel_group_binding_rejects_duplicate_dimensions() -> None:
    first = CommunicationGroupDescription("first", "tp", (0, 1))
    second = CommunicationGroupDescription("second", "tp", (0, 1))
    registry = CommunicatorRegistry(group_factory=lambda spec: spec.group_id)
    registry.acquire(first, "model-a")
    registry.acquire(second, "model-a")

    try:
        ParallelGroupBinding.from_registry(registry, (first, second))
    except ValueError as exc:
        assert "at most one group" in str(exc)
    else:
        raise AssertionError("Duplicate parallel dimensions were accepted")


def test_cuda_stream_binding_rebinds_runner_without_host_sync(
    monkeypatch,
) -> None:
    import torch

    events: list[str] = []

    class _Stream:
        def __init__(self, name: str, handle: int) -> None:
            self.name = name
            self.cuda_stream = handle
            self.device = 0

        def wait_stream(self, other) -> None:
            events.append(f"{self.name}.wait:{other.name}")

    caller = _Stream("caller", 1)
    previous = _Stream("previous", 2)
    selected = _Stream("selected", 3)
    output_copy = _Stream("output-copy", 4)
    current = [caller]

    @contextmanager
    def use_stream(stream):
        old = current[0]
        current[0] = stream
        events.append(f"enter:{stream.name}")
        try:
            yield
        finally:
            events.append(f"exit:{stream.name}")
            current[0] = old

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: current[0]
    )
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    runner = type("Runner", (), {})()
    runner.main_stream = previous
    runner.output_copy_stream = output_copy
    binding = CudaStreamBinding(selected)

    with binding.activate(runner) as active:
        assert active is binding
        assert runner.main_stream is selected
        assert current[0] is selected
        events.append("execute")

    assert current[0] is caller
    assert events == [
        "selected.wait:previous",
        "selected.wait:output-copy",
        "selected.wait:caller",
        "enter:selected",
        "execute",
        "exit:selected",
        "caller.wait:selected",
    ]


def test_cuda_graph_binding_activates_existing_manager_variant() -> None:
    stream = type("Stream", (), {"cuda_stream": 17})()
    streams = CudaStreamBinding(stream)
    runner = object()

    class _Manager:
        def __init__(self) -> None:
            self.calls = []

        def activate(self, selected_runner, key) -> None:
            self.calls.append((selected_runner, key))

    manager = _Manager()
    graph = CudaGraphBinding(manager, "shared", stream)

    graph.validate_stream(streams)
    graph.activate(runner)

    assert manager.calls == [(runner, "shared")]


def test_cuda_graph_binding_rejects_non_capture_stream() -> None:
    captured = type("Stream", (), {"cuda_stream": 17})()
    assigned = type("Stream", (), {"cuda_stream": 18})()
    graph = CudaGraphBinding(object(), "shared", captured)

    try:
        graph.validate_stream(CudaStreamBinding(assigned))
    except ValueError as exc:
        assert "different from its capture stream" in str(exc)
    else:
        raise AssertionError("Mismatched graph and execution streams were accepted")


def test_tagged_memory_binding_creates_then_reuses_model_pool() -> None:
    events: list[str] = []

    class _Allocator:
        @contextmanager
        def use_memory_pool(self, tag):
            events.append(f"create:{tag}")
            yield

        @contextmanager
        def reuse_memory_pool(self, tag):
            events.append(f"reuse:{tag}")
            yield

        def get_current_usage(self):
            return 4096

    binding = TaggedMemoryPoolBinding(_Allocator(), "generator-a")

    with binding.activate() as active:
        assert active is binding
    with binding.allocation_context("weights"):
        events.append("allocate:weights")
    with binding.allocation_context("weights"):
        events.append("allocate:more-weights")
    with binding.allocation_context("cache"):
        events.append("allocate:cache")

    assert events == [
        "create:generator-a:weights",
        "allocate:weights",
        "reuse:generator-a:weights",
        "allocate:more-weights",
        "create:generator-a:cache",
        "allocate:cache",
    ]
    assert binding.initialized_tags == frozenset(
        {"generator-a:weights", "generator-a:cache"}
    )
    assert binding.current_usage_bytes() == 4096


def test_container_composes_binder_and_resource_agnostic_exec_unit() -> None:
    events: list[str] = []
    runner = type("Runner", (), {})()
    exec_unit = _ExecUnit(runner, events)
    container = MiniContainer(
        exec_unit,
        VllmRunnerResourceBinder(runner),
    )
    assignment = ResourceAssignment(
        communicator=_ContextBinding("communicator", events),
        memory=_ContextBinding("memory", events),
        streams=_ContextBinding("streams", events),
        graph=_GraphBinding(events),
    )

    container.recv_job("batch")
    assert container.step(assignment) == "batch"
    assert runner.graph_was_bound
    assert events == [
        "enter:communicator",
        "enter:memory",
        "enter:streams",
        "activate:graph",
        "exec-unit:step",
        "exit:streams",
        "exit:memory",
        "exit:communicator",
    ]


def test_container_reuses_unchanged_resource_variant() -> None:
    events: list[str] = []
    runner = type("Runner", (), {})()
    exec_unit = _ExecUnit(runner, events)
    container = MiniContainer(
        exec_unit,
        VllmRunnerResourceBinder(runner),
    )
    assignment = ResourceAssignment(
        communicator=_ContextBinding("communicator", events),
        memory=_ContextBinding("memory", events),
        streams=_ContextBinding("streams", events),
        graph=_GraphBinding(events),
    )

    container.recv_job("first")
    container.recv_job("second")
    assert container.step(assignment) == "first"
    assert container.step(assignment) == "second"
    assert events.count("activate:graph") == 1


def test_binder_factory_selects_backend_only_at_container_creation() -> None:
    events: list[str] = []
    runner = type("Runner", (), {})()
    exec_unit = _ExecUnit(runner, events)
    exec_unit.spec = type("Spec", (), {"backend": "vllm"})()

    container = RunnerResourceBinderFactory.create_container(exec_unit)

    assert isinstance(container, MiniContainer)
    assert container.exec_unit is exec_unit


def test_vllm_binder_exposes_initialization_memory_binding() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    events: list[str] = []
    runner = type("Runner", (), {})()
    exec_unit = _ExecUnit(runner, events)
    exec_unit.spec = type("Spec", (), {"backend": "vllm"})()
    memory = _ContextBinding("memory", events)
    assignment = ResourceAssignment(
        communicator=_ContextBinding("communicator", events),
        memory=memory,
        streams=_ContextBinding("streams", events),
    )

    RunnerResourceBinderFactory.create_container(
        exec_unit,
        initialization_assignment=assignment,
    )

    assert runner._vllm_memory_binding is memory
    worker = type("WorkerView", (), {"model_runner": runner})()
    with Worker._maybe_get_memory_pool_context(worker, "kv_cache"):
        events.append("allocate:kv-cache")
    assert events == [
        "enter:memory",
        "allocate:kv-cache",
        "exit:memory",
    ]


class _ContainerManager:
    def __init__(self, binding=None) -> None:
        self.binding = binding
        self.acquired = []
        self.released = []

    def acquire(self, owner, description):
        self.acquired.append((owner, description))
        return self.binding

    def release(self, owner):
        self.released.append(owner)


class _ContainerStreamManager(_ContainerManager):
    def acquire(self, owner, variants):
        variants = tuple(variants)
        self.acquired.append((owner, variants))
        return {
            variant.name: _ContextBinding(
                f"stream:{variant.name}", self.binding
            )
            for variant in variants
        }


def test_managed_container_acquires_selects_and_releases_resources() -> None:
    events: list[str] = []
    runner = type("Runner", (), {})()
    exec_unit = _ExecUnit(runner, events)
    exec_unit.model_id = "model-a"
    exec_unit.spec = type(
        "Spec",
        (),
        {"backend": "vllm", "load_format": "default"},
    )()
    description = ResourceDescription(
        execution_variants=(
            ExecutionVariantDescription(
                "shared", StreamResourceDescription(kind="dedicated")
            ),
            ExecutionVariantDescription(
                "full", StreamResourceDescription(kind="default")
            ),
        ),
        default_variant="shared",
    )
    communicator = _ContextBinding("communicator", events)
    memory = _ContextBinding("memory", events)
    comm_manager = _ContainerManager(communicator)
    memory_manager = _ContainerManager(memory)
    stream_manager = _ContainerStreamManager(events)
    graph_manager = _ContainerManager()

    container = MiniContainer.acquire(
        exec_unit,
        description,
        comm_manager=comm_manager,
        memory_manager=memory_manager,
        stream_manager=stream_manager,
        graph_manager=graph_manager,
    )
    container.recv_job("batch")
    assert container.step(variant_name="full") == "batch"
    assert events[-7:] == [
        "enter:communicator",
        "enter:memory",
        "enter:stream:full",
        "exec-unit:step",
        "exit:stream:full",
        "exit:memory",
        "exit:communicator",
    ]

    registry = MiniContainerRegistry()
    registry.register(container)
    assert registry.get("model-a") is container
    registry.shutdown()
    assert stream_manager.released == ["model-a"]
    assert memory_manager.released == ["model-a"]
    assert comm_manager.released == ["model-a"]
