# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest

from vllm.experimental.worker.resource_desc import (
    CommunicationGroupDescription,
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)
from vllm.experimental.worker.resource_manager import (
    CommManager,
    GCCudaGraphManager,
    GCStreamManager,
    GraphVariantKey,
    MemoryManager,
)


def test_resource_description_selects_normalized_execution_variant() -> None:
    tp = CommunicationGroupDescription("model-tp", "tp", (0, 1))
    shared = ExecutionVariantDescription(
        name="shared",
        stream=StreamResourceDescription(
            kind="green_context",
            sm_count=34,
            partition_group="reasoner-generator",
        ),
        cuda_graph=True,
    )
    full = ExecutionVariantDescription(
        name="full",
        stream=StreamResourceDescription(kind="dedicated"),
        cuda_graph=True,
    )

    description = ResourceDescription(
        communication_groups=(tp,),
        memory=MemoryResourceDescription(
            kind="tagged_pool", capacity_bytes=8 << 30
        ),
        execution_variants=(shared, full),
        default_variant="shared",
    )

    assert description.variant() is shared
    assert description.variant("full") is full
    assert description.communication_groups == (tp,)


def test_green_context_description_requires_partition_information() -> None:
    with pytest.raises(ValueError, match="positive sm_count"):
        StreamResourceDescription(
            kind="green_context", partition_group="pair"
        )
    with pytest.raises(ValueError, match="partition_group"):
        StreamResourceDescription(kind="green_context", sm_count=34)


def test_resource_description_rejects_ambiguous_defaults() -> None:
    variants = (
        ExecutionVariantDescription(
            "shared", StreamResourceDescription(kind="dedicated")
        ),
        ExecutionVariantDescription(
            "full", StreamResourceDescription(kind="default")
        ),
    )

    with pytest.raises(ValueError, match="default_variant is required"):
        ResourceDescription(execution_variants=variants)


def test_resource_description_rejects_duplicate_parallel_dimensions() -> None:
    groups = (
        CommunicationGroupDescription("first", "tp", (0, 1)),
        CommunicationGroupDescription("second", "tp", (0, 1)),
    )

    with pytest.raises(ValueError, match="at most one communication group"):
        ResourceDescription(communication_groups=groups)


class _Group:
    def __init__(self, name: str) -> None:
        self.name = name
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


def test_comm_manager_materializes_binding_and_releases_owned_groups() -> None:
    created: dict[str, _Group] = {}

    def create_group(spec: CommunicationGroupDescription) -> _Group:
        group = _Group(spec.group_id)
        created[spec.group_id] = group
        return group

    manager = CommManager(group_factory=create_group)
    specs = (CommunicationGroupDescription("model-tp", "tp", (0, 1)),)

    binding = manager.acquire("model-a", specs)

    assert binding.require("tp") is created["model-tp"]
    assert manager.get("model-a") is binding
    assert manager.owners("model-tp") == frozenset({"model-a"})

    manager.release("model-a")

    assert created["model-tp"].destroy_calls == 1
    with pytest.raises(KeyError, match="Unknown communication owner"):
        manager.get("model-a")


def test_comm_manager_reuses_groups_between_model_owners() -> None:
    group = _Group("shared-tp")
    manager = CommManager(group_factory=lambda _: group)
    specs = (CommunicationGroupDescription("shared-tp", "tp", (0, 1)),)

    first = manager.acquire("model-a", specs)
    second = manager.acquire("model-b", specs)

    assert first.require("tp") is group
    assert second.require("tp") is group
    assert manager.owners("shared-tp") == frozenset(
        {"model-a", "model-b"}
    )

    manager.release("model-a")
    assert group.destroy_calls == 0
    manager.release("model-b")
    assert group.destroy_calls == 1


def test_comm_manager_rejects_duplicate_owner_acquisition() -> None:
    manager = CommManager(group_factory=lambda spec: _Group(spec.group_id))
    specs = (CommunicationGroupDescription("model-tp", "tp", (0, 1)),)
    manager.acquire("model-a", specs)

    with pytest.raises(ValueError, match="already owns communication"):
        manager.acquire("model-a", specs)


class _Allocator:
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def use_memory_pool(self, tag: str):
        self.events.append(f"create:{tag}")
        yield

    @contextmanager
    def reuse_memory_pool(self, tag: str):
        self.events.append(f"reuse:{tag}")
        yield

    def get_current_usage(self) -> int:
        return 4096


def test_memory_manager_creates_and_reuses_tagged_pool_binding() -> None:
    allocator = _Allocator()
    manager = MemoryManager(allocator_factory=lambda: allocator)
    description = MemoryResourceDescription(
        kind="tagged_pool", capacity_bytes=8 << 30
    )

    binding = manager.acquire("model-a", description)
    with binding.allocation_context("weights"):
        pass
    with binding.allocation_context("weights"):
        pass

    assert manager.get("model-a") is binding
    assert manager.description("model-a") == description
    assert allocator.events == [
        "create:model-a:weights",
        "reuse:model-a:weights",
    ]

    manager.release("model-a")
    with pytest.raises(KeyError, match="Unknown memory owner"):
        manager.get("model-a")

    reacquired = manager.acquire("model-a", description)
    assert reacquired is binding
    with reacquired.allocation_context("weights"):
        pass
    assert allocator.events[-1] == "reuse:model-a:weights"


def test_memory_manager_default_binding_does_not_create_allocator() -> None:
    factory_calls = 0

    def create_allocator():
        nonlocal factory_calls
        factory_calls += 1
        return _Allocator()

    manager = MemoryManager(allocator_factory=create_allocator)
    binding = manager.acquire(
        "model-a", MemoryResourceDescription(kind="default")
    )

    with binding.allocation_context("weights"):
        pass
    manager.release("model-a")

    assert factory_calls == 0


def test_memory_manager_rejects_description_change_for_retained_pool() -> None:
    manager = MemoryManager(allocator_factory=_Allocator)
    original = MemoryResourceDescription(
        kind="tagged_pool", capacity_bytes=8 << 30
    )
    manager.acquire("model-a", original)
    manager.release("model-a")

    with pytest.raises(ValueError, match="different resource description"):
        manager.acquire(
            "model-a",
            MemoryResourceDescription(
                kind="tagged_pool", capacity_bytes=4 << 30
            ),
        )


class _Stream:
    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle


class _GreenPair:
    def __init__(self, first_sm_count: int, device: int) -> None:
        self.total_sm_count = 170
        self.first_sm_count = first_sm_count
        self.second_sm_count = self.total_sm_count - first_sm_count
        self.first_stream_ptr = 1000 + device
        self.second_stream_ptr = 2000 + device
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_stream_manager_creates_disjoint_green_pair_and_full_variants() -> None:
    pairs: list[_GreenPair] = []

    def create_pair(sm_count: int, device: int) -> _GreenPair:
        pair = _GreenPair(sm_count, device)
        pairs.append(pair)
        return pair

    manager = GCStreamManager(
        0,
        default_stream_factory=lambda _: _Stream(10),
        dedicated_stream_factory=lambda _: _Stream(20),
        green_pair_factory=create_pair,
        external_stream_factory=lambda pointer, _: _Stream(pointer),
    )
    requirements = {
        "reasoner": (
            ExecutionVariantDescription(
                "shared",
                StreamResourceDescription(
                    kind="green_context",
                    sm_count=34,
                    partition_group="pair-0",
                ),
            ),
            ExecutionVariantDescription(
                "full", StreamResourceDescription(kind="dedicated")
            ),
        ),
        "generator": (
            ExecutionVariantDescription(
                "shared",
                StreamResourceDescription(
                    kind="green_context",
                    sm_count=136,
                    partition_group="pair-0",
                ),
            ),
            ExecutionVariantDescription(
                "full", StreamResourceDescription(kind="default")
            ),
        ),
    }

    bindings = manager.acquire_many(requirements)

    assert bindings["generator"]["shared"].stream.cuda_stream == 1000
    assert bindings["reasoner"]["shared"].stream.cuda_stream == 2000
    assert manager.get("reasoner", "full").stream.cuda_stream == 20
    assert manager.get("generator", "full").stream.cuda_stream == 10

    manager.release("reasoner")
    assert pairs[0].close_calls == 0
    manager.release("generator")
    assert pairs[0].close_calls == 1


def test_stream_manager_rejects_incomplete_green_partition_group() -> None:
    manager = GCStreamManager(
        0,
        default_stream_factory=lambda _: _Stream(10),
        dedicated_stream_factory=lambda _: _Stream(20),
        green_pair_factory=lambda count, device: _GreenPair(count, device),
        external_stream_factory=lambda pointer, _: _Stream(pointer),
    )
    variant = ExecutionVariantDescription(
        "shared",
        StreamResourceDescription(
            kind="green_context",
            sm_count=34,
            partition_group="incomplete",
        ),
    )

    with pytest.raises(ValueError, match="exactly two"):
        manager.acquire("reasoner", (variant,))


def test_graph_variant_key_only_describes_model_resource_variant() -> None:
    key = GraphVariantKey(owner_id="reasoner", variant_name="shared")

    assert key.owner_id == "reasoner"
    assert key.variant_name == "shared"

    with pytest.raises(ValueError, match="owner_id"):
        GraphVariantKey(owner_id="", variant_name="shared")
    with pytest.raises(ValueError, match="name"):
        GraphVariantKey(owner_id="reasoner", variant_name="")


class _GraphNativeManager:
    def __init__(self) -> None:
        self.active = None
        self.cleared = []
        self.graphs = {}

    def activate_resource_variant(self, key) -> None:
        self.active = key

    def clear_resource_variant(self, key) -> None:
        self.cleared.append(key)


class _GraphStreamBinding:
    def __init__(self, handle: int, events: list[str]) -> None:
        self.stream = _Stream(handle)
        self._events = events

    @contextmanager
    def execution_context(self):
        self._events.append(f"enter:{self.stream.cuda_stream}")
        try:
            yield self
        finally:
            self._events.append(f"exit:{self.stream.cuda_stream}")


def test_unified_graph_manager_reuses_one_native_manager_for_variants() -> None:
    events: list[str] = []
    native = _GraphNativeManager()
    manager = GCCudaGraphManager()
    manager.register_model("reasoner", native)
    shared = GraphVariantKey("reasoner", "shared")
    full = GraphVariantKey("reasoner", "full")

    def capture_native(*, size: int) -> int:
        events.append(f"capture:{native.active.variant_name}")
        return size

    assert manager.capture_variant(
        shared,
        _GraphStreamBinding(11, events),
        capture_native,
        size=8,
    ) == 8
    assert manager.capture_variant(
        full,
        _GraphStreamBinding(22, events),
        capture_native,
        size=9,
    ) == 9

    runner = type("Runner", (), {"cudagraph_manager": native})()
    manager.binding(shared).activate(runner)
    assert native.active == shared
    manager.binding(full).activate(runner)
    assert native.active == full
    assert runner.cudagraph_manager is native
    assert events == [
        "enter:11",
        "capture:shared",
        "exit:11",
        "enter:22",
        "capture:full",
        "exit:22",
    ]

    manager.release_model("reasoner")
    assert set(native.cleared) == {shared, full}


def test_graph_manager_routes_temporary_capture_provider_by_runner() -> None:
    native = _GraphNativeManager()
    manager = GCCudaGraphManager()
    manager.register_model("reasoner", native)
    runner = object()
    calls = []

    def handler(actual_runner, capture_native, *, profile_only):
        calls.append((actual_runner, profile_only))
        return capture_native(profile_only=profile_only) + 1

    def capture_native(*, profile_only):
        assert profile_only is True
        return 7

    manager.set_capture_handler("reasoner", runner, handler)
    assert manager.capture_all(
        runner, capture_native, profile_only=True
    ) == 8
    assert calls == [(runner, True)]

    manager.clear_capture_handler("reasoner")
    with pytest.raises(RuntimeError, match="No unique"):
        manager.capture_all(runner, capture_native, profile_only=True)
