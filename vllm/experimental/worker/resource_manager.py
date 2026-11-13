# SPDX-License-Identifier: Apache-2.0
"""Worker 本地物理资源的创建、复用和生命周期管理。"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from vllm.experimental.communication import CommunicationGroupDescription
from vllm.experimental.worker.binding import (
    CudaStreamBinding,
    DefaultMemoryBinding,
    GraphBinding,
    ParallelGroupBinding,
    TaggedMemoryPoolBinding,
)
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass
class _ManagedCommunicationGroup:
    """CommManager 内部持有的一组通信资源及其模型 owner。
    
    `description`: 通信组使用的资源描述符
    `group`: 通信组
    `owners`: 记录当前有哪些模型 Container 正在使用这个通信组，集合元素是模型的 model_id。

    理论上通信组可以相同，但是目前实践中，不同模型直接分配不同通信组（哪怕分布式通信关系是同构的）
    
    """

    # 资源描述里的内容
    description: CommunicationGroupDescription
    # 通信组
    group: Any

    owners: set[str]


class CommManager:
    """管理一个 Worker 进程内的模型通信组。

    Engine 负责规划 ``CommunicationGroupDescription``，本 Manager 只在当前 rank
    materialize、复用并释放这些组。它不决定 collective 的提交顺序，也不进行
    跨 rank 调度；这些约束以后由 Engine/UnitScheduler 处理。

    `_group_facotry`: 通信组的工厂
    `_groups`: 存储的若干通信组，key 是 group_id
    `_bindings`: 运行时动态绑定，key 是 owner_id，目前就是 model_id，例如 "generator"
    `_group_ids`: 存储的多个运行组id，key 同样是 owner_id/model_id， value 是该模型持有的全部 group_id

    """

    def __init__(
        self,
        group_factory: Callable[[CommunicationGroupDescription], Any]
        | None = None,
    ) -> None:
        self._group_factory = group_factory or self._create_group
        self._groups: dict[str, _ManagedCommunicationGroup] = {}
        self._bindings: dict[str, ParallelGroupBinding] = {}
        self._group_ids: dict[str, tuple[str, ...]] = {}
        # acquire/release 改变通信组生命周期，必须避免同一 Worker 内并发修改
        # 通信组及其 owner 引用。
        self._lock = threading.RLock()

    @staticmethod
    def _create_group(spec: CommunicationGroupDescription) -> Any:
        """
        内部方法：根据传入的资源需求描述创建通信组
        """
        from vllm.distributed.parallel_state import (
            get_tp_group,
            get_world_group,
            init_model_parallel_group,
        )

        world = get_world_group()
        # 默认tp组
        default_tp = get_tp_group()
        backend = spec.backend
        if backend is None:
            import torch

            # 默认tp组只用来获取通信后端
            # TODO: 未来这里一定要大改
            backend = str(
                torch.distributed.get_backend(default_tp.device_group)
            )
        return init_model_parallel_group(
            group_ranks=[list(spec.ranks)],
            local_rank=int(world.local_rank),
            backend=backend,
            group_name=spec.group_id,
            use_device_communicator=spec.use_device_communicator,
        )

    def _acquire_group(
        self, spec: CommunicationGroupDescription, owner: str
    ) -> Any:
        """
        内部方法，真正创建通信组
        """
        # 先看有没有现存的通信组，必须和当前的id一样
        existing = self._groups.get(spec.group_id)
        if existing is not None:
            if existing.description != spec:
                raise ValueError(
                    f"Communication group {spec.group_id!r} already exists "
                    "with a different resource description"
                )
            existing.owners.add(owner)
            return existing.group

        # 如果没有再创建并行组
        group = self._group_factory(spec)
        self._groups[spec.group_id] = _ManagedCommunicationGroup(
            description=spec,
            group=group,
            owners={owner},
        )
        return group

    def _release_groups(self, owner: str, group_ids: tuple[str, ...]) -> None:
        for group_id in reversed(group_ids):
            slot = self._groups[group_id]
            slot.owners.discard(owner)
            if not slot.owners:
                slot.group.destroy()
                del self._groups[group_id]

    def acquire(
        self,
        owner_id: str,
        specs: Iterable[CommunicationGroupDescription],
    ) -> ParallelGroupBinding:
        """为 Container 创建或复用通信组，并返回只负责接线的 Binding。"""
        # 校验id
        owner = str(owner_id).strip()
        if not owner:
            raise ValueError("Communication resource owner_id cannot be empty")
        normalized = tuple(
            CommunicationGroupDescription.from_rpc_payload(spec)
            for spec in specs
        )

        dimensions: set[str] = set()
        group_ids: set[str] = set()
        # 目前只支持同构TP
        # 根据spec记录属性
        for spec in normalized:
            if spec.parallel_dimension in dimensions:
                raise ValueError(
                    "CommManager accepts at most one group for parallel "
                    f"dimension {spec.parallel_dimension!r}"
                )
            if spec.group_id in group_ids:
                raise ValueError(
                    f"Duplicate communication group id {spec.group_id!r}"
                )
            dimensions.add(spec.parallel_dimension)
            group_ids.add(spec.group_id)

        # 修改前获取锁
        with self._lock:
            if owner in self._bindings:
                raise ValueError(
                    f"Owner {owner!r} already owns communication resources"
                )

            acquired_ids: list[str] = []
            groups: dict[str, Any] = {}
            try:
                for spec in normalized:
                    # 这里真正创建通信组
                    groups[spec.parallel_dimension] = self._acquire_group(
                        spec, owner
                    )
                    acquired_ids.append(spec.group_id)
                binding = ParallelGroupBinding(groups)
            except Exception:
                if acquired_ids:
                    self._release_groups(owner, tuple(acquired_ids))
                raise

            self._bindings[owner] = binding
            self._group_ids[owner] = tuple(acquired_ids)
            return binding

    def get(self, owner_id: str) -> ParallelGroupBinding:
        """取得一个 Container 已申请的通信 Binding。"""
        owner = str(owner_id)
        with self._lock:
            try:
                return self._bindings[owner]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown communication owner {owner!r}"
                ) from exc

    def release(self, owner_id: str) -> None:
        """释放 owner 对通信组的引用；最后一个 owner 负责销毁组。
        当然其实目前还是分开own比较合适
        """
        owner = str(owner_id)
        with self._lock:
            if owner not in self._bindings:
                raise KeyError(f"Unknown communication owner {owner!r}")
            group_ids = self._group_ids[owner]
            self._release_groups(owner, group_ids)
            del self._group_ids[owner]
            del self._bindings[owner]

    def owners(self, group_id: str) -> frozenset[str]:
        """返回通信组当前 owner；仅用于生命周期观测和测试。"""
        with self._lock:
            return frozenset(self._groups[group_id].owners)


class MemoryManager:
    """管理 Worker 内模型级显存 Binding。

    第一版只支持默认 PyTorch allocator 和现有 tagged memory pool。显存预算
    仅被记录，尚不执行 admission。由于底层 allocator 没有安全的按 tag 删除
    pool 接口，``release`` 只结束活跃所有权；已创建的 tagged pool 会保留到
    Worker/allocator shutdown，并允许同一 owner 之后复用。
    
    - _allocator
    延迟创建的底层显存 allocator。只有使用 tagged_pool 时才会初始化。

    - _bindings
    key 是 owner_id/model_id，value 是该模型当前正在使用的显存 Binding。

    - _descriptions
    key 是 owner_id/model_id，value 是该模型对应的 MemoryResourceDescription，用于记录显存池类型和预算。

    - _retired
    保存已经释放所有权、但底层显存池暂时不能安全销毁的 Binding。模型之后以相同资源描述重新注册时，可以复用原有 pool。

    - _lock
    保护上述共享状态，防止模型注册、释放和重新获取资源并发修改这些字典。RLock 允许同一线程嵌套进入。
    """

    def __init__(
        self,
        allocator_factory: Callable[[], Any] | None = None,
    ) -> None:
        if allocator_factory is None:
            from vllm.device_allocator import get_mem_allocator_instance

            allocator_factory = get_mem_allocator_instance
        self._allocator_factory = allocator_factory
        self._allocator: Any | None = None
        self._bindings: dict[
            str, DefaultMemoryBinding | TaggedMemoryPoolBinding
        ] = {}
        self._descriptions: dict[str, MemoryResourceDescription] = {}
        self._retired: dict[
            str,
            tuple[
                MemoryResourceDescription,
                DefaultMemoryBinding | TaggedMemoryPoolBinding,
            ],
        ] = {}
        self._lock = threading.RLock()

    def _get_allocator(self) -> Any:
        if self._allocator is None:
            self._allocator = self._allocator_factory()
            if self._allocator is None:
                raise RuntimeError(
                    "Memory allocator factory returned no allocator"
                )
        return self._allocator

    def acquire(
        self,
        owner_id: str,
        description: MemoryResourceDescription,
    ) -> DefaultMemoryBinding | TaggedMemoryPoolBinding:
        """为 Container 建立显存 Binding；不立即分配模型显存。"""
        owner = str(owner_id).strip()
        if not owner:
            raise ValueError("Memory resource owner_id cannot be empty")
        
        with self._lock:
            if owner in self._bindings:
                raise ValueError(
                    f"Owner {owner!r} already owns memory resources"
                )
            # 先找这个模型以前是否有释放过显存池
            retired = self._retired.pop(owner, None)
            if retired is not None:
                previous_description, binding = retired
                if previous_description != description:
                    self._retired[owner] = retired
                    raise ValueError(
                        f"Owner {owner!r} has retained memory pools with a "
                        "different resource description"
                    )
            elif description.kind == "default":
                # 否则创建新池子并进行管理
                binding = DefaultMemoryBinding()
            elif description.kind == "tagged_pool":
                binding = TaggedMemoryPoolBinding(
                    self._get_allocator(), tag_prefix=owner
                )
            else:
                raise ValueError(
                    f"Unsupported memory resource kind {description.kind!r}"
                )

            self._bindings[owner] = binding
            self._descriptions[owner] = description
            return binding

    def get(
        self, owner_id: str
    ) -> DefaultMemoryBinding | TaggedMemoryPoolBinding:
        """取得一个 Container 当前持有的显存 Binding。"""
        owner = str(owner_id)
        with self._lock:
            try:
                return self._bindings[owner]
            except KeyError as exc:
                raise KeyError(f"Unknown memory owner {owner!r}") from exc

    def description(self, owner_id: str) -> MemoryResourceDescription:
        """返回已规范化的需求；其中预算目前只用于观测。"""
        owner = str(owner_id)
        with self._lock:
            try:
                return self._descriptions[owner]
            except KeyError as exc:
                raise KeyError(f"Unknown memory owner {owner!r}") from exc

    def release(self, owner_id: str) -> None:
        """结束活跃所有权，并保留可能仍被 tensor 引用的 tagged pool。"""
        owner = str(owner_id)
        with self._lock:
            if owner not in self._bindings:
                raise KeyError(f"Unknown memory owner {owner!r}")
            binding = self._bindings.pop(owner)
            description = self._descriptions.pop(owner)
            if isinstance(binding, TaggedMemoryPoolBinding):
                self._retired[owner] = (description, binding)


@dataclass
class _GreenPartitionSlot:
    """  
    它表示一次 Green Context 设备分区及其参与模型。

    - owner
    持有底层 Green Context pair 的对象，例如 GreenContextPair。它负责保持两个 Green Context/stream 存活，并在最终释放时执行 close()。

    - member_owners
    使用这次分区的模型 model_id 集合，例如：{"reasoner", "generator"}"""

    owner: Any
    member_owners: set[str]


class GCStreamManager:
    """创建并持有 Worker 本地的 CUDA/Green Context stream。

    普通 ``default``/``dedicated`` stream 可以独立创建。属于同一个
    ``partition_group`` 的两个 Green Context 需求必须通过 ``acquire_many``
    一次提交，Manager 才能用一次设备资源切分保证二者 SM 集合互不重叠。
    """

    def __init__(
        self,
        device: int,
        *,
        default_stream_factory: Callable[[int], Any] | None = None,
        dedicated_stream_factory: Callable[[int], Any] | None = None,
        green_pair_factory: Callable[[int, int], Any] | None = None,
        external_stream_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        """
        - default_stream_factory(device)
            获取设备当前默认 stream。

        - dedicated_stream_factory(device)
            创建普通独立 CUDA stream。

        - green_pair_factory(generator_sm_count, reasoner_sm_count)
            一次性创建互斥的 Green Context 分区和底层 stream pair。

        - external_stream_factory(stream_pointer, device)
            把 Green Context 提供的底层 CUDA stream handle 包装成 PyTorch 可使用的 ExternalStream。       
        """
        self._device = int(device)
        if default_stream_factory is None:
            import torch

            default_stream_factory = torch.cuda.current_stream
        if dedicated_stream_factory is None:
            import torch

            dedicated_stream_factory = lambda index: torch.cuda.Stream(
                device=index
            )
        if green_pair_factory is None:
            from vllm.experimental.green_context import GreenContextPair

            green_pair_factory = GreenContextPair
        if external_stream_factory is None:
            import torch

            external_stream_factory = lambda pointer, index: (
                torch.cuda.ExternalStream(pointer, device=index)
            )

        self._default_stream_factory = default_stream_factory
        self._dedicated_stream_factory = dedicated_stream_factory
        self._green_pair_factory = green_pair_factory
        self._external_stream_factory = external_stream_factory
        self._bindings: dict[str, dict[str, CudaStreamBinding]] = {}
        self._owner_partition_groups: dict[str, set[str]] = {}
        self._partition_slots: dict[str, _GreenPartitionSlot] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        owner_id: str,
        variants: Iterable[ExecutionVariantDescription],
    ) -> dict[str, CudaStreamBinding]:
        """申请单个 owner 的资源；Green Context 仍必须形成完整二分组。"""
        result = self.acquire_many({owner_id: tuple(variants)})
        return result[str(owner_id).strip()]

    def acquire_many(
        self,
        requirements: Mapping[
            str, Iterable[ExecutionVariantDescription]
        ],
    ) -> dict[str, dict[str, CudaStreamBinding]]:
        """
        原子创建一批 owner 的 stream 变体并返回对应 Binding。
        因为green context是要创建多个分区的
        """
        normalized: dict[str, tuple[ExecutionVariantDescription, ...]] = {}
        # 校验并整理所有模型的 stream variant（shared、full，未来考虑支持多种分区方法）
        # 同时检查同一模型有没有重复的 variant 名称
        for raw_owner, raw_variants in requirements.items():
            owner = str(raw_owner).strip()
            if not owner:
                raise ValueError("Stream resource owner_id cannot be empty")
            variants = tuple(raw_variants)
            names = [variant.name for variant in variants]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Owner {owner!r} has duplicate stream variant names"
                )
            normalized[owner] = variants

        green_requirements: dict[
            str, list[tuple[str, ExecutionVariantDescription]]
        ] = defaultdict(list)

        # 同样是校验逻辑
        for owner, variants in normalized.items():
            for variant in variants:
                if variant.stream.kind == "green_context":
                    group = variant.stream.partition_group
                    assert group is not None
                    green_requirements[group].append((owner, variant))

        for group, members in green_requirements.items():
            if len(members) != 2:
                raise ValueError(
                    f"Green Context partition group {group!r} requires "
                    f"exactly two stream requirements, got {len(members)}"
                )

        with self._lock:
            conflicts = set(normalized).intersection(self._bindings)
            if conflicts:
                raise ValueError(
                    "Owners already own stream resources: "
                    f"{sorted(conflicts)!r}"
                )
            group_conflicts = set(green_requirements).intersection(
                self._partition_slots
            )
            if group_conflicts:
                raise ValueError(
                    "Green Context partition groups already exist: "
                    f"{sorted(group_conflicts)!r}"
                )

            created_bindings: dict[str, dict[str, CudaStreamBinding]] = {
                owner: {} for owner in normalized
            }
            created_pairs: dict[str, _GreenPartitionSlot] = {}
            try:
                for owner, variants in normalized.items():
                    # 这次循环先不创建gc分区和stream
                    for variant in variants:
                        kind = variant.stream.kind
                        if kind == "green_context":
                            continue
                        if kind == "default":
                            stream = self._default_stream_factory(self._device)
                        elif kind == "dedicated":
                            stream = self._dedicated_stream_factory(self._device)
                        else:
                            raise ValueError(
                                f"Unsupported stream resource kind {kind!r}"
                            )
                        created_bindings[owner][variant.name] = (
                            CudaStreamBinding(stream)
                        )

                for group, members in green_requirements.items():
                    # 先切出较大的请求，使 pair 的 remainder 对应较小请求；
                    # 按照SM大小排序即可
                    # 避免依赖 Mapping 的插入顺序决定物理分区。
                    ordered = sorted(
                        members,
                        key=lambda item: (
                            -int(item[1].stream.sm_count or 0),
                            item[0],
                            item[1].name,
                        ),
                    )
                    first_owner, first_variant = ordered[0]
                    second_owner, second_variant = ordered[1]
                    requested_first = int(first_variant.stream.sm_count or 0)
                    requested_second = int(second_variant.stream.sm_count or 0)
                    pair = self._green_pair_factory(
                        requested_first, self._device
                    )
                    try:
                        requested_total = requested_first + requested_second
                        if requested_total != int(pair.total_sm_count):
                            raise ValueError(
                                f"Green Context group {group!r} requests "
                                f"{requested_total} SMs, but device exposes "
                                f"{pair.total_sm_count}"
                            )
                        first_stream = self._external_stream_factory(
                            int(pair.first_stream_ptr), self._device
                        )
                        second_stream = self._external_stream_factory(
                            int(pair.second_stream_ptr), self._device
                        )
                    except Exception:
                        pair.close()
                        raise
                    created_bindings[first_owner][first_variant.name] = (
                        CudaStreamBinding(first_stream)
                    )
                    created_bindings[second_owner][second_variant.name] = (
                        CudaStreamBinding(second_stream)
                    )
                    created_pairs[group] = _GreenPartitionSlot(
                        owner=pair,
                        member_owners={first_owner, second_owner},
                    )
            except Exception:
                for slot in reversed(tuple(created_pairs.values())):
                    slot.owner.close()
                raise

            self._bindings.update(created_bindings)
            for owner in normalized:
                self._owner_partition_groups[owner] = {
                    group
                    for group, members in green_requirements.items()
                    if any(member_owner == owner for member_owner, _ in members)
                }
            self._partition_slots.update(created_pairs)
            return {
                owner: dict(bindings)
                for owner, bindings in created_bindings.items()
            }

    def get(self, owner_id: str, variant_name: str) -> CudaStreamBinding:
        """取得一个 owner 已创建的执行 stream Binding。"""
        owner = str(owner_id)
        with self._lock:
            try:
                return self._bindings[owner][variant_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown stream resource {owner!r}/{variant_name!r}"
                ) from exc

    def bindings(self, owner_id: str) -> dict[str, CudaStreamBinding]:
        """返回 owner 的变体映射副本，避免外部修改 Manager 状态。"""
        owner = str(owner_id)
        with self._lock:
            try:
                return dict(self._bindings[owner])
            except KeyError as exc:
                raise KeyError(f"Unknown stream owner {owner!r}") from exc

    def release(self, owner_id: str) -> None:
        """释放 owner 的 stream；最后一个成员负责销毁 Green Context pair。"""
        owner = str(owner_id)
        with self._lock:
            if owner not in self._bindings:
                raise KeyError(f"Unknown stream owner {owner!r}")

            # 先删除 ExternalStream Binding，再在最后一个成员退出时销毁其
            # 底层 Green Context stream。
            del self._bindings[owner]
            groups = self._owner_partition_groups.pop(owner)
            for group in groups:
                slot = self._partition_slots[group]
                slot.member_owners.discard(owner)
                if not slot.member_owners:
                    slot.owner.close()
                    del self._partition_slots[group]


@dataclass(frozen=True)
class GraphVariantKey:
    """统一 Graph Manager 中一个模型资源变体的稳定标识。

    batch size、token 数等执行形状仍由 vLLM 原生
    ``BatchExecutionDescriptor`` 表达，不重复放入资源层 key。
    也就是说,最终是GraphVariantKey和BatchExecutionDescriptor联合决定cuda graph的区别
    """

    owner_id: str
    variant_name: str

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("Graph variant owner_id cannot be empty")
        if not self.variant_name.strip():
            raise ValueError("Graph variant name cannot be empty")


@dataclass
class _GraphModelSlot:
    """  
    - native_manager
    该模型原生的 vLLM CUDA Graph Manager。一个模型只保留一个，所有资源变体共用它。

    - variants
    已经捕获的资源变体及其 stream，例如：

    {
      GraphVariantKey("reasoner", "shared"): shared_stream_binding,
      GraphVariantKey("reasoner", "full"): full_stream_binding,
    }

    - runner_id
    临时记录正在进行启动期 graph capture 的 Runner 身份，用于把 Runner 发出的 capture 请求路由到正确模型。

    - capture_handler
    启动期多变体捕获回调，负责按 shared/full 等资源变体重复调用原生 capture 流程。"""
    
    native_manager: Any
    variants: dict[GraphVariantKey, CudaStreamBinding]
    runner_id: int | None = None
    capture_handler: Callable[..., int] | None = None


class GCCudaGraphManager:
    """Worker 级统一 CUDA Graph Manager 接口。

    每个模型仍可复用 vLLM 原生创建的 ``ModelCudaGraphManager``，但一个模型
    的所有资源变体必须注册到同一个 native manager；禁止为 shared/full 分别
    构造 native manager。实现内部的完整 executable key 应为
    ``(GraphVariantKey, BatchExecutionDescriptor)``。

    目前多个变体的input buffer仍然共享(因为现阶段只是迁移)
    """

    def __init__(self) -> None:
        self._models: dict[str, _GraphModelSlot] = {}
        self._native_owners: dict[int, str] = {}
        self._lock = threading.RLock()

    def register_model(self, owner_id: str, native_manager: Any) -> None:
        """注册模型唯一的原生 graph manager。"""
        owner = str(owner_id).strip()
        if not owner:
            raise ValueError("Graph model owner_id cannot be empty")
        required = (
            "activate_resource_variant",
            "clear_resource_variant",
        )
        missing = [name for name in required if not hasattr(native_manager, name)]
        if missing:
            raise TypeError(
                "Native CUDA Graph manager lacks resource-variant APIs: "
                f"{missing!r}"
            )
        with self._lock:
            if owner in self._models:
                raise ValueError(
                    f"Graph model owner {owner!r} is already registered"
                )
            native_id = id(native_manager)
            previous_owner = self._native_owners.get(native_id)
            if previous_owner is not None:
                raise ValueError(
                    "One native CUDA Graph manager cannot be registered by "
                    f"both {previous_owner!r} and {owner!r}"
                )
            self._models[owner] = _GraphModelSlot(native_manager, {})
            self._native_owners[native_id] = owner

    def set_capture_handler(
        self,
        owner_id: str,
        runner: Any,
        handler: Callable[..., int],
    ) -> None:
        """临时接管一个 Runner 的启动期多变体 capture 入口。"""
        with self._lock:
            slot = self._require_model(owner_id)
            if slot.capture_handler is not None:
                raise RuntimeError(
                    f"Graph capture handler for {owner_id!r} already exists"
                )
            slot.runner_id = id(runner)
            slot.capture_handler = handler

    def clear_capture_handler(self, owner_id: str) -> None:
        """移除启动期 capture 回调，不影响已经捕获的变体。"""
        with self._lock:
            slot = self._require_model(owner_id)
            slot.runner_id = None
            slot.capture_handler = None

    def capture_all(
        self,
        runner: Any,
        capture_native: Callable[..., int],
        **kwargs: Any,
    ) -> int:
        """实现 GPUModelRunner 的 graph-variant provider 接口。"""
        with self._lock:
            matches = [
                slot
                for slot in self._models.values()
                if slot.runner_id == id(runner)
            ]
            if len(matches) != 1 or matches[0].capture_handler is None:
                raise RuntimeError(
                    "No unique CUDA Graph capture handler for this Runner"
                )
            handler = matches[0].capture_handler
        # capture 很慢，不能在持有 Manager 元数据锁时执行。
        return int(handler(runner, capture_native, **kwargs))

    def capture_variant(
        self,
        key: GraphVariantKey,
        stream: CudaStreamBinding,
        capture_native: Callable[..., int],
        **kwargs: Any,
    ) -> int:
        """在指定资源 stream 上调用原生 capture 路径捕获一种变体。"""
        with self._lock:
            slot = self._require_model(key.owner_id)
            if key in slot.variants:
                raise ValueError(f"CUDA Graph variant {key!r} already exists")
            slot.native_manager.activate_resource_variant(key)
            try:
                with stream.execution_context():
                    captured_bytes = int(capture_native(**kwargs))
            except Exception:
                slot.native_manager.clear_resource_variant(key)
                raise
            slot.variants[key] = stream
            return captured_bytes

    def binding(self, key: GraphVariantKey) -> GraphBinding:
        """返回不拥有 graph 生命周期、仅负责激活变体的 GraphBinding。"""
        from vllm.experimental.worker.binding import CudaGraphBinding

        with self._lock:
            slot = self._require_model(key.owner_id)
            try:
                stream = slot.variants[key]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown CUDA Graph variant {key!r}"
                ) from exc
            return CudaGraphBinding(self, key, stream.stream)

    def activate(self, runner: Any, key: GraphVariantKey) -> None:
        """选择后续 dispatch/replay 使用的资源变体，不替换 native manager。"""
        with self._lock:
            slot = self._require_model(key.owner_id)
            if key not in slot.variants:
                raise KeyError(f"Unknown CUDA Graph variant {key!r}")
            current = getattr(runner, "cudagraph_manager", None)
            if current is not None and current is not slot.native_manager:
                raise RuntimeError(
                    "Runner is attached to a different native CUDA Graph "
                    "manager"
                )
            runner.cudagraph_manager = slot.native_manager
            slot.native_manager.activate_resource_variant(key)

    def has_variant(self, key: GraphVariantKey) -> bool:
        with self._lock:
            slot = self._models.get(key.owner_id)
            return slot is not None and key in slot.variants

    def active_variant(self, owner_id: str) -> GraphVariantKey | None:
        with self._lock:
            slot = self._require_model(owner_id)
            active = getattr(
                slot.native_manager, "active_resource_variant", None
            )
            return active if isinstance(active, GraphVariantKey) else None

    def counts(self, owner_id: str) -> dict[str, int]:
        """返回各资源变体当前持有的 graph entry 数。"""
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphWrapper,
        )
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        with self._lock:
            slot = self._require_model(owner_id)
            original = slot.native_manager.active_resource_variant
            result: dict[str, int] = {}
            try:
                for key in slot.variants:
                    slot.native_manager.activate_resource_variant(key)
                    count = len(slot.native_manager.graphs)
                    count += sum(
                        len(wrapper.resource_variant_entries(key))
                        for wrapper in tuple(CUDAGraphWrapper._all_instances)
                    )
                    count += sum(
                        len(wrapper.resource_variant_entries(key))
                        for wrapper in tuple(
                            BreakableCUDAGraphWrapper._all_instances
                        )
                    )
                    result[key.variant_name] = count
            finally:
                slot.native_manager.activate_resource_variant(original)
            return result

    def release_model(self, owner_id: str) -> None:
        """释放一个模型的所有已登记 graph executable。

        原生 Runner 持有的长期输入/输出 buffer 仍随 Runner 生命周期释放；本层
        不冒充其所有者。后续 completion-event 状态也应由统一 Manager 显式持有
        后再在这里释放。
        """
        owner = str(owner_id)
        with self._lock:
            slot = self._require_model(owner)
            for key in tuple(slot.variants):
                slot.native_manager.clear_resource_variant(key)
            del self._native_owners[id(slot.native_manager)]
            del self._models[owner]

    def clear(self) -> None:
        """释放所有模型登记的资源变体。"""
        with self._lock:
            owners = tuple(self._models)
        for owner in owners:
            self.release_model(owner)

    def _require_model(self, owner_id: str) -> _GraphModelSlot:
        try:
            return self._models[owner_id]
        except KeyError as exc:
            raise KeyError(f"Unknown graph model owner {owner_id!r}") from exc
