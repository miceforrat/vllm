# SPDX-License-Identifier: Apache-2.0
"""Worker 本地资源管理器使用的规范化资源描述。

本模块目前只定义“需要什么”，不创建任何物理资源。描述对象不包含 backend、
Runner 或 Binding，因此 MiniContainer 可以用同一套结构向不同 ResourceManager
申请资源。ResourceManager 创建完成后的对象由 ``binding.py`` 中的 Binding 接入
Runner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

StreamKind = Literal["default", "dedicated", "green_context"]
MemoryKind = Literal["default", "tagged_pool"]


@dataclass(frozen=True)
class CommunicationGroupDescription:
    """Engine 规划、由各 Worker 实例化的通信资源描述。"""

    group_id: str
    parallel_dimension: str
    ranks: tuple[int, ...]
    backend: str | None = None
    use_device_communicator: bool = True

    @classmethod
    def from_rpc_payload(
        cls,
        payload: Mapping[str, Any] | CommunicationGroupDescription | Any,
    ) -> CommunicationGroupDescription:
        """恢复 RPC payload，并兼容旧 CommunicationGroupSpec。"""
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, Mapping):
            values = dict(payload)
        else:
            values = {
                "group_id": payload.group_id,
                "parallel_dimension": payload.parallel_dimension,
                "ranks": payload.ranks,
                "backend": payload.backend,
                "use_device_communicator": payload.use_device_communicator,
            }
        values["ranks"] = tuple(int(rank) for rank in values["ranks"])
        return cls(**values)


@dataclass(frozen=True)
class MemoryResourceDescription:
    """一个模型实例的显存资源需求。

    ``capacity_bytes`` 是供未来 admission 使用的预算上限，而不是要求 Manager
    在创建 Container 时立即预留同等大小的显存。``None`` 表示暂未声明预算。
    """

    kind: MemoryKind = "tagged_pool"
    capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.capacity_bytes is not None and self.capacity_bytes <= 0:
            raise ValueError("Memory capacity_bytes must be positive")

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | MemoryResourceDescription
    ) -> MemoryResourceDescription:
        return payload if isinstance(payload, cls) else cls(**dict(payload))


@dataclass(frozen=True)
class StreamResourceDescription:
    """一个执行变体所需的 CUDA stream 类型。

    ``partition_group`` 只标识哪些 Green Context 需求必须由 StreamManager 作为
    同一次设备分区统一规划；它不是已创建的 Green Context 或 stream 对象。
    """

    kind: StreamKind = "default"
    sm_count: int | None = None
    partition_group: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "green_context":
            if self.sm_count is None or self.sm_count <= 0:
                raise ValueError(
                    "A green_context stream requires a positive sm_count"
                )
            if not self.partition_group:
                raise ValueError(
                    "A green_context stream requires a partition_group"
                )
            return

        if self.sm_count is not None:
            raise ValueError(
                "sm_count is only valid for a green_context stream"
            )
        if self.partition_group is not None:
            raise ValueError(
                "partition_group is only valid for a green_context stream"
            )

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | StreamResourceDescription
    ) -> StreamResourceDescription:
        return payload if isinstance(payload, cls) else cls(**dict(payload))


@dataclass(frozen=True)
class ExecutionVariantDescription:
    """同一模型实例的一种可调度执行资源变体。

    例如 ``shared`` 可以使用受限 Green Context stream 并启用 CUDA Graph，
    ``full`` 可以使用完整设备 stream 和另一套 Graph variant。具体 graph 的
    capture/replay 仍由 GCCudaGraphManager 管理。
    """

    name: str
    stream: StreamResourceDescription
    cuda_graph: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Execution variant name cannot be empty")

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | ExecutionVariantDescription
    ) -> ExecutionVariantDescription:
        if isinstance(payload, cls):
            return payload
        values = dict(payload)
        values["stream"] = StreamResourceDescription.from_rpc_payload(
            values["stream"]
        )
        return cls(**values)


@dataclass(frozen=True)
class ResourceDescription:
    """MiniContainer 向各 ResourceManager 提交的规范化资源需求。

    本对象只描述模型生命周期资源及可选执行变体。它不携带已创建资源，也不
    根据模型 backend 推断需求。
    """

    communication_groups: tuple[CommunicationGroupDescription, ...] = ()
    memory: MemoryResourceDescription = field(
        default_factory=MemoryResourceDescription
    )
    execution_variants: tuple[ExecutionVariantDescription, ...] = ()
    default_variant: str | None = None

    def __post_init__(self) -> None:
        normalized_groups = tuple(
            CommunicationGroupDescription.from_rpc_payload(group)
            for group in self.communication_groups
        )
        object.__setattr__(self, "communication_groups", normalized_groups)
        # 通信组校验
        group_ids: set[str] = set()
        dimensions: set[str] = set()
        for group in self.communication_groups:
            if group.group_id in group_ids:
                raise ValueError(
                    f"Duplicate communication group id {group.group_id!r}"
                )
            if group.parallel_dimension in dimensions:
                raise ValueError(
                    "A ResourceDescription accepts at most one communication "
                    f"group for {group.parallel_dimension!r}"
                )
            group_ids.add(group.group_id)
            dimensions.add(group.parallel_dimension)

        # 执行变体（stream、graph校验）
        variant_names = [variant.name for variant in self.execution_variants]
        if len(variant_names) != len(set(variant_names)):
            raise ValueError("Execution variant names must be unique")

        if self.default_variant is None:
            if len(self.execution_variants) > 1:
                raise ValueError(
                    "default_variant is required when multiple execution "
                    "variants are declared"
                )
        elif self.default_variant not in set(variant_names):
            raise ValueError(
                f"Unknown default execution variant {self.default_variant!r}"
            )

    def variant(self, name: str | None = None) -> ExecutionVariantDescription:
        """按名称取得执行变体；未指定时返回默认变体。"""
        selected = self.default_variant if name is None else name
        if selected is None:
            if len(self.execution_variants) == 1:
                return self.execution_variants[0]
            raise KeyError("ResourceDescription has no default variant")
        for variant in self.execution_variants:
            if variant.name == selected:
                return variant
        raise KeyError(f"Unknown execution variant {selected!r}")

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | ResourceDescription
    ) -> ResourceDescription:
        """恢复 Engine 下发的规范化资源需求。"""
        if isinstance(payload, cls):
            return payload
        values = dict(payload)
        values["communication_groups"] = tuple(
            CommunicationGroupDescription.from_rpc_payload(group)
            for group in values.get("communication_groups", ())
        )
        values["memory"] = MemoryResourceDescription.from_rpc_payload(
            values.get("memory", {})
        )
        values["execution_variants"] = tuple(
            ExecutionVariantDescription.from_rpc_payload(variant)
            for variant in values.get("execution_variants", ())
        )
        return cls(**values)

    def to_rpc_payload(self) -> dict[str, Any]:
        """转换为仅包含 RPC 安全基础类型的 payload。"""
        from dataclasses import asdict

        return asdict(self)
