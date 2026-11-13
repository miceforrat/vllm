# SPDX-License-Identifier: Apache-2.0
"""模型无关的细粒度执行 slice 契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ExecutionSliceDescriptor:
    """模型侧与 EngineCore 共同使用的下一段工作描述。

    这个描述主要侧重于“下一段的工作是什么”，而没有描述资源

    ``sequence_no`` 是请求内单调递增的 slice 序号；``tags`` 描述跨模型可
    比较的稳定执行性质，供调度策略匹配。模型私有诊断信息仍放在
    ``metadata`` 目前的作用只是日志输出，调度器不能依赖这些字段做控制流判断（消除对于底层细节的控制），
    或者说我们不能仅针对“name”做控制，真正值得用作控制的应当是资源属性而不是私有的组件名）。该类型既是模型
    报告的下一 slice，也是 Scheduler 的输入，不再区分 Spec 与 Boundary。
    """

    sequence_no: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # slice对应的tag
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        normalized = frozenset(str(tag).strip() for tag in self.tags)
        if "" in normalized:
            raise ValueError("Execution slice tags cannot be empty")
        object.__setattr__(self, "tags", normalized)

    def to_payload(self) -> dict[str, Any]:
        return {
            "sequence_no": self.sequence_no,
            "tags": sorted(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | ExecutionSliceDescriptor,
    ) -> ExecutionSliceDescriptor:
        if isinstance(payload, cls):
            return payload
        return cls(
            sequence_no=int(payload["sequence_no"]),
            metadata=dict(payload.get("metadata", {})),
            tags=frozenset(str(tag) for tag in payload.get("tags", ())),
        )


@dataclass(frozen=True)
class ExecutionSliceResult:
    """一次 slice 推进的通用结果。"""

    sequence_no: int
    # 请求是否完成了
    request_done: bool
    output: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tags: frozenset[str] = field(default_factory=frozenset)

class ExecutionSliceSession(Protocol):
    """一个请求私有的、可反复推进的模型执行状态。
       用来支持一个model同时serving多个请求（不过目前还没有给omni在缓存等层面做更多支持——至少cosmos3还不行）
    """

    @property
    def request_id(self) -> str: ...

    def next_slice(self) -> ExecutionSliceDescriptor: ...

    def execute_slice(self) -> ExecutionSliceResult: ...

    def close(self) -> None: ...


class ExecutionSliceSessionFactory(Protocol):
    """为模型请求创建细粒度执行 session。
       每收到一个新请求，通过 create_session(request_id, work) 创建独立 Session；
    """

    def create_session(
        self,
        request_id: str,
        work: Any,
    ) -> ExecutionSliceSession: ...
