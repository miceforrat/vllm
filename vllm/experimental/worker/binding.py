# SPDX-License-Identifier: Apache-2.0
"""Worker 执行资源的统一 Binding 接口。

Binding 表示 ResourceManager 已经创建并选择好的物理资源。它只负责把资源
临时接入 Runner，不负责资源的创建、选择、调度或销毁。
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.experimental.communication import CommunicationGroupDescription


class CommunicatorBinding(Protocol):
    """将既有通信上下文接入当前执行。"""

    def activate(self) -> AbstractContextManager[Any]: ...

    def communication_context(self) -> AbstractContextManager[Any]: ...


# 用当前并行组替换函数内原有的并行组
class ParallelGroupBinding:
    """一次模型执行可见的并行通信组集合。

    通信组由 ``CommManager`` 按模型生命周期创建和持有；本对象只保存
    已有 group 的只读视图，并在执行期间将该视图交给 Runner。当前版本只描述
    并行组，不负责 collective 排序或跨 rank 调度。
    """

    def __init__(self, groups: Mapping[str, Any]) -> None:
        normalized: dict[str, Any] = {}
        for dimension, group in groups.items():
            name = str(dimension)
            if not name:
                raise ValueError("Parallel dimension cannot be empty")
            if group is None:
                raise ValueError(
                    f"Parallel group {name!r} has not been materialized"
                )
            normalized[name] = group
        self._groups = MappingProxyType(normalized)

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        specs: Iterable[CommunicationGroupDescription],
    ) -> ParallelGroupBinding:
        """兼容旧 Registry：引用既有组，不改变其所有权和引用计数。"""
        groups: dict[str, Any] = {}
        for spec in specs:
            dimension = str(spec.parallel_dimension)
            if dimension in groups:
                raise ValueError(
                    "A CommunicatorBinding accepts at most one group for "
                    f"parallel dimension {dimension!r}"
                )
            groups[dimension] = registry.get(spec.group_id)
        return cls(groups)

    @property
    def groups(self) -> Mapping[str, Any]:
        """返回不可变的 ``parallel_dimension -> group`` 映射。"""
        return self._groups

    def get(self, parallel_dimension: str) -> Any | None:
        return self._groups.get(parallel_dimension)

    def require(self, parallel_dimension: str) -> Any:
        try:
            return self._groups[parallel_dimension]
        except KeyError as exc:
            raise KeyError(
                f"No communication group for {parallel_dimension!r}"
            ) from exc

    @contextmanager
    def activate(self) -> Iterator[ParallelGroupBinding]:
        """让 vLLM/Omni getter 在当前执行中优先读取本 Binding。"""
        from vllm.distributed.parallel_state import (
            use_tensor_model_parallel_group,
        )

        with ExitStack() as stack:
            tp_group = self.get("tp")
            if tp_group is not None:
                stack.enter_context(
                    use_tensor_model_parallel_group(tp_group)
                )

            try:
                from vllm_omni.diffusion.distributed.parallel_state import (
                    use_parallel_group_provider,
                )
            except ModuleNotFoundError as exc:
                if exc.name is None or not exc.name.startswith("vllm_omni"):
                    raise
            else:
                stack.enter_context(use_parallel_group_provider(self))
            yield self

    def communication_context(
        self,
    ) -> AbstractContextManager[ParallelGroupBinding]:
        """兼容 Omni ``DiffusionCommunicatorProvider`` 接口。"""
        return self.activate()


class MemoryBinding(Protocol):
    """将既有显存上下文接入当前执行或分配过程。"""

    def activate(self) -> AbstractContextManager[Any]: ...

    def allocation_context(
        self, tag: str
    ) -> AbstractContextManager[Any]: ...


class DefaultMemoryBinding:
    """不改变 PyTorch 默认 allocator 的显存 Binding。"""

    @contextmanager
    def activate(self) -> Iterator[DefaultMemoryBinding]:
        yield self

    @contextmanager
    def allocation_context(
        self, tag: str
    ) -> Iterator[DefaultMemoryBinding]:
        del tag
        yield self


class TaggedMemoryPoolBinding:
    """把模型级 tagged memory pool 接入显存分配点。

    allocator 及其 pool 的生命周期由 MemoryManager 持有。Binding 只负责将
    ``模型前缀 + 局部用途`` 解析为稳定 tag，并在首次分配后复用同一 pool。
    """

    def __init__(self, allocator: Any, tag_prefix: str) -> None:
        if allocator is None:
            raise ValueError(
                "TaggedMemoryPoolBinding requires a materialized allocator"
            )
        normalized_prefix = str(tag_prefix).strip()
        if not normalized_prefix:
            raise ValueError("Memory-pool tag prefix cannot be empty")
        self._allocator = allocator
        self._tag_prefix = normalized_prefix
        self._initialized_tags: set[str] = set()
        # CuMemAllocator.current_tag 是进程内共享状态。只串行化实际的
        # allocation context，不影响不发生持久分配的模型执行。
        self._allocation_lock = threading.RLock()

    @property
    def allocator(self) -> Any:
        return self._allocator

    @property
    def tag_prefix(self) -> str:
        return self._tag_prefix

    @property
    def initialized_tags(self) -> frozenset[str]:
        return frozenset(self._initialized_tags)

    def _pool_tag(self, local_tag: str) -> str:
        normalized = str(local_tag).strip()
        if not normalized:
            raise ValueError("Memory-pool local tag cannot be empty")
        return f"{self._tag_prefix}:{normalized}"

    @contextmanager
    def activate(self) -> Iterator[TaggedMemoryPoolBinding]:
        """执行期作用域；具体 pool 仅在 allocation_context 中进入。"""
        yield self

    @contextmanager
    def allocation_context(
        self, tag: str
    ) -> Iterator[TaggedMemoryPoolBinding]:
        """
        真正切换显存池的函数
        """
        pool_tag = self._pool_tag(tag)
        with self._allocation_lock:
            creating = pool_tag not in self._initialized_tags
            if not creating:
                reuse = getattr(self._allocator, "reuse_memory_pool", None)
                if reuse is None:
                    raise RuntimeError(
                        "The configured allocator cannot re-enter an existing "
                        f"memory pool {pool_tag!r}"
                    )
                context = reuse(pool_tag)
            else:
                context = self._allocator.use_memory_pool(tag=pool_tag)
            with context:
                # CuMemAllocator 在 context enter 时完成 pool 注册；即使随后
                # 的模型分配失败，下次也必须走 reuse，而不能重复创建同名 pool。
                if creating:
                    self._initialized_tags.add(pool_tag)
                yield self

    def current_usage_bytes(self) -> int:
        """返回 allocator 当前跟踪的总用量，仅用于观测。"""
        return int(self._allocator.get_current_usage())


class StreamBinding(Protocol):
    """将既有 CUDA stream 资源接入指定 Runner。"""

    def activate(
        self, runner: Any, *, synchronize_transition: bool = True
    ) -> AbstractContextManager[Any]: ...

    def execution_context(self) -> AbstractContextManager[Any]: ...


class CudaStreamBinding:
    """把一个已经创建的 CUDA stream 接入 Runner。

    Binding 不拥有 stream，也不负责 Green Context 的创建和分区选择。切换
    vLLM Runner 的执行流时，通过设备端 stream dependency 保护上一轮计算和
    asynchronous output copy；不会执行 host-side ``synchronize()``。
    """

    def __init__(self, stream: Any) -> None:
        if stream is None:
            raise ValueError("CudaStreamBinding requires a materialized stream")
        self._stream = stream

    @property
    def stream(self) -> Any:
        return self._stream

    @staticmethod
    def _cached_main_stream(runner: Any) -> Any | None:
        if runner is None:
            return None
        values = getattr(runner, "__dict__", None)
        if not isinstance(values, dict):
            return None
        return values.get("main_stream")

    @staticmethod
    def _same_stream(left: Any, right: Any) -> bool:
        if left is right:
            return True
        left_handle = getattr(left, "cuda_stream", None)
        right_handle = getattr(right, "cuda_stream", None)
        return (
            left_handle is not None
            and right_handle is not None
            and int(left_handle) == int(right_handle)
        )

    @staticmethod
    def _bind_runner_main_stream(runner: Any, stream: Any) -> None:
        if runner is None:
            return
        runner_type = type(runner)
        values = getattr(runner, "__dict__", None)
        if not isinstance(values, dict):
            return
        if "main_stream" in values or hasattr(runner_type, "main_stream"):
            # GPUModelRunner.main_stream 是 cached_property。将缓存更新为实际
            # 执行流，使 AsyncOutput 在正确的 stream 上记录 handoff event。
            values["main_stream"] = stream

    @contextmanager
    def _activate(
        self,
        runner: Any | None,
        *,
        synchronize_transition: bool = True,
    ) -> Iterator[CudaStreamBinding]:
        import torch

        stream = self._stream
        caller = torch.cuda.current_stream(stream.device)
        if self._same_stream(caller, stream):
            # 不切换stream就直接返回
            self._bind_runner_main_stream(runner, stream)
            yield self
            return

        if not synchronize_transition:
            # 连续 slice 使用同一资源变体时，上一 slice 已在同一 stream
            # 排队；只恢复 CUDA current-stream 作用域，不重复建立与 caller
            # stream、output-copy stream 的交接依赖。
            self._bind_runner_main_stream(runner, stream)
            with torch.cuda.stream(stream):
                yield self
            return

        # 否则才需要执行同步
        previous = self._cached_main_stream(runner)
        if previous is not None and not self._same_stream(previous, stream):
            # TODO(cosmos3): 这是迁移期的保守兼容逻辑。跨 graph variant 的
            # compute/output-copy 完成关系最终应由统一 CUDA Graph Manager
            # 结合持久输出 buffer 的生命周期维护；届时 StreamBinding 只消费
            # Manager 提供的 completion dependency，不应自行识别两类 stream。
            # 这里写的太难看了，肯定要改
            stream.wait_stream(previous)
            for name in ("output_copy_stream", "async_output_copy_stream"):
                output_copy_stream = getattr(runner, name, None)
                if output_copy_stream is not None:
                    stream.wait_stream(output_copy_stream)

        # 保留调用者在进入 Binding 前已经提交的依赖。
        stream.wait_stream(caller)
        self._bind_runner_main_stream(runner, stream)
        try:
            with torch.cuda.stream(stream):
                yield self
        finally:
            # 只建立设备端完成依赖，不阻塞 host 线程。
            caller.wait_stream(stream)

    def activate(
        self, runner: Any, *, synchronize_transition: bool = True
    ) -> AbstractContextManager[CudaStreamBinding]:
        return self._activate(
            runner, synchronize_transition=synchronize_transition
        )

    def execution_context(
        self,
    ) -> AbstractContextManager[CudaStreamBinding]:
        """兼容 Omni ``DiffusionStreamProvider`` 接口。"""
        return self._activate(None)


class GraphBinding(Protocol):
    """激活 ResourceManager 已选择的 CUDA Graph variant。"""

    def validate_stream(self, streams: StreamBinding) -> None: ...

    def activate(self, runner: Any) -> None: ...


@dataclass(frozen=True)
class CudaGraphBinding:
    """统一 Graph Manager 中一个已经完成 capture 的资源变体引用。

    ``variant_key`` 对 Container 保持不透明；Binding 不创建 graph、不执行
    capture，也不拥有 manager 的生命周期。
    """

    manager: Any
    variant_key: Any
    capture_stream: Any

    @staticmethod
    def _stream_handle(stream: Any) -> int | None:
        handle = getattr(stream, "cuda_stream", None)
        return None if handle is None else int(handle)

    def validate_stream(self, streams: StreamBinding) -> None:
        """拒绝在不同于 capture stream 的资源变体上执行 graph。"""
        assigned = getattr(streams, "stream", None)
        if assigned is None:
            raise ValueError(
                "A CUDA Graph binding requires a concrete stream binding"
            )
        if assigned is self.capture_stream:
            return
        assigned_handle = self._stream_handle(assigned)
        captured_handle = self._stream_handle(self.capture_stream)
        if (
            assigned_handle is None
            or captured_handle is None
            or assigned_handle != captured_handle
        ):
            raise ValueError(
                "CUDA Graph variant was assigned to a stream different from "
                "its capture stream"
            )

    def activate(self, runner: Any) -> None:
        self.manager.activate(runner, self.variant_key)


@dataclass(frozen=True)
class ResourceAssignment:
    """ResourceManager 已解析完成的一次执行资源组合。"""

    communicator: CommunicatorBinding
    memory: MemoryBinding
    streams: StreamBinding
    graph: GraphBinding | None = None
