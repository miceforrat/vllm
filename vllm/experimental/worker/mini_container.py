# SPDX-License-Identifier: Apache-2.0
"""模型执行单元与 Worker 物理资源之间的生命周期容器。"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from vllm.experimental.worker.binding import ResourceAssignment
from vllm.experimental.worker.model_timeline import (
    ModelGPUTimeline,
    SliceExecutionRecord,
)
from vllm.experimental.worker.resource_manager import GraphVariantKey

if TYPE_CHECKING:
    from vllm.experimental.execution_slice import ExecutionSliceDescriptor
    from vllm.experimental.worker.binding import CudaStreamBinding
    from vllm.experimental.worker.resource_desc import ResourceDescription
    from vllm.experimental.worker.resource_manager import (
        CommManager,
        GCCudaGraphManager,
        GCStreamManager,
        MemoryManager,
    )
class MiniContainer:
    """持有一个 ExecUnit 及其模型生命周期资源。

    Container 只解析规范化的 ``ResourceDescription``，不会根据 backend
    推断资源。后端差异由 ``ExecUnit`` 处理，物理资源创建与销毁分别委托
    给对应 ResourceManager。
    """

    def __init__(
        self,
        exec_unit: Any,
        *,
        description: ResourceDescription | None = None,
        assignments: Mapping[str, ResourceAssignment] | None = None,
        comm_manager: CommManager | None = None,
        memory_manager: MemoryManager | None = None,
        stream_manager: GCStreamManager | None = None,
        graph_manager: GCCudaGraphManager | None = None,
        graph_registered: bool = False,
        scheduler_managed: bool = False,
        runtime_added: bool = False,
    ) -> None:
        self._exec_unit = exec_unit
        self._description = description
        self._assignments = dict(assignments or {})
        self._comm_manager = comm_manager
        self._memory_manager = memory_manager
        self._stream_manager = stream_manager
        self._graph_manager = graph_manager
        self._graph_registered = graph_registered
        self._scheduler_managed = scheduler_managed
        self._runtime_added = runtime_added
        self._variant_step_counts: Counter[str] = Counter()
        self._active_assignment: ResourceAssignment | None = None
        self._timeline = ModelGPUTimeline()
        self._slice_records: deque[SliceExecutionRecord] = deque(maxlen=4096)
        self._require_graph = True
        self._closed = False
        self._lock = threading.RLock()


    @classmethod
    def acquire(
        cls,
        exec_unit: Any,
        description: ResourceDescription,
        *,
        comm_manager: CommManager,
        memory_manager: MemoryManager,
        stream_manager: GCStreamManager,
        graph_manager: GCCudaGraphManager,
        stream_bindings: Mapping[str, CudaStreamBinding] | None = None,
        scheduler_managed: bool = False,
        runtime_added: bool = False,
    ) -> MiniContainer:
        """从各 Manager 申请资源，并完成 Runner 初始化接线。"""

        owner = exec_unit.model_id
        communicator = comm_manager.acquire(
            owner, description.communication_groups
        )
        memory = None
        graph_registered = False
        try:
            memory = memory_manager.acquire(owner, description.memory)
            streams = (
                stream_manager.acquire(owner, description.execution_variants)
                if stream_bindings is None
                else dict(stream_bindings)
            )
            expected = {
                variant.name for variant in description.execution_variants
            }
            if set(streams) != expected:
                raise ValueError(
                    "Pre-acquired stream variants do not match the resource "
                    f"description: expected={sorted(expected)!r}, "
                    f"actual={sorted(streams)!r}"
                )

            assignments = {
                variant.name: ResourceAssignment(
                    communicator=communicator,
                    memory=memory,
                    streams=streams[variant.name],
                )
                for variant in description.execution_variants
            }
            default_assignment = assignments[
                description.variant().name
            ]

            exec_unit.bind_resources(
                default_assignment,
                load_format=exec_unit.spec.load_format,
            )


            return cls(
                exec_unit,
                description=description,
                assignments=assignments,
                comm_manager=comm_manager,
                memory_manager=memory_manager,
                stream_manager=stream_manager,
                graph_manager=graph_manager,
                graph_registered=graph_registered,
                scheduler_managed=scheduler_managed,
                runtime_added=runtime_added,
            )
        except Exception:
            if graph_registered:
                graph_manager.release_model(owner)
            try:
                stream_manager.release(owner)
            except KeyError:
                pass
            if memory is not None:
                memory_manager.release(owner)
            comm_manager.release(owner)
            raise

    def register_graph_manager(self) -> None:
        """模型加载后，把已创建的原生 Graph manager 延迟接入统一管理器。"""
        with self._lock:
            self._ensure_open()
            if self._description is None or self._graph_manager is None:
                raise RuntimeError("Container does not own graph resources")
            requested = any(
                variant.cuda_graph
                for variant in self._description.execution_variants
            )
            if not requested:
                return
            if self._graph_registered:
                return
            native = getattr(
                self._exec_unit.runner, "cudagraph_manager", None
            )
            if native is None:
                raise RuntimeError(
                    f"ExecUnit {self.model_id!r} requested CUDA Graph "
                    "resources, but its Runner has not created the native "
                    "graph manager before compile"
                )
            self._graph_manager.register_model(self.model_id, native)
            self._graph_registered = True

    @property
    def exec_unit(self) -> Any:
        return self._exec_unit

    @property
    def model_id(self) -> str:
        return self._exec_unit.model_id

    @property
    def resource_description(self) -> ResourceDescription | None:
        return self._description

    @property
    def scheduler_managed(self) -> bool:
        """是否由继承自 GPUWorker 的原生 scheduler 路径推进。"""
        return self._scheduler_managed

    @property
    def runtime_added(self) -> bool:
        """是否在 Worker 启动完成后动态加入。"""
        return self._runtime_added


    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """在 Container 资源环境内加载该模型。"""
        with self.activate(require_graph=False):
            self._exec_unit.load_model(
                load_dummy_weights=load_dummy_weights
            )

    def determine_available_memory(self) -> int:
        """在 Container 资源环境内执行模型显存测量。"""
        with self.activate(require_graph=False):
            return self._exec_unit.determine_available_memory()

    def initialize_runtime(
        self,
        runtime_config: Any | None = None,
    ) -> None:
        """初始化模型执行前所需的后端运行时状态。"""
        with self.activate(require_graph=False):
            self._exec_unit.initialize_runtime(runtime_config)

    def compile_or_warm_up_model(self) -> Any:
        """在 Container 资源环境内完成编译、warmup 与图捕获。"""
        with self._lock:
            self._require_graph = False
        try:
            with self.activate(require_graph=False):
                return self._exec_unit.compile_or_warm_up_model()
        finally:
            with self._lock:
                self._require_graph = True

    def recv_job(
        self,
        work: Any,
        *,
        request_id: str | None = None,
    ) -> None:
        self._exec_unit.recv_job(work, request_id=request_id)

    def capture_graph_variant(
        self,
        variant_name: str,
        capture_native: Any,
        *,
        buffer_bundle: Any | None = None,
        **kwargs: Any,
    ) -> int:
        """通过统一 Graph Manager 捕获并接入一个声明过的资源变体。"""
        with self._lock:
            self._ensure_open()
            if self._description is None or self._graph_manager is None:
                raise RuntimeError("Container does not own graph resources")
            variant = self._description.variant(variant_name)
            if not variant.cuda_graph:
                raise ValueError(
                    f"Execution variant {variant_name!r} does not request "
                    "CUDA Graph"
                )
            assignment = self._assignments[variant.name]
            key = GraphVariantKey(self.model_id, variant.name)
            captured = self._graph_manager.capture_variant(
                key,
                assignment.streams,
                capture_native,
                buffer_bundle=buffer_bundle,
                **kwargs,
            )
            self._assignments[variant.name] = replace(
                assignment,
                graph=self._graph_manager.binding(key),
            )
            return captured

    def assignment(
        self,
        variant_name: str | None = None,
        *,
        require_graph: bool = True,
    ) -> ResourceAssignment:
        """返回指定执行变体的已解析资源组合。"""
        with self._lock:
            self._ensure_open()
            if self._description is None:
                raise RuntimeError(
                    "Legacy Container requires an explicit assignment"
                )
            variant = self._description.variant(variant_name)
            assignment = self._assignments[variant.name]
            if require_graph and variant.cuda_graph and assignment.graph is None:
                raise RuntimeError(
                    f"CUDA Graph variant {variant.name!r} has not been "
                    "captured"
                )
            return assignment

    def execute_slice(
        self,
        assignment: ResourceAssignment | None = None,
        *,
        descriptor: ExecutionSliceDescriptor | None = None,
        variant_name: str | None = None,
        work: Any | None = None,
        request_id: str | None = None,
        execution_epoch: int | None = None,
    ) -> Any:
        """在选定资源环境中推进 ExecUnit 一次。"""
        with self._lock:
            self._ensure_open()
            if work is not None:
                self._exec_unit.recv_job(work, request_id=request_id)
            selected_descriptor = descriptor or self._exec_unit.next_slice_spec(
                request_id
            )
            selected_name = variant_name
            if assignment is None and selected_name == "auto":
                raise RuntimeError(
                    "Automatic resource selection requires an EngineCore "
                    "ScheduledExecutionSlice"
                )
            selected = assignment or self.assignment(
                selected_name,
                require_graph=self._require_graph,
            )
            name = "default" if selected_name is None else selected_name
            resource_transition = selected is not self._active_assignment
            timeline_span = None
            try:
                with self._exec_unit.activate(
                    selected, resource_transition=resource_transition
                ):
                    execution_stream = selected.streams.stream
                    with self._timeline.slice(
                        execution_stream,
                        dependent_streams=self._dependent_streams(),
                        independent_tail_key=self._independent_tail_key(name),
                    ) as timeline_span: # 统一使用timeline.slice
                        result = self._exec_unit.execute_slice(
                            selected_descriptor,
                            request_id=request_id,
                        )
            except Exception as exc:
                self._active_assignment = selected
                self._variant_step_counts[name] += 1
                if timeline_span is not None:
                    self._record_slice(
                        selected_descriptor,
                        name,
                        request_id,
                        execution_epoch,
                        timeline_span,
                        None,
                        error=exc,
                    )
                raise
            self._active_assignment = selected
            self._variant_step_counts[name] += 1
            self._record_slice(
                selected_descriptor,
                name,
                request_id,
                execution_epoch,
                timeline_span,
                result,
            )
            return result

    def run_on_active_assignment(self, operation: Callable[[], Any]) -> Any:
        """在最近一次 slice 的资源环境中继续执行其后半段。（继续调用，不切换资源）

        vLLM 将一个 scheduler batch 拆成 ``execute_model`` 与
        ``sample_tokens`` 两次 Worker 调用。后者仍属于同一个执行 slice，
        必须复用前者选择的 stream/graph 环境；否则 sampling 与 AsyncOutput
        handoff 会退回调用线程的默认 stream，并越过资源变体切换边界。
        """
        with self._lock:
            self._ensure_open()
            selected = self._active_assignment
            if selected is None:
                raise RuntimeError(
                    f"Container {self.model_id!r} has no active resource "
                    "assignment"
                )
            with self._exec_unit.activate(
                selected, resource_transition=False
            ):
                execution_stream = selected.streams.stream
                with self._timeline.slice(
                    execution_stream,
                    dependent_streams=self._dependent_streams(),
                    independent_tail_key=self._independent_tail_key(
                        self._active_variant_name()
                    ),
                ) as continuation:
                    output = operation()
            # sampling 是上一 scheduler slice 的后半段，不另造一条记录；把
            # 记录的 GPU 结束 event 延伸到 sampling/output-copy 真正完成处。
            if self._slice_records:
                record = self._slice_records[-1]
                record.timeline.end_event = continuation.end_event
                record.timeline.host_end_seconds = (
                    continuation.host_end_seconds
                )
            return output

    def _dependent_streams(self) -> tuple[Any, ...]:
        """返回 Runner 在 slice 内使用的辅助 GPU stream。"""
        runner = getattr(self._exec_unit, "runner", None)
        if runner is None:
            return ()
        return tuple(
            stream
            for name in ("output_copy_stream", "async_output_copy_stream")
            if (stream := getattr(runner, name, None)) is not None
        )

    def _active_variant_name(self) -> str:
        selected = self._active_assignment
        for name, assignment in self._assignments.items():
            if assignment is selected:
                return name
        return "default"

    def _independent_tail_key(self, variant_name: str) -> str | None:
        description = self._description
        if (
            description is None
            or description.graph_transition_sync_mode != "variant_output"
        ):
            return None
        return variant_name

    def variant_step_counts(self) -> dict[str, int]:
        """返回该模型各资源变体的实际执行次数。"""
        with self._lock:
            return dict(self._variant_step_counts)

    def slice_execution_records(self) -> tuple[dict[str, Any], ...]:
        """返回 I6 观测记录；尚未完成的 GPU 区间不会触发 host 同步。"""
        with self._lock:
            return tuple(record.to_payload() for record in self._slice_records)

    @property
    def last_gpu_completion_event(self) -> Any | None:
        return self._timeline.last_done_event

    def partition_group(self, variant_name: str) -> str | None:
        if self._description is None:
            return None
        return self._description.variant(variant_name).stream.partition_group

    def participates_in_partition(self, partition_group: str) -> bool:
        return self._description is not None and any(
            variant.stream.partition_group == partition_group
            for variant in self._description.execution_variants
        )

    def _record_slice(
        self,
        descriptor: ExecutionSliceDescriptor,
        variant_name: str,
        request_id: str | None,
        execution_epoch: int | None,
        timeline_span: Any,
        result: Any,
        *,
        error: Exception | None = None,
    ) -> None:
        # profiling/warmup 以及少量兼容测试可能直接传入不带 slice 契约的
        # 占位对象。它们仍要经过 GPU timeline，但没有可供 I6 记录的模型
        # 语义，因此使用空 metadata/tag 和本地回退序号。
        metadata = getattr(descriptor, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        request_ids = tuple(
            str(value) for value in metadata.get("request_ids", ())
        )
        if not request_ids and request_id is not None:
            request_ids = (str(request_id),)
        tags = frozenset(getattr(descriptor, "tags", ()))
        operator_tags = sorted(
            tag for tag in tags if tag.startswith("component:")
        )
        operator_class = operator_tags[0] if operator_tags else (
            sorted(tags)[0] if tags else "unspecified"
        )
        actual_sm_count = None
        if self._description is not None:
            actual_sm_count = self._description.variant(
                None if variant_name == "default" else variant_name
            ).stream.sm_count
        if actual_sm_count is None:
            try:
                import torch

                stream = self._active_assignment.streams.stream
                actual_sm_count = int(
                    torch.cuda.get_device_properties(stream.device)
                    .multi_processor_count
                )
            except (AttributeError, RuntimeError):
                # CPU/fake-stream 单元测试没有物理 SM；字段保留为 None。
                pass
        predicted = metadata.get("predicted_seconds")
        if error is not None:
            result_summary = {
                "request_done": True,
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
            }
        elif result is None:
            # 原生 vLLM 的 profiling、warmup、非末级 PP，以及部分仅提交
            # GPU 工作的 execute_model 调用都可以合法返回 None。I6 仍记录
            # 这次 GPU slice，但不能把“没有 host 输出”解释成请求完成。
            result_summary = {
                "request_done": False,
                "status": "NO_HOST_OUTPUT",
            }
        else:
            if isinstance(result, dict):
                request_done = bool(result.get("request_done", False))
            else:
                request_done = bool(
                    getattr(result, "request_done", False)
                )
            result_summary = {
                "request_done": request_done,
            }
            if not hasattr(result, "request_done") and not (
                isinstance(result, dict) and "request_done" in result
            ):
                result_summary["status"] = "FRAMEWORK_OUTPUT"
        result_metadata = (
            result.get("metadata", {})
            if isinstance(result, dict)
            else getattr(result, "metadata", {}) if result is not None else {}
        )
        if isinstance(result_metadata, Mapping):
            for key in ("status", "finished_request_ids", "error"):
                if key in result_metadata:
                    result_summary[key] = result_metadata[key]
        self._slice_records.append(SliceExecutionRecord(
            model_id=self.model_id,
            request_ids=request_ids,
            operator_class=operator_class,
            resource_variant=variant_name,
            actual_sm_count=actual_sm_count,
            predicted_seconds=(
                None if predicted is None else float(predicted)
            ),
            execution_epoch=(
                None if execution_epoch is None else int(execution_epoch)
            ),
            sequence_no=int(
                getattr(
                    descriptor,
                    "sequence_no",
                    len(self._slice_records),
                )
            ),
            timeline=timeline_span,
            result=result_summary,
        ))

    @staticmethod
    def _result_field(result: Any, name: str) -> Any:
        if isinstance(result, dict):
            return result[name]
        return getattr(result, name)

    def run_to_completion(
        self,
        work: Any,
        *,
        variant_name: str | None = "auto",
        max_steps: int = 4096,
        result_metadata: Mapping[str, Any] | None = None,
        slice_provider: Callable[[ExecutionSliceDescriptor], Any] | None = None,
    ) -> dict[str, Any]:
        """连续推进一个已接收请求，直到 ExecUnit 报告完成。"""
        started = time.perf_counter()
        request_id = self._work_request_id(work)
        self.recv_job(work, request_id=request_id)
        for step_count in range(1, max_steps + 1):
            if slice_provider is not None:
                spec = self._exec_unit.next_slice_spec(request_id)
                scheduled = slice_provider(spec)
                selected_name = str(scheduled.resource_variant)
            else:
                selected_name = variant_name
            assignment = self.assignment(selected_name)
            result = self.execute_slice(
                assignment=assignment,
                descriptor=spec,
                variant_name=selected_name,
                request_id=request_id,
            )
            if bool(self._result_field(result, "request_done")):
                import torch

                completion = torch.cuda.Event()
                completion.record(assignment.streams.stream)
                completion.synchronize()
                return {
                    **dict(result_metadata or {}),
                    "model_id": self.model_id,
                    "seconds": time.perf_counter() - started,
                    "step_count": step_count,
                    "last_sequence_no": self._result_field(
                        result, "sequence_no"
                    ),
                }
        raise RuntimeError(
            f"Model request {self.model_id!r} exceeded {max_steps} "
            "execution slices"
        )

    @staticmethod
    def _work_request_id(work: Any) -> str | None:
        if isinstance(work, Mapping):
            value = work.get("request_id")
        else:
            value = getattr(work, "request_id", None)
        return None if value is None else str(value)

    @contextmanager
    def activate(
        self,
        variant_name: str | None = None,
        *,
        require_graph: bool = True,
    ):
        """激活一种资源环境，但把具体 Worker 操作留给调用者。

        原生 vLLM Worker 的加载、KV cache 初始化、编译和执行包含大量
        Runner 之外的协议逻辑，不能简单替换为 ``ExecUnit.execute_slice()``。这个
        作用域让新 Worker 保留原生流程，同时确保底层 Runner 使用 Container
        已取得的通信组、显存、stream 和 graph 变体。
        """
        with self._lock:
            self._ensure_open()
            selected = self.assignment(
                variant_name, require_graph=require_graph
            )
        with self._exec_unit.activate(selected):
            try:
                yield selected
            finally:
                # 生命周期操作可能改变 Runner/stream 内部状态；下一次 step
                # 必须重新建立一次完整资源交接。
                self._active_assignment = None

    def shutdown(self) -> None:
        """释放 Container 从各 Manager 取得的模型生命周期资源。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._exec_unit, "close", None)
            if close is not None:
                close()
            # ExecUnit 不负责资源生命周期；Container 销毁后也不能继续暴露
            # 一个已经关闭的 Runner 引用。
            self._exec_unit._model_runner = None
            if self._description is None:
                return
            owner = self.model_id
            if self._graph_registered and self._graph_manager is not None:
                self._graph_manager.release_model(owner)
            assert self._stream_manager is not None
            assert self._memory_manager is not None
            assert self._comm_manager is not None
            self._stream_manager.release(owner)
            self._memory_manager.release(owner)
            self._comm_manager.release(owner)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"MiniContainer {self.model_id!r} is closed")


class MiniContainerRegistry:
    """管理一个 Worker 内全部 MiniContainer 的唯一所有权。"""

    def __init__(self) -> None:
        self._containers: dict[str, MiniContainer] = {}
        self._lock = threading.RLock()

    def register(self, container: MiniContainer) -> None:
        with self._lock:
            if container.model_id in self._containers:
                raise ValueError(
                    f"MiniContainer {container.model_id!r} already exists"
                )
            self._containers[container.model_id] = container

    def get(self, model_id: str) -> MiniContainer:
        with self._lock:
            try:
                return self._containers[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown MiniContainer {model_id!r}") from exc

    def all(self) -> tuple[MiniContainer, ...]:
        with self._lock:
            return tuple(self._containers.values())

    def scheduler_managed(self) -> tuple[MiniContainer, ...]:
        """返回由原生 GPUWorker scheduler 驱动的 Container。"""
        with self._lock:
            return tuple(
                container
                for container in self._containers.values()
                if container.scheduler_managed
            )

    def require_scheduler_managed(self) -> MiniContainer:
        """返回当前兼容路径唯一的 scheduler-managed Container。"""
        containers = self.scheduler_managed()
        if len(containers) != 1:
            raise RuntimeError(
                "The native GPUWorker compatibility path requires exactly "
                f"one scheduler-managed MiniContainer, got {len(containers)}"
            )
        return containers[0]

    def remove(self, model_id: str) -> MiniContainer:
        with self._lock:
            container = self._containers.pop(model_id)
        container.shutdown()
        return container

    def shutdown(self) -> None:
        with self._lock:
            containers = tuple(reversed(tuple(self._containers.values())))
            self._containers.clear()
        for container in containers:
            container.shutdown()
