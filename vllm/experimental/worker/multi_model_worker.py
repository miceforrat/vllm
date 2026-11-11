# SPDX-License-Identifier: Apache-2.0
"""基于 MiniContainer 的通用多模型 GPU Worker。"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from vllm.experimental.deployment import EngineDeploymentPlan
from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.worker.exec_unit import LLMExecUnit, OmniModelExecUnit
from vllm.experimental.worker.mini_container import (
    MiniContainer,
    MiniContainerRegistry,
)
from vllm.experimental.worker.model_lifecycle import NativeGPUWorkerLifecycle
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
    MemoryManager,
)
from vllm.v1.core.sched.execution_units import ScheduledExecutionSlice
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class MultiModelWorker(GPUWorker):
    """统一拥有 ExecUnit、Container 与物理资源的 GPU Worker。

    第一阶段只把 Engine 启动模型接入新架构。加载权重、显存 profiling、
    KV cache、编译和执行协议仍复用原生 ``GPUWorker``，避免复制或分叉
    vLLM 的成熟生命周期。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 初始化各种基本管理单元
        self._container_registry = MiniContainerRegistry()
        self._comm_manager = CommManager()
        self._memory_manager = MemoryManager()
        self._stream_manager: GCStreamManager | None = None
        self._graph_manager = GCCudaGraphManager()
        self._model_jobs: dict[str, dict[str, Any]] = {}
        self._model_jobs_lock = threading.Lock()
        self._last_execution_epoch = 0

    @property
    def container_registry(self) -> MiniContainerRegistry:
        """暴露注册表"""
        return self._container_registry

    @property
    def scheduler_container(self) -> MiniContainer:
        """返回接入原生 GPUWorker scheduler 的兼容 Container。"""
        return self._container_registry.require_scheduler_managed()

    def _multi_model_runtime_config(self) -> dict[str, Any]:
        additional = getattr(self.vllm_config, "additional_config", None)
        return (
            additional.get("multi_model_runtime", {})
            if isinstance(additional, dict)
            else {}
        )

    def _planned_model_specs(self) -> tuple[ModelSpec, ...]:
        """读取 Engine 在 Worker 启动前规划的全部模型描述。"""
        runtime = self._multi_model_runtime_config()
        plan_payload = runtime.get("deployment_plan")
        if plan_payload is not None:
            plan = EngineDeploymentPlan.from_rpc_payload(plan_payload)
            specs = tuple(
                deployment.spec for deployment in plan.deployments
            )
        else:
            payloads = runtime.get("startup_models", ())
            specs = tuple(
                ModelSpec.from_rpc_payload(payload) for payload in payloads
            )
        rank = getattr(self, "rank", None)
        if rank is not None:
            specs = tuple(
                spec
                for spec in specs
                if not spec.worker_ranks or rank in spec.worker_ranks
            )
        if not specs:
            raise RuntimeError(
                "MultiModelWorker requires at least one Engine-planned "
                "startup model"
            )
        model_ids = [spec.model_id for spec in specs]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Engine startup model ids must be unique")
        primary = [spec for spec in specs if spec.backend == "vllm"]
        if len(primary) != 1:
            raise RuntimeError(
                "MultiModelWorker requires exactly one scheduler-managed "
                f"vllm startup model, got {len(primary)}"
            )
        unsupported = [
            spec.backend
            for spec in specs
            if spec.backend not in ("vllm", "vllm_omni")
        ]
        if unsupported:
            raise ValueError(f"Unsupported startup model backends: {unsupported!r}")
        return specs

    @staticmethod
    def _default_resource_description(spec: ModelSpec) -> ResourceDescription:
        """为尚未显式规划资源的原生 vLLM 模型建立兼容描述。

        通信拓扑已经由 Engine 写入 ``ModelSpec``。stream 与 memory 的完整
        规划尚未上移到 Engine，因此这里暂用默认 stream 与默认 allocator；
        该兼容入口之后应由 Engine 序列化的 ``ResourceDescription`` 替换。
        """
        return ResourceDescription(
            communication_groups=tuple(
                CommunicationGroupDescription.from_rpc_payload(group)
                for group in spec.communication_groups
            ),
            memory=MemoryResourceDescription(kind="default"),
            execution_variants=(
                ExecutionVariantDescription(
                    name="default",
                    stream=StreamResourceDescription(kind="default"),
                ),
            ),
            default_variant="default",
        )

    def _planned_resource_descriptions(
        self, specs: tuple[ModelSpec, ...]
    ) -> dict[str, ResourceDescription]:
        """读取并校验 Engine 规划的逐模型资源描述。"""
        runtime = self._multi_model_runtime_config()
        plan_payload = runtime.get("deployment_plan")
        if plan_payload is not None:
            plan = EngineDeploymentPlan.from_rpc_payload(plan_payload)
            payloads = {
                deployment.spec.model_id: deployment.resources
                for deployment in plan.deployments
            }
        else:
            payloads = runtime.get("startup_resources", {})
        descriptions: dict[str, ResourceDescription] = {}
        for spec in specs:
            payload = payloads.get(spec.model_id)
            if payload is None:
                if spec.backend != "vllm":
                    raise RuntimeError(
                        "A non-vLLM startup model requires an Engine-planned "
                        f"ResourceDescription: {spec.model_id!r}"
                    )
                description = self._default_resource_description(spec)
            else:
                description = ResourceDescription.from_rpc_payload(payload)
            planned_groups = tuple(
                CommunicationGroupDescription.from_rpc_payload(group)
                for group in spec.communication_groups
            )
            if description.communication_groups != planned_groups:
                raise ValueError(
                    f"Startup model {spec.model_id!r} has inconsistent "
                    "communication topology"
                )
            descriptions[spec.model_id] = description
        return descriptions

    def _create_model_runner(self):
        """原子创建全部启动 ExecUnit、stream 资源和 Container。
        GPUWorker本体会调用这个方法的,所以是方法override"""
        device_index = self.device.index
        if device_index is None:
            raise RuntimeError("CUDA device has no logical index")
        self._stream_manager = GCStreamManager(device_index)

        specs = self._planned_model_specs()
        descriptions = self._planned_resource_descriptions(specs)
        units = {
            spec.model_id: (
                LLMExecUnit(spec, self.vllm_config, self.device)
                if spec.backend == "vllm"
                else OmniModelExecUnit(spec, self.vllm_config, self.device)
            )
            for spec in specs
        }
        stream_bindings = self._stream_manager.acquire_many(
            {
                model_id: description.execution_variants
                for model_id, description in descriptions.items()
            }
        )

        created: list[MiniContainer] = []
        try:
            for spec in specs:
                container = MiniContainer.acquire(
                    units[spec.model_id],
                    descriptions[spec.model_id],
                    comm_manager=self._comm_manager,
                    memory_manager=self._memory_manager,
                    stream_manager=self._stream_manager,
                    graph_manager=self._graph_manager,
                    stream_bindings=stream_bindings[spec.model_id],
                    scheduler_managed=spec.backend == "vllm",
                    runtime_added=False,
                )
                try:
                    if spec.backend != "vllm":
                        container.load_model()
                    self._container_registry.register(container)
                except Exception:
                    container.shutdown()
                    raise
                created.append(container)
        except Exception:
            # 已进入 Container 的 owner 由 Registry 统一回滚；尚未进入
            # Container 的预创建 stream 需要在这里单独释放。
            self._container_registry.shutdown()
            created_ids = {container.model_id for container in created}
            for model_id in stream_bindings:
                if model_id not in created_ids:
                    try:
                        self._stream_manager.release(model_id)
                    except KeyError:
                        pass
            raise

        scheduler_container = self.scheduler_container
        scheduler_container.attach_lifecycle(
            NativeGPUWorkerLifecycle(
                self, scheduler_container, self._graph_manager
            )
        )
        scheduler_container.exec_unit.bind_step_executor(
            lambda work: GPUWorker.execute_model(self, work)
        )
        return scheduler_container.exec_unit.runner

    def register_model(
        self,
        spec_payload: dict[str, Any],
        resource_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """在当前 rank 追加创建并注册一个运行时模型 Container。主要作用于omni"""
        spec = ModelSpec.from_rpc_payload(spec_payload)
        if spec.worker_ranks and self.rank not in spec.worker_ranks:
            return {
                "model_id": spec.model_id,
                "rank": self.rank,
                "active": False,
            }
        if spec.backend != "vllm_omni":
            raise ValueError(
                "Runtime registration currently supports vllm_omni; the "
                "startup vllm model is registered during init_device"
            )
        description = ResourceDescription.from_rpc_payload(resource_payload)
        planned_groups = tuple(
            CommunicationGroupDescription.from_rpc_payload(group)
            for group in spec.communication_groups
        )
        if description.communication_groups != planned_groups:
            raise ValueError(
                "ModelSpec and ResourceDescription communication groups "
                "must describe the same Engine-planned topology"
            )
        if self._stream_manager is None:
            raise RuntimeError("Worker device resources are not initialized")
        if any(
            variant.stream.kind == "green_context"
            for variant in description.execution_variants
        ):
            raise ValueError(
                "Green Context models must be included in the atomic startup "
                "plan with every member of their partition_group"
            )

        exec_unit = OmniModelExecUnit(spec, self.vllm_config, self.device)
        container = MiniContainer.acquire(
            exec_unit,
            description,
            comm_manager=self._comm_manager,
            memory_manager=self._memory_manager,
            stream_manager=self._stream_manager,
            graph_manager=self._graph_manager,
            scheduler_managed=False,
            runtime_added=True,
        )
        try:
            container.load_model()
            self._container_registry.register(container)
        except Exception:
            container.shutdown()
            raise
        return {
            "model_id": spec.model_id,
            "rank": self.rank,
            "active": True,
        }

    def recv_model_work(self, model_id: str, work: Any) -> dict[str, Any]:
        """把新请求交给指定的长期 ExecUnit。也就是说, 必须要已有模型才能转交"""
        container = self._container_registry.get(model_id)
        container.recv_job(work)
        return {
            "model_id": model_id,
            "rank": self.rank,
            "pending_work": container.exec_unit.pending_work_count,
        }

    def step_model(
        self, model_id: str, variant_name: str | None = None
    ) -> Any:
        """在指定资源变体中推进模型一个执行单元。"""
        return self._container_registry.get(model_id).step(
            variant_name=variant_name
        )

    def execute_slice(
        self,
        scheduled: ScheduledExecutionSlice | dict[str, Any],
        *,
        work: Any | None = None,
    ) -> Any:
        """执行 EngineCore 已规划的一个 slice。"""
        decision = ScheduledExecutionSlice.from_payload(scheduled)
        # 不同模型可由不同线程消费 slice，因此本地到达顺序不保证等于
        # EngineCore 的全局决策顺序；epoch 用于观测，而非本地拒绝执行。
        self._last_execution_epoch = max(
            self._last_execution_epoch, decision.execution_epoch
        )
        container = self._container_registry.get(decision.model_id)
        return container.step(
            work=work,
            variant_name=decision.resource_variant,
        )

    def start_model_request(
        self,
        model_id: str,
        work: Any,
        variant_name: str | None = None,
        max_steps: int = 4096,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """接收请求并向 EngineCore 报告第一个可执行边界。"""
        job_id = uuid.uuid4().hex if job_id is None else str(job_id)
        if not job_id:
            raise ValueError("Model request job_id cannot be empty")
        with self._model_jobs_lock:
            if job_id in self._model_jobs:
                raise ValueError(f"Model job {job_id!r} already exists")
        container = self._container_registry.get(model_id)
        container.recv_job(work)
        kind, index = container.exec_unit.next_slice_spec()
        with self._model_jobs_lock:
            self._model_jobs[job_id] = {
                "model_id": model_id,
                "variant_name": variant_name,
                "max_steps": int(max_steps),
                "step_count": 0,
                "started": time.perf_counter(),
            }
        return {
            "job_id": job_id,
            "model_id": model_id,
            "rank": self.rank,
            "next_slice": {"kind": kind, "index": index},
        }

    def execute_model_request_slice(
        self,
        job_id: str,
        scheduled: ScheduledExecutionSlice | dict[str, Any],
    ) -> dict[str, Any]:
        """执行 EngineCore 下发的一个 slice，并报告下一个边界。"""
        with self._model_jobs_lock:
            try:
                state = self._model_jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model job {job_id!r}") from exc
            state["step_count"] += 1
            if state["step_count"] > state["max_steps"]:
                raise RuntimeError(
                    f"Model request {job_id!r} exceeded "
                    f"{state['max_steps']} execution units"
                )

        decision = ScheduledExecutionSlice.from_payload(scheduled)
        if decision.request_id != job_id:
            raise ValueError(
                f"Scheduled request {decision.request_id!r} does not match "
                f"Worker job {job_id!r}"
            )
        container = self._container_registry.get(state["model_id"])
        result = self.execute_slice(decision)
        request_done = bool(container._result_field(result, "request_done"))
        response = {
            "job_id": job_id,
            "model_id": state["model_id"],
            "rank": self.rank,
            "request_done": request_done,
            "kind": container._result_field(result, "kind"),
            "index": container._result_field(result, "index"),
            "step_count": state["step_count"],
        }
        if request_done:
            assignment = container.assignment(decision.resource_variant)
            import torch

            completion = torch.cuda.Event()
            completion.record(assignment.streams.stream)
            completion.synchronize()
            response["seconds"] = time.perf_counter() - state["started"]
            with self._model_jobs_lock:
                self._model_jobs.pop(job_id, None)
        else:
            kind, index = container.exec_unit.next_slice_spec()
            response["next_slice"] = {"kind": kind, "index": index}
        return response

    def unregister_model(self, model_id: str) -> dict[str, Any]:
        """销毁运行时模型，并释放其 Container 资源。"""
        container = self._container_registry.get(model_id)
        if not container.runtime_added:
            raise ValueError(
                "An Engine-owned startup model follows the Worker lifecycle "
                "and cannot be unregistered at runtime"
            )
        self._container_registry.remove(model_id)
        return {"model_id": model_id, "rank": self.rank, "active": False}

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.scheduler_container.load_model(
            load_dummy_weights=load_dummy_weights
        )

    def determine_available_memory(self) -> int:
        return self.scheduler_container.determine_available_memory()

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        self.scheduler_container.initialize_from_config(kv_cache_config)

    def compile_or_warm_up_model(self):
        return self.scheduler_container.compile_or_warm_up_model()

    def model_execution_status(
        self, model_id: str | None = None
    ) -> dict[str, Any]:
        """返回模型资源变体及 Worker-local UnitScheduler 状态。"""
        container = (
            self.scheduler_container
            if model_id is None
            else self._container_registry.get(model_id)
        )
        active = self._graph_manager.active_variant(container.model_id)
        variant_step_counts = {
            f"{item.model_id}:{variant}": count
            for item in self._container_registry.all()
            for variant, count in item.variant_step_counts().items()
        }
        return {
            "rank": self.rank,
            "model_id": container.model_id,
            "active_graph_variant": (
                None if active is None else active.variant_name
            ),
            "capture_counts": self._graph_manager.counts(
                container.model_id
            ),
            "variant_step_counts": variant_step_counts,
        }

    def execute_model(self, scheduler_output: Any):
        """把原生 SchedulerOutput 直接交给对应 Container 推进一步。"""
        scheduled = getattr(
            scheduler_output, "scheduled_execution_slice", None
        )
        if scheduled is not None:
            return self.execute_slice(scheduled, work=scheduler_output)
        # GPUWorker 在 profiling、warmup 和 CUDA Graph capture 期间会直接
        # 构造 SchedulerOutput；这些调用不经过 EngineCore，因而没有 slice
        # 决策。使用部署计划中的默认变体，不能把 ``auto`` 留给 Worker 推断。
        return self.scheduler_container.step(
            work=scheduler_output,
            variant_name=None,
        )

    def _shutdown_model_runner(self) -> None:
        """先销毁 Runner，再释放其 Container 资源引用。"""
        jobs_lock = getattr(self, "_model_jobs_lock", None)
        if jobs_lock is not None:
            with jobs_lock:
                self._model_jobs.clear()
        self._container_registry.shutdown()
        self.model_runner = None
        self._graph_manager.clear()
        
