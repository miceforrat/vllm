# SPDX-License-Identifier: Apache-2.0
"""Worker 本地、长期存活的模型执行单元。

:class:`ExecUnit` 对应一个已经实例化的模型，而不是单个 scheduler batch
或 diffusion step。同一个对象可以反复接收工作，并通过 :meth:`step`
逐步推进这些工作。

具体 ExecUnit 负责创建对应模型族的 Runner。
外部只需提供 ``ModelSpec``、vLLM 运行时配置和目标设备；这样 vLLM 与
vLLM-Omni 的 ExecUnit 可以尽量保持相同的构造参数。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.worker.binding import ResourceAssignment
from vllm.experimental.execution_slice import (
    ExecutionSliceDescriptor,
    ExecutionSliceResult,
)
from vllm.experimental.worker.resource_manager import GraphVariantKey
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig

_WorkT = TypeVar("_WorkT")
_ResultT = TypeVar("_ResultT")


# 兼容旧测试和外部导入；新代码应使用模型无关的 ExecutionSliceResult。
OmniStepResult = ExecutionSliceResult


@dataclass(frozen=True)
class _ResolvedOmniStage:
    """OmniModelExecUnit 创建 Runner 所需的配置视图。"""

    pipeline_config: Any
    stage_config: Any
    diffusion_config: Any


class ExecUnit(ABC, Generic[_WorkT, _ResultT]):
    """一个长期存活、可持续接收工作的模型执行实例。

    ExecUnit 负责模型族相关的 Runner 创建和执行语义。物理资源的
    所有权归 MiniContainer 和资源管理器。
    """

    def __init__(
        self,
        spec: ModelSpec,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.spec: ModelSpec = spec
        runner = self._create_model_runner(vllm_config, device)
        if runner is None:
            raise ValueError("An ExecUnit requires a materialized Runner")
        self._model_runner = runner

    @abstractmethod
    def _create_model_runner(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """创建该模型族对应的 Runner 和 Worker 本地模型实例。"""

    @property
    def runner(self):
        return self._model_runner

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    @abstractmethod
    def pending_work_count(self) -> int:
        """已经接收、但尚未被 ``execute_slice`` 推进的工作数量。"""

    @abstractmethod
    def recv_job(
        self,
        work: _WorkT,
        *,
        request_id: str | None = None,
    ) -> None:
        """为这个长期模型实例接收一项新工作。从而可以步进推进job"""

    @abstractmethod
    def execute_slice(
        self,
        descriptor: ExecutionSliceDescriptor,
        *,
        request_id: str | None = None,
    ) -> _ResultT:
        """推进一项已接收的工作，并返回 Runner 的执行结果。一般就是推进当前工作"""

    def next_slice_spec(
        self, request_id: str | None = None
    ) -> ExecutionSliceDescriptor:
        """返回下一次 ``execute_slice`` 的模型无关 slice 描述。"""
        del request_id
        return ExecutionSliceDescriptor(0)

    def abort_requests(self, request_ids: tuple[str, ...]) -> tuple[str, ...]:
        """终止由该模型实例持有的请求，并返回实际清理的请求 ID。"""
        raise NotImplementedError(
            f"ExecUnit {self.model_id!r} does not support request abort: "
            f"{request_ids!r}"
        )

    def bind_resources(
        self,
        assignment: ResourceAssignment,
        *,
        load_format: str = "default",
    ) -> None:
        """接入模型生命周期级资源。"""

    @contextmanager
    def activate(
        self,
        assignment: ResourceAssignment,
        *,
        resource_transition: bool = True,
    ):
        """临时激活一次执行使用的资源环境。"""
        yield

    @abstractmethod
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """加载模型"""

    @abstractmethod
    def determine_available_memory(self) -> int:
        """决定可用的显存数"""

    @abstractmethod
    def initialize_runtime(self, runtime_config: Any | None = None) -> None:
        """初始化模型执行前所需的后端运行时状态。"""

    @abstractmethod
    def compile_or_warm_up_model(self) -> Any:
        """预编译和warmup模型"""

    def close(self) -> None:
        """关闭execUnit"""

class LLMExecUnit(ExecUnit[Any, Any]):
    """面向原生 vLLM Model Runner 的 ExecUnit 实现。

    每项工作对应一个 vLLM ``SchedulerOutput``。基类接口不绑定这个具体类型，
    因而未来 Diffusion ExecUnit 可以使用不同的工作表示，而不必改变统一的
    构造方式和生命周期接口。
    """

    def __init__(
        self,
        spec: ModelSpec,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(spec, vllm_config, device)
        self._pending_work: deque[Any] = deque()
        self._work_lock = threading.Lock()
        # Runner 内部可能保存可变的 input batch 和 CUDA Graph 状态。多个调用方
        # 可以并发提交工作，但同一时刻只允许一个调用进入 Runner。
        self._step_lock = threading.Lock()
        self._step_executor: Callable[[Any], Any] | None = None
        # 原生 GPUWorker 生命周期依赖。Runner 必须先于 MiniContainer 创建，
        # 因此这些对象在 Container 建立后通过 attach_native_runtime 接入。
        self._worker: Any | None = None
        self._container: Any | None = None
        self._graph_manager: Any | None = None

    def attach_native_runtime(
        self,
        worker: Any,
        container: Any,
        graph_manager: Any,
    ) -> None:
        """接入原生 GPUWorker 生命周期及 Container Graph 编排依赖。"""
        if self._worker is not None:
            raise RuntimeError(
                f"LLM ExecUnit {self.model_id!r} already has a native runtime"
            )
        if container.exec_unit is not self:
            raise ValueError(
                f"Container {container.model_id!r} does not own ExecUnit "
                f"{self.model_id!r}"
            )
        self._worker = worker
        self._container = container
        self._graph_manager = graph_manager

    def _native_runtime(self) -> tuple[Any, Any, Any]:
        worker = self._worker
        container = self._container
        graph_manager = self._graph_manager
        if worker is None or container is None or graph_manager is None:
            raise RuntimeError(
                f"LLM ExecUnit {self.model_id!r} has no native runtime"
            )
        return worker, container, graph_manager

    def _create_model_runner(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> Any:
        """复用 GPUWorker 的选择规则创建对应版本的 Model Runner。"""
        if vllm_config.use_v2_model_runner:
            if vllm_config.is_mm_encoder_only:
                from vllm.v1.worker.mm_encoder_model_runner import (
                    MMEncoderModelRunner as GPUModelRunner,
                )
            else:
                from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        else:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        return GPUModelRunner(vllm_config, device)

    @property
    def pending_work_count(self) -> int:
        with self._work_lock:
            return len(self._pending_work)

    def recv_job(
        self,
        work: Any,
        *,
        request_id: str | None = None,
    ) -> None:
        del request_id
        if work is None:
            raise ValueError("LLMExecUnit cannot receive an empty work item")
        with self._work_lock:
            self._pending_work.append(work)

    def bind_step_executor(self, executor: Callable[[Any], Any]) -> None:
        """接入完整的 Worker batch 执行协议，只允许绑定一次。"""
        if self._step_executor is not None:
            raise RuntimeError(
                f"ExecUnit {self.model_id!r} already has a step executor"
            )
        self._step_executor = executor

    def execute_slice(
        self,
        descriptor: ExecutionSliceDescriptor,
        *,
        request_id: str | None = None,
    ) -> Any:
        """vLLM有自己的一套调度方式，用不上这个reqId和desc
            vLLM的实际执行描述已经包含在scheduler_output里面
        """
        del descriptor
        del request_id
        with self._step_lock:
            with self._work_lock:
                if not self._pending_work:
                    raise RuntimeError(
                        f"ExecUnit {self.model_id!r} has no runnable work"
                    )
                scheduler_output = self._pending_work.popleft()
            executor = self._step_executor
            if executor is None:
                executor = self.runner.execute_model
            return executor(scheduler_output)

    def bind_resources(
        self,
        assignment: ResourceAssignment,
        *,
        load_format: str = "default",
    ) -> None:
        """向原生 Worker 生命周期暴露模型级显存 Binding。"""
        del load_format
        self.runner._vllm_memory_binding = assignment.memory

    @contextmanager
    def activate(
        self,
        assignment: ResourceAssignment,
        *,
        resource_transition: bool = True,
    ):
        if assignment.graph is not None:
            assignment.graph.validate_stream(assignment.streams)
        with ExitStack() as stack:
            # stack用来动态组合多个 context manager，并保证按相反顺序退出。
            stack.enter_context(assignment.communicator.activate())
            stack.enter_context(assignment.memory.activate())
            stack.enter_context(
                assignment.streams.activate(
                    self.runner,
                    synchronize_transition=resource_transition,
                )
            )
            if assignment.graph is not None and resource_transition:
                # 如果decode开启graph执行，还需要额外切换graph环境
                assignment.graph.activate(self.runner)
            yield

    def close(self) -> None:
        runner, self._model_runner = self._model_runner, None
        if runner is not None:
            shutdown = getattr(runner, "shutdown", None)
            if shutdown is not None:
                shutdown()

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        worker, _, _ = self._native_runtime()
        GPUWorker.load_model(
            worker, load_dummy_weights=load_dummy_weights
        )

    def determine_available_memory(self) -> int:
        worker, _, _ = self._native_runtime()
        return GPUWorker.determine_available_memory(worker)

    def initialize_runtime(self, runtime_config: Any | None = None) -> None:
        """使用 Engine 规划的配置初始化 vLLM KV cache。"""
        if runtime_config is None:
            raise ValueError(
                f"LLM ExecUnit {self.model_id!r} requires a KV-cache "
                "runtime configuration"
            )
        worker, _, _ = self._native_runtime()
        GPUWorker.initialize_from_config(worker, runtime_config)

    def compile_or_warm_up_model(self) -> Any:
        worker, container, graph_manager = self._native_runtime()
        description = container.resource_description
        graph_variants = tuple(
            variant
            for variant in description.execution_variants
            if variant.cuda_graph
        )
        runner = container.exec_unit.runner
        provider_installed = False
        if graph_variants:
            expected = {"shared", "reasoner_full"}
            actual = {variant.name for variant in graph_variants}
            if actual != expected:
                raise ValueError(
                    "The first vLLM dual-graph integration requires exactly "
                    f"{sorted(expected)!r}, got {sorted(actual)!r}"
                )
            if getattr(runner, "graph_variant_capture_provider", None) is not None:
                raise RuntimeError(
                    "GPUModelRunner already has a graph variant capture provider"
                )
            container.register_graph_manager()
            graph_manager.set_capture_handler(
                container.model_id,
                runner,
                self._capture_graph_variants,
            )
            runner.graph_variant_capture_provider = graph_manager
            provider_installed = True
        try:
            return GPUWorker.compile_or_warm_up_model(worker)
        finally:
            if provider_installed:
                runner.graph_variant_capture_provider = None
                graph_manager.clear_capture_handler(container.model_id)

    def _capture_graph_variants(
        self,
        runner: Any,
        capture_shared: Any,
        **kwargs: Any,
    ) -> int:
        _, container, graph_manager = self._native_runtime()

        def capture_with_stream_patch(
            variant_name: str,
            capture: Any,
            **capture_kwargs: Any,
        ) -> int:
            from vllm.experimental.runtime_stream_patch import (
                install_vllm_graph_capture_stream_patch,
            )

            assignment = container.assignment(
                variant_name, require_graph=False
            )
            stream = getattr(assignment.streams, "stream", None)
            if stream is None:
                raise RuntimeError(
                    f"Graph variant {variant_name!r} has no capture stream"
                )
            patch = install_vllm_graph_capture_stream_patch(stream)
            try:
                return container.capture_graph_variant(
                    variant_name, capture, **capture_kwargs
                )
            finally:
                patch.restore()

        total = capture_with_stream_patch("shared", capture_shared, **kwargs)
        total += capture_with_stream_patch(
            "reasoner_full",
            lambda: self._capture_additional_graph(runner),
        )
        default_variant = container.resource_description.default_variant
        if default_variant is None:
            raise RuntimeError("vLLM model has no default graph variant")
        graph_manager.activate(
            runner,
            GraphVariantKey(container.model_id, default_variant),
        )
        return total

    @staticmethod
    def _capture_additional_graph(runner: Any) -> int:
        import torch

        from vllm.v1.worker.gpu.lora_utils import create_lora_capture_hook

        native = runner.cudagraph_manager
        input_buffers = runner.input_buffers
        if runner.pcp_manager is not None:
            input_buffers = runner.pcp_manager.input_buffers
        native.capture(
            runner.model,
            runner.model_state,
            input_buffers,
            runner.intermediate_tensors,
            runner.block_tables,
            runner.attn_groups,
            runner.kv_cache_config,
            pcp_manager=runner.pcp_manager,
            has_lora=runner.lora_config is not None,
            use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
            lora_capture_hook=create_lora_capture_hook(
                runner.lora_config, runner
            ),
            progress_bar_desc="Capturing full-SM vLLM graphs",
        )
        torch.cuda.current_stream().synchronize()
        runner.kv_connector.reset_capture_state()
        return 0

class _SingleRankGroupView:
    """把 vLLM 的单 rank PP group 暴露为 Omni 禁用并行维度。"""

    world_size = 1
    rank_in_group = 0
    ulysses_world_size = 1
    ulysses_rank = 0
    ring_world_size = 1
    ring_rank = 0
    allgather_world_size = 1
    allgather_rank = 0
    is_first_rank = True
    is_last_rank = True

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator

    @property
    def ulysses_group(self) -> Any:
        return self.coordinator.device_group

    @property
    def ring_group(self) -> Any:
        return self.coordinator.device_group

    @property
    def allgather_group(self) -> Any:
        return self.coordinator.device_group

    def __getattr__(self, name: str) -> Any:
        return getattr(self.coordinator, name)

class _OmniParallelGroupProvider:
    """补全 Omni 所需的 WORLD、TP 和禁用并行维度视图。"""

    _SINGLETON_DIMENSIONS = {
        "sp",
        "pp",
        "cfg",
        "dp",
        "fs",
        "hsdp_replicate",
    }

    def __init__(self, binding: Any) -> None:
        from vllm.distributed.parallel_state import get_pp_group

        self._binding = binding
        self._singleton = _SingleRankGroupView(get_pp_group())

    def get(self, parallel_dimension: str) -> Any | None:
        group = self._binding.get(parallel_dimension)
        if group is not None:
            return group
        if parallel_dimension == "world":
            from vllm.distributed.parallel_state import get_world_group

            return get_world_group()
        if parallel_dimension == "tp":
            from vllm.distributed.parallel_state import get_tp_group

            return get_tp_group()
        if parallel_dimension in self._SINGLETON_DIMENSIONS:
            return self._singleton
        return None

    @contextmanager
    def activate(self):
        from vllm_omni.diffusion.distributed.parallel_state import (
            use_parallel_group_provider,
        )

        with self._binding.activate():
            with use_parallel_group_provider(self):
                yield self

    def communication_context(self):
        return self.activate()

class OmniModelExecUnit(ExecUnit[Any, Any]):
    """Omni 模型实例及其 DiffusionModelRunner 适配层。"""

    def __init__(
        self,
        spec: ModelSpec,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self._diffusion_vllm_config: Any | None = None
        self._omni_diffusion_config: Any | None = None
        super().__init__(spec, vllm_config, device)
        self._step_scheduler = self._create_step_scheduler()
        self._step_scheduler.initialize(self._omni_diffusion_config)
        self._pending_scheduler_output: Any | None = None
        self._slice_sequence = 0
        self._work_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._communicator_provider: Any | None = None
        self._base_runtime_providers: Any | None = None

    @staticmethod
    def _create_step_scheduler() -> Any:
        from vllm_omni.diffusion.sched.step_scheduler import StepScheduler

        return StepScheduler()

    def _create_model_runner(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> Any:
        """解析 Omni 配置，并创建尚未加载的 DiffusionModelRunner。"""
        if self.spec.backend != "vllm_omni":
            raise ValueError(
                "OmniModelExecUnit requires the 'vllm_omni' backend, got "
                f"{self.spec.backend!r}"
            )

        resolved = self._resolve_omni_stage()
        return self._create_worker_hosted_runner(
            resolved=resolved,
            device=device,
            host_vllm_config=vllm_config,
        )

    def _resolve_omni_stage(self) -> Any:
        """通过 Omni 公共配置路径解析当前模型的单个 diffusion stage。
            
            复用omni的方法"""
        from vllm_omni.config.omni_config import (
            extract_diffusion_stage_config_kwargs,
        )
        from vllm_omni.config.resolver import resolve_omni_config
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.engine.stage_init_utils import (
            build_engine_args_dict_from_omni_stage_config,
        )

        model = self.spec.model_path
        overrides = dict(self.spec.backend_options)
        overrides.setdefault("model", model)
        resolution = resolve_omni_config(
            model,
            trust_remote_code=False,
            deploy_config_path=None,
            cli_overrides=overrides,
            stage_overrides=None,
            strategy_config_path=None,
        )
        if len(resolution.stage_configs) != 1:
            raise ValueError(
                "OmniModelExecUnit requires exactly one Omni stage; resolved "
                f"{len(resolution.stage_configs)}"
            )

        stage_config = resolution.stage_configs[0]
        stage_type = getattr(
            stage_config.stage_type, "value", stage_config.stage_type
        )

        # 这个模型/pipeline必须是diffusion类型
        if stage_type != "diffusion":
            raise ValueError(
                "OmniModelExecUnit requires a diffusion stage; resolved "
                f"{stage_config.stage_type!r}"
            )

        engine_args = build_engine_args_dict_from_omni_stage_config(
            stage_config,
            model,
        )
        diffusion_kwargs = extract_diffusion_stage_config_kwargs(
            engine_args,
            stage_id=stage_config.stage_id,
            include_engine_adapter_metadata=True,
        )
        diffusion_config = OmniDiffusionConfig(
            **{
                name: value
                for name, value in diffusion_kwargs.items()
                if value is not None
            }
        )
        diffusion_config.enrich_config()
        diffusion_config.num_gpus = diffusion_config.parallel_config.world_size
        return _ResolvedOmniStage(
            pipeline_config=resolution.pipeline_config,
            stage_config=stage_config,
            diffusion_config=diffusion_config,
        )

    def _create_worker_hosted_runner(
        self,
        *,
        resolved: Any,
        device: torch.device,
        host_vllm_config: VllmConfig,
    ) -> Any:
        """按 Engine 规划的并行配置实例化尚未加载的 Omni Runner。"""
        import torch

        from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
        from vllm_omni.diffusion.worker.diffusion_model_runner import (
            DiffusionModelRunner,
        )
        from vllm_omni.platforms import current_omni_platform

        torch.cuda.set_device(device)
        host_parallel = getattr(host_vllm_config, "parallel_config", None)
        if host_parallel is None:
            raise ValueError("Omni ExecUnit requires a planned parallel_config")
        tp_size = int(host_parallel.tensor_parallel_size)
        pp_size = int(host_parallel.pipeline_parallel_size)
        world_size = tp_size * pp_size

        od_config = resolved.diffusion_config
        self._omni_diffusion_config = od_config
        parallel = od_config.parallel_config
        unsupported = {
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "sequence_parallel_size": parallel.sequence_parallel_size,
            "cfg_parallel_size": parallel.cfg_parallel_size,
            "vae_patch_parallel_size": parallel.vae_patch_parallel_size,
        }
        unsupported = {
            name: value for name, value in unsupported.items() if value != 1
        }
        if parallel.use_hsdp:
            unsupported["use_hsdp"] = True
        if parallel.enable_expert_parallel:
            unsupported["enable_expert_parallel"] = True
        if unsupported:
            raise ValueError(
                "The homogeneous TP path requires other Omni parallel "
                f"dimensions to be disabled, got {unsupported}"
            )
        if world_size != tp_size:
            raise ValueError(
                "The homogeneous TP path requires TP to span WORLD: "
                f"world_size={world_size}, tp_size={tp_size}"
            )

        parallel.tensor_parallel_size = tp_size
        parallel.data_parallel_size = None
        parallel.resolve_data_parallel_size(world_size)
        od_config.num_gpus = world_size
        diffusion_vllm_config = create_diffusion_vllm_config(device, od_config)
        self._diffusion_vllm_config = diffusion_vllm_config
        current_omni_platform.init_diffusion_worker_vllm_config(
            diffusion_vllm_config
        )
        runner = DiffusionModelRunner(
            vllm_config=diffusion_vllm_config,
            od_config=od_config,
            device=device,
        )
        return runner

    @property
    def diffusion_vllm_config(self) -> Any:
        return self._diffusion_vllm_config

    @property
    def omni_diffusion_config(self) -> Any:
        return self._omni_diffusion_config

    @staticmethod
    def _restore_diffusion_request(request: Any) -> Any:
        """RPC 将 dataclass 展平为 dict 后，恢复 Omni 请求类型。"""
        if not isinstance(request, dict):
            return request

        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        data = dict(request)
        sampling = data.get("sampling_params")
        if isinstance(sampling, dict):
            sampling_fields = {
                field.name
                for field in fields(OmniDiffusionSamplingParams)
                if field.init
            }
            data["sampling_params"] = OmniDiffusionSamplingParams(
                **{
                    key: value
                    for key, value in sampling.items()
                    if key in sampling_fields
                }
            )
        request_fields = {
            field.name for field in fields(OmniDiffusionRequest) if field.init
        }
        return OmniDiffusionRequest(
            **{
                key: value
                for key, value in data.items()
                if key in request_fields
            }
        )

    @property
    def pending_work_count(self) -> int:
        with self._work_lock:
            return len(self._step_scheduler.active_request_ids)

    def recv_job(
        self,
        work: Any,
        *,
        request_id: str | None = None,
    ) -> None:
        if work is None:
            raise ValueError("OmniModelExecUnit cannot receive an empty work item")
        request = self._restore_diffusion_request(work)
        session_id = request_id
        if session_id is None:
            session_id = getattr(request, "request_id", None)
        if not session_id:
            raise ValueError("OmniModelExecUnit requires a request_id")
        session_id = str(session_id)
        with self._work_lock:
            if request.request_id != session_id:
                request.request_id = session_id
            self._step_scheduler.add_request(request)

    def abort_requests(self, request_ids: tuple[str, ...]) -> tuple[str, ...]:
        """在当前 slice 排空后清理 scheduler 与 Runner 的请求状态。"""
        from vllm_omni.diffusion.sched.interface import (
            DiffusionRequestStatus,
        )

        requested = tuple(dict.fromkeys(str(value) for value in request_ids))
        with self._step_lock:
            with self._work_lock:
                active = tuple(
                    request_id
                    for request_id in requested
                    if request_id in self._step_scheduler.active_request_ids
                )
                if not active:
                    return requested
                pending = self._pending_scheduler_output
                if pending is not None and any(
                    request_id in pending.scheduled_request_ids
                    for request_id in active
                ):
                    # 该 batch 尚未执行；丢弃后由 scheduler 为剩余请求重排。
                    self._pending_scheduler_output = None
                self._step_scheduler.finish_requests(
                    list(active), DiffusionRequestStatus.FINISHED_ABORTED
                )
                for request_id in active:
                    self._step_scheduler.pop_request_state(request_id)
            # Runner 持有 latents/cache/step metadata，必须与 scheduler 一并清理。
            release = getattr(self.runner, "release_step_requests", None)
            if release is not None:
                release(list(active))
            return requested

    def next_slice_spec(
        self, request_id: str | None = None
    ) -> ExecutionSliceDescriptor:
        """调度下一批请求，并返回这个 batch 的模型无关描述。"""
        del request_id
        with self._work_lock:
            if self._pending_scheduler_output is None:
                output = self._step_scheduler.schedule()
                if output.is_empty:
                    raise RuntimeError(
                        f"ExecUnit {self.model_id!r} has no runnable requests"
                    )
                self._pending_scheduler_output = output
            output = self._pending_scheduler_output
            request_ids = tuple(output.scheduled_request_ids)
            phases = {
                self._step_scheduler.execution_phase(request_id)[0]
                for request_id in request_ids
            }
            if len(phases) != 1:
                raise RuntimeError(
                    f"Mixed execution phases in one Omni batch: {phases!r}"
                )
            phase = next(iter(phases))
            tags = {"diffusion", phase}
            if phase == "vae":
                tags.add("granularity:block")
            return ExecutionSliceDescriptor(
                sequence_no=self._slice_sequence,
                tags=frozenset(tags),
                metadata={"request_ids": request_ids},
            )

    def execute_slice(
        self,
        descriptor: ExecutionSliceDescriptor,
        *,
        request_id: str | None = None,
    ) -> ExecutionSliceResult:
        """用 Omni StepScheduler 的一个 batch 推进一个 execution slice。"""
        del request_id
        with self._step_lock:
            expected = self.next_slice_spec()
            if descriptor.sequence_no != expected.sequence_no:
                raise RuntimeError(
                    f"Stale execution slice for {self.model_id!r}: "
                    f"scheduled={descriptor.sequence_no}, "
                    f"expected={expected.sequence_no}"
                )
            if descriptor.tags != expected.tags:
                raise RuntimeError(
                    f"Stale execution slice tags for {self.model_id!r}: "
                    f"scheduled={sorted(descriptor.tags)!r}, "
                    f"expected={sorted(expected.tags)!r}"
                )
            with self._work_lock:
                scheduler_output = self._pending_scheduler_output
                self._pending_scheduler_output = None
            runner_output = self.runner.execute_stepwise(scheduler_output)
            with self._work_lock:
                finished = self._step_scheduler.update_from_output(
                    scheduler_output, runner_output
                )
                # 原生 DiffusionEngine 会在把终止输出交给客户端后调用
                # pop_request_state()。Worker-hosted 路径没有那层 result
                # pump，因此必须在这里完成同样的 scheduler 回收；否则状态
                # 虽已 FINISHED，却会被误判成仍有下一段工作。
                terminal_requests: dict[str, dict[str, str | None]] = {}
                for finished_request_id in finished:
                    terminal_state = self._step_scheduler.pop_request_state(
                        finished_request_id
                    )
                    if terminal_state is None:
                        terminal_requests[finished_request_id] = {
                            "status": None,
                            "error": "Scheduler lost terminal request state",
                        }
                        continue
                    status = getattr(terminal_state, "status", None)
                    status_name = getattr(status, "name", None)
                    terminal_requests[finished_request_id] = {
                        "status": (
                            str(status_name)
                            if status_name is not None
                            else str(status)
                        ),
                        "error": getattr(terminal_state, "error", None),
                    }
                self._slice_sequence += 1
                all_done = not self._step_scheduler.active_request_ids
            outputs = {
                request_id: runner_output.get_request_output(request_id)
                for request_id in scheduler_output.scheduled_request_ids
            }
            return ExecutionSliceResult(
                sequence_no=descriptor.sequence_no,
                request_done=all_done,
                output=outputs,
                tags=descriptor.tags,
                metadata={
                    "request_ids": tuple(
                        scheduler_output.scheduled_request_ids
                    ),
                    "finished_request_ids": tuple(sorted(finished)),
                    "terminal_requests": terminal_requests,
                },
            )

    def bind_resources(
        self,
        assignment: ResourceAssignment,
        *,
        load_format: str,
    ) -> None:
        """创建并接入 Omni runtime providers；不加载模型。"""
        del load_format
        from vllm_omni.diffusion.worker.runtime_providers import (
            DiffusionRuntimeProviders,
        )

        communicator = _OmniParallelGroupProvider(
            assignment.communicator
        )
        self._communicator_provider = communicator
        providers = DiffusionRuntimeProviders(
            memory=assignment.memory,
            stream=assignment.streams,
            communicator=communicator,
        )
        self._base_runtime_providers = providers
        self.runner._container_runtime_providers = providers

    def _providers_for_assignment(
        self, assignment: ResourceAssignment
    ) -> Any:
        """为本次资源变体派生 Omni provider 视图。"""
        from vllm_omni.diffusion.worker.runtime_providers import (
            DiffusionRuntimeProviders,
        )

        base = self._base_runtime_providers
        if base is None:
            raise RuntimeError("Omni runtime providers are not initialized")
        return DiffusionRuntimeProviders(
            memory=assignment.memory,
            stream=assignment.streams,
            communicator=base.communicator,
            phase=base.phase,
        )

    @contextmanager
    def activate(
        self,
        assignment: ResourceAssignment,
        *,
        resource_transition: bool = True,
    ):
        if assignment.graph is not None:
            raise ValueError("Omni Runner does not accept a CUDA Graph binding")

        from vllm.config import set_current_vllm_config
        from vllm_omni.diffusion.forward_context import set_forward_context

        runner = self._model_runner
        if runner is None:
            raise RuntimeError("Omni Runner binding is closed")
        selected_providers = self._providers_for_assignment(assignment)
        previous_runtime = getattr(runner, "runtime_providers", None)
        previous_container = getattr(
            runner, "_container_runtime_providers", None
        )
        # DiffusionModelRunner 会在 execute_model 内再次进入 provider；必须让
        # 该内层上下文看到本次 Container 选择的 stream，而不是初始化时的
        # 默认变体。
        runner.runtime_providers = selected_providers
        runner._container_runtime_providers = selected_providers

        # 激活绑定
        try:
            with ExitStack() as stack:
                # stack用来动态组合多个 context manager，并保证按相反顺序退出。
                communicator = self._communicator_provider
                if communicator is None:
                    raise RuntimeError(
                        "Omni communicator provider is not initialized"
                    )
                stack.enter_context(communicator.activate())
                stack.enter_context(assignment.memory.activate())
                stack.enter_context(
                    assignment.streams.activate(
                        runner,
                        synchronize_transition=resource_transition,
                    )
                )
                # Omni 执行所需的软件运行时上下文，不是物理资源
                stack.enter_context(
                    set_current_vllm_config(self._diffusion_vllm_config)
                )
                stack.enter_context(
                    set_forward_context(
                        vllm_config=self._diffusion_vllm_config,
                        omni_diffusion_config=self._omni_diffusion_config,
                    )
                )
                yield
        finally:
            runner._container_runtime_providers = (
                self._base_runtime_providers
                if previous_container is None
                else previous_container
            )
            runner.runtime_providers = (
                self._base_runtime_providers
                if previous_runtime is None
                else previous_runtime
            )

    def close(self) -> None:
        with self._work_lock:
            self._pending_scheduler_output = None
            self._step_scheduler.close()
        runner, self._model_runner = self._model_runner, None
        if runner is None:
            return
        close = getattr(runner, "close", None)
        if close is None:
            close = getattr(runner, "shutdown", None)
        if close is not None:
            close()

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        load_format = (
            "dummy" if load_dummy_weights else self.spec.load_format
        )
        providers = getattr(self.runner, "_container_runtime_providers", None)
        if providers is None:
            raise RuntimeError(
                f"Omni ExecUnit {self.model_id!r} has no bound "
                "runtime providers"
            )
        self.runner.load_model(
            runtime_providers=providers,
            load_format=load_format,
        )

    def determine_available_memory(self) -> int:
        raise NotImplementedError(
            "Omni memory profiling has not been connected to the unified "
            "Container lifecycle"
        )

    def initialize_runtime(self, runtime_config: Any | None = None) -> None:
        """Omni 当前没有独立于模型加载的运行时初始化阶段。"""
        if runtime_config is not None:
            raise ValueError(
                f"Omni ExecUnit {self.model_id!r} does not accept an "
                "external runtime configuration"
            )

    def compile_or_warm_up_model(self) -> Any:
        prepare = getattr(self.runner, "compile_or_warm_up_model", None)
        return None if prepare is None else prepare()
