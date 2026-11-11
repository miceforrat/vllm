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
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from vllm.experimental.model_instance_registry import ModelSpec

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig

_WorkT = TypeVar("_WorkT")
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class OmniStepResult:
    """一次 ``step`` 已完成的 Omni 细粒度执行单元。"""

    kind: str
    index: int | None
    request_done: bool
    output: Any | None = None


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
        """已经接收、但尚未被 ``step`` 推进的工作数量。"""

    @abstractmethod
    def recv_job(self, work: _WorkT) -> None:
        """为这个长期模型实例接收一项新工作。从而可以步进推进job"""

    @abstractmethod
    def step(self) -> _ResultT:
        """推进一项已接收的工作，并返回 Runner 的执行结果。一般就是推进当前工作"""

    def next_slice_spec(self) -> tuple[str, int | None]:
        """返回下一次 ``step`` 的逻辑边界；默认模型只有普通 step。"""
        return ("model_step", None)


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

    def recv_job(self, work: Any) -> None:
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

    def step(self) -> Any:
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
        self._active_request: Any | None = None
        self._active_request_state: Any | None = None
        self._active_phase = "idle"
        self._vae_blocks_prepared = False
        super().__init__(spec, vllm_config, device)
        self._pending_work: deque[Any] = deque()
        self._work_lock = threading.Lock()
        self._step_lock = threading.Lock()

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

    def _start_next_request(self) -> None:
        with self._work_lock:
            if not self._pending_work:
                raise RuntimeError(
                    f"ExecUnit {self.model_id!r} has no runnable work"
                )
            request = self._pending_work.popleft()
        self._active_request = self._restore_diffusion_request(request)
        self._active_request_state = None
        self._active_phase = "preparing"
        self._vae_blocks_prepared = False

    def _finish_active_request(self) -> None:
        self._active_request = None
        self._active_request_state = None
        self._active_phase = "idle"
        self._vae_blocks_prepared = False

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
            return len(self._pending_work)

    def recv_job(self, work: Any) -> None:
        if work is None:
            raise ValueError("OmniModelExecUnit cannot receive an empty work item")
        with self._work_lock:
            self._pending_work.append(work)

    def next_slice_spec(self) -> tuple[str, int | None]:
        """只描述下一段工作，不在 ExecUnit 内选择物理资源。"""
        if self._active_request is None or self._active_phase == "preparing":
            return ("preparing", None)
        state = self._active_request_state
        if state is None:
            raise RuntimeError("Omni explicit request has no runtime state")
        if self._active_phase == "denoising":
            return ("denoising", int(state.step_index))
        if self._active_phase == "vae_blocks":
            return (
                "vae_blocks",
                int(state.extra.get("vae_block_index", 0)),
            )
        raise RuntimeError(
            f"Unknown Omni execution phase {self._active_phase!r}"
        )

    def step(self) -> OmniStepResult:
        """同步推进一个显式 preparing/denoising/vae_blocks 单元。"""
        with self._step_lock:
            if self._active_request is None:
                self._start_next_request()
            if self._active_phase == "preparing":
                assert self._active_request is not None
                self._active_request_state = (
                    self.runner.prepare_explicit_request(
                        self._active_request
                    )
                )
                self._active_phase = "denoising"
                return OmniStepResult("preparing", None, False)

            state = self._active_request_state
            if state is None:
                raise RuntimeError("Omni explicit request has no runtime state")
            if self._active_phase == "denoising":
                index = int(state.step_index)
                self.runner.execute_explicit_denoising_step(state)
                if state.denoise_completed:
                    self._active_phase = "vae_blocks"
                return OmniStepResult("denoising", index, False)

            if self._active_phase == "vae_blocks":
                if not self._vae_blocks_prepared:
                    self.runner.prepare_explicit_vae_blocks(state)
                    self._vae_blocks_prepared = True
                index = int(state.extra["vae_block_index"])
                done = self.runner.execute_explicit_vae_block(state)
                output = None
                if done:
                    output = self.runner.finalize_explicit_vae_blocks(state)
                result = OmniStepResult(
                    "vae_blocks", index, done, output
                )
                if done:
                    self._finish_active_request()
                return result

            raise RuntimeError(
                f"Unknown Omni execution phase {self._active_phase!r}"
            )
