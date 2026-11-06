# SPDX-License-Identifier: Apache-2.0
"""Worker-local factory for scheduler-owned and runtime-added runners."""

from __future__ import annotations

from typing import Any, Protocol

from vllm.experimental.communicator_registry import CommunicatorRegistry
from vllm.experimental.model_instance_registry import (
    CommunicationGroupSpec,
    ModelInstance,
    ModelSpec,
)


# 统一的Runner后端协议
class RunnerBackend(Protocol):
    """Backend-specific construction and teardown of one model instance."""

    # 这里传递worker不太好，未来等稳定了再考虑是不是单独创一个参数类
    def create(
        self,
        spec: ModelSpec,
        *,
        worker: Any,
        communicators: CommunicatorRegistry,
    ) -> ModelInstance: ...

    def destroy(
        self,
        instance: ModelInstance,
        *,
        communicators: CommunicatorRegistry,
    ) -> None: ...


# vLLM的后端
class VllmRunnerBackend:
    """Construct the native vLLM Runner through the common factory path."""

    def create(
        self,
        spec: ModelSpec,
        *,
        worker: Any,
        communicators: CommunicatorRegistry,
    ) -> ModelInstance:
        del communicators
        if worker.use_v2_model_runner:
            if worker.vllm_config.is_mm_encoder_only:
                from vllm.v1.worker.mm_encoder_model_runner import (
                    MMEncoderModelRunner as GPUModelRunner,
                )
            else:
                from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        else:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(worker.vllm_config, worker.device)
        return ModelInstance(
            spec=spec,
            runner=runner,
            scheduler_managed=True,
            runtime_added=False,
        )

    def destroy(
        self,
        instance: ModelInstance,
        *,
        communicators: CommunicatorRegistry,
    ) -> None:
        del communicators
        runner = instance.runner
        if runner is None:
            return
        shutdown = getattr(runner, "shutdown", None)
        if shutdown is not None:
            shutdown()
        instance.runner = None


# vllm-omni的后端
class OmniRunnerBackend:
    """Construct the existing Omni DiffusionModelRunner inside a vLLM Worker."""

    # 从spec里获取并行组配置
    # 这里因为我们只搞了同构TP，所以还验证了是否同构
    @staticmethod
    def _tp_group_spec(spec: ModelSpec) -> CommunicationGroupSpec:
        from vllm.distributed.parallel_state import get_tp_group

        tp_ranks = tuple(int(rank) for rank in get_tp_group().ranks)
        if tuple(spec.worker_ranks) != tp_ranks:
            raise ValueError(
                "The homogeneous TP gate requires Generator placement to match "
                f"the vLLM TP group exactly: placement={spec.worker_ranks}, "
                f"tp={tp_ranks}"
            )
        tp_specs = tuple(
            group
            for group in spec.communication_groups
            if group.parallel_dimension == "tp"
        )
        if len(tp_specs) != 1:
            raise ValueError(
                "An auxiliary runner requires exactly one Engine-planned TP "
                f"communication group, got {len(tp_specs)}"
            )
        tp_spec = tp_specs[0]
        if tuple(tp_spec.ranks) != tp_ranks:
            raise ValueError(
                "The homogeneous TP gate requires the Engine-planned group "
                f"to match vLLM TP ranks: planned={tp_spec.ranks}, tp={tp_ranks}"
            )
        return tp_spec

    def create(
        self,
        spec: ModelSpec,
        *,
        worker: Any,
        communicators: CommunicatorRegistry,
    ) -> ModelInstance:
        from vllm.experimental.omni_runner_adapter import (
            EmbeddedOmniDiffusionRunner,
            resolve_omni_diffusion_stage,
        )

        tp_spec = self._tp_group_spec(spec)
        resolved = resolve_omni_diffusion_stage(
            spec.model_path,
            cli_overrides=spec.backend_options,
        )
        adapter = EmbeddedOmniDiffusionRunner(
            resolved,
            device=str(worker.device),
            memory_pool_tag_prefix=spec.model_id,
        )
        endpoint = getattr(
            worker, "_vllm_execution_unit_scheduler_endpoint", None
        )
        if endpoint is not None:
            adapter.configure_execution_unit_scheduler(
                model_id=spec.model_id,
                address=endpoint[0],
                authkey_hex=endpoint[1],
            )
        tp_group = None
        try:
            tp_group = communicators.acquire(tp_spec, spec.model_id)
            execution_stream = getattr(
                worker, "_multi_model_generator_stream", None
            )
            if execution_stream is not None:
                adapter.set_execution_stream(execution_stream)
            # 和执行流资源选择相关
            phase_selector = (
                getattr(worker, "acquire_generator_phase_lease", None)
                if getattr(worker, "_multi_model_resource_arbiter", None)
                is not None
                else None
            )
            if phase_selector is not None:
                adapter.set_generator_phase_stream_selector(phase_selector)
            runner = adapter.start_on_worker(
                worker,
                tp_group=tp_group,
                load_format=spec.load_format,
            )
        except Exception:
            adapter.close()
            if tp_group is not None:
                communicators.release(spec.model_id, (tp_spec.group_id,))
            raise
        return ModelInstance(
            spec=spec,
            adapter=adapter,
            runner=runner,
            runtime_added=True,
            communication_group_ids=(tp_spec.group_id,),
        )

    def destroy(
        self,
        instance: ModelInstance,
        *,
        communicators: CommunicatorRegistry,
    ) -> None:
        # Drop the duplicate runner reference before adapter teardown so the
        # pipeline can destruct while Omni compatibility views remain active.
        instance.runner = None
        if instance.adapter is not None:
            instance.adapter.close()
            instance.adapter = None
        communicators.release(
            instance.spec.model_id, instance.communication_group_ids
        )


class RunnerFactory:
    """Create a common ModelInstance regardless of the model backend."""

    def __init__(
        self,
        worker: Any,
        communicators: CommunicatorRegistry,
    ) -> None:
        self._worker = worker
        self._communicators = communicators
        self._backends: dict[str, RunnerBackend] = {
            "vllm": VllmRunnerBackend(),
            "vllm_omni": OmniRunnerBackend(),
        }

    def register_backend(self, name: str, backend: RunnerBackend) -> None:
        if name in self._backends:
            raise ValueError(f"Runner backend {name!r} is already registered")
        self._backends[name] = backend

    def create(self, spec: ModelSpec) -> ModelInstance:
        try:
            backend = self._backends[spec.backend]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported runtime model backend: {spec.backend!r}"
            ) from exc
        return backend.create(
            spec,
            worker=self._worker,
            communicators=self._communicators,
        )

    def destroy(self, instance: ModelInstance) -> None:
        runner = instance.runner
        try:
            backend = self._backends[instance.spec.backend]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported runtime model backend: {instance.spec.backend!r}"
            ) from exc
        try:
            backend.destroy(instance, communicators=self._communicators)
        finally:
            # ``GPUWorker.model_runner`` is a compatibility reference to the
            # scheduler-managed instance, not a second owner. Clear it after
            # the backend has attempted teardown so native shutdown cannot
            # invoke a partially destroyed object twice.
            if getattr(self._worker, "model_runner", None) is runner:
                self._worker.model_runner = None
