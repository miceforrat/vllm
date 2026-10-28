# SPDX-License-Identifier: Apache-2.0
"""Adapters for embedding a vLLM-Omni diffusion runner in vLLM."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ResolvedOmniDiffusionStage:
    """Official Omni configuration views for one diffusion stage."""

    pipeline_config: Any
    stage_config: Any
    diffusion_config: Any


@dataclass(frozen=True)
class DiffusionRequestMemoryProfile:
    """vLLM-owned accounting around one original Omni runner request."""

    baseline_allocated_bytes: int
    baseline_reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    transient_allocated_headroom_bytes: int
    reserved_pool_headroom_bytes: int
    end_allocated_bytes: int
    end_reserved_bytes: int
    end_live_allocated_delta_bytes: int
    allocator_reserved_delta_bytes: int
    tagged_pool_before_bytes: int | None
    tagged_pool_after_bytes: int | None
    tagged_pool_growth_bytes: int | None


class _SingleRankGroupView:
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

    def __init__(self, coordinator: Any):
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


def resolve_omni_diffusion_stage(
    model: str,
    *,
    trust_remote_code: bool = False,
    deploy_config_path: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedOmniDiffusionStage:
    """Resolve one diffusion stage through vLLM-Omni's public config path.

    Imports are intentionally local so vLLM does not acquire a hard runtime
    dependency on vLLM-Omni unless this experimental adapter is used.
    """
    from vllm_omni.config.omni_config import (
        extract_diffusion_stage_config_kwargs,
    )
    from vllm_omni.config.resolver import resolve_omni_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.engine.stage_init_utils import (
        build_engine_args_dict_from_omni_stage_config,
    )

    overrides = dict(cli_overrides or {})
    overrides.setdefault("model", model)
    resolution = resolve_omni_config(
        model,
        trust_remote_code=trust_remote_code,
        deploy_config_path=deploy_config_path,
        cli_overrides=overrides,
        stage_overrides=None,
        strategy_config_path=None,
    )
    if len(resolution.stage_configs) != 1:
        raise ValueError(
            "The embedded runner currently requires exactly one Omni stage; "
            f"resolved {len(resolution.stage_configs)}"
        )

    stage_config = resolution.stage_configs[0]
    stage_type = getattr(stage_config.stage_type, "value", stage_config.stage_type)
    if stage_type != "diffusion":
        raise ValueError(
            "The embedded runner requires a diffusion stage; "
            f"resolved {stage_config.stage_type!r}"
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

    return ResolvedOmniDiffusionStage(
        pipeline_config=resolution.pipeline_config,
        stage_config=stage_config,
        diffusion_config=diffusion_config,
    )


class EmbeddedOmniDiffusionRunner:
    """Run Omni's original runner on vLLM-owned infrastructure."""

    def __init__(
        self,
        resolved: ResolvedOmniDiffusionStage,
        device: str = "cuda:0",
        use_vllm_memory_pool: bool = True,
        execution_stream: torch.cuda.Stream | None = None,
    ):
        self.resolved = resolved
        self.device = torch.device(device)
        self.vllm_config: Any | None = None
        self.runner: Any | None = None
        self.use_vllm_memory_pool = use_vllm_memory_pool
        self.memory_allocator: Any | None = None
        self.execution_stream = execution_stream
        self._owns_execution_stream = False
        self._green_context_owner: Any | None = None
        self._last_execution_stream_handle: int | None = None
        self._memory_pool_tags: set[str] = set()
        self._exit_stack = ExitStack()
        self._omni_state: Any | None = None
        self._saved_omni_groups: dict[str, Any] = {}

    @contextmanager
    def _memory_pool_context(self, tag: str) -> Any:
        """Create a tagged pool once, then explicitly reuse it."""
        if self.memory_allocator is None:
            raise RuntimeError("vLLM memory allocator is not initialized")
        if tag in self._memory_pool_tags:
            context = self.memory_allocator.reuse_memory_pool(tag)
        else:
            context = self.memory_allocator.use_memory_pool(tag=tag)
        with context:
            self._memory_pool_tags.add(tag)
            yield

    def memory_pool_usage(self) -> int | None:
        """Return bytes currently tracked by the shared vLLM allocator."""
        if self.memory_allocator is None:
            return None
        return int(self.memory_allocator.get_current_usage())

    @staticmethod
    def _stream_handle(stream: torch.cuda.Stream | None) -> int | None:
        if stream is None:
            return None
        return int(stream.cuda_stream)

    def create_execution_stream(self, *, priority: int = 0) -> torch.cuda.Stream:
        """Create and retain a vLLM-owned execution stream on this device."""
        if self.execution_stream is not None:
            raise RuntimeError("An execution stream is already installed")
        with torch.cuda.device(self.device):
            self.execution_stream = torch.cuda.Stream(
                device=self.device, priority=priority
            )
        self._owns_execution_stream = True
        return self.execution_stream

    def set_execution_stream(self, stream: torch.cuda.Stream | None) -> None:
        """Install an externally owned stream, including a future GC stream."""
        if stream is not None and torch.device(stream.device) != self.device:
            raise ValueError(
                f"Execution stream device {stream.device} does not match {self.device}"
            )
        self.execution_stream = stream
        self._owns_execution_stream = False

    def create_green_context_execution_stream(
        self, requested_sm_count: int
    ) -> torch.cuda.ExternalStream:
        """Create and retain an SM-partitioned Green Context execution stream."""
        if self.execution_stream is not None:
            raise RuntimeError("An execution stream is already installed")
        if self._green_context_owner is not None:
            raise RuntimeError("A Green Context is already installed")
        from vllm.experimental.green_context import GreenContextStream

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        owner = GreenContextStream(requested_sm_count, device_index)
        stream = torch.cuda.ExternalStream(owner.stream_ptr, device=device_index)
        self._green_context_owner = owner
        self.execution_stream = stream
        self._owns_execution_stream = True
        return stream

    def stream_control_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.execution_stream is not None,
            "owner": (
                "vllm" if self._owns_execution_stream else
                "external" if self.execution_stream is not None else None
            ),
            "expected_stream_handle": self._stream_handle(self.execution_stream),
            "last_actual_stream_handle": self._last_execution_stream_handle,
            "verified": (
                self.execution_stream is not None
                and self._last_execution_stream_handle
                == self._stream_handle(self.execution_stream)
            ),
            "green_context": (
                None
                if self._green_context_owner is None
                else {
                    "requested_sm_count": self._green_context_owner.requested_sm_count,
                    "actual_sm_count": self._green_context_owner.actual_sm_count,
                }
            ),
        }

    def execute_model(self, request: Any, **kwargs: Any) -> Any:
        """Execute Omni on the installed stream while preserving dependencies."""
        if self.runner is None:
            raise RuntimeError("Embedded Omni runner is not started")
        stream = self.execution_stream
        if stream is None:
            current = torch.cuda.current_stream(self.device)
            self._last_execution_stream_handle = self._stream_handle(current)
            return self.runner.execute_model(request, **kwargs)

        caller = torch.cuda.current_stream(self.device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            actual = torch.cuda.current_stream(self.device)
            self._last_execution_stream_handle = self._stream_handle(actual)
            output = self.runner.execute_model(request, **kwargs)
            completion = torch.cuda.Event()
            completion.record(stream)
        caller.wait_event(completion)
        return output

    def execute_model_profiled(
        self, request: Any, **kwargs: Any
    ) -> tuple[Any, DiffusionRequestMemoryProfile]:
        """Execute through Omni while measuring vLLM-style memory headroom.

        Omni remains responsible for the forward and reports its absolute
        reserved-memory high-water mark. The adapter converts that value into
        transient headroom above the pre-request baseline and separately
        reports persistent tagged-pool growth.
        """
        if self.runner is None:
            raise RuntimeError("Embedded Omni runner is not started")

        torch.cuda.synchronize(self.device)
        baseline_allocated = int(torch.cuda.memory_allocated(self.device))
        baseline_reserved = int(torch.cuda.memory_reserved(self.device))
        tagged_before = self.memory_pool_usage()

        output = self.execute_model(request, **kwargs)
        torch.cuda.synchronize(self.device)

        end_allocated = int(torch.cuda.memory_allocated(self.device))
        end_reserved = int(torch.cuda.memory_reserved(self.device))
        tagged_after = self.memory_pool_usage()
        peak_reserved = int(
            getattr(
                self.runner,
                "_last_peak_reserved_bytes",
                round(float(output.peak_memory_mb) * 1024**2),
            )
        )
        peak_allocated = int(
            getattr(self.runner, "_last_peak_allocated_bytes", end_allocated)
        )
        # Older/custom pipelines may not populate request peak fields.
        peak_reserved = max(peak_reserved, baseline_reserved, end_reserved)
        peak_allocated = max(peak_allocated, baseline_allocated, end_allocated)

        tagged_growth = (
            None
            if tagged_before is None or tagged_after is None
            else tagged_after - tagged_before
        )
        profile = DiffusionRequestMemoryProfile(
            baseline_allocated_bytes=baseline_allocated,
            baseline_reserved_bytes=baseline_reserved,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            transient_allocated_headroom_bytes=max(
                0, peak_allocated - baseline_allocated
            ),
            reserved_pool_headroom_bytes=max(
                0, peak_reserved - baseline_reserved
            ),
            end_allocated_bytes=end_allocated,
            end_reserved_bytes=end_reserved,
            end_live_allocated_delta_bytes=end_allocated - baseline_allocated,
            allocator_reserved_delta_bytes=end_reserved - baseline_reserved,
            tagged_pool_before_bytes=tagged_before,
            tagged_pool_after_bytes=tagged_after,
            tagged_pool_growth_bytes=tagged_growth,
        )
        return output, profile

    def _install_omni_group_views(
        self,
        *,
        world: Any,
        tp: Any,
        pp: Any,
    ) -> None:
        import vllm_omni.diffusion.distributed.parallel_state as omni_state

        self._omni_state = omni_state
        names = ("_WORLD", "_SP", "_PP", "_CFG", "_DP", "_FS")
        names += ("_HSDP_REPLICATE", "_EXPERT_PARALLEL_GROUP_RANKS")
        self._saved_omni_groups = {name: getattr(omni_state, name) for name in names}
        singleton = _SingleRankGroupView(tp)
        omni_state._WORLD = world
        omni_state._SP = singleton
        omni_state._PP = _SingleRankGroupView(pp)
        omni_state._CFG = singleton
        omni_state._DP = singleton
        omni_state._FS = singleton
        omni_state._HSDP_REPLICATE = singleton
        omni_state._EXPERT_PARALLEL_GROUP_RANKS = [[0]]

    def _restore_omni_group_views(self) -> None:
        if self._omni_state is not None:
            for name, value in self._saved_omni_groups.items():
                setattr(self._omni_state, name, value)

    def start(self, *, load_format: str = "default") -> Any:
        from vllm.config import set_current_vllm_config
        from vllm.device_allocator import get_mem_allocator_instance
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tp_group,
            get_world_group,
        )
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
        from vllm_omni.diffusion.forward_context import set_forward_context
        from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
        from vllm_omni.diffusion.worker.diffusion_model_runner import (
            DiffusionModelRunner,
        )
        from vllm_omni.platforms import current_omni_platform

        if self.runner is not None:
            raise RuntimeError("Embedded Omni runner is already started")
        if self.resolved.diffusion_config.parallel_config.world_size != 1:
            raise ValueError("The initial embedded runner supports one GPU only")

        torch.cuda.set_device(self.device)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        od_config = self.resolved.diffusion_config
        self.vllm_config = create_diffusion_vllm_config(self.device, od_config)
        current_omni_platform.init_diffusion_worker_vllm_config(self.vllm_config)
        self._exit_stack.enter_context(set_current_vllm_config(self.vllm_config))
        self._exit_stack.enter_context(
            set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=od_config,
            )
        )
        init_worker_distributed_environment(
            self.vllm_config,
            rank=0,
            distributed_init_method="env://",
            local_rank=0,
            backend="nccl",
        )
        self._install_omni_group_views(
            world=get_world_group(),
            tp=get_tp_group(),
            pp=get_pp_group(),
        )
        self.runner = DiffusionModelRunner(
            vllm_config=self.vllm_config,
            od_config=od_config,
            device=self.device,
        )
        memory_pool_context_fn = None
        if self.use_vllm_memory_pool:
            self.memory_allocator = get_mem_allocator_instance()
            if self.memory_allocator.get_current_usage() != 0:
                raise RuntimeError(
                    "The shared vLLM allocator already owns live allocations; "
                    "the initial adapter requires an empty allocator before "
                    "loading Diffusion weights"
                )
            memory_pool_context_fn = self._memory_pool_context
        self.runner.load_model(
            memory_pool_context_fn=memory_pool_context_fn,
            load_format=load_format,
        )
        return self.runner

    def close(self) -> None:
        if self.execution_stream is not None:
            self.execution_stream.synchronize()
        self.runner = None
        if self._green_context_owner is not None:
            self._green_context_owner.close()
            self._green_context_owner = None
            self.execution_stream = None
        self._restore_omni_group_views()
        self._saved_omni_groups.clear()
        self._omni_state = None
        self._exit_stack.close()

    def __enter__(self) -> EmbeddedOmniDiffusionRunner:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
