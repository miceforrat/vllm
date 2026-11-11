# SPDX-License-Identifier: Apache-2.0
"""GPU worker with complementary Green Context execution streams."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.experimental.archived.communicator_registry import CommunicatorRegistry
from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.archived.model_instance_registry import ModelInstanceRegistry
from vllm.experimental.archived.runner_factory import RunnerFactory
from vllm.experimental.worker.binding import CudaStreamBinding
from vllm.experimental.worker.resource_manager import (
    GCCudaGraphManager,
    GraphVariantKey,
)
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


def _pool_token_key(token: Any) -> str:
    """Return a stable, JSON-safe representation of an allocator pool id."""
    if isinstance(token, (tuple, list)):
        return "(" + ",".join(str(int(value)) for value in token) + ")"
    return repr(token)


def _cuda_memory_snapshot_summary() -> dict[str, Any]:
    """Aggregate CUDA allocator segments without retaining the raw snapshot."""
    try:
        segments = torch.cuda.memory_snapshot()
    except Exception as exc:  # pragma: no cover - version/runtime dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    pools: dict[str, dict[str, Any]] = {}
    for segment in segments:
        raw_pool = segment.get("segment_pool_id")
        if raw_pool is None:
            raw_pool = segment.get("pool_id")
        pool_key = (
            "unreported"
            if raw_pool is None
            else _pool_token_key(raw_pool)
        )
        aggregate = pools.setdefault(
            pool_key,
            {
                "segment_count": 0,
                "total_size_bytes": 0,
                "allocated_size_bytes": 0,
                "active_size_bytes": 0,
                "requested_size_bytes": 0,
                "streams": [],
                "allocation_sites": {},
            },
        )
        aggregate["segment_count"] += 1
        for field in (
            "total_size",
            "allocated_size",
            "active_size",
            "requested_size",
        ):
            aggregate[f"{field}_bytes"] += int(segment.get(field, 0))
        stream = segment.get("stream")
        if stream is not None and stream not in aggregate["streams"]:
            aggregate["streams"].append(stream)
        for block in segment.get("blocks", ()):
            if not str(block.get("state", "")).startswith("active_"):
                continue
            frames = block.get("frames") or segment.get("frames") or ()
            if frames:
                site = " <- ".join(
                    f"{frame.get('filename', '?')}:"
                    f"{frame.get('line', '?')}:"
                    f"{frame.get('name', '?')}"
                    for frame in frames[:8]
                )
            else:
                site = "<unrecorded>"
            sites = aggregate["allocation_sites"]
            row = sites.setdefault(
                site,
                {
                    "block_count": 0,
                    "size_bytes": 0,
                    "requested_size_bytes": 0,
                },
            )
            row["block_count"] += 1
            row["size_bytes"] += int(block.get("size", 0))
            row["requested_size_bytes"] += int(
                block.get("requested_size", 0)
            )

    for aggregate in pools.values():
        aggregate["streams"].sort(key=str)
        aggregate["allocation_sites"] = dict(
            sorted(
                aggregate["allocation_sites"].items(),
                key=lambda item: item[1]["requested_size_bytes"],
                reverse=True,
            )[:64]
        )
    return {
        "available": True,
        "segment_count": len(segments),
        "segment_fields": sorted(
            {key for segment in segments for key in segment}
        ),
        "pools": pools,
    }


def _graph_pool_summary(manager: Any) -> dict[str, Any]:
    primary = {
        _pool_token_key(graph.pool()) for graph in manager.graphs.values()
    }
    retained: set[str] = set()
    retained_pools_supported = True
    for graph in manager.graphs.values():
        pools = getattr(graph, "pools", None)
        if pools is None:
            retained_pools_supported = False
            continue
        retained.update(_pool_token_key(pool) for pool in pools())
    return {
        "manager_pool": _pool_token_key(manager.pool),
        "actual_primary_pools": sorted(primary),
        "actual_retained_pools": sorted(retained),
        "retained_pools_supported": retained_pools_supported,
    }


class MultiModelGPUWorker(GPUWorker):
    """Bind primary and auxiliary runners to complementary GC streams."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # These registries must exist before GPUWorker.__init__ reaches its
        # overridable Runner construction hook. Startup and runtime-added
        # models consequently use the same factory and ownership containers.
        self._vllm_communicator_registry = CommunicatorRegistry()
        self._vllm_model_instance_registry = ModelInstanceRegistry()
        self._vllm_runner_factory = RunnerFactory(
            self, self._vllm_communicator_registry
        )
        super().__init__(*args, **kwargs)
        self._multi_model_gc_pair: Any | None = None
        # 记录当前选择了哪条
        # reasoner的切换逻辑和generator略有区别所以需要这个变量
        self._multi_model_active_reasoner_stream: torch.cuda.Stream | None = None

        self._multi_model_reasoner_stream: torch.cuda.Stream | None = None
        self._multi_model_reasoner_full_stream: torch.cuda.Stream | None = None
        self._multi_model_generator_stream: torch.cuda.Stream | None = None
        self._multi_model_generator_full_stream: torch.cuda.Stream | None = None

        # 确保graph replay在捕获的stream上
        self._multi_model_graph_patch: Any | None = None

        # 仲裁器，决定请求使用什么流
        self._multi_model_resource_arbiter: Any | None = None

        # 一个模型只持有一个原生 ModelCudaGraphManager；shared/full 是其中的
        # 两个资源变体，而不是两套 manager。
        self._multi_model_graph_manager = GCCudaGraphManager()
        additional = getattr(self.vllm_config, "additional_config", None)
        runtime = (
            additional.get("multi_model_runtime", {})
            if isinstance(additional, dict)
            else {}
        )
        startup_models = runtime.get("startup_models", ())
        if len(startup_models) != 1:
            raise RuntimeError(
                "Graph resource initialization requires exactly one planned "
                f"startup model, got {len(startup_models)}"
            )
        self._multi_model_graph_owner = ModelSpec.from_rpc_payload(
            startup_models[0]
        ).model_id
        self._multi_model_shared_graph_key = GraphVariantKey(
            self._multi_model_graph_owner, "shared"
        )
        self._multi_model_full_graph_key = GraphVariantKey(
            self._multi_model_graph_owner, "reasoner_full"
        )

        # 控制量
        self._multi_model_graph_pool_mode = "isolated"
        self._multi_model_graph_buffer_mode = "isolated"
        self._multi_model_graph_capture_mode = "full"
        self._multi_model_partition_policy = "static"
        self._multi_model_graph_pools_shared = False
        self._multi_model_graph_buffers_shared = False
        # 记录上一批使用的 shared/reasoner_full 变体，用于识别切换。
        self._multi_model_last_graph_variant_mode: str | None = None

        # 记录最近一批 Reasoner GPU 工作提交后的 event，用于判断/等待该批 GPU 工作真正完成
        self._multi_model_primary_last_submit_event: torch.cuda.Event | None = None

        # 主要用于观测
        self._multi_model_graph_memory: dict[str, int] = {}
        self._multi_model_graph_snapshot: dict[str, Any] = {}
        self._multi_model_graph_startup: dict[str, float] = {}
        self._multi_model_graph_registry_switches = 0
        self._multi_model_graph_variant_switches = 0
        self._multi_model_graph_variant_switch_sync_seconds = 0.0
        self._multi_model_startup: dict[str, Any] = {}
        self._multi_model_startup_timeline: dict[str, float] = {}
        self._multi_model_primary_batch_index = 0
        self._multi_model_trace_partition_nvtx = False

    def _create_model_runner(self):
        """Materialize the Engine-planned startup model via RunnerFactory."""
        additional = getattr(self.vllm_config, "additional_config", None)
        runtime = (
            additional.get("multi_model_runtime", {})
            if isinstance(additional, dict)
            else {}
        )
        startup = runtime.get("startup_models", ())
        if len(startup) != 1:
            raise RuntimeError(
                "MultiModelGPUWorker requires exactly one Engine-planned "
                f"startup model during the compatibility bootstrap, got "
                f"{len(startup)}"
            )
        spec = ModelSpec.from_rpc_payload(startup[0])
        instance = self._vllm_runner_factory.create(spec)
        self._vllm_model_instance_registry.register(instance)
        return instance.runner

    # 设备级初始化（主要多加入了gc）
    def init_device(self) -> None:
        device_init_started = time.perf_counter()
        self._multi_model_startup_timeline[
            "device_and_distributed_init_start"
        ] = device_init_started
        super().init_device()
        self._multi_model_startup_timeline[
            "device_and_distributed_init_end"
        ] = time.perf_counter()
        self._multi_model_startup["device_and_distributed_init_seconds"] = (
            time.perf_counter() - device_init_started
        )
        stream_setup_started = time.perf_counter()
        self._multi_model_startup_timeline["resource_stream_setup_start"] = (
            stream_setup_started
        )
        config = self.vllm_config.additional_config
        if not isinstance(config, dict):
            return
        runtime = config.get("multi_model_runtime")
        if not isinstance(runtime, dict):
            return
        green_context = runtime.get("green_context")
        if not isinstance(green_context, dict):
            return
        self._multi_model_trace_partition_nvtx = bool(
            green_context.get("trace_partition_nvtx", False)
        )

        # 初始化gc streams
        # 目前是按照成对的方式初始化（虽然gc是可以创建多个分区的）
        from vllm.experimental.green_context import GreenContextPair
        from vllm.experimental.archived.multi_model_arbiter import (
            MultiModelResourceArbiter,
        )
        from vllm.experimental.runtime_stream_patch import (
            install_vllm_graph_capture_stream_patch,
        )

        # 目前只支持单分区
        # 未来可能需要支持多分区，但是现在暂且用不上
        requested = int(green_context["generator_sm_count"])
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        # 申请一套分区
        # reasoner吃剩饭
        pair = GreenContextPair(requested, device_index)
        self._multi_model_gc_pair = pair
        self._multi_model_generator_stream = torch.cuda.ExternalStream(
            pair.first_stream_ptr, device=device_index
        )
        self._multi_model_reasoner_stream = torch.cuda.ExternalStream(
            pair.second_stream_ptr, device=device_index
        )
        partition_policy = str(
            green_context.get("partition_policy", "static")
        )
        if partition_policy not in {"static", "dynamic"}:
            raise ValueError(
                "partition_policy must be 'static' or 'dynamic', "
                f"got {partition_policy!r}"
            )
        self._multi_model_partition_policy = partition_policy
        if partition_policy == "dynamic":
            self._multi_model_generator_full_stream = torch.cuda.Stream(
                device=device_index
            )
            # 创建资源仲裁器
            self._multi_model_resource_arbiter = MultiModelResourceArbiter(
                full_generator_stream=self._multi_model_generator_full_stream,
                shared_generator_stream=self._multi_model_generator_stream,
                full_primary_stream=self._create_reasoner_full_stream(
                    device_index
                ),
                shared_primary_stream=self._multi_model_reasoner_stream,
                generator_phase_policy=str(
                    green_context.get(
                        "generator_phase_policy", "demand_aware"
                    )
                ),
                trace_partition_nvtx=self._multi_model_trace_partition_nvtx,
            )
        self._multi_model_graph_patch = install_vllm_graph_capture_stream_patch(
            self._multi_model_reasoner_stream
        )
        self._multi_model_startup["resource_stream_setup_seconds"] = (
            time.perf_counter() - stream_setup_started
        )
        self._multi_model_startup_timeline["resource_stream_setup_end"] = (
            time.perf_counter()
        )

    def _create_reasoner_full_stream(
        self, device_index: int
    ) -> torch.cuda.Stream:
        stream = torch.cuda.Stream(device=device_index)
        self._multi_model_reasoner_full_stream = stream
        return stream

    @contextmanager
    def _multi_model_reasoner_context(self):
        stream = self._multi_model_reasoner_stream
        if stream is None:
            yield
        else:
            with torch.cuda.stream(stream):
                yield

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        self._multi_model_startup_timeline["model_load_start"] = started
        with self._multi_model_reasoner_context():
            result = super().load_model(*args, **kwargs)
        self._multi_model_startup["model_load_seconds"] = (
            time.perf_counter() - started
        )
        self._multi_model_startup_timeline["model_load_end"] = (
            time.perf_counter()
        )
        return result

    def determine_available_memory(self) -> int:
        started = time.perf_counter()
        self._multi_model_startup_timeline["memory_determination_start"] = started
        runner = self.model_runner
        phase_seconds: dict[str, float] = {}

        def timed_method(name: str, metric: str):
            original = getattr(runner, name, None)
            if original is None:
                return None

            def wrapped(*args: Any, **kwargs: Any):
                phase_started = time.perf_counter()
                self._multi_model_startup_timeline.setdefault(
                    f"{metric.removesuffix('_seconds')}_start", phase_started
                )
                try:
                    return original(*args, **kwargs)
                finally:
                    self._multi_model_startup_timeline[
                        f"{metric.removesuffix('_seconds')}_end"
                    ] = time.perf_counter()
                    phase_seconds[metric] = phase_seconds.get(metric, 0.0) + (
                        time.perf_counter() - phase_started
                    )

            setattr(runner, name, wrapped)
            return original

        # profile_run() consists primarily of an optional multimodal encoder
        # pass followed by the decoder/model dummy run.  The latter is also
        # where cached torch.compile/AOT artifacts are reconstructed, so do
        # not label the enclosing determine_available_memory() call as pure
        # "memory profiling".
        original_dummy_run = timed_method(
            "_dummy_run", "model_profile_and_compile_seconds"
        )
        original_mm_dummy_batch = timed_method(
            "_get_mm_dummy_batch", "encoder_input_preparation_seconds"
        )
        model = getattr(runner, "model", None)
        original_embed_multimodal = None
        if model is not None and hasattr(model, "embed_multimodal"):
            original_embed_multimodal = model.embed_multimodal

            def timed_embed_multimodal(*args: Any, **kwargs: Any):
                phase_started = time.perf_counter()
                self._multi_model_startup_timeline.setdefault(
                    "encoder_execution_start", phase_started
                )
                try:
                    return original_embed_multimodal(*args, **kwargs)
                finally:
                    self._multi_model_startup_timeline[
                        "encoder_execution_end"
                    ] = time.perf_counter()
                    phase_seconds["encoder_execution_seconds"] = (
                        phase_seconds.get("encoder_execution_seconds", 0.0)
                        + time.perf_counter()
                        - phase_started
                    )

            model.embed_multimodal = timed_embed_multimodal
        with self._multi_model_reasoner_context():
            try:
                result = super().determine_available_memory()
            finally:
                if original_dummy_run is not None:
                    runner._dummy_run = original_dummy_run
                if original_mm_dummy_batch is not None:
                    runner._get_mm_dummy_batch = original_mm_dummy_batch
                if original_embed_multimodal is not None:
                    model.embed_multimodal = original_embed_multimodal
        total = time.perf_counter() - started
        self._multi_model_startup_timeline["memory_determination_end"] = (
            time.perf_counter()
        )
        self._multi_model_startup["memory_determination_seconds"] = total
        self._multi_model_startup.update(phase_seconds)
        self._multi_model_startup["memory_determination_other_seconds"] = max(
            0.0, total - sum(phase_seconds.values())
        )
        self._multi_model_startup["memory_profiling_performed"] = not bool(
            self.cache_config.kv_cache_memory_bytes
        )
        return result

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        started = time.perf_counter()
        self._multi_model_startup_timeline["kv_cache_initialization_start"] = started
        with self._multi_model_reasoner_context():
            result = super().initialize_from_config(kv_cache_config)
        self._multi_model_startup["kv_cache_initialization_seconds"] = (
            time.perf_counter() - started
        )
        self._multi_model_startup_timeline["kv_cache_initialization_end"] = (
            time.perf_counter()
        )
        return result

    def compile_or_warm_up_model(self):
        dual_variants = self._configure_reasoner_graph_capture()
        runner = self.model_runner
        if dual_variants:
            if getattr(runner, "graph_variant_capture_provider", None) is not None:
                raise RuntimeError(
                    "GPUModelRunner already has a graph variant capture provider"
                )
            native_manager = getattr(runner, "cudagraph_manager", None)
            if native_manager is None:
                raise RuntimeError(
                    "Dual graph variants require a native CUDA Graph manager"
                )
            self._multi_model_graph_manager.register_model(
                self._multi_model_graph_owner, native_manager
            )
            self._multi_model_graph_manager.set_capture_handler(
                self._multi_model_graph_owner,
                runner,
                self._capture_reasoner_graph_variants,
            )
            runner.graph_variant_capture_provider = (
                self._multi_model_graph_manager
            )
        started = time.perf_counter()
        self._multi_model_startup_timeline[
            "primary_compile_warmup_capture_start"
        ] = started
        try:
            with self._multi_model_reasoner_context():
                result = super().compile_or_warm_up_model()
        finally:
            if dual_variants:
                runner.graph_variant_capture_provider = None
                self._multi_model_graph_manager.clear_capture_handler(
                    self._multi_model_graph_owner
                )
        primary_total = time.perf_counter() - started
        self._multi_model_startup_timeline[
            "primary_compile_warmup_capture_end"
        ] = time.perf_counter()
        additional_variant_setup = float(
            self._multi_model_graph_startup.get(
                "full_variant_capture_seconds", 0.0
            )
        ) + float(
            self._multi_model_graph_startup.get(
                "full_variant_cleanup_seconds", 0.0
            )
        )
        primary_setup = max(0.0, primary_total - additional_variant_setup)
        primary_capture = float(
            self._multi_model_graph_startup.get(
                "shared_graph_capture_seconds",
                getattr(self, "cuda_graph_capture_seconds", 0.0),
            )
        )
        self._multi_model_startup[
            "primary_compile_warmup_capture_seconds"
        ] = primary_setup
        self._multi_model_startup["compile_warmup_other_seconds"] = max(
            0.0, primary_setup - primary_capture
        )
        return result

    def _configure_reasoner_graph_capture(self) -> bool:
        """Resolve the complete graph-variant plan before capture starts."""
        config = self.vllm_config.additional_config
        runtime = (
            config.get("multi_model_runtime", {})
            if isinstance(config, dict)
            else {}
        )
        green = (
            runtime.get("green_context", {})
            if isinstance(runtime, dict)
            else {}
        )
        pool_mode = str(
            green.get("reasoner_graph_pool_mode", "shared")
        )
        if pool_mode not in {"isolated", "shared"}:
            raise ValueError(
                "reasoner_graph_pool_mode must be 'isolated' or 'shared', "
                f"got {pool_mode!r}"
            )
        self._multi_model_graph_pool_mode = pool_mode
        buffer_mode = str(
            green.get("reasoner_graph_buffer_mode", "isolated")
        )
        if buffer_mode not in {"isolated", "shared"}:
            raise ValueError(
                "reasoner_graph_buffer_mode must be 'isolated' or 'shared', "
                f"got {buffer_mode!r}"
            )
        self._multi_model_graph_buffer_mode = buffer_mode
        capture_mode = str(
            green.get("reasoner_graph_capture_mode", "full")
        )
        if capture_mode not in {
            "full",
            "piecewise",
            "breakable_piecewise",
        }:
            raise ValueError(
                "reasoner_graph_capture_mode must be 'full', 'piecewise', "
                f"or 'breakable_piecewise', got {capture_mode!r}"
            )
        self._multi_model_graph_capture_mode = capture_mode
        return bool(green.get("dual_reasoner_graph_variants", False))

    def _select_reasoner_graph_capture_mode(self, manager: Any) -> None:
        """让同一原生 manager 的所有资源变体使用等价 capture 计划。"""
        from vllm.compilation.cuda_graph import CUDAGraphMode

        mode = (
            CUDAGraphMode.FULL
            if self._multi_model_graph_capture_mode == "full"
            else CUDAGraphMode.PIECEWISE
        )
        manager._capture_descs = {
            mode: list(manager._capture_descs.get(mode, ()))
        }
        manager._candidates = {
            key: [desc for desc in descs if desc.cg_mode == mode]
            for key, descs in manager._candidates.items()
            if any(desc.cg_mode == mode for desc in descs)
        }

    def _capture_reasoner_graph_variants(
        self,
        runner: Any,
        capture_shared: Any,
        **kwargs: Any,
    ) -> int:
        """Capture and register every Reasoner variant in one trigger.

        vLLM still owns warmup and invokes this method once through its native
        ``capture_model`` hook.  The shared and full variants remain separate
        CUDA graph executables because they target different execution
        streams/Green Contexts.
        """
        native = getattr(runner, "cudagraph_manager", None)
        shared_stream = self._multi_model_reasoner_stream
        if native is None or shared_stream is None:
            raise RuntimeError("vLLM did not create the shared graph resources")
        self._select_reasoner_graph_capture_mode(native)

        shared_allocated_before = int(torch.cuda.memory_allocated(self.device))
        shared_reserved_before = int(torch.cuda.memory_reserved(self.device))
        shared_started = time.perf_counter()
        shared_device_bytes = self._multi_model_graph_manager.capture_variant(
            self._multi_model_shared_graph_key,
            CudaStreamBinding(shared_stream),
            capture_shared,
            **kwargs,
        )
        shared_capture_seconds = time.perf_counter() - shared_started
        self.cuda_graph_memory_bytes = shared_device_bytes
        self.cuda_graph_capture_seconds = shared_capture_seconds
        self.cuda_graph_allocated_delta_bytes = (
            int(torch.cuda.memory_allocated(self.device))
            - shared_allocated_before
        )
        self.cuda_graph_reserved_delta_bytes = (
            int(torch.cuda.memory_reserved(self.device))
            - shared_reserved_before
        )

        full_device_bytes = self._capture_reasoner_graph_variant(
            native
        )
        self._multi_model_graph_manager.activate(
            self.model_runner, self._multi_model_shared_graph_key
        )
        return shared_device_bytes + full_device_bytes

    def _capture_reasoner_graph_variant(
        self,
        native: Any,
    ) -> int:
        """Capture one additional FULL-decode resource variant."""
        if not self.use_v2_model_runner:
            raise RuntimeError("Dual graph variants require the V2 GPUModelRunner")
        full_stream = self._multi_model_reasoner_full_stream
        if full_stream is None:
            raise RuntimeError("A full-SM Reasoner stream was not created")
        runner = self.model_runner
        snapshot_after_shared = _cuda_memory_snapshot_summary()
        memory_history_status = "enabled"
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=100000,
            )
        except Exception as exc:  # pragma: no cover - version dependent
            memory_history_status = f"unavailable: {type(exc).__name__}: {exc}"

        from vllm.experimental.runtime_stream_patch import (
            install_vllm_graph_capture_stream_patch,
        )
        from vllm.v1.worker.gpu.lora_utils import create_lora_capture_hook

        # 同一原生 manager 天然共享 graph pool 和持久输入/输出 buffer；资源
        # 变体只区分 executable 与 capture/replay stream。
        self._multi_model_graph_pools_shared = True
        self._multi_model_graph_buffers_shared = True

        shared_patch = self._multi_model_graph_patch
        if shared_patch is not None:
            shared_patch.restore()
        full_patch = install_vllm_graph_capture_stream_patch(full_stream)
        torch.cuda.synchronize(self.device)
        capture_started = time.perf_counter()
        self._multi_model_startup_timeline["full_variant_capture_start"] = (
            capture_started
        )
        free_before = int(torch.accelerator.get_memory_info()[0])
        allocated_before = int(torch.cuda.memory_allocated(self.device))
        reserved_before = int(torch.cuda.memory_reserved(self.device))
        def capture_full_variant() -> int:
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
                use_aux_hidden_state_outputs=(
                    runner.use_aux_hidden_state_outputs
                ),
                lora_capture_hook=create_lora_capture_hook(
                    runner.lora_config, runner
                ),
                progress_bar_desc="Capturing full-SM Reasoner graphs",
            )
            full_stream.synchronize()
            runner.kv_connector.reset_capture_state()
            return 0

        try:
            self._multi_model_graph_manager.capture_variant(
                self._multi_model_full_graph_key,
                CudaStreamBinding(full_stream),
                capture_full_variant,
            )
        finally:
            full_patch.restore()
            self._multi_model_graph_patch = (
                install_vllm_graph_capture_stream_patch(
                    self._multi_model_reasoner_stream
                )
            )

        torch.cuda.synchronize(self.device)
        free_after = int(torch.accelerator.get_memory_info()[0])
        allocated_after = int(torch.cuda.memory_allocated(self.device))
        reserved_after = int(torch.cuda.memory_reserved(self.device))
        full_capture_seconds = time.perf_counter() - capture_started
        self._multi_model_startup_timeline["full_variant_capture_end"] = (
            time.perf_counter()
        )
        snapshot_after_full_before_cleanup = _cuda_memory_snapshot_summary()

        # Release ordinary unused allocator blocks left by the second capture
        # before reporting the steady-state serving footprint. Live graph-pool
        # allocations remain referenced by the native manager and must survive.
        cleanup_started = time.perf_counter()
        torch.accelerator.empty_cache()
        torch.cuda.synchronize(self.device)
        cleanup_seconds = time.perf_counter() - cleanup_started
        free_after_cleanup = int(torch.accelerator.get_memory_info()[0])
        allocated_after_cleanup = int(
            torch.cuda.memory_allocated(self.device)
        )
        reserved_after_cleanup = int(torch.cuda.memory_reserved(self.device))
        snapshot_after_full_after_cleanup = _cuda_memory_snapshot_summary()
        if memory_history_status == "enabled":
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as exc:  # pragma: no cover - version dependent
                memory_history_status = (
                    "enabled; disable failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        full_device_before_cleanup = free_before - free_after
        full_device_after_cleanup = free_before - free_after_cleanup
        self._multi_model_graph_memory = {
            "shared_graph_device_bytes": int(
                getattr(self, "cuda_graph_memory_bytes", 0)
            ),
            "shared_graph_allocated_delta_bytes": int(
                getattr(self, "cuda_graph_allocated_delta_bytes", 0)
            ),
            "shared_graph_reserved_delta_bytes": int(
                getattr(self, "cuda_graph_reserved_delta_bytes", 0)
            ),
            "full_variant_device_bytes_before_cleanup": (
                full_device_before_cleanup
            ),
            "full_variant_allocated_delta_bytes_before_cleanup": (
                allocated_after - allocated_before
            ),
            "full_variant_reserved_delta_bytes_before_cleanup": (
                reserved_after - reserved_before
            ),
            "full_variant_device_bytes_after_cleanup": (
                full_device_after_cleanup
            ),
            "full_variant_allocated_delta_bytes_after_cleanup": (
                allocated_after_cleanup - allocated_before
            ),
            "full_variant_reserved_delta_bytes_after_cleanup": (
                reserved_after_cleanup - reserved_before
            ),
            "full_variant_device_bytes_reclaimed": (
                free_after_cleanup - free_after
            ),
            "full_variant_reserved_bytes_reclaimed": (
                reserved_after - reserved_after_cleanup
            ),
            "total_graph_device_bytes_before_cleanup": int(
                getattr(self, "cuda_graph_memory_bytes", 0)
            )
            + full_device_before_cleanup,
            "total_graph_device_bytes_after_cleanup": int(
                getattr(self, "cuda_graph_memory_bytes", 0)
            )
            + full_device_after_cleanup,
        }
        self._multi_model_graph_startup = {
            "shared_graph_capture_seconds": float(
                getattr(self, "cuda_graph_capture_seconds", 0.0)
            ),
            "full_variant_capture_seconds": full_capture_seconds,
            "full_variant_cleanup_seconds": cleanup_seconds,
            "total_graph_capture_seconds": float(
                getattr(self, "cuda_graph_capture_seconds", 0.0)
            )
            + full_capture_seconds,
            "total_graph_setup_seconds": float(
                getattr(self, "cuda_graph_capture_seconds", 0.0)
            )
            + full_capture_seconds
            + cleanup_seconds,
        }
        self._multi_model_graph_snapshot = {
            "memory_history": memory_history_status,
            "shared_registry": _graph_pool_summary(native),
            "full_registry": _graph_pool_summary(native),
            "after_shared_capture": snapshot_after_shared,
            "after_full_capture_before_cleanup": (
                snapshot_after_full_before_cleanup
            ),
            "after_full_capture_after_cleanup": (
                snapshot_after_full_after_cleanup
            ),
        }

        counts = self._multi_model_graph_manager.counts(
            self._multi_model_graph_owner
        )
        shared_count = counts.get("shared", 0)
        full_count = counts.get("reasoner_full", 0)
        if shared_count <= 0 or full_count <= 0:
            raise RuntimeError(
                "Dual Reasoner graph capture produced an empty registry: "
                f"mode={self._multi_model_graph_capture_mode} "
                f"shared={shared_count} full={full_count}"
            )

        return full_device_after_cleanup

    def execute_model(self, scheduler_output: Any):
        # This snapshot is authored by EngineCore's Scheduler and travels with
        # the ordinary vLLM batch. Keep it at the Worker execution boundary so
        # device-side arbitration can consume Engine policy without reaching
        # back into Engine/Executor objects.
        execution_unit_state = getattr(
            scheduler_output, "execution_unit_state", None
        )
        if execution_unit_state is not None:
            self._multi_model_execution_unit_state = execution_unit_state
        arbiter = self._multi_model_resource_arbiter
        stream = self._multi_model_reasoner_stream
        epoch_id = None
        mode = "shared"
        if arbiter is not None:
            epoch_id = arbiter.note_primary_batch_submitted()
            selected, mode = arbiter.select_primary_batch_stream()
            if selected is not None:
                stream = selected
            if self._multi_model_graph_manager.has_variant(
                self._multi_model_shared_graph_key
            ):
                self._multi_model_graph_manager.activate(
                    self.model_runner,
                    (
                        self._multi_model_full_graph_key
                        if mode == "reasoner_full"
                        else self._multi_model_shared_graph_key
                    ),
                )
        previous = self._multi_model_active_reasoner_stream
        if previous is not None and previous is not stream:
            # A resource epoch changes only at a scheduler-batch boundary.
            # Drain both the prior compute lane and its asynchronous output
            # copies before rebinding GPUModelRunner.main_stream. AsyncOutput
            # records its handoff event on that cached stream; leaving it
            # bound to the previous GC lane lets the D2H copy race sampling
            # on the newly selected lane.
            switch_started = time.perf_counter()
            previous.synchronize()
            output_copy_stream = getattr(
                self.model_runner, "output_copy_stream", None
            )
            if output_copy_stream is not None:
                output_copy_stream.synchronize()
            self._multi_model_graph_variant_switches += 1
            self._multi_model_graph_variant_switch_sync_seconds += (
                time.perf_counter() - switch_started
            )
        if self._multi_model_last_graph_variant_mode not in (None, mode):
            self._multi_model_graph_registry_switches += 1
        self._multi_model_last_graph_variant_mode = mode
        self._multi_model_active_reasoner_stream = stream
        if stream is not None:
            # GPUModelRunner.main_stream is a cached_property. The normal
            # runner assumes a lifetime-stable execution stream, whereas this
            # worker deliberately changes streams at resource-epoch
            # boundaries. Override the cached value so AsyncOutput waits on
            # the stream that actually ran sampling for this batch.
            self.model_runner.main_stream = stream
        context = (
            torch.cuda.stream(stream)
            if stream is not None
            else nullcontext()
        )
        batch_index = self._multi_model_primary_batch_index
        self._multi_model_primary_batch_index += 1
        stream_handle = None if stream is None else int(stream.cuda_stream)
        epoch = "none" if epoch_id is None else str(epoch_id)
        label = (
            "reasoner/decode_batch/"
            f"batch={batch_index}/epoch={epoch}/mode={mode}/"
            f"stream={'default' if stream_handle is None else f'0x{stream_handle:x}'}"
        )
        trace_nvtx = self._multi_model_trace_partition_nvtx
        if trace_nvtx:
            torch.cuda.nvtx.range_push(label)
        primary_batch_finished = False
        try:
            with context:
                release = getattr(
                    self, "release_prepared_auxiliary_runners", None
                )
                if release is not None:
                    release()
                result = super().execute_model(scheduler_output)
                # This event is ordered after GPU work enqueued by this
                # scheduler batch. Do not synchronize it here: the marker is
                # observational and must not serialize normal decode.
                event = torch.cuda.Event()
                event.record(stream)
                self._multi_model_primary_last_submit_event = event
                if arbiter is not None:
                    arbiter.finish_primary_batch(mode, event)
                    primary_batch_finished = True
                if trace_nvtx:
                    torch.cuda.nvtx.mark(
                        "reasoner/gpu_submit_end/"
                        f"batch={batch_index}/epoch={epoch}/mode={mode}"
                    )
                return result
        finally:
            if arbiter is not None and not primary_batch_finished:
                # Unblock Generator admission on exceptional model execution.
                arbiter.finish_primary_batch(mode, None)
            if trace_nvtx:
                torch.cuda.nvtx.range_pop()

    def sample_tokens(self, grammar_output: Any):
        stream = (
            self._multi_model_active_reasoner_stream
            or self._multi_model_reasoner_stream
        )
        if stream is None:
            return super().sample_tokens(grammar_output)
        with torch.cuda.stream(stream):
            return super().sample_tokens(grammar_output)

    def acquire_generator_phase_lease(
        self,
        phase: str,
        step_index: int | None = None,
        selected_mode: str | None = None,
    ):
        arbiter = self._multi_model_resource_arbiter
        if arbiter is None:
            return None
        return arbiter.acquire_generator_phase(
            phase, step_index, selected_mode=selected_mode
        )

    def set_primary_model_demand(self, active: bool) -> dict[str, Any]:
        arbiter = self._multi_model_resource_arbiter
        if arbiter is None:
            raise RuntimeError("Dynamic Green Context arbitration is not enabled")
        if active:
            if self._multi_model_trace_partition_nvtx:
                torch.cuda.nvtx.mark("reasoner/admission_requested")
            arbiter.acquire_primary_demand()
            if self._multi_model_trace_partition_nvtx:
                torch.cuda.nvtx.mark("reasoner/request_admitted")
        else:
            event = self._multi_model_primary_last_submit_event
            drained = event is None or event.query()
            if self._multi_model_trace_partition_nvtx:
                torch.cuda.nvtx.mark(
                    "reasoner/request_returned/"
                    f"last_compute_event_drained={str(drained).lower()}"
                )
            arbiter.release_primary_demand()
            if self._multi_model_trace_partition_nvtx:
                torch.cuda.nvtx.mark("reasoner/demand_released")
        return arbiter.stats()

    def multi_model_resource_plan(self) -> dict[str, Any]:
        pair = self._multi_model_gc_pair
        if pair is None:
            return {"enabled": False}
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphWrapper,
            is_breakable_cudagraph_enabled,
        )

        patch = self._multi_model_graph_patch
        breakable_wrappers = list(BreakableCUDAGraphWrapper._all_instances)
        graph_variants_ready = self._multi_model_graph_manager.has_variant(
            self._multi_model_shared_graph_key
        )
        active_graph_variant = (
            self._multi_model_graph_manager.active_variant(
                self._multi_model_graph_owner
            )
            if graph_variants_ready
            else None
        )
        return {
            "enabled": True,
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "device": str(self.device),
            "requested_generator_sm_count": pair.requested_first_sm_count,
            "actual_generator_sm_count": pair.first_sm_count,
            "actual_reasoner_sm_count": pair.second_sm_count,
            "total_sm_count": pair.total_sm_count,
            "partition_policy": self._multi_model_partition_policy,
            "scheduler_execution_units": getattr(
                self, "_multi_model_execution_unit_state", None
            ),
            "generator_stream_handle": int(
                self._multi_model_generator_stream.cuda_stream
            ),
            "generator_full_stream_handle": (
                None
                if self._multi_model_generator_full_stream is None
                else int(self._multi_model_generator_full_stream.cuda_stream)
            ),
            "reasoner_stream_handle": int(
                self._multi_model_reasoner_stream.cuda_stream
            ),
            "reasoner_full_stream_handle": (
                None
                if self._multi_model_reasoner_full_stream is None
                else int(self._multi_model_reasoner_full_stream.cuda_stream)
            ),
            "graph_capture": None if patch is None else patch.stats(),
            "reasoner_graph_variants": (
                self._multi_model_graph_manager.counts(
                    self._multi_model_graph_owner
                )
                if graph_variants_ready
                else {}
            ),
            "reasoner_graph_capture_plan": {
                "orchestration": "single_vllm_capture_trigger",
                "mode": self._multi_model_graph_capture_mode,
                "variants": ["shared", "reasoner_full"],
                "active": (
                    None
                    if active_graph_variant is None
                    else active_graph_variant.variant_name
                ),
            },
            "reasoner_cudagraph": {
                "breakable_enabled": is_breakable_cudagraph_enabled(),
                "breakable_wrapper_count": len(breakable_wrappers),
                "breakable_entry_count": sum(
                    len(wrapper.entries) for wrapper in breakable_wrappers
                ),
                "breakable_capture_calls": sum(
                    wrapper.capture_calls for wrapper in breakable_wrappers
                ),
                "breakable_replay_calls": sum(
                    wrapper.replay_calls for wrapper in breakable_wrappers
                ),
                "breakable_eager_fallback_calls": sum(
                    wrapper.eager_fallback_calls
                    for wrapper in breakable_wrappers
                ),
                "all_variants_mode": self._multi_model_graph_capture_mode,
                "full_graphs_remain_standard": (
                    self._multi_model_graph_capture_mode == "full"
                ),
            },
            "reasoner_graph_pool": {
                "mode": self._multi_model_graph_pool_mode,
                "pools_shared": self._multi_model_graph_pools_shared,
                "registry_switches": (
                    self._multi_model_graph_registry_switches
                ),
                "stream_drains": (
                    self._multi_model_graph_variant_switches
                ),
                "stream_drain_seconds": (
                    self._multi_model_graph_variant_switch_sync_seconds
                ),
            },
            "reasoner_graph_buffers": {
                "mode": self._multi_model_graph_buffer_mode,
                "buffers_shared": self._multi_model_graph_buffers_shared,
            },
            "cuda_allocator": {
                "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "max_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
                "max_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.device)
                ),
            },
            "reasoner_graph_memory": dict(self._multi_model_graph_memory),
            "reasoner_graph_memory_snapshot": dict(
                self._multi_model_graph_snapshot
            ),
            "reasoner_graph_startup": dict(self._multi_model_graph_startup),
            "reasoner_worker_startup": dict(self._multi_model_startup),
            "reasoner_worker_startup_timeline": dict(
                self._multi_model_startup_timeline
            ),
            "dynamic_partition": (
                None
                if self._multi_model_resource_arbiter is None
                else self._multi_model_resource_arbiter.stats()
            ),
        }

    def shutdown(self) -> None:
        # Auxiliary runners are injected through MultiModelWorkerExtension,
        # rather than inherited directly.  Tear them down while the primary
        # distributed runtime and CUDA context are still valid.  The explicit
        # frontend close normally makes this a no-op; this is the worker-side
        # safety net for exceptions and engine-driven shutdown.
        close_auxiliary = getattr(self, "close_auxiliary_runners", None)
        try:
            if close_auxiliary is not None:
                close_auxiliary()
        finally:
            try:
                super().shutdown()
            finally:
                patch = self._multi_model_graph_patch
                if patch is not None:
                    patch.restore()
                    self._multi_model_graph_patch = None
                for stream in (
                    self._multi_model_generator_full_stream,
                    self._multi_model_generator_stream,
                    self._multi_model_reasoner_full_stream,
                    self._multi_model_reasoner_stream,
                ):
                    if stream is not None:
                        stream.synchronize()
                self._multi_model_generator_stream = None
                self._multi_model_generator_full_stream = None
                self._multi_model_reasoner_stream = None
                self._multi_model_reasoner_full_stream = None
                self._multi_model_active_reasoner_stream = None
                self._multi_model_resource_arbiter = None
                self._multi_model_graph_manager.clear()
                pair = self._multi_model_gc_pair
                if pair is not None:
                    pair.close()
                    self._multi_model_gc_pair = None

    def _shutdown_model_runner(self) -> None:
        """Destroy all remaining model instances through RunnerFactory.

        Runtime-added instances are normally removed by the frontend close
        path first. The startup vLLM instance remains in the same registry and
        reaches this hook at the native GPUWorker model-runner shutdown point.
        Draining the whole registry also makes exceptional shutdown safe.
        """
        registry = self._vllm_model_instance_registry
        instances = registry.drain()
        errors: list[BaseException] = []
        for instance in instances:
            try:
                self._vllm_runner_factory.destroy(instance)
            except BaseException as exc:
                errors.append(exc)
        # A partially initialized Worker may not have registered its startup
        # instance. Retain the native fallback for that exceptional case.
        if getattr(self, "model_runner", None) is not None:
            try:
                super()._shutdown_model_runner()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
