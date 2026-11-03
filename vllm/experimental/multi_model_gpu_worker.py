# SPDX-License-Identifier: Apache-2.0
"""GPU worker with complementary Green Context execution streams."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class MultiModelGPUWorker(GPUWorker):
    """Bind primary and auxiliary runners to complementary GC streams."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._multi_model_gc_pair: Any | None = None
        self._multi_model_reasoner_stream: torch.cuda.Stream | None = None
        self._multi_model_reasoner_full_stream: torch.cuda.Stream | None = None
        self._multi_model_active_reasoner_stream: torch.cuda.Stream | None = None
        self._multi_model_generator_stream: torch.cuda.Stream | None = None
        self._multi_model_generator_full_stream: torch.cuda.Stream | None = None
        self._multi_model_graph_patch: Any | None = None
        self._multi_model_resource_arbiter: Any | None = None
        self._multi_model_shared_graph_manager: Any | None = None
        self._multi_model_full_graph_manager: Any | None = None
        self._multi_model_graph_memory: dict[str, int] = {}
        self._multi_model_graph_startup: dict[str, float] = {}
        self._multi_model_startup: dict[str, float] = {}
        self._multi_model_primary_batch_index = 0
        self._multi_model_primary_last_submit_event: torch.cuda.Event | None = None
        self._multi_model_trace_partition_nvtx = False

    def init_device(self) -> None:
        super().init_device()
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

        from vllm.experimental.green_context import GreenContextPair
        from vllm.experimental.multi_model_arbiter import (
            MultiModelResourceArbiter,
        )
        from vllm.experimental.runtime_stream_patch import (
            install_vllm_graph_capture_stream_patch,
        )

        requested = int(green_context["generator_sm_count"])
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        pair = GreenContextPair(requested, device_index)
        self._multi_model_gc_pair = pair
        self._multi_model_generator_stream = torch.cuda.ExternalStream(
            pair.first_stream_ptr, device=device_index
        )
        self._multi_model_reasoner_stream = torch.cuda.ExternalStream(
            pair.second_stream_ptr, device=device_index
        )
        if bool(green_context.get("dynamic_step_scheduling", False)):
            self._multi_model_generator_full_stream = torch.cuda.Stream(
                device=device_index
            )
            self._multi_model_resource_arbiter = MultiModelResourceArbiter(
                full_generator_stream=self._multi_model_generator_full_stream,
                shared_generator_stream=self._multi_model_generator_stream,
                full_primary_stream=(
                    self._create_reasoner_full_stream(device_index)
                    if bool(
                        green_context.get(
                            "dynamic_reasoner_stream", False
                        )
                    )
                    else None
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
        with self._multi_model_reasoner_context():
            result = super().load_model(*args, **kwargs)
        self._multi_model_startup["model_load_seconds"] = (
            time.perf_counter() - started
        )
        return result

    def determine_available_memory(self) -> int:
        started = time.perf_counter()
        with self._multi_model_reasoner_context():
            result = super().determine_available_memory()
        self._multi_model_startup["memory_profiling_seconds"] = (
            time.perf_counter() - started
        )
        return result

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        started = time.perf_counter()
        with self._multi_model_reasoner_context():
            result = super().initialize_from_config(kv_cache_config)
        self._multi_model_startup["kv_cache_initialization_seconds"] = (
            time.perf_counter() - started
        )
        return result

    def compile_or_warm_up_model(self):
        started = time.perf_counter()
        with self._multi_model_reasoner_context():
            result = super().compile_or_warm_up_model()
        primary_total = time.perf_counter() - started
        primary_capture = float(
            getattr(self, "cuda_graph_capture_seconds", 0.0)
        )
        self._multi_model_startup[
            "primary_compile_warmup_capture_seconds"
        ] = primary_total
        self._multi_model_startup["compile_warmup_other_seconds"] = max(
            0.0, primary_total - primary_capture
        )
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
        if bool(green.get("dual_reasoner_graph_variants", False)):
            self._capture_full_reasoner_graph_variant()
        return result

    def _capture_full_reasoner_graph_variant(self) -> None:
        """Capture a second FULL-decode registry on the full-SM stream."""
        if not self.use_v2_model_runner:
            raise RuntimeError("Dual graph variants require the V2 GPUModelRunner")
        full_stream = self._multi_model_reasoner_full_stream
        if full_stream is None:
            raise RuntimeError("A full-SM Reasoner stream was not created")
        runner = self.model_runner
        shared = getattr(runner, "cudagraph_manager", None)
        if shared is None:
            raise RuntimeError("vLLM did not initialize a CUDA graph manager")

        from vllm.compilation.cuda_graph import CUDAGraphMode
        from vllm.experimental.runtime_stream_patch import (
            install_vllm_graph_capture_stream_patch,
        )
        from vllm.platforms import current_platform
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        from vllm.v1.worker.gpu.lora_utils import create_lora_capture_hook

        full = ModelCudaGraphManager(
            shared.vllm_config,
            shared.device,
            shared.cudagraph_mode,
            decode_query_len=shared.decode_query_len,
            lora_capture_cases=shared.lora_capture_cases,
            varlen_decode=shared.varlen_decode,
            ubatch_runner=shared.ubatch_runner,
        )
        # Graph-pool sharing is only safe when graphs are replayed in the same
        # order in which their allocations were captured. Resource variants
        # are selected independently, so the full-SM registry must not alias
        # static allocations owned by the shared-GC registry.
        full.pool = current_platform.graph_pool_handle()
        full._capture_descs = {
            CUDAGraphMode.FULL: list(
                full._capture_descs.get(CUDAGraphMode.FULL, ())
            )
        }
        full._candidates = {
            key: [d for d in descs if d.cg_mode == CUDAGraphMode.FULL]
            for key, descs in full._candidates.items()
            if any(d.cg_mode == CUDAGraphMode.FULL for d in descs)
        }

        shared_patch = self._multi_model_graph_patch
        if shared_patch is not None:
            shared_patch.restore()
        full_patch = install_vllm_graph_capture_stream_patch(full_stream)
        torch.cuda.synchronize(self.device)
        capture_started = time.perf_counter()
        free_before = int(torch.accelerator.get_memory_info()[0])
        allocated_before = int(torch.cuda.memory_allocated(self.device))
        reserved_before = int(torch.cuda.memory_reserved(self.device))
        try:
            input_buffers = runner.input_buffers
            if runner.pcp_manager is not None:
                input_buffers = runner.pcp_manager.input_buffers
            with torch.cuda.stream(full_stream):
                full.capture(
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

        # Release ordinary unused allocator blocks left by the second capture
        # before reporting the steady-state serving footprint. Live graph-pool
        # allocations remain referenced by ``full.graphs`` and must survive.
        cleanup_started = time.perf_counter()
        torch.accelerator.empty_cache()
        torch.cuda.synchronize(self.device)
        cleanup_seconds = time.perf_counter() - cleanup_started
        free_after_cleanup = int(torch.accelerator.get_memory_info()[0])
        allocated_after_cleanup = int(
            torch.cuda.memory_allocated(self.device)
        )
        reserved_after_cleanup = int(torch.cuda.memory_reserved(self.device))

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

        if not shared.graphs or not full.graphs:
            raise RuntimeError(
                "Dual Reasoner graph capture produced an empty registry: "
                f"shared={len(shared.graphs)} full={len(full.graphs)}"
            )

        self._multi_model_shared_graph_manager = shared
        self._multi_model_full_graph_manager = full
        runner.cudagraph_manager = shared

    def execute_model(self, scheduler_output: Any):
        arbiter = self._multi_model_resource_arbiter
        stream = self._multi_model_reasoner_stream
        epoch_id = None
        mode = "shared"
        if arbiter is not None:
            epoch_id = arbiter.note_primary_batch_submitted()
            selected, mode = arbiter.select_primary_batch_stream()
            if selected is not None:
                stream = selected
            if self._multi_model_shared_graph_manager is not None:
                self.model_runner.cudagraph_manager = (
                    self._multi_model_full_graph_manager
                    if mode == "reasoner_full"
                    else self._multi_model_shared_graph_manager
                )
        previous = self._multi_model_active_reasoner_stream
        if previous is not None and previous is not stream:
            # A resource epoch changes only at a scheduler-batch boundary.
            # Drain both the prior compute lane and its asynchronous output
            # copies before rebinding GPUModelRunner.main_stream. AsyncOutput
            # records its handoff event on that cached stream; leaving it
            # bound to the previous GC lane lets the D2H copy race sampling
            # on the newly selected lane.
            previous.synchronize()
            output_copy_stream = getattr(
                self.model_runner, "output_copy_stream", None
            )
            if output_copy_stream is not None:
                output_copy_stream.synchronize()
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
        self, phase: str, step_index: int | None = None
    ):
        arbiter = self._multi_model_resource_arbiter
        if arbiter is None:
            return None
        return arbiter.acquire_generator_phase(phase, step_index)

    def acquire_generator_step_lease(self, step_index: int):
        return self.acquire_generator_phase_lease("denoising", step_index)

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
        patch = self._multi_model_graph_patch
        return {
            "enabled": True,
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "device": str(self.device),
            "requested_generator_sm_count": pair.requested_first_sm_count,
            "actual_generator_sm_count": pair.first_sm_count,
            "actual_reasoner_sm_count": pair.second_sm_count,
            "total_sm_count": pair.total_sm_count,
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
            "reasoner_graph_variants": {
                "shared": (
                    0
                    if self._multi_model_shared_graph_manager is None
                    else len(self._multi_model_shared_graph_manager.graphs)
                ),
                "reasoner_full": (
                    0
                    if self._multi_model_full_graph_manager is None
                    else len(self._multi_model_full_graph_manager.graphs)
                ),
            },
            "reasoner_graph_memory": dict(self._multi_model_graph_memory),
            "reasoner_graph_startup": dict(self._multi_model_graph_startup),
            "reasoner_worker_startup": dict(self._multi_model_startup),
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
                self._multi_model_shared_graph_manager = None
                self._multi_model_full_graph_manager = None
                pair = self._multi_model_gc_pair
                if pair is not None:
                    pair.close()
                    self._multi_model_gc_pair = None
