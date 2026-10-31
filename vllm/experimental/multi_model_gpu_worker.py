# SPDX-License-Identifier: Apache-2.0
"""GPU worker with complementary Green Context execution streams."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class MultiModelGPUWorker(GPUWorker):
    """Bind primary and auxiliary runners to complementary GC streams."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._multi_model_gc_pair: Any | None = None
        self._multi_model_reasoner_stream: torch.cuda.Stream | None = None
        self._multi_model_generator_stream: torch.cuda.Stream | None = None
        self._multi_model_graph_patch: Any | None = None

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

        from vllm.experimental.green_context import GreenContextPair
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
        self._multi_model_graph_patch = install_vllm_graph_capture_stream_patch(
            self._multi_model_reasoner_stream
        )

    @contextmanager
    def _multi_model_reasoner_context(self):
        stream = self._multi_model_reasoner_stream
        if stream is None:
            yield
        else:
            with torch.cuda.stream(stream):
                yield

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        with self._multi_model_reasoner_context():
            return super().load_model(*args, **kwargs)

    def determine_available_memory(self) -> int:
        with self._multi_model_reasoner_context():
            return super().determine_available_memory()

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        with self._multi_model_reasoner_context():
            return super().initialize_from_config(kv_cache_config)

    def compile_or_warm_up_model(self):
        with self._multi_model_reasoner_context():
            return super().compile_or_warm_up_model()

    def execute_model(self, scheduler_output: Any):
        with self._multi_model_reasoner_context():
            release = getattr(
                self, "release_prepared_auxiliary_runners", None
            )
            if release is not None:
                release()
            return super().execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any):
        with self._multi_model_reasoner_context():
            return super().sample_tokens(grammar_output)

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
            "reasoner_stream_handle": int(
                self._multi_model_reasoner_stream.cuda_stream
            ),
            "graph_capture": None if patch is None else patch.stats(),
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
                    self._multi_model_generator_stream,
                    self._multi_model_reasoner_stream,
                ):
                    if stream is not None:
                        stream.synchronize()
                self._multi_model_generator_stream = None
                self._multi_model_reasoner_stream = None
                pair = self._multi_model_gc_pair
                if pair is not None:
                    pair.close()
                    self._multi_model_gc_pair = None
