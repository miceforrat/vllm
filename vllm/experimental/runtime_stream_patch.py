"""Explicit CUDA Graph capture-stream injection for in-process runtimes.

This is the minimal, version-local form of the stream patch already validated
by the dual-framework Cosmos3 runtime.  vLLM imports ``graph_capture`` by
value and has inner graph wrappers which independently query a capture stream,
so changing only the caller's current stream is insufficient.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class _Replacement:
    owner: Any
    name: str
    original: Callable[..., Any]
    replacement: Callable[..., Any]

    def restore(self) -> None:
        current = getattr(self.owner, self.name)
        if current is self.replacement:
            setattr(self.owner, self.name, self.original)
        elif current is not self.original:
            raise RuntimeError(f"{self.owner}.{self.name} changed after patch installation")


@dataclass
class GraphCaptureStreamPatch:
    stream: torch.cuda.Stream
    replacements: list[_Replacement]
    graph_capture_calls: int = 0
    manager_capture_stream_calls: int = 0
    compiled_capture_stream_calls: int = 0

    def restore(self) -> None:
        for replacement in reversed(self.replacements):
            replacement.restore()

    def stats(self) -> dict[str, int]:
        return {
            "stream_handle": int(self.stream.cuda_stream),
            "graph_capture_calls": self.graph_capture_calls,
            "manager_capture_stream_calls": self.manager_capture_stream_calls,
            "compiled_capture_stream_calls": self.compiled_capture_stream_calls,
        }


def install_vllm_graph_capture_stream_patch(
    stream: torch.cuda.Stream,
) -> GraphCaptureStreamPatch:
    """Force all vLLM graph-capture layers onto ``stream``.

    This mirrors the proven dual-framework patch.  It is intentionally scoped
    to capture; a captured graph retains the supplied stream for replay.
    """
    from vllm.compilation import cuda_graph as compilation_cuda_graph
    from vllm.distributed import parallel_state
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker.gpu import cudagraph_utils

    patch: GraphCaptureStreamPatch

    @contextmanager
    def graph_capture_on_external_stream(device):
        del device
        patch.graph_capture_calls += 1
        context = parallel_state.GraphCaptureContext(stream)
        with (
            parallel_state.get_tp_group().graph_capture(context),
            parallel_state.get_pp_group().graph_capture(context),
            parallel_state.get_dp_group().graph_capture(context),
        ):
            yield context

    replacements: list[_Replacement] = []
    for owner in (parallel_state, gpu_model_runner, cudagraph_utils):
        original = getattr(owner, "graph_capture")
        setattr(owner, "graph_capture", graph_capture_on_external_stream)
        replacements.append(
            _Replacement(owner, "graph_capture", original, graph_capture_on_external_stream)
        )

    manager = cudagraph_utils.ModelCudaGraphManager
    original_manager_stream = manager._capture_stream

    def capture_stream_on_external_green_context(self, desc):
        del self, desc
        patch.manager_capture_stream_calls += 1
        return stream

    manager._capture_stream = capture_stream_on_external_green_context
    replacements.append(
        _Replacement(
            manager,
            "_capture_stream",
            original_manager_stream,
            capture_stream_on_external_green_context,
        )
    )

    original_compiled_stream = compilation_cuda_graph.current_stream

    def compiled_capture_stream_on_external_green_context():
        patch.compiled_capture_stream_calls += 1
        return stream

    compilation_cuda_graph.current_stream = (
        compiled_capture_stream_on_external_green_context
    )
    replacements.append(
        _Replacement(
            compilation_cuda_graph,
            "current_stream",
            original_compiled_stream,
            compiled_capture_stream_on_external_green_context,
        )
    )
    patch = GraphCaptureStreamPatch(stream=stream, replacements=replacements)
    return patch
