# SPDX-License-Identifier: Apache-2.0
"""MiniContainer 使用的模型后端生命周期实现。"""

from __future__ import annotations

from typing import Any

import torch

from vllm.experimental.worker.resource_manager import GraphVariantKey
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class NativeGPUWorkerLifecycle:
    """把原生 GPUWorker 生命周期与双 Graph 捕获接入 Container。"""

    def __init__(self, worker: Any, container: Any, graph_manager: Any) -> None:
        self._worker = worker
        self._container = container
        self._graph_manager = graph_manager

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        GPUWorker.load_model(
            self._worker, load_dummy_weights=load_dummy_weights
        )

    def determine_available_memory(self) -> int:
        return GPUWorker.determine_available_memory(self._worker)

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        GPUWorker.initialize_from_config(self._worker, kv_cache_config)

    def compile_or_warm_up_model(self) -> Any:
        container = self._container
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
            self._graph_manager.set_capture_handler(
                container.model_id,
                runner,
                self._capture_graph_variants,
            )
            runner.graph_variant_capture_provider = self._graph_manager
            provider_installed = True
        try:
            return GPUWorker.compile_or_warm_up_model(self._worker)
        finally:
            if provider_installed:
                runner.graph_variant_capture_provider = None
                self._graph_manager.clear_capture_handler(container.model_id)

    def _capture_graph_variants(
        self,
        runner: Any,
        capture_shared: Any,
        **kwargs: Any,
    ) -> int:
        container = self._container

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
        self._graph_manager.activate(
            runner,
            GraphVariantKey(container.model_id, default_variant),
        )
        return total

    @staticmethod
    def _capture_additional_graph(runner: Any) -> int:
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


class OmniModelLifecycle:
    """将 Omni Runner 的初始化阶段接入统一 Container 生命周期。"""

    def __init__(self, exec_unit: Any) -> None:
        self._exec_unit = exec_unit

    @property
    def _runner(self) -> Any:
        return self._exec_unit.runner

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        load_format = (
            "dummy" if load_dummy_weights else self._exec_unit.spec.load_format
        )
        providers = getattr(self._runner, "_container_runtime_providers", None)
        if providers is None:
            raise RuntimeError(
                f"Omni ExecUnit {self._exec_unit.model_id!r} has no bound "
                "runtime providers"
            )
        self._runner.load_model(
            runtime_providers=providers,
            load_format=load_format,
        )

    def determine_available_memory(self) -> int:
        raise NotImplementedError(
            "Omni memory profiling has not been connected to the unified "
            "Container lifecycle"
        )

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        del kv_cache_config
        raise NotImplementedError(
            "Omni does not use the vLLM KV-cache initialization protocol"
        )

    def compile_or_warm_up_model(self) -> Any:
        prepare = getattr(self._runner, "compile_or_warm_up_model", None)
        return None if prepare is None else prepare()


class ModelLifecycleFactory:
    """在 Container 创建边界解析一次模型后端生命周期差异。"""

    @staticmethod
    def create(exec_unit: Any) -> Any | None:
        if exec_unit.spec.backend == "vllm_omni":
            return OmniModelLifecycle(exec_unit)
        if exec_unit.spec.backend == "vllm":
            # 原生 vLLM 兼容生命周期需要 Worker 协议，由 Worker 在创建全部
            # Container 后显式接入。
            return None
        raise ValueError(
            f"No model lifecycle is registered for {exec_unit.spec.backend!r}"
        )
