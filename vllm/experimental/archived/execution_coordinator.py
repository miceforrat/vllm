# SPDX-License-Identifier: Apache-2.0
"""已归档的 Worker-local 资源变体协调器。

动态分区决策现由 EngineCore ``ExecutionSliceScheduler`` 统一完成。本文件仅
保留旧实验实现，不能再由活动 Worker 路径导入。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from vllm.experimental.worker.mini_container import MiniContainer


class ContainerExecutionCoordinator:
    """协调多个 Container 的 demand、活跃请求和资源变体。

    这是 Worker-local UnitScheduler 的最小实现：它不理解 LLM 或 diffusion，
    只依据模型的资源描述，在 execution-unit 边界选择 ``shared`` 或独占变体。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._demand: dict[str, int] = {}
        self._active_requests: dict[str, int] = {}
        self._forced_variants: dict[str, str] = {}
        self._step_counts: dict[str, int] = {}

    def set_demand(self, model_id: str, active: bool) -> int:
        """声明或释放一个模型的待执行请求。"""
        with self._lock:
            count = self._demand.get(model_id, 0)
            count = count + 1 if active else max(0, count - 1)
            if count:
                self._demand[model_id] = count
            else:
                self._demand.pop(model_id, None)
            return count

    @contextmanager
    def request_scope(self, model_id: str) -> Iterator[None]:
        """标记一个模型请求正在由本 Worker 推进。"""
        with self._lock:
            self._active_requests[model_id] = (
                self._active_requests.get(model_id, 0) + 1
            )
        try:
            yield
        finally:
            with self._lock:
                count = self._active_requests.get(model_id, 0) - 1
                if count > 0:
                    self._active_requests[model_id] = count
                else:
                    self._active_requests.pop(model_id, None)

    def force_variant(self, model_id: str, variant_name: str | None) -> None:
        """设置模型的显式变体；传入 ``None`` 恢复自动选择。"""
        with self._lock:
            if variant_name is None:
                self._forced_variants.pop(model_id, None)
            else:
                self._forced_variants[model_id] = variant_name

    def select_variant(
        self,
        container: MiniContainer,
        requested: str | None,
    ) -> str | None:
        """在 execution-unit 边界选择已声明的资源变体。"""
        if requested not in (None, "auto"):
            return requested
        description = container.resource_description
        if description is None:
            return None
        model_id = container.model_id
        with self._lock:
            forced = self._forced_variants.get(model_id)
            contended = any(
                other != model_id and count > 0
                for source in (self._demand, self._active_requests)
                for other, count in source.items()
            )
        available = {
            variant.name for variant in description.execution_variants
        }
        if forced is not None:
            return forced
        if contended and "shared" in available:
            return "shared"
        full_candidates = (
            "full",
            f"{model_id}_full",
            "reasoner_full",
            "generator_full",
        )
        for candidate in full_candidates:
            if candidate in available:
                return candidate
        return description.default_variant

    def record_step(self, model_id: str, variant_name: str | None) -> None:
        name = "default" if variant_name is None else variant_name
        key = f"{model_id}:{name}"
        with self._lock:
            self._step_counts[key] = self._step_counts.get(key, 0) + 1

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "demand": dict(self._demand),
                "active_requests": dict(self._active_requests),
                "forced_variants": dict(self._forced_variants),
                "variant_step_counts": dict(self._step_counts),
            }
