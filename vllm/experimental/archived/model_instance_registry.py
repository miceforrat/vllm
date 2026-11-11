# SPDX-License-Identifier: Apache-2.0
"""旧 Worker-local ModelInstance 抽象；新路径已由 MiniContainerRegistry 取代。"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from vllm.experimental.model_instance_registry import ModelSpec


@dataclass
class ModelInstance:
    """旧实现中一个 Worker 本地的模型实例。"""

    spec: ModelSpec
    runner: Any
    adapter: Any | None = None
    scheduler_managed: bool = False
    runtime_added: bool = False
    communication_group_ids: tuple[str, ...] = ()
    pending_future: Future[Any] | None = None
    pending_start: threading.Event | None = None
    pending_started: threading.Event | None = None
    pending_job_id: str | None = None


class ModelInstanceRegistry:
    """旧 Worker 本地实例注册表；新路径使用 MiniContainerRegistry。"""

    def __init__(self) -> None:
        self._instances: dict[str, ModelInstance] = {}
        self._lock = threading.RLock()

    def register(self, instance: ModelInstance) -> None:
        model_id = instance.spec.model_id
        with self._lock:
            if model_id in self._instances:
                raise ValueError(f"Model {model_id!r} is already registered")
            self._instances[model_id] = instance

    def get(self, model_id: str) -> ModelInstance:
        with self._lock:
            try:
                return self._instances[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model: {model_id!r}") from exc

    def remove(self, model_id: str) -> ModelInstance:
        with self._lock:
            instance = self.get(model_id)
            if not instance.runtime_added:
                raise ValueError(
                    f"Model {model_id!r} is owned by the Engine lifecycle"
                )
            del self._instances[model_id]
            return instance

    def runtime_instances(self) -> tuple[ModelInstance, ...]:
        with self._lock:
            return tuple(
                instance
                for instance in self._instances.values()
                if instance.runtime_added
            )

    def runtime_items(self) -> tuple[tuple[str, ModelInstance], ...]:
        with self._lock:
            return tuple(
                (model_id, instance)
                for model_id, instance in self._instances.items()
                if instance.runtime_added
            )

    def values(self) -> tuple[ModelInstance, ...]:
        with self._lock:
            return tuple(self._instances.values())

    def drain(self) -> tuple[ModelInstance, ...]:
        with self._lock:
            instances = tuple(reversed(tuple(self._instances.values())))
            self._instances.clear()
            return instances

    def __contains__(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._instances
