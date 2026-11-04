# SPDX-License-Identifier: Apache-2.0
"""Worker-owned model specifications and runtime instance registry."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class CommunicationGroupSpec:
    """Engine-planned communication domain materialized by each worker."""

    group_id: str
    parallel_dimension: str
    ranks: tuple[int, ...]
    backend: str | None = None
    use_device_communicator: bool = True

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | "CommunicationGroupSpec"
    ) -> "CommunicationGroupSpec":
        if isinstance(payload, cls):
            return payload
        values = dict(payload)
        values["ranks"] = tuple(int(rank) for rank in values["ranks"])
        return cls(**values)


@dataclass(frozen=True)
class ModelSpec:
    """Serializable description of one model instance.

    ``capabilities`` describes what a deployed model can serve. The concrete
    task remains request-scoped, so one registered model may accept multiple
    request types without being re-registered.
    """

    model_id: str
    model_path: str
    backend: str
    capabilities: tuple[str, ...] = ()
    default_task: str | None = None
    worker_ranks: tuple[int, ...] = ()
    communication_groups: tuple[CommunicationGroupSpec, ...] = ()
    load_format: str = "default"
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rpc_payload(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        values = dict(payload)
        capabilities = values.get("capabilities", ())
        values["capabilities"] = tuple(str(item) for item in capabilities)
        values["worker_ranks"] = tuple(
            int(rank) for rank in values.get("worker_ranks", ())
        )
        values["communication_groups"] = tuple(
            CommunicationGroupSpec.from_rpc_payload(group)
            for group in values.get("communication_groups", ())
        )
        return cls(**values)

    def to_rpc_payload(self) -> dict[str, Any]:
        return asdict(self)

    def with_placement(
        self,
        *,
        worker_ranks: tuple[int, ...],
        communication_groups: tuple[CommunicationGroupSpec, ...],
    ) -> "ModelSpec":
        return replace(
            self,
            worker_ranks=worker_ranks,
            communication_groups=communication_groups,
        )


@dataclass
class ModelInstance:
    """One Worker-local materialization of a :class:`ModelSpec`."""

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
    """Own every model instance materialized by one vLLM Worker."""

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

    def __contains__(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._instances
