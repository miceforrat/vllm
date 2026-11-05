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

    def drain(self) -> tuple[ModelInstance, ...]:
        """Remove and return every instance for Worker-wide shutdown.

        Runtime removal still goes through :meth:`remove` and therefore
        cannot remove an Engine-owned startup model. Worker shutdown is the
        one lifecycle boundary that owns every rank-local instance.
        """
        with self._lock:
            instances = tuple(reversed(tuple(self._instances.values())))
            self._instances.clear()
            return instances

    def __contains__(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._instances


class ModelRegistry:
    """Engine-owned registry of every deployed model specification.

    Unlike :class:`ModelInstanceRegistry`, this registry contains deployment
    intent rather than rank-local Runner objects.  Startup and runtime-added
    models enter through the same API; ``default_model_id`` is only a
    compatibility routing default and does not grant a model special
    lifecycle ownership.
    """

    def __init__(self, *, default_model_id: str | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._default_model_id = default_model_id
        self._lock = threading.RLock()

    @property
    def default_model_id(self) -> str:
        with self._lock:
            if self._default_model_id is None:
                raise RuntimeError("The Engine ModelRegistry has no default model")
            return self._default_model_id

    def register(self, spec: ModelSpec, *, make_default: bool = False) -> None:
        with self._lock:
            if spec.model_id in self._specs:
                raise ValueError(f"Model {spec.model_id!r} is already registered")
            self._specs[spec.model_id] = spec
            if make_default or self._default_model_id is None:
                self._default_model_id = spec.model_id

    def get(self, model_id: str) -> ModelSpec:
        with self._lock:
            try:
                return self._specs[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model: {model_id!r}") from exc

    def remove(self, model_id: str) -> ModelSpec:
        with self._lock:
            if model_id == self._default_model_id:
                raise ValueError("Cannot remove the compatibility default model")
            try:
                return self._specs.pop(model_id)
            except KeyError as exc:
                raise KeyError(f"Unknown model: {model_id!r}") from exc

    def values(self) -> tuple[ModelSpec, ...]:
        with self._lock:
            return tuple(self._specs.values())

    def __contains__(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._specs


def create_engine_model_registry(
    vllm_config: Any, topology_planner: Any
) -> ModelRegistry:
    """Create the Engine registry and register its startup model uniformly."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    runtime = additional.setdefault("multi_model_runtime", {})
    model_id = str(runtime.get("default_model_id", "reasoner"))
    model_config = vllm_config.model_config
    capabilities = ("text_generation",)
    spec = ModelSpec(
        model_id=model_id,
        model_path=str(model_config.model),
        backend="vllm",
        capabilities=capabilities,
        default_task=capabilities[0],
    )
    planned = topology_planner.plan(spec).apply(spec)
    registry = ModelRegistry(default_model_id=model_id)
    registry.register(planned, make_default=True)
    # The same Engine-owned deployment description is serialized with the
    # vLLM config and consumed by every Worker RunnerFactory.
    runtime["startup_models"] = [planned.to_rpc_payload()]
    return registry


def restore_engine_model_registry(vllm_config: Any) -> ModelRegistry:
    """Restore the EngineCore registry from Engine-planned startup specs."""
    additional = getattr(vllm_config, "additional_config", None)
    runtime = (
        additional.get("multi_model_runtime", {})
        if isinstance(additional, Mapping)
        else {}
    )
    payloads = runtime.get("startup_models", ())
    if not payloads:
        raise RuntimeError("Engine config contains no planned startup models")
    specs = tuple(ModelSpec.from_rpc_payload(payload) for payload in payloads)
    default_model_id = str(
        runtime.get("default_model_id", specs[0].model_id)
    )
    registry = ModelRegistry(default_model_id=default_model_id)
    for spec in specs:
        registry.register(
            spec, make_default=spec.model_id == default_model_id
        )
    return registry
