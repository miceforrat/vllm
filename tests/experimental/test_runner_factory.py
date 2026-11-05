# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.experimental.communicator_registry import CommunicatorRegistry
from vllm.experimental.model_instance_registry import ModelInstance, ModelSpec
from vllm.experimental.runner_factory import RunnerFactory


class _Backend:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.destroyed: list[str] = []

    def create(
        self,
        spec: ModelSpec,
        *,
        worker: Any,
        communicators: CommunicatorRegistry,
    ) -> ModelInstance:
        self.created.append(spec.model_id)
        return ModelInstance(
            spec=spec,
            runner={"worker": worker},
            runtime_added=True,
        )

    def destroy(
        self,
        instance: ModelInstance,
        *,
        communicators: CommunicatorRegistry,
    ) -> None:
        self.destroyed.append(instance.spec.model_id)


def test_factory_routes_runtime_model_through_registered_backend() -> None:
    worker = object()
    factory = RunnerFactory(worker, CommunicatorRegistry())
    backend = _Backend()
    factory.register_backend("test", backend)
    spec = ModelSpec("generator", "model", "test")

    instance = factory.create(spec)
    factory.destroy(instance)

    assert instance.runner == {"worker": worker}
    assert backend.created == ["generator"]
    assert backend.destroyed == ["generator"]


def test_factory_routes_startup_vllm_model_through_backend() -> None:
    worker = object()
    factory = RunnerFactory(worker, CommunicatorRegistry())
    backend = _Backend()
    factory._backends["vllm"] = backend
    spec = ModelSpec("reasoner", "model", "vllm")

    instance = factory.create(spec)

    assert instance.runner == {"worker": worker}
    assert backend.created == ["reasoner"]


def test_factory_destroys_engine_owned_runner_through_backend() -> None:
    factory = RunnerFactory(object(), CommunicatorRegistry())
    backend = _Backend()
    factory._backends["vllm"] = backend
    instance = ModelInstance(
        spec=ModelSpec("reasoner", "model", "vllm"),
        runner=object(),
        scheduler_managed=True,
        runtime_added=False,
    )

    factory.destroy(instance)

    assert backend.destroyed == ["reasoner"]
