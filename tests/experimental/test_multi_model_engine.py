# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment
from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.multi_model_engine import MultiModelEngine
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)


def _resources() -> ResourceDescription:
    return ResourceDescription(
        memory=MemoryResourceDescription("default"),
        execution_variants=(
            ExecutionVariantDescription(
                "default", StreamResourceDescription("default")
            ),
        ),
        default_variant="default",
    )


def _plan() -> EngineDeploymentPlan:
    return EngineDeploymentPlan(
        deployments=(
            ModelDeployment(
                ModelSpec("reasoner", "model", "vllm"), _resources()
            ),
            ModelDeployment(
                ModelSpec("generator", "model", "vllm_omni"), _resources()
            ),
        ),
        default_model_id="reasoner",
    )


class _Frontend:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.rpc_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda timeout=None: None),
            execution_unit_rpc=self._execution_unit_rpc,
        )

    def _execution_unit_rpc(self, method: str, *args: Any) -> Any:
        self.rpc_calls.append((method, args))
        if method == "open_model_request":
            model_id = str(args[0])
            if model_id not in {"reasoner", "generator"}:
                raise KeyError(f"Unknown model: {model_id!r}")
            return {"model_id": model_id, "request_id": args[1]}
        if method == "submit_model_request":
            return {"accepted": True}
        if method == "wait_model_request_completion":
            return [{"job_id": args[0], "rank": 0, "done": True}]
        return {"method": method}

    def generate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.generate_calls.append(kwargs)
        self.llm_engine.prepared_request_dispatcher({"prepared": True})
        return [{"text": "ok"}]

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del timeout
        assert kwargs is None
        self.rpc_calls.append((method, args))
        if method == "start_model_request":
            return [{"job_id": args[-1], "rank": 0}]
        return [{"job_id": args[0], "rank": 0, "done": True}]


def test_create_injects_plan_and_worker() -> None:
    captured: dict[str, Any] = {}
    frontend = _Frontend()

    def factory(**kwargs: Any) -> _Frontend:
        captured.update(kwargs)
        return frontend

    engine = MultiModelEngine.create(_plan(), engine_factory=factory)
    try:
        assert captured["model"] == "model"
        assert captured["worker_cls"] == MultiModelEngine.WORKER_CLASS
        payload = captured["additional_config"]["multi_model_runtime"][
            "deployment_plan"
        ]
        assert EngineDeploymentPlan.from_rpc_payload(payload) == _plan()
    finally:
        engine.shutdown()


def test_submit_routes_native_and_worker_managed_requests() -> None:
    frontend = _Frontend()
    engine = MultiModelEngine(frontend, _plan())
    try:
        native = engine.submit(
            "reasoner",
            {"prompts": ["hello"], "sampling_params": "params"},
            use_tqdm=False,
        )
        assert native.result() == [{"text": "ok"}]
        assert frontend.generate_calls == [
            {
                "prompts": ["hello"],
                "sampling_params": "params",
                "use_tqdm": False,
            }
        ]

        auxiliary = engine.submit(
            "generator",
            {"prompt": "video"},
            request_id="request-1",
            variant="auto",
        )
        assert auxiliary.result() == [
            {"job_id": "request-1", "rank": 0, "done": True}
        ]
        assert frontend.rpc_calls == [
            (
                "open_model_request",
                (
                    "reasoner",
                    native.request_id,
                ),
            ),
            (
                "submit_model_request",
                (
                    "reasoner",
                    native.request_id,
                    {"prepared": True},
                    {},
                ),
            ),
            ("finish_model_request", ("reasoner", native.request_id)),
            (
                "open_model_request",
                (
                    "generator",
                    "request-1",
                ),
            ),
            (
                "submit_model_request",
                (
                    "generator",
                    "request-1",
                    {"prompt": "video"},
                    {"variant": "auto"},
                ),
            ),
            ("wait_model_request_completion", ("request-1",)),
            ("finish_model_request", ("generator", "request-1")),
        ]
    finally:
        engine.shutdown()


def test_submit_rejects_unknown_model() -> None:
    engine = MultiModelEngine(_Frontend(), _plan())
    try:
        with pytest.raises(KeyError, match="missing"):
            engine.submit("missing", {})
    finally:
        engine.shutdown()
