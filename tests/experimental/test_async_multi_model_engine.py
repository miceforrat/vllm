# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from typing import Any

from vllm.experimental.async_multi_model_engine import AsyncMultiModelEngine
from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment
from vllm.experimental.model_instance_registry import ModelSpec
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


class _CoreClient:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def call_utility_async(self, method: str, *args: Any) -> Any:
        self.owner.utility_calls.append((method, args))
        if method == "open_model_request":
            return {"model_id": args[0], "request_id": args[1]}
        if method == "submit_model_request":
            return {"accepted": True}
        if method == "finish_model_request":
            return {"finished": args[1]}
        if method == "wait_model_request_completion":
            return [{"job_id": args[0], "rank": 0, "done": True}]
        raise AssertionError(f"Unexpected utility method: {method}")


class _Engine(AsyncMultiModelEngine):
    def __init__(self) -> None:
        self.vllm_config = SimpleNamespace(
            additional_config={
                "multi_model_runtime": {
                    "deployment_plan": _plan().to_rpc_payload()
                }
            }
        )
        self.rpc_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.utility_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.native_kwargs: dict[str, Any] | None = None
        self.engine_core = _CoreClient(self)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del timeout, kwargs
        self.rpc_calls.append((method, args))
        if method == "start_model_request":
            return [{"job_id": args[-1], "rank": 0}]
        return [{"rank": 0}]

    async def _fake_native_stream(self):
        await self._dispatch_prepared_request({"prepared": True})
        yield {"text": "partial", "finished": False}
        yield {"text": "complete", "finished": True}

    def _native_output_stream(
        self, kwargs: dict[str, Any]
    ):
        self.native_kwargs = kwargs
        return self._fake_native_stream()

    def shutdown(self, timeout: float | None = None) -> None:
        del timeout


def test_native_submit_preserves_async_output_stream() -> None:
    async def run() -> None:
        engine = _Engine()
        handle = await engine.submit(
            "reasoner",
            {"prompt": "hello", "sampling_params": "params"},
            request_id="reasoner-1",
        )
        outputs = [output async for output in handle.stream()]
        assert outputs == [
            {"text": "partial", "finished": False},
            {"text": "complete", "finished": True},
        ]
        assert await handle.result() == outputs[-1]
        assert engine.native_kwargs == {
            "prompt": "hello",
            "sampling_params": "params",
            "request_id": "reasoner-1",
        }
        assert engine.utility_calls == [
            (
                "open_model_request",
                (
                    "reasoner",
                    "reasoner-1",
                ),
            ),
            (
                "submit_model_request",
                (
                    "reasoner",
                    "reasoner-1",
                    {"prepared": True},
                    {},
                ),
            ),
            ("finish_model_request", ("reasoner", "reasoner-1")),
        ]

    asyncio.run(run())


def test_worker_managed_submit_uses_async_collective_rpc() -> None:
    async def run() -> None:
        engine = _Engine()
        handle = await engine.submit(
            "generator",
            {"prompt": "video"},
            request_id="generator-1",
            variant="auto",
        )
        assert await handle.result() == [
            {"job_id": "generator-1", "rank": 0, "done": True}
        ]
        assert engine.rpc_calls == []
        assert engine.utility_calls == [
            (
                "open_model_request",
                (
                    "generator",
                    "generator-1",
                ),
            ),
            (
                "submit_model_request",
                (
                    "generator",
                    "generator-1",
                    {"prompt": "video"},
                    {"variant": "auto"},
                ),
            ),
            ("wait_model_request_completion", ("generator-1",))
            ,
            ("finish_model_request", ("generator", "generator-1")),
        ]

    asyncio.run(run())
