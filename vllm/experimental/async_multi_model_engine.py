# SPDX-License-Identifier: Apache-2.0
"""直接继承 AsyncLLM 的异步多模型 Engine。"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from contextvars import ContextVar
from typing import Any, Callable, Coroutine

from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment
from vllm.v1.engine.async_llm import AsyncLLM

_STREAM_END = object()
_MODEL_REQUEST_CONTEXT: ContextVar[tuple[str, str] | None] = ContextVar(
    "multi_model_request_context", default=None
)


class AsyncModelRequestHandle:
    """一个已经启动的异步模型请求。"""

    def __init__(
        self,
        model_id: str,
        request_id: str,
        task: asyncio.Task[Any],
        outputs: asyncio.Queue[Any],
    ) -> None:
        self.model_id = model_id
        self.request_id = request_id
        self._task = task
        self._outputs = outputs
        self._stream_claimed = False

    async def result(self, timeout: float | None = None) -> Any:
        """等待最终结果；超时不会自动取消底层请求。"""
        waitable = asyncio.shield(self._task)
        if timeout is None:
            return await waitable
        return await asyncio.wait_for(waitable, timeout)

    async def stream(self) -> AsyncIterator[Any]:
        """消费该请求的输出流；同一个 Handle 只允许消费一次。
            把请求结果以异步迭代器形式逐条返回
            主要用于返回 LLM tokens
        """
        if self._stream_claimed:
            raise RuntimeError("A model request stream can only be consumed once")
        self._stream_claimed = True
        while True:
            output = await self._outputs.get()
            if output is _STREAM_END:
                break
            yield output
        # 传播后台任务在输出流结束前记录的异常。
        await self._task

    def done(self) -> bool:
        return self._task.done()

    async def cancel(self) -> None:
        """取消本地等待；原生请求会由 generate() 执行 abort。"""
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task


class AsyncMultiModelEngine(AsyncLLM):
    """通过继承, 在原生 AsyncLLM 上增加部署查询和按模型路由的 submit 接口。"""

    WORKER_CLASS = (
        "vllm.experimental.worker.multi_model_worker.MultiModelWorker"
    )

    @classmethod
    def create(
        cls,
        deployment_plan: EngineDeploymentPlan,
        **engine_kwargs: Any,
    ) -> AsyncMultiModelEngine:
        """从部署计划创建 AsyncLLM/EngineCore/Executor/Worker 主链。"""
        from vllm.engine.arg_utils import AsyncEngineArgs

        # 这里目前,由于是继承,所以还是只允许一个native的vLLM
        # TODO: 这里的主从差异未来也要消灭?
        native = tuple(
            deployment
            for deployment in deployment_plan.deployments
            if deployment.spec.backend == "vllm"
        )
        if len(native) != 1:
            raise ValueError(
                "The compatibility AsyncLLM frontend currently requires "
                f"exactly one native vllm deployment, got {len(native)}"
            )
        
        # 配置worker
        configured_worker = engine_kwargs.get("worker_cls")
        if configured_worker not in (None, cls.WORKER_CLASS):
            raise ValueError(
                "AsyncMultiModelEngine requires its MultiModelWorker, got "
                f"{configured_worker!r}"
            )
        engine_kwargs["worker_cls"] = cls.WORKER_CLASS
        engine_kwargs.setdefault("model", native[0].spec.model_path)

        additional = dict(engine_kwargs.get("additional_config") or {})
        runtime = dict(additional.get("multi_model_runtime") or {})
        runtime["deployment_plan"] = deployment_plan.to_rpc_payload()
        additional["multi_model_runtime"] = runtime
        engine_kwargs["additional_config"] = additional
        return cls.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    @property
    def deployment_plan(self) -> EngineDeploymentPlan:
        additional = self.vllm_config.additional_config
        runtime = (
            additional.get("multi_model_runtime", {})
            if isinstance(additional, Mapping)
            else {}
        )
        payload = runtime.get("deployment_plan")
        if payload is None:
            raise RuntimeError("AsyncMultiModelEngine has no deployment plan")
        return EngineDeploymentPlan.from_rpc_payload(payload)

    @property
    def default_model_id(self) -> str:
        return self.deployment_plan.default_model_id

    def deployments(self) -> tuple[ModelDeployment, ...]:
        return self.deployment_plan.deployments

    def deployment(self, model_id: str) -> ModelDeployment:
        for deployment in self.deployment_plan.deployments:
            if deployment.spec.model_id == model_id:
                return deployment
        raise KeyError(f"Unknown model: {model_id!r}")

    def _native_output_stream(
        self, kwargs: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """调用继承自 AsyncLLM 的原生输出流。"""
        return AsyncLLM.generate(self, **kwargs)

    @staticmethod
    def _request_id(
        request: Mapping[str, Any] | Any,
        options: dict[str, Any],
    ) -> str:
        request_id = options.pop("request_id", None)
        if request_id is None and isinstance(request, Mapping):
            request_id = request.get("request_id")
        request_id = uuid.uuid4().hex if request_id is None else str(request_id)
        if not request_id:
            raise ValueError("request_id cannot be empty")
        return request_id

    @staticmethod
    def _start_task(
        coroutine: Coroutine[Any, Any, Any],
        outputs: asyncio.Queue[Any],
    ) -> asyncio.Task[Any]:
        async def run() -> Any:
            try:
                return await coroutine
            finally:
                await outputs.put(_STREAM_END)

        return asyncio.create_task(run())

    async def submit(
        self,
        model_id: str,
        request: Any,
        **options: Any,
    ) -> AsyncModelRequestHandle:
        """启动指定模型请求，并返回异步结果/流句柄。"""
        request_id = self._request_id(request, options)
        outputs: asyncio.Queue[Any] = asyncio.Queue()
        await self.execution_unit_rpc(
            "open_model_request", model_id, request_id
        )
        # 这里只选择输入/输出适配器；规范化请求的执行路由由 EngineCore 完成。
        deployment = self.deployment(model_id)
        if deployment.spec.backend == "vllm":
            return await self._run_vllm_preprocessor(
                model_id, request_id, request, options, outputs
            )
        return await self._run_container_preprocessor(
            model_id, request_id, request, options, outputs
        )

    async def _dispatch_prepared_request(self, request: Any) -> None:
        """把 AsyncLLM 生成的 EngineCoreRequest 交回 Core 统一路由。"""
        context = _MODEL_REQUEST_CONTEXT.get()
        if context is None:
            await super()._dispatch_prepared_request(request)
            return
        # 原生 vLLM 请求必须使用 EngineCoreClient 的强类型请求通道。经通用
        # utility RPC 的 ``Any`` 参数传输时，msgspec 会把 EngineCoreRequest
        # 还原成普通映射，Core 无法再执行原生 Scheduler 预处理。模型 session
        # 与 demand 已在 submit() 开头登记，因此这里直接复用 AsyncLLM 的
        # 标准 EngineCore 提交通道即可。
        await self.engine_core.add_request_async(request)

    async def _run_vllm_preprocessor(
        self,
        model_id: str,
        request_id: str,
        request: Any,
        options: dict[str, Any],
        outputs: asyncio.Queue[Any],
    ) -> AsyncModelRequestHandle:
        """使用原生 AsyncLLM 完成输入和输出处理。"""
        try:
            if not isinstance(request, Mapping):
                raise TypeError("A vLLM request must be a mapping")
            kwargs = dict(request)
            kwargs.pop("request_id", None)
            overlap = set(kwargs).intersection(options)
            if overlap:
                raise ValueError(
                    f"Duplicate native request options: {sorted(overlap)!r}"
                )
            kwargs.update(options)
            kwargs["request_id"] = request_id
            if "prompt" not in kwargs or "sampling_params" not in kwargs:
                raise ValueError(
                    "A native AsyncLLM request requires 'prompt' and "
                    "'sampling_params'"
                )
        except Exception:
            await self.execution_unit_rpc(
                "finish_model_request", model_id, request_id
            )
            raise

        async def execute_native() -> Any:
            last = None
            token = _MODEL_REQUEST_CONTEXT.set((model_id, request_id))
            try:
                async for output in self._native_output_stream(kwargs):
                    last = output
                    await outputs.put(output)
                return last
            finally:
                _MODEL_REQUEST_CONTEXT.reset(token)
                await self.execution_unit_rpc(
                    "finish_model_request", model_id, request_id
                )

        task = self._start_task(execute_native(), outputs)
        return AsyncModelRequestHandle(
            model_id, request_id, task, outputs
        )

    async def _run_container_preprocessor(
        self,
        model_id: str,
        request_id: str,
        request: Any,
        options: dict[str, Any],
        outputs: asyncio.Queue[Any],
    ) -> AsyncModelRequestHandle:
        """规范化 Container 请求，并交给 EngineCore 直接调度。"""
        try:
            await self.execution_unit_rpc(
                "submit_model_request",
                model_id,
                request_id,
                request,
                dict(options),
            )
        except Exception:
            await self.execution_unit_rpc(
                "finish_model_request", model_id, request_id
            )
            raise

        async def execute_worker_managed() -> Any:
            # EngineCore 返回 deferred utility Future；调度线程继续推进原生
            # 请求，Worker 完成后由既有 utility 输出通道主动通知 Frontend。
            try:
                result = await self.engine_core.call_utility_async(
                    "wait_model_request_completion",
                    request_id,
                )
                await outputs.put(result)
                return result
            finally:
                await self.execution_unit_rpc(
                    "finish_model_request", model_id, request_id
                )
        task = self._start_task(execute_worker_managed(), outputs)
        return AsyncModelRequestHandle(model_id, request_id, task, outputs)
