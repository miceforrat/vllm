# SPDX-License-Identifier: Apache-2.0
"""复用原生 vLLM 前端的轻量多模型 Engine。"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment


class ModelRequestHandle:
    """统一表示原生 vLLM 请求或 Worker-managed 模型请求。"""

    def __init__(
        self,
        model_id: str,
        request_id: str,
        result_provider: Callable[[float | None], Any],
    ) -> None:
        self.model_id = model_id
        self.request_id = request_id
        self._result_provider = result_provider
        self._lock = threading.Lock()
        self._resolved = False
        self._result: Any = None
        self._error: BaseException | None = None

    def result(self, timeout: float | None = None) -> Any:
        """等待请求完成；重复调用返回同一个结果。"""
        with self._lock:
            if not self._resolved:
                try:
                    self._result = self._result_provider(timeout)
                except BaseException as exc:
                    self._error = exc
                self._resolved = True
            if self._error is not None:
                raise self._error
            return self._result


@dataclass(frozen=True)
class _NativeRequest:
    kwargs: dict[str, Any]


class MultiModelEngine:
    """在原生 ``LLM`` 之上统一部署查询和模型请求入口。

    本类不创建 Executor、Worker 资源或 execution-unit 调度器。原生 vLLM
    EngineCore 仍驱动 scheduler-managed 模型；Worker-managed 模型通过 Worker
    控制 RPC 启动，并由 Worker 内部的 MiniContainer/ResourceManager 推进。
    """

    WORKER_CLASS = (
        "vllm.experimental.worker.multi_model_worker.MultiModelWorker"
    )

    def __init__(
        self,
        frontend: Any,
        deployment_plan: EngineDeploymentPlan,
        *,
        native_request_workers: int = 1,
    ) -> None:
        if native_request_workers <= 0:
            raise ValueError("native_request_workers must be positive")
        self._frontend = frontend
        self._plan = deployment_plan
        self._deployments = {
            deployment.spec.model_id: deployment
            for deployment in deployment_plan.deployments
        }
        self._native_executor = ThreadPoolExecutor(
            max_workers=native_request_workers,
            thread_name_prefix="multi-model-native",
        )
        self._closed = False
        self._lock = threading.RLock()
        self._request_context = threading.local()
        llm_engine = getattr(frontend, "llm_engine", None)
        if llm_engine is not None:
            llm_engine.prepared_request_dispatcher = (
                self._dispatch_prepared_vllm_request
            )

    @classmethod
    def create(
        cls,
        deployment_plan: EngineDeploymentPlan,
        *,
        engine_factory: Callable[..., Any] | None = None,
        native_request_workers: int = 1,
        **engine_kwargs: Any,
    ) -> MultiModelEngine:
        """创建原生 ``LLM``，并注入完整部署计划。"""
        native = tuple(
            deployment
            for deployment in deployment_plan.deployments
            if deployment.spec.backend == "vllm"
        )
        if len(native) != 1:
            raise ValueError(
                "The compatibility vLLM frontend currently requires exactly "
                f"one native vllm deployment, got {len(native)}"
            )

        configured_worker = engine_kwargs.get("worker_cls")
        if configured_worker not in (None, cls.WORKER_CLASS):
            raise ValueError(
                "MultiModelEngine requires its MultiModelWorker, got "
                f"{configured_worker!r}"
            )
        engine_kwargs["worker_cls"] = cls.WORKER_CLASS
        engine_kwargs.setdefault("model", native[0].spec.model_path)

        additional = dict(engine_kwargs.get("additional_config") or {})
        runtime = dict(additional.get("multi_model_runtime") or {})
        runtime["deployment_plan"] = deployment_plan.to_rpc_payload()
        additional["multi_model_runtime"] = runtime
        engine_kwargs["additional_config"] = additional

        if engine_factory is None:
            from vllm import LLM

            engine_factory = LLM
        frontend = engine_factory(**engine_kwargs)

        # Engine 可能已经补全拓扑和模型路径；对外暴露其权威计划。
        llm_engine = getattr(frontend, "llm_engine", None)
        config = getattr(llm_engine, "vllm_config", None)
        config_additional = getattr(config, "additional_config", None)
        if isinstance(config_additional, Mapping):
            planned_payload = config_additional.get(
                "multi_model_runtime", {}
            ).get("deployment_plan")
            if planned_payload is not None:
                deployment_plan = EngineDeploymentPlan.from_rpc_payload(
                    planned_payload
                )
        return cls(
            frontend,
            deployment_plan,
            native_request_workers=native_request_workers,
        )

    @property
    def deployment_plan(self) -> EngineDeploymentPlan:
        return self._plan

    @property
    def default_model_id(self) -> str:
        return self._plan.default_model_id

    def deployments(self) -> tuple[ModelDeployment, ...]:
        return self._plan.deployments

    def get_tokenizer(self) -> Any:
        """复用原生 vLLM frontend 的 tokenizer。"""
        self._ensure_open()
        return self._frontend.get_tokenizer()

    def worker_rpc(
        self,
        method: str,
        *,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """供诊断和生命周期操作使用。"""
        self._ensure_open()
        return self._frontend.collective_rpc(
            method, timeout=timeout, args=args, kwargs=kwargs
        )

    def deployment(self, model_id: str) -> ModelDeployment:
        try:
            return self._deployments[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {model_id!r}") from exc

    @staticmethod
    def _native_request(request: Any, options: Mapping[str, Any]) -> _NativeRequest:
        if not isinstance(request, Mapping):
            raise TypeError(
                "A native vLLM request must be a mapping of LLM.generate() "
                "keyword arguments"
            )
        kwargs = dict(request)
        overlap = set(kwargs).intersection(options)
        if overlap:
            raise ValueError(
                f"Duplicate native request options: {sorted(overlap)!r}"
            )
        kwargs.update(options)
        if "prompts" not in kwargs:
            raise ValueError("A native vLLM request requires 'prompts'")
        return _NativeRequest(kwargs)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MultiModelEngine is closed")

    def submit(
        self,
        model_id: str,
        request: Any,
        **options: Any,
    ) -> ModelRequestHandle:
        """按 ``model_id`` 路由请求，并返回统一完成句柄。"""
        with self._lock:
            self._ensure_open()
            request_id = str(options.pop("request_id", uuid.uuid4().hex))
            if not request_id:
                raise ValueError("request_id cannot be empty")
            self._frontend.llm_engine.execution_unit_rpc(
                "open_model_request", model_id, request_id
            )
            # 这里只选择输入/输出适配器；执行路由由 EngineCore 完成。
            deployment = self.deployment(model_id)
            if deployment.spec.backend == "vllm":
                return self._run_vllm_preprocessor(
                    model_id, request_id, request, options
                )
            return self._run_container_preprocessor(
                model_id, request_id, request, options
            )

    def _dispatch_prepared_vllm_request(self, request: Any) -> None:
        """把 LLMEngine 预处理后的请求交回 EngineCore 统一路由。"""
        context = getattr(self._request_context, "current", None)
        if context is None:
            raise RuntimeError("No active multi-model request context")
        model_id, request_id = context
        self._frontend.llm_engine.execution_unit_rpc(
            "submit_model_request",
            model_id,
            request_id,
            request,
            {},
        )

    def _run_vllm_preprocessor(
        self,
        model_id: str,
        request_id: str,
        request: Any,
        options: dict[str, Any],
    ) -> ModelRequestHandle:
        """使用原生 LLMEngine 完成输入和输出处理。"""
        try:
            native = self._native_request(request, options)
        except Exception:
            self._frontend.llm_engine.execution_unit_rpc(
                "finish_model_request", model_id, request_id
            )
            raise

        def execute_native() -> Any:
            self._request_context.current = (model_id, request_id)
            try:
                return self._frontend.generate(**native.kwargs)
            finally:
                self._request_context.current = None
                self._frontend.llm_engine.execution_unit_rpc(
                    "finish_model_request", model_id, request_id
                )

        try:
            future: Future[Any] = self._native_executor.submit(execute_native)
        except Exception:
            self._frontend.llm_engine.execution_unit_rpc(
                "finish_model_request", model_id, request_id
            )
            raise
        return ModelRequestHandle(
            model_id,
            request_id,
            lambda timeout: future.result(timeout=timeout),
        )

    def _run_container_preprocessor(
        self,
        model_id: str,
        request_id: str,
        request: Any,
        options: dict[str, Any],
    ) -> ModelRequestHandle:
        """规范化 Container 请求，并交给 EngineCore 直接调度。"""
        try:
            self._frontend.llm_engine.execution_unit_rpc(
                "submit_model_request",
                model_id,
                request_id,
                request,
                dict(options),
            )
        except Exception:
            self._frontend.llm_engine.execution_unit_rpc(
                "finish_model_request", model_id, request_id
            )
            raise

        def wait(timeout: float | None) -> Any:
            try:
                result = self._frontend.llm_engine.execution_unit_rpc(
                    "wait_model_request_completion", request_id
                )
                if isinstance(result, Future):
                    return result.result(timeout=timeout)
                return result
            finally:
                self._frontend.llm_engine.execution_unit_rpc(
                    "finish_model_request", model_id, request_id
                )

        return ModelRequestHandle(model_id, request_id, wait)

    def shutdown(self, timeout: float | None = None) -> None:
        """停止前端请求线程，并关闭底层 EngineCore client。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._native_executor.shutdown(wait=True, cancel_futures=False)
        llm_engine = getattr(self._frontend, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if shutdown is not None:
            shutdown(timeout=timeout)

    def __enter__(self) -> MultiModelEngine:
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()
