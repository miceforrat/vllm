# SPDX-License-Identifier: Apache-2.0
"""多模型请求的单一所有者控制面。"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.experimental.execution_slice import ExecutionSliceDescriptor


class RequestOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    model_id: str
    backend: str
    variant: str
    demand_active: bool
    terminal: bool
    outcome: RequestOutcome | None


@dataclass
class _RequestRecord:
    request_id: str
    model_id: str
    backend: str
    variant: str = "auto"
    future: Future[list[Any]] | None = None
    demand_active: bool = True
    terminal: bool = False
    outcome: RequestOutcome | None = None

    def snapshot(self) -> RequestSnapshot:
        return RequestSnapshot(
            request_id=self.request_id,
            model_id=self.model_id,
            backend=self.backend,
            variant=self.variant,
            demand_active=self.demand_active,
            terminal=self.terminal,
            outcome=self.outcome,
        )


@dataclass
class _ControlMessage:
    operation: str
    args: tuple[Any, ...]
    reply: Future[Any]


class RequestControlPlane:
    """在单一控制线程中管理所有请求状态。"""

    def __init__(self, execution_slices: Any) -> None:
        self._execution_slices = execution_slices
        self._records: dict[str, _RequestRecord] = {}
        self._messages: queue.Queue[_ControlMessage | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="model-request-control",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while (message := self._messages.get()) is not None:
            try:
                method = getattr(self, f"_handle_{message.operation}")
                message.reply.set_result(method(*message.args))
            except BaseException as error:
                message.reply.set_exception(error)

    def _call(self, operation: str, *args: Any) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("Model request control thread is not running")
        reply: Future[Any] = Future()
        self._messages.put(_ControlMessage(operation, args, reply))
        return reply.result()

    def open(
        self, request_id: str, model_id: str, backend: str
    ) -> RequestSnapshot:
        return self._call("open", request_id, model_id, backend)

    def _handle_open(
        self, request_id: str, model_id: str, backend: str
    ) -> RequestSnapshot:
        if request_id in self._records:
            raise ValueError(f"Model request {request_id!r} already exists")
        record = _RequestRecord(request_id, model_id, backend)
        self._records[request_id] = record
        self._execution_slices.set_model_demand(model_id, True)
        return record.snapshot()

    def prepare_submit(
        self, request_id: str, model_id: str, variant: str
    ) -> Future[list[Any]] | None:
        return self._call("prepare_submit", request_id, model_id, variant)

    def _handle_prepare_submit(
        self, request_id: str, model_id: str, variant: str
    ) -> Future[list[Any]] | None:
        record = self._require(request_id)
        if record.model_id != model_id:
            raise RuntimeError(
                f"Model request {request_id!r} belongs to "
                f"{record.model_id!r}, not {model_id!r}"
            )
        if record.terminal:
            raise RuntimeError(f"Model request {request_id!r} is finished")
        if record.future is not None:
            raise RuntimeError(f"Model request {request_id!r} was submitted")
        record.variant = variant
        if record.backend == "vllm":
            return None
        record.future = Future()
        return record.future

    def future(self, request_id: str) -> Future[list[Any]]:
        return self._call("future", request_id)

    def _handle_future(self, request_id: str) -> Future[list[Any]]:
        future = self._require(request_id).future
        if future is None:
            raise KeyError(f"Model request {request_id!r} has no completion Future")
        return future

    def plan_slice(
        self,
        model_id: str,
        driver_request_id: str,
        requested_variant: str,
        boundary: dict[str, Any],
    ) -> dict[str, Any]:
        return self._call(
            "plan_slice",
            model_id,
            driver_request_id,
            requested_variant,
            boundary,
        )

    def _handle_plan_slice(
        self,
        model_id: str,
        driver_request_id: str,
        requested_variant: str,
        boundary: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(boundary.get("metadata", {}))
        request_ids = tuple(str(value) for value in metadata.get("request_ids", ()))
        variants = {
            self._require(request_id).variant
            for request_id in request_ids
            if request_id in self._records
        }
        explicit = variants - {"auto"}
        if len(explicit) > 1:
            raise RuntimeError(
                "One execution wave contains conflicting resource variants: "
                f"{sorted(explicit)!r} for requests {request_ids!r}"
            )
        variant = next(iter(explicit)) if explicit else requested_variant
        if variant == "auto" and variants:
            variant = "auto"
        descriptor = ExecutionSliceDescriptor.from_payload(boundary)
        scheduled = self._execution_slices.schedule_slice(
            model_id,
            driver_request_id,
            ExecutionSliceDescriptor(
                sequence_no=descriptor.sequence_no,
                metadata={**metadata, "requested_variant": variant},
                tags=descriptor.tags,
            ),
        )
        return scheduled.to_payload()

    def report_workers(
        self, workers: list[dict[str, Any]], driver_request_id: str
    ) -> bool:
        return self._call("report_workers", workers, driver_request_id)

    def _handle_report_workers(
        self, workers: list[dict[str, Any]], driver_request_id: str
    ) -> bool:
        if not workers:
            raise RuntimeError("Model emitter received no Worker response")
        finished_ids = tuple(
            str(value) for value in workers[0].get("finished_request_ids", ())
        )
        if not finished_ids and bool(workers[0].get("request_done")):
            finished_ids = (driver_request_id,)
        terminal_requests = dict(
            workers[0].get("metadata", {}).get("terminal_requests", {})
        )
        for request_id in finished_ids:
            terminal = terminal_requests.get(request_id)
            status = None if terminal is None else terminal.get("status")
            error = None if terminal is None else terminal.get("error")
            if terminal is not None and (
                status != "FINISHED_COMPLETED" or error is not None
            ):
                detail = error or status or "unknown terminal error"
                self._finish(
                    request_id,
                    RequestOutcome.FAILED,
                    error=RuntimeError(
                        f"Model request {request_id!r} failed "
                        f"({status or 'UNKNOWN'}): {detail}"
                    ),
                )
                continue
            request_workers: list[dict[str, Any]] = []
            for worker in workers:
                output = dict(worker)
                for source, target in (
                    ("finished_seconds", "seconds"),
                    ("per_request_step_counts", "step_count"),
                    ("finished_media", "media"),
                ):
                    values = dict(worker.get(source, {}))
                    if request_id in values:
                        output[target] = values[request_id]
                if terminal is not None:
                    output["status"] = terminal.get("status")
                    output["error"] = terminal.get("error")
                request_workers.append(output)
            self._finish(
                request_id,
                RequestOutcome.COMPLETED,
                result=request_workers,
            )

        done = [bool(worker.get("request_done")) for worker in workers]
        if any(value != done[0] for value in done[1:]):
            raise RuntimeError("Workers disagreed on model request completion")
        return bool(done and done[0])

    def finish(
        self,
        request_id: str,
        outcome: RequestOutcome,
        *,
        result: list[Any] | None = None,
        error: BaseException | None = None,
    ) -> bool:
        return self._call("finish", request_id, outcome, result, error)

    def _handle_finish(
        self,
        request_id: str,
        outcome: RequestOutcome,
        result: list[Any] | None,
        error: BaseException | None,
    ) -> bool:
        return self._finish(request_id, outcome, result=result, error=error)

    def _finish(
        self,
        request_id: str,
        outcome: RequestOutcome,
        *,
        result: list[Any] | None = None,
        error: BaseException | None = None,
    ) -> bool:
        record = self._require(request_id)
        if record.terminal:
            return False
        record.terminal = True
        record.outcome = outcome
        if record.demand_active:
            self._execution_slices.set_model_demand(record.model_id, False)
            record.demand_active = False
        future = record.future
        if future is not None and not future.done():
            if outcome is RequestOutcome.COMPLETED:
                future.set_result(result or [])
            elif outcome is RequestOutcome.CANCELLED:
                future.cancel()
            else:
                future.set_exception(error or RuntimeError(outcome.value))
        return True

    def fail_model(self, model_id: str, error: BaseException) -> None:
        self._call("fail_model", model_id, error)

    def _handle_fail_model(self, model_id: str, error: BaseException) -> None:
        for request_id, record in tuple(self._records.items()):
            if record.model_id == model_id and not record.terminal:
                self._finish(request_id, RequestOutcome.FAILED, error=error)

    def snapshot(self, request_id: str) -> RequestSnapshot:
        return self._call("snapshot", request_id)

    def _handle_snapshot(self, request_id: str) -> RequestSnapshot:
        return self._require(request_id).snapshot()

    def close(self, request_id: str) -> bool:
        return self._call("close", request_id)

    def _handle_close(self, request_id: str) -> bool:
        record = self._require(request_id)
        if not record.terminal:
            raise RuntimeError(f"Model request {request_id!r} is not finished")
        del self._records[request_id]
        return any(
            item.model_id == record.model_id and item.demand_active
            for item in self._records.values()
        )

    def _require(self, request_id: str) -> _RequestRecord:
        try:
            return self._records[request_id]
        except KeyError as error:
            raise KeyError(f"Unknown model request {request_id!r}") from error

    def shutdown(self) -> None:
        if not self._thread.is_alive():
            return
        self._messages.put(None)
        self._thread.join()


@dataclass
class _EmitterCommand:
    operation: str
    args: tuple[Any, ...]
    reply: Future[Any]


class ModelEmitter:
    """在常驻线程中独占一个模型的 Worker 执行状态。"""

    def __init__(
        self,
        model_id: str,
        worker_rpc: Callable[..., list[dict[str, Any]]],
        control: RequestControlPlane,
    ) -> None:
        self.model_id = model_id
        self._worker_rpc = worker_rpc
        self._control = control
        self._commands: queue.Queue[_EmitterCommand] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name=f"model-emitter-{model_id}",
            daemon=True,
        )
        self._thread.start()

    def _call(self, operation: str, *args: Any) -> Any:
        reply: Future[Any] = Future()
        self._commands.put(_EmitterCommand(operation, args, reply))
        return reply.result()

    def submit(
        self,
        request_id: str,
        work: Any,
        variant: str,
        max_steps: int,
    ) -> list[dict[str, Any]]:
        return self._call("submit", request_id, work, variant, max_steps)

    def cancel(self, request_id: str) -> list[dict[str, Any]]:
        return self._call("cancel", request_id)

    def _run(self) -> None:
        workers: list[dict[str, Any]] | None = None
        driver_request_id: str | None = None
        driver_variant = "auto"
        running = True
        while running:
            command = self._commands.get() if workers is None else None
            commands = [] if command is None else [command]
            while True:
                try:
                    commands.append(self._commands.get_nowait())
                except queue.Empty:
                    break
            for command in commands:
                try:
                    if command.operation == "shutdown":
                        running = False
                        command.reply.set_result(None)
                        continue
                    if command.operation == "cancel":
                        request_id = str(command.args[0])
                        result = self._worker_rpc(
                            "abort_model_request",
                            args=(self.model_id, request_id),
                        )
                        if any(
                            not worker.get("stopped", False)
                            for worker in result
                        ):
                            raise RuntimeError(
                                f"Workers failed to abort {request_id!r}: "
                                f"{result!r}"
                            )
                        # 丢弃取消前取得的边界。若模型仍有请求，由 Worker 在
                        # 同一响应中给出清理后的新边界，后续只推进该边界。
                        if any(
                            int(worker.get("pending_work", 0)) > 0
                            for worker in result
                        ):
                            if any(
                                worker.get("next_slice") is None
                                for worker in result
                            ):
                                raise RuntimeError(
                                    "Workers retained model work after abort "
                                    "without returning a replacement boundary"
                                )
                            workers = result
                        else:
                            workers = None
                            driver_request_id = None
                            driver_variant = "auto"
                        command.reply.set_result(result)
                        continue
                    if command.operation != "submit":
                        raise RuntimeError(
                            f"Unknown emitter operation {command.operation!r}"
                        )
                    request_id, work, variant, max_steps = command.args
                    model_was_active = workers is not None
                    started = self._worker_rpc(
                        "start_model_request",
                        args=(
                            self.model_id,
                            work,
                            variant,
                            max_steps,
                            request_id,
                        ),
                    )
                    if not started:
                        raise RuntimeError(
                            f"No Worker accepted model request {request_id!r}"
                        )
                    opened = all(
                        worker.get("next_slice") is not None
                        for worker in started
                    )
                    if opened:
                        if model_was_active:
                            raise RuntimeError(
                                f"Model {self.model_id!r} opened two drivers"
                            )
                        workers = started
                        driver_request_id = str(request_id)
                        driver_variant = str(variant)
                    elif not model_was_active:
                        raise RuntimeError(
                            f"Idle model {self.model_id!r} accepted "
                            f"request {request_id!r} without a first boundary"
                        )
                    command.reply.set_result(started)
                except BaseException as error:
                    command.reply.set_exception(error)
                    self._recover(error)
                    workers = None
                    driver_request_id = None

            if not running or workers is None:
                continue
            try:
                assert driver_request_id is not None
                boundaries = [worker.get("next_slice") for worker in workers]
                if not boundaries or any(value is None for value in boundaries):
                    raise RuntimeError(
                        f"Workers returned no next slice for {driver_request_id!r}"
                    )
                if any(value != boundaries[0] for value in boundaries[1:]):
                    raise RuntimeError(
                        f"Workers disagreed on next slice: {boundaries!r}"
                    )
                scheduled = self._control.plan_slice(
                    self.model_id,
                    driver_request_id,
                    driver_variant,
                    boundaries[0],
                )
                workers = self._worker_rpc(
                    "execute_model_request_slice",
                    args=(driver_request_id, scheduled),
                )
                if self._control.report_workers(workers, driver_request_id):
                    workers = None
                    driver_request_id = None
                    driver_variant = "auto"
            except BaseException as error:
                self._recover(error)
                workers = None
                driver_request_id = None
                driver_variant = "auto"

    def _recover(self, error: BaseException) -> None:
        try:
            self._worker_rpc("reset_model_requests", args=(self.model_id,))
        except BaseException as reset_error:
            error = RuntimeError(
                f"Model emitter for {self.model_id!r} failed: {error}; "
                f"Worker reset also failed: {reset_error}"
            )
        self._control.fail_model(self.model_id, error)

    def shutdown(self) -> None:
        if not self._thread.is_alive():
            return
        self._call("shutdown")
        self._thread.join()
