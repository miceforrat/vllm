# SPDX-License-Identifier: Apache-2.0
"""已归档的 Worker -> EngineCore execution-slice admission socket。

旧实现由 Worker 在模型内部边界通过 Unix socket 反向请求 EngineCore 决策。
当前主路径改为 EngineCore 主动逐 slice 下发，因此不再使用该桥。这里仅保留
协议实现，便于追溯历史设计；现役代码不得导入本模块。
"""

from __future__ import annotations

import os
import secrets
import threading
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from vllm.v1.core.sched.execution_slice_scheduler import (
    ScheduledExecutionSlice,
)


class _LegacyBoundaryScheduler(Protocol):
    def handle_boundary(self, message: Mapping[str, Any]) -> dict[str, Any]: ...

    def finish_boundary_request(self, request_id: str) -> dict[str, Any]: ...


class ArchivedExecutionSliceAdmissionServer:
    """旧 Worker boundary socket server；当前主路径不再实例化。"""

    def __init__(self, scheduler: _LegacyBoundaryScheduler) -> None:
        self.scheduler = scheduler
        self.authkey = secrets.token_bytes(32)
        self.address = (
            f"/tmp/vllm-execution-slices-{os.getpid()}-"
            f"{secrets.token_hex(6)}.sock"
        )
        self._listener = Listener(
            self.address, family="AF_UNIX", authkey=self.authkey
        )
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            name="vllm-execution-slice-admission",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                connection = self._listener.accept()
            except (OSError, EOFError):
                return
            try:
                message = connection.recv()
                operation = message.get("operation", "boundary")
                if operation == "boundary":
                    result = self.scheduler.handle_boundary(message)
                elif operation == "finish":
                    result = self.scheduler.finish_boundary_request(
                        str(message["request_id"])
                    )
                else:
                    raise ValueError(
                        f"Unknown execution-slice operation: {operation!r}"
                    )
                connection.send({"ok": True, "result": result})
            except BaseException as exc:
                connection.send(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
                connection.close()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._listener.close()
        self._thread.join(timeout=1.0)
        Path(self.address).unlink(missing_ok=True)


class ArchivedExecutionSliceAdmissionClient:
    """旧 Worker 侧 socket client；当前主路径不再实例化。"""

    def __init__(self, address: str, authkey_hex: str) -> None:
        self._address = str(address)
        self._authkey = bytes.fromhex(authkey_hex)

    def _request(self, payload: Mapping[str, Any]) -> Any:
        connection = Client(
            self._address, family="AF_UNIX", authkey=self._authkey
        )
        try:
            connection.send(dict(payload))
            response = connection.recv()
        finally:
            connection.close()
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "unknown error")))
        return response["result"]

    def schedule_boundary(
        self,
        *,
        model_id: str,
        request_id: str,
        kind: str,
        index: int | None,
        requested_variant: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
    ) -> ScheduledExecutionSlice:
        result = self._request(
            {
                "operation": "boundary",
                "model_id": model_id,
                "request_id": request_id,
                "kind": kind,
                "index": index,
                "requested_variant": requested_variant or "auto",
                "tags": list(tags),
                "metadata": dict(metadata or {}),
            }
        )
        active = result.get("active_slice")
        if not isinstance(active, Mapping):
            raise RuntimeError("EngineCore returned no active execution slice")
        scheduled = active.get("scheduled_slice")
        if not isinstance(scheduled, Mapping):
            raise RuntimeError("EngineCore returned no resource decision")
        return ScheduledExecutionSlice.from_payload(scheduled)

    def finish(self, request_id: str) -> Any:
        return self._request(
            {"operation": "finish", "request_id": str(request_id)}
        )
