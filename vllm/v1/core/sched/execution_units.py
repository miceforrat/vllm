# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned execution units for non-token model work."""

from __future__ import annotations

import os
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExecutionUnitState(str, Enum):
    PENDING = "pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionUnitSpec:
    """One scheduler-visible boundary of non-token model execution."""

    kind: str
    index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutionUnitSpec":
        return cls(
            kind=str(payload["kind"]),
            index=None if payload.get("index") is None else int(payload["index"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "index": self.index,
            "metadata": dict(self.metadata),
        }


@dataclass
class ExecutionUnitRecord:
    spec: ExecutionUnitSpec
    state: ExecutionUnitState = ExecutionUnitState.PENDING
    selected_mode: str | None = None


@dataclass
class ExecutionRequestRecord:
    model_id: str
    request_id: str
    units: list[ExecutionUnitRecord]
    cursor: int = 0
    failed: bool = False

    @property
    def active(self) -> ExecutionUnitRecord | None:
        return None if self.cursor >= len(self.units) else self.units[self.cursor]

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.units)


class ExecutionUnitScheduler:
    """Engine-Scheduler lifecycle and admission state for execution units."""

    def __init__(self) -> None:
        self._requests: dict[str, ExecutionRequestRecord] = {}
        self._recent_requests: deque[dict[str, Any]] = deque(maxlen=256)
        self._primary_demand = 0
        self._revision = 0
        self._lock = threading.RLock()

    def register_request(
        self,
        model_id: str,
        request_id: str,
        units: Sequence[ExecutionUnitSpec | Mapping[str, Any]],
    ) -> dict[str, Any]:
        if request_id in self._requests:
            raise ValueError(f"Execution request {request_id!r} already exists")
        records = [
            ExecutionUnitRecord(
                unit
                if isinstance(unit, ExecutionUnitSpec)
                else ExecutionUnitSpec.from_payload(unit)
            )
            for unit in units
        ]
        if not records:
            raise ValueError("Execution request must contain at least one unit")
        self._requests[request_id] = ExecutionRequestRecord(
            model_id=str(model_id), request_id=str(request_id), units=records
        )
        self._revision += 1
        return self.snapshot()

    def offer_unit(
        self,
        model_id: str,
        request_id: str,
        unit: ExecutionUnitSpec | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append one unit discovered at a blocking pipeline boundary."""
        spec = (
            unit
            if isinstance(unit, ExecutionUnitSpec)
            else ExecutionUnitSpec.from_payload(unit)
        )
        request = self._requests.get(request_id)
        if request is None:
            request = ExecutionRequestRecord(
                model_id=str(model_id), request_id=str(request_id), units=[]
            )
            self._requests[request_id] = request
        elif request.model_id != model_id:
            raise ValueError(
                f"Execution request {request_id!r} belongs to "
                f"{request.model_id!r}, not {model_id!r}"
            )
        if request.active is not None:
            raise RuntimeError("Cannot offer a unit while another unit is active")
        request.units.append(ExecutionUnitRecord(spec))
        self._revision += 1
        return self._describe_request(request)

    def set_primary_demand(self, active: bool) -> dict[str, Any]:
        with self._lock:
            if active:
                self._primary_demand += 1
            else:
                if self._primary_demand <= 0:
                    raise RuntimeError("Primary demand is not active")
                self._primary_demand -= 1
            self._revision += 1
            return self.snapshot()

    def admit_next(self, request_id: str) -> dict[str, Any]:
        request = self._get(request_id)
        unit = request.active
        if unit is None:
            raise RuntimeError(f"Execution request {request_id!r} is complete")
        if unit.state is not ExecutionUnitState.PENDING:
            raise RuntimeError(f"Execution unit is not pending: {unit.state.value}")
        unit.selected_mode = "shared" if self._primary_demand else "generator_full"
        unit.state = ExecutionUnitState.ADMITTED
        self._revision += 1
        return self._describe_request(request)

    def start_unit(self, request_id: str) -> dict[str, Any]:
        request = self._get(request_id)
        unit = request.active
        if unit is None or unit.state is not ExecutionUnitState.ADMITTED:
            state = None if unit is None else unit.state.value
            raise RuntimeError(f"Execution unit is not admitted: {state}")
        unit.state = ExecutionUnitState.RUNNING
        self._revision += 1
        return self._describe_request(request)

    def complete_unit(
        self, request_id: str, *, failed: bool = False
    ) -> dict[str, Any]:
        request = self._get(request_id)
        unit = request.active
        if unit is None or unit.state is not ExecutionUnitState.RUNNING:
            state = None if unit is None else unit.state.value
            raise RuntimeError(f"Execution unit is not running: {state}")
        unit.state = ExecutionUnitState.FAILED if failed else ExecutionUnitState.COMPLETED
        request.failed = failed
        if not failed:
            request.cursor += 1
        self._revision += 1
        return self._describe_request(request)

    def remove_request(self, request_id: str) -> dict[str, Any]:
        request = self._get(request_id)
        self._recent_requests.append(self._describe_request(request))
        del self._requests[request_id]
        self._revision += 1
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self._revision,
                "primary_demand": self._primary_demand,
                "requests": [
                    self._describe_request(request)
                    for request in self._requests.values()
                ],
                "recent_requests": list(self._recent_requests),
            }

    def handle_boundary(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically advance one request at a Worker-reported boundary."""
        with self._lock:
            model_id = str(message["model_id"])
            request_id = str(message["request_id"])
            current = self._requests.get(request_id)
            if current is not None and current.active is not None:
                self.complete_unit(request_id)
            self.offer_unit(
                model_id,
                request_id,
                {
                    "kind": str(message["kind"]),
                    "index": message.get("index"),
                },
            )
            admitted = self.admit_next(request_id)
            self.start_unit(request_id)
            return admitted

    def finish_boundary_request(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            request = self._get(request_id)
            if request.active is not None:
                self.complete_unit(request_id)
            return self.remove_request(request_id)

    def _get(self, request_id: str) -> ExecutionRequestRecord:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown execution request: {request_id!r}") from exc

    @staticmethod
    def _describe_request(request: ExecutionRequestRecord) -> dict[str, Any]:
        active = request.active
        return {
            "model_id": request.model_id,
            "request_id": request.request_id,
            "cursor": request.cursor,
            "unit_count": len(request.units),
            "done": request.done,
            "failed": request.failed,
            "active_unit": None
            if active is None
            else {
                **active.spec.to_payload(),
                "state": active.state.value,
                "selected_mode": active.selected_mode,
            },
        }


class ExecutionUnitAdmissionServer:
    """Small local control socket connecting Worker boundaries to Scheduler."""

    def __init__(self, scheduler: ExecutionUnitScheduler) -> None:
        self.scheduler = scheduler
        self.authkey = secrets.token_bytes(32)
        self.address = f"/tmp/vllm-execution-units-{os.getpid()}-{secrets.token_hex(6)}.sock"
        self._listener = Listener(
            self.address, family="AF_UNIX", authkey=self.authkey
        )
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            name="vllm-execution-unit-admission",
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
                        f"Unknown execution-unit operation: {operation!r}"
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
