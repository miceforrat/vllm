# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned execution units for non-token model work."""

from __future__ import annotations

import os
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExecutionUnitState(str, Enum):
    PENDING = "pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ScheduledExecutionSlice:
    """EngineCore 为一次模型推进下发的调度结果。

    ``ExecUnit`` 是 Worker 中长期存活的模型实例；本类只描述
    一次 batch、denoising step 或 VAE block 的逻辑资源决策。
    """

    model_id: str
    request_id: str
    kind: str
    index: int | None
    resource_variant: str
    execution_epoch: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any] | ScheduledExecutionSlice
    ) -> ScheduledExecutionSlice:
        if isinstance(payload, cls):
            return payload
        return cls(
            model_id=str(payload["model_id"]),
            request_id=str(payload["request_id"]),
            kind=str(payload["kind"]),
            index=(
                None if payload.get("index") is None else int(payload["index"])
            ),
            resource_variant=str(payload["resource_variant"]),
            execution_epoch=int(payload["execution_epoch"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "request_id": self.request_id,
            "kind": self.kind,
            "index": self.index,
            "resource_variant": self.resource_variant,
            "execution_epoch": self.execution_epoch,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionUnitSpec:
    """EngineCore 可见的一个非 token 执行 slice 边界。"""

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
    scheduled_slice: ScheduledExecutionSlice | None = None


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
        self._primary_model_id: str | None = None
        self._model_demand: dict[str, int] = {}
        self._model_variants: dict[str, tuple[str, str | None]] = {}
        self._model_participants: dict[str, int] = {}
        self._boundary_decisions: dict[
            str, dict[tuple[str, int | None], dict[str, Any]]
        ] = {}
        self._request_finish_counts: dict[str, int] = {}
        self._next_execution_epoch = 0
        self._revision = 0
        self._lock = threading.RLock()

    def configure_model(
        self,
        model_id: str,
        *,
        full_variant: str,
        shared_variant: str | None,
        primary: bool = False,
        participant_count: int = 1,
    ) -> None:
        """Register Engine-planned logical variants for one model."""
        with self._lock:
            self._model_variants[str(model_id)] = (
                str(full_variant),
                None if shared_variant is None else str(shared_variant),
            )
            if participant_count <= 0:
                raise ValueError("Model participant count must be positive")
            self._model_participants[str(model_id)] = int(participant_count)
            if primary:
                self._primary_model_id = str(model_id)

    @property
    def has_resource_variants(self) -> bool:
        return any(shared is not None for _, shared in self._model_variants.values())

    @property
    def primary_model_id(self) -> str | None:
        return self._primary_model_id

    def set_model_demand(self, model_id: str, active: bool) -> dict[str, Any]:
        """Record request demand used by EngineCore partition decisions."""
        with self._lock:
            model_id = str(model_id)
            count = self._model_demand.get(model_id, 0)
            if active:
                count += 1
            elif count <= 0:
                raise RuntimeError(
                    f"Model demand is not active for {model_id!r}"
                )
            else:
                count -= 1
            if count:
                self._model_demand[model_id] = count
            else:
                self._model_demand.pop(model_id, None)
            self._revision += 1
            return self.snapshot()

    def schedule_slice(
        self,
        model_id: str,
        request_id: str,
        spec: ExecutionUnitSpec,
    ) -> ScheduledExecutionSlice:
        """Select one logical resource variant and assign a global epoch."""
        with self._lock:
            model_id = str(model_id)
            full_variant, shared_variant = self._model_variants.get(
                model_id, ("generator_full", "shared")
            )
            contended = self._primary_demand > 0 or any(
                other != model_id and count > 0
                for other, count in self._model_demand.items()
            ) or any(
                request.model_id != model_id and not request.done
                for request in self._requests.values()
            )
            variant = (
                shared_variant
                if contended and shared_variant is not None
                else full_variant
            )
            requested = spec.metadata.get("requested_variant")
            if requested not in (None, "auto"):
                requested = str(requested)
                allowed = {full_variant, shared_variant}
                if requested not in allowed:
                    raise ValueError(
                        f"Unknown resource variant {requested!r} for "
                        f"model {model_id!r}"
                    )
                variant = requested
            self._next_execution_epoch += 1
            return ScheduledExecutionSlice(
                model_id=model_id,
                request_id=str(request_id),
                kind=spec.kind,
                index=spec.index,
                resource_variant=variant,
                execution_epoch=self._next_execution_epoch,
                metadata=spec.metadata,
            )

    def schedule_model_batch(
        self, model_id: str, request_ids: Sequence[str]
    ) -> ScheduledExecutionSlice:
        """Schedule one native model batch as an execution slice."""
        joined = ",".join(str(item) for item in request_ids)
        return self.schedule_slice(
            model_id,
            joined or f"batch-{self._next_execution_epoch + 1}",
            ExecutionUnitSpec("model_batch"),
        )

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
            if self._primary_model_id is not None:
                model_id = self._primary_model_id
                if active:
                    self._model_demand[model_id] = (
                        self._model_demand.get(model_id, 0) + 1
                    )
                else:
                    count = self._model_demand.get(model_id, 0) - 1
                    if count > 0:
                        self._model_demand[model_id] = count
                    else:
                        self._model_demand.pop(model_id, None)
            return self.snapshot()

    def admit_next(self, request_id: str) -> dict[str, Any]:
        request = self._get(request_id)
        unit = request.active
        if unit is None:
            raise RuntimeError(f"Execution request {request_id!r} is complete")
        if unit.state is not ExecutionUnitState.PENDING:
            raise RuntimeError(f"Execution unit is not pending: {unit.state.value}")
        unit.scheduled_slice = self.schedule_slice(
            request.model_id, request.request_id, unit.spec
        )
        unit.selected_mode = unit.scheduled_slice.resource_variant
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
            boundary_key = (
                str(message["kind"]),
                None
                if message.get("index") is None
                else int(message["index"]),
            )
            cached = self._boundary_decisions.get(request_id, {}).get(
                boundary_key
            )
            if cached is not None:
                return cached
            current = self._requests.get(request_id)
            if current is not None and current.active is not None:
                self.complete_unit(request_id)
            self.offer_unit(
                model_id,
                request_id,
                {
                    "kind": str(message["kind"]),
                    "index": message.get("index"),
                    "metadata": {
                        "requested_variant": message.get(
                            "requested_variant", "auto"
                        )
                    },
                },
            )
            admitted = self.admit_next(request_id)
            self.start_unit(request_id)
            self._boundary_decisions.setdefault(request_id, {})[
                boundary_key
            ] = admitted
            return admitted

    def finish_boundary_request(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            if request_id not in self._requests:
                # 同一 TP 请求由多个 rank 完成；清理操作保持幂等。
                return self.snapshot()
            request = self._get(request_id)
            finished = self._request_finish_counts.get(request_id, 0) + 1
            expected = self._model_participants.get(request.model_id, 1)
            if finished < expected:
                self._request_finish_counts[request_id] = finished
                return self.snapshot()
            if request.active is not None:
                self.complete_unit(request_id)
            result = self.remove_request(request_id)
            self._request_finish_counts.pop(request_id, None)
            self._boundary_decisions.pop(request_id, None)
            return result

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
                "scheduled_slice": (
                    None
                    if active.scheduled_slice is None
                    else active.scheduled_slice.to_payload()
                ),
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


class ExecutionSliceAdmissionClient:
    """Worker 侧的轻量客户端；分区策略仍完全位于 EngineCore。"""

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
    ) -> ScheduledExecutionSlice:
        result = self._request(
            {
                "operation": "boundary",
                "model_id": model_id,
                "request_id": request_id,
                "kind": kind,
                "index": index,
                "requested_variant": requested_variant or "auto",
            }
        )
        active = result.get("active_unit")
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
