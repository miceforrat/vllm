# SPDX-License-Identifier: Apache-2.0
"""Cooperative resource arbitration for colocated model runners."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class GeneratorExecutionPhase(str, Enum):
    PREPARING = "preparing"
    DENOISING = "denoising"
    VAE = "vae"
    COMPLETE = "complete"


@dataclass
class GeneratorPhaseLease:
    """Stable execution decision for one schedulable Generator phase."""

    stream: Any
    mode: str
    phase: GeneratorExecutionPhase
    step_index: int | None
    _arbiter: "MultiModelResourceArbiter"
    _finished: bool = False

    def finish(self, completion_event: Any | None = None) -> None:
        if not self._finished:
            # 由仲裁器来托管最后的finish
            self._arbiter.finish_generator_phase(self, completion_event)
            self._finished = True

    @property
    def trace_nvtx(self) -> bool:
        return self._arbiter.trace_partition_nvtx


class MultiModelResourceArbiter:
    """Select full or partitioned Generator execution at safe boundaries.

    An arriving primary-model request first registers demand. If a full-SM
    Generator step is already in flight, registration waits until that step's
    explicit completion boundary. New Generator steps then select the shared
    stream until all primary-model demand has been released.
    """

    def __init__(
        self,
        *,
        full_generator_stream: Any,
        shared_generator_stream: Any,
        full_primary_stream: Any | None = None,
        shared_primary_stream: Any | None = None,
        generator_phase_policy: str = "demand_aware",
        trace_partition_nvtx: bool = False,
    ):
        if generator_phase_policy not in {"demand_aware", "fixed_shared"}:
            raise ValueError(
                "generator_phase_policy must be demand_aware or fixed_shared"
            )

        # 各类执行流资源
        # 这个未来可以考虑通用化，比如按照SM数量用map存储
        self._full_generator_stream = full_generator_stream
        self._shared_generator_stream = shared_generator_stream
        self._full_primary_stream = full_primary_stream
        self._shared_primary_stream = shared_primary_stream

        # 调度策略保存
        self._generator_phase_policy = generator_phase_policy
        # 观测相关
        self.trace_partition_nvtx = trace_partition_nvtx
        # 线程间状态同步条件变量
        self._condition = threading.Condition()

        # reasoner请求需求和资源占用周期、batch状态
        self._primary_demand = 0
        self._primary_epoch_id = 0
        self._primary_epoch_active = False
        self._primary_batches_submitted = 0
        self._primary_unscoped_batches = 0
        self._active_primary_batch_mode: str | None = None

        # reasoner统计量
        self._primary_full_batches = 0
        self._primary_shared_batches = 0
        self._primary_waits = 0


        # generation admission与租约状态
        self._generator_request_active = False
        self._generator_admission_pending = False
        self._generator_waits_for_primary = 0

        self._primary_completion_drains = 0
        self._active_generator_lease: GeneratorPhaseLease | None = None
        self._last_generator_phase: GeneratorExecutionPhase | None = None

        self._full_steps = 0
        self._shared_steps = 0
        self._phase_counts: dict[str, int] = {}
        self._generator_phase_trace: deque[dict[str, Any]] = deque(maxlen=256)
        # 延迟完成状态
        self._pending_generator_completion: tuple[
            Any, GeneratorExecutionPhase, int | None, str
        ] | None = None
        self._pending_primary_completion: tuple[Any, str] | None = None


    def _drain_generator_completion(
        self,
        completion: tuple[Any, GeneratorExecutionPhase, int | None, str],
        reason: str,
    ) -> None:
        event, phase, step_index, mode = completion
        event.synchronize()
        if self.trace_partition_nvtx:
            import torch

            step = "none" if step_index is None else str(step_index)
            torch.cuda.nvtx.mark(
                "generator/gpu_drained/"
                f"phase={phase.value}/step={step}/mode={mode}/reason={reason}"
            )

    def _drain_primary_completion(
        self, completion: tuple[Any, str], reason: str
    ) -> None:
        event, mode = completion
        event.synchronize()
        if self.trace_partition_nvtx:
            import torch

            torch.cuda.nvtx.mark(
                "reasoner/gpu_drained/"
                f"mode={mode}/reason={reason}"
            )

    def acquire_primary_demand(self) -> None:
        completion = None
        with self._condition:
            if self._primary_demand == 0:
                self._primary_epoch_id += 1
                self._primary_epoch_active = True
            self._primary_demand += 1
            while (
                self._active_generator_lease is not None
                and self._active_generator_lease.mode == "generator_full"
            ):
                self._primary_waits += 1
                self._condition.wait()
            pending = self._pending_generator_completion
            if pending is not None and pending[3] == "generator_full":
                completion = pending
        if completion is not None:
            # Demand may enter the complementary GC lane only after the
            # preceding full-SM Generator phase has actually drained.
            self._drain_generator_completion(completion, "primary_admission")
            with self._condition:
                if self._pending_generator_completion is completion:
                    self._pending_generator_completion = None
                self._condition.notify_all()

    def release_primary_demand(self) -> None:
        with self._condition:
            if self._primary_demand <= 0:
                raise RuntimeError("Primary-model demand is not active")
            self._primary_demand -= 1
            if self._primary_demand == 0:
                self._primary_epoch_active = False
            self._condition.notify_all()

    def note_primary_batch_submitted(self) -> int | None:
        """Record work in the current request-level resource epoch.

        vLLM may keep multiple scheduler batches in flight. Resource ownership
        therefore follows the outer demand epoch, rather than being acquired
        and released once per ``execute_model`` call.
        """
        with self._condition:
            if not self._primary_epoch_active:
                # Keep legacy/synchronous callers working while the Engine
                # integration is migrated to explicit demand epochs.
                self._primary_unscoped_batches += 1
                return None
            self._primary_batches_submitted += 1
            return self._primary_epoch_id

    def select_primary_batch_stream(self) -> tuple[Any | None, str]:
        """Select one stable lane for the next vLLM scheduler batch."""
        with self._condition:
            if self._active_primary_batch_mode is not None:
                raise RuntimeError(
                    "A second primary batch reached the Worker before the "
                    "previous batch finished host submission"
                )
            active = self._active_generator_lease
            if active is not None and active.mode == "generator_full":
                raise RuntimeError(
                    "Primary execution reached the Worker while a full-SM "
                    "Generator phase was active; demand admission must wait "
                    "for the Generator boundary first"
                )
            if (
                self._primary_epoch_active
                and not self._generator_request_active
                and not self._generator_admission_pending
                and self._full_primary_stream is not None
            ):
                self._primary_full_batches += 1
                self._active_primary_batch_mode = "reasoner_full"
                return self._full_primary_stream, "reasoner_full"
            self._primary_shared_batches += 1
            self._active_primary_batch_mode = "shared"
            return self._shared_primary_stream, "shared"

    def finish_primary_batch(
        self, mode: str, completion_event: Any | None
    ) -> None:
        """Publish the completion of one primary batch's host submission."""
        with self._condition:
            if self._active_primary_batch_mode != mode:
                raise RuntimeError(
                    "Finishing a stale primary batch: "
                    f"active={self._active_primary_batch_mode!r}, mode={mode!r}"
                )
            self._active_primary_batch_mode = None
            if mode == "reasoner_full" and completion_event is not None:
                self._pending_primary_completion = (completion_event, mode)
            self._condition.notify_all()

    def acquire_generator_phase(
        self,
        phase: GeneratorExecutionPhase | str,
        step_index: int | None = None,
        selected_mode: str | None = None,
    ) -> GeneratorPhaseLease | None:
        phase = GeneratorExecutionPhase(phase)
        completion = None
        primary_completion = None
        with self._condition:
            self._last_generator_phase = phase
            if phase is GeneratorExecutionPhase.COMPLETE:
                if self._active_generator_lease is not None:
                    raise RuntimeError(
                        "Generator cannot complete with an active phase lease"
                    )
                completion = self._pending_generator_completion
            else:
                first_phase = not self._generator_request_active
                if first_phase:
                    self._generator_admission_pending = True
                    if self._active_primary_batch_mode == "reasoner_full":
                        self._generator_waits_for_primary += 1
                    while self._active_primary_batch_mode == "reasoner_full":
                        self._condition.wait()
                    primary_completion = self._pending_primary_completion
                self._generator_request_active = True
                if self._active_generator_lease is not None:
                    raise RuntimeError("A Generator phase lease is already active")
                if selected_mode is not None and selected_mode not in {
                    "shared",
                    "generator_full",
                }:
                    raise ValueError(
                        f"Unsupported Generator execution mode: {selected_mode!r}"
                    )
                mode = selected_mode
                if mode is None and (
                    self._generator_phase_policy == "fixed_shared"
                    or self._primary_demand
                ):
                    mode = "shared"
                elif mode is None:
                    mode = "generator_full"
                if mode == "shared":
                    mode = "shared"
                    stream = self._shared_generator_stream
                    self._shared_steps += 1
                else:
                    stream = self._full_generator_stream
                    self._full_steps += 1
                pending = self._pending_generator_completion
                if pending is not None and pending[3] != mode:
                    if (
                        pending[3] == "generator_full"
                        and self._primary_demand > 0
                    ):
                        # Primary admission is the unique owner of the
                        # full->shared handoff drain.  Do not synchronize the
                        # same CUDA event concurrently from both host threads:
                        # wait until acquire_primary_demand() publishes that
                        # the event has drained and clears the handoff.
                        while self._pending_generator_completion is pending:
                            self._condition.wait()
                    else:
                        completion = pending
                lease = GeneratorPhaseLease(
                    stream, mode, phase, step_index, self
                )
                self._active_generator_lease = lease
                key = phase.value
                self._phase_counts[key] = self._phase_counts.get(key, 0) + 1
                self._generator_phase_trace.append(
                    {
                        "phase": key,
                        "step_index": step_index,
                        "mode": mode,
                        "primary_demand": self._primary_demand,
                    }
                )
                self._generator_admission_pending = False
        if primary_completion is not None:
            # The first Generator phase may use either the full lane or the
            # complementary GC lane, but neither may begin while a previously
            # submitted full-SM Reasoner batch is still executing.
            self._drain_primary_completion(
                primary_completion, "generator_admission"
            )
            with self._condition:
                self._primary_completion_drains += 1
                if self._pending_primary_completion is primary_completion:
                    self._pending_primary_completion = None
        if completion is not None:
            reason = (
                "request_complete"
                if phase is GeneratorExecutionPhase.COMPLETE
                else "mode_switch"
            )
            self._drain_generator_completion(completion, reason)
            with self._condition:
                if self._pending_generator_completion is completion:
                    self._pending_generator_completion = None
        if phase is GeneratorExecutionPhase.COMPLETE:
            with self._condition:
                key = phase.value
                self._phase_counts[key] = self._phase_counts.get(key, 0) + 1
                self._generator_request_active = False
                self._generator_admission_pending = False
                self._condition.notify_all()
            return None
        return lease

    def finish_generator_phase(
        self, lease: GeneratorPhaseLease, completion_event: Any | None = None
    ) -> None:
        with self._condition:
            if self._active_generator_lease is not lease:
                raise RuntimeError("Finishing a stale Generator step lease")
            self._active_generator_lease = None
            if completion_event is not None:
                self._pending_generator_completion = (
                    completion_event,
                    lease.phase,
                    lease.step_index,
                    lease.mode,
                )
            self._condition.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._condition:
            active = self._active_generator_lease
            return {
                "primary_demand": self._primary_demand,
                "primary_epoch_id": self._primary_epoch_id,
                "primary_epoch_active": self._primary_epoch_active,
                "primary_batches_submitted": self._primary_batches_submitted,
                "primary_unscoped_batches": self._primary_unscoped_batches,
                "primary_full_batches": self._primary_full_batches,
                "primary_shared_batches": self._primary_shared_batches,
                "active_primary_batch_mode": self._active_primary_batch_mode,
                "generator_admission_pending": (
                    self._generator_admission_pending
                ),
                "generator_waits_for_primary": (
                    self._generator_waits_for_primary
                ),
                "primary_completion_drains": (
                    self._primary_completion_drains
                ),
                "generator_phase_policy": self._generator_phase_policy,
                "active_generator_mode": None if active is None else active.mode,
                "active_generator_phase": (
                    None if active is None else active.phase.value
                ),
                "last_generator_phase": (
                    None
                    if self._last_generator_phase is None
                    else self._last_generator_phase.value
                ),
                "generator_request_active": self._generator_request_active,
                "generator_phase_counts": dict(self._phase_counts),
                "generator_phase_trace": list(self._generator_phase_trace),
                "generator_full_steps": self._full_steps,
                "shared_steps": self._shared_steps,
                "primary_waits": self._primary_waits,
            }
