# SPDX-License-Identifier: Apache-2.0
"""单模型 GPU 提交时间线。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GPUTimelineSpan:
    """一个 slice 的 GPU event 区间；查询不会隐式同步设备。"""

    start_event: Any
    end_event: Any
    host_start_seconds: float
    host_end_seconds: float | None = None
    origin_event: Any | None = None

    def elapsed_ms(self) -> float | None:
        query = getattr(self.end_event, "query", None)
        if query is not None and not query():
            return None
        try:
            return float(self.start_event.elapsed_time(self.end_event))
        except (AttributeError, RuntimeError):
            return None

    def gpu_timestamps_ms(self) -> tuple[float | None, float | None]:
        origin = self.origin_event
        query = getattr(self.end_event, "query", None)
        if origin is None or (query is not None and not query()):
            return None, None
        try:
            return (
                float(origin.elapsed_time(self.start_event)),
                float(origin.elapsed_time(self.end_event)),
            )
        except (AttributeError, RuntimeError):
            return None, None


@dataclass
class SliceExecutionRecord:
    """I6：一次 slice 留下的一条模型无关执行记录。"""

    model_id: str
    request_ids: tuple[str, ...]
    operator_class: str
    resource_variant: str
    actual_sm_count: int | None
    predicted_seconds: float | None
    execution_epoch: int | None
    sequence_no: int
    timeline: GPUTimelineSpan
    result: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        gpu_start_ms, gpu_end_ms = self.timeline.gpu_timestamps_ms()
        return {
            "model_id": self.model_id,
            "request_ids": self.request_ids,
            "operator_class": self.operator_class,
            "resource_variant": self.resource_variant,
            "actual_sm_count": self.actual_sm_count,
            "predicted_seconds": self.predicted_seconds,
            "execution_epoch": self.execution_epoch,
            "sequence_no": self.sequence_no,
            "gpu_elapsed_ms": self.timeline.elapsed_ms(),
            "gpu_start_ms": gpu_start_ms,
            "gpu_end_ms": gpu_end_ms,
            "host_start_seconds": self.timeline.host_start_seconds,
            "host_end_seconds": self.timeline.host_end_seconds,
            "result": dict(self.result),
        }


class ModelGPUTimeline:
    """使同一模型的所有 GPU slice 形成一条 event 链。"""

    def __init__(self, event_factory: Callable[[], Any] | None = None) -> None:
        self._event_factory = event_factory
        self._last_done_event: Any | None = None
        self._last_compute_event: Any | None = None
        self._resource_done_events: dict[str, Any] = {}
        self._origin_event: Any | None = None

    @property
    def last_done_event(self) -> Any | None:
        return self._last_done_event

    @contextmanager
    def slice(
        self,
        stream: Any,
        *,
        dependent_streams: Iterable[Any] = (),
        independent_tail_key: str | None = None,
    ) -> Iterator[GPUTimelineSpan]:
        """等待前一个 slice，并在当前 stream 上记录新边界。

        默认把辅助 stream 尾部工作并入全模型 completion chain。提供
        ``independent_tail_key`` 时，计算仍等待全模型上一计算边界，但辅助
        output-copy 只在再次复用同一 key 对应的持久 buffer 时等待。
        """
        if independent_tail_key is not None:
            with self._independent_tail_slice(
                stream,
                dependent_streams=dependent_streams,
                resource_key=independent_tail_key,
            ) as span:
                yield span
            return

        previous = self._last_done_event
        if previous is not None:
            stream.wait_event(previous)
        start = self._new_event()
        end = self._new_event()
        if self._origin_event is None:
            self._origin_event = start
        span = GPUTimelineSpan(
            start,
            end,
            time.perf_counter(),
            origin_event=self._origin_event,
        )
        start.record(stream)
        try:
            yield span
        finally:
            # Runner 可能把 output-copy/D2H 提交到辅助 stream。先把
            # 这些尾部提交并入主时间线，再记录 slice 完成点。
            for dependent in dependent_streams:
                if dependent is not None and not self._same_stream(
                    dependent, stream
                ):
                    stream.wait_stream(dependent)
            end.record(stream)
            span.host_end_seconds = time.perf_counter()
            self._last_done_event = end
            self._last_compute_event = end

    @contextmanager
    def _independent_tail_slice(
        self,
        stream: Any,
        *,
        dependent_streams: Iterable[Any],
        resource_key: str,
    ) -> Iterator[GPUTimelineSpan]:
        """串行模型计算，但仅在 buffer 复用时等待该档位的 output 尾部。"""
        previous_compute = self._last_compute_event
        if previous_compute is not None:
            stream.wait_event(previous_compute)
        previous_resource = self._resource_done_events.get(resource_key)
        if previous_resource is not None:
            stream.wait_event(previous_resource)

        start = self._new_event()
        compute_end = self._new_event()
        resource_end = self._new_event()
        if self._origin_event is None:
            self._origin_event = start
        span = GPUTimelineSpan(
            start,
            resource_end,
            time.perf_counter(),
            origin_event=self._origin_event,
        )
        start.record(stream)
        try:
            yield span
        finally:
            # 此 event 只覆盖模型主 stream 上的计算，下一资源变体可立即依赖它。
            compute_end.record(stream)
            self._last_compute_event = compute_end
            for dependent in dependent_streams:
                if dependent is not None and not self._same_stream(
                    dependent, stream
                ):
                    stream.wait_stream(dependent)
            # 同一 variant 的持久 output buffer 再次使用前必须等待此边界。
            resource_end.record(stream)
            self._resource_done_events[resource_key] = resource_end
            self._last_done_event = resource_end
            span.host_end_seconds = time.perf_counter()

    def _new_event(self) -> Any:
        factory = self._event_factory
        if factory is not None:
            return factory()
        import torch

        return torch.cuda.Event(enable_timing=True)

    @staticmethod
    def _same_stream(left: Any, right: Any) -> bool:
        if left is right:
            return True
        left_handle = getattr(left, "cuda_stream", None)
        right_handle = getattr(right, "cuda_stream", None)
        return (
            left_handle is not None
            and right_handle is not None
            and int(left_handle) == int(right_handle)
        )
