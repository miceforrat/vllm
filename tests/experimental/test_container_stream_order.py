# SPDX-License-Identifier: Apache-2.0
"""MiniContainer 跨资源变体的 GPU 提交顺序回归测试。"""

from contextlib import contextmanager
from typing import Any

import pytest
import torch

from vllm.experimental.execution_slice import ExecutionSliceDescriptor
from vllm.experimental.worker.binding import (
    CudaStreamBinding,
    DefaultMemoryBinding,
    ParallelGroupBinding,
    ResourceAssignment,
)
from vllm.experimental.worker.mini_container import MiniContainer
from vllm.experimental.worker.model_timeline import ModelGPUTimeline


def test_model_timeline_chains_every_slice_and_records_failures() -> None:
    records: list[tuple[str, str]] = []

    class _Event:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream: Any) -> None:
            records.append(("record", f"{self.name}:{stream.name}"))

    class _Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event: _Event) -> None:
            records.append(("wait", f"{event.name}:{self.name}"))

        def wait_stream(self, stream: Any) -> None:
            records.append(("join", f"{stream.name}:{self.name}"))

    events = iter((_Event("s0"), _Event("e0"), _Event("s1"), _Event("e1")))
    timeline = ModelGPUTimeline(event_factory=lambda: next(events))
    with timeline.slice(
        _Stream("shared"), dependent_streams=(_Stream("copy"),)
    ):
        pass
    with pytest.raises(RuntimeError, match="injected"):
        with timeline.slice(_Stream("full")):
            raise RuntimeError("injected")

    assert records == [
        ("record", "s0:shared"),
        ("join", "copy:shared"),
        ("record", "e0:shared"),
        ("wait", "e0:full"),
        ("record", "s1:full"),
        ("record", "e1:full"),
    ]


def test_model_timeline_only_joins_output_when_variant_buffer_reused() -> None:
    records: list[tuple[str, str]] = []

    class _Event:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream: Any) -> None:
            records.append(("record", f"{self.name}:{stream.name}"))

    class _Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event: _Event) -> None:
            records.append(("wait", f"{event.name}:{self.name}"))

        def wait_stream(self, stream: Any) -> None:
            records.append(("join", f"{stream.name}:{self.name}"))

    names = iter(
        (
            "shared-start",
            "shared-compute",
            "shared-output",
            "full-start",
            "full-compute",
            "full-output",
            "shared2-start",
            "shared2-compute",
            "shared2-output",
        )
    )
    timeline = ModelGPUTimeline(
        event_factory=lambda: _Event(next(names))
    )
    with timeline.slice(
        _Stream("shared"),
        dependent_streams=(_Stream("shared-copy"),),
        independent_tail_key="shared",
    ):
        pass
    with timeline.slice(
        _Stream("full"),
        dependent_streams=(_Stream("full-copy"),),
        independent_tail_key="full",
    ):
        pass
    with timeline.slice(
        _Stream("shared2"),
        independent_tail_key="shared",
    ):
        pass

    # full 只等 shared 的计算边界，不等 shared-output；再次使用 shared
    # buffer 时，同时等待最新 full 计算和旧 shared output-copy 完成。
    assert ("wait", "shared-compute:full") in records
    assert ("wait", "shared-output:full") not in records
    assert ("wait", "full-compute:shared2") in records
    assert ("wait", "shared-output:shared2") in records


def test_container_records_one_observation_per_execution_slice() -> None:
    class _Event:
        def record(self, stream: Any) -> None:
            del stream

        def query(self) -> bool:
            return True

        def elapsed_time(self, end: Any) -> float:
            del end
            return 1.5

    class _Stream:
        name = "shared"

        def wait_event(self, event: Any) -> None:
            del event

    class _Unit:
        model_id = "generator"
        runner = None

        @contextmanager
        def activate(self, assignment, *, resource_transition=True):
            del assignment, resource_transition
            yield

        def execute_slice(self, descriptor, *, request_id=None):
            del descriptor, request_id
            return {
                "request_done": False,
                "metadata": {"status": "RUNNING"},
            }

    assignment = ResourceAssignment(
        ParallelGroupBinding({}),
        DefaultMemoryBinding(),
        CudaStreamBinding(_Stream()),
    )
    container = MiniContainer(_Unit())
    container._timeline = ModelGPUTimeline(event_factory=_Event)
    container.execute_slice(
        assignment,
        descriptor=ExecutionSliceDescriptor(
            7,
            metadata={
                "request_ids": ("a", "b"),
                "predicted_seconds": 0.002,
            },
            tags=frozenset(("component:denoising",)),
        ),
        variant_name="shared",
        request_id="a",
        execution_epoch=11,
    )

    assert container.slice_execution_records() == ({
        "model_id": "generator",
        "request_ids": ("a", "b"),
        "operator_class": "component:denoising",
        "resource_variant": "shared",
        "actual_sm_count": None,
        "predicted_seconds": 0.002,
        "execution_epoch": 11,
        "sequence_no": 7,
        "gpu_elapsed_ms": 1.5,
        "gpu_start_ms": 1.5,
        "gpu_end_ms": 1.5,
        "host_start_seconds": container.slice_execution_records()[0][
            "host_start_seconds"
        ],
        "host_end_seconds": container.slice_execution_records()[0][
            "host_end_seconds"
        ],
        "result": {"request_done": False, "status": "RUNNING"},
    },)


def test_container_records_native_slice_without_host_output() -> None:
    class _Event:
        def record(self, stream: Any) -> None:
            del stream

        def query(self) -> bool:
            return True

        def elapsed_time(self, end: Any) -> float:
            del end
            return 0.0

    class _Stream:
        device = 0

        def wait_event(self, event: Any) -> None:
            del event

    class _Unit:
        model_id = "reasoner"
        runner = None

        @contextmanager
        def activate(self, assignment, *, resource_transition=True):
            del assignment, resource_transition
            yield

        def execute_slice(self, descriptor, *, request_id=None):
            del descriptor, request_id
            return None

    assignment = ResourceAssignment(
        ParallelGroupBinding({}),
        DefaultMemoryBinding(),
        CudaStreamBinding(_Stream()),
    )
    container = MiniContainer(_Unit())
    container._timeline = ModelGPUTimeline(event_factory=_Event)
    container.execute_slice(
        assignment,
        descriptor=object(),
        variant_name="default",
    )

    record = container.slice_execution_records()[0]
    assert record["result"] == {
        "request_done": False,
        "status": "NO_HOST_OUTPUT",
    }


def test_container_records_native_framework_output_without_done_field() -> None:
    class _Event:
        def record(self, stream: Any) -> None:
            del stream

        def query(self) -> bool:
            return True

        def elapsed_time(self, end: Any) -> float:
            del end
            return 0.0

    class _Stream:
        device = 0

        def wait_event(self, event: Any) -> None:
            del event

    class _FrameworkOutput:
        pass

    class _Unit:
        model_id = "reasoner"
        runner = None

        @contextmanager
        def activate(self, assignment, *, resource_transition=True):
            del assignment, resource_transition
            yield

        def execute_slice(self, descriptor, *, request_id=None):
            del descriptor, request_id
            return _FrameworkOutput()

    assignment = ResourceAssignment(
        ParallelGroupBinding({}),
        DefaultMemoryBinding(),
        CudaStreamBinding(_Stream()),
    )
    container = MiniContainer(_Unit())
    container._timeline = ModelGPUTimeline(event_factory=_Event)
    container.execute_slice(
        assignment,
        descriptor=object(),
        variant_name="default",
    )

    assert container.slice_execution_records()[0]["result"] == {
        "request_done": False,
        "status": "FRAMEWORK_OUTPUT",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_variant_switch_waits_for_latest_slice() -> None:
    shared_stream = torch.cuda.Stream()
    full_stream = torch.cuda.Stream()
    value = torch.zeros((), device="cuda")

    class _Unit:
        model_id = "generator"

        def __init__(self) -> None:
            self.sequence = 0
            self.reads: list[torch.Tensor] = []

        @contextmanager
        def activate(self, assignment, *, resource_transition=True):
            with assignment.streams.activate(
                None, synchronize_transition=resource_transition
            ):
                yield

        def execute_slice(self, descriptor, *, request_id=None):
            del descriptor, request_id
            self.reads.append(value.clone())
            torch.cuda._sleep(100_000_000)
            self.sequence += 1
            value.fill_(self.sequence)
            return {"request_done": False}

    def assignment(stream):
        return ResourceAssignment(
            ParallelGroupBinding({}),
            DefaultMemoryBinding(),
            CudaStreamBinding(stream),
        )

    unit = _Unit()
    container = MiniContainer(unit)
    shared = assignment(shared_stream)
    full = assignment(full_stream)

    # 三个 shared slice 后切到 full。full 必须等待最近的第三个 slice，
    # 而不是只等待第一次进入 shared 时经 caller 建立的旧边界。
    for selected in (shared, shared, shared, full):
        container.execute_slice(selected, descriptor=object())

    torch.cuda.synchronize()
    assert [int(item.item()) for item in unit.reads] == [0, 1, 2, 3]
