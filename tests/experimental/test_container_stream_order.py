# SPDX-License-Identifier: Apache-2.0
"""MiniContainer 跨资源变体的 GPU 提交顺序回归测试。"""

from contextlib import contextmanager

import pytest
import torch

from vllm.experimental.worker.binding import (
    CudaStreamBinding,
    DefaultMemoryBinding,
    ParallelGroupBinding,
    ResourceAssignment,
)
from vllm.experimental.worker.mini_container import MiniContainer


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
