# SPDX-License-Identifier: Apache-2.0
"""C2（N3）的确定性复现：不需要模型，只用开发者的 MiniContainer + CudaStreamBinding。

放到 vllm 仓库 tests/experimental/ 下运行：
  .venv/bin/python -m pytest tests/experimental/test_c2_stream_order_gpu.py -v
现状：reads == [0, 1, 2, 1]（第 4 个 slice 在另一条 stream 上，只等到了第 1 个 slice）
修复后：reads == [0, 1, 2, 3]
"""

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
def test_variant_switch_orders_all_previous_slices() -> None:
    s_shared, s_full = torch.cuda.Stream(), torch.cuda.Stream()
    buf = torch.zeros((), device="cuda")

    class _Unit:  # 与 DiffusionModelRunner 一样：没有 main_stream 属性
        model_id = "generator"

        def __init__(self) -> None:
            self.k = 0
            self.reads: list[torch.Tensor] = []

        @contextmanager
        def activate(self, assignment, *, resource_transition=True):
            with assignment.streams.activate(
                None, synchronize_transition=resource_transition
            ):
                yield

        def execute_slice(self, descriptor, *, request_id=None):
            self.reads.append(buf.clone())      # 读上一个 slice 写的值
            torch.cuda._sleep(100_000_000)      # 让本 slice 的写入推迟约 50 ms
            self.k += 1
            buf.fill_(self.k)
            return {"request_done": False}

    def assignment(stream):
        return ResourceAssignment(
            ParallelGroupBinding({}), DefaultMemoryBinding(), CudaStreamBinding(stream)
        )

    unit = _Unit()
    container = MiniContainer(unit)
    shared, full = assignment(s_shared), assignment(s_full)
    for selected in (shared, shared, shared, full):
        container.execute_slice(selected, descriptor=object())
    torch.cuda.synchronize()
    assert [int(t.item()) for t in unit.reads] == [0, 1, 2, 3]
