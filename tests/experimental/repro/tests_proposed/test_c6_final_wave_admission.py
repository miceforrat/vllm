# SPDX-License-Identifier: Apache-2.0
"""C6（E3 / W1）的确定性复现：不需要 GPU，不需要模型。

放到 vllm 仓库 tests/experimental/ 下运行：
  .venv/bin/python -m pytest tests/experimental/test_c6_final_wave_admission.py -v
现状：断言失败（B 拿到 next_slice=None，而 driver 报告模型空闲 → B 无人推进）
修复后：通过
"""

import threading
from types import SimpleNamespace

from vllm.experimental.execution_slice import (
    ExecutionSliceDescriptor,
    ExecutionSliceResult,
)
from vllm.experimental.worker.mini_container import MiniContainer
from vllm.experimental.worker.multi_model_worker import MultiModelWorker
from vllm.v1.core.sched.execution_slice_scheduler import ScheduledExecutionSlice


def test_request_arriving_in_final_wave_is_not_stranded(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *a, **k: SimpleNamespace(record=lambda *_: None, synchronize=lambda: None),
    )
    in_final_wave, release = threading.Event(), threading.Event()

    class _Unit:
        def __init__(self) -> None:
            self.active: list[str] = []

        @property
        def pending_work_count(self) -> int:
            return len(self.active)

        def recv_job(self, work, request_id=None) -> None:
            self.active.append(request_id)

        def next_slice_spec(self, request_id=None):
            return ExecutionSliceDescriptor(1, metadata={"request_ids": tuple(self.active)})

    unit = _Unit()

    class _Container:
        model_id = "g"
        exec_unit = unit
        _result_field = staticmethod(MiniContainer._result_field)

        def recv_job(self, work, request_id=None) -> None:
            unit.recv_job(work, request_id=request_id)

        def assignment(self, name=None):
            return SimpleNamespace(streams=SimpleNamespace(stream=None))

        def execute_slice(self, **kwargs):
            finished = tuple(unit.active)
            unit.active.clear()          # A 的最后一个 wave：all_done 已经算出
            in_final_wave.set()
            release.wait(5)              # 此时 B 到达
            return ExecutionSliceResult(0, True, metadata={"finished_request_ids": finished})

    worker = MultiModelWorker.__new__(MultiModelWorker)
    worker.rank = 0
    worker._last_execution_epoch = 0
    worker._model_jobs = {}
    worker._model_jobs_lock = threading.Lock()
    worker._container_registry = SimpleNamespace(get=lambda _: _Container())

    first = worker.start_model_request("g", object(), job_id="A")
    scheduled = ScheduledExecutionSlice(
        "g", "A", ExecutionSliceDescriptor.from_payload(first["next_slice"]), "shared", 1
    )
    out: dict = {}
    t = threading.Thread(
        target=lambda: out.setdefault("slice", worker.execute_model_request_slice("A", scheduled))
    )
    t.start()
    assert in_final_wave.wait(5)
    second = worker.start_model_request("g", object(), job_id="B")
    release.set()
    t.join(5)
    # B 必须有人推进：要么 B 拿到 next_slice（EngineCore 会新开 driver），要么当前 driver 继续。
    assert second["next_slice"] is not None or out["slice"]["request_done"] is False
