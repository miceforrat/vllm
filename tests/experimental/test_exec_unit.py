# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.execution_slice import (
    ExecutionSliceDescriptor,
)
from vllm.experimental.worker.exec_unit import (
    LLMExecUnit,
    OmniModelExecUnit,
)


class _Runner:
    def __init__(self) -> None:
        self.executed: list[Any] = []

    def execute_model(self, scheduler_output: Any) -> str:
        self.executed.append(scheduler_output)
        return f"result:{scheduler_output}"


def _execute_next(unit: Any, request_id: str | None = None) -> Any:
    descriptor = unit.next_slice_spec(request_id)
    return unit.execute_slice(descriptor, request_id=request_id)


def _create_unit(monkeypatch: pytest.MonkeyPatch) -> tuple[LLMExecUnit, _Runner]:
    runner = _Runner()
    monkeypatch.setattr(
        LLMExecUnit,
        "_create_model_runner",
        lambda _self, _config, _device: runner,
    )
    unit = LLMExecUnit(
        ModelSpec("reasoner", "model", "vllm"),
        SimpleNamespace(),
        object(),
    )
    return unit, runner


def test_llm_exec_unit_creates_and_owns_vllm_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, runner = _create_unit(monkeypatch)

    assert unit.runner is runner
    assert unit.model_id == "reasoner"


def test_llm_exec_unit_repeatedly_receives_and_advances_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, runner = _create_unit(monkeypatch)

    unit.recv_job("batch-0")
    unit.recv_job("batch-1")

    assert unit.pending_work_count == 2
    assert _execute_next(unit) == "result:batch-0"
    assert _execute_next(unit) == "result:batch-1"
    assert unit.pending_work_count == 0
    assert runner.executed == ["batch-0", "batch-1"]


def test_llm_exec_unit_can_use_worker_batch_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, runner = _create_unit(monkeypatch)
    executed: list[Any] = []
    unit.bind_step_executor(
        lambda work: executed.append(work) or f"worker-result:{work}"
    )

    unit.recv_job("batch")

    assert _execute_next(unit) == "worker-result:batch"
    assert executed == ["batch"]
    assert runner.executed == []


def test_llm_exec_unit_rejects_invalid_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, _ = _create_unit(monkeypatch)
    with pytest.raises(ValueError, match="empty work"):
        unit.recv_job(None)
    with pytest.raises(RuntimeError, match="no runnable work"):
        _execute_next(unit)


class _OmniRunner:
    def __init__(self) -> None:
        self.executed: list[Any] = []
        self.released: list[str] = []
        self.unit: OmniModelExecUnit | None = None

    def execute_stepwise(self, scheduler_output: Any) -> Any:
        request_ids = tuple(scheduler_output.scheduled_request_ids)
        self.executed.append(request_ids)
        outputs = {
            request_id: SimpleNamespace(request_id=request_id)
            for request_id in request_ids
        }
        return SimpleNamespace(get_request_output=outputs.get)

    def release_step_requests(self, request_ids: list[str]) -> None:
        self.released.extend(request_ids)


class _FakeStepScheduler:
    def __init__(self) -> None:
        self.progress: dict[str, int] = {}
        self.terminal: dict[str, Any] = {}

    def initialize(self, _config: Any) -> None:
        pass

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self.progress)

    def add_request(self, request: Any) -> str:
        self.progress[request.request_id] = 0
        return request.request_id

    def execution_phase(self, request_id: str) -> tuple[str, int]:
        cursor = self.progress[request_id]
        return ("denoising" if cursor < 2 else "vae", max(cursor - 2, 0))

    def schedule(self) -> Any:
        request_ids = tuple(self.progress)
        return SimpleNamespace(
            scheduled_request_ids=request_ids,
            is_empty=not request_ids,
        )

    def update_from_output(self, output: Any, _runner_output: Any) -> set[str]:
        finished: set[str] = set()
        for request_id in output.scheduled_request_ids:
            self.progress[request_id] += 1
            if self.progress[request_id] == 4:
                finished.add(request_id)
        for request_id in finished:
            self.progress.pop(request_id)
            self.terminal[request_id] = SimpleNamespace(
                status=SimpleNamespace(name="FINISHED_COMPLETED"),
                error=None,
            )
        return finished

    def pop_request_state(self, request_id: str) -> Any:
        return self.terminal.pop(request_id, None)

    def finish_requests(self, request_ids: list[str], status: Any) -> None:
        for request_id in request_ids:
            if request_id not in self.progress:
                continue
            self.progress.pop(request_id)
            self.terminal[request_id] = SimpleNamespace(
                status=status,
                error=None,
            )


def _create_omni_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[OmniModelExecUnit, _OmniRunner]:
    runner = _OmniRunner()
    resolved = object()
    monkeypatch.setattr(
        OmniModelExecUnit,
        "_resolve_omni_stage",
        lambda _self: resolved,
    )

    def create_runner(
        self: OmniModelExecUnit,
        *,
        resolved: Any,
        device: Any,
        host_vllm_config: Any,
    ) -> Any:
        self._test_creation = (resolved, device, host_vllm_config)
        runner.unit = self
        return runner

    monkeypatch.setattr(
        OmniModelExecUnit,
        "_create_worker_hosted_runner",
        create_runner,
    )
    monkeypatch.setattr(
        OmniModelExecUnit,
        "_create_step_scheduler",
        staticmethod(_FakeStepScheduler),
    )
    host_config = SimpleNamespace(name="host-config")
    unit = OmniModelExecUnit(
        ModelSpec(
            "generator",
            "omni-model",
            "vllm_omni",
            load_format="dummy",
            backend_options={"option": "value"},
        ),
        host_config,
        "cuda:3",
    )
    return unit, runner


def test_omni_exec_unit_creates_runner_without_owning_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, runner = _create_omni_unit(monkeypatch)

    assert unit.runner is runner
    resolved, device, host_config = unit._test_creation
    assert device == "cuda:3"
    assert resolved is not None
    assert host_config.name == "host-config"


def test_omni_exec_unit_repeatedly_receives_and_advances_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, runner = _create_omni_unit(monkeypatch)

    unit.recv_job(SimpleNamespace(request_id="job-0"), request_id="job-0")
    unit.recv_job(SimpleNamespace(request_id="job-1"), request_id="job-1")
    assert unit.pending_work_count == 2

    results = [_execute_next(unit) for _ in range(4)]
    assert [result.request_done for result in results] == [
        False,
        False,
        False,
        True,
    ]
    assert results[0].tags == frozenset(("diffusion", "denoising"))
    assert results[2].tags == frozenset(
        ("diffusion", "vae", "granularity:block")
    )
    assert runner.executed == [("job-0", "job-1")] * 4
    assert results[-1].metadata["terminal_requests"] == {
        "job-0": {"status": "FINISHED_COMPLETED", "error": None},
        "job-1": {"status": "FINISHED_COMPLETED", "error": None},
    }


def test_omni_exec_unit_abort_clears_scheduler_and_runner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, runner = _create_omni_unit(monkeypatch)
    unit.recv_job(SimpleNamespace(request_id="job-0"), request_id="job-0")
    unit.recv_job(SimpleNamespace(request_id="job-1"), request_id="job-1")

    # 模拟 EngineCore 已取得下一片、但尚未下发执行时发生取消。
    unit.next_slice_spec()
    assert unit.abort_requests(("job-0",)) == ("job-0",)

    assert unit.pending_work_count == 1
    assert runner.released == ["job-0"]
    descriptor = unit.next_slice_spec()
    assert descriptor.metadata["request_ids"] == ("job-1",)


def test_omni_exec_unit_rejects_invalid_backend_and_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="vllm_omni"):
        OmniModelExecUnit(
            ModelSpec("generator", "model", "vllm"),
            SimpleNamespace(),
            "cuda:0",
        )

    unit, _ = _create_omni_unit(monkeypatch)
    with pytest.raises(ValueError, match="empty work"):
        unit.recv_job(None)
    with pytest.raises(RuntimeError, match="no runnable requests"):
        unit.execute_slice(ExecutionSliceDescriptor(0))
