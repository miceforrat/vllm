# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.worker.exec_unit import (
    LLMExecUnit,
    OmniModelExecUnit,
    OmniStepResult,
)


class _Runner:
    def __init__(self) -> None:
        self.executed: list[Any] = []

    def execute_model(self, scheduler_output: Any) -> str:
        self.executed.append(scheduler_output)
        return f"result:{scheduler_output}"


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
    assert unit.step() == "result:batch-0"
    assert unit.step() == "result:batch-1"
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

    assert unit.step() == "worker-result:batch"
    assert executed == ["batch"]
    assert runner.executed == []


def test_llm_exec_unit_rejects_invalid_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit, _ = _create_unit(monkeypatch)
    with pytest.raises(ValueError, match="empty work"):
        unit.recv_job(None)
    with pytest.raises(RuntimeError, match="no runnable work"):
        unit.step()


class _OmniRunner:
    def __init__(self) -> None:
        self.executed: list[Any] = []
        self.unit: OmniModelExecUnit | None = None

    def prepare_explicit_request(self, request: Any) -> Any:
        self.executed.append(request)
        return _OmniState(request)

    def execute_explicit_denoising_step(self, state: Any) -> Any:
        state.step_index += 1
        return state

    def prepare_explicit_vae_blocks(self, state: Any) -> None:
        state.extra["vae_block_index"] = 0

    def execute_explicit_vae_block(self, state: Any) -> bool:
        state.extra["vae_block_index"] += 1
        return state.extra["vae_block_index"] >= 2

    def finalize_explicit_vae_blocks(self, state: Any) -> str:
        return f"omni-result:{state.request}"


class _OmniState:
    def __init__(self, request: Any) -> None:
        self.request = request
        self.step_index = 0
        self.extra: dict[str, Any] = {}

    @property
    def denoise_completed(self) -> bool:
        return self.step_index >= 2


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

    unit.recv_job("request-0")
    unit.recv_job("request-1")
    assert unit.pending_work_count == 2

    expected_prefix = [
        OmniStepResult("preparing", None, False),
        OmniStepResult("denoising", 0, False),
        OmniStepResult("denoising", 1, False),
    ]
    assert [unit.step() for _ in range(3)] == expected_prefix
    assert unit.step() == OmniStepResult("vae_blocks", 0, False)
    assert unit.step() == OmniStepResult(
        "vae_blocks", 1, True, "omni-result:request-0"
    )
    assert [unit.step() for _ in range(3)] == expected_prefix
    assert unit.step() == OmniStepResult("vae_blocks", 0, False)
    assert unit.step() == OmniStepResult(
        "vae_blocks", 1, True, "omni-result:request-1"
    )
    assert runner.executed == ["request-0", "request-1"]


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
    with pytest.raises(RuntimeError, match="no runnable work"):
        unit.step()
