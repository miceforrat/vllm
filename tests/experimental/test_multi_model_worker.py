# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.experimental.multi_model_worker import (
    CommunicationGroupSpec,
    ModelDeploymentSpec,
    MultiModelWorkerExtension,
    WorkerHostedRunnerClient,
)




class _FakeParallelConfig:
    tensor_parallel_size = 2
    pipeline_parallel_size = 1


class _FakeVllmConfig:
    parallel_config = _FakeParallelConfig()


class _FakeEngine:
    def __init__(self, replies: dict[str, list[Any]]):
        self.replies = replies
        self.vllm_config = _FakeVllmConfig()
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def collective_rpc(
        self, method: str, *, args: tuple[Any, ...]
    ) -> list[Any]:
        self.calls.append((method, args))
        return self.replies[method]


def test_client_submits_serializable_model_spec() -> None:
    engine = _FakeEngine(
        {
            "register_auxiliary_runner": [
                {"model_id": "generator-a", "active": True}
            ]
        }
    )
    client = WorkerHostedRunnerClient(engine)
    spec = ModelDeploymentSpec(
        model_id="generator-a",
        model_path="nvidia/Cosmos3-Edge",
        backend="vllm_omni",
        task="text_to_video",
        worker_ranks=(0, 1),
        communication_groups=(
            CommunicationGroupSpec(
                group_id="generator-a-tp",
                parallel_dimension="tp",
                ranks=(0, 1),
            ),
        ),
        backend_options={"cache_backend": "sea_cache"},
    )

    result = client.register(spec)

    assert result == [{"model_id": "generator-a", "active": True}]
    method, args = engine.calls[0]
    assert method == "register_auxiliary_runner"
    assert args[0]["worker_ranks"] == (0, 1)
    assert args[0]["communication_groups"][0]["group_id"] == "generator-a-tp"
    assert args[0]["backend_options"]["cache_backend"] == "sea_cache"


def test_worker_skips_model_not_placed_on_its_rank() -> None:
    worker = MultiModelWorkerExtension()
    worker.rank = 2

    result = worker.register_auxiliary_runner(
        ModelDeploymentSpec(
            model_id="generator-a",
            model_path="nvidia/Cosmos3-Edge",
            backend="vllm_omni",
            task="text_to_video",
            worker_ranks=(0, 1),
        ).to_rpc_payload()
    )

    assert result == {"model_id": "generator-a", "active": False, "rank": 2}
    assert worker._auxiliary_runner_slots() == {}

def test_client_requests_worker_topology() -> None:
    topology = [{"rank": 0}, {"rank": 1}]
    engine = _FakeEngine({"inspect_multi_model_topology": topology})
    client = WorkerHostedRunnerClient(engine)

    result = client.topology((0, 1))

    assert result == topology
    assert engine.calls == [
        ("inspect_multi_model_topology", ((0, 1),)),
    ]


def test_client_selects_only_tp_output_owner() -> None:
    expected = {"output": "rank-zero"}
    engine = _FakeEngine(
        {"execute_auxiliary_runner": [expected, None]}
    )
    client = WorkerHostedRunnerClient(engine)

    result = client.execute("generator-a", object())

    assert result is expected


def test_client_rejects_ambiguous_tp_outputs() -> None:
    engine = _FakeEngine(
        {"execute_auxiliary_runner": [{"rank": 0}, {"rank": 1}]}
    )
    client = WorkerHostedRunnerClient(engine)

    try:
        client.execute("generator-a", object())
    except RuntimeError as exc:
        assert "exactly one output-owner rank" in str(exc)
    else:
        raise AssertionError("Ambiguous TP outputs were accepted")

def test_client_timed_execute_returns_only_owner_timing() -> None:
    timing = {"host_start": 1.0, "host_end": 2.5, "seconds": 1.5}
    request = object()
    engine = _FakeEngine(
        {"execute_auxiliary_runner_timed": [timing, None]}
    )
    client = WorkerHostedRunnerClient(engine)

    result = client.execute_timed("generator-a", request)

    assert result is timing
    assert engine.calls == [
        (
            "execute_auxiliary_runner_timed",
            ("generator-a", request, False),
        )
    ]
    assert "output" not in result


def test_client_timed_execute_rejects_ambiguous_tp_results() -> None:
    engine = _FakeEngine(
        {
            "execute_auxiliary_runner_timed": [
                {"seconds": 1.0},
                {"seconds": 1.1},
            ]
        }
    )
    client = WorkerHostedRunnerClient(engine)

    try:
        client.execute_timed("generator-a", object())
    except RuntimeError as exc:
        assert "exactly one timed output-owner result" in str(exc)
    else:
        raise AssertionError("Ambiguous timed TP results were accepted")



def test_model_spec_restores_nested_communication_groups() -> None:
    original = ModelDeploymentSpec(
        model_id="generator-a",
        model_path="model",
        backend="vllm_omni",
        task="text_to_video",
        worker_ranks=(0, 1),
        communication_groups=(
            CommunicationGroupSpec(
                group_id="generator-a-tp",
                parallel_dimension="tp",
                ranks=(0, 1),
            ),
        ),
    )

    restored = ModelDeploymentSpec.from_rpc_payload(original.to_rpc_payload())

    assert restored == original
    assert isinstance(restored.communication_groups[0], CommunicationGroupSpec)


def test_worker_group_registry_reuses_and_releases_by_owner() -> None:
    class _FakeGroup:
        def __init__(self) -> None:
            self.destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1

    worker = MultiModelWorkerExtension()
    spec = CommunicationGroupSpec(
        group_id="shared-tp",
        parallel_dimension="tp",
        ranks=(0, 1),
    )
    group = _FakeGroup()
    from vllm.experimental.multi_model_worker import _CommunicationGroupSlot

    worker._vllm_communication_group_slots = {
        spec.group_id: _CommunicationGroupSlot(spec, group, {"runner-a"})
    }

    assert worker._acquire_communication_group(spec, "runner-b") is group
    worker._release_communication_groups("runner-a", (spec.group_id,))
    assert group.destroy_count == 0
    worker._release_communication_groups("runner-b", (spec.group_id,))
    assert group.destroy_count == 1
    assert worker._communication_group_slots() == {}


def test_client_derives_homogeneous_tp_topology_from_engine() -> None:
    engine = _FakeEngine(
        {"register_auxiliary_runner": [{"model_id": "generator-b"}]}
    )
    client = WorkerHostedRunnerClient(engine)
    spec = ModelDeploymentSpec(
        model_id="generator-b",
        model_path="model",
        backend="vllm_omni",
        task="text_to_video",
    )

    client.register(spec)

    payload = engine.calls[0][1][0]
    assert payload["worker_ranks"] == (0, 1)
    assert payload["communication_groups"] == (
        {
            "group_id": "generator-b-tp",
            "parallel_dimension": "tp",
            "ranks": (0, 1),
            "backend": None,
            "use_device_communicator": True,
        },
    )
