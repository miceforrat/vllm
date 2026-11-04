# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import vllm.experimental.multi_model_worker as multi_model_worker
from vllm.experimental.communicator_registry import CommunicatorRegistry
from vllm.experimental.model_instance_registry import (
    ModelInstance,
    ModelInstanceRegistry,
)
from vllm.experimental.multi_model_topology import TopologyPlan, TopologyPlanner
from vllm.experimental.multi_model_worker import (
    AsyncWorkerHostedRunnerClient,
    CommunicationGroupSpec,
    ModelSpec,
    MultiModelWorkerExtension,
    SchedulerModelRequest,
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
        self.multi_model_topology_planner = TopologyPlanner(
            self.vllm_config.parallel_config
        )
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def collective_rpc(
        self, method: str, *, args: tuple[Any, ...]
    ) -> list[Any]:
        self.calls.append((method, args))
        return self.replies[method]


class _FakeAsyncEngine(_FakeEngine):
    async def collective_rpc(
        self, method: str, *, args: tuple[Any, ...]
    ) -> list[Any]:
        self.calls.append((method, args))
        return self.replies[method]


def test_client_submits_serializable_model_spec() -> None:
    engine = _FakeEngine(
        {
            "register_model": [
                {"model_id": "generator-a", "active": True}
            ]
        }
    )
    client = WorkerHostedRunnerClient(engine)
    spec = ModelSpec(
        model_id="generator-a",
        model_path="nvidia/Cosmos3-Edge",
        backend="vllm_omni",
        capabilities=("text_to_video",),
        default_task="text_to_video",
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
    assert method == "register_model"
    assert args[0]["worker_ranks"] == (0, 1)
    assert args[0]["communication_groups"][0]["group_id"] == "generator-a-tp"
    assert args[0]["backend_options"]["cache_backend"] == "sea_cache"


def test_submit_routes_primary_model_to_scheduler_frontend() -> None:
    engine = _FakeEngine({})

    class _Frontend:
        llm_engine = engine

        def generate(self, prompt: Any, params: Any, **kwargs: Any) -> list[Any]:
            assert prompt == ["hello"]
            assert params == "sampling"
            assert kwargs == {"use_tqdm": False}
            return ["reasoner-output"]

    client = WorkerHostedRunnerClient(_Frontend())
    future = client.submit(
        "reasoner",
        SchedulerModelRequest(
            prompt=["hello"],
            sampling_params="sampling",
            request_id="request-1",
            generate_kwargs={"use_tqdm": False},
        ),
    )

    assert future.result() == ["reasoner-output"]
    assert engine.calls == []


def test_submit_routes_runtime_model_to_worker_rpc() -> None:
    engine = _FakeEngine(
        {
            "start_auxiliary_runner": [
                {"job_id": "unused", "started": True}
            ]
        }
    )
    client = WorkerHostedRunnerClient(engine)
    client._runtime_model_specs["generator-a"] = ModelSpec(
        "generator-a", "generator", "vllm_omni"
    )

    # Keep the assertion independent of the randomly generated job id.
    engine.collective_rpc = lambda method, *, args: [
        {"job_id": args[3], "started": True}
    ]
    future = client.submit("generator-a", "generator-request")

    assert future.model_id == "generator-a"


def test_worker_skips_model_not_placed_on_its_rank() -> None:
    worker = MultiModelWorkerExtension()
    worker.rank = 2

    result = worker.register_auxiliary_runner(
        ModelSpec(
            model_id="generator-a",
            model_path="nvidia/Cosmos3-Edge",
            backend="vllm_omni",
            capabilities=("text_to_video",),
            default_task="text_to_video",
            worker_ranks=(0, 1),
        ).to_rpc_payload(),
        TopologyPlan(
            model_id="generator-a",
            worker_ranks=(0, 1),
            communication_groups=(),
        ).to_rpc_payload(),
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


def test_client_updates_primary_model_demand() -> None:
    expected = [{"primary_demand": 1}, {"primary_demand": 1}]
    engine = _FakeEngine({"set_primary_model_demand": expected})
    client = WorkerHostedRunnerClient(engine)

    result = client.set_primary_demand(True)

    assert result == expected
    assert engine.calls == [("set_primary_model_demand", (True,))]


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


def test_client_starts_and_collects_external_async_job(monkeypatch) -> None:
    class _JobId:
        hex = "job-123"

    monkeypatch.setattr(multi_model_worker.uuid, "uuid4", _JobId)
    result = {"seconds": 2.5}
    engine = _FakeEngine(
        {
            "start_auxiliary_runner": [
                {"job_id": "job-123", "started": True},
                {"job_id": "job-123", "started": True},
            ],
            "auxiliary_runner_job_status": [
                {"job_id": "job-123", "done": True},
                {"job_id": "job-123", "done": True},
            ],
            "collect_auxiliary_runner": [result, None],
        }
    )
    client = WorkerHostedRunnerClient(engine)
    request = object()

    future = client.start_async("generator-a", request)

    assert future.model_id == "generator-a"
    assert future.job_id == "job-123"
    assert future.done()
    assert future.result() is result
    assert future.result() is result
    assert engine.calls == [
        (
            "start_auxiliary_runner",
            ("generator-a", request, False, "job-123"),
        ),
        (
            "auxiliary_runner_job_status",
            ("generator-a", "job-123"),
        ),
        (
            "collect_auxiliary_runner",
            ("generator-a", "job-123", None),
        ),
    ]


def test_async_client_starts_and_collects_external_job(monkeypatch) -> None:
    class _JobId:
        hex = "async-job-123"

    monkeypatch.setattr(multi_model_worker.uuid, "uuid4", _JobId)
    result = {"seconds": 1.25}
    engine = _FakeAsyncEngine(
        {
            "start_auxiliary_runner": [
                {"job_id": "async-job-123", "started": True},
                {"job_id": "async-job-123", "started": True},
            ],
            "auxiliary_runner_job_status": [
                {"job_id": "async-job-123", "done": True},
                {"job_id": "async-job-123", "done": True},
            ],
            "collect_auxiliary_runner": [result, None],
        }
    )
    client = AsyncWorkerHostedRunnerClient(engine)
    request = object()

    async def exercise() -> None:
        future = await client.start_async("generator-a", request)
        assert future.model_id == "generator-a"
        assert future.job_id == "async-job-123"
        assert await future.done()
        assert await future.result() is result
        assert await future.result() is result

    asyncio.run(exercise())
    assert engine.calls == [
        (
            "start_auxiliary_runner",
            ("generator-a", request, False, "async-job-123"),
        ),
        (
            "auxiliary_runner_job_status",
            ("generator-a", "async-job-123"),
        ),
        (
            "collect_auxiliary_runner",
            ("generator-a", "async-job-123", None),
        ),
    ]


def test_async_submit_routes_primary_model_to_async_llm() -> None:
    class _AsyncFrontend(_FakeAsyncEngine):
        async def generate(self, **kwargs: Any):
            assert kwargs["prompt"] == "hello"
            assert kwargs["sampling_params"] == "sampling"
            assert kwargs["request_id"] == "request-1"
            yield "partial"
            yield "final"

    client = AsyncWorkerHostedRunnerClient(_AsyncFrontend({}))

    async def exercise() -> None:
        future = await client.submit(
            "reasoner",
            SchedulerModelRequest(
                prompt="hello",
                sampling_params="sampling",
                request_id="request-1",
            ),
        )
        assert await future.result() == "final"

    asyncio.run(exercise())



def test_model_spec_restores_nested_communication_groups() -> None:
    original = ModelSpec(
        model_id="generator-a",
        model_path="model",
        backend="vllm_omni",
        capabilities=("text_to_video",),
        default_task="text_to_video",
        worker_ranks=(0, 1),
        communication_groups=(
            CommunicationGroupSpec(
                group_id="generator-a-tp",
                parallel_dimension="tp",
                ranks=(0, 1),
            ),
        ),
    )

    restored = ModelSpec.from_rpc_payload(original.to_rpc_payload())

    assert restored == original
    assert isinstance(restored.communication_groups[0], CommunicationGroupSpec)


def test_registry_keeps_primary_and_runtime_added_models() -> None:
    registry = ModelInstanceRegistry()
    primary = ModelInstance(
        spec=ModelSpec("reasoner", "reasoner", "vllm"),
        runner=object(),
        scheduler_managed=True,
    )
    generator = ModelInstance(
        spec=ModelSpec("generator", "generator", "vllm_omni"),
        runner=object(),
        runtime_added=True,
    )

    registry.register(primary)
    registry.register(generator)

    assert registry.values() == (primary, generator)
    assert registry.runtime_instances() == (generator,)
    assert registry.remove("generator") is generator
    assert registry.values() == (primary,)


def test_registry_does_not_remove_engine_owned_primary() -> None:
    registry = ModelInstanceRegistry()
    registry.register(
        ModelInstance(
            spec=ModelSpec("reasoner", "reasoner", "vllm"),
            runner=object(),
            scheduler_managed=True,
        )
    )

    try:
        registry.remove("reasoner")
    except ValueError as exc:
        assert "Engine lifecycle" in str(exc)
    else:
        raise AssertionError("Engine-owned primary model was removed")


def test_worker_group_registry_reuses_and_releases_by_owner() -> None:
    class _FakeGroup:
        def __init__(self) -> None:
            self.destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1

    spec = CommunicationGroupSpec(
        group_id="shared-tp",
        parallel_dimension="tp",
        ranks=(0, 1),
    )
    group = _FakeGroup()
    registry = CommunicatorRegistry(group_factory=lambda _: group)

    assert registry.acquire(spec, "runner-a") is group
    assert registry.acquire(spec, "runner-b") is group
    registry.release("runner-a", (spec.group_id,))
    assert group.destroy_count == 0
    registry.release("runner-b", (spec.group_id,))
    assert group.destroy_count == 1
    try:
        registry.get(spec.group_id)
    except KeyError:
        pass
    else:
        raise AssertionError("Released communicator remained registered")


def test_client_derives_homogeneous_tp_topology_from_engine() -> None:
    engine = _FakeEngine(
        {"register_model": [{"model_id": "generator-b"}]}
    )
    client = WorkerHostedRunnerClient(engine)
    spec = ModelSpec(
        model_id="generator-b",
        model_path="model",
        backend="vllm_omni",
        capabilities=("text_to_video",),
        default_task="text_to_video",
    )

    client.register(spec)

    spec_payload, topology_payload = engine.calls[0][1]
    assert spec_payload["worker_ranks"] == ()
    assert spec_payload["communication_groups"] == ()
    assert topology_payload["worker_ranks"] == (0, 1)
    assert topology_payload["communication_groups"] == (
        {
            "group_id": "generator-b-tp",
            "parallel_dimension": "tp",
            "ranks": (0, 1),
            "backend": None,
            "use_device_communicator": True,
        },
    )
