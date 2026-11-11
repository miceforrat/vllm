from multiprocessing.connection import Client
from types import SimpleNamespace

from vllm.experimental.model_instance_registry import ModelRegistry, ModelSpec
from vllm.v1.core.sched.execution_units import (
    ExecutionUnitAdmissionServer,
    ExecutionUnitSpec,
    ExecutionUnitScheduler,
)
from vllm.v1.engine.core import EngineCore


def _units():
    return [
        {"kind": "preparing"},
        {"kind": "denoising", "index": 0},
        {"kind": "vae", "index": 0},
    ]


def test_execution_unit_lifecycle_and_modes():
    scheduler = ExecutionUnitScheduler()
    scheduler.register_request("generator", "req-1", _units())

    admitted = scheduler.admit_next("req-1")
    assert admitted["active_unit"]["selected_mode"] == "generator_full"
    scheduler.start_unit("req-1")
    completed = scheduler.complete_unit("req-1")
    assert completed["cursor"] == 1

    scheduler.set_primary_demand(True)
    admitted = scheduler.admit_next("req-1")
    assert admitted["active_unit"]["selected_mode"] == "shared"
    scheduler.start_unit("req-1")
    scheduler.complete_unit("req-1")
    scheduler.set_primary_demand(False)

    scheduler.admit_next("req-1")
    scheduler.start_unit("req-1")
    completed = scheduler.complete_unit("req-1")
    assert completed["done"] is True


def test_execution_unit_rejects_invalid_transition():
    scheduler = ExecutionUnitScheduler()
    scheduler.register_request("generator", "req-1", _units())

    try:
        scheduler.start_unit("req-1")
    except RuntimeError as exc:
        assert "not admitted" in str(exc)
    else:
        raise AssertionError("start without admission should fail")


def test_execution_units_can_be_discovered_at_runtime_boundaries():
    scheduler = ExecutionUnitScheduler()
    offered = scheduler.offer_unit(
        "generator", "req-stream", {"kind": "denoising", "index": 0}
    )
    assert offered["unit_count"] == 1
    scheduler.admit_next("req-stream")
    scheduler.start_unit("req-stream")
    assert scheduler.complete_unit("req-stream")["done"] is True

    offered = scheduler.offer_unit(
        "generator", "req-stream", {"kind": "vae", "index": 0}
    )
    assert offered["cursor"] == 1
    assert offered["unit_count"] == 2


def test_admission_server_selects_mode_from_scheduler_demand():
    scheduler = ExecutionUnitScheduler()
    server = ExecutionUnitAdmissionServer(scheduler)
    try:
        scheduler.set_primary_demand(True)
        connection = Client(
            server.address, family="AF_UNIX", authkey=server.authkey
        )
        connection.send(
            {
                "operation": "boundary",
                "model_id": "generator",
                "request_id": "req-socket",
                "kind": "denoising",
                "index": 0,
            }
        )
        response = connection.recv()
        connection.close()
        assert response["ok"] is True
        assert (
            response["result"]["active_unit"]["selected_mode"]
            == "shared"
        )
    finally:
        server.close()


def test_engine_slice_scheduler_selects_planned_variants():
    scheduler = ExecutionUnitScheduler()
    scheduler.configure_model(
        "reasoner",
        full_variant="reasoner_full",
        shared_variant="shared",
        primary=True,
    )
    scheduler.configure_model(
        "generator",
        full_variant="full",
        shared_variant="shared",
    )

    alone = scheduler.schedule_slice(
        "generator", "g-1", ExecutionUnitSpec("denoising", 0)
    )
    assert alone.resource_variant == "full"

    scheduler.set_model_demand("reasoner", True)
    scheduler.set_model_demand("generator", True)
    colocated = scheduler.schedule_slice(
        "generator", "g-1", ExecutionUnitSpec("denoising", 1)
    )
    reasoner = scheduler.schedule_model_batch("reasoner", ("r-1",))
    assert colocated.resource_variant == "shared"
    assert reasoner.resource_variant == "shared"
    assert (
        alone.execution_epoch
        < colocated.execution_epoch
        < reasoner.execution_epoch
    )

    fixed = scheduler.schedule_slice(
        "generator",
        "g-2",
        ExecutionUnitSpec(
            "denoising", 2, {"requested_variant": "full"}
        ),
    )
    assert fixed.resource_variant == "full"


def test_tp_ranks_reuse_boundary_decision_until_all_finish():
    scheduler = ExecutionUnitScheduler()
    scheduler.configure_model(
        "generator",
        full_variant="full",
        shared_variant="shared",
        participant_count=2,
    )
    boundary = {
        "model_id": "generator",
        "request_id": "tp-request",
        "kind": "denoising",
        "index": 0,
    }
    first = scheduler.handle_boundary(boundary)
    second = scheduler.handle_boundary(boundary)
    assert first == second
    assert first["cursor"] == 0

    scheduler.finish_boundary_request("tp-request")
    assert scheduler.snapshot()["requests"]
    scheduler.finish_boundary_request("tp-request")
    assert scheduler.snapshot()["requests"] == []


def test_engine_core_owns_model_request_routing():
    registry = ModelRegistry(default_model_id="reasoner")
    registry.register(ModelSpec("reasoner", "model", "vllm"))
    registry.register(ModelSpec("generator", "model", "vllm_omni"))
    demand: list[tuple[str, bool]] = []
    worker_calls: list[tuple[str, tuple]] = []

    units = ExecutionUnitScheduler()
    units.configure_model(
        "reasoner",
        full_variant="reasoner_full",
        shared_variant="shared",
        primary=True,
    )
    units.configure_model(
        "generator", full_variant="generator_full", shared_variant="shared"
    )

    class _ExecutionUnits:
        def set_model_demand(self, model_id, active):
            demand.append((model_id, active))
            return units.set_model_demand(model_id, active)

        def schedule_slice(self, model_id, request_id, spec):
            return units.schedule_slice(model_id, request_id, spec)

    core = EngineCore.__new__(EngineCore)
    core.model_registry = registry
    core._model_request_sessions = {}
    core.scheduler = SimpleNamespace(
        execution_units=_ExecutionUnits()
    )
    def collective_rpc(method, timeout=None, args=(), kwargs=None):
        del timeout, kwargs
        worker_calls.append((method, args))
        if method == "start_model_request":
            return [
                {
                    "rank": 0,
                    "next_slice": {"kind": "preparing", "index": None},
                }
            ]
        if method == "execute_model_request_slice":
            return [{"rank": 0, "request_done": True}]
        raise AssertionError(method)

    core.model_executor = SimpleNamespace(collective_rpc=collective_rpc)
    core._model_request_completion_executor = None
    core._model_request_futures = {}

    native = core.open_model_request("reasoner", "r-1")
    auxiliary_session = core.open_model_request("generator", "g-1")
    auxiliary = core.submit_model_request(
        "generator", "g-1", {"prompt": "video"}, {"variant": "auto"}
    )

    assert native == {"model_id": "reasoner", "request_id": "r-1"}
    assert auxiliary_session == {
        "model_id": "generator",
        "request_id": "g-1",
    }
    assert auxiliary["accepted"] is True
    assert demand == [("reasoner", True), ("generator", True)]
    assert core.wait_model_request_completion("g-1").result() == [
        {"rank": 0, "request_done": True}
    ]
    assert worker_calls == [
        (
            "start_model_request",
            ("generator", {"prompt": "video"}, "auto", 4096, "g-1"),
        ),
        (
            "execute_model_request_slice",
            (
                "g-1",
                {
                    "model_id": "generator",
                    "request_id": "g-1",
                    "kind": "preparing",
                    "index": None,
                    "resource_variant": "shared",
                    "execution_epoch": 1,
                    "metadata": {"requested_variant": "auto"},
                },
            ),
        ),
    ]
    core._model_request_completion_executor.shutdown()
