from multiprocessing.connection import Client

from vllm.v1.core.sched.execution_units import (
    ExecutionUnitAdmissionServer,
    ExecutionUnitScheduler,
)


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
