# SPDX-License-Identifier: Apache-2.0

import threading

from vllm.experimental.multi_model_arbiter import (
    GeneratorExecutionPhase,
    MultiModelResourceArbiter,
)


def test_primary_demand_waits_for_full_generator_step() -> None:
    full_stream = object()
    shared_stream = object()
    arbiter = MultiModelResourceArbiter(
        full_generator_stream=full_stream,
        shared_generator_stream=shared_stream,
    )
    full_lease = arbiter.acquire_generator_step(0)
    assert full_lease.mode == "generator_full"
    assert full_lease.stream is full_stream

    admitted = threading.Event()

    def request_primary() -> None:
        arbiter.acquire_primary_demand()
        admitted.set()

    waiter = threading.Thread(target=request_primary)
    waiter.start()
    assert not admitted.wait(timeout=0.01)

    full_lease.finish()
    assert admitted.wait(timeout=1.0)
    waiter.join(timeout=1.0)

    shared_lease = arbiter.acquire_generator_step(1)
    assert shared_lease.mode == "shared"
    assert shared_lease.stream is shared_stream
    shared_lease.finish()

    arbiter.release_primary_demand()
    next_full_lease = arbiter.acquire_generator_step(2)
    assert next_full_lease.mode == "generator_full"
    next_full_lease.finish()

    stats = arbiter.stats()
    assert stats["generator_full_steps"] == 2
    assert stats["shared_steps"] == 1
    assert stats["primary_demand"] == 0


def test_generator_exposes_preparing_denoising_vae_and_complete() -> None:
    arbiter = MultiModelResourceArbiter(
        full_generator_stream="full",
        shared_generator_stream="shared",
    )

    for phase, step_index in (
        (GeneratorExecutionPhase.PREPARING, None),
        (GeneratorExecutionPhase.DENOISING, 0),
        (GeneratorExecutionPhase.DENOISING, 1),
        (GeneratorExecutionPhase.VAE, None),
    ):
        lease = arbiter.acquire_generator_phase(phase, step_index)
        assert lease is not None
        assert lease.phase is phase
        assert lease.step_index == step_index
        assert lease.mode == "generator_full"
        lease.finish()

    assert arbiter.acquire_generator_phase("complete") is None
    stats = arbiter.stats()
    assert stats["last_generator_phase"] == "complete"
    assert stats["generator_phase_counts"] == {
        "preparing": 1,
        "denoising": 2,
        "vae": 1,
        "complete": 1,
    }


def test_primary_resource_epoch_spans_multiple_scheduler_batches() -> None:
    arbiter = MultiModelResourceArbiter(
        full_generator_stream="full",
        shared_generator_stream="shared",
    )

    assert arbiter.note_primary_batch_submitted() is None
    arbiter.acquire_primary_demand()
    epoch = arbiter.note_primary_batch_submitted()
    assert epoch == arbiter.note_primary_batch_submitted()
    arbiter.acquire_primary_demand()
    assert epoch == arbiter.note_primary_batch_submitted()
    arbiter.release_primary_demand()
    arbiter.release_primary_demand()

    stats = arbiter.stats()
    assert stats["primary_epoch_id"] == epoch
    assert not stats["primary_epoch_active"]
    assert stats["primary_batches_submitted"] == 3
    assert stats["primary_unscoped_batches"] == 1


def test_fixed_shared_policy_never_changes_generator_partition() -> None:
    arbiter = MultiModelResourceArbiter(
        full_generator_stream="full",
        shared_generator_stream="shared",
        generator_phase_policy="fixed_shared",
    )

    for phase in ("preparing", "denoising", "vae"):
        lease = arbiter.acquire_generator_phase(phase)
        assert lease is not None
        assert lease.mode == "shared"
        assert lease.stream == "shared"
        lease.finish()

    stats = arbiter.stats()
    assert stats["generator_phase_policy"] == "fixed_shared"
    assert stats["generator_full_steps"] == 0
    assert stats["shared_steps"] == 3


def test_primary_uses_full_lane_after_generator_finishes() -> None:
    arbiter = MultiModelResourceArbiter(
        full_generator_stream="generator-full",
        shared_generator_stream="generator-shared",
        full_primary_stream="reasoner-full",
        shared_primary_stream="reasoner-shared",
    )
    arbiter.acquire_primary_demand()

    stream, mode = arbiter.select_primary_batch_stream()
    assert (stream, mode) == ("reasoner-full", "reasoner_full")
    arbiter.finish_primary_batch(mode, None)

    shared_generator = arbiter.acquire_generator_phase("denoising", 0)
    assert shared_generator is not None
    stream, mode = arbiter.select_primary_batch_stream()
    assert (stream, mode) == ("reasoner-shared", "shared")
    arbiter.finish_primary_batch(mode, None)
    shared_generator.finish()

    stream, mode = arbiter.select_primary_batch_stream()
    assert (stream, mode) == ("reasoner-shared", "shared")
    arbiter.finish_primary_batch(mode, None)
    arbiter.acquire_generator_phase("complete")
    stream, mode = arbiter.select_primary_batch_stream()
    assert (stream, mode) == ("reasoner-full", "reasoner_full")
    arbiter.finish_primary_batch(mode, None)
    arbiter.release_primary_demand()


def test_generator_admission_drains_active_full_primary_batch() -> None:
    class _Event:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            self.synchronized = True

    arbiter = MultiModelResourceArbiter(
        full_generator_stream="generator-full",
        shared_generator_stream="generator-shared",
        full_primary_stream="reasoner-full",
        shared_primary_stream="reasoner-shared",
    )
    arbiter.acquire_primary_demand()
    stream, mode = arbiter.select_primary_batch_stream()
    assert (stream, mode) == ("reasoner-full", "reasoner_full")

    admitted = threading.Event()
    leases = []

    def start_generator() -> None:
        leases.append(arbiter.acquire_generator_phase("preparing"))
        admitted.set()

    waiter = threading.Thread(target=start_generator)
    waiter.start()
    assert not admitted.wait(timeout=0.01)

    completion = _Event()
    arbiter.finish_primary_batch(mode, completion)
    assert admitted.wait(timeout=1.0)
    waiter.join(timeout=1.0)

    assert completion.synchronized
    assert leases[0] is not None
    assert leases[0].mode == "shared"
    stats = arbiter.stats()
    assert stats["generator_waits_for_primary"] == 1
    assert stats["primary_completion_drains"] == 1
    leases[0].finish()
    arbiter.acquire_generator_phase("complete")
    arbiter.release_primary_demand()
