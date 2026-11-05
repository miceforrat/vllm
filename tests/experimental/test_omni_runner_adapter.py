# SPDX-License-Identifier: Apache-2.0

import threading

from vllm.experimental.omni_runner_adapter import EmbeddedOmniDiffusionRunner


def test_vae_decode_is_split_into_coarse_temporal_segments() -> None:
    runner = EmbeddedOmniDiffusionRunner(object(), device="cpu")
    begins: list[tuple[str, int | None]] = []
    finishes: list[None] = []
    runner._begin_scheduled_phase = (  # type: ignore[method-assign]
        lambda phase, index=None: begins.append((phase, index))
    )
    runner._finish_scheduled_phase = (  # type: ignore[method-assign]
        lambda: finishes.append(None)
    )
    runner._vae_decode_active = True

    for _ in range(31):
        runner._before_vae_decoder_call(None, ())

    # Segment zero is opened by the _decode_latents wrapper. Decoder calls
    # 8, 16 and 24 open the remaining three scheduling segments.
    assert begins == [("vae", 1), ("vae", 2), ("vae", 3)]
    assert len(finishes) == 3
    assert runner._vae_decoder_calls == 31


def test_vae_decoder_hook_is_inert_outside_decode() -> None:
    runner = EmbeddedOmniDiffusionRunner(object(), device="cpu")
    runner._before_vae_decoder_call(None, ())
    assert runner._vae_decoder_calls == 0


def test_runtime_provider_bundle_is_model_scoped() -> None:
    runner = EmbeddedOmniDiffusionRunner(object(), device="cpu")
    runner._auxiliary_tp_group = object()
    runner.set_generator_phase_stream_selector(lambda phase, index: None)

    providers = runner.runtime_providers()

    assert providers.memory is runner
    assert providers.stream is runner
    assert providers.communicator is runner
    assert providers.phase is runner


def test_execution_unit_gate_blocks_until_engine_admission() -> None:
    runner = EmbeddedOmniDiffusionRunner(object(), device="cpu")
    runner._active_execution_request_id = "request-1"
    runner.enable_engine_execution_unit_gate()
    selected: list[str | None] = []

    thread = threading.Thread(
        target=lambda: selected.append(
            runner._await_execution_unit_admission("denoising", 3)
        )
    )
    thread.start()
    with runner._execution_unit_condition:
        runner._execution_unit_condition.wait_for(
            lambda: runner._pending_execution_unit is not None,
            timeout=1.0,
        )
        pending = dict(runner._pending_execution_unit or {})

    assert thread.is_alive()
    assert pending == {
        "request_id": "request-1",
        "sequence": 0,
        "kind": "denoising",
        "index": 3,
    }
    runner.admit_execution_unit(0, "shared")
    thread.join(timeout=1.0)

    assert selected == ["shared"]
    assert runner.pending_execution_unit() is None
