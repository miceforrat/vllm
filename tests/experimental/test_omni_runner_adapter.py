# SPDX-License-Identifier: Apache-2.0

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


def test_execution_unit_boundary_uses_engine_scheduler() -> None:
    runner = EmbeddedOmniDiffusionRunner(object(), device="cpu")
    runner._active_execution_request_id = "request-1"
    runner._execution_unit_model_id = "generator-a"
    runner._execution_unit_scheduler_address = "/tmp/scheduler.sock"
    messages: list[dict[str, object]] = []

    def call(message: dict[str, object]) -> dict[str, object]:
        messages.append(message)
        return {"active_unit": {"selected_mode": "shared"}}

    runner._execution_unit_scheduler_call = call  # type: ignore[method-assign]

    assert runner._await_execution_unit_admission("denoising", 3) == "shared"
    assert messages == [
        {
            "operation": "boundary",
            "model_id": "generator-a",
            "request_id": "request-1",
            "kind": "denoising",
            "index": 3,
        }
    ]
