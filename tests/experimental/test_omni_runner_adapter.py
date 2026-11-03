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

