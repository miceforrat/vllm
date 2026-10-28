# SPDX-License-Identifier: Apache-2.0
"""Execute a real request through Omni's original diffusion model runner."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vllm.experimental.omni_runner_adapter import (
    EmbeddedOmniDiffusionRunner,
    resolve_omni_diffusion_stage,
)


def _describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Image.Image):
        return {"type": "image", "size": list(value.size), "mode": value.mode}
    if isinstance(value, dict):
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(item) for item in value]
    return {"type": type(value).__name__}


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _sea_cache_stats(runner: Any) -> dict[str, Any] | None:
    pipeline = runner.pipeline
    transformer = getattr(pipeline, "transformer", None)
    registry = getattr(transformer, "_hook_registry", None)
    hook = registry.get_hook("sea_cache") if registry is not None else None
    if hook is None:
        return None
    states = getattr(getattr(hook, "state_manager", None), "_states", {})
    persistent_bytes = 0
    persistent_tensors = 0
    for state in states.values():
        tensors = list(getattr(state, "previous_indicator", None) or [])
        tensors += [item[1] for item in getattr(state, "history", [])]
        for tensor in tensors:
            if torch.is_tensor(tensor):
                persistent_tensors += 1
                persistent_bytes += tensor.numel() * tensor.element_size()
    return {
        "hook_type": f"{type(hook).__module__}.{type(hook).__name__}",
        "full_count": int(getattr(hook, "full_count", 0)),
        "skip_count": int(getattr(hook, "skip_count", 0)),
        "state_contexts": len(states),
        "persistent_tensors": persistent_tensors,
        "persistent_tensor_bytes": persistent_bytes,
    }


def _und_cache_stats(runner: Any) -> dict[str, Any] | None:
    transformer = getattr(runner.pipeline, "transformer", None)
    if transformer is None:
        return None
    return {
        "persistent_tensors": int(
            getattr(transformer, "_und_cache_persistent_tensors", 0)
        ),
        "persistent_tensor_bytes": int(
            getattr(transformer, "_und_cache_persistent_tensor_bytes", 0)
        ),
        "memory_pool_provider_installed": (
            getattr(transformer, "_cache_memory_pool_context_fn", None) is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument(
        "--prompt", default="A red robot repairs a bicycle in a bright workshop."
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--cache-backend", choices=("none", "sea_cache"), default="none"
    )
    parser.add_argument(
        "--execution-stream",
        choices=("default", "dedicated", "green"),
        default="default",
    )
    parser.add_argument("--green-sm-count", type=int, default=120)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/omni_runner_gate_r2")
    )
    args = parser.parse_args()

    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cache_overrides = {}
    if args.cache_backend == "sea_cache":
        cache_overrides = {
            "cache_backend": "sea_cache",
            "cache_config": {
                "sea_threshold": 0.25,
                "sea_residual_order": 1,
                "sea_max_consecutive_cached": 2,
                "sea_power_exp": 3.0,
            },
        }
    resolved = resolve_omni_diffusion_stage(
        args.model, cli_overrides=cache_overrides
    )
    embedded = EmbeddedOmniDiffusionRunner(resolved)
    load_start = time.perf_counter()
    runner = embedded.start(load_format="default")
    if args.execution_stream == "dedicated":
        embedded.create_execution_stream()
    elif args.execution_stream == "green":
        embedded.create_green_context_execution_stream(args.green_sm_count)
    load_seconds = time.perf_counter() - load_start
    model_load_pool_bytes = embedded.memory_pool_usage()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    for index in range(args.repeats):
        request = OmniDiffusionRequest(
            request_id=f"omni-runner-gate-r2-{index}",
            prompt={
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
            },
            sampling_params=OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_frames=1,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                output_type="np",
            ),
        )
        started = time.perf_counter()
        output, memory_profile = embedded.execute_model_profiled(request)
        elapsed = time.perf_counter() - started
        pool_bytes_before = memory_profile.tagged_pool_before_bytes
        pool_bytes_after = memory_profile.tagged_pool_after_bytes
        payload = output.output
        if payload is None and output.media is not None:
            payload = output.media.video.tensor
        artifact = args.output_dir / f"output_{index:02d}.pt"
        torch.save(_to_cpu(payload), artifact)
        output_description = _describe(payload)
        stage_durations = output.stage_durations
        peak_memory_mb = output.peak_memory_mb
        profile_data = dataclasses.asdict(memory_profile)
        del payload, output
        gc.collect()
        torch.cuda.synchronize()
        profile_data["quiescent_allocated_bytes"] = int(
            torch.cuda.memory_allocated(embedded.device)
        )
        profile_data["quiescent_reserved_bytes"] = int(
            torch.cuda.memory_reserved(embedded.device)
        )
        profile_data["quiescent_live_delta_bytes"] = (
            profile_data["quiescent_allocated_bytes"]
            - profile_data["baseline_allocated_bytes"]
        )
        measurements.append(
            {
                "index": index,
                "seconds": elapsed,
                "output": output_description,
                "stage_durations": stage_durations,
                "peak_memory_mb": peak_memory_mb,
                "memory_profile": profile_data,
                "memory_pool_bytes_before": pool_bytes_before,
                "memory_pool_bytes_after": pool_bytes_after,
                "memory_pool_growth_bytes": (
                    None
                    if pool_bytes_before is None or pool_bytes_after is None
                    else pool_bytes_after - pool_bytes_before
                ),
                "artifact": str(artifact),
            }
        )

    report = {
        "runner_type": f"{type(runner).__module__}.{type(runner).__name__}",
        "pipeline_type": type(runner.pipeline).__name__,
        "requested_cache_backend": args.cache_backend,
        "effective_cache_backend": (
            None
            if runner.cache_backend is None
            else f"{type(runner.cache_backend).__module__}."
            f"{type(runner.cache_backend).__name__}"
        ),
        "sea_cache_stats": _sea_cache_stats(runner),
        "und_cache_stats": _und_cache_stats(runner),
        "resource_owner": "vllm",
        "stream_control": embedded.stream_control_stats(),
        "vllm_memory_pool": {
            "provider": "vllm.device_allocator",
            "enabled": embedded.use_vllm_memory_pool,
            "model_load_tracked_bytes": model_load_pool_bytes,
            "final_tracked_bytes": embedded.memory_pool_usage(),
        },
        "load_seconds": load_seconds,
        "measurements": measurements,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    embedded.close()


if __name__ == "__main__":
    main()
