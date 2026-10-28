# SPDX-License-Identifier: Apache-2.0
"""Gate R3: execute vLLM Reasoner and the original Omni Runner in one PID."""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

# A second EngineCore process would invalidate this gate's ownership claim.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import torch

from vllm.experimental.omni_runner_adapter import (
    EmbeddedOmniDiffusionRunner,
    resolve_omni_diffusion_stage,
)


def _cuda_context() -> str | None:
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuInit(0) != 0:
        return None
    context = ctypes.c_void_p()
    if driver.cuCtxGetCurrent(ctypes.byref(context)) != 0 or not context.value:
        return None
    return hex(context.value)


def _gpu_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = []
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 3:
            result.append(
                {"pid": int(fields[0]), "name": fields[1], "used_mib": int(fields[2])}
            )
    return result


def _snapshot(label: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "label": label,
        "pid": os.getpid(),
        "cuda_context": _cuda_context(),
        "cuda_free_mib": free / 2**20,
        "cuda_total_mib": total / 2**20,
        "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "gpu_processes": _gpu_processes(),
    }


def _run_reasoner(reasoner: Any, prompt_text: str, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    messages = [{"role": "user", "content": prompt_text}]
    tokenizer = reasoner.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_token_ids = tokenizer.encode(prompt)
    started = time.perf_counter()
    outputs = reasoner.generate(
        [prompt],
        SamplingParams(temperature=0.0, seed=0, max_tokens=max_tokens),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    candidate = outputs[0].outputs[0]
    return {
        "seconds": elapsed,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": len(candidate.token_ids),
        "finish_reason": candidate.finish_reason,
        "text_sha256": hashlib.sha256(candidate.text.encode()).hexdigest(),
        "text_tail": candidate.text[-160:],
    }


def _find_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("video", "image", "images", "frames", "samples"):
            if key in value:
                tensor = _find_tensor(value[key])
                if tensor is not None:
                    return tensor
        for item in value.values():
            tensor = _find_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _find_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _run_generator(
    embedded: Any, args: argparse.Namespace, request_id: str
) -> dict[str, Any]:
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    request = OmniDiffusionRequest(
        request_id=request_id,
        prompt={"prompt": args.prompt, "negative_prompt": args.negative_prompt},
        sampling_params=OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_type="np",
            extra_args={"flow_shift": args.flow_shift},
        ),
    )
    started = time.perf_counter()
    output, memory_profile = embedded.execute_model_profiled(request)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = output.output
    if payload is None and output.media is not None:
        payload = output.media.video.tensor
    tensor = _find_tensor(payload)
    if tensor is None:
        raise RuntimeError(
            f"Generator output contains no tensor: {type(payload).__name__}"
        )
    return {
        "seconds": elapsed,
        "output_shape": list(tensor.shape),
        "output_dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all()),
        "peak_memory_mb": output.peak_memory_mb,
        "memory_profile": dataclasses.asdict(memory_profile),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--reasoner-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--reasoner-max-tokens", type=int, default=64)
    parser.add_argument("--reasoner-input", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--prompt", default="A red robot repairs a bicycle in a bright workshop."
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--generator-workload-manifest", type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--generator-cache", choices=("none", "sea_cache"), default="sea_cache"
    )
    parser.add_argument(
        "--generator-execution-stream",
        choices=("default", "dedicated", "green"),
        default="dedicated",
    )
    parser.add_argument("--generator-sm-count", type=int, default=120)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/omni_runner_gate_r3/report.json")
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be >= 0 and --repeats must be >= 1")

    reasoner_prompt = "Describe a safe plan for a robot to move a cup onto a shelf."
    if args.reasoner_input is not None:
        reasoner_input = json.loads(args.reasoner_input.read_text(encoding="utf-8"))
        reasoner_prompt = reasoner_input["prompt"]
        args.reasoner_max_tokens = int(
            reasoner_input.get("max_new_tokens", args.reasoner_max_tokens)
        )

    if args.generator_workload_manifest is not None:
        manifest = json.loads(
            args.generator_workload_manifest.read_text(encoding="utf-8")
        )
        workload = manifest["generator"]["canonical_workload"]
        args.prompt = workload["prompt"]
        args.negative_prompt = workload["negative_prompt"]
        args.height = int(workload["height"])
        args.width = int(workload["width"])
        args.num_frames = int(workload["num_frames"])
        args.steps = int(workload["num_inference_steps"])
        args.guidance_scale = float(workload["guidance_scale"])
        args.flow_shift = float(workload["flow_shift"])
        args.seed = int(workload["seed"])

    snapshots = [_snapshot("process_start")]
    cache_overrides = {}
    if args.generator_cache == "sea_cache":
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
    generator_load_start = time.perf_counter()
    generator = embedded.start(load_format="default")
    if args.generator_execution_stream == "dedicated":
        embedded.create_execution_stream()
    elif args.generator_execution_stream == "green":
        embedded.create_green_context_execution_stream(args.generator_sm_count)
    generator_load_seconds = time.perf_counter() - generator_load_start
    snapshots.append(_snapshot("generator_loaded"))

    from vllm import LLM

    execution = (
        {"enforce_eager": True}
        if args.reasoner_mode == "eager"
        else {
            "enforce_eager": False,
            "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        }
    )
    reasoner_load_start = time.perf_counter()
    reasoner = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        enable_prefix_caching=False,
        disable_log_stats=True,
        **execution,
    )
    reasoner_load_seconds = time.perf_counter() - reasoner_load_start
    snapshots.append(_snapshot("both_models_loaded"))

    # Match the historical dual-framework measurement order exactly:
    # finish all Reasoner trials, then finish all Generator trials.
    reasoner_warmups = []
    reasoner_measured = []
    generator_warmups = []
    generator_measured = []
    for index in range(args.warmups):
        result = _run_reasoner(reasoner, reasoner_prompt, args.reasoner_max_tokens)
        reasoner_warmups.append(result)
        print(
            f"reasoner warmup[{index}]={result['seconds']:.3f}s "
            f"tokens={result['completion_tokens']}",
            flush=True,
        )
    for index in range(args.repeats):
        result = _run_reasoner(reasoner, reasoner_prompt, args.reasoner_max_tokens)
        reasoner_measured.append(result)
        print(
            f"reasoner measured[{index}]={result['seconds']:.3f}s "
            f"tokens={result['completion_tokens']}",
            flush=True,
        )
    for index in range(args.warmups):
        result = _run_generator(
            embedded,
            args,
            request_id=f"omni-runner-gate-r3-warmup-{index:02d}",
        )
        generator_warmups.append(result)
        print(f"generator warmup[{index}]={result['seconds']:.3f}s", flush=True)
    for index in range(args.repeats):
        result = _run_generator(
            embedded,
            args,
            request_id=f"omni-runner-gate-r3-measured-{index:02d}",
        )
        generator_measured.append(result)
        print(f"generator measured[{index}]={result['seconds']:.3f}s", flush=True)

    warmup_results = [
        {
            "index": index,
            "reasoner": reasoner_warmups[index],
            "generator": generator_warmups[index],
            "serial_seconds": reasoner_warmups[index]["seconds"]
            + generator_warmups[index]["seconds"],
        }
        for index in range(args.warmups)
    ]
    measured_results = [
        {
            "index": index,
            "reasoner": reasoner_measured[index],
            "generator": generator_measured[index],
            "serial_seconds": reasoner_measured[index]["seconds"]
            + generator_measured[index]["seconds"],
        }
        for index in range(args.repeats)
    ]

    reasoner_result = dict(measured_results[-1]["reasoner"])
    generator_result = dict(measured_results[-1]["generator"])
    timing_summary = {
        "reasoner_median_seconds": statistics.median(
            row["reasoner"]["seconds"] for row in measured_results
        ),
        "generator_median_seconds": statistics.median(
            row["generator"]["seconds"] for row in measured_results
        ),
        "serial_median_seconds": statistics.median(
            row["serial_seconds"] for row in measured_results
        ),
    }
    snapshots.append(_snapshot("measurements_complete"))

    pids = {snapshot["pid"] for snapshot in snapshots}
    contexts = {snapshot["cuda_context"] for snapshot in snapshots}
    report = {
        "gate": "R3 sequential dual-model single-process",
        "model": args.model,
        "reasoner_type": f"{type(reasoner).__module__}.{type(reasoner).__name__}",
        "generator_runner_type": f"{type(generator).__module__}.{type(generator).__name__}",
        "generator_pipeline_type": type(generator.pipeline).__name__,
        "resource_owner": "vllm",
        "measurement": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "order": "blocked_reasoner_then_generator",
            "warmup_results": warmup_results,
            "measured_results": measured_results,
            "summary": timing_summary,
        },
        "workload": {
            "reasoner_input": (
                None if args.reasoner_input is None else str(args.reasoner_input)
            ),
            "reasoner_max_tokens": args.reasoner_max_tokens,
            "reasoner_fixed_length": False,
            "generator_manifest": (
                None
                if args.generator_workload_manifest is None
                else str(args.generator_workload_manifest)
            ),
            "generator": {
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "flow_shift": args.flow_shift,
                "seed": args.seed,
                "negative_prompt_empty": not bool(args.negative_prompt),
                "negative_prompt_chars": len(args.negative_prompt),
                "negative_prompt_sha256": hashlib.sha256(
                    args.negative_prompt.encode()
                ).hexdigest(),
            },
        },
        "generator_cache_backend": (
            None
            if generator.cache_backend is None
            else f"{type(generator.cache_backend).__module__}.{type(generator.cache_backend).__name__}"
        ),
        "generator_stream_control": embedded.stream_control_stats(),
        "diffusion_weights_memory_pool": {
            "provider": "vllm.device_allocator",
            "enabled": embedded.use_vllm_memory_pool,
            "tracked_bytes": embedded.memory_pool_usage(),
        },
        "same_pid": len(pids) == 1,
        "same_cuda_context": len(contexts) == 1 and None not in contexts,
        "generator_load_seconds": generator_load_seconds,
        "reasoner_load_seconds": reasoner_load_seconds,
        "reasoner": reasoner_result,
        "generator": generator_result,
        "snapshots": snapshots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    # Remove Omni's borrowed compatibility views before the vLLM owner tears
    # down its in-process EngineCore and distributed resources.
    embedded.close()
    reasoner.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
