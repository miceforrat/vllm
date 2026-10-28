# SPDX-License-Identifier: Apache-2.0
"""Gate 2: full-SM Reasoner/Generator concurrency without Green Contexts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import torch

from vllm.experimental.omni_runner_adapter import (
    EmbeddedOmniDiffusionRunner,
    resolve_omni_diffusion_stage,
)


def run_reasoner(reasoner: Any, prompt_text: str, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    tokenizer = reasoner.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt)
    started = time.perf_counter()
    outputs = reasoner.generate(
        [prompt],
        SamplingParams(temperature=0.0, seed=0, max_tokens=max_tokens),
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    candidate = outputs[0].outputs[0]
    return {
        "seconds": elapsed,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(candidate.token_ids),
        "finish_reason": candidate.finish_reason,
        "text_sha256": hashlib.sha256(candidate.text.encode()).hexdigest(),
    }


def build_generator_request(workload: dict[str, Any], request_id: str) -> Any:
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={
            "prompt": workload["prompt"],
            "negative_prompt": workload["negative_prompt"],
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=int(workload["height"]),
            width=int(workload["width"]),
            num_frames=int(workload["num_frames"]),
            num_inference_steps=int(workload["num_inference_steps"]),
            guidance_scale=float(workload["guidance_scale"]),
            seed=int(workload["seed"]),
            output_type="np",
            extra_args={"flow_shift": float(workload["flow_shift"])},
        ),
    )


def find_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = find_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = find_tensor(item)
            if tensor is not None:
                return tensor
    return None


def run_generator(
    embedded: EmbeddedOmniDiffusionRunner,
    workload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    request = build_generator_request(workload, request_id)
    started = time.perf_counter()
    output = embedded.execute_model(request)
    # Synchronize only the Generator stream. A device-wide synchronize here
    # would wait for Reasoner work and invalidate the concurrency experiment.
    if embedded.execution_stream is None:
        torch.cuda.current_stream().synchronize()
    else:
        embedded.execution_stream.synchronize()
    elapsed = time.perf_counter() - started
    payload = output.output
    if payload is None and output.media is not None:
        payload = output.media.video.tensor
    tensor = find_tensor(payload)
    if tensor is None:
        raise RuntimeError("Generator output contains no tensor")
    return {
        "seconds": elapsed,
        "output_shape": list(tensor.shape),
        "output_dtype": str(tensor.dtype),
    }


def concurrent_once(
    executor: ThreadPoolExecutor,
    *,
    reasoner: Any,
    embedded: EmbeddedOmniDiffusionRunner,
    reasoner_prompt: str,
    reasoner_max_tokens: int,
    generator_workload: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    barrier = threading.Barrier(3)

    def reasoner_worker() -> dict[str, Any]:
        barrier.wait(timeout=60.0)
        host_start = time.perf_counter()
        with torch.cuda.nvtx.range(f"gate2.reasoner.iter{index}"):
            result = run_reasoner(reasoner, reasoner_prompt, reasoner_max_tokens)
        host_end = time.perf_counter()
        return {"host_start": host_start, "host_end": host_end, **result}

    def generator_worker() -> dict[str, Any]:
        barrier.wait(timeout=60.0)
        host_start = time.perf_counter()
        with torch.cuda.nvtx.range(f"gate2.generator.iter{index}"):
            result = run_generator(
                embedded,
                generator_workload,
                request_id=f"single-framework-gate2-concurrent-{index:02d}",
            )
        host_end = time.perf_counter()
        return {"host_start": host_start, "host_end": host_end, **result}

    reasoner_future = executor.submit(reasoner_worker)
    generator_future = executor.submit(generator_worker)
    wall_start = time.perf_counter()
    barrier.wait(timeout=60.0)
    reasoner_result = reasoner_future.result()
    generator_result = generator_future.result()
    wall_end = time.perf_counter()
    overlap = max(
        0.0,
        min(reasoner_result["host_end"], generator_result["host_end"])
        - max(reasoner_result["host_start"], generator_result["host_start"]),
    )
    shorter = min(reasoner_result["seconds"], generator_result["seconds"])
    return {
        "index": index,
        "reasoner": reasoner_result,
        "generator": generator_result,
        "host_wall_seconds": wall_end - wall_start,
        "host_overlap_seconds": overlap,
        "host_overlap_over_shorter": overlap / shorter if shorter else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--reasoner-input", type=Path, required=True)
    parser.add_argument("--generator-workload-manifest", type=Path, required=True)
    parser.add_argument("--reasoner-mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--generator-cache", choices=("none", "sea_cache"), default="sea_cache")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--alone-repeats", type=int, default=3)
    parser.add_argument("--concurrent-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gate_2_no_gc/result.json"),
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.alone_repeats < 1 or args.concurrent_repeats < 1:
        parser.error("warmups and repeats must all be at least one")

    reasoner_input = json.loads(args.reasoner_input.read_text(encoding="utf-8"))
    reasoner_prompt = reasoner_input["prompt"]
    reasoner_max_tokens = int(reasoner_input["max_new_tokens"])
    manifest = json.loads(args.generator_workload_manifest.read_text(encoding="utf-8"))
    generator_workload = manifest["generator"]["canonical_workload"]

    cache_overrides: dict[str, Any]
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
    else:
        cache_overrides = {"cache_backend": "none"}

    embedded = EmbeddedOmniDiffusionRunner(
        resolve_omni_diffusion_stage(args.model, cli_overrides=cache_overrides)
    )
    generator = embedded.start(load_format="default")
    embedded.create_execution_stream()

    from vllm import LLM

    execution = (
        {"enforce_eager": True}
        if args.reasoner_mode == "eager"
        else {
            "enforce_eager": False,
            "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        }
    )
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

    warmups = []
    for index in range(args.warmups):
        r = run_reasoner(reasoner, reasoner_prompt, reasoner_max_tokens)
        g = run_generator(
            embedded,
            generator_workload,
            request_id=f"single-framework-gate2-warmup-{index:02d}",
        )
        warmups.append({"index": index, "reasoner": r, "generator": g})
        print(
            f"warmup[{index}] reasoner={r['seconds']:.3f}s "
            f"generator={g['seconds']:.3f}s",
            flush=True,
        )

    alone_reasoner = []
    alone_generator = []
    serial_walls = []
    for index in range(args.alone_repeats):
        serial_start = time.perf_counter()
        r = run_reasoner(reasoner, reasoner_prompt, reasoner_max_tokens)
        g = run_generator(
            embedded,
            generator_workload,
            request_id=f"single-framework-gate2-alone-{index:02d}",
        )
        serial_walls.append(time.perf_counter() - serial_start)
        alone_reasoner.append(r)
        alone_generator.append(g)
        print(
            f"alone[{index}] reasoner={r['seconds']:.3f}s "
            f"generator={g['seconds']:.3f}s serial={serial_walls[-1]:.3f}s",
            flush=True,
        )

    concurrent = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gate2") as executor:
        for index in range(args.concurrent_repeats):
            row = concurrent_once(
                executor,
                reasoner=reasoner,
                embedded=embedded,
                reasoner_prompt=reasoner_prompt,
                reasoner_max_tokens=reasoner_max_tokens,
                generator_workload=generator_workload,
                index=index,
            )
            concurrent.append(row)
            print(
                f"concurrent[{index}] reasoner={row['reasoner']['seconds']:.3f}s "
                f"generator={row['generator']['seconds']:.3f}s "
                f"wall={row['host_wall_seconds']:.3f}s",
                flush=True,
            )

    r_alone = statistics.median(x["seconds"] for x in alone_reasoner)
    g_alone = statistics.median(x["seconds"] for x in alone_generator)
    serial_wall = statistics.median(serial_walls)
    r_concurrent = statistics.median(x["reasoner"]["seconds"] for x in concurrent)
    g_concurrent = statistics.median(x["generator"]["seconds"] for x in concurrent)
    concurrent_wall = statistics.median(x["host_wall_seconds"] for x in concurrent)
    report = {
        "gate": "gate_2_full_sm_concurrency_no_green_context",
        "spatial_sharing_enabled": False,
        "gpu_kernel_overlap_verified": False,
        "model": args.model,
        "reasoner_mode": args.reasoner_mode,
        "generator_mode": "regional_compile",
        "generator_cache": args.generator_cache,
        "reasoner_fixed_length": False,
        "workload": {
            "reasoner_input": str(args.reasoner_input),
            "reasoner_prompt_tokens": alone_reasoner[-1]["prompt_tokens"],
            "reasoner_max_tokens": reasoner_max_tokens,
            "generator_manifest": str(args.generator_workload_manifest),
            "generator": {
                key: generator_workload[key]
                for key in (
                    "height", "width", "num_frames", "num_inference_steps",
                    "guidance_scale", "flow_shift", "seed",
                )
            },
            "negative_prompt_empty": not bool(generator_workload["negative_prompt"]),
        },
        "stream_control": embedded.stream_control_stats(),
        "warmups": warmups,
        "alone": {
            "reasoner": alone_reasoner,
            "generator": alone_generator,
            "serial_host_wall_seconds": serial_walls,
        },
        "concurrent": concurrent,
        "summary": {
            "reasoner_alone_seconds": r_alone,
            "generator_alone_seconds": g_alone,
            "serial_host_wall_seconds": serial_wall,
            "reasoner_concurrent_seconds": r_concurrent,
            "generator_concurrent_seconds": g_concurrent,
            "concurrent_host_wall_seconds": concurrent_wall,
            "reasoner_slowdown": r_concurrent / r_alone,
            "generator_slowdown": g_concurrent / g_alone,
            "throughput_gain_host_wall": serial_wall / concurrent_wall,
            "host_overlap_seconds": statistics.median(
                x["host_overlap_seconds"] for x in concurrent
            ),
            "host_overlap_over_shorter": statistics.median(
                x["host_overlap_over_shorter"] for x in concurrent
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)

    embedded.close()
    reasoner.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
