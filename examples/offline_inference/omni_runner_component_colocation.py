# SPDX-License-Identifier: Apache-2.0
"""Component-level GC colocation: vLLM decode versus one Cosmos3 stage."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import torch

from omni_runner_gc_concurrent_gate import (
    build_generator_request,
    find_tensor,
)
from vllm.experimental.omni_runner_adapter import (
    EmbeddedOmniDiffusionRunner,
    resolve_omni_diffusion_stage,
)


class Region:
    def __init__(self, name: str, barrier: threading.Barrier | None = None):
        self.name = name
        self.barrier = barrier
        self.entries = 0
        self.iterations = 0
        self.host_start: float | None = None
        self.host_end: float | None = None
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None

    def enter(self, stream: torch.cuda.Stream) -> None:
        self.entries += 1
        if self.entries != 1:
            raise RuntimeError(f"region {self.name!r} entered more than once")
        if self.barrier is not None:
            self.barrier.wait(timeout=300.0)
        self.host_start = time.perf_counter()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record(stream)

    def leave(self, stream: torch.cuda.Stream) -> None:
        if self.end_event is None:
            raise RuntimeError(f"region {self.name!r} left before entry")
        self.end_event.record(stream)
        self.end_event.synchronize()
        self.host_end = time.perf_counter()

    def result(self) -> dict[str, Any]:
        if any(value is None for value in (self.host_start, self.host_end, self.start_event, self.end_event)):
            raise RuntimeError(f"region {self.name!r} is incomplete")
        return {
            "gpu_seconds": self.start_event.elapsed_time(self.end_event) / 1000.0,
            "host_seconds": self.host_end - self.host_start,
            "host_start": self.host_start,
            "host_end": self.host_end,
            "entries": self.entries,
            "iterations": self.iterations,
        }


def reasoner_model_runner(reasoner: Any) -> Any:
    return (
        reasoner.llm_engine.engine_core.engine_core.model_executor
        .driver_worker.worker.model_runner
    )


@contextmanager
def decode_region_hook(reasoner: Any, region: Region):
    runner = reasoner_model_runner(reasoner)
    original = runner.execute_model

    @functools.wraps(original)
    def wrapped(scheduler_output: Any, *args: Any, **kwargs: Any):
        from vllm.v1.utils import compute_iteration_details

        details = compute_iteration_details(scheduler_output)
        if details.num_generation_requests:
            region.iterations += 1
            if region.entries == 0:
                region.enter(torch.cuda.current_stream())
        return original(scheduler_output, *args, **kwargs)

    runner.execute_model = wrapped
    try:
        yield
    finally:
        runner.execute_model = original


@contextmanager
def generator_region_hook(
    embedded: EmbeddedOmniDiffusionRunner,
    component: str,
    region: Region,
):
    pipeline = embedded.runner.pipeline
    if pipeline is None:
        raise RuntimeError("Omni diffusion pipeline is not loaded")
    attribute = "diffuse" if component == "diffusion_sampling" else "_decode_latents"
    if not hasattr(pipeline, attribute):
        raise RuntimeError(
            f"Pipeline {type(pipeline).__name__} has no component boundary {attribute!r}"
        )
    original = getattr(pipeline, attribute)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        stream = torch.cuda.current_stream()
        region.enter(stream)
        try:
            with torch.cuda.nvtx.range(f"component.generator.{component}"):
                return original(*args, **kwargs)
        finally:
            region.leave(stream)

    setattr(pipeline, attribute, wrapped)
    try:
        yield
    finally:
        setattr(pipeline, attribute, original)


def run_generator(
    embedded: EmbeddedOmniDiffusionRunner,
    workload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = embedded.execute_model(build_generator_request(workload, request_id))
    stream = embedded.execution_stream or torch.cuda.current_stream()
    stream.synchronize()
    payload = output.output
    if payload is None and output.media is not None:
        payload = output.media.video.tensor
    tensor = find_tensor(payload)
    if tensor is None:
        raise RuntimeError("Generator output contains no tensor")
    return {
        "request_seconds": time.perf_counter() - started,
        "output_shape": list(tensor.shape),
        "output_dtype": str(tensor.dtype),
    }


def run_reasoner_benchmark(
    reasoner: Any,
    prompt_text: str,
    max_tokens: int,
    stream: torch.cuda.Stream,
    fixed_tokens: bool,
) -> dict[str, Any]:
    from vllm import SamplingParams

    tokenizer = reasoner.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt)
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        outputs = reasoner.generate(
            [prompt],
            SamplingParams(
                temperature=0.0,
                seed=0,
                max_tokens=max_tokens,
                ignore_eos=fixed_tokens,
            ),
            use_tqdm=False,
        )
    candidate = outputs[0].outputs[0]
    return {
        "seconds": time.perf_counter() - started,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(candidate.token_ids),
        "finish_reason": candidate.finish_reason,
        "fixed_tokens": fixed_tokens,
    }


def measured_decode(
    reasoner: Any,
    prompt: str,
    max_tokens: int,
    stream: torch.cuda.Stream,
    barrier: threading.Barrier | None,
    fixed_tokens: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    region = Region("llm_decode", barrier)
    with decode_region_hook(reasoner, region):
        request = run_reasoner_benchmark(
            reasoner, prompt, max_tokens, stream, fixed_tokens
        )
    region.leave(stream)
    return request, region.result()


def measured_component(
    embedded: EmbeddedOmniDiffusionRunner,
    workload: dict[str, Any],
    component: str,
    request_id: str,
    barrier: threading.Barrier | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    region = Region(component, barrier)
    with generator_region_hook(embedded, component, region):
        request = run_generator(embedded, workload, request_id)
    return request, region.result()


def median(rows: list[dict[str, Any]], side: str, field: str) -> float:
    return statistics.median(float(row[side]["region"][field]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--reasoner-input", type=Path, required=True)
    parser.add_argument("--generator-workload-manifest", type=Path, required=True)
    parser.add_argument(
        "--component", choices=("vae_decode", "diffusion_sampling"), required=True
    )
    parser.add_argument(
        "--coverage",
        choices=("decode_covers_component", "component_covers_decode"),
        required=True,
    )
    parser.add_argument("--reasoner-mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--reasoner-max-tokens", type=int)
    parser.add_argument(
        "--reasoner-fixed-tokens",
        action="store_true",
        help="Ignore EOS to create a controlled decode-length benchmark.",
    )
    parser.add_argument("--generator-sm-count", type=int, default=136)
    parser.add_argument(
        "--gc-workqueue-mode",
        choices=("shared", "balanced"),
        default="shared",
        help=(
            "Use the legacy shared CUDA workqueue or attach one balanced "
            "workqueue to each Green Context."
        ),
    )
    parser.add_argument(
        "--gc-workqueue-concurrency-limit",
        type=int,
        default=1,
        help="Concurrency limit for each balanced CUDA workqueue.",
    )
    parser.add_argument(
        "--generator-workqueue-concurrency-limit",
        type=int,
        help="Override the balanced Workqueue limit for the Generator GC.",
    )
    parser.add_argument(
        "--reasoner-workqueue-concurrency-limit",
        type=int,
        help="Override the balanced Workqueue limit for the Reasoner GC.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--generator-cache", choices=("none", "sea_cache"), default="sea_cache")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--alone-repeats", type=int, default=1)
    parser.add_argument("--concurrent-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gc_workqueue_concurrency_limit <= 0:
        parser.error("--gc-workqueue-concurrency-limit must be positive")
    if (
        args.generator_workqueue_concurrency_limit is not None
        and args.generator_workqueue_concurrency_limit <= 0
    ):
        parser.error("--generator-workqueue-concurrency-limit must be positive")
    if (
        args.reasoner_workqueue_concurrency_limit is not None
        and args.reasoner_workqueue_concurrency_limit <= 0
    ):
        parser.error("--reasoner-workqueue-concurrency-limit must be positive")

    reasoner_input = json.loads(args.reasoner_input.read_text())
    prompt = reasoner_input["prompt"]
    max_tokens = args.reasoner_max_tokens or int(reasoner_input["max_new_tokens"])
    manifest = json.loads(args.generator_workload_manifest.read_text())
    workload = manifest["generator"]["canonical_workload"]

    cache_overrides: dict[str, Any] = {"cache_backend": "none"}
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
    embedded = EmbeddedOmniDiffusionRunner(
        resolve_omni_diffusion_stage(args.model, cli_overrides=cache_overrides)
    )
    embedded.start(load_format="default")

    from vllm.experimental.green_context import GreenContextPair

    device = torch.cuda.current_device()
    gc_pair = GreenContextPair(
        args.generator_sm_count,
        device,
        balanced_workqueues=args.gc_workqueue_mode == "balanced",
        workqueue_concurrency_limit=args.gc_workqueue_concurrency_limit,
        first_workqueue_concurrency_limit=(
            args.generator_workqueue_concurrency_limit
        ),
        second_workqueue_concurrency_limit=(
            args.reasoner_workqueue_concurrency_limit
        ),
    )
    generator_stream = torch.cuda.ExternalStream(gc_pair.first_stream_ptr, device=device)
    decode_stream = torch.cuda.ExternalStream(gc_pair.second_stream_ptr, device=device)
    embedded.set_execution_stream(generator_stream)

    from vllm import LLM
    from vllm.experimental.runtime_stream_patch import install_vllm_graph_capture_stream_patch

    execution = {"enforce_eager": True}
    graph_patch = None
    if args.reasoner_mode == "graph":
        execution = {
            "enforce_eager": False,
            "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        }
        graph_patch = install_vllm_graph_capture_stream_patch(decode_stream)
    with torch.cuda.stream(decode_stream):
        reasoner = LLM(
            model=args.model,
            tensor_parallel_size=1,
            max_num_seqs=1,
            max_model_len=max(4096, max_tokens + 512),
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            enable_prefix_caching=False,
            disable_log_stats=True,
            **execution,
        )

    print(
        f"partition generator={gc_pair.first_sm_count} decode={gc_pair.second_sm_count} "
        f"component={args.component} coverage={args.coverage} tokens={max_tokens}",
        flush=True,
    )

    warmups = []
    for i in range(args.warmups):
        r = run_reasoner_benchmark(
            reasoner, prompt, max_tokens, decode_stream, args.reasoner_fixed_tokens
        )
        g = run_generator(embedded, workload, f"component-warmup-{i}")
        warmups.append({"reasoner": r, "generator": g})
        print(f"warmup[{i}] reasoner={r['seconds']:.3f}s generator={g['request_seconds']:.3f}s", flush=True)

    alone = []
    for i in range(args.alone_repeats):
        rr, rd = measured_decode(
            reasoner, prompt, max_tokens, decode_stream, None,
            args.reasoner_fixed_tokens,
        )
        gr, gd = measured_component(
            embedded, workload, args.component, f"component-alone-{i}", None
        )
        alone.append({
            "index": i,
            "decode": {"request": rr, "region": rd},
            "component": {"request": gr, "region": gd},
        })
        print(
            f"alone[{i}] decode={rd['host_seconds']:.3f}s "
            f"{args.component}={gd['host_seconds']:.3f}s",
            flush=True,
        )

    concurrent = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="component-gate") as pool:
        for i in range(args.concurrent_repeats):
            barrier = threading.Barrier(2)
            rf = pool.submit(
                measured_decode, reasoner, prompt, max_tokens, decode_stream, barrier,
                args.reasoner_fixed_tokens,
            )
            gf = pool.submit(
                measured_component,
                embedded,
                workload,
                args.component,
                f"component-concurrent-{i}",
                barrier,
            )
            rr, rd = rf.result()
            gr, gd = gf.result()
            overlap = max(
                0.0,
                min(rd["host_end"], gd["host_end"])
                - max(rd["host_start"], gd["host_start"]),
            )
            row = {
                "index": i,
                "decode": {"request": rr, "region": rd},
                "component": {"request": gr, "region": gd},
                "overlap_seconds": overlap,
            }
            concurrent.append(row)
            print(
                f"concurrent[{i}] decode={rd['host_seconds']:.3f}s "
                f"{args.component}={gd['host_seconds']:.3f}s overlap={overlap:.3f}s",
                flush=True,
            )

    decode_alone = median(alone, "decode", "host_seconds")
    component_alone = median(alone, "component", "host_seconds")
    decode_colocated = median(concurrent, "decode", "host_seconds")
    component_colocated = median(concurrent, "component", "host_seconds")
    decode_covers = statistics.median(
        float(r["decode"]["region"]["host_end"] >= r["component"]["region"]["host_end"])
        for r in concurrent
    ) == 1.0
    component_covers = statistics.median(
        float(r["component"]["region"]["host_end"] >= r["decode"]["region"]["host_end"])
        for r in concurrent
    ) == 1.0
    expected_coverage = decode_covers if args.coverage == "decode_covers_component" else component_covers
    summary = {
        "decode_partitioned_alone_seconds": decode_alone,
        "component_partitioned_alone_seconds": component_alone,
        "decode_colocated_seconds": decode_colocated,
        "component_colocated_seconds": component_colocated,
        "decode_colocation_slowdown": decode_colocated / decode_alone,
        "component_colocation_slowdown": component_colocated / component_alone,
        "decode_covers_component": decode_covers,
        "component_covers_decode": component_covers,
        "requested_coverage_verified": expected_coverage,
        "overlap_seconds": statistics.median(r["overlap_seconds"] for r in concurrent),
    }
    report = {
        "experiment": "llm_decode_x_generator_component",
        "component": args.component,
        "requested_coverage": args.coverage,
        "partition": {
            "requested_generator_sm_count": args.generator_sm_count,
            "actual_generator_sm_count": gc_pair.first_sm_count,
            "actual_decode_sm_count": gc_pair.second_sm_count,
            "total_sm_count": gc_pair.total_sm_count,
        },
        "workqueue": {
            "mode": args.gc_workqueue_mode,
            "concurrency_limit": args.gc_workqueue_concurrency_limit,
            "requested_generator_concurrency_limit": (
                args.generator_workqueue_concurrency_limit
            ),
            "requested_reasoner_concurrency_limit": (
                args.reasoner_workqueue_concurrency_limit
            ),
            "context_configs": gc_pair.workqueue_configs,
        },
        "reasoner_mode": args.reasoner_mode,
        "generator_mode": "regional_compile",
        "generator_cache": args.generator_cache,
        "workload": {
            "reasoner_input": str(args.reasoner_input),
            "reasoner_max_tokens": max_tokens,
            "reasoner_fixed_tokens": args.reasoner_fixed_tokens,
            "generator_manifest": str(args.generator_workload_manifest),
        },
        "warmups": warmups,
        "alone": alone,
        "concurrent": concurrent,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    generator_stream.synchronize()
    decode_stream.synchronize()
    embedded.close()
    reasoner.llm_engine.engine_core.shutdown()
    if graph_patch is not None:
        graph_patch.restore()
    gc_pair.close()


if __name__ == "__main__":
    main()
