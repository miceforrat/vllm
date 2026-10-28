# SPDX-License-Identifier: Apache-2.0
"""Gate 3: complementary Green Context Reasoner/Generator concurrency."""

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


def run_reasoner(
    reasoner: Any,
    prompt_text: str,
    max_tokens: int,
    stream: torch.cuda.Stream,
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
    reasoner_stream: torch.cuda.Stream,
    generator_workload: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    barrier = threading.Barrier(3)

    def reasoner_worker() -> dict[str, Any]:
        barrier.wait(timeout=60.0)
        host_start = time.perf_counter()
        with torch.cuda.nvtx.range(f"gate3.concurrent.reasoner.iter{index}"):
            result = run_reasoner(
                reasoner, reasoner_prompt, reasoner_max_tokens, reasoner_stream
            )
        host_end = time.perf_counter()
        return {"host_start": host_start, "host_end": host_end, **result}

    def generator_worker() -> dict[str, Any]:
        barrier.wait(timeout=60.0)
        host_start = time.perf_counter()
        with torch.cuda.nvtx.range(f"gate3.concurrent.generator.iter{index}"):
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
    parser.add_argument("--generator-sm-count", type=int, default=136)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--alone-repeats", type=int, default=3)
    parser.add_argument("--concurrent-repeats", type=int, default=3)
    parser.add_argument(
        "--full-sm-reference",
        type=Path,
        help="Matched Gate 2 full-SM result used for the serial baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gate_3_green_context/result.json"),
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.alone_repeats < 1 or args.concurrent_repeats < 1:
        parser.error("warmups and repeats must all be at least one")

    reasoner_input = json.loads(args.reasoner_input.read_text(encoding="utf-8"))
    reasoner_prompt = reasoner_input["prompt"]
    reasoner_max_tokens = int(reasoner_input["max_new_tokens"])
    manifest = json.loads(args.generator_workload_manifest.read_text(encoding="utf-8"))
    generator_workload = manifest["generator"]["canonical_workload"]

    full_sm_reference = None
    if args.full_sm_reference is not None:
        full_sm_reference = json.loads(
            args.full_sm_reference.read_text(encoding="utf-8")
        )
        reference_workload = full_sm_reference["workload"]
        expected_generator = {
            key: generator_workload[key]
            for key in (
                "height", "width", "num_frames", "num_inference_steps",
                "guidance_scale", "flow_shift", "seed",
            )
        }
        if (
            int(reference_workload["reasoner_max_tokens"]) != reasoner_max_tokens
            or reference_workload["generator"] != expected_generator
            or bool(reference_workload["negative_prompt_empty"])
            != (not bool(generator_workload["negative_prompt"]))
        ):
            raise ValueError("The full-SM reference does not match this workload")

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

    from vllm.experimental.green_context import GreenContextPair

    device_index = torch.cuda.current_device()
    gc_pair = GreenContextPair(args.generator_sm_count, device_index)
    generator_stream = torch.cuda.ExternalStream(
        gc_pair.first_stream_ptr, device=device_index
    )
    reasoner_stream = torch.cuda.ExternalStream(
        gc_pair.second_stream_ptr, device=device_index
    )
    embedded.set_execution_stream(generator_stream)
    print(
        "GC partition: "
        f"requested_generator={args.generator_sm_count} "
        f"actual_generator={gc_pair.first_sm_count} "
        f"actual_reasoner={gc_pair.second_sm_count} "
        f"total={gc_pair.total_sm_count}",
        flush=True,
    )

    from vllm import LLM
    from vllm.experimental.runtime_stream_patch import (
        install_vllm_graph_capture_stream_patch,
    )

    execution = (
        {"enforce_eager": True}
        if args.reasoner_mode == "eager"
        else {
            "enforce_eager": False,
            "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        }
    )
    reasoner_graph_stream_patch = (
        install_vllm_graph_capture_stream_patch(reasoner_stream)
        if args.reasoner_mode == "graph"
        else None
    )
    # vLLM has three capture-stream entry points.  The explicit patch above
    # binds all of them; the outer stream context also covers eager warmup and
    # non-graph setup work.
    with torch.cuda.stream(reasoner_stream):
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
    if reasoner_graph_stream_patch is not None:
        graph_stats = reasoner_graph_stream_patch.stats()
        if graph_stats["graph_capture_calls"] == 0:
            raise RuntimeError("vLLM did not enter the patched graph-capture context")
        if (
            graph_stats["manager_capture_stream_calls"]
            + graph_stats["compiled_capture_stream_calls"]
            == 0
        ):
            raise RuntimeError("vLLM did not request the patched inner capture stream")
    else:
        graph_stats = None

    warmups = []
    for index in range(args.warmups):
        r = run_reasoner(
            reasoner, reasoner_prompt, reasoner_max_tokens, reasoner_stream
        )
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
        with torch.cuda.nvtx.range(f"gate3.alone.reasoner.iter{index}"):
            r = run_reasoner(
                reasoner, reasoner_prompt, reasoner_max_tokens, reasoner_stream
            )
        with torch.cuda.nvtx.range(f"gate3.alone.generator.iter{index}"):
            g = run_generator(
                embedded,
                generator_workload,
                request_id=f"single-framework-gate3-alone-{index:02d}",
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
                reasoner_stream=reasoner_stream,
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
    full_sm_summary = (
        None if full_sm_reference is None else full_sm_reference["summary"]
    )
    full_sm_serial_wall = (
        None
        if full_sm_summary is None
        else float(full_sm_summary["serial_host_wall_seconds"])
    )
    full_sm_reasoner = (
        None
        if full_sm_summary is None
        else float(full_sm_summary["reasoner_alone_seconds"])
    )
    full_sm_generator = (
        None
        if full_sm_summary is None
        else float(full_sm_summary["generator_alone_seconds"])
    )
    report = {
        "gate": "gate_3_complementary_green_context_concurrency",
        "spatial_sharing_enabled": True,
        "gpu_kernel_overlap_verified": False,
        "model": args.model,
        "reasoner_mode": args.reasoner_mode,
        "generator_mode": "regional_compile",
        "generator_cache": args.generator_cache,
        "reasoner_fixed_length": False,
        "partition": {
            "requested_generator_sm_count": args.generator_sm_count,
            "actual_generator_sm_count": gc_pair.first_sm_count,
            "actual_reasoner_sm_count": gc_pair.second_sm_count,
            "total_sm_count": gc_pair.total_sm_count,
        },
        "full_sm_reference": (
            None
            if full_sm_reference is None
            else {
                "path": str(args.full_sm_reference),
                "workload_match_verified": True,
                "reasoner_alone_seconds": full_sm_reasoner,
                "generator_alone_seconds": full_sm_generator,
                "serial_host_wall_seconds": full_sm_serial_wall,
            }
        ),
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
        "reasoner_graph_stream_control": graph_stats,
        "warmups": warmups,
        "alone": {
            "reasoner": alone_reasoner,
            "generator": alone_generator,
            "serial_host_wall_seconds": serial_walls,
        },
        "concurrent": concurrent,
        "summary": {
            "reasoner_full_sm_alone_seconds": full_sm_reasoner,
            "generator_full_sm_alone_seconds": full_sm_generator,
            "full_sm_serial_host_wall_seconds": full_sm_serial_wall,
            "reasoner_partitioned_alone_seconds": r_alone,
            "generator_partitioned_alone_seconds": g_alone,
            "partitioned_serial_host_wall_seconds": serial_wall,
            "reasoner_concurrent_seconds": r_concurrent,
            "generator_concurrent_seconds": g_concurrent,
            "concurrent_host_wall_seconds": concurrent_wall,
            "reasoner_sm_restriction_slowdown": (
                None if full_sm_reasoner is None else r_alone / full_sm_reasoner
            ),
            "generator_sm_restriction_slowdown": (
                None if full_sm_generator is None else g_alone / full_sm_generator
            ),
            "reasoner_colocation_slowdown": r_concurrent / r_alone,
            "generator_colocation_slowdown": g_concurrent / g_alone,
            "throughput_gain_full_sm_reference": (
                None
                if full_sm_serial_wall is None
                else full_sm_serial_wall / concurrent_wall
            ),
            "throughput_gain_partitioned_reference": serial_wall / concurrent_wall,
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

    generator_stream.synchronize()
    reasoner_stream.synchronize()
    embedded.close()
    reasoner.llm_engine.engine_core.shutdown()
    if reasoner_graph_stream_patch is not None:
        reasoner_graph_stream_patch.restore()
    gc_pair.close()


if __name__ == "__main__":
    main()
