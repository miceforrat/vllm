# SPDX-License-Identifier: Apache-2.0
"""Submit Worker-hosted Generator and AsyncLLM Reasoner independently.

This is the serving-style counterpart of ``omni_worker_owned_gate.py``.  The
older gate intentionally retains blocking ``LLM.generate`` for controlled
performance experiments; this gate validates that external callers can start
the Generator and later enqueue an LLM request through AsyncLLM.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.experimental.multi_model_worker import (
    AsyncWorkerHostedRunnerClient,
    ModelDeploymentSpec,
)
from vllm.v1.engine.async_llm import AsyncLLM



def make_generator_request(args: argparse.Namespace, request_id: str) -> Any:
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
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


async def run_reasoner(
    engine: AsyncLLM,
    prompt: str,
    max_tokens: int,
    request_id: str,
    fixed_tokens: bool = False,
) -> dict[str, Any]:
    tokenizer = engine.renderer.tokenizer
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(formatted_prompt)
    started = time.perf_counter()
    final_output = None
    async for output in engine.generate(
        prompt=formatted_prompt,
        sampling_params=SamplingParams(
            temperature=0.0,
            seed=0,
            max_tokens=max_tokens,
            min_tokens=max_tokens if fixed_tokens else 0,
            ignore_eos=fixed_tokens,
        ),
        request_id=request_id,
    ):
        final_output = output
    if final_output is None:
        raise RuntimeError("AsyncLLM request produced no output")
    candidate = final_output.outputs[0]
    ended = time.perf_counter()
    return {
        "host_start": started,
        "host_end": ended,
        "seconds": ended - started,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": len(candidate.token_ids),
        "finish_reason": candidate.finish_reason,
        "text_sha256": hashlib.sha256(candidate.text.encode()).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--model-id", default="generator-a")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--reasoner-max-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoner-fixed-tokens",
        action="store_true",
        help="Generate exactly --reasoner-max-tokens for controlled overlap.",
    )
    parser.add_argument(
        "--reasoner-mode", choices=("eager", "graph"), default="graph"
    )
    parser.add_argument("--reasoner-input", type=Path)
    parser.add_argument(
        "--reasoner-prompt",
        default="Describe a safe plan for a robot to move a cup onto a shelf.",
    )
    parser.add_argument("--reasoner-arrival-delay-seconds", type=float, default=0)
    parser.add_argument(
        "--submission-order",
        choices=("generator_first", "reasoner_first"),
        default="generator_first",
    )
    parser.add_argument("--generator-arrival-delay-seconds", type=float, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--worker-shutdown-timeout-seconds", type=int, default=30)
    parser.add_argument("--engine-shutdown-timeout-seconds", type=int, default=45)
    parser.add_argument("--prompt", default="A red robot repairs a bicycle.")
    parser.add_argument("--generator-workload-manifest", type=Path)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--generator-cache", choices=("none", "sea_cache"), default="none"
    )
    parser.add_argument(
        "--generator-stream",
        choices=("default", "dedicated", "green"),
        default="green",
    )
    parser.add_argument("--generator-sm-count", type=int, default=136)
    parser.add_argument(
        "--dynamic-step-scheduling",
        action="store_true",
        help="Select full or partitioned Generator streams per diffusion step.",
    )
    parser.add_argument(
        "--dynamic-reasoner-stream",
        action="store_true",
        help="Enable eager Reasoner full/shared stream selection per epoch.",
    )
    parser.add_argument(
        "--trace-partition-nvtx",
        action="store_true",
        help=(
            "Annotate every Reasoner scheduler batch and request/phase GPU "
            "boundary for Nsight Systems diagnosis."
        ),
    )
    parser.add_argument(
        "--nsys-capture-measured",
        action="store_true",
        help=(
            "Bracket only the measured colocated request with "
            "cudaProfilerStart/Stop for Nsys --capture-range=cudaProfilerApi."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/omni_worker_async_gate/report.json"),
    )
    args = parser.parse_args()
    if args.reasoner_arrival_delay_seconds < 0:
        parser.error("--reasoner-arrival-delay-seconds must be >= 0")
    if args.generator_arrival_delay_seconds < 0:
        parser.error("--generator-arrival-delay-seconds must be >= 0")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.worker_shutdown_timeout_seconds <= 0:
        parser.error("--worker-shutdown-timeout-seconds must be > 0")
    if args.dynamic_reasoner_stream and not args.dynamic_step_scheduling:
        parser.error(
            "--dynamic-reasoner-stream requires --dynamic-step-scheduling"
        )
    if (
        args.engine_shutdown_timeout_seconds
        <= args.worker_shutdown_timeout_seconds
    ):
        parser.error(
            "--engine-shutdown-timeout-seconds must exceed the Worker timeout"
        )
    return args


def apply_workload_files(args: argparse.Namespace) -> None:
    if args.reasoner_input is not None:
        data = json.loads(args.reasoner_input.read_text(encoding="utf-8"))
        args.reasoner_prompt = data["prompt"]
        args.reasoner_max_tokens = int(
            data.get("max_new_tokens", args.reasoner_max_tokens)
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


async def async_main(args: argparse.Namespace) -> None:
    apply_workload_files(args)
    os.environ["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = str(
        args.worker_shutdown_timeout_seconds
    )
    reasoner_execution = (
        {"enforce_eager": True}
        if args.reasoner_mode == "eager"
        else {
            "enforce_eager": False,
            "compilation_config": {
                "cudagraph_mode": "FULL_AND_PIECEWISE"
            },
        }
    )
    worker_options: dict[str, Any] = {}
    if args.generator_stream == "green":
        worker_options = {
            "worker_cls": (
                "vllm.experimental.multi_model_gpu_worker.MultiModelGPUWorker"
            ),
            "additional_config": {
                "multi_model_runtime": {
                    "green_context": {
                        "generator_sm_count": args.generator_sm_count,
                        "dynamic_step_scheduling": args.dynamic_step_scheduling,
                        "dynamic_reasoner_stream": args.dynamic_reasoner_stream,
                        "trace_partition_nvtx": args.trace_partition_nvtx,
                        "dual_reasoner_graph_variants": (
                            args.dynamic_reasoner_stream
                            and args.reasoner_mode == "graph"
                        ),
                    }
                }
            },
        }
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_num_seqs=1,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        enable_prefix_caching=False,
        disable_log_stats=True,
        worker_extension_cls=(
            "vllm.experimental.multi_model_worker.MultiModelWorkerExtension"
        ),
        **reasoner_execution,
        **worker_options,
    )
    reasoner_engine_init_started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args)
    reasoner_engine_init_seconds = (
        time.perf_counter() - reasoner_engine_init_started
    )
    client = AsyncWorkerHostedRunnerClient(engine)
    cache_options: dict[str, Any] = {"cache_backend": "none"}
    if args.generator_cache == "sea_cache":
        cache_options = {
            "cache_backend": "sea_cache",
            "cache_config": {
                "sea_threshold": 0.25,
                "sea_residual_order": 1,
                "sea_max_consecutive_cached": 2,
                "sea_power_exp": 3.0,
            },
        }
    spec = ModelDeploymentSpec(
        model_id=args.model_id,
        model_path=args.model,
        backend="vllm_omni",
        task="text_to_video" if args.num_frames > 1 else "text_to_image",
        backend_options=cache_options,
    )

    try:
        generator_registration_started = time.perf_counter()
        registered = await client.register(spec)
        generator_registration_seconds = (
            time.perf_counter() - generator_registration_started
        )
        if args.generator_stream == "green":
            stream = await client.resource_plan()
        else:
            stream = await client.configure_stream(
                args.model_id, kind=args.generator_stream
            )

        warmups = []
        for index in range(args.warmups):
            warmup_reasoner = await run_reasoner(
                engine,
                args.reasoner_prompt,
                args.reasoner_max_tokens,
                f"async-warmup-reasoner-{index:02d}",
                args.reasoner_fixed_tokens,
            )
            # Keep every Generator forward on the same long-lived auxiliary
            # worker thread. CUDA current-context and current-stream state is
            # thread-local; warming up through a direct EngineCore RPC and
            # serving through the thread pool makes GC/full stream ownership
            # ambiguous and also causes CUPTI to lose the serving thread.
            warmup_future = await client.start_async(
                args.model_id,
                make_generator_request(
                    args, f"async-warmup-generator-{index:02d}"
                ),
            )
            warmup_generator = await warmup_future.result()
            warmups.append(
                {
                    "reasoner_seconds": warmup_reasoner["seconds"],
                    "generator_seconds": warmup_generator["seconds"],
                    "generator_worker_thread_ident": warmup_generator[
                        "worker_thread_ident"
                    ],
                    "generator_worker_thread_name": warmup_generator[
                        "worker_thread_name"
                    ],
                }
            )

        if args.trace_partition_nvtx:
            torch.cuda.nvtx.range_push("worker.concurrent.request.0")
        if args.nsys_capture_measured:
            torch.cuda.cudart().cudaProfilerStart()
        wall_start = time.perf_counter()
        if args.submission_order == "generator_first":
            generator_future = await client.start_async(
                args.model_id,
                make_generator_request(args, "async-generator-00"),
            )
            generator_submitted = time.perf_counter()
            if args.reasoner_arrival_delay_seconds:
                await asyncio.sleep(args.reasoner_arrival_delay_seconds)
            demand_requested = time.perf_counter()
            demand_state = (
                await client.set_primary_demand(True)
                if args.dynamic_step_scheduling
                else None
            )
            demand_admitted = time.perf_counter()
            reasoner_task = asyncio.create_task(run_reasoner(
                engine,
                args.reasoner_prompt,
                args.reasoner_max_tokens,
                "async-reasoner-00",
                args.reasoner_fixed_tokens,
            ))
        else:
            demand_requested = time.perf_counter()
            demand_state = (
                await client.set_primary_demand(True)
                if args.dynamic_step_scheduling
                else None
            )
            demand_admitted = time.perf_counter()
            reasoner_task = asyncio.create_task(run_reasoner(
                engine,
                args.reasoner_prompt,
                args.reasoner_max_tokens,
                "async-reasoner-00",
                args.reasoner_fixed_tokens,
            ))
            if args.generator_arrival_delay_seconds:
                await asyncio.sleep(args.generator_arrival_delay_seconds)
            generator_future = await client.start_async(
                args.model_id,
                make_generator_request(args, "async-generator-00"),
            )
            generator_submitted = time.perf_counter()
        try:
            reasoner = await reasoner_task
            if args.trace_partition_nvtx:
                torch.cuda.nvtx.mark("reasoner/client_request_returned")
        finally:
            demand_released = (
                await client.set_primary_demand(False)
                if args.dynamic_step_scheduling
                else None
            )
        generator = await generator_future.result()
        if args.trace_partition_nvtx:
            torch.cuda.nvtx.mark("generator/client_request_collected")
        wall_end = time.perf_counter()
        if args.trace_partition_nvtx:
            torch.cuda.nvtx.range_pop()
        if args.nsys_capture_measured:
            torch.cuda.cudart().cudaProfilerStop()
        final_stream = await client.resource_plan()
        runner_metadata = await client.metadata(args.model_id)

        generator_thread_ids = {
            row["generator_worker_thread_ident"] for row in warmups
        }
        generator_thread_ids.add(generator["worker_thread_ident"])
        if len(generator_thread_ids) != 1:
            raise RuntimeError(
                "Generator execution escaped its owner thread: "
                f"thread_ids={sorted(generator_thread_ids)}"
            )

        overlap = max(
            0.0,
            min(reasoner["host_end"], generator["host_end"])
            - max(reasoner["host_start"], generator["host_start"]),
        )
        report = {
            "gate": "async_llm_with_worker_hosted_generator",
            "startup": {
                "reasoner_engine_init_seconds": reasoner_engine_init_seconds,
                "generator_registration_seconds": (
                    generator_registration_seconds
                ),
                "ready_seconds": (
                    reasoner_engine_init_seconds
                    + generator_registration_seconds
                ),
            },
            "registered_workers": registered,
            "stream_control_initial": stream,
            "stream_control": final_stream,
            "runner_metadata": runner_metadata,
            "warmups": warmups,
            "submission": {
                "order": args.submission_order,
                "generator_started_before_reasoner": (
                    args.submission_order == "generator_first"
                ),
                "reasoner_arrival_delay_seconds": (
                    args.reasoner_arrival_delay_seconds
                ),
                "generator_arrival_delay_seconds": (
                    args.generator_arrival_delay_seconds
                ),
                "generator_start_rpc_seconds": generator_submitted - wall_start,
                "generator_worker_thread_stable": True,
                "generator_worker_thread_ident": generator[
                    "worker_thread_ident"
                ],
                "primary_demand_admission_seconds": (
                    demand_admitted - demand_requested
                ),
                "demand_state_after_admission": demand_state,
                "demand_state_after_release": demand_released,
            },
            "reasoner": reasoner,
            "generator": generator,
            "host_overlap_seconds": overlap,
            "host_wall_seconds": wall_end - wall_start,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        await client.close()
        engine.shutdown(timeout=float(args.engine_shutdown_timeout_seconds))


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
