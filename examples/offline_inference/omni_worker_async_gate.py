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
from vllm.usage.usage_lib import UsageContext
from vllm.experimental.archived.multi_model_worker import (
    AsyncWorkerHostedRunnerClient,
    ModelSpec,
    SchedulerModelRequest,
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
    client: AsyncWorkerHostedRunnerClient,
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
    future = await client.submit(
        client.default_model_id,
        SchedulerModelRequest(
            prompt=formatted_prompt,
            sampling_params=SamplingParams(
                temperature=0.0,
                seed=0,
                max_tokens=max_tokens,
                min_tokens=max_tokens if fixed_tokens else 0,
                ignore_eos=fixed_tokens,
            ),
            request_id=request_id,
        ),
    )
    final_output = await future.result()
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
    parser.add_argument("--repeats", type=int, default=1)
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
        "--partition-policy",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "Keep both models on fixed complementary partitions, or "
            "dynamically switch both models between shared and full-SM lanes."
        ),
    )
    parser.add_argument(
        "--reasoner-graph-pool-mode",
        choices=("isolated", "shared"),
        default="shared",
        help=(
            "Keep the shared/full Reasoner CUDA Graph registries in separate "
            "memory pools, or reuse vLLM's global graph pool."
        ),
    )
    parser.add_argument(
        "--reasoner-graph-buffer-mode",
        choices=("isolated", "shared"),
        default="isolated",
        help="Share persistent output buffers across Reasoner graph variants.",
    )
    parser.add_argument(
        "--reasoner-graph-capture-mode",
        choices=("full", "piecewise", "breakable_piecewise"),
        default="full",
        help=(
            "Use one graph implementation for every Reasoner resource "
            "variant: monolithic FULL, torch.compile PIECEWISE, or "
            "Breakable PIECEWISE."
        ),
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
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.worker_shutdown_timeout_seconds <= 0:
        parser.error("--worker-shutdown-timeout-seconds must be > 0")
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
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = (
        "1"
        if args.reasoner_graph_capture_mode == "breakable_piecewise"
        else "0"
    )
    # 加载workload和设置环境变量
    apply_workload_files(args)
    os.environ["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = str(
        args.worker_shutdown_timeout_seconds
    )

    # 这部分组装AsyncEngineArgs
    # 这里涉及到graph capture的设置，full表示全量捕获，piecewise表示分片，breakable可以配合piecewise但不能配合full
    reasoner_execution = (
        {"enforce_eager": True}
        if args.reasoner_mode == "eager"
        else {
            "enforce_eager": False,
            "compilation_config": {
                "cudagraph_mode": (
                    "FULL"
                    if args.reasoner_graph_capture_mode == "full"
                    else "PIECEWISE"
                )
            },
        }
    )
    # 这里引入了关于多流之间graph管理的参数
    # reasoner_graph_buffer_mode这个参数表示的是输出buffer，但这个东西的开销远没有cublas workspace高
    # graph pool表示共用一个池子（当然，这样就对提交顺序有要求了）
    worker_options: dict[str, Any] = {}
    if args.generator_stream == "green":
        worker_options = {
            "worker_cls": (
                "vllm.experimental.archived.multi_model_gpu_worker.MultiModelGPUWorker"
            ),
            "additional_config": {
                "multi_model_runtime": {
                    "green_context": {
                        "generator_sm_count": args.generator_sm_count,
                        "partition_policy": args.partition_policy,
                        "trace_partition_nvtx": args.trace_partition_nvtx,
                        "dual_reasoner_graph_variants": (
                            args.partition_policy == "dynamic"
                            and args.reasoner_mode == "graph"
                        ),
                        "reasoner_graph_pool_mode": (
                            args.reasoner_graph_pool_mode
                        ),
                        "reasoner_graph_buffer_mode": (
                            args.reasoner_graph_buffer_mode
                        ),
                        "reasoner_graph_capture_mode": (
                            args.reasoner_graph_capture_mode
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
            "vllm.experimental.archived.multi_model_worker.MultiModelWorkerExtension"
        ),
        **reasoner_execution,
        **worker_options,
    )

    # 创建vLLM engine（使用AsyncLLM）
    reasoner_engine_init_started = time.perf_counter()
    config_started = time.perf_counter()
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    reasoner_config_creation_seconds = time.perf_counter() - config_started
    runtime_started = time.perf_counter()
    engine = AsyncLLM.from_vllm_config(
        vllm_config,
        usage_context=UsageContext.ENGINE_CONTEXT,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )
    reasoner_runtime_startup_seconds = time.perf_counter() - runtime_started
    reasoner_engine_init_seconds = time.perf_counter() - reasoner_engine_init_started
    # 创建辅助的runner管理器件
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

    # 目前还是存在明显的主从地位差别——omni是属于patch进来的，所以这里额外用了一个spec
    # 以及这里怎么把任务限制死了，未来是需要修改的
    spec = ModelSpec(
        model_id=args.model_id,
        model_path=args.model,
        backend="vllm_omni",
        capabilities=("text_to_image", "text_to_video"),
        default_task=(
            "text_to_video" if args.num_frames > 1 else "text_to_image"
        ),
        backend_options=cache_options,
    )

    try:
        generator_registration_started = time.perf_counter()
        # 注册generator
        # 未来考虑改成统一注册以及可以追加注册（不过现在就可以）
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
            # warmup
            warmup_reasoner = await run_reasoner(
                client,
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
            warmup_future = await client.submit(
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

        # 观测相关函数，和功能不相干
        def graph_pool_counters(
            plans: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            return [
                {"rank": plan["rank"], **plan.get("reasoner_graph_pool", {})}
                for plan in plans
            ]

        async def run_measured(index: int) -> dict[str, Any]:
            suffix = f"{index:02d}"
            plans_before = await client.resource_plan()
            counters_before = graph_pool_counters(plans_before)
            if args.trace_partition_nvtx:
                torch.cuda.nvtx.range_push(
                    f"worker.concurrent.request.{index}"
                )
            wall_start = time.perf_counter()
            if args.submission_order == "generator_first":
                generator_future = await client.submit(
                    args.model_id,
                    make_generator_request(args, f"async-generator-{suffix}"),
                )
                generator_submitted = time.perf_counter()
                if args.reasoner_arrival_delay_seconds:
                    await asyncio.sleep(args.reasoner_arrival_delay_seconds)
                reasoner_task = asyncio.create_task(run_reasoner(
                    client,
                    engine,
                    args.reasoner_prompt,
                    args.reasoner_max_tokens,
                    f"async-reasoner-{suffix}",
                    args.reasoner_fixed_tokens,
                ))
            else:
                reasoner_task = asyncio.create_task(run_reasoner(
                    client,
                    engine,
                    args.reasoner_prompt,
                    args.reasoner_max_tokens,
                    f"async-reasoner-{suffix}",
                    args.reasoner_fixed_tokens,
                ))
                if args.generator_arrival_delay_seconds:
                    await asyncio.sleep(args.generator_arrival_delay_seconds)
                generator_future = await client.submit(
                    args.model_id,
                    make_generator_request(args, f"async-generator-{suffix}"),
                )
                generator_submitted = time.perf_counter()

            reasoner = await reasoner_task
            if args.trace_partition_nvtx:
                torch.cuda.nvtx.mark("reasoner/client_request_returned")

            generator = await generator_future.result()
            if args.trace_partition_nvtx:
                torch.cuda.nvtx.mark("generator/client_request_collected")
            wall_end = time.perf_counter()
            if args.trace_partition_nvtx:
                torch.cuda.nvtx.range_pop()
            overlap = max(
                0.0,
                min(reasoner["host_end"], generator["host_end"])
                - max(reasoner["host_start"], generator["host_start"]),
            )
            plans_after = await client.resource_plan()
            counters_after = graph_pool_counters(plans_after)
            counter_deltas = []
            for before, after in zip(counters_before, counters_after):
                counter_deltas.append({
                    "rank": after["rank"],
                    "registry_switches": (
                        after.get("registry_switches", 0)
                        - before.get("registry_switches", 0)
                    ),
                    "stream_drains": (
                        after.get("stream_drains", 0)
                        - before.get("stream_drains", 0)
                    ),
                    "stream_drain_seconds": (
                        after.get("stream_drain_seconds", 0.0)
                        - before.get("stream_drain_seconds", 0.0)
                    ),
                })
            return {
                "index": index,
                "submission": {
                    "generator_start_rpc_seconds": (
                        generator_submitted - wall_start
                    ),
                    "primary_demand_admission_seconds": None,
                    "demand_managed_by_submit": (
                        args.partition_policy == "dynamic"
                    ),
                },
                "reasoner": reasoner,
                "generator": generator,
                "host_overlap_seconds": overlap,
                "host_wall_seconds": wall_end - wall_start,
                "graph_pool_counter_deltas": counter_deltas,
                "cuda_allocator_before": [
                    {
                        "rank": plan["rank"],
                        **plan.get("cuda_allocator", {}),
                    }
                    for plan in plans_before
                ],
                "cuda_allocator_after": [
                    {
                        "rank": plan["rank"],
                        **plan.get("cuda_allocator", {}),
                    }
                    for plan in plans_after
                ],
            }

        if args.nsys_capture_measured:
            torch.cuda.cudart().cudaProfilerStart()
        measurements = []
        for index in range(args.repeats):
            measurement = await run_measured(index)
            measurements.append(measurement)
            print(
                f"measured[{index}] "
                f"reasoner={measurement['reasoner']['seconds']:.3f}s "
                f"generator={measurement['generator']['seconds']:.3f}s "
                f"wall={measurement['host_wall_seconds']:.3f}s",
                flush=True,
            )
        if args.nsys_capture_measured:
            torch.cuda.cudart().cudaProfilerStop()
        final_stream = await client.resource_plan()
        runner_metadata = await client.metadata(args.model_id)

        raw_startup_timestamps: dict[str, float] = {
            "engine_init_start": reasoner_engine_init_started,
            "engine_init_end": (
                reasoner_engine_init_started + reasoner_engine_init_seconds
            ),
            "config_creation_start": config_started,
            "config_creation_end": runtime_started,
            "runtime_startup_start": runtime_started,
            "runtime_startup_end": (
                runtime_started + reasoner_runtime_startup_seconds
            ),
        }
        raw_startup_timestamps.update(
            getattr(engine, "startup_timeline", {})
        )
        raw_startup_timestamps.update(
            getattr(engine.renderer, "startup_timeline", {})
        )

        def normalized_timeline(
            timestamps: dict[str, float], origin: float
        ) -> dict[str, Any]:
            events: dict[str, dict[str, float]] = {}
            points: dict[str, float] = {}
            consumed: set[str] = set()
            for key, start in timestamps.items():
                if not key.endswith("_start"):
                    continue
                base = key[:-6]
                end_key = f"{base}_end"
                end = timestamps.get(end_key)
                if end is None:
                    continue
                events[base] = {
                    "start_offset_seconds": start - origin,
                    "end_offset_seconds": end - origin,
                    "duration_seconds": end - start,
                }
                consumed.update((key, end_key))
            for key, value in timestamps.items():
                if key not in consumed:
                    points[key] = value - origin
            return {"events": events, "points": points}

        startup_timeline = normalized_timeline(
            raw_startup_timestamps, reasoner_engine_init_started
        )
        worker_startup_timelines = []
        for plan in final_stream:
            worker_startup_timelines.append({
                "rank": plan["rank"],
                **normalized_timeline(
                    plan.get("reasoner_worker_startup_timeline", {}),
                    reasoner_engine_init_started,
                ),
            })

        timeline_events = startup_timeline["events"]
        timeline_points = startup_timeline["points"]
        worker_timeline = worker_startup_timelines[0]["events"]
        worker_startup = final_stream[0].get("reasoner_worker_startup", {})
        graph_startup = final_stream[0].get("reasoner_graph_startup", {})

        def event_duration(name: str) -> float:
            return float(timeline_events.get(name, {}).get("duration_seconds", 0.0))

        def worker_event(name: str) -> dict[str, float]:
            return worker_timeline.get(name, {})

        worker_start = float(
            worker_event("device_and_distributed_init").get(
                "start_offset_seconds", 0.0
            )
        )
        worker_end = float(
            worker_event("full_variant_capture").get(
                "end_offset_seconds",
                worker_event("primary_compile_warmup_capture").get(
                    "end_offset_seconds", worker_start
                ),
            )
        )
        startup_summary = {
            "engine_startup_total_seconds": reasoner_engine_init_seconds,
            "config_creation_seconds": reasoner_config_creation_seconds,
            "frontend_initialization_seconds": sum(
                event_duration(name)
                for name in (
                    "renderer_init",
                    "input_processor_init",
                    "output_processor_init",
                )
            ),
            "engine_core_spawn_seconds": max(
                0.0,
                float(timeline_points.get("first_engine_hello", 0.0))
                - float(timeline_points.get("core_process_start_end", 0.0)),
            ),
            "engine_core_pre_worker_seconds": max(
                0.0,
                worker_start
                - float(timeline_points.get("first_engine_hello", worker_start)),
            ),
            "worker_startup_seconds": max(0.0, worker_end - worker_start),
            "engine_ready_finalize_seconds": max(
                0.0, reasoner_engine_init_seconds - worker_end
            ),
            "frontend_mm_warmup_seconds": event_duration(
                "frontend_mm_warmup"
            ),
            "reasoner_device_runtime_init_seconds": float(
                worker_startup.get("device_and_distributed_init_seconds", 0.0)
            ),
            "colocation_resource_stream_setup_seconds": float(
                worker_startup.get("resource_stream_setup_seconds", 0.0)
            ),
            "reasoner_model_load_seconds": float(
                worker_startup.get("model_load_seconds", 0.0)
            ),
            "profile_and_memory_preparation_seconds": float(
                worker_startup.get("memory_determination_seconds", 0.0)
            ),
            "kv_cache_initialization_seconds": float(
                worker_startup.get("kv_cache_initialization_seconds", 0.0)
            ),
            "primary_compile_graph_setup_seconds": float(
                worker_startup.get(
                    "primary_compile_warmup_capture_seconds", 0.0
                )
            ),
            "additional_full_graph_capture_seconds": float(
                graph_startup.get("full_variant_capture_seconds", 0.0)
            ),
            "shared_graph_capture_seconds": float(
                graph_startup.get("shared_graph_capture_seconds", 0.0)
            ),
            "full_graph_capture_seconds": float(
                graph_startup.get("full_variant_capture_seconds", 0.0)
            ),
            "dual_graph_capture_total_seconds": float(
                graph_startup.get("total_graph_capture_seconds", 0.0)
            ),
            "memory_profiling_performed": bool(
                worker_startup.get("memory_profiling_performed", False)
            ),
        }

        runtime_breakdown = []
        worker_accounted_critical_path = 0.0
        for plan in final_stream:
            worker_startup = plan.get("reasoner_worker_startup", {})
            graph_startup = plan.get("reasoner_graph_startup", {})
            # primary_compile_warmup_capture_seconds already includes the
            # shared graph capture.  Only the additional full variant belongs
            # beside it in this additive accounting.
            worker_accounted = sum(
                float(worker_startup.get(name, 0.0))
                for name in (
                    "device_and_distributed_init_seconds",
                    "resource_stream_setup_seconds",
                    "model_load_seconds",
                    "memory_determination_seconds",
                    "kv_cache_initialization_seconds",
                    "primary_compile_warmup_capture_seconds",
                )
            ) + float(graph_startup.get("full_variant_capture_seconds", 0.0))
            worker_accounted_critical_path = max(
                worker_accounted_critical_path, worker_accounted
            )
            runtime_breakdown.append({
                "rank": plan["rank"],
                "worker_accounted_seconds": worker_accounted,
                "worker_startup": worker_startup,
                "graph_startup": graph_startup,
            })

        runtime_unaccounted_seconds = max(
            0.0,
            reasoner_runtime_startup_seconds - worker_accounted_critical_path,
        )

        generator_thread_ids = {
            row["generator_worker_thread_ident"] for row in warmups
        }
        generator_thread_ids.update(
            row["generator"]["worker_thread_ident"] for row in measurements
        )
        if len(generator_thread_ids) != 1:
            raise RuntimeError(
                "Generator execution escaped its owner thread: "
                f"thread_ids={sorted(generator_thread_ids)}"
            )

        last = measurements[-1]
        reasoner = last["reasoner"]
        generator = last["generator"]
        report = {
            "gate": "async_llm_with_worker_hosted_generator",
            "startup": {
                "reasoner_engine_init_seconds": reasoner_engine_init_seconds,
                "reasoner_config_creation_seconds": (
                    reasoner_config_creation_seconds
                ),
                "reasoner_runtime_startup_seconds": (
                    reasoner_runtime_startup_seconds
                ),
                "reasoner_worker_accounted_critical_path_seconds": (
                    worker_accounted_critical_path
                ),
                "reasoner_runtime_unaccounted_seconds": (
                    runtime_unaccounted_seconds
                ),
                "reasoner_runtime_breakdown": runtime_breakdown,
                "reasoner_startup_timeline": startup_timeline,
                "reasoner_worker_startup_timelines": (
                    worker_startup_timelines
                ),
                "summary": startup_summary,
                "graph_capture_mode": args.reasoner_graph_capture_mode,
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
                "generator_worker_thread_stable": True,
                "generator_worker_thread_ident": generator[
                    "worker_thread_ident"
                ],
            },
            "measurements": measurements,
            "reasoner": reasoner,
            "generator": generator,
            "host_overlap_seconds": last["host_overlap_seconds"],
            "host_wall_seconds": last["host_wall_seconds"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        # 关闭资源
        await client.close()
        await engine.shutdown_async(
            timeout=float(args.engine_shutdown_timeout_seconds)
        )


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
