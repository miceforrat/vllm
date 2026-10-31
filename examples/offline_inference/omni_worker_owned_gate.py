# SPDX-License-Identifier: Apache-2.0
"""Validate that one vLLM GPU worker owns primary and auxiliary runners."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import torch

from vllm.experimental.multi_model_worker import (
    ModelDeploymentSpec,
    WorkerHostedRunnerClient,
)


def _run_reasoner(reasoner: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    tokenizer = reasoner.get_tokenizer()
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(formatted_prompt)
    started = time.perf_counter()
    outputs = reasoner.generate(
        [formatted_prompt],
        SamplingParams(temperature=0.0, seed=0, max_tokens=max_tokens),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    candidate = outputs[0].outputs[0]
    return {
        "seconds": time.perf_counter() - started,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": len(candidate.token_ids),
        "finish_reason": candidate.finish_reason,
        "text_sha256": hashlib.sha256(candidate.text.encode()).hexdigest(),
    }


def _make_generator_request(args: argparse.Namespace, request_id: str) -> Any:
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


def _describe_output(output: Any) -> dict[str, Any]:
    def field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    payload = field(output, "output")
    media = field(output, "media")
    if payload is None and media is not None:
        video = field(media, "video")
        payload = field(video, "tensor")
    tensor = payload
    if isinstance(payload, dict):
        tensor = next(
            (value for value in payload.values() if torch.is_tensor(value)), None
        )
    if isinstance(payload, (list, tuple)):
        tensor = next((value for value in payload if torch.is_tensor(value)), None)
    return {
        "type": type(payload).__name__,
        "tensor_shape": None if tensor is None else list(tensor.shape),
        "peak_memory_mb": field(output, "peak_memory_mb"),
        "stage_durations": field(output, "stage_durations"),
    }


def _describe_memory_profile(profile: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(profile):
        return dataclasses.asdict(profile)
    if isinstance(profile, dict):
        return profile
    raise TypeError(
        "Unsupported generator memory-profile transport type: "
        f"{type(profile).__name__}"
    )


def _run_concurrent_once(
    *,
    reasoner: Any,
    client: Any,
    model_id: str,
    args: argparse.Namespace,
    reasoner_prompt: str,
    index: int,
) -> dict[str, Any]:
    client.prepare(
        model_id,
        _make_generator_request(args, f"worker-concurrent-{index:02d}"),
        profile_memory=False,
    )

    # A Green-Context worker releases its local Generator future immediately
    # before entering the Reasoner's execute_model(). Other worker types lack
    # that hook, so release explicitly before submitting the Reasoner.
    if args.generator_stream != "green":
        client.release_prepared()

    wall_start = time.perf_counter()
    reasoner_start = time.perf_counter()
    with torch.cuda.nvtx.range(f"worker.concurrent.reasoner.{index}"):
        reasoner_result = _run_reasoner(
            reasoner, reasoner_prompt, args.reasoner_max_tokens
        )
    reasoner_end = time.perf_counter()
    reasoner_result = {
        "host_start": reasoner_start,
        "host_end": reasoner_end,
        **reasoner_result,
    }

    generator_result = client.collect(model_id)
    if generator_result is None:
        raise RuntimeError("Generator owner rank returned no async result")
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


def _load_full_sm_reference(
    path: Path,
    *,
    reasoner_mode: str,
    reasoner_prompt_sha256: str,
    reasoner_max_tokens: int,
    generator_workload_sha256: str | None,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report["full_sm_concurrency"]["summary"]
    if summary is None:
        raise ValueError(f"Full-SM reference has no concurrency summary: {path}")

    reference_workload = report["workload"]
    checks = {
        "reasoner_mode": (
            report["reasoner_execution"]["mode"] == reasoner_mode
        ),
        "reasoner_prompt_sha256": (
            reference_workload["reasoner_prompt_sha256"]
            == reasoner_prompt_sha256
        ),
        "reasoner_max_tokens": (
            reference_workload["reasoner_max_tokens"] == reasoner_max_tokens
        ),
        "generator_workload_sha256": (
            reference_workload["generator_workload_sha256"]
            == generator_workload_sha256
        ),
        "spatial_sharing_disabled": (
            not report["full_sm_concurrency"]["spatial_sharing_enabled"]
        ),
    }
    failed = [name for name, matched in checks.items() if not matched]
    if failed:
        raise ValueError(
            f"Full-SM reference does not match this run: {failed}; path={path}"
        )
    return {
        "path": str(path),
        "checks": checks,
        "summary": summary,
    }


def _verify_resource_contract(
    resource_plan: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    *,
    reasoner_mode: str,
    generator_stream: str,
) -> dict[str, Any]:
    if generator_stream != "green":
        return {"required": False, "verified": None, "checks": {}}

    plans_by_rank = {row.get("rank"): row for row in resource_plan}
    metadata_by_rank = {row.get("rank"): row for row in metadata}
    matching_ranks = (
        None not in plans_by_rank
        and None not in metadata_by_rank
        and set(plans_by_rank) == set(metadata_by_rank)
        and bool(plans_by_rank)
    )
    worker_checks: dict[str, dict[str, bool]] = {}
    if matching_ranks:
        for rank in sorted(plans_by_rank):
            plan = plans_by_rank[rank]
            stream = metadata_by_rank[rank]["stream_control"]
            graph = plan["graph_capture"]
            graph_required = reasoner_mode == "graph"
            worker_checks[str(rank)] = {
                "resource_plan_enabled": plan["enabled"],
                "device_matches": plan["device"] == metadata_by_rank[rank]["device"],
                "complementary_sm_partition": (
                    plan["actual_generator_sm_count"]
                    + plan["actual_reasoner_sm_count"]
                    == plan["total_sm_count"]
                ),
                "distinct_execution_streams": (
                    plan["generator_stream_handle"]
                    != plan["reasoner_stream_handle"]
                ),
                "generator_external_stream_verified": (
                    stream["owner"] == "external" and stream["verified"]
                ),
                "generator_stream_matches_partition": (
                    stream["expected_stream_handle"]
                    == plan["generator_stream_handle"]
                    == stream["last_actual_stream_handle"]
                ),
                "reasoner_graph_stream_matches_partition": (
                    not graph_required
                    or (
                        graph is not None
                        and graph["stream_handle"]
                        == plan["reasoner_stream_handle"]
                    )
                ),
                "reasoner_graph_capture_observed": (
                    not graph_required
                    or (
                        graph is not None
                        and graph["graph_capture_calls"] > 0
                        and graph["manager_capture_stream_calls"] > 0
                        and graph["compiled_capture_stream_calls"] > 0
                    )
                ),
            }
    workers_verified = matching_ranks and all(
        all(checks.values()) for checks in worker_checks.values()
    )
    return {
        "required": True,
        "verified": workers_verified,
        "checks": {
            "matching_worker_ranks": matching_ranks,
            "workers": worker_checks,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--model-id", default="generator-a")
    parser.add_argument("--reasoner-max-tokens", type=int, default=64)
    parser.add_argument(
        "--reasoner-mode",
        choices=("eager", "graph"),
        default="graph",
        help="Reasoner execution mode; graph matches the optimized vLLM baseline.",
    )
    parser.add_argument("--reasoner-input", type=Path)
    parser.add_argument("--generator-workload-manifest", type=Path)
    parser.add_argument("--generator-warmups", type=int, default=0)
    parser.add_argument("--generator-repeats", type=int, default=1)
    parser.add_argument("--alone-repeats", type=int, default=0)
    parser.add_argument("--concurrent-repeats", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--worker-shutdown-timeout-seconds",
        type=int,
        default=30,
        help="Grace period for each vLLM Worker to release model resources.",
    )
    parser.add_argument(
        "--engine-shutdown-timeout-seconds",
        type=int,
        default=45,
        help=(
            "Grace period for EngineCore shutdown; must exceed the Worker "
            "cleanup grace."
        ),
    )
    parser.add_argument("--prompt", default="A red robot repairs a bicycle.")
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
        default="dedicated",
    )
    parser.add_argument("--generator-sm-count", type=int, default=120)
    parser.add_argument(
        "--gc-workqueue-mode",
        choices=("shared", "balanced"),
        default="shared",
        help=(
            "Workqueue policy attached to both Green Context descriptors. "
            "Balanced requires CUDA 13.1+."
        ),
    )
    parser.add_argument(
        "--gc-workqueue-concurrency-limit",
        type=int,
        default=1,
        help="Expected concurrent stream-ordered workloads per Green Context.",
    )
    parser.add_argument(
        "--full-sm-reference",
        type=Path,
        help=(
            "Full-SM report with an identical workload; used to compute the "
            "end-to-end throughput gain of a partitioned colocated run."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/omni_worker_owned_gate/report.json"),
    )
    args = parser.parse_args()
    if (
        args.generator_warmups < 0
        or args.generator_repeats < 1
        or args.alone_repeats < 0
        or args.concurrent_repeats < 0
    ):
        parser.error(
            "warmups and concurrency repeats must be >= 0; "
            "--generator-repeats must be >= 1"
        )
    if args.worker_shutdown_timeout_seconds <= 0:
        parser.error("--worker-shutdown-timeout-seconds must be > 0")
    if (
        args.engine_shutdown_timeout_seconds
        <= args.worker_shutdown_timeout_seconds
    ):
        parser.error(
            "--engine-shutdown-timeout-seconds must be greater than "
            "--worker-shutdown-timeout-seconds"
        )
    if args.gc_workqueue_concurrency_limit <= 0:
        parser.error("--gc-workqueue-concurrency-limit must be positive")
    if args.generator_stream != "green" and args.gc_workqueue_mode != "shared":
        parser.error("balanced workqueues require --generator-stream green")

    # MultiprocExecutor otherwise gives workers only five seconds to release
    # the primary model, CUDA graphs, KV cache and the auxiliary model.  Keep
    # the outer EngineCore grace larger so it cannot kill the process manager
    # while Worker cleanup is still making progress.
    os.environ["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = str(
        args.worker_shutdown_timeout_seconds
    )

    reasoner_prompt = (
        "Describe a safe plan for a robot to move a cup onto a shelf."
    )
    reasoner_input = None
    if args.reasoner_input is not None:
        reasoner_input = json.loads(
            args.reasoner_input.read_text(encoding="utf-8")
        )
        reasoner_prompt = reasoner_input["prompt"]
        args.reasoner_max_tokens = int(
            reasoner_input.get("max_new_tokens", args.reasoner_max_tokens)
        )

    generator_manifest = None
    if args.generator_workload_manifest is not None:
        generator_manifest = json.loads(
            args.generator_workload_manifest.read_text(encoding="utf-8")
        )
        workload = generator_manifest["generator"]["canonical_workload"]
        args.prompt = workload["prompt"]
        args.negative_prompt = workload["negative_prompt"]
        args.height = int(workload["height"])
        args.width = int(workload["width"])
        args.num_frames = int(workload["num_frames"])
        args.steps = int(workload["num_inference_steps"])
        args.guidance_scale = float(workload["guidance_scale"])
        args.flow_shift = float(workload["flow_shift"])
        args.seed = int(workload["seed"])

    from vllm import LLM

    reasoner_execution: dict[str, Any]
    if args.reasoner_mode == "eager":
        reasoner_execution = {"enforce_eager": True}
    else:
        reasoner_execution = {
            "enforce_eager": False,
            "compilation_config": {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
            },
        }

    worker_options: dict[str, Any] = {}
    if args.generator_stream == "green":
        worker_options = {
            "worker_cls": (
                "vllm.experimental.multi_model_gpu_worker."
                "MultiModelGPUWorker"
            ),
            "additional_config": {
                "multi_model_runtime": {
                    "green_context": {
                        "generator_sm_count": args.generator_sm_count,
                        "workqueue": {
                            "mode": args.gc_workqueue_mode,
                            "concurrency_limit": (
                                args.gc_workqueue_concurrency_limit
                            ),
                        },
                    }
                }
            },
        }

    reasoner = LLM(
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
    client = WorkerHostedRunnerClient(reasoner.llm_engine)

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
        registered = client.register(spec)
        if args.generator_stream == "green":
            stream = client.resource_plan()
        else:
            stream = client.configure_stream(
                args.model_id,
                kind=args.generator_stream,
                sm_count=None,
            )
        reasoner_result = _run_reasoner(
            reasoner,
            reasoner_prompt,
            args.reasoner_max_tokens,
        )
        generator_warmup_seconds = []
        for index in range(args.generator_warmups):
            started = time.perf_counter()
            client.execute(
                args.model_id,
                _make_generator_request(args, f"worker-owned-warmup-{index:02d}"),
                profile_memory=False,
            )
            torch.cuda.synchronize()
            generator_warmup_seconds.append(time.perf_counter() - started)

        generator_measured_seconds = []
        output = None
        memory_profile = None
        for index in range(args.generator_repeats):
            started = time.perf_counter()
            output, memory_profile = client.execute(
                args.model_id,
                _make_generator_request(args, f"worker-owned-measured-{index:02d}"),
                profile_memory=True,
            )
            torch.cuda.synchronize()
            generator_measured_seconds.append(time.perf_counter() - started)
        assert output is not None and memory_profile is not None
        generator_seconds = statistics.median(generator_measured_seconds)

        alone_rows = []
        for index in range(args.alone_repeats):
            serial_start = time.perf_counter()
            reasoner_alone = _run_reasoner(
                reasoner, reasoner_prompt, args.reasoner_max_tokens
            )
            generator_alone = client.execute_timed(
                args.model_id,
                _make_generator_request(args, f"worker-alone-{index:02d}"),
                profile_memory=False,
            )
            alone_rows.append(
                {
                    "index": index,
                    "reasoner": reasoner_alone,
                    "generator": generator_alone,
                    "serial_host_wall_seconds": (
                        time.perf_counter() - serial_start
                    ),
                }
            )

        concurrent_rows = []
        for index in range(args.concurrent_repeats):
            concurrent_rows.append(
                _run_concurrent_once(
                    reasoner=reasoner,
                    client=client,
                    model_id=args.model_id,
                    args=args,
                    reasoner_prompt=reasoner_prompt,
                    index=index,
                )
            )

        concurrency_summary = None
        if alone_rows and concurrent_rows:
            reasoner_alone_seconds = statistics.median(
                row["reasoner"]["seconds"] for row in alone_rows
            )
            generator_alone_seconds = statistics.median(
                row["generator"]["seconds"] for row in alone_rows
            )
            serial_host_wall_seconds = statistics.median(
                row["serial_host_wall_seconds"] for row in alone_rows
            )
            reasoner_concurrent_seconds = statistics.median(
                row["reasoner"]["seconds"] for row in concurrent_rows
            )
            generator_concurrent_seconds = statistics.median(
                row["generator"]["seconds"] for row in concurrent_rows
            )
            concurrent_host_wall_seconds = statistics.median(
                row["host_wall_seconds"] for row in concurrent_rows
            )
            concurrency_summary = {
                "reasoner_alone_seconds": reasoner_alone_seconds,
                "generator_alone_seconds": generator_alone_seconds,
                "serial_host_wall_seconds": serial_host_wall_seconds,
                "reasoner_concurrent_seconds": reasoner_concurrent_seconds,
                "generator_concurrent_seconds": generator_concurrent_seconds,
                "concurrent_host_wall_seconds": concurrent_host_wall_seconds,
                "reasoner_slowdown": (
                    reasoner_concurrent_seconds / reasoner_alone_seconds
                ),
                "generator_slowdown": (
                    generator_concurrent_seconds / generator_alone_seconds
                ),
                "throughput_gain_partitioned_reference": (
                    serial_host_wall_seconds / concurrent_host_wall_seconds
                ),
                "host_overlap_seconds": statistics.median(
                    row["host_overlap_seconds"] for row in concurrent_rows
                ),
                "host_overlap_over_shorter": statistics.median(
                    row["host_overlap_over_shorter"]
                    for row in concurrent_rows
                ),
            }

        reasoner_prompt_sha256 = hashlib.sha256(
            reasoner_prompt.encode()
        ).hexdigest()
        generator_workload_sha256 = (
            None if generator_manifest is None
            else generator_manifest["generator"].get(
                "canonical_workload_sha256"
            )
        )
        full_sm_reference = None
        if args.full_sm_reference is not None:
            full_sm_reference = _load_full_sm_reference(
                args.full_sm_reference,
                reasoner_mode=args.reasoner_mode,
                reasoner_prompt_sha256=reasoner_prompt_sha256,
                reasoner_max_tokens=args.reasoner_max_tokens,
                generator_workload_sha256=generator_workload_sha256,
            )
            if concurrency_summary is None:
                raise ValueError(
                    "--full-sm-reference requires alone and concurrent repeats"
                )
            full_summary = full_sm_reference["summary"]
            concurrency_summary.update(
                {
                    "reasoner_full_sm_alone_seconds": full_summary[
                        "reasoner_alone_seconds"
                    ],
                    "generator_full_sm_alone_seconds": full_summary[
                        "generator_alone_seconds"
                    ],
                    "full_sm_serial_host_wall_seconds": full_summary[
                        "serial_host_wall_seconds"
                    ],
                    "reasoner_sm_restriction_slowdown": (
                        concurrency_summary["reasoner_alone_seconds"]
                        / full_summary["reasoner_alone_seconds"]
                    ),
                    "generator_sm_restriction_slowdown": (
                        concurrency_summary["generator_alone_seconds"]
                        / full_summary["generator_alone_seconds"]
                    ),
                    "throughput_gain_full_sm_reference": (
                        full_summary["serial_host_wall_seconds"]
                        / concurrency_summary["concurrent_host_wall_seconds"]
                    ),
                }
            )

        metadata = client.metadata(args.model_id)
        resource_contract = _verify_resource_contract(
            stream,
            metadata,
            reasoner_mode=args.reasoner_mode,
            generator_stream=args.generator_stream,
        )
        # The output-owner worker sends the media tensor through the
        # multiprocessing RPC transport.  Keep only report metadata before
        # shutting workers down; retaining the transported tensor pins its
        # shared-memory export and can prevent that worker from exiting.
        output_description = _describe_output(output)
        del output
        gc.collect()
        report = {
            "gate": "worker_owned_primary_and_auxiliary_runners",
            "reasoner_execution": {
                "mode": args.reasoner_mode,
                **reasoner_execution,
            },
            "model_spec": dataclasses.asdict(spec),
            "registered_workers": registered,
            "runner_metadata": metadata,
            "stream_control": stream,
            "resource_contract": resource_contract,
            "workload": {
                "reasoner_input": (
                    None if args.reasoner_input is None
                    else str(args.reasoner_input)
                ),
                "reasoner_prompt_sha256": reasoner_prompt_sha256,
                "reasoner_max_tokens": args.reasoner_max_tokens,
                "reasoner_fixed_length": False,
                "generator_manifest": (
                    None if args.generator_workload_manifest is None
                    else str(args.generator_workload_manifest)
                ),
                "generator_workload_sha256": generator_workload_sha256,
                "generator": {
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "num_inference_steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "flow_shift": args.flow_shift,
                    "seed": args.seed,
                    "negative_prompt_sha256": hashlib.sha256(
                        args.negative_prompt.encode()
                    ).hexdigest(),
                },
            },
            "reasoner": reasoner_result,
            "generator": {
                "seconds": generator_seconds,
                "warmup_seconds": generator_warmup_seconds,
                "measured_seconds": generator_measured_seconds,
                "output": output_description,
                "memory_profile": _describe_memory_profile(memory_profile),
            },
            "full_sm_concurrency": {
                "execution_mode": args.generator_stream,
                "spatial_sharing_enabled": (
                    args.generator_stream == "green"
                ),
                "generator_result_transport": "timing_only",
                "full_sm_reference": full_sm_reference,
                "alone": alone_rows,
                "concurrent": concurrent_rows,
                "summary": concurrency_summary,
            },
            "ownership_verified": (
                {row["rank"] for row in metadata}
                == set(range(args.tp_size))
                and all(
                    row["active"]
                    and "GPUModelRunner" in row["primary_runner_type"]
                    and row["parallel_topology"]["tp_size"] == args.tp_size
                    and row["parallel_topology"]["tp_rank"] == row["rank"]
                    for row in metadata
                )
                and {
                    row["rank"]: (row["pid"], row["worker_id"])
                    for row in registered
                }
                == {
                    row["rank"]: (row["pid"], row["worker_id"])
                    for row in metadata
                }
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        client.close()
        reasoner.llm_engine.engine_core.shutdown(
            timeout=float(args.engine_shutdown_timeout_seconds)
        )


if __name__ == "__main__":
    main()
