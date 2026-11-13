# SPDX-License-Identifier: Apache-2.0
"""真实验证 MultiModelWorker 的 Container 注册与执行链。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment
from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.multi_model_engine import MultiModelEngine
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="generator")
    parser.add_argument("--reasoner-max-tokens", type=int, default=32)
    parser.add_argument(
        "--reasoner-mode", choices=("eager", "graph"), default="graph"
    )
    parser.add_argument("--reasoner-input", type=Path)
    parser.add_argument("--generator-workload-manifest", type=Path)
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
        "--generator-cache",
        choices=("none", "sea_cache"),
        default="sea_cache",
    )
    parser.add_argument("--skip-step-smoke", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrent-repeats", type=int, default=3)
    parser.add_argument(
        "--reasoner-arrival-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "并发阶段先提交 Generator，再延迟指定秒数提交 Reasoner；"
            "用于覆盖不同 diffusion/VAE 阶段的 admission。"
        ),
    )
    parser.add_argument(
        "--concurrent-generator-variant",
        choices=("auto", "shared"),
        default="auto",
        help=(
            "并发阶段 Generator 使用的资源变体；shared 用于与 auto 做同版 "
            "A/B 对照。"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--reasoner-sm-count", type=int, default=34)
    parser.add_argument("--generator-sm-count", type=int, default=136)
    args = parser.parse_args()

    reasoner_prompt = "Describe a safe plan for moving a cup onto a shelf."
    if args.reasoner_input is not None:
        reasoner_data = json.loads(
            args.reasoner_input.read_text(encoding="utf-8")
        )
        reasoner_prompt = reasoner_data["prompt"]
        args.reasoner_max_tokens = int(
            reasoner_data.get("max_new_tokens", args.reasoner_max_tokens)
        )
    expected_reasoner_prompt_tokens: int | None = None
    if args.generator_workload_manifest is not None:
        workload_manifest = json.loads(
            args.generator_workload_manifest.read_text(encoding="utf-8")
        )
        generator_data = workload_manifest["generator"][
            "canonical_workload"
        ]
        reasoner_identity = workload_manifest.get("reasoner", {})
        if "prompt_token_count" in reasoner_identity:
            expected_reasoner_prompt_tokens = int(
                reasoner_identity["prompt_token_count"]
            )
        args.prompt = generator_data["prompt"]
        args.negative_prompt = generator_data["negative_prompt"]
        args.height = int(generator_data["height"])
        args.width = int(generator_data["width"])
        args.num_frames = int(generator_data["num_frames"])
        args.steps = int(generator_data["num_inference_steps"])
        args.guidance_scale = float(generator_data["guidance_scale"])
        args.flow_shift = float(generator_data["flow_shift"])
        args.seed = int(generator_data["seed"])

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

    generator_capability = (
        "text_to_video" if args.num_frames > 1 else "text_to_image"
    )
    generator_spec = ModelSpec(
        model_id=args.model_id,
        model_path=args.model,
        backend="vllm_omni",
        capabilities=(generator_capability,),
        default_task=generator_capability,
        backend_options=cache_options,
    )
    partition_group = "reasoner-generator"
    reasoner_resources = ResourceDescription(
        memory=MemoryResourceDescription(kind="default"),
        execution_variants=(
            ExecutionVariantDescription(
                name="shared",
                stream=StreamResourceDescription(
                    kind="green_context",
                    sm_count=args.reasoner_sm_count,
                    partition_group=partition_group,
                ),
                cuda_graph=args.reasoner_mode == "graph",
            ),
            ExecutionVariantDescription(
                name="reasoner_full",
                stream=StreamResourceDescription(kind="dedicated"),
                cuda_graph=args.reasoner_mode == "graph",
            ),
        ),
        default_variant="reasoner_full",
    )
    generator_resources = ResourceDescription(
        memory=MemoryResourceDescription(kind="tagged_pool"),
        execution_variants=(
            ExecutionVariantDescription(
                name="shared",
                stream=StreamResourceDescription(
                    kind="green_context",
                    sm_count=args.generator_sm_count,
                    partition_group=partition_group,
                ),
            ),
            ExecutionVariantDescription(
                name="full",
                stream=StreamResourceDescription(kind="dedicated"),
            ),
        ),
        default_variant="shared",
    )
    deployment_plan = EngineDeploymentPlan(
        deployments=(
            ModelDeployment(
                spec=ModelSpec(
                    model_id="reasoner",
                    model_path=args.model,
                    backend="vllm",
                    capabilities=("text_generation",),
                    default_task="text_generation",
                ),
                resources=reasoner_resources,
            ),
            ModelDeployment(
                spec=generator_spec,
                resources=generator_resources,
            ),
        ),
        default_model_id="reasoner",
    )

    from vllm import SamplingParams
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = MultiModelEngine.create(
        deployment_plan,
        model=args.model,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        enforce_eager=args.reasoner_mode == "eager",
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    def generator_request(request_id: str) -> Any:
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

    phases: list[dict[str, Any]] = []
    if not args.skip_step_smoke:
        print("[step-smoke] starting", flush=True)
        request = generator_request("container-smoke-generator")
        engine.worker_rpc(
            "recv_model_work",
            args=(args.model_id, request),
        )

        for _ in range(args.steps + 64):
            rank_results = engine.worker_rpc(
                "step_model", args=(args.model_id, "shared")
            )
            first = rank_results[0]
            phase = {
                "kind": _field(first, "kind"),
                "index": _field(first, "index"),
                "request_done": bool(_field(first, "request_done")),
            }
            phases.append(phase)
            if phase["request_done"]:
                break
        else:
            raise RuntimeError(
                "Generator did not complete within the step bound"
            )
        print("[step-smoke] complete", flush=True)

    tokenizer = engine.get_tokenizer()
    formatted_reasoner_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": reasoner_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    reasoner_prompt_tokens = tokenizer.encode(formatted_reasoner_prompt)
    if (
        expected_reasoner_prompt_tokens is not None
        and len(reasoner_prompt_tokens) != expected_reasoner_prompt_tokens
    ):
        raise RuntimeError(
            "Reasoner workload tokenization does not match the manifest: "
            f"expected={expected_reasoner_prompt_tokens}, "
            f"actual={len(reasoner_prompt_tokens)}"
        )

    reasoner_params = SamplingParams(
        temperature=0.0, max_tokens=args.reasoner_max_tokens
    )

    def run_reasoner() -> tuple[float, Any]:
        started = time.perf_counter()
        outputs = engine.submit(
            "reasoner",
            {
                "prompts": [formatted_reasoner_prompt],
                "sampling_params": reasoner_params,
                "use_tqdm": False,
            },
        ).result()
        return time.perf_counter() - started, outputs[0].outputs[0]

    def start_generator(request_id: str, variant_name: str) -> Any:
        return engine.submit(
            args.model_id,
            generator_request(request_id),
            request_id=request_id,
            variant=variant_name,
        )

    def wait_generator(handle: Any) -> dict[str, Any]:
        rows = handle.result(timeout=600.0)
        if len(rows) != 1:
            raise RuntimeError("This first performance smoke requires TP=1")
        return rows[0]

    for index in range(args.warmups):
        reasoner_seconds, _ = run_reasoner()
        generator_result = wait_generator(
            start_generator(f"container-warmup-{index}", "full")
        )
        print(
            f"[warmup {index + 1}/{args.warmups}] "
            f"reasoner={reasoner_seconds:.3f}s "
            f"generator={float(generator_result['seconds']):.3f}s",
            flush=True,
        )

    reasoner_samples = []
    for index in range(args.repeats):
        seconds, _ = run_reasoner()
        reasoner_samples.append(seconds)
        print(
            f"[reasoner {index + 1}/{args.repeats}] {seconds:.3f}s",
            flush=True,
        )

    generator_samples = []
    for index in range(args.repeats):
        result = wait_generator(
            start_generator(f"container-generator-{index}", "full")
        )
        seconds = float(result["seconds"])
        generator_samples.append(seconds)
        print(
            f"[generator {index + 1}/{args.repeats}] {seconds:.3f}s",
            flush=True,
        )

    concurrent_rows = []
    completion = None
    for index in range(args.concurrent_repeats):
        wall_started = time.perf_counter()
        generator_handle = start_generator(
            f"container-concurrent-{index}",
            args.concurrent_generator_variant,
        )
        if args.reasoner_arrival_delay_seconds > 0:
            time.sleep(args.reasoner_arrival_delay_seconds)
        reasoner_started = time.perf_counter()
        reasoner_handle = engine.submit(
            "reasoner",
            {
                "prompts": [formatted_reasoner_prompt],
                "sampling_params": reasoner_params,
                "use_tqdm": False,
            },
        )
        reasoner_outputs = reasoner_handle.result()
        reasoner_seconds = time.perf_counter() - reasoner_started
        completion = reasoner_outputs[0].outputs[0]
        generator_result = wait_generator(generator_handle)
        concurrent_rows.append(
            {
                "reasoner_seconds": reasoner_seconds,
                "generator_seconds": float(generator_result["seconds"]),
                "host_wall_seconds": time.perf_counter() - wall_started,
                "reasoner_arrival_delay_seconds": (
                    args.reasoner_arrival_delay_seconds
                ),
            }
        )
        print(
            f"[concurrent {index + 1}/{args.concurrent_repeats}] "
            f"reasoner={reasoner_seconds:.3f}s "
            f"generator={float(generator_result['seconds']):.3f}s "
            f"wall={concurrent_rows[-1]['host_wall_seconds']:.3f}s",
            flush=True,
        )

    graph_variants = engine.worker_rpc("model_execution_status")

    if completion is None:
        _, completion = run_reasoner()
    report = {
        "deployment_plan": deployment_plan.to_rpc_payload(),
        "reasoner_graph_variants": {
            "status": graph_variants,
        },
        "generator_phases": phases,
        "reasoner_prompt_tokens": len(reasoner_prompt_tokens),
        "reasoner_completion_tokens": len(completion.token_ids),
        "reasoner_finish_reason": completion.finish_reason,
        "configuration": {
            "reasoner_mode": args.reasoner_mode,
            "generator_cache": args.generator_cache,
            "concurrent_generator_variant": (
                args.concurrent_generator_variant
            ),
            "warmups": args.warmups,
            "reasoner_arrival_delay_seconds": (
                args.reasoner_arrival_delay_seconds
            ),
            "repeats": args.repeats,
            "concurrent_repeats": args.concurrent_repeats,
        },
        "performance": {
            "reasoner_alone_samples": reasoner_samples,
            "reasoner_alone_median": statistics.median(reasoner_samples),
            "generator_alone_samples": generator_samples,
            "generator_alone_median": statistics.median(generator_samples),
            "concurrent": concurrent_rows,
            "reasoner_concurrent_median": (
                statistics.median(
                    row["reasoner_seconds"] for row in concurrent_rows
                )
                if concurrent_rows
                else None
            ),
            "generator_concurrent_median": (
                statistics.median(
                    row["generator_seconds"] for row in concurrent_rows
                )
                if concurrent_rows
                else None
            ),
        },
    }
    rendered = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    engine.shutdown()
    print(rendered)


if __name__ == "__main__":
    main()
