# SPDX-License-Identifier: Apache-2.0
"""使用 AsyncMultiModelEngine 运行真实 Reasoner × Generator 负载。"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from vllm.experimental.async_multi_model_engine import AsyncMultiModelEngine
from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment
from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="generator")
    parser.add_argument("--reasoner-input", type=Path, required=True)
    parser.add_argument(
        "--generator-workload-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--reasoner-mode", choices=("eager", "graph"), default="graph"
    )
    parser.add_argument(
        "--generator-cache",
        choices=("none", "sea_cache"),
        default="sea_cache",
    )
    parser.add_argument("--reasoner-sm-count", type=int, default=34)
    parser.add_argument("--generator-sm-count", type=int, default=136)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrent-repeats", type=int, default=5)
    parser.add_argument(
        "--concurrent-generator-variant",
        choices=("auto", "shared"),
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def deployment_plan(
    args: argparse.Namespace,
    generator_data: dict[str, Any],
) -> EngineDeploymentPlan:
    # 解析并构建部署计划
    cache_options: dict[str, Any] = {"cache_backend": "none"}
    # 启用seacache
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
    partition_group = "reasoner-generator"
    # 创建资源描述符
    # 这里可以设置多种资源变体，比如reasoner有两种：拿34SM或者full SM
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
    # 这里设置generator的资源描述
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
    # 上面这个写法不涉及TP
    # 这里只是离线测试，所以写死了capability
    capability = (
        "text_to_video"
        if int(generator_data["num_frames"]) > 1
        else "text_to_image"
    )
    # 返回最终生成的部署计划
    return EngineDeploymentPlan(
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
                spec=ModelSpec(
                    model_id=args.model_id,
                    model_path=args.model,
                    backend="vllm_omni",
                    capabilities=(capability,),
                    default_task=capability,
                    backend_options=cache_options,
                ),
                resources=generator_resources,
            ),
        ),
        default_model_id="reasoner",
    )


async def main() -> None:
    args = parse_args()
    reasoner_data = json.loads(args.reasoner_input.read_text())
    manifest = json.loads(args.generator_workload_manifest.read_text())
    generator_data = manifest["generator"]["canonical_workload"]
    plan = deployment_plan(args, generator_data)

    from vllm import SamplingParams
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # 创建多模型的引擎
    engine = AsyncMultiModelEngine.create(
        plan,
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
    try:
        tokenizer = engine.get_tokenizer()
        reasoner_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": reasoner_data["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        reasoner_params = SamplingParams(
            temperature=0.0,
            max_tokens=int(reasoner_data["max_new_tokens"]),
        )
        
        def generator_request(request_id: str) -> Any:
            return OmniDiffusionRequest(
                request_id=request_id,
                prompt={
                    "prompt": generator_data["prompt"],
                    "negative_prompt": generator_data["negative_prompt"],
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=int(generator_data["height"]),
                    width=int(generator_data["width"]),
                    num_frames=int(generator_data["num_frames"]),
                    num_inference_steps=int(
                        generator_data["num_inference_steps"]
                    ),
                    guidance_scale=float(generator_data["guidance_scale"]),
                    seed=int(generator_data["seed"]),
                    output_type="np",
                    extra_args={
                        "flow_shift": float(generator_data["flow_shift"])
                    },
                ),
            )

        async def run_reasoner(request_id: str) -> tuple[float, Any]:
            started = time.perf_counter()
            handle = await engine.submit(
                "reasoner",
                {
                    "prompt": reasoner_prompt,
                    "sampling_params": reasoner_params,
                },
                request_id=request_id,
            )
            result = await handle.result()
            return time.perf_counter() - started, result

        async def run_generator(
            request_id: str, variant: str
        ) -> tuple[float, dict[str, Any]]:
            started = time.perf_counter()
            handle = await engine.submit(
                args.model_id,
                generator_request(request_id),
                request_id=request_id,
                variant=variant,
            )
            rows = await handle.result(timeout=600.0)
            if len(rows) != 1:
                raise RuntimeError("This smoke currently requires TP=1")
            return time.perf_counter() - started, rows[0]

        for index in range(args.warmups):
            reasoner_seconds, _ = await run_reasoner(
                f"async-reasoner-warmup-{index}"
            )
            _, generator_result = await run_generator(
                f"async-generator-warmup-{index}", "full"
            )
            print(
                f"[warmup {index + 1}/{args.warmups}] "
                f"reasoner={reasoner_seconds:.3f}s "
                f"generator={float(generator_result['seconds']):.3f}s",
                flush=True,
            )

        reasoner_samples = []
        completion = None
        for index in range(args.repeats):
            seconds, completion = await run_reasoner(
                f"async-reasoner-{index}"
            )
            reasoner_samples.append(seconds)

        generator_samples = []
        for index in range(args.repeats):
            _, result = await run_generator(
                f"async-generator-{index}", "full"
            )
            generator_samples.append(float(result["seconds"]))

        # 跑并发
        concurrent_rows = []
        for index in range(args.concurrent_repeats):
            wall_started = time.perf_counter()
            reasoner_handle = await engine.submit(
                "reasoner",
                {
                    "prompt": reasoner_prompt,
                    "sampling_params": reasoner_params,
                },
                request_id=f"async-concurrent-reasoner-{index}",
            )
            reasoner_started = wall_started
            generator_handle = await engine.submit(
                args.model_id,
                generator_request(f"async-concurrent-generator-{index}"),
                request_id=f"async-concurrent-generator-{index}",
                variant=args.concurrent_generator_variant,
            )

            async def wait_reasoner() -> tuple[float, Any]:
                result = await reasoner_handle.result()
                return time.perf_counter() - reasoner_started, result

            async def wait_generator() -> tuple[float, dict[str, Any]]:
                rows = await generator_handle.result(timeout=600.0)
                return float(rows[0]["seconds"]), rows[0]

            (reasoner_seconds, completion), (
                generator_seconds,
                _,
            ) = await asyncio.gather(wait_reasoner(), wait_generator())
            concurrent_rows.append(
                {
                    "reasoner_seconds": reasoner_seconds,
                    "generator_seconds": generator_seconds,
                    "host_wall_seconds": time.perf_counter() - wall_started,
                }
            )
            print(
                f"[concurrent {index + 1}/{args.concurrent_repeats}] "
                f"reasoner={reasoner_seconds:.3f}s "
                f"generator={generator_seconds:.3f}s",
                flush=True,
            )

        graph_status = await engine.collective_rpc("model_execution_status")
        assert completion is not None
        final = completion.outputs[0]
        report = {
            "engine": "AsyncMultiModelEngine",
            "deployment_plan": engine.deployment_plan.to_rpc_payload(),
            "reasoner_graph_variants": graph_status,
            "reasoner_completion_tokens": len(final.token_ids),
            "performance": {
                "reasoner_alone_samples": reasoner_samples,
                "reasoner_alone_median": statistics.median(
                    reasoner_samples
                ),
                "generator_alone_samples": generator_samples,
                "generator_alone_median": statistics.median(
                    generator_samples
                ),
                "concurrent": concurrent_rows,
                "reasoner_concurrent_median": statistics.median(
                    row["reasoner_seconds"] for row in concurrent_rows
                ),
                "generator_concurrent_median": statistics.median(
                    row["generator_seconds"] for row in concurrent_rows
                ),
            },
        }
        rendered = json.dumps(report, indent=2, default=str)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        print(rendered)
    finally:
        await engine.shutdown_async()


if __name__ == "__main__":
    asyncio.run(main())
