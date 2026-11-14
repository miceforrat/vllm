# SPDX-License-Identifier: Apache-2.0
"""复现脚本的公共部分：完全复用开发者自己的入口。

- 部署计划：直接 import 开发者的
  examples/offline_inference/omni_async_multi_model_engine_smoke.py 里的 deployment_plan()，
  保证资源描述（34/136 SM、shared/full/reasoner_full 变体、SeaCache 参数）与开发者一致。
- 引擎：AsyncMultiModelEngine.create(...)，参数与该 smoke 相同。
- Worker 内观测：engine.collective_rpc(可调用对象)，探针见 _probes.py。
仅支持 TP=1。
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

# 探针以可调用对象经 RPC 下发，需要允许 cloudpickle（仅限本机调试）。
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cloudpickle  # noqa: E402

import _probes  # noqa: E402

# 按值序列化探针模块：EngineCore 进程不需要能 import 本目录。
cloudpickle.register_pickle_by_value(_probes)


# --------------------------------------------------------------------------- 参数
def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", required=True, help="Cosmos3-Edge 路径（与 smoke 的 --model 相同）")
    p.add_argument("--model-id", default="generator")
    p.add_argument("--reasoner-mode", choices=("eager", "graph"), default="graph")
    p.add_argument("--generator-cache", choices=("none", "sea_cache"), default="none",
                   help="默认关闭 SeaCache，保证每次计算量相同、结果可比")
    p.add_argument("--reasoner-sm-count", type=int, default=34)
    p.add_argument("--generator-sm-count", type=int, default=136)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=33, help="33 帧 → 9 个 VAE block")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--flow-shift", type=float, default=10.0)
    p.add_argument("--prompt", default="A red robot repairs a bicycle in a sunny garage.")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--reasoner-prompt", default="Rewrite this video prompt with rich visual details: a robot repairs a bicycle.")
    p.add_argument("--reasoner-max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--smoke-path", type=Path, default=None,
                   help="开发者 smoke 脚本路径；默认从已安装的 vllm 源码树推断")
    return p


def _load_dev_smoke(path: Path | None) -> Any:
    if path is None:
        import vllm

        root = Path(vllm.__file__).resolve().parents[1]
        path = root / "examples/offline_inference/omni_async_multi_model_engine_smoke.py"
    if not path.exists():
        raise FileNotFoundError(f"找不到开发者的 smoke 脚本：{path}（用 --smoke-path 指定）")
    spec = importlib.util.spec_from_file_location("dev_async_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _with_generator_options(plan: Any, model_id: str, extra: dict) -> Any:
    deployments = []
    for dep in plan.deployments:
        if dep.spec.model_id == model_id:
            options = {**dict(dep.spec.backend_options), **extra}
            dep = dataclasses.replace(dep, spec=dataclasses.replace(dep.spec, backend_options=options))
        deployments.append(dep)
    return dataclasses.replace(plan, deployments=tuple(deployments))


# --------------------------------------------------------------------------- 引擎
class Harness:
    def __init__(self, args: argparse.Namespace, generator_options: dict | None = None):
        self.args = args
        self.generator_options = dict(generator_options or {})
        self.engine = None
        self.tokenizer = None

    async def __aenter__(self) -> "Harness":
        a = self.args
        smoke = _load_dev_smoke(a.smoke_path)
        plan = smoke.deployment_plan(a, {"num_frames": a.num_frames})
        if self.generator_options:
            plan = _with_generator_options(plan, a.model_id, self.generator_options)
        from vllm.experimental.async_multi_model_engine import AsyncMultiModelEngine

        t0 = time.perf_counter()
        self.engine = AsyncMultiModelEngine.create(
            plan,
            model=a.model,
            tensor_parallel_size=1,
            max_num_seqs=1,
            max_model_len=4096,
            gpu_memory_utilization=a.gpu_memory_utilization,
            kv_cache_memory_bytes=a.kv_cache_memory_bytes,
            enforce_eager=a.reasoner_mode == "eager",
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
        self.tokenizer = self.engine.get_tokenizer()
        log(f"engine ready in {time.perf_counter() - t0:.1f}s")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.engine is not None:
            await self.engine.shutdown_async()

    # -- RPC 到 Worker（rank 0）
    async def rpc(self, fn: Any, *args: Any) -> Any:
        rows = await self.engine.collective_rpc(fn, args=args)
        return rows[0] if isinstance(rows, list) and rows else rows

    # -- 请求
    def generator_request(self, request_id: str, seed: int = 0, **override: Any) -> Any:
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        a = self.args
        g = {
            "height": a.height, "width": a.width, "num_frames": a.num_frames,
            "steps": a.steps, "guidance_scale": a.guidance_scale,
            "flow_shift": a.flow_shift, "prompt": a.prompt,
            "negative_prompt": a.negative_prompt,
        }
        g.update(override)
        return OmniDiffusionRequest(
            request_id=request_id,
            prompt={"prompt": g["prompt"], "negative_prompt": g["negative_prompt"]},
            sampling_params=OmniDiffusionSamplingParams(
                height=int(g["height"]), width=int(g["width"]),
                num_frames=int(g["num_frames"]),
                num_inference_steps=int(g["steps"]),
                guidance_scale=float(g["guidance_scale"]),
                seed=int(seed), output_type="np",
                extra_args={"flow_shift": float(g["flow_shift"])},
            ),
        )

    async def submit_generator(self, request_id: str, seed: int = 0, variant: str = "auto",
                               max_steps: int | None = None, **override: Any) -> Any:
        options: dict[str, Any] = {"request_id": request_id, "variant": variant}
        if max_steps is not None:
            options["max_steps"] = int(max_steps)
        return await self.engine.submit(self.args.model_id, self.generator_request(request_id, seed, **override), **options)

    async def submit_reasoner(self, request_id: str, max_tokens: int | None = None) -> Any:
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.args.reasoner_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        params = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=int(max_tokens or self.args.reasoner_max_tokens))
        return await self.engine.submit("reasoner", {"prompt": prompt, "sampling_params": params},
                                        request_id=request_id)

    async def run_generator(self, request_id: str, seed: int = 0, variant: str = "auto", **kw: Any) -> tuple[float, dict]:
        t0 = time.perf_counter()
        handle = await self.submit_generator(request_id, seed, variant, **kw)
        rows = await handle.result(timeout=self.args.timeout)
        return time.perf_counter() - t0, rows[0]

    async def variant_counts(self) -> dict:
        status = await self.engine.collective_rpc("model_execution_status", args=(self.args.model_id,))
        row = status[0] if isinstance(status, list) else status
        return dict(row.get("variant_step_counts", {}))


# --------------------------------------------------------------------------- 输出
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verdict(tag: str, present: bool | None, detail: str, fix: str) -> None:
    status = {True: "BUG PRESENT", False: "NOT OBSERVED", None: "INCONCLUSIVE"}[present]
    bar = "=" * 78
    print(bar)
    print(f"[{tag}] RESULT: {status}")
    print(f"  detail: {detail}")
    print(f"  where to look / fix: {fix}")
    print(bar, flush=True)


def run(main_coro: Any) -> None:
    asyncio.run(main_coro)
