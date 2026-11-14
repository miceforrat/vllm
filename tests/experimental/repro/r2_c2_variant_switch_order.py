# SPDX-License-Identifier: Apache-2.0
"""C2（上一轮 N3 + 本轮 O2）：生成器在两个资源变体之间切换时，跨 stream 没有先后依赖。

怀疑的病灶：
  vllm ffb34c2  binding.py CudaStreamBinding._activate
    - 非切换 slice（:298-305）只 torch.cuda.stream(stream)，不建任何依赖；
    - 切换 slice（:308-322）的 previous 取自 runner.main_stream，而 vllm-omni 的
      DiffusionModelRunner 没有这个属性，所以只等到了 caller（legacy default stream），
      caller 只覆盖"上一次切换 slice"，覆盖不到之后连续的同变体 slice。
  vllm-omni 5445e40 pipeline_cosmos3.py run_vae_block：每个 VAE block 读写 vae._feat_map，
    block 之间没有 host 同步，所以乱序会读到没写完的因果卷积缓存。

方法（全部在开发者自己的引擎里，用 RPC 探针完成，不改仓库文件）：
  1. 强制变体序列：去噪 slice 固定 shared；VAE slice 每 period 个在 shared/full 之间切换。
     两组运行的变体序列完全一样，所以数值路径（每个 kernel 用多少 SM）一样。
  2. 对照组 mode=sync：每个 slice 后 torch.cuda.synchronize()，消除一切并发 → 参考哈希。
  3. 现状组 mode=bug：不加保护，重复 N 次。
  4. 修复组 mode=eventfix：每个 slice 开始前 wait_event(上一个 slice 的完成 event)，即建议的修法。
  5. 放大窗口：每个 VAE block 的 decoder 前插入 torch.cuda._sleep(sleep_cycles)。
  另外打印 PyTorch allocator 的 num_alloc_retries / num_sync_all_streams（C14 / W6 的线索）。

运行：
  python r2_c2_variant_switch_order.py --model /path/to/Cosmos3-Edge --num-frames 65 --repeats 4
"""

from __future__ import annotations

from _common import Harness, base_parser, log, run, verdict

import _probes


async def one(h: Harness, rid: str, seed: int) -> dict:
    await h.run_generator(rid, seed=seed, variant="auto")
    st = await h.rpc(_probes.p_get_state)
    return {"video": st["hashes"].get(rid, {}).get("hash"),
            "latent": st["latent_hashes"].get(rid),
            "mem": st["memory_stats"]}


async def main() -> None:
    p = base_parser("C2 / N3 repro")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--period", type=int, default=2)
    p.add_argument("--sleep-cycles", type=int, default=50_000_000)
    p.set_defaults(num_frames=65)  # 65 帧 → 17 个 VAE block，切换次数更多
    args = p.parse_args()

    async with Harness(args) as h:
        log(await h.rpc(_probes.p_install_output_hasher, args.model_id))
        installed = await h.rpc(
            _probes.p_force_vae_variant_pattern,
            args.model_id,
            args.period,
            "shared",
            ("shared", "full"),
            "sync",
            args.sleep_cycles,
        )
        log(installed)
        if str(installed).startswith("ERROR:"):
            verdict("C2", None, str(installed), "修复复现探针后重跑")
            return

        await h.run_generator("r2-warm", seed=0)  # 热身（含编译）

        await h.rpc(_probes.p_set_pattern_mode, "sync")
        ref = await one(h, "r2-ref", seed=7)
        log(f"reference (sync each slice): video={ref['video']} latent={ref['latent']}")

        await h.rpc(_probes.p_set_pattern_mode, "bug")
        mem0 = ref["mem"]
        bug = [await one(h, f"r2-bug-{i}", seed=7) for i in range(args.repeats)]
        mem1 = bug[-1]["mem"]

        await h.rpc(_probes.p_set_pattern_mode, "eventfix")
        fix = [await one(h, f"r2-fix-{i}", seed=7) for i in range(args.repeats)]

        st = await h.rpc(_probes.p_get_state)
        n_vae = sum(1 for e in st["events"] if "vae" in e["tags"])

    log(f"current-code video hashes : {[r['video'] for r in bug]}")
    log(f"event-fix video hashes    : {[r['video'] for r in fix]}")
    log(f"latents equal to ref (bug): {[r['latent'] == ref['latent'] for r in bug]}")
    gib = 1 << 30
    log(f"reserved memory {mem0.get('reserved_bytes', 0) / gib:.2f} GiB (after sync-mode run) -> "
        f"{mem1.get('reserved_bytes', 0) / gib:.2f} GiB (after switching runs)  "
        "← 增量就是生成器在两条 stream 上各缓存一份激活的代价（C14），决定生成器能用几个档位")
    log(f"allocator retries {mem0.get('num_alloc_retries')} -> {mem1.get('num_alloc_retries')}, "
        f"sync_all_streams {mem0.get('num_sync_all_streams')} -> {mem1.get('num_sync_all_streams')} "
        f"(VAE slices seen in last window: {n_vae})")

    fix_txt = ("修复文档 C2：MiniContainer 维护每个模型的 last_done event，切换时 "
               "stream.wait_event(last_done)；CudaStreamBinding 去掉经 caller 的依赖（同时修 C17/F10）")
    if ref["video"] is None or ref["latent"] is None:
        verdict(
            "C2",
            None,
            "探针没有取得 reference video/latent 哈希，不能用 None 比较判定正确性",
            "修复输出哈希探针后重跑",
        )
        return
    bad = [r["video"] != ref["video"] for r in bug]
    fix_ok = all(r["video"] == ref["video"] for r in fix)
    if not all(r["latent"] == ref["latent"] for r in bug):
        verdict("C2", None, "去噪结果本身与对照组不同（latent 哈希不同），无法把差异归因到 VAE 乱序；"
                "检查是否有其他非确定性（例如 SeaCache 未关、cuDNN benchmark）", fix_txt)
    elif any(bad) and fix_ok:
        verdict("C2", True, f"{sum(bad)}/{len(bad)} 次现状运行的视频与对照组不同，"
                "模拟修复（event 链）后全部一致", fix_txt)
    elif any(bad):
        verdict("C2", True, f"{sum(bad)}/{len(bad)} 次现状运行的视频与对照组不同；"
                "但模拟修复也不完全一致，说明还有别的跨 stream 共享状态（例如 vae_decoded_chunks 的 torch.cat）", fix_txt)
    else:
        verdict("C2", False, "本次未观察到乱序。可把 --sleep-cycles 加大 4 倍、--period 改为 1 再试；"
                "也可先跑 tests_proposed/test_c2_stream_order_gpu.py（不依赖模型，确定性复现）", fix_txt)
    if (mem1.get("num_alloc_retries") or 0) > (mem0.get("num_alloc_retries") or 0):
        log("[C14] allocator 在变体切换期间发生了 alloc retry（会触发全设备同步并 cudaFree），见 修复文档 C14")


if __name__ == "__main__":
    run(main())
