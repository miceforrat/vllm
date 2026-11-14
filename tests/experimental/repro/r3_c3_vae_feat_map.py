# SPDX-License-Identifier: Apache-2.0
"""C3（本轮 W3 / O1）：两个请求同时处于 VAE 阶段时，共用 pipeline 上的 vae._feat_map。

怀疑的病灶（vllm-omni 5445e40）：
  pipeline_cosmos3.py  prepare_vae_blocks（:4156 clear_cache）、run_vae_block（:4177-4181 用 self.vae._feat_map）、
  finalize_vae_blocks（:4219 clear_cache）；diffusion_model_runner.py _execute_stepwise_decode_wave
  在一个 wave 里对每个请求依次调用它们。Wan VAE 的时间因果卷积缓存挂在共享的 self.vae 上，
  B 的 prepare 会清掉 A 的缓存，下一波 A 读到 B 的缓存。

前提：生成器 max_num_seqs≥2（默认 1 不会触发）。脚本通过部署计划的 backend_options 设为 2，
并先 RPC 读出 StepScheduler.max_num_running_reqs 确认生效。

判定：
  A 单独、B 单独各跑一次得到视频哈希；再背靠背提交 A、B（同一 seed）。
  B 晚一个 wave 进入，所以 B 少一个去噪步，使两者在同一个 wave 进入 VAE。
  - 若共置时两者的 latent 哈希与单独时相同（去噪无差异），但视频哈希不同 → C3 存在。
  - 修好后（按请求保存 feat_cache），共置视频哈希应与单独完全一致。

运行：
  python r3_c3_vae_feat_map.py --model /path/to/Cosmos3-Edge
"""

from __future__ import annotations

import asyncio

from _common import Harness, base_parser, log, run, verdict

import _probes


async def main() -> None:
    p = base_parser("C3 / W3 repro")
    args = p.parse_args()

    async with Harness(args, generator_options={"max_num_seqs": 2}) as h:
        info = await h.rpc(_probes.p_info, args.model_id)
        log(f"worker info: {info}")
        if info.get("generator_max_num_running_reqs") != 2:
            verdict("C3", None, f"max_num_seqs 没有生效（读到 {info.get('generator_max_num_running_reqs')}），"
                    "无法让两个请求进入同一个 wave", "检查 ModelSpec.backend_options 是否传到 OmniDiffusionConfig.max_num_seqs")
            return
        log(await h.rpc(_probes.p_install_output_hasher, args.model_id))
        await h.run_generator("r3-warm", seed=0, variant="full")

        # B 比 A 晚进入一个 wave，所以让 B 少一个去噪步：两者会在同一个 wave 里一起进入 VAE。
        b_steps = max(2, args.steps - 1)
        await h.run_generator("r3-A-alone", seed=11, variant="full")
        await h.run_generator("r3-B-alone", seed=22, variant="full", steps=b_steps)

        ha = await h.submit_generator("r3-A-pair", seed=11, variant="full")
        hb = await h.submit_generator("r3-B-pair", seed=22, variant="full", steps=b_steps)
        await asyncio.gather(ha.result(timeout=args.timeout), hb.result(timeout=args.timeout))
        st = await h.rpc(_probes.p_get_state)

    hv, hl = st["hashes"], st["latent_hashes"]
    shared_vae_waves = [e for e in st["events"] if "vae" in e["tags"]
                        and {"r3-A-pair", "r3-B-pair"} <= set(e["request_ids"])]
    log(f"VAE waves containing both A and B: {len(shared_vae_waves)}")
    for name in ("A", "B"):
        log(f"{name}: latent alone={hl.get(f'r3-{name}-alone')} pair={hl.get(f'r3-{name}-pair')} | "
            f"video alone={hv.get(f'r3-{name}-alone', {}).get('hash')} pair={hv.get(f'r3-{name}-pair', {}).get('hash')}")

    latent_same = all(hl.get(f"r3-{n}-alone") == hl.get(f"r3-{n}-pair") for n in ("A", "B"))
    video_same = all(hv.get(f"r3-{n}-alone", {}).get("hash") == hv.get(f"r3-{n}-pair", {}).get("hash") for n in ("A", "B"))
    fix = ("修复文档 C3：prepare/run/finalize_vae_blocks 改为每个请求在 state.extra 里持有自己的 feat_cache 列表，"
           "不再调用全局 clear_cache()")
    if not shared_vae_waves:
        verdict("C3", None, "两个请求没有在同一个 VAE wave 里（调度没有把它们放在一起），本次无法判定", fix)
    elif latent_same and not video_same:
        verdict("C3", True, "去噪结果完全相同，但同 wave 解码后视频不同 → VAE 因果缓存被另一个请求覆盖", fix)
    elif not latent_same:
        verdict("C3", None, "去噪阶段就已不同（同 wave 的去噪与单独运行数值不一致），需先排除这一点", fix)
    else:
        verdict("C3", False, "共置与单独的视频完全一致", "无需修改")


if __name__ == "__main__":
    run(main())
