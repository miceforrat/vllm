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
  先用一对校准请求测出第二个 submit 实际晚进入多少个 wave，再相应减少 B
  的去噪步数，使 A、B 在同一个 wave 进入 VAE。不能把该偏移写死：ModelEmitter
  会在两次前端 submit 之间继续推进模型。
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

        # ModelEmitter 不会为前端的两次 submit 建立隐式 barrier：A 被接纳后，
        # emitter 可能在 B 到达前推进一个或多个 wave。先实测这个偏移，避免
        # 用固定的 ``steps - 1`` 猜测时序。
        cal_a = "r3-cal-A"
        cal_b = "r3-cal-B"
        ha = await h.submit_generator(cal_a, seed=101, variant="full")
        hb = await h.submit_generator(cal_b, seed=102, variant="full")
        await asyncio.gather(
            ha.result(timeout=args.timeout), hb.result(timeout=args.timeout)
        )
        calibration = await h.rpc(_probes.p_get_state)
        cal_events = [
            event
            for event in calibration["events"]
            if {cal_a, cal_b} & set(event["request_ids"])
        ]
        first_b = next(
            (
                index
                for index, event in enumerate(cal_events)
                if cal_b in event["request_ids"]
            ),
            None,
        )
        if first_b is None:
            verdict(
                "C3",
                None,
                "校准请求 B 没有进入执行 wave，无法构造同 wave VAE 负载",
                "检查 ModelEmitter 请求接纳和 Omni StepScheduler batching",
            )
            return
        admission_lag = sum(
            cal_a in event["request_ids"]
            and cal_b not in event["request_ids"]
            and "vae" not in event["tags"]
            and "component:vae" not in event["tags"]
            for event in cal_events[: first_b + 1]
        )
        b_steps = args.steps - admission_lag
        log(
            f"calibrated B admission lag={admission_lag} wave(s); "
            f"A steps={args.steps}, B steps={b_steps}"
        )
        if b_steps < 2:
            verdict(
                "C3",
                None,
                f"校准得到的 admission lag={admission_lag} 过大，无法构造有效 B 请求",
                "增大 --steps 后重试",
            )
            return
        await h.run_generator("r3-A-alone", seed=11, variant="full")
        await h.run_generator("r3-B-alone", seed=22, variant="full", steps=b_steps)

        ha = await h.submit_generator("r3-A-pair", seed=11, variant="full")
        hb = await h.submit_generator("r3-B-pair", seed=22, variant="full", steps=b_steps)
        await asyncio.gather(ha.result(timeout=args.timeout), hb.result(timeout=args.timeout))
        st = await h.rpc(_probes.p_get_state)

    hv, hl = st["hashes"], st["latent_hashes"]
    def is_vae_wave(event: dict) -> bool:
        # ExecutionSlice 通用化后采用稳定的 namespace tag；同时接受旧版
        # ``vae``，方便同一脚本对比迁移前后的结果。
        tags = set(event["tags"])
        return "component:vae" in tags or "vae" in tags

    shared_vae_waves = [
        event
        for event in st["events"]
        if is_vae_wave(event)
        and {"r3-A-pair", "r3-B-pair"} <= set(event["request_ids"])
    ]
    log(f"VAE waves containing both A and B: {len(shared_vae_waves)}")
    if not shared_vae_waves:
        pair_events = [
            event
            for event in st["events"]
            if {"r3-A-pair", "r3-B-pair"} & set(event["request_ids"])
        ]
        log(f"pair event sample: {pair_events[-12:]}")
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
