# SPDX-License-Identifier: Apache-2.0
"""C1（上一轮 N4）：LLM 的 lm_head + 采样不在资源绑定里。

怀疑的病灶（vllm ffb34c2）：
  MultiModelWorker 没有覆写 sample_tokens；EngineCore 单独调用
  GPUWorker.sample_tokens（core.py:719-721 → gpu_worker.py:1264-1269），
  它跑在调用线程的 legacy default stream 上，而 forward 跑在变体 stream（GC 34 SM
  或 dedicated 整卡）上，二者之间没有依赖（binding.py:298-305 非切换分支不建依赖）。

三组对照（同一个 greedy 请求，生成器同时占着 136 SM，LLM 在 shared 档）：
  S 标准答案：每次 forward 后、采样前后都 torch.cuda.synchronize()；kernel 与现状相同，只消除竞争。
  B 现状：不改行为，只记录"采样所在 stream ≠ forward 所在 stream"的次数。
  F 模拟修法：采样放到 forward 的 stream 上。
判定：
  - B 中 sample_on_other_stream > 0                 → 结构性问题确认（不依赖时序）
  - 任何一轮 B 的 token ≠ S 的 token                 → 竞争已经造成输出错误
  - F 各轮彼此一致                                  → 修法消除了竞争
    （F 与 S 可能不逐个相同：lm_head 在 34 SM 的分区里可能选到不同的 cuBLAS kernel）
另外检查每一轮里 LLM 是否一直在 shared 档：若中途切到 reasoner_full，该轮数值路径不同，不计入比较。

运行：
  python r1_c1_sample_stream.py --model /path/to/Cosmos3-Edge --repeats 6
"""

from __future__ import annotations

import asyncio

from _common import Harness, base_parser, log, run, verdict

import _probes


async def concurrent_round(h: Harness, idx: int, tag: str) -> tuple[list[int], bool]:
    before = await h.variant_counts()
    # 先让生成器跑起来（产生 demand → LLM 被调度到 shared/34 SM），再提交 LLM。
    gen = await h.submit_generator(f"r1-{tag}-gen-{idx}", seed=idx, variant="shared",
                                   steps=max(h.args.steps, 20))
    await asyncio.sleep(0.5)
    llm = await h.submit_reasoner(f"r1-{tag}-llm-{idx}")
    out = await llm.result()
    after_llm = await h.variant_counts()
    await gen.result(timeout=h.args.timeout)
    key = "reasoner:reasoner_full"
    stayed_shared = after_llm.get(key, 0) == before.get(key, 0)
    return list(out.outputs[0].token_ids), stayed_shared


async def run_group(h: Harness, mode: str, n: int) -> list[list[int]]:
    await h.rpc(_probes.p_install_sample_probe, mode)
    seqs = []
    for i in range(n):
        toks, ok = await concurrent_round(h, i, mode)
        if ok:
            seqs.append(toks)
        else:
            log(f"[{mode}] round {i}: LLM switched to reasoner_full mid-request, excluded "
                "(生成器先结束了；可加大 --steps 或减小 --reasoner-max-tokens)")
    return seqs


async def main() -> None:
    p = base_parser("C1 / N4 repro")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument(
        "--observe-only",
        action="store_true",
        help=(
            "只记录 forward/sample stream，不插入 device synchronize，"
            "用于先确认结构性问题"
        ),
    )
    args = p.parse_args()

    async with Harness(args) as h:
        log(await h.rpc(_probes.p_install_sample_probe, "bug"))
        await (await h.submit_reasoner("r1-warm-alone")).result()
        await concurrent_round(h, -1, "warm")

        if args.observe_only:
            await h.rpc(_probes.p_reset_state)
            bug = await run_group(h, "bug", args.repeats)
            st = await h.rpc(_probes.p_get_state)
            calls = st["counters"].get("sample_calls", 0)
            other = st["counters"].get("sample_on_other_stream", 0)
            sample_stream = st["counters"].get("last_sample_stream")
            forward_stream = st["counters"].get("last_forward_stream")
            log(
                f"[observe-only] sample_tokens calls={calls}, "
                f"on a different stream than forward={other} "
                f"(sample stream={sample_stream}, "
                f"forward stream={forward_stream})"
            )
            if calls == 0:
                verdict(
                    "C1",
                    None,
                    "探针没有截到 sample_tokens 调用",
                    "检查当前 vLLM sample_tokens 调用路径",
                )
            elif sample_stream is None or forward_stream is None:
                verdict(
                    "C1",
                    None,
                    "探针没有取得非空的 sample/forward stream handle，"
                    "不能把 mismatch=0 判定为同 stream",
                    "检查 Runner main_stream 绑定和探针取值路径",
                )
            elif other > 0:
                verdict(
                    "C1",
                    True,
                    f"{other}/{calls} 次采样不在 forward 的 stream 上；"
                    "纯观测模式未改变 CUDA 时序",
                    "让 sample_tokens 复用当前 LLM slice 的资源 assignment",
                )
            else:
                verdict(
                    "C1",
                    False,
                    "采样与 forward 在同一 stream 上",
                    "无需修改",
                )
            return

        ref = await run_group(h, "sync", max(2, args.repeats // 2))
        await h.rpc(_probes.p_reset_state)
        bug = await run_group(h, "bug", args.repeats)
        st = await h.rpc(_probes.p_get_state)
        fix = await run_group(h, "fix", args.repeats)

    calls = st["counters"].get("sample_calls", 0)
    other = st["counters"].get("sample_on_other_stream", 0)
    log(f"[bug] sample_tokens calls={calls}, on a different stream than forward={other} "
        f"(sample stream={st['counters'].get('last_sample_stream')}, "
        f"forward stream={st['counters'].get('last_forward_stream')})")
    if not ref or not bug:
        verdict("C1", None, "有效轮次不足（LLM 中途换档太多），无法比较 token", "加大 --steps 后重跑")
        return
    ref_set = {tuple(x) for x in ref}
    if len(ref_set) > 1:
        log("WARNING: 标准答案组自身不一致——存在与 C1 无关的非确定性（例如原子加法类 kernel），"
            "以下 token 比较只作参考")
    ref0 = tuple(ref[0])
    n_bad = sum(tuple(x) != ref0 for x in bug)
    n_fix_variants = len({tuple(x) for x in fix}) if fix else 0
    log(f"[sync] distinct={len(ref_set)}  [bug] rounds != reference: {n_bad}/{len(bug)}  "
        f"[fix] distinct={n_fix_variants}/{len(fix)}  fix==reference: "
        f"{sum(tuple(x) == ref0 for x in fix)}/{len(fix)}")

    fix_txt = ("修复文档 C1：MultiModelWorker 覆写 sample_tokens，经 MiniContainer.run_on_active_assignment "
               "在 forward 的同一变体 stream 上执行（依赖 C2 的 _record_done）")
    if other > 0 and n_bad > 0:
        verdict("C1", True, f"{other}/{calls} 次采样不在 forward 的 stream 上；{n_bad}/{len(bug)} 轮输出与标准答案不同", fix_txt)
    elif other > 0:
        verdict("C1", True, f"{other}/{calls} 次采样不在 forward 的 stream 上（结构性问题确认）；本次负载下 token "
                "尚未出错，但 lm_head 用的是整卡 SM，分区隔离不成立", fix_txt)
    elif calls == 0:
        verdict("C1", None, "探针没有截到 sample_tokens 调用（可能是 vLLM 版本路径不同）", fix_txt)
    else:
        verdict("C1", False, "采样与 forward 在同一 stream 上", "无需修改")
    if n_fix_variants > 1:
        log("NOTE: 模拟修复组各轮之间仍不一致——说明还有别的竞争（例如 AsyncOutput 拷贝、postprocess 写 req_states）")


if __name__ == "__main__":
    run(main())
