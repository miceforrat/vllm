# SPDX-License-Identifier: Apache-2.0
"""C1：验证 LLM forward 与 sampling 的实际 GPU stream/依赖。

旧探针在 Worker.sample_tokens 外层取 current stream，观测点早于
MiniContainer 的资源 Binding，因此 mismatch=0 不能证明 sampling kernel 真正在
forward stream 上。本探针直接包装 ModelRunner.execute_model/sample_tokens，并在
forward stream 上延迟写 sentinel、在 sampling stream 上读取：

* positive：故意把 sampling 放到无依赖 rogue stream，探针必须同时看到 stream
  mismatch 和 sentinel 乱序，否则测试为 INCONCLUSIVE；
* current：不改变当前实现。只有实际 stream 相同且 sentinel 全部有序，才能判定
  C1 未观察到。
"""

from __future__ import annotations

import asyncio

from _common import Harness, base_parser, log, run, verdict

import _probes


async def concurrent_round(h: Harness, idx: int, tag: str) -> bool:
    before = await h.variant_counts()
    generator = await h.submit_generator(
        f"c1-{tag}-gen-{idx}",
        seed=idx,
        variant="shared",
        steps=max(h.args.steps, 20),
    )
    await asyncio.sleep(0.5)
    reasoner = await h.submit_reasoner(f"c1-{tag}-llm-{idx}")
    await reasoner.result()
    after = await h.variant_counts()
    await generator.result(timeout=h.args.timeout)
    return after.get("reasoner:reasoner_full", 0) == before.get(
        "reasoner:reasoner_full", 0
    )


async def run_group(
    h: Harness,
    mode: str,
    repeats: int,
    sleep_cycles: int,
) -> tuple[dict, int]:
    await h.rpc(_probes.p_reset_state)
    log(await h.rpc(
        _probes.p_install_sample_probe,
        mode,
        sleep_cycles,
    ))
    valid = 0
    for index in range(repeats):
        if await concurrent_round(h, index, mode):
            valid += 1
        else:
            log(
                f"[{mode}] round {index}: Reasoner switched to full; "
                "excluded from the shared-tier assertion"
            )
    return await h.rpc(_probes.p_get_sample_probe_state), valid


async def main() -> None:
    parser = base_parser("C1 actual sampling stream/dependency probe")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--positive-control-repeats", type=int, default=1)
    parser.add_argument(
        "--probe-sleep-cycles",
        type=int,
        default=50_000_000,
        help="forward sentinel 写入前插入的 GPU sleep，用于放大无依赖读窗口",
    )
    # 接受旧命令，但新版始终运行阳性对照；否则仍无法证明探针有效。
    parser.add_argument("--observe-only", action="store_true", help="兼容旧命令")
    args = parser.parse_args()

    async with Harness(args) as harness:
        await harness.rpc(
            _probes.p_install_sample_probe,
            "current",
            args.probe_sleep_cycles,
        )
        await (await harness.submit_reasoner("c1-warm-alone")).result()
        await concurrent_round(harness, -1, "warm")

        positive, positive_valid = await run_group(
            harness,
            "positive",
            args.positive_control_repeats,
            args.probe_sleep_cycles,
        )
        current, current_valid = await run_group(
            harness,
            "current",
            args.repeats,
            args.probe_sleep_cycles,
        )

    log(f"[positive] valid_rounds={positive_valid} state={positive}")
    log(f"[current] valid_rounds={current_valid} state={current}")

    detector_valid = (
        positive_valid > 0
        and positive.get("sample_calls", 0) > 0
        and positive.get("sample_on_other_stream", 0) > 0
        and positive.get("ordering_violations", 0) > 0
    )
    if not detector_valid:
        verdict(
            "C1",
            None,
            "阳性对照没有同时触发 stream mismatch 与 sentinel 乱序；"
            f"positive={positive!r}",
            "增大 --probe-sleep-cycles，确认 ModelRunner 调用路径后重跑",
        )
        return

    if current_valid == 0 or current.get("sample_calls", 0) == 0:
        verdict(
            "C1",
            None,
            "没有取得 shared 档位下的有效 sampling 观测",
            "加大 --steps 或减小 --reasoner-max-tokens 后重跑",
        )
        return

    mismatches = int(current.get("sample_on_other_stream", 0))
    ordering = int(current.get("ordering_violations", 0))
    present = mismatches > 0 or ordering > 0
    verdict(
        "C1",
        present,
        (
            f"阳性对照有效；当前路径 sample_calls="
            f"{current.get('sample_calls', 0)}, stream_mismatches={mismatches}, "
            f"ordering_violations={ordering}, sample_stream="
            f"{current.get('last_sample_stream')}, forward_stream="
            f"{current.get('last_forward_stream')}"
        ),
        (
            "让 sampling 经 MiniContainer.run_on_active_assignment 进入 forward "
            "assignment，并由 ModelGPUTimeline 建立依赖"
            if present else "无需修改"
        ),
    )


if __name__ == "__main__":
    run(main())
