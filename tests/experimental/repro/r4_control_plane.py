# SPDX-License-Identifier: Apache-2.0
"""控制面的几个问题，每次运行一个子测试（有的会让生成模型"坏掉"，所以每个子测试单独起引擎）。

  --test c7  出错请求被当作成功（E4）
  --test c8  生成媒体未随请求结果返回（N2）
  --test c9  多请求重叠时按请求的计时缺失/错误（E5/W5）
  --test c10 取消不会传到 Worker（E6/O5）
  --test c5  slice 预算记在驱动请求头上（E2/W2）→ 进而触发 C4
  --test c4  任一 driver 异常后模型永久失效（E1/W4/O3）

建议顺序：c7 → c9 → c10 → c5 → c4（后两个在问题存在时会让模型失效）。
运行示例：
  for t in c7 c9 c10 c5 c4; do python r4_control_plane.py --model /path/to/Cosmos3-Edge --test $t; done
默认用小负载（256×256、9 帧、4 步）以便快速跑完；控制面问题与负载大小无关。
"""

from __future__ import annotations

import asyncio
import time

from _common import Harness, base_parser, log, run, verdict

import _probes


async def test_c7(h: Harness) -> None:
    log(await h.rpc(_probes.p_inject_finalize_error, h.args.model_id, "e4-"))
    try:
        _, row = await h.run_generator("e4-target", seed=1, variant="full")
        verdict("C7", True, f"注入错误的请求正常返回了结果（seconds={row.get('seconds')}），"
                "会被计入成功率和延迟分布",
                "修复文档 C7：ExecUnit 把 pop_request_state 返回的状态与 error 放进 metadata，"
                "Worker 按请求上报，EngineCore 对非 FINISHED_COMPLETED 的请求 set_exception")
    except Exception as exc:  # noqa: BLE001
        verdict("C7", False, f"请求按预期失败：{exc!r}", "无需修改")


async def test_c8(h: Harness) -> None:
    _, row = await h.run_generator("c8-output", seed=1, variant="full")
    media = row.get("media")
    if not isinstance(media, dict) or not media.get("handle"):
        verdict(
            "C8",
            True,
            f"请求成功但结果中没有媒体句柄；返回字段={sorted(row)!r}",
            "Worker 保存完成请求的媒体 tensor，并在请求结果中返回 "
            "media:// handle；按需通过 fetch_media 做 D2H",
        )
        return

    handle = str(media["handle"])
    try:
        fetched = await h.engine.collective_rpc(
            "fetch_media", args=(handle, True)
        )
        records = [item for item in fetched if item is not None]
        if len(records) != 1:
            raise RuntimeError(
                f"Expected exactly one media owner, got {len(records)}"
            )
        record = records[0]
        tensor = record.get("tensor")
        valid = (
            tensor is not None
            and int(record.get("bytes", 0)) > 0
            and tuple(tensor.shape) == tuple(media.get("shape", ()))
        )
        verdict(
            "C8",
            not valid,
            f"handle={handle}，shape={media.get('shape')}，"
            f"bytes={record.get('bytes')}，D2H={record.get('d2h_seconds')}s",
            "无需修改" if valid else "修复媒体仓与 fetch_media 返回协议",
        )
    except Exception as exc:  # noqa: BLE001
        verdict(
            "C8",
            True,
            f"返回了媒体句柄 {handle!r}，但 fetch_media 失败：{exc!r}",
            "修复媒体句柄的 Worker 所有权、D2H 与释放路径",
        )


async def test_c9(h: Harness) -> None:
    await h.run_generator("c9-warm", seed=0, variant="full")
    t_a, _ = await h.run_generator("c9-solo", seed=1, variant="full")
    log(f"warmed single request wall time ≈ {t_a:.2f}s")
    started_a = time.perf_counter()
    ha = await h.submit_generator("c9-A", seed=1, variant="full")
    await asyncio.sleep(t_a / 2)
    started_b = time.perf_counter()
    hb = await h.submit_generator("c9-B", seed=2, variant="full")

    async def wait_timed(handle, started):
        rows = await handle.result(timeout=h.args.timeout)
        return rows, time.perf_counter() - started

    (ra, wall_a), (rb, wall_b) = await asyncio.gather(
        wait_timed(ha, started_a),
        wait_timed(hb, started_b),
    )
    sa, sb = ra[0].get("seconds"), rb[0].get("seconds")
    log(f"A rows[0]: seconds={sa} finished_seconds={ra[0].get('finished_seconds')}")
    log(f"B rows[0]: seconds={sb} finished_seconds={rb[0].get('finished_seconds')}")
    tolerance = 0.30

    def outside(value: object, expected: float) -> bool:
        if not isinstance(value, (int, float)):
            return True
        # 小负载下 EngineCore/Worker RPC 固定开销占比很高；相对误差与
        # 150 ms 绝对误差取较宽者，避免把控制面开销误判为计时归属错误。
        allowance = max(expected * tolerance, 0.15)
        return abs(float(value) - expected) > allowance

    bad = outside(sa, wall_a) or outside(sb, wall_b)
    ratio_a = None if not isinstance(sa, (int, float)) else float(sa) / wall_a
    ratio_b = None if not isinstance(sb, (int, float)) else float(sb) / wall_b
    verdict("C9", bad,
            f"A 的 seconds={sa}（前端实测={wall_a:.2f}，比例={ratio_a}），"
            f"B 的 seconds={sb}（前端实测={wall_b:.2f}，比例={ratio_b}）；"
            f"允许误差=±{tolerance:.0%}",
            "修复文档 C9：Worker 对每个完成的请求单独返回 seconds/step_count/status，"
            "EngineCore 把各自字段合并进各自的结果行；SLO 统计改在前端计时")


async def test_c10(h: Harness) -> None:
    await h.run_generator("c10-warm", seed=0, variant="full")
    handle = await h.submit_generator("c10-cancel", seed=1, variant="full", steps=max(h.args.steps, 20))
    await asyncio.sleep(h.args.cancel_delay)
    if handle.done():
        verdict("C10", None, "请求在取消探针触发前已经完成，无法验证取消路径", "缩短取消延迟或增大负载")
        return
    await handle.cancel()
    t_cancel = time.perf_counter()
    samples = []
    for _ in range(6):
        samples.append(await h.variant_counts())
        await asyncio.sleep(0.5)
    key = [k for k in samples[0] if k.startswith(f"{h.args.model_id}:")]
    totals = [sum(s.get(k, 0) for k in key) for s in samples]
    status = await h.engine.collective_rpc(
        "model_execution_status", args=(h.args.model_id,)
    )
    row = status[0] if isinstance(status, list) else status
    residual = (
        "c10-cancel" in row.get("model_job_ids", ())
        or int(row.get("pending_work", 0)) != 0
    )
    log(f"generator slice counters after cancel (every 0.5s): {totals}")
    grew = totals[-1] > totals[0]
    verdict("C10", grew or residual, f"取消后 {time.perf_counter() - t_cancel:.1f}s 内生成器 slice 计数从 {totals[0]} 增长到 {totals[-1]}"
            if grew else (f"取消后仍残留 Worker 状态: {row}" if residual else "取消后生成器不再推进，且 Worker 无残留请求状态"),
            "修复文档 C10：finish 时若 Future 未完成，登记 abort；driver 每个 slice 前把 abort 下发给 Worker；"
            "ExecUnit.abort_requests 清 StepScheduler 与 runner 状态；demand 在 Worker 确认停止后再释放")


async def test_c5(h: Harness) -> None:
    _, row = await h.run_generator("c5-solo", seed=1, variant="full")
    n = int(row.get("step_count", 0))
    if n <= 0:
        verdict("C5", None, f"读不到单请求的 slice 数（rows[0]={row}）", "")
        return
    budget = n + 2
    log(f"one request = {n} slices; submitting A and B back-to-back with max_steps={budget} each")
    ha = await h.submit_generator("c5-A", seed=1, variant="full", max_steps=budget)
    hb = await h.submit_generator("c5-B", seed=2, variant="full", max_steps=budget)
    errs = []
    for name, hd in (("A", ha), ("B", hb)):
        try:
            await hd.result(timeout=h.args.timeout)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{name}: {exc!r}")
    log(f"errors: {errs}")
    verdict("C5", any("exceeded" in e for e in errs),
            "B 自己的预算足够，却因为 A（驱动请求）的计数超限而失败" if errs else "A、B 都按各自预算完成",
            "修复文档 C5：按 wave 内每个请求各自计数（descriptor.metadata['request_ids']），"
            "超限只 abort 该请求；variant 也按本 wave 的请求决定")


async def test_c4(h: Harness) -> None:
    log(
        await h.rpc(
            _probes.p_install_one_shot_driver_failure,
            "c4-fail",
        )
    )
    injected_failed = False
    try:
        await h.run_generator("c4-fail", seed=1, variant="full")
        log("injected driver failure unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        injected_failed = "C4 injected" in repr(exc)
        log(f"injected request failed as expected: {exc!r}")
    status = await h.engine.collective_rpc(
        "model_execution_status", args=(h.args.model_id,)
    )
    state = status[0] if isinstance(status, list) else status
    residual = (
        "c4-fail" in state.get("model_job_ids", ())
        or int(state.get("pending_work", 0)) != 0
    )
    try:
        _, row = await h.run_generator("c4-after", seed=2, variant="full")
        broken = not injected_failed or residual
        detail = (
            f"故障注入生效={injected_failed}，残留状态={residual}，"
            f"后续正常请求成功（seconds={row.get('seconds')}）"
        )
        verdict("C4", broken, detail, "无需修改" if not broken else
                "driver 异常退出前调用 Worker.reset_model_requests")
    except Exception as exc:  # noqa: BLE001
        verdict("C4", True, f"一次异常之后，正常请求也失败：{exc!r}（之后所有生成请求都会这样，直到重启）",
                "修复文档 C4：driver 异常退出前调用 Worker.reset_model_requests "
                "（清 _model_jobs、StepScheduler、_pending_scheduler_output、runner 状态与 vae.clear_cache）")


async def main() -> None:
    p = base_parser("control-plane repro")
    p.add_argument(
        "--test",
        choices=("c7", "c8", "c9", "c10", "c5", "c4"),
        required=True,
    )
    p.add_argument(
        "--cancel-delay",
        type=float,
        default=0.1,
        help="C10 提交请求后等待多久触发取消（秒）",
    )
    p.set_defaults(height=256, width=256, num_frames=9, steps=4)
    args = p.parse_args()
    async with Harness(args) as h:
        await {
            "c7": test_c7,
            "c8": test_c8,
            "c9": test_c9,
            "c10": test_c10,
            "c5": test_c5,
            "c4": test_c4,
        }[args.test](h)


if __name__ == "__main__":
    run(main())
