# SPDX-License-Identifier: Apache-2.0
"""C6（E3 / W1 / O4）：新请求在上一批"最后一个 slice"窗口内到达，会被收进 StepScheduler 却没人推进。

怀疑的病灶（vllm ffb34c2）：
  multi_model_worker.py start_model_request（:371-398）用"_model_jobs 里有没有同模型 job"判断是否已有 driver；
  execute_model_request_slice（:421、:451-464）用 execute_slice 里早先算出的 all_done 判断空闲，
  然后删掉该模型的全部 job。两步不在同一把锁里。core.py:1288-1301 看到 driver 还活着就不新开。

方法：用探针在"Worker 已判定空闲、还没删 job"之间睡 delay 秒（只放大窗口，不改逻辑），
在这个窗口里提交请求 B，看 B 能否完成。

运行：
  python r5_c6_lost_wakeup.py --model /path/to/Cosmos3-Edge
"""

from __future__ import annotations

import asyncio
import time

from _common import Harness, base_parser, log, run, verdict

import _probes


async def main() -> None:
    p = base_parser("C6 / W1 repro")
    p.add_argument("--delay", type=float, default=2.0)
    p.set_defaults(height=256, width=256, num_frames=9, steps=4)
    args = p.parse_args()

    async with Harness(args) as h:
        t_single, _ = await h.run_generator("c6-warm", seed=0, variant="full")
        log(f"single request ≈ {t_single:.2f}s")
        log(await h.rpc(_probes.p_install_final_window_delay, args.delay, args.model_id))
        await h.rpc(_probes.p_reset_state)

        ha = await h.submit_generator("c6-A", seed=1, variant="full")
        t0 = time.perf_counter()
        while True:  # 等到 A 进入"已空闲、未删 job"的窗口
            st = await h.rpc(_probes.p_get_state)
            if st.get("final_window"):
                break
            if time.perf_counter() - t0 > args.timeout:
                verdict("C6", None, "没等到 A 的最后一个 slice", "")
                return
            await asyncio.sleep(0.05)
        hb = await h.submit_generator("c6-B", seed=2, variant="full")
        await ha.result(timeout=args.timeout)
        budget = 3 * t_single + args.delay + 10
        try:
            await hb.result(timeout=budget)
            verdict("C6", False, "B 在窗口内到达后仍然完成", "无需修改")
        except Exception as exc:  # noqa: BLE001
            verdict("C6", True, f"B 在 {budget:.0f}s 内没有完成（{type(exc).__name__}: {exc}）；"
                    "它被放进了 StepScheduler，但 driver 已判定空闲并退出",
                    "修复文档 C6：把'判断有无 driver + 入队 + 登记 job'和'重新确认空闲 + 删 job'放进同一把 "
                    "_model_jobs_lock；空闲判定改为锁内读 pending_work_count；"
                    "单元测试见 tests_proposed/test_c6_final_wave_admission.py")


if __name__ == "__main__":
    run(main())
