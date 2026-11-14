# SPDX-License-Identifier: Apache-2.0
"""C11（F1 + N5）：生成器每个 slice 里有多少次隐式 host 同步，分别来自哪一行。

怀疑的病灶（vllm-omni 5445e40）：
  - UniPC 调度器每步把 CPU 标量 .to(device) / torch.tensor(..., device=)（scheduling_flow_unipc_multistep.py:301-442）；
  - SeaCache 的 .cpu() / .item()（seacache/hook.py:196-239、276-288；sea_filter.py:45）；
  - pipeline_cosmos3.py:4024 _step_cfg_is_active 的 timestep.item()。
这些同步让 driver 线程在每步都等 GPU，也让"LLM 线程抢 GIL/驱动锁"直接变成生成器分区上的空泡。

方法：在 Worker 进程里打开 torch.cuda.set_sync_debug_mode("warn")，按 slice 阶段（denoise / vae）
统计同步次数，并记录触发位置（Python 栈里最近的 vllm_omni / diffusers / vllm 帧）。
建议分别用 --generator-cache none 与 --generator-cache sea_cache 各跑一次。

运行：
  python r6_c11_sync_points.py --model /path/to/Cosmos3-Edge --generator-cache sea_cache
"""

from __future__ import annotations

from _common import Harness, base_parser, log, run, verdict

import _probes


async def main() -> None:
    p = base_parser("C11 / N5 sync-point counter")
    args = p.parse_args()
    async with Harness(args) as h:
        await h.run_generator("c11-warm", seed=0, variant="full")
        log(await h.rpc(_probes.p_install_sync_counter, args.model_id))
        await h.rpc(_probes.p_reset_state)
        await h.run_generator("c11-run", seed=1, variant="full")
        rep = await h.rpc(_probes.p_get_sync_report)

    c = rep["counters"]
    n_d, n_v = c.get("slices_denoise", 0), c.get("slices_vae", 0)
    s_d, s_v = rep["by_phase"].get("denoise", 0), rep["by_phase"].get("vae", 0)
    log(f"denoise slices={n_d}, syncs={s_d} ({s_d / max(n_d, 1):.1f}/slice); "
        f"vae slices={n_v}, syncs={s_v} ({s_v / max(n_v, 1):.1f}/slice)")
    for site, cnt in rep["top_sites"]:
        log(f"  {cnt:5d}  {site}")
    verdict("C11", (s_d / max(n_d, 1)) >= 1,
            f"每个去噪 slice 平均 {s_d / max(n_d, 1):.1f} 次 host 同步（cache={args.generator_cache}）",
            "修复文档 C11：sigmas/timesteps 预先放到 GPU（或 pinned + non_blocking）；SeaCache 判定在 GPU 上算、"
            "延迟一步读回；_step_cfg_is_active 改用 CPU 侧的 timestep 副本；控制面改为单一非阻塞循环")


if __name__ == "__main__":
    run(main())
