# SPDX-License-Identifier: Apache-2.0
"""在 Worker 所在进程里执行的探针（通过 collective_rpc 以可调用对象下发）。

用法约定：
- 每个 p_* 函数的第一个参数是 vLLM 传入的 WorkerWrapperBase；用 _w() 取出 MultiModelWorker。
- 探针只做 monkeypatch / 读状态，不修改仓库里的任何文件；进程退出即失效。
- 探针状态挂在 worker 对象上（w._repro_state），因为 cloudpickle 按值传函数，
  模块级全局变量在两次 RPC 之间不会保留。
- 需要环境变量 VLLM_ALLOW_INSECURE_SERIALIZATION=1（_common.py 会设置）。
- 仅支持 TP=1（UniProcExecutor：EngineCore 与 Worker 在同一进程）。
"""

from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- 基础
def _w(wrapper: Any) -> Any:
    return getattr(wrapper, "worker", wrapper)


def _state(w: Any) -> dict:
    st = w.__dict__.get("_repro_state")
    if st is None:
        st = {"hashes": {}, "latent_hashes": {}, "events": [], "counters": {}}
        w.__dict__["_repro_state"] = st
    return st


def _container(w: Any, model_id: str) -> Any:
    return w._container_registry.get(model_id)


def _unit(w: Any, model_id: str) -> Any:
    return _container(w, model_id).exec_unit


def _pipeline(w: Any, model_id: str) -> Any:
    return _unit(w, model_id).runner.pipeline


def _tensors(obj: Any, depth: int = 0) -> list:
    """在 RunnerOutput / DiffusionOutput / dict / list 中找数组（按遍历顺序）。"""
    import numpy as np
    import torch

    if depth > 6 or obj is None:
        return []
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for key in sorted(obj, key=str):
            out += _tensors(obj[key], depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out += _tensors(item, depth + 1)
        return out
    out = []
    for name in ("result", "media", "video", "tensor", "output", "latents"):
        if hasattr(obj, name):
            out += _tensors(getattr(obj, name), depth + 1)
    return out


def _hash_tensors(tensors: list) -> str | None:
    import hashlib
    import numpy as np
    import torch

    if not tensors:
        return None
    h = hashlib.sha1()
    for t in tensors:
        # 这里的 .cpu() 是一次 host 同步：只在请求最后一个 slice 之后调用，
        # 不影响前面 slice 之间的并发关系。
        h.update(str(tuple(t.shape)).encode())
        if isinstance(t, torch.Tensor):
            # NumPy 没有原生 bfloat16 dtype；按 uint8 读取连续 tensor 的原始
            # 存储字节，既覆盖 BF16，也不会因转换 float32 改变位模式。
            raw = t.detach().contiguous().view(-1).view(torch.uint8)
            h.update(raw.cpu().numpy().tobytes())
        else:
            array = np.ascontiguousarray(t)
            h.update(array.view(np.uint8).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- 通用读写
def p_info(wrapper: Any, model_id: str = "generator") -> dict:
    w = _w(wrapper)
    info = {"worker_class": type(w).__name__, "models": []}
    for item in w._container_registry.all():
        info["models"].append(item.model_id)
    try:
        sched = _unit(w, model_id)._step_scheduler
        info["generator_max_num_running_reqs"] = getattr(
            sched, "max_num_running_reqs", None
        )
    except Exception as exc:  # noqa: BLE001
        info["generator_error"] = repr(exc)
    return info


def p_get_state(wrapper: Any) -> dict:
    import torch

    w = _w(wrapper)
    st = _state(w)
    stats = torch.cuda.memory_stats()
    keys = ("num_alloc_retries", "num_sync_all_streams", "num_device_alloc", "num_device_free")
    mem = {k: stats.get(k) for k in keys}
    mem["reserved_bytes"] = int(torch.cuda.memory_reserved())
    mem["allocated_bytes"] = int(torch.cuda.memory_allocated())
    return {
        "hashes": dict(st["hashes"]),
        "latent_hashes": dict(st["latent_hashes"]),
        "events": list(st["events"])[-200:],
        "counters": dict(st["counters"]),
        "final_window": st.get("final_window"),
        "memory_stats": mem,
    }


def p_reset_state(wrapper: Any) -> str:
    w = _w(wrapper)
    st = _state(w)
    st["hashes"].clear()
    st["latent_hashes"].clear()
    st["events"].clear()
    st["counters"].clear()
    st.pop("final_window", None)
    return "ok"


def p_install_one_shot_driver_failure(
    wrapper: Any,
    request_prefix: str = "c4-fail",
) -> str:
    """在请求已经入队后，让首个匹配 slice 抛出一次 Worker 异常。"""
    w = _w(wrapper)
    st = _state(w)
    if w.__dict__.get("_repro_c4_driver_failure"):
        return "already installed"
    original = w.execute_model_request_slice

    def wrapped(job_id: str, scheduled: Any) -> Any:
        if (
            str(job_id).startswith(request_prefix)
            and not st.get("c4_driver_failure_injected", False)
        ):
            st["c4_driver_failure_injected"] = True
            raise RuntimeError("C4 injected post-admission driver failure")
        return original(job_id, scheduled)

    w.execute_model_request_slice = wrapped
    w.__dict__["_repro_c4_driver_failure"] = True
    return "installed"


def _bump(st: dict, key: str, n: int = 1) -> None:
    st["counters"][key] = st["counters"].get(key, 0) + n


# --------------------------------------------------------------------------- 输出与 latent 哈希（C2/C3/C7 用）
def p_install_output_hasher(wrapper: Any, model_id: str = "generator") -> str:
    """包一层 OmniModelExecUnit.execute_slice：请求完成时在 Worker 内把视频哈希下来。

    原因：C8（N2）——视频不会随 RPC 返回，只能在 Worker 内取。
    """
    w = _w(wrapper)
    st = _state(w)
    unit = _unit(w, model_id)
    if unit.__dict__.get("_repro_hasher"):
        return "already installed"
    orig = unit.execute_slice

    def wrapped(descriptor, *, request_id=None):
        res = orig(descriptor, request_id=request_id)
        meta = dict(getattr(res, "metadata", {}) or {})
        finished = set(meta.get("finished_request_ids", ()))
        ids = tuple(meta.get("request_ids", ()))
        st["events"].append({"tags": sorted(getattr(res, "tags", ())), "request_ids": ids})
        for rid, out in (getattr(res, "output", None) or {}).items():
            if rid in finished:
                err = getattr(getattr(out, "result", None), "error", None)
                st["hashes"][rid] = {"hash": _hash_tensors(_tensors(out)), "error": err}
        return res

    unit.execute_slice = wrapped
    unit.__dict__["_repro_hasher"] = True

    pipeline = _pipeline(w, model_id)
    orig_prepare = pipeline.prepare_vae_blocks

    def prepare(state):
        # 去噪刚结束、VAE 还没开始时的 latents：用来区分"去噪不同"还是"VAE 不同"
        st["latent_hashes"][str(state.request_id)] = _hash_tensors([state.latents])
        return orig_prepare(state)

    pipeline.prepare_vae_blocks = prepare
    return "installed"


# --------------------------------------------------------------------------- C1（N4）：采样所在 stream
def p_install_sample_probe(wrapper: Any, mode: str = "bug") -> str:
    """包一层 Worker.sample_tokens / execute_model，记录采样运行在哪条 stream 上。

    mode:
      "bug"  —— 不改行为，只记录（现状）；
      "sync" —— 标准答案：每次 execute_model 之后、sample_tokens 前后都 torch.cuda.synchronize()，
                消除一切跨 stream 竞争；kernel 与现状完全相同（采样仍在默认 stream），所以数值可直接比较；
      "fix"  —— 模拟修法：把采样放到 forward 所在的 main_stream（变体 stream）上执行。
                注意：lm_head 这时跑在 34 SM 的分区里，cuBLAS 选的 kernel 可能与整卡不同，
                所以 fix 组与 sync 组的 token 不一定逐个相同；判断修法是否正确看 fix 组自身是否稳定。
    """
    import torch

    w = _w(wrapper)
    st = _state(w)
    st["sample_mode"] = mode
    if w.__dict__.get("_repro_sample_probe"):
        return f"already installed (mode -> {mode})"
    orig_sample = type(w).sample_tokens.__get__(w)
    orig_exec = type(w).execute_model.__get__(w)

    def execute_model(*args, **kwargs):
        out = orig_exec(*args, **kwargs)
        if st.get("sample_mode") == "sync":
            torch.cuda.synchronize()
        return out

    def sample_tokens(*args, **kwargs):
        runner = getattr(w, "model_runner", None)
        main = runner.__dict__.get("main_stream") if runner is not None else None
        cur = torch.cuda.current_stream()
        _bump(st, "sample_calls")
        st["counters"]["last_sample_stream"] = hex(int(cur.cuda_stream))
        st["counters"]["last_forward_stream"] = (
            None if main is None else hex(int(main.cuda_stream))
        )
        if main is not None and int(cur.cuda_stream) != int(main.cuda_stream):
            _bump(st, "sample_on_other_stream")
        mode_now = st.get("sample_mode")
        if mode_now == "fix" and main is not None:
            with torch.cuda.stream(main):
                return orig_sample(*args, **kwargs)
        if mode_now == "sync":
            torch.cuda.synchronize()
            out = orig_sample(*args, **kwargs)
            torch.cuda.synchronize()
            return out
        return orig_sample(*args, **kwargs)

    w.execute_model = execute_model
    w.sample_tokens = sample_tokens
    w.__dict__["_repro_sample_probe"] = True
    return f"installed (mode={mode})"


# --------------------------------------------------------------------------- C2（N3）：强制变体切换 + 放大窗口
def p_force_vae_variant_pattern(
    wrapper: Any,
    model_id: str = "generator",
    period: int = 2,
    denoise_variant: str = "shared",
    vae_variants: tuple = ("shared", "full"),
    mode: str = "bug",
    sleep_cycles: int = 50_000_000,
) -> str:
    """让生成器的 VAE slice 每 period 个在两个变体之间切换一次。

    mode:
      "bug"     —— 不加任何保护，复现现状；
      "sync"    —— 每个 slice 后 torch.cuda.synchronize()（对照组：消除一切并发）；
      "eventfix"—— 模拟建议的修法：每个 slice 开始前在当前 stream 上 wait_event(上一个 slice 的完成 event)。
    sleep_cycles：在每个 VAE block 的 decoder 之前插入 torch.cuda._sleep，
      让旧 stream 上的 block 写得更晚，从而放大乱序窗口（0 表示不放大）。
    """
    import dataclasses
    import torch

    from vllm.v1.core.sched.execution_slice_scheduler import ExecutionSliceScheduler

    w = _w(wrapper)
    st = _state(w)
    st["pattern"] = {"period": int(period), "mode": mode, "vae_count": {}}

    # 当前 EngineCore/Scheduler 不保证被 Python GC tracking，不能依赖
    # gc.get_objects() 搜实例。探针与 Scheduler 位于同一个 EngineCore 进程，
    # 因而临时包装类方法即可覆盖实际调度实例。
    if not getattr(ExecutionSliceScheduler, "_repro_pattern", False):
        orig_schedule = ExecutionSliceScheduler.schedule_slice

        def schedule_slice(self, mid, request_id, descriptor):
            out = orig_schedule(self, mid, request_id, descriptor)
            pat = st.get("pattern")
            if pat is None or str(mid) != model_id:
                return out
            if "vae" in descriptor.tags:
                n = pat["vae_count"].get(request_id, 0)
                pat["vae_count"][request_id] = n + 1
                variant = vae_variants[(n // pat["period"]) % len(vae_variants)]
            else:
                variant = denoise_variant
            return dataclasses.replace(out, resource_variant=variant)

        ExecutionSliceScheduler.schedule_slice = schedule_slice
        ExecutionSliceScheduler._repro_pattern = True

    unit = _unit(w, model_id)
    if not unit.__dict__.get("_repro_order"):
        orig_exec = unit.execute_slice

        def execute_slice(descriptor, *, request_id=None):
            pat = st.get("pattern") or {}
            stream = torch.cuda.current_stream()
            if pat.get("mode") == "eventfix" and st.get("last_done") is not None:
                stream.wait_event(st["last_done"])
            try:
                return orig_exec(descriptor, request_id=request_id)
            finally:
                if pat.get("mode") == "eventfix":
                    ev = torch.cuda.Event()
                    ev.record(stream)
                    st["last_done"] = ev
                elif pat.get("mode") == "sync":
                    torch.cuda.synchronize()

        unit.execute_slice = execute_slice
        unit.__dict__["_repro_order"] = True

    pipeline = _pipeline(w, model_id)
    if sleep_cycles and not pipeline.__dict__.get("_repro_vae_sleep"):
        orig_block = pipeline.run_vae_block

        def run_vae_block(state):
            # 在 decoder 之前排一个 GPU 空转：本 block 的写入被推迟，
            # 若下一个 block 在另一条 stream 上且没有依赖，就会先读到旧的 feat_cache。
            torch.cuda._sleep(int(sleep_cycles))
            return orig_block(state)

        pipeline.run_vae_block = run_vae_block
        pipeline.__dict__["_repro_vae_sleep"] = True
    return f"pattern installed: period={period}, mode={mode}, sleep_cycles={sleep_cycles}"


def p_set_pattern_mode(wrapper: Any, mode: str) -> str:
    w = _w(wrapper)
    st = _state(w)
    if "pattern" not in st:
        return "ERROR: pattern not installed"
    st["pattern"]["mode"] = mode
    st["pattern"]["vae_count"] = {}
    st.pop("last_done", None)
    return f"mode={mode}"


# --------------------------------------------------------------------------- C6（E3/W1）：放大"最后一个 slice"窗口
def p_install_final_window_delay(
    wrapper: Any, delay_s: float = 2.0, model_id: str = "generator"
) -> str:
    """在 Worker.execute_slice 返回"模型已空闲"之后、Worker 删除 job 之前睡 delay_s 秒。

    这正是 start_model_request 与 execute_model_request_slice 不在同一把锁里的窗口。
    """
    import time

    w = _w(wrapper)
    st = _state(w)
    st["final_delay"] = float(delay_s)
    if w.__dict__.get("_repro_final_delay"):
        return "already installed"
    orig = type(w).execute_slice.__get__(w)

    from vllm.v1.core.sched.execution_slice_scheduler import ScheduledExecutionSlice

    def execute_slice(scheduled, *, work=None):
        res = orig(scheduled, work=work)
        decision = ScheduledExecutionSlice.from_payload(scheduled)
        if decision.model_id != model_id:  # LLM 的 batch 也走这里，跳过
            return res
        done = res.get("request_done") if isinstance(res, dict) else getattr(res, "request_done", False)
        if done and st.get("final_delay", 0) > 0:
            st["final_window"] = time.time()
            time.sleep(st["final_delay"])
        return res

    w.execute_slice = execute_slice
    w.__dict__["_repro_final_delay"] = True
    return "installed"


# --------------------------------------------------------------------------- C7（E4）：注入错误
def p_inject_finalize_error(wrapper: Any, model_id: str = "generator", rid_prefix: str = "e4-") -> str:
    """让 request_id 以 rid_prefix 开头的请求在最后一个 VAE block 后出错。

    runner 会把异常转成 DiffusionOutput(error=...)，StepScheduler 记为 FINISHED_ERROR。
    """
    w = _w(wrapper)
    pipeline = _pipeline(w, model_id)
    if pipeline.__dict__.get("_repro_inject"):
        return "already installed"
    orig = pipeline.finalize_vae_blocks

    def finalize(state):
        out = orig(state)  # 照常合并视频并释放缓存
        if str(state.request_id).startswith(rid_prefix):
            raise RuntimeError("repro-injected failure")
        return out

    pipeline.finalize_vae_blocks = finalize
    pipeline.__dict__["_repro_inject"] = True
    return "installed"


# --------------------------------------------------------------------------- C11（N5）：统计每个 slice 的 host 同步次数
def p_install_sync_counter(wrapper: Any, model_id: str = "generator") -> str:
    import threading
    import traceback
    import warnings

    import torch

    w = _w(wrapper)
    st = _state(w)
    tls = threading.local()
    st["sync_sites"] = {}
    st["sync_by_phase"] = {}

    orig_show = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "synchroniz" in text.lower():
            phase = getattr(tls, "phase", "other")
            st["sync_by_phase"][phase] = st["sync_by_phase"].get(phase, 0) + 1
            site = "?"
            for frame in reversed(traceback.extract_stack()[:-1]):
                fn = frame.filename
                if ("vllm_omni" in fn or "diffusers" in fn or "vllm" in fn) and "warnings" not in fn:
                    site = f"{fn.split('site-packages/')[-1]}:{frame.lineno}"
                    break
            st["sync_sites"][site] = st["sync_sites"].get(site, 0) + 1
            return None
        return orig_show(message, category, filename, lineno, file, line)

    warnings.simplefilter("always")
    warnings.showwarning = showwarning
    torch.cuda.set_sync_debug_mode("warn")

    unit = _unit(w, model_id)
    orig_exec = unit.execute_slice

    def execute_slice(descriptor, *, request_id=None):
        phase = "vae" if "vae" in descriptor.tags else "denoise"
        tls.phase = phase
        key = f"slices_{phase}"
        st["counters"][key] = st["counters"].get(key, 0) + 1
        try:
            return orig_exec(descriptor, request_id=request_id)
        finally:
            tls.phase = "other"

    unit.execute_slice = execute_slice
    return "installed"


def p_get_sync_report(wrapper: Any) -> dict:
    w = _w(wrapper)
    st = _state(w)
    sites = sorted(st.get("sync_sites", {}).items(), key=lambda kv: -kv[1])[:15]
    return {"by_phase": dict(st.get("sync_by_phase", {})), "counters": dict(st["counters"]), "top_sites": sites}
