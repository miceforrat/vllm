# vLLM 单框架接入 vLLM-Omni：阶段实现与实验报告

> 日期：2026-10-28  
> 状态：单 GPU、单进程、双模型串行/并发/Green Context 原型已跑通  
> vLLM commit：`fd783be9ce`（分支 `experiment/omni-model-backend`）  
> vLLM-Omni commit：`4fdf39e15`（分支 `experiment/cosmos3-dual-framework`）

## 1. 当前结论

当前实现没有把 Cosmos3 Pipeline 或 Omni `DiffusionModelRunner` 复制进 vLLM，也没有在同一进程中启动第二套 Omni Engine。实际边界是：

```text
同一个 Python PID / CUDA primary context
├── vLLM LLM / Reasoner
│   └── vLLM 原生 LLM Engine、Paged KV、prefill/decode、CUDA Graph
└── EmbeddedOmniDiffusionRunner（薄适配层）
    └── 原版 Omni DiffusionModelRunner
        └── Cosmos3OmniDiffusersPipeline
            ├── condition preparation
            ├── diffusion sampling + SeaCache
            └── VAE decode / fast path
```

vLLM 负责基础资源的创建与所有权；Omni Runner 保留模型加载、Pipeline、扩散缓存和优化执行。当前已经接管或适配三类关键资源：

1. 并行组：vLLM 初始化基础 WORLD/TP/PP，Omni 单卡全局视图借用这些 group。
2. Memory context：权重、SeaCache、Cosmos3 UND cache 的持久 tensor 进入 vLLM tagged memory pool；请求临时峰值单独观测。
3. Stream/GC：Generator 可绑定 vLLM stream 或 GC external stream；LLM graph capture 可显式绑定互斥 GC stream。

这仍是实验原型，而非通用多卡实现。尤其是并行组目前只支持 `world_size=1`，Memory 管理是“同一个 allocator/pool provider 下的分类所有权”，并不是所有缓存共享一套可互换 block arena。

## 2. 代码与工作量边界

### 2.1 vLLM 仓库提交

`fd783be9ce` 共 3210 additions，构成为：

| 类别 | 文件 | 行数 | 性质 |
|---|---|---:|---|
| 核心适配 | `vllm/experimental/omni_runner_adapter.py` | 460 | Runner 生命周期、并行组、memory provider、stream/event |
| GC 资源 | `vllm/experimental/green_context.py` | 262 | CUDA Driver API Green Context 与互斥分区 |
| Graph 适配 | `vllm/experimental/runtime_stream_patch.py` | 125 | vLLM CUDA Graph capture stream 注入 |
| allocator API | `vllm/device_allocator/cumem.py` | +39 | tagged pool 的显式复用接口 |
| Gate/实验入口 | `examples/offline_inference/*.py` | 2054 | 串行、无 GC、GC、组件影响验证，不属于框架核心 |
| 标准负载 | `examples/offline_inference/workloads/*.json` | 24 | 固定 T2V workload |
| 设计文档 | `docs/design/cosmos3_single_framework/vllm_omni_runner_resource_adapter_design.md` | 246 | 设计草案 |

所以“接入 Omni 之外”的 vLLM 核心技术代码约为 886 行：Adapter 460、GC 262、Graph patch 125、allocator 39。约 2k 行 Gate 代码主要用于验证和测量，不能算成生产运行时的必要体积。

### 2.2 vLLM-Omni 配套提交

`4fdf39e15` 修改 4 个文件，88 additions / 13 deletions：

| 文件 | 作用 |
|---|---|
| `diffusion_model_runner.py` | 把 Worker 提供的 memory context 转交 Pipeline/DiT，并暴露最近请求峰值 |
| `cache/seacache/backend.py` | 将 Pipeline 上的 memory provider 传入 SeaCache hook |
| `cache/seacache/hook.py` | 只把长期保留的 indicator/residual clone 放入 cache pool |
| `models/cosmos3/transformer_cosmos3.py` | 把长期保留的 UND K/V 和频率 tensor 放入 `und_cache` pool |

这些是最小 resource-provider patch，不改变 diffusion/VAE 算法，也不替换 Omni Runner。

## 3. Runner 接入与资源所有权

核心入口是 `EmbeddedOmniDiffusionRunner`：

```python
embedded = EmbeddedOmniDiffusionRunner(
    resolve_omni_diffusion_stage(model, cli_overrides=cache_overrides)
)
generator = embedded.start(load_format="default")
```

`resolve_omni_diffusion_stage()` 继续调用 Omni 官方 resolver、stage config 和 `OmniDiffusionConfig`，没有重新实现 Cosmos3 配置解析。Adapter 只要求解析结果恰好包含一个 diffusion stage。

`start()` 的关键顺序是：

```python
init_worker_distributed_environment(...)
self._install_omni_group_views(
    world=get_world_group(),
    tp=get_tp_group(),
    pp=get_pp_group(),
)
self.runner = DiffusionModelRunner(
    vllm_config=self.vllm_config,
    od_config=od_config,
    device=self.device,
)
self.runner.load_model(
    memory_pool_context_fn=memory_pool_context_fn,
    load_format=load_format,
)
```

这里的含义是：

- vLLM 建立 CUDA device、基础分布式环境和 group coordinator；
- Adapter 构造原始 Omni `DiffusionModelRunner`；
- Omni 继续负责 Pipeline/权重加载、compile、SeaCache、VAE fast path；
- vLLM 通过 provider 提供 memory pool 和执行 stream。

### 3.1 为什么需要借用并行组

Omni 单卡 diffusion 实现仍通过模块级 `parallel_state` 全局变量获取 WORLD/SP/PP/CFG/DP 等 group。若在 vLLM 已初始化后再运行 Omni 全套 group 初始化，会重复创建或覆盖同一进程中的基础状态。

当前单卡适配把 vLLM 已创建的 group coordinator 映射为 Omni 只读视图：

```python
singleton = _SingleRankGroupView(tp)
omni_state._WORLD = world
omni_state._SP = singleton
omni_state._PP = _SingleRankGroupView(pp)
omni_state._CFG = singleton
omni_state._DP = singleton
omni_state._FS = singleton
omni_state._HSDP_REPLICATE = singleton
```

关闭时恢复原始模块变量，而不是让 Omni 销毁 vLLM 拥有的 process group：

```python
def close(self):
    ...
    self._restore_omni_group_views()
    self._exit_stack.close()
```

这不是通用 topology registry，只是单 rank compatibility view。未来若 LLM 与 Generator 使用不同 TP/SP，需要把模块级全局变量替换为 model-scoped registry/provider，并保证所有 rank 以一致顺序创建 communicator。

### 3.2 Memory context 管理

#### 权重池

Adapter 复用 vLLM `CuMemAllocator`：第一次进入 tag 使用 `use_memory_pool()`，后续持久缓存增长通过新增的 `reuse_memory_pool()` 回到同一 pool。

```python
@contextmanager
def _memory_pool_context(self, tag):
    if tag in self._memory_pool_tags:
        context = self.memory_allocator.reuse_memory_pool(tag)
    else:
        context = self.memory_allocator.use_memory_pool(tag=tag)
    with context:
        self._memory_pool_tags.add(tag)
        yield
```

`reuse_memory_pool()` 是对 vLLM allocator 的 39 行扩展。它不改变原有 one-shot `use_memory_pool()` 语义，只允许后续 cache 将持久 tensor 放回已经创建的 tagged `MemPool`。

#### SeaCache

SeaCache 原本在请求过程中动态创建并长期保留 indicator 和 residual。配套 Omni patch 只对最终 retained clone 使用 provider：

```python
with self._cache_memory_context():
    indicator.append(filtered.clone())

with self._cache_memory_context():
    cached_residual = residual.clone()
state.history.append((step, cached_residual))
```

FFT/filter 等临时 tensor 仍由普通 PyTorch allocator 管理，避免把请求临时峰值错误标记为持久 cache。

验证结果：SeaCache 首次增长 2 MiB，随后四轮增长均为 0；实际长期 tensor 为 6 个、约 1.10 MB。2 MiB 是 allocator 粒度，不等于 tensor payload。

#### Cosmos3 UND cache

Cosmos3 Transformer 的 reasoner K/V 与频率 tensor 会跨 denoising step 保留。patch 在 provider 存在时创建精确尺寸 clone：

```python
with context_fn("und_cache"):
    self.cached_kv = [
        (
            k[:, :max_real_len].contiguous().clone(),
            v[:, :max_real_len].contiguous().clone(),
        )
        for k, v in cached_kv_full
    ]
    self.cached_freqs_gen = tuple(freq.clone() for freq in freqs_gen)
```

验证结果：58 个持久 tensor、约 2.21 MB payload；pool 首次增长 6 MiB，后续增长为 0。

#### 请求临时显存

Adapter 没有重写 activation allocator，而是记录请求前后与峰值：

```python
baseline_allocated = torch.cuda.memory_allocated(device)
output = self.execute_model(request)
peak_allocated = runner._last_peak_allocated_bytes
transient_headroom = peak_allocated - baseline_allocated
```

标准 832×480、121 帧、35 step 负载下：

- 双模型加载后 Torch allocated 约 13.17 GiB；
- warm Generator peak reserved 约 21.7–22.2 GiB；
- steady Generator transient allocated headroom 约 4.62 GiB；
- persistent tagged pool 约 7.99–8.37 GiB，取决于 workload/cache 状态。

因此目前做到的是统一 allocator/provider 与可观测预算，并未把 LLM KV、SeaCache、UND cache、activation 变成同一种可互换 block。

### 3.3 Stream 与 Event 管理

Generator 不再隐式依赖调用线程的默认 stream。Adapter 可以创建 vLLM-owned stream，也可以接收 GC external stream：

```python
def set_execution_stream(self, stream):
    self.execution_stream = stream
```

一次执行的依赖契约是：

```python
caller = torch.cuda.current_stream(self.device)
stream.wait_stream(caller)
with torch.cuda.stream(stream):
    output = self.runner.execute_model(request)
    completion = torch.cuda.Event()
    completion.record(stream)
caller.wait_event(completion)
```

这保证输入依赖先于 Generator，且调用者后续工作等待 Generator completion；没有使用 device-wide synchronize。`stream_control_stats()` 同时记录 expected/actual handle。普通 dedicated stream 和 120-SM GC stream 的验证均为 `verified=true`。

### 3.4 CUDA Graph capture stream

仅仅在外层使用 `with torch.cuda.stream(gc_stream)` 不足以约束 vLLM Graph。原因是 vLLM：

1. 多个模块通过 by-value import 持有 `graph_capture`；
2. `ModelCudaGraphManager` 会再次选择 capture stream；
3. compiled graph 路径会调用独立的 `current_stream()`。

所以 `install_vllm_graph_capture_stream_patch()` 同时覆盖三层：

```python
for owner in (parallel_state, gpu_model_runner, cudagraph_utils):
    owner.graph_capture = graph_capture_on_external_stream

ModelCudaGraphManager._capture_stream = capture_stream_on_external_green_context
compilation_cuda_graph.current_stream = compiled_capture_stream_on_external_green_context
```

Nsys 2026.1 software trace 已确认：Reasoner graph/non-graph 工作位于 Reasoner GC，Generator 位于互斥 Generator GC。该 patch 是当前 vLLM 版本绑定的实验实现，升级 vLLM 后必须重新审计入口。

## 4. 单框架推进的三个步骤

这里重点列“接入原始 Omni Runner 以后，我们额外修改了什么”。

### 4.1 第一步：同 PID 串行双模型

目标是证明一个 Python PID 和 CUDA primary context 内可以同时驻留：

- vLLM Reasoner；
- 原始 Omni DiffusionModelRunner + Cosmos3 Pipeline。

新增内容：

1. `EmbeddedOmniDiffusionRunner` 生命周期；
2. vLLM 分布式基础环境与 Omni 单 rank group view；
3. vLLM memory pool provider；
4. 原始 DiffusionRequest/Output 桥接；
5. PID、CUDA context、显存快照与输出验证。

证据：

- `same_pid=true`；
- `same_cuda_context=true`；
- Generator runner 类型仍是 `vllm_omni...DiffusionModelRunner`；
- Pipeline 类型仍是 `Cosmos3OmniDiffusersPipeline`；
- 1350-token Reasoner warm median 约 4.34 s；
- 标准 T2V warm median 约 16.88 s；
- 串行 host wall median 约 21.15 s。

对应入口：`omni_runner_dual_model_gate.py`。

### 4.2 第二步：无 GC 的双 stream 并发

这一阶段没有修改模型算法，也没有 spatial partition。新增的是：

1. Generator dedicated stream；
2. Reasoner/Generator 两个 host thread 同时提交；
3. Event 依赖而非 device synchronize；
4. full-SM alone、serial、concurrent wall 的一致计时；
5. 2×2 eager/graph 与 eager/compile 模式消融；
6. Nsys 验证 GPU timeline。

Graph Reasoner + compiled Generator 的三轮结果：

| 指标 | 时间/比率 |
|---|---:|
| Reasoner alone | 4.380 s |
| Generator alone | 16.816 s |
| Full-SM serial wall | 21.170 s |
| Concurrent wall | 20.302 s |
| Reasoner slowdown | 4.634× |
| Generator slowdown | 1.059× |
| Full-SM throughput gain | 1.043× |

这一步证明“host 区间完全重叠”不等于良好 GPU 共置。未做 spatial isolation 时，短 Reasoner 被 Generator kernel 链严重压制，系统吞吐只提高约 4.3%。

对应入口：`omni_runner_concurrent_gate.py`。

### 4.3 第三步：互斥 GC、Graph 重捕获与组件影响

新增内容：

1. CUDA Driver API `GreenContextPair`；
2. 从同一次 SM split 创建互斥的 Generator/Reasoner GC；
3. Generator 绑定 external GC stream；
4. Reasoner graph 在目标 GC 上重新 capture；
5. full-SM、partitioned-alone、colocated 三类基线；
6. Nsys 2026.1 `cuda-sw` trace 验证 stream/greenContextId；
7. 精确 hook vLLM decode、Omni `diffuse()` 和 `_decode_latents()`。

`GreenContextPair` 不能用两个独立 `GreenContextStream` 替代；后者不会保证资源不重叠。当前实现从一个 SM resource split 同时生成 requested 分区和 remainder，再分别创建 context/stream。

136:34 的整请求三轮结果：

| 指标 | 时间/比率 |
|---|---:|
| Full-SM Reasoner alone | 4.380 s |
| Full-SM Generator alone | 16.816 s |
| Partitioned Reasoner alone | 6.579 s |
| Partitioned Generator alone | 20.765 s |
| Colocated Reasoner | 7.489 s |
| Colocated Generator | 21.322 s |
| Reasoner colocation slowdown | 1.138× |
| Generator colocation slowdown | 1.027× |
| Throughput gain vs full-SM serial | 0.993× |
| Throughput gain vs partitioned serial | 1.283× |

必须区分两个 throughput reference：相对 partitioned serial 有 1.283×，但相对真正 full-SM serial 只有 0.993×。这组 1350-token Reasoner 与 T2V 不配平，不能据此宣称部署吞吐收益。

对应入口：`omni_runner_gc_concurrent_gate.py`。

## 5. 组件级双向影响实验

### 5.1 测量边界

组件 runner 没有拆 Pipeline：

- diffusion sampling：hook `Cosmos3OmniDiffusersPipeline.diffuse()`；
- VAE decode：hook `_decode_latents()`；
- LLM decode：hook vLLM ModelRunner 的首个 generation iteration，prefill 完成后才进入 barrier。

```python
attribute = (
    "diffuse" if component == "diffusion_sampling"
    else "_decode_latents"
)
```

两侧到达精确边界后由 barrier 同时释放。每种组件测两次：组件覆盖 Decode、Decode 覆盖组件。报告同时记录每轮实际 completion token 和 `requested_coverage_verified`，避免把未配平实验误当成完整覆盖。

实验配置：Generator 120 SM、Reasoner 50 SM；Reasoner graph；Generator regional compile + SeaCache；每组 warmup 1、alone 3、concurrent 3。

### 5.2 三轮正式结果

| 覆盖方向 | Alone：Decode / 组件 | Colocated：Decode / 组件 | Decode slowdown | 组件 slowdown |
|---|---:|---:|---:|---:|
| Decode 覆盖 VAE | 5.745 / 6.266 s | 9.183 / 6.390 s | **1.599×** | 1.020× |
| VAE 覆盖 Decode | 2.254 / 6.265 s | 5.491 / 6.338 s | **2.436×** | 1.012× |
| Sampling 覆盖 Decode | 5.164 / 16.421 s | 5.871 / 16.929 s | **1.137×** | 1.031× |
| Decode 覆盖 Sampling | 20.130 / 16.375 s | 25.275 / 17.692 s | **1.256×** | 1.080× |

四组均为 3/3 轮覆盖方向成立。大多数 CV <0.5%；VAE 覆盖 Decode 时 Decode colocated CV 为 1.85%。

绝对增量：

| 场景 | Decode 增量 | 组件增量 |
|---|---:|---:|
| Decode 覆盖 VAE | +3.438 s | +0.124 s |
| VAE 覆盖 Decode | +3.237 s | +0.072 s |
| Sampling 覆盖 Decode | +0.707 s | +0.508 s |
| Decode 覆盖 Sampling | +5.145 s | +1.317 s |

### 5.3 当前可以写出的结论

1. VAE decode 对 LLM decode 是强烈且近似单向的干扰。两种覆盖方向下 Decode 都增加约 3.2–3.4 s，而 VAE 只增加 1–2%。
2. 2.436× 不表示机制突然变差；600-token Decode 的独占基线只有 2.254 s，相同的约 3.24 s 绝对惩罚被更小分母放大。
3. Diffusion sampling 干扰较温和但更双向。短 Decode 完全落在 Sampling 内时慢 13.7%；长 Decode 覆盖 Sampling 时 Decode 慢 25.6%，Sampling 慢 8.0%。
4. 现象与“Omni VAE fast path 形成连续 kernel 链、减少提交空隙”假设一致，但还没有通过 kernel/queue 消融证明根因，文档中不能将其写成定论。

对应入口：`omni_runner_component_colocation.py`。

## 6. 哪些优化被保留

当前单框架路径仍使用原始 Omni Runner，因此保留：

- Cosmos3 Pipeline 与官方配置解析；
- regional `torch.compile`；
- SeaCache；
- Cosmos3 UND cache；
- DistributedAutoencoderKLWan；
- Wan VAE lossless fast path、cuDNN causal convolution 与 Triton fused kernels；
- Pipeline 原有 condition/sampling/VAE 执行顺序。

Reasoner 侧保留 vLLM：

- Paged KV；
- 原生 prefill/decode scheduler；
- vLLM CUDA Graph replay；
- vLLM LLM request/output 语义。

当前没有为 Cosmos3 diffusion/VAE 新增 CUDA Graph，因为所用 Omni 路径没有启用这一优化；Generator 的 `regional_compile` 不等于 CUDA Graph。

## 7. 仍然存在的限制

1. **单卡限定**：Omni group view 只支持 world size 1；不同 TP/SP 并行度尚未实现。
2. **模块级全局状态**：当前会临时替换 Omni `parallel_state` 变量；多个 Omni 模型同时初始化需要 model-scoped registry。
3. **Memory 不等于统一 block manager**：权重/SeaCache/UND cache 有统一 provider 和 tag，但 LLM KV、diffusion cache 和 activation 不能动态互借同一批 blocks。
4. **Graph patch 与版本绑定**：vLLM graph capture 三个入口可能随版本改变。
5. **GC 动态调整未实现**：分区创建后固定；尚未做阶段边界的安全 resize/variant switch。
6. **VAE 干扰根因未闭合**：已经稳定测到影响，但还缺同区间 Nsys queue/kernel 分类和必要消融。
7. **实验输出未提交**：原始 PT/Nsys/SQLite/JSON 约 969 MB，保留在本地 `outputs/`；本文固化关键数字，Git 不包含大体积产物。

## 8. 代码入口索引

| 内容 | 路径 |
|---|---|
| Runner 资源适配 | `vllm/experimental/omni_runner_adapter.py` |
| GC Driver API | `vllm/experimental/green_context.py` |
| vLLM graph capture stream patch | `vllm/experimental/runtime_stream_patch.py` |
| tagged pool reuse | `vllm/device_allocator/cumem.py` |
| 串行双模型 Gate | `examples/offline_inference/omni_runner_dual_model_gate.py` |
| 无 GC 并发 Gate | `examples/offline_inference/omni_runner_concurrent_gate.py` |
| GC 整请求 Gate | `examples/offline_inference/omni_runner_gc_concurrent_gate.py` |
| 组件双向影响 Gate | `examples/offline_inference/omni_runner_component_colocation.py` |
| 标准 T2V workload | `examples/offline_inference/workloads/cosmos3_t2v_832x480_f121_s35_empty_negative.json` |

配套 Omni patch：

| 内容 | 路径 |
|---|---|
| Runner provider/peak | `vllm_omni/diffusion/worker/diffusion_model_runner.py` |
| SeaCache provider | `vllm_omni/diffusion/cache/seacache/{backend,hook}.py` |
| UND cache provider | `vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py` |


## 9. 代码导读：从请求入口读到资源底层

本节给出建议阅读路径和关键实现。片段来自当前提交；定位时优先搜索类名或函数名，不依赖容易漂移的行号。

### 9.1 主调用链

```text
omni_runner_dual_model_gate.py::main
  ├─ resolve_omni_diffusion_stage
  │    └─ Omni 官方 resolve_omni_config / OmniDiffusionConfig
  ├─ EmbeddedOmniDiffusionRunner.start
  │    ├─ create_diffusion_vllm_config
  │    ├─ init_worker_distr ibuted_environment
  │    ├─ _install_omni_group_views
  │    ├─ 原始 Omni DiffusionModelRunner
  │    └─ DiffusionModelRunner.load_model(memory_pool_context_fn=...)
  ├─ vLLM LLM.generate                         # Reasoner
  └─ EmbeddedOmniDiffusionRunner.execute_model # Generator
       └─ Omni DiffusionModelRunner.execute_model
            └─ Cosmos3OmniDiffusersPipeline
```

Gate 负责负载、计时和证据记录；`EmbeddedOmniDiffusionRunner` 才是运行时适配层。先用下面命令找到 Gate 的组装点：

```bash
rg -n "resolve_omni_diffusion_stage|EmbeddedOmniDiffusionRunner|embedded.start|LLM\(" \
  examples/offline_inference/omni_runner_dual_model_gate.py
```

### 9.2 配置仍然走 Omni 官方 resolver

文件：`vllm/experimental/omni_runner_adapter.py`

函数：`resolve_omni_diffusion_stage()`

```python
resolution = resolve_omni_config(
    model,
    trust_remote_code=trust_remote_code,
    deploy_config_path=deploy_config_path,
    cli_overrides=overrides,
    stage_overrides=None,
    strategy_config_path=None,
)
stage_config = resolution.stage_configs[0]
engine_args = build_engine_args_dict_from_omni_stage_config(stage_config, model)
diffusion_kwargs = extract_diffusion_stage_config_kwargs(
    engine_args,
    stage_id=stage_config.stage_id,
    include_engine_adapter_metadata=True,
)
diffusion_config = OmniDiffusionConfig(**filtered_kwargs)
diffusion_config.enrich_config()
```

Adapter 只额外要求解析结果为一个 diffusion stage。模型注册、Pipeline class、cache 和 compile 配置仍由 Omni 解释，因此没有复制 Cosmos3 配置解析器。

### 9.3 `start()` 如何形成单进程双模型

函数：`EmbeddedOmniDiffusionRunner.start()`

```python
self.vllm_config = create_diffusion_vllm_config(self.device, od_config)
current_omni_platform.init_diffusion_worker_vllm_config(self.vllm_config)

self._exit_stack.enter_context(set_current_vllm_config(self.vllm_config))
self._exit_stack.enter_context(
    set_forward_context(
        vllm_config=self.vllm_config,
        omni_diffusion_config=od_config,
    )
)

init_worker_distributed_environment(
    self.vllm_config,
    rank=0,
    distributed_init_method="env://",
    local_rank=0,
    backend="nccl",
)
self._install_omni_group_views(
    world=get_world_group(), tp=get_tp_group(), pp=get_pp_group()
)

self.runner = DiffusionModelRunner(
    vllm_config=self.vllm_config,
    od_config=od_config,
    device=self.device,
)
self.runner.load_model(
    memory_pool_context_fn=memory_pool_context_fn,
    load_format=load_format,
)
```

关键点是直接实例化原始 `DiffusionModelRunner`，没有启动第二套 Omni Engine、Orchestrator 或 Worker 进程。建议依次在 `start()`、`DiffusionModelRunner.load_model()`、`DiffusersPipelineLoader.load_model()` 和 Cosmos3 Pipeline 调用入口打断点。

### 9.4 并行组借用的真实实现

Omni 模型代码通过模块级 `parallel_state._SP/_CFG/...` 读取 group，而不是从 Runner 实例获取 provider。单卡兼容视图将这些入口映射到 vLLM 已建立的 coordinator：

```python
class _SingleRankGroupView:
    world_size = 1
    rank_in_group = 0
    ulysses_world_size = 1
    ring_world_size = 1

    def __init__(self, coordinator):
        self.coordinator = coordinator

    @property
    def ulysses_group(self):
        return self.coordinator.device_group

    def __getattr__(self, name):
        return getattr(self.coordinator, name)
```

安装位于 `_install_omni_group_views()`：

```python
singleton = _SingleRankGroupView(tp)
omni_state._WORLD = world
omni_state._SP = singleton
omni_state._PP = _SingleRankGroupView(pp)
omni_state._CFG = singleton
omni_state._DP = singleton
omni_state._FS = singleton
omni_state._HSDP_REPLICATE = singleton
```

关闭时 `_restore_omni_group_views()` 恢复原模块变量，但不销毁 vLLM 拥有的 process group。这也是当前实现明确限制为单 rank 的原因；多卡版本不能只删除检查，而需要 model-scoped topology registry。

### 9.5 Memory provider 跨仓库调用链

```text
EmbeddedOmniDiffusionRunner._memory_pool_context(tag)
  → DiffusionModelRunner.load_model(memory_pool_context_fn=...)
      ├─ weights
      ├─ pipeline._cache_memory_pool_context_fn
      └─ DiT._cache_memory_pool_context_fn
          ├─ SeaCache retained clone → sea_cache
          └─ Cosmos3 retained UND K/V/freq → und_cache
```

vLLM Adapter 决定创建还是复用 tagged pool：

```python
if tag in self._memory_pool_tags:
    context = self.memory_allocator.reuse_memory_pool(tag)
else:
    context = self.memory_allocator.use_memory_pool(tag=tag)
with context:
    self._memory_pool_tags.add(tag)
    yield
```

`reuse_memory_pool()` 位于 `vllm/device_allocator/cumem.py`：

```python
@contextmanager
def reuse_memory_pool(self, tag: str):
    data = self.pointer_to_data[tag]
    old_tag = self.current_tag
    self.current_tag = tag
    try:
        with _reuse_memory_pool(data):
            yield
    finally:
        self.current_tag = old_tag
```

Omni patch 位于另一个仓库。`DiffusionModelRunner.load_model()` 只接受抽象 context factory：

```python
def get_memory_context():
    if memory_pool_context_fn is not None:
        return memory_pool_context_fn("weights")
    return nullcontext()

with get_memory_context():
    self.pipeline = model_loader.load_model(...)

self.pipeline._cache_memory_pool_context_fn = memory_pool_context_fn
for module_name in self.pipeline._dit_modules:
    module._cache_memory_pool_context_fn = memory_pool_context_fn
```

SeaCache 只把最终长期保留的 clone 放进 pool，而不是包围整个 forward：

```python
with self._cache_memory_context():
    indicator.append(filtered.clone())
with self._cache_memory_context():
    cached_residual = residual.clone()
```

UND cache 在 `transformer_cosmos3.py::_retain_und_cache()` 中遵循同样规则；没有 provider 时保留 Omni 原行为，有 provider 时才在 `und_cache` tag 中 clone K/V 和频率 tensor。

### 9.6 Stream/Event 契约

入口：`EmbeddedOmniDiffusionRunner.execute_model()`

```python
caller = torch.cuda.current_stream(self.device)
stream.wait_stream(caller)
with torch.cuda.stream(stream):
    actual = torch.cuda.current_stream(self.device)
    self._last_execution_stream_handle = int(actual.cuda_stream)
    output = self.runner.execute_model(request)
    completion = torch.cuda.Event()
    completion.record(stream)
caller.wait_event(completion)
```

数据依赖是：调用者已提交工作 → Generator stream 等待 caller → Generator 执行并记录 event → caller 等待 completion。这里避免 device-wide synchronize。`stream_control_stats()` 比较 expected/actual handle，用来发现底层逃回其他 stream。

普通并发由 Adapter 创建 stream：

```python
embedded.create_execution_stream()
```

GC 场景由 runtime 注入 external stream：

```python
generator_stream = torch.cuda.ExternalStream(gc_pair.first_stream_ptr)
embedded.set_execution_stream(generator_stream)
```

### 9.7 为什么 `GreenContextPair` 必须来自同一次 split

文件：`vllm/experimental/green_context.py`

```python
cuDeviceGetDevResource(device, available, CU_DEV_RESOURCE_TYPE_SM)
cuDevSmResourceSplitByCount(
    first_resource,
    group_count,
    available,
    second_resource,  # 同一次调用返回 remainder
    flags,
    requested_first_sm_count,
)
```

随后对 requested resource 和 remainder 分别创建 descriptor、GC 和 stream。两个独立的 `GreenContextStream` 都从完整设备请求资源，不能提供非重叠证明；`GreenContextPair` 的互斥性来自同一次 split。

当前 descriptor 只含 SM resource。CUDA 13.2 的 balanced workqueue 支持应在这里扩展 resource descriptor，不需要改变 Adapter 的 `set_execution_stream()` API。

### 9.8 CUDA Graph 为什么 patch 三类入口

文件：`vllm/experimental/runtime_stream_patch.py`

vLLM 多个模块 by-value import 了 `graph_capture`：

```python
for owner in (parallel_state, gpu_model_runner, cudagraph_utils):
    owner.graph_capture = graph_capture_on_external_stream
```

Graph manager 又会自行选择 capture stream：

```python
ModelCudaGraphManager._capture_stream = capture_stream_on_external_green_context
```

compiled graph 路径还会独立查询 current stream：

```python
compilation_cuda_graph.current_stream = (
    compiled_capture_stream_on_external_green_context
)
```

所以外层 `with torch.cuda.stream(reasoner_gc_stream)` 仍不足够。`graph_capture_calls`、`manager_capture_stream_calls` 和 `compiled_capture_stream_calls` 三个计数器用于证明 capture 确实经过补丁。`restore()` 会按反向顺序还原符号，并检测安装后是否又被修改。

### 9.9 各 Gate 对应的代码问题

| Gate | 文件 | 阅读重点 |
|---|---|---|
| 配置 | `omni_runner_config_gate.py` | 官方 resolver 最小输出 |
| 串行双模型 | `omni_runner_dual_model_gate.py` | 同 PID/context、初始化与关闭顺序 |
| full-SM 并发 | `omni_runner_concurrent_gate.py` | dedicated stream、host thread、serial/concurrent wall |
| 互斥 GC | `omni_runner_gc_concurrent_gate.py` | `GreenContextPair`、external stream、graph recapture、三类基线 |
| 组件影响 | `omni_runner_component_colocation.py` | decode hook、`diffuse`/`_decode_latents` hook、coverage 验证 |

组件 Gate 没有复制 Pipeline，只临时 hook 原方法。重点搜索：

```text
decode_region_hook
generator_region_hook
Region.enter / Region.leave
requested_coverage_verified
```

### 9.10 推荐阅读顺序和检索命令

推荐顺序：

1. `omni_runner_config_gate.py`；
2. `omni_runner_dual_model_gate.py::main()`；
3. Adapter 的 `resolve_omni_diffusion_stage()`；
4. Adapter 的 `start()` / `close()`；
5. `_SingleRankGroupView` / `_install_omni_group_views()`；
6. `_memory_pool_context()`，然后跨仓库阅读 Omni memory patch；
7. `execute_model()` / `stream_control_stats()`；
8. `GreenContextPair`；
9. `install_vllm_graph_capture_stream_patch()`；
10. 最后阅读并发和组件 Gate 的测量代码。

```bash
cd /data/cjh/cosmos3-workspace/vllm-single-framework
rg -n "^(class|def) " vllm/experimental/*.py
rg -n "EmbeddedOmniDiffusionRunner|GreenContextPair|install_vllm_graph" \
  examples/offline_inference/*.py

git show fd783be9ce -- \
  vllm/experimental \
  vllm/device_allocator/cumem.py
```

跨仓库查看 Omni patch：

```bash
cd /data/cjh/cosmos3-workspace/vllm-omni

git show 4fdf39e15 -- \
  vllm_omni/diffusion/worker/diffusion_model_runner.py \
  vllm_omni/diffusion/cache/seacache \
  vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py
```
