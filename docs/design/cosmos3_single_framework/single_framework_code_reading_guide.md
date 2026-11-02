# vLLM 单框架多 Runner 改造：架构、资源接入与代码导读

> 日期：2026-10-30  
> 文档性质：面向当前实验分支的代码阅读材料，不是上游 vLLM 既有能力说明。  
> vLLM 仓库：`/workspace/vllm-single-framework`，分支 `experiment/omni-model-backend`。  
> vLLM 基线：`v0.31.0`（`db9527a468`）。  
> vLLM-Omni 配套分支：`experiment/cosmos3-dual-framework`，commit `4fdf39e15`。  
> 当前状态：同构 TP=2、每卡一个 Worker、每个 Worker 同时托管 LLM Runner 与 Omni Diffusion Runner，互补 Green Context 和独立 TP communicator 已跑通；通用 serving 调度、异构并行和跨 communicator 排序协议尚未完成。

本文回答两个问题：

1. 为了让 vLLM 在自己的 Worker 内托管原版 vLLM-Omni `DiffusionModelRunner`，我们到底修改了哪些层？
2. 阅读代码时，应如何从 Engine/Executor 一直追到 Worker、Runner、Pipeline、memory、stream 和通信组？

本文不把 benchmark/Gate 代码冒充正式运行时。文中会明确区分：

- **已有实现**：当前工作树中已经存在并通过某个 Gate 验证；
- **实验编排**：仅在 `examples/offline_inference` 中用于验证，还未接入生产 scheduler；
- **后续设计**：架构上需要，但当前尚未实现。

---

## 1. 背景

### 1.1 vLLM 架构综述

当前使用的是 vLLM V1 多进程架构。对本项目最重要的层次是：

```text
LLM / API frontend
  └─ LLMEngine client
      └─ EngineCore process
          ├─ Scheduler
          ├─ KV-cache block manager
          └─ Executor
              └─ GPU Worker process（通常一张 GPU 一个进程）
                  └─ GPUModelRunner
                      └─ torch.nn.Module / Cosmos3 Reasoner
```

各层职责不能混淆：

| 层 | vLLM 原生职责 | 本项目关注点 |
|---|---|---|
| Frontend / `LLM` | tokenization、请求输入输出、离线 API | 当前 Gate 从这里发起 Reasoner 请求和辅助 Runner RPC |
| EngineCore | 调度请求、管理请求状态和 KV block、形成 `SchedulerOutput` | 目前只调度主 LLM；尚不知道 Diffusion 请求的 readiness/priority |
| Executor | 按 TP/PP/DP 拓扑把调用分发到 Worker | 当前复用 `collective_rpc()` 向所有 Worker 注册/执行辅助 Runner |
| Worker | 拥有具体 GPU、CUDA context、主 Runner、分布式状态 | 当前改造的核心所有权边界；一个 Worker 开始拥有多个 Runner |
| ModelRunner | 加载模型、准备 tensor、执行 forward、capture/replay graph | LLM 仍用原生 `GPUModelRunner`；Generator 复用 Omni Runner |
| Model | 实际模型模块和 kernel | 不把 Cosmos3 Generator 重写为 vLLM AR model |

vLLM 官方架构仍假定“每个 Worker 一个主 ModelRunner”。我们的实现没有把第二个模型硬塞进 `GPUModelRunner`，而是给 Worker 增加辅助 Runner registry：

```text
GPU Worker rank i
├─ primary runner: vLLM GPUModelRunner（Reasoner shard i）
└─ auxiliary runner registry
   └─ generator-a: Omni DiffusionModelRunner（Generator shard i）
```

这样做的原因是 LLM 与 diffusion 的执行契约不同：

- LLM 是 token/KV/scheduler 驱动；
- Diffusion 是 latent/timestep/pipeline 驱动；
- 用 AR `GPUModelRunner` 执行 DiT 会迫使我们重写 Omni 的 Pipeline、SeaCache、VAE fast path、compile 和请求协议。

因此，当前路线是 **vLLM-owned Worker + Omni-owned DiffusionModelRunner + resource adapter**。

建议先阅读 vLLM 自带文档：

- `docs/design/arch_overview.md`
- `vllm/v1/engine/core.py`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/worker/gpu_worker.py`

### 1.2 vLLM-Omni 架构综述

vLLM-Omni 的完整部署路径大致为：

```text
Omni Engine / Orchestrator
  └─ Stage / StagePool
      └─ DiffusionWorker
          └─ DiffusionModelRunner
              └─ Cosmos3OmniDiffusersPipeline
                  ├─ condition preparation
                  ├─ diffusion sampling / DiT / SeaCache
                  └─ VAE decode fast path
```

本项目没有把 Omni Engine、Orchestrator、StagePool 和 DiffusionWorker 一起搬进 vLLM。保留边界从 `DiffusionModelRunner` 开始：

```text
保留：
  DiffusionModelRunner
  Cosmos3OmniDiffusersPipeline
  官方配置 resolver
  loader
  regional torch.compile
  SeaCache
  UND cache
  DistributedAutoencoderKLWan / Wan VAE fast path

替换或适配：
  Worker 生命周期
  CUDA device/context 所有权
  process-group 创建与销毁
  memory context provider
  execution stream
  请求跨进程传输
```

配置仍走 Omni 官方路径。`resolve_omni_diffusion_stage()` 调用：

```python
resolution = resolve_omni_config(...)
stage_config = resolution.stage_configs[0]
engine_args = build_engine_args_dict_from_omni_stage_config(...)
diffusion_kwargs = extract_diffusion_stage_config_kwargs(...)
diffusion_config = OmniDiffusionConfig(**filtered_kwargs)
diffusion_config.enrich_config()
```

因此我们没有复制一套 Cosmos3 配置解析器，也没有复制四千行 Pipeline。当前 Adapter 的职责是把 Omni Runner 放进 vLLM 已有的资源域。

关键代码：

- `vllm/experimental/omni_runner_adapter.py::resolve_omni_diffusion_stage`
- vLLM-Omni：`vllm_omni/diffusion/worker/diffusion_model_runner.py`
- vLLM-Omni：`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`

### 1.3 任务介绍与标准负载

当前代表性任务是视频 Agent 中的 Prompt Upsampler + T2V：

```text
用户短 prompt
  → Reasoner / Prompt Upsampler（LLM）
  → 结构化 temporal caption
  → Cosmos3 T2V Generator
```

这条真实 DAG 有数据依赖；当前共置 Gate 为研究 steady-state throughput，近似流水化为“请求 i 的 Generator 与请求 i+1 的 Reasoner 并发”。它不等同于把同一个请求中存在依赖的两个阶段强行并发。

#### Reasoner 负载

文件：

`examples/offline_inference/workloads/cosmos3_agentic_t2v_matched/reasoner.json`

参数：

- Cosmos3-Edge Reasoner；
- Prompt Upsampler 官方风格模板；
- prompt 288 tokens；
- `max_new_tokens=1350`；
- `do_sample=false`；
- 当前 Worker Gate 使用真实 length stop，不强制 fixed-length；标准输出在本轮通常达到 1350 tokens。

Reasoner 优化：

- vLLM Paged KV；
- vLLM 原生 scheduler/runner；
- `FULL_AND_PIECEWISE` CUDA Graph；
- TP=2 时使用 vLLM 原生 TP communicator。

#### Generator 负载

文件：

`examples/offline_inference/workloads/cosmos3_agentic_t2v_matched/generator_manifest.json`

参数：

| 参数 | 值 |
|---|---:|
| checkpoint | `nvidia/Cosmos3-Edge` |
| resolution | 832×480 |
| frames | 121 |
| duration / fps | 5 秒 / 24 fps |
| diffusion steps | 35 |
| guidance scale | 6.0 |
| flow shift | 10.0 |
| seed | 0 |
| cache | SeaCache |

Generator 继续使用 Omni 的 `Cosmos3OmniDiffusersPipeline`，包括 diffusion sampling、SeaCache 和 VAE decode。

#### 当前 TP2 + GC 代表结果

报告：

`outputs/omni_worker_tp/gc_tp2_graph_worker_groups_v14_retry.json`

每张 RTX 5090 的 170 SM 被分为 Generator 136 SM、Reasoner 34 SM。两卡均使用同样的分区：

| 指标 | 结果 |
|---|---:|
| Reasoner partitioned alone | 3.841 s |
| Generator partitioned alone | 17.367 s |
| Reasoner colocated | 4.471 s |
| Generator colocated | 17.669 s |
| Reasoner slowdown | 1.164× |
| Generator slowdown | 1.017× |
| partitioned-serial reference throughput gain | 1.200× |

该结果验证的是资源接入和同构 TP 可运行性。它不是最终部署收益：Reasoner 与 Generator 未配平，而且吞吐参考是 partitioned-alone serial，不是 full-SM serial。

---

## 2. 单框架改造 Overview

### 2.1 需要接入的资源

“单进程/单框架”不只是把两个 Python 对象放进一个 PID。至少要处理以下资源：

| 资源 | 为什么要接入 | 当前状态 |
|---|---|---|
| Device / CUDA primary context | 防止两个 Worker 各自初始化设备和上下文 | 已完成：由 vLLM Worker 唯一拥有 |
| Runner 生命周期 | 同一 Worker 加载、查找、执行和关闭多个 Runner | 已完成实验实现 |
| Parallel topology | Engine/Executor 决定 rank placement、TP/PP/DP | 同构 TP 已完成；异构拓扑未完成 |
| Communicator | 不同 Runner 可能需要不同 NCCL collective domain | 辅助 TP group registry 已完成 |
| Omni module-level group view | Omni 内部仍读取 `_WORLD/_SP/_PP/...` | 兼容映射已完成，但仍是模块级状态 |
| Memory allocator/context | 权重和持久 cache 必须进入同一 Worker 的内存域 | tagged pool provider 已完成 |
| KV/cache ownership | LLM KV、SeaCache、UND cache 生命周期不同 | 分类接入完成；统一 block arena 未完成 |
| Activation/temporary memory | 影响 admission 和 OOM | 只做峰值 profile，未做统一分配 |
| CUDA stream/event | 两个 Runner 需要独立提交并建立数据依赖 | 已完成 |
| Green Context / SM partition | 提供空间隔离而非仅 time multiplexing | 互补 GC 已完成 |
| CUDA Graph capture/replay | Reasoner graph 必须捕获在目标 GC stream | 版本绑定 patch 已完成 |
| Request/RPC serialization | dataclass 和大媒体 tensor 跨进程传输 | 请求恢复已完成；结果 transport 仍需收敛 |
| Scheduling | 决定何时执行哪个 Runner、何时共置 | 当前仅 Gate 手工编排，未进入 Engine scheduler |
| Collective launch ordering | 多 communicator 并发可能产生跨 Rank 顺序环 | 尚未实现 distributed turnstile |
| Shutdown | Runner、线程、group、stream、graph 的逆序销毁 | 顺序已接入；grace-time 修复待复测 |

除了用户列出的 parallel/memory/stream，最容易漏掉的是：**execution context、request transport、CUDA Graph、生命周期和 collective ordering**。

### 2.2 资源管理接入到哪些层次

#### 当前实现

```text
Frontend / Gate
  └─ WorkerHostedRunnerClient
      ├─ 从 Engine config 读取 TP/PP，形成 ModelDeploymentSpec
      └─ llm_engine.collective_rpc(...)

Engine / Executor（原生）
  └─ 把同一个 RPC 分发到各 Worker

MultiModelGPUWorker（我们的 Worker 子类）
  ├─ 拥有 GPU / primary Runner
  ├─ 创建互补 GreenContextPair
  ├─ 绑定 Reasoner GC stream
  └─ 在 shutdown 中关闭辅助资源

MultiModelWorkerExtension（注入 Worker 的多 Runner 托管扩展）
  ├─ auxiliary runner registry
  ├─ communication-group registry
  ├─ worker-local thread pool
  └─ register / execute / prepare / collect / close RPC

EmbeddedOmniDiffusionRunner（Adapter）
  ├─ Omni config/Runner/Pipeline
  ├─ memory context
  ├─ execution context / TP override
  ├─ stream/event
  └─ Omni group compatibility views
```

当前 Engine 只“间接参与”：`WorkerHostedRunnerClient._plan_homogeneous_tp()` 从 `llm_engine.vllm_config.parallel_config` 读取 topology，再通过 RPC 下发。EngineCore scheduler 本身还没有辅助模型请求队列。

#### 目标层次

后续更严格的职责划分应为：

| 层 | 目标职责 |
|---|---|
| Serving / Request scheduler | DAG readiness、优先级、batching、admission、结果路由 |
| Engine / Executor | model placement、TP/DP/PP topology、全 Rank 一致的执行 epoch |
| Worker runtime | 每卡 Runner registry、group registry、stream/GC、memory budget、graph registry |
| Runner Adapter | 将 Worker 资源映射到具体后端，不自行决定全局 placement |
| Backend Runner | 保留模型算法、cache 语义、compile 和 forward |

尚未实现的关键部分：

1. Diffusion 请求没有进入 vLLM scheduler；
2. Engine 没有做跨请求的动态 placement/SM partition 决策；
3. 没有跨 Runner collective conflict graph；
4. 没有 distributed turnstile 保证跨 communicator host-enqueue order；
5. 没有按 shape/GC/topology 管理多套 CUDA Graph variant；
6. 没有跨 Runner 的统一显存 admission；目前需要手工缩小 Reasoner 的 KV cache budget，为 Generator 权重、cache、activation 和持久 artifact 预留空间。

### 2.3 改造路线如何演进到当前形态

当前代码不是一次设计完成的，理解几个阶段有助于区分仍在仓库中的 Gate 与最终边界：

| 阶段 | 尝试 | 得到的结论 |
|---|---|---|
| Model-layer Gate | 在 vLLM 侧直接构造 Cosmos3 model/Pipeline | 能运行最小生成，但继续走下去会重建 Omni Runner 的 loader、cache、compile 和执行协议 |
| Runner-level Gate | 直接保留原版 `DiffusionModelRunner` | 确立当前最小复用边界 |
| 单 PID 串行 | 主进程中同时驻留 vLLM LLM 与 Embedded Omni Runner | 验证同一 CUDA primary context、权重共驻和配置兼容 |
| 单 PID 并发/GC | dedicated stream、互补 GC、Reasoner graph recapture | 验证资源接入和干扰特征，但 Runner 还不在 vLLM GPU Worker 下面 |
| Worker-owned | 用 Worker extension 注册第二个 Runner | 把 device、allocator、stream 和生命周期所有权真正下沉到 Worker |
| 同构 TP2 | Engine-derived topology + Worker auxiliary TP group | 验证每卡一 Worker、每 Worker 双 Runner、双 communicator |

因此，早期 `omni_runner_*_gate.py` 仍有实验价值，但当前正式阅读主线应从 `omni_worker_owned_gate.py` 和 `vllm/experimental/multi_model_*` 开始。Model-layer 的旧实现不是当前目标架构。

---

## 3. Memory 接入：构思与实现

### 3.1 所有权目标

当前目标不是把所有 tensor 改成同一种 block，而是：

1. 只有 vLLM Worker 建立 CUDA allocator/resource domain；
2. Omni 权重和持久 cache 能进入 vLLM 可识别的 tagged pool；
3. 不改变 SeaCache/UND cache 的算法和 tensor layout；
4. 临时 activation 继续由 PyTorch 分配，但测量其真实峰值；
5. LLM KV cache 预算不应在 Generator 加载前吞尽显存。

### 3.2 `CuMemAllocator` 的最小扩展

文件：`vllm/device_allocator/cumem.py`

上游 `use_memory_pool(tag)` 偏向一次性创建。SeaCache 会在运行期维护持久状态；UND cache 则在每次 generation 的首次 UND forward 创建 request-scoped K/V，并在该 generation 的 denoising steps 间复用。两者都需要在运行期重新进入已有 tagged pool，因此新增：

```python
@contextmanager
def reuse_memory_pool(self, tag: str):
    data = self.allocator_and_pools.get(tag)
    if data is None:
        raise KeyError(...)
    old_tag = self.current_tag
    self.current_tag = tag
    try:
        with torch.cuda.memory.use_mem_pool(data[0]):
            yield
    finally:
        self.current_tag = old_tag
```

这个 patch 保留上游 one-shot 创建语义，只新增显式重入已有 pool 的入口。

### 3.3 Adapter 如何选择 pool

文件：`vllm/experimental/omni_runner_adapter.py`

```python
@contextmanager
def _memory_pool_context(self, tag):
    pool_tag = f"{self.memory_pool_tag_prefix}:{tag}"
    if pool_tag in self._memory_pool_tags:
        context = self.memory_allocator.reuse_memory_pool(pool_tag)
    else:
        context = self.memory_allocator.use_memory_pool(tag=pool_tag)
    with context:
        self._memory_pool_tags.add(pool_tag)
        yield
```

Worker 注册 `generator-a` 时传入 `memory_pool_tag_prefix="generator-a"`，因此不同模型不会都写入裸 `weights/cache` tag：

```text
generator-a:weights
generator-a:cache
generator-a:und_cache
```

### 3.4 vLLM-Omni provider patch

配套 commit `4fdf39e15` 修改 4 个 Omni 文件，88 additions / 13 deletions。

#### 权重

`DiffusionModelRunner.load_model()` 接受：

```python
memory_pool_context_fn: Callable[[str], AbstractContextManager] | None
```

权重加载在：

```python
with memory_pool_context_fn("weights"):
    self.pipeline = model_loader.load_model(...)
```

#### SeaCache

provider 被挂到 Pipeline，再传给 SeaCache hook。只有最终长期保留的 clone 进入 cache pool：

```python
with self._cache_memory_context():
    cached_residual = residual.clone()
```

FFT/filter 的临时 tensor 不进入持久 pool，避免把 activation 当作 cache。

#### UND cache

Cosmos3 Transformer 的跨 denoising-step K/V 和频率 tensor 在：

`vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py::_retain_und_cache`

中进入 `und_cache` context。

### 3.5 请求临时显存测量

`execute_model_profiled()` 在请求前后读取 PyTorch allocator：

```python
baseline_allocated = torch.cuda.memory_allocated(device)
baseline_reserved = torch.cuda.memory_reserved(device)
output = self.execute_model(request)
peak_allocated = runner._last_peak_allocated_bytes
peak_reserved = runner._last_peak_reserved_bytes
```

返回 `DiffusionRequestMemoryProfile`，区分：

- persistent tagged-pool growth；
- transient allocated headroom；
- reserved-pool headroom；
- 请求结束后仍 live 的 delta。

### 3.6 当前没有统一的部分

目前不是统一 block manager：

- LLM KV 仍由 vLLM KV manager 管理；
- SeaCache 仍由 Omni 管理语义和淘汰；
- UND cache 仍由 Cosmos3 module 管理；
- activation/latents 仍由 PyTorch caching allocator 管理；
- CUDA Graph private pool、NCCL workspace、compile artifact 没有统一 admission；
- 三类 cache 不能从同一批预分配 blocks 动态借用容量。

当前 Gate 使用 `--kv-cache-memory-bytes 1073741824` 手工限制 LLM KV。这是实验配置，不是通用 memory planner。

后续合理的最小方案不是再造 allocator，而是在 Engine admission 层建立预算：

```text
weights residency
+ LLM KV reservation
+ persistent diffusion caches
+ graph pools
+ measured transient headroom
+ NCCL/workspace safety margin
<= device budget
```

---

## 4. Stream 接入：构思与实现

### 4.1 Adapter 的普通 stream/event 契约

`EmbeddedOmniDiffusionRunner.execute_model()` 不让 Generator 默默落回 default stream：

```python
caller = torch.cuda.current_stream(self.device)
stream.wait_stream(caller)
with torch.cuda.stream(stream):
    output = self.runner.execute_model(request)
    completion = torch.cuda.Event()
    completion.record(stream)
caller.wait_event(completion)
```

语义是：

```text
caller 已提交的输入工作
  → Generator stream 等待 caller
  → Generator 执行
  → Generator 记录 completion event
  → caller 后续工作等待 completion
```

这避免正常路径中的 device-wide synchronize。`stream_control_stats()` 同时记录 expected/actual stream handle，用于发现内部逃逸。

### 4.2 Green Context 与互补分区

文件：`vllm/experimental/green_context.py`

`GreenContextPair` 从同一次 `cuDevSmResourceSplitByCount()` 得到 requested SM resource 与 remainder：

```text
完整 SM resource
  └─ 一次 split
      ├─ Generator resource（例如 136 SM）
      └─ Reasoner remainder（例如 34 SM）
```

必须来自同一次 split；两个独立创建的 GC stream 不能证明 SM 不重叠。

### 4.3 为什么 GC 在 Worker `init_device()` 创建

文件：`vllm/experimental/multi_model_gpu_worker.py`

```python
def init_device(self):
    super().init_device()
    pair = GreenContextPair(requested, device_index)
    self._multi_model_generator_stream = torch.cuda.ExternalStream(
        pair.first_stream_ptr
    )
    self._multi_model_reasoner_stream = torch.cuda.ExternalStream(
        pair.second_stream_ptr
    )
```

Reasoner 的加载、KV 初始化、compile/warmup、graph capture 和执行都需要在同一个长期稳定 GC stream 上，因此 Worker 覆盖：

```python
load_model()
determine_available_memory()
initialize_from_config()
compile_or_warm_up_model()
execute_model()
sample_tokens()
```

这些方法都进入 `_multi_model_reasoner_context()`。

Generator Runner 注册完成后，Worker 把互补的 Generator stream 注入 Adapter：

```python
execution_stream = worker._multi_model_generator_stream
adapter.set_execution_stream(execution_stream)
```

### 4.4 Reasoner CUDA Graph stream patch

仅在 Worker 外层使用 `torch.cuda.stream(reasoner_stream)` 不够。vLLM 内部有三种 stream 获取路径：

1. by-value import 的 `graph_capture`；
2. `ModelCudaGraphManager._capture_stream()`；
3. compiled graph 路径的 `current_stream()`。

文件：`vllm/experimental/runtime_stream_patch.py`

```python
for owner in (parallel_state, gpu_model_runner, cudagraph_utils):
    owner.graph_capture = graph_capture_on_external_stream

ModelCudaGraphManager._capture_stream = capture_stream_on_external_green_context
compilation_cuda_graph.current_stream = compiled_capture_stream_on_external_green_context
```

`GraphCaptureStreamPatch.stats()` 记录三类调用计数。TP2 v14 报告中每个 rank 均为：

```text
graph_capture_calls = 1
manager_capture_stream_calls = 1
compiled_capture_stream_calls = 58
```

这证明 graph capture 经过目标 Reasoner GC stream，而不是只证明 replay 时外层 current stream 相同。

### 4.5 当前并发编排

辅助 Runner 的并发目前是实验性 host 编排：

1. 每个 Worker 的 `prepare_auxiliary_runner()` 在线程池中创建任务，先等待 `threading.Event`；
2. Engine 正常下发 Reasoner `execute_model()`；
3. `MultiModelGPUWorker.execute_model()` 在进入主 Runner 前调用 `release_prepared_auxiliary_runners()`；
4. 同一 Worker 内两个线程分别向两个 GC stream 提交；
5. frontend 分别 collect Reasoner 和 Generator 结果。

这能构造并发，但还不是 production scheduler。

### 4.6 尚缺的 collective 排序协议

Reasoner 原生 TP communicator 与 Generator auxiliary TP communicator 共享设备和 rank 集合。不同 rank 若出现相反的跨 communicator enqueue 顺序，可能形成循环等待。

当前 `threading.Event` 只提供 worker-local 启动门，不是 distributed turnstile。后续需要：

```text
Engine/Executor 分配全局 boundary epoch
  → 所有 rank 按同一 host-enqueue order 通过 turnstile
  → 对冲突 collective 建立选择性的 stream event edge
  → 非冲突计算继续异步
```

不能简单把所有 Runner 串行化；应只排序 rank 集合交叉且存在 collective 冲突的 segment。

---

## 5. 单 Worker 多 Runner 的实现方法

### 5.1 Worker class 与 extension 的分工

创建 vLLM `LLM` 时配置：

```python
worker_cls = "vllm.experimental.multi_model_gpu_worker.MultiModelGPUWorker"
worker_extension_cls = (
    "vllm.experimental.multi_model_worker.MultiModelWorkerExtension"
)
```

两者不是一回事：

- `MultiModelGPUWorker` 是真正的 Worker 子类，负责 device、GC stream、主 Runner 执行边界和 shutdown；
- `MultiModelWorkerExtension` 通过 vLLM 原生 worker extension 注入 RPC 方法，负责辅助 Runner/group registry。

Extension 机制是 vLLM 已有的 RPC 扩展入口，但“多模型 registry”是我们新增的语义。

### 5.2 声明式部署描述

文件：`vllm/experimental/multi_model_worker.py`

```python
@dataclass(frozen=True)
class ModelDeploymentSpec:
    model_id: str
    model_path: str
    backend: str
    task: str
    worker_ranks: tuple[int, ...]
    communication_groups: tuple[CommunicationGroupSpec, ...]
    load_format: str
    backend_options: Mapping[str, Any]
```

它描述“加载什么、放在哪些 rank、使用哪些通信组”，不携带实际 CUDA/NCCL 对象，因而可以跨 RPC 序列化。

### 5.3 Frontend 到 Worker 的注册调用链

```text
WorkerHostedRunnerClient.register(spec)
  → _plan_homogeneous_tp(spec)
  → llm_engine.collective_rpc("register_auxiliary_runner", payload)
  → 每个 Worker 的 MultiModelWorkerExtension.register_auxiliary_runner()
      → resolve_omni_diffusion_stage()
      → acquire auxiliary TP group
      → EmbeddedOmniDiffusionRunner.start_on_worker()
      → 保存 _RunnerSlot
```

`_RunnerSlot` 保存：

```python
spec
adapter
runner
communication_group_ids
pending_future
pending_start
```

这使 Worker，而不是外部 Gate，成为 Runner 生命周期所有者。

### 5.4 为什么是 `start_on_worker()` 而不是旧 `start()`

旧 `start()` 面向单卡原型，会自行调用 `init_worker_distributed_environment()`。Worker-owned 路径不能重复初始化，因此 `start_on_worker()`：

- 校验 Worker 已有 device/config；
- 读取 vLLM WORLD/TP/PP；
- 接受 Worker registry 创建的 auxiliary TP group；
- 不创建第二个 Worker 或 Omni Engine；
- 在 Worker 的 allocator 和 execution context 下加载模型。

### 5.5 执行与结果所有者

TP 中每个 rank 都必须执行 forward 和 collective，但只有一个 rank 应把大结果返回 frontend：

```python
output_rank = min(slot.spec.worker_ranks)
return output if self.rank == output_rank else None
```

计时路径 `execute_auxiliary_runner_timed()` 不传输媒体 tensor，只返回：

```python
{"host_start": ..., "host_end": ..., "seconds": ...}
```

这是必要的：把 121 帧视频 tensor 反复通过 multiprocessing RPC 会引入同步、shared-memory 和关闭问题。

目前普通 profiled 路径仍会传输一次媒体结果。v15 暴露过 output-owner TP0 清理较慢和 shared-memory 残留；当前代码已增加 Worker shutdown 兜底和分层 grace 配置，但 30/45 秒修复尚待复测。长期应改为 Worker 侧生成 output descriptor/artifact，避免 raw video tensor 进入控制面 RPC。

### 5.6 关闭顺序

`close_auxiliary_runners()` 当前顺序：

```text
release pending start gates
→ 等待 pending future
→ 丢弃 registry 的 runner 重复引用
→ adapter.close()
→ release/destroy auxiliary communication groups
→ clear slots
→ shutdown worker-local thread pool
```

`MultiModelGPUWorker.shutdown()` 再提供异常路径兜底：

```text
close auxiliary runners
→ GPUWorker.shutdown()（primary runner）
→ restore graph patch
→ synchronize/drop GC streams
→ close GreenContextPair
```

---

## 6. 并行组接入：构思与实现

### 6.1 为什么不能只复用 vLLM 默认 TP group

Reasoner 和 Generator 可以拥有相同 rank 集合，但仍可能需要：

- 不同 collective 序列；
- 不同 graph capture 生命周期；
- 不同 communicator buffer/warmup；
- 未来不同 TP/DP/EP 组合；
- 独立销毁而不破坏主模型。

因此当前同构 TP Gate 使用相同 ranks、独立 communicator：

```text
Reasoner TP: vLLM 原生 tp:0
Generator TP: generator-a-tp:0
rank members: [0, 1]
```

“同一 rank 集合”不等于“同一个 communicator”。

### 6.2 Engine 规划，Worker 创建

`WorkerHostedRunnerClient._plan_homogeneous_tp()` 从 Engine config 读取：

```python
parallel = llm_engine.vllm_config.parallel_config
tp_size = parallel.tensor_parallel_size
pp_size = parallel.pipeline_parallel_size
ranks = tuple(range(tp_size))
```

然后生成：

```python
CommunicationGroupSpec(
    group_id=f"{model_id}-tp",
    parallel_dimension="tp",
    ranks=ranks,
)
```

这里体现目标职责：Engine/Executor 规划 topology，Worker 在自己的进程中 materialize communicator。

当前只支持：

- PP=1；
- Generator placement 与 vLLM TP ranks 完全一致；
- Omni SP/CFG/DP/HSDP/EP 都为 1/关闭。

### 6.3 Worker communication-group registry

每个 Worker 保存：

```python
@dataclass
class _CommunicationGroupSlot:
    spec: CommunicationGroupSpec
    group: GroupCoordinator
    owners: set[str]
```

创建使用 vLLM 原生 `init_model_parallel_group()`：

```python
group = init_model_parallel_group(
    group_ranks=[list(spec.ranks)],
    local_rank=world.local_rank,
    backend=backend,
    group_name=spec.group_id,
    use_device_communicator=True,
)
```

registry 通过 `owners` 引用计数复用同一 group spec；最后一个 owner 释放时调用 `GroupCoordinator.destroy()`。

### 6.4 执行时 TP group override

问题是 vLLM/Omni 代码大量调用模块级 `get_tp_group()`。Generator 使用独立 group 时，不能永久覆盖全局 `_TP`，否则 Reasoner 会拿错 communicator。

文件：`vllm/distributed/parallel_state.py`

当前增加：

```python
_TP_CONTEXT_OVERRIDE: ContextVar[GroupCoordinator | None]
_TP_COMPILE_OVERRIDE: GroupCoordinator | None

@contextmanager
def use_tensor_model_parallel_group(group):
    token = _TP_CONTEXT_OVERRIDE.set(group)
    _TP_COMPILE_OVERRIDE = group
    try:
        yield group
    finally:
        _TP_CONTEXT_OVERRIDE.reset(token)
```

`get_tp_group()`：

- eager 路径读取 `ContextVar`，允许不同 host thread 使用不同 group；
- Dynamo/AOT compiling 时避免 trace `ContextVar.get()`，改用短期静态 compile override；
- 没有 override 时保持上游 `_TP` 行为。

Adapter 的 `_execution_context()` 将 TP override 与两个配置 context 组合：

```python
with (
    use_tensor_model_parallel_group(auxiliary_tp_group),
    set_current_vllm_config(diffusion_vllm_config),
    set_forward_context(...),
):
    yield
```

因此不仅 group，Generator 在执行时还拿到自己的 `VllmConfig` 和 Omni forward context。

### 6.5 Omni group compatibility views

Omni 仍通过模块变量读取 `_WORLD/_SP/_PP/_CFG/_DP/_FS/...`。`_install_omni_group_views()` 当前映射：

```python
omni_state._WORLD = auxiliary_tp_group
omni_state._SP = singleton_pp_view
omni_state._PP = singleton_pp_view
omni_state._CFG = singleton_pp_view
omni_state._DP = singleton_pp_view
omni_state._FS = singleton_pp_view
omni_state._HSDP_REPLICATE = singleton_pp_view
omni_state._EXPERT_PARALLEL_GROUP_RANKS = [list(tp.ranks)]
```

这里用 vLLM PP singleton coordinator 作为所有关闭维度的 backing group，避免“声明 world_size=1，却暴露多 rank TP device_group”的不一致。

这个兼容层仍然是当前实现最需要谨慎的地方：它修改 Omni 模块级全局变量。当前每个 Worker 只有一个 Omni auxiliary Runner，因此可控；多个 Omni Runner 拥有不同 topology 时必须改成 model-scoped getter/provider，不能来回覆盖全局变量。

### 6.6 TP2 验证证据

v14 报告中两个 Worker：

```text
rank 0: tp_rank=0, tp_size=2, auxiliary_tp_group=generator-a-tp:0
rank 1: tp_rank=1, tp_size=2, auxiliary_tp_group=generator-a-tp:0
communicator_isolated=true
```

日志显示 Generator group 使用独立 PYNCCL communicator，Reasoner 保留原生 TP communicator。

### 6.7 尚未实现的并行能力

1. Generator TP 与 Reasoner TP 不同；
2. Generator TP、LLM DP 的异构映射；
3. 多个 LLM replica 与一个跨卡 Generator 的 placement；
4. Omni SP、CFG、VAE patch parallel、HSDP、EP；
5. 跨 group 一致的创建顺序协议；
6. 跨 communicator distributed turnstile；
7. CUDA Graph 中 collective 与动态 group/stream variant 的管理。

---

## 7. Technical 修改位置与工作量总结

### 7.1 vLLM 当前 technical 代码

相对 `v0.31.0` 基线，并包含当前未提交工作树：

| 文件 | 当前新增行数/文件行数 | 作用 | 状态 |
|---|---:|---|---|
| `vllm/device_allocator/cumem.py` | +39 | tagged MemPool 重入 | 已提交 |
| `vllm/experimental/green_context.py` | 262 | GC Driver API 与互补 SM split | 已提交 |
| `vllm/experimental/runtime_stream_patch.py` | 125 | Reasoner graph capture stream 注入 | 已提交 |
| `vllm/experimental/omni_runner_adapter.py` | 658 | resolver、Runner、memory、stream、group compatibility | 已提交基础 + 当前扩展 |
| `vllm/distributed/parallel_state.py` | +39 | execution-scoped TP override | 当前未提交 |
| `vllm/experimental/multi_model_gpu_worker.py` | 147 | Worker GC、主 Runner stream、shutdown | 当前未提交 |
| `vllm/experimental/multi_model_worker.py` | 567 | Runner/group registry、RPC、并发编排 | 当前未提交 |
| **合计** | **约 1837 行** | technical prototype | — |

这里的 1837 行是原型总量，不等于最终最小 patch。后续可以收敛：

- 把 Gate 特有的并发 prepare/release 从通用 registry 分离；
- 把 topology planner 移入 Engine/Executor 正式配置；
- 用统一 ResourcePlan 减少 Worker/Adapter 重复校验；
- 将版本绑定 graph monkey patch 替换为上游可注入接口。

### 7.2 vLLM-Omni 配套 patch

commit `4fdf39e15`：88 additions / 13 deletions。

| 文件 | 变更 | 作用 |
|---|---:|---|
| `diffusion_model_runner.py` | +13 | memory provider 与请求峰值 |
| `cache/seacache/backend.py` | +3 | provider 下传 |
| `cache/seacache/hook.py` | +25/−8 | retained cache clone 进入 provider |
| `models/cosmos3/transformer_cosmos3.py` | +47/−5 | UND cache 进入 provider |

这部分是对 Omni 的最小 provider patch，没有复制 Runner/Pipeline，也没有改变 diffusion/VAE 数学逻辑。

### 7.3 不计入 technical 核心的代码

| 类别 | 当前规模 | 说明 |
|---|---:|---|
| `omni_worker_owned_gate.py` | 752 行 | 负载、计时、报告、回归 Gate |
| `omni_worker_tp_topology_gate.py` | 99 行 | TP topology 验证 |
| `tests/experimental/test_multi_model_worker.py` | 242 行 | 单元测试 |
| workload JSON | 数据 | 固定输入与 hash |

这些代码重要，但不应计入部署 runtime 的必要工作量。

### 7.4 当前修改的功能归类

从技术类别看，主要修改仍集中在四类：

```text
1. Runner/Worker 生命周期
   - 一个 Worker 托管多个 Runner
   - RPC 注册、执行、关闭

2. 并行组
   - Engine-derived topology
   - Worker group registry
   - execution-scoped TP override
   - Omni compatibility views

3. Memory
   - tagged pool reuse
   - weights / SeaCache / UND cache provider
   - request peak profile

4. Stream / GC / Graph
   - complementary GC
   - execution stream + event
   - Reasoner graph capture stream patch
```

请求 scheduler、dynamic partition、distributed turnstile 和统一 admission 仍是下一阶段，而不是当前已完成代码。

### 7.5 推荐代码阅读顺序

第一遍先理解控制流：

1. `examples/offline_inference/omni_worker_owned_gate.py::main`
2. `WorkerHostedRunnerClient`
3. `ModelDeploymentSpec` / `CommunicationGroupSpec`
4. `MultiModelWorkerExtension.register_auxiliary_runner`
5. `EmbeddedOmniDiffusionRunner.start_on_worker`
6. `execute_auxiliary_runner` / `prepare` / `collect`
7. `MultiModelGPUWorker.execute_model`

第二遍理解资源：

1. `EmbeddedOmniDiffusionRunner._execution_context`
2. `use_tensor_model_parallel_group`
3. `_acquire_communication_group`
4. `_install_omni_group_views`
5. `_memory_pool_context`
6. Adapter `execute_model`
7. `GreenContextPair`
8. `install_vllm_graph_capture_stream_patch`

第三遍跨仓库阅读 Omni：

1. `DiffusionModelRunner.load_model`
2. `DiffusionModelRunner.execute_model`
3. Cosmos3 Pipeline `__call__`/generation path
4. SeaCache backend/hook
5. Cosmos3 `_retain_und_cache`
6. VAE fast path

建议检索命令：

```bash
cd /workspace/vllm-single-framework

rg -n "class (MultiModel|WorkerHosted|EmbeddedOmni)" vllm/experimental
rg -n "register_auxiliary|execute_auxiliary|close_auxiliary" \
  vllm/experimental/multi_model_worker.py
rg -n "use_tensor_model_parallel_group|_install_omni_group_views" \
  vllm/distributed/parallel_state.py \
  vllm/experimental/omni_runner_adapter.py
rg -n "memory_pool_context|reuse_memory_pool" \
  vllm/experimental/omni_runner_adapter.py \
  vllm/device_allocator/cumem.py
rg -n "GreenContextPair|graph_capture|_capture_stream" \
  vllm/experimental
```

### 7.6 阅读时必须保留的判断

1. 当前已经是“同一 vLLM Worker 内多 Runner”，不再只是主进程里创建两个 Engine；
2. 当前还不是“统一多模型 serving Engine”，因为 Engine scheduler 尚未管理 Diffusion 请求；
3. 保留 Omni Runner 是为了保留其优化和降低模型接入成本，不意味着资源所有权仍归 Omni Worker；
4. memory provider 实现的是共同 allocator 域和分类 pool，不是统一 cache block manager；
5. 独立 communicator 解决命名和生命周期隔离，但不自动解决跨 communicator launch order；
6. 同构 TP=2 已验证，异构 TP/DP/EP 不能从当前结果直接外推；
7. 当前 graph patch、Omni module-global view 都是版本绑定的实验兼容层，正式化需要更稳定的注入接口。
