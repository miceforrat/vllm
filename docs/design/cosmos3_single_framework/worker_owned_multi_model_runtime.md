# Worker-owned 多模型运行时：当前实现、资源边界与实验结论

## 1. 文档范围

本文记录当前“把 Reasoner 与 Generator 放进同一套 vLLM Worker 资源域”的实现。它回答四个问题：

1. Runner 如何进入 Worker，而不再由外部脚本各自持有两个 Engine；
2. 并行组、CUDA stream、Green Context（GC）和 CUDA Graph 目前归谁管理；
3. memory context 到底统一到了哪一层；
4. 当前实验说明了什么，以及同构 TP 应如何继续推进。

这里的“单框架”不是重写 Omni 的模型实现。Generator 仍复用 Omni 的原生 `DiffusionModelRunner`、pipeline、loader、SeaCache 与 VAE fast path；变化的是：其生命周期和物理资源入口被挂到 vLLM GPU Worker 下。

## 2. 当前层次结构

```text
LLMEngine / Scheduler
        |
Executor (当前 Gate 为 UniProc)
        |
MultiModelGPUWorker                  每张 GPU 的物理资源所有者
        |-- primary ModelRunner      vLLM Reasoner
        |-- MultiModelWorkerExtension
        |       `-- EmbeddedOmniDiffusionRunner
        |               `-- Omni DiffusionModelRunner
        |                       `-- Cosmos3 pipeline / model
        |
        `-- GC pair / streams / graph-capture patch / device allocator
```

关键变化是两个模型现在落在同一个 Worker、PID、CUDA primary context 和设备 allocator 域中。上层仍通过 `LLMEngine.collective_rpc()` 调用每个 Worker，而不是绕过 Executor 直接操作另一个 Engine。

## 3. Worker 如何托管第二个 Runner

### 3.1 声明式部署描述

`ModelDeploymentSpec` 描述辅助模型放在哪些 rank、采用哪个 backend、执行什么任务：

```python
@dataclass(frozen=True)
class ModelDeploymentSpec:
    model_id: str
    model_path: str
    backend: str
    task: str
    worker_ranks: tuple[int, ...] = (0,)
    load_format: str = "default"
    backend_options: Mapping[str, Any] = field(default_factory=dict)
```

目前 `worker_ranks` 已进入接口，但执行结果聚合仍只支持单 Worker；它是后续同构 TP 的落点，不代表 TP 已完成。

### 3.2 在已初始化 Worker 内加载 Omni Runner

`MultiModelWorkerExtension.register_auxiliary_runner()` 在目标 Worker 内解析 Omni stage，并调用：

```python
adapter = EmbeddedOmniDiffusionRunner(
    resolved,
    device=str(self.device),
    memory_pool_tag_prefix=spec.model_id,
)
runner = adapter.start_on_worker(self, load_format=spec.load_format)
```

`start_on_worker()` 不再创建第二套设备进程或第二套 vLLM Engine。它复用当前 Worker 已有的：

- device 与 CUDA primary context；
- vLLM world/TP/PP group；
- vLLM device allocator；
- Worker 生命周期和 RPC 通道。

Omni 的 `DiffusionModelRunner` 与 pipeline 本身保留，避免重新实现其 loader、regional compile、SeaCache、VAE fast path 等优化。

### 3.3 调用路径

前端通过 `WorkerHostedRunnerClient` 发 collective RPC：

```python
results = llm_engine.collective_rpc(
    "execute_auxiliary_runner",
    args=(model_id, request, profile_memory),
)
```

因此当前执行路径是：

```text
client -> LLMEngine -> Executor -> Worker RPC
       -> auxiliary runner slot -> Omni DiffusionModelRunner
```

这与早期“脚本同时持有 vLLM Engine 和 Omni Engine”不同：现在 Worker 才是两个 Runner 的共同设备所有者。

## 4. 资源归属

### 4.1 并行组

Worker-hosted adapter 读取 vLLM 已初始化的 group：

```python
self._install_omni_group_views(
    world=get_world_group(),
    tp=get_tp_group(),
    pp=get_pp_group(),
)
```

随后把它们安装为 Omni parallel-state 的兼容视图。当前 Gate 是 world size 1，因此只证明了“没有创建冲突的第二套 group”。尚未证明多卡 collective、输出聚合和 shutdown 顺序。

同构并行首先采用 Reasoner TP=2、Generator TP=2：两者共用相同两张卡和相同 rank placement，但使用独立 TP communication domain。异构 rank 集合留到同构链路稳定以后。

### 4.2 GC 与 stream

`MultiModelGPUWorker.init_device()` 创建互补 GC：

```python
pair = GreenContextPair(requested_generator_sm_count, device_index)
generator_stream = torch.cuda.ExternalStream(pair.first_stream_ptr)
reasoner_stream = torch.cuda.ExternalStream(pair.second_stream_ptr)
```

当前约定是 Generator 获得主分区，Reasoner 获得补集。Worker 将 Reasoner 的模型加载、KV 初始化、warmup、执行和 sampling 包在 Reasoner stream 中；注册 Generator 后，adapter 自动绑定 `_multi_model_generator_stream`。

这使 stream 选择不再只存在于 benchmark 脚本，而成为 Worker 的执行契约。

### 4.3 Reasoner CUDA Graph

vLLM graph capture 必须发生在 Reasoner 对应的 GC stream 上。Worker 初始化 GC 后安装 capture-stream patch：

```python
self._multi_model_graph_patch = (
    install_vllm_graph_capture_stream_patch(reasoner_stream)
)
```

`multi_model_resource_plan()` 会报告 capture stream 和调用计数。当前报告已经验证：Reasoner execution stream 与 graph capture stream handle 相同，且 capture 入口实际被调用。

Generator 当前没有额外强制 CUDA Graph；它保留 Omni 自己支持的 diffusion compile、SeaCache 与 VAE fast path。

## 5. Memory context：已统一什么，尚未统一什么

### 5.1 简短结论

**目前完成了统一的 Worker 所有权、vLLM allocator 入口和带标签的 allocation context；尚未完成统一显存预算器或可动态互借的统一 block pool。**

因此可以说“Generator 的关键持久分配接入 vLLM memory context”，不能说“Reasoner 和 Generator 的全部显存已经由一个全局内存管理器统一调度”。

### 5.2 已完成的部分

Worker 内的 adapter 获取 vLLM allocator：

```python
self.memory_allocator = get_mem_allocator_instance()
memory_pool_context_fn = self._memory_pool_context
self.runner.load_model(
    memory_pool_context_fn=memory_pool_context_fn,
    load_format=load_format,
)
```

每类持久对象使用带模型前缀的 tag；第一次创建 pool，之后显式复用：

```python
pool_tag = f"{model_id}:{tag}"
if pool_tag in self._memory_pool_tags:
    context = allocator.reuse_memory_pool(pool_tag)
else:
    context = allocator.use_memory_pool(tag=pool_tag)
```

目前 hook 覆盖：

| 对象 | context/tag | 状态 |
|---|---|---|
| Generator 权重与持久 buffer | `weights` | 已接入 vLLM allocator context |
| SeaCache 持久 tensor | `cache` | 已接入同一 context 入口 |
| Cosmos3 UND K/V 等请求内缓存 | `cache` | 已接入同一 context 入口 |
| 请求前后 allocator 状态 | allocated/reserved/peak/tagged growth | 可观测 |

Omni 侧为此保留了一个很小的资源注入接口：`DiffusionModelRunner.load_model(memory_pool_context_fn=...)`，并把该函数下传给 pipeline、transformer 和 SeaCache hook。业务语义仍由 Omni/Cosmos3 维护，vLLM 只提供 allocation context。

### 5.3 尚未完成的部分

| 对象 | 当前实际管理者 | 尚缺能力 |
|---|---|---|
| Reasoner Paged KV | vLLM KV cache manager | 未与 diffusion cache 共享 block 池 |
| SeaCache / UND cache | Omni/Cosmos3 语义，vLLM allocator context 承载分配 | 不能与 Paged KV 动态交换容量 |
| diffusion/VAE activation | pipeline + PyTorch caching allocator | 只有 peak/headroom 观测，没有请求级预留与 admission |
| CUDA Graph private pool | graph/backend | 未纳入统一可回收预算 |
| compile artifact / workspace | backend/compiler | 生命周期仍由各 backend 管理 |
| 多模型总预算 | 无统一仲裁器 | 尚不能按优先级压缩、驱逐或拒绝请求 |

几个 pool 最终可能都落在同一 CUDA primary context 和 PyTorch caching allocator 上，但这不等于容量可互借：tagged pool 是所有权和生命周期边界，不是 vLLM Paged-KV 那样的统一 block allocator。

### 5.4 当前 profiling 能力

`execute_model_profiled()` 记录：

- 请求前 allocated/reserved；
- 请求期 peak allocated/reserved；
- transient headroom；
- 请求结束后的 live delta；
- tagged pool growth。

这些数据足以做容量规划和未来 admission control 的输入，但当前尚未驱动调度决策。

## 6. Worker Gate 与当前性能结果

代表性负载由 `omni_worker_owned_gate.py` 驱动，Reasoner 使用 graph，Generator 使用匹配的 T2V workload，并保留 SeaCache/Omni 优化。

证据文件：

- `outputs/omni_worker_owned_gate/full_sm_concurrent_quick.json`
- `outputs/omni_worker_owned_gate/gc_graph_g136_with_reference.json`

### 6.1 Full-SM、不分区并发

| 指标 | 时间 |
|---|---:|
| Reasoner alone | 4.360 s |
| Generator alone | 17.499 s |
| Full-SM 串行 wall | 21.861 s |
| Reasoner concurrent | 21.012 s |
| Generator concurrent | 18.469 s |
| concurrent wall | 21.016 s |
| 相对 Full-SM 串行吞吐 | 1.040x |

不做空间隔离时，短 Reasoner 被 Generator kernel 链严重压制，slowdown 为 4.819x；仅有 host overlap 不等于有效 GPU 共置。

### 6.2 GC 136:34

| 指标 | 时间/倍率 |
|---|---:|
| Reasoner partitioned alone | 6.587 s |
| Generator partitioned alone | 21.641 s |
| partitioned serial wall | 28.229 s |
| Reasoner colocated | 7.624 s |
| Generator colocated | 22.187 s |
| colocated wall | 22.188 s |
| Reasoner 共置 slowdown | 1.158x |
| Generator 共置 slowdown | 1.025x |
| 相对 partitioned serial 吞吐 | 1.272x |
| 相对 Full-SM serial 吞吐 | 0.985x |

GC 已把互相干扰压回可控范围，表现与此前双框架实验接近；但静态分区本身使 Reasoner 和 Generator 分别膨胀 1.511x、1.237x，所以相对真正 Full-SM 串行基线还没有净收益。

此外 Reasoner 只覆盖 Generator 前约 7.6 s，而 Generator 总长约 22.2 s。Reasoner 结束后，34 SM 仍被静态保留，这是下一阶段 yield/阶段边界重分区的直接动机。

## 6.1 Engine 规划拓扑，Worker 拥有通信组

当前 `WorkerHostedRunnerClient` 从所属 `llm_engine.vllm_config.parallel_config` 派生同构 TP placement，并补全 `ModelDeploymentSpec` 中的 `CommunicationGroupSpec`：

```python
CommunicationGroupSpec(
    group_id="generator-a-tp",
    parallel_dimension="tp",
    ranks=(0, 1),
)
```

Worker 侧维护 communication-group registry。它根据 group ID 创建或复用 `GroupCoordinator`，记录使用该通信域的 Runner owner，并在最后一个 owner 释放后销毁 group。`EmbeddedOmniDiffusionRunner` 不再调用 `init_model_parallel_group()`，只接收 Worker 提供的 group handle，并在执行上下文中通过 `use_tensor_model_parallel_group()` 消费它。

因此当前 ownership 为：

```text
Engine / Executor: placement、并行维度、rank topology
Worker:            GroupCoordinator、device communicator、引用计数与销毁
Runner adapter:    绑定并使用通信组，不拥有其生命周期
```

当前仍限制为同构 TP：Generator placement 必须与默认 vLLM TP ranks 完全一致。这个限制避免在尚未实现全 rank 一致的 subgroup partition 创建协议前，错误支持异构或交叉 rank groups。

## 7. 当前限制

1. Worker 托管、TP=2 输出 ownership、独立 auxiliary communicator 和并发 collective 已完成验证；更大的 TP degree 尚未验证。
2. Omni parallel state 仍通过兼容视图接入 Worker-owned communication domain；PP/SP/CFG/DP/HSDP/EP 尚未接入。
3. GC 是启动时静态分区，尚无阶段边界 yield、resize 或 graph variant 切换。
4. 没有统一的跨模型 admission control 和显存预算仲裁。
5. 辅助 Runner 目前只支持 `vllm_omni` backend，但 registry/spec 接口已为其他 backend 留出位置。

## 8. 下一步：同构 TP 优先

建议按以下三个 Gate 推进：

### P1：拓扑验证

- 启动 vLLM TP=2；
- 每个 Worker 输出 global/local rank、device、world/TP/PP group identity；
- 仅注册辅助 Runner，不执行模型；
- 验证两个 rank 都进入同一个 `worker_ranks=(0, 1)` deployment。

### P2：同构双 Runner

- Reasoner TP=2；
- Generator 复用相同 rank placement，但由 Worker 创建独立 TP communication domain；
- 删除 `world_size == 1` 和单结果返回假设；
- 定义 Generator 输出的 rank ownership 与 gather 规则；
- 验证串行正确性和 shutdown。

### P3：每卡互补 GC

- 每个 Worker 在本地 GPU 创建同样比例的互补 GC；
- Reasoner graph 在各自本地 Reasoner GC stream capture；
- Generator 在各自本地 Generator GC stream 执行；
- 验证 collective、stream 和 GC 不产生跨 rank 等待环；
- 最后再做并发与吞吐测量。

在 P3 稳定前，不引入 A 用 4 卡、B/C 各用 2 卡的异构 rank 集合，也不同时引入动态 yield，避免把 communicator、调度和资源迁移问题混在一次实验中。

## 9. 代码索引

- `vllm/experimental/multi_model_worker.py`：deployment spec、Worker runner registry、RPC client。
- `vllm/experimental/multi_model_gpu_worker.py`：GC 创建、Reasoner stream 绑定、graph capture patch、资源报告。
- `vllm/experimental/omni_runner_adapter.py`：保留 Omni Runner，注入 vLLM group、memory context 与 execution stream。
- `vllm/experimental/green_context.py`：互补 GC 的底层封装。
- `vllm/experimental/runtime_stream_patch.py`：vLLM Reasoner graph capture stream 适配。
- `examples/offline_inference/omni_worker_owned_gate.py`：Worker-owned 串行、alone、并发和 Full-SM reference Gate。

