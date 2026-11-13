# 单框架 Top-down 代码导读 v2：从 `main()` 到多模型资源调度

> 更新日期：2026-11-05  
> 入口脚本：[`examples/offline_inference/omni_worker_owned_gate.py`](../../../examples/offline_inference/omni_worker_owned_gate.py)  
> 适用范围：当前 Worker-owned Runner、Engine execution unit、动态 Green Context 与双 CUDA Graph 变体实现。

本文从实际可运行的 `main()` 出发，沿启动和请求执行的真实调用链向下阅读。vLLM 原生路径只交代职责；本项目新增的模型注册、拓扑、通信组、资源仲裁、Omni provider 和 Graph 变体会展开说明。最后逐项对照目标 `MultiModelEngine` 架构，区分“机制已经存在”和“抽象仍未收敛”。

---

## 1. 一眼看清当前真实结构

```text
omni_worker_owned_gate.py::main()
  |
  | 当前遗留 bootstrap：先由原生 LLM 创建 Reasoner
  | （这是尚未消除的不对称，不是目标架构）
  v
LLM / LLMEngine
  |-- ModelRegistry                          Engine frontend mirror
  |-- TopologyPlanner                         Engine frontend 持有
  |-- WorkerHostedRunnerClient                当前统一提交/注册入口
  |
  v
EngineCore
  |-- ModelRegistry                          Engine authoritative state
  |-- Scheduler
  |    `-- ExecutionSliceScheduler             跨模型 slice 资源变体选择
  `-- Executor                                原生 vLLM Executor
         |
         v
      MultiModelGPUWorker
        |-- MultiModelWorkerExtension          多模型 RPC/生命周期接口
        |-- RunnerFactory
        |    |-- VllmRunnerBackend
        |    |    `-- GPUModelRunner
        |    `-- OmniRunnerBackend
        |         `-- EmbeddedOmniDiffusionRunner
        |              `-- Omni DiffusionModelRunner / Cosmos3 pipeline
        |-- ModelInstanceRegistry             同等持有两类 instance
        |-- CommunicatorRegistry
        |-- MultiModelResourceArbiter          shared/full stream 仲裁
        `-- ModelGraphVariantManager           Reasoner Graph 变体
```

当前已经不是“双 Engine 拼接”：Reasoner 和 Generator 最终都由同一组 vLLM Worker 拥有。两类 Runner 都由 `RunnerFactory` backend 创建并进入同一 `ModelInstanceRegistry`；Engine frontend 与 EngineCore 也分别持有同一部署状态的 ModelRegistry。尚未完成的是最外层 `LLM(...)`/`VllmConfig.model_config` 仍承担第一个 scheduling domain 的 bootstrap，尚未形成接收 `ModelSpec[]` 的独立 `MultiModelEngine` 外壳。

本文出现的 `primary`、`primary_demand` 等名称均指**当前兼容实现中的调度角色/代码字段**，不表示目标架构中 Reasoner 永久高于其他模型。目标状态下，它们应改写为请求需求与资源类，而不是模型主从关系。

---

## 2. 从 `main()` 开始

入口是 [`omni_worker_owned_gate.py:337`](../../../examples/offline_inference/omni_worker_owned_gate.py#L337)。其生命周期可以压缩为：

```text
main()
  1. 解析 CLI 和 matched workload
  2. 组装当前 bootstrap Reasoner 的 vLLM config 与 multi_model_runtime config
  3. 经遗留 LLM 入口创建 EngineCore -> Executor -> Worker
  4. 创建 WorkerHostedRunnerClient
  5. 构造 ModelSpec，注册 Generator
  6. 创建 Green Context/stream，完成 warmup
  7. 通过 submit(model_id, request) 运行两类请求
  8. 收集 execution-unit、stream、memory、Graph 与性能报告
  9. 关闭追加注册的 Runner，再关闭当前 bootstrap Engine
```

### 2.1 配置如何进入 Worker

脚本通过 `worker_cls=MultiModelGPUWorker` 和 `worker_extension_cls=MultiModelWorkerExtension` 启用共置能力；动态资源策略通过 `additional_config.multi_model_runtime.green_context` 下发。

其中：

- `partition_policy=dynamic` 同时启用 shared/full 动态分区和 Engine execution-unit admission；
- `generator_sm_count` 决定 Generator/Reasoner 的互补 SM 分区；
- `reasoner_graph_pool_mode`、`reasoner_graph_buffer_mode` 决定 Graph 变体资源策略；
- `generator_phase_policy=demand_aware` 让 Generator 根据 primary demand 选择 full 或 shared。

这部分目前仍是实验配置结构，还不是正式的 `MultiModelEngineConfig`。

### 2.2 `ModelSpec` 只描述部署，不固定请求任务

脚本构造 [`ModelSpec`](../../../vllm/experimental/model_instance_registry.py#L25)，包含：

- `model_id`、`model_path`、`backend`；
- capabilities/default task；
- Engine 规划后补入的 Worker ranks 和 communication groups；
- backend-specific options，例如 SeaCache。

具体 T2I/T2V 任务仍由请求决定，因此同一已注册 Generator 不需要因为请求类型变化而重新注册。

---

## 3. Engine frontend：目前的模型注册与统一提交入口

### 3.1 `TopologyPlanner` 已经位于 Engine frontend

同步和异步 Engine frontend 分别在：

- [`vllm/v1/engine/llm_engine.py:65`](../../../vllm/v1/engine/llm_engine.py#L65)
- [`vllm/v1/engine/async_llm.py:134`](../../../vllm/v1/engine/async_llm.py#L134)

创建 [`TopologyPlanner`](../../../vllm/experimental/multi_model_topology.py#L54)。`WorkerHostedRunnerClient.register()` 调用 `planner.plan(spec)`，再把显式 `TopologyPlan` 随 RPC 下发 Worker，见 [`multi_model_worker.py:531`](../../../vllm/experimental/multi_model_worker.py#L531)。

当前 gate 只实现同构 TP：Generator placement 必须与所属 vLLM TP ranks 一致。Engine 规划、Worker 物化的方向已正确，但还没有通用 placement solver、PP/DP/异构子集规划。

### 3.2 `submit(model_id, request)` 已统一入口，但还不是 Engine router

[`WorkerHostedRunnerClient.submit()`](../../../vllm/experimental/multi_model_worker.py#L468) 根据 `model_id` 分流：

- primary Reasoner：调用原生 vLLM frontend `generate()`；
- runtime-added Generator：调用 Worker RPC，返回统一 future/handle。

请求 payload 可以不同，控制入口统一。这已经验证了 `UnifiedRequestRouter` 的接口形态，但路由表仍在 client 的 `_runtime_model_specs` 中，primary/runtime-added 仍走两个分支，所以它是 router prototype，不是 Engine 内正式 router。

### 3.3 Reasoner demand 的分段计时

同一入口现在记录：

- `admission_wait_seconds`；
- `model_execution_seconds`；
- `demand_release_seconds`；
- `submit_total_seconds`。

对齐 `delay=0` 的 1350-token 回归中，Reasoner model execution 中位数约 7.507 s，端到端约 7.512 s，与组件化前一致；说明 execution-unit 接入没有造成可测退化。`delay=9.5s` 变慢来自请求进入后段 denoising/VAE overlap，而非重构本身。

---

## 4. EngineCore/Scheduler：Execution slices 已进入调度层

### 4.1 调度入口

[`ExecutionSliceScheduler`](../../../vllm/v1/core/sched/execution_slice_scheduler.py) 管理：

```text
ExecutionSliceDescriptor + model demand
  -> full/shared variant
  -> ScheduledExecutionSlice
```

每个 slice 携带 `sequence_no/tags/metadata`。当前 Cosmos3 暴露：

- preparing；
- 每个 denoising step；
- VAE segments；
- VAE blocks。

其他模型是否存在 demand，决定新 slice 选择 `shared` 还是模型的 full variant。

### 4.2 EngineCore 如何主动推进 Worker

动态模式下，EngineCore 取得 Worker 返回的下一 slice boundary，调用 `ExecutionSliceScheduler.schedule_slice()`，再通过 RPC 将 `ScheduledExecutionSlice` 主动下发 Worker。

旧 Worker 反向请求 Engine 的 socket admission 路径已归档。当前 Omni provider 显式推进一个 slice 后返回下一 boundary，不再需要本地 Server/Client。

### 4.3 当前 Scheduler 的边界

它已经管理 slice 的资源变体决策，但还不是完整 ResourceScheduler：

- 决策规则仍只有“存在其他模型 demand -> shared，否则 full”；
- 不做多模型排队、公平性、优先级、deadline/SLO 或容量 admission；
- 不直接拥有 CUDA stream/event；Worker 根据决策激活实际资源；
- 尚未形成具有公平性和优先级的统一可调度实体队列。

---

## 5. Executor：拓扑运输通道，而非新的多模型调度器

Executor 仍是 vLLM 原生 Executor。当前改造复用它完成：

- 按 TP/PP 创建 Worker；
- `collective_rpc` 把 TopologyPlan、execution-unit endpoint 和模型注册命令广播到 Worker；
- Worker 方法返回 metadata/status。

没有新增第二套 executor，也没有让 runtime-added Runner 自己拉进程。这保证了单框架进程所有权。

仍缺少的是：由 Engine 给 Executor 一次性下发“多个模型的完整 placement plan”，以及 Executor 针对不同 rank 子集进行选择性 RPC。当前 homogeneous TP gate 对所有所属 Worker 广播即可。

---

## 6. Worker：当前完成度最高的一层

### 6.1 所有 startup/runtime model 都进入统一 `ModelInstanceRegistry`

[`MultiModelGPUWorker`](../../../vllm/experimental/multi_model_gpu_worker.py#L131) 在原生 Worker 初始化进入 Runner 构造点前创建 Worker-local [`ModelInstanceRegistry`](../../../vllm/experimental/model_instance_registry.py#L98) 和 `RunnerFactory`。Engine-planned startup spec 通过重写的 `_create_model_runner()` 进入 Factory；后续 runtime spec 通过 `register_model()` 进入同一个 Factory。

因此 Worker 不再先创建裸 `self.model_runner` 再事后包装。`self.model_runner` 只是原生 vLLM 代码所需的兼容引用，真实实例所有权位于 `ModelInstanceRegistry`。

### 6.2 `RunnerFactory`

[`RunnerFactory`](../../../vllm/experimental/runner_factory.py#L149) 提供统一：

```text
create(vllm_spec)       -> VllmRunnerBackend -> GPUModelRunner
create(omni_spec)       -> OmniRunnerBackend -> EmbeddedOmniDiffusionRunner
destroy(instance)       -> 对应 backend 生命周期
destroy(instance)
```

backend 通过 `RunnerBackend` protocol 扩展。目前同时注册 `vllm -> VllmRunnerBackend` 与 `vllm_omni -> OmniRunnerBackend`。两者共享 Factory/Registry 契约；vLLM backend 当前直接返回原生 GPUModelRunner，后续仍可增加显式 adapter 来统一 provider 和销毁接口。

### 6.3 `CommunicatorRegistry`

`CommManager` 根据 Engine 下发的共享 `CommunicationGroupDescription` 在每个 Worker/rank 物化通信组，并记录 owner/refcount。Omni backend 从 Manager acquire，而不是从 Worker 反向读取 Engine 拓扑。

当前只验证同构 TP；Omni module-level group compatibility view 仍由 adapter 临时安装。长期应让模型代码直接使用 runner-scoped communicator provider。

### 6.4 `MultiModelResourceArbiter`

[`MultiModelResourceArbiter`](../../../vllm/experimental/multi_model_arbiter.py#L47) 是当前实际的设备资源安全仲裁器：

- 跟踪 primary demand/epoch；
- 为 Reasoner batch 选择 `reasoner_full` 或 `shared` stream；
- 为 Generator phase 选择 `generator_full` 或 `shared` stream；
- 用 CUDA event 建立切换安全边界；
- full/shared 发生变化时 drain 必需依赖；
- mode 不变时保持同 stream 自然顺序，避免每 phase host synchronize。

它目前只管理一个 primary 与一个 Generator 的 pair，尚不是通用 N-model resource allocator。

### 6.5 `ModelGraphVariantManager`

[`ModelGraphVariantManager`](../../../vllm/experimental/reasoner_graph_variants.py#L25) 统一 Graph variant 的 capture/register/activate。Worker 在 [`compile_or_warm_up_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L403) 接入 vLLM 原生 capture trigger，捕获：

- `shared`；
- `reasoner_full`。

两种变体采用同一种配置的 capture mode，并可共享 graph pool。当前 manager 已具有长期接口雏形，但仍绑定 Reasoner runner 的 manager/entry-set 结构，没有成为任意 Runner 都能声明 variant dimensions 的通用服务。

---

## 7. Omni adapter：最小侵入地接入 Worker 资源

[`EmbeddedOmniDiffusionRunner`](../../../vllm/experimental/omni_runner_adapter.py#L148) 复用 Omni `DiffusionModelRunner` 和 Cosmos3 pipeline，主要做四类适配。

### 7.1 Stream provider

`set_execution_stream()` 设置基础 stream；动态运行时通过 `set_generator_phase_stream_selector()` 从 Worker arbiter 获取 phase lease。phase 结束时记录 completion event 并归还 lease，不在每个 phase 无条件 host synchronize。

### 7.2 Memory provider

`allocation_context()`/`request_context()` 把 Omni 权重、SeaCache、UND cache 和请求期分配接入 vLLM memory-pool context/tag。它解决归属和观测，但尚未形成 KV/Sea/UND 可互换 block 的统一 admission pool。

### 7.3 Communicator provider

`communication_context()` 把 Worker `CommunicatorRegistry` 物化的 TP group 映射为 Omni 所需 view。当前仍兼容 Omni module-level globals；provider API 已加入 Omni 分支，但模型代码还没有完全去全局化。

### 7.4 Phase/execution-unit provider

adapter 包装 Cosmos3 原有 step metadata 与 VAE decode 边界，并把它们报告给 Engine execution-unit scheduler。分段发生在 pipeline 已有安全边界，未复制 solver/cache/kernel 实现。

这仍带有 Cosmos3-specific hook。通用方向是 Omni Runner 暴露标准 execution-unit iterator/provider，各 pipeline 只负责提供最小的 phase/yield boundary。

---

## 8. 一次动态请求的完整运行链

```text
脚本
  -> client.submit(generator_id, OmniRequest)
       -> Worker thread 执行 EmbeddedOmniDiffusionRunner.execute_model()
            -> Cosmos3 到达 preparing/denoising/VAE boundary
            -> adapter 请求 Engine execution-unit admission
            -> Engine Scheduler 根据 primary_demand 返回 mode
            -> Worker arbiter 获取对应 stream lease
            -> Omni 在该 stream 继续原生 phase
            -> completion event + lease.finish()

随后或同时：

脚本
  -> client.submit(reasoner_id, SchedulerModelRequest)
       -> Engine set_primary_demand(True)
       -> 等 Generator 到达可切换边界
       -> vLLM scheduler 产生 LLM batch
       -> MultiModelGPUWorker.execute_model()
       -> arbiter 选择 shared/reasoner_full stream
       -> 激活匹配 CUDA Graph variant
       -> 原生 GPUModelRunner 执行
       -> batch completion event
       -> 请求结束 set_primary_demand(False)
```

关键点是：Engine Scheduler 决定 execution-unit mode，Worker arbiter 保证 CUDA 层安全，Runner adapter 暴露模型内部边界。三者职责不能混为一层。

---

## 9. 与目标架构逐项对照

目标：

```text
MultiModelEngine
  ├─ ModelRegistry
  ├─ TopologyPlanner
  ├─ UnifiedRequestRouter
  └─ ResourceScheduler
       |
  Executor
       |
  Worker
  ├─ RunnerFactory
  ├─ ModelInstanceRegistry
  │    ├─ vLLMRunnerAdapter
  │    └─ OmniRunnerAdapter
  ├─ CommunicatorRegistry
  ├─ ExecutionResourceManager
  │    ├─ stream / Green Context
  │    └─ memory context
  └─ MultiVariantGraphManager
```

| 目标构件 | 当前对应物 | 状态 | 主要缺口 |
|---|---|---:|---|
| `MultiModelEngine` | 原生 LLM/AsyncLLM + client + EngineCore 扩展 | 约 45% | 尚无直接接收 `ModelSpec[]` 的独立 Engine 外壳和多 scheduling-domain lifecycle |
| Engine `ModelRegistry` | frontend mirror + EngineCore authoritative registry | 约 70% | 缺完整部署事务、失败回滚和持久状态机 |
| `TopologyPlanner` | `multi_model_topology.py` | 约 65% | 仅同构 TP gate；缺 placement solver、rank 子集、PP/DP/异构拓扑 |
| `UnifiedRequestRouter` | `submit(model_id, request)` | 约 60% | 仍是 frontend client 分支，不是 Engine 原生 router/queue |
| `ResourceScheduler` | `ExecutionSliceScheduler` + Worker arbiter | 约 55% | 策略与 CUDA 安全机制分散；缺 N-model、公平性、SLO、统一队列 |
| Executor 接入 | 原生 Executor + collective RPC | 约 55% | 缺按 placement rank 子集部署/调用，多模型 plan 不是一等配置 |
| `RunnerFactory` | `runner_factory.py` | 约 85% | vLLM/Omni 均经 Factory；销毁/provider 契约仍需进一步统一 |
| `ModelInstanceRegistry` | `model_instance_registry.py` | 约 85% | Worker 内已统一；Engine registry 尚未对应 |
| `vLLMRunnerAdapter` | `VllmRunnerBackend` + 原生 GPUModelRunner | 约 55% | 创建已统一；尚无显式 adapter/resource-provider 契约 |
| `OmniRunnerAdapter` | `EmbeddedOmniDiffusionRunner` | 约 75% | Cosmos3 hooks、module-level group compatibility 尚待通用化 |
| `CommunicatorRegistry` | `communicator_registry.py` | 约 70% | 仅验证同构 TP；缺多维 group 与跨 communicator ordering protocol |
| `ExecutionResourceManager` | GreenContext + arbiter + adapter memory context | 约 55% | 还不是统一对象；stream/memory admission 分散，memory 主要是归属而非容量调度 |
| `MultiVariantGraphManager` | `ModelGraphVariantManager` | 约 70% | 当前偏 Reasoner/stream variant；缺任意模型、shape/batch/partition 统一 key 和预算策略 |

这些百分比表示“距离目标抽象的工程完成度”，不是代码正确率。

---

## 10. 到目标架构还有多远

从底层可行性看，最难的机制已经跑通：

1. 同一 Worker 多 Runner；
2. Engine 规划拓扑、Worker 创建通信组；
3. shared/full Green Context 与 stream 安全切换；
4. 双 CUDA Graph 变体；
5. Omni memory/stream/communicator/phase provider；
6. Engine Scheduler 原生记录 diffusion execution units；
7. 同步与异步统一提交入口；
8. 正确性、TP gate、性能与压力测试。

因此距离目标不是“还缺一次底层突破”，而是“两轮架构收敛 + 一轮通用化”。

### 第一轮：Engine 控制面收敛

- 建立 Engine-owned `ModelRegistry`；
- 让 Engine 外壳直接接收 `ModelSpec[]`，移除单一 `model_config` 的入口语义；
- 把 client 路由迁入 `UnifiedRequestRouter`；
- 注册、加载、ready、failed、remove 成为 Engine 生命周期；
- TopologyPlan 作为模型部署状态保存，而非一次 RPC 临时参数。

完成后才能真正称为 `MultiModelEngine`。

### 第二轮：Worker 资源组件收敛

- 把 Green Context、stream、event、memory context 收进 `ExecutionResourceManager`；
- 让 ResourceScheduler 只做策略，ResourceManager 只做设备机制；
- 给 vLLM runner 增加显式 adapter，与 Omni adapter 使用同一 provider 接口；
- 把 Graph manager 从 Reasoner-specific key 扩展为 model/shape/batch/partition variant key。

### 第三轮：从 Cosmos3 gate 走向通用 serving

- Omni 标准 execution-unit provider，pipeline 只提供最小 yield hooks；
- 多 auxiliary model、rank 子集和异构拓扑；
- TP/DP/PP/CFG 等多通信域及跨 communicator ordering；
- N-model admission、公平性、优先级、SLO 与取消；
- 模型输出 artifact/descriptor，避免大 tensor 进入控制面；
- 统一 memory capacity admission，而不仅是 allocator ownership/tagging。

若目标只限定为“单机、同构 TP、一个 LLM + 一个 Cosmos3 Generator 的研究原型”，当前约完成 **75%–80%**；若目标是“通用生产级 MultiModelEngine”，约完成 **55%–60%**。差距主要集中在 Engine 控制面、N-model 策略和通用 backend 契约，而不是已经验证过的 CUDA 共置机制。

---

## 11. 推荐的后续代码顺序

1. 先定义 `MultiModelEngine`/Engine `ModelRegistry` 的接口与状态机，不移动底层代码；
2. 将 `WorkerHostedRunnerClient.register/submit` 变成该 Engine API 的薄封装；
3. 再引入显式 `vLLMRunnerAdapter`，统一两类 Runner 生命周期和 provider 契约；
4. 抽取 `ExecutionResourceManager`，让 arbiter 仅保留策略状态；
5. 最后扩展 topology/resource scheduler 到多模型和 rank 子集。

这样可以保留当前已通过的性能基线，并让每次重构都能继续用现有 warm/repeat、TP gate 和 execution-unit stress test 做回归。
