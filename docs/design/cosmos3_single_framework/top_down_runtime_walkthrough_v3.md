# 单框架 Top-down 代码导读 v3：统一模型创建后的真实运行链

> 更新日期：2026-11-05  
> 主入口：[`examples/offline_inference/omni_worker_owned_gate.py`](../../../examples/offline_inference/omni_worker_owned_gate.py)  
> 阅读目标：从 `main()` 沿真实调用链理解 Engine 模型注册、Worker Runner 创建、请求路由、动态分区、CUDA Graph 变体和 Omni execution unit。

本文只描述当前代码实际做了什么。为了兼容 vLLM，代码中仍有 `primary`、`auxiliary`、`self.model_runner` 等旧命名；它们不是目标架构中的模型主从关系。当前真正的模型身份由 `model_id + ModelSpec` 决定。

---

## 1. 当前结构与剩余兼容壳

```text
omni_worker_owned_gate.py::main()
  |
  | 兼容 bootstrap：LLM/VllmConfig 仍提供第一个模型的配置
  v
LLMEngine / AsyncLLM                         Engine frontend
  |-- ModelRegistry                         frontend mirror
  |-- TopologyPlanner
  `-- WorkerHostedRunnerClient              当前统一 submit/register API
        |
        v
EngineCore
  |-- ModelRegistry                         authoritative deployment state
  |-- Scheduler
  |-- ExecutionSliceScheduler                diffusion unit admission/state
  `-- Executor
        |
        v
MultiModelGPUWorker
  |-- RunnerFactory
  |    |-- VllmRunnerBackend -> GPUModelRunner
  |    `-- OmniRunnerBackend -> EmbeddedOmniDiffusionRunner
  |-- ModelInstanceRegistry                 同等持有两类 ModelInstance
  |-- CommunicatorRegistry
  |-- MultiModelResourceArbiter
  `-- ModelGraphVariantManager
```

已经统一的是：Engine 以 `ModelSpec` 记录模型；Worker 上两类 Runner 都由 `RunnerFactory` 创建并进入同一个 `ModelInstanceRegistry`。

还未统一的是：最外层仍通过单模型语义的 `LLM/VllmConfig.model_config` 引导第一个 scheduling domain；`submit()` 仍根据 backend 走 vLLM frontend 或 Worker RPC 两条执行通道。

---

## 2. 从 `main()` 开始读

入口：[`omni_worker_owned_gate.py:337`](../../../examples/offline_inference/omni_worker_owned_gate.py#L337)。

按代码顺序，`main()` 做六件事：

1. 解析 workload、Graph、Green Context、重复次数等参数；
2. 在 `additional_config["multi_model_runtime"]` 写入 Worker 类和资源策略；
3. 通过兼容 `LLM(...)` 入口启动 Engine、Executor 和 Worker；
4. 创建 `WorkerHostedRunnerClient`，注册 Cosmos3 Generator 的 `ModelSpec`；
5. 使用统一 `submit(model_id, request)` 跑 Reasoner/Generator alone 与 concurrent；
6. 汇总 Worker registry、execution unit、stream/Graph/memory 和时间报告。

建议先阅读以下几个局部：

- 参数与配置组装：[`omni_worker_owned_gate.py:337`](../../../examples/offline_inference/omni_worker_owned_gate.py#L337)
- Engine/Client 创建和 Generator 注册：在同一 `main()` 中搜索 `WorkerHostedRunnerClient` 与 `client.register`
- 请求执行：搜索 `_run_reasoner`、`_run_generator` 和 `_run_concurrent_once`
- 报告组装：搜索 `scheduler_timing`、`stream_control` 与 `full_sm_concurrency`

### 2.1 `ModelSpec` 不是请求 task

[`ModelSpec`](../../../vllm/experimental/model_instance_registry.py#L34) 描述部署意图：

```text
model_id / model_path / backend
capabilities / default_task
worker_ranks / communication_groups
load_format / backend_options
```

具体 T2I、T2V 或文本生成参数仍在 request 中。`default_task` 只是兼容默认值，不要求同一模型实例只能处理一种请求类型。

---

## 3. 启动链：Engine 如何规划并下发第一个模型

### 3.1 Engine frontend 创建 `ModelRegistry`

同步和异步 frontend 都创建 `TopologyPlanner` 与 Engine-owned registry：

- [`llm_engine.py`](../../../vllm/v1/engine/llm_engine.py)
- [`async_llm.py`](../../../vllm/v1/engine/async_llm.py)
- [`create_engine_model_registry()`](../../../vllm/experimental/model_instance_registry.py#L209)

`create_engine_model_registry()` 将兼容 `vllm_config.model_config` 转为普通 `ModelSpec(backend="vllm")`，调用 `TopologyPlanner.plan()`，随后：

1. 注册到 frontend `ModelRegistry`；
2. 将规划后的 payload 写入 `multi_model_runtime.startup_models`；
3. 随 `VllmConfig` 传给 EngineCore 和每个 Worker。

`default_model_id` 只是未显式指定模型时的兼容路由默认值，不赋予该模型更高的资源或生命周期地位。

### 3.2 EngineCore 恢复权威 registry

[`restore_engine_model_registry()`](../../../vllm/experimental/model_instance_registry.py#L237) 从 `startup_models` 恢复 EngineCore 的 registry。EngineCore 通过：

- [`register_model_spec()`](../../../vllm/v1/engine/core.py#L1074)
- `remove_model_spec()`
- [`model_registry_state()`](../../../vllm/v1/engine/core.py#L1086)

维护权威部署状态。frontend registry 是本地镜像；运行时注册会同时更新 Worker、EngineCore 和 frontend mirror。

当前注册顺序还不是事务：Worker 创建成功后若 EngineCore/frontend 更新失败，尚缺完整 rollback。这是后续 Engine deployment state machine 的工作。

### 3.3 Executor 只运输计划

Executor 仍使用 vLLM 原生进程模型和 `collective_rpc`。它负责把 startup config、运行时注册、execution-unit endpoint 等送到 Worker，但不自行决定模型 placement。

[`TopologyPlanner`](../../../vllm/experimental/multi_model_topology.py#L54) 属于 Engine 控制面；[`CommunicatorRegistry`](../../../vllm/experimental/communicator_registry.py#L19) 属于 Worker 物化层。这样避免 Worker 反向读取 Engine 来规划拓扑。

---

## 4. Worker 启动：两个 backend 走同一个 Factory

### 4.1 原生 Worker 提供可覆写创建点

原生 [`GPUWorker._create_model_runner()`](../../../vllm/v1/worker/gpu_worker.py#L515) 保留默认 GPUModelRunner 创建逻辑。`GPUWorker.init_device()` 不再内联构造 Runner，而是调用这个 hook。

### 4.2 `MultiModelGPUWorker` 在父类初始化前创建组件

[`MultiModelGPUWorker.__init__()`](../../../vllm/experimental/multi_model_gpu_worker.py#L137) 先建立：

```text
CommunicatorRegistry
ModelInstanceRegistry
RunnerFactory
```

随后才进入 `GPUWorker.__init__()`。原因是父类初始化期间会调用可覆写的 `_create_model_runner()`，Factory 和 registry 必须提前存在。

### 4.3 startup vLLM Runner 也经过 Factory

[`MultiModelGPUWorker._create_model_runner()`](../../../vllm/experimental/multi_model_gpu_worker.py#L194) 读取 Engine 规划的 `startup_models`，然后：

```python
spec = ModelSpec.from_rpc_payload(startup[0])
instance = self._vllm_runner_factory.create(spec)
self._vllm_model_instance_registry.register(instance)
return instance.runner
```

因此当前不是“Worker 原生创建 Reasoner，再给 Registry 补一个壳”。`VllmRunnerBackend` 真正负责创建 GPUModelRunner，返回的 `ModelInstance` 真正进入统一 registry。

返回 `instance.runner` 是因为大量原生 vLLM 代码仍访问 `self.model_runner`。它是兼容引用，不是另一份 Runner，也不是独立所有权容器。

### 4.4 `RunnerFactory` 的两个 backend

[`runner_factory.py`](../../../vllm/experimental/runner_factory.py) 中：

- [`VllmRunnerBackend`](../../../vllm/experimental/runner_factory.py#L35) 创建 GPUModelRunner；
- [`OmniRunnerBackend`](../../../vllm/experimental/runner_factory.py#L74) 创建 `EmbeddedOmniDiffusionRunner`，获取通信组并加载 Omni pipeline；
- [`RunnerFactory`](../../../vllm/experimental/runner_factory.py#L188) 按 `spec.backend` 选择 backend。

两者都返回相同的 `ModelInstance`，并由相同的 `RunnerFactory.destroy()` 分发到对应 backend 销毁。区别是执行域：vLLM instance 当前 `scheduler_managed=True`；Omni instance 当前在 Worker thread 中执行，并通过 execution-unit provider 与 Engine Scheduler 协作。

### 4.5 两级 registry 的职责

[`ModelRegistry`](../../../vllm/experimental/model_instance_registry.py#L154)：Engine 级部署意图，不含 CUDA 对象。

[`ModelInstanceRegistry`](../../../vllm/experimental/model_instance_registry.py#L98)：每个 Worker/rank 上的实际对象，包含 Runner、adapter、通信组引用和 pending job。

```text
Engine ModelRegistry:       model-a -> ModelSpec
Worker rank 0 registry:     model-a -> rank-0 ModelInstance
Worker rank 1 registry:     model-a -> rank-1 ModelInstance
```

---

## 5. 运行时追加注册链

入口是 [`WorkerHostedRunnerClient.register()`](../../../vllm/experimental/multi_model_worker.py#L503)。真实顺序是：

```text
ModelSpec
  -> Engine TopologyPlanner.plan(spec)
  -> RPC register_model(spec, topology_plan)
  -> 每个目标 Worker 的 RunnerFactory.create(planned_spec)
  -> Worker ModelInstanceRegistry.register(instance)
  -> EngineCore register_model_spec(planned_spec)
  -> frontend ModelRegistry.register(planned_spec)
```

Omni backend 在创建时将 Engine 下发的 `CommunicationGroupDescription` 物化成 runner-owned TP group。当前 homogeneous TP gate 要求 Generator placement 与 vLLM TP ranks 相同；这不是最终拓扑模型的理论限制。

---

## 6. 统一请求入口及其尚存分支

同步入口：[`WorkerHostedRunnerClient.submit()`](../../../vllm/experimental/multi_model_worker.py#L447)。  
异步入口：[`AsyncWorkerHostedRunnerClient.submit()`](../../../vllm/experimental/multi_model_worker.py#L768)。

调用者只提交：

```python
submit(model_id, request)
```

入口先从 Engine `ModelRegistry` 读取 spec，再根据 backend 路由：

- `backend == "vllm"`：调用 frontend `generate()`，由原生 Scheduler 驱动；
- 其他 backend：调用 `start_async()`，由 Worker-hosted Runner 执行。

因此 API 已统一，执行通道尚未统一。真正的 `UnifiedRequestRouter` 还应位于 Engine 内部，把 backend adapter、request codec、队列和 scheduling domain 作为模型注册信息，而不是在 client 中写 `if spec.backend != "vllm"`。

### 6.1 Reasoner demand 与时间字段

动态策略下，请求执行前后调用 `set_model_demand(model_id, True/False)`，以模型 ID 记录当前资源需求。

报告中的：

- `admission_wait_seconds`：等待 Generator 安全边界；
- `model_execution_seconds`：frontend generate 的实际时间；
- `demand_release_seconds`：释放需求状态；
- `submit_total_seconds`：以上总和。

注意动态策略的 `alone` 默认会取得相应 full-SM lane。`concurrent / alone` 会同时包含“分区限制”和“共置干扰”，不能直接解释成纯 colocation slowdown。纯干扰必须使用相同 SM 分区的 partitioned-alone 基线。

---

## 7. Engine Scheduler 如何调度 execution slice

[`ExecutionSliceScheduler`](../../../vllm/v1/core/sched/execution_slice_scheduler.py) 接收 EngineCore 已经取得的 slice boundary，根据当前各模型 demand 选择资源变体，并分配全局 execution epoch。

```text
ExecutionSliceDescriptor + model demand
  -> resource variant (full/shared)
  -> ScheduledExecutionSlice
  -> Worker
```

vLLM 的原生 scheduler batch 和 Omni provider 产生的细粒度 slice 都进入这一入口。Cosmos3 当前产生：

```text
preparing
denoising step 0 ... N-1
vae segment 0 ... M-1
VAE block 0 ... M-1
```

旧 Worker 通过 Unix socket 反向请求调度的 `ExecutionSliceAdmissionServer/Client` 已归档至 `vllm/experimental/archived/execution_slice_admission.py`。当前路径由 EngineCore 主动逐 slice 下发，不再存在反向 admission server。

它目前解决的是“Engine 看得见并决定每个 slice 的资源变体”，还不是完整 N-model ResourceScheduler：没有公平性、deadline/SLO、跨模型优先级或统一 capacity admission。

---

## 8. Omni adapter 如何暴露 phase 而不拆掉 Runner

[`EmbeddedOmniDiffusionRunner`](../../../vllm/experimental/omni_runner_adapter.py#L148) 保留一次 [`execute_model()`](../../../vllm/experimental/omni_runner_adapter.py#L799)，没有把每个 denoising step 变成外部 RPC。

adapter 在 Cosmos3 已有安全边界安装 hook：

- step metadata：每个 denoising step；
- `_decode_latents`/decoder hook：VAE segments；
- 请求首尾：preparing/complete。

每个边界向 execution-unit provider 请求 mode，再从 Worker arbiter 取得 stream lease。这样保留 Omni 原有 solver、cache 和优化 kernel，只把“何时可以 yield/切换”暴露给 Engine。

Omni 分支已经提供 phase/stream/memory/communicator provider；当前 adapter 仍有 Cosmos3-specific hook，长期应由 Omni Runner 定义标准 execution-unit iterator，具体 pipeline 只提供最小 boundary patch。

---

## 9. Worker 设备资源路径

### 9.1 Stream 与 Green Context

[`MultiModelResourceArbiter`](../../../vllm/experimental/multi_model_arbiter.py#L47) 持有四条资源 lane：

```text
shared_primary_stream       shared_generator_stream
full_primary_stream         full_generator_stream
```

它根据 demand 和 Generator phase：

- 选择 full/shared lane；
- 在切换边界记录或等待 CUDA event；
- mode 不变时依赖同一 stream 的自然顺序，不做每 phase host synchronize；
- 防止 full-SM phase 与另一模型新提交发生非法重叠。

Engine execution-unit scheduler 决定 mode；Worker arbiter 保证 CUDA 层安全。两者不能合并成同一个职责。

### 9.2 Memory context

Omni adapter 的 allocation/request context 将权重、SeaCache、UND cache 和请求期分配纳入 vLLM allocator context/tag。当前实现解决“归谁、如何复用、如何观测”，没有把 KV/Sea/UND 变成可互借 block 的统一显存池，也没有 Engine 级显存 admission。

### 9.3 Communicator

Engine 规划共享的 `CommunicationGroupDescription`，Worker `CommManager` 创建并按 owner/refcount 管理，Omni adapter 安装兼容 group view。长期接口应是 model/runner-scoped communicator provider，彻底避免 Omni module-level group globals。

---

## 10. 多 CUDA Graph 变体

[`ModelGraphVariantManager`](../../../vllm/experimental/reasoner_graph_variants.py#L25) 统一保存、捕获和激活：

```text
shared
reasoner_full
```

Worker 在 [`compile_or_warm_up_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L437) 接入原生 vLLM capture 生命周期；执行 batch 时在 [`execute_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L891) 根据 arbiter 选择对应 variant。

当前两种 variant 使用等价 capture 接口和相同 capture mode，可配置共享 Graph pool。manager 仍主要适配 vLLM model runner 的 entry sets；最终应使用通用 key：

```text
(model_id, shape/batch class, partition mode, compile mode)
```

并具备显存预算、淘汰和按需捕获策略。

---

## 11. 一次 concurrent 请求的完整时序

```text
main()
  -> client.submit(generator_id, request)
  -> start_async RPC
  -> Worker ModelInstanceRegistry.get(generator_id)
  -> EmbeddedOmniDiffusionRunner.execute_model()
  -> preparing/denoising/VAE boundary
  -> Engine ExecutionSliceScheduler chooses generator_full/shared
  -> Worker arbiter grants stream lease
  -> phase kernels enqueue
  -> completion event + lease.finish()

同时：

main()
  -> client.submit(reasoner_id, SchedulerModelRequest)
  -> set demand
  -> wait only if a conflicting full Generator unit is active
  -> frontend generate -> vLLM Scheduler batch
  -> MultiModelGPUWorker.execute_model()
  -> arbiter chooses reasoner_full/shared
  -> activate matching CUDA Graph variant
  -> GPUModelRunner executes
  -> record batch completion event
  -> request end -> release demand
```

控制面决策、Worker CUDA 安全机制、模型内部 yield boundary 是三个不同层次。

---

## 12. 统一 Shutdown 生命周期

运行时追加的 Omni instance 经 `RunnerFactory.destroy()`：关闭 adapter/pipeline，释放 communicator owner。startup vLLM instance 不能被普通 runtime remove，但 Worker 整体退出时也由 `RunnerFactory.destroy()` 调用 `VllmRunnerBackend.destroy()`。

`GPUWorker.shutdown()` 在原本释放原生 model runner 的位置调用 `_shutdown_model_runner()` hook；`MultiModelGPUWorker` 覆写该 hook，drain 整个 `ModelInstanceRegistry` 并逐个交给 Factory。这样没有提前改变原生清理顺序，也避免 Factory 与父类重复销毁。

当前统一契约是：

```text
create -> load -> warmup -> ready -> drain -> destroy
```

Engine registry 负责部署意图，Worker registry 保存 rank-local materialization。尚待补充的是更细的 Engine 状态转换和注册失败 rollback，而不是 Runner 销毁入口。

---

## 13. 当前机制与代码位置总表

| 机制 | 主要代码 | 当前状态 |
|---|---|---|
| 部署描述 | `model_instance_registry.py::ModelSpec` | 已统一 |
| Engine 模型表 | `ModelRegistry` + `create/restore_engine_model_registry` | startup/runtime 已接入，缺事务状态机 |
| 拓扑规划 | `multi_model_topology.py::TopologyPlanner` | Engine-owned；当前主要验证同构 TP |
| Worker 实例表 | `ModelInstanceRegistry` | 两类 Runner 同等注册 |
| Runner 生命周期 | `runner_factory.py` + `GPUWorker._shutdown_model_runner()` | vLLM/Omni 均经 Factory 创建和销毁 |
| 请求入口 | `WorkerHostedRunnerClient.submit` | API 统一，执行域仍分支 |
| 通信组 | `communicator_registry.py` | Engine spec、Worker materialization |
| 动态分区 | `multi_model_arbiter.py` | full/shared 切换与 event 安全边界已验证 |
| Diffusion slices | `execution_slice_scheduler.py` | Engine Scheduler 已管理 slice 状态/mode |
| Omni 适配 | `omni_runner_adapter.py` | phase/stream/memory/comm provider 已接入 |
| Graph 变体 | `reasoner_graph_variants.py` + Worker hooks | shared/full 双变体已验证 |
| 性能观测 | gate report + Worker startup/Graph counters | 实验完整，部分字段命名仍需去 primary 化 |

---

## 14. 距离目标架构还有什么

目标仍是：

```text
MultiModelEngine
  |-- ModelRegistry
  |-- TopologyPlanner
  |-- UnifiedRequestRouter
  `-- ResourceScheduler
        |
        v
Executor
        |
        v
Worker
  |-- RunnerFactory
  |-- ModelInstanceRegistry
  |-- CommunicatorRegistry
  |-- ExecutionResourceManager
  `-- MultiVariantGraphManager
```

目前底层 CUDA 与 Worker 可行性已经基本打通，主要差距在控制面收敛：

1. 新建真正接收 `ModelSpec[]` 的 `MultiModelEngine`，移除首模型只能来自 `VllmConfig.model_config` 的约束；
2. 将 client 中 backend 分支迁入 Engine `UnifiedRequestRouter`；
3. 把 register/load/ready/fail/remove 做成可 rollback 的部署事务；
4. 把 `primary_demand` 改为模型无关的 resource claim/activity set；
5. 将 arbiter、Green Context、event、memory context 收敛为 `ExecutionResourceManager`；
6. 给 vLLM backend 增加显式 Runner adapter，与 Omni 使用同一生命周期/provider 契约；
7. 支持多个 vLLM scheduling domain、rank 子集、异构 TP/PP/DP 和 N-model 策略。

因此当前不是完整 `MultiModelEngine`，但也不再是“一个原生模型加一个外挂模型”：Engine 已有统一部署表，Worker 已有统一创建和实例所有权；剩余最重要的工作是把最外层 bootstrap、请求路由和生命周期状态机统一起来。

---

## 15. 推荐阅读顺序

dead path 清理后，按下面顺序读即可覆盖当前活跃调用链：

下面第 1–10 项均以 `vllm-single-framework/` 仓库根目录为基准：

1. [`examples/offline_inference/omni_worker_owned_gate.py::main()`](../../../examples/offline_inference/omni_worker_owned_gate.py#L337)
2. [`vllm/experimental/model_instance_registry.py`](../../../vllm/experimental/model_instance_registry.py)：`ModelSpec`、`ModelRegistry`、`ModelInstanceRegistry`，重点看 `create_engine_model_registry()` 与 `restore_engine_model_registry()`
3. [`vllm/experimental/runner_factory.py`](../../../vllm/experimental/runner_factory.py#L194)：`RunnerFactory.create/destroy`
4. [`vllm/experimental/multi_model_gpu_worker.py`](../../../vllm/experimental/multi_model_gpu_worker.py#L194)：`MultiModelGPUWorker._create_model_runner()` 与 `_shutdown_model_runner()`
5. [`vllm/experimental/multi_model_worker.py`](../../../vllm/experimental/multi_model_worker.py#L399)：`WorkerHostedRunnerClient.register/submit`
6. [`vllm/experimental/multi_model_topology.py`](../../../vllm/experimental/multi_model_topology.py#L54) 与 [`vllm/experimental/communicator_registry.py`](../../../vllm/experimental/communicator_registry.py#L19)：`TopologyPlanner` 与 `CommunicatorRegistry`
7. [`vllm/v1/core/sched/execution_slice_scheduler.py`](../../../vllm/v1/core/sched/execution_slice_scheduler.py#L73)：`ExecutionSliceScheduler`
8. [`vllm/experimental/omni_runner_adapter.py`](../../../vllm/experimental/omni_runner_adapter.py#L147)：`EmbeddedOmniDiffusionRunner`，重点看 `configure_execution_unit_scheduler()` 与 `_await_execution_unit_admission()`；旧的本地 Condition admission gate 已删除
9. [`vllm/experimental/multi_model_arbiter.py`](../../../vllm/experimental/multi_model_arbiter.py#L42)：`MultiModelResourceArbiter`，从 `acquire_generator_phase()`/`finish_generator_phase()` 阅读，不再经过旧 step-only alias
10. [`vllm/experimental/reasoner_graph_variants.py`](../../../vllm/experimental/reasoner_graph_variants.py#L25)：`ModelGraphVariantManager`

Omni 侧的修改以 vLLM-Omni 仓库根目录为基准，继续阅读：

11. `vllm_omni/diffusion/worker/runtime_providers.py`：定义 model-scoped memory/stream/phase/communicator provider 契约
12. `vllm_omni/diffusion/worker/diffusion_model_runner.py`：消费 runtime providers，并在模型加载、请求执行和关闭阶段进入相应 context

读完这些位置，再回到 gate 的 reporting 部分，就能区分实际控制状态与实验观测量。旧 `auxiliary` 命名仍存在于当前活跃 Worker RPC 中，不能仅凭名称视为 dead path；已经删除的是不再被当前调用链使用的兼容 alias 和本地 admission fallback。
