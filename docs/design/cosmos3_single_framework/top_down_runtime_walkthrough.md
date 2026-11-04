# 单框架 Top-down 代码导读：从启动脚本到双 Runner 动态执行

> 更新日期：2026-11-03
> 适用代码：当前 `experiment/omni-model-backend` 工作树
> 阅读目标：从一个可运行脚本出发，沿真实调用链理解启动、模型注册、请求执行、动态分区、CUDA Graph 和关闭流程。
> 入口脚本：[`examples/offline_inference/omni_worker_async_gate.py`](../../../examples/offline_inference/omni_worker_async_gate.py)

本文刻意采用 top-down 顺序。vLLM 原生的 Engine、Scheduler、Executor、Worker 只说明它们在调用链中的作用；篇幅集中在本项目新增的构件、资源所有权和版本绑定适配上。

---

## 1. 从 `main()` 开始的最外层调用链

### 1.1 Python 入口

脚本真正的起点在 [`omni_worker_async_gate.py:820-825`](../../../examples/offline_inference/omni_worker_async_gate.py#L820)：

```python
def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
```

从 Python 语义上看，执行顺序是：

```text
__main__
  -> main()
       -> parse_args()              解析并校验 CLI
       -> asyncio.run(...)          创建顶层 asyncio event loop
            -> async_main(args)     完整启动、运行、报告和关闭生命周期
```

因此后文所有流程都位于一次 `asyncio.run()` 生命周期内。Generator 的异步 RPC future、Reasoner 的 `AsyncLLM.generate()`、动态 demand admission 都共享这个前端 event loop；GPU Worker 内部另有进程和辅助执行线程，不能把二者混为一谈。

### 1.2 `parse_args()`：决定本次会创建哪些机制

[`parse_args()`](../../../examples/offline_inference/omni_worker_async_gate.py#L102) 不只是 benchmark 参数解析，它决定后续运行时结构：

| 参数组 | 代表参数 | 影响 |
|---|---|---|
| Reasoner | `--tp-size`、`--reasoner-mode` | vLLM 并行拓扑和 eager/graph 路径 |
| Generator | `--steps`、`--generator-cache` | Omni 请求和 SeaCache |
| 静态分区 | `--generator-stream green`、`--generator-sm-count` | 是否创建互补 Green Context pair |
| 动态调度 | `--partition-policy static/dynamic` | 固定互补分区，或为两个模型同时启用 shared/full 动态切换 |
| Graph | `--reasoner-graph-*-mode` | Graph capture、pool 和 buffer 策略 |
| 观测 | `--trace-partition-nvtx`、`--nsys-capture-measured` | NVTX 和 profiler capture range |

参数校验也在这里提前完成，见 [`omni_worker_async_gate.py`](../../../examples/offline_inference/omni_worker_async_gate.py)。

### 1.3 `async_main()` 的生命周期骨架

[`async_main()`](../../../examples/offline_inference/omni_worker_async_gate.py#L257) 可以先按生命周期读成下面九步，后续章节再逐层进入每一步内部：

```text
1. 设置编译/关闭相关环境变量，加载 workload
2. 组装 AsyncEngineArgs
3. 创建 vLLM config 和 AsyncLLM（启动 EngineCore/Executor/Worker）
4. 创建 AsyncWorkerHostedRunnerClient
5. 构造 ModelSpec，并向 Worker 的统一 registry 追加 Generator
6. warmup Reasoner 和 Generator
7. 按提交顺序异步运行 measured requests
8. 收集 resource plan、metadata、时间线并写 report.json
9. finally: client.close()，然后 engine.shutdown()
```

其中第 3 步先建立 Reasoner 和 Worker 资源域，第 5 步才把 Generator 加载进已经存在的 Worker；这体现了“vLLM Worker 拥有辅助 Runner”，而不是两个 Engine 各自启动后再由脚本拼接。

### 1.4 全局调用图

```text
__main__ -> main() -> asyncio.run(async_main(args))
  |
  | AsyncEngineArgs：Reasoner 配置 + 自定义 Worker/Extension
  v
AsyncLLM / EngineCore / Executor                         vLLM 原生主干
  |
  | 每个 rank 创建一个 Worker
  v
MultiModelGPUWorker                                     本项目新增 Worker 子类
  |-- primary model_runner: GPUModelRunner              原生 Reasoner
  |-- GreenContextPair                                  互补 SM 分区
  |-- MultiModelResourceArbiter                         动态资源仲裁
  |-- ReasonerGraphVariantRegistry                      shared/full Graph 变体
  `-- MultiModelWorkerExtension                         注入式多 Runner RPC 能力
        |
        | register_model(ModelSpec)
        v
      EmbeddedOmniDiffusionRunner                       最小 Omni adapter
        `-- Omni DiffusionModelRunner                   原版 Runner
              `-- Cosmos3 pipeline
                    |-- preparing
                    |-- denoising step 0..N-1
                    `-- VAE segments
```

这里的“单框架”边界是：

- vLLM Engine/Executor 决定进程和 rank 拓扑；
- vLLM Worker 是 GPU、CUDA context、stream、communicator 和 Runner 生命周期的共同所有者；
- Reasoner 继续使用 vLLM 原生 `GPUModelRunner`；
- Generator 继续复用 Omni `DiffusionModelRunner` 和 Cosmos3 pipeline；
- Adapter 只替换资源入口和暴露安全调度边界，不复制 Omni pipeline。

---

## 2. 进入 `async_main()`：配置如何变成运行时

### 2.1 环境和 workload 初始化

`async_main()` 首先设置 Breakable CUDA Graph 开关、Worker shutdown timeout，并调用 [`apply_workload_files()`](../../../examples/offline_inference/omni_worker_async_gate.py#L234) 用文件中的 matched workload 覆盖 CLI 默认值，见 [`omni_worker_async_gate.py:257-266`](../../../examples/offline_inference/omni_worker_async_gate.py#L257)。

这里必须发生在构造 `AsyncEngineArgs` 之前，因为部分环境变量和 Graph mode 会进入 vLLM 编译/启动路径。

### 2.2 两类请求如何构造

入口脚本顶部构造原生 Omni 请求：

- [`make_generator_request()`](../../../examples/offline_inference/omni_worker_async_gate.py#L36)：生成 `OmniDiffusionRequest`；
- [`run_reasoner()`](../../../examples/offline_inference/omni_worker_async_gate.py#L59)：通过 `AsyncLLM.generate()` 提交 Reasoner。

请求协议仍由各自框架拥有。脚本没有发明统一 tensor request，而是在共同 Worker 之上保留两种 Runner 的请求契约。

### 2.3 `AsyncEngineArgs` 分成两层

[`async_main()`](../../../examples/offline_inference/omni_worker_async_gate.py#L257) 把参数分为：

1. vLLM 原生 Reasoner 参数：模型、TP、KV cache、eager/graph；
2. `additional_config.multi_model_runtime.green_context`：我们的共置资源参数。

关键代码在 [`omni_worker_async_gate.py:267-325`](../../../examples/offline_inference/omni_worker_async_gate.py#L267)：

```python
worker_options = {
    "worker_cls": "vllm.experimental.multi_model_gpu_worker.MultiModelGPUWorker",
    "additional_config": {
        "multi_model_runtime": {
            "green_context": {
                "generator_sm_count": ...,
                "partition_policy": "static" | "dynamic",
                "dual_reasoner_graph_variants": ...,
                "reasoner_graph_pool_mode": ...,
                "reasoner_graph_buffer_mode": ...,
                "reasoner_graph_capture_mode": ...,
            }
        }
    },
}
```

同时通过 `worker_extension_cls` 注入：

```python
"vllm.experimental.multi_model_worker.MultiModelWorkerExtension"
```

两者职责不同：

| 构件 | 职责 |
|---|---|
| `MultiModelGPUWorker` | GPU 资源、stream、动态仲裁、Graph 变体、执行覆写 |
| `MultiModelWorkerExtension` | 辅助 Runner/通信组 registry，以及对外 RPC 方法 |

### 2.4 AsyncLLM 启动

脚本在 [`omni_worker_async_gate.py:326-339`](../../../examples/offline_inference/omni_worker_async_gate.py#L326) 创建配置和 `AsyncLLM`。之后的 EngineCore、Executor 和 Worker 拉起属于 vLLM 原生流程，可简写为：

```text
AsyncLLM.from_vllm_config()
  -> EngineCoreClient
  -> EngineCore process
  -> Executor
  -> one Worker process per GPU/rank
```

当前在以下位置增加了**启动观测**，不改变调度语义：

- [`vllm/v1/engine/async_llm.py`](../../../vllm/v1/engine/async_llm.py)：renderer、input processor、EngineCore client 时间线；
- [`vllm/v1/engine/core_client.py`](../../../vllm/v1/engine/core_client.py)：client init 和 READY 等待；
- [`vllm/v1/engine/utils.py`](../../../vllm/v1/engine/utils.py)：进程拉起、HELLO/READY barrier；
- [`vllm/renderers/base.py`](../../../vllm/renderers/base.py)：前端多模态 warmup 时间线。

这些字段用于拆启动时间，后续规整时可独立于核心共置机制处理。

---

## 3. Worker 启动：在每张卡上建立共同资源域

### 3.1 Worker 对象初始化

[`MultiModelGPUWorker.__init__()`](../../../vllm/experimental/multi_model_gpu_worker.py#L131) 初始化 Worker-owned 状态：

- Green Context pair 和四种执行 stream；
- `MultiModelResourceArbiter`；
- `ReasonerGraphVariantRegistry`；
- Graph pool/buffer/capture mode；
- stream 切换、Graph 切换和启动观测计数器。

此时还没有创建 CUDA 资源；这里只建立所有权槽位。

### 3.2 `init_device()`：创建互补 SM 分区

[`MultiModelGPUWorker.init_device()`](../../../vllm/experimental/multi_model_gpu_worker.py#L163) 先调用原生 `GPUWorker.init_device()`，确保设备和分布式环境由 vLLM 初始化，然后才创建共置资源。

核心步骤：

1. [`GreenContextPair`](../../../vllm/experimental/green_context.py#L132) 调 CUDA Driver API，把设备 SM 拆成两个互补集合；
2. 将两个 driver stream 包装为 PyTorch `ExternalStream`；
3. Generator 使用第一分区，Reasoner 使用补集；
4. 动态调度开启时，再创建 primary-context full-SM streams；
5. 创建 `MultiModelResourceArbiter`；
6. 安装 Reasoner CUDA Graph capture-stream patch。

对应代码在 [`multi_model_gpu_worker.py:192-238`](../../../vllm/experimental/multi_model_gpu_worker.py#L192)。

`GreenContextPair` 的重要性质是两个资源来自**同一次 split 的 group 与 remainder**。不能用两个互不相关的 `GreenContextStream` 代替，否则无法保证 SM 集合互斥。Driver API 细节见：

- [`cuDeviceGetDevResource`](../../../vllm/experimental/green_context.py#L162)
- [`cuDevSmResourceSplitByCount`](../../../vllm/experimental/green_context.py#L180)
- [`cuGreenCtxCreate/cuGreenCtxStreamCreate`](../../../vllm/experimental/green_context.py#L197)

### 3.3 Reasoner 加载、memory preparation 和 KV 初始化

以下方法仍调用 vLLM 原生实现，只是把执行放到 Reasoner shared stream，并记录时间：

- [`load_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L262)
- [`determine_available_memory()`](../../../vllm/experimental/multi_model_gpu_worker.py#L275)
- [`initialize_from_config()`](../../../vllm/experimental/multi_model_gpu_worker.py#L362)

因此 Reasoner 的权重、Paged KV、scheduler 和 model runner 仍由 vLLM 管理。我们没有另写 LLM loader 或 KV manager。

---

## 4. Reasoner 编译与双资源 CUDA Graph

### 4.1 为什么要两套 Graph 变体

同一个 Reasoner 会在两种资源状态运行：

- `shared`：Generator 活跃，Reasoner 在互补 GC stream 上运行；
- `reasoner_full`：Generator 不活跃，Reasoner 在 full-SM stream 上运行。

AOT 编译代码可以跨 stream 复用，但 CUDA Graph executable 绑定捕获 stream/context 和稳定地址，所以两种资源环境分别捕获。

### 4.2 单一 vLLM capture trigger

[`compile_or_warm_up_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L375) 没有在脚本外另跑一次捕获。它临时包装原生 `runner.capture_model`，让 vLLM 的一次原生 capture trigger 依次完成两个资源变体：

```text
vLLM compile_or_warm_up_model()
  -> runner.capture_model()                    单一原生触发点
       -> capture shared
       -> register shared
       -> capture reasoner_full
       -> register reasoner_full
       -> reactivate shared
```

完整编排在 [`_capture_reasoner_graph_variants()`](../../../vllm/experimental/multi_model_gpu_worker.py#L497)。

### 4.3 Graph capture stream patch

只在外层 `torch.cuda.stream()` 不足以控制 vLLM 内层 Graph wrapper 的 capture stream，因为部分模块按值导入 `graph_capture`，另一些 wrapper 独立查询 current stream。

[`install_vllm_graph_capture_stream_patch()`](../../../vllm/experimental/runtime_stream_patch.py#L54) 对当前 vLLM 版本的三个入口做窄 patch：

- `parallel_state/gpu_model_runner/cudagraph_utils.graph_capture`；
- `ModelCudaGraphManager._capture_stream`；
- compiled CUDA Graph 的 `current_stream`。

这是当前最版本绑定的部分，规整时应优先演化为 vLLM 原生的 `capture_stream_provider` 接口，而不是继续扩大 monkey patch 面。

### 4.4 Graph registry

[`ReasonerGraphVariantRegistry`](../../../vllm/experimental/reasoner_graph_variants.py#L23) 保存每个变体的：

- 名称；
- stream；
- `ModelCudaGraphManager`；
- PIECEWISE wrapper entry dictionaries；
- capture 数量。

[`activate()`](../../../vllm/experimental/reasoner_graph_variants.py#L56) 在运行前切换 `runner.cudagraph_manager`；PIECEWISE 模式还会切换 wrapper 的 concrete entry registry。Registry 管的是生命周期和选择，不声称两套 CUDA Graph executable 可以合并成一个。

### 4.5 三种统一 capture mode

[`_configure_reasoner_graph_capture()`](../../../vllm/experimental/multi_model_gpu_worker.py#L426) 要求 shared/full 两种资源变体使用同一种模式：

- `full`
- `piecewise`
- `breakable_piecewise`

Breakable 的 host-only 诊断计数在 [`vllm/compilation/breakable_cudagraph.py`](../../../vllm/compilation/breakable_cudagraph.py)，用于区分“wrapper 存在”和“请求实际发生 capture/replay/fallback”。

---

## 5. 注册 Generator：Engine 规划，Worker 实例化

### 5.1 外部部署描述

Reasoner Engine ready 后，脚本创建 [`ModelSpec`](../../../vllm/experimental/model_instance_registry.py)，随后调用：

```python
registered = await client.register(spec)
```

入口见 [`omni_worker_async_gate.py:339-370`](../../../examples/offline_inference/omni_worker_async_gate.py#L339)。

`ModelSpec` 是统一的外部配置接口，包含：

- `model_id/model_path`；
- backend、能力集合和默认 task；具体 task 仍由请求决定；
- placement ranks；
- communication group specs；
- backend-specific options（例如 SeaCache）。

### 5.2 Engine-side 同构 TP 规划

[`WorkerHostedRunnerClient._plan_homogeneous_tp()`](../../../vllm/experimental/multi_model_worker.py#L547) 从所属 Engine 的 `parallel_config` 读取 TP/PP，并生成 `CommunicationGroupSpec`。

当前 Gate 的约束是：

- PP=1；
- Generator placement 与 Reasoner TP ranks 完全一致；
- Engine/client 决定拓扑描述，Worker 不反向依赖 Executor 对象。

随后 [`AsyncWorkerHostedRunnerClient.register()`](../../../vllm/experimental/multi_model_worker.py#L741) 通过原生 `AsyncLLM.collective_rpc()` 把相同部署描述发给各 Worker。Executor/RPC 分发属于 vLLM 原生能力。

### 5.3 Worker-side ModelInstanceRegistry

[`ModelInstanceRegistry`](../../../vllm/experimental/model_instance_registry.py) 同时登记 Engine 启动时创建的 Reasoner 和运行时追加的 Generator。两者都表示为 `ModelSpec + ModelInstance`，但生命周期属性不同：Reasoner 是 `scheduler_managed`，Generator 是 `runtime_added`。

[`MultiModelWorkerExtension.register_model()`](../../../vllm/experimental/multi_model_worker.py) 在每个参与 rank 上实例化运行时模型；旧 `register_auxiliary_runner()` 仅作为兼容别名保留。关闭运行时模型不会移除由 Engine 生命周期拥有的 Reasoner。

1. 校验 placement 和 TP group；
2. 解析 Omni 官方配置；
3. 创建 `EmbeddedOmniDiffusionRunner`；
4. 获取/创建辅助 TP communicator；
5. 在现有 Worker 上启动原版 Omni Runner；
6. 注入 Generator execution stream 和 phase selector；
7. 把对象存入 `_RunnerSlot`。

`MultiModelWorkerExtension` 不是第二个 Worker，也不执行调度；它是注入现有 Worker 的 Runner/communicator registry 和 RPC surface。

### 5.4 辅助 communicator

[`_acquire_communication_group()`](../../../vllm/experimental/multi_model_worker.py#L440) 使用 vLLM `init_model_parallel_group()` 创建 Runner-owned TP group；[`_release_communication_groups()`](../../../vllm/experimental/multi_model_worker.py#L483) 用 owner 引用计数销毁。

当前选择“相同 ranks、独立 communicator”，原因是 Runner 可能有独立 collective 顺序和生命周期。尚未实现多个交叉 communicator 的 distributed turnstile，因此当前支持范围仍是同构 TP Gate，不应外推为任意异构并行安全。

---

## 6. Omni Adapter：保留 Runner，只替换资源入口

核心文件是 [`vllm/experimental/omni_runner_adapter.py`](../../../vllm/experimental/omni_runner_adapter.py)。

### 6.1 配置解析

[`ResolvedOmniDiffusionStage`](../../../vllm/experimental/omni_runner_adapter.py#L16) 保存 Omni 官方 config views。Adapter 继续调用 Omni resolver/enrich 逻辑，没有维护第二套 Cosmos3 配置解析器。

### 6.2 Memory context

[`_memory_pool_context()`](../../../vllm/experimental/omni_runner_adapter.py#L199) 将 Omni 暴露的持久分配点放入 vLLM allocator 的 tagged pool：

- 首次 `use_memory_pool(tag=...)`；
- 后续 `reuse_memory_pool(tag)`；
- tag 以 `model_id` 为前缀，避免多辅助模型冲突。

[`start_on_worker()`](../../../vllm/experimental/omni_runner_adapter.py#L801) 把 context provider 传给 Omni `DiffusionModelRunner.load_model()`。当前已接入 Generator 权重、SeaCache 和 UND cache 的持久分配点，但这仍是共同 allocator/context，不是让 diffusion cache 与 Paged KV 共享同一个 block arena。

### 6.3 Omni group compatibility views

`start_on_worker()` 读取 vLLM world/TP/PP 和辅助 TP group，校验 Omni 其余并行维度，然后安装 Omni module-level compatibility views。执行期间 [`_runner_execution_context()`](../../../vllm/experimental/omni_runner_adapter.py#L217) 用 `use_tensor_model_parallel_group()` 选择当前 Runner 的 TP group。

这保留了 Omni 内部现有 `get_tp_group()` 调用，但仍属于兼容层；长期更干净的方向是 model/runner-scoped communicator provider。

### 6.4 Pipeline phase hook

[`set_generator_phase_stream_selector()`](../../../vllm/experimental/omni_runner_adapter.py#L335) 是动态调度的关键最小 patch。它没有重写 pipeline，而是包住已有边界：

- `_set_denoise_step_metadata`：每个 denoising step；
- `_clear_denoise_step_metadata`：denoising 结束；
- `_decode_latents`：VAE 粗阶段；
- VAE decoder forward pre-hook：把长 VAE decode 再分成若干 segment。

每个阶段通过 [`_begin_scheduled_phase()`](../../../vllm/experimental/omni_runner_adapter.py#L447) 获取 lease，并进入所选 stream 的完整 scope；通过 [`_finish_scheduled_phase()`](../../../vllm/experimental/omni_runner_adapter.py#L487) 记录 completion event。

这里特意不在每个 phase 结束后 host synchronize。相邻 phase 若 mode 不变，依赖同一 stream 的自然顺序；只有发生 full/shared 切换、另一模型 admission 或请求完成时，arbiter 才等待 event。

---

## 7. 请求运行：Generator 和 Reasoner 如何异步进入同一 Worker

### 7.1 Generator 异步提交

脚本通过：

```python
generator_future = await client.start_async(model_id, request)
```

调用链为：

```text
AsyncWorkerHostedRunnerClient.start_async()
  -> AsyncLLM.collective_rpc("start_auxiliary_runner")
  -> each Worker: MultiModelWorkerExtension.start_auxiliary_runner()
  -> Worker-local long-lived ThreadPoolExecutor
  -> EmbeddedOmniDiffusionRunner.execute_model()
  -> Omni DiffusionModelRunner.execute_model()
```

异步 client 和 future 位于：

- [`AsyncWorkerHostedRunnerClient`](../../../vllm/experimental/multi_model_worker.py#L731)
- [`AsyncWorkerHostedRunnerFuture`](../../../vllm/experimental/multi_model_worker.py#L864)

使用同一个长生命周期 auxiliary thread 很重要：CUDA current context/current stream 是 thread-local 状态，warmup 和 serving 若换线程，会使 stream 所有权及 CUPTI 观测含糊。

每个 TP rank 都执行 Generator forward/collective，但只有最小 rank 返回媒体输出，见 [`execute_auxiliary_runner()`](../../../vllm/experimental/multi_model_worker.py#L214)。

### 7.2 Reasoner admission 和请求提交

动态调度时，脚本先调用：

```python
await client.set_primary_demand(True)
```

被 admission 后才创建 `run_reasoner()` task。请求结束后在 `finally` 中释放 demand。完整并发编排见 [`omni_worker_async_gate.py:414-480`](../../../examples/offline_inference/omni_worker_async_gate.py#L414)。

Reasoner 请求本身仍走 vLLM 原生：

```text
AsyncLLM.generate()
  -> EngineCore Scheduler
  -> SchedulerOutput
  -> Executor.execute_model()
  -> MultiModelGPUWorker.execute_model()
  -> GPUModelRunner
```

也就是说，Generator 目前通过 utility RPC/辅助线程运行；Reasoner 仍由原生 scheduler 连续批处理。

---

## 8. 动态资源仲裁

### 8.1 仲裁状态机

[`MultiModelResourceArbiter`](../../../vllm/experimental/multi_model_arbiter.py#L46) 管理三种运行状态：

```text
无 Reasoner demand：Generator -> generator_full
两者都有 demand：Generator -> shared，Reasoner -> shared
无 Generator request：Reasoner -> reasoner_full
```

调度单位不是任意 kernel，而是显式安全边界：

- Generator：preparing、每个 denoising step、VAE segment、complete；
- Reasoner：一个 vLLM scheduler batch，外层以 request demand epoch 约束。

### 8.2 为什么需要 lease 和 event

host 端结束一次 Python 调用不等于 GPU 已完成。每个 phase/batch 因此发布 CUDA completion event：

- Generator 在 [`_finish_scheduled_phase()`](../../../vllm/experimental/omni_runner_adapter.py#L487) 发布；
- Reasoner 在 [`MultiModelGPUWorker.execute_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L822) 发布。

Arbiter 只在资源所有权可能冲突时 drain：

- full Generator 正运行时，Reasoner admission 等到 phase boundary；
- full Reasoner batch 已提交时，Generator admission 等它真正完成；
- Generator 从 full 切 shared 或反向切换时，等待前一 lane 的 event；
- 连续同 mode phase 不同步。

核心逻辑：

- [`acquire_primary_demand()`](../../../vllm/experimental/multi_model_arbiter.py#L129)
- [`select_primary_batch_stream()`](../../../vllm/experimental/multi_model_arbiter.py#L178)
- [`acquire_generator_phase()`](../../../vllm/experimental/multi_model_arbiter.py#L221)
- [`finish_generator_phase()`](../../../vllm/experimental/multi_model_arbiter.py#L316)

### 8.3 Reasoner 执行时切换 stream 和 Graph

[`MultiModelGPUWorker.execute_model()`](../../../vllm/experimental/multi_model_gpu_worker.py#L822) 在原生 Worker forward 外增加：

1. 向 arbiter 登记 scheduler batch；
2. 选择 `shared` 或 `reasoner_full` stream；
3. 激活对应 CUDA Graph registry；
4. 若资源 variant 改变，drain 前一 compute stream 和 output-copy stream；
5. 更新 `GPUModelRunner.main_stream`；
6. 在所选 stream scope 内调用原生 `super().execute_model()`；
7. 记录 batch completion event。

这里更新 `main_stream` 是因为它是 cached property，而 AsyncOutput 必须在实际执行 sampling 的 stream 与 copy stream 之间建立正确依赖。

---

## 9. AsyncOutput event pool：解决共置下的 host/CUDA event 争用

Reasoner 每个 decode batch 都会把 sampling 结果异步 D2H。原生 `Stream.wait_stream()` 会创建临时 event；与密集 VAE kernel submission 共置时，这条路径曾出现明显 host/CUDA driver gap。

当前新增：

- [`AsyncOutputEventPool`](../../../vllm/v1/worker/gpu/async_utils.py#L125)：固定数量的 `{handoff_event, copy_done_event}` slot；
- [`AsyncOutput`](../../../vllm/v1/worker/gpu/async_utils.py#L184)：复用 slot，用 `record + wait_event` 代替每轮临时 event；
- [`GPUModelRunner` 初始化](../../../vllm/v1/worker/gpu/model_runner.py#L221)：按 `VLLM_ASYNC_OUTPUT_EVENT_POOL_SIZE` 创建 pool；
- [`AsyncOutput` 构造点](../../../vllm/v1/worker/gpu/model_runner.py#L2154)：传入 pool。

slot 只有在 D2H copy 完成并被 `get_output()` 消费后才归还。pool 耗尽时当前实现回退到一次性 event，而不是在 Worker RPC 内阻塞，避免 async scheduling 死锁。

它与动态 GC 仲裁是两类机制：前者减少输出 handoff 的 event/driver 争用，后者决定模型使用哪个 SM/stream 资源域。

---

## 10. 结果收集、观测和关闭

### 10.1 结果收集

`generator_future.result()` 最终调用 Worker `collect_auxiliary_runner`。所有 TP rank 完成，但 client 要求恰好一个 output-owner result。长期 serving 路径仍应把 raw video tensor 改成 Worker-local artifact/descriptor，避免大型媒体结果进入控制面 RPC。

### 10.2 运行时观测

[`multi_model_resource_plan()`](../../../vllm/experimental/multi_model_gpu_worker.py#L959) 汇总每个 rank 的：

- 实际 SM 数和 stream handle；
- Graph capture/variant/counter；
- Graph pool、buffer 和 memory snapshot；
- CUDA allocator；
- Worker 启动时间线；
- 动态仲裁状态和 phase trace。

脚本把这些信息写入 JSON report。NVTX 标记覆盖 Generator phase、Reasoner batch、admission 和 GPU drain，可用于 Nsys 验证 host 决策与 GPU 时间线是否一致。

### 10.3 关闭顺序

[`MultiModelGPUWorker.shutdown()`](../../../vllm/experimental/multi_model_gpu_worker.py#L1067) 的顺序是：

1. 在 primary distributed runtime 仍有效时关闭辅助 Runner；
2. 调原生 Worker shutdown；
3. 恢复 Graph capture patch；
4. synchronize 并释放 streams；
5. 清空 arbiter/Graph registry；
6. 销毁 Green Context pair。

`MultiModelWorkerExtension` 同时负责关闭 auxiliary thread pool、adapter、Runner 和通信组。当前仍见过 EngineCore grace timeout/shared-memory 警告，这属于尚待收敛的生命周期问题，不能把“请求结果已正确返回”视为 shutdown 已完全正式化。

---

## 11. 当前新增机制与代码位置总表

| 机制 | 主要代码 | 当前作用 |
|---|---|---|
| 外部异步双请求 Gate | [`omni_worker_async_gate.py`](../../../examples/offline_inference/omni_worker_async_gate.py) | 独立提交 Generator/Reasoner，生成报告和 Nsys 边界 |
| 统一模型描述与实例 registry | [`ModelSpec/ModelInstanceRegistry`](../../../vllm/experimental/model_instance_registry.py) | 同时描述主模型与运行时追加模型，管理能力、placement 和生命周期 |
| Worker 模型注册/RPC | [`MultiModelWorkerExtension`](../../../vllm/experimental/multi_model_worker.py) | 注册、执行、异步 job、收集和关闭运行时追加的 Omni Runner |
| 同步/异步前端 client | [`WorkerHostedRunnerClient`](../../../vllm/experimental/multi_model_worker.py#L537)、[`AsyncWorkerHostedRunnerClient`](../../../vllm/experimental/multi_model_worker.py#L731) | 经 Engine collective RPC 控制 Worker-hosted Runner |
| 辅助通信组 registry | [`_acquire_communication_group()`](../../../vllm/experimental/multi_model_worker.py#L440) | Engine 描述拓扑，Worker 创建/引用计数/销毁 communicator |
| 多模型 GPU Worker | [`MultiModelGPUWorker`](../../../vllm/experimental/multi_model_gpu_worker.py#L131) | 共同物理资源所有者和执行覆写入口 |
| 互补 SM 分区 | [`GreenContextPair`](../../../vllm/experimental/green_context.py#L132) | 同一 GPU 上创建不重叠 GC streams |
| 动态资源仲裁 | [`MultiModelResourceArbiter`](../../../vllm/experimental/multi_model_arbiter.py#L46) | 在 full/shared 三状态间按安全边界切换 |
| Omni Runner adapter | [`EmbeddedOmniDiffusionRunner`](../../../vllm/experimental/omni_runner_adapter.py#L151) | 保留 Omni Runner/pipeline，注入 Worker 资源 |
| vLLM memory context provider | [`_memory_pool_context()`](../../../vllm/experimental/omni_runner_adapter.py#L199) | 为权重、SeaCache、UND cache 提供 tagged allocation context |
| Omni TP compatibility | [`start_on_worker()`](../../../vllm/experimental/omni_runner_adapter.py#L801) | 将 vLLM/辅助 groups 映射给 Omni 当前接口 |
| Generator phase slicing | [`set_generator_phase_stream_selector()`](../../../vllm/experimental/omni_runner_adapter.py#L335) | 暴露 denoising step 与 VAE segment 调度边界 |
| Graph capture stream patch | [`runtime_stream_patch.py`](../../../vllm/experimental/runtime_stream_patch.py) | 确保 vLLM 所有 capture 层使用目标 GC stream |
| Reasoner Graph variant registry | [`reasoner_graph_variants.py`](../../../vllm/experimental/reasoner_graph_variants.py) | 统一持有/切换 shared 和 reasoner_full executables |
| 三种统一 Graph 模式 | [`_configure_reasoner_graph_capture()`](../../../vllm/experimental/multi_model_gpu_worker.py#L426) | FULL、PIECEWISE、Breakable PIECEWISE 对照 |
| Breakable 实际使用计数 | [`breakable_cudagraph.py`](../../../vllm/compilation/breakable_cudagraph.py) | capture/replay/fallback 诊断，不改执行语义 |
| AsyncOutput event pool | [`async_utils.py`](../../../vllm/v1/worker/gpu/async_utils.py#L117) | 复用 D2H handoff events，减少 decode 共置 gap |
| 启动关键路径观测 | `async_llm.py`、`core_client.py`、`engine/utils.py`、`renderers/base.py` | 只做时间线诊断，非核心资源机制 |

---

## 12. 规整代码时建议保留的边界

后续规整不应按“文件行数少”作为目标，而应保持以下依赖方向：

```text
外部部署配置
  -> Engine/Client 规划逻辑拓扑
  -> Worker 实例化物理资源和 Runner
  -> Runner adapter 消费 resource provider
  -> Omni pipeline 只暴露阶段边界，不拥有 vLLM scheduler
```

建议优先处理：

1. 将 `runtime_stream_patch.py` 演化成 vLLM 内部正式的 capture-stream provider；
2. 将 `MultiModelGPUWorker` 中启动 profiling/memory snapshot 与核心执行逻辑拆文件；
3. 把 Graph variant capture/切换完整下沉到独立 manager，而不仅是 registry；
4. 将 phase hook 变成 Omni Runner/Pipeline 的小型正式 callback protocol，移除 method replacement；
5. 把辅助 Runner RPC、部署规划和 Worker-local registry 分成 control-plane 与 worker-runtime 两层；
6. 正式化 output artifact transport 和 shutdown；
7. 在扩展到交叉通信组前实现跨 communicator 的一致 enqueue-order 协议。

当前最应避免的是重新实现 Omni `DiffusionModelRunner` 或 Cosmos3 pipeline。现有工作真正需要收敛的是资源 provider、调度边界和生命周期接口，而不是模型计算代码。
