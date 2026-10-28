# vLLM 最小接入 vLLM-Omni Diffusion：保留 ModelRunner、替换资源适配层

> 设计状态：Proposal / Codex 实施草案
> 日期：2026-10-27
> 第一目标：在 vLLM GPU Worker 所属的同一 Python 进程中执行 vLLM LLM 和 vLLM-Omni Cosmos3 Generator，并尽量保留 Omni 的模型优化。
> 后续目标：单卡多模型并发、CUDA Stream 分离、Green Context (GC) 空间共置；随后扩展到多卡多模型组管理。

## 0. 执行摘要

**推荐方案：vLLM-owned Infrastructure + Omni-owned DiffusionModelRunner + Thin Resource Adapter。**

不以 `modeling` 层为主要移植边界；也不整体嵌入 Omni DiffusionWorker / Diffusion Engine。原版 Omni `DiffusionModelRunner` 与 Cosmos3 Pipeline、Attention、Cache、Compilation 尽量保留；由 vLLM 现有 Worker 拥有设备、基础分布式环境、共享内存预算和执行调度，再用少量 Bridge / Provider 解决硬编码的全局状态及资源初始化问题。

**关键限定**：
1. 相同宿主进程不自动保证相同 GPU Worker 进程；必须核查实际进程拓扑（PID）以及 CUDA context。
2. 保留 Runner 不等于零改动 Runner：全局 Group Getter、Platform Hooks、Forward Context、Profiling/Memory Pool、内部 Stream 及 Graph 路径仍需逐项审计。
3. 保留 Omni 的优化应逐项实测，不可默认 Cosmos3 的所有模型/版本支持全部优化；`torch.compile` ≠ CUDA Graph replay。
4. vLLM 本身的 AR ModelRunner 不应被拿来直接执行 DiT；它只保留给 LLM。
5. 首阶段以单卡、串行请求、固定形状、无 Offload、无 Diffusion CUDA Graph 为主；随后引入并发。

## 1. 背景与问题

目标是把 video agent 中的 LLM 与视频生成模型部署于**同一个 GPU Worker Python 进程**，为后续单卡 spatial sharing / GC 提供可控实验平台。现有独立 vLLM 与 vLLM-Omni serving 往往对应不同 Worker / 进程生命周期；这不利于验证统一上下文中的资源接管和模型级执行协作。

原设想是在 vLLM 的 `modeling` 层仅注册 Cosmos3 的 Generator 模型。该路径看似最小，但 vLLM AR Runner 的 token-oriented 执行契约与 Diffusion Pipeline 的 step/latents/sampler-oriented 契约不同；仅移植模型层无法自动得到 Omni 现成的编译、缓存、Offload、分布式和执行前后处理。于是会持续向上重建执行逻辑，风险在于最终形成第二套 DiffusionModelRunner。

本方案把重用边界上移一层：**保留 Omni DiffusionModelRunner，资源提供者改由 vLLM 主导。**

## 2. 路线对比与取舍

| 路线 | 可保留的 Omni 能力 | 主要工作 | 主要风险 | 建议 |
|---|---|---|---|---|
| A. modeling-only：将 Generator 做成 vLLM 模型 | 模型层算子（若自行适配） | 重写 Pipeline 驱动、Sampler、缓存生命周期、请求/输出协议、优化接线 | 看似侵入小，实则 Diffusion 运行时重建 | 不作为首选；适用于将来特定单组件融合 |
| B. Runner-level：保留 Omni DiffusionModelRunner | Pipeline、加载、编译、Cache/Offload 路径、执行逻辑 | Worker 资源适配、请求桥接、全局状态隔离 | Runner 依赖 Omni 模块和全局 Hook，不是纯 DI | **首选** |
| C. Worker / Engine-level：完整套入 Omni Runtime | Omni 全部流程 | 调和双方 Scheduler/Worker/IPC/初始化 | 容易保留双套进程和资源控制权 | 不适合当前单进程目标 |

**重要区分**：不是否定 modeling-level 的必要性。Cosmos3 的具体 `modeling`、Attention kernel 仍被 Omni Pipeline 调用；只是我们不再把 *vLLM 的 AR modeling 注册* 作为主接入点。

## 3. 建议目标结构

```text
Single GPU Worker Python Process (one PID per GPU rank)
└── vLLM-owned Worker / Resource Runtime
    ├── Device + CUDA context setup
    ├── Distributed process group base / NCCL lifecycle
    ├── Shared Memory Budget / admission
    ├── Stream + Event / execution arbitration
    ├── Model-aware configuration / group context
    │
    ├── Original vLLM AR ModelRunner
    │   └── LLM prefill / decode + paged KV + native AR CUDA Graph
    │
    └── OmniRunnerAdapter
        └── Original Omni DiffusionModelRunner
            ├── Cosmos3OmniDiffusersPipeline
            ├── Omni Diffusion Attention / Compilation
            ├── Diffusion cache / internal KV / latents / VAE
            ├── Sampler / denoise workflow
            └── optional Offload / diffusion-specific Graph (later)
```

**所有权原则**：vLLM 拥有底层基础设施的创建、销毁和跨模型仲裁；Omni 保留 Diffusion 内部数据结构与算法。Adapter 将资源从前者映射给后者，不能让两个 Worker 独立初始化/终结同一基础设施。

## 4. 接入接口与依赖边界

### 4.1 Adapter（设计示意，非上游现有 API）

```python
class OmniRunnerAdapter:
    def __init__(self, runner, resources, model_id="cosmos3"):
        self.runner = runner
        self.resources = resources
        self.model_id = model_id

    def load_model(self):
        with self.resources.model_scope(self.model_id):
            self.runner.load_model(
                memory_pool_context_fn=self.resources.memory_pool_context
            )

    def execute(self, request):
        with self.resources.model_scope(self.model_id):
            with self.resources.execution_stream(self.model_id):
                return self.runner.execute_model(request)

    def close(self):
        ...  # model-local cleanup; do not tear down shared world/NCCL
```

外层包装**不保证**全部资源调用都改路由：Runner、Pipeline 可能在内部直接引用全局变量、创建 Stream 或调用 Platform Hooks。所以下面的依赖审计是落地前提。

### 4.2 Resource Provider 备选接口

```python
class DiffusionResourceProvider:
    def get_group(self, model_id, parallel_dimension): ...
    def model_scope(self, model_id): ...
    def memory_pool_context(self, kind): ...
    def execution_stream(self, model_id): ...
    def graph_resource(self, model_id, shape_signature): ...
    def profile_and_account(self, model_id): ...
```

优先使用上下文包装与现有扩展点；**不能使用不安全的 monkey-patch 在多个并行模型之间反复覆盖 `_TP`、`_SP` 等模块变量**。

## 5. 三类资源适配矩阵

| 子系统 | 由 vLLM 统一提供 | 保留 Omni 实现 | 新增/修改适配 | 第一阶段 |
|---|---|---|---|---|
| GPU Device / CUDA Context | Worker 设备初始化 | Runner 接收 `torch.device` | 避免二次初始化与错误释放 | 必须 |
| Base Process Group / WORLD | vLLM/PyTorch distributed 初始化 | Omni collective 实现 | 单卡跳过复杂组初始化；多卡须独立 Registry | 单卡最简 |
| TP / SP / CFG / HSDP | 组创建生命周期、跨模型命名空间 | Omni Rank Layout、必要通信算法 | Model-aware Group Registry / Getter 路由 / 所有 rank 一致建组 | 暂不实现多卡 |
| Weights | 全局预算和加载时机 | Omni model loader | 内存池上下文 / 账本 / 加载顺序 | 必须 |
| LLM KV | vLLM native KV 管理 | 不适用 | KV 预算不得吞占 Diffusion 工作集 | 必须 |
| Diffusion KV / Cache-DiT / Latents | 上限、峰值测量、隔离 | Omni 内部算法、缓存布局 | 显存测量及释放/复用协议 | 保留原语义 |
| Offload | 全局内存策略 | Omni Offload | 与 Stream、Graph、权重移动同步 | 初版关闭 |
| CUDA Stream | vLLM 创建、分配执行作用域 | Omni Pipeline 本地异步能力 | Event 依赖、跨 Stream Tensor lifetime | 初版串行 |
| AR CUDA Graph | vLLM 原始路径 | 不适用 | 不动原 AR 路径 | 保留 |
| Diffusion CUDA Graph | 全局预算和捕获生命周期约束 | Omni 已有模型专有实现（仅已支持模型） | Cosmos3 专项 capture/replay + shape key | 初版关闭 |
| Forward / Config Context | 按模型作用域 | `set_forward_context` 等 Omni 逻辑 | 处理 thread-/module-local 全局状态冲突 | 必须 |

### 5.1 Parallel Groups

现有 Omni 分布式实现引用 vLLM `parallel_state`，并在初始化过程中写入 vLLM 全局 `_TP` 等变量。故不能直接在已运行的 vLLM Worker 中再调用 Omni 全套初始化。目标是：只初始化一次基础 `torch.distributed`/WORLD；将 Omni 并行布局初始化改为注册到模型命名空间。单卡 TP=SP=CFG=1 时优先用最小化 Mock/退化组而非先实现多卡全量逻辑。

多卡后续问题：`new_group` 必须确保所有 rank 顺序一致；不能仅构造本地 Registry；还要核查直接 import 的 getter、custom collectives、HSDP mesh、NCCL buffer warm-up、capture safety。

### 5.2 Memory

统一预算 **≠** 统一 allocator。默认仍使用 PyTorch allocator 与 Omni Diffusion 缓存布局。共享内存账本至少统计：LLM weights、LLM KV、Omni weights、Diffusion 峰值 activation/latents、persistent cache、CUDA Graph 私有 pool、NCCL/workspace、保留余量。vLLM 的 KV 分配必须在确定共驻权重和 Diffusion 工作集之后计算或配置；避免两套 Worker 独立宣称可用显存。

Runner 的 `load_model(memory_pool_context_fn=...)` 是可用的权重加载上下文插入点，但不是所有未来分配的完整拦截点。初期只采用预算/准入/峰值 profile，暂不实现复杂 allocator pooling。

### 5.3 Stream

先在相同 Worker 中串行调用 LLM 与 Diffusion 验证正确性；下一步给模型独立 execution stream，使用 `torch.cuda.Event` 建依赖；检查 Pipeline 是否内部切换 Stream、是否存在默认 Stream 操作、Tensor 的 `record_stream`、输出同步时刻；利用 Nsight Systems 检查两模型 Kernel 能否真正 overlap。

两条 Stream 不代表独立 SM 分区。若使用 Green Context，另设实验验证该 GPU/驱动/CUDA 版本的 binding、Graph capture 和并发行为。

### 5.4 CUDA Graph

原 vLLM AR Graph 路径保持原状。Diffusion 的 compile / cache 与 Graph 应分别对待。先确认 Cosmos3 在目标 Omni commit 中有没有模型专有、实际生效的 DiT capture/replay；若无，则仅为固定 shape、固定步长的 DiT forward 提供可选 Graph Adapter，管理静态缓冲区、capture stream、replay、随机状态、内存 pool、cache 更新。不要将整个含 Python 调度和动态 I/O 的 Pipeline 默认视为可捕获。

## 6. 第四个必须处理的接口：Execution Contract

Omni `execute_model()` 接受 Diffusion Request，返回 DiffusionOutput；vLLM AR Runner 则使用 token-based 的调度、KV 和输出协议。不能把 Diffusion 请求直接包装成普通 AR decode。

建议在 vLLM Worker 上增加最小 dispatch：`LLMRequest -> ARRunner`，`DiffusionRequest -> OmniRunnerAdapter`，仅在 Worker 内提供请求级执行入口。第一阶段用测试入口绕过复杂 Server Scheduler；单机通过后再对齐 serving API、request routing 和结果输送。

Omni Runner 存在 `execute_stepwise()`，但**是否适配 Cosmos3 必须验证**；当前通用 stepwise 路径存在 cache_backend 限制，不得将其与 Cosmos3 Cache-DiT 同时可用视为既成事实。后续 step-level colocation 作为单独里程碑。

## 7. Codex 源码审计任务（在编辑代码之前）

1. **版本锁定**：记录 vLLM、Omni 的完整 commit hash，PyTorch/CUDA/GPU、是否使用上游包还是 editable install；确认彼此版本要求。
2. **Process Boundary**：画出 Scheduler -> Worker -> ModelRunner 的真实 Python 进程树、Device 初始化顺序；明确哪个进程中实例化 Runner。
3. **Runner Dependencies**：从 Omni `DiffusionModelRunner.__init__`、`load_model`、`execute_model`、`execute_stepwise` 出发，枚举直接与间接触达的 `torch.distributed`、`vllm.distributed`、Platform、Memory、Stream、Graph、Forward Context、模块级状态。
4. **Loader / Pipeline**：核查 Cosmos3 `Cosmos3OmniDiffusersPipeline` 是否依赖额外进程、隐含组件加载、Reasoner/Generator 双路径或特定检查点组织；**确认所选 checkpoint 中 Generator 是否可单独启动**。
5. **Optimization Inventory**：针对指定 Cosmos3 权重逐项列出 Cache-DiT、TP/SP/CFG、Offload、torch.compile、CUDA Graph、stepwise 的确切支持证据与最小触发条件。
6. **Patch Surface**：提出 Wrapper 可解决、需小型上游 Patch、暂无法解决的三级清单；对每个 Patch 给出代码路径/调用链/回归风险。

不要在未经以上审计前直接重构 vLLM Scheduler 或复制 Omni Worker。

## 8. 分阶段实现、检查点与退场机制

### P0：Baseline 与依赖审计
- 在未修改 Omni 的情况下，用其官方路径成功运行选定 Cosmos3 Generator 任务，并记录 latency/peak memory/输出、编译及缓存配置。
- 另以原生 vLLM 启动 LLM，记录 Worker PID、KV budget、Graph 状态。
- 提供依赖调用图和“哪些初始化具有全局副作用”的明细。

**通过标准**：两个独立基线均可重现，版本固定，资源调用图清楚。

### P1：同一个 Worker，原版 Runner，串行执行
- 让已初始化 vLLM Worker 实例化 `DiffusionModelRunner(vllm_config, od_config, device)`。
- 绕过 Omni 自己的 Worker 设备/WORLD 初始化，优先使用单 GPU 退化组。
- 用 Runner 原生 `load_model` 加载 Cosmos3，并以原生 DiffusionRequest 执行。
- 串行调用 LLM 与 Cosmos3 的固定请求，打印 PID、GPU ID、关键 CUDA context / stream 信息。

**通过标准**：一个 GPU Worker PID 中先后执行两模型、无额外模型 Worker 子进程、无 global group/context 冲突、输出与独立基线误差在合理范围。失败则报告具体函数/调用栈，不能通过无说明移除优化来掩盖。

### P2：共驻内存预算
- 双模型长期驻留，记录显存组成和 Diffusion 峰值；保护原生 LLM KV。
- 固定 workload 下循环请求、确保无显存泄漏及模型状态交叉污染。

**通过标准**：多次运行无 OOM/泄漏，实际占用不超过配置预算，LLM KV 未被覆盖。

### P3：独立 Stream 与真正并发
- 保持 P1/P2 串行 baseline，另用独立 Stream 交错/并发提交。
- Nsight Systems 验证 Kernel 时间线；比较 wall time、两个请求的 latency/throughput、正确性。

**通过标准**：执行与同步正确；能实际观察重叠则进入 GC 实验，否则报告瓶颈而非假定有空间共享。

### P4：保留优化并做逐项消融
- 在新 Runner 接入路径中验证 Omni `torch.compile`、Cache-DiT 等特性与官方独立运行的效果差异。
- 将 Graph、Offload、stepwise 分别列为附加项；逐项启用，不能默认互相兼容。

**通过标准**：同一模型相同输入/shape/seed 下报告每项优化是否真的生效，比较独立 Omni 与接入路径的速度、显存、正确性。

### P5：多卡 Model-aware Group Registry（可延期）
- 统一 base world 初始化，按模型注册 TP/SP/CFG 等，并检查每个 rank 同步顺序。
- 压测模型 A 使用不同 ranks/TP，模型 B 使用不同 ranks/SP 的组合。

**通过标准**：无死锁、无通信组覆盖、跨模型 collectives 正确；退出时模型局部资源释放不破坏 WORLD。

## 9. 推荐仓库结构（拟新增，而非声称上游已存在）

```text
integration/omni_runtime/
  adapter.py              # OmniRunnerAdapter, lifecycle
  resource_provider.py    # common provider protocol
  group_registry.py       # model-scoped groups (P5)
  memory_budget.py        # shared budget / profiler (P2)
  stream_manager.py       # stream & event bindings (P3)
  request_bridge.py       # request/result dispatch
  compat/                 # very small version-pinned upstream patches
  tests/
    test_single_process_serial.py
    test_memory_residency.py
    test_stream_correctness.py
    test_optimization_parity.py
```

原则：**不复制 Omni Runner 文件到自家目录；依赖官方源代码并在小范围 Patch / wrapper 下运行**。对版本兼容分支和 patch 位置进行显式记录。

## 10. 第一轮 Codex 任务单（可直接复制）

> 请先做可行性审计、再实现最小 PoC。固定当前本地 vLLM 和 vLLM-Omni commit；不要直接把 Cosmos3 Generator 塞进 vLLM AR modeling 注册路径，也不要启动第二套 Omni Worker/Engine。目标是在 vLLM GPU Worker 所在的同一 Python PID 中构造**原版** Omni `DiffusionModelRunner`，用其原生 Pipeline 和 DiffusionRequest 运行一次 Cosmos3 Generator 推理，并串行运行一个 LLM 请求。vLLM 拥有设备初始化和基础资源生命周期，Omni Runner 保留模型加载、编译、缓存和执行功能。先做单 GPU，TP/SP/CFG=1，禁用 CPU Offload 和 Diffusion CUDA Graph。审计 Runner 的所有全局依赖，逐项说明哪些可以 wrapper 注入，哪些需要 patch。第一轮产出：①版本/进程拓扑/依赖报告；②运行入口与最小变更 diff；③完整可复制命令；④两模型单 PID 日志；⑤正确性、显存、延迟数据；⑥未解决问题清单。**不要将同一个 API Server PID 与同一个 GPU Worker PID 混淆；不要声称所有 Omni 优化均已经保留，须逐项实测。**

## 11. 待验证的技术风险

- 当前 Cosmos3 支持列表覆盖的是多模态 Pipeline（可能涉及 UND / GEN），并不自动保证任意 Cosmos3 checkpoint 的 Generator 可独立装载。
- PyTorch / vLLM / Omni 的全局状态（`parallel_state`、forward context、platform 初始化）未必能仅靠 ResourceProvider 做依赖注入；必要时对 Getter 做最小 patch。
- Omni Pipeline 内部可能管理自己的 Stream / staging buffer / synchronization，外层 Stream context 并不覆盖内部所有操作。
- 若 LLM KV 使用了设备绝大部分可用内存，后加载 Diffusion 将失败；必须明确内存预算与加载顺序。
- GC 空间分区与 CUDA Graph、NCCL、异步异构执行的兼容性需要目标设备实测。
- 使用固定 shape 对比性能时，需严格控制首次编译、预热、Diffusion cache 命中、seed 和生成步数。

## 12. 已核实的上游参考（以所锁定 commit 为准）

- Omni Runner：`https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/worker/diffusion_model_runner.py`。Runner 文档明确：Worker 管基础设施，Runner 管模型加载、编译、Offload、Cache 和执行；`load_model` 暴露 `memory_pool_context_fn`；`execute_model` 与 `execute_stepwise` 为不同执行入口。
- Omni distributed：`https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/distributed/parallel_state.py`。检查 vLLM 全局分布式状态、`_TP`/`_DP`/`_PP` 关联与组创建流程。
- Omni Cosmos3 Registry：`https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/registry.py`。`Cosmos3OmniDiffusersPipeline` 映射至 `cosmos3/pipeline_cosmos3`。
- Omni Model Support：`https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md`。
- Omni Diffusion Feature Matrix：`https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md`。
- Omni CPU Offload：`https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/cpu_offload.md`。

> 此文档是设计方案，不宣称代码已经集成或经过 5090/8 卡验证。`main` 上游会变化，所有文件路径、API 和能力声明需在实际冻结的 commit 上重新确认。
