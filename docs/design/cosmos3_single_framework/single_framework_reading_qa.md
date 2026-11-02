# vLLM 单框架多 Runner：代码阅读问答

本文记录阅读 `single_framework_code_reading_guide.md` 过程中产生的问题、当前结论和后续待办。

为避免混淆，每个回答尽量区分：

- vLLM 原生机制；
- vLLM-Omni 原生机制；
- 本项目已经实现的修改；
- 尚未实现或仍需验证的设计。

## 1. 背景与总体架构

### Q1：未来是否需要迁移到 AsyncLLM？

如果目标发展为在线并发服务，未来很可能需要接入 `AsyncLLM`，但它不是当前 Worker/Runner 改造的替代方案，而是位于其上方的异步服务入口。

```text
AsyncLLM / Serving API
        ↓
EngineCore / Scheduler
        ↓
Executor
        ↓
Worker
        ↓
Reasoner Runner + Generator Runner
```

当前工作的重点在下半部分：让同一个 Worker 托管多个 Runner，并协调并行组、CUDA Stream/Green Context、显存和执行生命周期。`AsyncLLM` 主要负责上半部分，包括异步请求、取消、流式输出、多客户端并发、持续批处理和背压。

因此，更准确的演进路线不是立即把当前实现“迁移到 AsyncLLM”，而是：

1. 先稳定 Worker-owned 多 Runner 执行底座；
2. 将 Generator 请求纳入 Engine/Scheduler 可以识别的一等请求模型；
3. 再由 `AsyncLLM` 或同层异步控制面统一接收 LLM、Diffusion 和复合请求。

当前 Generator 仍主要通过辅助 Runner RPC 和 Worker 内线程池启动。此时直接采用 `AsyncLLM`，只能改善 Reasoner 请求入口，不能自动解决 Reasoner 与 Generator 的依赖调度、共置决策或 GC 分区。

结论：

- 离线执行、资源机制与可行性验证阶段，不需要立即迁移；
- 生产级并发 Serving 阶段，应考虑接入 `AsyncLLM`；
- 当前 GC、Stream、通信组和多 Runner Worker 工作仍是异步 Serving 所需的执行基础；
- 后续核心工作是多模型请求类型与 Scheduler 集成，而不是重写现有资源管理机制。



## 2. 单框架改造 Overview

### Q2：原版 vLLM 的 communicator 是 Worker 创建的吗？

需要区分“拓扑由谁决定”“进程组由谁初始化”和“NCCL communicator 何时真正建立”。

- Engine/Executor 根据 \`ParallelConfig\` 决定 world size、rank、TP/PP/DP 拓扑，并启动或连接各个 Worker；
- 每个 Worker 在 \`Worker.init_device()\` 中调用 \`init_worker_distributed_environment()\`；
- 该函数在 Worker 进程内初始化 \`torch.distributed\` 和 vLLM 的 \`_WORLD/_TP/_PP/_DP/...\` \`GroupCoordinator\`；
- 底层 NCCL communicator 可能在 process group 创建阶段建立，也可能延迟到第一次 collective，取决于 PyTorch/NCCL 的初始化路径。

因此可以简写为：**原版 communicator 的拓扑由 Engine/Executor 配置，逻辑通信组由每个 Worker 进程创建或加入，底层 NCCL communicator 由分布式后端建立。**

本项目的辅助 Generator group 沿用同样的分层：客户端从 Engine config 形成 \`CommunicationGroupSpec\`，再由 \`MultiModelWorkerExtension._acquire_communication_group()\` 在各 Worker 上调用 vLLM 原生 \`init_model_parallel_group()\` 实例化。

### Q3：Omni module-level group view 中的“模块级状态”是什么意思？

这里的 module 是 Python 模块，例如 \`vllm_omni.diffusion.distributed.parallel_state\`。该模块直接保存 \`_WORLD/_SP/_PP/_CFG/_DP/_FS\` 等变量。

这些变量不是某个 \`DiffusionModelRunner\` 实例的成员，也不是某次请求的局部变量，而是**当前 Worker 进程内该 Python 模块唯一的一组全局变量**。同一进程中的所有线程和所有 Omni Runner 都会看到同一份值；多进程部署时，每个 Worker 进程各有一份独立副本。

当前 Adapter 的 \`_install_omni_group_views()\` 会保存旧值，再把这些模块变量映射到 Worker 已创建的 world、辅助 TP group 和 singleton group。这样可以兼容 Omni 原有的 \`get_*_group()\` 调用，但存在边界：

- 当前单 Worker、单辅助 Omni Runner、同构 TP Gate 可以工作；
- 如果一个 Worker 同时运行两个需要不同拓扑的 Omni Runner，它们无法仅靠这一组模块全局变量同时表达各自拓扑；
- 在线程并发切换这些变量会有竞态；未来需要 group provider、实例级 execution context 或 context-local 状态，不能只依赖反复覆盖模块全局变量。

### Q4：\`MultiModelWorkerExtension（注入 Worker）\` 是否应写成“注入 Runner”？

不应改成“注入 Runner”。这里被注入的对象确实是 Worker。

vLLM 原生提供 \`worker_extension_cls\`：Worker 构造时会动态修改 Worker class 的继承关系，把 Extension 的方法加入 Worker，以便 Executor 通过 \`collective_rpc\` 调用这些方法。本项目配置的是：

\`\`\`python
worker_extension_cls = (
    "vllm.experimental.multi_model_worker.MultiModelWorkerExtension"
)
\`\`\`

Extension 随后在 Worker 内注册、持有和执行辅助 Runner。因此准确表述应是：

> \`MultiModelWorkerExtension\`：注入 Worker 的多 Runner 托管扩展。

也就是“能力注入 Worker，Runner 注册到该能力中”。Runner 是被托管对象，不是 Extension 的注入目标。

### Q5：“没有统一显存 admission，只能手工设置 KV cache budget”是什么意思？

这句话原本范围写得过大。vLLM 原生并非没有显存 admission：它会根据 \`gpu_memory_utilization\`、模型权重、profile 得到的 activation 峰值和 CUDA Graph 估算，计算可用于 LLM KV cache 的空间；也可以用 \`kv_cache_memory_bytes\` 显式指定 KV cache 大小。

当前缺少的是**跨 Reasoner 与 Generator 两个 Runner 的统一显存 admission**：系统在加载模型或接收请求前，尚不能统一判断下面这些资源的总峰值是否会超过设备预算：

- Reasoner 权重与 Paged KV cache；
- Reasoner CUDA Graph pool；
- Generator 权重；
- SeaCache 与 UND cache；
- diffusion/VAE activation 峰值；
- compile artifact、NCCL workspace 和其他持久/临时缓冲。

目前虽然部分 Generator 资源已经接入同一 Worker 的 allocator/memory context，并且可以观测和分类，但 Engine 的 KV cache 规划器不会自动替 Generator 的未来峰值留出完整预算。因此实验中需要手工缩小 Reasoner 的 \`gpu_memory_utilization\` 或 \`kv_cache_memory_bytes\`，给 Generator 留出空间。

这里的 admission 是“在加载/接受请求之前，根据全体 Runner 的预算决定接受、拒绝、降低 batch、缩小 cache 或调整 placement”，不是指重新实现一个 CUDA allocator。


## 3. Memory 接入

### Q6：UND cache 不是 forward once 后一直复用吗？为什么文档说它会在后续请求中增量创建持久 tensor？

是的。UND cache 的语义是 encode once，并在同一次 generation 的全部 denoising steps 中复用。首次 diffusion forward 在 `cached_kv is None` 时运行 UND language model，并由 `_retain_und_cache()` 保存 K/V 和 RoPE frequency；后续 diffusion steps 直接读取这组 cache。

它不是跨所有请求永久复用。Pipeline 会在 generation 开始和结束调用 `transformer.reset_cache()`，因为新 prompt、shape、控制条件或 CFG branch 可能需要不同的 UND 状态。因此这里的“持久”是相对单次 forward 的临时 activation：cache 跨 denoising steps 存活，但生命周期仍是 request/generation scoped。

`reuse_memory_pool("und_cache")` 的作用不是让同一请求的 UND cache 不断增长，而是让每个新 generation 首次创建 K/V 时重新进入已经存在的 tagged pool。generation 结束后 tensor 引用清除，pool 保留并供后续请求再次使用。原代码导读中的表述已经相应修正。

### Q7：`reuse_memory_pool()` 中临时关闭 `expandable_segments` 是做什么的？

`expandable_segments` 是 PyTorch CUDA caching allocator 用来降低动态尺寸负载碎片的一种 segment 增长机制。vLLM 的 `CuMemAllocator` 使用 `CUDAPluggableAllocator`、独立 `MemPool` 和 cuMem map/unmap；PyTorch 的 expandable segments 与这条 MemPool 路径存在兼容性问题，所以进入或重入 tagged pool 时暂时关闭它，退出后恢复。

`current_alloc_conf()` 先读取完整配置；`with_conf_flag()` 只翻转 `expandable_segments`，保留 `max_split_size_mb`、`garbage_collection_threshold` 等其他字段；finally 中再恢复 `prev_conf`。不能只写 `expandable_segments:False`，因为 PyTorch 会把 allocator 配置整体重新解析，并重置未出现的其他选项。

需要注意，allocator setting 是进程级状态。未来如果允许多个线程同时创建持久分配，还需要审视这段临时切换的同步范围；它解决的是 MemPool 兼容性，不等于完整的并发 allocator 隔离。

### Q8：在哪里查看我们对 vLLM-Omni 的修改，例如 SeaCache 的 `cached_residual.clone()`？

配套 checkout：`/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/.runtime/vllm_omni_tf517/vllm-omni`。

分支为 `experiment/cosmos3-dual-framework`，对应提交为 `4fdf39e15`（`[Experimental] Expose diffusion cache memory providers`）。该提交修改：

- `vllm_omni/diffusion/worker/diffusion_model_runner.py`：接收 memory provider；
- `vllm_omni/diffusion/cache/seacache/backend.py`：向 hook 传递 provider；
- `vllm_omni/diffusion/cache/seacache/hook.py`：在 cache pool 中 clone 长期保留的 residual/indicator；
- `vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py`：在 `und_cache` pool 中保留 UND K/V 和 frequency。

SeaCache 的 `_cache_memory_context()` 和 `cached_residual = residual.clone()` 位于 `vllm_omni/diffusion/cache/seacache/hook.py`；UND cache 的关键函数是 `transformer_cosmos3.py::_retain_und_cache()`。

```bash
cd /data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/.runtime/vllm_omni_tf517/vllm-omni
git show --stat 4fdf39e15
git show 4fdf39e15
git diff 4fdf39e15^ 4fdf39e15 -- vllm_omni/diffusion/cache/seacache/hook.py
git diff 4fdf39e15^ 4fdf39e15 -- vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py
```

虽然分支名仍含 `dual-framework`，当前单框架原型也直接复用了这份 checkout 中的 provider patch，并没有在 vLLM 仓库复制第二份 Omni 实现。

## 4. Stream、Green Context 与 CUDA Graph

### Q9：并行情况下，是否每个 rank 都有大小统一但彼此独立的 GC partition pair？

是的，当前同构 TP 实现基本如此。每个 Worker/rank 绑定一张本地 GPU，并在自己的 `MultiModelGPUWorker.init_device()` 中创建一个本地 `GreenContextPair`。该 pair 从该 GPU 的 SM resource 做一次 split：first partition 给 Generator，remainder 给 Reasoner。每个 rank 拥有独立的 CUDA primary context、Green Context handle 和 stream handle；不存在跨 GPU 的“全局 Green Context”。

所有 rank 从同一份 `additional_config.multi_model_runtime.green_context.generator_sm_count` 读取 requested SM 数，因此在同型号 GPU 的同构部署中，请求大小一致，通常 actual split 也一致。但这不是 CUDA 自动提供的跨 rank 保证：系统仍应收集并验证每个 rank 的 actual SM count；异构 GPU 或驱动取整规则不一致时不能直接假设相同。

所以应描述为：**拓扑和分区策略全局一致，GC 资源在每个 rank/GPU 上独立创建。** CUDA 只保证单个 pair 内两份 SM resource 来自同一次 split、互不重叠；跨 rank 的同步、collective 顺序和模式切换仍由运行时负责。

### Q10：Reasoner CUDA Graph stream monkey patch 未来是否应改成正式代码接口？如果改，应怎么改？

是。当前 `runtime_stream_patch.py` 是实验阶段用于证明可行性的版本绑定 monkey patch。它同时替换 by-value import 的 `graph_capture`、`ModelCudaGraphManager._capture_stream()` 和 compiled graph 路径的 `current_stream()`，说明 vLLM 当前没有单一、显式的 capture-stream 注入点。长期保留全局 monkey patch 会有版本脆弱、进程内多 Runner 相互覆盖以及 restore 顺序等问题。

正式化时不应把 GC 逻辑硬编码进通用 CUDA Graph 代码，而应增加一个显式的 execution resource/capture stream provider：

1. Worker 为每个 Runner 持有 `ExecutionResourceContext`，其中包含执行 stream、capture stream、GC/partition identity 和 graph registry；
2. ModelRunner/GraphManager 构造时接收该 context 或 `capture_stream_provider`；
3. `graph_capture()` 接受显式 stream/context，不再依赖模块级 `torch.cuda.current_stream()`；
4. `ModelCudaGraphManager._capture_stream()` 从实例 context 取 stream；
5. compiled CUDA Graph 路径同样接收该 context，移除独立查询 current stream 的旁路；
6. replay 必须在与 variant 对应的 GC execution stream 上 launch；
7. graph key 至少包含 Runner、device、shape/batch、execution mode 和 GC/SM partition identity。分区改变时选择或重新 capture 对应 variant，不能把旧 graph 当成可任意迁移对象。

显式实例级依赖优于仅用 `ContextVar`：graph capture 可能跨 compile helper、线程或嵌套 manager，资源所有权放在 Worker/Runner context 中更容易验证。当前问题只做设计记录，不修改实现。

### Q11：把 diffusion 按 step（甚至 layer）切成调度片段，在边界检查 LLM，并在无 LLM 时独占 GPU、LLM 到达后转为共置，这个思路是否成立？

成立。这可以称为 cooperative yielding、phase-aware scheduling 或 quanta-based co-scheduling，与 PD 共置中把长 Prefill 切段、让 Decode 在安全边界推进的思想相近。CUDA kernel 一旦提交后很难由应用层低成本抢占，因此真正可控的切入点是尚未提交下一段工作时的 host-side boundary。

建议先以 diffusion step 为最小调度 quantum：

```text
Generator full-SM quantum 完成
  → 记录/等待 completion event
  → scheduler 检查 active/ready LLM backlog
      ├─ 无 LLM：下一 diffusion step 继续 full-SM
      └─ 有 LLM：进入 partitioned epoch
           ├─ Generator 后续 step 使用 Generator GC stream
           └─ LLM prefill/decode 使用互补 Reasoner GC stream
```

当 LLM 队列清空后，也不能在任意 kernel 中途恢复 full-SM；应等到下一个安全的 diffusion/LLM 边界，确认相关 stream/event 和 collective 已完成，再切回 full-SM epoch。

这个设计需要补足以下机制：

- 两套执行模式：Generator full-SM stream/context，以及 Generator/Reasoner 的互补 GC pair；full-SM 工作与 partitioned 工作不能同时提交，否则分区隔离失去意义；
- epoch/state machine：`GEN_EXCLUSIVE`、`COLOCATED`，以及有序 drain/switch；
- 所有 TP ranks 必须采用相同的 epoch 和切换边界，不能各 rank 根据本地线程状态自行决定；
- 如果片段内包含 collective，需要 distributed turnstile 或等价协议保证各 rank 和跨 communicator 的 host-enqueue order；
- compile/CUDA Graph 需要按 execution mode 和 partition 保存 variant；
- hysteresis、最小驻留时间和公平性，避免 LLM 请求抖动导致每个 step 频繁切换或 diffusion 饥饿。

按 layer 切分可以降低 LLM 到达后的等待时间，但代价明显更高：更多 Python/调度开销，更细的 event，可能破坏 regional compile/CUDA Graph 边界，并增加 collective 排序难度。因此合理路线是先验证 per-step yielding，再根据 step 尾延迟决定是否细化到 block/layer group，而不是一开始 per-layer。

这套机制当前尚未实现；当前 Gate 使用固定 GC pair，并在 Worker 内通过线程和 event 一次性启动并发负载。

## 5. 单 Worker 多 Runner

### Q12：未来是否可以不区分 `MultiModelGPUWorker` 和 `MultiModelWorkerExtension`，让 LLM 与 multimodal model 成为等价的共置参与者？

可以，而且这是更合理的长期方向，但需要区分“调度与资源语义上的对等”和“底层 Runner 实现完全相同”。

当前拆成两部分是原型阶段的最小侵入策略：

- `MultiModelGPUWorker` 是真正的 Worker 子类，覆盖 device 生命周期、Reasoner stream/GC、Graph capture 和 shutdown；
- `MultiModelWorkerExtension` 利用 vLLM 原生 `worker_extension_cls` 动态继承机制，为现有 Worker 添加辅助 Runner registry、通信组 registry、线程池和 collective RPC；
- vLLM LLM Runner 仍是 primary runner，Omni Diffusion Runner 被当作 auxiliary runner。

这种不对称来自 vLLM 当前 Engine/Scheduler 的单模型语义，而不是最终系统必须认为 LLM 更重要。正式化后可以把两部分收敛为一个 first-class `MultiModelGPUWorker`，内部持有统一的 `RunnerRegistry`：

```text
MultiModelGPUWorker
  ├─ RunnerSlot(reasoner-a, kind=llm, adapter=VllmRunnerAdapter)
  ├─ RunnerSlot(generator-a, kind=diffusion, adapter=OmniRunnerAdapter)
  ├─ ResourceRegistry
  │    ├─ streams / GC
  │    ├─ communicator groups
  │    ├─ memory budgets
  │    └─ graph variants
  └─ execution / shutdown coordinator
```

每个 `RunnerSlot` 可以实现统一的生命周期协议，例如 `load`、`prepare`、`execute_quantum`、`profile`、`cancel` 和 `close`。Engine/Executor 根据 placement plan 把任意模型部署到某组 Worker；Scheduler 在共置决策中将 LLM、diffusion、VAE 等都视为资源竞争参与者。

但不应为了形式对称强迫它们使用完全相同的内部执行接口：LLM 有 token scheduler、Paged KV 和 continuous batching；diffusion 有 timestep、SeaCache、VAE 和 regional compile。合理做法是统一外层资源与生命周期协议，保留 backend-specific adapter。

渐进式迁移时，可以先保留 vLLM primary LLM runner 的原生路径，同时也把它注册为一个可观察的 `RunnerSlot`；等多模型请求正式进入 Engine Scheduler 后，再移除动态 `worker_extension_cls`。因此 extension 不是最终架构必需品，而是当前降低 vLLM 源码修改量的接入手段。

### Q13：以后不通过控制面 RPC 传 raw video tensor，视频结果应如何保存和传递？

应把结果分成 control plane descriptor 和 data-plane artifact 两部分。只有 designated output owner（通常是每个模型并行组的 TP0）负责物化结果；其他 ranks 在完成必要同步后释放本地输出。

Worker 侧建议采用如下流程：

```text
GPU output tensor
  → 专用 export stream 做异步 D2H
  → CUDA event 标记数据可用
  → 有界后台 writer/encoder
  → 临时文件写入并 atomic rename
  → RPC 只返回 ArtifactDescriptor
```

`ArtifactDescriptor` 至少包含：

```text
artifact_id / request_id
owner_worker / rank
URI 或共享路径
MIME type / encoding
shape / dtype / frame count / fps
size_bytes
checksum
ready/error state
lifetime / refcount metadata
```

最终用户输出与中间结果应区别处理：

- 最终视频：在 Worker 侧编码为 MP4/WebM；图片编码为 PNG/JPEG，RPC 返回路径或对象存储 URI；
- 精确回归或 benchmark：保存未压缩/无损 tensor artifact，例如 safetensors，避免视频编码改变数值；
- 同 Worker/GPU 的后续阶段（Critic、Refinement）：优先不落盘，通过 Worker-local `ArtifactRegistry` 返回 handle，并用 CUDA event 表达 readiness；
- 同机跨进程：必要时使用共享内存、CUDA IPC 或受控的本地 artifact store；
- 跨节点或需要持久化：上传共享文件系统或对象存储，控制面只传 descriptor。

为了不让视频编码阻塞推理，D2H、编码和写盘必须进入有容量限制的后台队列；队列也需要 admission/backpressure。不能无限堆积待导出视频，否则只是把 GPU OOM 转换成 pinned host memory 或磁盘队列爆炸。

当前最小可行实现可以先采用：TP0 将 tensor 异步复制到 CPU，后台保存到请求专属目录，完成后通过 collective RPC 返回小型 descriptor。等复合流水真正接入 Scheduler，再增加 Worker-local handle 和跨阶段零拷贝复用。

#### Q13 补充：只让一个 rank 保存，是否仍然存在传输问题？这是否是非共置场景也存在的问题？

是。指定 TP0 为 output owner 不能消除所有传输，它只明确了所有权并避免重复输出。至少可能存在三类数据移动：

1. **Rank 间汇聚**：如果最终 tensor 在 TP/SP/VAE parallel ranks 间分片，必须先 gather/reduce 到 output owner，或者由各 rank 写 shard 后再重组；如果每个 rank 已经持有完整复制结果，则不需要这一步，只需非 owner rank 丢弃副本。
2. **GPU 到 CPU/编码器**：使用 CPU encoder 或普通文件 I/O 时，TP0 仍需执行 D2H；若使用 GPU encoder/NVENC，可以减少原始帧 D2H，但编码后的 bitstream 最终仍需进入网络或存储。
3. **Worker 到消费者/存储**：最终 artifact 仍需写入本地盘、共享文件系统、对象存储或发送给客户端。

这些都是模型 serving 的输出物化问题，并不是 co-location 特有问题。即便只部署一个 Generator，也必须决定并行输出如何汇聚、由谁编码以及如何传给客户端。

co-location 会放大但不会创造这个问题：

- 大输出 tensor 保留时间过长会挤占与 LLM KV/cache 共享的显存；
- gather、D2H 和编码会与共置模型竞争 copy engine、内存带宽、CPU 和 pinned memory；
- 输出队列阻塞可能延长 Worker 持有请求资源的时间；
- 需要让 artifact 导出也进入统一 admission/backpressure。

因此设计目标不是“零传输”，而是：

- 只做算法与输出语义要求的必要传输；
- 明确 output owner，避免各 rank 重复保存；
- 大数据走 data plane，控制面只传 descriptor；
- 能在同一 Worker 内继续消费的中间 tensor 尽量用 handle/event 复用，不提前 D2H 或落盘；
- 将 gather、导出和释放纳入可观测的生命周期。

在当前 Cosmos3 路径正式实现 artifact writer 前，还需要先用实际输出布局确认 VAE 输出是 TP0-only、各 rank replicated，还是某个维度 sharded；不能仅凭“TP0 保存”假设不存在 rank 间传输。

## 6. 并行组接入

### Q14：当前 TP group override 不够理想，未来是否应改为独立 TP group，并映射到 model-level communicator？

是。需要先澄清：当前 Generator 已经拥有物理上独立的 `GroupCoordinator`/NCCL communication domain；`use_tensor_model_parallel_group()` override 不是临时创建 communicator，而是在执行期间把遗留的模块级 `get_tp_group()` 查询路由到 Generator group。问题在于访问方式仍依赖隐式全局接口。

最终 communicator 应绑定到一个模型部署实例或 `RunnerSlot`，而不是绑定到进程全局 `_TP`，也不应按每个请求重新创建：

```text
ModelDeployment / RunnerSlot
  └─ ParallelExecutionContext
       ├─ world_group
       ├─ tp_group
       ├─ pp_group
       ├─ dp_group
       ├─ sp/cfg groups
       └─ collective ordering metadata
```

Engine/Executor 负责生成 topology/placement plan；每个 Worker 的 `CommunicationGroupRegistry` 按 group ID 创建或复用本地 coordinator；Runner 注册时拿到稳定的 `ParallelExecutionContext`，模型层和 collective op 使用其中的 group。该 group 生命周期应与模型 replica/RunnerSlot 一致，而不是与单次请求一致。

最理想且 compile-friendly 的方式是在模型/层构造时显式注入 communicator，例如模块保存 `self.tp_group`，collective 直接使用该成员。这样：

- Reasoner 与 Generator 可以在同一线程或不同线程并发，不依赖修改 `_TP`；
- `torch.compile` 不必追踪 `ContextVar` 或全局状态切换；
- CUDA Graph capture 时 communicator 与 graph variant 的绑定明确；
- 多个辅助 Runner 可以各自拥有不同的 group。

但 vLLM 和 Omni 现有大量代码直接调用模块级 `get_tp_group()`，一次性改完成本较高。因此合理迁移路径是：

1. 保留当前 scoped override 作为 legacy compatibility shim；
2. 新增 Worker-owned `CommunicationGroupRegistry` 和 Runner-level `ParallelExecutionContext`；
3. Adapter/provider 先让 legacy `get_tp_group()` 从当前 Runner context 解析；
4. 逐步把关键模型层和 collective op 改为构造时显式接收 group；
5. 当所有并发路径不再读取模块全局 `_TP` 后，删除 override。

仅把 `_TP` 永久改成 Generator group 是错误的，因为 Reasoner 会拿错 communicator；在线程之间反复交换模块全局 `_TP` 同样有竞态。当前 eager 路径使用 ContextVar、compile 路径使用受限 static override，只是为了越过现有 API 边界，并不代表最终设计。

还要注意：独立 communicator 解决的是所有权与错误路由，不自动解决多个 communicator 在同一批 ranks 上并发 collective 的跨 rank 入队顺序。未来仍需 scheduler/turnstile 对冲突 collective 建立一致顺序。

### Q15：之前 profile 的计时是否包含通信部分？

需要按实验和指标区分。

#### TP=2 Worker-owned Gate 的 host timing

包含模型执行关键路径上的 TP/NCCL 通信。Reasoner 的 `_run_reasoner()` 在 `reasoner.generate()` 后调用 `torch.cuda.synchronize()`，因此结束时间晚于该设备上 Reasoner 计算和 collective 完成。Generator 的 `execute_auxiliary_runner_timed()` 会让每个 TP rank 执行 forward 及其 collectives，然后调用 `slot.adapter.synchronize_execution_stream()`，再记录 `host_end`。所以 `reasoner_alone_seconds`、`generator_alone_seconds`、concurrent durations 和 `concurrent_host_wall_seconds` 都包含运行期 collective 造成的执行和等待。

但 timed path 特意不返回 raw image/video tensor，因此不包含大媒体 payload 通过 `collective_rpc` 序列化和传回控制进程的时间。它测的是模型计算、模型内部通信、stream/event 等待和必要的 host orchestration，不是最终 artifact 导出的完整服务延迟。

#### 早期 TP=1 实验

TP group 是 singleton，没有真正的跨 rank TP collective。因此这些数据不能用于判断通信干扰或通信扩展性。只有最新 TP=2（以及以后更高并行度）结果才包含实际跨 GPU 通信。

#### Warmup 与 communicator 初始化

当前正式测量在 warmup 后进行，所以运行期每轮 collective 包含在内，而首次 NCCL communicator lazy initialization、首次 kernel/JIT 等启动成本通常已被 warmup 吸收。这是 steady-state latency，不是 cold-start latency。

#### Nsys 指标

- NVTX 区域 wall time 包含区域内计算、NCCL、stream wait、host gap 和调度停顿；
- 如果查询该区域内全部 CUDA kernels，NCCL kernel 也会被统计；
- 如果分析脚本只筛选 attention/GEMM/diffusion/VAE kernel，筛选后的 kernel sum 可能不含 NCCL，即使区域 wall 已经包含其影响；
- kernel sum 不能直接等同于端到端时间，因为不同 stream 的 kernel 可以重叠。

因此当前性能表中的 host wall/slowdown 是“包含模型内部通信的稳态执行时间”；但若要单独量化通信，需要在 Nsys SQLite 中把 `nccl*` kernel、collective API、通信 stream 和 compute stream 分组统计，并比较 alone 与 colocated。
