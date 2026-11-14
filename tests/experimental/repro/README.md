# 复现脚本（在 5090 上运行；全部基于开发者自己的引擎和 smoke 部署计划）

这些脚本**不修改仓库里的任何文件**。它们 import 开发者的
`examples/offline_inference/omni_async_multi_model_engine_smoke.py` 里的 `deployment_plan()`，
用 `AsyncMultiModelEngine` 起引擎（参数与该 smoke 相同），再通过
`engine.collective_rpc(可调用对象)` 把探针（`_probes.py`）送进 Worker 进程做 monkeypatch 和观测。
所以环境依赖与开发者的 smoke 完全一致。

> 注意：这些脚本在我们这边**没有 GPU、没有运行过**，是按 ffb34c2 / 5445e40 的代码静态写成的。
> 第一次运行若遇到属性名不符之类的小问题，按报错改一两行即可；判定逻辑不受影响。

## 前提
- TP=1（UniProcExecutor：EngineCore 与 Worker 同进程；探针依赖这一点）。
- 环境变量 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`（`_common.py` 会自动设置；仅限本机调试）。
- 两个仓库分别在 `experiment/omni-model-backend@ffb34c2` 与 `experiment/cosmos3-dual-framework@5445e40`。

## 一览
| 脚本 | 对应问题（编号） | 时长量级 | 会不会把模型"弄坏" |
|---|---|---|---|
| `tests_proposed/test_c6_final_wave_admission.py` | C6 | 秒级，无需 GPU | 否 |
| `tests_proposed/test_c2_stream_order_gpu.py` | C2 | 秒级，需 GPU，无需模型 | 否 |
| `r1_c1_sample_stream.py` | C1（LLM 采样不在分区内） | 数分钟 | 否 |
| `r2_c2_variant_switch_order.py` | C2（变体切换乱序）+ C14 线索 | 数分钟 | 否 |
| `r3_c3_vae_feat_map.py` | C3（VAE 因果缓存跨请求） | 数分钟 | 否 |
| `r4_control_plane.py --test c7/c9/c10/c5/c4` | C7、C9、C10、C5、C4 | 每个 1–3 分钟 | c5、c4 在问题存在时会 |
| `r5_c6_lost_wakeup.py` | C6（真引擎） | 1–2 分钟 | 问题存在时 B 挂住直到超时 |
| `r6_c11_sync_points.py` | C11（隐式同步计数） | 1–2 分钟 | 否 |

## 运行
```bash
M=/path/to/Cosmos3-Edge
# 1) 不需要模型的两个单测（拷到 vllm 仓库 tests/experimental/ 下）
.venv/bin/python -m pytest tests/experimental/test_c6_final_wave_admission.py tests/experimental/test_c2_stream_order_gpu.py -v
# 2) 正确性
python r1_c1_sample_stream.py --model $M
python r2_c2_variant_switch_order.py --model $M
python r3_c3_vae_feat_map.py --model $M
# 3) 控制面（每个子测试单独起引擎）
for t in c7 c9 c10 c5 c4; do python r4_control_plane.py --model $M --test $t; done
python r5_c6_lost_wakeup.py --model $M
# 4) 性能线索
python r6_c11_sync_points.py --model $M --generator-cache none
python r6_c11_sync_points.py --model $M --generator-cache sea_cache
```
每个脚本最后都会打印一个框：
```
==============================================================================
[C2] RESULT: BUG PRESENT | NOT OBSERVED | INCONCLUSIVE
  detail: ……
  where to look / fix: 修复文档 C2 ……
==============================================================================
```

## 输出怎么对应到病灶
| 看到的输出 | 说明 | 病灶 | 改哪里（完整代码见 Cosmos3_GC_code_fixes_and_tests.md 第 4 节） |
|---|---|---|---|
| r1：`sample_on_other_stream > 0` | lm_head + 采样跑在 legacy default stream（整卡 SM），与 forward 无依赖 | C1 | `MultiModelWorker` 覆写 `sample_tokens`，经 `MiniContainer.run_on_active_assignment` 在同一变体 stream 执行 |
| r1：现状组有轮次与标准答案组（每步强制同步）不同 | 采样读到没写完的 hidden_states，输出已经出错 | C1 | 同上 |
| r1：模拟修复组各轮之间仍不一致 | 还有别的跨 stream 竞争（AsyncOutput 拷贝、postprocess） | C1 | 把 postprocess 也放进同一 slice（F2 的 ModelTimeline） |
| r2：现状视频哈希 ≠ 对照（sync）哈希，event-fix 后一致 | 切换变体时新 stream 没等旧 stream 上的连续 slice | C2 | `MiniContainer` 增加 per-model `last_done` event；`CudaStreamBinding._activate` 去掉经 caller 的依赖 |
| r2：latent 哈希就不同 | 去噪阶段有别的非确定性 | （先排查） | 关 SeaCache、固定 cuDNN 算法后重跑 |
| r2：reserved 显存在切换运行后明显增加 | 生成器在两条 stream 上各缓存一份激活 | C14 | 限制生成器使用的档位数；层块走 graph 固定内存池 |
| r2：`num_alloc_retries` 增长 | caching allocator 按 stream 分池，变体切换后显存碎片化触发重试（全设备同步） | C14 | 生成器激活值走固定 stream 的显存池，或切换前 `record_stream` |
| test_c2：`[0, 1, 2, 1]` | 同 C2 | C2 | 同上 |
| r3：latent 相同、视频不同 | VAE `_feat_map` 被同 wave 的另一个请求覆盖 | C3 | `prepare/run/finalize_vae_blocks` 每个请求持有自己的 feat_cache |
| r4 c7：注入错误的请求正常返回 | 失败被当作成功 | C7 | ExecUnit 上报 `status/error`，EngineCore 对失败请求 `set_exception` |
| r4 c9：A 没有 `seconds`，或 B 的 `seconds`≈A+B | 计时按驱动请求算 | C9 | Worker 对每个完成请求单独返回 `seconds` |
| r4 c10：取消后 slice 计数继续增长 | 取消没有传到 Worker | C10 | abort 路径（EngineCore→Worker→ExecUnit→runner） |
| r4 c5：B 报 `exceeded …`，而报出的 id 是 A | 预算记在驱动请求头上 | C5 | 按 wave 内每个请求各自计数 |
| r4 c4：坏请求之后正常请求报 `Workers returned no next slice` | 异常后残留状态，模型永久失效 | C4 | 提交时校验 variant；driver 异常时 `reset_model_requests` |
| r5 / test_c6：B 超时 / 断言失败 | admission 与"判定空闲"不在同一把锁 | C6 | 两段放进同一把 `_model_jobs_lock`，锁内重读 `pending_work_count` |
| r6：每个去噪 slice ≥1 次同步，top sites 指向 UniPC / SeaCache | 隐式 host 同步 | C11 | sigmas 常驻 GPU、SeaCache 判定在 GPU 上算并延迟一步读回 |

## 建议的修复顺序
C1 + C2（加 test_c2 回归）→ C4、C5、C6、C7、C9、C10（Worker 簿记整体替换，加 test_c6 回归）→
开 `max_num_seqs≥2` 之前修 C3 → 再看 C11、C14 这类性能项。
