# RTX 5090 / CUDA 13.2 最小环境迁移与 Green Context Workqueue 验证

本文用于把当前 Cosmos3 单进程、单框架实验迁移到新的 RTX 5090 服务器。目标不是复制旧容器，而是以最少组件重建可复现环境，并优先验证 CUDA 13.1 起增加的 Green Context workqueue 资源能力。

## 1. 迁移边界

必须迁移：

- `vllm-single-framework`：vLLM 主仓库和单框架接入代码；
- `vllm-omni`：保留 Omni `DiffusionModelRunner`、Cosmos3 pipeline 与最小资源接口补丁；
- `nvidia/Cosmos3-Edge` 权重；
- 一份固定 Reasoner 输入和一份固定 T2V workload manifest；
- 精确的软件版本与验证命令。

不应直接复制：

- 旧 `.venv` 或 `.runtime`。虚拟环境包含绝对路径，并且容易带入旧 CUDA 动态库；
- `outputs/`、profiling 结果、core dump；
- 整个 Hugging Face cache。只复制 Cosmos3-Edge snapshot，或者在新机器重新下载；
- 旧容器中手工安装且未记录的系统包。

CUDA 13.2 驱动能够运行 cu130 构建的 PyTorch/vLLM。第一阶段因此继续使用已经验证过的 cu130 Python 二进制栈，仅用新主机的 R595 驱动和 CUDA 13.2 Toolkit 验证 workqueue。不要一开始同时升级 PyTorch、vLLM、Omni 和 Transformers，否则无法区分环境问题与 workqueue 行为。

## 2. 当前代码存档状态

### 2.1 vLLM 单框架仓库

仓库：`miceforrat/vllm`

分支：`experiment/omni-model-backend`

当前已提交关键点：

```text
f9f5e2fe55 [Docs] Summarize single-process Omni runtime experiments
fd783be9ce [Experimental] Embed Omni diffusion runner in vLLM runtime
```

当前仍未提交：

```text
examples/offline_inference/cuda_green_context_workqueue_probe.cu
outputs/
```

其中探针必须提交；`outputs/` 不应提交。

### 2.2 vLLM-Omni 仓库

仓库：`miceforrat/vllm-omni`

分支：`experiment/cosmos3-dual-framework`

当前关键提交：

```text
4fdf39e15 [Experimental] Expose diffusion cache memory providers
7d1efa856 pin transformers 5.17 for Cosmos3 dual-engine runtime
```

该仓库当前干净，Omni 侧补丁已经提交。

### 2.3 cosmos-framework 仓库

分支：`experiment/vllm-omni-single-framework`

当前仍有以下未跟踪项：

```text
.runtime/
core
profiling_tools/docs/images/
profiling_tools/runners/single_process_dual_engine_matched.py
```

`.runtime/` 和 `core` 不应提交。后两个目录/文件只有在需要保留旧双框架实验时才单独审查和提交；它们不是新机器启动单框架路径的必要条件。

## 3. 当前推荐路线

新服务器暂时无法访问 Docker Hub，当前先采用宿主机路径：

1. 使用宿主机 CUDA 13.2 Toolkit 运行 standalone workqueue probe；
2. probe 通过后，创建独立的用户态 Python venv；
3. venv 中继续使用已验证的 cu130 PyTorch/vLLM 二进制栈；
4. 不修改系统 Python，不向系统 site-packages 安装包；
5. Docker 网络恢复后，再把同一流程迁入容器。

`registry-1.docker.io:443 connect: connection refused` 表示 registry 网络、DNS、代理或防火墙问题，并不能证明镜像 tag 不存在。宿主机方案会暂时失去容器级系统依赖隔离，但足以验证 GC workqueue 和完成第一轮 matched workload。

## 4. 新服务器的宿主机验收

先确认驱动、GPU 和 Docker GPU runtime：

```bash
nvidia-smi
docker version
docker info | grep -i runtime
```

首先确认宿主机 Toolkit：

```bash
which nvcc
nvcc --version
ls -ld /usr/local/cuda*
ldconfig -p | grep -E 'libcuda|libcudart' | head -20
```

必须确认 RTX 5090 正常可见、驱动支持 CUDA 13.2、`nvcc` 为 CUDA 13.1/13.2，并且 `libcuda.so.1` 来自驱动。

如果 Docker registry 恢复可达，再运行纯 CUDA 容器测试。镜像的具体 patch tag 以服务器可拉取的 CUDA 13.2 devel tag 为准：

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.2.0-devel-ubuntu22.04 \
  nvidia-smi
```

Docker 验收条件：

- GPU 显示 RTX 5090；
- 驱动属于支持 CUDA 13.2 的 R595 系列或更高；
- 容器内可以看到 GPU；
- `/usr/local/cuda/bin/nvcc --version` 显示 CUDA 13.2。

### 4.1 Docker Hub 故障诊断与离线导入

```bash
getent hosts registry-1.docker.io
curl -Iv --connect-timeout 10 https://registry-1.docker.io/v2/
env | grep -i proxy
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'
```

`curl` 返回 `401 Unauthorized` 表示 registry 网络可达；`connection refused` 或 timeout 表示链路仍未打通。shell 代理不一定会被 Docker daemon 使用。

离线镜像可以在联网机器用 `docker save` 导出，并在新服务器用 `docker load` 导入。

## 5. Docker 路径（网络恢复后的推荐方案）

为了编译 workqueue 探针，使用 `devel` 而不是 `runtime` 镜像：

```bash
docker run -it \
  --name cosmos3-cu132 \
  --gpus all \
  --network host \
  --ipc host \
  --cap-add SYS_ADMIN \
  --security-opt seccomp=unconfined \
  -v /data/cjh/cosmos3-workspace:/workspace \
  -v /data/cjh/cosmos3-cache:/root/.cache \
  nvidia/cuda:13.2.0-devel-ubuntu22.04 \
  bash
```

说明：

- `SYS_ADMIN` 主要为 Nsys/Ncu profiling 预留；只跑功能验证时可以去掉；
- 新服务器目录可以改变，但容器内统一映射为 `/workspace`，减少脚本路径改动；
- 如果不复制旧 cache，就先不挂载第二个目录，在新服务器下载后再固定 snapshot。

容器内安装最小系统工具：

```bash
apt-get update
apt-get install -y git git-lfs curl build-essential ca-certificates
git lfs install

curl -fsSLo /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p /opt/conda
source /opt/conda/etc/profile.d/conda.sh

conda create -n vllm-omni-co-cu132 python=3.13 pip -y
conda activate vllm-omni-co-cu132
python -m pip install uv
```

## 6. 拉取代码

```bash
mkdir -p /data/cjh/cosmos3-workspace
cd /data/cjh/cosmos3-workspace

git clone git@github.com:miceforrat/vllm.git vllm-single-framework
cd /data/cjh/cosmos3-workspace/vllm-single-framework
git checkout experiment/omni-model-backend

cd /data/cjh/cosmos3-workspace
git clone git@github.com:miceforrat/vllm-omni.git vllm-omni
cd /data/cjh/cosmos3-workspace/vllm-omni
git checkout experiment/cosmos3-dual-framework
```

固定并记录实际提交：

```bash
git -C /data/cjh/cosmos3-workspace/vllm-single-framework rev-parse HEAD
git -C /data/cjh/cosmos3-workspace/vllm-omni rev-parse HEAD
git -C /data/cjh/cosmos3-workspace/vllm-single-framework status --short
git -C /data/cjh/cosmos3-workspace/vllm-omni status --short
```

## 7. 使用 Conda 重建隔离的 Python 栈

旧环境的已知版本为：

```text
Python       3.13
torch        2.13.0+cu130
torchvision  0.28.0+cu130
vllm         0.31.0+precompiled
vllm-omni    0.31.0rc2.dev288
transformers 5.17.0
triton       3.7.1
flashinfer   0.7.0.post1
```

推荐继续使用 cu130 Python 栈。CUDA 13.2 驱动向后兼容 cu130 用户态二进制；不要把 `/usr/local/cuda-13.2/lib64` 强行放到 `LD_LIBRARY_PATH` 前面覆盖 wheel 自带依赖。

先确认 Conda：

```bash
conda --version
```

如果宿主机没有 Conda，可以无 root 安装 Miniconda：

```bash
curl -fsSLo /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version
```

如果当前 shell 尚未初始化 Conda：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

创建固定名称的环境：

```bash
conda create -n vllm-omni-co-cu132 python=3.13 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-omni-co-cu132

python -m pip install --upgrade pip
python -m pip install uv
```

Conda 负责环境隔离；环境内使用 `uv pip` 是为了明确选择 cu130 PyTorch backend：

```bash
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=20

uv pip install --python "$(which python)" --torch-backend=cu130 \
  "vllm==0.31.0" \
  "transformers==5.17.0"

uv pip install --python "$(which python)" --no-deps \
  -e /data/cjh/cosmos3-workspace/vllm-omni
```

不要复制旧 `.venv`，不要使用 Conda base 环境，也不要执行 `sudo pip install`。如果 Omni import 报缺包，根据旧环境的 freeze 补齐依赖，不要升级核心版本。

验证并记录环境：

```bash
python - <<"PY"
import importlib.metadata as m
import torch

for name in ("torch", "torchvision", "vllm", "vllm-omni", "transformers", "triton", "flashinfer-python"):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, "NOT INSTALLED")

print("torch CUDA:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

mkdir -p /data/cjh/runtime/vllm-omni-co-cu132
python -m pip freeze > /data/cjh/runtime/vllm-omni-co-cu132/freeze.txt
conda env export --no-builds > /data/cjh/runtime/vllm-omni-co-cu132/environment.yml
```

## 8. 通用运行环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-omni-co-cu132

export PYTHONPATH=/data/cjh/cosmos3-workspace/vllm-single-framework:/data/cjh/cosmos3-workspace/vllm-omni
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_DEEP_GEMM=0

# 权重已经复制到本地 cache 后再启用。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 先保持旧实验基线；后续单独消融。
export CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_SCALE_LAUNCH_QUEUES=4x
```

`CUDA_SCALE_LAUNCH_QUEUES=4x` 是此前实验中的调优项，不是 correctness 要求。首次迁移验证也应补跑 unset/default 对照。

## 9. Gate 0：优先验证 GC workqueue

这一步不加载模型，只验证新驱动是否真正接受 workqueue resource type，以及两个 Green Context 能否各自建立 balanced workqueue。

```bash
cd /data/cjh/cosmos3-workspace/vllm-single-framework

nvcc \
  -std=c++17 \
  -O2 \
  -arch=sm_120 \
  examples/offline_inference/cuda_green_context_workqueue_probe.cu \
  -o /tmp/cuda_green_context_workqueue_probe

/tmp/cuda_green_context_workqueue_probe \
  | tee /data/cjh/workqueue_probe_cuda132.log
```

预期结果：

- SM resource query 成功；
- workqueue config query 成功，不再返回 `cudaErrorInvalidResourceType (914)`；
- 能创建两个带 `CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED` 的 GC descriptor；
- 两个 stream 上的验证 kernel 完成；
- 最终打印 `PASS`。

如果仍返回 914，首先记录以下信息，不要先改框架：

```bash
nvidia-smi
nvcc --version
ldd /tmp/cuda_green_context_workqueue_probe | grep -E 'cuda|cudart'
```

这可以区分“Toolkit 有头文件”与“实际加载的驱动不支持该 resource type”。

## 10. Gate 1：最小模型加载

```bash
cd /data/cjh/cosmos3-workspace/vllm-single-framework

python -u examples/offline_inference/omni_runner_config_gate.py \
  --model nvidia/Cosmos3-Edge \
  --load-format default
```

验收：

- 使用原生 Omni `DiffusionModelRunner`；
- pipeline 为 Cosmos3；
- `resource_owner` 为 `vllm`；
- 无第二个 Python worker 进程；
- 模型加载后显存与旧服务器同量级。

## 11. Gate 2：单进程双模型串行

先使用小负载排除迁移错误：

```bash
python -u examples/offline_inference/omni_runner_dual_model_gate.py \
  --model nvidia/Cosmos3-Edge \
  --reasoner-mode graph \
  --reasoner-max-tokens 256 \
  --height 256 \
  --width 256 \
  --num-frames 1 \
  --steps 2 \
  --warmups 1 \
  --repeats 1 \
  --output /workspace/outputs/dual_model_cuda132_smoke.json
```

随后再换成固定 Reasoner 输入和此前 matched T2V manifest。当前脚本仍引用旧 Cosmos workspace 里的 workload 文件，因此迁移前应把这两个小型 JSON 复制到 `vllm-single-framework/examples/offline_inference/workloads/`，消除对整个 cosmos-framework 输出目录的依赖。

## 12. Gate 3：把 workqueue 接入 GC runtime

只有 Gate 0、1、2 都通过后才做这一项。

建议按以下顺序实现：

1. 在 `vllm/experimental/green_context.py` 的 GC descriptor 中同时加入 SM resource 与 workqueue resource；
2. 为 Reasoner GC 和 Generator GC 各创建一个 balanced workqueue；
3. 保持现有 GC stream record/rebind 逻辑不变；
4. 在报告里记录 workqueue scope、descriptor 创建结果和 stream handle；
5. 先跑 VAE decode × LLM decode；
6. 再跑 diffusion sampling × LLM decode；
7. 与旧服务器上的同负载、同 SM 分区结果对照。

必须保留三组对照：

```text
A. GC + 无显式 workqueue
B. GC + balanced workqueue
C. full-SM 串行基线
```

不要只看共置 makespan。至少报告：

- full-SM alone；
- partitioned alone；
- colocated；
- 两端各自 slowdown；
- 实测 serial host wall；
- concurrent host wall；
- Nsys 中的 GPU kernel overlap 与 stream/context 归属。

## 13. 迁移前需要提交的最小白名单

在旧机器上由用户执行：

```bash
cd /data/cjh/cosmos3/cosmos3-workspace/vllm-single-framework

git add \
  examples/offline_inference/cuda_green_context_workqueue_probe.cu \
  docs/design/rtx5090_cuda132_minimal_bootstrap.md

git commit -m '[Experimental] Add CUDA GC workqueue migration probe'
git push
```

确认没有误提交运行产物：

```bash
git status --short
git diff --cached --stat
```

`outputs/` 保留在本机或单独 rsync，不进入 Git。

## 14. 最小成功标准

迁移完成不等于“容器启动成功”。本阶段的验收标准是：

1. 两个仓库能从远端精确 checkout；
2. cu130 Python 栈在 CUDA 13.2/R595 主机上加载 RTX 5090；
3. Cosmos3-Edge 单进程双模型 smoke test 通过；
4. standalone workqueue probe 打印 `PASS`；
5. GC runtime 接入 workqueue 后，Nsys 证明两个 workload 仍在各自 GC stream/context 上；
6. 用 matched workload 量化 workqueue 是否把 VAE/LLM 或 sampling/LLM 的干扰降回旧基模附近。

