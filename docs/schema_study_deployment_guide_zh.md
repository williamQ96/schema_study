# 部署指南：Docker Hub → Apptainer → Mercury

[English](schema_study_deployment_guide.md) | **简体中文**

本指南使用公开仓库 `williamQ96/schema_study` 的目录结构。日常输入准备和 packet 操作见[使用指南](schema_study_user_guide_zh.md)。所有 shell 示例按 Linux/Bash 编写。

## 1. 部署内容及机器要求

| 项目 | 本版约定 |
| --- | --- |
| OCI 镜像 | `docker.io/plalelab/schema-study:0.1.0-cuda13`，Linux amd64 |
| 容器用户空间 | Ubuntu 24.04，CUDA 13.0.2 / cuDNN |
| 目标宿主 | Mercury：RHEL 9.8，4×H200 NVL 141GB，128 CPU 核，约 1.1TiB RAM |
| 包含 | 四类流程源码、prompt/schema、解析与推理依赖、启动入口 |
| 外部提供 | PDF/layout/input、dataset、模型 checkpoint、配置、API 凭据、输出目录 |

Mercury 配置记录的是用户提供的目标规格；是否可用，以现场 `doctor` 为准。CUDA 13 要求 NVIDIA 驱动 major 至少 580；这只是必要条件，仍需实际 CUDA 分配与模型测试。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

宿主不需要 Docker daemon：Apptainer 可直接拉取 Docker Hub OCI 镜像并转成 SIF。安装 Apptainer、驱动及站点调度配置由机器管理员按本地策略完成，本仓库不假定已有 Slurm 或特定 partition。

## 2. 取得固定版本的源码与 SIF

```bash
git clone https://github.com/williamQ96/schema_study.git
cd schema_study
git checkout v0.1.0
PROJECT="$(pwd)"
# 注意：启动器的 REPO 参数是 Python package 目录，不是 clone 根目录。
REPO="$PROJECT/high_fidelity_schema_study"
WORK="$HOME/schema-study-work"
mkdir -p "$WORK"/{images,sources,outputs,models,cache}
SIF="$WORK/images/schema-study-0.1.0-cuda13.sif"
SOURCES="$WORK/sources"
OUTPUTS="$WORK/outputs"
MODELS="$WORK/models"
CACHE="$WORK/cache"

apptainer pull "$SIF" docker://docker.io/plalelab/schema-study:0.1.0-cuda13
sha256sum "$SIF" > "$SIF.sha256"
git rev-parse HEAD > "$WORK/source-commit.txt"
```

正式实验建议把 release 中的 OCI digest 替换进拉取地址：`docker://docker.io/plalelab/schema-study@sha256:...`。OCI digest 标识镜像内容；SIF SHA-256 标识实际转换后的文件，两者不能互换。相同 OCI 的重复转换可能得到不同 SIF 字节，因此应保留实际使用的 SIF，并用其哈希冻结实验。[Apptainer OCI 文档](https://apptainer.org/docs/user/main/docker_and_oci.html)

为了使运行代码可审计，启动器会用指定 `$REPO` 只读覆盖镜像内代码。固定 Git commit 后再 compose/freeze；改源码后使用新实验版本。原研究仓库中 compose 的配置不能直接套用本公开包：源码集合及哈希不同。

## 3. 挂载约定

| 宿主变量 | 容器路径 | 权限/内容 |
| --- | --- | --- |
| `REPO` | `/workspace/high_fidelity_schema_study` | 只读，Python package 源码 |
| `SOURCES` | `/inputs` | 只读，PDF/layout/input、原始 dataset、解析 bundle、corpus 和配置 |
| `OUTPUTS` | `/outputs` | 可写，解析结果、批次、packet、部署报告 |
| `MODELS` | `/models` | 只读，已下载的完整模型 snapshot |
| `CACHE` | `/cache` | 可写，运行缓存/offload |

所有目录必须先存在；输出、缓存和只读来源使用不同目录。路径不能含 Apptainer bind 语法中的冒号、逗号或换行。Dataset 已有输入接口，但不嵌入镜像；`run` 不会自动发现或解析未登记的 dataset。

```bash
launch() {
  bash "$PROJECT/container/apptainer/run.sh" \
    "$SIF" "$REPO" "$SOURCES" "$OUTPUTS" "$MODELS" "$CACHE" -- "$@"
}

# 合成材料、模拟后端；不下载/加载模型、不访问 API。
MERCURY_CPU_ONLY=1 launch offline-demo --output /outputs/software-smoke-v1
```

启动器使用 `--cleanenv --containall`，仅转发允许的运行配置和指定的 secret 环境变量。`MERCURY_DRY_RUN=1` 可以显示实际启动参数而不运行容器。

## 4. GPU 预检

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
# 已由调度器设置 CUDA_VISIBLE_DEVICES 时直接继承，不覆盖。
# 仅在已分配四卡、且调度器未设置该变量时使用下一行：
# export MERCURY_CUDA_DEVICES=0,1,2,3
launch doctor --gpu-count 4 --output /outputs/doctor-mercury-v1.json
```

GPU 模式要求显式设备选择。`MERCURY_CUDA_DEVICES` 与调度器的 `CUDA_VISIBLE_DEVICES` 冲突会拒绝启动。单卡/双卡对应 `--gpu-count 1/2` 和选定的可见设备数。GPU 透传使用 Apptainer `--nv`。[GPU 文档](https://apptainer.org/docs/user/main/gpu.html)

Doctor 记录宿主与容器 OS、NVIDIA-SMI 物理设备、Torch 可见设备、CUDA runtime、驱动、显存、拓扑和小规模分配/同步检查。CPU 检查可用 `MERCURY_CPU_ONLY=1 launch doctor --gpu-count 0 ...`；它不能证明 GPU 已通过。Doctor 不加载模型，也不能保证长上下文不会 OOM。

## 5. 选择模型：无需修改 Python 模型名单

先复制配置到宿主 sources，再编辑副本：

```bash
mkdir -p "$SOURCES/checkpoints"
cp "$REPO/config/mercury_models.example.json" "$SOURCES/models.mercury-v1.json"
cp "$REPO/config/mercury_selection.example.json" "$SOURCES/selection.mercury-v1.json"
```

| 文件/字段 | 必须填写的内容 |
| --- | --- |
| catalog 每个 profile | 模型 ID、不可变 revision（可获取时）、部署/endpoint、真实 context window、参数能力、量化/模板 |
| Transformers `model_id` | 容器内 `/models/<snapshot>`，模型提前下载；正式运行禁止隐式 Hub 下载 |
| `runtime.checkpoint_manifest` | 完整 checkpoint 清单路径和文件字节 SHA-256 |
| selection `locals` | 三个本地 profile ID |
| `soft_reference` | frontier profile ID；仍是可错参照 |
| `classifier` | 独立 profile ID 及 catalog profile hash，可以复用本地 A 的部署 |
| `resource_preset` | `h200_1gpu`、`h200_2gpu` 或 `h200_4gpu` |
| `replicates` / `parameters` | 重复 seed 列表及各角色参数，省略时继承 base |

为每个实际使用的本地 checkpoint 建清单。清单输出必须在模型目录之外：

```bash
MERCURY_CPU_ONLY=1 launch checkpoint \
  --model-path /models/model-a-snapshot --revision ACTUAL_IMMUTABLE_REVISION \
  --output /outputs/model-a.manifest.json
cp "$OUTPUTS/model-a.manifest.json" "$SOURCES/checkpoints/"
sha256sum "$SOURCES/checkpoints/model-a.manifest.json"
```

把输出哈希写入对应 profile：

```json
"checkpoint_manifest": {
  "path": "/inputs/checkpoints/model-a.manifest.json",
  "file_bytes_sha256": "这里填写 sha256sum 的 64 位十六进制结果"
}
```

对 B/C 和独立 classifier 的实际 checkpoint 重复。清单读取所有模型文件字节；HF snapshot 符号链接的目标也必须在容器内可读。文件新增、遗漏或改动都会使验证失败。

真实身份、参数能力和上下文完成 qualification 后，把选中 profile 状态改为 `frozen`。再运行：

```bash
MERCURY_CPU_ONLY=1 launch catalog --catalog /inputs/models.mercury-v1.json
```

把选定 classifier 输出的 `profile_sha256` 填入 selection。更换主模型不会自动更换 classifier；修改 catalog 后应重新取 hash。

默认抽取参数为 temperature 0.7、top_p 0.8、top_k 20、repetition_penalty 1.0、max_output_tokens 16384，三个 seed 为 1729/2718/3141；分类器示例为 greedy。参数是实验条件，不是机器性能推荐。selection 中的角色参数对象为**整组替换**，要写全实际需要的参数。

后端支持 Transformers、兼容 Chat Completions 和 Responses。若新模型架构不被镜像内 Transformers 支持，需要新依赖镜像；若供应商协议不同，需要新 adapter。兼容接口不意味着 seed、top_k、reasoning 等参数等价，不支持的设置会被拒绝。HTTP 还需配置服务对应的精确 token-count endpoint，或与实际请求哈希绑定的 token-count observation；未知计数不能绕过上下文门禁。

## 6. Compose 与冻结

```bash
MERCURY_CPU_ONLY=1 launch compose \
  --base /workspace/high_fidelity_schema_study/config/four_category_experiment_v1.json \
  --catalog /inputs/models.mercury-v1.json \
  --selection /inputs/selection.mercury-v1.json \
  --host /workspace/high_fidelity_schema_study/config/mercury_host_v1.json \
  --output /outputs/experiment.draft-v1.json \
  --report /outputs/compose.draft-v1.json
```

审阅 draft 后用新文件名冻结：

```bash
SIF_SHA256="$(sha256sum "$SIF" | awk '{print $1}')"
MERCURY_CPU_ONLY=1 launch compose \
  --base /workspace/high_fidelity_schema_study/config/four_category_experiment_v1.json \
  --catalog /inputs/models.mercury-v1.json \
  --selection /inputs/selection.mercury-v1.json \
  --host /workspace/high_fidelity_schema_study/config/mercury_host_v1.json \
  --freeze --image-sha256 "$SIF_SHA256" \
  --output /outputs/experiment.frozen-v1.json \
  --report /outputs/compose.frozen-v1.json
```

占位模型、缺少 classifier pin、未冻结 profile、不支持参数、过小 context 或缺少 checkpoint 清单均阻断 freeze。静态 `configuration_ready` 不代表运行已经通过；正式 `run` 再查 SIF、源码、checkpoint 全部字节、GPU 及每个请求上下文。

## 7. 启动与续跑

完成[使用指南](schema_study_user_guide_zh.md)的 sources/corpus 准备后，以下命令才会触发真实推理和可能收费的 API：

```bash
# 通过自己的凭据管理方式先设置 OPENAI_API_KEY；不要写入 JSON 或镜像。
export MERCURY_SECRET_ENV_NAMES=OPENAI_API_KEY
launch run --config /outputs/experiment.frozen-v1.json \
  --corpus /inputs/corpus-v1.json --source-root /inputs \
  --output /outputs/batch-v1 --allow-live
```

首次可追加 `--max-jobs 1` 做有界运行；完成后同一命令继续同一个 batch 目录。续跑重验部署并复用通过 replay 验证的结果，摘要保存在 `batch-v1/batches/*.json`。Mercury `run` 没有 `--summary-output` 参数。

运行 packet/verify 或 dataset 准备时切回 `MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli`，CPU 操作用 `MERCURY_CPU_ONLY=1`。缺少独立人工评价不阻止 packet 生成；语义准确率保持未评估。

## 8. 自行重建镜像

在 clone 根目录、联网 Linux Docker 环境运行：

```bash
docker build --platform linux/amd64 \
  -f container/docker/Dockerfile \
  --build-arg IMAGE_VERSION=0.1.0 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t schema-study:local .
```

镜像内 `/opt/requirements/pip-freeze.txt` 记录解析后的 Python 依赖；本次发行的副本为 [pip-freeze-0.1.0-cuda13.txt](../container/docker/pip-freeze-0.1.0-cuda13.txt)。requirements 的部分版本范围和 base tag 允许未来解析结果变化；严格复现保留实际镜像 digest/SIF 及该依赖清单。更换 CUDA base 不会自动改变 pip PyTorch wheel 的 CUDA 版本。

也可从配方直接构建仅含运行库的 SIF，然后仍用启动器挂载源码：

```bash
bash "$PROJECT/container/apptainer/build.sh" "$WORK/images/schema-study-local.sif"
```

构建会联网安装依赖，不下载权重；需要站点允许的 Apptainer 构建权限。脚本不会自行升级权限。直接构建产生新的 SIF 哈希，须重新 freeze。

## 9. 常见故障与边界

| 症状 | 处理 |
| --- | --- |
| `No module named high_fidelity_schema_study` | 确认 REPO 指向 clone 内同名 package，而不是 clone 根目录 |
| GPU selection 缺失/冲突 | 使用调度器分配的可见设备；不要另设冲突值 |
| `actual_sif_identity_mismatch_or_missing` | 使用冻结时的 SIF 和提供实际哈希的启动器；新 SIF 建新配置 |
| `active_source_identity_mismatch` | checkout 冻结版本；不要把研究目录中的旧冻结配置复用于公开源码包 |
| checkpoint identity 不匹配 | 检查全部模型文件和 symlink 目标，按新版本重建清单 |
| 请求 token count 不可得 | 核实供应商计数接口或提供精确请求绑定的 observation |
| dataset 文件不可读 | 文件放入 SOURCES，corpus 引用相对路径，输出写 /outputs |
| 输出已存在 | 单次产物换新版本名；批次续跑使用原 batch 目录 |

当前 runner 串行调度，Transformers 每次调用后释放模型；不启动 vLLM、不提供 tensor parallel 或跨节点 InfiniBand/NCCL 配置。`device_map=auto` 为层放置，每卡 120GiB 是权重放置预算，不含完整 KV cache/激活峰值，也不是显存硬限制。4×141GB 不能当成单一 GPU 显存空间。[Accelerate 大模型推理说明](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference)

容器软件验证、Mercury 现场资格验证、真实实验及独立抽样语义评价分别记录。当前发行不宣称后面三项已经完成。
