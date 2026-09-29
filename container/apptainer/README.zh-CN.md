# Mercury：四类解析与可替换模型运行环境

这个入口运行当前 `four_category` 双来源流程。Paper 保留 layout-aware 全文和四类索引；dataset 由独立解析器处理，不进入模型请求。三个本地槽位和一个 frontier soft-reference 槽位分别选择 profile，分类器另行绑定。

**当前交付是代码、配方与离线验收。尚未在 mercury 构建 SIF、加载真实模型或执行实验。** 旧 `container/phase1`、OSC 脚本和历史冻结产物保持原样。

## 配置边界

| 文件 | 修改内容 |
| --- | --- |
| `config/mercury_models.example.json` | 任意模型 ID/本地目录、revision、后端、上下文、量化、chat template、能力声明、API endpoint |
| `config/mercury_selection.example.json` | 三个本地 profile、soft reference、独立 classifier ID/hash、重复次数及参数覆盖 |
| `config/mercury_host_v1.json` | mercury 的用户报告硬件，以及 1/2/4 卡资源预设 |
| `config/four_category_experiment_v1.json` | 共享任务、taxonomy、默认生成参数和既有流水线配置；作为输入读取 |

示例模型和上下文仍是占位值，不能用于正式冻结。复制为新文件后填写。换模型不需要修改 Python 中的模型家族列表；模型必须能被所选后端实际加载并通过该任务的 contract 检查。新架构可能仍需升级镜像中的 Transformers/依赖，或新增供应商 adapter。

选择文件中的参数对象是**整组替换**，不是隐式合并。默认本地抽取使用 `do_sample=true, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, max_output_tokens=16384`，seeds 为 1729/2718/3141。这些是可修改的研究配置，不是 H200 的最佳性能设置。示例分类器明确使用 greedy `do_sample=false`；共享提示内容未改变。

## 构建和挂载

在 Linux 上使用 Bash；所有命令中的路径均需替换为 mercury 上实际位置。脚本不假定存在 Slurm、某个 partition、管理员权限或某个网络文件系统。

```bash
REPO=/path/to/high_fidelity_schema_study
SIF=/path/to/images/four-category-mercury-v1.sif
SOURCES=/path/to/frozen-sources
OUTPUTS=/path/to/mercury-results
MODELS=/path/to/local-models
CACHE=/path/to/mercury-cache
mkdir -p "$SOURCES" "$OUTPUTS" "$MODELS" "$CACHE"
cd "$REPO"
bash container/apptainer/build.sh "$SIF"
```

默认 base 为当前工程使用的 `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`，运行依赖读取既有 `requirements-phase1-inference.txt` 和 `requirements_four_category_offline.txt`。可用 `CUDA_IMAGE=registry/image@sha256:...` 显式替换 base；需要依赖升级时使用新版本 requirements 和新 SIF，不改历史镜像。单独更换 CUDA base 并不会自动更换 pip 的 PyTorch CUDA wheel。

构建安装依赖需要网络，但不会下载模型。失败时脚本停止，不自动切换权限。镜像保存 `/opt/requirements/pip-freeze.txt`；外部 `SIF.manifest.json` 保存 SIF 字节 SHA-256、配方和 requirements 哈希、base 引用及 Apptainer 版本。默认 base tag 与版本范围依赖不保证重新构建字节一致；**实验绑定实际生成的 SIF 哈希**。后续严格复现应保留该 SIF，并锁定 OCI digest / resolved dependency lock。

RHEL 是宿主系统，Ubuntu 是容器用户空间。启动器在宿主 `/etc/os-release` 可用时将其只读挂载，doctor 分别记录 host/container 元数据；缺少挂载时 host 信息保留 unknown。CUDA 13 的 NVIDIA 驱动 major 下限为 580；这只是必要条件，实际 PyTorch CUDA runtime、GPU allocation probe 和所选模型仍要分别检查。参见 [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

脚本将源代码、输入和模型只读挂载为 `/workspace/high_fidelity_schema_study`、`/inputs`、`/models`，结果和缓存读写挂载为 `/outputs`、`/cache`。源代码不烘焙进镜像，正式配置绑定实际源文件字节哈希。source-root 应包含 corpus 引用的 PDF、layout/input v3、dataset bundle 及其原始来源。API 密钥只从显式环境变量传入，不放入配置文件或镜像。

## 先做零推理检查

定义一个便于复用的 shell 函数：

```bash
launch() {
  bash "$REPO/container/apptainer/run.sh" \
    "$SIF" "$REPO" "$SOURCES" "$OUTPUTS" "$MODELS" "$CACHE" -- "$@"
}

# 在容器内运行完整 mock demo，包含续跑与 packet 验证。
MERCURY_CPU_ONLY=1 launch offline-demo --output /outputs/software-smoke-v1

# 直接使用 mercury 配置/预检 CLI。
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
# 若调度器已经提供 CUDA_VISIBLE_DEVICES，保留它，不设置另一组设备。
# 仅在已获四卡分配且变量未设置时：
export MERCURY_CUDA_DEVICES=0,1,2,3
launch doctor --gpu-count 4 --output /outputs/mercury-doctor-v1.json
```

已有 `CUDA_VISIBLE_DEVICES` 与 `MERCURY_CUDA_DEVICES` 不一致时脚本拒绝启动；没有明确 GPU 分配也不默认占用全部卡。单/双卡任务相应选择 `h200_1gpu` / `h200_2gpu` 并传入 1/2 个获分配设备。推荐在调度系统中直接继承其 GPU UUID/可见设备设置。[Apptainer GPU 文档](https://apptainer.org/docs/user/main/gpu.html)说明了 `--nv` 和 `APPTAINERENV_CUDA_VISIBLE_DEVICES` 的作用。

Doctor 不加载权重；它区分 NVIDIA-SMI 物理设备清单、Torch 可见设备、GPU topology、驱动、实际 CUDA runtime、显存和一次小分配/同步结果。它不证明模型可用上下文、吞吐或长上下文不会 OOM。CPU 模式的通过也不代表 GPU 已合格。

## 绑定模型与生成配置

本地模型预先存放于 `$MODELS`，使用容器内 `/models/...` 路径。不要让 Hub 下载在正式批次中隐式发生。远程自定义模型代码只有 profile 显式 `trust_remote_code=true` 才允许加载；它和 tokenizer 文件也在 checkpoint 清单内。

为每个选用的本地 checkpoint 生成完整文件清单，输出放在模型目录之外。此操作读取全部权重字节，可能耗时，但不加载模型：

```bash
MERCURY_CPU_ONLY=1 launch checkpoint \
  --model-path /models/model-a-snapshot --revision REPLACE_WITH_ACTUAL_REVISION \
  --output /outputs/model-a.manifest.json
```

对 B/C 及独立 classifier checkpoint 重复。把生成的清单放入 `$SOURCES/checkpoints/`，然后在 catalog 对应 profile 的 `runtime` 中填写：

```json
"checkpoint_manifest": {
  "path": "/inputs/checkpoints/model-a.manifest.json",
  "file_bytes_sha256": "REPLACE_WITH_MANIFEST_FILE_SHA256"
}
```

清单同时包含文件字节哈希与 canonical JSON self-hash，不能混用。模型目录的新增、遗漏、篡改文件均使验证失败；HF snapshot 文件符号链接可用，但链接目标也必须在容器内可读，例如挂载完整的 cache/snapshot 父目录。清单不能位于 checkpoint 目录内。真实模型、revision、context window、参数能力和 checkpoint 绑定填写完整并经过实际 qualification 后，再把 profile 的状态设为 `frozen`。

获取 classifier 对应的 **catalog profile hash**：

```bash
MERCURY_CPU_ONLY=1 launch catalog --catalog /inputs/models.mercury-v1.json
```

把输出中选定 classifier 的 hash 填入 selection 的 `classifier.profile_sha256`。它是独立选择，不会随 local slot 更换自动更新。资源预设应用后生成的有效 profile 另有 hash，compose 自动将它绑定到实验。

先生成可审阅的 draft：

```bash
MERCURY_CPU_ONLY=1 launch compose \
  --base /workspace/high_fidelity_schema_study/config/four_category_experiment_v1.json \
  --catalog /inputs/models.mercury-v1.json \
  --selection /inputs/selection.mercury-v1.json \
  --host /workspace/high_fidelity_schema_study/config/mercury_host_v1.json \
  --output /outputs/experiment.mercury-draft-v1.json \
  --report /outputs/compose-draft-v1.json
```

正式 freeze 使用新的输出文件名，追加 `--freeze --image-sha256 <实际 SIF 文件 SHA-256>`。示例占位值、无法容纳输出预算的 context、未绑定 classifier、未冻结 profile、缺少 checkpoint 清单、参数不受支持或资源配置冲突都会失败。`configuration_ready` 只表示静态配置通过；实际执行前仍重新验证 SIF、源代码、checkpoint 字节和 GPU 条件。

## 后端替换

`transformers` 是直接本地加载方式，支持任意可由当前 `AutoModelForCausalLM`/tokenizer 加载的 text-generation checkpoint；没有旧的三个模型家族限制。`chat_completions` 对接已部署的兼容服务；`responses` 对接支持该协议的 frontier API。模型大小不会改变 soft reference 的非 gold 身份。

本地 Transformers 默认 `bfloat16` 和 `sdpa`；可显式切换 dtype、attention、BitsAndBytes 4/8-bit、CPU/disk offload。未声明的设置会被拒绝。资源预设为每个可见 GPU 配置 120GiB 的**权重放置预算**，CPU 放置上限为 128GiB；这些不是 KV cache 或进程总显存硬上限。`0/1/...` 是**当前可见设备集内部的 ordinal**，不是固定物理卡号。依据 [Accelerate 的大模型文档](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference)，`device_map` 表示层放置；此实现不声称实现 tensor parallel，也不把 564GB 当成单一显存空间。

兼容 HTTP 的 `do_sample`、`top_k`、`repetition_penalty` 支持情况要单独声明；示例中的 `local_http_example_v1` 不接受 `do_sample`。换后端若不支持原参数，compose 会失败，应明确制定并版本化新的运行条件。不能删掉参数后声称设置不变。HTTP 服务自身的模型 revision、量化、启动参数、镜像和设备分配应单独保存；客户端不能验证服务端未暴露的设置。

HTTP 请求须有精确输入 token 数才能过上下文门禁。可配置显式 token-count API，或使用与实际请求 hash 绑定的 `runtime.token_count_observations`。不要用字符数或另一个模型的 tokenizer 冒充精确计数。token-count 请求只在正式 `--allow-live` 运行时发生；原始请求/响应进入 run record，不属于生成调用。示例 frontier 使用 `runtime.token_counter={"kind":"responses_input_tokens","endpoint":"https://api.openai.com/v1/responses/input_tokens","timeout_seconds":30}`；vLLM 兼容服务使用 `kind="vllm_tokenize"` 和同源 `/tokenize` endpoint。不同源地址或重定向被拒绝。两种协议依据 [OpenAI token-count 文档](https://developers.openai.com/api/docs/guides/token-counting) 和 [vLLM tokenize 协议](https://docs.vllm.ai/en/v0.15.0/api/vllm/entrypoints/serve/tokenize/protocol/)，实际供应商是否支持仍需检查。

## 运行、续跑和 packet

```bash
# 仅在确实决定启动实际推理时执行以下命令。
# 密钥已由自己的凭据方式设置在 OPENAI_API_KEY 中，不在命令行写密钥值。
export MERCURY_SECRET_ENV_NAMES=OPENAI_API_KEY
launch run --config /outputs/experiment.mercury-frozen-v1.json \
  --corpus /inputs/corpus.json --source-root /inputs \
  --output /outputs/batch-v1 --allow-live
```

`run` 无论从默认入口还是 mercury 入口调用，都经过 mercury 的部署门禁。每次启动/续跑保存独立 `deployment_checks/*.json`；通过后调用现有 resumable runner。重复同一命令可复用通过完整 replay 验证的结果；更改模型/任务/输入必须形成新配置与条件。可用 `--max-jobs` 做有界试运行。

处理后切回 `MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli` 使用原有 `packet` / `verify` 命令；批次摘要位于 `batch-v1/batches/`，选择需要的实际摘要文件。Dataset 解析、plan、split、packet 和离线验证可在 `MERCURY_CPU_ONLY=1` 下执行。没有人工标签不阻止 integrity/provenance packet 生成，也不产生正式准确率。

当前 runner **串行运行**，直接 Transformers 路径每次调用释放模型与 CUDA cache，避免不同大模型残留并存；它不承诺最高吞吐。此镜像不启动 vLLM 服务、不配置跨节点 NCCL/InfiniBand、不自动把三模型同时塞进四卡。后续高吞吐可通过独立常驻服务使用既有 HTTP adapter，服务与客户端分别冻结。

`MERCURY_DRY_RUN=1` 可检查安全引用后的启动 argv。`OMP_NUM_THREADS` 默认 8，可按分配调整；不是强制占满 128 个 CPU。实际 GPU 互联、NUMA、模型上下文和吞吐由 mercury 上的预检及 qualification 决定。保留 3 次重复有助于描述生成变异，不构成“消除模型偏差或随机性”的保证。
