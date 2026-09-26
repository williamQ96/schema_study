# Schema Study 用户指南（中文）

[English](schema_study_user_guide.md) | **简体中文**

本指南介绍如何把论文 PDF 和数据文件组织成可追溯的四类别研究输入，并在 Mercury 容器中运行离线准备、硬件预检、冻结配置后的模型调用与结果核验。论文与数据集是两条独立输入路径：数据文件由确定性解析器处理，不会进入论文分类或抽取请求。
源码：[GitHub v0.1.0 release](https://github.com/williamQ96/schema_study/releases/tag/v0.1.0)；OCI 镜像：[Docker Hub `plalelab/schema-study:0.1.0-cuda13`](https://hub.docker.com/r/plalelab/schema-study/tags?name=0.1.0-cuda13)。版本、取得 SIF、挂载和硬件预检见[部署指南](schema_study_deployment_guide_zh.md)。

新用户建议按顺序完成：先运行零推理软件示例；准备并绑定论文 v3 输入；独立解析数据集；审核解析状态并建立 many-to-many 语料清单；计划任务；最后才选择模型、冻结配置并执行硬件与模型 qualification。未完成真实模型和硬件 qualification 的配置保持 `draft`，不能通过 `--allow-live` 绕过冻结门禁。

## 1. Mercury 容器中的命令入口

以下命令假设已按部署指南创建 `launch` 函数，容器把只读来源挂载到 `/inputs`，把可写结果挂载到 `/outputs`。不要在容器里写入 `/inputs`。新发布布局中 Python 包位于 `/workspace/high_fidelity_schema_study`。先选定 CLI 模块：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
```

容器启动及挂载由部署指南中的 `launch` 函数负责。`prepare-paper`、`parse-datasets`、`plan`、`run`、`packet`、`verify`、`split` 和 `offline-demo` 均由此 CLI 提供。`doctor`、`catalog`、`checkpoint`、`compose` 是 Mercury profile 管理 CLI；需要时切换模块：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
```

所有 JSON 输出均采用新文件名；这些命令不会覆盖已有文件。路径必须与容器挂载一致。自己的源文件、profile 和配置应绑定到发布版 package 的预期目录。

来源准备、数据解析、计划和 packet 操作需用 `MERCURY_CPU_ONLY=1 launch ...`，避免 GPU 透传或分配；真实批次运行时不要设置它。

先运行合成软件检查。它只使用内置虚构来源与 mock transport，不访问模型服务：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch offline-demo --output /outputs/synthetic-demo-01
```

报告中的 `live_requests: 0` 表示没有真实模型调用。示例产物不是研究数据、标签或准确率结果。

## 2. 准备论文来源（独立于数据集）

`prepare-paper` 验证既有输入，不会从 PDF 自动生成 layout。若手头只有 PDF，可在有项目依赖的公开包 Python 环境调用现有 preprocessing API，再运行 CLI 校验。以下脚本在宿主机执行，并把所有新文件写入 `$SOURCES`；`paper_id` 将成为 v3 input 的 document ID：

```python
from pathlib import Path
from high_fidelity_schema_study.paper_layout_evidence import preprocess_layout_pdf
from high_fidelity_schema_study.paper_layout_evidence_v3 import upgrade_bundle
from high_fidelity_schema_study.four_category.common import write_new

from_root = Path("/host/path/frozen-sources")  # 已含 papers/p01/source.pdf
paper_id = "P01"
folder = from_root / "papers/p01"
pdf = folder / "source.pdf"
legacy, _, _ = preprocess_layout_pdf(paper_id, pdf)
layout, reading, paper_input, _ = upgrade_bundle(legacy)
write_new(folder / "layout-v3.json", layout)
with (folder / "reading.txt").open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(reading)
write_new(folder / "input-v3.json", paper_input)
```

预处理读取 PDF 文本层，不执行 OCR，也不恢复图片或图表语义。每篇论文绑定四个匹配文件：原始 PDF、`paper-evidence-layout/v3` JSON、UTF-8 reading-text 和 `paper-evidence-input/v3` JSON。PDF 字节必须与 layout 哈希一致。CLI 校验这些准备结果，不评判视觉内容是否完整；重做时使用新目录，不覆盖已冻结文件。

把这些文件放在来源目录中，例如宿主机上的 `$SOURCES`。在容器中调用 `prepare-paper`；输出先写入 `/outputs`：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch prepare-paper \
  --layout /inputs/papers/p01/layout-v3.json \
  --reading /inputs/papers/p01/reading.txt \
  --input /inputs/papers/p01/input-v3.json \
  --pdf /inputs/papers/p01/source.pdf \
  --root /inputs \
  --output /outputs/paper-source-p01.json
```

它会校验 PDF/文本/JSON 字节哈希、v3 结构、全文提取词覆盖审计和证据单元唯一性。覆盖指布局文件保存的提取词，不是 PDF 全部视觉信息；原始 PDF 应保留以供独立审计。失败时先修复准备材料，不要手工伪造 hash 或 source descriptor。

完成后在宿主机把生成的 `paper-source-p01.json` 复制进来源目录，例如 `$SOURCES/paper-sources/p01.json`。这样语料里的 source descriptor 及其所引用的原始文件都能从只读 `/inputs` 解析。复制动作在宿主机执行，不要让容器写入 `/inputs`。

## 3. 独立解析数据来源

### 支持范围与解释边界

适配器识别 CSV、TSV、JSON、JSONL/NDJSON、XML、XSD、ARFF、XLSX、HDF5、NetCDF、Parquet 和 Zarr-v2 元数据目录。格式提示可以显式提供，但底层读取器可能需要对应依赖；格式识别不保证某个文件完整解析。损坏、缺少依赖或有界解析限制会在该来源结果中报告 `failed`、`unsupported` 或 `partial`，批次里其它来源仍可分别处理。不要把 `complete` 批次状态理解为所有来源都通过；逐条检查 `status` 和 `issues`。

解析事实区分 `declared`（文件内明确声明）、`observed`（解析时直接观察到）、`inferred`（有限规则推断）与 `unknown`。名字后缀的单位提示属于推断。样本值不是受控词表，也不证明 NOT NULL；数值统计不构成语义枚举。为控制资源，读取存在格式特定边界，默认 `--sample-limit 200`；部分格式只读元数据，不扫描完整值或块数据。Zarr 仅处理 v2 元数据，不读取 chunks，也不支持 Zarr v3。

### 建立 manifest 并运行批处理

在 `$SOURCES/datasets/` 放置数据文件和可选 sidecar。Manifest 中的 `path`/`sidecars` 相对 `/inputs`；`source_id` 必须唯一，`family_id` 可选，用于后续把相关来源分组。下面示例为格式，需换成真实文件名与来源 ID：

```json
{
  "schema_version": "four-category-dataset-jobs/v1",
  "sources": [
    {"source_id": "station-observations", "path": "datasets/stations.csv", "family_id": "station-network"},
    {"source_id": "station-description", "path": "datasets/stations.json", "format_hint": "json",
     "sidecars": ["datasets/stations.schema.json"], "family_id": "station-network"}
  ]
}
```

把 manifest 放入 `$SOURCES/manifests/dataset-jobs-v1.json` 后，在容器中让批处理写到可写的 `/outputs`：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch parse-datasets \
  --manifest /inputs/manifests/dataset-jobs-v1.json \
  --source-root /inputs \
  --output /outputs/dataset-cache-v1 \
  --sample-limit 200 \
  --summary-output /outputs/dataset-batch-v1.json
```

`--max-jobs N` 可限制本次处理数；之后再次运行可继续。不可变 bundle 会复用，但来源仍会读取并重放验证，不保证节省来源 I/O。`/outputs/dataset-batch-v1.json` 包含真实 `source_id`、`dataset_id`、`bundle_sha256`、`bundle_path` 和各自状态。
### 将 bundle 安全带入只读来源

语料里的 `bundle_path` 必须相对 `--source-root`（此部署中即 `/inputs`）。批处理缓存写在 `/outputs`，因此在宿主机完成以下两步：逐个检查成功结果；把被接受的 bundle 复制到 `$SOURCES/dataset-bundles/`。不要在容器中写 `/inputs`，也不要编辑或重新生成 bundle 的 hash。建议以工具返回的 `bundle_sha256` 作为目标文件名，便于与结果核对。

以下脚本在宿主机运行；把三个本地路径设置为实际目录。它只复制批次里状态为 `pass` 的 bundle，并核对 canonical self-hash 与批次记录。目标文件采用排他创建，已存在时停止，避免静默覆盖。其它状态由版本化准入规则决定重试、排除或抽样复核，不能自动视为通过；不要求人工逐份批准。

```python
import hashlib
import json
from pathlib import Path

batch_file = Path("/host/path/mercury-results/dataset-batch-v1.json")
cache = Path("/host/path/mercury-results/dataset-cache-v1")
source_root = Path("/host/path/frozen-sources")
destination = source_root / "dataset-bundles"
destination.mkdir(parents=True, exist_ok=True)
batch = json.loads(batch_file.read_text(encoding="utf-8"))
for row in batch["results"]:
    if row.get("status") != "pass":
        print(f"Review/exclude {row['source_id']}: {row.get('status')}")
        continue
    source = cache / row["bundle_path"]
    bundle = json.loads(source.read_text(encoding="utf-8"))
    claimed = bundle.get("bundle_sha256")
    unhashed = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    canonical = json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != claimed or claimed != row["bundle_sha256"] or bundle.get("dataset_id") != row["dataset_id"]:
        raise ValueError(f"batch/bundle mismatch: {row['source_id']}")
    relative = Path("dataset-bundles") / f"{row['bundle_sha256']}.json"
    with source.open("rb") as incoming, (source_root / relative).open("xb") as outgoing:
        while chunk := incoming.read(1024 * 1024):
            outgoing.write(chunk)
    print(f"Copied {row['source_id']} to {relative}")
```

目标路径由真实 `bundle_sha256` 生成。若来源为 `partial`、`unsupported` 或 `failed`，先审查 `issues` 与覆盖范围，再决定是否排除或重新处理；不要把失败结果伪装成通过。
## 4. 建立真实 corpus（paper ↔ dataset）

Corpus 关联是 many-to-many。先确认每篇论文是否关联一个或多个数据集；一个数据集也可能关联多篇论文。每个 match 应写明 `candidate` 或 `verified`，并记录真实的来源依据。`candidate` 表示待核对候选，不是科学关联结论。不要用论文题名或文件名相似度代替证据。

下面的宿主机 Python helper 从 `paper-source` 和批次输出读取实际 ID/hash/path，并按真实 hash 推导已复制的 bundle 路径；不会编造 ID 或 hash。Corpus `paper_id` 必须等于 source descriptor 的 `source_identity.document_id`，helper 会直接读取。按研究记录填写真实的 `source_id` 和 linkage evidence；无独立来源依据时省略 family ID。路径均为宿主机绝对路径示例：

```python
import json
from pathlib import Path

source_root = Path("/host/path/frozen-sources")
paper_source_file = source_root / "paper-sources/p01.json"
dataset_batch_file = Path("/host/path/mercury-results/dataset-batch-v1.json")
corpus_file = source_root / "corpus-v1.json"

# 仅填写 manifest 中的真实 source_id 和可审计的关联依据。
paper_family_id = None
links = [
    {"source_id": "station-observations", "status": "candidate",
     "evidence": ["填写 paper-source 实际记录的页码、句子或外部持久标识"]}
]

paper_source = json.loads(paper_source_file.read_text(encoding="utf-8"))
paper_id = paper_source["source_identity"]["document_id"]
batch = json.loads(dataset_batch_file.read_text(encoding="utf-8"))
batch_rows = {row["source_id"]: row for row in batch["results"]}
datasets, matches = {}, []
for link in links:
    row = batch_rows[link["source_id"]]
    if row.get("status") != "pass":
        raise ValueError(f"dataset source is not admitted: {link['source_id']}")
    bundle_path = Path("dataset-bundles") / f"{row['bundle_sha256']}.json"
    if not (source_root / bundle_path).is_file():
        raise FileNotFoundError(source_root / bundle_path)
    dataset = {"dataset_id": row["dataset_id"], "bundle_path": bundle_path.as_posix(),
               "bundle_sha256": row["bundle_sha256"], **({"family_id": row["family_id"]} if row.get("family_id") else {})}
    datasets[dataset["dataset_id"]] = dataset
    matches.append({"match_id": f"{paper_id}-{link['source_id']}", "paper_id": paper_id,
                    "dataset_id": dataset["dataset_id"],
                    "linkage": {"status": link["status"], "evidence": link["evidence"]}})

corpus = {"schema_version": "four-category-corpus/v1",
          "papers": [{"paper_id": paper_id, "source": paper_source,
                      **({"family_id": paper_family_id} if paper_family_id else {})}],
          "datasets": list(datasets.values()), "matches": matches}
with corpus_file.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
print(f"Wrote {corpus_file}; papers={len(corpus['papers'])}, datasets={len(corpus['datasets'])}, matches={len(matches)}")
```

`family_id` 应根据真实来源关系填写；不清楚时省略，不要声称独立。多个论文时把 paper rows 和 match rows 按相同 schema 扩展即可，paper 的 `source` 必须是 `prepare-paper` 实际生成并复制到来源树内的完整 JSON 对象。所有 bundle 和 paper artifacts 必须在 `/inputs` 内可读且由 corpus 的相对路径准确引用。

## 5. 计划、配置模型与实际运行

当前模板是 draft：模型未选择、上下文 sentinel 不可执行、分类器 hash 未绑定、`inference_enabled` 为 false。不能拿它做正式实时推理。先检查 jobs 展开，不发送模型请求：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch plan \
  --config /workspace/high_fidelity_schema_study/config/four_category_experiment_v1.json \
  --corpus /inputs/corpus-v1.json \
  --output /outputs/plan-v1.json
```

如果要运行 Mercury profile 流程，先配置 model catalog、selection 与 mercury host 资源预设，再用 `mercury catalog` 检查模型 profile，`mercury checkpoint` 生成本地 checkpoint 清单，`mercury compose` 创建新的 draft 或显式 freeze 条件。必须把分类器 profile ID 与 profile SHA-256 单独绑定；换分类器或 profile 后需要显式重新 pin。具体 compose/doctor 和 SIF 操作见部署指南。

实验设计要求三个可替换本地 profile、一个单独的 frontier soft-reference profile 和一个明确绑定的自动分类器 profile。分类器用于对论文全部证据单元作四类、none 或 uncertain 标注；分类结果是可失败的导航索引，不是人工标签或 gold。之后本地/参考抽取使用同一完整 paper input 与同一冻结 index；数据集内容不会加入论文 prompt。支持的 backend adapter、请求参数能力、token 数精确计数、checkpoint、镜像、代码和 GPU 条件都需按实际 deployment profile 校验。

只有配置和 profiles 经过 qualification、配置正式标为 frozen 且 `inference_enabled=true` 后，才执行真实 `run`。显式 `--allow-live` 是必要的用户动作，但它不会把 draft 变成 frozen，也不会替代模型/硬件 qualification。一次运行例：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
export MERCURY_SECRET_ENV_NAMES=OPENAI_API_KEY
unset MERCURY_CPU_ONLY
launch run \
  --config /outputs/experiment.frozen-v1.json \
  --corpus /inputs/corpus-v1.json \
  --source-root /inputs \
  --output /outputs/runs-v1 \
  --allow-live
```

先按部署指南运行 `doctor`。`run` 会再次做部署及硬件门禁，失败时不派发推理。结果目录可续跑。Mercury CLI 每次会在 `runs-v1/deployment_checks/` 写部署检查报告，批次摘要保存在 `runs-v1/batches/`；需要 handoff 时选择实际生成的摘要 JSON，不要假定 hash 文件名。

## 6. Packet、验证与独立评估

实时或 mock batch 完成后，构建 provenance packet，再执行独立验证。下面路径应替换为同一批次的真实文件：

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
launch packet \
  --config /outputs/experiment.frozen-v1.json \
  --corpus /inputs/corpus-v1.json \
  --batch /outputs/runs-v1/batches/REPLACE_WITH_ACTUAL_BATCH_SHA256.json \
  --run-root /outputs/runs-v1 \
  --source-root /inputs \
  --output /outputs/packet-v1

launch verify --packet /outputs/packet-v1 --source-root /inputs \
  --output /outputs/verification-v1.json

launch split --corpus /inputs/corpus-v1.json --source-root /inputs \
  --audit-fraction 0.2 --seed 1729 --output /outputs/audit-split-v1.json
```

Packet 检查字节完整性与来源/请求/派生重放，不等于语义正确。`split` 以连通来源家族为单位规划独立人工抽样，含 untagged 内容以便估计遗漏；默认 fraction 是工作量设置，不是统计功效保证。人工规则校准和独立样本评估须由研究者另行执行与记录。本仓库不自动生成真实人工标签、准确率或科学结论。

## 7. 输出位置与常见阻塞

| 位置 | 内容 | 应否改写 |
| --- | --- | --- |
| `/inputs` | 只读 PDF、layout/input v3、dataset、sidecar、corpus、配置、bundle | 不在容器内写入 |
| `/outputs` | paper-source、dataset batch/cache、plan、run attempts、index、summary、packet、报告 | 使用新名称，不覆盖旧结果 |
| `/models` | 容器可读的本地模型目录 | 预先准备并绑定 checkpoint manifest |
| `/cache` | 可写运行缓存 | 按部署配置使用 |

常见阻塞包括：paper PDF 与 layout 哈希不同；source descriptor 引用不在 `/inputs` 的路径；dataset manifest path 越界；批次虽然 complete 但某条解析是 partial/failed；复制到来源根的 bundle 与 batch hash 不匹配；experiment task/taxonomy hash 过期；classifier profile pin 不匹配；HTTP token 计数不是精确且与实际消息 hash 绑定；GPU mask、Torch 可见卡数、driver floor 或分配 probe 不符合预期；profile 参数未获 adapter 声明支持。修复来源/配置并创建新输出文件，勿手工改写封印字段、run record 或 hash。
