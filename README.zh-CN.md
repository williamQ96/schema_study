# Schema Study

[English](README.md) | **简体中文**

Paper–dataset schema evidence pipeline: **structure, encoding, value, syntax**.

论文侧保留 layout-aware 全文，冻结共享分类索引和任务规范，供三个可替换的本地模型及一个 frontier soft reference 使用。Dataset 侧由格式解析器独立生成四类证据。两条路径在 evaluation packet 汇合，终点是完整性与来源验证；soft reference 不作为 gold，验证通过也不代表语义准确率。

版本事实：v0.1.0 已包含 dataset 四类解析基础接口。本次 V13 源码增加了明确资源范围的 dataset 字段目录、证据重放检查、Scheduler V2 和修订后的 paper 抽取合约。当前 GPU 输出资格测试仍在进行，十篇全量推理尚未重新启动。详见英文 [dataset 架构说明](docs/dataset_four_category_architecture.md) 与 [V13 状态](docs/v13_source_status_2026-09-28.md)；Docker Hub 的 `0.1.0-cuda13` 镜像仍属于上一版源码。

## 从这里开始

| 任务 | 文档 |
| --- | --- |
| 跑通示例、准备 paper/dataset、创建 corpus、运行与验证 packet | [使用指南](docs/schema_study_user_guide_zh.md) |
| 在 Mercury 部署、下载镜像、转为 Apptainer、选择模型及冻结配置 | [部署指南](docs/schema_study_deployment_guide_zh.md) |

- GitHub: [williamQ96/schema_study](https://github.com/williamQ96/schema_study)
- Docker Hub: [plalelab/schema-study](https://hub.docker.com/r/plalelab/schema-study)
- 镜像版本：`plalelab/schema-study:0.1.0-cuda13`（Linux amd64）
- 发布验证及镜像 digest：见 [Releases](https://github.com/williamQ96/schema_study/releases)。
- 本版软件验收：[113 项 Linux 容器测试通过及完整离线流程记录](docs/release-validation-v0.1.0.json)。

## 五分钟离线检查

以下 Linux/Bash 示例不需要 GPU、模型权重或 API key，也不会调用模型。

```bash
docker pull plalelab/schema-study:0.1.0-cuda13
mkdir -p "$PWD/schema-study-results"
docker run --rm \
  --mount type=bind,src="$PWD/schema-study-results",dst=/outputs \
  plalelab/schema-study:0.1.0-cuda13 \
  offline-demo --output /outputs/demo-v1
```

成功后查看 `schema-study-results/demo-v1/report.json`。示例使用合成材料和模拟后端，覆盖三模型重复、独立分类器、soft reference、断点续跑及 packet 验证。再次做完整 demo 时使用新目录；实际批次续跑使用同一个 batch 目录。

## 仓库结构

```text
high_fidelity_schema_study/
  four_category/         # 任务、模型适配器、dataset、批次、packet、Mercury
  extractors/            # 格式解析插件
  config/                # 共享实验规范及尚未选定真实模型的示例配置
  templates/             # 原文 prompt、taxonomy、输出 JSON schema
container/
  docker/                # 可构建的 OCI 镜像
  apptainer/             # SIF 配方、构建与安全挂载启动器
docs/                    # 使用和部署文档
tests/                   # 合成/模拟软件测试
source-export-manifest.json # 发布源码文件字节 SHA-256
```

本仓库是四类流程的可运行源码发行包。原始论文、dataset、模型权重、API 凭据和历史实验结果由使用者在外部目录管理。源码 export manifest 覆盖其列出的文件，发布后增加的 CI 或 release 元数据不隐含在该清单内。

## 本地开发与测试

```bash
git clone https://github.com/williamQ96/schema_study.git
cd schema_study
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_four_category_offline.txt
python -m pytest -q tests
python -m high_fidelity_schema_study.four_category.cli --help
```

离线依赖足以运行 parser、mock demo 和软件测试。真实 Transformers 推理依赖 Linux CUDA 镜像中的额外运行库。

## 当前能力与实验状态

- Dataset 格式：CSV/TSV、JSON/JSONL、XML/XSD、HDF5、NetCDF、Parquet、Zarr v2、ARFF、XLSX；读取范围和 declared/observed/inferred/unknown 分开记录。
- 模型后端：Transformers、OpenAI-compatible Chat Completions、Responses。模型选择由 profile 决定；无法支持的参数会明确拒绝。
- `run` 经过模型、源码、SIF、checkpoint 和硬件门禁；输入与结果保留来源身份，失败按任务隔离。
- Mercury 目标为 4×H200 NVL；其实际 GPU、驱动、模型上下文和吞吐仍需在该机器上 qualification。软件离线通过不能替代硬件或研究效果验证。
- 当前 runner 串行调度；三个本地模型默认各三次重复。重复用于描述变异，不能宣称消除模型偏差或随机性。
- 公开示例配置是 draft，占位模型不能启动正式实验。没有执行新的模型实验，也没有生成真人标签。
