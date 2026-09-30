# Schema Study

**English** | [简体中文](README.zh-CN.md)

Paper–dataset schema evidence pipeline: **structure, encoding, value, syntax**.

The dataset path applies all four categories to evidence from dataset files and
declared metadata, independently of the paper models. Facts retain their source,
reading scope, and `declared` / `observed` / `inferred` basis. The paper path
uses the same taxonomy for navigation before extraction. See the
[dataset architecture and version history](docs/dataset_four_category_architecture.md).

The paper workflow retains the full layout-aware text and freezes a shared classification index and task specification for three interchangeable local models and one frontier soft reference. The dataset workflow independently produces evidence through format parsers. Both paths meet in an evaluation packet, ending at an integrity and provenance gate. A soft reference is fallible; passing this gate does not establish semantic accuracy.

## Start here

| Task | Documentation |
| --- | --- |
| Run on Mercury from environment setup through Scheduler V2, monitoring and result collection | [Mercury runbook with complete command sequences](docs/mercury_runbook.md) |
| Run the demo, prepare paper/dataset inputs, build a corpus, run jobs and verify packets | [User guide](docs/schema_study_user_guide.md) |
| Deploy on Mercury, obtain a Docker/Apptainer image, select models and freeze a configuration | [Deployment guide](docs/schema_study_deployment_guide.md) |
| Read the documentation in Chinese | [中文首页](README.zh-CN.md) · [使用指南](docs/schema_study_user_guide_zh.md) · [部署指南](docs/schema_study_deployment_guide_zh.md) |

- GitHub: [williamQ96/schema_study](https://github.com/williamQ96/schema_study)
- Docker Hub: [plalelab/schema-study](https://hub.docker.com/r/plalelab/schema-study)
- Image: `plalelab/schema-study:0.1.0-cuda13` (Linux amd64)
- Image digest and release records: [Releases](https://github.com/williamQ96/schema_study/releases).
- Release validation: [113 passing Linux container tests and the complete offline workflow](docs/release-validation-v0.1.0.json).

The Docker image and the v0.1.0 tag describe the **previous frozen release**.
This branch also contains the V13 source snapshot and Scheduler V2. The current
V13 repair candidate is undergoing GPU output qualification; it is not in the
0.1.0 image and is not a completed fidelity evaluation. See the
[V13 source status](docs/v13_source_status_2026-09-28.md).

## Five-minute offline check

This Linux/Bash example needs no GPU, model weights or API key, and makes no model calls.

```bash
docker pull plalelab/schema-study:0.1.0-cuda13
mkdir -p "$PWD/schema-study-results"
docker run --rm \
  --mount type=bind,src="$PWD/schema-study-results",dst=/outputs \
  plalelab/schema-study:0.1.0-cuda13 \
  offline-demo --output /outputs/demo-v1
```

Read `schema-study-results/demo-v1/report.json` after completion. The demo uses synthetic inputs and mock backends to exercise repeated local-model slots, a separate classifier role, a soft reference, resume behavior and packet verification. Use a new directory for another complete demo; resume an actual batch in its existing batch directory.

## Repository layout

```text
high_fidelity_schema_study/
  four_category/         # Tasks, model adapters, datasets, batches, packets, Mercury
  extractors/            # Format parser plugins
  config/                # Shared specification and draft model configurations
  templates/             # Exact prompts, taxonomy and output JSON schemas
container/
  docker/                # OCI image recipe
  apptainer/             # SIF recipe, build script and bind-mount launcher
docs/                    # English guides and optional Chinese translations
tests/                   # Synthetic and mock software tests
scripts/                 # V13 qualification and operational helpers
scheduler_v2/            # Group dispatch, recovery, verification and queue health
source-export-manifest.json # File-byte SHA-256 inventory for the current export
```

This repository distributes the runnable four-category workflow. Users manage original papers, datasets, model weights, API credentials and historical results in external directories. The export manifest covers only its listed files; CI and release metadata are not implicitly included. The original v0.1.0 inventory remains available in its tag and release assets.

## Local development and tests

```bash
git clone https://github.com/williamQ96/schema_study.git
cd schema_study
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_four_category_offline.txt
PYTHONPATH=.:scripts python -m pytest -q tests
python -m high_fidelity_schema_study.four_category.cli --help
```

The offline dependencies support parsing, mock demos and software tests. Real Transformers inference also requires the additional runtime libraries in the Linux CUDA image.

## Capabilities and experiment status

- Dataset formats: CSV/TSV, JSON/JSONL, XML/XSD, HDF5, NetCDF, Parquet, Zarr v2, ARFF and XLSX. Evidence records distinguish declared, observed, inferred and unknown facts and retain the reading scope.
- Model backends: Transformers, OpenAI-compatible Chat Completions and Responses. Profiles select the models; unsupported parameters are explicitly rejected.
- `run` verifies the model configuration, source code, SIF, checkpoint and hardware before dispatch. Sources and results retain their identities, and failures are isolated per task.
- Mercury targets 4×H200 NVL. Its actual GPUs, driver, model context limits and throughput still require qualification on that machine. Offline software validation does not establish hardware readiness or research effectiveness.
- The v0.1.0 CLI runner is serial. V13 adds a separate Scheduler V2 path with group-level dispatch; the ten-paper production rerun remains paused until the current output qualification passes. Three repetitions describe variation; they do not eliminate model bias or randomness.
- Example profiles remain drafts. The V13 source snapshot does not publish private source data, model weights, credentials, human annotations, or a new Docker image.
