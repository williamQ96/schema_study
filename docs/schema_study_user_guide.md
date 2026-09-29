# Schema Study user guide

**English** | [简体中文](schema_study_user_guide_zh.md)

This guide explains how to organize paper PDFs and dataset files into traceable four-category inputs, prepare them offline, run a qualified frozen configuration on Mercury and verify the resulting packets. Papers and datasets follow separate paths: deterministic dataset parsers never supply content to paper classification or extraction requests.

Source release: [GitHub v0.1.0](https://github.com/williamQ96/schema_study/releases/tag/v0.1.0). Image: [Docker Hub `plalelab/schema-study:0.1.0-cuda13`](https://hub.docker.com/r/plalelab/schema-study/tags?name=0.1.0-cuda13). See the [deployment guide](schema_study_deployment_guide.md) for SIF conversion, mounts, model profiles and hardware checks.

Start with the offline demo. Then prepare and bind paper v3 inputs, parse datasets independently, apply admission rules and build a many-to-many corpus manifest. Inspect the job plan before selecting models and freezing a configuration. Configurations that have not completed model and hardware qualification remain drafts; `--allow-live` does not bypass the freeze requirements.

## 1. Command entry points

The examples assume you have defined the `launch` function from the deployment guide. Sources are mounted read-only at `/inputs`; results go to writable `/outputs`. The Python package is mounted at `/workspace/high_fidelity_schema_study`. Select the workflow CLI:

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
```

This CLI provides `prepare-paper`, `parse-datasets`, `plan`, `run`, `packet`, `verify`, `split` and `offline-demo`. The launcher routes `run` through the Mercury deployment gate. For `doctor`, `catalog`, `checkpoint` and `compose`, select the Mercury configuration CLI:

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
```

Use new filenames for standalone JSON outputs; the commands do not overwrite existing files. Paths must match the container mounts. Use `MERCURY_CPU_ONLY=1 launch ...` for preparation, parsing, planning and packet operations. Do not set CPU-only mode for a GPU inference batch.

Switch back to the workflow CLI and run the synthetic check. It uses built-in fictional sources and mock transports, with no model service calls:

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch offline-demo --output /outputs/synthetic-demo-01
```

The report records `live_requests: 0`. Its outputs are software fixtures, not research observations, human labels or accuracy results.

## 2. Prepare paper sources independently

`prepare-paper` validates existing artifacts; it does not generate a layout from a PDF. If you only have a PDF, use the existing preprocessing API in a Python environment with the public package and its offline dependencies installed. Run this script on the host. It writes new artifacts under your source root, and `paper_id` becomes the v3 input's document ID:

```python
from pathlib import Path
from high_fidelity_schema_study.paper_layout_evidence import preprocess_layout_pdf
from high_fidelity_schema_study.paper_layout_evidence_v3 import upgrade_bundle
from high_fidelity_schema_study.four_category.common import write_new

from_root = Path("/host/path/frozen-sources")  # Contains papers/p01/source.pdf
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

Preprocessing reads the PDF text layer. It performs no OCR and does not recover image or figure semantics. Each paper binds four matching artifacts: the original PDF, a `paper-evidence-layout/v3` JSON file, UTF-8 reading text and a `paper-evidence-input/v3` JSON file. The PDF bytes must match the layout's source hash. Validation checks these artifacts, not visual completeness. Use a new directory when rebuilding a frozen input.

Place the artifacts under `$SOURCES`, then validate them in the container. Write the source descriptor to `/outputs`:

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

Validation checks source identities, v3 structure, extracted-word coverage and evidence-unit uniqueness. Coverage refers to the extracted words stored in the layout, not every visual element of the PDF. Keep the original PDF for independent auditing. Fix the preparation artifacts if validation fails; do not fabricate hashes or source descriptors.

On the host, copy the generated `paper-source-p01.json` to `$SOURCES/paper-sources/p01.json`. The descriptor and its referenced original artifacts must resolve within the read-only `/inputs` source tree. Perform this copy on the host, not from inside the container.

## 3. Parse dataset sources independently

The four dataset categories have an explicit evidence contract: `structure`
describes objects and fields, `encoding` records representation and storage,
`value` records declared codes and unit semantics, and `syntax` records lexical
and ordering conventions. A fact may belong to more than one category. The
parser also records whether support is declared, observed, inferred, or
unknown. The [dataset architecture note](dataset_four_category_architecture.md)
explains the new scoped field catalog and its limits.

### Supported formats and interpretation

The adapter recognizes CSV, TSV, JSON, JSONL/NDJSON, XML, XSD, ARFF, XLSX, HDF5, NetCDF, Parquet and Zarr v2 metadata directories. A format hint can be supplied explicitly. Recognition does not guarantee complete parsing: damaged files, missing dependencies and bounded-reading limits can produce `failed`, `unsupported` or `partial` results while other sources continue. A batch marked `complete` has accounted for its jobs; inspect each source's `status` and `issues` separately.

Facts distinguish `declared` statements, directly `observed` content, rule-based `inferred` candidates and `unknown` information. A unit suggested by a field-name suffix is inferred. Sample values are not a declared vocabulary or a NOT NULL constraint, and numeric summaries are not semantic enumerations. Reading limits depend on the format; `--sample-limit` defaults to 200. Some formats expose only metadata. Zarr support reads v2 metadata, not chunks, and does not support Zarr v3.

### Create a manifest and run parsing

Place files and optional sidecars under `$SOURCES/datasets/`. Manifest `path` and `sidecars` entries are relative to `/inputs`. Each `source_id` must be unique; an optional `family_id` groups related sources. Replace the following example with actual filenames and source IDs:

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

Save the manifest at `$SOURCES/manifests/dataset-jobs-v1.json`, then write parsed outputs to the writable output mount:

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch parse-datasets \
  --manifest /inputs/manifests/dataset-jobs-v1.json \
  --source-root /inputs \
  --output /outputs/dataset-cache-v1 \
  --sample-limit 200 \
  --summary-output /outputs/dataset-batch-v1.json
```

Use `--max-jobs N` to bound an invocation, then resume with the same cache directory and a new summary filename. Immutable bundles can be reused, but their sources are still read and replay-verified; caching does not promise to avoid source I/O. The summary includes actual `source_id`, `dataset_id`, `bundle_sha256`, `bundle_path` and status values.

### Bring accepted bundles into the source tree

Corpus `bundle_path` values must be relative to `--source-root`, which is `/inputs` here. Parsing writes its cache under `/outputs`, so apply your admission rules and copy accepted bundles to `$SOURCES/dataset-bundles/` on the host. Do not edit a bundle or regenerate its hash. Naming the copied file with its returned `bundle_sha256` makes the mapping explicit.

The following host script copies only bundles with status `pass`, checking the canonical JSON self-hash and batch identity. Replace its three host paths. It creates each destination exclusively and stops if the file already exists. Versioned admission rules decide whether other statuses trigger retry, exclusion or sampled review; individual human approval is not required for every source.

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

For `partial`, `unsupported` or `failed` sources, inspect the reported coverage and issues before admitting or reprocessing them. Do not relabel a failure as a pass. Full source derivation is checked again during packet verification.

## 4. Build a paper–dataset corpus

Relationships are many-to-many: one paper can reference several datasets, and a dataset can appear in several papers. Each match records a `candidate` or `verified` linkage and its actual supporting evidence. A candidate is provisional. Similar paper titles or filenames do not establish a scientific correspondence.

The host helper below reads actual IDs and hashes from the paper descriptor and dataset batch, then derives paths for the copied bundles. Corpus `paper_id` must equal the descriptor's `source_identity.document_id`. Fill in the real `source_id` and linkage evidence; omit a family ID when its source relationship is unknown.

```python
import json
from pathlib import Path

source_root = Path("/host/path/frozen-sources")
paper_source_file = source_root / "paper-sources/p01.json"
dataset_batch_file = Path("/host/path/mercury-results/dataset-batch-v1.json")
corpus_file = source_root / "corpus-v1.json"

# Use actual manifest source IDs and auditable linkage evidence.
paper_family_id = None
links = [
    {"source_id": "station-observations", "status": "candidate",
     "evidence": ["REPLACE_WITH_ACTUAL_PAGE_QUOTE_OR_PERSISTENT_IDENTIFIER"]}
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

For more papers, extend the paper and match rows using the same schema. Each paper's `source` is the complete descriptor returned by `prepare-paper`, not a filename. All referenced paper artifacts and dataset bundles must be readable within `/inputs` through their recorded relative paths. Do not claim independence when source-family relationships are unknown.

## 5. Plan jobs, configure models and run

The supplied configuration is a draft: models are unselected, context placeholders are non-executable, the classifier hash is unbound and `inference_enabled` is false. Inspect job expansion without sending requests:

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch plan \
  --config /workspace/high_fidelity_schema_study/config/four_category_experiment_v1.json \
  --corpus /inputs/corpus-v1.json \
  --output /outputs/plan-v1.json
```

For Mercury, configure the model catalog, selection and host resource preset. Use `catalog` to inspect profile identities, `checkpoint` to inventory local model files and `compose` to produce a new draft or explicitly frozen configuration. Bind the classifier profile ID and hash independently. Changing that profile requires an explicit new pin. The deployment guide provides the complete commands.

The design uses three interchangeable local profiles, a separate frontier soft-reference profile and an explicitly bound automatic classifier role. The classifier can reuse a selected model deployment. Its four-category, none or uncertain assignments provide fallible navigation over all paper evidence units; they are not human labels or gold. Local and reference extraction share the same full paper input and frozen index. Dataset content never enters these prompts.

Adapters, supported request parameters, exact token counts, checkpoint identity, image, source code and GPUs must match the deployment profile. After qualification, freeze the configuration and enable inference. `--allow-live` permits real calls, but does not freeze a draft or replace qualification. The following command can perform real inference and billable API calls:

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

Run `doctor` as described in the deployment guide first. `run` repeats deployment and hardware checks before dispatch. Resume in the same result directory. Each invocation records a deployment report under `runs-v1/deployment_checks/`; batch summaries are written to `runs-v1/batches/`. Select an actual summary file rather than guessing its hash-based filename.

## 6. Build and verify a packet; plan independent evaluation

After a real or mock batch, build the provenance packet and verify it independently. Replace the batch placeholder with the actual summary from that run:

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

Packet verification checks file bytes and source, request and derivation replay. It does not establish semantic correctness. `split` groups connected source families when planning independent human evaluation, including untagged content to help detect omissions. The audit fraction is a workload setting, not a statistical-power guarantee. Researchers perform and record rule calibration and independent sampled evaluation separately. The pipeline does not generate human labels, accuracy estimates or scientific conclusions automatically.

## 7. Output locations and common blockers

| Location | Contents | Handling |
| --- | --- | --- |
| `/inputs` | Read-only PDFs, layout/input v3, datasets, sidecars, corpus, configurations and bundles | Do not write from inside the container |
| `/outputs` | Descriptors, parsing cache/batches, plans, attempts, indexes, summaries, packets and reports | Use new standalone output names; resume batches in their existing directory |
| `/models` | Locally prepared model checkpoints | Bind an inventory manifest before use |
| `/cache` | Writable runtime cache | Configure it for the deployment |

Common blockers include mismatched PDF/layout hashes, references outside `/inputs`, manifest paths escaping the source root, partial or failed dataset results, copied bundles with mismatched identities, stale task/taxonomy hashes, an incorrect classifier pin, missing exact request-bound token counts, incompatible GPU visibility or drivers, failed CUDA allocation probes and unsupported profile parameters. Fix the source or configuration and create a new versioned artifact. Do not edit sealed hashes or run records to force admission.
