# Deployment guide: Docker Hub → Apptainer → Mercury

**English** | [简体中文](schema_study_deployment_guide_zh.md)

This guide uses the public `williamQ96/schema_study` repository layout. See the [user guide](schema_study_user_guide.md) for input preparation and packet operations. Shell examples use Linux/Bash.

## 1. Deployment contents and host requirements

| Item | Release configuration |
| --- | --- |
| OCI image | `docker.io/plalelab/schema-study:0.1.0-cuda13`, Linux amd64 |
| Container user space | Ubuntu 24.04, CUDA 13.0.2 / cuDNN |
| Target host | Mercury: RHEL 9.8, 4×H200 NVL with 141GB each, 128 CPU cores, approximately 1.1TiB RAM |
| Included | Four-category workflow, prompts/schemas, parsing and inference dependencies, command entry point |
| Supplied externally | PDF/layout/input artifacts, datasets, model checkpoints, configurations, API credentials and output directories |

The Mercury configuration records the reported target hardware. Run `doctor` on the actual host to check readiness. CUDA 13 requires NVIDIA driver major version 580 or newer; that is a necessary condition, not a substitute for CUDA allocation and model checks. See [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The host does not need a Docker daemon: Apptainer can pull the OCI image directly from Docker Hub and convert it to SIF. Administrators should install Apptainer and the driver according to site policy. The repository does not assume Slurm, a particular partition or administrator privileges.

## 2. Obtain the versioned source and SIF

```bash
git clone https://github.com/williamQ96/schema_study.git
cd schema_study
git checkout v0.1.0
PROJECT="$(pwd)"
# REPO is the Python package directory, not the clone root.
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

For a formal experiment, replace the image tag with the release's OCI digest: `docker://docker.io/plalelab/schema-study@sha256:...`. An OCI digest identifies the registry image; the SIF SHA-256 identifies the converted file. These hashes are not interchangeable. Repeated conversion of the same OCI image can produce different SIF bytes, so retain the actual SIF and bind its hash to the experiment. See [Apptainer OCI documentation](https://apptainer.org/docs/user/main/docker_and_oci.html).

The launcher mounts your selected `$REPO` read-only over the image's embedded package. Pin the source commit before composing and freezing a configuration. Use a new experiment version when source code changes. A configuration frozen in the larger research checkout cannot be assumed to match this public package: the source file set and hashes differ.

## 3. Mount layout

| Host variable | Container path | Access and contents |
| --- | --- | --- |
| `REPO` | `/workspace/high_fidelity_schema_study` | Read-only Python package directory |
| `SOURCES` | `/inputs` | Read-only PDFs, layout/input artifacts, raw datasets, parsed bundles, corpus and configurations |
| `OUTPUTS` | `/outputs` | Writable parsing results, batches, packets and deployment reports |
| `MODELS` | `/models` | Read-only, previously downloaded model snapshots |
| `CACHE` | `/cache` | Writable runtime cache/offload storage |

Create all directories first. Keep output and cache directories distinct from each other and from read-only source mounts. Paths cannot contain colons, commas or newlines that conflict with Apptainer bind syntax. Dataset input is supported through mounts; datasets are not embedded in the image, and `run` does not discover or parse unregistered files automatically.

```bash
launch() {
  bash "$PROJECT/container/apptainer/run.sh" \
    "$SIF" "$REPO" "$SOURCES" "$OUTPUTS" "$MODELS" "$CACHE" -- "$@"
}

# Synthetic inputs and mock backends; no model loading, downloads or API calls.
MERCURY_CPU_ONLY=1 launch offline-demo --output /outputs/software-smoke-v1
```

The launcher uses `--cleanenv --containall`, forwarding only controlled runtime settings and explicitly named secret variables. Set `MERCURY_DRY_RUN=1` to inspect its invocation without starting the container.

## 4. Check GPU readiness

```bash
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
# Preserve CUDA_VISIBLE_DEVICES when provided by the scheduler.
# Use the following only after four GPUs are allocated and no scheduler mask is set:
# export MERCURY_CUDA_DEVICES=0,1,2,3
launch doctor --gpu-count 4 --output /outputs/doctor-mercury-v1.json
```

GPU mode requires an explicit device selection. The launcher rejects a `MERCURY_CUDA_DEVICES` setting that conflicts with scheduler-provided `CUDA_VISIBLE_DEVICES`. For one or two allocated GPUs, use the corresponding visible-device selection and `--gpu-count 1` or `2`. GPU access uses Apptainer `--nv`; see the [GPU documentation](https://apptainer.org/docs/user/main/gpu.html).

Doctor records host and container OS metadata, physical devices from NVIDIA-SMI, Torch-visible devices, CUDA runtime, driver, memory, topology and a small allocation/synchronization probe. A CPU check such as `MERCURY_CPU_ONLY=1 launch doctor --gpu-count 0 ...` does not establish GPU readiness. Doctor does not load model weights or guarantee that a long-context workload will fit in memory.

## 5. Select models through profiles

Copy the example catalog and selection into your host source directory, then edit those copies:

```bash
mkdir -p "$SOURCES/checkpoints"
cp "$REPO/config/mercury_models.example.json" "$SOURCES/models.mercury-v1.json"
cp "$REPO/config/mercury_selection.example.json" "$SOURCES/selection.mercury-v1.json"
```

| File or field | Required information |
| --- | --- |
| Each catalog profile | Model ID, immutable revision where available, deployment/endpoint, actual context window, parameter capabilities, quantization and template settings |
| Transformers `model_id` | Container path `/models/<snapshot>`; download weights beforehand, as formal runs do not download implicitly |
| `runtime.checkpoint_manifest` | Complete checkpoint inventory path and its file-byte SHA-256 |
| Selection `locals` | Three local profile IDs |
| `soft_reference` | Frontier profile ID; its output remains a fallible reference |
| `classifier` | Independently selected profile ID and catalog profile hash; it may reuse local A's deployment |
| `resource_preset` | `h200_1gpu`, `h200_2gpu` or `h200_4gpu` |
| `replicates` / `parameters` | Repetition seeds and role parameters; omitted values inherit the base configuration |

Inventory every local checkpoint that will be used. Write its manifest outside the model directory:

```bash
MERCURY_CPU_ONLY=1 launch checkpoint \
  --model-path /models/model-a-snapshot --revision ACTUAL_IMMUTABLE_REVISION \
  --output /outputs/model-a.manifest.json
cp "$OUTPUTS/model-a.manifest.json" "$SOURCES/checkpoints/"
sha256sum "$SOURCES/checkpoints/model-a.manifest.json"
```

Insert the returned manifest file hash into the corresponding profile:

```json
"checkpoint_manifest": {
  "path": "/inputs/checkpoints/model-a.manifest.json",
  "file_bytes_sha256": "REPLACE_WITH_64_HEX_CHARACTERS_FROM_SHA256SUM"
}
```

Repeat for B/C and any separate classifier checkpoint. Inventory creation reads every model file's bytes. HF snapshot symlink targets must also be accessible inside the container. Added, missing or changed files cause validation to fail.

After qualifying the actual model identity, parameter capabilities and context window, set the selected profiles to `frozen`. Inspect catalog hashes:

```bash
MERCURY_CPU_ONLY=1 launch catalog --catalog /inputs/models.mercury-v1.json
```

Copy the selected classifier's `profile_sha256` into the selection file. Replacing an extraction model does not automatically replace the classifier. Obtain a new hash after changing the catalog profile.

Default local extraction uses temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0 and max_output_tokens 16384, with seeds 1729/2718/3141. The example classifier is greedy. These are experiment settings, not hardware-performance recommendations. A role's parameter object in the selection **replaces the whole object**; specify all required parameters.

Backends include Transformers, compatible Chat Completions services and Responses. A model architecture unsupported by the image's Transformers version requires a new dependency image; a different provider protocol requires an adapter. Interface compatibility does not imply equivalent seed, sampling or reasoning controls. Unsupported settings are rejected. HTTP backends also need a supported exact token-count endpoint or an observation bound to the actual request hash. Unknown counts cannot bypass the context gate.

## 6. Compose and freeze a configuration

```bash
MERCURY_CPU_ONLY=1 launch compose \
  --base /workspace/high_fidelity_schema_study/config/four_category_experiment_v1.json \
  --catalog /inputs/models.mercury-v1.json \
  --selection /inputs/selection.mercury-v1.json \
  --host /workspace/high_fidelity_schema_study/config/mercury_host_v1.json \
  --output /outputs/experiment.draft-v1.json \
  --report /outputs/compose.draft-v1.json
```

Inspect the draft, then freeze to new output filenames:

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

Placeholder model identities, missing classifier pins, unfrozen profiles, unsupported parameters, insufficient context windows and missing checkpoint inventories block freezing. Static `configuration_ready` does not establish execution readiness. `run` checks the actual SIF, source code, full checkpoint bytes, hardware and each request's context requirements again.

## 7. Run and resume

After preparing sources and the corpus with the [user guide](schema_study_user_guide.md), the following command permits real inference and potentially billable API calls:

```bash
# Set OPENAI_API_KEY through your credential manager, not in JSON or the image.
export MERCURY_SECRET_ENV_NAMES=OPENAI_API_KEY
launch run --config /outputs/experiment.frozen-v1.json \
  --corpus /inputs/corpus-v1.json --source-root /inputs \
  --output /outputs/batch-v1 --allow-live
```

For a bounded first invocation, add `--max-jobs 1`. Resume with the same configuration and batch directory. Each invocation rechecks the deployment and reuses only replay-validated results. Summaries are stored under `batch-v1/batches/*.json`. The Mercury `run` command has no `--summary-output` option.

For packet/verify or dataset preparation, switch back to `MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli` and use `MERCURY_CPU_ONLY=1` for CPU operations. Missing independent human evaluation does not block packet construction; semantic accuracy remains unevaluated.

## 8. Build an image yourself

From the clone root, in a Linux Docker environment with network access:

```bash
docker build --platform linux/amd64 \
  -f container/docker/Dockerfile \
  --build-arg IMAGE_VERSION=0.1.0 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t schema-study:local .
```

The image records resolved Python dependencies at `/opt/requirements/pip-freeze.txt`; this release's copy is [pip-freeze-0.1.0-cuda13.txt](../container/docker/pip-freeze-0.1.0-cuda13.txt). Some requirement ranges and the base image tag can resolve differently in a later build. For strict reproduction, retain the actual image digest/SIF and resolved dependency inventory. Changing the CUDA base does not automatically change the CUDA version of the pip-installed PyTorch wheel.

Alternatively, build a runtime-only SIF from the native recipe, then mount source code with the launcher:

```bash
bash "$PROJECT/container/apptainer/build.sh" "$WORK/images/schema-study-local.sif"
```

Building installs dependencies over the network, but does not download model weights. It requires the Apptainer build permissions allowed by the site; the script does not elevate privileges automatically. A rebuilt SIF has a new file identity and requires a new configuration freeze.

## 9. Troubleshooting and operating limits

| Symptom | Action |
| --- | --- |
| `No module named high_fidelity_schema_study` | Point REPO at the package directory inside the clone, not the clone root |
| Missing or conflicting GPU selection | Use the scheduler's allocated visible devices without a conflicting override |
| `actual_sif_identity_mismatch_or_missing` | Use the frozen SIF and the launcher that supplies its actual hash; freeze a new configuration for a new SIF |
| `active_source_identity_mismatch` | Check out the frozen source version; do not reuse a research-checkout configuration with a different public source tree |
| Checkpoint identity mismatch | Check all files and symlink targets; inventory a changed checkpoint as a new version |
| Exact request token count unavailable | Check the provider's count endpoint or supply an exact request-bound observation |
| Dataset source unavailable | Place it under SOURCES, reference it relatively in the corpus and write outputs under /outputs |
| Output already exists | Choose a new standalone artifact name; use the existing directory when resuming a batch |

The current runner is serial and releases Transformers models after each call. It does not start vLLM or configure tensor parallelism, cross-node InfiniBand or NCCL. `device_map=auto` places model layers across devices. The per-GPU 120GiB weight-placement budget does not include all KV-cache and activation peaks, and is not a hard limit on memory use. Four 141GB GPUs are not a single GPU memory space. See [Accelerate's large-model inference explanation](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference).

Container software validation, Mercury hardware qualification, real experiments and independent sampled semantic evaluation are separate records. The release does not claim that the latter three have been completed.
