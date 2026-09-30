# Mercury runbook: environment setup, inputs, inference and Scheduler V2

Updated 29 September 2026 (America/Los_Angeles). Commands use **Linux Bash**.
Run compute commands on `mercury.nic.uoregon.edu`, after arranging site access
and GPU use. Substitute your own account for `YOUR_UO_ACCOUNT`.

The runnable repository is [williamQ96/schema_study](https://github.com/williamQ96/schema_study).
This guide covers a new installation, a complete mock scheduler run, preparation
for real inference, operation of a prepared live deployment, and inspection of
William's completed P06 run. It does not start an experiment merely by installing
the environment.

## Contents

1. [Log in and choose storage](#1-log-in-and-choose-storage)
2. [Install or select Apptainer](#2-install-or-select-apptainer)
3. [Clone the code and create the control environment](#3-clone-the-code-and-create-the-control-environment)
4. [Obtain an image and run the CPU smoke test](#4-obtain-an-image-and-run-the-cpu-smoke-test)
5. [Run Scheduler V2 from scratch with mock models](#5-run-scheduler-v2-from-scratch-with-mock-models)
6. [Prepare a real paper and dataset](#6-prepare-a-real-paper-and-dataset)
7. [Stage weights and freeze model configuration](#7-stage-weights-and-freeze-model-configuration)
8. [Prepare a real Scheduler V2 deployment](#8-prepare-a-real-scheduler-v2-deployment)
9. [Start, monitor, pause, drain and resume](#9-start-monitor-pause-drain-and-resume)
10. [Finalization, verification and collection](#10-finalization-verification-and-collection)
11. [Use William's existing Mercury installation](#11-use-williams-existing-mercury-installation)
12. [Troubleshooting](#12-troubleshooting)

### Which version am I running?

| Component | Identity / purpose |
| --- | --- |
| Source used by this tutorial | `25daf4762d9c38c43d666e1d95a18036262513f9`; public V13 r2 snapshot plus Scheduler V2 |
| Published runtime image | `plalelab/schema-study:0.1.0-cuda13`; an earlier release, sufficient for the documented offline demo |
| William's P06 V13 r3 | Separately frozen source and inputs on Mercury; the public r2 checkout is **not** that condition |
| Site allocation | Grants access to physical GPUs; arrange this separately |
| Scheduler V2 | Application coordinator, resident workers, replay validators, independent watchdog and CPU finalizer |

Do not check out `v0.1.0` if you need `scheduler_v2`: that tag predates it.
The commands below pin the published source instead of following a changing
`main`. New images, source trees, inputs, profiles or sampling settings require
a new condition. Mock results and passing packet checks do not establish
semantic fidelity.

## 1. Log in and choose storage

**On your workstation:** complete the gateway logins required for your account.
The September deployment used Orthus and Sphinx, then jumped through Orthus.
Site authentication prompts are interactive; no password belongs in a command.

```bash
ssh YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu
# Exit that session, then complete the second gateway login if required.
ssh YOUR_UO_ACCOUNT@sphinx.nic.uoregon.edu
# Exit, then open Mercury:
ssh -J YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu YOUR_UO_ACCOUNT@mercury.nic.uoregon.edu
```

**On Mercury, in a Bash session:**

```bash
hostname -f
uname -m
python3 --version
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
df -h "$HOME" /var/tmp
command -v sbatch || true
command -v squeue || true
```

The observed deployment host is `mercury`, Linux x86_64, with four H200 NVL
GPUs. On the read-only check for this guide, `sbatch` and `squeue` were absent
from the login environment's PATH. This guide therefore uses the application
scheduler directly and does not invent a Slurm partition or allocation command.
Confirm the site's current allocation procedure before live GPU use.

Use shared storage for source, inputs, checkpoints and results; use host-local
storage for the SQLite ledger, locks and scratch. Shared free space is not your
personal quota. The following creates a new work directory each time:

```bash
umask 077
export RUN_ID="mercury-$(date -u +%Y%m%dT%H%M%SZ)"
export WORK="$HOME/schema-study-work/$RUN_ID"
export LOCAL_BASE="/var/tmp/schema-study-$(id -un)"
mkdir -p "$WORK"/{images,sources,outputs,models,cache,tools,oci-cache}
mkdir -p "$LOCAL_BASE/$RUN_ID"/{tmp,control}
export APPTAINER_CACHEDIR="$WORK/oci-cache"
export APPTAINER_TMPDIR="$LOCAL_BASE/$RUN_ID/tmp"
df -h "$WORK" "$APPTAINER_TMPDIR"
```

Allow space for the SIF, OCI layers and conversion scratch, plus the selected
weights and inputs. The model and Caravan archives can dominate disk use.
Avoid colons, commas and newlines in bind paths.

## 2. Install or select Apptainer

If an administrator supplies Apptainer, select that installation and skip the
installer. `module avail apptainer` is useful only if your shell has `module`.
William's existing installation is described in section 11.

```bash
command -v apptainer || true
command -v module >/dev/null 2>&1 && module avail apptainer
```

If Apptainer is unavailable and the site permits a user installation, use the
[official relocatable installer](https://apptainer.org/docs/admin/1.5/installation.html#install-unprivileged-from-pre-built-binaries).
It requires `curl`, `rpm2cpio`, `cpio`, and supported user namespaces/FUSE.
The downloaded versioned script supports `-v` to select the Apptainer version.

```bash
command -v curl
command -v rpm2cpio
command -v cpio
curl -fsSL \
  https://raw.githubusercontent.com/apptainer/apptainer/v1.5.4/tools/install-unprivileged.sh \
  -o "$WORK/tools/install-unprivileged.sh"
sha256sum "$WORK/tools/install-unprivileged.sh" > "$WORK/tools/installer.sha256"
bash "$WORK/tools/install-unprivileged.sh" -v 1.5.4 "$WORK/tools/apptainer-1.5.4"
export PATH="$WORK/tools/apptainer-1.5.4/bin:$PATH"
apptainer --version
```

Do not replace the site's Singularity installation or modify system container
configuration. If a pull reports a duplicate `unqualified-search-registries`
key, see the specific remedy in section 12.

## 3. Clone the code and create the control environment

```bash
git clone https://github.com/williamQ96/schema_study.git "$WORK/source"
git -C "$WORK/source" checkout --detach 25daf4762d9c38c43d666e1d95a18036262513f9
export PROJECT="$WORK/source"
export REPO="$PROJECT/high_fidelity_schema_study"
export SOURCES="$WORK/sources"
export OUTPUTS="$WORK/outputs"
export MODELS="$WORK/models"
export CACHE="$WORK/cache"
git -C "$PROJECT" rev-parse HEAD > "$WORK/source-commit.txt"
cd "$PROJECT"

python3.12 -m venv "$WORK/control-venv"
source "$WORK/control-venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT/requirements_four_category_offline.txt"
python -m pip check
python -m pip freeze --all > "$WORK/control-pip-freeze.txt"
export CONTROL_PYTHON="$WORK/control-venv/bin/python"
export PYTHONPATH="$PROJECT:$PROJECT/scripts"
python -m high_fidelity_schema_study.four_category.cli --help
python -m scheduler_v2.cli --help
```

Mercury's checked `python3` was Python 3.12.13; use a site-provided Python 3.12
if `python3.12` is not on your PATH. The host environment handles parsing and
CPU orchestration. GPU workers use the separate container environment; do not
install a second CUDA/Torch stack in the host control venv.

Check the relevant CPU paths before introducing real inputs:

```bash
python -m pytest -q \
  tests/test_scheduler_v2_cli.py \
  tests/test_scheduler_v2_bootstrap.py \
  tests/test_scheduler_v2_policy.py
python -m high_fidelity_schema_study.four_category.cli offline-demo \
  --output "$OUTPUTS/host-offline-demo"
python -m json.tool "$OUTPUTS/host-offline-demo/report.json"
```

Keep the exported variables for subsequent sections. When returning later,
restore them from your chosen work directory; do not create a second RUN_ID
when you intend to inspect or resume an existing run.

```bash
declare -p RUN_ID WORK LOCAL_BASE PROJECT REPO SOURCES OUTPUTS MODELS CACHE \
  CONTROL_PYTHON PYTHONPATH PATH APPTAINER_CACHEDIR APPTAINER_TMPDIR \
  > "$WORK/runbook-env.sh"
# Later, substitute your existing absolute WORK directory:
# source /ABSOLUTE/WORK/runbook-env.sh
```

After choosing SIF below, append `declare -p SIF` to that file. Functions such
as `launch` and `sched` must also be redefined when opening a new shell.

## 4. Obtain an image and run the CPU smoke test

### Published image: offline smoke and basic deployment

```bash
export SIF="$WORK/images/schema-study-0.1.0-cuda13.sif"
apptainer pull "$SIF" \
  docker://docker.io/plalelab/schema-study@sha256:c6abbb7f853e295dcce0634029eb991a1f1d488bb548c986b40998b1476db45e
sha256sum "$SIF" > "$SIF.sha256"
declare -p SIF >> "$WORK/runbook-env.sh"

launch() {
  bash "$PROJECT/container/apptainer/run.sh" \
    "$SIF" "$REPO" "$SOURCES" "$OUTPUTS" "$MODELS" "$CACHE" -- "$@"
}
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.cli
MERCURY_CPU_ONLY=1 launch offline-demo --output /outputs/container-offline-demo
python -m json.tool "$OUTPUTS/container-offline-demo/report.json"
```

The launcher overlays the selected source directory read-only. The SIF and
source are separate identities. OCI digests and converted SIF byte hashes are
also different; retain both. Expected smoke observations are zero live model
requests and a passing packet verification report.

### Live Scheduler V2: check the image's interpreter and libraries

The live V2 worker command expects `/opt/phase1-venv/bin/python`. An image
built with the native `four-category.def` instead uses
`/opt/four-category-venv/bin/python`, so that native recipe is not a drop-in
image for this V2 launcher. Use an appropriate retained image or build through
`container/docker/Dockerfile`. No newly qualified V13 image is implied by the
old public tag.

```bash
apptainer exec --cleanenv "$SIF" /opt/phase1-venv/bin/python -c \
  'import torch, transformers, xgrammar; print(torch.__version__, transformers.__version__)'
```

If the retained image does not support your model architecture or runtime
settings, build a new image. **On a Linux amd64 Docker build machine**, clone
the same source commit and run:

```bash
git clone https://github.com/williamQ96/schema_study.git schema-study-build
cd schema-study-build
git checkout --detach 25daf4762d9c38c43d666e1d95a18036262513f9
docker build --platform linux/amd64 -f container/docker/Dockerfile \
  --build-arg IMAGE_VERSION=mercury-runbook \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t schema-study:mercury-runbook .
docker image inspect schema-study:mercury-runbook > image-inspect.json
docker save schema-study:mercury-runbook -o schema-study-runtime.tar
sha256sum schema-study-runtime.tar > schema-study-runtime.tar.sha256
scp -o ProxyJump=YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu \
  schema-study-runtime.tar schema-study-runtime.tar.sha256 image-inspect.json \
  YOUR_UO_ACCOUNT@mercury.nic.uoregon.edu:/ABSOLUTE/PATH/TO/YOUR/WORK/images/
```

**Back on Mercury:** use the absolute WORK path created in section 1.
[Docker archives use `docker-archive:PATH`](https://apptainer.org/docs/user/1.5/docker_and_oci.html#containers-in-docker-archive-files),
with a single colon.

```bash
(cd "$WORK/images" && sha256sum -c schema-study-runtime.tar.sha256)
export SIF="$WORK/images/schema-study-runbook.sif"
apptainer build "$SIF" "docker-archive:$WORK/images/schema-study-runtime.tar"
sha256sum "$SIF" > "$SIF.sha256"
declare -p SIF >> "$WORK/runbook-env.sh"
apptainer exec --cleanenv "$SIF" /opt/phase1-venv/bin/python -m pip freeze --all \
  > "$WORK/images/runtime-pip-freeze.txt"
```

Build requirements contain some ranges; a later build can resolve different
dependencies. This creates a new runtime to qualify, rather than reproducing
William's retained image byte-for-byte.

After arranging use of the intended devices, check CUDA. Use the site-supplied
mask if present; the following explicit mask is only for an allocation of all
four physical GPUs:

```bash
# Only if all four devices were allocated and no site mask is already set:
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
launch doctor --gpu-count 4 --output /outputs/gpu-doctor.json
```

`doctor` checks small CUDA allocations. It does not qualify model loading,
full-context prefill/decode, or output quality.

## 5. Run Scheduler V2 from scratch with mock models

This is the complete application-scheduler exercise. It uses a generated paper,
CSV, deterministic transports and isolated mock lease directories. It needs no
SIF, real GPU allocation, checkpoints or API keys. The synthetic GPU labels in
its policy are logical test slots. It does not reserve production GPUs.

```bash
export MOCK_JOB="$WORK/mock-v2"
export MOCK_LOCAL="$LOCAL_BASE/$RUN_ID/mock-v2"
python - <<'PY'
import os
from pathlib import Path
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.workflow import tasks_for_config
from high_fidelity_schema_study.four_category.scheduler import compile_condition, default_policy
from scheduler_v2.io import file_sha, write_once

job = Path(os.environ['MOCK_JOB'])
if job.exists():
    raise SystemExit('Choose a new mock directory; do not overwrite a prior run.')
project = Path(os.environ['PROJECT']).resolve()
local = Path(os.environ['MOCK_LOCAL']).resolve()
config, corpus = make_fixture(job / 'sources')
config['classification'].update(
    profile_id=config['roles']['locals'][0], protocol='classification-anchors/v3',
    grouping={'max_units': 1, 'max_target_chars': 4000})
config['extraction_input_protocol'] = 'extraction-pointers/v5'
config['extraction_grouping'] = {
    'max_windows': 1, 'max_target_chars': 240, 'max_mentions': 32, 'max_facts': 48}
config['execution']['grouped_mode'] = 'all-groups/v1'
for profile in config['profiles']:
    profile['context_window'] = 131072
    profile['runtime']['structured_output'] = {'engine': 'xgrammar', 'version': '0.2.8', 'channel': 'json'}
config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
config['task_hashes'] = {k: t['task_sha256'] for k, t in tasks_for_config(config).items()}
policy = default_policy()
for i, worker in enumerate(policy['workers']):
    if worker['gpu_ids']:
        worker['gpu_ids'] = ['GPU-RUNBOOK-MOCK-' + str(i)]
condition = compile_condition(config, corpus, policy, live=False)
write_once(job / 'frozen-condition.json', condition)
deployment = {
    'deployment_id': 'runbook-mock', 'root': str(job / 'run'),
    'local_root': str(local), 'sources': str(job / 'sources'),
    'science_source': str(project), 'runtime_source': str(project),
    'condition_file_sha256': file_sha(job / 'frozen-condition.json'),
    'operational_files': {p.relative_to(project).as_posix(): file_sha(p)
                          for p in sorted((project / 'scheduler_v2').glob('*.py'))},
    'execution_identity': {'mode': 'synthetic_mock', 'source_commit': '25daf4762d9c38c43d666e1d95a18036262513f9'},
    'authority_lock': str(local / 'dispatch-authority.lock'),
    'local_leases': str(local / 'mock-leases'),
    'legacy_leases': str(local / 'mock-legacy-leases'),
    'qualification': str(job / 'unused-mock-qualification.json')}
write_once(job / 'deployment.json', deployment)
print(job / 'deployment.json')
PY

python -m scheduler_v2.bootstrap --config "$MOCK_JOB/deployment.json" \
  --condition "$MOCK_JOB/frozen-condition.json"
python -m scheduler_v2.cli --config "$MOCK_JOB/deployment.json" verify
python -m scheduler_v2.cli --config "$MOCK_JOB/deployment.json" start
python -m scheduler_v2.cli --config "$MOCK_JOB/deployment.json" status
```

`start` returns after launching the supervisor. Wait for `terminal: true`, then
for `run/finalization-status.json` to report `status: pass`:

```bash
watch -n 5 "$CONTROL_PYTHON -m scheduler_v2.cli --config $MOCK_JOB/deployment.json status"
# Ctrl-C exits watch; it does not terminate the scheduler.
python -m scheduler_v2.report --root "$MOCK_JOB/run"
python -m json.tool "$MOCK_JOB/run/finalization-status.json"
python -m json.tool "$MOCK_JOB/packet-release.json"
```

The supervisor finalizes independently. The watchdog remains a separate
process after completion; its identity and logs stay in MOCK_LOCAL. Keep that
directory while inspecting the run. Finalization can pass while a returned
mock answer has rejections: inspect admission separately from terminal status.

## 6. Prepare a real paper and dataset

For a **new native text-layer tutorial input**, stage one PDF and one dataset.
The CSV example below can be replaced with another supported format. Supply
actual linkage evidence from the paper; do not invent a match from filenames.

**On the workstation:**

```bash
scp -o ProxyJump=YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu paper.pdf \
  YOUR_UO_ACCOUNT@mercury.nic.uoregon.edu:/ABSOLUTE/WORK/sources/paper.pdf
scp -o ProxyJump=YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu data.csv \
  YOUR_UO_ACCOUNT@mercury.nic.uoregon.edu:/ABSOLUTE/WORK/sources/data.csv
```

**On Mercury:** the following binds and parses the actual files, then writes
a complete corpus manifest using the resulting IDs and hashes.

```bash
export LINK_EVIDENCE='REPLACE with the actual page quote or persistent identifier linking this paper and dataset'
python - <<'PY'
import os
from pathlib import Path
from high_fidelity_schema_study.paper_layout_evidence import preprocess_layout_pdf
from high_fidelity_schema_study.paper_layout_evidence_v3 import upgrade_bundle
from high_fidelity_schema_study.four_category.paper import prepare_paper
from high_fidelity_schema_study.four_category.dataset import parse_dataset
from high_fidelity_schema_study.four_category.common import write_new

root = Path(os.environ['SOURCES'])
evidence = os.environ['LINK_EVIDENCE']
if evidence.startswith('REPLACE'):
    raise SystemExit('Set actual paper/dataset linkage evidence first.')
folder = root / 'P01-native-v3'
folder.mkdir(exist_ok=False)
legacy, _, _ = preprocess_layout_pdf('P01', root / 'paper.pdf')
layout, reading, paper_input, _ = upgrade_bundle(legacy)
write_new(folder / 'layout.json', layout)
write_new(folder / 'input.json', paper_input)
with (folder / 'reading.txt').open('x', encoding='utf-8') as stream:
    stream.write(reading)
paper = prepare_paper(folder / 'layout.json', folder / 'reading.txt',
                      folder / 'input.json', root / 'paper.pdf', root=root)
dataset = parse_dataset(root / 'data.csv', root=root, sample_limit=200)
if dataset['status'] != 'pass':
    raise SystemExit('Dataset needs review: ' + repr(dataset.get('issues')))
write_new(root / 'dataset-bundle.json', dataset)
write_new(root / 'paper-source.json', paper)
corpus = {
    'schema_version': 'four-category-corpus/v1',
    'papers': [{'paper_id': 'P01', 'source': paper}],
    'datasets': [{'dataset_id': dataset['dataset_id'],
                  'bundle_path': 'dataset-bundle.json', 'bundle_sha256': dataset['bundle_sha256']}],
    'matches': [{'match_id': 'P01-dataset', 'paper_id': 'P01',
                 'dataset_id': dataset['dataset_id'],
                 'linkage': {'status': 'candidate', 'evidence': [evidence]}}]}
write_new(root / 'corpus.json', corpus)
PY
```

This uses PDF text and native layout, **not MinerU/OCR or V13 r3 regions**.
Its matching grouped tutorial protocol is `classification-anchors/v3` with
`extraction-pointers/v5`. It is a new experiment, not a recreation of P06.
For a larger corpus, use the manifest-based `parse-datasets` commands and
many-to-many linkage examples in the [user guide](schema_study_user_guide.md).
CSV sampling and absent normalized values do not establish full legal domains.

### MinerU-prepared V13 inputs

V13 uses a separately prepared source bundle. To inspect or replay William's
run, retain its saved PDF, MinerU outputs, adapters, layout/input artifacts,
region/group plans and manifests together. Do not replace them with the native
v3 input above. The public checkout does not include those original inputs.

For new MinerU preprocessing, provision its separate image and complete parser
model directory first. The published wrapper exposes the following commands:

```bash
export MINERU_SIF=/ABSOLUTE/PATH/TO/mineru.sif
export MINERU_MODELS=/ABSOLUTE/PATH/TO/mineru-models
mkdir -p "$OUTPUTS/mineru" "$CACHE/mineru"
bash "$PROJECT/container/apptainer/run-mineru.sh" \
  "$MINERU_SIF" "$REPO" "$SOURCES" "$OUTPUTS/mineru" "$MINERU_MODELS" "$CACHE/mineru" \
  -- inventory --model-root /models --output /outputs/model-manifest.json
bash "$PROJECT/container/apptainer/run-mineru.sh" \
  "$MINERU_SIF" "$REPO" "$SOURCES" "$OUTPUTS/mineru" "$MINERU_MODELS" "$CACHE/mineru" \
  -- parse --pdf /inputs/paper.pdf --output /outputs/P01-attempt-01 \
  --model-root /models --model-manifest /outputs/model-manifest.json --tier basic --ocr-mode txt
```

The Basic wrapper is CPU-only but uses learned parser weights; it performs no
schema-LLM inference. It runs offline and cannot acquire missing parser models.
Parsing alone does not create a frozen V13 corpus: the version-specific adapter
and source replay must also pass. Site-specific `prepare_v13_sources.py` and
repair launchers expect their own frozen condition/baseline files. Obtain the
matching bundle from its owner rather than substituting arbitrary JSON.

## 7. Stage weights and freeze model configuration

Real inference requires three selected local profiles and a separately bound
classifier. The frontier profile is distinct and is deferred by Scheduler V2.
No model weights or private research inputs are in GitHub or the image.

If the weights are not already staged, download a full immutable revision.
The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
supports `hf download --revision` and `--local-dir`:

```bash
python -m pip install huggingface_hub
# For a gated model, complete hf auth login interactively first.
export MODEL_REPOSITORY='REPLACE_ORG/REPLACE_MODEL'
export MODEL_REVISION='REPLACE_WITH_FULL_IMMUTABLE_COMMIT'
hf download "$MODEL_REPOSITORY" --revision "$MODEL_REVISION" \
  --local-dir "$MODELS/local-a"
```

Repeat for local-b/local-c. Review each model's license and architecture before
selection. Use the actual qualified context window, not the template sentinel
`1`. Inventory checkpoints **after** download is finished:

```bash
mkdir -p "$SOURCES/checkpoints"
python -m high_fidelity_schema_study.four_category.mercury checkpoint \
  --model-path "$MODELS/local-a" --revision "$MODEL_REVISION" \
  --output "$SOURCES/checkpoints/local-a.json"
sha256sum "$SOURCES/checkpoints/local-a.json"
cp "$REPO/config/mercury_models.example.json" "$SOURCES/models.json"
cp "$REPO/config/mercury_selection.example.json" "$SOURCES/selection.json"
```

Repeat the inventory command for B/C with **their own revisions**. Edit the
copied JSON, using an editor available on the host, for example:

```bash
vi "$SOURCES/models.json"
vi "$SOURCES/selection.json"
python -m high_fidelity_schema_study.four_category.mercury catalog \
  --catalog "$SOURCES/models.json"
```

Fill in absolute **host** model paths, immutable revisions, checkpoint manifest
paths/file hashes, measured context windows, runtime settings, structured-output
settings and supported parameters. The live V2 launcher mounts each model path
at the same absolute path inside the container. Set qualified selected profiles
to `frozen`; paste the reported classifier profile hash into selection.
Parameter objects replace an entire role parameter object.

For this local-only tutorial, explicitly replace the selected frontier slot
with a **mock profile that will be deferred**. This avoids requiring a remote
provider configuration to freeze a run that makes no reference requests. It
does not produce a soft reference and must not be labeled a frontier result:

```bash
python - <<'PY'
import os
from pathlib import Path
from high_fidelity_schema_study.four_category.common import read_json, write_new
root = Path(os.environ['SOURCES'])
catalog = read_json(root / 'models.json')
selection = read_json(root / 'selection.json')
reference_id = selection['soft_reference']
catalog['profiles'] = [p for p in catalog['profiles'] if p['profile_id'] != reference_id]
catalog['profiles'].append({'profile_id': reference_id, 'backend': 'mock',
    'deployment': 'mock', 'model_id': 'DEFERRED-TUTORIAL-MOCK', 'revision': 'mock-deferred-v1',
    'endpoint': None, 'status': 'frozen', 'context_window': 131072,
    'capabilities': {'supported_parameters': ['max_output_tokens'], 'supports_system_role': True},
    'runtime': {}})
write_new(root / 'models.local-only.json', catalog)
PY
```

For the native-v3 tutorial, create a matching grouped base configuration:

```bash
python - <<'PY'
import os
from pathlib import Path
from high_fidelity_schema_study.four_category.common import read_json, write_new
from high_fidelity_schema_study.four_category.workflow import tasks_for_config
base = read_json(Path(os.environ['REPO']) / 'config/four_category_experiment_v1.json')
base['classification'].update(protocol='classification-anchors/v3',
    grouping={'max_units': 48, 'max_target_chars': 12000})
base['extraction_input_protocol'] = 'extraction-pointers/v5'
base['extraction_grouping'] = {
    'max_windows': 96, 'max_target_chars': 12000, 'max_mentions': 32, 'max_facts': 48}
base['execution']['grouped_mode'] = 'all-groups/v1'
base['task_hashes'] = {k: t['task_sha256'] for k, t in tasks_for_config(base).items()}
write_new(Path(os.environ['SOURCES']) / 'base-grouped.json', base)
PY

export SIF_SHA256="$(sha256sum "$SIF" | awk '{print $1}')"
python -m high_fidelity_schema_study.four_category.mercury compose \
  --base "$SOURCES/base-grouped.json" --catalog "$SOURCES/models.local-only.json" \
  --selection "$SOURCES/selection.json" --host "$REPO/config/mercury_host_v1.json" \
  --freeze --image-sha256 "$SIF_SHA256" \
  --output "$SOURCES/experiment.frozen.json" --report "$OUTPUTS/compose.json"
python -m high_fidelity_schema_study.four_category.cli plan \
  --config "$SOURCES/experiment.frozen.json" --corpus "$SOURCES/corpus.json" \
  --output "$OUTPUTS/job-plan.json"
```

Composition validates identities and static settings. It does **not** perform
capacity or output qualification. The basic compiler requires three distinct
replicates; the historical single-repeat P06 condition was frozen through its
separate r3 path. Do not edit that historical condition to satisfy this tutorial.

## 8. Prepare a real Scheduler V2 deployment

This section takes the **frozen experiment** and corpus from sections 6–7,
measures capacity, and creates a worker policy and qualification record. For a
V13 condition, obtain the version-specific frozen bundle and use its source
and qualification workflow instead; the native tutorial does not implement
V13 r3's input or output qualification.

### Capacity and worker policy

For the native tutorial, select a **common layout** for all local models:
`h200_1gpu`, `h200_2gpu`, or `h200_4gpu` in `selection.json`, with an allocation
of the same width. This creates one resident GPU worker that runs the selected
models in sequence. It exercises V2 without assuming that arbitrary models
have already qualified for the historical heterogeneous layout.

Get the UUIDs and physical memory of the devices you are allowed to use:

```bash
nvidia-smi --query-gpu=index,uuid,memory.total --format=csv
# Set only the UUIDs in your actual allocation; do not select another owner's GPUs.
# Example shape for a four-GPU allocation:
# export CUDA_VISIBLE_DEVICES=GPU-UUID0,GPU-UUID1,GPU-UUID2,GPU-UUID3
export CAPACITY_JOB="$WORK/tutorial-capacity"
export QUALIFICATION="$CAPACITY_JOB/qualification.json"
export POLICY="$CAPACITY_JOB/policy.qualified.json"
export LOCAL_LEASES="$LOCAL_BASE/gpu-leases"
export LEGACY_LEASES="$WORK/legacy-gpu-leases"
# If sharing William's deployments, use the existing legacy namespace instead:
# export LEGACY_LEASES=/storage/users/williamq/schema-study-deployment-20260926/scheduler-gpu-leases
```

The following is **live capacity testing**, not a CPU installation check. It
uses the same source/SIF/profile settings as the planned worker, holds both
lease namespaces, refuses occupied devices and bounds each owned probe to 30
minutes. It retains raw measurements even on failure. Full-context prefill and
eight synthetic decode tokens are capacity evidence, not paper-output quality.

```bash
cat > "$WORK/tools/measure-capacity.py" <<'PY'
import os, signal, subprocess, time
from contextlib import ExitStack
from pathlib import Path
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.common import read_json
from high_fidelity_schema_study.four_category.mercury import code_identity, checkpoint_binding_errors
from high_fidelity_schema_study.four_category.scheduler import default_policy
from scheduler_v2.io import Lock, digest, file_sha, write_once
from scheduler_v2.processes import identity, same, gpu_processes

job = Path(os.environ['CAPACITY_JOB']).resolve()
job.mkdir(exist_ok=False)
project = Path(os.environ['PROJECT']).resolve()
sources = Path(os.environ['SOURCES']).resolve()
config = read_json(sources / 'experiment.frozen.json')
active = set(config['roles']['locals']) | {config['classification']['profile_id']}
profiles = [p for p in config['profiles'] if p['profile_id'] in active]
errors = checkpoint_binding_errors({'profiles': profiles}, verify_bytes=True)
if errors:
    raise SystemExit(repr(errors))
gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
if len(gpus) not in (1, 2, 4) or len(set(gpus)) != len(gpus) or not all(g.startswith('GPU-') for g in gpus):
    raise SystemExit('Use the actual allocated GPU UUIDs (one, two or four).')
if config['deployment']['required_gpu_count'] != len(gpus):
    raise SystemExit('Allocation width differs from the composed resource preset.')
image = Path(os.environ['SIF']).resolve()
if config['deployment']['image_file_bytes_sha256'] != file_sha(image):
    raise SystemExit('Image differs from the composed experiment.')
lease_roots = [Path(os.environ[k]) for k in ('LOCAL_LEASES', 'LEGACY_LEASES')]
for root in lease_roots:
    root.mkdir(parents=True, exist_ok=True)
memory_csv = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total',
    '--format=csv,noheader,nounits'], text=True)
physical = {line.split(',')[0].strip(): int(line.split(',')[1].strip()) * 1024**2
            for line in memory_csv.splitlines()}
worker = {'worker_id': 'models', 'gpu_ids': gpus, 'cpu_threads': 8,
    'ram_bytes': 128 * 1024**3, 'allowed_profile_ids': sorted(active),
    'gpu_budget_bytes': [physical[g] for g in gpus], 'gpu_reserve_bytes': 12 * 1024**3}
rows = {}
for profile in profiles:
    pid = profile['profile_id']
    profile_path = job / 'profiles' / (pid + '.json')
    write_once(profile_path, profile)
    output = job / pid
    output.mkdir()
    scratch = Path(os.environ['LOCAL_BASE']) / os.environ['RUN_ID'] / 'capacity' / pid
    scratch.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            for root in lease_roots:
                stack.enter_context(Lock(root / (digest(gpu) + '.lock')))
        if any(g in gpus for g, _ in gpu_processes()):
            raise SystemExit('Allocated GPU is occupied; resolve ownership before probing.')
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))
               and k not in {'APPTAINER_BIND', 'APPTAINER_BINDPATH', 'SINGULARITY_BIND', 'SINGULARITY_BINDPATH'}}
        env.update(APPTAINERENV_PYTHONPATH='/scientific',
            APPTAINERENV_CUDA_VISIBLE_DEVICES=','.join(gpus), APPTAINERENV_OMP_NUM_THREADS='8',
            APPTAINERENV_HF_HUB_OFFLINE='1', APPTAINERENV_TRANSFORMERS_OFFLINE='1',
            APPTAINERENV_HF_HOME='/scratch/hf', APPTAINERENV_XDG_CACHE_HOME='/scratch/cache',
            APPTAINERENV_TMPDIR='/scratch')
        command = ['apptainer', 'exec', '--nv', '--cleanenv', '--containall']
        for src, dst, mode in [(project, '/scientific', 'ro'), (job, '/probe', 'rw'),
                (scratch, '/scratch', 'rw'), (sources, sources, 'ro'),
                (Path(profile['model_id']), profile['model_id'], 'ro')]:
            command += ['--bind', f'{src}:{dst}:{mode}']
        command += [str(image), '/opt/phase1-venv/bin/python', '-m',
            'high_fidelity_schema_study.four_category.resident_qualification',
            '--profile', '/probe/profiles/' + pid + '.json', '--output', '/probe/' + pid + '/measurement']
        write_once(output / 'intent.json', {'command': command, 'time': time.time()})
        with (output / 'process.log').open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                stdin=subprocess.DEVNULL, start_new_session=True)
            expected = identity(child.pid)
            if not expected:
                raise RuntimeError('Child process identity unavailable.')
            write_once(output / 'process.json', expected)
            try:
                code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                if same(expected):
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if same(expected):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
                code = -1
        if any(g in gpus for g, _ in gpu_processes()):
            raise RuntimeError('Probe release not confirmed; reconcile before further work.')
        result = output / 'measurement/result.json'
        measured = read_json(result) if result.exists() else {'status': 'failed'}
        if result.exists() and (measured.get('profile_sha256') != profile_hash(profile)
                or measured.get('source_tree_sha256') != digest(code_identity())):
            raise RuntimeError('Capacity measurement identity mismatch.')
        rows[profile_hash(profile)] = measured
        if code:
            rows[profile_hash(profile)]['status'] = 'failed'
        write_once(output / 'release.json', {'exit_code': code, 'gpu_release_verified': True})
record = {'status': 'pass' if all(r['status'] == 'pass' for r in rows.values()) else 'failed',
    'purpose': 'synthetic_capacity_not_output_qualification',
    'image_file_bytes_sha256': file_sha(image), 'source_tree_sha256': digest(code_identity()),
    'profiles': rows}
write_once(Path(os.environ['QUALIFICATION']), record)
if record['status'] != 'pass':
    raise SystemExit('Capacity failed; keep measurements and do not start production.')
policy = default_policy()
policy['workers'] = [worker, {'worker_id': 'dataset-0', 'gpu_ids': [],
    'cpu_threads': 2, 'ram_bytes': 8 * 1024**3}]
policy['qualification'] = {'path': os.environ['QUALIFICATION'],
    'file_bytes_sha256': file_sha(Path(os.environ['QUALIFICATION']))}
write_once(Path(os.environ['POLICY']), policy)
PY
python "$WORK/tools/measure-capacity.py"
```

The later `live_gate` also checks the measured peaks plus reserve against GPU
budgets and CPU peaks against RAM budgets. A measured pass can still fail
that headroom gate. Keep profile settings/source unchanged between probe and
freeze. Bind paths must satisfy the restrictions in section 1.

For an existing V13 candidate, the retained capacity launcher acquires both
lease namespaces, checks process ownership, and imposes that timeout:

```bash
# Only for a NEW, fully staged V13 candidate with these exact input files:
# previous-deployment.json, config.json, profiles/, policy-pending.json.
# Do not point this at an old candidate or William's completed P06 job.
export V13_JOB=/ABSOLUTE/PATH/TO/NEW/STAGED/V13/CANDIDATE
python "$PROJECT/scripts/v13_capacity_launcher.py"
```

That helper is site-specific: its scratch path names William's account.
For another account, use an explicitly adapted and pinned launcher with your
own paths; do not run it blindly. A new model/layout needs measured capacity
and actual request token/grammar/output checks before production. Retain failed
measurements and raw qualification answers. The previous direct-P06 output
qualification waiver does not carry over to a new experiment.

The prepared policy must use `matrix-scheduler-policy/v1`, defer the soft
reference, and cover every active local/classifier profile with disjoint GPU
workers plus a CPU dataset worker. For **live admission**, use allocated GPU
UUIDs, matching per-GPU byte budgets and a positive reserve. Qualification must
bind the actual source/SIF and each effective profile hash, GPU width,
full-context pass, observed GPU peaks and CPU peak. Use measured values;
`doctor` observations cannot replace them.

For reference, William's r3 placements were Qwen on one GPU, Muse on one GPU,
and Gemma on two GPUs. The public example selection has a common placement
preset; it does not automatically create that heterogeneous layout. Freeze and
qualify each effective profile/worker pairing before using such a layout.

### Freeze and bootstrap a new deployment (CPU only)

Use the policy and measured record above, or the matching externally qualified
V13 inputs. Keep them outside the scientific source package. The following
script checks live admission before staging V2:

```bash
export LIVE_JOB="$WORK/live-v2"
export LIVE_LOCAL="$LOCAL_BASE/$RUN_ID/live-v2"
# QUALIFICATION, POLICY, LOCAL_LEASES and LEGACY_LEASES were exported above.
# Preserve the site's allocation. For direct-site access, use only UUIDs
# whose use you have arranged, e.g. CUDA_VISIBLE_DEVICES=GPU-...,GPU-...
test -n "${CUDA_VISIBLE_DEVICES:-}"

python - <<'PY'
import os
import shutil
from pathlib import Path
from high_fidelity_schema_study.four_category.common import read_json
from high_fidelity_schema_study.four_category.scheduler import compile_condition, live_gate
from scheduler_v2.io import digest, file_sha, write_once

project = Path(os.environ['PROJECT']).resolve()
job = Path(os.environ['LIVE_JOB']).resolve()
local = Path(os.environ['LIVE_LOCAL']).resolve()
sources = Path(os.environ['SOURCES']).resolve()
image = Path(os.environ['SIF']).resolve()
qualification = Path(os.environ['QUALIFICATION']).resolve()
if job.exists():
    raise SystemExit('New deployment directory required.')
config = read_json(sources / 'experiment.frozen.json')
policy = read_json(Path(os.environ['POLICY']))
if policy['qualification']['path'] != str(qualification):
    raise SystemExit('Policy must bind this qualification path and its file hash.')
condition = compile_condition(config, read_json(sources / 'corpus.json'), policy, live=True)
live_gate(condition, image, sources)
write_once(job / 'frozen-condition.json', condition)
deployment = {
    'deployment_id': os.environ['RUN_ID'] + '-live-v2',
    'root': str(job / 'run'), 'local_root': str(local), 'sources': str(sources),
    'science_source': str(project), 'runtime_source': str(project),
    'condition_file_sha256': file_sha(job / 'frozen-condition.json'),
    'operational_files': {p.relative_to(project).as_posix(): file_sha(p)
                          for p in sorted((project / 'scheduler_v2').glob('*.py'))},
    'execution_identity': {'source_tree_sha256': digest(condition['source_file_bytes_sha256']),
                           'image_file_bytes_sha256': file_sha(image)},
    'authority_lock': str(local / 'dispatch-authority.lock'),
    'qualification': str(qualification), 'image': str(image),
    'apptainer': shutil.which('apptainer'),
    'local_leases': os.environ['LOCAL_LEASES'],
    'legacy_leases': os.environ['LEGACY_LEASES']}
if not deployment['apptainer']:
    raise SystemExit('Apptainer unavailable.')
write_once(job / 'deployment.json', deployment)
PY
python -m scheduler_v2.bootstrap --config "$LIVE_JOB/deployment.json" \
  --condition "$LIVE_JOB/frozen-condition.json"
python -m scheduler_v2.cli --config "$LIVE_JOB/deployment.json" verify
```

`science_source` is the **parent** of the `high_fidelity_schema_study` package;
`runtime_source` is the parent of `scheduler_v2`. In this clone both are
PROJECT. On William's deployments they are separate frozen directories.
Bootstrap source-replays dataset jobs, freezes group plans, leaves model jobs
pending and marks soft reference deferred. It performs no paper-model calls.

If cooperating with an existing deployment, use its agreed **same** GPU lease
namespaces, rather than the fresh standalone paths in this example. Locks do
not grant site allocation, and independent accounts' lease files do not fence
each other. Never bootstrap inside SOURCES or place the local ledger on NFS.
Use `migrate`/`handoff` only for an explicitly planned continuation; they are
not required for a fresh run.

## 9. Start, monitor, pause, drain and resume

Use this only for your prepared new live deployment. `start` permits real
model inference. It is not part of installing the environment.

```bash
export DEPLOYMENT="$LIVE_JOB/deployment.json"
sched() { "$CONTROL_PYTHON" -m scheduler_v2.cli --config "$DEPLOYMENT" "$@"; }
sched verify
# Clear extra inherited bind specifications before launching V2 workers.
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
sched start
sched status
sched reconcile
```

The supervisor launches the coordinator and watchdog separately. Resident
workers own their models/RNG; two CPU validators replay results independently.
The supervisor can restart the coordinator within bounded limits and invokes
the separate CPU finalizer when the snapshot is terminal. `status` reports
stages, group coverage, worker waiting reasons and maintenance state.

Read the actual roots from the deployment rather than assuming a directory:

```bash
export RESULT_ROOT="$(python -c 'import json,os; print(json.load(open(os.environ["DEPLOYMENT"]))["root"])')"
export CONTROL_ROOT="$(python -c 'import json,os; print(json.load(open(os.environ["DEPLOYMENT"]))["local_root"])')"
watch -n 10 "$CONTROL_PYTHON -m scheduler_v2.cli --config $DEPLOYMENT status"
# Ctrl-C stops watch only.
tail -n 60 "$CONTROL_ROOT/coordinator.log"
find "$CONTROL_ROOT/workers" -name process.log -print
python -m scheduler_v2.report --root "$RESULT_ROOT"
```

Use a stable command ID when retrying the same control action. Submission is
not acknowledgment: check `control_receipts/`, worker heartbeats and release
records before treating it as applied.

```bash
sched pause --command-id maintenance-01-pause
sched status
# Or target a worker ID copied from status/condition.json:
sched pause --worker ACTUAL_WORKER_ID --command-id worker-maintenance-01
sched resume --worker ACTUAL_WORKER_ID --command-id worker-resume-01
sched resume --command-id maintenance-01-resume

# Cooperative release at a safe group boundary, for planned maintenance:
sched drain --command-id shutdown-01-drain
sched status
sched reconcile
# Resume the same owned deployment only after the maintenance is complete:
sched resume --command-id shutdown-01-resume
```

Global and worker-specific maintenance are independent; clear each scope with
its corresponding resume. A pause/drain deadline alerts after 15 minutes by
default; expiry never authorizes a forced takeover. Do not run a second worker
launcher or delete locks because a model appears idle.

To temporarily reserve **one configured GPU** for an owned probe:

```bash
sched reserve --gpu ACTUAL_GPU_UUID --duration 3600 --grace 900 \
  --reservation-key probe-01
sched status
# After confirming the owned probe released its resources:
sched cancel-reservation --reservation-key probe-01 --command-id probe-01-cancel
```

The limit is three hours with at most 900 seconds grace. Repeating a reservation
key must retain the same GPU/duration. Reservation submission alone does not
permit GPU use. The probe runner waits for release, acquires both namespaces,
enforces the child's deadline and confirms release before canceling:

```bash
python -m scheduler_v2.probe --config "$DEPLOYMENT" --reservation-key probe-01 \
  --manifest /ABSOLUTE/PATH/TO/probe-manifest.json \
  --manifest-sha256 ACTUAL_MANIFEST_FILE_SHA256
```

The manifest supplies `argv` and a `files` mapping of absolute executable,
script and image paths to file-byte SHA-256, including the executable. Preserve
the runner's child-identity receipts. Never signal an unrecognized process.

### Optional Telegram notifications

```bash
# Private JSON: enabled=true, bot_token, bot_username, integer chat_id.
# Keep it outside Git, inputs, shared/local run roots and scientific source.
chmod 600 /ABSOLUTE/PRIVATE/telegram.json
sched start --telegram-secret /ABSOLUTE/PRIVATE/telegram.json
```

Pass the secret when launching the deployment; an already running supervisor
retains its launch settings. No secret values belong in deployment JSON, shell
history, images or reports. Telegram delivery is separate from a signed Codex
observer/receiver bridge; `start` does not provision such a bridge.

## 10. Finalization, verification and collection

```bash
sched status
python -m json.tool "$RESULT_ROOT/finalization-status.json"
python -m json.tool "$(dirname "$RESULT_ROOT")/packet-release.json"
python -m scheduler_v2.report --root "$RESULT_ROOT"
```

Wait for a terminal snapshot and `finalization-status.json` with `status: pass`.
`completed_with_rejections` is an accounted terminal outcome, not a successful
complete schema extraction. Preserve rejected responses. Packet integrity,
source derivation, coverage and semantic correctness remain separate.

For an independent CPU replay of a V2 packet, use its matching frozen source
on PYTHONPATH:

```bash
export PACKET="$(dirname "$RESULT_ROOT")/packets/ACTUAL_PAPER_ID"
python - <<'PY'
import json, os
from pathlib import Path
from scheduler_v2.finalizer import verify_packet
config = json.loads(Path(os.environ['DEPLOYMENT']).read_text())
condition = json.loads((Path(config['root']) / 'condition.json').read_text())
report = verify_packet(Path(os.environ['PACKET']), Path(config['sources']),
                       expected_condition=condition['condition'])
print(json.dumps(report, indent=2))
PY
```

The old workflow `cli verify` is for its own serial packet format; use the V2
finalizer verifier for V2 scheduled packets. Copy only closed results. Example
collection on **Mercury**, followed by download on **your workstation**:

```bash
# Mercury, after finalization passed:
export RELEASE_ROOT="$(dirname "$RESULT_ROOT")"
tar -czf "$WORK/schema-results.tar.gz" -C "$RELEASE_ROOT" packets packet-release.json
sha256sum "$WORK/schema-results.tar.gz" > "$WORK/schema-results.tar.gz.sha256"
```

```bash
# Workstation:
scp -o ProxyJump=YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu \
  YOUR_UO_ACCOUNT@mercury.nic.uoregon.edu:/ABSOLUTE/WORK/schema-results.tar.gz .
scp -o ProxyJump=YOUR_UO_ACCOUNT@orthus.nic.uoregon.edu \
  YOUR_UO_ACCOUNT@mercury.nic.uoregon.edu:/ABSOLUTE/WORK/schema-results.tar.gz.sha256 .
# The generated checksum names the host's absolute archive path. Verify locally:
EXPECTED_SHA="$(awk '{print $1}' schema-results.tar.gz.sha256)"
printf '%s  schema-results.tar.gz\n' "$EXPECTED_SHA" | sha256sum -c -
```

For full provenance, also preserve deployment/condition files, input and source
manifests, runtime identities, raw answers, judgments, immutable journals and
qualification records. Do not copy the host-local verification key or secrets
into a public result archive.

## 11. Use William's existing Mercury installation

These paths belong to William's account. A colleague using a different account
needs arranged read access or separately copied artifacts; do not assume your
HOME resolves to William's storage. This installation need not be rebuilt.

**After logging into Mercury as William or an authorized environment:**

```bash
export WILLIAM_BASE=/storage/users/williamq/schema-study-deployment-20260926
source "$WILLIAM_BASE/deployment-env.sh"
apptainer --version
MERCURY_CPU_ONLY=1 bash "$WILLIAM_BASE/workflow.sh" --help
MERCURY_CPU_ONLY=1 bash "$WILLIAM_BASE/workflow.sh" offline-demo \
  --output "/outputs/colleague-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
```

`workflow.sh` points at the earlier `release-0.1.0` source/image, not the r3
scientific runtime. For the **completed P06 V13 r3 continuation**, select its
actual deployment and frozen source from its configuration:

```bash
export P06_JOB="$WILLIAM_BASE/jobs/v13-p06-direct-20260928-r3-v1"
export DEPLOYMENT="$P06_JOB/continuation-replay-mailbox-v1/deployment-config.json"
export CONTROL_PYTHON="$WILLIAM_BASE/jobs/matrix-scheduler-20260926-v7/control-venv/bin/python"
export RUNTIME_SOURCE="$("$CONTROL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["DEPLOYMENT"]))["runtime_source"])')"
export SCIENCE_SOURCE="$("$CONTROL_PYTHON" -c 'import json,os; print(json.load(open(os.environ["DEPLOYMENT"]))["science_source"])')"
export PYTHONPATH="$RUNTIME_SOURCE:$SCIENCE_SOURCE"
sched() { "$CONTROL_PYTHON" -m scheduler_v2.cli --config "$DEPLOYMENT" "$@"; }
sched verify
sched status
sched reconcile
"$CONTROL_PYTHON" -m json.tool \
  "$P06_JOB/continuation-replay-mailbox-v1/run/finalization-status.json"
ls -lh "$P06_JOB/exports"
"$CONTROL_PYTHON" -m json.tool "$P06_JOB/exports/schema-summary.json"
```

Read-only configuration inspection for this guide confirmed those paths, the
separate frozen scientific/runtime sources, and the retained structured SIF:

```text
jobs/full-inference-20260927-v1/image-build-v2/schema-study-structured-20260927.sif
```

P06's three model cells already ended as `completed_with_rejections`; all 528
delivered members passed SHA-256 verification. Attribute facts retained in
those exports are zero; schema fidelity is not established. **Do not run
`start`, `resume`, old observers or direct-run launchers against this completed
job.** A new experiment needs a separately frozen condition and dispatch root.
The original direct-run controller was superseded by the nested continuation.

The older September 28 full-queue V2 job used
`jobs/scheduler-v2-20260928-v1/deployment-config-v5.json` and `runtime-v5`.
Those are historical operational identities, not an instruction to restart
the paused ten-paper queue. The `v13r3` repair source is not included in the
pinned public r2 commit; exact r3 reproduction requires its retained source
manifest, complete input bundle and configuration.

## 12. Troubleshooting

| Symptom | Check / action |
| --- | --- |
| `apptainer: command not found` | Select the site module or source William's opt-in deployment environment; otherwise install to your own directory |
| `python3.12` unavailable | Select the site's Python 3.12 environment; the checked Mercury python3 resolved to 3.12.13 |
| `No module named scheduler_v2` | Use the public V13 source commit and its clone root on PYTHONPATH; v0.1.0 lacks the scheduler |
| `No module named high_fidelity_schema_study` | PYTHONPATH/science_source must contain the package's parent, not just the package directory |
| `/opt/phase1-venv/bin/python` missing | Use the Dockerfile-based/retained V2 image; the native SIF recipe has a different venv path |
| Missing model architecture/runtime library | Build and qualify a new dependency image; preserve its resolved dependency inventory |
| Conflicting CUDA masks | Keep the site's allocation; remove a conflicting manual MERCURY_CUDA_DEVICES override |
| `waiting_for_snapshot` | Start may still be initializing; inspect supervisor/coordinator logs and recorded process identity |
| `dependency_wait` / `awaiting_validation` | Check classifier/index availability or the CPU replay backlog; GPU idleness alone is not a crash |
| `condition_identity_changed` or source/hash mismatch | Restore the matching frozen source/config/image; never edit hashes to force a pass |
| `v2_requires_frozen_all_groups_jobs` | The experiment must freeze grouped_mode `all-groups/v1` and a compatible grouped protocol |
| `three_distinct_replicates_required` | The generic compiler expects three seeds; do not reinterpret the separately frozen single-repeat P06 condition |
| Qualification absent / headroom insufficient | Obtain matching measured qualification and free assigned resources; do not replace measurement with doctor or increase budgets to bypass failure |
| `persistent_host_local_filesystem_required` | Move the new ledger/scratch to local storage such as /var/tmp; shared SQLite is not supported |
| Lease already owned / foreign GPU process | Reconcile current identities and contact the resource owner; do not delete locks or signal another user's process |
| Output already exists / immutable conflict | Choose a new artifact version; resume the same run only with identical frozen inputs |
| NFS unavailable / journal mismatch | New dispatch fails closed; preserve evidence and restore storage/verified journal before recovery |
| Coordinator restart did not help | Run reconcile; supervisor recovery is bounded and does not authorize duplicate coordinators |
| Terminal snapshot but no packet release | Inspect finalization-status.json and finalizer.log; terminal generation is not completed delivery |

If an OCI pull fails because the system registry file has duplicate keys,
preserve that file. When site policy permits a per-user override and you do not
already have one, create a new configuration:

```bash
mkdir -p "$HOME/.config/containers"
test ! -e "$HOME/.config/containers/registries.conf"
(set -o noclobber; cat > "$HOME/.config/containers/registries.conf" <<'EOF'
unqualified-search-registries = ["docker.io"]
short-name-mode = "enforcing"
EOF
)
```

If an existing user file is present, review it instead of overwriting it. Image
commands in this guide already name the full registry and retain TLS checks.
For installation or user namespace failures, follow the
[Apptainer installation requirements](https://apptainer.org/docs/admin/1.5/installation.html#system-requirements)
and the site's administrator instructions.

### Validation scope of this runbook

The source/CLI signatures were inspected at the pinned public commit. The
William-specific environment and deployment paths were checked read-only on
Mercury. In an isolated Mercury validation directory, **18 scheduler tests
passed**, the host offline demo passed, and the documented V2 mock run reached
passing finalization and independent packet byte/derivation replay. The native
input/corpus builder, grouped base and deferred-reference configuration examples
also ran successfully. Fenced Bash blocks passed syntax checks and embedded
Python programs compiled. The validation's owned background processes were
then stopped; historical queues were not restarted. See the
[validation record](mercury_runbook_validation_2026-09-29.json).

Live model runs, image builds, weight downloads and new GPU qualifications were
not executed to validate documentation. Those steps remain conditional on the
operator's actual inputs, allocation, profiles, runtime and measured gates.

The public source status and older guides are dated snapshots. Consult the
actual frozen job's manifests and terminal reports when assessing a historical
result, and preserve all original answers and judgments when preparing a new
version.
