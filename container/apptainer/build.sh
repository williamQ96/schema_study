#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf 'Usage: %s OUTPUT.sif\n' "${0##*/}" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"
recipe="$script_dir/four-category.def"
inference_requirements="$repo_root/requirements-phase1-inference.txt"
offline_requirements="$repo_root/requirements_four_category_offline.txt"
cuda_image="${CUDA_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"

mkdir -p -- "$(dirname -- "$1")"
output="$(realpath -m -- "$1")"
manifest="$output.manifest.json"
if [[ -e "$output" || -L "$output" ]]; then
    printf 'Refusing to overwrite existing SIF: %s\n' "$output" >&2
    exit 2
fi
if [[ -e "$manifest" || -L "$manifest" ]]; then
    printf 'Refusing to overwrite existing build manifest: %s\n' "$manifest" >&2
    exit 2
fi
if [[ ! -f "$inference_requirements" || ! -f "$offline_requirements" || ! -f "$recipe" ]]; then
    printf 'Build inputs are missing under repository root: %s\n' "$repo_root" >&2
    exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
    printf 'Apptainer is not installed or is not available on PATH.\n' >&2
    exit 127
fi

apptainer_version="$(apptainer --version 2>&1 || true)"
if [[ -z "$apptainer_version" ]]; then
    apptainer_version="unknown (apptainer --version failed)"
fi

printf 'Building runtime-only Mercury image from %s\n' "$cuda_image"
printf 'Image output: %s\n' "$output"
if ! (cd -- "$repo_root" && apptainer build --build-arg "CUDA_IMAGE=$cuda_image" "$output" "$recipe"); then
    printf 'Apptainer build failed. No fallback privilege mode was attempted; inspect Apptainer/fakeroot permissions and the build log.\n' >&2
    exit 1
fi
if [[ ! -f "$output" ]]; then
    printf 'Apptainer reported success but did not create the requested SIF: %s\n' "$output" >&2
    exit 1
fi

python3 - "$output" "$manifest" "$cuda_image" "$apptainer_version" "$recipe" "$inference_requirements" "$offline_requirements" <<'PY'
import hashlib
import json
import os
import sys

image, manifest, base, apptainer_version, recipe, inference, offline = sys.argv[1:]

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema_version": "mercury-apptainer-build/v1",
    "image": os.path.basename(image),
    "image_bytes": os.path.getsize(image),
    "image_sha256": digest(image),
    "base_image_uri": base,
    "apptainer_version": apptainer_version,
    "recipe_sha256": digest(recipe),
    "requirements_sha256": {
        os.path.basename(inference): digest(inference),
        os.path.basename(offline): digest(offline),
    },
    "runtime_scope": "RHEL 9.8 NVIDIA runtime-only Apptainer lane; no weights or credentials embedded",
}
with open(manifest, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    stream.write("\n")
print(f"Build manifest: {manifest}")
PY
