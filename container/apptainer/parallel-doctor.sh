#!/usr/bin/env bash
# Local-reviewable multi-container hardware probe. No model loading or inference.
# Default prints commands; --execute must be supplied later on the allocated host.
set -Eeuo pipefail
umask 077
mode=--dry-run
if [[ ${1:-} == --execute || ${1:-} == --dry-run ]]; then mode="$1"; shift; fi
[[ $# == 7 ]] || { echo 'Usage: parallel-doctor.sh [--dry-run|--execute] SIF REPO INPUTS PROBE_ROOT MODELS CACHE_ROOT GPUS_PER_WORKER' >&2; exit 2; }
image="$1" repo="$2" inputs="$3" probe="$4" models="$5" cache="$6" width="$7"
[[ "$width" =~ ^(1|2|4)$ ]] || { echo 'GPUS_PER_WORKER must be 1, 2, or 4' >&2; exit 2; }
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || { echo 'Explicit allocated CUDA_VISIBLE_DEVICES is required' >&2; exit 2; }
[[ -z ${MERCURY_CUDA_DEVICES:-} || "$MERCURY_CUDA_DEVICES" == "$CUDA_VISIBLE_DEVICES" ]] || { echo 'Conflicting allocation masks' >&2; exit 2; }
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 4 ]] || { echo 'This Mercury probe requires four explicitly allocated devices' >&2; exit 2; }
declare -A seen=()
for gpu in "${devices[@]}"; do
    [[ "$gpu" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ && ! ${seen[$gpu]+exists} ]] || { echo 'Invalid/duplicate device allocation' >&2; exit 2; }
    seen[$gpu]=1
done
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run.sh"
pids=()
if [[ "$mode" == --execute ]]; then
    [[ -f "$image" && -d "$probe" && -d "$cache" ]] || { echo 'Image and separate probe/cache roots must exist' >&2; exit 2; }
    [[ "$(realpath "$probe")" != "$(realpath "$cache")" ]] || { echo 'Probe and cache roots must differ' >&2; exit 2; }
    # Pre-create every exclusive directory before launching any process.
    for ((i=0; i<4; i+=width)); do
        [[ ! -e "$probe/worker-$i" && ! -e "$cache/worker-$i" ]] || { echo 'Use fresh probe/cache roots' >&2; exit 2; }
    done
    for ((i=0; i<4; i+=width)); do mkdir "$probe/worker-$i" "$cache/worker-$i"; done
fi
for ((i=0; i<4; i+=width)); do
    mask="$(IFS=,; printf '%s' "${devices[*]:i:width}")"
    command=(env "CUDA_VISIBLE_DEVICES=$mask" "MERCURY_CUDA_DEVICES=$mask"
        MERCURY_CPU_ONLY=0 MERCURY_DRY_RUN=0 MERCURY_SECRET_ENV_NAMES=
        MERCURY_PYTHON_MODULE=high_fidelity_schema_study.four_category.mercury
        OMP_NUM_THREADS=8 bash "$launcher" "$image" "$repo" "$inputs"
        "$probe/worker-$i" "$models" "$cache/worker-$i" -- doctor
        --gpu-count "$width" --output /outputs/hardware.json)
    if [[ "$mode" == --dry-run ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
    else
        "${command[@]}" > "$probe/worker-$i/doctor.log" 2>&1 &
        pids+=("$!")
    fi
done
result=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else result=1; fi
done
exit "$result"
