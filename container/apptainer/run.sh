#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf 'Usage: %s IMAGE.sif REPO SOURCE_ROOT OUTPUT_ROOT MODEL_ROOT CACHE_ROOT [-- CLI_ARGS...]\n' "${0##*/}" >&2
}

if [[ $# -lt 6 ]]; then
    usage
    exit 2
fi

image="$1"
repo="$2"
source_root="$3"
output_root="$4"
model_root="$5"
cache_root="$6"
shift 6
if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        usage
        exit 2
    fi
    shift
fi

die() { printf '%s\n' "$*" >&2; exit 2; }

# Ignore host-injected container environment overrides; this wrapper explicitly
# forwards only its controlled runtime settings and named secrets below.
while IFS='=' read -r inherited_name _; do
    case "$inherited_name" in APPTAINERENV_*|SINGULARITYENV_*) unset "$inherited_name" ;; esac
done < <(env)
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH || true

image="$(realpath -e -- "$image")" || die "Image does not exist: $image"
[[ -f "$image" ]] || die "Image must be a SIF file: $image"
image_sha256="$(sha256sum -- "$image" | awk '{print $1}')" || die 'Cannot compute SIF SHA-256'
[[ "$image_sha256" =~ ^[a-f0-9]{64}$ ]] || die 'Could not compute a valid SIF SHA-256'
export APPTAINERENV_MERCURY_IMAGE_SHA256="$image_sha256"
if command -v apptainer >/dev/null 2>&1; then
    apptainer_version="$(apptainer --version 2>&1 || true)"
    [[ -n "$apptainer_version" ]] && export APPTAINERENV_MERCURY_APPTAINER_VERSION="$apptainer_version"
fi

path_has_bind_separator() {
    [[ "$1" == *:* || "$1" == *,* || "$1" == *$'\n'* || "$1" == *$'\r'* ]]
}

canonical_dir() {
    local label="$1" raw="$2" result
    [[ -d "$raw" ]] || die "$label must be an existing directory: $raw"
    result="$(realpath -e -- "$raw")" || die "Cannot resolve $label: $raw"
    path_has_bind_separator "$raw" && die "$label contains a colon, comma, or newline that Apptainer bind syntax cannot represent safely"
    path_has_bind_separator "$result" && die "Resolved $label contains a colon, comma, or newline that Apptainer bind syntax cannot represent safely"
    printf '%s' "$result"
}

repo="$(canonical_dir REPO "$repo")"
source_root="$(canonical_dir SOURCE_ROOT "$source_root")"
output_root="$(canonical_dir OUTPUT_ROOT "$output_root")"
model_root="$(canonical_dir MODEL_ROOT "$model_root")"
cache_root="$(canonical_dir CACHE_ROOT "$cache_root")"
for writable_path in "$output_root" "$cache_root"; do
    for read_only_path in "$repo" "$source_root" "$model_root"; do
        [[ "$writable_path" != "$read_only_path" ]] || die 'A writable bind directory cannot equal a read-only bind directory'
    done
done
[[ "$output_root" != "$cache_root" ]] || die 'OUTPUT_ROOT and CACHE_ROOT must be distinct directories'

cpu_only="${MERCURY_CPU_ONLY:-0}"
[[ "$cpu_only" == 0 || "$cpu_only" == 1 ]] || die 'MERCURY_CPU_ONLY must be 0 or 1'
gpu_devices=""
if [[ "$cpu_only" == 1 ]]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES=""
else
    if [[ ${MERCURY_CUDA_DEVICES+x} ]]; then
        gpu_devices="$MERCURY_CUDA_DEVICES"
        if [[ ${CUDA_VISIBLE_DEVICES+x} && "$CUDA_VISIBLE_DEVICES" != "$gpu_devices" ]]; then
            die 'MERCURY_CUDA_DEVICES conflicts with scheduler-provided CUDA_VISIBLE_DEVICES; scheduler selection is not overridden'
        fi
    else
        [[ ${CUDA_VISIBLE_DEVICES+x} ]] || die 'GPU mode requires CUDA_VISIBLE_DEVICES or explicit MERCURY_CUDA_DEVICES'
        gpu_devices="$CUDA_VISIBLE_DEVICES"
    fi
    [[ -n "$gpu_devices" ]] || die 'GPU mode received an empty GPU selection; use MERCURY_CPU_ONLY=1 for CPU execution'
    gpu_atom='([0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)'
    [[ "$gpu_devices" =~ ^${gpu_atom}(,${gpu_atom})*$ ]] || die 'GPU selection must be comma-separated ordinals, GPU UUIDs, or MIG identifiers'
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu_devices"
fi

omp_threads="${OMP_NUM_THREADS:-8}"
[[ "$omp_threads" =~ ^[1-9][0-9]*$ ]] || die 'OMP_NUM_THREADS must be a positive integer'
export APPTAINERENV_OMP_NUM_THREADS="$omp_threads"
export APPTAINERENV_PYTHONNOUSERSITE=1

python_module="${MERCURY_PYTHON_MODULE:-high_fidelity_schema_study.four_category.cli}"
case "$python_module" in
    high_fidelity_schema_study.four_category.cli|high_fidelity_schema_study.four_category.mercury) ;;
    *) die 'MERCURY_PYTHON_MODULE must be one of the approved four-category CLI modules' ;;
esac
if [[ "${1:-}" == "run" ]]; then
    case "$python_module" in
        high_fidelity_schema_study.four_category.cli) python_module=high_fidelity_schema_study.four_category.mercury ;;
    esac
fi
export APPTAINERENV_MERCURY_PYTHON_MODULE="$python_module"

dry_run="${MERCURY_DRY_RUN:-0}"
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || die 'MERCURY_DRY_RUN must be 0 or 1'

secret_names="${MERCURY_SECRET_ENV_NAMES:-}"
if [[ -n "$secret_names" ]]; then
    IFS=',' read -r -a secret_name_array <<< "$secret_names"
    for name in "${secret_name_array[@]}"; do
        name="${name//[[:space:]]/}"
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die 'MERCURY_SECRET_ENV_NAMES must be a comma-separated list of valid environment variable names'
        [[ ${!name+x} ]] || die "Allowlisted secret environment variable is not set: $name"
        case "$name" in
            PATH|PYTHONPATH|PYTHONHOME|PYTHONUSERBASE|PYTHONNOUSERSITE|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|OMP_NUM_THREADS|HF_*|TRANSFORMERS_*|MERCURY_*|APPTAINERENV_*|SINGULARITYENV_*|LD_PRELOAD|LD_LIBRARY_PATH|APPTAINER_BIND|APPTAINER_BINDPATH|SINGULARITY_BIND|SINGULARITY_BINDPATH)
                die "Reserved runtime variable cannot be forwarded as a secret: $name" ;;
        esac
        export "APPTAINERENV_${name}=${!name}"
    done
fi

launcher=(apptainer run --cleanenv --containall)
if [[ "$cpu_only" != 1 ]]; then
    launcher+=(--nv)
fi
launcher+=(
    --bind "$repo:/workspace/high_fidelity_schema_study:ro"
    --bind "$source_root:/inputs:ro"
    --bind "$output_root:/outputs:rw"
    --bind "$model_root:/models:ro"
    --bind "$cache_root:/cache:rw"
)
if [[ -f /etc/os-release ]]; then
    launcher+=(--bind "/etc/os-release:/run/mercury/host-os-release:ro")
fi
launcher+=(
    --pwd /workspace/high_fidelity_schema_study
    "$image"
    "$@"
)

if [[ "$dry_run" == 1 ]]; then
    printf 'MERCURY_APPTAINER_MODULE=%s\n' "$python_module"
    printf 'MERCURY_APPTAINER_ARGV='
    printf '%q ' "${launcher[@]}"
    printf '\n'
    exit 0
fi
exec "${launcher[@]}"
