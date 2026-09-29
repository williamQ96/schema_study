#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf 'Usage: %s IMAGE.sif REPO PDF_ROOT OUTPUT_ROOT MODEL_ROOT CACHE_ROOT -- inventory --model-root /models --output /outputs/manifest.json\n' "${0##*/}" >&2
    printf '   or: %s IMAGE.sif REPO PDF_ROOT OUTPUT_ROOT MODEL_ROOT CACHE_ROOT -- parse --pdf /inputs/paper.pdf --output /outputs/new-attempt --model-root /models --model-manifest /outputs/manifest.json [--tier basic] [--ocr-mode txt|ocr|auto]\n' "${0##*/}" >&2
}
die() { printf '%s\n' "$*" >&2; exit 2; }

[[ $# -ge 8 ]] || { usage; exit 2; }
image="$1" repo="$2" pdf_root="$3" output_root="$4" model_root="$5" cache_root="$6"
shift 6
[[ "$1" == -- ]] || { usage; exit 2; }
shift
command_name="$1"
shift

# Host container-injection variables cannot add mounts or override offline mode.
while IFS='=' read -r inherited_name _; do
    case "$inherited_name" in APPTAINERENV_*|SINGULARITYENV_*) unset "$inherited_name" ;; esac
done < <(env)
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH || true

image="$(realpath -e -- "$image")" || die 'SIF image missing'
[[ -f "$image" ]] || die 'SIF image must be a file'
path_ok() { [[ "$1" != *:* && "$1" != *,* && "$1" != *$'\n'* && "$1" != *$'\r'* ]]; }
directory() {
    local label="$1" raw="$2" result
    [[ -d "$raw" ]] || die "$label must be an existing directory"
    result="$(realpath -e -- "$raw")" || die "$label cannot be resolved"
    path_ok "$raw" && path_ok "$result" || die "$label contains an unsafe bind separator"
    printf '%s' "$result"
}
repo="$(directory REPO "$repo")"
pdf_root="$(directory PDF_ROOT "$pdf_root")"
output_root="$(directory OUTPUT_ROOT "$output_root")"
model_root="$(directory MODEL_ROOT "$model_root")"
cache_root="$(directory CACHE_ROOT "$cache_root")"
for writable in "$output_root" "$cache_root"; do
    for readonly in "$repo" "$pdf_root" "$model_root"; do
        [[ "$writable" != "$readonly" && "$writable" != "$readonly/"* && "$readonly" != "$writable/"* ]] || die 'Writable and read-only roots overlap'
    done
done
[[ "$output_root" != "$cache_root" && "$output_root" != "$cache_root/"* && "$cache_root" != "$output_root/"* ]] || die 'OUTPUT_ROOT and CACHE_ROOT overlap'

safe_container_path() {
    local path="$1" prefix="$2" resolved
    [[ "$path" == "$prefix/"* && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || die "Path must be under $prefix"
    resolved="$(realpath -m -- "$path")"
    [[ "$resolved" == "$prefix/"* ]] || die "Path escapes $prefix"
}

args=("$command_name")
case "$command_name" in
    inventory)
        [[ $# -eq 4 && "$1" == --model-root && "$2" == /models && "$3" == --output ]] || { usage; exit 2; }
        safe_container_path "$4" /outputs
        args+=(--model-root /models --output "$4")
        ;;
    parse)
        [[ $# -ge 8 && "$1" == --pdf && "$3" == --output && "$5" == --model-root && "$6" == /models && "$7" == --model-manifest ]] || { usage; exit 2; }
        safe_container_path "$2" /inputs
        safe_container_path "$4" /outputs
        safe_container_path "$8" /outputs
        args+=(--pdf "$2" --output "$4" --model-root /models --model-manifest "$8")
        shift 8
        if [[ $# -ge 2 && "$1" == --tier ]]; then
            [[ "$2" == basic ]] || die 'Only Basic tier is qualified by this CPU runner'
            args+=(--tier basic)
            shift 2
        fi
        if [[ $# -ge 2 && "$1" == --ocr-mode ]]; then
            [[ "$2" == txt || "$2" == ocr || "$2" == auto ]] || die 'Invalid OCR mode'
            args+=(--ocr-mode "$2")
            shift 2
        fi
        [[ $# -eq 0 ]] || { usage; exit 2; }
        ;;
    *) usage; exit 2 ;;
esac

export APPTAINERENV_CUDA_VISIBLE_DEVICES=""
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
launcher=(apptainer run --cleanenv --containall
    --bind "$repo:/workspace/high_fidelity_schema_study:ro"
    --bind "$pdf_root:/inputs:ro"
    --bind "$output_root:/outputs:rw"
    --bind "$model_root:/models:ro"
    --bind "$cache_root:/cache:rw"
    --pwd /workspace/high_fidelity_schema_study
    "$image" "${args[@]}")

dry_run="${MINERU_DRY_RUN:-0}"
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || die 'MINERU_DRY_RUN must be 0 or 1'
if [[ "$dry_run" == 1 ]]; then
    printf 'MINERU_SIF_FILE_BYTES_SHA256=%s\n' "$(sha256sum -- "$image" | awk '{print $1}')"
    printf 'MINERU_APPTAINER_ARGV='
    printf '%q ' "${launcher[@]}"
    printf '\n'
    exit 0
fi
exec "${launcher[@]}"
