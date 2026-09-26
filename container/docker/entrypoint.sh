#!/usr/bin/env bash
set -Eeuo pipefail

module="${MERCURY_PYTHON_MODULE:-high_fidelity_schema_study.four_category.cli}"
case "$module" in
    high_fidelity_schema_study.four_category.cli|high_fidelity_schema_study.four_category.mercury) ;;
    *) printf 'Unsupported MERCURY_PYTHON_MODULE: %s\n' "$module" >&2; exit 2 ;;
esac

# The general CLI is appropriate for preparation and inspection commands.
# Live inference must pass through the Mercury image/code/hardware gate.
if [[ "${1:-}" == run && "$module" == high_fidelity_schema_study.four_category.cli ]]; then
    module=high_fidelity_schema_study.four_category.mercury
fi

exec python -m "$module" "$@"
