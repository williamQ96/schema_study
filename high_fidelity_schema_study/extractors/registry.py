from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from ..models import DatasetSchema
from .base import (
    ExtractionIssue,
    ExtractionOutcome,
    ExtractionRequest,
    ExtractorCapability,
    FormatDecision,
    FormatSignal,
    StructuredExtractionError,
)
from .csv_extractor import extract_csv_schema
from .hdf5_extractor import extract_hdf5_schema
from .json_extractor import extract_json_schema
from .netcdf_extractor import extract_netcdf_schema, hdf5_has_netcdf_markers
from .parquet_extractor import extract_parquet_schema
from .xml_extractor import extract_xml_schema
from .zarr_extractor import extract_zarr_schema
from ..unified_schema import build_unified_schema_envelope


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
PARQUET_MAGIC = b"PAR1"
NETCDF_CLASSIC_MAGICS = (b"CDF\x01", b"CDF\x02", b"CDF\x05")
ZIP_MAGIC = b"PK\x03\x04"

FORMAT_ALIASES = {
    "auto": None,
    "binary": "raw_binary",
    "raw": "raw_binary",
    "raw_binary": "raw_binary",
    "timeseries": "csv",
    "time_series": "csv",
    "h5": "hdf5",
}


def _extract_csv(request: ExtractionRequest) -> DatasetSchema:
    return extract_csv_schema(request.path, sample_limit=request.sample_limit)


def _extract_hdf5(request: ExtractionRequest) -> DatasetSchema:
    return extract_hdf5_schema(request.path)


def _extract_json(request: ExtractionRequest) -> DatasetSchema:
    return extract_json_schema(request.path, sample_limit=request.sample_limit)


def _extract_netcdf(request: ExtractionRequest) -> DatasetSchema:
    return extract_netcdf_schema(request.path, sample_limit=request.sample_limit)


def _extract_zarr(request: ExtractionRequest) -> DatasetSchema:
    return extract_zarr_schema(request.path, sample_limit=request.sample_limit)


def _extract_parquet(request: ExtractionRequest) -> DatasetSchema:
    return extract_parquet_schema(request.path)


def _extract_xml(request: ExtractionRequest) -> DatasetSchema:
    return extract_xml_schema(request.path, sample_limit=request.sample_limit)


EXTRACTOR_CAPABILITIES: Dict[str, ExtractorCapability] = {
    "csv": ExtractorCapability(
        extractor_id="csv_conservative_profiler",
        version="1.0.0",
        formats=["csv"],
        suffixes=[".csv", ".tsv"],
        magic_prefixes=[],
        operations=["enumerate_fields", "sample_values", "infer_physical_types", "profile_time_series"],
        determinism_class="strict_with_sampled_profiles",
        evidence_types=["csv_header", "sample_rows", "column_name_unit_hint"],
    ),
    "hdf5": ExtractorCapability(
        extractor_id="h5py_structure_traversal",
        version="1.0.0",
        formats=["hdf5"],
        suffixes=[".h5", ".hdf5", ".hdf", ".he5"],
        magic_prefixes=[HDF5_MAGIC.hex()],
        operations=["enumerate_groups", "enumerate_fields", "read_shapes", "read_attributes"],
        determinism_class="strict",
        evidence_types=["hdf5_dataset_path", "hdf5_attribute"],
    ),
    "json": ExtractorCapability(
        extractor_id="json_bounded_structure_extractor",
        version="0.1.0",
        formats=["json"],
        suffixes=[".json", ".jsonl", ".ndjson"],
        magic_prefixes=[],
        operations=[
            "observe_bounded_paths",
            "observe_type_sets",
            "observe_nullability",
            "observe_missingness",
            "observe_array_shapes",
            "read_json_schema_declarations",
        ],
        determinism_class="strict_with_bounded_sample",
        evidence_types=["json_sample_observation", "json_schema_declaration"],
    ),
    "netcdf": ExtractorCapability(
        extractor_id="netcdf_cf_deterministic_extractor",
        version="1.0.0",
        formats=["netcdf"],
        suffixes=[".nc", ".cdf"],
        magic_prefixes=[magic.hex() for magic in NETCDF_CLASSIC_MAGICS] + [HDF5_MAGIC.hex()],
        operations=[
            "enumerate_dimensions",
            "enumerate_groups",
            "enumerate_variables",
            "read_shapes",
            "read_attributes",
            "interpret_cf_coordinates",
            "decode_cf_time",
        ],
        determinism_class="strict_with_sampled_profiles",
        evidence_types=["netcdf_dimension", "netcdf_group", "netcdf_variable", "netcdf_attribute"],
    ),
    "parquet": ExtractorCapability(
        extractor_id="parquet_arrow_metadata_extractor",
        version="0.1.0",
        formats=["parquet"],
        suffixes=[".parquet"],
        magic_prefixes=[PARQUET_MAGIC.hex()],
        operations=[
            "enumerate_arrow_fields",
            "read_arrow_types",
            "read_nullability",
            "read_row_groups",
            "read_encodings",
            "read_compression",
            "read_footer_statistics",
            "read_key_value_metadata",
        ],
        determinism_class="strict_metadata_only",
        evidence_types=[
            "arrow_schema_field",
            "arrow_field_metadata",
            "parquet_schema_column",
            "parquet_row_group",
            "parquet_column_chunk",
            "parquet_footer_statistics",
        ],
    ),
    "zarr": ExtractorCapability(
        extractor_id="zarr_v2_metadata_extractor",
        version="0.2.0",
        formats=["zarr"],
        suffixes=[".zarr"],
        magic_prefixes=[],
        operations=[
            "enumerate_groups",
            "enumerate_arrays",
            "read_array_metadata",
            "read_attributes",
            "interpret_xarray_dimensions",
            "interpret_cf_coordinates",
        ],
        determinism_class="strict_metadata_only",
        evidence_types=[
            "zarr_group_metadata",
            "zarr_array_metadata",
            "zarr_attribute",
            "zarr_consolidated_metadata",
        ],
        resource_kinds=["directory_store"],
    ),
    "xml": ExtractorCapability(
        extractor_id="xml_xsd_bounded_structure_extractor",
        version="0.1.0",
        formats=["xml"],
        suffixes=[".xml", ".xsd"],
        magic_prefixes=[],
        operations=[
            "observe_bounded_elements",
            "observe_attributes",
            "observe_namespaces",
            "observe_repeated_paths",
            "read_xsd_declarations",
            "compare_xsi_type_observation",
        ],
        determinism_class="strict_with_bounded_sample",
        evidence_types=["xml_sample_observation", "xsd_element_declaration", "xsd_attribute_declaration"],
    ),
}

EXTRACTOR_RUNNERS: Dict[str, Callable[[ExtractionRequest], DatasetSchema]] = {
    "csv": _extract_csv,
    "hdf5": _extract_hdf5,
    "json": _extract_json,
    "netcdf": _extract_netcdf,
    "parquet": _extract_parquet,
    "xml": _extract_xml,
    "zarr": _extract_zarr,
}


def list_capabilities() -> list[ExtractorCapability]:
    return [EXTRACTOR_CAPABILITIES[key] for key in sorted(EXTRACTOR_CAPABILITIES)]


def normalize_format_hint(format_hint: Optional[str]) -> Optional[str]:
    if format_hint is None:
        return None
    normalized = format_hint.strip().lower()
    return FORMAT_ALIASES.get(normalized, normalized)


def _magic_signal(prefix: bytes) -> Optional[FormatSignal]:
    if prefix.startswith(HDF5_MAGIC):
        return FormatSignal("hdf5", "magic", 1.0, "HDF5 signature")
    if prefix.startswith(PARQUET_MAGIC):
        return FormatSignal("parquet", "magic", 1.0, "Parquet signature")
    if any(prefix.startswith(magic) for magic in NETCDF_CLASSIC_MAGICS):
        return FormatSignal("netcdf", "magic", 1.0, "NetCDF classic signature")
    if prefix.startswith(ZIP_MAGIC):
        return FormatSignal("zip_container", "magic", 0.98, "ZIP container signature")
    return None


def _suffix_signal(path: Path) -> Optional[FormatSignal]:
    suffix = path.suffix.lower()
    suffix_formats = {
        ".csv": "csv",
        ".tsv": "csv",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".hdf": "hdf5",
        ".he5": "hdf5",
        ".nc": "netcdf",
        ".cdf": "netcdf",
        ".parquet": "parquet",
        ".zarr": "zarr",
        ".fits": "fits",
        ".fit": "fits",
        ".grib": "grib",
        ".grb": "grib",
        ".json": "json",
        ".jsonl": "json",
        ".ndjson": "json",
        ".xml": "xml",
        ".xsd": "xml",
        ".avro": "avro",
    }
    file_format = suffix_formats.get(suffix)
    if file_format is None:
        return None
    return FormatSignal(file_format, "suffix", 0.72, f"filename suffix {suffix}")


def _looks_like_delimited_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(8192)
        if not sample or b"\x00" in sample:
            return False
        text = sample.decode("utf-8-sig")
        dialect = csv.Sniffer().sniff(text, delimiters=",\t;|")
        rows = list(csv.reader(text.splitlines()[:5], dialect=dialect))
        return bool(rows) and len(rows[0]) >= 2 and all(len(row) == len(rows[0]) for row in rows[1:])
    except (UnicodeDecodeError, csv.Error, OSError):
        return False


def _structured_text_signal(path: Path) -> Optional[FormatSignal]:
    try:
        if path.stat().st_size > 1024 * 1024:
            return None
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return FormatSignal(
            "json",
            "complete_json_probe",
            0.9,
            "complete UTF-8 JSON document",
        )
    return None


def _directory_signal(path: Path) -> Optional[FormatSignal]:
    if (path / "zarr.json").is_file():
        return FormatSignal("zarr", "zarr_v3_metadata", 1.0, "Zarr v3 metadata document zarr.json")
    markers = [name for name in (".zgroup", ".zarray", ".zmetadata") if (path / name).is_file()]
    if markers:
        return FormatSignal(
            "zarr",
            "zarr_store_metadata",
            1.0,
            f"Zarr store metadata documents: {', '.join(markers)}",
        )
    return None


def _actual_resource_kind(path: Path) -> str:
    return "directory_store" if path.is_dir() else "file"


def detect_format(request: ExtractionRequest) -> FormatDecision:
    path = Path(request.path)
    signals: list[FormatSignal] = []
    strong_signal: Optional[FormatSignal]
    if path.is_dir():
        strong_signal = _directory_signal(path)
    else:
        with path.open("rb") as handle:
            prefix = handle.read(16)
        strong_signal = _magic_signal(prefix)
        if (
            strong_signal is not None
            and strong_signal.format == "hdf5"
            and path.suffix.lower() in {".nc", ".cdf"}
            and hdf5_has_netcdf_markers(path)
        ):
            strong_signal = FormatSignal(
                "netcdf",
                "container_magic_plus_netcdf_marker",
                1.0,
                "HDF5 container with explicit NetCDF4 marker attributes",
            )
    if strong_signal is not None:
        signals.append(strong_signal)

    hint = normalize_format_hint(request.format_hint)
    hint_signal = None
    if hint is not None:
        hint_signal = FormatSignal(hint, "explicit_hint", 0.99, f"requested format hint {request.format_hint}")
        signals.append(hint_signal)

    suffix = _suffix_signal(path)
    if suffix is not None:
        signals.append(suffix)

    structured_text = (
        _structured_text_signal(path) if path.is_file() else None
    )
    if structured_text is not None:
        signals.append(structured_text)

    if path.is_file() and _looks_like_delimited_text(path):
        signals.append(FormatSignal("csv", "text_probe", 0.8, "consistent delimited UTF-8 text sample"))

    if strong_signal is not None and hint_signal is not None and strong_signal.format != hint_signal.format:
        return FormatDecision(
            selected_format=None,
            basis="conflicting_strong_signals",
            support_score=0.0,
            signals=signals,
            conflicted=True,
        )
    if strong_signal is not None:
        return FormatDecision(strong_signal.format, strong_signal.basis, strong_signal.support_score, signals)
    if hint_signal is not None:
        return FormatDecision(hint_signal.format, hint_signal.basis, hint_signal.support_score, signals)
    if (
        structured_text is not None
        and suffix is not None
        and structured_text.format != suffix.format
    ):
        return FormatDecision(
            selected_format=None,
            basis="conflicting_text_and_suffix_signals",
            support_score=0.0,
            signals=signals,
            conflicted=True,
        )
    if structured_text is not None:
        return FormatDecision(
            structured_text.format,
            structured_text.basis,
            structured_text.support_score,
            signals,
        )
    if suffix is not None:
        return FormatDecision(suffix.format, suffix.basis, suffix.support_score, signals)

    text_signal = next((signal for signal in signals if signal.basis == "text_probe"), None)
    if text_signal is not None:
        return FormatDecision(text_signal.format, text_signal.basis, text_signal.support_score, signals)
    return FormatDecision(None, "no_supported_signal", 0.0, signals)


def build_raw_binary_schema(path: Path, file_id: Optional[str] = None) -> DatasetSchema:
    is_directory = path.is_dir()
    if is_directory:
        prefix_hex = None
        byte_size = None
        entry_count = sum(1 for item in path.rglob("*") if item.is_file())
    else:
        with path.open("rb") as handle:
            prefix_hex = handle.read(16).hex(" ")
        byte_size = path.stat().st_size
        entry_count = None
    return DatasetSchema(
        dataset_id=path.stem,
        file_id=file_id or path.name,
        file_format="raw_binary",
        data_modality="unknown",
        fields=[],
        metadata={
            "resource_kind": "directory_store" if is_directory else "file",
            "byte_size": byte_size,
            "entry_count": entry_count,
            "magic_hex_prefix": prefix_hex,
            "field_claim_policy": "abstain_without_spec_backed_extractor",
            "extraction_errors": [],
        },
        notes=[
            "The payload does not have a selected spec-backed deterministic extractor.",
            "The high-fidelity policy records file-level evidence and abstains from unsupported field claims.",
        ],
    )


def _abstention_issue(decision: FormatDecision) -> ExtractionIssue:
    if decision.conflicted:
        return ExtractionIssue(
            code="unknown_conflicting_format_signals",
            stage="format_detection",
            severity="warning",
            message="Strong format signals conflict; no extractor was selected.",
            details={"formats": sorted({signal.format for signal in decision.signals})},
        )
    if decision.selected_format is None:
        return ExtractionIssue(
            code="unknown_no_signature",
            stage="format_detection",
            severity="warning",
            message="No trustworthy format signature, hint, suffix, or text probe selected a parser.",
        )
    if decision.selected_format in {"raw_binary", "zip_container"}:
        return ExtractionIssue(
            code="unknown_container_unparsed",
            stage="extractor_selection",
            severity="warning",
            message="The container or raw payload is recognized but is not parsed into field claims.",
            details={"selected_format": decision.selected_format},
        )
    return ExtractionIssue(
        code="unknown_no_spec_backed_extractor",
        stage="extractor_selection",
        severity="warning",
        message="The format is recognized but no spec-backed extractor is registered.",
        details={"selected_format": decision.selected_format},
    )


def _attach_unknown_traces(
    schema: DatasetSchema,
    request: ExtractionRequest,
    issues: list[ExtractionIssue],
) -> None:
    """Record what was inspected and why an unresolved property stayed unknown."""

    sampling = schema.metadata.get("sampling", {})
    sampling_scope = {
        "configured_sample_limit": request.sample_limit,
        "sampled_rows": schema.metadata.get("sampled_rows"),
        "observed_records": sampling.get("observed_records")
        if isinstance(sampling, dict)
        else None,
        "sample_truncated": sampling.get("sample_truncated")
        if isinstance(sampling, dict)
        else None,
    }
    sampling_scope = {
        key: value for key, value in sampling_scope.items() if value is not None
    }
    requirements = {
        "physical_type": "a parser-observed or format-declared physical representation",
        "logical_type": "an unambiguous declared type or value pattern",
        "semantic_type": "explicit domain metadata or unambiguous field-level semantics",
        "unit": "an explicit unit declaration or a supported unambiguous unit convention",
    }
    issue_codes = [issue.code for issue in issues]
    for field in schema.fields:
        evidence = [
            {
                "evidence_type": item.evidence_type,
                "source": item.source,
                "detail": item.detail,
            }
            for item in field.source_evidence
        ]
        for property_name in ("physical_type", "logical_type", "semantic_type", "unit"):
            value = getattr(field, property_name)
            if value not in (None, "", "unknown"):
                continue
            reason = field.uncertainty_reason or f"{property_name}_support_not_observed"
            field.unknown_trace[property_name] = {
                "state": "unknown",
                "rationale": reason,
                "inspected_evidence": evidence,
                "missing_requirements": [requirements[property_name]],
                "ambiguity": [reason] if "ambig" in reason.lower() else [],
                "conflicts": [code for code in issue_codes if "conflict" in code],
                "sampling_scope": sampling_scope,
            }


def extract_path(request: ExtractionRequest) -> ExtractionOutcome:
    path = Path(request.path)
    if not path.exists() or not (path.is_file() or path.is_dir()):
        decision = FormatDecision(None, "input_unavailable", 0.0, [])
        return ExtractionOutcome(
            status="failed",
            format_decision=decision,
            extractor=None,
            schema=None,
            issues=[
                ExtractionIssue(
                    code="parser_error",
                    stage="file_intake",
                    severity="error",
                    message="Input path does not exist or is not a file.",
                    details={"path": str(path)},
                )
            ],
        )
    actual_resource_kind = _actual_resource_kind(path)
    requested_resource_kind = request.resource_kind.strip().lower()
    if requested_resource_kind not in {"auto", "file", "directory_store"}:
        return ExtractionOutcome(
            status="failed",
            format_decision=FormatDecision(None, "invalid_resource_kind", 0.0, []),
            extractor=None,
            schema=None,
            issues=[
                ExtractionIssue(
                    "resource_kind_mismatch",
                    "file_intake",
                    "error",
                    "Requested resource kind is not supported.",
                    {"requested_resource_kind": request.resource_kind},
                )
            ],
        )
    if requested_resource_kind != "auto" and requested_resource_kind != actual_resource_kind:
        return ExtractionOutcome(
            status="failed",
            format_decision=FormatDecision(None, "resource_kind_mismatch", 0.0, []),
            extractor=None,
            schema=None,
            issues=[
                ExtractionIssue(
                    "resource_kind_mismatch",
                    "file_intake",
                    "error",
                    "Requested resource kind does not match the selected path.",
                    {
                        "requested_resource_kind": requested_resource_kind,
                        "actual_resource_kind": actual_resource_kind,
                    },
                )
            ],
        )

    try:
        decision = detect_format(request)
    except OSError as exc:
        return ExtractionOutcome(
            status="failed",
            format_decision=FormatDecision(None, "file_intake_error", 0.0, []),
            extractor=None,
            schema=None,
            issues=[ExtractionIssue("parser_error", "file_intake", "error", str(exc))],
        )
    capability = EXTRACTOR_CAPABILITIES.get(decision.selected_format or "")
    runner = EXTRACTOR_RUNNERS.get(decision.selected_format or "")
    if (
        decision.conflicted
        or capability is None
        or runner is None
        or actual_resource_kind not in capability.resource_kinds
    ):
        return ExtractionOutcome(
            status="abstained",
            format_decision=decision,
            extractor=None,
            schema=build_raw_binary_schema(path),
            issues=[_abstention_issue(decision)],
        )

    try:
        schema = runner(request)
    except StructuredExtractionError as exc:
        return ExtractionOutcome(
            status=exc.status,
            format_decision=decision,
            extractor=capability,
            schema=None,
            issues=[ExtractionIssue(exc.code, exc.stage, "error" if exc.status == "failed" else "warning", str(exc), exc.details)],
        )
    except RuntimeError as exc:
        code = "dependency_unavailable" if "required" in str(exc).lower() else "parser_error"
        return ExtractionOutcome(
            status="failed",
            format_decision=decision,
            extractor=capability,
            schema=None,
            issues=[ExtractionIssue(code, "extraction", "error", str(exc))],
        )
    except Exception as exc:
        return ExtractionOutcome(
            status="failed",
            format_decision=decision,
            extractor=capability,
            schema=None,
            issues=[ExtractionIssue("parser_error", "extraction", "error", str(exc))],
        )

    extraction_errors = schema.metadata.get("extraction_errors", [])
    issues: list[ExtractionIssue] = []
    status = "success"
    if extraction_errors:
        status = "partial"
        issues.append(
            ExtractionIssue(
                code="partial_extraction",
                stage="extraction",
                severity="warning",
                message="The extractor returned a schema with one or more recoverable errors.",
                details={"error_count": len(extraction_errors)},
            )
        )
        issues.extend(
            ExtractionIssue(
                str(error.get("code", "partial_extraction")),
                "extraction",
                "warning",
                str(error.get("error", "Recoverable extraction error.")),
                {"path": error.get("path")},
            )
            for error in extraction_errors
        )
    if decision.selected_format == "csv" and schema.metadata.get("sampled_rows") == request.sample_limit:
        issues.append(
            ExtractionIssue(
                code="sampling_insufficient",
                stage="profiling",
                severity="note",
                message="Profile statistics are based on the configured sample limit.",
                details={"sample_limit": request.sample_limit},
            )
        )
    if decision.selected_format == "json" and schema.metadata.get("sampling", {}).get("sample_truncated"):
        issues.append(
            ExtractionIssue(
                code="sampling_insufficient",
                stage="profiling",
                severity="note",
                message="Observed JSON structure is based on the configured sample limit.",
                details={"sample_limit": request.sample_limit},
            )
        )
    if decision.selected_format == "xml" and schema.metadata.get("sampling", {}).get("sample_truncated"):
        issues.append(
            ExtractionIssue(
                code="sampling_insufficient",
                stage="profiling",
                severity="note",
                message="Observed XML structure is based on the configured element sample limit.",
                details={"sample_limit": request.sample_limit},
            )
        )
    _attach_unknown_traces(schema, request, issues)
    return ExtractionOutcome(status, decision, capability, schema, issues)


def extract_unified(request: ExtractionRequest) -> Dict[str, object]:
    """Return the versioned unified envelope without changing legacy outcomes."""
    return build_unified_schema_envelope(extract_path(request))


def capability_for_format(file_format: str) -> Optional[ExtractorCapability]:
    return EXTRACTOR_CAPABILITIES.get(file_format)


def registered_formats() -> Iterable[str]:
    return sorted(EXTRACTOR_CAPABILITIES)
