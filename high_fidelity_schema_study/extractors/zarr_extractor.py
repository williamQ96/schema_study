from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Dict, List, Optional

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

from ..models import DatasetSchema, EvidenceRecord, FieldSchema
from ..temporal_semantics import analyze_temporal_semantics
from ..unit_normalization import normalize_unit_claim
from .base import StructuredExtractionError


CF_TIME_UNIT_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+?)\s*$",
    flags=re.IGNORECASE,
)
SUPPORTED_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
KNOWN_CALENDARS = SUPPORTED_STANDARD_CALENDARS | {
    "julian",
    "noleap",
    "no_leap",
    "365_day",
    "360_day",
    "all_leap",
    "366_day",
}
STANDARD_NAME_SEMANTICS = {
    "time": "observation_time",
    "latitude": "latitude",
    "longitude": "longitude",
    "height": "elevation",
    "altitude": "elevation",
    "depth": "depth",
    "air_temperature": "air_temperature",
    "sea_water_temperature": "water_temperature",
    "sea_water_salinity": "salinity",
    "air_pressure": "surface_pressure",
    "relative_humidity": "relative_humidity",
    "precipitation_amount": "precipitation",
    "wind_speed": "wind_speed",
}
METADATA_NAMES = {".zgroup", ".zarray", ".zattrs", ".zmetadata"}
REQUIRED_ARRAY_KEYS = {"zarr_format", "shape", "chunks", "dtype", "compressor", "fill_value", "order", "filters"}


def _claim(value: Any, state: str, reason_code: str, evidence_refs: List[str]) -> Dict[str, Any]:
    return {
        "value": value,
        "state": state,
        "reason_code": reason_code,
        "evidence_refs": evidence_refs,
    }


def _portable_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _evidence(relative_source: str, evidence_type: str, detail: str, confidence: float = 1.0) -> EvidenceRecord:
    return EvidenceRecord(
        tier="explicit_metadata",
        evidence_type=evidence_type,
        source=relative_source,
        detail=detail,
        confidence=confidence,
    )


def _safe_metadata_key(key: str) -> bool:
    path = PurePosixPath(key)
    return bool(key) and not path.is_absolute() and ".." not in path.parts and path.name in METADATA_NAMES


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("metadata document must contain a JSON object")
    return payload


def _validate_zarr_format(payload: Dict[str, Any], source: str) -> None:
    version = payload.get("zarr_format")
    if version == 3:
        raise StructuredExtractionError(
            "unsupported_zarr_version",
            "extraction",
            "Zarr v3 metadata is recognized but the current extractor supports Zarr v2 only.",
            status="abstained",
            details={"source": source, "zarr_format": version},
        )
    if version != 2:
        raise ValueError("zarr_format must be 2")


def _validate_array_metadata(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    _validate_zarr_format(payload, source)
    missing = sorted(REQUIRED_ARRAY_KEYS - set(payload))
    if missing:
        raise ValueError(f"missing required .zarray keys: {missing}")

    shape = payload["shape"]
    chunks = payload["chunks"]
    if (
        not isinstance(shape, list)
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape)
        or not isinstance(chunks, list)
        or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in chunks)
        or len(shape) != len(chunks)
    ):
        raise ValueError("shape and chunks must be same-rank integer arrays with valid sizes")
    try:
        np.dtype(_normalize_dtype_descriptor(payload["dtype"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dtype: {payload['dtype']!r}") from exc
    if payload["order"] not in {"C", "F"}:
        raise ValueError("order must be C or F")
    if payload["compressor"] is not None and not isinstance(payload["compressor"], dict):
        raise ValueError("compressor must be an object or null")
    if payload["filters"] is not None and not isinstance(payload["filters"], list):
        raise ValueError("filters must be an array or null")
    separator = payload.get("dimension_separator", ".")
    if separator not in {".", "/"}:
        raise ValueError("dimension_separator must be '.' or '/'")
    return payload


def _normalize_dtype_descriptor(value: Any) -> Any:
    """Restore tuples lost when NumPy structured dtypes are serialized as JSON."""
    if not isinstance(value, list):
        return value
    fields = []
    for entry in value:
        if not isinstance(entry, list) or len(entry) not in {2, 3}:
            raise ValueError("structured dtype fields must be 2- or 3-item arrays")
        name = tuple(entry[0]) if isinstance(entry[0], list) else entry[0]
        field_type = _normalize_dtype_descriptor(entry[1])
        if len(entry) == 2:
            fields.append((name, field_type))
            continue
        shape = entry[2]
        if not isinstance(shape, list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in shape
        ):
            raise ValueError("structured dtype field shapes must be non-negative integer arrays")
        fields.append((name, field_type, tuple(shape)))
    return fields


def _collect_metadata_documents(
    root: Path,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    documents: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, str] = {}
    errors: List[Dict[str, Any]] = []
    root_resolved = root.resolve()
    inventory = {
        "metadata_document_count": 0,
        "observed_non_metadata_file_count": 0,
        "pruned_array_directory_count": 0,
    }

    consolidated_path = root / ".zmetadata"
    if consolidated_path.exists():
        try:
            consolidated = _read_json_object(consolidated_path)
            if consolidated.get("zarr_consolidated_format") != 1 or not isinstance(consolidated.get("metadata"), dict):
                raise ValueError(".zmetadata must declare zarr_consolidated_format=1 and an object metadata map")
            for key, payload in consolidated["metadata"].items():
                if not isinstance(key, str) or not _safe_metadata_key(key) or not isinstance(payload, dict):
                    errors.append(
                        {
                            "path": ".zmetadata",
                            "code": "zarr_consolidated_entry_invalid",
                            "error": f"invalid consolidated metadata entry: {key!r}",
                        }
                    )
                    continue
                documents[key] = payload
                sources[key] = f".zmetadata#{key}"
        except ValueError as exc:
            errors.append({"path": ".zmetadata", "code": "zarr_metadata_malformed", "error": str(exc)})

    consolidated_array_nodes = {
        _node_path(key)
        for key in documents
        if key.endswith(".zarray")
    }
    for directory_text, directory_names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory = Path(directory_text)
        node_path = _portable_relative(directory, root) if directory != root else "/"
        if ".zarray" in filenames or node_path in consolidated_array_nodes:
            inventory["pruned_array_directory_count"] += len(directory_names)
            directory_names[:] = []
        inventory["observed_non_metadata_file_count"] += sum(
            name not in METADATA_NAMES for name in filenames
        )
        for name in sorted(set(filenames) & (METADATA_NAMES - {".zmetadata"})):
            path = directory / name
            try:
                if not path.resolve().is_relative_to(root_resolved):
                    errors.append(
                        {
                            "path": _portable_relative(path, root),
                            "code": "zarr_path_outside_store",
                            "error": "metadata path resolves outside the selected store root",
                        }
                    )
                    continue
                key = _portable_relative(path, root)
                direct_payload = _read_json_object(path)
                if key in documents and documents[key] != direct_payload:
                    errors.append(
                        {
                            "path": key,
                            "code": "zarr_consolidated_conflict",
                            "error": "consolidated metadata conflicts with the direct metadata document",
                        }
                    )
                documents[key] = direct_payload
                sources[key] = key
            except ValueError as exc:
                errors.append(
                    {
                        "path": _portable_relative(path, root),
                        "code": "zarr_metadata_malformed",
                        "error": str(exc),
                    }
                )

    inventory["metadata_document_count"] = len(documents)
    return documents, sources, errors, inventory


def _node_path(metadata_key: str) -> str:
    parent = PurePosixPath(metadata_key).parent.as_posix()
    return "/" if parent == "." else parent


def _attrs_for(node_path: str, documents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    key = ".zattrs" if node_path == "/" else f"{node_path}/.zattrs"
    payload = documents.get(key, {})
    return payload if isinstance(payload, dict) else {}


def _semantic_from_attrs(attrs: Dict[str, Any]) -> str:
    standard_name = str(attrs.get("standard_name", "")).strip().lower()
    if standard_name in STANDARD_NAME_SEMANTICS:
        return STANDARD_NAME_SEMANTICS[standard_name]
    axis = str(attrs.get("axis", "")).strip().upper()
    units = str(attrs.get("units", "")).strip().lower()
    if axis == "T" or CF_TIME_UNIT_RE.match(units):
        return "observation_time"
    if axis == "X" or units in {"degrees_east", "degree_east", "degrees_e", "degree_e"}:
        return "longitude"
    if axis == "Y" or units in {"degrees_north", "degree_north", "degrees_n", "degree_n"}:
        return "latitude"
    if axis == "Z":
        return "depth" if str(attrs.get("positive", "")).lower() == "down" else "elevation"
    return "unknown"


def _logical_from_semantic(semantic_type: str, is_dimension_coordinate: bool) -> str:
    if semantic_type == "observation_time":
        return "temporal_coordinate"
    if semantic_type in {"latitude", "longitude", "depth", "elevation"} or is_dimension_coordinate:
        return "coordinate"
    if semantic_type != "unknown":
        return "measurement"
    return "unknown"


def _coordinate_claim(
    field_path: str,
    field_name: str,
    dimensions: List[str],
    attrs: Dict[str, Any],
    auxiliary_paths: set[str],
) -> tuple[List[str], Dict[str, Any]]:
    roles: List[str] = []
    refs: List[str] = []
    hints: List[str] = []
    axis = str(attrs.get("axis", "")).strip().upper()
    standard_name = str(attrs.get("standard_name", "")).strip().lower()
    units = str(attrs.get("units", "")).strip().lower()
    if len(dimensions) == 1 and dimensions[0] == field_name:
        roles.append("dimension_coordinate")
        refs.append("attribute:_ARRAY_DIMENSIONS")
    if field_path in auxiliary_paths:
        roles.append("auxiliary_coordinate")
        refs.append("attribute:coordinates")
    if axis in {"X", "Y", "Z", "T"}:
        roles.append(f"axis_{axis.lower()}")
        refs.append("attribute:axis")
        hints.append(axis)
    if standard_name in {"latitude", "longitude", "time", "height", "altitude", "depth"}:
        roles.append(f"standard_{standard_name}")
        refs.append("attribute:standard_name")
        hints.append(
            {
                "latitude": "Y",
                "longitude": "X",
                "time": "T",
                "height": "Z",
                "altitude": "Z",
                "depth": "Z",
            }[standard_name]
        )
    if units in {"degrees_north", "degree_north", "degrees_east", "degree_east"}:
        roles.append("geospatial_coordinate")
        refs.append("attribute:units")
        hints.append("Y" if "north" in units else "X")
    roles = list(dict.fromkeys(roles))
    conflicted = len(set(hints)) > 1
    return roles, _claim(
        roles,
        "conflicted" if conflicted else "supported" if roles else "unknown",
        "conflicting_coordinate_evidence"
        if conflicted
        else "explicit_coordinate_evidence"
        if roles
        else "no_coordinate_evidence",
        list(dict.fromkeys(refs)),
    )


def _dimension_claim(attrs: Dict[str, Any], rank: int) -> tuple[List[str], Dict[str, Any]]:
    value = attrs.get("_ARRAY_DIMENSIONS")
    refs = ["attribute:_ARRAY_DIMENSIONS"] if value is not None else []
    if value is None:
        return [], _claim([], "unknown", "array_dimensions_missing", refs)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        return [], _claim(value, "conflicted", "array_dimensions_invalid", refs)
    if len(value) != rank:
        return value, _claim(value, "conflicted", "array_dimensions_rank_conflict", refs)
    return value, _claim(value, "supported", "explicit_xarray_dimensions", refs)


def _resolve_auxiliary_coordinate_paths(
    array_nodes: Dict[str, Dict[str, Any]],
    documents: Dict[str, Dict[str, Any]],
    sources: Dict[str, str],
) -> tuple[set[str], List[Dict[str, Any]]]:
    known_paths = set(array_nodes)
    resolved: set[str] = set()
    gaps: List[Dict[str, Any]] = []
    for owner_path in sorted(known_paths):
        coordinates = _attrs_for(owner_path, documents).get("coordinates")
        if not isinstance(coordinates, str):
            continue
        parent = PurePosixPath(".") if owner_path == "/" else PurePosixPath(owner_path).parent
        attrs_key = ".zattrs" if owner_path == "/" else f"{owner_path}/.zattrs"
        for token in coordinates.split():
            token_path = PurePosixPath(token)
            if token_path.is_absolute() or ".." in token_path.parts:
                candidate = None
            else:
                candidate = (parent / token_path).as_posix()
                if candidate == ".":
                    candidate = "/"
            if candidate in known_paths:
                resolved.add(candidate)
            else:
                gaps.append(
                    {
                        "code": "unresolved_coordinate_reference",
                        "state": "unknown",
                        "owner_path": owner_path,
                        "reference": token,
                        "evidence_source": sources.get(attrs_key, attrs_key),
                        "evidence_refs": [f"{sources.get(attrs_key, attrs_key)}#coordinates"],
                    }
                )
    return resolved, gaps


def _calendar_claim(attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not (CF_TIME_UNIT_RE.match(str(attrs.get("units", ""))) or "calendar" in attrs):
        return None
    calendar = str(attrs.get("calendar", "standard")).strip().lower()
    if calendar in KNOWN_CALENDARS:
        return _claim(
            calendar,
            "supported",
            "explicit_cf_calendar_metadata_only",
            ["attribute:units", "attribute:calendar"],
        )
    return _claim(
        calendar,
        "conflicted",
        "unsupported_calendar",
        ["attribute:units", "attribute:calendar"],
    )


def extract_zarr_schema(path: str, sample_limit: int = 200) -> DatasetSchema:
    del sample_limit
    if np is None:
        raise RuntimeError("numpy is required for Zarr metadata validation but is not installed in this environment.")

    root = Path(path)
    if not root.is_dir():
        raise StructuredExtractionError(
            "resource_kind_mismatch",
            "file_intake",
            "Zarr metadata extraction requires a local directory store.",
        )
    if (root / "zarr.json").exists():
        raise StructuredExtractionError(
            "unsupported_zarr_version",
            "extraction",
            "Zarr v3 metadata is recognized but the current extractor supports Zarr v2 only.",
            status="abstained",
            details={"source": "zarr.json"},
        )

    documents, sources, extraction_errors, store_inventory = _collect_metadata_documents(root)
    if not any(key.endswith((".zgroup", ".zarray")) for key in documents):
        code = "zarr_metadata_missing" if not extraction_errors else "zarr_metadata_malformed"
        status = "abstained" if code == "zarr_metadata_missing" else "failed"
        raise StructuredExtractionError(
            code,
            "extraction",
            "No valid Zarr v2 group or array metadata could be extracted.",
            status=status,
            details={"error_count": len(extraction_errors)},
        )

    groups: List[Dict[str, Any]] = []
    fields: List[FieldSchema] = []
    analyses: List[Dict[str, Any]] = []
    calendar_claims: Dict[str, Dict[str, Any]] = {}
    dimensions: Dict[str, Dict[str, Any]] = {}
    compatibility_gaps: List[Dict[str, Any]] = []

    array_nodes = {
        _node_path(key): payload
        for key, payload in documents.items()
        if key.endswith(".zarray")
    }
    auxiliary_paths, coordinate_gaps = _resolve_auxiliary_coordinate_paths(array_nodes, documents, sources)
    compatibility_gaps.extend(coordinate_gaps)

    for key, payload in sorted(documents.items()):
        if not key.endswith(".zgroup"):
            continue
        try:
            _validate_zarr_format(payload, sources[key])
            node_path = _node_path(key)
            attrs = _attrs_for(node_path, documents)
            group_evidence = [_evidence(sources[key], "zarr_group_metadata", "zarr_format=2")]
            attrs_key = ".zattrs" if node_path == "/" else f"{node_path}/.zattrs"
            if attrs_key in documents:
                group_evidence.append(_evidence(sources[attrs_key], "zarr_attribute", "explicit group attributes", 0.99))
            groups.append(
                {
                    "path": node_path,
                    "kind": "group",
                    "attributes": attrs,
                    "source_evidence": group_evidence,
                }
            )
        except StructuredExtractionError:
            raise
        except ValueError as exc:
            extraction_errors.append({"path": key, "code": "zarr_metadata_malformed", "error": str(exc)})

    for node_path, payload in sorted(array_nodes.items()):
        key = ".zarray" if node_path == "/" else f"{node_path}/.zarray"
        try:
            metadata = _validate_array_metadata(payload, sources[key])
            attrs = _attrs_for(node_path, documents)
            field_name = root.stem if node_path == "/" else PurePosixPath(node_path).name
            dimensions_value, dimension_claim = _dimension_claim(attrs, len(metadata["shape"]))
            dimensions_list = dimensions_value if dimension_claim["state"] == "supported" else []
            roles, coordinate_claim = _coordinate_claim(
                node_path,
                field_name,
                dimensions_list,
                attrs,
                auxiliary_paths,
            )
            coordinate_conflicted = coordinate_claim["state"] == "conflicted"
            semantic_type = "unknown" if coordinate_conflicted else _semantic_from_attrs(attrs)
            is_dimension_coordinate = "dimension_coordinate" in roles
            logical_type = _logical_from_semantic(semantic_type, is_dimension_coordinate)
            units = attrs.get("units")
            evidence = [
                _evidence(
                    sources[key],
                    "zarr_array_metadata",
                    "keys=dtype,shape,chunks,compressor,fill_value,order,filters,dimension_separator",
                )
            ]
            attrs_key = ".zattrs" if node_path == "/" else f"{node_path}/.zattrs"
            if attrs_key in documents:
                evidence.append(_evidence(sources[attrs_key], "zarr_attribute", "explicit array attributes", 0.99))
            unit_normalization = normalize_unit_claim(
                str(units) if units is not None else None,
                [item.evidence_type for item in evidence],
            )
            fill_value = metadata.get("fill_value")
            field = FieldSchema(
                field_name=field_name,
                field_path=node_path,
                physical_type=str(metadata["dtype"]),
                logical_type=logical_type,
                semantic_type=semantic_type,
                nullable=fill_value is not None,
                shape=list(metadata["shape"]),
                unit=str(units) if units is not None else None,
                unit_normalization=unit_normalization,
                description=str(attrs.get("long_name")) if attrs.get("long_name") is not None else None,
                source_evidence=evidence,
                confidence=1.0 if semantic_type != "unknown" or logical_type != "unknown" else 0.9,
                uncertainty_reason=(
                    "Conflicting convention-backed coordinate evidence"
                    if coordinate_conflicted
                    else None
                    if semantic_type != "unknown" or logical_type != "unknown"
                    else "Zarr/Xarray/CF metadata does not support a precise logical or semantic claim"
                ),
                extraction_method="zarr_v2_metadata_extractor",
            )
            fields.append(field)
            for index, dimension_name in enumerate(dimensions_list):
                candidate = {
                    "size": metadata["shape"][index],
                    "source_arrays": [node_path],
                    "source_evidence": [
                        asdict(_evidence(sources[attrs_key], "zarr_attribute", f"_ARRAY_DIMENSIONS[{index}]={dimension_name!r}", 0.99))
                    ],
                }
                existing = dimensions.get(dimension_name)
                if existing and existing["size"] != candidate["size"]:
                    existing["state"] = "conflicted"
                    existing["reason_code"] = "dimension_size_conflict"
                    existing["source_arrays"].append(node_path)
                    existing["source_evidence"].extend(candidate["source_evidence"])
                    compatibility_gaps.append(
                        {
                            "code": "dimension_size_conflict",
                            "state": "conflicted",
                            "dimension": dimension_name,
                            "observed_sizes": sorted({existing["size"], candidate["size"]}),
                            "source_arrays": list(existing["source_arrays"]),
                            "evidence_refs": [
                                item["source"]
                                for item in existing["source_evidence"]
                            ],
                        }
                    )
                elif existing:
                    existing["source_arrays"].append(node_path)
                    existing["source_evidence"].extend(candidate["source_evidence"])
                elif not existing:
                    candidate.update({"state": "supported", "reason_code": "explicit_xarray_dimensions"})
                    dimensions[dimension_name] = candidate
            calendar_claim = _calendar_claim(attrs)
            if calendar_claim is not None:
                calendar_claims[node_path] = calendar_claim
            analyses.append(
                {
                    "field_path": node_path,
                    "metadata_source": sources[key],
                    "attributes": attrs,
                    "array_metadata": {
                        "dtype": metadata["dtype"],
                        "shape": metadata["shape"],
                        "chunks": metadata["chunks"],
                        "compressor": metadata["compressor"],
                        "fill_value": metadata["fill_value"],
                        "order": metadata["order"],
                        "filters": metadata["filters"],
                        "dimension_separator": metadata.get("dimension_separator", "."),
                    },
                    "dimensions": dimensions_value,
                    "dimension_claim": dimension_claim,
                    "coordinate_roles": roles,
                    "coordinate_role_claim": coordinate_claim,
                    "unit_claim": _claim(
                        field.unit,
                        "supported" if field.unit is not None else "unknown",
                        "explicit_units_attribute" if field.unit is not None else "missing_units_attribute",
                        ["attribute:units"] if field.unit is not None else [],
                    ),
                }
            )
        except StructuredExtractionError:
            raise
        except ValueError as exc:
            extraction_errors.append({"path": key, "code": "zarr_metadata_malformed", "error": str(exc)})

    if not fields and not groups:
        raise StructuredExtractionError(
            "zarr_metadata_malformed",
            "extraction",
            "Zarr metadata documents were present but none could be validated.",
            details={"error_count": len(extraction_errors)},
        )

    temporal = analyze_temporal_semantics({field.field_name: [] for field in fields}, fields)
    temporal["temporal_analysis"]["issues"].append(
        {
            "code": "metadata_only_no_temporal_samples",
            "severity": "note",
            "detail": "Metadata-only Zarr extraction does not read chunk payloads, so no canonical time axis is selected from numeric arrays.",
        }
    )
    schema = DatasetSchema(
        dataset_id=root.stem,
        file_id=root.name,
        file_format="zarr",
        data_modality="multidimensional",
        fields=fields,
        groups=groups,
        metadata={
            "resource_kind": "directory_store",
            "zarr_format": 2,
            "metadata_only": True,
            "chunk_payloads_read": False,
            "value_observation": {
                "state": "unknown",
                "reason_code": "chunk_payloads_not_read",
                "properties": [
                    "missing_count",
                    "missing_ratio",
                    "unique_ratio",
                    "value_range",
                    "example_values",
                ],
            },
            "documents_discovered": sorted(documents),
            "store_inventory": {
                **store_inventory,
                "walk_strategy": "array_boundary_aware",
                "payload_bytes_read": 0,
            },
            "dimensions": dimensions,
            "zarr_analysis": {
                "arrays": analyses,
                "calendar_claims": calendar_claims,
                "temporal": temporal["temporal_analysis"],
                "compatibility_gaps": compatibility_gaps,
            },
            "extraction_errors": extraction_errors,
        },
        notes=[
            "The current Zarr extractor reads Zarr v2 metadata only and does not read chunk payloads.",
            "Convention-backed claims remain unknown or conflicted when explicit metadata is absent or inconsistent.",
            "Value-level statistics remain unknown because chunk payloads are not read.",
        ],
    )
    return schema
