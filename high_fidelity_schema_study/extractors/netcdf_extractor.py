from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

from ..deterministic_profile import attach_dataset_profile
from ..models import DatasetSchema, EvidenceRecord, FieldSchema
from ..temporal_semantics import analyze_temporal_semantics
from ..unit_normalization import normalize_unit_claim

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None

try:
    from scipy.io import netcdf_file  # type: ignore
except ImportError:  # pragma: no cover
    netcdf_file = None


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
CF_TIME_UNIT_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+?)\s*$",
    flags=re.IGNORECASE,
)
SUPPORTED_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
KNOWN_CALENDARS = SUPPORTED_STANDARD_CALENDARS | {"julian", "noleap", "no_leap", "365_day", "360_day", "all_leap", "366_day"}
NETCDF4_MARKER_ATTRIBUTES = {"_NCProperties", "_Netcdf4Coordinates", "_Netcdf4Dimid"}

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


def hdf5_has_netcdf_markers(path: Path) -> bool:
    if h5py is None:
        return False
    try:
        with h5py.File(path, "r") as handle:
            if NETCDF4_MARKER_ATTRIBUTES.intersection(handle.attrs.keys()):
                return True
            found = False

            def inspect(_name: str, node: Any) -> Optional[bool]:
                nonlocal found
                if NETCDF4_MARKER_ATTRIBUTES.intersection(node.attrs.keys()):
                    found = True
                    return True
                return None

            handle.visititems(inspect)
            return found
    except OSError:
        return False


def _normalize_value(value: Any) -> Any:
    if h5py is not None and isinstance(value, (h5py.Reference, h5py.RegionReference)):
        return "hdf5_reference"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _normalize_value(value.item())
    if isinstance(value, np.ndarray):
        return [_normalize_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _attrs(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _normalize_value(value) for key, value in mapping.items()}


def _dtype_name(dtype: Any) -> str:
    normalized = np.dtype(dtype)
    if normalized.kind in {"S", "U", "O"}:
        return "string"
    return str(normalized)


def _evidence(
    path: Path,
    field_path: str,
    evidence_type: str,
    detail: str,
    confidence: float = 1.0,
) -> EvidenceRecord:
    return EvidenceRecord(
        tier="explicit_metadata",
        evidence_type=evidence_type,
        source=f"{path}:{field_path}",
        detail=detail,
        confidence=confidence,
    )


def _claim(value: Any, state: str, reason_code: str, evidence_refs: List[str]) -> Dict[str, Any]:
    return {
        "value": value,
        "state": state,
        "reason_code": reason_code,
        "evidence_refs": evidence_refs,
    }


def _semantic_from_attrs(_name: str, attrs: Dict[str, Any]) -> str:
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
    if semantic_type in {"latitude", "longitude", "depth", "elevation"}:
        return "coordinate"
    if semantic_type.endswith("_identifier"):
        return "identifier"
    if semantic_type != "unknown":
        return "measurement"
    if is_dimension_coordinate:
        return "coordinate"
    return "unknown"


def _time_origin(text: str) -> Optional[datetime]:
    normalized = text.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    if normalized.endswith(" UTC"):
        normalized = f"{normalized[:-4]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed


def _decode_cf_time(values: Iterable[Any], units: str, calendar: str) -> tuple[List[str], Dict[str, Any]]:
    match = CF_TIME_UNIT_RE.match(units)
    evidence = ["attribute:units", "attribute:calendar"]
    if match is None:
        return [], _claim("unknown", "unknown", "malformed_cf_time_units", evidence)
    normalized_calendar = calendar.strip().lower() if calendar else "standard"
    if normalized_calendar not in KNOWN_CALENDARS:
        return [], _claim(normalized_calendar, "conflicted", "unsupported_calendar", evidence)
    if normalized_calendar not in SUPPORTED_STANDARD_CALENDARS:
        return [], _claim(normalized_calendar, "supported", "explicit_cf_calendar_not_decoded", evidence)

    origin = _time_origin(match.group(2))
    if origin is None:
        return [], _claim(normalized_calendar, "conflicted", "malformed_cf_time_origin", evidence)
    unit = match.group(1).lower()
    multipliers = {
        "second": 1,
        "seconds": 1,
        "minute": 60,
        "minutes": 60,
        "hour": 3600,
        "hours": 3600,
        "day": 86400,
        "days": 86400,
    }
    decoded: List[str] = []
    try:
        for value in values:
            if value is None:
                continue
            number = float(value)
            if np.isnan(number):
                continue
            instant = origin + timedelta(seconds=number * multipliers[unit])
            decoded.append(instant.isoformat().replace("+00:00", "Z"))
    except (TypeError, ValueError, OverflowError):
        return [], _claim(normalized_calendar, "conflicted", "cf_time_decode_failed", evidence)
    return decoded, _claim(normalized_calendar, "supported", "explicit_cf_calendar", evidence)


def _sample_values(values: Any, limit: int) -> List[Any]:
    try:
        flattened = np.asarray(values).reshape(-1)
    except Exception:
        return []
    sampled = []
    for value in flattened[:limit]:
        normalized = _normalize_value(value)
        if isinstance(normalized, float) and np.isnan(normalized):
            continue
        sampled.append(normalized)
    return sampled


def _coordinate_roles(
    variable_name: str,
    dimensions: List[str],
    attrs: Dict[str, Any],
    dimension_names: set[str],
    auxiliary_names: set[str],
) -> tuple[List[str], Dict[str, Any]]:
    roles: List[str] = []
    evidence_refs: List[str] = []
    axis_hints: List[str] = []
    axis = str(attrs.get("axis", "")).strip().upper()
    standard_name = str(attrs.get("standard_name", "")).strip().lower()
    units = str(attrs.get("units", "")).strip().lower()
    if variable_name in dimension_names and dimensions == [variable_name]:
        roles.append("dimension_coordinate")
        evidence_refs.append("structure:dimension_coordinate")
    if variable_name in auxiliary_names:
        roles.append("auxiliary_coordinate")
        evidence_refs.append("attribute:coordinates")
    if axis in {"X", "Y", "Z", "T"}:
        roles.append(f"axis_{axis.lower()}")
        evidence_refs.append("attribute:axis")
        axis_hints.append(axis)
    if standard_name in {"latitude", "longitude", "time", "height", "altitude", "depth"}:
        roles.append(f"standard_{standard_name}")
        evidence_refs.append("attribute:standard_name")
        axis_hints.append(
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
        evidence_refs.append("attribute:units")
        axis_hints.append("Y" if "north" in units else "X")
    unique_roles = list(dict.fromkeys(roles))
    conflicted = len(set(axis_hints)) > 1
    state = "conflicted" if conflicted else "supported" if unique_roles else "unknown"
    reason_code = (
        "conflicting_cf_coordinate_evidence"
        if conflicted
        else "explicit_cf_coordinate_evidence"
        if unique_roles
        else "no_coordinate_evidence"
    )
    return unique_roles, _claim(unique_roles, state, reason_code, evidence_refs)


def _build_field(
    path: Path,
    field_path: str,
    name: str,
    dtype: Any,
    shape: List[int],
    dimensions: List[str],
    attrs: Dict[str, Any],
    samples: List[Any],
    dimension_names: set[str],
    auxiliary_names: set[str],
    bounds_names: set[str],
) -> tuple[FieldSchema, Dict[str, Any]]:
    is_dimension_coordinate = name in dimension_names and dimensions == [name]
    is_bounds_variable = name in bounds_names
    roles, coordinate_claim = _coordinate_roles(name, dimensions, attrs, dimension_names, auxiliary_names)
    coordinate_conflicted = coordinate_claim["state"] == "conflicted"
    semantic_type = (
        "time_bounds"
        if is_bounds_variable
        else "unknown"
        if coordinate_conflicted
        else _semantic_from_attrs(name, attrs)
    )
    logical_type = "relationship" if is_bounds_variable else _logical_from_semantic(semantic_type, is_dimension_coordinate)
    units = attrs.get("units")
    evidence = [_evidence(path, field_path, "netcdf_variable", f"dimensions={dimensions}; shape={shape}; dtype={_dtype_name(dtype)}")]
    for attr_name in ("standard_name", "long_name", "units", "axis", "coordinates", "calendar", "bounds", "_FillValue", "missing_value"):
        if attr_name in attrs:
            evidence.append(_evidence(path, field_path, "netcdf_attribute", f"{attr_name}={attrs[attr_name]!r}", 0.99))
    unit_normalization = normalize_unit_claim(
        str(units) if units is not None else None,
        [item.evidence_type for item in evidence],
    )
    missing_markers = [
        attrs[key]
        for key in ("_FillValue", "missing_value")
        if key in attrs
    ]
    missing_count = sum(1 for sample in samples if sample in missing_markers)
    field = FieldSchema(
        field_name=name,
        field_path=field_path,
        physical_type=_dtype_name(dtype),
        logical_type=logical_type,
        semantic_type=semantic_type,
        nullable=bool(missing_markers),
        shape=shape,
        unit=str(units) if units is not None else None,
        unit_normalization=unit_normalization,
        description=str(attrs.get("long_name")) if attrs.get("long_name") is not None else None,
        example_values=[str(value) for value in samples[:3]],
        missing_count=missing_count,
        source_evidence=evidence,
        confidence=1.0 if semantic_type != "unknown" or logical_type != "unknown" else 0.9,
        uncertainty_reason=(
            "Conflicting CF coordinate evidence"
            if coordinate_conflicted
            else None
            if semantic_type != "unknown" or logical_type != "unknown"
            else "CF metadata does not support a precise logical or semantic claim"
        ),
        extraction_method="netcdf_cf_deterministic_extractor",
    )
    analysis = {
        "field_path": field_path,
        "dimensions": dimensions,
        "attributes": attrs,
        "coordinate_roles": roles,
        "coordinate_role_claim": coordinate_claim,
        "is_bounds_variable": is_bounds_variable,
        "bounds_claim": _claim(
            is_bounds_variable,
            "supported" if is_bounds_variable else "unknown",
            "referenced_by_bounds_attribute" if is_bounds_variable else "not_referenced_as_bounds",
            ["attribute:bounds"] if is_bounds_variable else [],
        ),
        "unit_claim": _claim(
            field.unit,
            "supported" if field.unit is not None else "unknown",
            "explicit_units_attribute" if field.unit is not None else "missing_units_attribute",
            ["attribute:units"] if field.unit is not None else [],
        ),
    }
    return field, analysis


def _extract_classic(path: Path, sample_limit: int) -> tuple[DatasetSchema, Dict[str, List[Any]]]:
    if netcdf_file is None:
        raise RuntimeError("scipy is required for NetCDF classic extraction")
    fields: List[FieldSchema] = []
    analyses: List[Dict[str, Any]] = []
    samples_by_path: Dict[str, List[Any]] = {}
    extraction_errors: List[Dict[str, str]] = []
    with netcdf_file(str(path), "r", mmap=False) as handle:
        file_attrs = _attrs(getattr(handle, "_attributes", {}))
        dimensions = {
            name: {
                "size": size,
                "unlimited": size is None,
                "source_evidence": [
                    _evidence(path, name, "netcdf_dimension", f"size={size}; unlimited={size is None}")
                ],
            }
            for name, size in handle.dimensions.items()
        }
        dimension_names = set(dimensions)
        raw_variables = handle.variables
        auxiliary_names = {
            token
            for variable in raw_variables.values()
            for token in str(_normalize_value(getattr(variable, "coordinates", ""))).split()
            if token
        }
        bounds_names = {
            str(_normalize_value(getattr(variable, "bounds", "")))
            for variable in raw_variables.values()
            if getattr(variable, "bounds", None)
        }
        for name, variable in raw_variables.items():
            try:
                attrs = _attrs(getattr(variable, "_attributes", {}))
                values = _sample_values(variable.data, sample_limit)
                samples_by_path[name] = values
                field, analysis = _build_field(
                    path,
                    name,
                    name,
                    variable.data.dtype,
                    list(variable.data.shape),
                    list(variable.dimensions),
                    attrs,
                    values,
                    dimension_names,
                    auxiliary_names,
                    bounds_names,
                )
                fields.append(field)
                analyses.append(analysis)
            except Exception as exc:
                extraction_errors.append({"path": name, "error": str(exc)})

    return DatasetSchema(
        dataset_id=path.stem,
        file_id=path.name,
        file_format="netcdf",
        data_modality="multidimensional",
        fields=fields,
        metadata={
            "backend": "scipy.io.netcdf_file",
            "dimensions": dimensions,
            "file_attributes": file_attrs,
            "cf_analysis": {"variables": analyses},
            "extraction_errors": extraction_errors,
        },
    ), samples_by_path


def _extract_hdf5_netcdf(path: Path, sample_limit: int) -> tuple[DatasetSchema, Dict[str, List[Any]]]:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5-backed NetCDF extraction")
    fields: List[FieldSchema] = []
    groups: List[Dict[str, Any]] = []
    analyses: List[Dict[str, Any]] = []
    samples_by_path: Dict[str, List[Any]] = {}
    extraction_errors: List[Dict[str, str]] = []
    with h5py.File(path, "r") as handle:
        file_attrs = _attrs(handle.attrs)
        dataset_nodes: Dict[str, Any] = {}

        def collect(name: str, node: Any) -> None:
            path_name = f"/{name}" if name else "/"
            if isinstance(node, h5py.Group):
                groups.append(
                    {
                        "path": path_name,
                        "kind": "group",
                        "attributes": _attrs(node.attrs),
                        "source_evidence": [
                            _evidence(path, path_name, "netcdf_group", "HDF5-backed NetCDF group")
                        ],
                    }
                )
            else:
                dataset_nodes[path_name] = node

        handle.visititems(collect)
        dimension_nodes = {
            path_name: node
            for path_name, node in dataset_nodes.items()
            if _normalize_value(node.attrs.get("CLASS")) == "DIMENSION_SCALE"
            or "_Netcdf4Dimid" in node.attrs
        }
        dimension_names = {path_name.rsplit("/", 1)[-1] for path_name in dimension_nodes}
        auxiliary_names = {
            token
            for node in dataset_nodes.values()
            for token in str(_normalize_value(node.attrs.get("coordinates", ""))).split()
            if token
        }
        bounds_names = {
            str(_normalize_value(node.attrs.get("bounds")))
            for node in dataset_nodes.values()
            if node.attrs.get("bounds") is not None
        }
        dimensions: Dict[str, Dict[str, Any]] = {
            path_name.rsplit("/", 1)[-1]: {
                "size": node.shape[0],
                "unlimited": False,
                "source_evidence": [
                    _evidence(
                        path,
                        path_name,
                        "netcdf_dimension",
                        f"dimension scale size={node.shape[0]}; unlimited=False",
                    )
                ],
            }
            for path_name, node in dimension_nodes.items()
            if node.ndim == 1
        }
        for field_path, node in dataset_nodes.items():
            name = field_path.rsplit("/", 1)[-1]
            attrs = _attrs(node.attrs)
            dimension_labels = []
            for index, dimension in enumerate(node.dims):
                scale_names = list(dimension.keys())
                label = dimension.label
                dimension_labels.append(
                    scale_names[0].rsplit("/", 1)[-1]
                    if scale_names
                    else label or f"dim_{index}"
                )
            if field_path in dimension_nodes:
                dimension_labels = [name]
            try:
                values = _sample_values(node[...], sample_limit)
                samples_by_path[field_path] = values
                field, analysis = _build_field(
                    path,
                    field_path,
                    name,
                    node.dtype,
                    list(node.shape),
                    dimension_labels,
                    attrs,
                    values,
                    dimension_names,
                    auxiliary_names,
                    bounds_names,
                )
                fields.append(field)
                analyses.append(analysis)
            except Exception as exc:
                extraction_errors.append({"path": field_path, "error": str(exc)})

    return DatasetSchema(
        dataset_id=path.stem,
        file_id=path.name,
        file_format="netcdf",
        data_modality="multidimensional",
        fields=fields,
        groups=groups,
        metadata={
            "backend": "h5py_netcdf4_compatible",
            "dimensions": dimensions,
            "file_attributes": file_attrs,
            "cf_analysis": {"variables": analyses},
            "extraction_errors": extraction_errors,
        },
    ), samples_by_path


def _attach_cf_temporal(schema: DatasetSchema, samples_by_path: Dict[str, List[Any]]) -> None:
    fields_by_path = {field.field_path: field for field in schema.fields}
    analysis_by_path = {
        item["field_path"]: item
        for item in schema.metadata["cf_analysis"]["variables"]
    }
    temporal_samples: Dict[str, List[str]] = {
        field.field_name: [str(value) for value in samples_by_path.get(field.field_path, [])]
        for field in schema.fields
    }
    calendar_claims: Dict[str, Dict[str, Any]] = {}
    temporal_candidate_paths: List[str] = []
    for field_path, field in fields_by_path.items():
        attrs = analysis_by_path[field_path]["attributes"]
        if analysis_by_path[field_path].get("is_bounds_variable"):
            continue
        units = str(attrs.get("units", ""))
        axis = str(attrs.get("axis", "")).upper()
        standard_name = str(attrs.get("standard_name", "")).lower()
        if not (CF_TIME_UNIT_RE.match(units) or axis == "T" or standard_name == "time"):
            continue
        temporal_candidate_paths.append(field_path)
        calendar = str(attrs.get("calendar", "standard"))
        decoded, calendar_claim = _decode_cf_time(samples_by_path.get(field_path, []), units, calendar)
        temporal_samples[field.field_name] = decoded
        calendar_claims[field_path] = calendar_claim

    analysis = analyze_temporal_semantics(temporal_samples, schema.fields) if temporal_candidate_paths else {
        "time_series": {},
        "temporal_analysis": {
            "analysis_version": "phase12_temporal_v1",
            "candidates": [],
            "selected_candidate": None,
            "property_claims": {},
            "validator_results": {"candidate_count": 0, "eligible_candidate_count": 0, "measurement_count": 0, "selection_abstained": True},
            "issues": [],
        },
    }
    cf_temporal = {
        "candidate_paths": temporal_candidate_paths,
        "calendar_claims": calendar_claims,
        "decoded_with_temporal_semantics": any(temporal_samples.get(fields_by_path[path].field_name) for path in temporal_candidate_paths),
        "selection": analysis["temporal_analysis"],
    }
    schema.metadata["cf_analysis"]["temporal"] = cf_temporal
    if analysis["time_series"]:
        schema.metadata["time_series"] = analysis["time_series"]
        schema.metadata["temporal_analysis"] = analysis["temporal_analysis"]
        schema.data_modality = "time_series"
    elif len(temporal_candidate_paths) > 1:
        schema.metadata["cf_analysis"]["issues"] = [
            {
                "code": "ambiguous_cf_time_candidates",
                "severity": "warning",
                "candidate_paths": temporal_candidate_paths,
            }
        ]


def extract_netcdf_schema(path: str, sample_limit: int = 200) -> DatasetSchema:
    if np is None:
        raise RuntimeError("numpy is required for NetCDF extraction but is not installed in this environment.")

    netcdf_path = Path(path)
    with netcdf_path.open("rb") as handle:
        prefix = handle.read(8)
    if prefix.startswith(HDF5_MAGIC):
        schema, samples = _extract_hdf5_netcdf(netcdf_path, sample_limit)
    else:
        schema, samples = _extract_classic(netcdf_path, sample_limit)
    _attach_cf_temporal(schema, samples)
    return attach_dataset_profile(schema)
