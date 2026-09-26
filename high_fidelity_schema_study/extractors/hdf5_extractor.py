from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..models import DatasetSchema, EvidenceRecord, FieldSchema
from ..deterministic_profile import attach_dataset_profile
from ..unit_normalization import normalize_unit_claim

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None


def _normalize_attr_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_dataset_dtype(node: Any) -> str:
    string_info = h5py.check_string_dtype(node.dtype)
    if string_info is not None:
        return "string"
    return str(node.dtype)


def _semantic_type_from_hdf5(path_name: str, attrs: Dict[str, Any]) -> str:
    lowered_path = path_name.lower()
    long_name = str(attrs.get("long_name", "")).lower()
    description = str(attrs.get("description", "")).lower()
    combined = " ".join([lowered_path, long_name, description])

    if "station_id" in lowered_path or "station_code" in lowered_path:
        return "station_identifier"
    if "timestamp" in lowered_path:
        return "observation_time"
    if lowered_path.endswith("/time") and "hours since" in str(attrs.get("units", "")).lower():
        return "forecast_hour"
    if "air temperature" in combined:
        return "air_temperature"
    if "seawater temperature" in combined or ("temp" in lowered_path and "salinity" in combined):
        return "water_temperature"
    if lowered_path.endswith("/temp") and "station_" in lowered_path:
        return "air_temperature"
    if lowered_path.endswith("/temperature"):
        return "air_temperature"
    if "humidity" in combined:
        return "relative_humidity"
    if "pressure" in combined:
        return "surface_pressure"
    if "salinity" in combined:
        return "salinity"
    if "depth" in combined or lowered_path.endswith("/depth_m"):
        return "depth"
    if "quality control flag" in combined or lowered_path.endswith("/qc_flag"):
        return "quality_flag"
    return "unknown"


def _logical_type_from_hdf5(path_name: str, dtype: str, semantic_type: str) -> str:
    lowered_path = path_name.lower()
    if semantic_type.endswith("_identifier"):
        return "identifier"
    if semantic_type in {"observation_time", "forecast_hour"}:
        return "time_axis"
    if semantic_type in {"depth", "latitude", "longitude"}:
        return "coordinate"
    if semantic_type in {"air_temperature", "water_temperature", "relative_humidity", "surface_pressure", "salinity"}:
        return "measurement"
    if semantic_type == "quality_flag":
        return "label"
    if lowered_path.endswith("/platform") and dtype == "string":
        return "label"
    return "unknown"


def extract_hdf5_schema(path: str) -> DatasetSchema:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 extraction but is not installed in this environment.")

    hdf5_path = Path(path)
    groups: List[Dict[str, Any]] = []
    fields: List[FieldSchema] = []
    extraction_errors: List[Dict[str, str]] = []

    with h5py.File(hdf5_path, "r") as handle:
        def visit_group(group: Any, prefix: str = "") -> None:
            for key in group.keys():
                child_name = f"{prefix}/{key}" if prefix else key
                path_name = f"/{child_name}" if child_name else "/"
                try:
                    node = group[key]
                except Exception as exc:
                    extraction_errors.append(
                        {
                            "path": path_name,
                            "error": str(exc),
                        }
                    )
                    continue

                attrs = {attr_key: _normalize_attr_value(value) for attr_key, value in node.attrs.items()}
                if isinstance(node, h5py.Group):
                    groups.append(
                        {
                            "path": path_name,
                            "kind": "group",
                            "attributes": attrs,
                        }
                    )
                    visit_group(node, child_name)
                    continue

                dtype = _normalize_dataset_dtype(node)
                shape = list(node.shape)
                unit = attrs.get("unit") or attrs.get("units")
                description = attrs.get("long_name") or attrs.get("description")
                semantic_type = _semantic_type_from_hdf5(path_name, attrs)
                logical_type = _logical_type_from_hdf5(path_name, dtype, semantic_type)
                evidence = [
                    EvidenceRecord(
                        tier="explicit_metadata",
                        evidence_type="hdf5_dataset_path",
                        source=str(hdf5_path),
                        detail=path_name,
                        confidence=1.0,
                    )
                ]
                if unit is not None:
                    evidence.append(
                        EvidenceRecord(
                            tier="explicit_metadata",
                            evidence_type="hdf5_attribute",
                            source=path_name,
                            detail=f"unit={unit}",
                            confidence=0.99,
                        )
                    )
                unit_normalization = normalize_unit_claim(
                    str(unit) if unit is not None else None,
                    [item.evidence_type for item in evidence],
                )

                fields.append(
                    FieldSchema(
                        field_name=path_name.rsplit("/", 1)[-1] if path_name != "/" else "/",
                        field_path=path_name,
                        physical_type=dtype,
                        logical_type=logical_type,
                        semantic_type=semantic_type,
                        shape=shape,
                        unit=unit,
                        unit_normalization=unit_normalization,
                        description=str(description) if description is not None else None,
                        source_evidence=evidence,
                        confidence=1.0,
                        extraction_method="h5py_structure_traversal",
                    )
                )

        visit_group(handle)

    schema = DatasetSchema(
        dataset_id=hdf5_path.stem,
        file_id=hdf5_path.name,
        file_format="hdf5",
        data_modality="hierarchical",
        fields=fields,
        groups=groups,
        metadata={
            "field_count": len(fields),
            "group_count": len(groups),
            "extraction_errors": extraction_errors,
        },
    )
    return attach_dataset_profile(schema)
