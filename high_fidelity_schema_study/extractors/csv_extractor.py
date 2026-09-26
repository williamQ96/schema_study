from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..models import DatasetSchema, EvidenceRecord, FieldSchema
from ..deterministic_profile import attach_dataset_profile
from ..temporal_semantics import analyze_temporal_semantics, parse_datetime
from ..unit_normalization import normalize_unit_claim


ML_GPU_TRAINING_SEMANTIC_MAPPING = {
    "name": "training_run_name",
    "samples": "sample_count",
    "input_dim_w": "input_width",
    "input_dim_h": "input_height",
    "input_dim_c": "input_channel_count",
    "output_dim": "output_dimension",
    "optimizer": "optimizer",
    "epochs": "epoch_count",
    "batch": "batch_size",
    "learn_rate": "learning_rate",
    "tf_version": "tensorflow_version",
    "cuda_version": "cuda_version",
    "batch_time": "batch_duration",
    "epoch_time": "epoch_duration",
    "fit_time": "fit_duration",
    "npz_path": "model_artifact_path",
    "gpu_make": "gpu_vendor",
    "gpu_name": "gpu_model",
    "gpu_arch": "gpu_architecture",
    "gpu_cc": "gpu_compute_capability",
    "gpu_core_count": "gpu_core_count",
    "gpu_sm_count": "gpu_sm_count",
    "gpu_memory_size": "gpu_memory_size",
    "gpu_memory_type": "gpu_memory_type",
    "gpu_memory_bw": "gpu_memory_bandwidth",
    "gpu_tensor_core_count": "gpu_tensor_core_count",
    "max_memory_util": "gpu_memory_utilization",
    "avg_memory_util": "gpu_memory_utilization",
    "max_gpu_util": "gpu_utilization",
    "avg_gpu_util": "gpu_utilization",
    "max_gpu_temp": "gpu_temperature",
    "avg_gpu_temp": "gpu_temperature",
}

TEMPERATURE_UNIT_SUFFIXES = {
    "_c": "Celsius",
    "_f": "Fahrenheit",
    "_k": "Kelvin",
}


def _canonicalize_csv_headers(raw_headers: List[str]) -> tuple[List[str], List[Dict[str, object]]]:
    fieldnames: List[str] = []
    mappings: List[Dict[str, object]] = []
    occurrences: Dict[str, int] = {}
    used = set()
    for index, raw_header in enumerate(raw_headers):
        if raw_header.strip():
            base = raw_header
            reason = "preserved"
        else:
            base = f"__unnamed_column_{index + 1}"
            reason = "blank_header"
        occurrence = occurrences.get(base, 0) + 1
        occurrences[base] = occurrence
        candidate = base if occurrence == 1 else f"{base}__duplicate_{occurrence}"
        collision = occurrence
        while candidate in used:
            collision += 1
            candidate = f"{base}__duplicate_{collision}"
            reason = "duplicate_header"
        if occurrence > 1:
            reason = "duplicate_header"
        used.add(candidate)
        fieldnames.append(candidate)
        mappings.append(
            {
                "column_index": index,
                "raw_header": raw_header,
                "field_path": candidate,
                "canonicalization_reason": reason,
            }
        )
    return fieldnames, mappings


def _name_tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _has_temperature_name_token(column_name: str) -> bool:
    return bool(_name_tokens(column_name) & {"temp", "temperature"})


def _has_environmental_temperature_context(column_name: str, fieldnames: List[str]) -> bool:
    lowered = column_name.lower()
    tokens = _name_tokens(column_name)
    fieldname_set = {name.lower() for name in fieldnames}
    context_tokens = set().union(*(_name_tokens(name) for name in fieldnames)) if fieldnames else set()
    return (
        bool(tokens & {"air", "ambient", "weather", "surface"})
        or "air_temp" in lowered
        or "temperature_air" in lowered
        or bool(context_tokens & {"weather", "meteorology", "meteo", "humidity", "precip", "rain", "wind"})
        or bool(fieldname_set & {"station_id", "station_name"})
    )


def _is_gpu_temperature_name(column_name: str) -> bool:
    tokens = _name_tokens(column_name)
    return "gpu" in tokens and bool(tokens & {"temp", "temperature"})


def _has_ml_gpu_training_context(fieldnames: List[str]) -> bool:
    fieldname_set = {name.lower() for name in fieldnames}
    return {"optimizer", "learn_rate", "gpu_name", "cuda_version"} <= fieldname_set


def _looks_like_int(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith(("+", "-")):
        text = text[1:]
    return text.isdigit()


def _has_leading_zero(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    signless = text[1:] if text.startswith(("+", "-")) else text
    return len(signless) > 1 and signless.startswith("0") and signless.isdigit()


def _looks_like_float(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return any(char in text for char in ".eE")


def _semantic_type_from_name(column_name: str, fieldnames: List[str]) -> str:
    lowered = column_name.lower()
    fieldname_set = {name.lower() for name in fieldnames}

    exact_mapping = {
        "station_id": "station_identifier",
        "station_name": "station_name",
        "sample_id": "sample_identifier",
        "sensor_id": "sensor_identifier",
        "buoy_id": "buoy_identifier",
        "site_code": "site_identifier",
        "collection_date": "collection_date",
        "ts_utc": "observation_time",
        "event_time": "observation_time",
        "dt_obs": "observation_time",
        "power_kw": "power",
        "voltage_v": "voltage",
        "status": "operational_status",
        "ph": "acidity_ph",
        "do_mg_l": "dissolved_oxygen",
        "turbidity_ntu": "turbidity",
        "notes": "free_text_note",
        "qc_flag": "quality_flag",
        "flag": "quality_flag",
        "rec": "record_identifier",
        "zc": "postal_zone_code",
        "device": "device_identifier",
        "salinity_psu": "salinity",
    }
    if _has_ml_gpu_training_context(fieldnames) and lowered in ML_GPU_TRAINING_SEMANTIC_MAPPING:
        return ML_GPU_TRAINING_SEMANTIC_MAPPING[lowered]
    if lowered in exact_mapping:
        return exact_mapping[lowered]
    if lowered.endswith("_id") or lowered == "id" or "identifier" in lowered:
        return lowered.removesuffix("_id") + "_identifier" if lowered.endswith("_id") and lowered != "id" else "identifier"
    if "timestamp" in lowered or lowered.startswith("time") or lowered.endswith("date") or lowered in {"ts_utc", "dt_obs", "event_time"}:
        return "observation_time"
    if lowered in {"lat", "latitude", "latitude_deg"}:
        return "latitude"
    if lowered in {"lon", "longitude", "longitude_deg"}:
        return "longitude"
    if lowered in {"lat", "latitude", "lon", "longitude"}:
        return "coordinate"
    if "elevation" in lowered:
        return "elevation"
    if "precip" in lowered:
        return "precipitation"
    if "humidity" in lowered:
        return "relative_humidity"
    if "pressure" in lowered:
        return "surface_pressure"
    if "wind_speed" in lowered:
        return "wind_speed"
    if _is_gpu_temperature_name(column_name):
        return "gpu_temperature"
    if _has_temperature_name_token(column_name):
        aquatic_context = any(token in fieldname_set for token in {"salinity_psu", "buoy_id"}) or "water" in lowered
        if aquatic_context:
            return "water_temperature"
        if _has_environmental_temperature_context(column_name, fieldnames):
            return "air_temperature"
        return "unknown"
    if any(token in lowered for token in ("value", "val")):
        return "unknown"
    return "unknown"


def _unit_from_name(column_name: str, semantic_type: str = "unknown") -> Optional[str]:
    lowered = column_name.lower()
    for suffix, unit in TEMPERATURE_UNIT_SUFFIXES.items():
        if lowered.endswith(suffix):
            if semantic_type in {"air_temperature", "water_temperature", "gpu_temperature"} or _has_temperature_name_token(column_name):
                return unit
            return None
    suffix_mapping = {
        "_mm": "millimeter",
        "_m": "meter",
        "_kg": "kilogram",
        "_deg": "degree",
        "_pct": "percent",
        "_kw": "kilowatt",
        "_v": "volt",
        "_ntu": "NTU",
        "_psu": "PSU",
        "_mg_l": "milligram_per_liter",
        "_m_s": "meter_per_second",
    }
    for suffix, unit in suffix_mapping.items():
        if lowered.endswith(suffix):
            return unit
    return None


def _logical_type_from_field(fieldname: str, physical_type: str, semantic_type: str) -> str:
    lowered = fieldname.lower()
    if semantic_type.endswith("_identifier") or semantic_type == "identifier":
        return "identifier"
    if semantic_type in {"latitude", "longitude", "elevation", "depth"}:
        return "coordinate"
    if semantic_type in {
        "precipitation",
        "relative_humidity",
        "surface_pressure",
        "wind_speed",
        "air_temperature",
        "water_temperature",
        "gpu_temperature",
        "power",
        "voltage",
        "acidity_ph",
        "dissolved_oxygen",
        "turbidity",
        "salinity",
        "batch_duration",
        "epoch_duration",
        "fit_duration",
        "gpu_memory_size",
        "gpu_memory_bandwidth",
        "gpu_memory_utilization",
        "gpu_utilization",
    }:
        return "measurement"
    if semantic_type in {
        "station_name",
        "operational_status",
        "quality_flag",
        "free_text_note",
        "training_run_name",
        "optimizer",
        "tensorflow_version",
        "cuda_version",
        "model_artifact_path",
        "gpu_vendor",
        "gpu_model",
        "gpu_architecture",
        "gpu_memory_type",
    }:
        return "label"
    if semantic_type in {
        "collection_date",
        "observation_time",
        "sample_count",
        "input_width",
        "input_height",
        "input_channel_count",
        "output_dimension",
        "epoch_count",
        "batch_size",
        "learning_rate",
        "gpu_compute_capability",
        "gpu_core_count",
        "gpu_sm_count",
        "gpu_tensor_core_count",
    }:
        return "attribute"
    if "flag" in lowered or "status" in lowered or "note" in lowered or lowered.endswith("_name"):
        return "label"
    if physical_type in {"int", "float"} and any(token in lowered for token in ("temp", "precip", "pressure", "power", "voltage", "humidity", "wind", "rain", "ph", "turbidity")):
        return "measurement"
    return "unknown"


def _conservative_physical_type(values: List[str]) -> str:
    non_empty = [value for value in values if value.strip()]
    if not non_empty:
        return "string"
    if any(_has_leading_zero(value) for value in non_empty):
        return "string"
    if all(parse_datetime(value) is not None for value in non_empty):
        return "datetime"
    if all(_looks_like_int(value) for value in non_empty):
        return "int"
    if all(_looks_like_int(value) or _looks_like_float(value) for value in non_empty):
        return "float"
    return "string"


def _value_range(physical_type: str, values: List[str]) -> Optional[List[float]]:
    if physical_type not in {"int", "float"}:
        return None
    numeric_values = [float(value) for value in values if value.strip()]
    if not numeric_values:
        return None
    return [min(numeric_values), max(numeric_values)]


def extract_csv_schema(path: str, sample_limit: int = 200) -> DatasetSchema:
    csv_path = Path(path)
    delimiter = "\t" if csv_path.suffix.lower() == ".tsv" else ","
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file has no header row: {csv_path}")
        fieldnames, header_mapping = _canonicalize_csv_headers(raw_headers)
        column_samples: Dict[str, List[str]] = {fieldname: [] for fieldname in fieldnames}
        sampled_rows = 0
        row_width_conflict_count = 0
        for row in reader:
            if sampled_rows >= sample_limit:
                break
            sampled_rows += 1
            if len(row) != len(fieldnames):
                row_width_conflict_count += 1
            for index, fieldname in enumerate(fieldnames):
                value = row[index] if index < len(row) else ""
                column_samples[fieldname].append(value.strip())

    fields: List[FieldSchema] = []
    for fieldname, header in zip(fieldnames, header_mapping):
        samples = column_samples[fieldname]
        physical_type = _conservative_physical_type(samples)
        raw_header = str(header["raw_header"])
        semantic_input = raw_header if raw_header.strip() else fieldname
        semantic_type = _semantic_type_from_name(semantic_input, fieldnames)
        logical_type = _logical_type_from_field(
            semantic_input, physical_type, semantic_type
        )
        unit = _unit_from_name(semantic_input, semantic_type)
        nullable = any(not value.strip() for value in samples)
        non_empty = [value for value in samples if value.strip()]
        unique_ratio = round(len(set(non_empty)) / len(non_empty), 4) if non_empty else None
        evidence = [
            EvidenceRecord(
                tier="structural",
                evidence_type="csv_header",
                source=str(csv_path),
                detail=(
                    f"column_index={header['column_index']}; "
                    f"raw_header={raw_header!r}; field_path={fieldname!r}"
                ),
                confidence=1.0,
            ),
            EvidenceRecord(
                tier="statistical",
                evidence_type="sample_rows",
                source=str(csv_path),
                detail=f"sampled_rows={sampled_rows}",
                confidence=0.9,
            ),
        ]
        if unit is not None:
            evidence.append(
                EvidenceRecord(
                    tier="structural",
                    evidence_type="column_name_unit_hint",
                    source=fieldname,
                    detail=f"inferred unit from column suffix: {unit}",
                    confidence=0.82,
                )
            )
        if semantic_type == "identifier" and physical_type == "string":
            evidence.append(
                EvidenceRecord(
                    tier="structural",
                    evidence_type="identifier_name_pattern",
                    source=fieldname,
                    detail="column name suggests identifier semantics",
                    confidence=0.9,
                )
            )
        unit_normalization = normalize_unit_claim(
            unit,
            [item.evidence_type for item in evidence],
        )

        field = FieldSchema(
            field_name=fieldname,
            field_path=fieldname,
            physical_type=physical_type,
            logical_type=logical_type,
            semantic_type=semantic_type,
            nullable=nullable,
            unique_ratio=unique_ratio,
            unit=unit,
            unit_normalization=unit_normalization,
            example_values=list(dict.fromkeys(non_empty[:3])),
            missing_count=sum(1 for value in samples if not value.strip()),
            value_range=_value_range(physical_type, samples),
            source_evidence=evidence,
            confidence=1.0 if physical_type in {"datetime", "int", "float"} else 0.9,
            uncertainty_reason=(
                "conservative fallback to string to avoid over-claiming"
                if physical_type == "string" and non_empty
                else None
            ),
            extraction_method="csv_conservative_profiler",
        )
        fields.append(field)

    dataset_schema = DatasetSchema(
        dataset_id=csv_path.stem,
        file_id=csv_path.name,
        file_format="csv",
        data_modality="tabular",
        fields=fields,
        metadata={
            "sampled_rows": sampled_rows,
            "column_count": len(fieldnames),
            "csv_header_mapping": header_mapping,
            "row_width_conflict_count": row_width_conflict_count,
        },
    )

    temporal_result = analyze_temporal_semantics(column_samples, fields)
    dataset_schema.metadata["temporal_analysis"] = temporal_result["temporal_analysis"]
    time_series_metadata = temporal_result["time_series"]
    if time_series_metadata:
        dataset_schema.data_modality = "time_series"
        dataset_schema.metadata["time_series"] = time_series_metadata

    return attach_dataset_profile(dataset_schema)
