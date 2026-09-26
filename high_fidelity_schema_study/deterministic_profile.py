"""Deterministic profiling helpers for schema artifacts.

The profiles in this module are intentionally conservative. They summarize
signals already visible in extracted fields instead of making new semantic
claims.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .models import DatasetSchema, FieldSchema


GENERIC_SEMANTIC_TYPES = {"unknown", "quality_flag", "free_text_note"}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _field_missing_profile(field: FieldSchema, sampled_rows: int | None) -> Dict[str, Any]:
    missing_ratio = _safe_ratio(field.missing_count, sampled_rows or 0)
    return {
        "field_path": field.field_path,
        "missing_count": field.missing_count,
        "missing_ratio": missing_ratio,
        "nullable": field.nullable,
        "basis": "sampled_rows" if sampled_rows is not None else "field_metadata",
    }


def _identifier_quality(field: FieldSchema, sampled_rows: int | None) -> Dict[str, Any]:
    non_missing_rows = None
    duplicate_count = None
    if sampled_rows is not None:
        non_missing_rows = sampled_rows - field.missing_count
        if field.unique_ratio is not None and non_missing_rows >= 0:
            duplicate_count = max(non_missing_rows - round(field.unique_ratio * non_missing_rows), 0)

    unique_ratio = field.unique_ratio
    missing_ratio = _safe_ratio(field.missing_count, sampled_rows or 0)
    if unique_ratio is None:
        quality = "not_profiled"
    elif missing_ratio is not None and missing_ratio > 0:
        quality = "incomplete_identifier"
    elif unique_ratio >= 0.98:
        quality = "unique_identifier"
    elif unique_ratio >= 0.5:
        quality = "group_identifier"
    else:
        quality = "low_cardinality_identifier"

    return {
        "field_path": field.field_path,
        "semantic_type": field.semantic_type,
        "unique_ratio": unique_ratio,
        "missing_ratio": missing_ratio,
        "non_missing_rows": non_missing_rows,
        "duplicate_count": duplicate_count,
        "quality": quality,
    }


def build_dataset_profile(schema: DatasetSchema) -> Dict[str, Any]:
    sampled_rows = schema.metadata.get("sampled_rows")
    if not isinstance(sampled_rows, int):
        sampled_rows = None

    missing_fields = [_field_missing_profile(field, sampled_rows) for field in schema.fields]
    fields_with_missing = [
        field_profile
        for field_profile in missing_fields
        if field_profile["missing_count"] > 0
    ]
    total_missing_cells = sum(field.missing_count for field in schema.fields)

    identifier_fields = [
        field
        for field in schema.fields
        if field.logical_type == "identifier" or field.semantic_type.endswith("_identifier")
    ]
    identifier_profiles = [_identifier_quality(field, sampled_rows) for field in identifier_fields]
    best_identifier_fields = [
        profile["field_path"]
        for profile in identifier_profiles
        if profile["quality"] in {"unique_identifier", "group_identifier"}
    ]

    return {
        "missingness": {
            "basis": "sampled_rows" if sampled_rows is not None else "field_metadata",
            "sampled_rows": sampled_rows,
            "field_count": len(schema.fields),
            "fields_with_missing_count": len(fields_with_missing),
            "total_missing_cells": total_missing_cells,
            "fields": missing_fields,
        },
        "identifier_quality": {
            "candidate_count": len(identifier_profiles),
            "best_identifier_fields": best_identifier_fields,
            "fields": identifier_profiles,
        },
    }


def attach_dataset_profile(schema: DatasetSchema) -> DatasetSchema:
    schema.metadata["deterministic_profile"] = build_dataset_profile(schema)
    return schema


def _fields_by_semantic(schema: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for field in schema.get("fields", []):
        semantic_type = field.get("semantic_type", "unknown")
        if semantic_type in GENERIC_SEMANTIC_TYPES:
            continue
        grouped.setdefault(semantic_type, []).append(field)
    return grouped


def _relationship_kind(fields_a: Sequence[Mapping[str, Any]], fields_b: Sequence[Mapping[str, Any]]) -> str:
    left_logical_types = {field.get("logical_type") for field in fields_a}
    right_logical_types = {field.get("logical_type") for field in fields_b}
    logical_types = left_logical_types | right_logical_types
    if left_logical_types == {"identifier"} and right_logical_types == {"identifier"}:
        return "shared_identifier_semantic"
    if left_logical_types == {"time_axis"} and right_logical_types == {"time_axis"}:
        return "shared_time_axis_semantic"
    if "time_axis" in logical_types or any(
        field.get("semantic_type") == "observation_time"
        for field in list(fields_a) + list(fields_b)
    ):
        return "shared_temporal_semantic"
    if left_logical_types == {"coordinate"} and right_logical_types == {"coordinate"}:
        return "shared_coordinate_semantic"
    if left_logical_types == {"measurement"} and right_logical_types == {"measurement"}:
        return "shared_measurement_semantic"
    return "shared_field_semantic"


def _relationship_confidence(kind: str) -> float:
    if kind == "shared_identifier_semantic":
        return 0.75
    if kind in {"shared_time_axis_semantic", "shared_coordinate_semantic"}:
        return 0.65
    if kind in {"shared_temporal_semantic", "shared_measurement_semantic"}:
        return 0.55
    return 0.45


def infer_multi_file_relationships(schemas: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    schema_list = list(schemas)
    relationships: List[Dict[str, Any]] = []

    for left, right in combinations(schema_list, 2):
        left_id = left.get("study_dataset_id") or left.get("dataset_id")
        right_id = right.get("study_dataset_id") or right.get("dataset_id")
        left_by_semantic = _fields_by_semantic(left)
        right_by_semantic = _fields_by_semantic(right)
        shared_semantics = sorted(set(left_by_semantic) & set(right_by_semantic))
        for semantic_type in shared_semantics:
            left_fields = left_by_semantic[semantic_type]
            right_fields = right_by_semantic[semantic_type]
            kind = _relationship_kind(left_fields, right_fields)
            relationships.append(
                {
                    "left_dataset_id": left_id,
                    "right_dataset_id": right_id,
                    "relationship_type": kind,
                    "semantic_type": semantic_type,
                    "left_fields": [field["field_path"] for field in left_fields],
                    "right_fields": [field["field_path"] for field in right_fields],
                    "confidence": _relationship_confidence(kind),
                    "evidence": "shared deterministic logical/semantic field labels",
                }
            )

    relationship_counts: Dict[str, int] = {}
    for relationship in relationships:
        kind = relationship["relationship_type"]
        relationship_counts[kind] = relationship_counts.get(kind, 0) + 1

    return {
        "profile_type": "internal_multi_file_relationships",
        "dataset_count": len(schema_list),
        "relationship_count": len(relationships),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "relationships": relationships,
        "notes": [
            "Relationships are deterministic candidates, not asserted joins.",
            "Generic semantic types such as unknown and quality_flag are excluded.",
            "Confidence reflects label strength only; value-level join validation is future work.",
        ],
    }
