"""Backward-compatible adapter over the Phase 12 temporal subsystem."""

from __future__ import annotations

from typing import Dict, List

from ..models import FieldSchema
from ..temporal_semantics import (
    analyze_temporal_semantics,
    infer_frequency,
    is_strong_time_field_name,
    looks_like_datetime_series,
    parse_datetime,
)


def infer_time_series_metadata(
    column_samples: Dict[str, List[str]],
    fields: List[FieldSchema],
) -> Dict[str, object]:
    return analyze_temporal_semantics(column_samples, fields)["time_series"]


__all__ = [
    "analyze_temporal_semantics",
    "infer_frequency",
    "infer_time_series_metadata",
    "is_strong_time_field_name",
    "looks_like_datetime_series",
    "parse_datetime",
]
