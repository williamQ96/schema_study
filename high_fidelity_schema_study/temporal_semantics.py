from __future__ import annotations

from collections import Counter
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Optional

from .models import FieldSchema


DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)

TIMEZONE_OFFSET_RE = re.compile(r"([+-])(\d{2}):?(\d{2})$")


def is_strong_time_field_name(field_name: str) -> bool:
    lowered = field_name.lower()
    return any(
        token in lowered
        for token in ("timestamp", "event_time", "datetime", "time_", "_time", "ts_", "_ts")
    ) or lowered in {"time", "timestamp", "event_time", "ts", "ts_utc", "date"}


def parse_datetime(value: str) -> Optional[datetime]:
    text = value.strip()
    if not text:
        return None
    iso_text = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        pass
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def looks_like_datetime_series(values: Iterable[str]) -> bool:
    non_empty = [value for value in values if value.strip()]
    return bool(non_empty) and all(parse_datetime(value) is not None for value in non_empty)


def infer_frequency(deltas_seconds: List[int]) -> str:
    if not deltas_seconds:
        return "unknown"
    dominant_delta, count = Counter(deltas_seconds).most_common(1)[0]
    if count / len(deltas_seconds) < 0.75:
        return "mixed"
    mapping = {
        1: "1 second",
        60: "1 minute",
        300: "5 minutes",
        360: "6 minutes",
        600: "10 minutes",
        900: "15 minutes",
        1200: "20 minutes",
        1800: "30 minutes",
        3600: "1 hour",
        86400: "1 day",
    }
    return mapping.get(dominant_delta, f"{dominant_delta} seconds")


def _claim(value: Any, state: str, reason_code: str, evidence_refs: List[str]) -> Dict[str, Any]:
    return {
        "value": value,
        "state": state,
        "reason_code": reason_code,
        "evidence_refs": evidence_refs,
    }


def _timezone_claim(values: List[str]) -> Dict[str, Any]:
    non_empty = [value.strip() for value in values if value.strip()]
    evidence = ["sample_values:timezone_markers"]
    if not non_empty:
        return _claim("unknown", "unknown", "sampling_insufficient", evidence)

    aware_offsets: List[str] = []
    naive_count = 0
    for value in non_empty:
        if value.endswith(("Z", "z")):
            aware_offsets.append("+00:00")
            continue
        match = TIMEZONE_OFFSET_RE.search(value)
        if match:
            sign, hours, minutes = match.groups()
            aware_offsets.append(f"{sign}{hours}:{minutes}")
        else:
            naive_count += 1

    if aware_offsets and naive_count:
        return _claim(
            "unknown",
            "conflicted",
            "mixed_naive_and_aware_datetimes",
            evidence,
        )
    if not aware_offsets:
        return _claim(
            "unknown",
            "unknown",
            "timezone_not_encoded_in_values",
            evidence,
        )

    unique_offsets = sorted(set(aware_offsets))
    if len(unique_offsets) > 1:
        return _claim(
            "unknown",
            "conflicted",
            "conflicting_timezone_offsets",
            evidence,
        )
    timezone = "UTC" if unique_offsets[0] == "+00:00" else unique_offsets[0]
    return _claim(timezone, "supported", "explicit_timezone_offset", evidence)


def _time_properties(parsed_values: List[datetime]) -> Dict[str, Any]:
    if len(parsed_values) < 2:
        return {
            "frequency": "unknown",
            "regularity": "unknown",
            "missing_intervals": None,
            "deltas_seconds": [],
        }

    awareness = {value.tzinfo is not None for value in parsed_values}
    if len(awareness) > 1:
        return {
            "frequency": "unknown",
            "regularity": "unknown",
            "missing_intervals": None,
            "deltas_seconds": [],
        }

    ordered = sorted(parsed_values)
    deltas = [
        int((current - previous).total_seconds())
        for previous, current in zip(ordered, ordered[1:])
        if current >= previous
    ]
    frequency = infer_frequency(deltas)
    regularity = "unknown"
    missing_intervals = None
    if deltas:
        _, count = Counter(deltas).most_common(1)[0]
        dominance = count / len(deltas)
        if frequency == "mixed":
            regularity = "irregular"
        elif dominance >= 0.95:
            regularity = "regular"
        elif dominance >= 0.75:
            regularity = "mostly_regular"
        else:
            regularity = "irregular"

        if frequency not in {"unknown", "mixed"}:
            target_delta = Counter(deltas).most_common(1)[0][0]
            if target_delta > 0:
                expected_steps = int((ordered[-1] - ordered[0]).total_seconds() / target_delta) + 1
                missing_intervals = max(expected_steps - len(ordered), 0)
            else:
                # Repeated timestamps often represent several observations at
                # one time. Without a grouping key, cadence is not defined.
                frequency = "unknown"
                regularity = "unknown"

    return {
        "frequency": frequency,
        "regularity": regularity,
        "missing_intervals": missing_intervals,
        "deltas_seconds": deltas,
    }


def _measurement_candidates(fields: List[FieldSchema], excluded_names: set[str]) -> List[str]:
    def is_numeric(physical_type: str) -> bool:
        lowered = physical_type.lower().lstrip("<>|=")
        return lowered in {"int", "float"} or "int" in lowered or "float" in lowered or lowered.startswith(("i", "u", "f"))

    return [
        field.field_name
        for field in fields
        if field.field_name not in excluded_names
        and is_numeric(field.physical_type)
        and field.logical_type not in {"identifier", "coordinate", "temporal_coordinate", "time_axis"}
    ]


def _assembled_datetime_series(column_samples: Dict[str, List[str]]) -> tuple[Optional[List[str]], List[datetime]]:
    key_map = {key.lower(): key for key in column_samples}
    if any(part not in key_map for part in ("year", "month", "day")):
        return None, []

    ordered_parts = [key_map["year"], key_map["month"], key_map["day"]]
    ordered_parts.extend(key_map[part] for part in ("hour", "minute", "second") if part in key_map)
    parsed_values: List[datetime] = []
    for index in range(len(column_samples[key_map["year"]])):
        try:
            parsed_values.append(
                datetime(
                    int(column_samples[key_map["year"]][index]),
                    int(column_samples[key_map["month"]][index]),
                    int(column_samples[key_map["day"]][index]),
                    int(column_samples[key_map["hour"]][index]) if "hour" in key_map else 0,
                    int(column_samples[key_map["minute"]][index]) if "minute" in key_map else 0,
                    int(column_samples[key_map["second"]][index]) if "second" in key_map else 0,
                )
            )
        except (IndexError, ValueError):
            return None, []
    return ordered_parts, parsed_values


def _candidate_for_field(field: FieldSchema, samples: List[str]) -> Optional[Dict[str, Any]]:
    non_empty = [value for value in samples if value.strip()]
    parsed = [parse_datetime(value) for value in non_empty]
    parsed_values = [value for value in parsed if value is not None]
    parse_rate = round(len(parsed_values) / len(non_empty), 4) if non_empty else 0.0
    if field.physical_type != "datetime" and parse_rate < 0.8:
        return None

    score = 0.0
    reasons: List[str] = []
    if field.physical_type == "datetime":
        score += 0.4
        reasons.append("physical_datetime")
    elif parse_rate >= 0.8:
        score += 0.35
        reasons.append("high_datetime_parse_rate")
    if is_strong_time_field_name(field.field_name):
        score += 0.4
        reasons.append("strong_time_name")
    if field.semantic_type == "observation_time":
        score += 0.15
        reasons.append("observation_time_semantic")
    if field.logical_type == "time_axis":
        score += 0.15
        reasons.append("existing_time_axis_logical")
    elif field.logical_type == "temporal_coordinate":
        score += 0.35
        reasons.append("explicit_temporal_coordinate")
    return {
        "field": field.field_name,
        "field_path": field.field_path,
        "score": round(min(score, 1.0), 4),
        "parse_rate": parse_rate,
        "reasons": reasons,
        "parsed_values": parsed_values,
        "raw_values": non_empty,
    }


def analyze_temporal_semantics(
    column_samples: Dict[str, List[str]],
    fields: List[FieldSchema],
) -> Dict[str, Any]:
    field_by_name = {field.field_name: field for field in fields}
    candidates = [
        candidate
        for field in fields
        if (candidate := _candidate_for_field(field, column_samples.get(field.field_name, []))) is not None
    ]

    assembled_parts, assembled_values = _assembled_datetime_series(column_samples)
    if assembled_parts:
        candidates.append(
            {
                "field": "computed_from_parts",
                "field_path": "computed_from_parts",
                "score": 0.85,
                "parse_rate": 1.0,
                "reasons": ["assembled_year_month_day"],
                "parsed_values": assembled_values,
                "raw_values": [],
                "computed_from": assembled_parts,
            }
        )

    public_candidates = [
        {key: value for key, value in candidate.items() if key not in {"parsed_values", "raw_values"}}
        for candidate in sorted(candidates, key=lambda item: (-item["score"], item["field"]))
    ]
    issues: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    eligible = sorted(
        [candidate for candidate in candidates if candidate["score"] >= 0.7],
        key=lambda item: (-item["score"], item["field"]),
    )
    if eligible:
        if len(eligible) > 1 and eligible[0]["score"] - eligible[1]["score"] <= 0.05:
            issues.append(
                {
                    "code": "ambiguous_time_axis_candidates",
                    "severity": "warning",
                    "detail": "Multiple temporal candidates have approximately equal deterministic support.",
                    "candidates": [eligible[0]["field"], eligible[1]["field"]],
                }
            )
        else:
            selected = eligible[0]
    elif candidates:
        issues.append(
            {
                "code": "no_supported_time_axis_candidate",
                "severity": "note",
                "detail": "Temporal-looking fields exist, but none meet the deterministic selection threshold.",
            }
        )

    excluded = set(selected.get("computed_from", [])) if selected else set()
    if selected and selected["field"] != "computed_from_parts":
        excluded.add(selected["field"])
    measurements = _measurement_candidates(fields, excluded)
    if selected and not measurements:
        issues.append(
            {
                "code": "temporal_candidate_without_measurements",
                "severity": "note",
                "detail": "A temporal field exists, but no numeric measurements support time-series organization.",
            }
        )
        selected = None

    property_claims: Dict[str, Dict[str, Any]] = {}
    time_series: Dict[str, Any] = {}
    selected_public = None
    if selected is not None:
        selected_public = {key: value for key, value in selected.items() if key not in {"parsed_values", "raw_values"}}
        properties = _time_properties(selected["parsed_values"])
        timezone_claim = (
            _timezone_claim(selected["raw_values"])
            if selected["field"] != "computed_from_parts"
            else _claim("unknown", "unknown", "timezone_not_encoded_in_values", ["assembled_date_parts"])
        )
        field_evidence = [f"field:{selected['field']}:name", f"field:{selected['field']}:sample_values"]
        property_claims = {
            "field": _claim(selected["field"], "supported", "ranked_temporal_candidate", field_evidence),
            "type": _claim("datetime", "supported", "datetime_parse_success", field_evidence),
            "timezone": timezone_claim,
            "frequency": _claim(
                properties["frequency"],
                "derived" if properties["frequency"] != "unknown" else "unknown",
                "sample_delta_profile" if properties["frequency"] != "unknown" else "sampling_insufficient",
                ["sample_values:ordered_deltas"],
            ),
            "regularity": _claim(
                properties["regularity"],
                "derived" if properties["regularity"] != "unknown" else "unknown",
                "sample_delta_profile" if properties["regularity"] != "unknown" else "sampling_insufficient",
                ["sample_values:ordered_deltas"],
            ),
            "missing_intervals": _claim(
                properties["missing_intervals"],
                "derived" if properties["missing_intervals"] is not None else "unknown",
                "dominant_interval_grid" if properties["missing_intervals"] is not None else "irregular_or_insufficient_sampling",
                ["sample_values:ordered_deltas"],
            ),
        }
        time_axis = {
            "field": selected["field"],
            "type": "datetime",
            "timezone": timezone_claim["value"],
            "frequency": properties["frequency"],
            "regularity": properties["regularity"],
            "missing_intervals": properties["missing_intervals"],
        }
        if selected.get("computed_from"):
            time_axis["computed_from"] = selected["computed_from"]
        if selected["field"] != "computed_from_parts":
            field_by_name[selected["field"]].logical_type = "time_axis"
            if field_by_name[selected["field"]].semantic_type == "unknown":
                field_by_name[selected["field"]].semantic_type = "observation_time"
        identifier_candidates = [
            field.field_name
            for field in fields
            if field.semantic_type.endswith("_identifier") or field.logical_type == "identifier"
        ]
        time_series = {
            "time_axis": time_axis,
            "series_identifier": identifier_candidates[0] if identifier_candidates else None,
            "measurements": measurements,
        }

    return {
        "time_series": time_series,
        "temporal_analysis": {
            "analysis_version": "phase12_temporal_v1",
            "candidates": public_candidates,
            "selected_candidate": selected_public,
            "property_claims": property_claims,
            "validator_results": {
                "candidate_count": len(candidates),
                "eligible_candidate_count": len(eligible),
                "measurement_count": len(measurements),
                "selection_abstained": selected is None,
            },
            "issues": issues,
        },
    }
