from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List


UNIFIED_SCHEMA_VERSION = "1.0.0"
CLAIM_STATES = [
    "observed",
    "declared",
    "derived",
    "supported",
    "conflicted",
    "unknown",
    "unsupported",
    "abstained",
]


def _dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("Unified schema input must be an extraction outcome or dictionary.")


def _state_for_field(field: Dict[str, Any]) -> str:
    method = str(field.get("extraction_method", ""))
    if "declaration" in method:
        return "declared"
    if field.get("uncertainty_reason") == "sample_bounded_observation":
        return "observed"
    return "supported"


def _walk_named_lists(value: Any, names: set[str]) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names and isinstance(item, list):
                for entry in item:
                    yield key, entry
            yield from _walk_named_lists(item, names)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_named_lists(item, names)


def _walk_state_claims(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        if value.get("state") in CLAIM_STATES and (
            "reason_code" in value or "evidence_refs" in value
        ):
            yield value
        for item in value.values():
            yield from _walk_state_claims(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_state_claims(item)


def build_unified_schema_envelope(outcome: Any) -> Dict[str, Any]:
    payload = _dict(outcome)
    schema = payload.get("schema") or {}
    metadata = schema.get("metadata", {})
    extractor = payload.get("extractor")
    evidence: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    evidence_by_key: Dict[tuple[Any, ...], str] = {}

    def add_evidence(item: Dict[str, Any]) -> str:
        key = (
            item.get("tier"),
            item.get("evidence_type"),
            item.get("source"),
            item.get("detail"),
        )
        if key not in evidence_by_key:
            evidence_id = f"e{len(evidence) + 1:04d}"
            evidence_by_key[key] = evidence_id
            evidence.append({"evidence_id": evidence_id, **item})
        return evidence_by_key[key]

    for field in schema.get("fields", []):
        refs = [add_evidence(item) for item in field.get("source_evidence", [])]
        state = _state_for_field(field)
        claims.append(
            {
                "claim_id": f"c{len(claims) + 1:04d}",
                "subject": field.get("field_path"),
                "property": "physical_type",
                "value": field.get("physical_type"),
                "state": state,
                "reason_code": field.get("uncertainty_reason") or "extractor_field_claim",
                "evidence_refs": refs,
            }
        )
        for property_name in ("logical_type", "semantic_type", "unit", "nullable"):
            value = field.get(property_name)
            if value is None or value == "unknown":
                continue
            claims.append(
                {
                    "claim_id": f"c{len(claims) + 1:04d}",
                    "subject": field.get("field_path"),
                    "property": property_name,
                    "value": value,
                    "state": state if property_name == "nullable" else "supported",
                    "reason_code": "extractor_field_property",
                    "evidence_refs": refs,
                }
            )

    seen_generic: set[str] = set()
    for item in _walk_state_claims(metadata):
        signature = repr(item)
        if signature in seen_generic:
            continue
        seen_generic.add(signature)
        generic_refs = [
            add_evidence(
                {
                    "tier": "claim_reference",
                    "evidence_type": "format_specific_reference",
                    "source": str(ref),
                    "detail": "Evidence reference preserved from format-specific analysis.",
                    "confidence": 1.0,
                }
            )
            for ref in item.get("evidence_refs", [])
        ]
        claims.append(
            {
                "claim_id": f"c{len(claims) + 1:04d}",
                "subject": item.get("field") or item.get("field_path") or "dataset",
                "property": item.get("property") or "format_specific_claim",
                "value": item.get("value"),
                "state": item["state"],
                "reason_code": item.get("reason_code"),
                "evidence_refs": generic_refs,
            }
        )

    conflict_items = [
        {"category": name, "detail": item}
        for name, item in _walk_named_lists(metadata, {"conflicts", "issues", "compatibility_gaps"})
    ]
    outcome_issues = payload.get("issues", [])
    unsupported = [
        issue for issue in outcome_issues if "unsupported" in str(issue.get("code", ""))
    ]
    abstentions = outcome_issues if payload.get("status") == "abstained" else []
    format_specific = {
        key: value
        for key, value in metadata.items()
        if key.endswith("_analysis") or key in {"dimensions", "store_inventory", "row_groups"}
    }
    temporal = {
        key: metadata[key]
        for key in ("temporal_analysis", "time_series")
        if key in metadata
    }
    return {
        "schema_envelope_version": UNIFIED_SCHEMA_VERSION,
        "identity": {
            "dataset_id": schema.get("dataset_id"),
            "file_id": schema.get("file_id"),
            "file_format": schema.get("file_format"),
            "data_modality": schema.get("data_modality"),
            "resource_kind": metadata.get("resource_kind", "file"),
        },
        "format_detection": payload.get("format_decision"),
        "extractor_capability": extractor,
        "outcome": {
            "status": payload.get("status"),
            "issues": outcome_issues,
        },
        "physical_structure": {
            "fields": schema.get("fields", []),
            "groups": schema.get("groups", []),
            "dimensions": metadata.get("dimensions", {}),
            "format_specific": format_specific,
        },
        "logical_roles": [
            {"field_path": field.get("field_path"), "logical_type": field.get("logical_type")}
            for field in schema.get("fields", [])
            if field.get("logical_type") not in {None, "unknown"}
        ],
        "semantic_hints": [
            {"field_path": field.get("field_path"), "semantic_type": field.get("semantic_type")}
            for field in schema.get("fields", [])
            if field.get("semantic_type") not in {None, "unknown"}
        ],
        "units": [
            {
                "field_path": field.get("field_path"),
                "unit": field.get("unit"),
                "normalization": field.get("unit_normalization"),
            }
            for field in schema.get("fields", [])
            if field.get("unit") is not None or field.get("unit_normalization") is not None
        ],
        "temporal_semantics": temporal,
        "relationships": {
            "dimensions": metadata.get("dimensions", {}),
            "groups": schema.get("groups", []),
        },
        "claims": claims,
        "evidence": evidence,
        "provenance": {
            "source_file_id": schema.get("file_id"),
            "extractor_id": (extractor or {}).get("extractor_id"),
            "extractor_version": (extractor or {}).get("version"),
            "determinism_class": (extractor or {}).get("determinism_class"),
            "transformation": "deterministic_unified_envelope_projection",
        },
        "conflicts": conflict_items,
        "abstentions": abstentions,
        "unsupported_features": unsupported,
        "evaluation_metadata": {
            "claim_state_vocabulary": CLAIM_STATES,
            "source_outcome_status": payload.get("status"),
            "backward_compatibility": "Legacy ExtractionOutcome.schema remains authoritative and unchanged.",
            "claim_boundary": "The envelope normalizes existing extractor claims; it does not promote new semantic truth.",
        },
    }
