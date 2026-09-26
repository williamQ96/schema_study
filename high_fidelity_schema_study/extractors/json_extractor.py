from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import DatasetSchema, EvidenceRecord, FieldSchema
from .base import StructuredExtractionError


MAX_JSON_BYTES = 16 * 1024 * 1024


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _evidence(source: str, evidence_type: str, detail: str) -> EvidenceRecord:
    return EvidenceRecord("sample_observation", evidence_type, source, detail, 1.0)


def _records_from_document(payload: Any) -> tuple[List[Any], str]:
    if isinstance(payload, list):
        return payload, "top_level_array"
    return [payload], "single_document"


def _read_records(path: Path, sample_limit: int) -> tuple[List[Any], Dict[str, Any]]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise StructuredExtractionError(
            "sampling_insufficient",
            "extraction",
            "JSON document exceeds the bounded parser byte limit.",
            status="abstained",
            details={"max_json_bytes": MAX_JSON_BYTES, "byte_size": path.stat().st_size},
        )
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: List[Any] = []
        total_lines = 0
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    total_lines += 1
                    if len(records) >= sample_limit:
                        continue
                    records.append(json.loads(line))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StructuredExtractionError("parser_error", "extraction", str(exc)) from exc
        return records, {
            "organization": "json_lines",
            "observed_record_count": len(records),
            "known_total_record_count": total_lines,
            "sample_truncated": total_lines > len(records),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredExtractionError("parser_error", "extraction", str(exc)) from exc
    records, organization = _records_from_document(payload)
    return records[:sample_limit], {
        "organization": organization,
        "observed_record_count": min(len(records), sample_limit),
        "known_total_record_count": len(records),
        "sample_truncated": len(records) > sample_limit,
    }


def _declared_json_schema(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("type"), (str, list)) or not isinstance(payload.get("properties"), dict):
        return None
    if "$schema" not in payload and "$id" not in payload:
        return None
    return payload


def _declared_fields(schema: Dict[str, Any]) -> tuple[List[FieldSchema], List[Dict[str, Any]]]:
    required = set(schema.get("required", [])) if isinstance(schema.get("required"), list) else set()
    fields: List[FieldSchema] = []
    claims: List[Dict[str, Any]] = []

    def visit(properties: Dict[str, Any], prefix: str, required_names: set[str]) -> None:
        for name, declaration in properties.items():
            if not isinstance(declaration, dict):
                continue
            path = f"{prefix}.{name}"
            declared_type = declaration.get("type", "unknown")
            types = declared_type if isinstance(declared_type, list) else [declared_type]
            source = f"json_schema:{path}"
            nullable = "null" in types or name not in required_names
            fields.append(
                FieldSchema(
                    field_name=name,
                    field_path=path,
                    physical_type=" | ".join(str(item) for item in types),
                    logical_type="unknown",
                    semantic_type="unknown",
                    nullable=nullable,
                    description=declaration.get("description"),
                    source_evidence=[_evidence(source, "json_schema_declaration", "JSON Schema property declaration")],
                    extraction_method="json_schema_declaration",
                )
            )
            claims.append(
                {
                    "field_path": path,
                    "state": "declared",
                    "declared_types": types,
                    "required": name in required_names,
                    "description": declaration.get("description"),
                    "evidence_refs": [source],
                }
            )
            nested = declaration.get("properties")
            nested_required = set(declaration.get("required", [])) if isinstance(declaration.get("required"), list) else set()
            if isinstance(nested, dict):
                visit(nested, path, nested_required)

    visit(schema["properties"], "$", required)
    return fields, claims


def _observed_fields(records: List[Any]) -> tuple[List[FieldSchema], List[Dict[str, Any]]]:
    types: Dict[str, set[str]] = defaultdict(set)
    record_presence: Dict[str, set[int]] = defaultdict(set)
    null_counts: Dict[str, int] = defaultdict(int)
    array_lengths: Dict[str, List[int]] = defaultdict(list)
    array_element_types: Dict[str, set[str]] = defaultdict(set)

    def visit(value: Any, path: str, record_index: int) -> None:
        value_type = _json_type(value)
        types[path].add(value_type)
        record_presence[path].add(record_index)
        if value is None:
            null_counts[path] += 1
            return
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, f"{path}.{name}", record_index)
        elif isinstance(value, list):
            array_lengths[path].append(len(value))
            for child in value:
                array_element_types[path].add(_json_type(child))
                visit(child, f"{path}[]", record_index)

    for index, record in enumerate(records):
        visit(record, "$", index)

    fields: List[FieldSchema] = []
    claims: List[Dict[str, Any]] = []
    total = len(records)
    for path in sorted(types):
        if path == "$":
            continue
        observed_types = sorted(types[path])
        missing_count = total - len(record_presence[path])
        source = f"json_sample:{path}"
        field_name = path.rsplit(".", 1)[-1]
        fields.append(
            FieldSchema(
                field_name=field_name,
                field_path=path,
                physical_type=" | ".join(observed_types),
                logical_type="unknown",
                semantic_type="unknown",
                nullable=null_counts[path] > 0 or missing_count > 0,
                missing_count=missing_count,
                source_evidence=[
                    _evidence(source, "json_sample_observation", "Observed JSON path and bounded type set")
                ],
                extraction_method="bounded_json_structure_observation",
                uncertainty_reason="sample_bounded_observation",
            )
        )
        claim: Dict[str, Any] = {
            "field_path": path,
            "state": "observed",
            "observed_types": observed_types,
            "observed_record_count": len(record_presence[path]),
            "sample_record_count": total,
            "missing_count": missing_count,
            "null_count": null_counts[path],
            "evidence_refs": [source],
        }
        if path in array_lengths:
            element_types = sorted(array_element_types[path])
            claim["array"] = {
                "observed_lengths": sorted(set(array_lengths[path])),
                "element_types": element_types,
                "heterogeneous": len(element_types) > 1,
            }
        claims.append(claim)
    return fields, claims


def extract_json_schema(path: str, sample_limit: int = 200) -> DatasetSchema:
    json_path = Path(path)
    records, sample = _read_records(json_path, sample_limit)
    declared = _declared_json_schema(records[0]) if sample["organization"] == "single_document" and records else None
    if declared is not None:
        fields, declarations = _declared_fields(declared)
        examples = declared.get("examples", [])
        example_records = examples[:sample_limit] if isinstance(examples, list) else []
        _example_fields, observations = _observed_fields(example_records) if example_records else ([], [])
        declared_map = {item["field_path"]: item for item in declarations}
        conflicts = []
        for observation in observations:
            declaration = declared_map.get(observation["field_path"])
            if declaration is None:
                continue
            declared_types = set(str(item) for item in declaration["declared_types"])
            observed_types = set(observation["observed_types"])
            compatible = observed_types.issubset(declared_types) or (
                "number" in declared_types and observed_types.issubset({"number", "integer"})
            )
            if not compatible:
                conflicts.append(
                    {
                        "field_path": observation["field_path"],
                        "code": "declared_observed_type_conflict",
                        "declared_types": sorted(declared_types),
                        "observed_types": sorted(observed_types),
                        "evidence_refs": [
                            *declaration["evidence_refs"],
                            *observation["evidence_refs"],
                        ],
                    }
                )
        analysis = {
            "mode": "declared_json_schema",
            "declared_fields": declarations,
            "observed_fields": observations,
            "conflicts": conflicts,
        }
        data_modality = "schema_document"
    else:
        fields, observations = _observed_fields(records)
        analysis = {
            "mode": "bounded_observed_structure",
            "declared_fields": [],
            "observed_fields": observations,
            "conflicts": [],
        }
        data_modality = "semi_structured"

    return DatasetSchema(
        dataset_id=json_path.stem,
        file_id=json_path.name,
        file_format="json",
        data_modality=data_modality,
        fields=fields,
        metadata={
            "resource_kind": "file",
            "json_analysis": analysis,
            "sampling": {
                **sample,
                "sample_limit": sample_limit,
                "claim_scope": "bounded_sample" if declared is None else "declared_document",
                "schema_example_count": len(example_records) if declared is not None else 0,
            },
            "value_observation": {
                "state": "observed_bounded_sample" if declared is None else "not_applicable",
                "reason_code": "bounded_json_structure_sample" if declared is None else "json_schema_declaration",
            },
            "extraction_errors": [],
        },
        notes=[
            "Observed JSON structure is sample-bounded and is not a universal schema claim.",
            "JSON Schema properties are preserved as declarations and are not treated as observed data.",
            "Semantic roles are not inferred from JSON field names.",
        ],
    )
