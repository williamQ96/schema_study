from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..models import DatasetSchema, EvidenceRecord, FieldSchema

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover
    pa = None
    pq = None


def _normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _normalize(value.item())
        except ValueError:
            pass
    return value


def _decode_metadata(metadata: Optional[Dict[bytes, bytes]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    return {str(_normalize(key)): _normalize(value) for key, value in metadata.items()}


def _evidence(source: str, evidence_type: str, detail: str) -> EvidenceRecord:
    return EvidenceRecord(
        tier="explicit_metadata",
        evidence_type=evidence_type,
        source=source,
        detail=detail,
        confidence=1.0,
    )


def _logical_type(arrow_type: Any) -> str:
    if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type) or pa.types.is_time(arrow_type):
        return "temporal"
    if pa.types.is_struct(arrow_type):
        return "struct"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return "list"
    if pa.types.is_map(arrow_type):
        return "map"
    if pa.types.is_dictionary(arrow_type):
        return "dictionary"
    return "unknown"


def _type_descriptor(arrow_type: Any) -> Dict[str, Any]:
    descriptor: Dict[str, Any] = {
        "arrow_type": str(arrow_type),
        "type_id": str(arrow_type.id),
    }
    if pa.types.is_timestamp(arrow_type):
        descriptor.update({"unit": arrow_type.unit, "timezone": arrow_type.tz})
    elif pa.types.is_time(arrow_type) or pa.types.is_duration(arrow_type):
        descriptor["unit"] = arrow_type.unit
    elif pa.types.is_decimal(arrow_type):
        descriptor.update({"precision": arrow_type.precision, "scale": arrow_type.scale})
    elif pa.types.is_fixed_size_binary(arrow_type):
        descriptor["byte_width"] = arrow_type.byte_width
    elif pa.types.is_fixed_size_list(arrow_type):
        descriptor["list_size"] = arrow_type.list_size
    elif pa.types.is_dictionary(arrow_type):
        descriptor.update(
            {
                "index_type": str(arrow_type.index_type),
                "value_type": str(arrow_type.value_type),
                "ordered": arrow_type.ordered,
            }
        )
    return descriptor


def _child_fields(field: Any) -> Iterable[tuple[str, Any]]:
    arrow_type = field.type
    if pa.types.is_struct(arrow_type):
        return [(child.name, child) for child in arrow_type]
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return [("element", arrow_type.value_field)]
    if pa.types.is_map(arrow_type):
        return [("key", arrow_type.key_field), ("value", arrow_type.item_field)]
    return []


def _arrow_fields(schema: Any) -> tuple[List[FieldSchema], List[Dict[str, Any]]]:
    fields: List[FieldSchema] = []
    analysis: List[Dict[str, Any]] = []

    def visit(field: Any, path: str, parent_path: Optional[str]) -> None:
        source = f"arrow_schema:{path}"
        logical = _logical_type(field.type)
        metadata = _decode_metadata(field.metadata)
        evidence = [_evidence(source, "arrow_schema_field", "Arrow field declaration, type, and nullability")]
        if metadata:
            evidence.append(_evidence(source, "arrow_field_metadata", "Arrow field key-value metadata"))
        fields.append(
            FieldSchema(
                field_name=field.name,
                field_path=path,
                physical_type=str(field.type),
                logical_type=logical,
                nullable=field.nullable,
                source_evidence=evidence,
                extraction_method="parquet_footer_arrow_schema",
            )
        )
        children = list(_child_fields(field))
        claim: Dict[str, Any] = {
            "field_path": path,
            "parent_path": parent_path,
            "name": field.name,
            "nullable": field.nullable,
            "metadata": metadata,
            "type": _type_descriptor(field.type),
            "container": bool(children),
            "leaf": not children,
            "evidence_refs": [source],
        }
        if pa.types.is_timestamp(field.type):
            claim["temporal_claim"] = {
                "state": "declared",
                "reason_code": "arrow_timestamp_type",
                "unit": field.type.unit,
                "timezone": field.type.tz if field.type.tz is not None else "unknown",
                "evidence_refs": [source],
            }
        analysis.append(claim)
        for child_name, child in children:
            visit(child, f"{path}.{child_name}", path)

    for root_field in schema:
        visit(root_field, root_field.name, None)
    return fields, analysis


def _statistics(column: Any, evidence_ref: Optional[str] = None) -> Dict[str, Any]:
    statistics = column.statistics
    evidence_refs = [evidence_ref] if evidence_ref is not None else []
    if statistics is None:
        return {
            "state": "unknown",
            "reason_code": "footer_statistics_absent",
            "source": "parquet_footer",
            "evidence_refs": evidence_refs,
        }
    result = {
        "state": "observed",
        "reason_code": "parquet_footer_statistics",
        "source": "parquet_footer",
        "evidence_refs": evidence_refs,
        "has_min_max": statistics.has_min_max,
        "null_count": statistics.null_count,
        "distinct_count": statistics.distinct_count,
        "num_values": statistics.num_values,
        "physical_type": statistics.physical_type,
    }
    if statistics.has_min_max:
        result.update(
            {
                "min_raw": _normalize(statistics.min_raw),
                "max_raw": _normalize(statistics.max_raw),
            }
        )
        try:
            result.update({"min": _normalize(statistics.min), "max": _normalize(statistics.max)})
        except Exception as exc:
            result.update(
                {
                    "min_max_representation": "raw_physical_values",
                    "decode_issue": {
                        "code": "footer_statistics_decode_unavailable",
                        "message": str(exc),
                    },
                }
            )
    return result


def _parquet_columns(metadata: Any) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    columns: List[Dict[str, Any]] = []
    by_path: Dict[str, Dict[str, Any]] = {}
    for index in range(metadata.num_columns):
        column = metadata.schema.column(index)
        item = {
            "column_index": index,
            "path": column.path,
            "name": column.name,
            "physical_type": column.physical_type,
            "logical_type": str(column.logical_type),
            "converted_type": column.converted_type,
            "max_definition_level": column.max_definition_level,
            "max_repetition_level": column.max_repetition_level,
            "evidence_refs": [f"parquet_footer:schema.column[{index}]"],
        }
        columns.append(item)
        by_path[column.path] = item
    return columns, by_path


def _row_groups(metadata: Any) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        columns = []
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            evidence_ref = f"parquet_footer:row_group[{row_group_index}].column[{column_index}]"
            columns.append(
                {
                    "column_index": column_index,
                    "path": column.path_in_schema,
                    "physical_type": column.physical_type,
                    "compression": column.compression,
                    "encodings": list(column.encodings),
                    "num_values": column.num_values,
                    "total_compressed_size": column.total_compressed_size,
                    "total_uncompressed_size": column.total_uncompressed_size,
                    "statistics": _statistics(column, evidence_ref),
                    "evidence_refs": [evidence_ref],
                }
            )
        groups.append(
            {
                "row_group_index": row_group_index,
                "num_rows": row_group.num_rows,
                "total_byte_size": row_group.total_byte_size,
                "sorting_columns": [str(item) for item in row_group.sorting_columns],
                "columns": columns,
                "evidence_refs": [f"parquet_footer:row_group[{row_group_index}]"],
            }
        )
    return groups


def extract_parquet_schema(path: str) -> DatasetSchema:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for Parquet extraction but is not installed in this environment.")

    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)
    metadata = parquet_file.metadata
    arrow_schema = parquet_file.schema_arrow
    fields, arrow_analysis = _arrow_fields(arrow_schema)
    parquet_columns, parquet_columns_by_path = _parquet_columns(metadata)

    for item in arrow_analysis:
        parquet_column = parquet_columns_by_path.get(item["field_path"])
        if parquet_column is not None:
            item["parquet_column"] = parquet_column
            item["evidence_refs"].extend(parquet_column["evidence_refs"])
    arrow_leaves = [item for item in arrow_analysis if item["leaf"]]
    if len(arrow_leaves) == len(parquet_columns):
        for arrow_leaf, parquet_column in zip(arrow_leaves, parquet_columns):
            arrow_leaf["parquet_column"] = parquet_column
            arrow_leaf["physical_path_mapping"] = {
                "state": "derived",
                "reason_code": "arrow_leaf_parquet_column_ordinal",
                "arrow_path": arrow_leaf["field_path"],
                "parquet_path": parquet_column["path"],
                "evidence_refs": parquet_column["evidence_refs"],
            }
            arrow_leaf["evidence_refs"] = list(
                dict.fromkeys([*arrow_leaf["evidence_refs"], *parquet_column["evidence_refs"]])
            )

    key_value_metadata = _decode_metadata(metadata.metadata)
    return DatasetSchema(
        dataset_id=parquet_path.stem,
        file_id=parquet_path.name,
        file_format="parquet",
        data_modality="tabular",
        fields=fields,
        metadata={
            "backend": f"pyarrow {pa.__version__}",
            "resource_kind": "file",
            "num_rows": metadata.num_rows,
            "num_columns": metadata.num_columns,
            "num_row_groups": metadata.num_row_groups,
            "created_by": metadata.created_by,
            "format_version": metadata.format_version,
            "serialized_size": metadata.serialized_size,
            "key_value_metadata": key_value_metadata,
            "arrow_schema_metadata": _decode_metadata(arrow_schema.metadata),
            "parquet_analysis": {
                "arrow_fields": arrow_analysis,
                "parquet_columns": parquet_columns,
                "row_groups": _row_groups(metadata),
            },
            "value_observation": {
                "state": "unknown",
                "reason_code": "row_values_not_read",
                "row_values_read": 0,
                "statistics_source": "parquet_footer_only",
            },
            "extraction_errors": [],
        },
        notes=[
            "Parquet and Arrow schema metadata were extracted without reading row values.",
            "Min/max, null counts, and distinct counts are reported only when present in the Parquet footer.",
            "Arrow logical and temporal types are declarations; semantic roles are not inferred from field names.",
        ],
    )
