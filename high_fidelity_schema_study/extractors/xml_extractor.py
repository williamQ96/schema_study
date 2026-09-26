from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from ..models import DatasetSchema, EvidenceRecord, FieldSchema
from .base import StructuredExtractionError


MAX_XML_BYTES = 16 * 1024 * 1024
XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
XSI_TYPE = f"{{{XSI_NAMESPACE}}}type"
INTEGER_RE = re.compile(r"^[+-]?\d+$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$")


def _local_name(name: str) -> str:
    return name.split("}", 1)[-1] if name.startswith("{") else name


def _scalar_type(text: Optional[str]) -> str:
    value = (text or "").strip()
    if not value:
        return "empty"
    if value.lower() in {"true", "false"}:
        return "boolean"
    if INTEGER_RE.match(value):
        return "integer"
    if NUMBER_RE.match(value):
        return "number"
    return "string"


def _xsd_scalar_type(value: str) -> str:
    local = value.split(":")[-1]
    if local in {"byte", "short", "int", "integer", "long", "nonNegativeInteger", "positiveInteger"}:
        return "integer"
    if local in {"decimal", "float", "double"}:
        return "number"
    if local == "boolean":
        return "boolean"
    if local in {"string", "normalizedString", "token", "date", "dateTime", "time"}:
        return "string"
    return "unknown"


def _evidence(source: str, evidence_type: str, detail: str, tier: str = "sample_observation") -> EvidenceRecord:
    return EvidenceRecord(tier, evidence_type, source, detail, 1.0)


def _parse(path: Path) -> ET.Element:
    size = path.stat().st_size
    if size > MAX_XML_BYTES:
        raise StructuredExtractionError(
            "sampling_insufficient",
            "extraction",
            "XML document exceeds the bounded parser byte limit.",
            status="abstained",
            details={"max_xml_bytes": MAX_XML_BYTES, "byte_size": size},
        )
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise StructuredExtractionError("parser_error", "extraction", str(exc)) from exc


def _declared_xsd(root: ET.Element) -> tuple[List[FieldSchema], Dict[str, Any]]:
    fields: List[FieldSchema] = []
    declarations: List[Dict[str, Any]] = []

    def owned_attributes(node: ET.Element) -> List[ET.Element]:
        result: List[ET.Element] = []
        for child in node:
            if child.tag == f"{{{XSD_NAMESPACE}}}element":
                continue
            if child.tag == f"{{{XSD_NAMESPACE}}}attribute":
                result.append(child)
            result.extend(owned_attributes(child))
        return result

    def visit(node: ET.Element, prefix: str) -> None:
        name = node.attrib.get("name")
        if name:
            path = f"{prefix}/{name}" if prefix else f"/{name}"
            declared_type = node.attrib.get("type", "unknown")
            min_occurs = node.attrib.get("minOccurs", "1")
            max_occurs = node.attrib.get("maxOccurs", "1")
            source = f"xsd_declaration:{path}"
            fields.append(
                FieldSchema(
                    field_name=name,
                    field_path=path,
                    physical_type=declared_type,
                    nullable=min_occurs == "0",
                    source_evidence=[
                        _evidence(source, "xsd_element_declaration", "XSD element declaration", "explicit_metadata")
                    ],
                    extraction_method="xsd_declaration",
                )
            )
            declarations.append(
                {
                    "field_path": path,
                    "kind": "element",
                    "state": "declared",
                    "declared_type": declared_type,
                    "min_occurs": min_occurs,
                    "max_occurs": max_occurs,
                    "evidence_refs": [source],
                }
            )
            next_prefix = path
        else:
            next_prefix = prefix
        for attribute in owned_attributes(node):
            attribute_name = attribute.attrib.get("name")
            if not attribute_name or not next_prefix:
                continue
            path = f"{next_prefix}/@{attribute_name}"
            source = f"xsd_declaration:{path}"
            declared_type = attribute.attrib.get("type", "unknown")
            required = attribute.attrib.get("use") == "required"
            if not any(item["field_path"] == path for item in declarations):
                fields.append(
                    FieldSchema(
                        field_name=f"@{attribute_name}",
                        field_path=path,
                        physical_type=declared_type,
                        nullable=not required,
                        source_evidence=[
                            _evidence(source, "xsd_attribute_declaration", "XSD attribute declaration", "explicit_metadata")
                        ],
                        extraction_method="xsd_declaration",
                    )
                )
                declarations.append(
                    {
                        "field_path": path,
                        "kind": "attribute",
                        "state": "declared",
                        "declared_type": declared_type,
                        "required": required,
                        "evidence_refs": [source],
                    }
                )
        for child in node:
            if child.tag == f"{{{XSD_NAMESPACE}}}element":
                visit(child, next_prefix)
            elif child.tag in {
                f"{{{XSD_NAMESPACE}}}complexType",
                f"{{{XSD_NAMESPACE}}}sequence",
                f"{{{XSD_NAMESPACE}}}choice",
                f"{{{XSD_NAMESPACE}}}all",
            }:
                visit(child, next_prefix)

    for element in root:
        if element.tag == f"{{{XSD_NAMESPACE}}}element":
            visit(element, "")
    return fields, {
        "mode": "declared_xsd",
        "declared_fields": declarations,
        "observed_fields": [],
        "conflicts": [],
        "namespaces": {"xsd": XSD_NAMESPACE},
    }


def _observed_xml(root: ET.Element, sample_limit: int) -> tuple[List[FieldSchema], Dict[str, Any], Dict[str, Any]]:
    types: Dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    max_sibling_occurs: Dict[str, int] = defaultdict(int)
    declarations: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    namespaces: Dict[str, str] = {}
    visited = 0
    truncated = False

    def record_namespace(name: str) -> None:
        if name.startswith("{"):
            uri = name[1:].split("}", 1)[0]
            if uri not in namespaces.values():
                namespaces[f"ns{len(namespaces)}"] = uri

    def visit(node: ET.Element, path: str) -> None:
        nonlocal visited, truncated
        if visited >= sample_limit:
            truncated = True
            return
        visited += 1
        record_namespace(node.tag)
        observed_type = _scalar_type(node.text)
        types[path].add(observed_type)
        counts[path] += 1
        xsi_type = node.attrib.get(XSI_TYPE)
        if xsi_type:
            source = f"xml_attribute:{path}/@{{{XSI_NAMESPACE}}}type"
            declaration = {
                "field_path": path,
                "state": "declared",
                "declared_type": xsi_type,
                "reason_code": "xsi_type_declaration",
                "evidence_refs": [source],
            }
            declarations.append(declaration)
            expected = _xsd_scalar_type(xsi_type)
            if expected != "unknown" and observed_type not in {expected, "empty"}:
                conflicts.append(
                    {
                        "field_path": path,
                        "code": "declared_observed_type_conflict",
                        "declared_type": xsi_type,
                        "observed_type": observed_type,
                        "evidence_refs": [source, f"xml_sample:{path}"],
                    }
                )
        for name, value in node.attrib.items():
            record_namespace(name)
            attribute_path = f"{path}/@{name}"
            types[attribute_path].add(_scalar_type(value))
            counts[attribute_path] += 1
        children = list(node)
        sibling_counts = Counter(child.tag for child in children)
        for tag, count in sibling_counts.items():
            max_sibling_occurs[f"{path}/{tag}"] = max(max_sibling_occurs[f"{path}/{tag}"], count)
        for child in children:
            visit(child, f"{path}/{child.tag}")

    root_path = f"/{root.tag}"
    visit(root, root_path)
    fields: List[FieldSchema] = []
    observations: List[Dict[str, Any]] = []
    for path in sorted(types):
        observed_types = sorted(types[path])
        source = f"xml_sample:{path}"
        name = path.rsplit("/", 1)[-1]
        fields.append(
            FieldSchema(
                field_name=name,
                field_path=path,
                physical_type=" | ".join(observed_types),
                nullable="empty" in observed_types,
                source_evidence=[_evidence(source, "xml_sample_observation", "Observed bounded XML path and type set")],
                extraction_method="bounded_xml_structure_observation",
                uncertainty_reason="sample_bounded_observation",
            )
        )
        observations.append(
            {
                "field_path": path,
                "state": "observed",
                "observed_types": observed_types,
                "occurrence_count": counts[path],
                "max_sibling_occurs": max_sibling_occurs.get(path, 1),
                "repeated": max_sibling_occurs.get(path, 1) > 1,
                "evidence_refs": [source],
            }
        )
    return fields, {
        "mode": "bounded_observed_structure",
        "declared_fields": declarations,
        "observed_fields": observations,
        "conflicts": conflicts,
        "namespaces": namespaces,
    }, {
        "visited_element_count": visited,
        "sample_limit": sample_limit,
        "sample_truncated": truncated,
        "claim_scope": "bounded_sample",
    }


def extract_xml_schema(path: str, sample_limit: int = 200) -> DatasetSchema:
    xml_path = Path(path)
    root = _parse(xml_path)
    if root.tag == f"{{{XSD_NAMESPACE}}}schema":
        fields, analysis = _declared_xsd(root)
        sampling = {
            "visited_element_count": 0,
            "sample_limit": sample_limit,
            "sample_truncated": False,
            "claim_scope": "declared_document",
        }
        modality = "schema_document"
    else:
        fields, analysis, sampling = _observed_xml(root, sample_limit)
        modality = "semi_structured"
    return DatasetSchema(
        dataset_id=xml_path.stem,
        file_id=xml_path.name,
        file_format="xml",
        data_modality=modality,
        fields=fields,
        metadata={
            "resource_kind": "file",
            "xml_analysis": analysis,
            "sampling": sampling,
            "extraction_errors": [],
        },
        notes=[
            "Observed XML structure is sample-bounded and is not a universal schema claim.",
            "XSD and xsi:type declarations remain separate from observed instance structure.",
            "External schemas and entities are not loaded.",
            "Semantic roles are not inferred from XML names.",
        ],
    )
