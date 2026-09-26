from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceRecord:
    tier: str
    evidence_type: str
    source: str
    detail: str
    confidence: float


@dataclass
class FieldSchema:
    field_name: str
    field_path: str
    physical_type: str
    logical_type: str = "unknown"
    semantic_type: str = "unknown"
    nullable: bool = True
    unique_ratio: Optional[float] = None
    shape: Optional[List[int]] = None
    unit: Optional[str] = None
    unit_normalization: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    example_values: List[str] = field(default_factory=list)
    missing_count: int = 0
    value_range: Optional[List[float]] = None
    source_evidence: List[EvidenceRecord] = field(default_factory=list)
    confidence: float = 1.0
    uncertainty_reason: Optional[str] = None
    unknown_trace: Dict[str, Any] = field(default_factory=dict)
    extraction_method: str = ""


@dataclass
class DatasetSchema:
    dataset_id: str
    file_id: str
    file_format: str
    data_modality: str = "unknown"
    fields: List[FieldSchema] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
