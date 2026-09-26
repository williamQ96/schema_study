from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..models import DatasetSchema


@dataclass
class ExtractionRequest:
    path: str
    format_hint: Optional[str] = None
    sample_limit: int = 200
    resource_kind: str = "auto"


@dataclass
class FormatSignal:
    format: str
    basis: str
    support_score: float
    detail: str


@dataclass
class FormatDecision:
    selected_format: Optional[str]
    basis: str
    support_score: float
    signals: List[FormatSignal] = field(default_factory=list)
    conflicted: bool = False


@dataclass
class ExtractorCapability:
    extractor_id: str
    version: str
    formats: List[str]
    suffixes: List[str]
    magic_prefixes: List[str]
    operations: List[str]
    determinism_class: str
    evidence_types: List[str]
    resource_kinds: List[str] = field(default_factory=lambda: ["file"])


@dataclass
class ExtractionIssue:
    code: str
    stage: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionOutcome:
    status: str
    format_decision: FormatDecision
    extractor: Optional[ExtractorCapability]
    schema: Optional[DatasetSchema]
    issues: List[ExtractionIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "format_decision": asdict(self.format_decision),
            "extractor": asdict(self.extractor) if self.extractor is not None else None,
            "schema": self.schema.to_dict() if self.schema is not None else None,
            "issues": [asdict(issue) for issue in self.issues],
        }


class StructuredExtractionError(RuntimeError):
    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        *,
        status: str = "failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.status = status
        self.details = details or {}


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, path: str) -> DatasetSchema:
        raise NotImplementedError
