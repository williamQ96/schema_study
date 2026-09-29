"""Deterministic navigation sentinel for a condition with no category classifier."""
from __future__ import annotations

from .common import digest, seal, seal_errors, taxonomy
from .evidence_schema import validate_paper_input
from .paper import source_identity, unit_catalog


INDEX_VERSION = "paper-category-index/disabled-v1"


def make_disabled_index(paper_input: dict, taxonomy_value: dict | None = None) -> dict:
    validate_paper_input(paper_input)
    tax = taxonomy() if taxonomy_value is None else taxonomy_value
    if not isinstance(tax, dict):
        raise ValueError("taxonomy_must_be_object")
    entries = [{"unit_id": uid, "page": unit["page"], "availability": "unavailable",
                "prediction": None, "machine_reason": "disabled_by_experimental_condition"}
               for uid, unit in unit_catalog(paper_input).items()]
    return seal({"schema_version": INDEX_VERSION, "taxonomy_sha256": digest(tax),
                 "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input),
                 "entries": entries, "authority": "navigation_disabled_no_classifier_generation",
                 "semantic_review": "not_established"}, "index_sha256")


def verify_disabled_index(index: dict, paper_input: dict,
                          taxonomy_value: dict | None = None) -> list[str]:
    errors = seal_errors(index, "index_sha256")
    try:
        if index != make_disabled_index(paper_input, taxonomy_value):
            errors.append("disabled_navigation_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("disabled_navigation_replay:" + str(exc))
    return errors
