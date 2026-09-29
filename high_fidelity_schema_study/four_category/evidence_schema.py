"""Schema dispatch for versioned paper evidence artifacts.

The layout and compact input are a single source identity.  Keeping their
version pairing here prevents a v4 input from being admitted against v3 layout
metadata (or vice versa) by an otherwise permissive downstream consumer.
"""
from __future__ import annotations

import jsonschema

from .common import ROOT, read_json

LAYOUT_INPUT_PAIRS = {
    ("paper-evidence-layout/v3", "paper-evidence-input/v3"): "v3",
    ("paper-evidence-layout/v4", "paper-evidence-input/v4"): "v4",
    ("paper-evidence-layout/v5", "paper-evidence-input/v5"): "v5",
    ("paper-evidence-layout/v6", "paper-evidence-input/v6"): "v6",
}


def evidence_version(layout: dict, paper_input: dict) -> str:
    if not isinstance(layout, dict) or not isinstance(paper_input, dict):
        raise ValueError("evidence_layout_and_input_must_be_objects")
    pair = (layout.get("schema_version"), paper_input.get("schema_version"))
    try:
        return LAYOUT_INPUT_PAIRS[pair]
    except (KeyError, TypeError):
        raise ValueError("mixed_or_unsupported_evidence_versions") from None


def validate_paper_input(paper_input: dict) -> None:
    if not isinstance(paper_input, dict):
        raise ValueError("paper_input_must_be_object")
    version = paper_input.get("schema_version")
    if version not in {"paper-evidence-input/v3", "paper-evidence-input/v4", "paper-evidence-input/v5",
                       "paper-evidence-input/v6"}:
        raise ValueError("unsupported_paper_input_version")
    schema_path = ROOT / "templates" / f"paper_evidence_input_{version.rsplit('/', 1)[1]}.schema.json"
    jsonschema.validate(paper_input, read_json(schema_path))


def validate_layout_bundle(layout: dict, reading_text: str, paper_input: dict, *, pdf_path=None,
                           baseline_input: dict | None = None, middle_json: dict | None = None,
                           parser_identity: dict | None = None) -> list[str]:
    """Run the exact validator for the matching layout/input pair."""
    version = evidence_version(layout, paper_input)
    try:
        validate_paper_input(paper_input)
    except jsonschema.ValidationError as exc:
        return ["input_schema:" + exc.message]
    if version == "v6":
        if baseline_input is None or middle_json is None or parser_identity is None or pdf_path is None:
            return ["v6_external_origins_required"]
        from .common import file_digest
        from .mineru_adapter import validate_mineru_bundle
        return validate_mineru_bundle(baseline_input, middle_json, parser_identity,
                                     layout, reading_text, paper_input,
                                     pdf_sha256=file_digest(pdf_path))
    if version == "v3":
        from ..paper_layout_evidence import validate_layout_bundle as validate_v3
        return validate_v3(layout, reading_text, paper_input)
    if version == "v5":
        schema = read_json(ROOT / "templates/paper_evidence_layout_v5.schema.json")
        schema_errors = ["layout_schema:" + error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(layout)]
        if schema_errors:
            return schema_errors
        from ..paper_layout_evidence_v5 import validate_layout_bundle_v5
        return validate_layout_bundle_v5(layout, reading_text, paper_input, pdf_path=pdf_path)
    from ..paper_layout_evidence_v4 import validate_layout_bundle_v4
    return validate_layout_bundle_v4(layout, reading_text, paper_input)
