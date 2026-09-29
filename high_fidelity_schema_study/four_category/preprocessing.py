"""Frozen, independently switched paper parsing and structural navigation condition."""
from __future__ import annotations

import copy
from pathlib import Path

from . import extraction_v10, structural_classifier
from .backends import profile_hash, validate_profile
from .common import canonical_bytes, contained, digest, read_json, seal, seal_errors
from .disabled_navigation import make_disabled_index
from .paper import source_identity, unit_catalog, verify_paper


CONFIG_VERSION = "paper-preprocessing-condition-config/v1"
PACKAGE_VERSION = "paper-preprocessing-condition/v1"
TASK_VERSION = "paper-structure-aware-extraction-task/v1"
PARSER_PROFILE_KEYS = {"version", "tier", "small_backend", "model_manifest_sha256"}
COMMON_INSTRUCTIONS = (
    "Extract schema observations from the primary source using the frozen exact-quote contract. "
    "The structural annotations below are fallible navigation hints. They are not evidence, "
    "do not establish semantic correctness, and cannot replace an exact primary source quote. "
    "Use the source-role metadata only to interpret local reading order and table topology."
)


def make_config(*, mineru_enabled: bool, auxiliary_enabled: bool,
                parser_profile: dict | None = None, auxiliary_profile: dict | None = None) -> dict:
    if type(mineru_enabled) is not bool or type(auxiliary_enabled) is not bool:
        raise ValueError("preprocessing_switches_must_be_booleans")
    if mineru_enabled:
        if (not isinstance(parser_profile, dict) or set(parser_profile) != PARSER_PROFILE_KEYS
                or any(not isinstance(parser_profile[key], str) or not parser_profile[key]
                       for key in PARSER_PROFILE_KEYS)):
            raise ValueError("enabled_mineru_requires_frozen_parser_profile")
    elif parser_profile is not None:
        raise ValueError("disabled_mineru_rejects_parser_profile")
    if auxiliary_enabled:
        if not isinstance(auxiliary_profile, dict) or validate_profile(auxiliary_profile):
            raise ValueError("enabled_auxiliary_requires_valid_profile")
    elif auxiliary_profile is not None:
        raise ValueError("disabled_auxiliary_rejects_profile")
    return seal({"schema_version": CONFIG_VERSION, "mineru_enabled": mineru_enabled,
                 "auxiliary_enabled": auxiliary_enabled,
                 "parser_profile": copy.deepcopy(parser_profile),
                 "auxiliary_profile": copy.deepcopy(auxiliary_profile),
                 "auxiliary_profile_sha256": profile_hash(auxiliary_profile) if auxiliary_enabled else None},
                "config_sha256")


def _config_errors(config: dict) -> list[str]:
    errors = seal_errors(config, "config_sha256")
    try:
        expected = make_config(mineru_enabled=config["mineru_enabled"],
                               auxiliary_enabled=config["auxiliary_enabled"],
                               parser_profile=config["parser_profile"],
                               auxiliary_profile=config["auxiliary_profile"])
        if config != expected:
            errors.append("preprocessing_config_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("preprocessing_config_invalid:" + str(exc))
    return errors


def make_extraction_task() -> dict:
    return seal({"schema_version": TASK_VERSION, "kind": "extraction",
                 "base_task": extraction_v10.make_task(),
                 "common_instructions": COMMON_INSTRUCTIONS,
                 "navigation_protocol": "disabled-category-navigation/v1",
                 "hint_authority": "fallible_auxiliary_navigation_not_schema_truth"}, "task_sha256")


def _task(task: dict | None) -> dict:
    expected = make_extraction_task()
    if task is not None and task != expected:
        raise ValueError("preprocessing_extraction_task_mismatch")
    return expected


def _input(root: Path, source: dict) -> dict:
    return read_json(contained(root, source["artifacts"]["input"]["path"]))


def _parser_identity(root: Path, source: dict) -> dict:
    return read_json(contained(root, source["artifacts"]["parser_identity"]["path"]))


def _condition_errors(root: Path, baseline_source: dict, config: dict,
                      mineru_source: dict | None, annotations: dict | None) -> tuple[list[str], dict | None]:
    errors = _config_errors(config)
    if errors:
        return errors, None
    errors.extend("baseline:" + issue for issue in verify_paper(baseline_source, root))
    if errors:
        return errors, None
    baseline_input = _input(root, baseline_source)
    if baseline_input.get("schema_version") == "paper-evidence-input/v6":
        errors.append("baseline_must_precede_mineru_v6")
    if config["mineru_enabled"]:
        if mineru_source is None:
            errors.append("enabled_mineru_source_required")
        else:
            errors.extend("mineru:" + issue for issue in verify_paper(mineru_source, root))
            if not errors:
                selected = _input(root, mineru_source)
                if selected.get("schema_version") != "paper-evidence-input/v6":
                    errors.append("mineru_source_must_be_v6")
                if (source_identity(selected)["document_id"] != source_identity(baseline_input)["document_id"]
                        or source_identity(selected)["paper_sha256"] != source_identity(baseline_input)["paper_sha256"]):
                    errors.append("mineru_baseline_paper_identity_mismatch")
                raw_baseline = read_json(contained(root, mineru_source["artifacts"]["baseline_input"]["path"]))
                if raw_baseline != baseline_input:
                    errors.append("mineru_bound_baseline_input_mismatch")
                identity = _parser_identity(root, mineru_source)
                actual_profile = {key: identity.get(key) for key in PARSER_PROFILE_KEYS}
                if actual_profile != config["parser_profile"]:
                    errors.append("mineru_parser_profile_mismatch")
    elif mineru_source is not None:
        errors.append("disabled_mineru_rejects_source")
    selected_input = None if errors else _input(root, mineru_source if config["mineru_enabled"] else baseline_source)
    if config["auxiliary_enabled"]:
        if annotations is None:
            errors.append("enabled_auxiliary_annotations_required")
        elif not isinstance(annotations, dict):
            errors.append("auxiliary_annotations_must_be_object")
        elif selected_input is not None:
            if annotations.get("task") != structural_classifier.make_task():
                errors.append("auxiliary_task_mismatch")
            if annotations.get("profile") != config["auxiliary_profile"] or annotations.get("profile_sha256") != config["auxiliary_profile_sha256"]:
                errors.append("auxiliary_profile_mismatch")
            errors.extend("auxiliary:" + issue for issue in structural_classifier.verify_annotations(
                annotations, selected_input, task=structural_classifier.make_task(),
                profile=config["auxiliary_profile"]))
    elif annotations is not None:
        errors.append("disabled_auxiliary_rejects_annotations")
    return errors, selected_input


def prepare_condition(root: Path, baseline_source: dict, config: dict, *,
                      mineru_source: dict | None = None, annotations: dict | None = None) -> dict:
    root = Path(root)
    errors, selected = _condition_errors(root, baseline_source, config, mineru_source, annotations)
    if errors:
        raise ValueError(errors)
    return seal({"schema_version": PACKAGE_VERSION, "config": copy.deepcopy(config),
                 "baseline_source": copy.deepcopy(baseline_source),
                 "mineru_source": copy.deepcopy(mineru_source),
                 "annotations": copy.deepcopy(annotations),
                 "selected_input_sha256": digest(selected),
                 "source_identity": source_identity(selected),
                 "navigation_index_sha256": make_disabled_index(selected)["index_sha256"],
                 "task_sha256": make_extraction_task()["task_sha256"]}, "package_sha256")


def verify_condition(package: dict, root: Path) -> list[str]:
    errors = seal_errors(package, "package_sha256")
    try:
        expected = prepare_condition(root, package["baseline_source"], package["config"],
                                     mineru_source=package["mineru_source"],
                                     annotations=package["annotations"])
        if package != expected:
            errors.append("preprocessing_condition_derivation_mismatch")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append("preprocessing_condition_replay:" + str(exc))
    return errors


def _context(package: dict, root: Path, group: dict, task: dict | None):
    errors = verify_condition(package, root)
    if errors:
        raise ValueError(errors)
    frozen = _task(task)
    selected_source = package["mineru_source"] if package["config"]["mineru_enabled"] else package["baseline_source"]
    selected = _input(Path(root), selected_source)
    index = make_disabled_index(selected)
    errors = extraction_v10.group_errors(group, selected)
    if errors:
        raise ValueError(errors)
    return frozen, selected, index


def _local_metadata(paper_input: dict, group: dict) -> list[dict]:
    windows = extraction_v10.window_catalog(paper_input)
    units = {}
    table_ids = {}
    for page in paper_input["pages"]:
        for unit in page["text_regions"]:
            units[unit["unit_id"]] = unit
        for table in page["tables"]:
            tid = table["table_id"]
            for unit in [table["caption"], *table.get("header_units", [])]:
                units[unit["unit_id"]] = unit
                table_ids[unit["unit_id"]] = tid
            for row in table["rows"]:
                units[row["unit_id"]] = row
                table_ids[row["unit_id"]] = tid
                for cell in row["cells"]:
                    units[cell["unit_id"]] = cell
                    table_ids[cell["unit_id"]] = tid
    return [{"window_id": wid, "unit_id": windows[wid]["unit_id"],
             "region_kind": units[windows[wid]["unit_id"]].get("region_kind", windows[wid]["source_kind"]),
             "text_origin": units[windows[wid]["unit_id"]].get("text_origin"),
             "structure_status": units[windows[wid]["unit_id"]].get("structure_status"),
             "table_id": table_ids.get(windows[wid]["unit_id"])} for wid in group["window_ids"]]


def _hints(package: dict, paper_input: dict, group: dict) -> dict:
    if not package["config"]["auxiliary_enabled"]:
        return {"status": "disabled", "authority": "no_auxiliary_classifier_generation", "group_outcomes": [],
                "annotations": []}
    bundle = package["annotations"]
    windows = extraction_v10.window_catalog(paper_input)
    target = {windows[wid]["unit_id"] for wid in group["window_ids"]}
    outcomes = [row for row in bundle["group_outcomes"] if target.intersection(row["unit_ids"])]
    ids = {row["group_id"] for row in outcomes}
    hints = [row for row in bundle["annotations"] if row["group_id"] in ids]
    return {"status": "available" if hints else "unavailable", "authority": bundle["authority"],
            "group_outcomes": copy.deepcopy(outcomes), "annotations": copy.deepcopy(hints)}


def render_group(package: dict, root: Path, group: dict, *, task: dict | None = None) -> list[dict]:
    frozen, paper_input, index = _context(package, root, group, task)
    messages = extraction_v10.render_group(frozen["base_task"], paper_input, index, group)
    additional = {"diagnostic_binding": {"package_sha256": package["package_sha256"],
                                          "task_sha256": frozen["task_sha256"]},
                  "local_source_roles": _local_metadata(paper_input, group),
                  "structural_hints": _hints(package, paper_input, group)}
    messages[-1]["content"] += "\n\nSTRUCTURAL PREPROCESSING CONTEXT (advisory):\n" + frozen["common_instructions"] + "\n" + canonical_bytes(additional).decode("utf-8")
    return messages


def response_schema(package: dict, root: Path, group: dict, task: dict | None = None) -> dict:
    frozen, paper_input, index = _context(package, root, group, task)
    return extraction_v10.response_schema(frozen["base_task"], paper_input, index, group)


def admit_response(raw: str, package: dict, root: Path, group: dict,
                   task: dict | None = None) -> dict:
    frozen, paper_input, index = _context(package, root, group, task)
    try:
        payload, normalization = extraction_v10.parse_response(raw)
        errors = extraction_v10.validate_response(payload, paper_input, index, group,
                                                  task=frozen["base_task"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "contract_invalid", "payload": None,
                "normalization": None, "validation_errors": ["response_parse:" + str(exc)]}
    status = "contract_invalid" if errors else extraction_v10.completion_status(payload, group)
    return {"status": status, "payload": payload, "normalization": normalization,
            "validation_errors": errors}
