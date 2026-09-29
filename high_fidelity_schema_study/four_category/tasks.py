"""Provider-independent task contracts and reproducible prompt rendering."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, asdict

import jsonschema

from .common import ROOT, canonical_bytes, digest, read_json, seal, seal_errors, taxonomy
from .paper import categorized_input, source_identity, unit_catalog
from .evidence_schema import validate_paper_input


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    system_prompt: str
    user_template: str
    output_schema: dict
    taxonomy: dict
    schema_version: str = "four-category-task/v1"
    admission_rules_version: str = "four-category-admission/v1"

    def to_dict(self) -> dict:
        return seal(asdict(self), "task_sha256")


def load_task(kind: str, *, taxonomy_value: dict | None = None) -> dict:
    if kind not in {"classification", "extraction"}:
        raise ValueError("unknown task kind")
    prefix, suffix, schema = ("four_category_classification", "v1", "paper_category_response_v1.schema.json") if kind == "classification" else ("paper_to_schema", "v4", "paper_derived_schema_v4.schema.json")
    return TaskSpec(kind,
                    (ROOT / f"templates/{prefix}_system_{suffix}.txt").read_text(encoding="utf-8"),
                    (ROOT / f"templates/{prefix}_user_{suffix}.txt").read_text(encoding="utf-8"),
                    read_json(ROOT / "templates" / schema), copy.deepcopy(taxonomy_value or taxonomy())).to_dict()


def task_errors(task: dict) -> list[str]:
    errors = seal_errors(task, "task_sha256")
    if errors:
        return errors
    if task.get('schema_version') == 'four-category-task/v13r2' and task.get('kind') == 'extraction':
        from .extraction_v13r2 import task_errors as v13r2_errors
        return v13r2_errors(task)
    if task.get('schema_version') == 'four-category-task/v13':
        from .extraction_v13 import task_errors as v13_errors
        return v13_errors(task)
    if task.get('schema_version') == 'four-category-task/v10':
        from .extraction_v10 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v10_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v10_task']
    if task.get('schema_version') == 'four-category-task/v9':
        from .extraction_v9 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v9_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v9_task']
    if task.get('schema_version') == 'four-category-task/v8':
        from .extraction_v8 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v8_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v8_task']
    if task.get('schema_version') == 'four-category-task/v7':
        from .extraction_v7 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v7_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v7_task']
    if task.get('schema_version') == 'four-category-task/v6':
        from .extraction_v6 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v6_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v6_task']
    if task.get('schema_version') == 'four-category-task/v5':
        from .extraction_v5 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v5_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v5_task']
    if task.get('schema_version') == 'four-category-task/v13r2' and task.get('kind') == 'classification':
        from .classification_v13r2 import task_errors as classification_task_errors
        return classification_task_errors(task)
    if task.get('schema_version') == 'four-category-task/v4':
        from .classification_v3 import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_classification_v3_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_classification_v3_task']
    if task.get('schema_version') == 'four-category-task/v3':
        from .extraction_view import make_task
        try:
            return [] if task == make_task(task['taxonomy']) else ['unsupported_extraction_v2_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_extraction_v2_task']
    if task.get('schema_version') == 'four-category-task/v2':
        from .classification_v2 import make_task
        try:
            expected = make_task(task['taxonomy'])
            return [] if task == expected else ['unsupported_classification_v2_task']
        except (KeyError, TypeError, ValueError):
            return ['invalid_classification_v2_task']
    kind = task.get("kind")
    if kind not in {"classification", "extraction"}:
        return ["unsupported_task_kind"]
    template = task.get("user_template")
    if not isinstance(template, str) or not isinstance(task.get("system_prompt"), str):
        return ["task_prompts_must_be_strings"]
    expected = {"TAXONOMY_JSON", "TARGET_SCHEMA_JSON", "SOURCE_IDENTITY_JSON"}
    expected |= {"PAPER_INPUT_JSON", "UNIT_CATALOG_JSON"} if kind == "classification" else {"CATEGORIZED_INPUT_JSON"}
    found = re.findall(r"\{\{([A-Z_]+)\}\}", template)
    if set(found) != expected or len(found) != len(expected):
        errors.append("task_marker_contract_invalid")
    return errors


def render_task(task: dict, paper_input: dict, index: dict | None = None) -> list[dict]:
    if task_errors(task):
        raise ValueError("task identity mismatch")
    if task.get('schema_version') == 'four-category-task/v3':
        from .extraction_view import render_task_v2
        return render_task_v2(task, paper_input, index)
    # Reject unrelated extra top-level data; input itself must be a validated projection.
    validate_paper_input(paper_input)
    values = {"TAXONOMY_JSON": task["taxonomy"], "TARGET_SCHEMA_JSON": task["output_schema"],
              "SOURCE_IDENTITY_JSON": source_identity(paper_input)}
    if task["kind"] == "classification":
        values["PAPER_INPUT_JSON"] = paper_input
        values["UNIT_CATALOG_JSON"] = [{k: v for k, v in unit.items() if k != "text"} for unit in unit_catalog(paper_input).values()]
    elif task["kind"] == "extraction":
        if index is None:
            raise ValueError("extraction requires frozen automatic index")
        values["CATEGORIZED_INPUT_JSON"] = categorized_input(paper_input, index, taxonomy_value=task["taxonomy"])
    else:
        raise ValueError("unknown task kind")
    template = task["user_template"]
    for name, value in values.items():
        marker = "{{" + name + "}}"
        if template.count(marker) != 1:
            raise ValueError(f"prompt marker must occur exactly once: {name}")
    # Inspect the template rather than the paper; paper text can legitimately contain braces.
    import re
    if set(re.findall(r"\{\{([A-Z_]+)\}\}", task["user_template"])) != set(values):
        raise ValueError("unexpected task prompt marker")
    rendered = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: canonical_bytes(values[m[1]]).decode("utf-8"), template)
    return [{"role": "system", "content": task["system_prompt"]}, {"role": "user", "content": rendered}]


def request_identity(task: dict, messages: list[dict]) -> dict:
    return {"task_sha256": task["task_sha256"], "messages_canonical_sha256": digest(messages)}
