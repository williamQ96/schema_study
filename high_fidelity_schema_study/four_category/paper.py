"""Full-paper input, automatic index admission and extraction validation.

Semantic correctness is deliberately not inferred from a passing locator gate.
Evidence source identity is bound to an exact versioned layout/input pair.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import jsonschema

from ..paper_layout_evidence import resolve_evidence_quote
from ..paper_layout_evidence_v3 import word_coverage_audit
from .evidence_schema import evidence_version, validate_layout_bundle
from .common import ROOT, CATEGORIES, contained, digest, file_digest, identity, read_json, seal, seal_errors, strict_json, taxonomy


def source_identity(paper_input: dict) -> dict:
    return {"document_id": paper_input["paper_id"], "paper_sha256": paper_input["source_pdf_sha256"],
            "preprocessing_sha256": paper_input["layout_preprocessing_sha256"]}


def unit_catalog(paper_input: dict) -> dict[str, dict]:
    units = {}
    for page in paper_input["pages"]:
        items = [(u, u.get("region_kind", "paragraph")) for u in page["text_regions"]]
        for table in page["tables"]:
            items.append((table["caption"], "table_caption"))
            items.extend((u, "table_header") for u in table.get("header_units", []))
            for row in table["rows"]:
                items.append((row, "table_row"))
                items.extend((u, "table_cell") for u in row["cells"])
        for unit, kind in items:
            uid = unit["unit_id"]
            if uid in units:
                raise ValueError(f"duplicate unit: {uid}")
            units[uid] = {"unit_id": uid, "page": page["page"], "source_kind": kind, "text": unit["text"]}
    return units


def prepare_paper(layout_path: Path, reading_path: Path, input_path: Path, pdf_path: Path, *, root: Path,
                  ancestor_layout_path: Path | None = None, mineru_raw_path: Path | None = None,
                  baseline_input_path: Path | None = None,
                  parser_identity_path: Path | None = None) -> dict:
    paper_input = read_json(input_path)
    if not isinstance(paper_input, dict):
        raise ValueError("paper_input_must_be_object")
    paths = {"layout": layout_path, "reading_text": reading_path, "input": input_path, "pdf": pdf_path}
    if paper_input.get("schema_version") == "paper-evidence-input/v5":
        if ancestor_layout_path is None:
            raise ValueError("v5_source_requires_bound_v4_ancestor_layout")
        paths["spatial_ancestor_v4"] = ancestor_layout_path
    if paper_input.get("schema_version") == "paper-evidence-input/v6":
        if any(path is None for path in (mineru_raw_path, baseline_input_path, parser_identity_path)):
            raise ValueError("v6_source_requires_bound_external_origins")
        paths.update(mineru_raw=mineru_raw_path, baseline_input=baseline_input_path,
                     parser_identity=parser_identity_path)
    artifacts = {name: identity(Path(path), root) for name, path in paths.items()}
    value = seal({"schema_version": "four-category-paper-source/v1", "source_identity": source_identity(paper_input),
                  "artifacts": artifacts}, "source_sha256")
    errors = verify_paper(value, root)
    if errors:
        raise ValueError(errors)
    return value


def verify_paper(source: dict, root: Path) -> list[str]:
    errors = seal_errors(source, "source_sha256")
    try:
        input_path = contained(root, source["artifacts"]["input"]["path"])
        paper_input = read_json(input_path)
        if not isinstance(paper_input, dict):
            raise ValueError("paper_input_must_be_object")
        names = ["layout", "reading_text", "input", "pdf"]
        if paper_input.get("schema_version") == "paper-evidence-input/v5":
            names.append("spatial_ancestor_v4")
        if paper_input.get("schema_version") == "paper-evidence-input/v6":
            names.extend(("mineru_raw", "baseline_input", "parser_identity"))
            if set(source["artifacts"]) != set(names):
                errors.append("v6_source_artifact_set_mismatch")
        paths = {}
        for name in names:
            item = source["artifacts"][name]
            path = contained(root, item["path"])
            paths[name] = path
            if item.get("hash_mode") != "file_bytes" or file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                errors.append(f"{name}:file_identity_mismatch")
        layout = read_json(paths["layout"])
        reading = paths["reading_text"].read_text(encoding="utf-8")
        version = evidence_version(layout, paper_input)
        if file_digest(paths["pdf"]) != layout["source_pdf_sha256"]:
            errors.append("pdf_layout_identity_mismatch")
        if source_identity(paper_input) != source["source_identity"]:
            errors.append("source_identity_mismatch")
        if version == "v6":
            baseline_input = read_json(paths["baseline_input"])
            parser_identity = read_json(paths["parser_identity"])
            if not isinstance(baseline_input, dict):
                errors.append("v6_baseline_input_must_be_object")
            elif (baseline_input.get("source_pdf_sha256") != paper_input.get("source_pdf_sha256")
                  or baseline_input.get("paper_id") != paper_input.get("paper_id")):
                errors.append("v6_baseline_origin_identity_mismatch")
            if (not isinstance(parser_identity, dict)
                    or parser_identity.get("raw_file_bytes_sha256") != file_digest(paths["mineru_raw"])):
                errors.append("v6_parser_raw_file_identity_mismatch")
            errors.extend(validate_layout_bundle(layout, reading, paper_input, pdf_path=paths["pdf"],
                                                 baseline_input=baseline_input,
                                                 middle_json=read_json(paths["mineru_raw"]),
                                                 parser_identity=parser_identity))
        else:
            errors.extend(validate_layout_bundle(layout, reading, paper_input,
                                                 pdf_path=paths["pdf"] if version == "v5" else None))
        if version == "v5":
            errors.extend(_v5_ancestor_errors(layout, read_json(paths["spatial_ancestor_v4"])))
        if version == "v3" and word_coverage_audit(layout)["status"] != "pass":
            errors.append("source_word_coverage_failed")
        unit_catalog(paper_input)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"paper_source:{exc}")
    return errors


def _v5_ancestor_errors(layout: dict, ancestor: dict) -> list[str]:
    """Bind a v5 visibility projection to the actual, validated v4 artifact."""
    errors = []
    try:
        jsonschema.validate(ancestor, read_json(ROOT / "templates/paper_evidence_layout_v4.schema.json"))
        if ancestor.get("schema_version") != "paper-evidence-layout/v4":
            return ["v5_ancestor_not_v4"]
        if ancestor.get("source_pdf_sha256") != layout.get("source_pdf_sha256"):
            errors.append("v5_ancestor_pdf_mismatch")
        if ancestor.get("paper_id") != layout.get("paper_id"):
            errors.append("v5_ancestor_paper_mismatch")
        if ancestor.get("preprocessing_sha256") != layout.get("derived_from_v4_preprocessing_sha256"):
            errors.append("v5_ancestor_preprocessing_mismatch")
        if len(ancestor.get("pages", [])) != len(layout.get("pages", [])):
            errors.append("v5_ancestor_page_count_mismatch")
        else:
            for old_page, new_page in zip(ancestor["pages"], layout["pages"], strict=True):
                if old_page.get("source_words") != new_page.get("source_words"):
                    errors.append(f"v5_ancestor_source_words_mismatch:p{new_page.get('page')}")
        from ..paper_layout_evidence import assign_document_spans, sha256_bytes, canonical_json_bytes
        from ..paper_layout_evidence_v4 import compact_model_input_v4, validate_layout_bundle_v4
        body = dict(ancestor)
        recorded = body.pop("preprocessing_sha256", None)
        if recorded != sha256_bytes(canonical_json_bytes(body)):
            errors.append("v5_ancestor_hash_mismatch")
        ancestor_pages = copy.deepcopy(ancestor["pages"])
        ancestor_reading = assign_document_spans(ancestor_pages)
        if ancestor_pages != ancestor["pages"]:
            errors.append("v5_ancestor_span_derivation_mismatch")
        errors.extend(validate_layout_bundle_v4(ancestor, ancestor_reading, compact_model_input_v4(ancestor)))
    except jsonschema.ValidationError as exc:
        errors.append("v5_ancestor_schema:" + exc.message)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        errors.append("v5_ancestor_validation:" + str(exc))
    return errors


def classification_errors(response: dict, paper_input: dict) -> list[str]:
    schema = read_json(ROOT / "templates/paper_category_response_v1.schema.json")
    errors = ["classification_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(response)]
    if errors:
        return errors
    if response["source_identity"] != source_identity(paper_input):
        errors.append("classification_source_identity_mismatch")
    catalog = unit_catalog(paper_input)
    ids = [e["unit_id"] for e in response["entries"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_classification_unit")
    if set(ids) != set(catalog):
        errors.append("classification_unit_set_mismatch")
    for entry in response["entries"]:
        unit = catalog.get(entry["unit_id"])
        if not unit:
            continue
        if entry["page"] != unit["page"]:
            errors.append("classification_page_mismatch:" + entry["unit_id"])
        if entry["state"] == "classified":
            if not entry["categories"] or not entry["evidence_spans"]:
                errors.append("classified_entry_requires_categories_and_evidence")
        elif entry["categories"]:
            errors.append("noncategory_state_has_categories")
        for span in entry["evidence_spans"]:
            if not (0 <= span["start"] < span["end"] <= len(unit["text"])) or unit["text"][span["start"]:span["end"]] != span["quote"]:
                errors.append("classification_span_quote_mismatch:" + entry["unit_id"])
    return errors


def build_index(raw_text: str, paper_input: dict, producer: dict, *, taxonomy_value: dict | None = None) -> dict:
    response = strict_json(raw_text)
    errors = classification_errors(response, paper_input)
    if errors:
        raise ValueError(errors)
    for key in ("run_id", "profile_sha256", "task_sha256", "request_sha256"):
        if not producer.get(key):
            raise ValueError(f"index producer requires {key}")
    order = {uid: i for i, uid in enumerate(unit_catalog(paper_input))}
    return seal({"schema_version": "paper-category-index/v1", "taxonomy_sha256": digest(taxonomy_value or taxonomy()),
                 "source_identity": source_identity(paper_input), "paper_input_canonical_sha256": digest(paper_input),
                 "entries": sorted(response["entries"], key=lambda x: order[x["unit_id"]]),
                 "raw_response": raw_text, "producer": copy.deepcopy(producer),
                 "authority": "automated_fallible_navigation_not_gold", "semantic_review": "not_established"}, "index_sha256")


def verify_index(index: dict, paper_input: dict, *, taxonomy_value: dict | None = None) -> list[str]:
    if index.get('schema_version') == 'paper-category-index/disabled-v1':
        from .disabled_navigation import verify_disabled_index
        return verify_disabled_index(index, paper_input, taxonomy_value)
    if index.get('schema_version') == 'paper-category-index/v4':
        from .classification_navigation_v4 import verify_index_v4
        from .replay_cache import cached_index_replay
        return cached_index_replay(verify_index_v4, index, paper_input, taxonomy_value)
    if index.get('schema_version') == 'paper-category-index/v3':
        from .classification_v3 import verify_index_v3
        return verify_index_v3(index, paper_input, taxonomy_value=taxonomy_value)
    if index.get('schema_version') == 'paper-category-index/v2':
        from .classification_v2 import verify_index_v2
        return verify_index_v2(index, paper_input, taxonomy_value=taxonomy_value)
    errors = seal_errors(index, "index_sha256")
    try:
        replayed = build_index(index["raw_response"], paper_input, index["producer"], taxonomy_value=taxonomy_value)
        if replayed != index:
            errors.append("classification_raw_derivation_mismatch")
    except (ValueError, TypeError, KeyError, jsonschema.SchemaError) as exc:
        errors.append(f"classification_replay:{exc}")
    return errors


def categorized_input(paper_input: dict, index: dict, *, taxonomy_value: dict | None = None) -> dict:
    tax = taxonomy_value or taxonomy()
    errors = verify_index(index, paper_input, taxonomy_value=tax)
    if errors:
        raise ValueError(errors)
    return seal({"schema_version": "paper-categorized-input/v1", "taxonomy_sha256": digest(tax),
                 "paper_input": copy.deepcopy(paper_input),
                 "category_index": {"index_sha256": index["index_sha256"], "entries": copy.deepcopy(index["entries"]),
                                    "authority": "automated_fallible_navigation_not_gold"}}, "categorized_input_sha256")


def verify_categorized(value: dict, paper_input: dict, index: dict, *, taxonomy_value: dict | None = None) -> list[str]:
    try:
        return [] if value == categorized_input(paper_input, index, taxonomy_value=taxonomy_value) else ["categorized_input_derivation_mismatch"]
    except (ValueError, TypeError, KeyError) as exc:
        return [f"categorized_input:{exc}"]


def pointer(value: Any, path: str) -> Any:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("invalid JSON pointer")
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not part.isdecimal() or str(int(part)) != part:
                raise ValueError("invalid array index")
            value = value[int(part)]
        else:
            value = value[part]
    return value


def assertion_pointers(claim: dict) -> list[str]:
    targets = ["/reported_name"]
    for name in ("parent_claim_id", "canonical_path"):
        if claim.get(name) is not None:
            targets.append("/" + name)
    for name in ("datatype", "shape_or_dimensions"):
        if claim[name]["status"] != "unknown":
            targets.append("/" + name)
    for name, value in claim["semantic_annotations"].items():
        if value["status"] != "unknown":
            targets.append("/semantic_annotations/" + name)
    for name in ("constraints", "relationships", "encoding", "syntax"):
        targets.extend(f"/{name}/{i}" for i, value in enumerate(claim[name]) if value["status"] != "unknown")
    return targets


def extraction_errors(payload: dict, paper_input: dict, *, layout: dict | None = None, reading_text: str | None = None) -> list[str]:
    schema = read_json(ROOT / "templates/paper_derived_schema_v4.schema.json")
    errors = ["extraction_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
    if errors:
        return errors
    if payload["source_document"] != source_identity(paper_input):
        errors.append("extraction_source_identity_mismatch")
    ids = [c["claim_id"] for c in payload["claims"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_claim_id")
    catalog = unit_catalog(paper_input)
    parents = {c["claim_id"]: c["parent_claim_id"] for c in payload["claims"]}
    for cid in parents:
        seen, parent = {cid}, parents[cid]
        while parent in parents:
            if parent in seen:
                errors.append(f"{cid}:parent_cycle")
                break
            seen.add(parent)
            parent = parents[parent]
    for claim in payload["claims"]:
        cid = claim["claim_id"]
        if claim["parent_claim_id"] is not None and claim["parent_claim_id"] not in ids:
            errors.append(f"{cid}:invalid_parent")
        if any(r["target_claim_id"] not in ids for r in claim["relationships"]):
            errors.append(f"{cid}:invalid_relationship_target")
        evs = {e["evidence_id"]: e for e in claim["evidence"]}
        if len(evs) != len(claim["evidence"]):
            errors.append(f"{cid}:duplicate_evidence_id")
        for ev in evs.values():
            unit = catalog.get(ev["unit_id"])
            if not unit or ev["document_id"] != paper_input["paper_id"] or ev["page"] != unit["page"]:
                errors.append(f"{cid}:invalid_evidence_identity")
                continue
            if ev["source_kind"] != unit["source_kind"]:
                errors.append(f"{cid}:evidence_kind_mismatch")
            if layout is not None and reading_text is not None:
                if resolve_evidence_quote(layout, reading_text, ev["unit_id"], ev["quote_or_cell_text"])["status"] != "resolved":
                    errors.append(f"{cid}:unresolved_quote")
            elif ev["quote_or_cell_text"] not in unit["text"]:
                errors.append(f"{cid}:quote_not_found")
            if ev["supports"] != "whole_claim":
                try:
                    pointer(claim, ev["supports"])
                except (ValueError, KeyError, IndexError, TypeError):
                    errors.append(f"{cid}:unresolved_support_pointer")
        required = set(assertion_pointers(claim))
        targets = []
        for ann in claim["category_annotations"]:
            target = ann["target_pointer"]
            targets.append(target)
            if target not in required:
                errors.append(f"{cid}:category_target_is_not_asserted_fact:{target}")
            for eid in ann["evidence_ids"]:
                if eid not in evs or evs[eid]["supports"] != target:
                    errors.append(f"{cid}:category_evidence_target_mismatch")
        if len(targets) != len(set(targets)):
            errors.append(f"{cid}:duplicate_category_target")
        if set(targets) != required:
            errors.append(f"{cid}:assertion_category_coverage_mismatch")
        qualified = [claim["datatype"], claim["shape_or_dimensions"], *claim["semantic_annotations"].values(), *claim["encoding"], *claim["syntax"]]
        for value in qualified:
            if (value["status"] == "unknown") != (value["value"] is None):
                errors.append(f"{cid}:qualified_status_value_conflict")
        if claim["claim_state"] == "inferred" and not claim["inference_basis"]:
            errors.append(f"{cid}:inference_basis_required")
    return errors


def fact_view(payload: dict) -> list[dict]:
    """Unique assertions; overlapping categories never multiply the global count."""
    if payload.get('schema_version') in {'paper-derived-observations/v6', 'paper-derived-observations/v7',
                                        'paper-derived-observations/v8', 'paper-derived-observations/v9',
                                        'paper-derived-observations/v10', 'paper-derived-observations/v13',
                                        'paper-derived-observations/v13r2'}:
        facts = []
        for fact in payload['facts']:
            claim = fact['claim']
            assertion = claim['assertion']
            state = assertion['status']
            facts.append({**fact, 'subject_id': fact['subject_mention_id'],
                          'predicate': claim['predicate'], 'value': assertion['value'],
                          'basis': 'declared' if state == 'reported' else state,
                          'source_state': state, 'inference_basis': assertion['basis'],
                          'target': claim.get('target'), 'origin': 'paper',
                          'basis_authority': 'model_asserted_not_semantically_adjudicated',
                          'identity_resolution': 'unresolved_model_hypotheses_only'})
        return facts
    if payload.get('schema_version') == 'paper-derived-observations/v5':
        # This is a projection of an independently verified bundle, not an
        # admission function and not an object-identity resolution step.
        return [{**fact, 'subject_id': fact['subject_mention_id'],
                 'basis': 'declared' if fact['status'] == 'reported' else fact['status'],
                 'source_state': fact['status'], 'origin': 'paper',
                 'basis_authority': 'model_asserted_not_semantically_adjudicated',
                 'identity_resolution': 'unresolved_model_hypotheses_only'} for fact in payload['facts']]
    facts = []
    for claim in payload["claims"]:
        for ann in claim["category_annotations"]:
            target = ann["target_pointer"]
            raw = pointer(claim, target)
            state = raw.get("status", claim["claim_state"]) if isinstance(raw, dict) else claim["claim_state"]
            facts.append({"fact_id": claim["claim_id"] + "#" + target, "subject_id": claim["claim_id"],
                          "predicate": target, "value": raw.get("value", raw) if isinstance(raw, dict) else raw,
                          "categories": ann["categories"], "basis": "declared" if state in {"supported", "reported"} else "unknown" if state == "conflicted" else state,
                          "source_state": state,
                          "basis_authority": "model_asserted_not_semantically_adjudicated",
                          "evidence_ids": ann["evidence_ids"], "origin": "paper"})
    return facts


def category_counts(facts: list[dict]) -> dict:
    unique = {f["fact_id"]: f for f in facts}
    if len(unique) != len(facts):
        raise ValueError("duplicate fact identity")
    return {"unique_facts": len(unique), "categories": {c: sum(c in f["categories"] for f in facts) for c in CATEGORIES},
            "category_counts_overlap": True}
