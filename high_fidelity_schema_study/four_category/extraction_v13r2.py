"""V13 repair candidate: direct, typed page-window evidence references.

The frozen V13 module remains unchanged. Groups retain complete-page
responsibility, while adjacent pages are supplied only as supporting context.
"""
from __future__ import annotations

import copy
import difflib
import json

import jsonschema

from . import extraction_v10 as v10, extraction_v12 as v12
from .common import canonical_bytes, digest, seal, seal_errors
from .paper import source_identity, verify_index

VERSION = "four-category-task/v13r2"
PROTOCOL = "extraction-page-regions/v13r2"
BUNDLE_VERSION = "paper-derived-observations/v13r2"
RESPONSE_VERSION = "paper-extraction-group-response/v8"
PROMPT_PROJECTION = "direct-typed-page-windows/compact-contract-v2"
DEFAULT_POLICY = {"max_windows": 512, "max_target_chars": 60000,
                  "max_mentions": 128, "max_facts": 256}
GUIDANCE = """Extract only claims supported by this paper. No datasets, codebooks,
expected names, or other extraction outputs are available. The target PDF page
is your extraction responsibility. Adjacent pages are supporting context only;
distant pages are not included, so explicitly mark unresolved cross-page
definitions uncertain. Cite evidence only with the exact W identifier shown
beside its complete text. Every object needs assigned-page primary evidence;
supporting windows may be from adjacent pages. Table reviews may cite only that
table's listed eligible_evidence_ids. Tables with possible duplicate text are
separate source views; never transfer evidence between their IDs. Preserve one
object when a claim appears in multiple views and cite each view's own W IDs as
appropriate; shared text does not prove locator equivalence. Per-window quote
options are paired directly with their W IDs. Preserve
wording, report uncertainty, and do not infer semantic correctness from schema
validity. Output exactly the supplied JSON schema."""
GUIDANCE += """
The displayed output contract abbreviates repeated per-window enumerations;
the decoder enforces the fully bound schema. Emit coverage once for every
primary_eligible window in source order, table_reviews once for every table in
table_candidates order, and feature_coverage once for every feature candidate
in its displayed order. A fact's primary_quote must be one of the quote_options
beside its primary W ID. Supporting pages have text but no primary quote options.
Navigation groups attach one fallible classification to the listed W IDs; they
do not remove any source text or forbid a different classification judgment.
"""


def make_task(taxonomy_value=None):
    base = v10.make_task(taxonomy_value)
    nested = v12.make_task()
    return seal({"kind": "extraction", "schema_version": VERSION,
                 "admission_rules_version": PROTOCOL, "system_prompt": nested["system_prompt"] + GUIDANCE,
                 "output_schema": base["output_schema"], "taxonomy": base["taxonomy"],
                 "v12_contract_sha256": nested["task_sha256"],
                 "grouping": "complete_pdf_pages/v1", "evidence_ids": "typed_windows/v1",
                 "context_scope": "target_page_plus_adjacent_pdf_pages/v1",
                 "prompt_projection": PROMPT_PROJECTION}, "task_sha256")


def task_errors(task):
    try:
        return [] if task == make_task(task["taxonomy"]) else ["v13r2_task_mismatch"]
    except (KeyError, TypeError, ValueError):
        return ["invalid_v13r2_task"]


def _policy(policy):
    value = copy.deepcopy(DEFAULT_POLICY if policy is None else policy)
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_POLICY)
            or any(type(v) is not int or v <= 0 for v in value.values())):
        raise ValueError("invalid_v13r2_group_policy")
    return value


def _wid(n):
    return f"W{n:06d}"


def _catalog(paper):
    return v10.window_catalog(paper)


def _maps(paper):
    rows = _catalog(paper)
    return ({_wid(r["window_index"]): r["window_index"] for r in rows},
            {r["window_index"]: _wid(r["window_index"]) for r in rows})


def plan_groups(paper_input, policy=None):
    limits = _policy(policy)
    pages = sorted({p["page"] for p in paper_input["pages"]})
    groups = []
    all_windows = _catalog(paper_input)
    for i, page in enumerate(pages):
        region = v12.plan_region(paper_input, [page], max_windows=limits["max_windows"],
                                 max_target_chars=limits["max_target_chars"],
                                 max_mentions=limits["max_mentions"], max_facts=limits["max_facts"])
        context_pages = [p for p in pages if abs(p - page) <= 1]
        context_ids = [w["window_index"] for w in all_windows if w["page"] in context_pages]
        value = {**region, "index": i, "policy": copy.deepcopy(limits),
                 "context_pages": context_pages, "context_window_ids": context_ids,
                 "context_scope": "target_plus_adjacent_pdf_pages/v1",
                 "group_id": digest({"version": PROTOCOL, "region_sha256": region["group_sha256"],
                                     "index": i, "context_window_ids": context_ids})}
        groups.append(value)
    return groups


def _region(group):
    return {k: copy.deepcopy(group[k]) for k in
            ("schema_version", "pages", "source_pdf_sha256", "paper_input_canonical_sha256",
             "window_ids", "limits", "group_sha256")}


def _check(task, paper, group):
    if task_errors(task):
        raise ValueError("v13r2_task_invalid")
    groups = plan_groups(paper, group["policy"])
    if group not in groups:
        raise ValueError("v13r2_group_derivation_mismatch")


def _context(paper, group):
    forward, reverse = _maps(paper)
    catalog = _catalog(paper)
    base = v12.context(paper, _region(group))
    tables = []
    table_unit_ids = {}
    table_eligible = {}
    table_texts = {}
    table_text_windows = {}
    table_all_text_windows = {}
    for t in base["table_candidates"]:
        eligible = sorted({reverse[w["window_index"]] for w in catalog
                           if w["unit_id"] in t["unit_ids"] and w["page"] in group["pages"]})
        table_eligible[t["table_id"]] = eligible
        table_unit_ids[t["table_id"]] = set(t["unit_ids"])
        table_texts[t["table_id"]] = "\n".join(
            catalog[forward[w]]["text"] for w in eligible)
        text_windows = {}
        all_text_windows = {}
        for wid in eligible:
            text = catalog[forward[wid]]["text"]
            all_text_windows.setdefault(text, []).append(wid)
            if len(text.strip()) >= 24:
                text_windows.setdefault(text, []).append(wid)
        table_text_windows[t["table_id"]] = text_windows
        table_all_text_windows[t["table_id"]] = all_text_windows
    for t in base["table_candidates"]:
        eligible = table_eligible[t["table_id"]]
        peers = [peer for peer in base["table_candidates"] if peer["page"] == t["page"]
                 and peer["table_id"] != t["table_id"]]
        comparisons = []
        for peer in peers:
            peer_ids = table_eligible[peer["table_id"]]
            shared = sorted(set(eligible) & set(peer_ids))
            identical_texts = sorted(set(table_all_text_windows[t["table_id"]]) & set(table_all_text_windows[peer["table_id"]]))
            all_identical = [{"left_window_id": left, "right_window_id": right}
                             for text in identical_texts
                             for left in table_all_text_windows[t["table_id"]][text]
                             for right in table_all_text_windows[peer["table_id"]][text]]
            all_identical.sort(key=lambda row: (row["left_window_id"], row["right_window_id"]))
            identical = [{"left_window_id": left, "right_window_id": right}
                         for text in sorted(set(table_text_windows[t["table_id"]]) & set(table_text_windows[peer["table_id"]]))
                         for left in table_text_windows[t["table_id"]][text]
                         for right in table_text_windows[peer["table_id"]][text]]
            identical.sort(key=lambda row: (row["left_window_id"], row["right_window_id"]))
            comparisons.append({"table_id": peer["table_id"],
                                "text_similarity": round(difflib.SequenceMatcher(
                                    None, table_texts[t["table_id"]], table_texts[peer["table_id"]]).ratio(), 6),
                                "shared_evidence_ids": shared,
                                "all_exact_text_window_match_count": len(all_identical),
                                "all_exact_text_window_matches": all_identical[:12],
                                "identical_text_window_match_count": len(identical),
                                "identical_text_window_matches": identical[:12]})
        exact_peers = [x["table_id"] for x in comparisons
                       if table_texts[t["table_id"]] == table_texts[x["table_id"]]]
        overlap_peers = [x["table_id"] for x in comparisons if x["all_exact_text_window_match_count"]]
        tables.append({"table_id": t["table_id"], "page": t["page"], "purpose_hint": t["purpose"],
                       "caption": t.get("caption", ""), "headers": copy.deepcopy(t.get("headers", [])),
                       "eligible_evidence_ids": eligible,
                       "duplicate_view_status": ("exact_table_text_duplicate_equivalence_unproven" if exact_peers
                                                  else "exact_window_text_overlap_equivalence_unproven" if overlap_peers
                                                  else "no_exact_text_duplicate_detected"),
                       "same_page_text_overlap_table_ids": overlap_peers,
                       "compared_same_page_table_ids": [x["table_id"] for x in comparisons],
                       "same_page_comparisons": comparisons})
    cells = []
    for n, c in enumerate(base["feature_cell_candidates"]):
        cells.append({"candidate_id": f"FC{n:04d}", "table_id": c["table_id"],
                      "label": c.get("label", c.get("text", "")),
                      "evidence_ids": sorted({reverse[w["window_index"]] for w in _catalog(paper)
                           if w["unit_id"] in {c["unit_id"], c.get("row_unit_id")}
                           and w["page"] in group["pages"]})})
    windows = [{"window_id": reverse[n], "page": r["page"], "source_kind": r["source_kind"],
                "role": "primary_eligible" if n in group["window_ids"] else "supporting_context",
                "text": r["text"], "quote_options": v10.quote_options(r["text"]) if n in group["window_ids"] else []}
               for r in catalog for n in [r["window_index"]] if n in group["context_window_ids"]]
    return {"target_pages": copy.deepcopy(group["pages"]), "context_pages": copy.deepcopy(group["context_pages"]),
            "context_scope": group["context_scope"], "distant_cross_page_evidence": "not_provided",
            "source_windows": windows, "table_candidates": tables, "feature_cell_candidates": cells,
            "_table_unit_ids": table_unit_ids, "_v12": base}


def response_schema(task, paper_input, index, index_group):
    _check(task, paper_input, index_group)
    schema = v12.response_schema(v12.make_task(), paper_input, index, _region(index_group))
    schema["$id"] = RESPONSE_VERSION
    schema["properties"]["schema_version"] = {"const": RESPONSE_VERSION}
    ctx = _context(paper_input, index_group)
    vis = [x["window_id"] for x in ctx["source_windows"]]
    primary = [_wid(n) for n in index_group["window_ids"]]
    schema["properties"]["coverage"]["prefixItems"] = [
        {"type": "object", "additionalProperties": False, "required": ["window_id", "state"],
         "properties": {"window_id": {"const": _wid(n)}, "state": {"enum": ["reviewed", "uncertain", "overflow"]}}}
        for n in index_group["window_ids"]]
    schema["properties"]["coverage"]["items"] = {"type": "object"}
    schema["properties"]["coverage"]["minItems"] = schema["properties"]["coverage"]["maxItems"] = len(primary)
    objects = schema["properties"]["objects"]["items"]["properties"]
    source_ref = schema["$defs"]["sourceWindows"]
    source_ref["items"] = {"type": "string", "enum": vis}
    facts = objects["facts"]["items"]
    fprops = facts["properties"]
    fprops.pop("primary_anchor", None)
    if "primary_anchor" in facts["required"]:
        facts["required"].remove("primary_anchor")
    fprops["primary_window_id"] = {"type": "string", "enum": primary}
    target_quotes = list(dict.fromkeys(q for n in index_group["window_ids"]
                                       for q in v10.quote_options(_catalog(paper_input)[n]["text"])))
    fprops["primary_quote"] = {"type": "string", "enum": target_quotes} if target_quotes else {"type": "string", "minLength": 1}
    if not target_quotes:
        facts["maxItems"] = 0
    fprops["support_windows"]["items"] = {"type": "string", "enum": vis}
    facts["required"].extend(["primary_window_id", "primary_quote"])
    # Require object-level primary anchor separately from generic source context.
    objects["primary_evidence_id"] = {"type": "string", "enum": primary}
    schema["properties"]["objects"]["items"]["required"].append("primary_evidence_id")
    for review, table in zip(schema["properties"]["table_reviews"].get("prefixItems", []), ctx["table_candidates"]):
        if not table["eligible_evidence_ids"]:
            raise ValueError("table_has_no_eligible_evidence:" + table["table_id"])
        review["properties"]["source_windows"] = {
            "type": "array", "minItems": 1,
            "maxItems": source_ref['maxItems'],
            "items": {"type": "string", "enum": table["eligible_evidence_ids"]},
            "uniqueItems": True}
    for row, cell in zip(schema["properties"]["feature_coverage"].get("prefixItems", []), ctx["feature_cell_candidates"]):
        row["properties"].pop("unit_id", None)
        row["required"].remove("unit_id")
        row["properties"]["candidate_id"] = {"const": cell["candidate_id"]}
        row["required"].append("candidate_id")
    return schema


def display_schema(schema):
    """Abbreviate repeated ID/quote enums for display, never for the decoder."""
    result = copy.deepcopy(schema)
    for name, id_key, description in (
        ('coverage', 'window_id', 'Each primary_eligible W ID, in source_windows order.'),
        ('table_reviews', 'table_id', 'Each table_id, in table_candidates order; cite its eligible_evidence_ids only.'),
        ('feature_coverage', 'candidate_id', 'Each candidate_id, in feature_cell_candidates order.'),
    ):
        array = result['properties'][name]
        members = array.pop('prefixItems', [])
        if members:
            item = copy.deepcopy(members[0])
            item['properties'][id_key] = {'type': 'string', 'description': description}
            array['items'] = item
        array['description'] = description

    def abbreviate(value, key=None):
        if isinstance(value, list):
            return [abbreviate(item) for item in value]
        if not isinstance(value, dict):
            return value
        if key == 'primary_quote' and 'enum' in value:
            return {'type': 'string', 'description': 'Exact quote_options text beside primary_window_id.'}
        choices = value.get('enum')
        if choices and all(isinstance(v, str) and v.startswith('W') and v[1:].isdigit() for v in choices):
            return {'type': 'string', 'description': 'Exact displayed W ID; primary evidence must be on the target page.'}
        return {name: abbreviate(item, name) for name, item in value.items()}
    return abbreviate(result)


def render_group(task, paper_input, index, group):
    schema = response_schema(task, paper_input, index, group)
    nav_errors = verify_index(index, paper_input, taxonomy_value=task["taxonomy"])
    if nav_errors:
        raise ValueError("v13r2_navigation_index_invalid:" + ",".join(nav_errors[:4]))
    unit_pages = {row["unit_id"]: row["page"] for row in v10.unit_catalog(paper_input).values()}
    entries = {row["unit_id"]: row for row in index["entries"]}
    navigation_groups = {}
    target = _context(paper_input, group)
    raw_windows = {_wid(row['window_index']): row for row in _catalog(paper_input)}
    for w in target["source_windows"]:
        raw = raw_windows[w['window_id']]
        entry = entries[raw["unit_id"]]
        prediction = entry.get("prediction") if entry.get("availability") == "available" else entry
        classification = {'availability': entry.get('availability', 'available'),
                          'state': prediction.get('state') if isinstance(prediction, dict) else None,
                          'categories': copy.deepcopy(prediction.get('categories', [])) if isinstance(prediction, dict) else []}
        key = digest(classification)
        navigation_groups.setdefault(key, {**classification, 'window_ids': []})['window_ids'].append(w['window_id'])
    body = {"task_sha256": task["task_sha256"], "target_schema": display_schema(schema),
            "decoder_schema_sha256": digest(schema), "prompt_projection": PROMPT_PROJECTION,
            "source_identity": source_identity(paper_input),
            "paper_input_canonical_sha256": digest(paper_input),
            "source_pdf_sha256": paper_input["source_pdf_sha256"],
            "navigation_index_sha256": index["index_sha256"],
            "target": target,
            "navigation_context": {"scope": "target_plus_adjacent_pdf_pages/v1",
                                   "index_sha256": index["index_sha256"],
                                   "authority": index.get("authority", "automated_fallible_navigation_not_gold"),
                                   "groups": list(navigation_groups.values())}}
    body["target"].pop("_table_unit_ids", None)
    body["target"].pop("_v12", None)
    return [{"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": canonical_bytes(body).decode("utf-8")}]


parse_response = v10.parse_response


def _internalize(payload, paper, group):
    p = copy.deepcopy(payload)
    forward, reverse = _maps(paper)
    p["schema_version"] = v12.RESPONSE_VERSION
    for c in p["coverage"]:
        c["window_id"] = forward[c["window_id"]]
    for o in p["objects"]:
        o.pop("primary_evidence_id", None)
        o["source_windows"] = [forward[x] for x in o["source_windows"]]
        for f in o["facts"]:
            n = forward[f.pop("primary_window_id")]
            quote = f.pop("primary_quote")
            options = v10.quote_options(_catalog(paper)[n]["text"])
            try:
                quote_index = options.index(quote)
            except ValueError:
                quote_index = 0  # The V13R2 validator reports the invalid quote.
            f["primary_anchor"] = f"w{n}:q{quote_index}"
            f["support_windows"] = [forward[x] for x in f["support_windows"]]
            if f["claim"]["kind"] == "link":
                f["claim"]["target"]["source_windows"] = [
                    forward[x] for x in f["claim"]["target"]["source_windows"]]
    ctx = _context(paper, group)
    for r, t in zip(p["table_reviews"], ctx["_v12"]["table_candidates"]):
        r["source_windows"] = [forward[x] for x in r["source_windows"]]
    for row, cell in zip(p["feature_coverage"], ctx["_v12"]["feature_cell_candidates"]):
        row["unit_id"] = cell["unit_id"]
        row.pop("candidate_id", None)
    return p


def validate_response(payload, paper_input, index, group, task=None):
    try:
        schema = response_schema(make_task() if task is None else task, paper_input, index, group)
        errors = ["v13r2_schema:" + e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(payload)]
        if errors:
            return errors
        ids, reverse = _maps(paper_input)
        windows = _catalog(paper_input)
        target = set(group["window_ids"])
        for i, obj in enumerate(payload["objects"]):
            if obj["primary_evidence_id"] not in {_wid(n) for n in target}:
                errors.append(f"object_primary_not_assigned_page:{i}")
            elif not any(windows[ids[obj["primary_evidence_id"]]]["page"] == group["pages"][0]
                         for _ in [0]):
                errors.append(f"object_primary_page_mismatch:{i}")
            if obj["primary_evidence_id"] not in obj["source_windows"]:
                errors.append(f"object_primary_not_in_source_windows:{i}")
            for fno, fact in enumerate(obj["facts"]):
                wi = ids[fact["primary_window_id"]]
                if fact["primary_window_id"] not in {_wid(n) for n in target}:
                    errors.append(f"fact_primary_not_assigned_page:{i}:{fno}")
                if fact["primary_quote"] not in v10.quote_options(windows[wi]["text"]):
                    errors.append(f"fact_quote_not_exact_option:{i}:{fno}")
        internal = _internalize(payload, paper_input, group)
        # The v12 validator preserves nested fact, feature-cell, and table gates.
        errors.extend(v12.validate_response(internal, paper_input, index, _region(group)))
        return errors
    except (KeyError, TypeError, ValueError) as exc:
        return ["v13r2_validation_context:" + str(exc)]


def completion_status(payload, group, paper_input=None):
    if paper_input is None:
        raise ValueError("v13r2_paper_required_for_completeness")
    internal = _internalize(payload, paper_input, group)
    return "incomplete" if v12.completeness(internal, paper_input, _region(group)) else "success"


def materialize_group(paper_input, index, task, group, record):
    from .workflow import replay_run
    errors = replay_run(record, task, paper_input, index=index, extraction_group=group)
    if errors or record["status"] != "success" or record["task"] != task or record.get("extraction_group") != group:
        raise ValueError("v13r2_record_not_admitted")
    payload, normalization = parse_response(record["backend_result"]["raw_text"])
    if payload != record["parsed_response"] or normalization != record.get("response_normalization"):
        raise ValueError("v13r2_raw_replay_mismatch")
    errors = validate_response(payload, paper_input, index, group, task)
    if errors or completion_status(payload, group, paper_input) != "success":
        raise ValueError("v13r2_response_invalid_or_incomplete:" + ",".join(errors[:4]))
    internal = _internalize(payload, paper_input, group)
    flat = v12.to_v10(internal, paper_input, _region(group))
    prefix, identity = group["group_id"], record["job"]["parent_job_id"] + ":" + group["group_id"]
    coverage = [{"group_id": prefix, **copy.deepcopy(c)} for c in payload["coverage"]]
    mentions = [{"mention_id": identity + ":m" + str(i), "group_id": prefix, **copy.deepcopy(m),
                 "source_evidence": [v10._window_evidence(w, paper_input) for w in internal["objects"][i]["source_windows"]],
                 "normalized_label_authority": "model_asserted_not_verified"}
                for i, m in enumerate(flat["mentions"])]
    facts = []
    for i, f in enumerate(flat["facts"]):
        row = copy.deepcopy(f)
        row["subject_mention_id"] = identity + ":m" + str(row.pop("subject_mention"))
        facts.append({"fact_id": identity + ":f" + str(i), "group_id": prefix, **row,
                      "primary_evidence": v10._primary_quote_evidence(f["primary_window"], f["primary_quote"], paper_input),
                      "support_evidence": [v10._window_evidence(w, paper_input) for w in f["support_windows"]],
                      "target_evidence": None if f["claim"]["kind"] != "link" else
                      [v10._window_evidence(w, paper_input) for w in f["claim"]["target"]["source_windows"]]})
    return {"group_id": prefix, "record_sha256": record["record_sha256"], "mentions": mentions,
            "facts": facts, "coverage": coverage}


def build_bundle(paper_input, index, task, groups, records):
    expected = plan_groups(paper_input, groups[0]["policy"] if groups else None)
    if groups != expected or len(groups) != len(records):
        raise ValueError("v13r2_group_plan_mismatch")
    parts = [materialize_group(paper_input, index, task, g, r) for g, r in zip(groups, records)]
    bindings = {(r["profile_sha256"], r["job"]["parent_job_id"], r["index_sha256"]) for r in records}
    if len(bindings) > 1:
        raise ValueError("v13r2_cross_record_binding_mismatch")
    binding = next(iter(bindings), (None, None, None))
    return seal({"schema_version": BUNDLE_VERSION, "source_identity": source_identity(paper_input),
                 "paper_input_canonical_sha256": digest(paper_input), "index_sha256": index["index_sha256"],
                 "task_sha256": task["task_sha256"], "groups": copy.deepcopy(groups),
                 "profile_sha256": binding[0], "parent_job_id": binding[1],
                 "record_refs": [{"group_id": p["group_id"], "run_id": r["run_id"],
                                  "record_sha256": p["record_sha256"]} for p, r in zip(parts, records)],
                 "coverage": [x for p in parts for x in p["coverage"]],
                 "mentions": [x for p in parts for x in p["mentions"]],
                 "facts": [x for p in parts for x in p["facts"]],
                 "observation_count": sum(len(p["facts"]) for p in parts),
                 "mention_record_count": sum(len(p["mentions"]) for p in parts),
                 "resolved_object_count": None, "semantic_correctness": "not_established",
                 "identity_resolution": "unresolved_model_hypotheses_only",
                 "label_evidence_scope": "cited_window_context_not_name_equivalence_or_entailment"}, "bundle_sha256")


def verify_bundle(bundle, paper_input, index, task, records=None):
    errors = seal_errors(bundle, "bundle_sha256")
    if records is None:
        return errors + ["v13r2_bundle_records_required_for_replay"]
    try:
        if bundle != build_bundle(paper_input, index, task, bundle["groups"], records):
            errors.append("v13r2_bundle_derivation_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("v13r2_bundle_replay:" + str(exc))
    return errors
