"""Versioned V13 model view retaining every paper window and exact quote.

Exact layout coordinates and original unit IDs stay bound in the frozen source
artifact. The model view projects them to numbered windows and unit ordinals.
"""
from __future__ import annotations

import copy
import json
import re

from .common import canonical_bytes, digest


def compact_full_source(source: dict) -> dict:
    src = copy.deepcopy(source)
    columns = src["unit_columns"]
    uid_pos = columns.index("unit_id")
    bbox_pos = columns.index("bbox")
    ids = {row[uid_pos]: i for i, row in enumerate(src["units"])}
    if len(ids) != len(src["units"]):
        raise ValueError("duplicate_unit_ids")
    for row in src["units"]:
        row.pop(bbox_pos)
    src["unit_columns"].pop(bbox_pos)
    # The complete page/table topology remains in the verified source artifact.
    # Target-page structural candidates remain in the target section; the full
    # paper context here keeps every page, unit ID, kind and numbered text.
    for key in ("pages", "page_columns", "table_columns", "column_columns", "row_columns"):
        src.pop(key)
    for column in ("reading_order", "structure_status"):
        if column in src["unit_columns"]:
            position = src["unit_columns"].index(column)
            for row in src["units"]:
                row.pop(position)
            src["unit_columns"].pop(position)
    # Reversible ID patterns retain all source identities with fewer repeated
    # prefixes and zero-padded digit runs in model-visible text.
    patterns = []
    kinds = []
    for row in src["units"]:
        uid = row[src["unit_columns"].index("unit_id")]
        pieces = re.split(r"(\d+)", uid)
        static = pieces[::2]
        numeric = pieces[1::2]
        pattern = [static, [len(value) for value in numeric]]
        if pattern not in patterns:
            patterns.append(pattern)
        row[src["unit_columns"].index("unit_id")] = [patterns.index(pattern), *[int(value) for value in numeric]]
        kind_pos = src["unit_columns"].index("source_kind")
        if row[kind_pos] not in kinds:
            kinds.append(row[kind_pos])
        row[kind_pos] = kinds.index(row[kind_pos])
    src["unit_id_patterns"] = patterns
    src["source_kind_dictionary"] = kinds
    # Native recovery can duplicate exact substrings already visible in a
    # recognized window. Refer only to non-recovery windows, so references are
    # acyclic and mechanically reversible without semantic matching.
    win_pos = src["unit_columns"].index("numbered_windows")
    kind_pos = src["unit_columns"].index("source_kind")
    recognized = [(wid, value) for row in src["units"]
                  if src["source_kind_dictionary"][row[kind_pos]] != "recovered_source_text"
                  for wid, value in row[win_pos] if isinstance(value, str)]
    recognized.sort(key=lambda pair: len(pair[1]))
    for row in src["units"]:
        if src["source_kind_dictionary"][row[kind_pos]] != "recovered_source_text":
            continue
        for window in row[win_pos]:
            value = window[1]
            if len(value) < 12:
                continue
            for source_wid, source_text in recognized:
                if len(source_text) < len(value):
                    continue
                offset = source_text.find(value)
                if offset >= 0 and source_wid != window[0]:
                    window[1] = ["W", source_wid, offset, offset + len(value)]
                    break
    # Exact repeated text is stored once. Unique source text stays readable.
    counts = {}
    for row in src["units"]:
        for _, value in row[src["unit_columns"].index("numbered_windows")]:
            if isinstance(value, str):
                counts[value] = counts.get(value, 0) + 1
    dictionary = sorted(value for value, count in counts.items() if count > 1 and len(value) >= 20)
    lookup = {value: i for i, value in enumerate(dictionary)}
    for row in src["units"]:
        for window in row[src["unit_columns"].index("numbered_windows")]:
            if isinstance(window[1], str) and window[1] in lookup:
                window[1] = ["T", lookup[window[1]]]
    src["text_dictionary"] = dictionary
    nav_dictionary = []
    nav_positions = []
    for entry in src.pop("index_entries"):
        if entry not in nav_dictionary:
            nav_dictionary.append(entry)
        nav_positions.append(nav_dictionary.index(entry))
    src["navigation_dictionary"] = nav_dictionary
    src["navigation_by_unit_index"] = nav_positions
    src["schema_version"] = "paper-extraction-numbered-source/v13-compact"
    src["prompt_projection"] = {
        "unit_array_index": "Zero-based index into units; stable unit_id remains in units.",
        "navigation": "navigation_by_unit_index selects an index_entries-shaped row from navigation_dictionary.",
        "unit_id": "unit_id_patterns[pattern] gives static parts and decimal widths; remaining tuple values reconstruct each exact original unit ID.",
        "source_kind": "Integer selects the exact source kind from source_kind_dictionary.",
        "window_text": "String is exact text; [T,n] selects exact text_dictionary[n].",
        "recovery_span": "[W,source_window,start,end] is the exact Unicode substring of an earlier recognized source window at [start,end).",
        "layout": "Full page/table topology, exact boxes and reading order remain in the verified source artifact; assigned-page candidates are in target.",
        "evidence": "Every page number, source kind, numbered window text and stable unit ID is retained."}
    return src


def expand_unit_projection(source: dict) -> list[tuple[str, int, str, list[list]]]:
    """Reconstruct every identity, role and numbered window for audit tests."""
    result = []
    positions = {name: i for i, name in enumerate(source["unit_columns"])}
    raw_windows = {wid: value for row in source["units"]
                   for wid, value in row[positions["numbered_windows"]]}

    def text(value):
        if isinstance(value, str):
            return value
        if value[0] == "T":
            return source["text_dictionary"][value[1]]
        if value[0] == "W":
            return text(raw_windows[value[1]])[value[2]:value[3]]
        raise ValueError("unknown_compact_text_reference")

    for row in source["units"]:
        code = row[positions["unit_id"]]
        static, widths = source["unit_id_patterns"][code[0]]
        uid = static[0] + "".join(str(number).zfill(width) + suffix
                                  for number, width, suffix in zip(code[1:], widths, static[1:]))
        windows = [[wid, text(value)]
                   for wid, value in row[positions["numbered_windows"]]]
        result.append((uid, row[positions["page"]],
                       source["source_kind_dictionary"][row[positions["source_kind"]]], windows))
    return result


def explanatory_schema(schema: dict) -> dict:
    """Readable prompt guide; decoder still receives the exact original schema."""
    return {"schema_version": schema["properties"]["schema_version"]["const"],
            "decoder_schema_sha256": digest(schema),
            "decoder_contract": "The structured decoder enforces the full exact JSON Schema; this is a guide to its field meanings, not a replacement schema.",
            "coverage": "One {window_id,state} for every target window in order; state is reviewed, uncertain or overflow.",
            "objects": "Array of {kind,normalized_label,source_windows,context,facts}; kind is dataset|table|group|array|record|variable|field|dimension; context is string|null; facts are nested under their owning object.",
            "facts": "Each fact has {support_windows:[window IDs],categories:[structure|encoding|value|syntax],claim,primary_anchor:anchor ID}. Attribute claim is {kind:'attribute',predicate,assertion}. Link claim is {kind:'link',predicate:'parent'|'relationship'|'same_as',target:{kind,normalized_label,source_windows},assertion}. Assertion is {status:'reported',value:string,basis:null}, {status:'inferred',value:string,basis:string}, or {status:'unknown',value:null,basis:null}. Select primary_anchor from target anchors.",
            "table_reviews": "One {table_id,purpose,field_axis,rationale,source_windows} per target table, in target order.",
            "feature_coverage": "One {unit_id,decision,emitted_labels,rationale} per target feature cell, in target order. Emitted labels must bind field/variable objects.",
            "limits": {key: schema["properties"][key].get("maxItems") for key in
                       ("coverage", "objects", "table_reviews", "feature_coverage")}}


def compact_messages(messages: list[dict]) -> list[dict]:
    output = copy.deepcopy(messages)
    body = json.loads(output[1]["content"])
    body["full_source"] = compact_full_source(body["full_source"])
    body["target_schema"] = explanatory_schema(body["target_schema"])
    # Anchor quotes duplicate source-window text. Keep exact identity and
    # Unicode code-point spans into the corresponding numbered window.
    compact_source = body["full_source"]
    lookup = {window_id: text for _, _, _, windows in expand_unit_projection(compact_source)
              for window_id, text in windows}
    anchors = body["target"].get("anchors", [])
    body["target"]["anchor_columns"] = ["anchor_id", "window_id", "quote_start", "quote_end"]
    body["target"]["anchors"] = [[a["anchor_id"], a["window_id"],
                                    lookup[a["window_id"]].index(a["quote"]),
                                    lookup[a["window_id"]].index(a["quote"]) + len(a["quote"])]
                                   for a in anchors]
    body["target"]["anchor_projection"] = "Quote is the exact Unicode substring of the named numbered window at [quote_start,quote_end)."
    output[1]["content"] = canonical_bytes(body).decode("utf-8")
    return output


def compact_messages_ordinal(messages: list[dict]) -> list[dict]:
    """Official model-view projection with exact source IDs bound externally.

    The verified source still owns exact unit IDs. Model evidence uses numbered
    windows, and target candidates retain their named IDs. This prompt binds
    the ordered unit-ID list by hash rather than displaying every ID.
    """
    output = compact_messages(messages)
    body = json.loads(output[1]["content"])
    source = body["full_source"]
    expanded = expand_unit_projection(source)
    unit_ids = [row[0] for row in expanded]
    unit_indexes = {uid: index for index, uid in enumerate(unit_ids)}
    source["unit_id_sequence_sha256"] = digest(unit_ids)
    source["unit_reference"] = "U0..U(n-1) are row indexes in units; exact unit IDs are pinned by unit_id_sequence_sha256 in the verified source artifact. Numbered window IDs are unchanged."
    uid_pos = source["unit_columns"].index("unit_id")
    for row in source["units"]:
        row.pop(uid_pos)
    source["unit_columns"].pop(uid_pos)
    source.pop("unit_id_patterns")
    source["prompt_projection"].pop("unit_id")
    page_pos = source["unit_columns"].index("page")
    page_runs = []
    for index, row in enumerate(source["units"]):
        page = row[page_pos]
        if not page_runs or page_runs[-1][0] != page:
            page_runs.append([page, index, index + 1])
        else:
            page_runs[-1][2] = index + 1
        row.pop(page_pos)
    source["unit_columns"].pop(page_pos)
    source["page_runs"] = page_runs
    source["prompt_projection"]["page_runs"] = "[page,start,end) assigns page to contiguous rows in units."
    for table in body["target"].get("table_candidates", []):
        if "unit_ids" in table:
            table["unit_indexes"] = [unit_indexes[uid] for uid in table.pop("unit_ids")]
        if "header_unit_ids" in table:
            table["header_unit_indexes"] = [unit_indexes[uid] for uid in table.pop("header_unit_ids")]
    for cell in body["target"].get("feature_cell_candidates", []):
        cell["unit_index"] = unit_indexes[cell["unit_id"]]
        cell["row_unit_index"] = unit_indexes[cell["row_unit_id"]]
    output[1]["content"] = canonical_bytes(body).decode("utf-8")
    return output


def expand_ordinal_projection(source: dict) -> list[tuple[int, str, list[list]]]:
    """Audit exact page, role and text with immutable numeric window IDs."""
    positions = {name: i for i, name in enumerate(source["unit_columns"])}
    raw_windows = {wid: value for row in source["units"]
                   for wid, value in row[positions["numbered_windows"]]}

    def text(value):
        if isinstance(value, str):
            return value
        if value[0] == "T":
            return source["text_dictionary"][value[1]]
        if value[0] == "W":
            return text(raw_windows[value[1]])[value[2]:value[3]]
        raise ValueError("unknown_compact_text_reference")

    pages = [None] * len(source["units"])
    for page, start, end in source["page_runs"]:
        pages[start:end] = [page] * (end - start)
    if any(page is None for page in pages):
        raise ValueError("ordinal_page_coverage_incomplete")
    return [(pages[i], source["source_kind_dictionary"][row[positions["source_kind"]]],
             [[wid, text(value)] for wid, value in row[positions["numbered_windows"]]])
            for i, row in enumerate(source["units"])]
