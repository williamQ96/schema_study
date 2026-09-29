"""Spatial, lossless v4 layout from hash-bound v3 source-word observations.

This module does not alter v2/v3 artifacts. Its word partition is conservative:
when a candidate's header and rows do not support cells, it remains citable text.
"""
from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paper_layout_evidence import (
    TABLE_CAPTION, TABLE_REFERENCE_LEADS, assign_document_spans, bbox,
    canonical_json_bytes, group_lines, header_groups, sha256_bytes, sha256_file,
    split_segments, table_region, text_region, words_text,
)


LAYOUT_VERSION = "paper-evidence-layout/v4"
INPUT_VERSION = "paper-evidence-input/v4"
PROFILE = "frozen-pdf-words-spatial-table-v4"
_GAP = 15.0


def _center(word: Mapping[str, Any]) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2


def _lane(word: Mapping[str, Any], split_x: float | None) -> str:
    if split_x is None:
        return "full"
    return "left" if _center(word) < split_x else "right"


def _page_split(words: Sequence[dict], width: float, height: float) -> float | None:
    """Find a supported central gutter, without assuming every band is two-column."""
    body = [w for w in words if height * .065 <= float(w["top"]) < height * .925]
    if len(body) < 40:
        return None
    candidates = [width * fraction / 100 for fraction in range(43, 58)]
    scored = []
    for x in candidates:
        left = sum(_center(w) < x for w in body)
        right = len(body) - left
        crossing = sum(float(w["x0"]) < x < float(w["x1"]) for w in body)
        if min(left, right) >= max(12, len(body) * .15):
            scored.append((crossing, abs(x - width / 2), x))
    if not scored:
        return None
    crossing, _, chosen = min(scored)
    return chosen if crossing / len(body) <= .04 else None


def _caption_candidates(words: Sequence[dict], split_x: float | None) -> list[dict]:
    captions = []
    for lane in (("left", "right") if split_x is not None else ("full",)):
        local = [w for w in words if _lane(w, split_x) == lane]
        for line in group_lines(local):
            for segment in split_segments(line, gap=35):
                value = words_text(segment)
                match = TABLE_CAPTION.match(value)
                if not match:
                    continue
                remainder = value[match.end():].strip()
                lead = remainder.split(maxsplit=1)[0].casefold().strip(":,-") if remainder else ""
                if (lead in TABLE_REFERENCE_LEADS
                        or len(re.findall(r"\bTable\s*\d+[A-Za-z]?\b", value, re.I)) > 1
                        or remainder.startswith((")", "]"))):
                    continue
                captions.append({"number": match.group(1), "words": segment, "top": min(w["top"] for w in segment), "lane": lane})
    return sorted(captions, key=lambda c: (c["top"], min(w["x0"] for w in c["words"])))


def _horizontal_rules(page: Any) -> list[tuple[float, float, float]]:
    # PDF generators may encode one visual rule as several touching line
    # objects, or as a thin rectangle. Merge only collinear touching edges.
    line_edges = [item for item in page.edges if item.get("object_type") == "line"]
    edge_source = line_edges if line_edges else [item for item in page.edges
                                             if item.get("object_type") == "rect_edge"]
    edges = []
    for item in edge_source:
        x0, x1 = float(item["x0"]), float(item["x1"])
        top, bottom = float(item["top"]), float(item["bottom"])
        if item.get("orientation") == "h" and x1 - x0 >= 15 and abs(top - bottom) <= 1:
            edges.append((top, x0, x1))
    merged: list[tuple[float, float, float]] = []
    for y, x0, x1 in sorted(edges):
        if merged and abs(y - merged[-1][0]) <= 1 and x0 <= merged[-1][2] + 2:
            previous = merged.pop()
            merged.append((previous[0], min(previous[1], x0), max(previous[2], x1)))
        else:
            merged.append((y, x0, x1))
    return [rule for rule in merged if rule[2] - rule[1] >= 40]


def _rule_after(rules: Sequence[tuple[float, float, float]], y: float,
                x0: float, x1: float, limit: float) -> float | None:
    good = [ry for ry, rx0, rx1 in rules if y < ry < limit
            and rx0 <= x0 + 8 and rx1 >= x1 - 8]
    return max(good) if good else None


def _header(line: Sequence[dict], page_width: float, *, gap: float = _GAP) -> list[list[dict]] | None:
    groups = split_segments(line, gap=gap)
    if not 2 <= len(groups) <= 12:
        return None
    starts = [min(w["x0"] for w in g) for g in groups]
    if any(b - a < 18 for a, b in zip(starts, starts[1:])):
        return None
    if max(w["x1"] for w in line) - min(w["x0"] for w in line) < page_width * .15:
        return None
    return groups


def _split_cells(line: Sequence[dict], starts: Sequence[float]) -> list[list[dict]]:
    cells: list[list[dict]] = [[] for _ in starts]
    for word in sorted(line, key=lambda w: (w["x0"], w["word_index"])):
        index = 0
        for candidate in range(1, len(starts)):
            if float(word["x0"]) >= starts[candidate] - min(5.0, (starts[candidate] - starts[candidate - 1]) / 4):
                index = candidate
        cells[index].append(word)
    return cells


def _candidate_table(caption: dict, words: Sequence[dict], split_x: float | None,
                     rules: Sequence[tuple[float, float, float]], width: float,
                     height: float, next_caption_top: float) -> tuple[dict | None, set[int], dict]:
    """Return a table only if its local header and repeated rows are coherent."""
    lane = caption["lane"]
    local = [w for w in words if _lane(w, split_x) == lane
             and caption["top"] - 2 <= float(w["top"]) < next_caption_top]
    lines = group_lines(local)
    caption_line = next((i for i, line in enumerate(lines)
                         if set(w["word_index"] for w in caption["words"]) <=
                         set(w["word_index"] for w in line)), None)
    audit = {"table_number": caption["number"], "page_lane": lane,
             "caption_source_word_indexes": [w["word_index"] for w in caption["words"]]}
    if caption_line is None:
        return None, set(), {**audit, "status": "uncertain", "reason": "caption_line_not_found"}
    header_i = None
    groups = None
    for i in range(caption_line + 1, min(len(lines), caption_line + 7)):
        if min(w["top"] for w in lines[i]) - caption["top"] > 65:
            break
        candidate = _header(lines[i], width)
        if candidate is not None:
            header_i, groups = i, candidate
            break
    if header_i is None or groups is None:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_coherent_header"}
    starts = [min(w["x0"] for w in g) for g in groups]
    cap_words = [w for line in lines[caption_line:header_i] for w in line]
    header_words = lines[header_i]
    table_x0 = min(w["x0"] for w in cap_words + header_words)
    table_x1 = max(w["x1"] for w in cap_words + header_words)
    header_bottom = max(w["bottom"] for w in header_words)
    bottom_rule = _rule_after(rules, header_bottom + 1, table_x0, table_x1, next_caption_top)
    max_y = bottom_rule if bottom_rule is not None else min(next_caption_top, height * .925)
    rows: list[list[list[dict]]] = []
    row_line_indexes: list[int] = []
    rejected: list[dict] = []
    for i in range(header_i + 1, len(lines)):
        line = lines[i]
        top = min(w["top"] for w in line)
        if top >= max_y:
            break
        if top - max(w["bottom"] for w in lines[i - 1]) > 30:
            break
        cells = _split_cells(line, starts)
        populated = sum(bool(cell) for cell in cells)
        first = bool(cells[0])
        if first and populated >= 2:
            rows.append(cells)
            row_line_indexes.append(i)
        elif rows and not first and populated:
            for col, cell in enumerate(cells):
                rows[-1][col].extend(cell)
            row_line_indexes.append(i)
        else:
            rejected.append({"line": i, "top": top})
            break
    if len(rows) < 2 or len(rows) < max(2, len(row_line_indexes) // 3):
        return None, set(), {**audit, "status": "uncertain", "reason": "insufficient_repeated_rows"}
    if rejected and bottom_rule is None:
        # A prose line may be a note or an adjacent stream. Stop before it;
        # preserve that line in normal text rather than guessing a cell.
        max_y = rejected[0]["top"]
    selected_lines = lines[caption_line:header_i + 1] + [lines[i] for i in row_line_indexes]
    selected_words = [w for line in selected_lines for w in line]
    assigned = [w["word_index"] for w in cap_words + header_words]
    assigned.extend(w["word_index"] for row in rows for cell in row for w in cell)
    if len(assigned) != len(set(assigned)) or set(assigned) != {w["word_index"] for w in selected_words}:
        return None, set(), {**audit, "status": "uncertain", "reason": "nonpartitioned_candidate"}
    if split_x is not None and bottom_rule is None:
        # A word-only candidate cannot claim one side of a wide table as a
        # complete table. Repeated rows aligned across the page are positive
        # evidence that the page-global gutter cuts through this local band.
        row_tops = [min(float(w["top"]) for w in lines[i]) for i in row_line_indexes]
        opposite = [w for w in words if _lane(w, split_x) != lane]
        supported = sum(any(abs(float(w["top"]) - y) <= 2.5 for w in opposite)
                        for y in row_tops)
        if len(row_tops) >= 2 and supported >= max(2, len(row_tops) * .5):
            band_lines = [line for line in group_lines(words)
                          if min(float(w["top"]) for w in line) >= caption["top"]
                          and min(float(w["top"]) for w in line) < next_caption_top]
            contiguous = []
            for line in band_lines:
                if contiguous and min(float(w["top"]) for w in line) - max(float(w["bottom"]) for w in contiguous[-1]) > 18:
                    break
                contiguous.append(line)
            band = [w for line in contiguous for w in line]
            if band:
                return None, set(), {**audit, "status": "uncertain",
                    "reason": "word_only_candidate_has_coaligned_other_lane_rows",
                    "unparsed_bbox": [min(float(w["x0"]) for w in band),
                                      min(float(w["top"]) for w in band),
                                      max(float(w["x1"]) for w in band),
                                      max(float(w["bottom"]) for w in band) + .1]}
    # A lane can be a local column or a full-width band. A local table is
    # rejected if its own words cross into the other lane.
    if split_x is not None and lane != "full" and any(_lane(w, split_x) != lane for w in selected_words):
        return None, set(), {**audit, "status": "uncertain", "reason": "cross_lane_words"}
    return ({"caption_words": cap_words, "header_groups": groups, "header_words": header_words,
             "rows": rows, "selected_words": selected_words, "selected_lines": selected_lines,
             "bottom_rule": bottom_rule, "lane": lane, "number": caption["number"]},
            set(assigned),
            {**audit, "status": "structured_geometry_consistent", "source_word_count": len(assigned),
             "row_count": len(rows), "column_count": len(starts), "bottom_rule": bottom_rule})


def _candidate_below_ruled_table(caption: dict, words: Sequence[dict], pdf_page: Any,
                                  rules: Sequence[tuple[float, float, float]], width: float,
                                  height: float) -> tuple[dict | None, set[int], dict]:
    """Admit a caption-below, full-width grid only when PDF rulings bound it."""
    audit = {"table_number": caption["number"], "page_lane": "full",
             "caption_source_word_indexes": [w["word_index"] for w in caption["words"]]}
    wide = sorted((y, x0, x1) for y, x0, x1 in rules if x1 - x0 >= width * .3
                  and y < caption["top"] and x0 - 12 <= min(float(w["x0"]) for w in caption["words"]) <= x1 + 12)
    if not wide or caption["top"] - wide[-1][0] > 12:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_adjacent_full_width_bottom_rule"}
    bottom, x0, x1 = wide[-1]
    possible_top = [item for item in wide[:-1] if bottom - 550 < item[0] < bottom - 25
                    and abs(item[1] - x0) < 8 and abs(item[2] - x1) < 8]
    if not possible_top:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_matching_full_width_top_rule"}
    top = min(item[0] for item in possible_top)
    if x1 - x0 < width * .6:
        # These bounded caption-below grids may contain stacked equations or
        # multi-line cells. Their row boundaries are not established by the
        # outer border alone, so retain the complete grid and caption as text.
        local = [w for w in words if x0 - 8 <= _center(w) <= x1 + 8
                 and caption["top"] - 1 <= float(w["top"]) < min(height * .925, bottom + 85)]
        caption_lines = group_lines(local)
        cap_bottom = max(float(w["bottom"]) for w in caption["words"])
        for line in caption_lines:
            line_top = min(float(w["top"]) for w in line)
            if line_top - cap_bottom > 13:
                break
            cap_bottom = max(cap_bottom, max(float(w["bottom"]) for w in line))
        return None, set(), {**audit, "status": "uncertain",
            "reason": "bounded_caption_below_grid_has_unproven_rows",
            "unparsed_bbox": [x0 - 8, top - 1, x1 + 8, cap_bottom + .1]}
    vertical = sorted({round(float(item["x0"]), 2) for item in pdf_page.lines
                       if abs(float(item["x1"]) - float(item["x0"])) <= 1
                       and float(item["bottom"]) - float(item["top"]) >= 5
                       and top - 1 <= float(item["top"]) <= top + 25
                       and x0 - 2 <= float(item["x0"]) <= x1 + 2})
    if len(vertical) < 3 or vertical[0] > x0 + 5 or vertical[-1] < x1 - 5:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_full_width_cell_boundaries"}

    def cells_for(line: Sequence[dict]) -> list[list[dict]]:
        cells = [[] for _ in range(len(vertical) - 1)]
        for word in sorted(line, key=lambda w: (w["x0"], w["word_index"])):
            center = _center(word)
            for col, (left, right) in enumerate(zip(vertical, vertical[1:])):
                if left - 2 <= center < right + (2 if col == len(cells) - 1 else 0):
                    cells[col].append(word)
                    break
        return cells

    grid_words = [w for w in words if top <= float(w["top"]) < bottom
                  and x0 - 2 <= _center(w) <= x1 + 2]
    grid_lines = group_lines(grid_words)
    if len(grid_lines) < 3:
        return None, set(), {**audit, "status": "uncertain", "reason": "grid_has_too_few_lines"}
    header_line = grid_lines[0]
    header_groups = cells_for(header_line)
    if sum(bool(g) for g in header_groups) < 2:
        return None, set(), {**audit, "status": "uncertain", "reason": "grid_header_unreadable"}
    rows: list[list[list[dict]]] = []
    for line in grid_lines[1:]:
        cells = cells_for(line)
        if sum(len(cell) for cell in cells) != len(line):
            return None, set(), {**audit, "status": "uncertain", "reason": "grid_word_outside_cells"}
        if sum(bool(cell) for cell in cells) >= 2:
            rows.append(cells)
        elif rows:
            for col, cell in enumerate(cells):
                rows[-1][col].extend(cell)
        else:
            return None, set(), {**audit, "status": "uncertain", "reason": "grid_row_unreadable"}
    if len(rows) < 2:
        return None, set(), {**audit, "status": "uncertain", "reason": "grid_has_too_few_rows"}
    caption_lines = group_lines([w for w in words if caption["top"] - 1 <= float(w["top"]) < min(height * .925, bottom + 85)])
    cap_words = []
    previous_bottom = None
    for line in caption_lines:
        line_top = min(float(w["top"]) for w in line)
        if previous_bottom is not None and line_top - previous_bottom > 13:
            break
        cap_words.extend(line)
        previous_bottom = max(float(w["bottom"]) for w in line)
    if not set(w["word_index"] for w in caption["words"]) <= set(w["word_index"] for w in cap_words):
        return None, set(), {**audit, "status": "uncertain", "reason": "caption_continuation_not_contiguous"}
    selected = grid_words + cap_words
    indexes = [w["word_index"] for w in selected]
    if len(indexes) != len(set(indexes)):
        return None, set(), {**audit, "status": "uncertain", "reason": "grid_caption_word_overlap"}
    return ({"caption_words": cap_words, "header_groups": header_groups,
             "header_words": header_line, "rows": rows,
             "selected_words": selected, "selected_lines": grid_lines + caption_lines,
             "bottom_rule": bottom, "lane": "full", "number": caption["number"]},
            set(indexes),
            {**audit, "status": "structured_geometry_consistent", "source_word_count": len(indexes),
             "row_count": len(rows), "column_count": len(header_groups), "bottom_rule": bottom,
             "caption_position": "below"})


def _candidate_above_ruled_table(caption: dict, words: Sequence[dict],
                                  rules: Sequence[tuple[float, float, float]], width: float,
                                  height: float, next_caption_top: float,
                                  pdf_page: Any | None = None) -> tuple[dict | None, set[int], dict]:
    """Use caption-aligned top/header/bottom rulings as positive table scope."""
    audit = {"table_number": caption["number"], "page_lane": caption["lane"],
             "caption_source_word_indexes": [w["word_index"] for w in caption["words"]]}
    ruled = sorted((y, x0, x1) for y, x0, x1 in rules if x1 - x0 >= width * .3)
    caption_bottom = max(float(w["bottom"]) for w in caption["words"])
    caption_x = min(float(w["x0"]) for w in caption["words"])
    tops = [item for item in ruled if 0 <= item[0] - caption_bottom <= 30
            and item[1] - 5 <= caption_x <= item[2] + 5]
    if not tops:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_adjacent_ruled_top"}
    top, x0, x1 = tops[0]
    later = [item for item in ruled if item[0] > top + 7
             and abs(item[1] - x0) < 8 and abs(item[2] - x1) < 8]
    if not later or later[0][0] - top > 30:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_full_width_header_rule"}
    header_bottom = later[0][0]
    bottoms = [item for item in later[1:] if item[0] > header_bottom + 15
               and item[0] < min(next_caption_top, height * .93)]
    if not bottoms:
        return None, set(), {**audit, "status": "uncertain", "reason": "no_full_width_bottom_rule"}
    run = [bottoms[0]]
    for item in bottoms[1:]:
        if item[0] - run[-1][0] > 40:
            break
        run.append(item)
    bottom = run[-1][0]
    audit["unparsed_bbox"] = [x0, float(caption["top"]), x1, bottom]
    header_lines = group_lines([w for w in words if top - 1 <= float(w["top"]) < header_bottom
                                and x0 - 2 <= _center(w) <= x1 + 2])
    if not header_lines:
        return None, set(), {**audit, "status": "uncertain", "reason": "full_width_header_unreadable"}
    header_line = max(header_lines, key=len)
    groups = _header(header_line, width, gap=7)
    if groups is None or max(w["x1"] for w in header_line) - min(w["x0"] for w in header_line) < width * .2:
        return None, set(), {**audit, "status": "uncertain", "reason": "full_width_header_unreadable"}
    starts = [min(w["x0"] for w in g) for g in groups]
    page_edges = pdf_page.edges if pdf_page is not None else []
    line_edges = [edge for edge in page_edges if edge.get("object_type") == "line"]
    vertical_source = line_edges if line_edges else [edge for edge in page_edges
                                                     if edge.get("object_type") == "rect_edge"]
    vertical = sorted({round(float(edge["x0"]), 1) for edge in vertical_source
                       if edge.get("orientation") == "v"
                       and abs(float(edge["x1"]) - float(edge["x0"])) <= 1
                       and float(edge["bottom"]) - float(edge["top"]) >= 5
                       and float(edge["top"]) < bottom and float(edge["bottom"]) > top
                       and x0 - 2 <= float(edge["x0"]) <= x1 + 2})
    clustered = []
    for x in vertical:
        if not clustered or x - clustered[-1] > 1.2:
            clustered.append(x)
    boundaries = (clustered if len(clustered) == len(groups) + 1
                  and clustered[0] <= x0 + 2 and clustered[-1] >= x1 - 2 else None)

    def cells_for(line: Sequence[dict]) -> list[list[dict]]:
        if boundaries is None:
            return _split_cells(line, starts)
        cells: list[list[dict]] = [[] for _ in groups]
        for word in sorted(line, key=lambda w: (w["x0"], w["word_index"])):
            center = _center(word)
            for col, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
                if left - 2 <= center < right + (2 if col == len(cells) - 1 else 0):
                    cells[col].append(word)
                    break
        return cells
    for continuation in header_lines:
        if continuation is header_line:
            continue
        cells = cells_for(continuation)
        for col, cell in enumerate(cells):
            groups[col].extend(cell)
    body_lines = group_lines([w for w in words if header_bottom <= float(w["top"]) < bottom
                              and x0 - 2 <= _center(w) <= x1 + 2])
    dense_lines = [line for line in body_lines if len(split_segments(line, gap=7)) >= 2]
    if sum(len(split_segments(line, gap=7)) > len(groups) for line in dense_lines) >= 2:
        return None, set(), {**audit, "status": "uncertain",
                             "reason": "ruled_body_has_more_physical_columns_than_header"}
    if len(groups) >= 2 and len(dense_lines) >= 3:
        shifted = sum(min(float(w["x0"]) for w in line) > starts[0] + 35
                      for line in dense_lines)
        if shifted >= max(3, len(dense_lines) * .5):
            return None, set(), {**audit, "status": "uncertain",
                                 "reason": "ruled_body_lead_not_aligned_with_header"}
    rows: list[list[list[dict]]] = []
    for line in body_lines:
        cells = cells_for(line)
        if sum(len(cell) for cell in cells) != len(line):
            return None, set(), {**audit, "status": "uncertain", "reason": "ruled_cell_word_outside_boundaries"}
        if bool(cells[0]) and sum(bool(cell) for cell in cells) >= 2:
            rows.append(cells)
        elif rows and sum(bool(cell) for cell in cells):
            for col, cell in enumerate(cells):
                rows[-1][col].extend(cell)
        else:
            return None, set(), {**audit, "status": "uncertain", "reason": "full_width_row_unreadable"}
    if len(rows) < 2:
        return None, set(), {**audit, "status": "uncertain", "reason": "full_width_rows_insufficient"}
    cap_lines = group_lines([w for w in words if caption["top"] - 1 <= float(w["top"]) < top
                             and x0 - 2 <= _center(w) <= x1 + 2])
    cap_words = [w for line in cap_lines for w in line]
    if not set(w["word_index"] for w in caption["words"]) <= set(w["word_index"] for w in cap_words):
        return None, set(), {**audit, "status": "uncertain", "reason": "full_width_caption_unreadable"}
    selected_lines = cap_lines + header_lines + body_lines
    selected = [w for line in selected_lines for w in line]
    indexes = [w["word_index"] for w in selected]
    if len(indexes) != len(set(indexes)):
        return None, set(), {**audit, "status": "uncertain", "reason": "full_width_word_overlap"}
    scope = "full" if x1 - x0 >= width * .6 else "bounded"
    return ({"caption_words": cap_words, "header_groups": groups,
             "header_words": [w for line in header_lines for w in line],
             "rows": rows, "selected_words": selected, "selected_lines": selected_lines,
             "bottom_rule": bottom, "lane": scope, "number": caption["number"]},
            set(indexes),
            {**audit, "status": "structured_geometry_consistent", "source_word_count": len(indexes),
             "row_count": len(rows), "column_count": len(groups), "bottom_rule": bottom,
             "caption_position": "above", "ruling_scope": scope})


def _table_record(candidate: dict, page_number: int, table_index: int) -> tuple[dict, dict]:
    table_id = f"v4-p{page_number:04d}-t{table_index:02d}"
    columns = []
    for col, group in enumerate(candidate["header_groups"]):
        columns.append({"column_index": col, "header": words_text(group), "bbox": bbox(group),
                        "source_word_indexes": [w["word_index"] for w in group]})
    rows = []
    for row_index, cells in enumerate(candidate["rows"], 1):
        cell_records = []
        for col, cell in enumerate(cells):
            cell_records.append({"unit_id": f"{table_id}-r{row_index:03d}-c{col+1:02d}",
                                 "row_index": row_index, "column_index": col,
                                 "column_header": columns[col]["header"], "text": words_text(cell),
                                 "bbox": bbox(cell), "source_word_indexes": [w["word_index"] for w in cell]})
        rows.append({"row_id": f"{table_id}-r{row_index:03d}", "row_index": row_index, "cells": cell_records})
    selected = candidate["selected_words"]
    table = {"table_id": table_id, "table_number": candidate["number"],
             "caption": words_text(candidate["caption_words"]), "bbox": bbox(selected),
             "spatial_lane": candidate["lane"],
             "detection_method": "caption_local_lane_header_rows_rules_v4",
             "detection_status": "structured_geometry_consistent_not_semantically_adjudicated",
             "caption_source_word_indexes": [w["word_index"] for w in candidate["caption_words"]],
             "columns": columns, "rows": rows,
             "source_word_indexes": sorted(w["word_index"] for w in selected),
             "_source_lines": candidate["selected_lines"]}
    region = table_region(table, page_number)
    region["spatial_lane"] = candidate["lane"]
    table["caption_unit"]["source_word_indexes"] = table["caption_source_word_indexes"]
    for row in table["rows"]:
        row["source_word_indexes"] = [i for cell in row["cells"] for i in cell["source_word_indexes"]]
    table["header_units"] = [{"unit_id": f"{table_id}-header-c{col['column_index']+1:02d}",
                              "text": col["header"], "region_char_span": col["region_char_span"],
                              "column_index": col["column_index"], "bbox": col["bbox"],
                              "source_word_indexes": col["source_word_indexes"]} for col in columns]
    region["source_word_indexes"] = [t["source_word_index"] for t in region["tokens"]]
    table["source_word_indexes"] = region["source_word_indexes"]
    return table, region


def _text_regions(page_number: int, words: Sequence[dict], split_x: float | None,
                  obstacles: Sequence[tuple[str, float, float]], uncertain: Sequence[dict],
                  width: float, height: float) -> tuple[list[dict], list[tuple[float, float]]]:
    """Partition words by local full-width interruptions and column bands."""
    regions = []
    index = 1
    header = [w for w in words if float(w["top"]) < height * .065]
    footer = [w for w in words if float(w["bottom"]) > height * .925]
    middle = [w for w in words if w not in header and w not in footer]
    for kind, selected in (("header", header),):
        if selected:
            regions.append(text_region(f"v4-p{page_number:04d}-r{index:04d}", page_number, kind, None, selected))
            index += 1
    full_blocks: list[list[dict]] = []
    if split_x is not None:
        for line in group_lines(middle):
            left = [w for w in line if _lane(w, split_x) == "left"]
            right = [w for w in line if _lane(w, split_x) == "right"]
            if not left or not right or len(line) < 6:
                continue
            ordered = sorted(line, key=lambda w: (w["x0"], w["word_index"]))
            gutter_gaps = [float(b["x0"]) - float(a["x1"]) for a, b in zip(ordered, ordered[1:])
                           if min(abs(float(a["x1"]) - split_x), abs(float(b["x0"]) - split_x)) < width * .09]
            if any(gap > 12 for gap in gutter_gaps):
                continue
            seam = min(float(w["x0"]) for w in right) - max(float(w["x1"]) for w in left)
            if seam > 12 or max(w["x1"] for w in line) - min(w["x0"] for w in line) < width * .55:
                continue
            if (full_blocks and min(w["top"] for w in line) - max(w["bottom"] for w in full_blocks[-1]) <= 18):
                full_blocks[-1].extend(line)
            else:
                full_blocks.append(list(line))
    full_indexes = {w["word_index"] for block in full_blocks for w in block}
    full_barriers = [(min(w["top"] for w in block), max(w["bottom"] for w in block)) for block in full_blocks]
    for block in full_blocks:
        region = text_region(f"v4-p{page_number:04d}-r{index:04d}", page_number,
                             "body_full_width", None, block)
        regions.append(region)
        index += 1
    lane_names = ("left", "right") if split_x is not None else ("full",)
    for lane in lane_names:
        local = [w for w in middle if w["word_index"] not in full_indexes and _lane(w, split_x) == lane]
        bounds = sorted({edge for o_lane, top, bottom in obstacles
                         if o_lane in (lane, "full") for edge in (top, bottom)}
                        | {edge for top, bottom in full_barriers for edge in (top, bottom)})
        bins: list[list[dict]] = [[] for _ in range(len(bounds) + 1)]
        for word in local:
            bin_index = sum(float(word["top"]) >= edge for edge in bounds)
            bins[bin_index].append(word)
        for bucket in bins:
            if not bucket:
                continue
            kind = "body_column" if lane != "full" else "body_full_width"
            status = None
            for candidate in uncertain:
                if candidate["page_lane"] == lane and any(w["word_index"] in candidate["word_indexes"] for w in bucket):
                    kind = "table_unparsed"
                    status = "structure_uncertain"
                    break
            region = text_region(f"v4-p{page_number:04d}-r{index:04d}", page_number,
                                 kind, {"left": 1, "right": 2}.get(lane), bucket)
            if status:
                region["structure_status"] = status
            regions.append(region)
            index += 1
    if footer:
        regions.append(text_region(f"v4-p{page_number:04d}-r{index:04d}", page_number, "footer", None, footer))
    return regions, full_barriers


def _build_page(old_page: dict, pdf_page: Any) -> tuple[dict, list[dict]]:
    page_number = int(old_page["page"])
    width, height = float(old_page["width_points"]), float(old_page["height_points"])
    words = copy.deepcopy(old_page["source_words"])
    out_of_bounds = [w["word_index"] for w in words
                     if float(w["x0"]) < -1 or float(w["x1"]) > width + 1
                     or float(w["top"]) < -1 or float(w["bottom"]) > height + 1]
    severe_source_anomaly = len(out_of_bounds) >= 50 and len(out_of_bounds) / max(1, len(words)) > .05
    split_x = _page_split(words, width, height)
    captions = [] if severe_source_anomaly else _caption_candidates(words, split_x)
    rules = _horizontal_rules(pdf_page)
    used: set[int] = set()
    tables, table_regions, unparsed_regions, obstacles, audits, uncertain = [], [], [], [], [], []
    for caption in captions:
        later = [c["top"] for c in captions if c["lane"] == caption["lane"] and c["top"] > caption["top"]]
        next_top = min(later) if later else height * .925
        available = [w for w in words if w["word_index"] not in used]
        candidate, indexes, audit = _candidate_below_ruled_table(caption, available, pdf_page,
                                                                  rules, width, height)
        if candidate is None and "unparsed_bbox" not in audit:
            below_reason = audit["reason"]
            candidate, indexes, audit = _candidate_above_ruled_table(caption, available, rules,
                                                                      width, height, next_top,
                                                                      pdf_page=pdf_page)
            if candidate is None and below_reason == "no_adjacent_full_width_bottom_rule" and audit["reason"] == "no_adjacent_ruled_top":
                candidate, indexes, audit = _candidate_table(caption, available,
                                                               split_x, rules, width, height, next_top)
        audit["page"] = page_number
        if candidate is None:
            if "unparsed_bbox" in audit:
                x0, top, x1, bottom = audit["unparsed_bbox"]
                selected = [w for w in available if top - 1 <= float(w["top"]) < bottom
                            and x0 - 2 <= _center(w) <= x1 + 2]
                indexes = {w["word_index"] for w in selected}
                if indexes:
                    used.update(indexes)
                    region = text_region(f"v4-p{page_number:04d}-unparsed-{len(unparsed_regions)+1:02d}",
                                         page_number, "table_unparsed", None, selected)
                    region["structure_status"] = "structure_uncertain"
                    region["spatial_lane"] = "full" if x1 - x0 >= width * .6 else "bounded"
                    unparsed_regions.append(region)
                    obstacle_lane = region["spatial_lane"]
                    if obstacle_lane == "bounded":
                        obstacle_lane = "left" if split_x is None or x0 < split_x else "right"
                    obstacles.append((obstacle_lane, top, bottom))
                    audit["unparsed_unit_id"] = region["unit_id"]
                    audit["unparsed_source_word_count"] = len(indexes)
                    audits.append(audit)
                    continue
            # Mark a local, bounded caption neighborhood as uncertain; all its
            # words still flow into text regions, never manufactured cells.
            audit["word_indexes"] = [w["word_index"] for w in words if _lane(w, split_x) == caption["lane"]
                                     and caption["top"] <= w["top"] < min(next_top, caption["top"] + 90)]
            uncertain.append(audit)
            audits.append(audit)
            continue
        if indexes & used:
            raise ValueError(f"overlapping_table_words:p{page_number:04d}")
        used.update(indexes)
        table, region = _table_record(candidate, page_number, len(tables) + 1)
        tables.append(table)
        table_regions.append(region)
        top = min(w["top"] for w in candidate["selected_words"])
        bottom = max(w["bottom"] for w in candidate["selected_words"])
        obstacle_lane = candidate["lane"]
        if obstacle_lane == "bounded":
            obstacle_lane = "left" if split_x is None or table["bbox"][0] < split_x else "right"
        obstacles.append((obstacle_lane, top, bottom))
        audit["table_id"] = table["table_id"]
        audit["bbox"] = table["bbox"]
        audits.append(audit)
    remaining = [w for w in words if w["word_index"] not in used]
    text_regions, full_text_barriers = _text_regions(page_number, remaining, split_x,
                                                     obstacles, uncertain, width, height)
    # Column-major reading order within bands separated by full-width content.
    # P05 reads its entire left lane before the independent right figure/prose.
    barriers = sorted(full_text_barriers + [(top, bottom) for lane, top, bottom in obstacles if lane == "full"])
    def key(region: dict) -> tuple:
        kind = region["region_kind"]
        if kind == "header":
            return (-1, 0, float(region["bbox"][1]), region["unit_id"])
        if kind == "footer":
            return (len(barriers) + 1, 0, float(region["bbox"][1]), region["unit_id"])
        top = float(region["bbox"][1])
        stage = sum(bottom <= top for _, bottom in barriers)
        is_full = (region.get("column_index") is None and
                   (kind == "body_full_width" or region.get("spatial_lane") == "full"))
        table_lane = region.get("spatial_lane")
        if table_lane == "bounded":
            table_lane = "left" if split_x is None or region["bbox"][0] < split_x else "right"
        lane = 3 if is_full and barriers else (region.get("column_index") or
                                               {"left": 1, "right": 2}.get(table_lane, 1))
        return (stage, lane, top, region["unit_id"])
    regions = sorted(text_regions + table_regions + unparsed_regions, key=key)
    for order, region in enumerate(regions, 1):
        region["reading_order"] = order
    page = {"page": page_number, "width_points": old_page["width_points"],
            "height_points": old_page["height_points"],
            "layout_mode": "two_column" if split_x is not None else "single_column_or_full_width",
            "spatial_split_x": round(split_x, 3) if split_x is not None else None,
            "source_word_count": len(words), "regions": regions, "tables": tables,
            "source_words": words}
    if out_of_bounds:
        audits.append({"page": page_number, "status": "source_observation_anomaly",
                       "reason": "saved_word_bbox_outside_pdf_page_bounds",
                       "out_of_bounds_word_count": len(out_of_bounds),
                       "out_of_bounds_word_indexes": out_of_bounds,
                       "structured_tables_declined": severe_source_anomaly,
                       "release_blocker": severe_source_anomaly})
    return page, audits


def build_spatial_page(source_page: dict, pdf_page: Any) -> tuple[dict, list[dict]]:
    """Stable page-level geometry API for a later visibility-aware profile."""
    return _build_page(source_page, pdf_page)


def compact_model_input_v4(layout: Mapping[str, Any]) -> dict:
    if layout.get("schema_version") != LAYOUT_VERSION:
        raise ValueError("v4_layout_required")
    from .paper_layout_evidence import compact_model_input
    projected = compact_model_input(layout)
    projected["schema_version"] = INPUT_VERSION
    for source_page, target_page in zip(layout["pages"], projected["pages"]):
        statuses = {r["unit_id"]: r["structure_status"] for r in source_page["regions"]
                    if "structure_status" in r}
        for region in target_page["text_regions"]:
            if region["unit_id"] in statuses:
                region["structure_status"] = statuses[region["unit_id"]]
    projected.pop("model_input_sha256")
    projected["model_input_sha256"] = sha256_bytes(canonical_json_bytes(projected))
    return projected


def build_bundle(v3_layout: dict, *, pdf_path: Path) -> tuple[dict, str, dict, list[dict], dict]:
    """Build v4 artifacts in memory; callers choose a new output directory."""
    import pdfplumber
    if v3_layout.get("schema_version") != "paper-evidence-layout/v3":
        raise ValueError("v3_layout_required")
    pdf_path = Path(pdf_path)
    if sha256_file(pdf_path) != v3_layout.get("source_pdf_sha256"):
        raise ValueError("pdf_identity_mismatch")
    old_without_hash = dict(v3_layout)
    old_hash = old_without_hash.pop("preprocessing_sha256", None)
    if old_hash != sha256_bytes(canonical_json_bytes(old_without_hash)):
        raise ValueError("v3_preprocessing_identity_mismatch")
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != len(v3_layout["pages"]):
            raise ValueError("pdf_page_count_mismatch")
        pairs = [_build_page(old_page, pdf_page) for old_page, pdf_page in zip(v3_layout["pages"], pdf.pages)]
    pages = [page for page, _ in pairs]
    decisions = [decision for _, items in pairs for decision in items]
    reading = assign_document_spans(pages)
    layout = {key: copy.deepcopy(value) for key, value in v3_layout.items()
              if key not in {"pages", "preprocessing_sha256", "derived_from_v2_preprocessing_sha256"}}
    layout.update({"schema_version": LAYOUT_VERSION, "preprocessing_profile": PROFILE,
                   "derived_from_v3_preprocessing_sha256": old_hash,
                   "geometry_source": {"source": "same_sha256_pdf_horizontal_rules",
                                       "pdf_sha256": v3_layout["source_pdf_sha256"],
                                       "extractor": "pdfplumber", "version": pdfplumber.__version__},
                   "pages": pages, "page_count": len(pages),
                   "region_count": sum(len(p["regions"]) for p in pages),
                   "table_count": sum(len(p["tables"]) for p in pages),
                   "layout_token_count": sum(len(r["tokens"]) for p in pages for r in p["regions"]),
                   "reading_text_sha256": sha256_bytes(reading.encode("utf-8"))})
    layout["span_conventions"]["character_offsets"] = "Unicode code-point offsets into reading_text.txt; start inclusive, end exclusive"
    layout["preprocessing_sha256"] = sha256_bytes(canonical_json_bytes(layout))
    model_input = compact_model_input_v4(layout)
    mapping = []
    for old_page, new_page in zip(v3_layout["pages"], pages):
        old_units = [r for r in old_page["regions"] if r["region_kind"] != "table"]
        old_units.extend(u for t in old_page["tables"] for u in
                         [t["caption_unit"], *t.get("header_units", []), *t["rows"],
                          *(cell for row in t["rows"] for cell in row["cells"])])
        old_sets = {u["unit_id"]: set(u.get("source_word_indexes", [])) for u in old_units}
        new_units = [r for r in new_page["regions"] if r["region_kind"] != "table"]
        new_units.extend(u for t in new_page["tables"] for u in
                         [t["caption_unit"], *t["header_units"], *t["rows"],
                          *(cell for row in t["rows"] for cell in row["cells"])])
        for unit in new_units:
            indexes = set(unit.get("source_word_indexes", []))
            mapping.append({"v4_unit_id": unit["unit_id"], "page": new_page["page"],
                            "v3_overlaps": [{"unit_id": oid, "source_word_overlap": len(indexes & oset)}
                                            for oid, oset in old_sets.items() if indexes & oset]})
    audit = {"schema_version": "paper-layout-spatial-audit/v4", "paper_id": layout["paper_id"],
             "source_pdf_sha256": layout["source_pdf_sha256"],
             "source_v3_preprocessing_sha256": old_hash,
             "candidate_decisions": decisions,
             "structured_count": sum(d["status"] == "structured_geometry_consistent" for d in decisions),
             "uncertain_count": sum(d["status"] == "uncertain" for d in decisions),
             "source_observation_anomaly_count": sum(d["status"] == "source_observation_anomaly" for d in decisions),
             "release_blockers": [d for d in decisions if d.get("release_blocker")]}
    errors = validate_layout_bundle_v4(layout, reading, model_input)
    if errors:
        raise ValueError("v4_validation_failed:" + ",".join(errors[:8]))
    return layout, reading, model_input, mapping, audit


def validate_layout_bundle_v4(layout: Mapping[str, Any], reading: str,
                              model_input: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if layout.get("schema_version") != LAYOUT_VERSION or model_input.get("schema_version") != INPUT_VERSION:
        return ["v4_schema_pair_required"]
    if layout.get("preprocessing_profile") != PROFILE:
        errors.append("profile_mismatch")
    if layout.get("geometry_source", {}).get("pdf_sha256") != layout.get("source_pdf_sha256"):
        errors.append("geometry_pdf_identity_mismatch")
    if layout.get("reading_text_sha256") != sha256_bytes(reading.encode("utf-8")):
        errors.append("reading_hash_mismatch")
    payload = dict(layout)
    observed = payload.pop("preprocessing_sha256", None)
    if observed != sha256_bytes(canonical_json_bytes(payload)):
        errors.append("preprocessing_hash_mismatch")
    if dict(model_input) != compact_model_input_v4(layout):
        errors.append("compact_derivation_mismatch")
    if len(layout.get("pages", [])) != layout.get("page_count"):
        errors.append("page_count_mismatch")
    ids: set[str] = set()
    token_index = 0
    for page in layout.get("pages", []):
        source = {w["word_index"]: w for w in page["source_words"]}
        if len(source) != len(page["source_words"]) or len(source) != page["source_word_count"]:
            errors.append(f"source_word_inventory_mismatch:p{page['page']:04d}")
        assigned = []
        if [r["reading_order"] for r in page["regions"]] != list(range(1, len(page["regions"]) + 1)):
            errors.append(f"reading_order_mismatch:p{page['page']:04d}")
        for region in page["regions"]:
            uid = region["unit_id"]
            if uid in ids:
                errors.append("duplicate_unit_id:" + uid)
            ids.add(uid)
            span = region["document_char_span"]
            if reading[span["start"]:span["end"]] != region["text"]:
                errors.append("region_span_mismatch:" + uid)
            actual = [t["source_word_index"] for t in region["tokens"]]
            if actual != region["source_word_indexes"]:
                errors.append("region_word_declaration_mismatch:" + uid)
            assigned.extend(actual)
            for token in region["tokens"]:
                sw = source.get(token["source_word_index"])
                if sw is None or token["text"] != sw["text"] or token["bbox"] != [sw[k] for k in ("x0", "top", "x1", "bottom")]:
                    errors.append("token_source_mismatch:" + uid)
                if token["document_token_index"] != token_index:
                    errors.append("token_index_mismatch:" + uid)
                token_index += 1
                ts = token["document_char_span"]
                if reading[ts["start"]:ts["end"]] != token["text"]:
                    errors.append("token_span_mismatch:" + uid)
        counts = Counter(assigned)
        if set(counts) != set(source) or any(n != 1 for n in counts.values()):
            errors.append(f"word_partition_mismatch:p{page['page']:04d}")
        table_regions = {r["table_id"]: r for r in page["regions"] if r.get("table_id")}
        for table in page["tables"]:
            tid = table["table_id"]
            region = table_regions.get(tid)
            if region is None or set(table["source_word_indexes"]) != set(region["source_word_indexes"]):
                errors.append("table_region_words_mismatch:" + tid)
            split_x = page.get("spatial_split_x")
            lane = table.get("spatial_lane")
            if lane not in {"left", "right", "full", "bounded"}:
                errors.append("table_spatial_lane_missing:" + tid)
            elif split_x is not None and lane in {"left", "right"} and region is not None:
                if any(_lane(source[token["source_word_index"]], split_x) != lane
                       for token in region["tokens"] if token["source_word_index"] in source):
                    errors.append("table_cross_lane_word:" + tid)
            units = [table["caption_unit"], *table["header_units"], *table["rows"],
                     *(cell for row in table["rows"] for cell in row["cells"])]
            content_indexes = list(table["caption_unit"].get("source_word_indexes", []))
            content_indexes.extend(i for unit in table["header_units"] for i in unit.get("source_word_indexes", []))
            content_indexes.extend(i for row in table["rows"] for cell in row["cells"]
                                   for i in cell.get("source_word_indexes", []))
            if Counter(content_indexes) != Counter(table["source_word_indexes"]):
                errors.append("table_unit_word_partition_mismatch:" + tid)
            for unit in units:
                uid = unit["unit_id"]
                if uid in ids:
                    errors.append("duplicate_unit_id:" + uid)
                ids.add(uid)
                span = unit["document_char_span"]
                if reading[span["start"]:span["end"]] != unit["text"]:
                    errors.append("table_unit_span_mismatch:" + uid)
            for row in table["rows"]:
                if len(row["cells"]) != len(table["columns"]):
                    errors.append("cell_column_count_mismatch:" + tid)
                for cell in row["cells"]:
                    if cell["column_header"] != table["columns"][cell["column_index"]]["header"]:
                        errors.append("cell_header_mismatch:" + cell["unit_id"])
    if token_index != layout.get("layout_token_count"):
        errors.append("token_count_mismatch")
    if sum(len(p["regions"]) for p in layout["pages"]) != layout.get("region_count"):
        errors.append("region_count_mismatch")
    if sum(len(p["tables"]) for p in layout["pages"]) != layout.get("table_count"):
        errors.append("table_count_mismatch")
    return errors
