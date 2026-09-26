from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LAYOUT_SCHEMA_VERSION = "paper-evidence-layout/v2"
MODEL_INPUT_SCHEMA_VERSION = "paper-evidence-input/v2"
PROFILE = "pdfplumber-word-bbox-reading-order-table-cell-span-v2"
TABLE_CAPTION = re.compile(r"^Table\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
TABLE_REFERENCE_LEADS = {
    "also",
    "and",
    "are",
    "can",
    "compare",
    "compares",
    "contain",
    "contains",
    "could",
    "describe",
    "describes",
    "give",
    "gives",
    "illustrate",
    "illustrates",
    "is",
    "list",
    "lists",
    "may",
    "present",
    "presents",
    "provide",
    "provides",
    "report",
    "reports",
    "show",
    "shows",
    "summarize",
    "summarizes",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalized_word(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def whitespace_normalized_with_source_map(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace while retaining an offset map into ``value``.

    The generated evidence text is already NFKC. The additional NFKC call
    makes this helper safe for model-supplied quotes while the source map stays
    exact for the stored evidence text used as ``value``.
    """
    normalized = unicodedata.normalize("NFKC", value)
    output: list[str] = []
    source_map: list[tuple[int, int]] = []
    index = 0
    while index < len(normalized):
        if normalized[index].isspace():
            end = index + 1
            while end < len(normalized) and normalized[end].isspace():
                end += 1
            if output and end < len(normalized):
                output.append(" ")
                source_map.append((index, end))
            index = end
            continue
        output.append(normalized[index])
        source_map.append((index, index + 1))
        index += 1
    return "".join(output), source_map


def iter_layout_units(payload: Mapping[str, Any], page: int | None = None) -> Iterable[dict[str, Any]]:
    """Yield each independently citable region, table caption, row, and cell."""
    for page_record in payload.get("pages", []):
        if page is not None and int(page_record["page"]) != page:
            continue
        for region in page_record.get("regions", []):
            if region.get("region_kind") != "table":
                yield region
        for table in page_record.get("tables", []):
            yield table["caption_unit"]
            yield from table.get("header_units", [])
            for row in table.get("rows", []):
                yield row
                yield from row.get("cells", [])


def resolve_evidence_quote(
    payload: Mapping[str, Any], reading_text: str, unit_id: str, quote: str
) -> dict[str, Any]:
    """Resolve a quote to stable document character and layout-token spans.

    Matching is exact after only NFKC plus whitespace collapse. It performs no
    fuzzy, semantic, or case-insensitive matching, so the evidence gate remains
    deterministic and auditable across models.
    """
    units = [unit for unit in iter_layout_units(payload) if unit.get("unit_id") == unit_id]
    if len(units) != 1:
        return {
            "status": "unit_not_unique" if units else "unit_not_found",
            "unit_id": unit_id,
            "match_count": 0,
            "matches": [],
        }
    unit = units[0]
    normalized_text, source_map = whitespace_normalized_with_source_map(str(unit.get("text", "")))
    normalized_quote, _ = whitespace_normalized_with_source_map(quote)
    matches: list[dict[str, Any]] = []
    if normalized_quote:
        offset = 0
        while True:
            found = normalized_text.find(normalized_quote, offset)
            if found < 0:
                break
            local_start = source_map[found][0]
            local_end = source_map[found + len(normalized_quote) - 1][1]
            unit_document_start = int(unit["document_char_span"]["start"])
            document_span = {"start": unit_document_start + local_start, "end": unit_document_start + local_end}
            covered_tokens = []
            for page_record in payload.get("pages", []):
                for region in page_record.get("regions", []):
                    for token in region.get("tokens", []):
                        token_span = token["document_char_span"]
                        if token_span["end"] > document_span["start"] and token_span["start"] < document_span["end"]:
                            covered_tokens.append(int(token["document_token_index"]))
            token_span = {
                "start": min(covered_tokens) if covered_tokens else int(unit["document_token_span"]["start"]),
                "end": max(covered_tokens) + 1 if covered_tokens else int(unit["document_token_span"]["start"]),
            }
            matches.append(
                {
                    "unit_char_span": {"start": local_start, "end": local_end},
                    "document_char_span": document_span,
                    "document_token_span": token_span,
                    "resolved_text": reading_text[document_span["start"]:document_span["end"]],
                }
            )
            offset = found + 1
    return {
        "status": "resolved" if len(matches) == 1 else ("ambiguous" if matches else "quote_not_found"),
        "matching_rule": "NFKC_then_whitespace_collapse_case_sensitive_exact_v1",
        "unit_id": unit_id,
        "match_count": len(matches),
        "matches": matches,
    }


def find_evidence_units(
    payload: Mapping[str, Any], reading_text: str, quote: str, page: int | None = None
) -> list[dict[str, Any]]:
    """Find all units that deterministically resolve ``quote`` on a page."""
    resolved = []
    for unit in iter_layout_units(payload, page=page):
        result = resolve_evidence_quote(payload, reading_text, str(unit["unit_id"]), quote)
        if result["match_count"]:
            resolved.append(result)
    return resolved


def bbox(words: Sequence[Mapping[str, Any]]) -> list[float]:
    if not words:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(float(word["x0"]) for word in words), 3),
        round(min(float(word["top"]) for word in words), 3),
        round(max(float(word["x1"]) for word in words), 3),
        round(max(float(word["bottom"]) for word in words), 3),
    ]


def group_lines(words: Sequence[dict[str, Any]], y_tolerance: float = 2.5) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"], item["word_index"])):
        best: list[dict[str, Any]] | None = None
        best_delta = y_tolerance + 1
        for line in reversed(lines[-8:]):
            delta = abs(float(line[0]["top"]) - float(word["top"]))
            if delta <= y_tolerance and delta < best_delta:
                best = line
                best_delta = delta
        if best is None:
            lines.append([word])
        else:
            best.append(word)
    for line in lines:
        line.sort(key=lambda item: (item["x0"], item["word_index"]))
    return sorted(lines, key=lambda line: (min(item["top"] for item in line), min(item["x0"] for item in line)))


def split_segments(line: Sequence[dict[str, Any]], gap: float = 35.0) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    for word in sorted(line, key=lambda item: item["x0"]):
        if not segments or float(word["x0"]) - float(segments[-1][-1]["x1"]) > gap:
            segments.append([word])
        else:
            segments[-1].append(word)
    return segments


def words_text(words: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(str(word["text"]) for word in words if str(word.get("text", "")).strip())


def extract_page_words(page: Any) -> list[dict[str, Any]]:
    extracted = page.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        x_tolerance=2,
        y_tolerance=2,
        extra_attrs=["fontname", "size"],
    )
    words: list[dict[str, Any]] = []
    for index, item in enumerate(extracted):
        text = normalized_word(str(item.get("text", "")))
        if not text:
            continue
        words.append(
            {
                "word_index": index,
                "text": text,
                "x0": round(float(item["x0"]), 3),
                "top": round(float(item["top"]), 3),
                "x1": round(float(item["x1"]), 3),
                "bottom": round(float(item["bottom"]), 3),
                "fontname": str(item.get("fontname", "")),
                "size": round(float(item.get("size", 0.0)), 3),
            }
        )
    return words


def detect_table_zones(lines: Sequence[list[dict[str, Any]]], page_height: float) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    for line in lines:
        for segment in split_segments(line):
            text = words_text(segment)
            match = TABLE_CAPTION.match(text)
            remainder = text[match.end():].strip() if match else ""
            lead = remainder.split(maxsplit=1)[0].casefold().strip(":,-") if remainder else ""
            table_mentions = len(re.findall(r"\bTable\s*\d+[A-Za-z]?\b", text, re.IGNORECASE))
            is_prose_reference = (
                text.rstrip().endswith((".", ";"))
                or lead in TABLE_REFERENCE_LEADS
                or table_mentions > 1
                or remainder.startswith((")", "]"))
            )
            if match and not is_prose_reference:
                captions.append(
                    {
                        "number": match.group(1),
                        "text": text,
                        "words": segment,
                        "top": min(word["top"] for word in segment),
                        "bottom": max(word["bottom"] for word in segment),
                    }
                )
    captions.sort(key=lambda item: (item["top"], min(word["x0"] for word in item["words"])))
    zones: list[dict[str, Any]] = []
    for index, caption in enumerate(captions):
        next_top = captions[index + 1]["top"] if index + 1 < len(captions) else page_height * 0.94
        # Apply half-open table boundaries at word level. Adjacent captions can
        # share a pdfplumber line with the preceding table's last row; selecting
        # the whole line would duplicate or drop those boundary words.
        candidate_lines = []
        for line in lines:
            selected_words = [
                word for word in line
                if float(caption["top"]) <= float(word["top"]) < float(next_top)
            ]
            if selected_words:
                candidate_lines.append(selected_words)
        selected: list[list[dict[str, Any]]] = []
        previous_bottom: float | None = None
        for line in candidate_lines:
            top = min(float(word["top"]) for word in line)
            if previous_bottom is not None and selected and top - previous_bottom > 32.0:
                break
            selected.append(line)
            previous_bottom = max(float(word["bottom"]) for word in line)
        zone_words = [word for line in selected for word in line]
        if len(selected) < 3 or len(zone_words) < 8:
            continue
        zones.append(
            {
                "table_number": caption["number"],
                "caption": caption["text"],
                "caption_words": caption["words"],
                "lines": selected,
                "top": min(word["top"] for word in zone_words),
                "bottom": max(word["bottom"] for word in zone_words),
                "bbox": bbox(zone_words),
            }
        )
    return zones


def header_groups(line: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return split_segments(line, gap=15.0)


def build_table(zone: Mapping[str, Any], page_number: int, table_index: int) -> dict[str, Any]:
    lines = list(zone["lines"])
    caption_bottom = max(word["bottom"] for word in zone["caption_words"])
    after_caption = [line for line in lines if min(word["top"] for word in line) > caption_bottom]
    header_line: list[dict[str, Any]] | None = None
    groups: list[list[dict[str, Any]]] = []
    for line in after_caption:
        candidate = header_groups(line)
        if len(candidate) >= 2:
            header_line = line
            groups = candidate
            break
    if header_line is None:
        header_line = after_caption[0] if after_caption else list(zone["caption_words"])
        groups = [header_line]
    starts = [min(word["x0"] for word in group) for group in groups]

    def split_cells(line: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        cells = [[] for _ in starts]
        for word in line:
            column = 0
            for candidate_index, start in enumerate(starts):
                if word["x0"] + 3 >= start:
                    column = candidate_index
                else:
                    break
            cells[column].append(word)
        return cells

    header_cells = split_cells(header_line)
    rows: list[list[list[dict[str, Any]]]] = []
    header_top = min(word["top"] for word in header_line)
    data_lines = [line for line in after_caption if min(word["top"] for word in line) > header_top + 2]
    for line in data_lines:
        cells = split_cells(line)
        nonempty = [index for index, cell in enumerate(cells) if cell]
        if not nonempty:
            continue
        first_text = words_text(cells[0])
        has_tail_value = any(cells[index] for index in range(2, len(cells)))
        is_header_continuation = not rows and not cells[0] and has_tail_value
        if is_header_continuation:
            for index, cell in enumerate(cells):
                header_cells[index].extend(cell)
            continue
        new_row = bool(cells[0]) and (has_tail_value or not rows or (first_text[:1].isupper()))
        if new_row:
            rows.append(cells)
        elif rows:
            for index, cell in enumerate(cells):
                rows[-1][index].extend(cell)
        else:
            rows.append(cells)

    table_id = f"p{page_number:04d}-t{table_index:02d}"
    columns = []
    for index, cell_words in enumerate(header_cells):
        columns.append(
            {
                "column_index": index,
                "header": words_text(cell_words),
                "bbox": bbox(cell_words),
                "source_word_indexes": [word["word_index"] for word in cell_words],
            }
        )
    row_records = []
    for row_index, cells in enumerate(rows, start=1):
        cell_records = []
        for column_index, cell_words in enumerate(cells):
            cell_records.append(
                {
                    "unit_id": f"{table_id}-r{row_index:03d}-c{column_index + 1:02d}",
                    "row_index": row_index,
                    "column_index": column_index,
                    "column_header": columns[column_index]["header"],
                    "text": words_text(cell_words),
                    "bbox": bbox(cell_words),
                    "source_word_indexes": [word["word_index"] for word in cell_words],
                }
            )
        row_records.append({"row_id": f"{table_id}-r{row_index:03d}", "row_index": row_index, "cells": cell_records})
    return {
        "table_id": table_id,
        "table_number": str(zone["table_number"]),
        "caption": str(zone["caption"]),
        "bbox": list(zone["bbox"]),
        "detection_method": "caption_anchor_plus_word_geometry_v2",
        "detection_status": "deterministic_candidate_not_semantically_adjudicated",
        "caption_source_word_indexes": [word["word_index"] for word in zone["caption_words"]],
        "columns": columns,
        "rows": row_records,
        "source_word_indexes": sorted({word["word_index"] for line in lines for word in line}),
    }


def text_region(region_id: str, page_number: int, kind: str, column_index: int | None, words: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lines = group_lines(list(words))
    text_parts: list[str] = []
    token_records: list[dict[str, Any]] = []
    cursor = 0
    for line_index, line in enumerate(lines):
        if line_index:
            text_parts.append("\n")
            cursor += 1
        for word_index, word in enumerate(line):
            if word_index:
                text_parts.append(" ")
                cursor += 1
            start = cursor
            value = word["text"]
            text_parts.append(value)
            cursor += len(value)
            token_records.append(
                {
                    "text": value,
                    "region_char_span": {"start": start, "end": cursor},
                    "source_word_index": word["word_index"],
                    "bbox": [word["x0"], word["top"], word["x1"], word["bottom"]],
                }
            )
    text = "".join(text_parts)
    return {
        "region_id": region_id,
        "unit_id": region_id,
        "page": page_number,
        "region_kind": kind,
        "column_index": column_index,
        "bbox": bbox(words),
        "text": text,
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "tokens": token_records,
        "source_word_indexes": [word["word_index"] for line in lines for word in line],
    }


def table_region(table: dict[str, Any], page_number: int) -> dict[str, Any]:
    parts: list[str] = []
    tokens: list[dict[str, Any]] = []
    cursor = 0

    def append_words(source_words: Sequence[dict[str, Any]], separator: str) -> dict[str, int]:
        nonlocal cursor
        if separator:
            parts.append(separator)
            cursor += len(separator)
        start = cursor
        for index, word in enumerate(source_words):
            if index:
                parts.append(" ")
                cursor += 1
            token_start = cursor
            parts.append(word["text"])
            cursor += len(word["text"])
            tokens.append(
                {
                    "text": word["text"],
                    "region_char_span": {"start": token_start, "end": cursor},
                    "source_word_index": word["word_index"],
                    "bbox": [word["x0"], word["top"], word["x1"], word["bottom"]],
                }
            )
        return {"start": start, "end": cursor}

    word_map = {
        word["word_index"]: word
        for line in table.pop("_source_lines")
        for word in line
    }
    caption_words = [word_map[index] for index in table["caption_source_word_indexes"]]
    caption_span = append_words(caption_words, "")
    table["caption_unit"] = {
        "unit_id": f"{table['table_id']}-caption",
        "text": words_text(caption_words),
        "region_char_span": caption_span,
    }
    for column_index, column in enumerate(table["columns"]):
        source_words = [word_map[index] for index in column["source_word_indexes"]]
        column["region_char_span"] = append_words(source_words, "\n" if column_index == 0 else "\t")
    for row in table["rows"]:
        row_start: int | None = None
        row_end = cursor
        for cell_index, cell in enumerate(row["cells"]):
            source_words = [word_map[index] for index in cell["source_word_indexes"]]
            cell["region_char_span"] = append_words(source_words, "\n" if cell_index == 0 else "\t")
            if row_start is None:
                row_start = cell["region_char_span"]["start"]
            row_end = cell["region_char_span"]["end"]
        row["unit_id"] = row["row_id"]
        row["region_char_span"] = {
            "start": row_start if row_start is not None else row_end,
            "end": row_end,
        }
    text = "".join(parts)
    for row in table["rows"]:
        span = row["region_char_span"]
        row["text"] = text[span["start"]:span["end"]]
    return {
        "region_id": table["table_id"],
        "unit_id": table["table_id"],
        "page": page_number,
        "region_kind": "table",
        "column_index": None,
        "bbox": table["bbox"],
        "text": text,
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "tokens": tokens,
        "source_word_indexes": table["source_word_indexes"],
        "table_id": table["table_id"],
    }


def body_regions_for_band(
    words: Sequence[dict[str, Any]], page_number: int, page_width: float, region_start: int
) -> tuple[list[dict[str, Any]], int, str]:
    if not words:
        return [], region_start, "empty"
    midpoint = page_width / 2
    left = [word for word in words if (word["x0"] + word["x1"]) / 2 < midpoint]
    right = [word for word in words if (word["x0"] + word["x1"]) / 2 >= midpoint]
    two_column = len(left) >= 10 and len(right) >= 10
    regions: list[dict[str, Any]] = []
    if two_column:
        for column_index, column_words in enumerate((left, right), start=1):
            region_id = f"p{page_number:04d}-r{region_start:04d}"
            regions.append(text_region(region_id, page_number, "body_column", column_index, column_words))
            region_start += 1
        mode = "two_column"
    else:
        region_id = f"p{page_number:04d}-r{region_start:04d}"
        regions.append(text_region(region_id, page_number, "body_full_width", None, words))
        region_start += 1
        mode = "single_column_or_full_width"
    return regions, region_start, mode


def build_page(page: Any, page_number: int) -> dict[str, Any]:
    words = extract_page_words(page)
    lines = group_lines(words)
    zones = detect_table_zones(lines, float(page.height))
    tables = []
    for table_index, zone in enumerate(zones, start=1):
        table = build_table(zone, page_number, table_index)
        table["_source_lines"] = zone["lines"]
        tables.append(table)
    table_word_indexes = {index for table in tables for index in table["source_word_indexes"]}
    header_limit = float(page.height) * 0.065
    footer_limit = float(page.height) * 0.925
    header_words = [word for word in words if word["word_index"] not in table_word_indexes and word["top"] < header_limit]
    footer_words = [word for word in words if word["word_index"] not in table_word_indexes and word["bottom"] > footer_limit]
    body_words = [
        word for word in words
        if word["word_index"] not in table_word_indexes
        and word not in header_words
        and word not in footer_words
    ]
    events: list[tuple[float, str, Any]] = []
    if header_words:
        events.append((min(word["top"] for word in header_words), "header", header_words))
    for table in tables:
        events.append((table["bbox"][1], "table", table))
    zone_bounds = sorted([(table["bbox"][1], table["bbox"][3]) for table in tables])
    band_edges = [header_limit] + [edge for pair in zone_bounds for edge in pair] + [footer_limit]
    for start, end in zip(band_edges[::2], band_edges[1::2]):
        # Use the extraction anchor (`top`) rather than a glyph midpoint. In
        # two-column PDFs, words from the opposite column can share a visual
        # line with a caption while having slightly different glyph heights.
        band_words = [word for word in body_words if start <= float(word["top"]) < end]
        if band_words:
            events.append((start, "body_band", band_words))
    if not tables and body_words:
        events = [event for event in events if event[1] != "body_band"]
        events.append((header_limit, "body_band", body_words))
    if footer_words:
        events.append((footer_limit, "footer", footer_words))
    events.sort(key=lambda item: (item[0], {"header": 0, "body_band": 1, "table": 2, "footer": 3}[item[1]]))

    regions: list[dict[str, Any]] = []
    region_index = 1
    detected_modes: list[str] = []
    for _, kind, value in events:
        if kind == "table":
            regions.append(table_region(value, page_number))
        elif kind == "body_band":
            built, region_index, mode = body_regions_for_band(value, page_number, float(page.width), region_index)
            regions.extend(built)
            detected_modes.append(mode)
        else:
            region_id = f"p{page_number:04d}-r{region_index:04d}"
            regions.append(text_region(region_id, page_number, kind, None, value))
            region_index += 1
    for order, region in enumerate(regions, start=1):
        region["reading_order"] = order
    for table in tables:
        table.pop("_source_lines", None)
    return {
        "page": page_number,
        "width_points": round(float(page.width), 3),
        "height_points": round(float(page.height), 3),
        "layout_mode": "two_column" if "two_column" in detected_modes else "single_column_or_full_width",
        "source_word_count": len(words),
        "regions": regions,
        "tables": tables,
        "source_words": words,
    }


def assign_document_spans(pages: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    char_cursor = 0
    token_cursor = 0
    for page in pages:
        for region in page["regions"]:
            if parts:
                parts.append("\n\n")
                char_cursor += 2
            start = char_cursor
            parts.append(region["text"])
            char_cursor += len(region["text"])
            region["document_char_span"] = {"start": start, "end": char_cursor}
            region_token_start = token_cursor
            for token in region["tokens"]:
                token["document_token_index"] = token_cursor
                token["document_char_span"] = {
                    "start": start + token["region_char_span"]["start"],
                    "end": start + token["region_char_span"]["end"],
                }
                token_cursor += 1
            region["document_token_span"] = {"start": region_token_start, "end": token_cursor}
            if region.get("table_id"):
                table = next(item for item in page["tables"] if item["table_id"] == region["table_id"])
                units = (
                    [table["caption_unit"]]
                    + list(table.get("header_units", []))
                    + list(table["rows"])
                    + [cell for row in table["rows"] for cell in row["cells"]]
                )
                for unit in units:
                    span = unit["region_char_span"]
                    unit["document_char_span"] = {"start": start + span["start"], "end": start + span["end"]}
                    covered = [
                        token["document_token_index"] for token in region["tokens"]
                        if token["region_char_span"]["start"] >= span["start"]
                        and token["region_char_span"]["end"] <= span["end"]
                    ]
                    unit["document_token_span"] = {
                        "start": min(covered) if covered else region_token_start,
                        "end": max(covered) + 1 if covered else region_token_start,
                    }
    return "".join(parts) + "\n"


def compact_model_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    pages = []
    for page in payload["pages"]:
        text_regions = [
            {
                "unit_id": region["unit_id"],
                "page": region["page"],
                "region_kind": region["region_kind"],
                "column_index": region["column_index"],
                "reading_order": region["reading_order"],
                "bbox": region["bbox"],
                "document_char_span": region["document_char_span"],
                "document_token_span": region["document_token_span"],
                "text": region["text"],
            }
            for region in page["regions"]
            if region["region_kind"] != "table"
        ]
        tables = []
        for table in page["tables"]:
            tables.append(
                {
                    "table_id": table["table_id"],
                    "table_number": table["table_number"],
                    "caption": table["caption_unit"],
                    "bbox": table["bbox"],
                    "detection_status": table["detection_status"],
                    "columns": [{"column_index": column["column_index"], "header": column["header"]} for column in table["columns"]],
                    "rows": [
                        {
                            "row_id": row["row_id"],
                            "unit_id": row["unit_id"],
                            "text": row["text"],
                            "document_char_span": row["document_char_span"],
                            "document_token_span": row["document_token_span"],
                            "cells": [
                                {
                                    "unit_id": cell["unit_id"],
                                    "column_index": cell["column_index"],
                                    "column_header": cell["column_header"],
                                    "document_char_span": cell["document_char_span"],
                                    "document_token_span": cell["document_token_span"],
                                    "text": cell["text"],
                                }
                                for cell in row["cells"]
                            ],
                        }
                        for row in table["rows"]
                    ],
                }
            )
            if "header_units" in table:
                tables[-1]["header_units"] = table["header_units"]
        pages.append({"page": page["page"], "layout_mode": page["layout_mode"], "text_regions": text_regions, "tables": tables})
    result = {
        "schema_version": "paper-evidence-input/v3" if payload["schema_version"] == "paper-evidence-layout/v3" else MODEL_INPUT_SCHEMA_VERSION,
        "paper_id": payload["paper_id"],
        "source_pdf_sha256": payload["source_pdf_sha256"],
        "layout_preprocessing_sha256": payload["preprocessing_sha256"],
        "span_conventions": payload["span_conventions"],
        "pages": pages,
    }
    result["model_input_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def validate_layout_bundle(
    payload: Mapping[str, Any], reading_text: str, model_input: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") not in {LAYOUT_SCHEMA_VERSION, "paper-evidence-layout/v3"}:
        errors.append("layout_schema_version_invalid")
    if model_input.get("schema_version") not in {MODEL_INPUT_SCHEMA_VERSION, "paper-evidence-input/v3"}:
        errors.append("model_input_schema_version_invalid")
    if payload.get("reading_text_sha256") != sha256_bytes(reading_text.encode("utf-8")):
        errors.append("reading_text_sha256_mismatch")
    payload_without_hash = dict(payload)
    observed_preprocessing_hash = payload_without_hash.pop("preprocessing_sha256", None)
    if observed_preprocessing_hash != sha256_bytes(canonical_json_bytes(payload_without_hash)):
        errors.append("preprocessing_sha256_mismatch")
    input_without_hash = dict(model_input)
    observed_input_hash = input_without_hash.pop("model_input_sha256", None)
    if observed_input_hash != sha256_bytes(canonical_json_bytes(input_without_hash)):
        errors.append("model_input_sha256_mismatch")
    if dict(model_input) != compact_model_input(payload):
        errors.append("model_input_derivation_mismatch")

    unit_ids: set[str] = set()
    expected_token_index = 0
    for page in payload.get("pages", []):
        source_indexes = {word["word_index"] for word in page.get("source_words", [])}
        source_words = {word["word_index"]: word for word in page.get("source_words", [])}
        if len(source_indexes) != len(page.get("source_words", [])):
            errors.append(f"duplicate_source_word_index:p{page['page']:04d}")
        assigned: list[int] = []
        for region in page.get("regions", []):
            unit_id = region["unit_id"]
            if unit_id in unit_ids:
                errors.append(f"duplicate_unit_id:{unit_id}")
            unit_ids.add(unit_id)
            span = region["document_char_span"]
            if reading_text[span["start"]:span["end"]] != region["text"]:
                errors.append(f"region_char_span_mismatch:{unit_id}")
            actual_indexes = [token["source_word_index"] for token in region["tokens"]]
            assigned.extend(actual_indexes)
            if set(actual_indexes) != set(region["source_word_indexes"]):
                errors.append(f"region_word_declaration_mismatch:{unit_id}")
            for token in region["tokens"]:
                source_word = source_words.get(token["source_word_index"])
                if source_word is None or token["text"] != source_word["text"]:
                    errors.append(f"token_source_text_mismatch:{unit_id}:{token['source_word_index']}")
                if token["document_token_index"] != expected_token_index:
                    errors.append(f"token_index_noncontiguous:{unit_id}")
                expected_token_index += 1
                token_span = token["document_char_span"]
                if reading_text[token_span["start"]:token_span["end"]] != token["text"]:
                    errors.append(f"token_char_span_mismatch:{unit_id}:{token['document_token_index']}")
        if set(assigned) != source_indexes:
            missing = sorted(source_indexes - set(assigned))
            extra = sorted(set(assigned) - source_indexes)
            errors.append(f"page_word_coverage_mismatch:p{page['page']:04d}:missing={missing[:5]}:extra={extra[:5]}")
        if len(assigned) != len(set(assigned)):
            errors.append(f"page_word_assignment_duplicate:p{page['page']:04d}")
        for table in page.get("tables", []):
            for unit in [table["caption_unit"], *table.get("header_units", []), *table["rows"], *[cell for row in table["rows"] for cell in row["cells"]]]:
                unit_id = unit["unit_id"]
                if unit_id in unit_ids:
                    errors.append(f"duplicate_unit_id:{unit_id}")
                unit_ids.add(unit_id)
                span = unit["document_char_span"]
                if reading_text[span["start"]:span["end"]] != unit["text"]:
                    errors.append(f"table_unit_char_span_mismatch:{unit_id}")
    if payload.get("layout_token_count") != expected_token_index:
        errors.append("layout_token_count_mismatch")
    if payload.get("region_count") != sum(len(page.get("regions", [])) for page in payload.get("pages", [])):
        errors.append("region_count_mismatch")
    if payload.get("table_count") != sum(len(page.get("tables", [])) for page in payload.get("pages", [])):
        errors.append("table_count_mismatch")
    return errors


def preprocess_layout_pdf(paper_id: str, pdf_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pages = [build_page(page, page_number) for page_number, page in enumerate(pdf.pages, start=1)]
    reading_text = assign_document_spans(pages)
    payload: dict[str, Any] = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "paper_id": paper_id,
        "source_pdf_sha256": sha256_file(pdf_path),
        "preprocessing_profile": PROFILE,
        "extractor": {
            "name": "pdfplumber",
            "version": getattr(pdfplumber, "__version__", "unknown"),
            "word_arguments": {"use_text_flow": False, "keep_blank_chars": False, "x_tolerance": 2, "y_tolerance": 2},
        },
        "normalization": {"unicode": "NFKC", "line_endings": "LF", "dehyphenation": False, "ocr": False},
        "coordinate_system": "PDF points, top-left origin, bbox=[x0,top,x1,bottom]",
        "span_conventions": {
            "character_offsets": "Unicode code-point offsets into reading_text_v2.txt; start inclusive, end exclusive",
            "token_offsets": "model-neutral pdf layout-word indexes in document reading order; start inclusive, end exclusive",
            "table_cells": "row-major cell units with stable page/table/row/column IDs",
        },
        "page_count": len(pages),
        "region_count": sum(len(page["regions"]) for page in pages),
        "table_count": sum(len(page["tables"]) for page in pages),
        "layout_token_count": sum(len(region["tokens"]) for page in pages for region in page["regions"]),
        "reading_text_sha256": sha256_bytes(reading_text.encode("utf-8")),
        "pages": pages,
    }
    payload["preprocessing_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    model_input = compact_model_input(payload)
    return payload, reading_text, model_input


def write_outputs(output_dir: Path, payload: Mapping[str, Any], reading_text: str, model_input: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_path = output_dir / "evidence_layout_v2.json"
    text_path = output_dir / "reading_text_v2.txt"
    input_path = output_dir / "evidence_input_v2.json"
    layout_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    text_path.write_text(reading_text, encoding="utf-8", newline="\n")
    input_path.write_text(json.dumps(model_input, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "paper_id": payload["paper_id"],
        "source_pdf_sha256": payload["source_pdf_sha256"],
        "preprocessing_sha256": payload["preprocessing_sha256"],
        "model_input_sha256": model_input["model_input_sha256"],
        "evidence_layout_file_sha256": sha256_file(layout_path),
        "reading_text_file_sha256": sha256_file(text_path),
        "evidence_input_file_sha256": sha256_file(input_path),
        "page_count": payload["page_count"],
        "region_count": payload["region_count"],
        "table_count": payload["table_count"],
        "layout_token_count": payload["layout_token_count"],
    }


def build_corpus(corpus_path: Path, experiment_root: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    records = []
    for paper in corpus["papers"]:
        lane = experiment_root / Path(paper["preprocessing_artifact"]).parent
        payload, reading_text, model_input = preprocess_layout_pdf(paper["paper_id"], lane / "paper.pdf")
        if payload["source_pdf_sha256"] != paper["paper_sha256"]:
            raise ValueError(f"source PDF identity mismatch for {paper['paper_id']}")
        validation_errors = validate_layout_bundle(payload, reading_text, model_input)
        if validation_errors:
            raise ValueError(f"layout validation failed for {paper['paper_id']}: {validation_errors[:10]}")
        record = write_outputs(lane, payload, reading_text, model_input)
        record["validation_status"] = "pass"
        record["candidate_id"] = lane.name
        records.append(record)
    manifest = {
        "schema_version": "paper-layout-preprocessing-manifest/v2",
        "status": "generated_unfrozen_pending_quality_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_corpus_manifest": str(corpus_path),
        "source_corpus_manifest_sha256": sha256_file(corpus_path),
        "preprocessing_profile": PROFILE,
        "record_count": len(records),
        "records": records,
    }
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build layout-aware paper evidence v2 without modifying v1")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_corpus(args.corpus, args.experiment_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
