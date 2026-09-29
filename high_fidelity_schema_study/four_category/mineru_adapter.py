"""Deterministic DocVortex MiddleJson -> independent paper evidence v6 adapter.

This module consumes serialized data. MinerU is intentionally not imported here.
The baseline and raw parser output must also be retained as byte-bound artifacts.
"""
from __future__ import annotations

import copy
from html.parser import HTMLParser
import hashlib
import re

from .common import digest

LAYOUT_VERSION = "paper-evidence-layout/v6"
INPUT_VERSION = "paper-evidence-input/v6"
HEX = re.compile(r"^[0-9a-f]{64}$")
TEXT_TYPES = {"text", "title", "paragraph", "paragraph_title", "doc_title", "list", "code", "algorithm",
              "footnote", "header", "footer", "table_caption", "table_footnote", "image_caption", "chart_caption",
              "page_footnote", "page_number", "ref_text"}
VISUAL_TYPES = {"image", "image_body", "chart", "chart_body", "interline_equation", "equation"}


class _TableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict]] = []
        self.row: list[dict] | None = None
        self.cell: dict | None = None
        self.depth = 0
        self.invalid = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.depth += 1
            if self.depth > 1:
                self.invalid = True
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in {"th", "td"} and self.row is not None:
            try:
                rowspan, colspan = int(attrs.get("rowspan", 1)), int(attrs.get("colspan", 1))
            except (TypeError, ValueError):
                self.invalid = True
                rowspan, colspan = 1, 1
            if rowspan < 1 or colspan < 1:
                self.invalid = True
            self.cell = {"role": "header" if tag == "th" else "data", "rowspan": rowspan,
                         "colspan": colspan, "text": ""}

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.cell is not None:
            self.cell["text"] = " ".join(self.cell["text"].split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.depth:
            self.depth -= 1


def _text(value) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(filter(None, (_text(item) for item in value)))
    if isinstance(value, dict):
        if value.get("type") == "table" or "html" in value:
            return ""
        return _text(value.get("content", "")) or _text(value.get("lines", [])) or _text(value.get("spans", []))
    return ""


def _find_html(value) -> str | None:
    if isinstance(value, str):
        return value if "<table" in value.casefold() else None
    if isinstance(value, dict):
        html = value.get("html")
        if isinstance(html, str) and html.strip():
            return html
        for key in ("content", "blocks", "lines", "spans"):
            found = _find_html(value.get(key))
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_html(item)
            if found:
                return found
    return None


def _bbox(value, width, height):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(type(v) not in {int, float} for v in value):
        raise ValueError("mineru_bbox_invalid")
    x0, y0, x1, y1 = (float(v) for v in value)
    if not (0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1):
        raise ValueError("mineru_bbox_out_of_page")
    return [x0 * width, y0 * height, x1 * width, y1 * height]


def _native_units(page):
    for item in page.get("text_regions", []):
        yield item
    for table in page.get("tables", []):
        yield table["caption"]
        yield from table.get("header_units", [])
        for row in table.get("rows", []):
            yield row
            yield from row.get("cells", [])


def _native_records(page):
    for unit in page.get("text_regions", []):
        yield {"unit": unit, "bbox": unit.get("bbox"), "role": "text_region"}
    for table in page.get("tables", []):
        bbox = table.get("bbox")
        yield {"unit": table["caption"], "bbox": table["caption"].get("bbox") or bbox, "role": "caption"}
        for unit in table.get("header_units", []):
            yield {"unit": unit, "bbox": unit.get("bbox") or bbox, "role": "header"}
        for row in table.get("rows", []):
            yield {"unit": row, "bbox": row.get("bbox") or bbox, "role": "row"}
            for unit in row.get("cells", []):
                yield {"unit": unit, "bbox": unit.get("bbox") or bbox, "role": "cell"}


def _overlap(a, b):
    if not a or not b:
        return False
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _normalized(text):
    return " ".join(text.split())


def _alignment(text, bbox, native, consumed, preferred_role=None):
    norm = _normalized(text)
    ranked = []
    for record in native:
        unit = record["unit"]
        uid = unit["unit_id"]
        if uid in consumed:
            continue
        candidate = _normalized(unit["text"])
        if not norm or not candidate:
            continue
        if not _overlap(bbox, record["bbox"]):
            continue
        # Native text is covered only if its complete normalized string appears.
        # A recognized substring of a larger native unit leaves that native unit
        # available for recovery rather than silently losing the unmatched part.
        if candidate == norm or candidate in norm:
            ranked.append((0 if record["role"] == preferred_role else 1,
                           0 if candidate == text else 1, -len(candidate), uid))
    if not ranked:
        return {"status": "recognized_unaligned", "native_unit_id": None, "similarity": None}
    _, exactness, _, uid = min(ranked)
    consumed.add(uid)
    return {"status": "exact_native_text" if exactness == 0 else "normalized_native_text",
            "native_unit_id": uid, "similarity": 1.0}


def _recognized_records(regions, tables):
    for unit in regions:
        if unit.get("text_origin") == "mineru_recognized" and unit.get("text"):
            yield {"text": unit["text"], "bbox": unit.get("bbox")}
    for table in tables:
        if table.get("source_origin") != "mineru_recognized":
            continue
        units = [table["caption"], *[cell for row in table["rows"] for cell in row["cells"]]]
        if not any(row["cells"] for row in table["rows"]):
            units.extend(table["header_units"])
            units.extend(table["rows"])
        for unit in units:
            if unit.get("text"):
                yield {"text": unit["text"], "bbox": unit.get("bbox") or table.get("bbox")}


def _uncovered_fragments(text, bbox, recognized):
    """Exact token-sequence coverage; return native substrings still uncited."""
    matches = list(re.finditer(r"\S+", text))
    tokens = [m.group() for m in matches]
    covered = [False] * len(tokens)
    for rec in recognized:
        if not _overlap(bbox, rec["bbox"]):
            continue
        needle = rec["text"].split()
        if not needle:
            continue
        if len(needle) > len(tokens):
            if any(needle[j:j + len(tokens)] == tokens for j in range(len(needle) - len(tokens) + 1)):
                covered = [True] * len(tokens)
            continue
        for start in range(len(tokens) - len(needle) + 1):
            if tokens[start:start + len(needle)] == needle and not any(covered[start:start + len(needle)]):
                covered[start:start + len(needle)] = [True] * len(needle)
                break
    fragments = []
    start = None
    for i, flag in enumerate(covered + [True]):
        if not flag and start is None:
            start = i
        if flag and start is not None:
            fragments.append(text[matches[start].start():matches[i - 1].end()])
            start = None
    return fragments, sum(covered), len(tokens)


def _unit(uid, page, kind, order, bbox, text, origin, alignment=None, **extra):
    return {"unit_id": uid, "page": page, "region_kind": kind, "reading_order": order,
            "column_index": None, "bbox": bbox, "text": text, "text_origin": origin,
            "native_alignment": alignment or {"status": "not_applicable", "native_unit_id": None, "similarity": None},
            "document_char_span": {"start": 0, "end": 0},
            "document_token_span": {"start": 0, "end": 0}, **extra}


def _table(block, page, index, order, bbox, native, consumed):
    tid = f"v6-p{page:04d}-t{index:04d}"
    children = block.get("blocks") or block.get("content") or []
    caption = " ".join(filter(None, (_text(child) for child in children
                                    if isinstance(child, dict) and child.get("type") == "table_caption")))
    html = _find_html(block)
    parsed = _TableHTML()
    if html:
        parsed.feed(html)
        parsed.close()
    complete = bool(html and parsed.rows and not parsed.invalid and parsed.depth == 0)
    raw_rows = parsed.rows if complete else []
    # A malformed HTML table remains citable as unparsed recognized text.
    if html and not complete:
        fallback = _TableHTML()
        fallback.feed(html)
        flat = " ".join(c["text"] for row in fallback.rows for c in row if c["text"])
        raw_rows = [[{"role": "data", "rowspan": 1, "colspan": 1, "text": flat}]] if flat else []
    caption_unit = _unit(f"{tid}-caption", page, "table_caption", order, bbox, caption,
                         "mineru_recognized", _alignment(caption, bbox, native, consumed, "caption"))
    rows, header_units, columns = [], [], []
    width = max((sum(cell["colspan"] for cell in row) for row in raw_rows), default=0)
    for col in range(width):
        columns.append({"column_index": col, "header": ""})
    occupied = set()
    for ri, source_row in enumerate(raw_rows):
        cells, col = [], 0
        for ci, source_cell in enumerate(source_row):
            while (ri, col) in occupied:
                col += 1
            text = source_cell["text"]
            cell = _unit(f"{tid}-r{ri:04d}-c{ci:04d}", page, "table_cell", order, bbox, text,
                         "mineru_recognized", _alignment(text, bbox, native, consumed, "cell"), column_index=col,
                         column_header="", row_span=source_cell["rowspan"], col_span=source_cell["colspan"],
                         cell_role=source_cell["role"])
            cells.append(cell)
            if source_cell["role"] == "header":
                for target in range(col, min(width, col + source_cell["colspan"])):
                    if not columns[target]["header"]:
                        columns[target]["header"] = text
            for rr in range(ri, ri + source_cell["rowspan"]):
                for cc in range(col, col + source_cell["colspan"]):
                    if rr != ri or cc != col:
                        occupied.add((rr, cc))
            col += source_cell["colspan"]
        row_text = " | ".join(cell["text"] for cell in cells)
        rows.append({"row_id": f"{tid}-r{ri:04d}", "unit_id": f"{tid}-r{ri:04d}", "text": row_text,
                     "cells": cells, "document_char_span": {"start": 0, "end": 0},
                     "document_token_span": {"start": 0, "end": 0}, "region_kind": "table_row",
                     "text_origin": "mineru_recognized",
                     "native_alignment": {"status": "recognized_unaligned", "native_unit_id": None, "similarity": None},
                     "row_role": "header" if cells and all(c["cell_role"] == "header" for c in cells) else "data"})
    for col in columns:
        if col["header"]:
            header_units.append(_unit(f"{tid}-h{col['column_index']:04d}", page, "table_header", order,
                                      bbox, col["header"], "mineru_recognized",
                                      _alignment(col["header"], bbox, native, consumed, "header")))
    return {"table_id": tid, "table_number": None, "bbox": bbox, "caption": caption_unit,
            "header_units": header_units, "columns": columns, "rows": rows,
            "detection_status": "structured" if complete else "structure_unresolved",
            "structure_status": "parsed_html" if complete else "unparsed_or_missing_html",
            "raw_html": html, "source_block_index": index, "source_origin": "mineru_recognized"}


def _assign_spans(pages):
    parts = []
    char = token = 0
    def assign(unit):
        nonlocal char, token
        if parts:
            parts.append("\n\n")
            char += 2
        text = unit["text"]
        unit["document_char_span"] = {"start": char, "end": char + len(text)}
        n = len(text.split())
        unit["document_token_span"] = {"start": token, "end": token + n}
        parts.append(text)
        char += len(text)
        token += n
    for page in pages:
        for item in page["text_regions"]:
            assign(item)
        for table in page["tables"]:
            assign(table["caption"])
            for unit in table["header_units"]:
                assign(unit)
            for row in table["rows"]:
                assign(row)
                for cell in row["cells"]:
                    assign(cell)
    return "".join(parts) + "\n"


def _layout_tokens(unit):
    start = unit["document_char_span"]["start"]
    token_start = unit["document_token_span"]["start"]
    return [{"text": match.group(), "document_token_index": token_start + i,
             "document_char_span": {"start": start + match.start(), "end": start + match.end()}}
            for i, match in enumerate(re.finditer(r"\S+", unit["text"]))]


def _layout_regions(page, reading):
    regions = []
    for unit in page["text_regions"]:
        record = copy.deepcopy(unit)
        record["tokens"] = _layout_tokens(unit)
        regions.append(record)
    for table in page["tables"]:
        units = list(_native_units({"text_regions": [], "tables": [table]}))
        tokens = [token for unit in units for token in _layout_tokens(unit)]
        if not units:
            continue
        start = min(unit["document_char_span"]["start"] for unit in units)
        end = max(unit["document_char_span"]["end"] for unit in units)
        regions.append({"unit_id": table["table_id"] + "-layout-region", "region_kind": "table",
                        "page": page["page"], "table_id": table["table_id"],
                        "text": reading[start:end],
                        "document_char_span": {"start": start, "end": end},
                        "document_token_span": {"start": min((t["document_token_index"] for t in tokens), default=0),
                                                "end": max((t["document_token_index"] for t in tokens), default=-1) + 1},
                        "tokens": tokens})
    return regions


def build_mineru_bundle(baseline_input, middle_json, parser_identity, *, pdf_sha256):
    if not isinstance(baseline_input, dict) or not isinstance(middle_json, dict) or not isinstance(parser_identity, dict):
        raise ValueError("mineru_inputs_must_be_objects")
    if not isinstance(pdf_sha256, str) or not HEX.fullmatch(pdf_sha256):
        raise ValueError("pdf_sha256_invalid")
    if baseline_input.get("source_pdf_sha256") != pdf_sha256:
        raise ValueError("baseline_pdf_mismatch")
    if middle_json.get("schema") != "docvortex.middle" or middle_json.get("schema_version") != "2.0":
        raise ValueError("docvortex_middle_v2_required")
    producer = middle_json.get("metadata", {}).get("producer", {})
    if producer.get("name") != "mineru" or not producer.get("version"):
        raise ValueError("mineru_producer_missing")
    identity_keys = {"schema_version", "parser", "version", "source_pdf_sha256", "raw_canonical_sha256",
                     "raw_file_bytes_sha256", "model_manifest_sha256", "config_sha256", "tier", "ocr_mode",
                     "parser_identity_sha256"}
    if not identity_keys <= parser_identity.keys() or parser_identity.get("schema_version") != "paper-parser-identity/v1":
        raise ValueError("parser_identity_contract_invalid")
    if any(not isinstance(parser_identity[key], str) or not HEX.fullmatch(parser_identity[key])
           for key in ("source_pdf_sha256", "raw_canonical_sha256", "raw_file_bytes_sha256",
                       "model_manifest_sha256", "config_sha256", "parser_identity_sha256")):
        raise ValueError("parser_identity_hash_invalid")
    if parser_identity.get("parser") != "mineru" or parser_identity.get("version") != producer["version"]:
        raise ValueError("parser_identity_mismatch")
    if parser_identity["source_pdf_sha256"] != pdf_sha256:
        raise ValueError("parser_pdf_mismatch")
    if parser_identity["raw_canonical_sha256"] != digest(middle_json):
        raise ValueError("parser_raw_canonical_mismatch")
    if parser_identity["parser_identity_sha256"] != digest({k: v for k, v in parser_identity.items() if k != "parser_identity_sha256"}):
        raise ValueError("parser_identity_seal_mismatch")
    if middle_json.get("metadata", {}).get("file_suffix", "pdf").lower() != "pdf":
        raise ValueError("mineru_pdf_source_required")
    extension = middle_json.get("extensions", {})
    mineru = extension.get("mineru", {})
    geometry = extension.get("docvortex_layout", {})
    if not mineru.get("tier") or not mineru.get("parse_mode") or geometry.get("version") != 1:
        raise ValueError("mineru_extensions_missing")
    if parser_identity["tier"] != mineru["tier"] or parser_identity["ocr_mode"] != mineru["parse_mode"]:
        raise ValueError("parser_mode_mismatch")
    base_pages = baseline_input.get("pages", [])
    source_pages = middle_json.get("pages", [])
    geo_pages = geometry.get("pages", [])
    if not base_pages or not isinstance(source_pages, list) or not isinstance(geo_pages, list):
        raise ValueError("mineru_pages_missing")
    if not middle_json.get("is_full_document"):
        raise ValueError("mineru_partial_document_unsupported")
    if [p.get("page_idx") for p in source_pages] != list(range(len(base_pages))) or [p.get("page_idx") for p in geo_pages] != list(range(len(base_pages))):
        raise ValueError("mineru_page_binding_mismatch")
    pages = []
    audit_pages = []
    for pi, (base, source, geo) in enumerate(zip(base_pages, source_pages, geo_pages, strict=True)):
        page = pi + 1
        if base.get("page") != page:
            raise ValueError("baseline_page_binding_mismatch")
        width, height = geo.get("width_pt"), geo.get("height_pt")
        if type(width) not in {int, float} or type(height) not in {int, float} or width <= 0 or height <= 0:
            raise ValueError("mineru_page_geometry_invalid")
        native = list(_native_records(base))
        consumed = set()
        regions, tables, block_audit = [], [], []
        blocks = source.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("mineru_blocks_invalid")
        for bi, block in enumerate(blocks):
            if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                raise ValueError("mineru_block_invalid")
            kind = block["type"]
            bbox = _bbox(block.get("bbox"), width, height)
            audit = {"block_index": bi, "type": kind, "bbox": bbox, "raw": copy.deepcopy(block),
                     "disposition": None}
            block_audit.append(audit)
            if kind == "table":
                tables.append(_table(block, page, bi, bi, bbox, native, consumed))
                audit["disposition"] = "structured_table" if tables[-1]["structure_status"] == "parsed_html" else "unresolved_table"
            elif kind in TEXT_TYPES:
                text = _text(block)
                if text:
                    regions.append(_unit(f"v6-p{page:04d}-b{bi:04d}", page, kind, bi, bbox, text,
                                         "mineru_recognized", _alignment(text, bbox, native, consumed, "text_region"), source_block_index=bi))
                audit["disposition"] = "recognized_text" if text else "empty_text_block"
            elif kind in VISUAL_TYPES:
                text = _text(block)
                if text:
                    regions.append(_unit(f"v6-p{page:04d}-b{bi:04d}", page, kind, bi, bbox, text,
                                         "mineru_recognized", _alignment(text, bbox, native, consumed, "text_region"), source_block_index=bi))
                audit["disposition"] = "recognized_visual_text" if text else "nontext_visual"
            else:
                # Unknown blocks are retained verbatim and any textual content is citable.
                text = _text(block)
                if text:
                    regions.append(_unit(f"v6-p{page:04d}-b{bi:04d}", page, "unsupported_block", bi,
                                         bbox, text, "mineru_recognized", _alignment(text, bbox, native, consumed, "text_region"),
                                         structure_status="unresolved", source_block_index=bi, original_type=kind))
                audit["disposition"] = "unsupported_text_recovered" if text else "unsupported_nontext"
        recognized = list(_recognized_records(regions, tables))
        ledger = {"native_tokens": 0, "exactly_covered_native_tokens": 0, "recovered_native_tokens": 0,
                  "recognized_units": len(recognized), "recognized_only_units": 0,
                  "exact_native_matches": 0, "normalized_native_matches": 0,
                  "duplicate_views": 0, "native_units": []}
        for unit in regions:
            if unit["text_origin"] == "mineru_recognized":
                status = unit["native_alignment"]["status"]
                if status == "recognized_unaligned":
                    ledger["recognized_only_units"] += 1
                elif status == "exact_native_text":
                    ledger["exact_native_matches"] += 1
                elif status == "normalized_native_text":
                    ledger["normalized_native_matches"] += 1

        recovered = []
        def fragments_for(unit, bbox):
            fragments, covered, total = _uncovered_fragments(unit.get("text", ""), bbox, recognized)
            ledger["native_tokens"] += total
            ledger["exactly_covered_native_tokens"] += covered
            ledger["recovered_native_tokens"] += total - covered
            if covered and total - covered:
                ledger["duplicate_views"] += 1
            ledger["native_units"].append({"unit_id": unit["unit_id"], "total_tokens": total,
                                            "exactly_covered_tokens": covered,
                                            "recovered_tokens": total - covered})
            return fragments

        def extra_fragments(source_unit, texts, bbox, tag):
            for fi, text in enumerate(texts):
                recovered.append(_unit(f"v6-p{page:04d}-{tag}-f{fi:04d}", page,
                                       "recovered_source_text", len(blocks) + len(recovered),
                                       copy.deepcopy(bbox), text, "native_baseline_recovery",
                                       native_unit_id=source_unit["unit_id"], fragment_index=fi))

        for ni, unit in enumerate(base.get("text_regions", [])):
            extra_fragments(unit, fragments_for(unit, unit.get("bbox")), unit.get("bbox"), f"recovery-{ni:04d}")

        native_table_recoveries = 0
        for ti, native_table in enumerate(base.get("tables", [])):
            recovery = copy.deepcopy(native_table)
            table_bbox = recovery.get("bbox")
            recovery["table_id"] = f"v6-p{page:04d}-native-t{ti:04d}"
            recovery["source_origin"] = "native_baseline_recovery"
            recovery["source_table_id"] = native_table["table_id"]
            recovery["structure_status"] = "partial_native_recovery"
            caption = recovery["caption"]
            caption_parts = fragments_for(caption, caption.get("bbox") or table_bbox)
            caption["text"] = caption_parts[0] if caption_parts else ""
            extra_fragments(native_table["caption"], caption_parts[1:], table_bbox, f"native-t{ti:04d}-caption")
            headers = []
            for hi, header in enumerate(recovery.get("header_units", [])):
                parts = fragments_for(header, header.get("bbox") or table_bbox)
                if parts:
                    header["text"] = parts[0]
                    headers.append(header)
                    extra_fragments(native_table["header_units"][hi], parts[1:], table_bbox,
                                    f"native-t{ti:04d}-header-{hi:04d}")
            recovery["header_units"] = headers
            rows = []
            for ri, row in enumerate(recovery.get("rows", [])):
                row_parts = fragments_for(row, row.get("bbox") or table_bbox)
                row["text"] = row_parts[0] if row_parts else ""
                extra_fragments(native_table["rows"][ri], row_parts[1:], table_bbox,
                                f"native-t{ti:04d}-row-{ri:04d}")
                cells = []
                for ci, cell in enumerate(row.get("cells", [])):
                    parts = fragments_for(cell, cell.get("bbox") or table_bbox)
                    if parts:
                        cell["text"] = parts[0]
                        cells.append(cell)
                        extra_fragments(native_table["rows"][ri]["cells"][ci], parts[1:], table_bbox,
                                        f"native-t{ti:04d}-r{ri:04d}-c{ci:04d}")
                row["cells"] = cells
                if row["text"] or cells:
                    rows.append(row)
            recovery["rows"] = rows
            if not (caption["text"] or headers or rows):
                continue
            for ui, unit in enumerate(_native_units({"text_regions": [], "tables": [recovery]})):
                original_id = unit["unit_id"]
                unit["unit_id"] = f"{recovery['table_id']}-u{ui:04d}"
                unit.setdefault("page", page)
                unit.setdefault("region_kind", "table_caption" if ui == 0 else "table_source")
                unit.setdefault("reading_order", len(blocks) + ti)
                unit.setdefault("bbox", copy.deepcopy(table_bbox))
                unit["text_origin"] = "native_baseline_recovery"
                unit["native_unit_id"] = original_id
                unit["native_alignment"] = {"status": "not_applicable", "native_unit_id": None, "similarity": None}
            for ri, row in enumerate(recovery["rows"]):
                row["row_id"] = f"{recovery['table_id']}-r{ri:04d}"
            tables.append(recovery)
            native_table_recoveries += 1
        regions.extend(recovered)
        pages.append({"page": page, "layout_mode": "mineru_order_with_native_recovery", "text_regions": regions,
                      "tables": tables})
        audit_pages.append({"page": page, "page_idx": pi, "width_pt": width, "height_pt": height,
                            "blocks": block_audit, "recovered_native_region_count": len(recovered),
                            "native_table_recovery_count": native_table_recoveries,
                            "baseline_page": copy.deepcopy(base), "coverage_ledger": ledger})
    reading = _assign_spans(pages)
    for audit_page, paper_page in zip(audit_pages, pages, strict=True):
        # The input and layout must point to the same final reading offsets.
        # These copies must occur after _assign_spans mutates the input units.
        audit_page["regions"] = _layout_regions(paper_page, reading)
        audit_page["tables"] = [{**copy.deepcopy(table), "caption_unit": copy.deepcopy(table["caption"])}
                                for table in paper_page["tables"]]
    layout = {"schema_version": LAYOUT_VERSION, "paper_id": baseline_input["paper_id"],
              "source_pdf_sha256": pdf_sha256, "baseline_input_sha256": digest(baseline_input),
              "middle_json_sha256": digest(middle_json), "raw_middle_json": copy.deepcopy(middle_json),
              "parser_identity": copy.deepcopy(parser_identity),
              "mineru_output_identity": {"schema": middle_json["schema"], "schema_version": middle_json["schema_version"],
                                          "producer_version": producer["version"], "tier": mineru["tier"],
                                          "parse_mode": mineru["parse_mode"]},
              "page_count": len(pages), "pages": audit_pages, "reading_text_sha256": hashlib.sha256(reading.encode("utf-8")).hexdigest(),
              "span_conventions": {"page": "one_based_input_from_zero_based_docvortex", "bbox": "PDF points, top-left",
                                   "text": "recognized text distinguished from native baseline recovery"}}
    layout["preprocessing_sha256"] = digest(layout)
    paper_input = {"schema_version": INPUT_VERSION, "paper_id": layout["paper_id"],
                   "source_pdf_sha256": pdf_sha256, "layout_preprocessing_sha256": layout["preprocessing_sha256"],
                   "span_conventions": copy.deepcopy(layout["span_conventions"]), "pages": pages}
    paper_input["model_input_sha256"] = digest(paper_input)
    return layout, reading, paper_input


def rebuild_mineru_bundle(baseline_input, middle_json, parser_identity, *, pdf_sha256):
    return build_mineru_bundle(baseline_input, middle_json, parser_identity, pdf_sha256=pdf_sha256)


def validate_mineru_bundle(baseline_input, middle_json, parser_identity, layout, reading_text, paper_input, *, pdf_sha256):
    try:
        expected = build_mineru_bundle(baseline_input, middle_json, parser_identity, pdf_sha256=pdf_sha256)
    except (KeyError, TypeError, ValueError) as exc:
        return ["mineru_rebuild:" + str(exc)]
    errors = []
    for name, actual, derived in zip(("layout", "reading_text", "paper_input"),
                                     (layout, reading_text, paper_input), expected, strict=True):
        if actual != derived:
            errors.append("mineru_" + name + "_derivation_mismatch")
    return errors
