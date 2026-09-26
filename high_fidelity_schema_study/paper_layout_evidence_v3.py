"""Lossless v3 repair from frozen PDF word observations; v2 remains immutable."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

from .paper_layout_evidence import (
    assign_document_spans, canonical_json_bytes, compact_model_input, group_lines,
    iter_layout_units, sha256_bytes, text_region, validate_layout_bundle,
)
from .research_repair_integrity import file_hash, inventory, read_json, write_new


def word_coverage_audit(layout: dict) -> dict:
    pages = []
    for page in layout["pages"]:
        actual = Counter(token["source_word_index"] for region in page["regions"] for token in region["tokens"])
        source = {word["word_index"]: word for word in page["source_words"]}
        missing = [word for key, word in source.items() if not actual[key]]
        if missing or any(n != 1 for n in actual.values()) or set(actual) - set(source):
            pages.append({"page": page["page"], "missing_words": missing,
                          "duplicate_indexes": [k for k, n in actual.items() if n > 1],
                          "extra_indexes": sorted(set(actual) - set(source)),
                          "context": [{"word_index": w["word_index"], "text": " ".join(source[k]["text"] for k in sorted(source) if abs(k - w["word_index"]) <= 5)} for w in missing],
                          "impact_type": "source_word_not_rendered_requires_human_semantic_review"})
    return {"paper_id": layout["paper_id"], "status": "fail" if pages else "pass", "missing_word_count": sum(len(p["missing_words"]) for p in pages), "pages": pages}


def upgrade_bundle(old: dict) -> tuple[dict, str, dict, list[dict]]:
    """Keep rendered v2 content, recover unused source words, expose headers.

    Source words were extracted from the hash-bound PDF, not from model output.
    New IDs are explicitly namespaced even where their contents did not change.
    """
    layout = copy.deepcopy(old)
    layout["schema_version"] = "paper-evidence-layout/v3"
    layout["preprocessing_profile"] = "frozen-pdf-words-lossless-table-header-v3"
    layout["derived_from_v2_preprocessing_sha256"] = old["preprocessing_sha256"]
    layout["span_conventions"]["character_offsets"] = "Unicode code-point offsets into reading_text.txt; start inclusive, end exclusive"
    mapping = []
    for page in layout["pages"]:
        used = {t["source_word_index"] for region in page["regions"] for t in region["tokens"]}
        missing = [w for w in page["source_words"] if w["word_index"] not in used]
        for region in page["regions"]:
            region["source_word_indexes"] = [t["source_word_index"] for t in region["tokens"]]
        # Preserve adjacent lines as citable paragraphs, without accidentally
        # joining text from the opposite PDF column on the same visual line.
        recovered_groups = []
        midpoint = page["width_points"] / 2
        for side in (False, True):
            lines = group_lines([w for w in missing if ((w["x0"] + w["x1"]) / 2 >= midpoint) == side])
            current, bottom = [], None
            for line in lines:
                top = min(w["top"] for w in line)
                if current and top - bottom > 18:
                    recovered_groups.append(current)
                    current = []
                current.extend(line)
                bottom = max(w["bottom"] for w in line)
            if current:
                recovered_groups.append(current)
        for index, words in enumerate(recovered_groups, 1):
            page["regions"].append(text_region(f"p{page['page']:04d}-recovered-{index:04d}", page["page"], "recovered_source_text", None, words))
        for table in page["tables"]:
            table["header_units"] = []
            for column in table["columns"]:
                table["header_units"].append({
                    "unit_id": f"{table['table_id']}-header-c{column['column_index'] + 1:02d}",
                    "text": column["header"], "region_char_span": column["region_char_span"],
                    "column_index": column["column_index"], "bbox": column["bbox"],
                    "source_word_indexes": column["source_word_indexes"],
                })
            region = next(r for r in page["regions"] if r.get("table_id") == table["table_id"])
            table["source_word_indexes"] = region["source_word_indexes"]
        page["regions"].sort(key=lambda r: (r["bbox"][1], r["bbox"][0], r["unit_id"]))
        for order, region in enumerate(page["regions"], 1):
            region["reading_order"] = order
        for unit in iter_layout_units({"pages": [page]}):
            unit.setdefault("page", page["page"])
        # Rename identifiers, including row_id and region/table references.
        def rename(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"unit_id", "region_id", "table_id", "row_id"} and isinstance(item, str):
                        value[key] = "v3-" + item
                    else:
                        rename(item)
            elif isinstance(value, list):
                for item in value:
                    rename(item)
        rename(page)
    reading = assign_document_spans(layout["pages"])
    layout["layout_token_count"] = sum(len(r["tokens"]) for p in layout["pages"] for r in p["regions"])
    layout["region_count"] = sum(len(p["regions"]) for p in layout["pages"])
    layout["reading_text_sha256"] = sha256_bytes(reading.encode())
    layout.pop("preprocessing_sha256", None)
    layout["preprocessing_sha256"] = sha256_bytes(canonical_json_bytes(layout))
    model_input = compact_model_input(layout)
    old_units = {u["unit_id"]: u for u in iter_layout_units(old)}
    for unit in iter_layout_units(layout):
        old_id = unit["unit_id"].removeprefix("v3-")
        mapping.append({"v3_unit_id": unit["unit_id"], "v2_unit_id": old_id if old_id in old_units else None,
                        "text_identical": old_id in old_units and old_units[old_id]["text"] == unit["text"],
                        "relation": "retained_content_new_spans" if old_id in old_units else "newly_citable"})
    errors = validate_layout_bundle(layout, reading, model_input)
    if errors:
        raise ValueError(errors)
    return layout, reading, model_input, mapping


def build_corpus(phase_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    corpus = read_json(phase_root / "frozen_corpus_manifest_v1.json")
    output.mkdir(parents=True)
    audits, sources = [], []
    for paper in corpus["papers"]:
        lane = phase_root / Path(paper["preprocessing_artifact"]).parent
        old = read_json(lane / "evidence_layout_v2.json")
        if file_hash(lane / "paper.pdf") != paper["paper_sha256"] or old["source_pdf_sha256"] != paper["paper_sha256"]:
            raise ValueError(f"PDF identity mismatch: {paper['paper_id']}")
        sources.extend(inventory(lane / name, phase_root) for name in ("paper.pdf", "evidence_layout_v2.json", "reading_text_v2.txt", "evidence_input_v2.json"))
        audit = word_coverage_audit(old)
        audit["validation_errors"] = validate_layout_bundle(old, (lane / "reading_text_v2.txt").read_text(encoding="utf-8"), read_json(lane / "evidence_input_v2.json"))
        audits.append(audit)
        layout, text, model_input, mapping = upgrade_bundle(old)
        dest = output / paper["paper_id"]
        write_new(dest / "layout.json", layout)
        with (dest / "reading_text.txt").open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
        write_new(dest / "input.json", model_input)
        write_new(dest / "v2_v3_unit_mapping.json", mapping)
    write_new(output / "legacy_coverage_audit.json", {"schema_version": "legacy-layout-coverage-audit/v1", "missing_word_count": sum(a["missing_word_count"] for a in audits), "papers": audits})
    manifest = {"schema_version": "paper-layout-corpus/v3", "purpose": "annotation_and_future_ablation_only_not_used_by_frozen_inference", "source_root": "phase_root", "sources": sources,
                "files": [inventory(p, output) for p in sorted(output.rglob('*')) if p.is_file()]}
    write_new(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_corpus(args.phase_root, args.output)
