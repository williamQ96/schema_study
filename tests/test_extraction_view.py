from __future__ import annotations

import copy

import pytest

from high_fidelity_schema_study.four_category.classification_v2 import build_index_v2, make_task as classification_task, plan_groups
from high_fidelity_schema_study.four_category.common import seal
from high_fidelity_schema_study.four_category.extraction_view import make_task, make_view, render_task_v2, verify_view
from high_fidelity_schema_study.four_category.tasks import load_task

from .test_classification_v2 import paper, record, response


def paper_with_table():
    value = paper()
    page = value["pages"][0]
    page["tables"] = [{"table_id": "t1", "table_number": 7, "bbox": [1.25, 2, 3, 4.5],
                       "caption": {"unit_id": "cap", "page": 1, "text": "Table 7: terms"},
                       "columns": [{"column_index": 0, "header": "Term"}, {"column_index": 1, "header": "Meaning"}],
                       "header_units": [{"unit_id": "h0", "page": 1, "text": "Term", "bbox": [1, 2, 3, 4]}],
                       "rows": [{"row_id": "r1", "unit_id": "row", "text": "x = value",
                                 "cells": [{"unit_id": "c0", "text": "x", "column_index": 0},
                                           {"unit_id": "c1", "text": "value", "column_index": 1}]}]}]
    return value


def fixture_index(p):
    task = classification_task()
    groups = plan_groups(p)
    records = [record(p, task, group, response(p, group)) for group in groups]
    return build_index_v2(p, task, groups, records)


def test_complete_text_bbox_and_table_topology():
    p = paper_with_table()
    index = fixture_index(p)
    view = make_view(p, index)
    columns = view["unit_columns"]
    units = [dict(zip(columns, row)) for row in view["units"]]
    assert [u["unit_id"] for u in units] == [e["unit_id"] for e in index["entries"]]
    assert units[0]["text"] == "é😀 alpha"
    assert units[0]["bbox"] == [0, 0, 1, 1]
    assert next(u for u in units if u["unit_id"] == "h0")["bbox"] == [1, 2, 3, 4]
    assert next(u for u in units if u["unit_id"] == "c1")["text"] == "value"
    page = dict(zip(view["page_columns"], view["pages"][0]))
    table = dict(zip(view["table_columns"], page["tables"][0]))
    assert table["bbox"] == [1.25, 2, 3, 4.5]
    assert table["caption_unit_id"] == "cap" and table["header_unit_ids"] == ["h0"]
    assert table["columns"] == [[0, "Term"], [1, "Meaning"]]
    assert table["rows"] == [["row", ["c0", "c1"]]]
    assert len(view["index_entries"]) == len(view["units"])
    assert set(view["index_columns"]) == {"state", "categories"}
    assert verify_view(view, p, index) == []


def test_resealed_projection_tamper_is_rejected():
    p = paper_with_table(); index = fixture_index(p)
    view = make_view(p, index)
    changed = copy.deepcopy(view); changed["units"][0][3] += " invented"
    changed = seal(changed, "view_sha256")
    assert "compact_view_derivation_mismatch" in verify_view(changed, p, index)
    changed = copy.deepcopy(view); changed["pages"][0][3][0][6][0][1].reverse()
    changed = seal(changed, "view_sha256")
    assert "compact_view_derivation_mismatch" in verify_view(changed, p, index)


def test_deterministic_prompt_and_v1_task_unchanged():
    p = paper_with_table(); index = fixture_index(p)
    task = make_task()
    assert task["schema_version"] == "four-category-task/v3"
    assert task["output_schema"] == load_task("extraction")["output_schema"]
    one = render_task_v2(task, p, index)
    assert one == render_task_v2(task, p, index)
    assert "é😀 alpha" in one[1]["content"] and "classification_quotes" in one[1]["content"]
    assert '"document_char_span":{' not in one[1]["content"]
    old = load_task("extraction")
    assert old["schema_version"] == "four-category-task/v1"
    with pytest.raises(ValueError):
        render_task_v2(old, p, index)
