"""Paper-only candidate regions for axis tables and grouped feature inventories.

Region detection is navigation, not gold or a completeness certificate. A region
can contain several names, family labels, descriptors and damaged cell joins.
"""
from __future__ import annotations

import re
from .fidelity import table_roles

VERSION = "paper-feature-regions/v2"


def regions(paper: dict) -> list[dict]:
    result = table_roles(paper)
    tables = {t["table_id"]: t for p in paper["pages"] for t in p["tables"]}
    for role in result:
        table = tables[role["table_id"]]
        body = " ".join(c["text"] for r in table["rows"] for c in r["cells"])
        cue = bool(re.search(r"\bfeatures?\b", role["caption"], re.I))
        grouped = cue and bool(re.search(r"\b(?:main\s+features?|subfeatures?|\w+\s+features)\b", body, re.I))
        role["grouped_inventory_candidate"] = grouped
        role["candidate_inventory_exhaustive"] = False
        role["detection_version"] = VERSION
        if grouped:
            role["purpose"] = "grouped_feature_inventory_candidate"
            role["feature_cells"] = [
                {"label": c["text"], "unit_id": c["unit_id"], "row_unit_id": row["unit_id"],
                 "multiplicity": "zero_one_or_many", "segmentation": "model_must_review_source"}
                for row in table["rows"] for c in row["cells"] if c["text"].strip()]
        else:
            for cell in role["feature_cells"]:
                cell["multiplicity"] = "zero_one_or_many"
        role["feature_inventory_cue"] = cue or bool(role["feature_cells"])
    return result
