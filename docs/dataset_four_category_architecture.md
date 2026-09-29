# Dataset evidence in four categories

The paper and dataset paths share the versioned
[`four_category_taxonomy_v1.json`](../high_fidelity_schema_study/templates/four_category_taxonomy_v1.json),
but they establish facts independently. Paper classification helps a model find
potential evidence in the PDF. Dataset facts come from deterministic parsers
and explicitly bound metadata; they are never inserted into a paper model
request. A match between the two paths is a downstream evaluation decision.

| Category | Dataset evidence example | What it does not prove |
| --- | --- | --- |
| Structure | Header names, file schema, array shape, declared type | A sampled type is a universal constraint |
| Encoding | CSV serialization, HDF5 container, stored representation | The model's JSON response is the dataset's format |
| Value | Declared enum, codebook, explicit unit attribute | Two sampled values define the allowed domain |
| Syntax | Delimiter, quoting, field order, namespace, raw lexical token | A normalized value preserves its original spelling |

Facts may carry more than one category. Each fact links to evidence and records
`declared`, `observed`, `inferred`, or `unknown` support and its actual reading
scope. The source file's byte hash and parser version bind the representation.
The `parse-datasets` command can process a manifest and replay cached bundles;
individual failed or partial files retain their status. It makes no model calls.

## Version history and new scope

The published v0.1.0 source **already included** `dataset-four-category/v1`,
`parse-datasets`, and the four-category taxonomy. This source update extends
that foundation with V13 paper extraction, Scheduler V2, and an explicit
`scoped-dataset-catalog/v1` layer. The new catalog replays a selected dataset
evidence bundle, binds its original bytes, and inventories fields for an
explicit resource scope. It currently certifies complete header inventories
for CSV, TSV, and a selected XLSX sheet. It records unselected sheets as
excluded evidence. Other formats have four-category evidence bundles but do
not automatically receive a complete scoped catalog through this layer.

The automatic correspondence code uses this catalog as a bounded candidate
set. It cannot turn an observed dataset column into a paper-visible fact, infer
an undocumented alias, or establish `G_visible` on its own. Dataset coverage,
paper entailment, and independent semantic accuracy need separate evaluation.
The original dataset bundle interface remains available; the scoped catalog is
a new versioned derivative, not a rewrite of historical bundles.

Code: [`dataset.py`](../high_fidelity_schema_study/four_category/dataset.py),
[`dataset_batch.py`](../high_fidelity_schema_study/four_category/dataset_batch.py),
[`dataset_catalog.py`](../high_fidelity_schema_study/four_category/dataset_catalog.py),
[`fidelity_v2.py`](../high_fidelity_schema_study/four_category/fidelity_v2.py).
