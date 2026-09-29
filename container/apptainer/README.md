# Apptainer source recipes

**English** | [简体中文](README.zh-CN.md)

This directory holds recipes and wrappers for the public paper–dataset
workflow. `four-category.def` installs the offline parser and structured-output
dependencies. `mineru.def` packages the optional MinerU preprocessing path.
The two paths are selected independently. Paper extraction requires a frozen
paper input; dataset parsing reads its own files and does not call the paper
models.

The published Docker Hub `0.1.0-cuda13` image and the v0.1.0 source tag are an
earlier, reproducible release. This repository also carries a V13 revision-2
source snapshot. No newly qualified V13 SIF or OCI image is published by this
source update. Build a new image with a distinct identifier and verify its
actual byte hash, model compatibility and GPU capacity before using it for a
frozen run.

See the [English deployment guide](../../docs/schema_study_deployment_guide.md)
for the v0.1.0 image path, mounts and launch commands. The
[V13 source status](../../docs/v13_source_status_2026-09-28.md) records the
current qualification boundary.
