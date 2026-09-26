from high_fidelity_schema_study.four_category.dataset_batch import parse_batch


def test_dataset_batch_scale_reuse_resume_and_failure_isolation(tmp_path):
    (tmp_path / "shared.csv").write_text("id\n001\n002\n", encoding="utf-8")
    manifest = {"schema_version": "four-category-dataset-jobs/v1", "sources": [
        {"source_id": f"source-{i}", "path": "shared.csv" if i != 3 else "missing.csv"} for i in range(1000)]}
    output = tmp_path / "parse"
    first = parse_batch(manifest, source_root=tmp_path, output=output, sample_limit=1, max_jobs=5)
    assert first["status"] == "partial" and first["reused_this_invocation"] == 3
    second = parse_batch(manifest, source_root=tmp_path, output=output, sample_limit=1)
    assert second["status"] == "complete" and len(second["results"]) == 1000
    assert second["reused_this_invocation"] == 999 and second["model_calls"] == 0
    assert second["results"][3]["status"] == "failed"
    assert len({r["dataset_id"] for r in second["results"] if "dataset_id" in r}) == 1


def test_parser_cache_binds_source_taxonomy_and_sample_configuration(tmp_path):
    path = tmp_path / "x.csv"; path.write_text("x\n001\n002\n", encoding="utf-8")
    manifest = {"schema_version": "four-category-dataset-jobs/v1", "sources": [{"source_id": "x", "path": "x.csv"}]}
    output = tmp_path / "out"
    first = parse_batch(manifest, source_root=tmp_path, output=output, sample_limit=1)
    second = parse_batch(manifest, source_root=tmp_path, output=output, sample_limit=2)
    assert first["results"][0]["cache_key"] != second["results"][0]["cache_key"]
    path.write_text("x\n003\n", encoding="utf-8")
    third = parse_batch(manifest, source_root=tmp_path, output=output, sample_limit=1)
    assert first["results"][0]["cache_key"] != third["results"][0]["cache_key"]
