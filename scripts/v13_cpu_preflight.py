"""Compile and count every V13 page request without loading model weights.

Run with the frozen source release's parent on PYTHONPATH::

    PYTHONPATH=/path/to/source-parent python /job/v13_cpu_preflight.py \
      --job /job --profile-id qwen3_8_27b-bf16-v1

The job contains config.json, corpus.json and source_bundle/. The extraction
index here is explicitly disabled because classification has not run; production
must recount requests against its actual verified classification index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from high_fidelity_schema_study.four_category import backends, classification_v3, extraction_v13, structured_output
from high_fidelity_schema_study.four_category.classification_v2 import plan_groups as plan_classification
from high_fidelity_schema_study.four_category.common import canonical_bytes, digest, read_json
from high_fidelity_schema_study.four_category.disabled_navigation import make_disabled_index
from high_fidelity_schema_study.four_category.extraction_protocol import module_for_task
from high_fidelity_schema_study.four_category.paper import verify_paper
from high_fidelity_schema_study.four_category.workflow import production_request, tasks_for_config


def _token_ids(tokenizer, messages, profile):
    value = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=False, **profile["runtime"].get("chat_template_kwargs", {}))
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat_template_unexpected_batch")
        value = value[0]
    if not isinstance(value, list) or any(type(token) is not int for token in value):
        raise ValueError("chat_template_token_ids_invalid")
    return value


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(value)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def preflight(job: Path, profile_id: str, *, shard_index=0, shard_count=1) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    if torch.cuda.device_count() != 0:
        raise ValueError("cpu_preflight_gpu_visible")
    torch.set_num_threads(2)

    job = job.resolve()
    config = read_json(job / "config.json")
    corpus = read_json(job / "corpus.json")
    source = job / "source_bundle"
    profiles = [row for row in config["profiles"] if row["profile_id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError("profile_id_not_unique_or_missing")
    profile = profiles[0]
    if profile["backend"] != "transformers":
        raise ValueError("cpu_preflight_requires_local_transformers_profile")
    if profile_id not in config["roles"]["locals"] and profile_id != config["classification"]["profile_id"]:
        raise ValueError("profile_not_assigned_to_local_or_classifier_role")
    if config["extraction_input_protocol"] not in {extraction_v13.PROTOCOL, "extraction-page-regions/v13r2"}:
        raise ValueError("job_extraction_protocol_is_not_v13_supported")
    if config["classification"]["protocol"] not in {"classification-anchors/v3", "classification-anchors/v13r2"}:
        raise ValueError("job_classification_protocol_is_not_v13_supported")
    tasks = tasks_for_config(config)
    tokenizer = AutoTokenizer.from_pretrained(profile["model_id"], revision=profile["revision"],
                                               local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(profile["model_id"], local_files_only=True,
                                               trust_remote_code=False)
    generation = GenerationConfig.from_pretrained(profile["model_id"], local_files_only=True)
    vocabulary = model_config.get_text_config().vocab_size
    stub = SimpleNamespace(generation_config=generation,
                           get_output_embeddings=lambda: SimpleNamespace(
                               weight=SimpleNamespace(shape=(vocabulary, 1))))
    rows = []
    failures = []
    include_classifier = profile_id == config["classification"]["profile_id"]
    expected_papers = len(corpus["papers"])
    if expected_papers != 10:
        raise ValueError("expected_ten_papers_for_v13_full_run")
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid_preflight_shard")
    selected_papers = [row for i, row in enumerate(corpus["papers"]) if i % shard_count == shard_index]
    first_replicate = config["replicates"][0]
    extraction_parameters = {**config["local_parameters"], "seed": first_replicate["seed"]}

    def inspect(kind, paper_id, paper, group, messages, schema, parameters):
        identity = {"kind": kind, "paper_id": paper_id,
                    "group_id": group["group_id"], "group_index": group["index"]}
        try:
            tokens = _token_ids(tokenizer, messages, profile)
            request = backends.request_for(profile, messages, parameters, response_schema=schema)
            processor, applied = structured_output.logits_processor(request, tokenizer, stub)
            if processor is None or not isinstance(applied, dict):
                raise ValueError("grammar_processor_or_provenance_missing")
            allowance = parameters["max_output_tokens"]
            headroom = profile["context_window"] - len(tokens) - allowance
            row = {**identity, "status": "pass" if headroom >= 0 else "context_overflow",
                   "paper_input_canonical_sha256": digest(paper),
                   "request_sha256": digest(request), "messages_sha256": digest(messages),
                   "token_ids_sha256": hashlib.sha256(canonical_bytes(tokens)).hexdigest(),
                   "input_tokens": len(tokens), "output_allowance": allowance,
                   "context_window": profile["context_window"], "context_headroom": headroom,
                   "schema_bytes": len(canonical_bytes(schema)),
                   "schema_sha256": hashlib.sha256(canonical_bytes(schema)).hexdigest(),
                   "grammar_compiled": True, "structured_output_applied": applied,
                   "source_windows": len(group["window_ids"]) if kind == "extraction" else None,
                   "source_units": len(group["unit_ids"]) if kind == "classification" else None}
            rows.append(row)
            if headroom < 0:
                failures.append({**identity, "error": "context_overflow"})
        except Exception as exc:
            rows.append({**identity, "status": "error", "grammar_compiled": False,
                         "error": type(exc).__name__ + ":" + str(exc)})
            failures.append({**identity, "error": type(exc).__name__ + ":" + str(exc)})
        print(json.dumps({"progress": len(rows), "kind": kind, "paper_id": paper_id,
                          "group": group["index"], "status": rows[-1]["status"]}), flush=True)

    for paper_row in selected_papers:
        paper_id = paper_row["paper_id"]
        source_record = paper_row["source"]
        if not {"mineru_raw", "baseline_input", "parser_identity"} <= set(source_record["artifacts"]):
            raise ValueError("paper_is_not_frozen_mineru_source:" + paper_id)
        errors = verify_paper(source_record, source)
        if errors:
            raise ValueError("paper_source_invalid:" + paper_id + ":" + repr(errors[:3]))
        paper = read_json(source / source_record["artifacts"]["input"]["path"])
        index = make_disabled_index(paper, tasks["extraction"]["taxonomy"])
        if include_classifier:
            groups = plan_classification(paper, config["classification"]["grouping"])
            for group in groups:
                if config["classification"]["protocol"] == "classification-anchors/v13r2":
                    messages, schema = production_request(tasks["classification"], paper, classification_group=group)
                else:
                    messages = classification_v3.render_group(tasks["classification"], paper, group)
                    schema = tasks["classification"]["output_schema"]
                inspect("classification", paper_id, paper, group, messages,
                        schema, config["classification"]["parameters"])
        extraction_module = module_for_task(tasks["extraction"])
        groups = extraction_module.plan_groups(paper, config["extraction_grouping"])
        for group in groups:
            if config["extraction_input_protocol"] == "extraction-page-regions/v13r2":
                messages, schema = production_request(tasks["extraction"], paper, index=index, extraction_group=group)
            else:
                messages = extraction_v13.render_group(tasks["extraction"], paper, index, group)
                schema = extraction_v13.response_schema(tasks["extraction"], paper, index, group)
            inspect("extraction", paper_id, paper, group, messages, schema, extraction_parameters)

    report = {"schema_version": "v13-cpu-preflight/v1", "status": "pass" if not failures else "fail",
              "profile_id": profile_id, "profile_sha256": digest(profile),
              "config_sha256": digest(config), "corpus_sha256": digest(corpus),
              "paper_count": expected_papers,
              "shard_index": shard_index, "shard_count": shard_count,
              "checked_papers": [row["paper_id"] for row in selected_papers],
              "extraction_group_count": sum(r["kind"] == "extraction" for r in rows),
              "classification_group_count": sum(r["kind"] == "classification" for r in rows),
              "extraction_replicate_id_for_request_hash": first_replicate["replicate_id"],
              "extraction_seed_for_request_hash": first_replicate["seed"],
              "generation_calls": 0, "model_weights_loaded": False,
              "index_condition": "synthetic_disabled_navigation_for_cpu_preflight_only",
              "production_requirement": "Recount every request with the actual verified classification index before generation.",
              "rows": rows, "failures": failures}
    suffix = "" if shard_count == 1 else "-shard-" + str(shard_index)
    _atomic_json(job / "audit" / ("cpu-preflight-" + profile_id + suffix + ".json"), report)
    print(json.dumps({"status": report["status"], "profile_id": profile_id,
                      "extraction_groups": report["extraction_group_count"],
                      "classification_groups": report["classification_group_count"],
                      "failures": len(failures)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    result = preflight(args.job, args.profile_id, shard_index=args.shard_index, shard_count=args.shard_count)
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
