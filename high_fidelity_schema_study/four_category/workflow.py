"""Content-bound, resumable execution without corpus-size or model-name constants.

Live calls are opt-in, and real token-count qualification must precede them.
Offline transports exercise exactly the same request/admission/cache path.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .backends import invoke, profile_hash, validate_profile, backend_record_errors
from .common import ROOT, digest, file_digest, now, read_json, seal, seal_errors, strict_json, write_new, contained
from .paper import build_index, classification_errors, extraction_errors, verify_paper, verify_index
from .tasks import load_task, render_task, task_errors


def experiment_errors(config: dict, *, execution: bool = False) -> list[str]:
    if not isinstance(config, dict):
        return ["experiment_must_be_object"]
    errors = []
    if config.get("schema_version") != "four-category-experiment/v1":
        errors.append("experiment_version_invalid")
    profiles = config.get("profiles", [])
    if not isinstance(profiles, list) or any(not isinstance(p, dict) for p in profiles):
        return errors + ["profiles_must_be_objects"]
    keys = [p.get("profile_id") for p in profiles]
    if len(keys) != len(set(keys)):
        errors.append("duplicate_profile_id")
    roles = config.get("roles", {})
    if not isinstance(roles, dict):
        return errors + ["roles_must_be_object"]
    local = roles.get("locals", [])
    if not isinstance(local, list) or any(not isinstance(x, str) for x in local) or len(local) != 3 or len(set(local)) != 3:
        errors.append("three_distinct_local_roles_required")
        local = []
    for name in list(local) + [roles.get("soft_reference")]:
        if name not in keys:
            errors.append("unbound_model_role:" + str(name))
    if roles.get("soft_reference") in local:
        errors.append("soft_reference_must_have_separate_profile")
    classification = config.get("classification")
    if not isinstance(classification, dict) or not isinstance(classification.get("parameters"), dict):
        return errors + ["classification_parameters_required"]
    classifier = classification.get("profile_id")
    for key in ("local_parameters", "reference_parameters", "execution", "dataset_parser"):
        if not isinstance(config.get(key), dict):
            errors.append(key + ":object_required")
    if errors:
        return errors
    if execution and classifier not in keys:
        errors.append("classifier_binding_required")
    elif classifier is not None and classifier not in keys:
        errors.append("classifier_profile_missing")
    if execution and classifier in keys:
        profile = next(p for p in profiles if p["profile_id"] == classifier)
        if classification.get("profile_sha256") != profile_hash(profile):
            errors.append("classifier_profile_pin_required_or_mismatched")
    replicates = config.get("replicates", [])
    if not isinstance(replicates, list) or not replicates or any(not isinstance(r, dict) or isinstance(r.get("replicate_id"), bool) or not isinstance(r.get("replicate_id"), int) or isinstance(r.get("seed"), bool) or not isinstance(r.get("seed"), int) for r in replicates):
        errors.append("replicates_require_ids_and_integer_seeds")
    elif len({r["replicate_id"] for r in replicates}) != len(replicates):
        errors.append("duplicate_replicate_id")
    maximum = config.get("execution", {}).get("max_attempts", 2)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1 or maximum > 10:
        errors.append("max_attempts_must_be_1_to_10")
    if config["execution"].get("retry_statuses") != ["transport_error"]:
        errors.append("retry_policy_must_be_transport_error_only")
    if config["execution"].get("concurrency_per_backend", 1) != 1:
        errors.append("this_runner_supports_serial_execution_only")
    limit = config["dataset_parser"].get("sample_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        errors.append("dataset_sample_limit_must_be_positive_integer")
    for profile in profiles:
        if profile.get("backend") != "mock":
            expected = ("local" if profile["profile_id"] in local else
                        "remote" if profile["profile_id"] == roles.get("soft_reference") else None)
            if expected is not None and profile.get("deployment") != expected:
                errors.append("model_role_deployment_mismatch:" + str(profile.get("profile_id")))
    if execution:
        for profile in profiles:
            errors.extend(f"{profile.get('profile_id')}:{e}" for e in validate_profile(profile, for_execution=True))
        if any(p.get("backend") != "mock" for p in profiles) and (config.get("status") != "frozen" or not config.get("inference_enabled")):
            errors.append("live_experiment_not_enabled_and_frozen")
    return errors


def tasks_for_config(config: dict) -> dict:
    tax = read_json(contained(ROOT, config["taxonomy_path"]))
    if not isinstance(tax.get("categories"), dict) or set(tax["categories"]) != {"structure", "encoding", "value", "syntax"}:
        raise ValueError("taxonomy must preserve the four category identifiers")
    return {kind: load_task(kind, taxonomy_value=tax) for kind in ("classification", "extraction")}


def corpus_errors(corpus: dict) -> list[str]:
    if not isinstance(corpus, dict):
        return ["corpus_must_be_object"]
    errors = []
    if corpus.get("schema_version") != "four-category-corpus/v1":
        errors.append("corpus_version_invalid")
    for key, id_key in (("papers", "paper_id"), ("datasets", "dataset_id"), ("matches", "match_id")):
        records = corpus.get(key)
        if not isinstance(records, list) or any(not isinstance(x, dict) or not isinstance(x.get(id_key), str) or not x[id_key] for x in records):
            errors.append(key + ":invalid_records")
        elif len({x[id_key] for x in records}) != len(records):
            errors.append(key + ":duplicate_id")
    if errors:
        return errors
    pids = {p["paper_id"] for p in corpus["papers"]}
    dids = {d["dataset_id"] for d in corpus["datasets"]}
    pairs = set()
    for row in corpus["matches"]:
        pair = (row.get("paper_id"), row.get("dataset_id"))
        if pair[0] not in pids or pair[1] not in dids:
            errors.append("unresolved_match_endpoint:" + row["match_id"])
        if pair in pairs:
            errors.append("duplicate_paper_dataset_pair:" + row["match_id"])
        pairs.add(pair)
        if not isinstance(row.get("linkage"), dict) or row["linkage"].get("status") not in {"candidate", "verified"} or not row["linkage"].get("evidence"):
            errors.append("match_linkage_evidence_required:" + row["match_id"])
    for paper in corpus["papers"]:
        source = paper.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("source_identity"), dict) or source["source_identity"].get("document_id") != paper["paper_id"] or not source.get("source_sha256"):
            errors.append("paper_id_source_mismatch:" + paper["paper_id"])
    return errors


def plan_jobs(config: dict, corpus: dict, *, tasks: dict | None = None) -> dict:
    errors = experiment_errors(config) + corpus_errors(corpus)
    if errors:
        raise ValueError(errors)
    tasks = tasks or tasks_for_config(config)
    expected_tasks = config.get("task_hashes") or {k: t["task_sha256"] for k, t in tasks_for_config(config).items()}
    if set(tasks) != {"classification", "extraction"} or {k: t["task_sha256"] for k, t in tasks.items()} != expected_tasks:
        raise ValueError("task_hashes_do_not_match_experiment")
    for kind, task in tasks.items():
        if task_errors(task):
            raise ValueError(task_errors(task))
        if task["kind"] != kind or task["output_schema"] != load_task(kind)["output_schema"]:
            raise ValueError("unsupported_task_output_contract_requires_versioned_validator")
    if tasks["classification"]["taxonomy"] != tasks["extraction"]["taxonomy"]:
        raise ValueError("task taxonomy mismatch")
    profiles = {p["profile_id"]: p for p in config["profiles"]}
    adapter_identity = {"version": "four-category-backends/v1", "file_bytes_sha256": file_digest(Path(invoke.__code__.co_filename))}
    jobs = []
    for paper in corpus["papers"]:
        def job(kind, profile_id, parameters, replicate_id, dependencies):
            profile = profiles.get(profile_id)
            value = {"kind": kind, "paper_id": paper["paper_id"], "source_sha256": paper["source"]["source_sha256"],
                     "backend_adapter": adapter_identity,
                     "profile_id": profile_id, "profile_sha256": profile_hash(profile) if profile else None,
                     "parameters": copy.deepcopy(parameters), "replicate_id": replicate_id,
                     "task_sha256": tasks["classification" if kind == "classification" else "extraction"]["task_sha256"],
                     "dependencies": dependencies}
            value["job_id"] = digest(value)
            jobs.append(value)
            return value["job_id"]
        cid = job("classification", config["classification"].get("profile_id"), config["classification"]["parameters"], None, [])
        for profile_id in config["roles"]["locals"]:
            for rep in config["replicates"]:
                job("local_extraction", profile_id, {**config["local_parameters"], "seed": rep["seed"]}, rep["replicate_id"], [cid])
        job("soft_reference", config["roles"]["soft_reference"], config["reference_parameters"], 1, [cid])
    return seal({"schema_version": "four-category-job-plan/v1", "experiment_sha256": digest(config),
                 "task_hashes": expected_tasks,
                 "corpus_sha256": digest(corpus), "jobs": jobs, "paper_count": len(corpus["papers"]),
                 "matchset_count": len(corpus["matches"]), "dataset_count": len(corpus["datasets"]),
                 "execution_ready": not experiment_errors(config, execution=True),
                 "readiness_errors": experiment_errors(config, execution=True)}, "plan_sha256")


def context_preflight(profile: dict, messages: list[dict], parameters: dict, counter: Callable | None = None,
                      *, allow_live: bool = False, token_transport=None) -> dict:
    """Unknown provider tokenization blocks execution, never silently truncates.

The counter hook accepts (profile, messages) and returns an auditable count record.
It can be supplied by a provider-specific counter or a previously captured count
bound to this exact request. Synthetic mock counts are never called model tokens.
"""
    count = None
    if profile.get("runtime", {}).get("token_counter") is not None:
        from .token_counting import count_tokens
        observation = count_tokens(profile, messages, allow_live=allow_live, transport=token_transport)
        count = {"input_tokens": observation["input_tokens"], "counter_id": observation["counter_id"],
                 "exact": observation["exact"], "token_count_record": observation}
    elif counter is not None:
        count = counter(profile, messages)
    elif profile["backend"] == "mock":
        count = {"input_tokens": sum(len(m["content"]) for m in messages), "counter_id": "mock-codepoints-not-model-tokens", "exact": True}
    elif profile["backend"] == "transformers":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            profile["model_id"], revision=profile["revision"], local_files_only=True,
            trust_remote_code=profile["runtime"].get("settings", {}).get("trust_remote_code", False),
        )
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **profile["runtime"].get("chat_template_kwargs", {}))
        count = {"input_tokens": len(tokens), "counter_id": "qualified-local-chat-tokenizer", "exact": True}
    else:
        count = profile.get("runtime", {}).get("token_count_observations", {}).get(digest(messages))
    if not isinstance(count, dict) or not count.get("exact") or not count.get("counter_id") or isinstance(count.get("input_tokens"), bool) or not isinstance(count.get("input_tokens"), int) or count["input_tokens"] < 0:
        return {"status": "blocked", "reason": "exact_request_token_count_unavailable", "request_sha256": digest(messages),
                **({"token_count_record": count["token_count_record"]} if isinstance(count, dict) and "token_count_record" in count else {})}
    cap = parameters.get("max_output_tokens")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        return {"status": "blocked", "reason": "invalid_output_budget"}
    fits = count["input_tokens"] + cap <= profile["context_window"]
    return {"status": "pass" if fits else "blocked", "reason": None if fits else "context_overflow", **count,
            "request_sha256": digest(messages), "profile_sha256": profile_hash(profile),
            "output_allowance": cap, "context_window": profile["context_window"]}


def replay_run(record: dict, task: dict, paper_input: dict, *, index: dict | None = None,
               layout: dict | None = None, reading_text: str | None = None) -> list[str]:
    errors = seal_errors(record, "record_sha256")
    try:
        messages = render_task(task, paper_input, index)
        if record["task"] != task or record["messages"] != messages or record["request_sha256"] != digest(messages):
            errors.append("request_derivation_mismatch")
        if record["profile_sha256"] != profile_hash(record["profile"]):
            errors.append("profile_identity_mismatch")
        if record["job"]["profile_sha256"] != record["profile_sha256"] or record["job"]["task_sha256"] != task["task_sha256"]:
            errors.append("job_contract_identity_mismatch")
        if record["paper_input_canonical_sha256"] != digest(paper_input) or record["index_sha256"] != (index["index_sha256"] if index else None):
            errors.append("run_input_identity_mismatch")
        expected_id = digest({"job_id": record["job"]["job_id"], "index_sha256": record["index_sha256"], "attempt": record["attempt"]})
        if record["run_id"] != expected_id:
            errors.append("run_id_derivation_mismatch")
        result = record["backend_result"]
        observation = record["context_preflight"].get("token_count_record")
        if observation is not None:
            from .token_counting import counter_record_errors
            errors.extend(counter_record_errors(observation, record["profile"], messages))
            budget_count = record["context_preflight"]
            if budget_count.get("status") == "pass" and (budget_count.get("input_tokens") != observation.get("input_tokens")
                    or budget_count.get("counter_id") != observation.get("counter_id")
                    or budget_count.get("exact") != observation.get("exact")):
                errors.append("token_counter_budget_mismatch")
        elif record["profile"].get("runtime", {}).get("token_counter") is not None:
            errors.append("token_counter_evidence_missing")
        if record["context_preflight"]["status"] == "pass":
            budget = record["context_preflight"]
            if (budget["request_sha256"] != digest(messages) or budget["profile_sha256"] != record["profile_sha256"]
                    or budget["context_window"] != record["profile"]["context_window"]
                    or budget["output_allowance"] != record["job"]["parameters"]["max_output_tokens"]
                    or not budget.get("exact") or not budget.get("counter_id")
                    or not isinstance(budget["input_tokens"], int) or budget["input_tokens"] < 0
                    or budget["input_tokens"] + budget["output_allowance"] > budget["context_window"]):
                errors.append("context_budget_binding_mismatch")
            errors.extend(backend_record_errors(result, record["profile"], messages, record["job"]["parameters"]))
        elif result.get("dispatch_started") or result.get("request") is not None or result["status"] != "invalid_request":
            errors.append("blocked_context_has_dispatch")
        # Failed transport attempts are authentic missing generations, not empty schemas.
        if result["status"] == "success":
            try:
                payload = strict_json(result["raw_text"])
                checks = classification_errors(payload, paper_input) if task["kind"] == "classification" else extraction_errors(payload, paper_input, layout=layout, reading_text=reading_text)
            except (ValueError, TypeError, KeyError) as exc:
                payload, checks = None, ["response_parse:" + str(exc)]
            if record["parsed_response"] != payload or record["validation_errors"] != checks:
                errors.append("raw_response_replay_mismatch")
            expected = "success" if not checks else "contract_invalid"
        else:
            expected = result["status"]
            if record["parsed_response"] is not None:
                errors.append("failed_generation_has_payload")
        if record["status"] != expected:
            errors.append("run_status_replay_mismatch")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append("run_replay:" + str(exc))
    return errors


def _attempt(job, profile, task, paper_input, index, layout, reading_text, number, *, allow_live, transport, counter,
             token_transport=None):
    messages = render_task(task, paper_input, index)
    try:
        budget = context_preflight(profile, messages, job["parameters"], counter, allow_live=allow_live,
                                   token_transport=token_transport)
    except Exception as exc:
        budget = {"status": "blocked", "reason": "token_counter_error:" + str(exc)}
    if budget["status"] != "pass":
        result = {"status": "invalid_request", "raw_text": None, "raw_response": None, "request": None,
                  "errors": [budget["reason"]], "requested_model": profile.get("model_id"), "returned_model": None,
                  "requested_parameters": job["parameters"], "dispatch_started": False, "live_request_started": False,
                  "execution_mode": "mock" if profile["backend"] == "mock" else "injected_transport" if transport else "live"}
    else:
        result = invoke(profile, messages, job["parameters"], allow_live=allow_live, transport=transport)
    payload, checks = None, []
    if result["status"] == "success":
        try:
            payload = strict_json(result["raw_text"])
            checks = classification_errors(payload, paper_input) if task["kind"] == "classification" else extraction_errors(payload, paper_input, layout=layout, reading_text=reading_text)
        except (ValueError, TypeError, KeyError) as exc:
            checks = ["response_parse:" + str(exc)]
    condition = {"job_id": job["job_id"], "index_sha256": index["index_sha256"] if index else None}
    status = ("contract_invalid" if checks else "success") if result["status"] == "success" else result["status"]
    return seal({"schema_version": "four-category-run/v1", "run_id": digest({**condition, "attempt": number}),
                 "created_at_utc": now(), "attempt": number, "job": job, "profile": profile,
                 "profile_sha256": profile_hash(profile), "task": task, "paper_input_canonical_sha256": digest(paper_input),
                 "index_sha256": condition["index_sha256"], "messages": messages, "request_sha256": digest(messages),
                 "context_preflight": budget, "backend_result": result, "status": status,
                 "parsed_response": payload, "validation_errors": checks,
                 "semantic_correctness": "not_established", "mode": "offline_mock" if profile["backend"] == "mock" else "model_generation"}, "record_sha256")


def run_batch(config: dict, corpus: dict, *, source_root: Path, output: Path, tasks: dict | None = None,
              transports: dict | None = None, token_counter: Callable | None = None,
              allow_live: bool = False, max_jobs: int | None = None, token_transports: dict | None = None) -> dict:
    errors = experiment_errors(config, execution=True) + corpus_errors(corpus)
    if errors:
        return {"status": "blocked", "errors": errors, "model_calls": 0}
    if not allow_live and any(p["backend"] != "mock" for p in config["profiles"]):
        return {"status": "blocked", "errors": ["live_execution_not_requested"], "model_calls": 0}
    try:
        tasks = tasks or tasks_for_config(config)
        plan = plan_jobs(config, corpus, tasks=tasks)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "blocked", "errors": ["plan_preflight:" + str(exc)], "model_calls": 0}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for task in tasks.values():
        task_path = output / "tasks" / (task["task_sha256"] + ".json")
        if not task_path.exists():
            write_new(task_path, task)
        elif read_json(task_path) != task:
            raise ValueError("cached task identity collision")
    plan_path = output / "plans" / (plan["plan_sha256"] + ".json")
    if not plan_path.exists():
        write_new(plan_path, plan)
    elif read_json(plan_path) != plan:
        raise ValueError("cached plan identity collision")
    profiles = {p["profile_id"]: p for p in config["profiles"]}
    papers = {p["paper_id"]: p for p in corpus["papers"]}
    contexts, indexes, completed, slots = {}, {}, {}, []
    calls = 0
    max_attempts = config.get("execution", {}).get("max_attempts", 2)
    for job in plan["jobs"]:
        if max_jobs is not None and len(slots) >= max_jobs:
            break
        paper_id = job["paper_id"]
        if paper_id not in contexts:
            source = papers[paper_id]["source"]
            issues = verify_paper(source, source_root)
            if issues:
                contexts[paper_id] = {"errors": issues}
            else:
                paths = {k: contained(source_root, v["path"]) for k, v in source["artifacts"].items()}
                contexts[paper_id] = {"errors": [], "input": read_json(paths["input"]), "layout": read_json(paths["layout"]),
                                      "reading": paths["reading_text"].read_text(encoding="utf-8")}
        context = contexts[paper_id]
        if context["errors"] or any(dep not in completed for dep in job["dependencies"]):
            slots.append({"job_id": job["job_id"], "paper_id": paper_id, "status": "blocked_dependency", "errors": context["errors"] or ["classification_unavailable"], "record_paths": []})
            continue
        index = indexes.get(paper_id) if job["kind"] != "classification" else None
        task = tasks["classification" if job["kind"] == "classification" else "extraction"]
        cache_key = digest({"job_id": job["job_id"], "index_sha256": index["index_sha256"] if index else None})
        run_dir = output / "runs" / cache_key
        run_dir.mkdir(parents=True, exist_ok=True)
        saved = sorted(run_dir.glob("attempt-*.json"))
        try:
            records = [read_json(p) for p in saved]
        except (OSError, ValueError) as exc:
            slots.append({"job_id": job["job_id"], "paper_id": paper_id, "status": "cache_invalid", "errors": [str(exc)],
                          "record_paths": [p.relative_to(output).as_posix() for p in saved]})
            continue
        replay_errors = []
        if ([r.get("attempt") for r in records if isinstance(r, dict)] != list(range(1, len(records) + 1))
                or len(records) > max_attempts or any(r.get("status") != "transport_error" for r in records[:-1] if isinstance(r, dict))):
            replay_errors.append("cache_attempt_sequence_or_retry_invalid")
        for record in records:
            replay_errors.extend(replay_run(record, task, context["input"], index=index, layout=context["layout"], reading_text=context["reading"]))
            if record.get("job") != job:
                replay_errors.append("cached_job_mismatch")
        if replay_errors:
            slots.append({"job_id": job["job_id"], "paper_id": paper_id, "status": "cache_invalid", "errors": replay_errors,
                          "record_paths": [p.relative_to(output).as_posix() for p in saved]})
            continue
        while len(records) < max_attempts and (not records or records[-1]["status"] == "transport_error"):
            record = _attempt(job, profiles[job["profile_id"]], task, context["input"], index, context["layout"], context["reading"],
                              len(records) + 1, allow_live=allow_live, transport=(transports or {}).get(job["profile_id"]),
                              counter=token_counter, token_transport=(token_transports or {}).get(job["profile_id"]))
            calls += int(record["backend_result"].get("dispatch_started", False))
            path = run_dir / f"attempt-{len(records) + 1:04d}.json"
            write_new(path, record)
            saved.append(path)
            records.append(record)
        selected = records[-1]
        if selected["status"] == "success":
            completed[job["job_id"]] = selected
            if job["kind"] == "classification":
                producer = {"run_id": selected["run_id"], "profile_sha256": selected["profile_sha256"],
                            "task_sha256": task["task_sha256"], "request_sha256": selected["request_sha256"]}
                indexes[paper_id] = build_index(selected["backend_result"]["raw_text"], context["input"], producer, taxonomy_value=task["taxonomy"])
                ipath = output / "indexes" / (indexes[paper_id]["index_sha256"] + ".json")
                if not ipath.exists():
                    write_new(ipath, indexes[paper_id])
                elif read_json(ipath) != indexes[paper_id]:
                    raise ValueError("index cache identity mismatch")
        slots.append({"job_id": job["job_id"], "paper_id": paper_id, "status": selected["status"],
                      "selected_run_id": selected["run_id"], "record_paths": [p.relative_to(output).as_posix() for p in saved]})
    summary = seal({"schema_version": "four-category-batch/v1", "status": "complete" if len(slots) == len(plan["jobs"]) else "partial",
                    "plan_sha256": plan["plan_sha256"], "slots": slots, "index_paths": {p: "indexes/" + i["index_sha256"] + ".json" for p, i in indexes.items()},
                    "all_slots_accounted": len(slots) == len(plan["jobs"]), "backend_attempts_this_invocation": calls,
                    "real_model_calls": 0 if all(p["backend"] == "mock" for p in config["profiles"]) else None,
                    "dispatch_count_semantics": "adapter dispatches; provider or GPU computation is not observable from dispatch alone",
                    "semantic_evaluation": "pending_independent_sampled_reference"}, "batch_sha256")
    summary_path = output / "batches" / (summary["batch_sha256"] + ".json")
    if not summary_path.exists():
        write_new(summary_path, summary)
    return summary


def evaluation_split(corpus: dict, *, audit_fraction: float = .2, seed: int = 1729, source_root: Path | None = None) -> dict:
    """Group connected paper/dataset identities before a reproducible random split.

This selects cases, not labels. Inspect whole sampled units, including untagged
ones. The fraction is a configurable workload default, not a precision claim.
"""
    if not 0 < audit_fraction < 1:
        raise ValueError("audit fraction must be between zero and one")
    errors = corpus_errors(corpus)
    if errors:
        raise ValueError(errors)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(b)] = find(a)
    families = {}
    for paper in corpus["papers"]:
        key = "p:" + paper["paper_id"]
        labels = ["pdf:" + paper["source"]["source_identity"]["paper_sha256"]]
        if paper.get("family_id"):
            labels.append("paper-family:" + paper["family_id"])
        for label in labels:
            union(key, families.setdefault(label, key))
    unknown_dataset_families = []
    for dataset in corpus["datasets"]:
        key = "d:" + dataset["dataset_id"]
        labels = []
        if dataset.get("family_id"):
            labels.append("dataset-family:" + dataset["family_id"])
        source_hash = dataset.get("source_content_sha256")
        if source_root is not None:
            bundle = read_json(contained(source_root, dataset["bundle_path"]))
            if bundle["bundle_sha256"] != dataset["bundle_sha256"] or seal_errors(bundle, "bundle_sha256"):
                raise ValueError("dataset split source identity mismatch")
            source_hash = next(s["sha256"] for s in bundle["sources"] if s["role"] == "dataset")
        if source_hash:
            labels.append("dataset-content:" + source_hash)
        if not labels:
            # Unknown families cannot be presumed independent from each other.
            labels.append("unresolved-dataset-family")
            unknown_dataset_families.append(dataset["dataset_id"])
        for label in labels:
            union(key, families.setdefault(label, key))
    for match in corpus["matches"]:
        union("p:" + match["paper_id"], "d:" + match["dataset_id"])
    groups = defaultdict(list)
    for match in corpus["matches"]:
        groups[find("p:" + match["paper_id"])].append(match["match_id"])
    groups = [sorted(v) for v in groups.values()]
    groups.sort(key=lambda g: digest({"seed": seed, "members": g}))
    number = max(1, round(len(groups) * audit_fraction)) if len(groups) > 1 else 0
    return {"schema_version": "four-category-evaluation-split/v1", "seed": seed, "audit_fraction": audit_fraction,
            "calibration_groups": groups[number:], "independent_audit_groups": groups[:number],
            "status": "planned" if number else "insufficient_independent_groups",
            "unknown_dataset_families_grouped_conservatively": unknown_dataset_families,
            "human_labels": None, "include_untagged_evidence": True, "accuracy": None}
