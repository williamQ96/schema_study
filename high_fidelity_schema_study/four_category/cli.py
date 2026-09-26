"""Command-line entry points for the additive four-category study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .common import read_json, write_new
from .dataset import parse_dataset
from .dataset_batch import parse_batch
from .offline import make_fixture, mock_transport
from .packet import build_packet, verify_packet
from .paper import prepare_paper
from .workflow import evaluation_split, plan_jobs, run_batch


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-category paper/dataset study workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    paper = commands.add_parser("prepare-paper", help="Bind existing v3 paper artifacts and PDF bytes")
    for name in ("layout", "reading", "input", "pdf", "root", "output"):
        paper.add_argument("--" + name, required=True, type=Path)

    dataset = commands.add_parser("parse-dataset", help="Parse existing dataset bytes independently")
    dataset.add_argument("--path", required=True, type=Path)
    dataset.add_argument("--root", required=True, type=Path)
    dataset.add_argument("--format-hint")
    dataset.add_argument("--sample-limit", type=int, default=200)
    dataset.add_argument("--taxonomy", type=Path, help="Explicit taxonomy JSON; default is built-in v1")
    dataset.add_argument("--sidecar", action="append", type=Path, default=[])
    dataset.add_argument("--output", required=True, type=Path)

    datasets = commands.add_parser("parse-datasets", help="Parse or resume manifest-listed dataset sources")
    datasets.add_argument("--manifest", required=True, type=Path)
    datasets.add_argument("--source-root", required=True, type=Path)
    datasets.add_argument("--output", required=True, type=Path, help="Cache directory, usually within source-root")
    datasets.add_argument("--sample-limit", type=int, default=200)
    datasets.add_argument("--taxonomy", type=Path, help="Explicit taxonomy JSON; default is built-in v1")
    datasets.add_argument("--max-jobs", type=int)
    datasets.add_argument("--summary-output", type=Path, help="Write the returned batch to a new JSON file")

    plan = commands.add_parser("plan", help="Create a content-bound job plan")
    for name in ("config", "corpus", "output"):
        plan.add_argument("--" + name, required=True, type=Path)

    run = commands.add_parser("run", help="Execute or resume a planned batch")
    for name in ("config", "corpus", "source-root", "output"):
        run.add_argument("--" + name, required=True, type=Path)
    run.add_argument("--allow-live", action="store_true", help="Permit qualified frozen real backends")
    run.add_argument("--max-jobs", type=int)
    run.add_argument("--summary-output", type=Path, help="Write the returned batch to a new JSON file")

    packet = commands.add_parser("packet", help="Build a new portable handoff packet")
    for name in ("config", "corpus", "batch", "run-root", "source-root", "output"):
        packet.add_argument("--" + name, required=True, type=Path)

    verify = commands.add_parser("verify", help="Verify packet bytes and independent derivations")
    verify.add_argument("--packet", required=True, type=Path)
    verify.add_argument("--source-root", required=True, type=Path)
    verify.add_argument("--expected-experiment-sha256", help="Trusted external SHA-256 of canonical experiment JSON")
    verify.add_argument("--output", type=Path, help="Write a new JSON report as well as printing it")

    split = commands.add_parser("split", help="Plan grouped independent semantic audit cases")
    split.add_argument("--corpus", required=True, type=Path)
    split.add_argument("--source-root", type=Path, help="Read bound dataset sources to group identical content")
    split.add_argument("--audit-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=1729)
    split.add_argument("--output", required=True, type=Path)

    demo = commands.add_parser("offline-demo", help="Create synthetic, zero-live-call fixture and packet")
    demo.add_argument("--output", required=True, type=Path)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _args(argv)
    if args.command == "prepare-paper":
        result = prepare_paper(args.layout, args.reading, args.input, args.pdf, root=args.root)
        write_new(args.output, result)
    elif args.command == "parse-dataset":
        result = parse_dataset(args.path, root=args.root, format_hint=args.format_hint,
                               sample_limit=args.sample_limit, sidecars=args.sidecar,
                               taxonomy=read_json(args.taxonomy) if args.taxonomy else None)
        write_new(args.output, result)
    elif args.command == "parse-datasets":
        result = parse_batch(read_json(args.manifest), source_root=args.source_root, output=args.output,
                             sample_limit=args.sample_limit, max_jobs=args.max_jobs,
                             taxonomy_value=read_json(args.taxonomy) if args.taxonomy else None)
        if args.summary_output:
            write_new(args.summary_output, result)
    elif args.command == "plan":
        result = plan_jobs(read_json(args.config), read_json(args.corpus))
        write_new(args.output, result)
    elif args.command == "run":
        result = run_batch(read_json(args.config), read_json(args.corpus), source_root=args.source_root,
                           output=args.output, allow_live=args.allow_live, max_jobs=args.max_jobs)
        if args.summary_output:
            write_new(args.summary_output, result)
    elif args.command == "packet":
        result = build_packet(read_json(args.config), read_json(args.corpus), read_json(args.batch),
                              run_root=args.run_root, source_root=args.source_root, output=args.output)
    elif args.command == "verify":
        result = verify_packet(args.packet, source_root=args.source_root,
                               expected_experiment_sha256=args.expected_experiment_sha256)
        if args.output:
            write_new(args.output, result)
    elif args.command == "split":
        result = evaluation_split(read_json(args.corpus), audit_fraction=args.audit_fraction, seed=args.seed, source_root=args.source_root)
        write_new(args.output, result)
    else:
        destination = args.output.resolve()
        if destination.exists():
            raise FileExistsError(f"offline-demo output already exists: {destination}")
        destination.mkdir(parents=True)
        source_root = destination / "source"
        config, corpus = make_fixture(source_root)
        write_new(destination / "config.json", config)
        write_new(destination / "corpus.json", corpus)
        transports = {profile["profile_id"]: mock_transport for profile in config["profiles"]}
        run_root = destination / "runs"
        first = run_batch(config, corpus, source_root=source_root, output=run_root, transports=transports)
        packet = build_packet(config, corpus, first, run_root=run_root, source_root=source_root,
                              output=destination / "packet")
        resumed = run_batch(config, corpus, source_root=source_root, output=run_root, transports=transports)
        result = {"schema_version": "four-category-offline-demo/v1", "purpose": "synthetic_software_test_only",
                  "execution_mode": "injected_mock_transport", "live_requests": 0,
                  "first_batch": first, "resume_batch": resumed, "packet_verification": packet,
                  "config_path": "config.json", "corpus_path": "corpus.json", "packet_path": "packet"}
        write_new(destination / "report.json", result)
    _json(result)
    return 0 if result.get("status") not in {"fail", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
