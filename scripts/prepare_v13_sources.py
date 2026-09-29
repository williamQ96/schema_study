"""Freeze all cohort PDFs through MinerU Basic CPU; keep datasets independent."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from high_fidelity_schema_study.four_category.common import digest, file_digest, read_json, write_new
from high_fidelity_schema_study.four_category.mineru_adapter import build_mineru_bundle
from high_fidelity_schema_study.four_category.paper import prepare_paper, verify_paper
from high_fidelity_schema_study.four_category.preprocessing import make_config, prepare_condition, verify_condition, PARSER_PROFILE_KEYS


def prepare(args):
    old = read_json(args.condition)
    corpus = copy.deepcopy(old['corpus'])
    assert len(corpus['papers']) == 10
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / 'source_bundle'
    shutil.copytree(args.baseline, bundle)
    # Every source member must match the actual prior production condition.
    for row in corpus['papers']:
        issues = verify_paper(row['source'], bundle)
        if issues:
            raise ValueError('baseline_invalid:' + row['paper_id'] + ':' + repr(issues[:3]))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    logs = args.output / 'parser-logs'
    logs.mkdir()
    def parse(row):
        pid = row['paper_id']
        out = bundle / 'mineru_raw' / pid
        cmd = [str(args.python), '-X', 'utf8', '-m', 'high_fidelity_schema_study.four_category.mineru_runner',
               'parse', '--pdf', str(bundle / row['source']['artifacts']['pdf']['path']),
               '--output', str(out), '--model-root', str(args.models), '--model-manifest', str(args.model_manifest),
               '--tier', 'basic', '--ocr-mode', 'txt']
        with (logs / (pid + '.log')).open('wb') as log:
            finished = subprocess.run(cmd, env=env, stdout=log, stderr=log, timeout=1800)
        receipt = read_json(out / 'receipt.json')
        if finished.returncode or receipt['status'] != 'success':
            raise ValueError('mineru_parse_failed:' + pid)
        return pid, receipt
    parsed = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(parse, row) for row in corpus['papers']]):
            pid, receipt = future.result()
            parsed[pid] = receipt
            print(json.dumps({'paper': pid, 'pages': receipt['pages'], 'parse_status': receipt['status']}), flush=True)
    conditions = {}
    for row in corpus['papers']:
        pid, baseline = row['paper_id'], row['source']
        native = read_json(bundle / baseline['artifacts']['input']['path'])
        raw = bundle / 'mineru_raw' / pid
        parser = read_json(raw / 'parser_identity.json')
        for member in parsed[pid]['files']:
            if file_digest(raw / member['path']) != member['file_bytes_sha256']:
                raise ValueError('parser_receipt_member_changed:' + pid)
        layout, reading, paper = build_mineru_bundle(native, read_json(raw / 'middle.json'), parser,
                                                   pdf_sha256=native['source_pdf_sha256'])
        derived = bundle / 'mineru' / pid
        derived.mkdir(parents=True)
        write_new(derived / 'layout.json', layout)
        write_new(derived / 'input.json', paper)
        (derived / 'reading_text.txt').write_text(reading, encoding='utf8', newline='\n')
        source = prepare_paper(derived / 'layout.json', derived / 'reading_text.txt', derived / 'input.json',
                               bundle / baseline['artifacts']['pdf']['path'], root=bundle,
                               mineru_raw_path=raw / 'middle.json',
                               baseline_input_path=bundle / baseline['artifacts']['input']['path'],
                               parser_identity_path=raw / 'parser_identity.json')
        switch = make_config(mineru_enabled=True, auxiliary_enabled=False,
                             parser_profile={key: parser[key] for key in PARSER_PROFILE_KEYS})
        package = prepare_condition(bundle, baseline, switch, mineru_source=source)
        assert not verify_condition(package, bundle)
        write_new(bundle / 'conditions' / (pid + '.json'), package)
        row['source'] = source
        # Keep the cohort row's explicit source identity synchronized when present.
        if 'source_sha256' in row:
            row['source_sha256'] = source['source_sha256']
        conditions[pid] = {'package_sha256': digest(package), 'parser_identity_sha256': parser['parser_identity_sha256'],
                           'pages': len(paper['pages']), 'source_sha256': source['source_sha256']}
        print(json.dumps({'paper': pid, 'source_status': 'verified', 'pages': len(paper['pages'])}), flush=True)
    corpus['cohort_id'] = 'v13-mineru-' + corpus.get('cohort_id', 'ten-paper-cohort')
    write_new(args.output / 'corpus.json', corpus)
    report = {'pipeline': 'V13', 'mineru_enabled': True, 'auxiliary_enabled': False,
              'paper_count': len(corpus['papers']), 'dataset_asset_count': len(corpus['datasets']),
              'match_count': len(corpus['matches']), 'corpus_canonical_sha256': digest(corpus),
              'conditions': conditions, 'parser_model_manifest_file_sha256': file_digest(args.model_manifest),
              'parser_environment': str(args.python), 'dataset_inputs_passed_to_paper_parser': False,
              'model_generation_calls': 0}
    write_new(args.output / 'source-preparation.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('condition', 'baseline', 'output', 'python', 'models', 'model-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a)))
