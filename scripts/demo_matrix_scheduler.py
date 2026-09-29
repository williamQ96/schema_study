"""Local synthetic three-paper run; no HPC, weights, credentials or messages."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from high_fidelity_schema_study.four_category.common import read_json, write_new
from high_fidelity_schema_study.four_category.offline import make_fixture
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.paper_layout_evidence import preprocess_layout_pdf
from high_fidelity_schema_study.paper_layout_evidence_v3 import upgrade_bundle
from high_fidelity_schema_study.four_category.paper import prepare_paper
from high_fidelity_schema_study.four_category.scheduler import default_policy, run_scheduler
from high_fidelity_schema_study.four_category.scheduler_report import report
from high_fidelity_schema_study.four_category.queue_health import thresholds, cycle, render_alert
from high_fidelity_schema_study.four_category.scheduler_io import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new demonstration directory')
    sources = args.output/'sources'
    config, corpus = make_fixture(sources)
    config['classification']['profile_id'] = config['roles']['locals'][0]
    config['classification']['profile_sha256'] = profile_hash(config['profiles'][0])
    for n in (2, 3):
        pid = 'SYNTHETIC_P0'+str(n)
        prefix = sources/pid; prefix.mkdir()
        pdf = sources/'synthetic-paper.pdf'
        old, _, _ = preprocess_layout_pdf(pid, pdf)
        layout, reading, paper_input, _ = upgrade_bundle(old)
        write_new(prefix/'layout.json', layout); write_new(prefix/'input.json', paper_input)
        (prefix/'reading.txt').write_text(reading, encoding='utf-8')
        paper = prepare_paper(prefix/'layout.json', prefix/'reading.txt', prefix/'input.json', pdf, root=sources)
        corpus['papers'].append({'paper_id': pid, 'source': paper, 'family_id': 'synthetic-paper-family'})
        corpus['matches'].append({**copy.deepcopy(corpus['matches'][0]), 'match_id': 'SYNTHETIC_MATCH_'+str(n), 'paper_id': pid})
    policy = default_policy(); policy.update(poll_s=.05, heartbeat_s=.05)
    policy['health']['poll_s'] = .05
    write_new(args.output/'config.json', config); write_new(args.output/'corpus.json', corpus)
    write_new(args.output/'policy.json', policy)
    summary = run_scheduler(config, corpus, policy, root=args.output/'run', sources=sources,
                            leases=args.output/'leases', start_watchdog=True)
    write_new(args.output/'performance.json', report(args.output/'run'))
    # Inject a separate historical clock fixture, never contaminate live run state.
    fault = args.output/'health-fault-demo'
    atomic_json(fault/'snapshot.json', {'condition': 'synthetic-health-fixture', 'time': 1, 'terminal': False,
                                      'ready_jobs': 0, 'workers': []})
    cycle(fault, thresholds(), at=1000)
    event = next(iter(read_json(fault/'health/state.json')['outbox'].values()))['event']
    (args.output/'alert-preview-zh.txt').write_text(render_alert(event, fault, 'zh'), encoding='utf-8')
    print({'matrices': len(summary['matrices']), 'slots': summary['slot_counts'], 'live_model_calls': 0, 'telegram_sends': 0})


if __name__ == '__main__':
    main()
