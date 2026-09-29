"""Freeze a separate V13 revision 2 candidate; never mutate the first V13 run."""
from pathlib import Path
import argparse
import copy
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT / 'scripts')]
from mercury_scheduler_v2_access import connect, execute
from high_fidelity_schema_study.four_category.common import digest, file_digest
from high_fidelity_schema_study.four_category.mercury import code_identity
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.workflow import tasks_for_config, experiment_errors

LOCAL = ROOT / 'data/experiments/v13_repair_2026_09_28_r2'
OLD = ROOT / 'data/experiments/v13_full_2026_09_28_v1'
REMOTE = '/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-repair-20260928-r2'
PYTHON = '/home/users/williamq/schema-study-deployment-20260926/jobs/matrix-scheduler-20260926-v7/control-venv/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('immutable_candidate_conflict:' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)


def stage(candidate):
    from high_fidelity_schema_study.four_category import extraction_v13r2, classification_v13r2
    remote = REMOTE + '/candidate-' + format(candidate, '02d')
    staged = LOCAL / ('candidate-' + format(candidate, '02d'))
    config = json.loads((OLD / 'release-2/config.json').read_text())
    old_hash = digest(config)
    identity = code_identity()
    config.update(experiment_id='mercury-v13-revision2-20260928-c' + str(candidate),
                  extraction_input_protocol=extraction_v13r2.PROTOCOL,
                  extraction_grouping=copy.deepcopy(extraction_v13r2.DEFAULT_POLICY),
                  notes=['V13 revision 2; an independent condition, never a replacement of historical returns.',
                         'MinerU ON; structural auxiliary OFF; taxonomy classification ON with constrained group-local source.',
                         'Complete target page and immediate neighboring pages; one typed W evidence namespace.',
                         'Muse reasoning bounded at 2048; at least 8192 final tokens reserved within 16384 total.',
                         'Whitespace bounded; ordered grammar-schema identity pinned. Model weights and sampling unchanged.',
                         'Same ten papers and eleven dataset assets. Soft reference deferred. No semantic accuracy claim.'])
    config['classification']['protocol'] = classification_v13r2.PROTOCOL
    config['deployment'].update(previous_config_sha256=old_hash, source_tree_sha256=digest(identity))
    for profile in config['profiles']:
        if profile['backend'] != 'transformers':
            continue
        runtime = profile['runtime']
        runtime['deployment_identity']['source_tree_canonical_sha256'] = digest(identity)
        output = runtime['structured_output']
        if output['channel'] == 'muse-atem/v1':
            output.update(channel='muse-atem/v2', reasoning_max_tokens=2048,
                          final_min_tokens=8192, max_whitespace_cnt=2)
        else:
            output.update(channel='json/v2', max_whitespace_cnt=2)
    config['classification']['profile_sha256'] = profile_hash(next(
        p for p in config['profiles'] if p['profile_id'] == config['classification']['profile_id']))
    config['task_hashes'] = {k: v['task_sha256'] for k, v in tasks_for_config(config).items()}
    errors = experiment_errors(config)
    if errors:
        raise ValueError(errors)
    policy = json.loads((OLD / 'release-2/policy-pending.json').read_text())
    policy['qualification'] = None
    write(staged / 'config.json', config)
    write(staged / 'policy-pending.json', policy)
    write(staged / 'corpus.json', json.loads((OLD / 'prepared/corpus.json').read_text()))
    write(staged / 'previous-deployment.json', json.loads((OLD / 'previous-deployment.json').read_text()))
    write(staged / 'science-manifest.json', dict(files=identity, source_tree_sha256=digest(identity)))
    for p in config['profiles']:
        if p['backend'] == 'transformers':
            write(staged / 'profiles' / (p['profile_id'] + '.json'), p)
    operations = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in (ROOT / 'scheduler_v2').glob('*.py')}
    write(staged / 'runtime-manifest.json', dict(files=operations))
    helpers = ['v13_cpu_preflight.py', 'v13_capacity_launcher.py', 'v13_repair_probe.py',
               'v13_repair_launcher.py', 'freeze_v13_repair.py']
    helper_hashes = {name: file_digest(ROOT / 'scripts' / name) for name in helpers}
    write(staged / 'qualification-tools.json', {'files': helper_hashes})
    archive = staged / 'release.tar.gz'
    with archive.open('xb') as output, tarfile.open(fileobj=output, mode='w:gz') as tar:
        for name in identity:
            tar.add(ROOT / name, arcname='source/high_fidelity_schema_study/' + name)
        for name in operations:
            tar.add(ROOT / name, arcname='runtime-v1/' + name)
        tar.add(OLD / 'prepared/source_bundle', arcname='source_bundle')
        for path in staged.rglob('*'):
            if path.is_file() and path != archive:
                tar.add(path, arcname=path.relative_to(staged).as_posix())
        for name in helpers:
            tar.add(ROOT / 'scripts' / name, arcname=name)
    gateway, client = connect()
    try:
        with client.open_sftp() as sftp:
            sftp.mkdir(remote)
            sftp.put(str(archive), remote + '/release.tar.gz')
    finally:
        client.close(); gateway.close()
    result = execute('''import json,tarfile,hashlib
from pathlib import Path
j=Path(''' + repr(remote) + ''')
a=j/'release.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()==''' + repr(file_digest(archive)) + '''
assert not (j/'source').exists()
with tarfile.open(a) as t:t.extractall(j,filter='data')
for manifest,prefix in [('science-manifest.json','source/high_fidelity_schema_study'),('runtime-manifest.json','runtime-v1'),('qualification-tools.json','')]:
 for name,h in json.loads((j/manifest).read_text())['files'].items():
  assert hashlib.sha256((j/prefix/name).read_bytes()).hexdigest()==h,name
print(json.dumps({'status':'staged','candidate':str(j),'archive_bytes':a.stat().st_size}))''', 180)
    write(staged / 'stage-receipt.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=int, required=True)
    args = parser.parse_args()
    if args.candidate < 1:
        raise ValueError('positive_candidate_required')
    stage(args.candidate)
