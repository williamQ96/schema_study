"""Freeze and stage the fresh ten-paper V13 run; never reuse prior answers."""
from pathlib import Path
import copy
import argparse
import json
import shlex
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT / 'scripts')]
from mercury_scheduler_v2_access import connect, execute
from high_fidelity_schema_study.four_category.common import digest, file_digest
from high_fidelity_schema_study.four_category.mercury import code_identity
from high_fidelity_schema_study.four_category.backends import profile_hash
from high_fidelity_schema_study.four_category.workflow import tasks_for_config, experiment_errors
from high_fidelity_schema_study.four_category.extraction_v13 import PROTOCOL, DEFAULT_POLICY

LOCAL = ROOT / 'data/experiments/v13_full_2026_09_28_v1'
REMOTE = '/storage/users/williamq/schema-study-deployment-20260926/jobs/v13-full-20260928-v1'
PYTHON = '/home/users/williamq/schema-study-deployment-20260926/jobs/matrix-scheduler-20260926-v7/control-venv/bin/python'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == raw, str(path)
    else:
        path.write_bytes(raw)

def stage(revision=1):
    remote = REMOTE if revision == 1 else REMOTE + '/candidate-' + format(revision, '02d')
    old = json.loads((LOCAL / 'previous-condition.json').read_text())
    deployment = json.loads((LOCAL / 'previous-deployment.json').read_text())
    config = copy.deepcopy(old['config'])
    identity = code_identity()
    config.update(experiment_id='mercury-v13-mineru-full-20260928-v1',
                  preprocessing=dict(pipeline='V13', mineru_enabled=True, auxiliary_enabled=False),
                  extraction_input_protocol=PROTOCOL, extraction_grouping=DEFAULT_POLICY,
                  notes=['Fresh V13: full MinerU v6 sources, V12 nested-object contract in resumable V13 page groups.',
                         'Latest Scheduler V2; unchanged model weights, placement and generation parameters.',
                         'All ten papers; P05 day/hour are distinct physical dataset assets.',
                         'No prior answers imported. Rejected and truncated returns remain results. Soft reference deferred.',
                         'Structural auxiliary disabled; independently frozen Qwen taxonomy classification retained.'])
    config['deployment'] = dict(schema_version='matrix-scheduler-deployment/v1', previous_config_sha256=digest(old['config']),
                                source_tree_sha256=digest(identity), image_file_bytes_sha256=deployment['execution_identity']['image_file_sha256'],
                                placement='heterogeneous_1_1_2', scheduler_version='v2')
    for profile in config['profiles']:
        if profile['backend'] == 'transformers':
            profile['runtime']['deployment_identity']['source_tree_canonical_sha256'] = digest(identity)
    config['classification']['profile_sha256'] = profile_hash(next(p for p in config['profiles'] if p['profile_id'] == config['classification']['profile_id']))
    config['task_hashes'] = {k:v['task_sha256'] for k,v in tasks_for_config(config).items()}
    assert not experiment_errors(config), experiment_errors(config)
    policy = copy.deepcopy(old['policy'])
    policy['qualification'] = None
    staged = LOCAL / ('release' if revision == 1 else 'release-' + str(revision))
    write(staged / 'config.json', config)
    write(staged / 'policy-pending.json', policy)
    write(staged / 'corpus.json', json.loads((LOCAL / 'prepared/corpus.json').read_text()))
    write(staged / 'science-manifest.json', dict(files=identity, source_tree_sha256=digest(identity)))
    for p in config['profiles']:
        if p['backend'] == 'transformers':
            write(staged / 'profiles' / (p['profile_id'] + '.json'), p)
    operations = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in (ROOT / 'scheduler_v2').glob('*.py')}
    write(staged / 'runtime-manifest.json', dict(files=operations))
    archive = LOCAL / ('v13-release.tar.gz' if revision == 1 else 'v13-release-' + str(revision) + '.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as tar:
        for name in identity:
            tar.add(ROOT / name, arcname='source/high_fidelity_schema_study/' + name)
        for name in operations:
            tar.add(ROOT / name, arcname='runtime-v1/' + name)
        tar.add(LOCAL / 'prepared/source_bundle', arcname='source_bundle')
        for path in staged.rglob('*'):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(staged).as_posix())
    gateway, client = connect()
    try:
        with client.open_sftp() as sftp:
            if revision > 1:
                sftp.mkdir(remote)
            sftp.put(str(archive), remote + '/v13-release.tar.gz')
    finally:
        client.close(); gateway.close()
    result = execute('''import json,tarfile,hashlib
from pathlib import Path
j=Path(''' + repr(remote) + ''')
a=j/'v13-release.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()==''' + repr(file_digest(archive)) + '''
assert not (j/'source').exists()
with tarfile.open(a) as t: t.extractall(j,filter='data')
for manifest,prefix in [('science-manifest.json','source/high_fidelity_schema_study'),('runtime-manifest.json','runtime-v1')]:
 for name,h in json.loads((j/manifest).read_text())['files'].items():
  assert hashlib.sha256((j/prefix/name).read_bytes()).hexdigest()==h,name
print(json.dumps(dict(status='staged',source_files=len(json.loads((j/'science-manifest.json').read_text())['files']),archive_bytes=a.stat().st_size)))''', 180)
    write(LOCAL / ('stage-receipt.json' if revision == 1 else 'stage-receipt-' + str(revision) + '.json'), result)
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=int, default=1)
    args = parser.parse_args()
    if args.revision < 1:
        raise ValueError('positive_revision_required')
    stage(args.revision)
