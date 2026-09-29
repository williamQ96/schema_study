import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import deploy_production_health_bridge_v13 as deploy


def previous_config():
    return {'condition': 'a' * 64, 'bot_id': 123, 'bot_username': 'study_bot',
            'chat_id': -100123, 'repo': 'D:/repo', 'python_paths': ['C:/deps'],
            'codex_exe': 'C:/codex.exe', 'rollout_path': 'C:/rollouts',
            'thread_id': 'thread-1', 'hmac_key': 'b' * 64}


def frozen_bytes(*, preprocessing=None, condition='c' * 64):
    return json.dumps({'condition': condition, 'config': {'preprocessing': preprocessing or
                      {'pipeline': 'V13', 'mineru_enabled': True, 'auxiliary_enabled': False}},
                       'jobs': []}).encode()


def test_v13_config_pair_binds_fresh_hash_mineru_and_private_state(tmp_path):
    raw = frozen_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    local, remote, actual = deploy.build_configs(
        raw, previous_config(), expected_condition_sha256=expected,
        private=tmp_path, repo=tmp_path, hmac_key='d' * 64)
    assert actual == expected
    assert local['condition'] == remote['condition'] == 'c' * 64
    assert local['condition_file_bytes_sha256'] == remote['condition_file_bytes_sha256'] == expected
    assert local['hmac_key'] == remote['hmac_key'] == 'd' * 64
    assert all(local[k] == remote[k] for k in ('bot_id', 'bot_username', 'chat_id'))
    assert local['production_root'] == deploy.PRODUCTION_ROOT
    assert local['worker_state_root'] == '/var/tmp/schema-study-williamq/v13-full-20260928-v1'
    assert local['scheduler_version'] == 'v2'
    assert local['local_config_path'].endswith('production-health-bridge-v13-local.json')
    assert local['state_dir'].endswith('production-health-bridge-v13-state')
    assert local['instructions_path'].endswith('production_health_codex_bridge_v13.md')
    assert remote['bridge_root'] == deploy.BRIDGE_ROOT


@pytest.mark.parametrize('raw,sha,error', [
    (frozen_bytes(), '0' * 64, 'fresh_condition_file_hash_mismatch'),
    (frozen_bytes(condition='a' * 64), None, 'fresh_condition_reuses_previous_identity'),
    (frozen_bytes(preprocessing={'pipeline': 'V12', 'mineru_enabled': False,
                                 'auxiliary_enabled': False}), None, 'v13_mineru_preprocessing_required'),
])
def test_v13_rejects_stale_condition_or_preprocessing(raw, sha, error):
    with pytest.raises(ValueError, match=error):
        deploy.build_configs(raw, previous_config(),
                             expected_condition_sha256=sha or hashlib.sha256(raw).hexdigest(),
                             hmac_key='d' * 64)


def test_v13_rejects_reused_hmac_key():
    raw = frozen_bytes()
    with pytest.raises(ValueError, match='new_private_hmac_key_required'):
        deploy.build_configs(raw, previous_config(),
                             expected_condition_sha256=hashlib.sha256(raw).hexdigest(),
                             hmac_key=previous_config()['hmac_key'])


def test_v13_cli_requires_explicit_fresh_condition_hash():
    with pytest.raises(SystemExit, match='2'):
        deploy.main([])
