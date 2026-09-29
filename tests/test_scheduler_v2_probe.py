import sys
import pytest
from scheduler_v2.io import atomic_json, file_sha
from scheduler_v2.probe import specification


def test_probe_requires_pinned_executable_and_active_one_gpu_reservation(tmp_path):
    config = {'local_root': str(tmp_path)}
    manifest = tmp_path / 'probe.json'
    atomic_json(manifest, {'argv': [sys.executable, '-c', 'pass'], 'files': {sys.executable: file_sha(sys.executable)}})
    reservation = {'key': 'probe', 'gpu_ids': ['GPU0'], 'expires_at': 200}
    atomic_json(tmp_path/'snapshot.json', {'controls': {'reservations': [reservation]}})
    assert specification(config, 'probe', manifest, file_sha(manifest), 100)[1] == reservation
    with pytest.raises(ValueError, match='expired'):
        specification(config, 'probe', manifest, file_sha(manifest), 201)
    with pytest.raises(ValueError, match='not_acknowledged'):
        specification(config, 'other', manifest, file_sha(manifest), 100)
    with pytest.raises(ValueError, match='identity_changed'):
        specification(config, 'probe', manifest, 'wrong', 100)
    reservation['gpu_ids'].append('GPU1')
    atomic_json(tmp_path/'snapshot.json', {'controls': {'reservations': [reservation]}})
    with pytest.raises(ValueError, match='reservation_invalid'):
        specification(config, 'probe', manifest, file_sha(manifest), 100)


def test_probe_does_not_accept_unpinned_shell_text(tmp_path):
    manifest = tmp_path / 'probe.json'
    atomic_json(manifest, {'argv': 'python run.py', 'files': {}})
    with pytest.raises(ValueError, match='explicit_list'):
        specification({'local_root': str(tmp_path)}, 'x', manifest, file_sha(manifest), 0)
