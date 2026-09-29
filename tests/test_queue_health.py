from high_fidelity_schema_study.four_category.queue_health import thresholds, observations, evaluate, render_alert


def snap(**overrides):
    value = {'condition': 'x' * 64, 'time': 100, 'terminal': False,
             'workers': [{'worker_id': 'gpu0', 'job_id': None, 'heartbeat_at': 100}],
             'ready_jobs': 0, 'eligible_idle_workers': []}
    value.update(overrides)
    return value


def test_fraction_threshold_validation():
    for value in (-.1, 1.1, float('nan')):
        try:
            thresholds({'classification_failure_fraction': value})
        except ValueError:
            pass
        else:
            assert False, value


def test_systemic_classifier_alert_before_terminal_and_is_operational_only():
    stages = {'classification': {'success': 0, 'contract_invalid': 9, 'invalid_request': 1},
              'local_extraction': {'success': 11}, 'dataset_parse': {'success': 11},
              'soft_reference': {'deferred': 10}}
    s = snap(stage_counts=stages)
    issues = observations(s, 100, thresholds())
    assert any(issue[0] == 'systemic_classification_rejection' for issue in issues)
    event = {'kind': 'systemic_classification_rejection', 'scope': 'classification',
             'transition': 'open', 'time': 100, 'condition': 'x'*64,
             'facts': {'stage_counts': stages, 'completed': 10, 'failures': 10}}
    rendered = render_alert(event, '/reports')
    assert 'Classifier: contract_invalid=9, invalid_request=1, success=0' in rendered
    assert 'Extraction: success=11' in rendered and 'Dataset: success=11' in rendered
    assert 'Deferred reference: deferred=10' in rendered
    assert 'accuracy' in rendered.lower()


def test_dependency_wait_idle_worker_is_excluded_from_skew_and_imbalance():
    s = snap(ready_jobs=2, eligible_idle_workers=['gpu0'], workers=[
        {'worker_id': 'gpu0', 'job_id': None, 'heartbeat_at': 100,
         'compatible_ready_jobs': 0, 'dependency_wait_jobs': 3},
        {'worker_id': 'gpu1', 'job_id': 'busy', 'heartbeat_at': 100, 'progress_at': 100,
         'gpu_metrics': [{'utilization_pct': 95, 'total_bytes': 100, 'used_bytes': 20}]}])
    assert not ({i[0] for i in observations(s, 100, thresholds())} &
                {'allocation_imbalance', 'sustained_utilization_skew'})


def test_genuine_compatible_idle_worker_remains_imbalance():
    s = snap(ready_jobs=2, eligible_idle_workers=['gpu0'], workers=[
        {'worker_id': 'gpu0', 'job_id': None, 'heartbeat_at': 100,
         'compatible_ready_jobs': 2, 'dependency_wait_jobs': 0},
        {'worker_id': 'gpu1', 'job_id': 'busy', 'heartbeat_at': 100, 'progress_at': 100}])
    assert 'allocation_imbalance' in {i[0] for i in observations(s, 100, thresholds())}


def test_old_snapshots_retain_idle_worker_behavior():
    s = snap(ready_jobs=2, eligible_idle_workers=['gpu0'], workers=[
        {'worker_id': 'gpu0', 'job_id': None, 'heartbeat_at': 100, 'progress_at': 100},
        {'worker_id': 'gpu1', 'job_id': 'busy', 'heartbeat_at': 100, 'progress_at': 100}])
    assert 'allocation_imbalance' in {i[0] for i in observations(s, 100, thresholds())}


def test_stale_snapshot_does_not_apply_dependency_idle_queue_context():
    s = snap(ready_jobs=2, time=0, workers=[
        {'worker_id': 'gpu0', 'job_id': 'job', 'heartbeat_at': 1000, 'progress_at': -2000,
         'compatible_ready_jobs': 0, 'dependency_wait_jobs': 3}])
    issues = observations(s, 1000, thresholds())
    assert 'worker_no_progress' in {i[0] for i in issues}


def test_dependency_waiting_idle_worker_still_reports_low_gpu_headroom():
    s = snap(workers=[{'worker_id': 'gpu0', 'job_id': None, 'heartbeat_at': 100,
                       'compatible_ready_jobs': 0, 'dependency_wait_jobs': 3,
                       'gpu_reserve_bytes': 50,
                       'gpu_metrics': [{'utilization_pct': 10, 'total_bytes': 100,
                                        'used_bytes': 80}]}])
    issues = observations(s, 100, thresholds())
    assert 'gpu_headroom_low' in {i[0] for i in issues}
