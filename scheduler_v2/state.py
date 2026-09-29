"""Single-writer, replayable group scheduler ledger.

The shared directory contains immutable events. SQLite is only a local index of
those events and may be deleted and rebuilt without changing an outcome.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid

from .io import digest, file_sha, read, write_once


TERMINAL = frozenset({'success', 'truncated', 'refused', 'contract_invalid',
                      'invalid_request', 'transport_error', 'infrastructure_failed',
                      'generation_incomplete', 'incomplete', 'completed_with_rejections',
                      'blocked_dependency', 'deferred', 'unavailable', 'rejected'})
ACTIVE = frozenset({'assigned', 'pending_verification'})


class Ledger:
    def __init__(self, local_root, shared_root, deployment_id):
        if not isinstance(deployment_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', deployment_id):
            raise ValueError('invalid_deployment_id')
        self.local_root = Path(local_root)
        self.shared_root = Path(shared_root)
        self.deployment_id = deployment_id
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.journal = self.shared_root / 'scheduler_v2' / deployment_id / 'events'
        self.journal.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.local_root / 'scheduler_v2.sqlite', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL,
                body TEXT NOT NULL, status TEXT NOT NULL, result_ref TEXT, index_ref TEXT);
            CREATE TABLE IF NOT EXISTS groups (job_id TEXT NOT NULL, group_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, status TEXT NOT NULL, generation_returned INTEGER,
                attempt_refs TEXT, ready_at REAL, assignment_id TEXT,
                PRIMARY KEY(job_id,group_id));
            CREATE TABLE IF NOT EXISTS assignments (assignment_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL, group_id TEXT NOT NULL, worker_id TEXT NOT NULL,
                incarnation TEXT NOT NULL, epoch INTEGER NOT NULL, status TEXT NOT NULL,
                raw_ref TEXT, result_ref TEXT, verdict TEXT, started_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS controls (key TEXT PRIMARY KEY, action TEXT NOT NULL,
                payload TEXT NOT NULL, at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS recoveries (assignment_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL, at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        self._verify_existing_journal()
        self._sync()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.db.close()

    def _meta(self, key, default=None):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set_meta(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, json.dumps(value)))

    def _verify_existing_journal(self):
        """A surviving local index must still match the immutable shared log."""
        expected = 1
        for row in self.db.execute('SELECT seq,sha FROM events ORDER BY seq'):
            if row['seq'] != expected:
                raise ValueError('journal_sequence_gap')
            path = self.journal / f'{row["seq"]:08d}.json'
            if not path.is_file() or digest(read(path)) != row['sha']:
                raise ValueError('journal_existing_event_mismatch')
            expected += 1
        for path in self.journal.glob('*.json'):
            try:
                seq = int(path.stem)
            except ValueError as exc:
                raise ValueError('journal_filename_invalid') from exc
            if seq < 1 or seq > expected - 1 and seq != expected and not (self.journal / f'{seq-1:08d}.json').is_file():
                raise ValueError('journal_sequence_gap')

    def _sync(self):
        expected = (self.db.execute('SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0] + 1)
        while (path := self.journal / f'{expected:08d}.json').is_file():
            seq = expected
            event = read(path)
            if event.get('seq') != seq or event.get('deployment_id') != self.deployment_id:
                raise ValueError('journal_identity_mismatch')
            if event.get('prev_sha') != self._meta('last_sha'):
                raise ValueError('journal_chain_mismatch')
            with self.db:
                self._apply(event)
                self.db.execute('INSERT INTO events VALUES(?,?)', (seq, digest(event)))
                self._set_meta('last_sha', digest(event))
            expected += 1

    def _emit(self, kind, **fields):
        self._sync()
        seq = self.db.execute('SELECT COALESCE(MAX(seq),0)+1 FROM events').fetchone()[0]
        event = {'schema_version': 'scheduler-v2-event/v1', 'seq': seq,
                 'deployment_id': self.deployment_id, 'prev_sha': self._meta('last_sha'),
                 'kind': kind, **fields}
        write_once(self.journal / f'{seq:08d}.json', event)
        self._sync()
        return event

    def _apply(self, event):
        kind = event['kind']
        if kind == 'initialize':
            for order, job in enumerate(event['jobs']):
                jid = job['job_id']
                imp = event['imported'].get(jid, {})
                self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)',
                                (jid, order, json.dumps(job, sort_keys=True),
                                 imp.get('parent_status', 'pending'), json.dumps(imp.get('result_ref')),
                                 json.dumps(imp.get('index_ref'))))
                for position, gid in enumerate(event['group_plans'].get(jid, [])):
                    group = imp.get('groups', {}).get(gid, {})
                    self.db.execute('INSERT INTO groups VALUES(?,?,?,?,?,?,?,?)',
                                    (jid, gid, position, group.get('status', 'pending'),
                                     int(group['generation_returned']) if 'generation_returned' in group else None,
                                     json.dumps(group.get('attempt_refs')), None, None))
            self._set_meta('plan_sha', event['plan_sha'])
            self._set_meta('epoch', 0)
        elif kind == 'epoch':
            self._set_meta('epoch', event['epoch'])
        elif kind == 'worker':
            self._set_meta('worker:' + event['worker_id'],
                           {'incarnation': event['incarnation'], 'gpu_ids': event.get('gpu_ids')})
        elif kind == 'ready':
            self.db.execute('UPDATE groups SET ready_at=COALESCE(ready_at,?) WHERE job_id=? AND group_id=?',
                            (event['at'], event['job_id'], event['group_id']))
        elif kind == 'claim':
            a = event['assignment']
            self.db.execute('INSERT INTO assignments VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                            (a['assignment_id'], a['job_id'], a['group_id'], a['worker_id'],
                             a['incarnation'], a['epoch'], 'assigned', None, None, None, a['started_at']))
            self.db.execute("UPDATE groups SET status='assigned',assignment_id=? WHERE job_id=? AND group_id=?",
                            (a['assignment_id'], a['job_id'], a['group_id']))
            body = json.loads(self.db.execute('SELECT body FROM jobs WHERE job_id=?',
                                             (a['job_id'],)).fetchone()[0])
            lane = ('classification' if body.get('kind') == 'classification' else
                    'extraction' if body.get('kind') in ('extraction', 'local_extraction') else 'dataset')
            policy = self._meta('policy_state', {})
            policy['lane_streak'] = (policy.get('lane_streak', 0) + 1
                                     if policy.get('last_lane') == lane else 1)
            policy['last_lane'] = lane
            if lane == 'extraction':
                policy['last_extraction_paper'] = body.get('paper_id')
            elif lane == 'classification':
                policy['current_classifier_paper'] = body.get('paper_id')
            self._set_meta('policy_state', policy)
            by_worker = self._meta('policy_state_by_worker', {})
            worker_policy = by_worker.get(a['worker_id'], {})
            worker_policy['lane_streak'] = (worker_policy.get('lane_streak', 0) + 1
                                            if worker_policy.get('last_lane') == lane else 1)
            worker_policy['last_lane'] = lane
            if lane == 'extraction':
                worker_policy['last_extraction_paper'] = body.get('paper_id')
            elif lane == 'classification':
                worker_policy['current_classifier_paper'] = body.get('paper_id')
            by_worker[a['worker_id']] = worker_policy
            self._set_meta('policy_state_by_worker', by_worker)
        elif kind == 'acknowledge':
            self.db.execute("UPDATE assignments SET status='pending_verification',raw_ref=? WHERE assignment_id=?",
                            (json.dumps(event['raw_ref']), event['assignment_id']))
            self.db.execute("UPDATE groups SET status='pending_verification' WHERE assignment_id=?",
                            (event['assignment_id'],))
        elif kind == 'closed_raw':
            self.db.execute("UPDATE assignments SET status='pending_verification',raw_ref=?,epoch=? WHERE assignment_id=?",
                            (json.dumps(event['raw_ref']), event['epoch'], event['assignment_id']))
            self.db.execute("UPDATE groups SET status='pending_verification' WHERE assignment_id=?",
                            (event['assignment_id'],))
        elif kind == 'commit':
            aid, verdict = event['assignment_id'], event['verdict']
            self.db.execute("UPDATE assignments SET status='committed',verdict=?,result_ref=? WHERE assignment_id=?",
                            (json.dumps(verdict), json.dumps(event['result_ref']), aid))
            self.db.execute('UPDATE groups SET status=?,generation_returned=?,attempt_refs=? WHERE assignment_id=?',
                            (verdict['status'], int(verdict['generation_returned']),
                             json.dumps(verdict.get('attempt_refs', [])), aid))
        elif kind == 'recover':
            aid = event['assignment_id']
            worker_id = self.db.execute('SELECT worker_id FROM assignments WHERE assignment_id=?',
                                        (aid,)).fetchone()[0]
            self.db.execute('INSERT INTO recoveries VALUES(?,?,?)', (aid, worker_id, event['at']))
            self.db.execute("UPDATE assignments SET status='recovered' WHERE assignment_id=?", (aid,))
            self.db.execute("UPDATE groups SET status='pending',assignment_id=NULL WHERE assignment_id=?", (aid,))
        elif kind == 'adopt':
            self.db.execute('UPDATE assignments SET epoch=?,incarnation=? WHERE assignment_id=?',
                            (event['epoch'], event['incarnation'], event['assignment_id']))
        elif kind == 'parent':
            self.db.execute('UPDATE jobs SET status=?,result_ref=?,index_ref=? WHERE job_id=?',
                            (event['status'], json.dumps(event['result_ref']),
                             json.dumps(event['index_ref']), event['job_id']))
        elif kind == 'blocked_dependency':
            self.db.execute("UPDATE jobs SET status='blocked_dependency' WHERE job_id=?", (event['job_id'],))
            self.db.execute("UPDATE groups SET status='blocked_dependency' WHERE job_id=? AND status='pending'",
                            (event['job_id'],))
        elif kind == 'control':
            self.db.execute('INSERT INTO controls VALUES(?,?,?,?)',
                            (event['key'], event['action'], json.dumps(event['payload']), event['at']))
        else:
            raise ValueError('unknown_journal_event:' + kind)

    def initialize(self, jobs, group_plans, imported=None):
        self._sync()
        jobs = copy.deepcopy(list(jobs))
        group_plans = copy.deepcopy(group_plans)
        imported = copy.deepcopy(imported or {})
        ids = [j['job_id'] for j in jobs]
        if len(ids) != len(set(ids)) or set(group_plans) - set(ids):
            raise ValueError('invalid_job_plan')
        for jid, groups in group_plans.items():
            if len(groups) != len(set(groups)) or not all(isinstance(g, str) and g for g in groups):
                raise ValueError('invalid_group_plan')
        for jid, value in list(imported.items()):
            if jid not in ids:
                raise ValueError('unknown_imported_job')
            if 'groups' not in value and 'parent_status' not in value:
                imported[jid] = {'groups': value}
            if set(imported[jid].get('groups', {})) - set(group_plans.get(jid, [])):
                raise ValueError('unknown_imported_group')
        plan_sha = digest({'jobs': jobs, 'group_plans': group_plans, 'imported': imported})
        prior = self._meta('plan_sha')
        if prior is not None:
            if prior != plan_sha:
                raise ValueError('ledger_plan_identity_mismatch')
            return self.snapshot()
        self._emit('initialize', jobs=jobs, group_plans=group_plans, imported=imported, plan_sha=plan_sha)
        return self.snapshot()

    def new_epoch(self):
        if self._meta('plan_sha') is None:
            raise ValueError('ledger_not_initialized')
        value = self._meta('epoch', 0) + 1
        self._emit('epoch', epoch=value)
        return value

    def register_worker(self, worker_id, incarnation, gpu_ids=None):
        if not worker_id or not incarnation:
            raise ValueError('worker_identity_required')
        gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        identity = {'incarnation': incarnation, 'gpu_ids': gpu_ids}
        if self._meta('worker:' + worker_id) != identity:
            self._emit('worker', worker_id=worker_id, incarnation=incarnation, gpu_ids=gpu_ids)

    def mark_ready(self, job_id, group_id, now=None):
        row = self._group(job_id, group_id)
        if row['status'] != 'pending':
            return False
        if row['ready_at'] is None:
            self._emit('ready', job_id=job_id, group_id=group_id, at=self._now(now))
        return True

    def _group(self, job_id, group_id):
        row = self.db.execute('SELECT * FROM groups WHERE job_id=? AND group_id=?', (job_id, group_id)).fetchone()
        if not row:
            raise KeyError((job_id, group_id))
        return dict(row)

    @staticmethod
    def _now(value):
        return time.time() if value is None else float(value)

    def claim(self, job_id, group_id, worker_id, incarnation, epoch, now=None):
        self._sync()
        identity = self._meta('worker:' + worker_id, {})
        if epoch != self._meta('epoch') or identity.get('incarnation') != incarnation:
            return None
        row = self._group(job_id, group_id)
        global_paused = worker_paused = global_draining = worker_draining = False
        for control in self.db.execute("SELECT action,payload FROM controls WHERE action IN ('pause','resume','drain') ORDER BY rowid"):
            target = json.loads(control['payload']).get('worker_id')
            if control['action'] == 'drain':
                if target is None:
                    global_draining = True
                elif target == worker_id:
                    worker_draining = True
            else:
                if target is None:
                    global_paused = control['action'] == 'pause'
                    if control['action'] == 'resume':
                        global_draining = False
                elif target == worker_id:
                    worker_paused = control['action'] == 'pause'
                    if control['action'] == 'resume':
                        worker_draining = False
        if (global_paused or worker_paused or global_draining or worker_draining
                or self.db.execute('SELECT status FROM jobs WHERE job_id=?',
                                   (job_id,)).fetchone()['status'] != 'pending'):
            return None
        reservations = self._active_reservations()
        if reservations and (identity.get('gpu_ids') is None or any(
                set(identity['gpu_ids']).intersection(r['gpu_ids']) for r in reservations)):
            return None
        if row['status'] != 'pending' or row['ready_at'] is None:
            return None
        if self.db.execute("SELECT 1 FROM assignments WHERE worker_id=? AND incarnation=? AND status='assigned'", (worker_id, incarnation)).fetchone():
            return None
        active = [g['status'] for g in self.db.execute('SELECT status FROM groups WHERE job_id=?', (job_id,))]
        if 'assigned' in active or active.count('pending_verification') >= 2:
            return None
        aid = uuid.uuid4().hex
        assignment = {'assignment_id': aid, 'job_id': job_id, 'group_id': group_id,
                      'worker_id': worker_id, 'incarnation': incarnation,
                      'epoch': epoch, 'started_at': self._now(now)}
        self._emit('claim', assignment=assignment)
        return assignment

    def _live_assignment(self, assignment_id, worker_id, incarnation, epoch, expected):
        self._sync()
        row = self.db.execute('SELECT * FROM assignments WHERE assignment_id=?', (assignment_id,)).fetchone()
        return (dict(row) if row and row['status'] == expected and row['worker_id'] == worker_id
                and row['incarnation'] == incarnation and row['epoch'] == epoch
                and epoch == self._meta('epoch')
                and self._meta('worker:' + worker_id, {}).get('incarnation') == incarnation else None)

    def acknowledge(self, assignment_id, raw_ref, worker_id, incarnation, epoch, now=None):
        if self._live_assignment(assignment_id, worker_id, incarnation, epoch, 'pending_verification'):
            row = self.db.execute('SELECT raw_ref FROM assignments WHERE assignment_id=?', (assignment_id,)).fetchone()
            return json.loads(row[0]) == raw_ref
        if not self._live_assignment(assignment_id, worker_id, incarnation, epoch, 'assigned'):
            return False
        self._emit('acknowledge', assignment_id=assignment_id, raw_ref=copy.deepcopy(raw_ref), at=self._now(now))
        return True

    def commit(self, assignment_id, verdict, result_ref, now=None):
        self._sync()
        row = self.db.execute('SELECT * FROM assignments WHERE assignment_id=?', (assignment_id,)).fetchone()
        if not row or row['status'] != 'pending_verification' or row['epoch'] != self._meta('epoch'):
            return False
        if verdict.get('status') not in TERMINAL or type(verdict.get('generation_returned')) is not bool:
            raise ValueError('invalid_group_verdict')
        self._emit('commit', assignment_id=assignment_id, verdict=copy.deepcopy(verdict),
                   result_ref=copy.deepcopy(result_ref), at=self._now(now))
        return True

    def adopt(self, assignment_id, worker_id, incarnation, epoch, evidence):
        self._sync()
        row = self.db.execute('SELECT * FROM assignments WHERE assignment_id=?', (assignment_id,)).fetchone()
        if not row or row['status'] not in ACTIVE or row['worker_id'] != worker_id or row['incarnation'] != incarnation:
            return False
        if epoch != self._meta('epoch'):
            return False
        if row['status'] == 'pending_verification':
            raw = json.loads(row['raw_ref'])
            if evidence != {'raw_ref_verified': True, 'raw_ref_sha256': digest(raw)}:
                return False
        elif evidence != {'process_incarnation_confirmed': True}:
            return False
        self._emit('adopt', assignment_id=assignment_id, incarnation=incarnation, epoch=epoch)
        return True

    def adopt_closed_raw(self, assignment_id, epoch, now=None):
        """Adopt a completed immutable receipt after its worker has exited.

        The original dispatch and receipt identities are checked from shared
        write-once files; no process liveness assertion is fabricated.
        """
        self._sync()
        a = self.db.execute('SELECT * FROM assignments WHERE assignment_id=?', (assignment_id,)).fetchone()
        if not a or a['status'] != 'assigned' or epoch != self._meta('epoch'):
            return False
        dispatch_path = self.shared_root / 'dispatches' / (assignment_id + '.json')
        raw_path = (self.shared_root / 'worker_receipts' / a['worker_id'] /
                    a['incarnation'] / ('result-' + assignment_id + '.json'))
        if not dispatch_path.is_file() or not raw_path.is_file() or dispatch_path.is_symlink() or raw_path.is_symlink():
            return False
        dispatch = read(dispatch_path)
        receipt = read(raw_path)
        expected = {key: a[key] for key in ('assignment_id', 'job_id', 'group_id',
                                           'worker_id', 'incarnation', 'epoch')}
        expected['started_at'] = a['started_at']
        if any(dispatch.get(k) != value or receipt.get(k) != value for k, value in expected.items()):
            raise ValueError('closed_raw_dispatch_identity_mismatch')
        if dispatch.get('condition') != receipt.get('condition'):
            raise ValueError('closed_raw_condition_mismatch')
        if receipt.get('status') != 'returned' or not isinstance(receipt.get('outcome'), dict):
            return False
        raw_ref = {'path': raw_path.relative_to(self.shared_root).as_posix(),
                   'file_bytes_sha256': file_sha(raw_path)}
        self._emit('closed_raw', assignment_id=assignment_id, epoch=epoch,
                   dispatch_sha256=file_sha(dispatch_path), raw_ref=raw_ref, at=self._now(now))
        return raw_ref

    def recover(self, now=None, evidence=None):
        """Release proven abandoned assignments, at most two per worker-hour."""
        self._sync()
        at = self._now(now)
        evidence = evidence or {}
        recovered = []
        # Acknowledged raw output is awaiting verification, never regeneration.
        rows = self.db.execute("SELECT * FROM assignments WHERE status='assigned' ORDER BY started_at,assignment_id").fetchall()
        for row in rows:
            proof = evidence.get(row['assignment_id'])
            if proof != {'owner_released': True, 'incarnation': row['incarnation']}:
                continue
            used = self.db.execute('SELECT COUNT(*) FROM recoveries WHERE worker_id=? AND at>? AND at<=?',
                                   (row['worker_id'], at - 3600, at)).fetchone()[0]
            if used >= 2:
                continue
            self._emit('recover', assignment_id=row['assignment_id'], at=at, evidence=proof)
            recovered.append(row['assignment_id'])
        return recovered

    def set_parent(self, job_id, status, result_ref, index_ref=None, now=None):
        self._sync()
        row = self.db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        if status not in TERMINAL:
            raise ValueError('parent_status_not_terminal')
        current = dict(row)
        if current['status'] in TERMINAL:
            return (current['status'] == status and json.loads(current['result_ref']) == result_ref
                    and json.loads(current['index_ref']) == index_ref)
        if self.db.execute('SELECT 1 FROM groups WHERE job_id=? AND status NOT IN (' + ','.join('?' * len(TERMINAL)) + ')',
                           (job_id, *TERMINAL)).fetchone():
            return False
        self._emit('parent', job_id=job_id, status=status, result_ref=copy.deepcopy(result_ref),
                   index_ref=copy.deepcopy(index_ref), at=self._now(now))
        return True

    def block_dependency(self, job_id, dependency_id, now=None):
        self._sync()
        job = self.db.execute('SELECT status FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        dependency = self.db.execute('SELECT status,index_ref FROM jobs WHERE job_id=?', (dependency_id,)).fetchone()
        if not job or not dependency:
            raise KeyError((job_id, dependency_id))
        if job['status'] != 'pending':
            return False
        if dependency['status'] not in TERMINAL or dependency['status'] == 'success' or json.loads(dependency['index_ref']):
            return False
        if self.db.execute("SELECT 1 FROM groups WHERE job_id=? AND status IN ('assigned','pending_verification')",
                           (job_id,)).fetchone():
            return False
        self._emit('blocked_dependency', job_id=job_id, dependency_id=dependency_id, at=self._now(now))
        return True

    def control(self, action, key, payload=None, now=None):
        self._sync()
        if action not in {'pause', 'resume', 'drain', 'reserve', 'cancel_reservation'} or not key:
            raise ValueError('invalid_control')
        at = self._now(now)
        payload = copy.deepcopy(payload or {})
        prior = self.db.execute('SELECT action,payload FROM controls WHERE key=?', (key,)).fetchone()
        if prior:
            if prior['action'] != action or json.loads(prior['payload']) != payload:
                raise ValueError('control_idempotency_conflict')
            return False
        if action in {'pause', 'drain'}:
            deadline_values = [name for name in ('deadline_at', 'deadline', 'deadline_s') if name in payload]
            if len(deadline_values) > 1:
                raise ValueError('maintenance_deadline_ambiguous')
            if deadline_values:
                name = deadline_values[0]
                value = payload[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('maintenance_deadline_invalid')
                if name == 'deadline_s' and value <= 0 or name != 'deadline_s' and value <= at:
                    raise ValueError('maintenance_deadline_invalid')
        if action == 'reserve':
            if not isinstance(payload.get('gpu_ids'), list) or len(payload['gpu_ids']) != 1:
                raise ValueError('one_gpu_reservation_required')
            if not at < payload.get('expires_at', 0) <= at + 10800:
                raise ValueError('reservation_max_three_hours')
            if not at <= payload.get('grace_until', at) <= at + 900:
                raise ValueError('reservation_grace_max_fifteen_minutes')
            if self._active_reservations():
                raise ValueError('active_reservation_exists')
        elif action == 'cancel_reservation':
            if payload.get('resource_release_ack') is not True or not payload.get('reservation_key'):
                raise ValueError('reservation_release_ack_required')
        self._emit('control', action=action, key=key, payload=payload, at=at)
        return True

    def _active_reservations(self):
        active = {}
        for row in self.db.execute("SELECT key,action,payload FROM controls WHERE action IN ('reserve','cancel_reservation') ORDER BY rowid"):
            payload = json.loads(row['payload'])
            if row['action'] == 'reserve':
                active[row['key']] = payload
            else:
                active.pop(payload['reservation_key'], None)
        return list(active.values())

    def snapshot(self):
        self._sync()
        jobs = []
        for row in self.db.execute('SELECT * FROM jobs ORDER BY ordinal'):
            job = dict(row)
            job['job'] = json.loads(job.pop('body'))
            job['result_ref'] = json.loads(job['result_ref'])
            job['index_ref'] = json.loads(job['index_ref'])
            job['groups'] = []
            for g in self.db.execute('SELECT * FROM groups WHERE job_id=? ORDER BY ordinal', (job['job_id'],)):
                group = dict(g)
                group['attempt_refs'] = json.loads(group['attempt_refs'])
                if group['generation_returned'] is not None:
                    group['generation_returned'] = bool(group['generation_returned'])
                job['groups'].append(group)
            jobs.append(job)
        assignments = []
        for a in self.db.execute('SELECT * FROM assignments ORDER BY started_at,assignment_id'):
            item = dict(a)
            for key in ('raw_ref', 'result_ref', 'verdict'):
                item[key] = json.loads(item[key]) if item[key] is not None else None
            assignments.append(item)
        controls = {'paused': False, 'draining': False, 'paused_workers': [],
                    'draining_workers': [], 'reservations': [], 'maintenance': []}
        for c in self.db.execute('SELECT * FROM controls ORDER BY rowid'):
            payload = json.loads(c['payload'])
            target = payload.get('worker_id')
            if c['action'] in ('pause', 'resume'):
                if target is None:
                    controls['paused'] = c['action'] == 'pause'
                    if c['action'] == 'resume':
                        controls['draining'] = False
                elif c['action'] == 'pause' and target not in controls['paused_workers']:
                    controls['paused_workers'].append(target)
                elif c['action'] == 'resume':
                    controls['paused_workers'] = [w for w in controls['paused_workers'] if w != target]
                    controls['draining_workers'] = [w for w in controls['draining_workers'] if w != target]
                if c['action'] == 'resume':
                    controls['maintenance'] = [m for m in controls['maintenance'] if m['worker_id'] != target]
            elif c['action'] == 'drain':
                if target is None:
                    controls['draining'] = True
                elif target not in controls['draining_workers']:
                    controls['draining_workers'].append(target)
            if c['action'] in ('pause', 'drain'):
                deadline = (payload['deadline_at'] if 'deadline_at' in payload else
                            payload['deadline'] if 'deadline' in payload else
                            c['at'] + payload.get('deadline_s', 900))
                controls['maintenance'] = [m for m in controls['maintenance'] if
                                           not (m['worker_id'] == target and m['action'] == c['action'])]
                controls['maintenance'].append({'key': c['key'], 'action': c['action'],
                                                'worker_id': target, 'issued_at': c['at'],
                                                'deadline': deadline})
            elif c['action'] == 'reserve':
                controls['reservations'].append({'key': c['key'], **payload})
            elif c['action'] == 'cancel_reservation':
                controls['reservations'] = [r for r in controls['reservations'] if r['key'] != payload.get('reservation_key')]
        return {'epoch': self._meta('epoch', 0), 'jobs': jobs,
                'assignments': assignments, 'controls': controls,
                'policy_state': self._meta('policy_state', {}),
                'policy_state_by_worker': self._meta('policy_state_by_worker', {})}
