# SPDX-License-Identifier: Apache-2.0
"""Explicit retries preserve successful evidence and keep unsuccessful attempts."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import time


RETRYABLE = {'failed', 'blocked', 'interrupted'}


def controller_path(payload):
    """Check the pinned management scripts without changing the measured source."""
    from experiments import digest
    controller = payload['retry_controller']
    directory = Path(controller['repo'])
    files = {p.name: digest(p) for p in sorted((directory / 'bench').iterdir())
             if p.suffix in ('.py', '.sh') and p.is_file()}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if identity != controller['sha256'] or digest(directory / 'bench/experiments.py') != payload['snapshot']['runner']:
        raise ValueError('pinned retry controller changed; retry preparation must be repeated')
    return directory


def source(store, session, job):
    row = store.get(job)
    if not store.db.execute('SELECT 1 FROM subscribers s WHERE s.job=? AND s.session=? AND NOT EXISTS '
                            '(SELECT 1 FROM withdrawals w WHERE w.job=s.job AND w.session=s.session)',
                            (job, session)).fetchone():
        raise ValueError('retry requires your current subscription; withdrawn requests cannot be retried')
    if row['state'] not in RETRYABLE:
        raise ValueError('retry requires failed, blocked, or interrupted state; '
                         'incomplete evidence needs result refresh or an explicit additional sample')
    if row['payload']['spec']['kind'] not in {'cpu', 'pair', 'probe'}:
        raise ValueError('retry the consuming experiment, not its internal baseline reservation')
    from experiments import TERMINAL
    for dependency in row['payload']['spec']['depends_on']:
        state = store.get(dependency)['state']
        if state in TERMINAL and state != 'succeeded':
            raise ValueError('prerequisite ' + dependency + ' is ' + state +
                             '; resolve it and submit the updated dependency before retrying')
    return row


def retry(store, session, job, reason, repo=None, *, launch=True):
    """Publish one freshly attested attempt; reuse compatible in-flight or valid work.

    This is separate from an independent --repeat sample: repeat_reason stays
    empty, so CPU cache, shared CPU claims and shared baselines remain enabled.
    No failed prerequisite is rewritten or implicitly relaunched.
    """
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', session) or not isinstance(reason, str) or not reason.strip():
        raise ValueError('retry needs a valid session and a nonempty reason')
    reason = reason.strip()
    original = source(store, session, job)
    from experiments import normalize, git, worker_lock, ensure_worker, HERE
    from experiment_submission import Context, reuse
    revision = original['payload']['spec']['revision']
    # Most jobs already own a detached checkout. Keep that exact source even
    # after an agent has advanced its original working directory.
    candidates = [Path(original['payload']['repo'])]
    if repo is not None and Path(repo) not in candidates:
        candidates.append(Path(repo))
    checkout = None
    for candidate in candidates:
        try:
            if candidate.is_dir() and git(candidate, 'rev-parse', 'HEAD') == revision:
                checkout = candidate
                break
        except (OSError, subprocess.SubprocessError):
            continue
    if checkout is None:
        raise ValueError('saved revision checkout is unavailable; restore ' + revision + ' before retrying')
    saved = dict(original['payload']['spec'])
    if saved['kind'] == 'pair' and 'baseline_policy' not in saved:
        saved['baseline_policy'] = 'confirm'  # Old manifests required three samples.
    spec = normalize(saved, checkout)
    payload = Context(checkout).payload(spec)
    if Context(checkout).payload(spec) != payload:
        raise ValueError('retry source, deployment or CPU environment changed during preparation')
    # Historical source checkouts contain historical management code. Run the
    # freshly attested controller, while REPO/FLEET/LEVER still identify the
    # saved source and its reviewed workload/boot scripts.
    from fleet_pin import pin
    controller = pin(HERE.parent, store.root)
    payload['retry_controller'] = dict(repo=str(controller), sha256=controller.name)
    controller_path(payload)
    with store.transaction():
        # Subscription withdrawal or state changes during attestation cannot
        # create a replacement on behalf of a consumer that no longer wants it.
        current = source(store, session, job)
        if current['payload'] != original['payload']:
            raise ValueError('source request changed during retry preparation')
        answer = store.submit(session, payload, retry_failed=True)
        store.db.execute('INSERT OR IGNORE INTO retry_attempts VALUES(?,?,?,?,?)',
                         (job, answer['id'], session, reason, time.time()))
        store.event(job, 'retry_requested', dict(session=session, attempt=answer['id'], reason=reason))
        store.event(answer['id'], 'retry_of', dict(session=session, source=job, reason=reason))
    lock = worker_lock(store, answer['id'])
    if lock is not None:
        with lock, store.transaction():
            if reuse(store, answer['id'], payload):
                answer.update(state='succeeded', cache_hit=True)
    answer.update(retry_of=job, reason=reason)
    if launch:
        ensure_worker(store, answer['id'])
    return answer
