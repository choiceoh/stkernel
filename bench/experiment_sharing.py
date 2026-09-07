# SPDX-License-Identifier: Apache-2.0
"""Serialize identical in-flight CPU evidence, including across commit SHAs."""
import fcntl
from pathlib import Path
import time


def cpu_claim(store, job, payload):
    from experiments import RetiredJob
    directory = store.root / 'cpu-claims'
    directory.mkdir(exist_ok=True)
    stream = (directory / (payload['cpu_identity']['key'] + '.lock')).open('a+')
    deadline = time.monotonic() + payload['spec']['timeout_s']
    joined = None
    try:
        while True:
            if store.get(job)['state'] == 'retired':
                raise RetiredJob(job)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stream.seek(0)
                owner = stream.read().strip()
                if owner and owner != joined:
                    # Internal dependency keeps an in-flight owner alive when
                    # its original subscriber replaces their own request.
                    with store.db:
                        store.db.execute('INSERT OR IGNORE INTO dependencies VALUES(?,?,?)',
                                         (job, owner, 'cpu-shared'))
                    joined = owner
                    store.state(job, 'waiting_cpu_evidence', {'cpu_owner': owner})
                if time.monotonic() >= deadline:
                    raise ValueError('time budget exceeded waiting for identical CPU evidence')
                time.sleep(.1)
        stream.seek(0)
        stream.truncate()
        stream.write(job)
        stream.flush()
        return stream, joined
    except BaseException:
        stream.close()
        raise


def joined_failure(store, joined, payload):
    from experiments import TERMINAL
    if not joined:
        return None
    row = store.get(joined)
    if (row['payload'].get('cpu_identity') == payload.get('cpu_identity')
            and row['state'] in TERMINAL and row['state'] != 'succeeded'):
        return dict(evidence='cpu-only', reason='the shared in-flight CPU check did not pass',
                    cpu_owner=joined, tested_revision=row['payload']['spec']['revision'],
                    source_state=row['state'], source_result=row['result'])
    return None
