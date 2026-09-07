# SPDX-License-Identifier: Apache-2.0
"""Bound CPU processes and check readiness without reserving a GPU."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


def normalize(raw=None):
    raw = {} if raw is None else raw
    defaults = dict(cpu_memory_mb=4096, cpu_slots=1, disk_mb=0, node_memory_mb=0,
                    nodes=[], disk_path='/')
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError('unknown resources fields')
    value = dict(defaults, **raw)
    for k in ('cpu_memory_mb', 'cpu_slots', 'disk_mb', 'node_memory_mb'):
        if type(value[k]) is not int or not 0 <= value[k] <= 1048576:
            raise ValueError('invalid resource budget: ' + k)
    if value['cpu_memory_mb'] < 16 or not 1 <= value['cpu_slots'] <= 32:
        raise ValueError('CPU requires >=16 MiB and 1..32 slots')
    if (not isinstance(value['nodes'], list) or len(value['nodes']) > 8 or
            any(not isinstance(n, str) or not re.fullmatch(r'(?:[\w.-]+@)?[\w.-]+', n) for n in value['nodes'])):
        raise ValueError('invalid readiness nodes')
    if not isinstance(value['disk_path'], str) or not value['disk_path'].startswith('/') or '\0' in value['disk_path']:
        raise ValueError('disk_path must be absolute')
    return value


@lru_cache(maxsize=1)
def total_memory():
    path = Path('/proc/meminfo')
    if path.exists():
        return int(next(line for line in path.read_text().splitlines() if line.startswith('MemTotal:')).split()[1]) // 1024
    return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)) // 1048576


def memory():
    path = Path('/proc/meminfo')
    if path.exists():
        rows = {k: int(v.split()[0]) // 1024 for k, v in (l.split(':', 1) for l in path.read_text().splitlines())}
        return rows['MemTotal'], rows['MemAvailable']
    total = total_memory()
    output = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+)', output)[1])
    pages = sum(int(re.search(r'^' + name + r':\s+(\d+)', output, re.M)[1])
                for name in ('Pages free', 'Pages inactive', 'Pages speculative'))
    return total, pages * page // 1048576


def readiness(resources):
    if not resources['disk_mb'] and not resources['node_memory_mb']:
        return []
    def check(node):
        if node == 'local':
            available = memory()[1]
            disk = shutil.disk_usage(resources['disk_path']).free // 1048576
        else:
            script = "import json,shutil; from pathlib import Path; " + \
                "m={k:int(v.split()[0])//1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}; " + \
                "print(json.dumps([m['MemAvailable'],shutil.disk_usage(" + repr(resources['disk_path']) + ").free//1048576]))"
            available, disk = json.loads(subprocess.check_output(
                ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', node,
                 'python3 -c ' + shlex.quote(script)], text=True, timeout=12))
        row = dict(node=node, available_mb=available, disk_free_mb=disk, at=time.time())
        if available < resources['node_memory_mb'] or disk < resources['disk_mb']:
            raise ValueError('readiness refused: ' + json.dumps(row))
        return row
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(check, resources['nodes'] or ['local']))


def group_snapshot(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return []
    except PermissionError:
        pass  # A denied signal probe does not prevent read-only ps inspection.
    # Children start in a new session whose SID == PGID. Linux ps selects that
    # session; BSD/macOS ps selects the process group. Filter PGID in both cases.
    selector = '-s' if sys.platform.startswith('linux') else '-g'
    process = subprocess.run(['ps', selector, str(pgid), '-o', 'pgid=,stat=,rss='],
                             text=True, capture_output=True, timeout=5)
    if process.returncode and (process.returncode != 1 or process.stderr.strip()):
        raise subprocess.CalledProcessError(process.returncode, process.args, process.stdout, process.stderr)
    return [(fields[1],int(fields[2])) for line in process.stdout.splitlines()
            if len(fields := line.split()) == 3 and int(fields[0]) == pgid]


def group_active(pgid):
    return any(not state.startswith('Z') for state,_ in group_snapshot(pgid))


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return group_active(pid)
    except PermissionError:
        return True


def acquire(store, job, resources):
    from experiments import TERMINAL
    policy_path = Path(store.get(job)['payload']['paths']['FLEET_DIR']) / 'cpu-policy.json'
    total = total_memory()
    policy = dict(slots=2, memory_mb=min(16384, total // 2), reserve_mb=min(2048, total // 8))
    if policy_path.exists():
        custom = json.loads(policy_path.read_text())
        if not isinstance(custom, dict) or set(custom) - set(policy):
            raise ValueError('invalid cpu-policy.json')
        policy.update(custom)
    if any(type(v) is not int or v < 0 for v in policy.values()) or not policy['slots']:
        raise ValueError('invalid CPU pool policy')
    if resources['cpu_slots'] > policy['slots'] or resources['cpu_memory_mb'] > policy['memory_mb']:
        raise ValueError('CPU request exceeds the configured pool capacity')
    with store.transaction():
        for row in store.db.execute('SELECT job,pid FROM cpu_leases').fetchall():
            if not alive(row['pid']):
                store.db.execute('DELETE FROM cpu_leases WHERE job=?', (row['job'],))
        for row in store.db.execute('SELECT w.*,j.state FROM cpu_waiters w JOIN jobs j ON j.id=w.job').fetchall():
            if (row['state'] in TERMINAL or not alive(row['pid']) or row['slots'] > policy['slots']
                    or row['memory_mb'] > policy['memory_mb']):
                store.db.execute('DELETE FROM cpu_waiters WHERE job=?', (row['job'],))
        if store.get(job)['state'] in TERMINAL:
            return False
        lease = store.db.execute('SELECT pid FROM cpu_leases WHERE job=?', (job,)).fetchone()
        if lease:
            return lease['pid'] == os.getpid()
        store.db.execute('INSERT INTO cpu_waiters(job,pid,slots,memory_mb) VALUES(?,?,?,?) '
                         'ON CONFLICT(job) DO UPDATE SET pid=excluded.pid',
                         (job, os.getpid(), resources['cpu_slots'], resources['cpu_memory_mb']))
        if store.db.execute('SELECT job FROM cpu_waiters ORDER BY ticket LIMIT 1').fetchone()['job'] != job:
            return False
        slots, ram = store.db.execute('SELECT COALESCE(sum(slots),0),COALESCE(sum(memory_mb),0) FROM cpu_leases').fetchone()
        if slots + resources['cpu_slots'] > policy['slots'] or ram + resources['cpu_memory_mb'] > policy['memory_mb']:
            return False
        # Only the head waiter with room in the pool probes current host RAM.
        if memory()[1] < resources['cpu_memory_mb'] + policy['reserve_mb']:
            return False
        store.db.execute('INSERT OR REPLACE INTO cpu_leases VALUES(?,?,?,?)',
                         (job, os.getpid(), resources['cpu_slots'], resources['cpu_memory_mb']))
        store.db.execute('DELETE FROM cpu_waiters WHERE job=?', (job,))
        return True


def stop(proc):
    # A completed leader can leave running grandchildren in its own session.
    # Always clean that process group before releasing its RAM reservation.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if group_active(proc.pid):
            raise
        proc.wait()
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        proc.poll()
        if not group_active(proc.pid):
            break
        time.sleep(.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if group_active(proc.pid):
            raise
    proc.wait()


def run_cpu(store, job, command, payload):
    resources = payload['spec']['resources']
    deadline = time.monotonic() + payload['spec']['timeout_s']
    proc = None
    acquired = False
    try:
        store.state(job, 'waiting_cpu')
        while not acquire(store, job, resources):
            if store.get(job)['state'] == 'retired':
                from experiments import RetiredJob
                raise RetiredJob(job)
            if time.monotonic() >= deadline:
                return 124, 'CPU queue time budget exceeded'
            time.sleep(.05)
        acquired = True
        store.state(job, 'running')
        proc = subprocess.Popen(command, cwd=payload['repo'], start_new_session=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        with store.db:
            store.db.execute('UPDATE cpu_leases SET pid=? WHERE job=?', (proc.pid, job))
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                stop(proc)
                return 124, 'CPU time budget exceeded (including resource wait)'
            rss = sum(rss for _,rss in group_snapshot(proc.pid))
            if rss > resources['cpu_memory_mb'] * 1024:
                stop(proc)
                return 137, 'CPU process group exceeded its declared RAM budget'
            try:
                proc.wait(timeout=max(.001,min(.1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        return proc.returncode, None
    finally:
        if proc:
            stop(proc)
        with store.db:
            store.db.execute('DELETE FROM cpu_waiters WHERE job=?', (job,))
            if acquired:
                store.db.execute('DELETE FROM cpu_leases WHERE job=?', (job,))
