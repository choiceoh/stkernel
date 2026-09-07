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
import time
from concurrent.futures import ThreadPoolExecutor


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


def memory():
    path = Path('/proc/meminfo')
    if path.exists():
        rows = {k: int(v.split()[0]) // 1024 for k, v in (l.split(':', 1) for l in path.read_text().splitlines())}
        return rows['MemTotal'], rows['MemAvailable']
    total = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)) // 1048576
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


def group_active(pgid):
    output = subprocess.check_output(['ps', '-eo', 'pgid=,stat='], text=True)
    return any(int(fields[0]) == pgid and not fields[1].startswith('Z')
               for line in output.splitlines() if len(fields := line.split()) == 2)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return group_active(pid)
    except PermissionError:
        return True


def acquire(store, job, resources):
    policy_path = Path(store.get(job)['payload']['paths']['FLEET_DIR']) / 'cpu-policy.json'
    total, available = memory()
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
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        for row in store.db.execute('SELECT job,pid FROM cpu_leases').fetchall():
            if not alive(row['pid']):
                store.db.execute('DELETE FROM cpu_leases WHERE job=?', (row['job'],))
        slots, ram = store.db.execute('SELECT COALESCE(sum(slots),0),COALESCE(sum(memory_mb),0) FROM cpu_leases').fetchone()
        if (slots + resources['cpu_slots'] > policy['slots'] or ram + resources['cpu_memory_mb'] > policy['memory_mb']
                or available < resources['cpu_memory_mb'] + policy['reserve_mb']):
            return False
        store.db.execute('INSERT OR REPLACE INTO cpu_leases VALUES(?,?,?,?)',
                         (job, os.getpid(), resources['cpu_slots'], resources['cpu_memory_mb']))
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
    store.state(job, 'waiting_cpu')
    while not acquire(store, job, resources):
        if store.get(job)['state'] == 'retired':
            from experiments import RetiredJob
            raise RetiredJob(job)
        if time.monotonic() >= deadline:
            return 124, 'CPU queue time budget exceeded'
        time.sleep(.25)
    proc = None
    try:
        store.state(job, 'running')
        proc = subprocess.Popen(command, cwd=payload['repo'], start_new_session=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        with store.db:
            store.db.execute('UPDATE cpu_leases SET pid=? WHERE job=?', (proc.pid, job))
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                stop(proc)
                return 124, 'CPU time budget exceeded (including resource wait)'
            text = subprocess.check_output(['ps', '-eo', 'pgid=,rss='], text=True)
            rss = sum(int(fields[1]) for line in text.splitlines()
                      if len(fields := line.split()) == 2 and int(fields[0]) == proc.pid)
            if rss > resources['cpu_memory_mb'] * 1024:
                stop(proc)
                return 137, 'CPU process group exceeded its declared RAM budget'
            time.sleep(.1)
        return proc.returncode, None
    finally:
        if proc:
            stop(proc)
        with store.db:
            store.db.execute('DELETE FROM cpu_leases WHERE job=?', (job,))
