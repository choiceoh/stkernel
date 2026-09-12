"""Run the frozen native ST B/A/B bracket on srv2 through the official lease.

No measurement is sent to a different owner's endpoint. Each boot gets two
unchanged onepass invocations and a fresh copy of the same cache seed.
"""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from engine.base import fleet_lease

ROOT = Path(os.environ.get('ST_DECODE22_EVIDENCE_ROOT', '/home/choiceoh/glm53-logs/st-decode22-consumer-v2'))
SOURCE = Path('/home/choiceoh/st-decode22')
SEED = Path('/home/choiceoh/glm53-cache-decode22-seed-e12cb4b5')
HOSTS = (None, 'choiceoh@10.10.10.1', 'choiceoh@10.10.10.3', 'choiceoh@10.10.10.4')
PORT = 18122
BASE_URL = f'http://127.0.0.1:{PORT}'
BASELINE = 'baseline-' + os.environ['ST_DECODE22_BASELINE']
CANDIDATE = 'candidate-' + os.environ['ST_DECODE22_CANDIDATE']
ARMS = (('B1', BASELINE), ('A', CANDIDATE), ('B2', BASELINE))


def node(host, argv, **kwargs):
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, shlex.join(argv)] if host else argv
    return subprocess.run(command, check=True, text=True, **kwargs)


def event(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def get(path, body=None, timeout=15):
    request = urllib.request.Request(BASE_URL + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def held_by(owner):
    return (fleet_lease.read() or {}).get('owner') == owner


def wait_free(owner):
    deadline, previous = time.monotonic() + 3 * 3600, None
    asked = set()
    # The existing prefill campaign's user-authorized window was recorded in
    # its lease. Carry the deadline across the owner's intervening retries.
    not_before = float(os.environ.get('ST_DECODE22_NOT_BEFORE', '0'))
    while time.monotonic() < deadline:
        lease = fleet_lease.read()
        holder = Path('/home/choiceoh/glm53-logs/fleet/holder')
        queued = holder.read_text().strip() if holder.exists() else ''
        current = ((lease or {}).get('owner'), queued)
        if current != previous:
            event(f'{owner}: waiting for {current}')
            previous = current
        if lease is None and not queued and time.time() >= not_before:
            return
        if lease:
            # Respect an already authorized measurement window and an earlier
            # handoff request. Do not overwrite someone else's place in line.
            priority = float((lease.get('state') or {}).get('priority_until') or 0)
            key = (lease.get('owner'), lease.get('since'))
            if time.time() >= max(priority, not_before) and key not in asked and not lease.get('yield_to'):
                fleet_lease.request_yield(owner, reason='ST decode B/A/B canonical onepass; GPU numerical gates passed')
                asked.add(key)
                event(f'{owner}: requested normal yield from {key[0]}')
        time.sleep(10)
    raise TimeoutError('fleet remained occupied for three hours')


def prepare_cache(arm):
    cache = Path(f'/home/choiceoh/glm53-cache-decode22-{arm}')
    for host in HOSTS:
        script = 'import pathlib,subprocess,sys; src,dst=map(pathlib.Path,sys.argv[1:]); assert src.is_dir(); ' \
                 'subprocess.run(["cp","-a","--reflink=auto",str(src),str(dst)],check=True) if not dst.exists() else None'
        node(host, ['python3', '-c', script, str(SEED), str(cache)], timeout=180)
    return cache


def save_logs(directory):
    for rank, host in enumerate(HOSTS):
        with (directory / f'rank{rank}.log').open('w') as stream:
            node(host, ['docker', 'logs', '--timestamps', 'st-glm53'], stdout=stream, stderr=subprocess.STDOUT, timeout=90)
        with (directory / f'rank{rank}-inspect.json').open('w') as stream:
            node(host, ['docker', 'inspect', 'st-glm53'], stdout=stream, timeout=20)


def run_arm(arm, source_name):
    owner = 'codex/st-decode22-' + arm
    source = SOURCE / source_name
    directory = ROOT / arm
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'launch.log').exists(), 'arm already attempted; preserve it before making a new attempt'
    assert source.is_dir()
    (directory / 'source-commit.txt').write_text((source / 'source-commit.txt').read_text())
    wait_free(owner)
    cache = prepare_cache(arm)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    env = {key: value for key, value in os.environ.items() if not key.startswith(('STK_', 'FLEET_WORKLOAD'))}
    env.update(ST_IMAGE='st-engine:decode22-' + arm.lower(), PORT=str(PORT), ST_PRODUCTION='1', ST_KV_GIB='7',
               RANKS_DIR='/home/choiceoh/models/st-glm53-nvidia-tp4-9391', CKPT=str(source / 'st-glm53-meta'),
               DRAFTER='/home/choiceoh/models/GLM-5.3-Flash-DFlash2', ST_ENGINE_DIR='/home/choiceoh/st-releases/decode22-' + arm,
               CACHE_DIR=str(cache), ST_TIER_DIR=str(directory / 'tier'), ST_DUMP_DIR=str(directory / 'dumps'),
               LEASE_OWNER=owner, LEASE_MINUTES='25', LEASE_NOTE='ST decode22 ' + arm + '; two frozen canonical onepass runs')
    launcher = ['bash', str(source / 'launchers/start-st-glm53.sh')]
    event(arm + ': launching four ranks')
    try:
        with (directory / 'launch.log').open('x') as stream:
            subprocess.run(launcher, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if not held_by(owner):
                raise RuntimeError('lease lost during boot')
            try:
                card = get('/v1/models')['data'][0]
                break
            except (OSError, ValueError, KeyError):
                time.sleep(10)
        else:
            raise TimeoutError('ST did not become ready within twenty minutes')
        env.update(GLM53_API_PORT=str(PORT), BENCH_MODEL=card['id'])
        for number in (1, 2):
            if not held_by(owner):
                raise RuntimeError('lease lost before onepass')
            result_dir = directory / f'pass{number}'
            result_dir.mkdir()
            env['EVIDENCE_DIR'] = str(result_dir)
            (result_dir / 'prefix-reset.json').write_text(json.dumps(get('/v1/prefix/reset', {})) + '\n')
            command = [sys.executable, str(Path(__file__).with_name('run_onepass.py')), '--name', f'decode22-{arm}-{number}',
                       '--ctx', '2000,32000,128000', '--max-tokens', '400', '--num-spec', '6', '--seed', '7',
                       '--require-exclusive', '--fixed-decode-tokens', '1024', '--fixed-decode-reps', '3',
                       '--out', str(result_dir / 'raw.jsonl')]
            event(f'{arm}: canonical onepass {number}/2')
            with (result_dir / 'console.log').open('x') as stream:
                rc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=2400).returncode
            (result_dir / 'exit-code.txt').write_text(str(rc) + '\n')
            record = json.loads((result_dir / 'raw.jsonl').read_text().splitlines()[-1])
            event(f'{arm} pass{number}: rc={rc} quality={record.get("quality")} ' +
                  f'fixed_step_s={record.get("decode", {}).get("fixed_pooled_step_s")}')
            if rc or record.get('evidence_issues'):
                raise RuntimeError('onepass measurement failed; artifacts retained')
            if record['quality']['ok'] != record['quality']['total'] or record['korean']['dirty']:
                raise RuntimeError('quality gate failed; artifacts retained')
        (directory / 'complete.json').write_text(json.dumps({'arm': arm, 'owner': owner, 'completed_at': time.time()}) + '\n')
    finally:
        if held_by(owner):
            try:
                save_logs(directory)
            finally:
                if held_by(owner):
                    event(arm + ': official stop and lease release')
                    subprocess.run(launcher + ['stop'], env=env, check=True, timeout=180)
        else:
            event(arm + ': lease belongs elsewhere; no stop issued')


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, source_name in ARMS:
        run_arm(name, source_name)
    event('B/A/B complete; retained identity, workload, quality, counters, and all-rank logs')
