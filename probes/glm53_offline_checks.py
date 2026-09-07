#!/usr/bin/env python3
"""Run pinned probes during an owned boot turn, restoring stopped containers.

No deploy, image replacement, container deletion, or weaker memory guard.
An entirely absent incoming fleet is supported; the standard public restore
is then required unless another boot is queued. Partial fleets are refused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.request

NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
PINS = ()  # populated only by the fixed MoE-M64 CLI plan below

# Full environment values are never logged. Configuration hashes cover them.
INSPECT = r'''
import base64,hashlib,json,pathlib,re,subprocess
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def inspect(name):
    names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
    if name not in names:return None
    c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    overlays={m['Destination']:hashlib.sha256(pathlib.Path(m['Source']).read_bytes()).hexdigest()
        for m in c['Mounts'] if m['Source'].startswith('/home/choiceoh/overlays/glm53/')}
    manifest=pathlib.Path('/home/choiceoh/overlays/glm53/manifest.tsv')
    cmd=' '.join(c['Config'].get('Cmd') or [])
    match=re.search(r'echo ([A-Za-z0-9+/=]+) \| base64 -d',cmd)
    if match:cmd=base64.b64decode(match.group(1),validate=True).decode()
    port=re.search(r'--port\s+(\d+)',cmd)
    return dict(id=c['Id'],image=c['Image'],running=c['State']['Running'],
        started=c['State']['StartedAt'],auto_remove=c['HostConfig']['AutoRemove'],
        config=digest(c['Config']),host_config=digest(c['HostConfig']),
        mounts=digest(sorted(c['Mounts'],key=lambda m:m['Destination'])),overlays=overlays,
        manifest=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
        port=int(port.group(1)) if port else None)
'''


def remote(node, code, timeout=45):
    cmd = ['python3', '-c', code] if node == 'local' else [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
        'choiceoh@' + node, 'python3 -c ' + shlex.quote(code)]
    try:
        return json.loads(subprocess.check_output(cmd, text=True, timeout=timeout,stderr=subprocess.PIPE))
    except subprocess.CalledProcessError as exc:
        # The command contains an entire source/config-hash payload. Printing
        # its repr recursively obscures the actual remote failure.
        raise RuntimeError(f'{node}: remote exit {exc.returncode}: {(exc.stderr or "")[-1600:]}') from None
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{node}: remote timeout after {timeout}s') from None


def name(node):
    return 'glm53' if node == 'local' else 'glm53-worker'


def snapshot():
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = pool.map(lambda n: remote(n, INSPECT + '\nprint(json.dumps(inspect(' + repr(name(n)) + ')))'), NODES)
        return dict(zip(NODES, values))


def check_holder():
    session = os.environ.get('FLEET_SESSION')
    if not session or not re.fullmatch(r'[A-Za-z0-9_-]+', session):
        raise RuntimeError('owned FLEET_SESSION required')
    holder = Path(os.environ.get('FLEET_DIR', '/home/choiceoh/glm53-logs/fleet')) / 'holder'
    fields = holder.read_text().strip().split('|')
    if len(fields) != 7 or fields[0] != session or fields[2] != socket.gethostname().split('.')[0] or fields[6] != 'boot':
        raise RuntimeError('current normal boot hold is not ours')
    ancestor = os.getpid()
    while ancestor > 1:
        if ancestor == int(fields[1]):
            return
        status = Path(f'/proc/{ancestor}/status').read_text()
        ancestor = int(re.search(r'^PPid:\s+(\d+)', status, re.M).group(1))
    raise RuntimeError('fleet holder is not an ancestor of this process')


def validate_before(states):
    if set(states) != set(NODES):
        raise RuntimeError('four-node inventory required')
    if all(v is None for v in states.values()):
        return 'absent'
    if not all(v is not None and v['running'] and not v['auto_remove'] and
               v['image'] == IMAGE and v['overlays'] and v['manifest'] for v in states.values()):
        raise RuntimeError('require four running, persistent, pinned-image containers or four absent containers')
    if states['local']['port'] not in (8000, 18000):
        raise RuntimeError('unknown incoming endpoint')
    return 'present'


def identity(state):
    return {k: v for k, v in state.items() if k not in ('running', 'started')}


def transition(node, before, action):
    if action not in ('stop','start'):
        raise ValueError('only stop/start transitions are supported')
    check_holder()
    code = INSPECT + '\n' + f'''
expected={before!r}
current=inspect({name(node)!r})
immutable=lambda s:{{k:v for k,v in s.items() if k not in ('running','started')}}
if current is None or immutable(current)!=immutable(expected):
    raise RuntimeError('container identity/config/source changed before {action}')
cmd=['docker',{action!r}] + (['--time','45'] if {action!r}=='stop' else []) + [expected['id']]
subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,timeout=75)
after=inspect({name(node)!r})
if after is None or immutable(after)!=immutable(expected) or after['running']!={action == 'start'!r}:
    raise RuntimeError('transition did not preserve expected container state')
print(json.dumps(after))
'''
    return remote(node, code, timeout=100)


def transition_all(before, action):
    # Settle every action before proceeding; one failed stop still triggers
    # recovery of every original container in the outer finally block.
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {n: pool.submit(transition, n, before[n], action) for n in NODES}
        results, errors = {}, {}
        for node, task in tasks.items():
            try:
                results[node] = task.result()
            except Exception as exc:
                errors[node] = repr(exc)
    if errors:
        raise RuntimeError(str(errors))
    return results


def healthy(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def idle(port):
    if not healthy(port):
        raise RuntimeError('incoming serving is not healthy')
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as r:
        metrics = r.read().decode()
    for metric in ('num_requests_running', 'num_requests_waiting'):
        values = re.findall(r'^vllm:' + metric + r'(?:\{[^\n]*\})?\s+([^\s]+)', metrics, re.M)
        if not values or any(float(v) != 0 for v in values):
            raise RuntimeError('incoming traffic is not idle: ' + metric)


def wait_restore(before, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = snapshot()
        if any(states[n] is None or identity(states[n]) != identity(before[n]) for n in NODES):
            raise RuntimeError('restore identity/config/source mismatch')
        if not all(states[n]['running'] for n in NODES):
            raise RuntimeError('a restored container exited')
        if healthy(before['local']['port']):
            return states
        time.sleep(10)
    raise RuntimeError('original endpoint did not recover health')


def with_paused(before, run, save):
    """Always recover the original set, including a partially failed stop."""
    try:
        save('stopped.json', transition_all(before, 'stop'))
        return run()
    finally:
        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            save('restarted.json', transition_all(before, 'start'))
            save('restored.json', wait_restore(before))
        finally:
            signal.signal(signal.SIGTERM, previous)


def pinned(path, revision):
    actual = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', path, 'status', '--porcelain'], text=True).strip()
    if actual != revision or dirty:
        raise RuntimeError('frozen source changed: ' + path)


def probe_api_preflight(path, revision):
    """Read/check the frozen distributed API on all ranks before stop/start."""
    code = "import json,subprocess\n" + f"path={str(path)!r}\nrevision={revision!r}\n" + """
assert subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()==revision
assert not subprocess.check_output(['git','-C',path,'status','--porcelain'],text=True).strip()
result=subprocess.check_output(['python3',path+'/probes/glm53_moe_m64_check.py','--transport','bf16','--check-api'],text=True)
print(json.dumps(json.loads(result)))
"""
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(NODES, pool.map(lambda node: remote(node, code), NODES)))


def run_probe(cmd, path, log):
    child = subprocess.Popen(cmd, cwd=path, start_new_session=True,
        env=dict(os.environ, IMAGE=IMAGE, MOE_STREAM_LOCAL_ONLY='1', PREFILL32_LOCAL_ONLY='1'),
        stdout=log, stderr=subprocess.STDOUT)
    try:
        return child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        # The shell normally cleans all four ranks. Also remove only this
        # hold's unique probe containers if a timeout/kill prevented its trap.
        check_holder()
        prefix = 'moe-m64-' + os.environ['FLEET_SESSION'] + '-'
        for node in NODES:
            remote(node, "import json,subprocess; prefix=" + repr(prefix) + "\n"
                "names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()\n"
                "for container in names:\n"
                "    if container.startswith(prefix): subprocess.run(['docker','rm','-f',container],check=True,timeout=45,stdout=subprocess.DEVNULL)\n"
                "print(json.dumps(dict(cleaned=True)))", timeout=90)


def restore_public(out, save, result):
    check_holder()
    repo = Path('/home/choiceoh/stkernel')
    needed = subprocess.run(['bash', str(repo/'bench/fleet.sh'), 'restore-needed',
                             os.environ['FLEET_SESSION']], capture_output=True)
    if needed.returncode == 1 and needed.stdout.strip().startswith(b'no ('):
        result['public_restore'] = 'handed to queued boot per fleet policy'
        return
    if needed.returncode != 0 or not needed.stdout.strip().startswith(b'yes ('):
        raise RuntimeError('fleet restore decision unavailable')
    env = dict(os.environ, REPO=str(repo), IMAGE=IMAGE, LEGS='none',
               GLM53_API_PORT='8000', GLM53_API_HOST='0.0.0.0', HEAD='10.10.10.2',
               HEAD_URL='http://10.10.10.2:8000', HEALTH_BUDGET_S='1800')
    with (out/'public-restore.log').open('x') as log:
        subprocess.run(['bash', str(repo/'bench/ab-lever.sh'), 'PREFILLGATESRESTORE', ''],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not healthy(8000):
        raise RuntimeError('public restore health failed')
    restored = snapshot()
    if validate_before(restored) != 'present' or restored['local']['port'] != 8000:
        raise RuntimeError('public restore four-node check failed')
    save('public-restored.json', restored)
    result['public_restore'] = 'healthy standard public configuration'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--probe-source', type=Path, required=True)
    ap.add_argument('--probe-revision', required=True)
    args = ap.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.probe_revision):
        ap.error('exact frozen probe commit required')
    global PINS
    label = 'moe-m64'
    command = ['bash', 'probes/run_glm53_moe_m64_tp4_check.sh']
    PINS = ((label, str(args.probe_source.resolve()), args.probe_revision, command),)
    args.out.mkdir(parents=True, exist_ok=False)
    def save(file, value):
        (args.out / file).write_text(json.dumps(value, indent=2) + '\n')
    def interrupted(signum, frame):
        raise InterruptedError('termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    result = dict(started=time.time(), exit_code=1, probes={})
    try:
        check_holder()
        pinned(str(Path(__file__).resolve().parents[1]), os.environ['OFFLINE_SOURCE_REV'])
        for _, path, rev, _ in PINS:
            pinned(path, rev)
            save('api-preflight.json', probe_api_preflight(path, rev))
        resources = {}
        for node in NODES:
            resources[node] = remote(node, "import json,shutil; print(json.dumps(dict(disk_free_gib=shutil.disk_usage('/home/choiceoh').free/2**30)))")
        save('resources.json', resources)
        if any(v['disk_free_gib'] < 128 for v in resources.values()):
            raise RuntimeError('128 GiB per-node disk reserve required before the offline turn')
        before = snapshot()
        save('before.json', before)
        mode = validate_before(before)
        if mode == 'present':
            idle(before['local']['port'])
        def run():
            for label, path, rev, cmd in PINS:
                check_holder()
                pinned(path, rev)
                print('START ' + label + ' ' + time.strftime('%F %T'), flush=True)
                entry = dict(revision=rev, started=time.time(), command=cmd)
                result['probes'][label] = entry
                with (args.out / (label + '.log')).open('x') as log:
                    # Each frozen runner supplies its own 12-minute timeout
                    # and removes only its own probe container on exit.
                    entry['exit_code'] = run_probe(cmd, path, log)
                entry['ended'] = time.time()
                save('progress.json', result)
                print('DONE ' + label + ' rc=' + str(entry['exit_code']), flush=True)
        if mode == 'present':
            with_paused(before, run, save)
            result['restored_original'] = True
            if before['local']['port'] != 8000:
                restore_public(args.out, save, result)
        else:
            try:
                run()
            finally:
                # Last-holder public recovery still runs if a probe aborts.
                previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    restore_public(args.out, save, result)
                finally:
                    signal.signal(signal.SIGTERM, previous)
        result['exit_code'] = 0 if all(v['exit_code'] == 0 for v in result['probes'].values()) else 1
    except BaseException as exc:
        result['error'] = repr(exc)
    finally:
        result['ended'] = time.time()
        save('completion.json', result)
        print(json.dumps(result), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
