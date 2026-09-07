"""Owned fleet boot, guarded captures, trace archival and exact-setting restore."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import urllib.request

from snapshot import snapshot

JOB = Path(os.environ['PROFILE_JOB'])
REPO = Path(os.environ['REPO'])
BASE = 'http://127.0.0.1:18000'
NODES = ['local', '10.10.10.1', '10.10.10.3', '10.10.10.4']


def save(name, obj):
    (JOB / name).write_text(json.dumps(obj, indent=2) + '\n')


def remote(host, script):
    cmd = ['python3', '-c', script] if host == 'local' else [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
        'choiceoh@' + host, 'python3 -c ' + shlex.quote(script)]
    return json.loads(subprocess.check_output(cmd, text=True, timeout=45))


def traces():
    script = "import json,pathlib; print(json.dumps({str(p):[p.stat().st_size,p.stat().st_mtime] for p in pathlib.Path('/home/choiceoh/vllm-prof').rglob('*.pt.trace.json.gz')}))"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(NODES, pool.map(lambda n: remote(n, script), NODES)))


def archive_traces(name, before):
    deadline = time.monotonic() + 240
    previous, stable = None, 0
    while time.monotonic() < deadline:
        now = traces()
        fresh = {node: {p: stat for p, stat in files.items()
                        if p not in before[node] or stat != before[node][p]}
                 for node, files in now.items()}
        ready = all(any('rank' in p and s[0] > 1000 for p, s in fs.items())
                    for fs in fresh.values())
        stable = stable + 1 if ready and fresh == previous else 0
        if stable >= 2:
            break
        previous = fresh
        time.sleep(3)
    else:
        raise RuntimeError('all four rank traces did not finish flushing')
    save(name + '.trace-sources.json', fresh)
    target = JOB / 'traces' / name
    for node, files in fresh.items():
        out = target / node
        out.mkdir(parents=True, exist_ok=True)
        for path in files:
            dest = out / Path(path).name
            if node == 'local':
                shutil.copy2(path, dest)
            else:
                subprocess.run(['scp', '-q', '-o', 'BatchMode=yes',
                                'choiceoh@' + node + ':' + path, str(dest)], check=True)
    print('archived traces ' + name, flush=True)


def normalized(env):
    out = {k:v for k,v in env.items() if k.startswith('VLLM_GLM53_')}
    for key, value in [('FUSE_MHC', '0'), ('DIRECT_NCCL', '0'),
                       ('FP8_AG_MIN_TOKENS', '-1'), ('FP8_RS_MIN_TOKENS', '-1')]:
        out.setdefault('VLLM_GLM53_PREFILL_SP_' + key, value)
    return out


def verify(before, after, stage):
    issues = []
    for node in NODES:
        b, a = before[node], after[node]
        for key in ['image', 'mounts', 'manifest']:
            if b[key] != a[key]:
                issues.append(node + ': changed ' + key)
        if normalized(b['env']) != normalized(a['env']):
            issues.append(node + ': changed model settings')
    save(stage + '-issues.json', issues)
    if issues:
        raise RuntimeError(str(issues))


def boot(name, env):
    print('boot ' + name + ' ' + datetime.datetime.now().isoformat(), flush=True)
    with (JOB / (name + '.boot.log')).open('w') as log:
        subprocess.run(['bash', str(REPO / 'bench/ab-lever.sh'), name, ''],
                       env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=2400)
    print('healthy ' + name + ' ' + datetime.datetime.now().isoformat(), flush=True)


def main():
    before = snapshot()
    save('production-before.json', before)
    head = before['local']
    expected = {}
    for line in (REPO/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        src, dest, contract = line.split('\t')
        expected[dest] = hashlib.sha256((REPO/'build/glm53'/src).read_bytes()).hexdigest()
    for node, state in before.items():
        if state['disk_free_gib'] < 128:
            raise RuntimeError(node + ': less than 128 GiB disk reserve')
        if {k:v['sha256'] for k,v in state['mounts'].items()} != expected:
            raise RuntimeError(node + ': mounted code differs from frozen source')
        if normalized(state['env']) != normalized(head['env']):
            raise RuntimeError(node + ': model settings differ between ranks')

    env = dict(os.environ)
    env.update({k:v for k,v in head['env'].items() if k.startswith('VLLM_')})
    env.update(IMAGE=head['image'], LEGS='none', PREFILL_WARMUP='0',
               HEALTH_BUDGET_S='1800', BENCH_MODEL='glm-5.3-flash',
               KV_TOKENS='524288', MAX_LEN='262144',
               GLM53_API_PORT='18000', GLM53_API_HOST='127.0.0.1',
               HEAD='127.0.0.1', HEAD_URL=BASE)
    # Pin all graph/scheduling dimensions visible in the actual serving args.
    for flag, key in [('max-num-batched-tokens','MAX_BATCHED'),
                      ('max-num-seqs','MAX_SEQS'),
                      ('max-cudagraph-capture-size','GRAPH_CAP'),
                      ('gpu-memory-utilization','GMU')]:
        match = re.search('--' + flag + r'\s+(\S+)', head['command'])
        if match:
            env[key] = match.group(1)
    env['SPEC_K'] = head['env']['VLLM_GLM53_SPEC_K']
    save('measurement-overrides.json', {k:env[k] for k in [
        'KV_TOKENS','MAX_LEN','MAX_BATCHED','MAX_SEQS','GRAPH_CAP','GMU','SPEC_K',
        'VLLM_GLM53_NVFP4_STATIC_SCALE','GLM53_API_PORT','IMAGE']})
    changed = False
    result = {'started': datetime.datetime.now().isoformat(), 'exit_code': 1}
    try:
        changed = True
        boot('PATTR0907', env)
        measured = snapshot()
        save('measurement-before.json', measured)
        verify(before, measured, 'measurement')
        # The controls warm the process's exact request shapes. Profiles are
        # followed by a second clean control to bound instrumentation overhead.
        for capture, suffix in [(False,'B1'), (True,'P'), (False,'B2')]:
            for ctx in [32000,128000]:
                name = f'PATTR{ctx//1000}K{suffix}'
                inventory = traces() if capture else None
                cmd = ['python3', str(REPO/'bench/onepass_memory.py'),
                       '--minimum-gib', '12', '--report', str(JOB/(name+'.memory.jsonl')),
                       '--', 'python3', str(JOB/'glm53_prefill_attribution.py'),
                       '--repo', str(REPO), '--out', str(JOB), '--name', name,
                       '--ctx', str(ctx)]
                if capture:
                    cmd.append('--capture')
                print('request ' + name + ' ' + datetime.datetime.now().isoformat(), flush=True)
                with (JOB/(name+'.log')).open('w') as log:
                    subprocess.run(cmd, env=env, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=1200)
                print('completed ' + name, flush=True)
                if capture:
                    archive_traces(name, inventory)
        after = snapshot()
        save('measurement-after.json', after)
        verify(before, after, 'measurement-after')
        result['exit_code'] = 0
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        if changed:
            # Always restore the exact public settings we observed, even if a
            # later fleet boot is queued. This job owns all of its transitions.
            restore = dict(env)
            restore.update(KV_TOKENS='2000000', MAX_LEN='1048576',
                           GLM53_API_PORT='8000', GLM53_API_HOST='0.0.0.0',
                           HEAD='10.10.10.2', HEAD_URL='http://10.10.10.2:8000')
            try:
                boot('PATTR0907RESTORE', restore)
                restored = snapshot()
                save('production-after.json', restored)
                verify(before, restored, 'production-restore')
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10) as r:
                    result['production_health'] = r.status
                result['restored'] = True
            except BaseException as exc:
                result['restore_error'] = repr(exc)
                result['exit_code'] = 1
        result['ended'] = datetime.datetime.now().isoformat()
        save('completion.json', result)
        (JOB/'exit_code').write_text(str(result['exit_code'])+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
