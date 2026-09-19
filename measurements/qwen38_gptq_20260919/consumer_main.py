"""Continue the owned calibration window with a clean, pinned main consumer bracket.

The old shell is stopped after dispatching all projection workers. This
controller waits for their files, ends that shell, then runs the main bracket.
The paused parent is resumed in finally so the canonical hold always cleans up.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request


NODES = ('10.10.10.2', '10.10.10.1', '10.10.10.3', '10.10.10.4')
BASE = Path('/home/choiceoh/glm53-logs/qwen38-gptq-330k-20260919')
OLD = BASE / 'fleet-20260920b'
OWNER = 'queue/q38gptq-330k-0920d'
FIT = '/home/choiceoh/glm53-cache/qwen38-gptq-330k-20260919/fit330'
RTN = '/home/choiceoh/glm53-cache/qwen38-gptq-main-20260920/rtn'


def node(rank, command, *, check=True):
    cmd = ['bash', '-lc', command] if rank == 0 else ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'choiceoh@' + NODES[rank], command]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def owned():
    lease = json.loads(Path('/home/choiceoh/glm53-logs/st-fleet.lock').read_bytes())
    holder = Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
    assert lease['owner'] == OWNER and holder[0] == OWNER.removeprefix('queue/') and holder[-1] == 'boot'


def main():
    global OWNER
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--helpers', type=Path, required=True)
    p.add_argument('--session', default=OWNER.removeprefix('queue/'))
    p.add_argument('--standalone', action='store_true')
    p.add_argument('--suffix', default='')
    args = p.parse_args()
    OWNER = 'queue/' + args.session
    out = BASE / ('main-' + args.sha[:12] + args.suffix)
    out.mkdir(exist_ok=False)
    env = dict(os.environ, ST_LEASE_OWNER=OWNER, ST_WINDOW_PARENT='1', FLEET_SESSION=OWNER.removeprefix('queue/'),
               PORT='8001', GLM53_API_PORT='8001', BENCH_MODEL='qwen3.8-flash-next', SPEC_K='3', ST_SPEC_K='3',
               ST_HC_FP8='0', ST_MTP_PRECISION='bf16', ST_MTP_EXPERTS='bf16', ST_SELF_CALIBRATE='0',
               ST_TAP_MTP_INPUTS='0', ST_SHARED_OVERLAP='one', ST_DRAFT_CANDIDATES='0', ST_DRAFT_THRESHOLD='0.1',
               ST_ENGINE_DIR='/home/choiceoh/st-engine-qwen38-gptq-main-4436',
               ST_IMAGE='st-engine:qwen38-gptq-main-4436', ONEPASS_PROFILE='extended',
               ONEPASS_JSONL=str(out / 'onepass.jsonl'), ONEPASS_ST_CONTAINER='st-qwen38', ST_BRACKET_SHA=args.sha)
    for key in ('ST_MTP_TUNED', 'ST_DRAFT_INDEX', 'ST_PACK_ROOT', 'ST_CALIBRATION_ROWS', 'PYTHONPATH'):
        env.pop(key, None)
    helper = args.helpers / 'measurements/qwen38_gptq_20260919'
    harness = ['python3', '-u', str(helper / 'run_main_onepass.py'), '--tree', str(args.tree), '--helpers', str(args.helpers)]
    launcher = ['bash', str(args.tree / 'launchers/start-st-qwen38.sh')]
    child = None

    def report(stage, **fields):
        value = dict(stage=stage, time=time.time(), pid=os.getpid(), main_sha=args.sha, **fields)
        (out / 'status.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    def run(command, path, *, extra=None, accepted=(0,)):
        nonlocal child
        owned()
        with path.open('w') as log:
            child = subprocess.Popen(command, cwd=args.tree, env=dict(env, **(extra or {})), stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                owned()
                time.sleep(2)
        rc = child.returncode
        child = None
        if rc not in accepted:
            raise RuntimeError(f'{path.name}: command exited {rc}')
        return rc

    def audit_rank(rank, label, root, *, snapshot=False):
        dest = out / f'{label}-audit-rank{rank}.json'
        command = ['docker', 'exec', '-e', f'PYTHONPATH=/repo:{out}/tools', 'st-qwen38', 'python3',
                   str(out / 'tools/audit_main.py'), '--reference', str(OLD / f'B330pack-audit-rank{rank}.json'),
                   '--out', str(dest), '--root', root]
        if not snapshot:
            command += ['--boot', str(out / f'{label}-boot-rank{rank}.json')]
            command += ['--packs', str(out / f'evaluated-audit-rank{rank}.json')] if label == 'B330' else ['--rtn']
        result = node(rank, shlex.join(command))
        (out / f'{label}-audit-rank{rank}.log').write_text(result.stdout + result.stderr)
        if rank:
            subprocess.run(['scp', '-q', f'choiceoh@{NODES[rank]}:{dest}', str(dest)], check=True)

    try:
        owned()
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.tree, text=True).strip() == args.sha
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.tree, text=True).strip()
        engine = subprocess.check_output(['git', 'rev-parse', 'HEAD:engine'], cwd=args.tree, text=True).strip()
        env['ST_BRACKET_TREE'] = engine
        run(harness + ['--preflight'], out / 'preflight.log')
        (out / 'source.json').write_text(json.dumps(dict(main_sha=args.sha, engine_tree=engine,
            engine_and_launcher_unmodified=True, helper_source_sha=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=args.helpers, text=True).strip(),
            helper_files={n: hashlib.sha256((helper / n).read_bytes()).hexdigest()
                          for n in ('consumer_main.py', 'run_main_onepass.py', 'audit_main.py')}), indent=2) + '\n')
        if not args.standalone:
            (BASE / 'latest-origin-handoff/consumer-controller-started.json').write_text(json.dumps(dict(pid=os.getpid(), out=str(out))))
        report('waiting_for_projection_scores')
        for rank in range(4):
            node(rank, f'mkdir -p {out}/tools/probes')
            for source, dest in [(helper / 'audit_main.py', out / 'tools/audit_main.py'),
                                 (args.helpers / 'probes/qwen38_gptq_audit.py', out / 'tools/probes/qwen38_gptq_audit.py')]:
                if rank == 0:
                    dest.write_bytes(source.read_bytes())
                else:
                    subprocess.run(['scp', '-q', str(source), f'choiceoh@{NODES[rank]}:{dest}'], check=True)
        if not args.standalone:
            deadline = time.monotonic() + 1500
            while True:
                owned()
                if (BASE / 'latest-origin-handoff/paused.json').exists() and all(
                        node(r, f'test -s {OLD}/projection-rank{r}.json', check=False).returncode == 0 for r in range(4)):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('projection scores did not complete')
                time.sleep(2)
        for rank in range(1, 4):
            for name in (f'projection-rank{rank}.json', f'B330pack-audit-rank{rank}.json', f'B330pack-boot-rank{rank}.json'):
                subprocess.run(['scp', '-q', f'choiceoh@{NODES[rank]}:{OLD}/{name}', str(OLD / name)], check=True)
        if args.standalone:
            previous = BASE / ('main-' + args.sha[:12])
            for rank in range(4):
                node(rank, f'cp {previous}/evaluated-audit-rank{rank}.json {out}/evaluated-audit-rank{rank}.json')
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda r: audit_rank(r, 'evaluated', '/cache/qwen38-gptq-330k-20260919/fit330', snapshot=True), range(4)))
            os.kill(266443, signal.SIGTERM)
            os.kill(266443, signal.SIGCONT)
            while Path('/proc/266443/cmdline').exists() and Path('/proc/266443/cmdline').read_bytes():
                owned()
                time.sleep(1)
        report('switching_to_latest_main')
        for rank in range(4):
            for cache in (FIT, RTN):
                node(rank, f'sudo -n install -d -m 755 {cache}/cu132; mountpoint -q {cache}/cu132 || sudo -n mount --bind /home/choiceoh/glm53-cache/cu132 {cache}/cu132')
        first_images = first_runtime = None
        for label, cache in (('A1', RTN), ('B330', FIT), ('A2', RTN)):
            env.update(CACHE_DIR=cache, ST_TIER_DIR=str(out / ('tier-' + label)))
            report('booting', arm=label)
            run(launcher, out / f'{label}-launch.log')
            deadline = time.monotonic() + 1800
            while True:
                owned()
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8001/v1/models', timeout=3) as response:
                        models = json.load(response)
                    assert any(m['id'] == 'qwen3.8-flash-next' for m in models['data'])
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(label + ' boot readiness')
                    time.sleep(2)
            images, runtimes = [], []
            for rank in range(4):
                node(rank, f'cp /home/choiceoh/glm53-logs/st-qwen38-dumps/boot-rank{rank}.json {out}/{label}-boot-rank{rank}.json')
                inspected = json.loads(node(rank, 'docker inspect st-qwen38').stdout)[0]
                mount = next(m for m in inspected['Mounts'] if m['Destination'] == '/cache')
                assert mount['Source'] == cache
                (out / f'{label}-container-rank{rank}.json').write_text(json.dumps(inspected, indent=2) + '\n')
                images.append(inspected['Image'])
                raw = node(rank, 'docker exec st-qwen38 cat /opt/st/runtime-manifest.json').stdout
                (out / f'{label}-runtime-rank{rank}.json').write_text(raw)
                runtimes.append(raw)
            assert len(set(runtimes)) == 1
            if first_images is None:
                first_images, first_runtime = images, runtimes
            else:
                assert images == first_images and runtimes == first_runtime
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda r: audit_rank(r, label, '/cache'), range(4)))
            for index in (1, 2):
                report('measuring', arm=label, run=index)
                req = urllib.request.Request('http://127.0.0.1:8001/v1/prefix/reset', data=b'{}', headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    response.read()
                rc = run(harness + ['--name', 'q38gptq-main-' + label, '--num-spec', '3', '--require-exclusive'],
                         out / f'{label}-onepass-{index}.log', extra=dict(ONEPASS_RUN_INDEX=str(index)), accepted=(0, 2))
                (out / f'{label}-onepass-{index}.rc').write_text(str(rc) + '\n')
                records = [json.loads(line) for line in (out / 'onepass.jsonl').read_text().splitlines()]
                rec = records[-1]
                assert rec['arm_sha'] == args.sha and rec['run_index'] == index and rec['boot_id']
                assert rec['recording']['status'] == 'complete' and rec['engine'] == 'st' and not rec.get('rehearsal')
            for rank in range(4):
                (out / f'{label}-rank{rank}.log').write_text(node(rank, 'docker logs st-qwen38 2>&1').stdout)
                if rank:
                    subprocess.run(['scp', '-q', f'choiceoh@{NODES[rank]}:{out}/{label}-boot-rank{rank}.json', str(out)], check=True)
            run(launcher + ['stop'], out / f'{label}-stop.log')
        report('finished', note='Read individual quality and evidence gates; process completion is not a passing result')
    except BaseException as error:
        report('failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        try:
            owned()
            subprocess.run(launcher + ['stop'], cwd=args.tree, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for rank in range(4):
                for cache in (FIT, RTN):
                    node(rank, f'mountpoint -q {cache}/cu132 && sudo -n umount {cache}/cu132', check=False)
        finally:
            if args.standalone:
                try:
                    owned()
                    (Path('/home/choiceoh/glm53-logs/st-bracket') / args.session / 'stop').touch()
                except (OSError, AssertionError):
                    pass
            else:
                for pid, sig in ((266443, signal.SIGTERM), (266443, signal.SIGCONT), (241445, signal.SIGCONT)):
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    main()
