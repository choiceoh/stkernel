"""Continue the owned consumer without treating quality exit 2 as a crash.

The original controller used check=True for onepass, which would stop the
hold after a fully recorded quality failure instead of running pass 2. This
controller binds its exact parent/child identities before taking over the
existing pass. It never signals the consumer, a container or the fleet hold.
The frozen source, canonical workload and original hold deadline stay intact.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request


SHA = '8c8b031b94bb80175cfccd49cfb6329d18cdecae'
SESSION = 'st-decode-ranksum0913'
REPO = Path('/home/choiceoh/stkernel-worktrees/codex-st-oneshot-rank-sum')
LOGS = Path('/home/choiceoh/glm53-logs')
OUT = LOGS / 'st-decode-ranksum-8c8b031b-functional'
FIRST_RUN = LOGS / 'onepass-runs/20260912T213057-9725b41d901c'
OLD_CONTROLLER = 984315
FIRST_CONSUMER = 1065892
ORIGINAL = LOGS / 'st-decode-ranksum-functional-then-onepass.py'
ORIGINAL_HASH = 'f7518edf30651bc9993bbeefd7d3673035bf57df7c3efe2166578b3f4f1f3936'
NODES = (None, 'choiceoh@10.10.10.1', 'choiceoh@10.10.10.3', 'choiceoh@10.10.10.4')


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start=fields[19],
                    args=(root / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0'),
                    cwd=str((root / 'cwd').resolve()))
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    report = dict(source=SHA, session=SESSION, status='binding', passes=[],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reason='a completed quality exit 2 must retain the boot for pass 2')
    path = OUT / 'continuation.json'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)

    functional = json.loads((OUT / 'functional.json').read_text())
    assert functional['status'] == 'PASS' and functional['source'] == SHA
    containers = functional['containers']

    def rank_owned(item):
        node, identity = item
        cmd = ['docker', 'inspect', identity['id']]
        if node:
            cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', node, shlex.join(cmd)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
        value = json.loads(result.stdout)[0]
        env = dict(e.split('=', 1) for e in value['Config']['Env'] if '=' in e)
        return (value['Id'] == identity['id'] and value['State']['Running']
                and value['State']['StartedAt'] == identity['started']
                and value['Image'] == identity['image']
                and env.get('ST_LEASE_OWNER') == 'queue/' + SESSION
                and env.get('ST_RELEASE') == SHA[:12])

    def check_ranks():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            assert all(pool.map(rank_owned, zip(NODES, containers))), 'owned rank changed or exited'

    def check_record(directory, index):
        record = json.loads((directory / 'record.json').read_text())
        assert record['recording']['status'] == 'complete', 'canonical pass did not finish'
        assert record['arm_sha'] == SHA and record['run_index'] == index
        assert record['session'] == SESSION and record['cold'] == 'reset'
        assert record['boot_id'].split('|')[0] == containers[0]['id']
        assert len(record['c4']) == 5 and len(record['diagnostics']) == 6
        result = dict(index=index, run_id=record['run_id'], complete=True,
                      evidence_issues=record.get('evidence_issues', []),
                      quality=record.get('quality'), quality_c4=record.get('quality_c4'),
                      recording=record['recording'])
        report['passes'].append(result)
        save()
        return record

    check_ranks()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == SHA
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_HASH
    old, child = process(OLD_CONTROLLER), process(FIRST_CONSUMER)
    assert old and child and child['ppid'] == OLD_CONTROLLER
    assert old['args'] == [b'python3', b'-u', os.fsencode(ORIGINAL)]
    assert child['args'] == [b'python3', b'-u', b'bench/onepass.py', b'--name', b'rank-sum', b'--require-exclusive']
    assert child['cwd'] == str(REPO)
    report.update(old_controller=dict(pid=OLD_CONTROLLER, start=old['start']),
                  existing_consumer=dict(pid=FIRST_CONSUMER, start=child['start']),
                  containers=containers, status='handoff')
    save()
    bound = False
    try:
        # SIGTERM is the Python controller's default termination action.
        # Signal this exact PID only: its canonical onepass child continues.
        os.kill(OLD_CONTROLLER, signal.SIGTERM)
        bound = True
        for _ in range(50):
            parent = process(OLD_CONTROLLER)
            if not parent or parent['start'] != old['start'] or parent['state'] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('the previous controller did not exit')
        current = process(FIRST_CONSUMER)
        assert current and current['start'] == child['start'] and current['state'] not in ('Z', 'T')
        report['status'] = 'waiting for existing pass 1'
        save()
        print('controller handed over; existing canonical pass 1 and all four ranks continue', flush=True)
        while True:
            current = process(FIRST_CONSUMER)
            if not current or current['start'] != child['start'] or current['state'] == 'Z':
                break
            check_ranks()
            time.sleep(30)
        check_record(FIRST_RUN, 1)
        check_ranks()
        report['status'] = 'running pass 2'
        save()
        request = urllib.request.Request('http://127.0.0.1:18123/v1/prefix/reset',
                                         data=b'{}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            json.load(response)
        environment = dict(os.environ, GLM53_API_PORT='18123', BENCH_MODEL='glm-5.3-flash',
                           ONEPASS_JSONL=str(LOGS / 'st-decode-ranksum-8c8b031b.jsonl'),
                           ST_BRACKET_SHA=SHA, ST_BRACKET_COLD='reset', FLEET_SESSION=SESSION,
                           ONEPASS_RUN_INDEX='2')
        print('canonical pass 1 is fully recorded; pass 2 starts on the same unchanged boot', flush=True)
        with (OUT / 'onepass-2.log').open('x') as stream:
            result = subprocess.run(['python3', '-u', 'bench/onepass.py', '--name', 'rank-sum', '--require-exclusive'],
                                    cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode in (0, 2), f'canonical pass 2 exited {result.returncode}'
        records = [json.loads(line) for line in Path(environment['ONEPASS_JSONL']).read_text().splitlines()]
        second = [r for r in records if r.get('arm_sha') == SHA and r.get('run_index') == 2 and r.get('session') == SESSION]
        assert len(second) == 1, 'expected one completed pass 2 from this boot'
        check_record(LOGS / 'onepass-runs' / second[0]['run_id'], 2)
        check_ranks()
        report.update(status='complete', complete_onepasses=2, same_boot=True,
                      evidence_valid=all(not r['evidence_issues'] for r in report['passes']))
        save()
        (OUT / 'consumer-complete.json').write_text(json.dumps(report, indent=2) + '\n')
        print('two complete canonical passes recorded; quality failures retained independently', flush=True)
    except BaseException as error:
        report.update(status='incomplete', error=repr(error))
        save()
        raise
    finally:
        if bound:
            (LOGS / 'st-bracket' / SESSION / 'stop').touch()
            print('requested the canonical hold to stop its own boot and release', flush=True)


if __name__ == '__main__':
    main()
