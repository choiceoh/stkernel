"""Run the 330K campaign inside a canonical st-hold reservation on srv2.

The stock hold first boots the deployed GLM release. Once it reports ready, stop
only that reservation's GLM boot and use its documented session window for Qwen.
The queue retains its holder and lease for the whole campaign, so every pooled
single-GPU lane can see the four GPUs are reserved. Do not release its lease by
hand; end the hold with its own stop file after Qwen has stopped.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def owned_window(lease, holder, session):
    return (lease.get('owner') == 'queue/' + session and lease.get('kind') == 'queue'
            and len(holder) == 7 and holder[0] == session and holder[6] == 'boot')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--control-tree', type=Path, default=Path('/home/choiceoh/stkernel'))
    p.add_argument('--hold-sha', required=True)
    p.add_argument('--session', default='q38gptq-330k-0920')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.hold_sha):
        raise ValueError('hold source must be a resolved commit')
    if not re.fullmatch('[A-Za-z0-9_-]+', args.session):
        raise ValueError('invalid session name')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for cmd, expected in ((['git', 'rev-parse', 'HEAD'], args.sha),
                          (['git', 'status', '--porcelain'], '')):
        if subprocess.check_output(cmd, cwd=args.tree, text=True).strip() != expected:
            raise ValueError('Qwen source checkout changed or is dirty')
    driver = args.tree / 'measurements/qwen38_gptq_20260919/collect_window.sh'
    subprocess.run(['bash', str(driver), 'fleet330', '--preflight'], cwd=args.tree, check=True)
    logs = Path('/home/choiceoh/glm53-logs')
    stopfile = logs / 'st-bracket' / args.session / 'stop'
    fleet = args.control_tree / 'bench/fleet.sh'
    hold, experiment, observer = None, None, None
    def report(stage, **details):
        path = args.out / 'status.json'
        record = dict(stage=stage, time=time.time(), pid=os.getpid(), source_sha=args.sha,
                      reservation=args.session, hold_sha=args.hold_sha, **details)
        path.with_suffix('.tmp').write_text(json.dumps(record, indent=2) + '\n')
        path.with_suffix('.tmp').replace(path)
        print(json.dumps(record), flush=True)
    def owned():
        try:
            lease = json.loads((logs / 'st-fleet.lock').read_text())
            holder = (logs / 'fleet/holder').read_text().strip().split('|')
            return owned_window(lease, holder, args.session)
        except (OSError, ValueError):
            return False
    def terminate(signum, _frame):
        raise InterruptedError('signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    env = dict(os.environ, REPO=str(args.control_tree))
    env.pop('PYTHONPATH', None)  # canonical control policy requires its own imports
    env.pop('ST_LEASE_OWNER', None)
    rc = 1
    try:
        with (args.out / 'reservation.log').open('x') as log:
            hold = subprocess.Popen(['bash', str(fleet), 'st-hold', args.session, args.hold_sha, '150',
                'Operator requested fleet: collect 330K Qwen rows, GPTQ pack/error scoring, RTN-330K-RTN onepass'],
                cwd=args.control_tree, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        report('waiting_for_canonical_reservation', reservation_pid=hold.pid)
        deadline = time.monotonic() + 12 * 3600
        while True:
            if hold.poll() is not None:
                raise RuntimeError('canonical hold exited before ready: ' + str(hold.returncode))
            text = (args.out / 'reservation.log').read_text()
            if re.search(r'holding [0-9a-f]{12} on http://[^\s]+ for up to 150m', text) and owned():
                break
            if time.monotonic() > deadline:
                raise TimeoutError('canonical reservation wait expired')
            time.sleep(5)
        report('reserved_switching_owned_boot_to_qwen')
        release = Path('/home/choiceoh/st-releases') / args.hold_sha[:12]
        window_env = dict(os.environ, ST_LEASE_OWNER='queue/' + args.session, ST_WINDOW_PARENT='1',
                          FLEET_SESSION=args.session)
        subprocess.run(['bash', str(release / 'launchers/start-st-glm53.sh'), 'stop'],
                       cwd=release, env=dict(window_env, ST_ENGINE_DIR=str(release)), check=True)
        if not owned():
            raise RuntimeError('reservation lost while switching the owned model')
        observer = subprocess.Popen(['bash', str(driver.with_name('observe_fleet.sh')), str(args.tree),
            str(logs / 'qwen38-gptq-330k-20260919/fleet-20260920/occupancy')],
            cwd=args.tree, env=window_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with (args.out / 'experiment.log').open('x') as log:
            experiment = subprocess.Popen(['bash', str(driver), 'fleet330'], cwd=args.tree,
                env=window_env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        report('running_on_fleet', experiment_pid=experiment.pid)
        while experiment.poll() is None:
            if hold.poll() is not None or not owned():
                raise RuntimeError('canonical reservation ended before the Qwen campaign')
            time.sleep(5)
        rc = experiment.returncode
        report('finished' if rc == 0 else 'failed', returncode=rc,
               note='Read onepass return codes separately; driver success does not establish quality')
    except BaseException as exc:
        report('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if experiment is not None and experiment.poll() is None:
            experiment.terminate()  # The driver's owner-checked EXIT trap stops Qwen.
            experiment.wait()
        if observer is not None and observer.poll() is None:
            observer.terminate()
            observer.wait()
        if hold is not None and hold.poll() is None:
            if owned():
                stopfile.parent.mkdir(parents=True, exist_ok=True)
                stopfile.touch()  # The stock hold releases through the canonical supervisor.
                try:
                    hold.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    hold.terminate()
                    hold.wait()
            else:
                hold.terminate()
                hold.wait()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
