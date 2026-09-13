"""Compile the next two probes on CPUs during pass-2 preparation only.

The immutable serving containers, source, caches and GPU queue are untouched.
Both private containers use runc, expose no GPU, and have bounded CPU/memory.
If preparation ends before compilation, stop only that owned CPU container.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/choiceoh/glm53-logs')
OUT = ROOT / 'st-decode-next-cpu0913'
CONTROL = ROOT / 'st-decode-ranksum-8c8b031b-functional/continuation.json'
SOURCE = '8c8b031b94bb80175cfccd49cfb6329d18cdecae'
SESSION = 'st-decode-ranksum0913'
BOOT = '8f055b03662995db50df27977fd51996737a1a4962c835838e39c38988d997e0'
IMAGE = 'sha256:a13be2698a579983329b82a1fc2a98d6462346a323981ba17e97f27c3b7fbd7b'
LABEL = 'codex.scope=st-decode-next-cpu0913'


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def phase():
    control = json.loads(CONTROL.read_text())
    if control['status'] in ('complete', 'incomplete'):
        return 'finished'
    if control['status'] != 'running pass 2':
        return 'waiting'
    candidates = sorted((ROOT / 'onepass-runs').glob('*/record.json'),
                        key=lambda path: path.stat().st_mtime, reverse=True)[:8]
    for path in candidates:
        record = json.loads(path.read_text())
        if (record.get('session') == SESSION and record.get('arm_sha') == SOURCE
                and record.get('run_index') == 2):
            if record['boot_id'].split('|')[0] != BOOT:
                raise RuntimeError('pass 2 changed the engine boot')
            return record['recording']['phase']
    return 'waiting'


def stop_cpu(name):
    # No server name or ST launcher appears in this controller.
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if result.returncode:
        return
    identity = json.loads(result.stdout)[0]
    if (identity['Config'].get('Labels', {}).get('codex.scope') != LABEL.split('=', 1)[1]
            or identity['HostConfig']['Runtime'] != 'runc'
            or identity['HostConfig'].get('DeviceRequests')):
        raise RuntimeError('refusing to stop a container outside the CPU compile scope')
    subprocess.run(['docker', 'stop', '--time', '3', name], check=True, capture_output=True)


def main():
    report = dict(status='waiting for pass-2 preparation', started=timestamp(), jobs=[],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime='runc', gpu_used=False, cpuset='16-19', cpus=2, memory_gib=12,
                  image=IMAGE, source=SOURCE, session=SESSION, boot_id=BOOT)
    OUT.mkdir(exist_ok=True)
    def save():
        path = OUT / 'controller.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)
    save()
    deadline = time.monotonic() + 3 * 3600
    while phase() != 'prepare-c1':
        if phase() == 'finished' or time.monotonic() >= deadline:
            report.update(status='skipped: no preparation window', ended=timestamp())
            save()
            return
        time.sleep(10)
    for lane, sha, module in (
            ('mhc', '4626491bd3c2a6c2e535bc9611b67aeabd1e8058', 'probes.engine_mhc_single_compile'),
            ('moe', 'c13f280214b076abbb236fc613003c62655fb090', 'probes.engine_moe_waves_compile')):
        if phase() != 'prepare-c1':
            report['status'] = 'remaining jobs skipped: preparation ended'
            break
        output = OUT / (lane + '-output')
        output.mkdir(exist_ok=True)
        name = 'cpu-decode-' + lane + '-' + sha[:8]
        row = dict(lane=lane, source=sha, container=name, started=timestamp(), phase='prepare-c1')
        report['jobs'].append(row)
        report['status'] = 'compiling ' + lane
        save()
        args = ['docker', 'run', '--rm', '--name', name, '--label', LABEL,
                '--runtime=runc', '--network=none', '--cpuset-cpus=16-19', '--cpus=2', '--memory=12g',
                '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'MAX_JOBS=1',
                '--mount', f'type=bind,src={OUT / (lane + "-source")},dst=/src,readonly',
                '--mount', f'type=bind,src={output},dst=/out', '--workdir=/src',
                '--entrypoint=python3', IMAGE, '-m', module, '--output', '/out/compile.json']
        if lane == 'mhc':
            args += ['--build-root', '/out/build']
        with (output / 'compile.log').open('x') as stream:
            process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT)
            try:
                expiry = time.monotonic() + 180
                while process.poll() is None:
                    observed = phase()
                    if observed != 'prepare-c1' or time.monotonic() >= expiry:
                        row['stopped'] = 'preparation ended' if observed != 'prepare-c1' else '180-second CPU limit'
                        stop_cpu(name)
                        break
                    time.sleep(2)
                row.update(exit_code=process.wait(timeout=10), ended=timestamp(), end_phase=phase())
            except BaseException:
                stop_cpu(name)
                raise
        row['status'] = 'PASS' if row['exit_code'] == 0 and (output / 'compile.json').is_file() else 'FAIL'
        save()
    else:
        report['status'] = 'complete'
    report['ended'] = timestamp()
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
