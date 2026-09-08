#!/usr/bin/env python3
"""Normal fleet diagnostic: one post-context binding query and exact restore."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

import glm53_probe_lifecycle as lifecycle
import glm53_ep_sanitizer as sanitizer
from run_glm53_ep_local_offline import resources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        ap.error('full frozen source revision required')
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(started=time.time(), source_revision=args.revision, exit_code=1,
                  performance_acceptance=False, diagnostic_only=True)
    name = 'ep-binding-' + uuid.uuid4().hex
    owned = False

    def save(filename, value):
        (out/filename).write_text(json.dumps(value, indent=2)+'\n')

    def interrupted(signum, frame):
        raise InterruptedError('termination requested')

    signal.signal(signal.SIGTERM, interrupted)

    def cleanup():
        lifecycle.check_holder()
        if not owned:
            return
        inspected = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True)
        if inspected.returncode:
            names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
            if name in names:
                raise RuntimeError('owned diagnostic inspect failed')
            return
        state = json.loads(inspected.stdout)[0]
        if (state['Image'] != lifecycle.IMAGE or state['Config'].get('Labels', {}).get(
                'glm53.ep-binding.session') != os.environ['FLEET_SESSION']):
            raise RuntimeError('owned diagnostic container identity changed')
        subprocess.run(['docker', 'rm', '-f', state['Id']], check=True, timeout=45,
                       stdout=subprocess.DEVNULL)

    def run():
        nonlocal owned
        lifecycle.check_holder()
        lifecycle.pinned(str(root), args.revision)
        save('resources-offline.json', resources(True))
        states = lifecycle.snapshot()
        if any(state is not None and state['running'] for state in states.values()):
            raise RuntimeError('original serving must be paused on all four nodes')
        command = ['docker', 'run', '--rm', '--name', name, '--label',
                   'glm53.ep-binding.session='+os.environ['FLEET_SESSION'], '--gpus', 'all',
                   '--network=none', '--memory=4g', '--memory-swap=4g', '--cpus=2',
                   '--pids-limit=128', '--entrypoint=/usr/bin/env',
                   '--mount', f'type=bind,source={root},target=/repo,readonly',
                   '--mount', f'type=bind,source={out},target=/evidence']
        command += sanitizer.mount_args(receipt) + [lifecycle.IMAGE]
        command += sanitizer.command('memcheck') + [
            'python3', '/repo/probes/glm53_ep_binding_check.py',
            '--output', '/evidence/binding.json']
        result['command'] = command
        result['cell_started'] = time.time()
        owned = True
        try:
            with (out/'memcheck.log').open('x') as log:
                checked = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            result['sanitizer_exit_code'] = checked.returncode
            text = (out/'memcheck.log').read_text()
            result['log_sha256'] = hashlib.sha256((out/'memcheck.log').read_bytes()).hexdigest()
            result['api_lookup_error_count'] = len(re.findall(
                r'^=+ Program hit CUDA_ERROR_INVALID_VALUE.*cuGetProcAddress_v2\.', text, re.M))
            # Record expected diagnostics without ever treating nonzero as pass.
            if checked.returncode:
                raise RuntimeError('binding diagnostic sanitizer exit '+str(checked.returncode))
            result['sanitizer_summary'] = sanitizer.validate_summary(out/'memcheck.log', 'memcheck')
            inner = json.loads((out/'binding.json').read_text())
            if inner.get('verdict') != 'OBSERVED' or inner.get('performance_acceptance') is not False:
                raise RuntimeError('binding diagnostic did not complete its API observations')
        finally:
            result['cell_ended'] = time.time()
            cleanup()
            save('progress.json', result)

    try:
        lifecycle.check_holder()
        lifecycle.pinned(str(root), args.revision)
        receipt = sanitizer.preflight(lifecycle.IMAGE, out/'sanitizer-preflight.json')
        save('resources-before.json', resources(False))
        before = lifecycle.snapshot()
        save('before.json', before)
        mode = lifecycle.validate_before(before)
        result['incoming_mode'] = mode
        if mode in ('present', 'stopped'):
            if mode == 'present':
                lifecycle.idle(before['local']['port'])
            lifecycle.with_paused(before, run, save, before_restore=cleanup)
            if mode == 'present' and before['local']['port'] != 8000:
                lifecycle.restore_public(out, save, result)
        else:
            try:
                run()
            finally:
                previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    cleanup()
                    lifecycle.restore_public(out, save, result)
                finally:
                    signal.signal(signal.SIGTERM, previous)
        result['exit_code'] = 0
    except BaseException as exc:
        result['error'] = repr(exc)
    finally:
        result['ended'] = time.time()
        result['restored_original'] = (out/'restored.json').exists()
        save('completion.json', result)
        print(json.dumps(result), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
