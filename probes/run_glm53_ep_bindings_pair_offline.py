#!/usr/bin/env python3
"""One normal fleet pause: two fresh binding diagnostics and exact restore.

A clean candidate can establish only minimal compatibility. Baseline failures
remain failures, including the process exit status; no full GPU/MoE acceptance
or performance result is produced by this runner.
"""
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
from glm53_ep_bindings_capsule import validate_capsule
from run_glm53_ep_local_offline import resources

CAPSULE = '/opt/glm53-bindings-capsule'
VERSIONS = {'baseline': '13.3.1', 'candidate': '13.0.3'}


def v5_error_window(text):
    """Require all 34 lookup reports inside the unique first-count window."""
    lines = text.splitlines()
    markers = ('EP_BINDING_CONTEXT_BEGIN', 'EP_BINDING_CONTEXT_READY',
               'EP_BINDING_IDENTITY_VERIFIED', 'EP_BINDING_DEVICE_COUNT_BEGIN',
               'EP_BINDING_DEVICE_COUNT_END')
    positions = []
    for marker in markers:
        found = [index for index, line in enumerate(lines) if line == marker]
        if len(found) != 1:
            return False
        positions.append(found[0])
    if positions != sorted(positions):
        return False
    errors = [index for index, line in enumerate(lines)
              if re.match(r'^=+ Program hit ', line)]
    return len(errors) == 34 and all(positions[-2] < index < positions[-1] for index in errors)


def observation_valid(inner, arm, manifest_sha256):
    return (inner.get('verdict') == 'OBSERVED' and inner.get('arm') == arm
            and inner.get('performance_acceptance') is False
            and inner.get('full_gpu_acceptance') is False
            and inner.get('capsule_manifest_sha256') == manifest_sha256
            and inner.get('cuda_bindings') == VERSIONS[arm]
            and inner.get('cuda_initialized_before') is False
            and inner.get('cuda_initialized_after') is True
            and inner.get('count_result') == [0, 1]
            and inner.get('version_result') == [0, 13000]
            and inner.get('binding_identity', {}).get('version') == VERSIONS[arm])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--capsule-root', type=Path, required=True)
    ap.add_argument('--manifest-sha256', required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        ap.error('full frozen source revision required')
    if not re.fullmatch(r'[0-9a-f]{64}', args.manifest_sha256):
        ap.error('externally pinned capsule manifest SHA256 required')
    root = Path(__file__).resolve().parents[1]
    capsule, out = args.capsule_root.resolve(strict=True), args.out.resolve()
    if (out.is_relative_to(root) or out.is_relative_to(capsule)
            or any(c in str(path) for path in (root, capsule, out) for c in (',', '\n', '\r'))):
        ap.error('evidence must be outside source/capsule and mount paths must be literal')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(started=time.time(), source_revision=args.revision, exit_code=1,
                  capsule_manifest_sha256=args.manifest_sha256, diagnostic_only=True,
                  performance_acceptance=False, full_gpu_acceptance=False,
                  pair_completed=False, verdict='FAIL', cells=[])
    owned = set()

    def save(filename, value):
        (out/filename).write_text(json.dumps(value, indent=2)+'\n')

    def interrupted(signum, frame):
        raise InterruptedError('termination requested')

    signal.signal(signal.SIGTERM, interrupted)

    def cleanup():
        lifecycle.check_holder()
        for name in sorted(owned):
            inspected = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True)
            if inspected.returncode:
                names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
                if name in names:
                    raise RuntimeError('owned binding-pair inspect failed')
                continue
            state = json.loads(inspected.stdout)[0]
            if (state['Image'] != lifecycle.IMAGE or state['Config'].get('Labels', {}).get(
                    'glm53.ep-bindings-pair.session') != os.environ['FLEET_SESSION']):
                raise RuntimeError('owned binding-pair container identity changed')
            subprocess.run(['docker', 'rm', '-f', state['Id']], check=True, timeout=45,
                           stdout=subprocess.DEVNULL)

    def cell(arm):
        # Each GO rechecks both source and capsule, including after baseline.
        lifecycle.check_holder()
        lifecycle.pinned(str(root), args.revision)
        validate_capsule(capsule, args.manifest_sha256)
        save('resources-'+arm+'.json', resources(True))
        states = lifecycle.snapshot()
        if any(state is not None and state['running'] for state in states.values()):
            raise RuntimeError('original serving must be paused on all four nodes')
        name = 'ep-bindings-'+arm+'-'+uuid.uuid4().hex
        names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
        if name in names:
            raise RuntimeError('binding-pair container name already exists')
        command = ['docker', 'run', '--rm', '--name', name, '--label',
                   'glm53.ep-bindings-pair.session='+os.environ['FLEET_SESSION'],
                   '--gpus', 'all', '--network=none', '--memory=4g', '--memory-swap=4g',
                   '--cpus=2', '--pids-limit=128', '--entrypoint=/usr/bin/env',
                   '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONNOUSERSITE=1',
                   '-e', 'PYTHONPATH='+(CAPSULE if arm == 'candidate' else ''),
                   '--mount', f'type=bind,source={root},target=/repo,readonly',
                   '--mount', f'type=bind,source={capsule},target={CAPSULE},readonly',
                   '--mount', f'type=bind,source={out},target=/evidence']
        command += sanitizer.mount_args(receipt)+[lifecycle.IMAGE]
        command += sanitizer.command('memcheck')+[
            'python3', '-B', '/repo/probes/glm53_ep_bindings_pair_check.py', '--arm', arm,
            '--capsule-root', CAPSULE, '--manifest-sha256', args.manifest_sha256,
            '--output', '/evidence/'+arm+'.json']
        entry = dict(arm=arm, version=VERSIONS[arm], command=command,
                     started=time.time(), verdict='FAIL', observation_valid=False)
        result['cells'].append(entry)
        owned.add(name)
        try:
            with (out/(arm+'.log')).open('x') as log:
                checked = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            entry['sanitizer_exit_code'] = checked.returncode
            log_path = out/(arm+'.log')
            text = log_path.read_text()
            entry['log_sha256'] = hashlib.sha256(log_path.read_bytes()).hexdigest()
            entry['api_lookup_error_count'] = len(re.findall(
                r'^=+ Program hit CUDA_ERROR_INVALID_VALUE.*cuGetProcAddress_v2\.', text, re.M))
            # Parse the observation even on error-exitcode 86. Keep that failure
            # and run the next fresh container; do not turn it into a PASS.
            inner = json.loads((out/(arm+'.json')).read_text())
            entry['observation_sha256'] = hashlib.sha256((out/(arm+'.json')).read_bytes()).hexdigest()
            entry['observation_valid'] = observation_valid(inner, arm, args.manifest_sha256)
            entry['matches_v5_reproducer'] = (
                arm == 'baseline' and entry['observation_valid'] and checked.returncode == 86
                and entry['api_lookup_error_count'] == 34
                and len(re.findall(r'^=+ Program hit ', text, re.M)) == 34
                and v5_error_window(text)
                and not re.search(r'^\s*=+\s*(?:ERROR|FATAL)\s*:', text, re.I | re.M)
                and re.findall(r'^=+ (ERROR SUMMARY:.*)$', text, re.M) == ['ERROR SUMMARY: 34 errors'])
            if checked.returncode:
                raise RuntimeError('binding diagnostic sanitizer exit '+str(checked.returncode))
            entry['sanitizer_summary'] = sanitizer.validate_summary(log_path, 'memcheck')
            if not entry['observation_valid'] or entry['api_lookup_error_count']:
                raise RuntimeError('binding identity/count/version observation is incomplete')
            entry['verdict'] = 'CLEAN_DIAGNOSTIC'
        except InterruptedError:
            raise
        except Exception as exc:
            entry['error'] = repr(exc)
        finally:
            entry['ended'] = time.time()
            cleanup()
            save('progress.json', result)

    def run():
        cell('baseline')
        cell('candidate')
        validate_capsule(capsule, args.manifest_sha256)
        result['capsule_unchanged_after_pair'] = True
        result['pair_completed'] = True
        baseline, candidate = result['cells']
        if ((baseline['verdict'] == 'CLEAN_DIAGNOSTIC' or baseline.get('matches_v5_reproducer'))
                and candidate['verdict'] == 'CLEAN_DIAGNOSTIC'):
            result['verdict'] = 'COMPATIBILITY_OBSERVED'
        # Nonzero baseline errors remain visible to fleet/process consumers.
        if all(c['verdict'] == 'CLEAN_DIAGNOSTIC' for c in result['cells']):
            result['diagnostics_exit_code'] = 0
        else:
            result['diagnostics_exit_code'] = 1

    try:
        lifecycle.check_holder()
        lifecycle.pinned(str(root), args.revision)
        manifest = validate_capsule(capsule, args.manifest_sha256)
        save('capsule-manifest.json', manifest)
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
        result['exit_code'] = result['diagnostics_exit_code']
    except BaseException as exc:
        result.update(error=repr(exc), verdict='FAIL')
    finally:
        result['ended'] = time.time()
        result['restored_original'] = (out/'restored.json').exists()
        save('completion.json', result)
        print(json.dumps(result), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
