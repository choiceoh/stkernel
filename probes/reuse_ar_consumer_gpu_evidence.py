#!/usr/bin/env python3
"""Reuse all 15 AR/MHC GPU gates only for unchanged tested code and profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from ar_consumer_probe import AR_OWNERSHIP_SIZES

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
SOURCES = (
    'overlay/modules/glm53_megakernel/glm53_megakernel.cu',
    'overlay/modules/glm53_megakernel/glm53_megakernel.py',
    'overlay/modules/tp_oneshot_ar/dsv4_oneshot_ar.cu',
    'overlay/modules/tp_oneshot_ar/dsv4_oneshot_shim.py',
    'probes/ar_consumer_delay.cu',
)
INPUTS = ('overlay', 'build', 'profiles', 'probes/ar_consumer_probe.py',
          'probes/run_ar_consumer_gpu.py', 'probes/ar_consumer_delay.cu',
          'probes/mhc_reuse_bench.py', 'probes/megakernel_glm53_bench.py')
NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')


def verify(source, out=None):
    files = {}
    def read(name):
        if name not in files:
            files[name] = (source / name).read_bytes()
        return files[name]
    def record(name):
        return json.loads(read(name))
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

    revision = read('source.commit').decode().strip()
    assert re.fullmatch('[0-9a-f]{40}', revision), 'missing exact GPU source commit'
    assert not git('status', '--porcelain'), 'current source must be clean'
    assert not git('diff', '--name-only', revision, 'HEAD', '--', *INPUTS), 'tested GPU inputs changed'
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    admission = record('admission.json')
    assert admission['revision'] == revision and admission['image'] == IMAGE
    expected_stages = {'local-' + stage + '-rank0' for stage in ('probe', 'memcheck', 'racecheck')}
    expected_stages |= {stage + '-rank' + str(rank)
                        for stage in ('probe', 'memcheck', 'racecheck') for rank in range(4)}
    completed = admission['completed']
    assert len(completed) == 15 and {r['stage'] for r in completed} == expected_stages
    for entry in completed:
        name = entry['stage']
        rank = int(name[-1])
        distributed = not name.startswith('local-')
        r = record(name + '.json')
        c = record(name + '.container.json')
        assert entry['node'] == NODES[rank] and r['rank'] == rank
        assert r['status'] == 'PASS' and r['mode'] == ('distributed' if distributed else 'delayed-producer')
        assert r['torch'] == '2.13.0+cu130' and r['cuda'] == '13.0' and r['device'] == 'NVIDIA GB10'
        assert entry['source_sha256'] == r['source_sha256'] == hashes
        assert r['mhc_warmup_capture'] == 'PASS'
        assert len(r['mhc_pre_view_cases']) == 6
        assert {(v['consumer'], v['input_value']) for v in r['mhc_pre_view_cases'] if v['passed']} == {
            (early, value) for early in (False, True) for value in (.03125, 0., -.0625)}
        assert len(r['cases']) == 36
        assert {(v['tokens'], v['fp32_fn'], v['seed']) for v in r['cases'] if v['pass'] and v['exact_outputs'] == 6} == {
            (t, fp32, seed) for t in (1, 2, 6, 8, 16, 32) for fp32 in (False, True) for seed in (17, 0, 29)}
        ownership = {(n, seed) for n in AR_OWNERSHIP_SIZES for seed in (17, 0, 29)} if distributed else set()
        assert len(r['ar_ownership_cases']) == len(ownership)
        assert {(v['elements'], v['seed']) for v in r['ar_ownership_cases'] if v['pass']} == ownership
        assert c['state']['ExitCode'] == 0 and not c['state']['OOMKilled'] and c['image'] == IMAGE
        limit = 24 if 'racecheck' in name else 8
        assert c['memory_limit'] == c['memory_swap_limit'] == limit * 1024**3 and c['cpus'] == '14-17'
        log = read(name + '.log').decode()
        if 'memcheck' in name:
            assert 'ERROR SUMMARY: 0 errors' in log
        if 'racecheck' in name:
            assert entry['kernel_filter'] == '(mk_|k_oneshot|ar_consumer_delay)'
            assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log
    result = dict(source_directory=str(source.resolve()), source_commit=revision,
                  serving_commit=git('rev-parse', 'HEAD'), stages_passed=15, source_sha256=hashes,
                  artifacts_sha256={name: hashlib.sha256(value).hexdigest() for name, value in files.items()})
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
        for name, value in files.items():
            (out / name).write_bytes(value)
        (out / 'gpu-evidence-reuse.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('out', type=Path, nargs='?')
    args = ap.parse_args()
    print(json.dumps(verify(args.source, args.out), sort_keys=True))
