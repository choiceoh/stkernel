#!/usr/bin/env python3
"""Validate and reuse complete AR/MHC stage cohorts with exact identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from ar_consumer_probe import AR_OWNERSHIP_SIZES
from ar_consumer_gpu_identity import IMAGE, NODES, collect_runtime, runtime_key

ROOT = Path(__file__).resolve().parents[1]
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
AUTO_INPUTS = INPUTS + ('probes/reuse_ar_consumer_gpu_evidence.py',
                       'probes/ar_consumer_gpu_identity.py')
GROUPS = ('local-probe', 'local-memcheck', 'local-racecheck', 'probe', 'memcheck', 'racecheck')
RACECHECK_KERNELS = '(mk_|k_oneshot|ar_consumer_delay)'
# Reviewed orchestration-only migration: CUDA commands, cases, image and
# sanitizer options are unchanged. An arbitrary later runner edit is not equal.
LEGACY_RUNNER_EQUIVALENCE = {
    '312ca9166e94c55162d0000e412540be466591312a2536e4eb10135f96758e48':
        'dd7fef7a973aa5ec511df7beb770b473c4671f4e094abe55a10cac5ed50503dc',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def legacy_source_unchanged(revision):
    changed = _git('diff', '--name-only', revision, 'HEAD', '--', *INPUTS).splitlines()
    if changed == ['probes/run_ar_consumer_gpu.py']:
        name = changed[0]
        original = subprocess.check_output(['git', '-C', str(ROOT), 'show', revision + ':' + name])
        require(LEGACY_RUNNER_EQUIVALENCE.get(digest(original)) == digest((ROOT / name).read_bytes()),
                'GPU runner changed outside the reviewed cache-only migration')
    else:
        require(not changed, 'tested GPU inputs changed')


def source():
    """Content identity permits unrelated commits, never dirty tested inputs."""
    require(not _git('status', '--porcelain'), 'current source must be clean')
    revision = _git('rev-parse', 'HEAD')
    tree = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-rz', 'HEAD', '--', *AUTO_INPUTS])
    inputs = {}
    for entry in tree.split(b'\0'):
        if entry:
            metadata, name = entry.split(b'\t', 1)
            inputs[name.decode()] = digest(metadata)
    require(bool(inputs), 'tested source identity is empty')
    return {'revision': revision,
            'inputs_sha256': inputs,
            'source_sha256': {name: digest((ROOT / name).read_bytes()) for name in SOURCES}}


def group_stages(group):
    require(group in GROUPS, 'unknown GPU stage group: ' + str(group))
    ranks = (0,) if group.startswith('local-') else range(4)
    return tuple(group + '-rank' + str(rank) for rank in ranks)


def _record(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def _validate_cases(report, distributed):
    require(report.get('mhc_warmup_capture') == 'PASS', 'MHC warmup/capture incomplete')
    pre = report['mhc_pre_view_cases']
    require(isinstance(pre, list) and len(pre) == 6, 'MHC pre-view coverage incomplete')
    require(all(type(row['consumer']) is bool and row['passed'] is True for row in pre),
            'MHC pre-view case did not pass')
    require({(row['consumer'], row['input_value']) for row in pre} == {
        (early, value) for early in (False, True) for value in (.03125, 0., -.0625)},
        'MHC pre-view coverage mismatch')
    cases = report['cases']
    require(isinstance(cases, list) and len(cases) == 36, 'exact segment coverage incomplete')
    require(all(type(row['tokens']) is int and type(row['fp32_fn']) is bool
                and type(row['seed']) is int and row['pass'] is True
                and type(row['exact_outputs']) is int and row['exact_outputs'] == 6
                for row in cases), 'exact segment case did not pass')
    require({(row['tokens'], row['fp32_fn'], row['seed']) for row in cases} == {
        (t, fp32, seed) for t in (1, 2, 6, 8, 16, 32)
        for fp32 in (False, True) for seed in (17, 0, 29)}, 'exact segment coverage mismatch')
    ownership = report['ar_ownership_cases']
    expected = {(n, seed) for n in AR_OWNERSHIP_SIZES for seed in (17, 0, 29)} if distributed else set()
    require(isinstance(ownership, list) and len(ownership) == len(expected),
            'AR ownership coverage incomplete')
    require(all(type(row['elements']) is int and type(row['seed']) is int
                and row['pass'] is True for row in ownership), 'AR ownership case did not pass')
    require({(row['elements'], row['seed']) for row in ownership} == expected,
            'AR ownership coverage mismatch')


def _validate_sanitizer(name, entry, log):
    if 'memcheck' not in name and 'racecheck' not in name:
        return
    errors = re.findall(r'ERROR SUMMARY: (\d+) errors?\b', log)
    require(all(int(count) == 0 for count in errors), 'nonzero sanitizer error summary')
    require(not re.search(r'^=========\s+(?:Error:|Warning:|Fatal)', log, re.MULTILINE),
            'sanitizer diagnostic error or warning')
    if 'memcheck' in name:
        require(bool(errors), 'missing clean memcheck summary')
    else:
        require(entry.get('kernel_filter') == RACECHECK_KERNELS, 'racecheck kernel filter changed')
        hazards = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', log)
        require(bool(hazards) and all(all(int(count) == 0 for count in row) for row in hazards),
                'missing clean racecheck summary')


def validate_group(out, group, completed, runtime=None):
    """One validator admits both fresh runs and copied stage evidence.

    Return (ordered rank entries, artifact bytes, provenance). A distributed
    group must already have all four completed ranks from this one directory.
    """
    out = Path(out)
    stages = group_stages(group)
    require(isinstance(completed, list), 'completed receipts must be a list')
    all_stages = {stage for item in GROUPS for stage in group_stages(item)}
    names = [entry['stage'] for entry in completed]
    require(len(names) == len(set(names)) and set(names) <= all_stages,
            'duplicate or unknown completed stage')
    entries = {entry['stage']: entry for entry in completed if entry['stage'] in stages}
    require(set(entries) == set(stages), 'incomplete GPU cohort: ' + group)
    if runtime is not None:
        runtime_key(runtime, group)
    hashes = source()['source_sha256']
    artifacts = {}
    for name in stages:
        entry = entries[name]
        rank = int(name[-1])
        distributed = not group.startswith('local-')
        for suffix in ('.json', '.container.json', '.log'):
            artifacts[name + suffix] = (out / (name + suffix)).read_bytes()
        report = _record(artifacts[name + '.json'])
        container = _record(artifacts[name + '.container.json'])
        require(entry['node'] == NODES[rank] and type(report['rank']) is int
                and report['rank'] == rank, 'rank/node mismatch: ' + name)
        require(report['status'] == 'PASS'
                and report['mode'] == ('distributed' if distributed else 'delayed-producer'),
                'incomplete or wrong probe mode: ' + name)
        require(report['torch'] == '2.13.0+cu130' and report['cuda'] == '13.0'
                and report['device'] == 'NVIDIA GB10', 'probe runtime mismatch: ' + name)
        require(entry['source_sha256'] == report['source_sha256'] == hashes,
                'tested source hashes changed: ' + name)
        _validate_cases(report, distributed)
        state = container['state']
        require(type(state['ExitCode']) is int and state['ExitCode'] == 0
                and state['OOMKilled'] is False and container['image'] == IMAGE,
                'unclean container exit or image mismatch: ' + name)
        limit = (24 if 'racecheck' in name else 8) * 1024**3
        require(type(container['memory_limit']) is int and type(container['memory_swap_limit']) is int
                and container['memory_limit'] == container['memory_swap_limit'] == limit
                and container['cpus'] == '14-17', 'container execution limits changed: ' + name)
        _validate_sanitizer(name, entry, artifacts[name + '.log'].decode())
    provenance = {'source_directory': str(out.resolve()), 'group': group,
                  'stages_passed': len(stages),
                  'artifacts_sha256': {name: digest(value) for name, value in artifacts.items()}}
    return [entries[name] for name in stages], artifacts, provenance


def verify_group(source_dir, group, runtime, *, require_manifest=True):
    """Reuse one sealed cohort only under its original source/runtime identity."""
    source_dir = Path(source_dir)
    admission = _record((source_dir / 'admission.json').read_bytes())
    metadata = {'source.commit': (source_dir / 'source.commit').read_bytes()}
    revision = metadata['source.commit'].decode().strip()
    require(re.fullmatch('[0-9a-f]{40}', revision), 'missing exact GPU source commit')
    require(admission['revision'] == revision and admission['image'] == IMAGE,
            'GPU admission identity mismatch')
    current = source()
    if require_manifest:
        metadata.update({name: (source_dir / name).read_bytes()
                         for name in ('source.json', 'runtime.json')})
        manifest = admission.get('artifacts_sha256')
        require(isinstance(manifest, dict), 'missing artifact integrity manifest')
        require(all(manifest.get(name) == digest(value) for name, value in metadata.items()),
                'GPU identity metadata changed after publication')
        original = _record(metadata['source.json'])
        require(original['revision'] == revision, 'GPU source metadata revision mismatch')
        require(original['inputs_sha256'] == current['inputs_sha256']
                and original['source_sha256'] == current['source_sha256'], 'tested GPU inputs changed')
        previous_runtime = _record(metadata['runtime.json'])
        require(runtime_key(previous_runtime, group) == runtime_key(runtime, group),
                'GPU runtime/topology changed: ' + group)
    else:
        # Explicit legacy reuse preserves its historical source-only contract.
        legacy_source_unchanged(revision)
    entries, artifacts, provenance = validate_group(source_dir, group, admission['completed'], runtime)
    if require_manifest:
        require(all(manifest.get(name) == digest(value) for name, value in artifacts.items()),
                'GPU artifact changed after publication')
    provenance.update(source_commit=revision, serving_commit=current['revision'],
                      runtime_attested=require_manifest)
    return entries, artifacts, provenance


def verify(source, out=None):
    """Explicit historical all-stage reuse; it does not attest today's hosts."""
    source = Path(source)
    files = {name: (source / name).read_bytes() for name in ('source.commit', 'admission.json')}
    revision = files['source.commit'].decode().strip()
    admission = _record(files['admission.json'])
    require(len(admission['completed']) == 15, 'all 15 GPU gates required for legacy reuse')
    for group in GROUPS:
        _, artifacts, _ = verify_group(source, group, None, require_manifest=False)
        files.update(artifacts)
    hashes = {name: digest((ROOT / name).read_bytes()) for name in SOURCES}
    result = dict(source_directory=str(source.resolve()), source_commit=revision,
                  serving_commit=_git('rev-parse', 'HEAD'), stages_passed=15, source_sha256=hashes,
                  runtime_attested=False, legacy=True,
                  artifacts_sha256={name: digest(value) for name, value in files.items()})
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
