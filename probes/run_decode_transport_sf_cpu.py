#!/usr/bin/env python3
"""CPU contracts and production compiler gates; no GPU, serving or timings.

Run on srv2 through fleet.sh run --cpu. Each container uses runc, no device
visibility, no network, two CPU cores and a fixed memory/swap cap. Source is
read-only and every compiler uses a fresh evidence-owned cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'


def moe_mounts():
    names = {path.name for path in (ROOT/'overlay/modules/glm53_moe').glob('*.py')}
    mounts = []
    for line in (ROOT/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        source, target, _ = line.split('\t')
        if source in names:
            mounts += ['--mount', f'type=bind,src={ROOT}/build/glm53/{source},dst={target},readonly']
            names.remove(source)
    assert not names, ('unmounted MoE sources', sorted(names))
    return mounts


def stages():
    yield 'contracts', ['bench/cpu_checks.py', '--suite', 'core',
        '--test', 'tests/test_ar_transport_next.py',
        '--test', 'tests/test_megakernel_ar_consumer_regressions.py',
        '--test', 'tests/test_moe_reform_sf_pack.py',
        '--test', 'tests/test_decode_next_cpu_runner.py',
        '--test', 'tests/test_moe_sf6_owner.py',
        '--test', 'tests/test_moe_sf6_dispatch.py',
        '--test', 'tests/test_moe_static_sf6_direct.py',
        '--test', 'tests/test_moe_dynamic_sf6.py',
        '--out', '/evidence/result.json'], '6g', False
    for compact, inline in (('0','0'), ('1','0'), ('0','1'), ('1','1')):
        yield 'transport-'+compact+inline, ['probes/decode_transport_compile.py',
            '--compact', compact, '--inline', inline,
            '--out', '/evidence/result.json'], '6g', False
    for rows in (2, 6, 8, 16):
        yield 'sf-m'+str(rows), ['probes/b12x_static_compile_check.py',
            '--specs', 't,r|t,r,sf6', '--m', str(rows), '--max-rows', '640',
            '--dynamic', 'tiled' if rows == 16 else ''], '3g', True
    yield 'sf-compat', ['probes/b12x_static_compile_check.py',
        '--specs', 'u|v|t|t,q', '--m', '8', '--max-rows', '640'], '3g', True
    yield 'sf-expand', ['probes/moe_reform_sf6_check.py', '--cpu',
        '--out', '/evidence/result.json'], '3g', True
    yield 'sf-unpack-codegen', ['probes/sf6_unpack_compile.py', '--cpu',
        '--out', '/evidence/result.json'], '3g', True
    for tile_m in (128,):
        yield 'sf-direct-tm'+str(tile_m), ['probes/b12x_static_compile_check.py',
            '--specs', 't,r,sf6', '--m', '80', '--max-rows', '640',
            '--dynamic', 'sf6', '--tile-m', str(tile_m)], '3g', True
    yield 'sf-wrapper', ['probes/sf6_wrapper_cpu_check.py', '--cpu',
        '--out', '/evidence/result.json'], '3g', True


def option(payload, name, default=''):
    return payload[payload.index(name) + 1] if name in payload else default


def validate_compile_log(payload, text):
    """A generic PASS cannot substitute for every requested static/dynamic arm."""
    expected = [value.strip() for value in option(payload, '--specs').split('|') if value.strip()]
    assert expected and len(expected) == len(set(expected)), ('invalid requested specs', expected)
    actual = re.findall(r'^\[([^\]]+)\] compiled in ', text, re.M)
    dynamic = option(payload, '--dynamic')
    expected += {'': [], 'rowmajor': ['dynamic tiled=False'],
                 'tiled': ['dynamic tiled=True'],
                 'both': ['dynamic tiled=False', 'dynamic tiled=True'],
                 'sf6': ['dynamic tiled=True sf6=True tm='+option(payload,'--tile-m','128')]}[dynamic]
    assert sorted(actual) == sorted(expected), ('requested kernels did not all compile', expected, actual)
    assert re.search(r'^VERDICT: PASS\s*$', text, re.M), 'missing compiler verdict'


def validate_stage(stage, payload, output):
    if stage == 'contracts':
        result = json.loads((output/'result.json').read_text())
        assert result['passed'] is True and result['coverage_complete'] is True, result
    elif stage.startswith('transport-') or stage in ('sf-expand', 'sf-wrapper', 'sf-unpack-codegen'):
        result = json.loads((output/'result.json').read_text())
        assert result['status'] == 'PASS', result
        if stage == 'sf-unpack-codegen':
            assert result['mode'] == 'cpu' and result['cuda_initialized'] is False, result
            assert result['evidence'] == 'isolated-unpack-compile-only', result
            assert sorted((c['words'], c['arm']) for c in result['cases']) == [
                (words, arm) for words in (1, 4, 8) for arm in ('scalar', 'u8x4')], result
            for case in result['cases']:
                assert set(case['artifacts']) == {'ptx', 'cubin', 'sass'}, case
                for artifact in case['artifacts'].values():
                    path = output / artifact['path']
                    assert path.parent == output and path.is_file(), artifact
                    contents = path.read_bytes()
                    assert contents and hashlib.sha256(contents).hexdigest() == artifact['sha256'], artifact
        if stage.startswith('transport-'):
            assert result['evidence'] == 'compile-only', result
            assert result['modes'] == [int(option(payload, '--compact')),
                                       int(option(payload, '--inline')), 0, 0, 0], result
            assert result['cuda_initialized'] is False, result
        if stage == 'sf-wrapper':
            assert result['mode'] == 'cpu' and result['cuda_initialized'] is False, result
            assert [(c['rows'], c['backend'], c['repeats'], c['status']) for c in result['cases']] == [
                (6,'static',3,'PASS'), (16,'static',3,'PASS'),
                (513,'dynamic',3,'PASS'), (4096,'dynamic',3,'PASS')], result
            assert result['rejected_invalid_cases'] == 4, result
    else:
        validate_compile_log(payload, (output/'stdout.log').read_text())


def write_report(report, path):
    selected = report['selected_stages']
    complete_selected = (len(report['stages']) == len(selected)
                         and {item['name'] for item in report['stages']} == set(selected)
                         and all(item.get('passed') is True
                                 and item.get('cleanup', {}).get('passed') is True
                                 for item in report['stages']))
    report['selected_passed'] = complete_selected
    report['coverage_complete'] = complete_selected and set(selected) == set(report['expected_stages'])
    report['passed'] = report['coverage_complete']
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(path)


def cleanup_container(name):
    """Stop only this runner's container; retain failure and bounded fallback."""
    attempts = []
    for command in (['docker','stop','-t','1',name], ['docker','rm','-f',name]):
        attempt = dict(command=command)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            attempt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            attempts.append(attempt)
            if result.returncode == 0 or 'no such container' in (result.stdout + result.stderr).lower():
                return dict(passed=True, attempts=attempts)
        except Exception as exc:
            attempt['error'] = repr(exc)
            attempts.append(attempt)
    return dict(passed=False, attempts=attempts)


def run_stage(stage, payload, command, name, output, report, report_path):
    record = dict(name=stage, command=command, passed=False)
    report['stages'].append(record)
    write_report(report, report_path)
    failure = None
    try:
        with (output/'stdout.log').open('w') as log:
            record['returncode'] = subprocess.run(command, stdout=log,
                stderr=subprocess.STDOUT, timeout=900).returncode
        if record['returncode'] != 0:
            raise RuntimeError(f'{stage} failed: {output}/stdout.log')
        validate_stage(stage, payload, output)
        record['passed'] = True
    except Exception as exc:
        failure = exc
        record['error'] = repr(exc)
    finally:
        if (output/'stdout.log').exists():
            record['log_sha256'] = hashlib.sha256((output/'stdout.log').read_bytes()).hexdigest()
        # Persist the original failure BEFORE cleanup, which can itself fail.
        write_report(report, report_path)
        try:
            record['cleanup'] = cleanup_container(name)
        except Exception as exc:
            record['cleanup'] = dict(passed=False, error=repr(exc))
        if not record['cleanup']['passed']:
            record['passed'] = False
        write_report(report, report_path)
    if failure is not None:
        raise failure
    if not record['cleanup']['passed']:
        raise RuntimeError(f'{stage} container cleanup failed: {report_path}')
    print('PASS '+stage, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', action='append', choices=[name for name, *_ in stages()])
    args = parser.parse_args()
    args.out = args.out.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip(), 'clean source required'
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    assert available >= 12 * 1024**3, ('need 12 GiB available before CPU gate', available)
    args.out.mkdir(parents=True, exist_ok=False)
    image = subprocess.check_output(['docker','image','inspect',IMAGE,'--format={{.Id}}'], text=True).strip()
    assert image == IMAGE, image
    report = dict(evidence='cpu-only', image=image,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        expected_stages=[name for name, *_ in stages()],
        selected_stages=[name for name, *_ in stages() if not args.stage or name in args.stage],
        passed=False, stages=[])
    write_report(report, args.out/'report.json')
    for stage, payload, memory, mounts in stages():
        if args.stage and stage not in args.stage:
            continue
        output = args.out/stage
        output.mkdir()
        name = f'decode-next-cpu-{os.getpid()}-{stage}'
        command = ['docker','run','--rm','--runtime=runc','--network=none','--name',name,
            '--cpuset-cpus=14-15', '--memory='+memory, '--memory-swap='+memory,
            '-e','NVIDIA_VISIBLE_DEVICES=void', '-e','CUDA_VISIBLE_DEVICES=',
            '-e','MAX_JOBS=1', '-e','OMP_NUM_THREADS=1', '-e','MKL_NUM_THREADS=1',
            '-e','CUTE_DSL_ARCH=sm_121a', '-e','MK_PKG_PATH=/usr/local/lib/python3.12/dist-packages',
            '-e','VLLM_CACHE_ROOT=/evidence/cache', '-e','FLASHINFER_WORKSPACE_BASE=/evidence/flashinfer',
            '-e','VLLM_DSV4_OSAR_BUILD_ROOT=/evidence/build',
            '--mount',f'type=bind,src={ROOT},dst=/repo,readonly',
            '--mount',f'type=bind,src={output},dst=/evidence',
            '--mount','type=bind,src=/usr/bin/git,dst=/usr/bin/git,readonly',
            '--mount','type=bind,src=/usr/lib/git-core,dst=/usr/lib/git-core,readonly',
            '-e','GIT_CONFIG_COUNT=1', '-e','GIT_CONFIG_KEY_0=safe.directory',
            '-e','GIT_CONFIG_VALUE_0=/repo']
        if mounts:
            command += moe_mounts()
        command += ['--workdir','/repo','--entrypoint','python3',IMAGE,*payload]
        run_stage(stage, payload, command, name, output, report, args.out/'report.json')
    assert report['stages'], 'no stages selected'
    write_report(report, args.out/'report.json')
    label = 'complete CPU gates' if report['coverage_complete'] else 'selected CPU stages (full coverage incomplete)'
    print('PASS '+label+'; device numerics and speed remain unmeasured', flush=True)


if __name__ == '__main__':
    main()
