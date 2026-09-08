#!/usr/bin/env python3
"""Run exact EP decode compilation in a bounded, no-device CPU container."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import glm53_ep_capsule_runtime as capsule_runtime
from glm53_ep_short_decode_compile import source_receipt


def validate_artifacts(output, result):
    output = output.resolve(strict=True)
    expected_paths = set()

    def check(relative, digest, suffix):
        path = Path(relative)
        assert (not path.is_absolute() and path.parts and '..' not in path.parts
                and str(path) == relative and path.suffix == suffix), relative
        absolute = output/path
        assert absolute.resolve(strict=True) == absolute and absolute.is_file(), relative
        assert relative not in expected_paths, 'duplicate artifact: '+relative
        content = absolute.read_bytes()
        assert content and hashlib.sha256(content).hexdigest() == digest, relative
        expected_paths.add(relative)
        return absolute

    for name, suffix in (('micro_artifacts', '.ptx'), ('micro_resources', '.cubin')):
        assert len(result[name]) >= 2, 'both micro kernels need artifacts'
        for row in result[name]:
            assert not row['path'].startswith('prepare/'), row['path']
            path = check(row['path'], row['sha256'], suffix)
            if name == 'micro_resources':
                assert path.with_suffix('.resources.log').read_text() == row['resources'], row['path']
    expected_labels = {
        '-'.join((kind, ids, weight, mapping))
        for kind in ('mapped', 'empty', 'offset')
        for ids, weight, mapping in itertools.product(
            ('i32', 'i64'), ('fp32', 'fp16', 'bf16'),
            ('i32', 'i64') if kind == 'mapped' else ('i32',))}
    variants = result['prepare_variants']
    assert len(variants) == len(expected_labels) and {row['label'] for row in variants} == expected_labels
    for row in variants:
        for suffix in ('.ptx', '.cubin'):
            check('prepare/'+row['label']+'/kernel'+suffix, row[suffix[1:]+'_sha256'], suffix)
    actual_paths = {str(path.relative_to(output)) for suffix in ('*.ptx', '*.cubin')
                    for path in output.rglob(suffix)}
    assert actual_paths == expected_paths, 'compiled artifact file set differs'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capsule-root',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    a=p.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.image):p.error('immutable local image ID required')
    root=Path(__file__).resolve().parents[1]
    capsule=capsule_runtime.validate_capsule_input(a.capsule_root,a.manifest_sha256)
    output=a.output.resolve()
    if any(output.is_relative_to(x) or x.is_relative_to(output) for x in (root,capsule)):
        p.error('output must be separate from source and capsule')
    if output.exists():p.error('fresh output directory required')
    available=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<12*1024*1024:p.error('CPU compile needs 12 GiB available; serving memory is not reclaimed')
    expected_sources=source_receipt(root)
    output.mkdir(parents=True)
    command=['docker','run','--rm','--runtime=runc','--network=none','--memory=4g','--memory-swap=4g','--cpus=2','--pids-limit=128',
        '-e','NVIDIA_VISIBLE_DEVICES=void','-e','MAX_JOBS=1','--entrypoint=python3','-v',f'{root}:/repo:ro','-v',f'{output}:/evidence']
    for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,target,*_=line.split('\t')
        if '/flashinfer/' in target or name=='flashinfer_b12x_moe.py':
            command+=['-v',f'{root / "build/glm53" / name}:{target}:ro']
    command+=capsule_runtime.docker_capsule_args(capsule,a.manifest_sha256)
    command += [a.image,'-B','/repo/probes/glm53_ep_short_decode_compile.py','--output','/evidence',
        '--capsule-root',capsule_runtime.CAPSULE_MOUNT,'--manifest-sha256',a.manifest_sha256]
    completed=subprocess.run(command)
    capsule_runtime.validate_capsule_input(capsule,a.manifest_sha256)
    if completed.returncode:return completed.returncode
    result=json.loads((output/'result.json').read_text())
    assert result['verdict']=='PASS' and result['phase']=='complete' and result['cuda_initialized'] is False
    assert result['binding_runtime_rechecked'] is True
    capsule_runtime.validate_runtime_receipt(result['binding_runtime'])
    assert len(result['micro_keys'])==2 and len(result['prepare_variants'])==24
    assert result['contracts']['tests_run']>0 and not any(result['contracts'][k] for k in ('failures','errors','skips'))
    assert source_receipt(root)==expected_sources,'compile/test source changed'
    for name, expected in expected_sources.items():
        assert result.get(name)==expected,'missing or changed source receipt: '+name
    validate_artifacts(output,result)
    return 0

if __name__=='__main__':raise SystemExit(main())
