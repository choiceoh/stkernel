#!/usr/bin/env python3
"""Normal CPU fleet payload: four EP tiled static and two dynamic lowerings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import glm53_ep_capsule_runtime as capsule_runtime
from glm53_ep_tiled_compile import source_receipt, STATIC_ROWS, DYNAMIC_ROWS


def validate_artifacts(output,result):
    output=output.resolve(strict=True)
    expected=set()
    for kind,rows in (('static',STATIC_ROWS),('dynamic',DYNAMIC_ROWS)):
        passes=result[kind+'_passes']
        assert [p['arm'] for p in passes]==[kind+'/M'+str(m) for m in rows]
        for passed in passes:
            for name,suffix in (('artifacts','.ptx'),('resources','.cubin')):
                assert len(passed[name])==1
                for row in passed[name]:
                    rel=Path(row['path'])
                    assert not rel.is_absolute() and '..' not in rel.parts and str(rel)==row['path']
                    assert str(rel.parent)==passed['arm'] and rel.suffix==suffix
                    path=output/rel
                    assert path.resolve(strict=True)==path and path.is_file()
                    assert str(rel) not in expected
                    raw=path.read_bytes()
                    assert raw and hashlib.sha256(raw).hexdigest()==row['sha256']
                    expected.add(str(rel))
                    if suffix=='.cubin':
                        assert path.with_suffix('.resources.log').read_text()==row['resources']
    actual={str(p.relative_to(output)) for suffix in ('*.ptx','*.cubin') for p in output.rglob(suffix)}
    assert actual==expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capsule-root',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    a=p.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',a.image):p.error('immutable image ID required')
    root=Path(__file__).resolve().parents[1]
    capsule=capsule_runtime.validate_capsule_input(a.capsule_root,a.manifest_sha256)
    output=a.output.resolve()
    if any(output.is_relative_to(x) or x.is_relative_to(output) for x in (root,capsule)):
        p.error('output must be separate from source/capsule')
    if output.exists():p.error('fresh evidence directory required')
    available=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available<12*1024*1024:p.error('12 GiB available required; no serving memory reclaim')
    sources=source_receipt(root)
    output.mkdir(parents=True)
    command=['docker','run','--rm','--runtime=runc','--network=none','--memory=4g',
        '--memory-swap=4g','--cpus=2','--pids-limit=128','-e','NVIDIA_VISIBLE_DEVICES=void',
        '-e','MAX_JOBS=1','--entrypoint=python3','-v',f'{root}:/repo:ro','-v',f'{output}:/evidence']
    for line in (root/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,target,*_=line.split('\t')
        if '/flashinfer/' in target or name in ('flashinfer_b12x_moe.py','gpu_worker.py'):
            command+=['-v',f'{root/"build/glm53"/name}:{target}:ro']
    command+=capsule_runtime.docker_capsule_args(capsule,a.manifest_sha256)
    command += [a.image,'-B','/repo/probes/glm53_ep_tiled_compile.py','--output','/evidence',
        '--capsule-root',capsule_runtime.CAPSULE_MOUNT,'--manifest-sha256',a.manifest_sha256]
    completed=subprocess.run(command)
    capsule_runtime.validate_capsule_input(capsule,a.manifest_sha256)
    if completed.returncode:return completed.returncode
    result=json.loads((output/'result.json').read_text())
    assert result['verdict']=='PASS' and result['phase']=='complete' and result['cuda_initialized'] is False
    assert result['binding_runtime_rechecked'] is True and result['compile_only'] is True
    assert not any(k in result for k in ('error','cleanup_error','recheck_error'))
    capsule_runtime.validate_runtime_receipt(result['binding_runtime'])
    assert source_receipt(root)==sources
    for key,value in sources.items():assert result[key]==value
    validate_artifacts(output,result)
    return 0


if __name__=='__main__':raise SystemExit(main())
