#!/usr/bin/env python3
"""Run exact EP decode compilation in a bounded, no-device CPU container."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import glm53_ep_capsule_runtime as capsule_runtime


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
    for target,digest in result['mounted_sources'].items():
        source=root/'build/glm53'/Path(target).name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,target
    return 0

if __name__=='__main__':raise SystemExit(main())
