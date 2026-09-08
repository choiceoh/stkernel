"""Read distribution requirements in a bounded, no-device pinned container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
INNER = r'''
import hashlib,importlib.metadata as md,json,pathlib,sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
devices=sorted(str(p) for p in pathlib.Path('/dev').glob('nvidia*'))
assert not devices, devices
distributions=list(md.distributions())
versions={canonicalize_name(d.metadata['Name']):d.version for d in distributions}
focus={'cuda-bindings','cuda-python','cuda-core','cuda-pathfinder','nvidia-cutlass-dsl','flashinfer-python','vllm','torch'}
records=[]
for d in distributions:
    name=canonicalize_name(d.metadata['Name'])
    dependencies=[]
    for raw in d.requires or []:
        req=Requirement(raw)
        dep=canonicalize_name(req.name)
        if dep not in focus: continue
        active=req.marker is None or req.marker.evaluate({'extra':''})
        current=versions.get(dep)
        selected='13.0.3' if dep in {'cuda-bindings','cuda-python'} else current
        dependencies.append(dict(requirement=raw,name=dep,active=active,current=current,
            current_satisfies=current is not None and req.specifier.contains(current,prereleases=True),
            pair13_0_3_satisfies=selected is not None and req.specifier.contains(selected,prereleases=True)))
    if name in focus or any(x['name'] in {'cuda-bindings','cuda-python'} for x in dependencies):
        metadata=d.read_text('METADATA')
        records.append(dict(name=name,version=d.version,path=str(d._path),
            metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
            requires_dist=d.requires or [],relevant_requirements=dependencies))
result=dict(scope='Installed metadata only; no Torch/CUDA imports or package changes',
    distribution_count=len(distributions),python=sys.version,machine=__import__('platform').machine(),
    exposed_device_nodes=devices,imported_accelerator_modules=[n for n in sys.modules if n=='torch' or n=='cuda' or n.startswith(('torch.','cuda.'))],
    installed_focus={k:versions.get(k) for k in sorted(focus)},records=records)
assert not result['imported_accelerator_modules']
pathlib.Path('/evidence/metadata.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
args = ap.parse_args()
available = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
if available < 12*1024*1024:
    raise SystemExit('Existing 12 GiB host memory guard; no metadata container launched')
out = args.out.resolve()
out.mkdir(exist_ok=False)
command = ['docker','run','--rm','--runtime=runc','--network=none','--memory=512m',
    '--memory-swap=512m','--cpus=1','--pids-limit=64','-e','NVIDIA_VISIBLE_DEVICES=void',
    '-e','CUDA_VISIBLE_DEVICES=','--entrypoint=python3',
    '--mount',f'type=bind,source={out},target=/evidence',IMAGE,'-c',INNER]
receipt = dict(started=time.time(),image=IMAGE,host_mem_available_kib=available,
    command=command,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    with (out/'container.log').open('x') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    receipt['exit_code'] = result.returncode
    if result.returncode:
        raise RuntimeError('metadata container failed')
finally:
    receipt['ended'] = time.time()
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'exit_code':receipt['exit_code'],'output':str(out),'image':IMAGE}))
