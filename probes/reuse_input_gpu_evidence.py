#!/usr/bin/env python3
"""Reuse completed GPU gates only when their exact code and profile still match."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
source,out=map(Path,sys.argv[1:])
revision=(source/'source.commit').read_text().strip()
paths=('overlay/modules/glm53_megakernel/glm53_megakernel.cu',
       'overlay/modules/glm53_megakernel/glm53_megakernel.py',
       'probes/gemm_input_serving_gate.py','probes/gemm_input_reuse.py',
       'profiles/glm53.env')
hashes={}
for path in paths:
    previous=subprocess.check_output(['git','-C',str(root),'show',revision+':'+path])
    current=(root/path).read_bytes()
    assert current==previous, ('GPU evidence source changed',path)
    hashes[path]=hashlib.sha256(current).hexdigest()
cuda_sha=hashes[paths[0]]
files=[]
for name in ('production-gate','racecheck','memcheck'):
    record=json.loads((source/(name+'.json')).read_text())
    assert record['status']=='PASS' and record['source_sha256']==cuda_sha
    assert record['boot_gate'] and record['alternating_graph_replays']==40
    log=(source/(name+'.log')).read_text()
    if name=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log
    if name=='memcheck':assert 'ERROR SUMMARY: 0 errors' in log
    files.extend((name+'.json',name+'.log'))
out.mkdir(parents=True,exist_ok=True)
for name in files:shutil.copyfile(source/name,out/name)
(out/'gpu-source.commit').write_text(revision+'\n')
(out/'gpu-evidence-reuse.json').write_text(json.dumps({
    'source_directory':str(source),'source_commit':revision,'code_sha256':hashes,
    'artifacts_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
    'reason':'Main updated benchmark orchestration only; tested CUDA, driver, fixture and profile are byte-identical.'},indent=2)+'\n')
print('PASS exact-code GPU evidence reuse',revision)
