#!/usr/bin/env python3
"""Read-only all-rank proof after the maintenance runner restores main."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request

IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
KNOBS = {'VLLM_GLM53_MK_INPUT_CTA':'2', 'VLLM_GLM53_MK_INPUT_REUSE':'1',
         'VLLM_GLM53_B12X_STATIC_V2':'t'}
FILES = {
    'overlay/modules/glm53_megakernel/glm53_megakernel.cu':
        '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_megakernel.cu',
    'overlay/modules/glm53_moe/moe_static_kernel_v4.py':
        '/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel_v4.py',
}
for name in ('moe_dispatch.py','moe_static_kernel_v5.py'):
    FILES['overlay/modules/glm53_moe/'+name] = (
        '/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/'+name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--restore-repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as reply:
        assert reply.status == 200
    commit = subprocess.check_output(['git','-C',str(args.restore_repo),'rev-parse','HEAD'],text=True).strip()
    approved = subprocess.check_output(['git','-C',str(args.restore_repo),'rev-parse','origin/main'],text=True).strip()
    assert commit == approved
    expected = {target:hashlib.sha256((args.restore_repo/source).read_bytes()).hexdigest()
                for source,target in FILES.items()}
    rows = []
    for rank,node in enumerate((2,1,3,4)):
        name = 'glm53' if rank == 0 else 'glm53-worker'
        # Inspect only selected fields in the remote process; unrelated env
        # values (including any credentials) never enter the evidence.
        code = ('import json,subprocess; '
                f'j=json.loads(subprocess.check_output(["docker","inspect",{name!r}],text=True))[0]; '
                'e=dict(x.split("=",1) for x in j["Config"]["Env"]); '
                f'print(json.dumps(dict(id=j["Id"],image=j["Image"],running=j["State"]["Running"],'
                f'knobs={{k:e.get(k) for k in {list(KNOBS)!r}}})))')
        def run(command):
            if node != 2:
                command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
                         f'choiceoh@10.10.10.{node}',shlex.join(command)]
            return subprocess.check_output(command,text=True,timeout=30)
        row=json.loads(run(['python3','-c',code]))
        assert row['image'] == IMAGE and row['running'] and row['knobs'] == KNOBS,row
        hashes={line.split()[1]:line.split()[0] for line in
                run(['docker','exec',name,'sha256sum',*expected]).splitlines()}
        assert hashes == expected,hashes
        rows.append(dict(rank=rank,host=f'srv{node}',source_sha256=hashes,**row))
    report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),health=200,
                approved_main=commit,image=IMAGE,knobs=KNOBS,ranks=rows)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS HTTP 200, all four ranks, approved-main source hashes and default knobs')


if __name__ == '__main__':
    main()
