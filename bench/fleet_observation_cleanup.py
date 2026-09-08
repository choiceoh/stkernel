#!/usr/bin/env python3
"""Remove this holder's diagnostic clones before public restore or handoff.

Runs in the frozen fleet supervisor after the payload exits, including SIGKILL.
It has no dependency on the payload, probe checkout, HTTP API or model runtime.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

NODES=('local','10.10.10.1','10.10.10.3','10.10.10.4')
SCRIPT=r'''
import json,subprocess
name='glm53-observe-'+session
names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
if name in names:
    c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    if (c['Config'].get('Labels') or {}).get('codex.glm.prefill-observation')!=session:
        raise RuntimeError('refusing foreign observation container')
    subprocess.run(['docker','rm','-f',c['Id']],check=True,stdout=subprocess.DEVNULL,timeout=75)
    names=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
    if name in names:raise RuntimeError('observation container still present')
print(json.dumps(dict(removed=True)))
'''


def cleanup(session,call=subprocess.run):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',session):raise ValueError('invalid session')
    def one(node):
        code='session='+repr(session)+'\n'+SCRIPT
        cmd=['python3','-c',code]
        if node!='local':cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','choiceoh@'+node,shlex.join(cmd)]
        result=call(cmd,capture_output=True,text=True,timeout=100)
        if result.returncode or json.loads(result.stdout)!=dict(removed=True):
            raise RuntimeError('clone cleanup failed: '+node)
        return node
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,node) for node in NODES]
        errors=[]
        for future in futures:
            try:future.result()
            except Exception as exc:errors.append(str(exc))
    if errors:raise RuntimeError(str(errors))


def main():
    session=os.environ['FLEET_SESSION']
    holder=(Path(os.environ['FLEET_DIR'])/'holder').read_text().strip().split('|')
    if len(holder)!=7 or holder[0]!=session or holder[6]!='boot':raise RuntimeError('owned boot hold required')
    ancestor=os.getpid()
    while ancestor>1 and ancestor!=int(holder[1]):
        ancestor=int(re.search(r'^PPid:\s+(\d+)',Path(f'/proc/{ancestor}/status').read_text(),re.M)[1])
    if ancestor!=int(holder[1]):raise RuntimeError('fleet holder is not an ancestor')
    cleanup(session)
    print('FLEET_OBSERVATION_CLONES_REMOVED',flush=True)


if __name__=='__main__':main()
