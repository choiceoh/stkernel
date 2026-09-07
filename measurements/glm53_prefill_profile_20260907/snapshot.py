import concurrent.futures
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

SCRIPT = r'''
import base64,hashlib,json,pathlib,re,shutil,subprocess,time
name=next(n for n in subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines() if n.startswith('glm53'))
c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
env=dict(e.split('=',1) for e in c['Config']['Env'] if '=' in e)
cmd=' '.join(c['Config'].get('Cmd') or [])
p=re.search(r'echo ([A-Za-z0-9+/=]+) \| base64 -d',cmd)
if p:cmd=base64.b64decode(p.group(1),validate=True).decode()
mounts={m['Destination']: {'source':m['Source'],'sha256':hashlib.sha256(pathlib.Path(m['Source']).read_bytes()).hexdigest()} for m in c['Mounts'] if m['Source'].startswith('/home/choiceoh/overlays/glm53/')}
memory={l.split(':')[0]:int(l.split()[1]) for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.split(':')[0] in ['MemTotal','MemAvailable']}
print(json.dumps(dict(epoch=time.time(),id=c['Id'],started=c['State']['StartedAt'],image=c['Image'],command=cmd,env={k:v for k,v in env.items() if k.startswith(('VLLM_','NCCL_','CUDA_','TORCH_'))},mounts=mounts,manifest=pathlib.Path('/home/choiceoh/overlays/glm53/manifest.tsv').read_text(),memory_kib=memory,disk_free_gib=shutil.disk_usage('/home/choiceoh').free/2**30)))
'''


def snapshot():
    def one(host):
        cmd = ['python3', '-c', SCRIPT] if host == 'local' else [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            'choiceoh@' + host, 'python3 -c ' + shlex.quote(SCRIPT)]
        return host, json.loads(subprocess.check_output(cmd, text=True, timeout=30))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(one, ['local', '10.10.10.1', '10.10.10.3', '10.10.10.4']))


if __name__ == '__main__':
    result = snapshot()
    pathlib.Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {'memory_gib': v['memory_kib']['MemAvailable']/1048576,
                          'disk_gib': v['disk_free_gib'], 'image': v['image'],
                          'static_scale': v['env'].get('VLLM_GLM53_NVFP4_STATIC_SCALE')}
                      for k,v in result.items()}))
