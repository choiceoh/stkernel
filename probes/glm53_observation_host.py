"""Host-local preparation of a named diagnostic clone; never remove serving."""
import base64
import copy
import gzip
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import stat
import subprocess

OWNER = 'codex.glm.prefill-observation'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def host_config_identity(config):
    """Docker serializes an unset OOM-disable flag as null or false.

    Both leave the default OOM killer enabled. This changes only comparison,
    never the submitted configuration. Explicit true remains distinct; no other
    nullable resource field (notably MemorySwappiness) is normalized.
    """
    result=copy.deepcopy(config)
    if 'OomKillDisable' in result:
        value=result['OomKillDisable']
        if value is not None and type(value) is not bool:
            raise ValueError('invalid OomKillDisable representation')
        if value is None:result['OomKillDisable']=False
    return result


def host_config_differences(requested, actual):
    expected=host_config_identity(requested);observed=host_config_identity(actual)
    return sorted(k for k,v in expected.items() if k not in observed or observed[k]!=v)


def clone_payload(container, *, directory, source, session):
    payload = copy.deepcopy(container['Config'])
    host = copy.deepcopy(container['HostConfig'])
    if host.get('NetworkMode') != 'host' or host.get('AutoRemove') or host.get('Mounts'):
        raise ValueError('persistent host-network serving with explicit bind mounts required')
    if payload.get('Entrypoint') != ['/bin/bash']:
        raise ValueError('expected static bash serving entrypoint')
    env = dict(item.split('=',1) for item in payload['Env'])
    for key in ('VLLM_GLM53_B12X_PREFILL_M64','VLLM_GLM53_PREFILL_SP_RS_INT8','VLLM_GLM53_DEV_LAB'):
        if env.get(key,'0') != '0':raise ValueError('experimental option active: '+key)
    command=payload.get('Cmd',[])
    if len(command)!=2 or command[0]!='-c':raise ValueError('unknown serving command shape')
    match=re.fullmatch(r'echo ([A-Za-z0-9+/=]+) \| base64 -d > /tmp/serve\.sh; bash /tmp/serve\.sh',command[1])
    if not match:raise ValueError('static base64 serving command required')
    script=base64.b64decode(match[1],validate=True).decode()
    if len(re.findall(r'^vllm serve ',script,re.M))!=1 or '--worker-extension-cls' in script:
        raise ValueError('one unextended vllm serving process required')
    for pattern,replacement in ((r'--host\s+0\.0\.0\.0\b','--host 127.0.0.1'),(r'--port\s+8000\b','--port 18000')):
        script,count=re.subn(pattern,replacement,script)
        if count!=1:raise ValueError('approved public endpoint required before cloning')
    redirect='> /glmlogs/glm53.log 2>&1'
    if script.count(redirect)!=1 or '--profiler-config' not in script:
        raise ValueError('known log redirection and profiler configuration required')
    script=script.replace(redirect,'--worker-extension-cls glm53_prefill_observer.WorkerExtension '
        '--middleware glm53_prefill_observer.middleware '+redirect)
    encoded=base64.b64encode(script.encode()).decode()
    payload['Cmd']=['-c','echo '+encoded+' | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh']
    env['PYTHONPATH']='/observer'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    payload['Env']=[key+'='+value for key,value in env.items()]
    payload['Image']=container['Image']
    payload['Labels']={**(payload.get('Labels') or {}),OWNER:session,OWNER+'.original':container['Id']}
    binds=[];seen=set()
    for bind in host.get('Binds') or []:
        parts=bind.split(':')
        if len(parts) not in (2,3):raise ValueError('unsupported bind syntax')
        src,dest=parts[:2]
        if dest=='/observer':raise ValueError('observer already mounted')
        if dest in ('/prof','/glmlogs'):
            seen.add(dest);src=str(Path(directory)/dest[1:])
        binds.append(':'.join([src,dest,*parts[2:]]))
    if seen!={'/prof','/glmlogs'}:raise ValueError('profiler and log binds required')
    binds.append(str(Path(source)/'probes')+':/observer:ro')
    host['Binds']=binds
    payload['HostConfig']=host
    return payload


def inspect(name):
    result=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
    if result.returncode:
        subprocess.run(['docker','info'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=15)
        return None
    return json.loads(result.stdout)[0]


def owned(name, session):
    c=inspect(name)
    if c and (c['Config'].get('Labels') or {}).get(OWNER)!=session:
        raise RuntimeError('refusing foreign diagnostic container')
    return c


def create(name, payload):
    connection=http.client.HTTPConnection('localhost',timeout=30)
    connection.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    connection.sock.settimeout(30)
    try:
        connection.sock.connect('/var/run/docker.sock')
        connection.request('POST','/containers/create?name='+name,json.dumps(payload),{'Content-Type':'application/json'})
        response=connection.getresponse();result=json.loads(response.read())
        if response.status!=201:raise RuntimeError('Docker create failed: '+str(result.get('message')))
        return result['Id']
    finally:connection.close()


def dispatch(*,action,name,session,directory,source,original=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',session) or name!='glm53-observe-'+session:
        raise ValueError('owned observation name required')
    directory=Path(directory);source=Path(source)
    if not directory.is_absolute() or not source.is_absolute() or directory.is_symlink() or source.is_symlink():
        raise ValueError('absolute regular source and output paths required')
    if action=='prepare':
        if inspect(name) is not None:raise RuntimeError('diagnostic name already exists')
        c=inspect(original)
        if c is None or not c['State']['Running']:raise RuntimeError('incoming serving absent')
        payload=clone_payload(c,directory=directory,source=source,session=session)
        with socket.socket() as port_check:port_check.bind(('127.0.0.1',18000))
        directory.mkdir(parents=True,exist_ok=False)
        for folder in ('prof','glmlogs'):(directory/folder).mkdir()
        cid=create(name,payload)
        actual=owned(name,session)
        if actual['Id']!=cid or actual['State']['Running']:
            raise RuntimeError('prepared clone identity mismatch')
        # Extra daemon-populated fields are allowed, but every field copied or
        # explicitly changed above must have its requested value.
        if any(actual['Config'].get(k)!=v for k,v in payload.items() if k!='HostConfig'):
            raise RuntimeError('Docker changed requested clone configuration')
        differences=host_config_differences(payload['HostConfig'],actual['HostConfig'])
        if differences:
            raise RuntimeError('Docker changed requested clone host configuration: '+', '.join(differences))
        return dict(id=cid,original_id=c['Id'],config=digest(actual['Config']),
                    host_config=digest(host_config_identity(actual['HostConfig'])),
                    host_config_raw=digest(actual['HostConfig']),
                    oom_kill_disable_raw=actual['HostConfig'].get('OomKillDisable'),image=actual['Image'])
    c=owned(name,session)
    if action=='logs':
        path=directory/'glmlogs/glm53.log'
        if not path.exists():return dict(exists=False)
        if path.is_symlink() or not path.is_file():raise ValueError('nonregular diagnostic log')
        raw=path.read_bytes()
        return dict(exists=True,sha256=hashlib.sha256(raw).hexdigest(),
                    gzip_base64=base64.b64encode(gzip.compress(raw,mtime=0)).decode())
    if action=='remove':
        if c:subprocess.run(['docker','rm','-f',c['Id']],check=True,stdout=subprocess.DEVNULL,timeout=75)
        return dict(removed=True)
    if c is None:raise RuntimeError('diagnostic clone missing')
    if action=='start':
        subprocess.run(['docker','start',c['Id']],check=True,stdout=subprocess.DEVNULL,timeout=30)
        c=owned(name,session)
    if action in ('start','state'):
        return dict(id=c['Id'],running=c['State']['Running'],started=c['State']['StartedAt'],
                    config=digest(c['Config']),host_config=digest(host_config_identity(c['HostConfig'])),
                    host_config_raw=digest(c['HostConfig']),
                    oom_kill_disable_raw=c['HostConfig'].get('OomKillDisable'),image=c['Image'])
    if action in ('traces','trace_hashes'):
        traces={}
        for p in (directory/'prof').rglob('*'):
            if not p.name.endswith(('.pt.trace.json','.pt.trace.json.gz')):continue
            st=p.lstat()
            traces[str(p)]=dict(device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,
                               size=st.st_size,symlink=p.is_symlink(),regular=stat.S_ISREG(st.st_mode))
            if action=='trace_hashes':
                if not stat.S_ISREG(st.st_mode) or p.is_symlink():raise ValueError('nonregular trace')
                with p.open('rb') as stream:traces[str(p)]['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        return traces
    raise ValueError('unsupported host action')
