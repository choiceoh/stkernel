"""Only the two cache modules and one vote policy may vary in private clones."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket

import glm53_observation_host as host

KNOB = 'VLLM_GLM53_RANK_CACHE_CPU_VOTE'
PREFIX = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/'
MODULES = ('glm53_rank_cache.py', 'glm53_startup_cache.py')
FIXED_ENV = {'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET'}


def source_hashes(source):
    result = {}
    for name in MODULES:
        path = Path(source)/'build/glm53'/name
        raw = path.read_bytes()
        if path.is_symlink() or raw != (Path(source)/'overlay/modules/glm53_model'/name).read_bytes():
            raise ValueError('candidate source and composed copy differ')
        result[PREFIX+name] = hashlib.sha256(raw).hexdigest()
    return result


def original_identity(container):
    return dict(id=container['Id'], image=container['Image'], config=host.digest(container['Config']),
                host_config=host.digest(host.host_config_identity(container['HostConfig'])))


def clone_payload(container, *, directory, source, session, policy):
    if policy not in ('0', '1') or type(policy) is not str:
        raise ValueError('explicit 0/1 vote policy required')
    payload = host.clone_payload(container, directory=directory, source=source, session=session)
    source_hashes(source)
    env = dict(e.split('=', 1) for e in payload['Env'])
    if env.get(KNOB, '0') != '0' or env.get('VLLM_DISTRIBUTED_USE_SPLIT_GROUP', '0') != '0':
        raise ValueError('approved device-vote non-split original required')
    if env.get('VLLM_GLM53_RANK_CACHE') != '/cache/glm53-ranks':
        raise ValueError('rank artifact cache must be enabled at its approved path')
    if 'NCCL_DEBUG_FILE' in env:
        raise ValueError('NCCL diagnostics must remain in the owned serving log')
    env.update(FIXED_ENV)
    env[KNOB] = policy
    payload['Env'] = [k+'='+v for k, v in env.items()]
    script = base64.b64decode(payload['Cmd'][1].split()[1], validate=True).decode()
    for suffix in ('WorkerExtension', 'middleware'):
        old = 'glm53_prefill_observer.'+suffix
        if script.count(old) != 1:
            raise ValueError('observer command replacement is ambiguous')
        script = script.replace(old, 'glm53_cpu_vote_memory.'+suffix)
    payload['Cmd'] = ['-c', 'echo '+base64.b64encode(script.encode()).decode()
                      +' | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh']
    binds = []; seen = set()
    for bind in payload['HostConfig']['Binds']:
        parts = bind.split(':')
        if parts[1] in {PREFIX+name for name in MODULES}:
            if parts[1] in seen or parts[2:] != ['ro']:
                raise ValueError('one read-only bind per cache module required')
            seen.add(parts[1])
            parts[0] = str(Path(source)/'build/glm53'/Path(parts[1]).name)
        binds.append(':'.join(parts))
    if seen != {PREFIX+name for name in MODULES}:
        raise ValueError('both original cache module binds required')
    payload['HostConfig']['Binds'] = binds
    return payload


def dispatch(*, action, name, session, directory, source, original=None, policy='0', expected_original=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session) or name != 'glm53-observe-'+session:
        raise ValueError('owned observation name required')
    directory, source = Path(directory), Path(source)
    if any(not p.is_absolute() or p.is_symlink() for p in (directory, source)):
        raise ValueError('absolute regular source and output paths required')
    if action != 'prepare':
        result = host.dispatch(action=action, name=name, session=session, directory=str(directory),
                               source=str(source), original=original)
        if action in ('state', 'start'):
            result['candidate_sources'] = source_hashes(source)
        return result
    if host.inspect(name) is not None:
        raise RuntimeError('diagnostic name already exists')
    incoming = host.inspect(original)
    if incoming is None:
        raise RuntimeError('original serving absent')
    identity = original_identity(incoming)
    if expected_original is None:
        if not incoming['State']['Running']:
            raise RuntimeError('initial preparation requires running originals')
    elif identity != expected_original or incoming['State']['Running']:
        raise RuntimeError('subsequent arm requires the exact paused original')
    reserve = 128 + (64 if expected_original is None else 0)
    if shutil.disk_usage('/home/choiceoh').free < reserve*2**30:
        raise RuntimeError('128 GiB disk reserve plus 64 GiB initial cache headroom required')
    payload = clone_payload(incoming, directory=directory, source=source, session=session, policy=policy)
    with socket.socket() as port_check:
        port_check.bind(('127.0.0.1', 18000))
    directory.mkdir(parents=True, exist_ok=False)
    for folder in ('prof', 'glmlogs'):
        (directory/folder).mkdir()
    cid = host.create(name, payload)
    actual = host.owned(name, session)
    if actual['Id'] != cid or actual['State']['Running']:
        raise RuntimeError('prepared clone identity mismatch')
    if any(actual['Config'].get(k) != v for k, v in payload.items() if k != 'HostConfig'):
        raise RuntimeError('Docker changed requested vote clone configuration')
    if host.host_config_differences(payload['HostConfig'], actual['HostConfig']):
        raise RuntimeError('Docker changed requested vote clone host configuration')
    return dict(id=cid, original=identity, config=host.digest(actual['Config']),
                host_config=host.digest(host.host_config_identity(actual['HostConfig'])),
                image=actual['Image'], candidate_sources=source_hashes(source))
