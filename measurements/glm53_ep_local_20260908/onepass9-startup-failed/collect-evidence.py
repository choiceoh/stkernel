"""Read-only onepass9 terminal collection; writes only this archive directory."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REV = 'e4425b984e3455614744f0f3072916b1b296bd0f'
SESSION = 'eplocalonepass0909v9'
TICKET = '17889108153442432'
NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
originals = {}


def save(name, raw, origin, **extra):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == raw, name
    else:
        path.write_bytes(raw)
    originals[name] = dict(origin=origin, bytes=len(raw), sha256=sha(raw), **extra)


def packed(name, raw, origin):
    save(name, gzip.compress(raw, mtime=0), origin,
         original_bytes=len(raw), original_sha256=sha(raw))


def record(name, value, origin):
    save(name, (json.dumps(value, indent=2, sort_keys=True)+'\n').encode(), origin)


REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,time
P=pathlib.Path
root=P('/home/choiceoh/stkernel-ep-onepass-0909-9')
job=P('/tmp/glm53-ep-onepass-0909-9')
def identity():
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}).decode().strip()
 return dict(head=git('rev-parse','HEAD'),status=git('status','--porcelain'),shallow=git('rev-parse','--is-shallow-repository'))
def fleet():
 p=subprocess.run(['bash',str(root/'bench/fleet.sh'),'show','eplocalonepass0909v9','--ticket','17889108153442432','--json'],capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 data=json.loads(p.stdout)
 keys=('session','ticket','state','phase','started_at','payload_finished_at','finished_at','payload_returncode','returncode','outcome','log_path','recovery_policy','recovery_deferred','supervisor_alive','payload_seconds')
 return {k:data[k] for k in keys if k in data}
def holder_own():
 p=P('/home/choiceoh/glm53-logs/fleet/holder')
 return p.exists() and p.read_text().split('|',1)[0]=='eplocalonepass0909v9'
r=dict(captured_at=time.time(),source_before=identity(),fleet_before=fleet(),own_holder_before=holder_own(),files={},absent=[])
assert r['fleet_before']['phase']=='finished' and r['fleet_before']['payload_returncode']==1 and not r['own_holder_before']
paths=[job/'onepass.jsonl',job/'verdicts.jsonl',P(r['fleet_before']['log_path'])]
paths += [P('/home/choiceoh/glm53-logs')/('boot-EPONEPASS9'+arm+'.log') for arm in ('B1','A')]
for path in paths:
 if not path.exists():r['absent'].append(str(path));continue
 before=path.stat();raw=path.read_bytes();after=path.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and len(raw)<128*2**20
 r['files'][str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(raw).decode()}
r.update(source_after=identity(),fleet_after=fleet(),own_holder_after=holder_own())
print(json.dumps(r))
'''
completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'choiceoh@srv2', 'python3 -B -'],
                           input=REMOTE.encode(), capture_output=True)
assert completed.returncode == 0, completed.stderr.decode()
remote = json.loads(completed.stdout)
assert remote['source_before'] == remote['source_after']
assert remote['source_before']['head'] == REV and remote['source_before']['status'] == ''
assert remote['fleet_before'] == remote['fleet_after'] and not remote['own_holder_after']
for origin, item in remote.pop('files').items():
    raw = base64.b64decode(item['data'])
    assert sha(raw) == item['sha256']
    name = Path(origin).name
    if name == 'onepass.jsonl':
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 1 and rows[0]['name'] == 'EPONEPASS9B1'
        save('records/onepass.jsonl', raw, origin)
    elif name == 'verdicts.jsonl':
        save('records/verdicts.jsonl', raw, origin)
    elif name.startswith('boot-'):
        packed('boot/'+name+'.gz', raw, origin)
    else:
        packed('fleet/terminal-run.log.gz', raw, origin)
record('terminal-capture.json', remote, 'read-only terminal fleet/source/holder checks')


def gitfile(name):
    return subprocess.check_output(['git', 'show', REV+':'+name], cwd=ROOT)


manifest = gitfile('build/glm53/manifest.tsv')
deployed_manifest = b'# source_commit='+REV.encode()+b'\n'+manifest
mounts = {line.split('\t')[1]: sha(gitfile('build/glm53/'+line.split('\t')[0]))
          for line in manifest.decode().splitlines() if line and not line.startswith('#')}
save('source/frozen-manifest.tsv', manifest, 'git '+REV)
record('source/mounted-hashes.json', mounts, 'git '+REV)

private = Path('/tmp/glm53-onepass9-live-B1-observer')
raw_identity = (private/'identity.json').read_bytes()
identity = json.loads(raw_identity)
assert (identity['revision'], identity['arm'], identity['session']) == (REV, 'B1', SESSION)
parser_bytes = (private/'launch-parser.py').read_bytes()
assert sha(parser_bytes) == identity['parser_sha256'] and parser_bytes == gitfile('bench/glm53_launch_metadata.py')
parser = runpy.run_path(str(private/'launch-parser.py'))


def stable(container):
    return {k:container[k] for k in ('Id', 'Image', 'Config', 'HostConfig', 'RestartCount')} | {
        'Mounts':sorted(container['Mounts'], key=lambda x:json.dumps(x, sort_keys=True)),
        'StartedAt':container['State']['StartedAt'], 'Pid':container['State']['Pid']}


summary = {k:identity[k] for k in ('schema', 'arm', 'revision', 'source', 'session', 'captured_at', 'parser_sha256')}
summary.update(private_identity_sha256=sha(raw_identity), raw_inspect_archived=False, nodes={})
for rank, node in enumerate(NODES):
    detail = identity['nodes'][node]
    before_pack = (private/(node+'.inspect.before.json.gz')).read_bytes()
    after_pack = (private/(node+'.inspect.after.json.gz')).read_bytes()
    before_raw, after_raw = gzip.decompress(before_pack), gzip.decompress(after_pack)
    before, after = json.loads(before_raw)[0], json.loads(after_raw)[0]
    assert stable(before) == stable(after) and before['State']['Running'] and after['State']['Running']
    assert (before['Id'], before['Image'], before['State']['StartedAt']) == (detail['id'], detail['image'], detail['started_at'])
    assert before['Image'] == 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
    topology = parser['launch_parallelism'](before['Config']['Cmd'])
    assert topology == detail['topology'] and topology['node_rank'] == rank and not topology['enabled']
    env = {}
    for item in before['Config']['Env']:
        key, value = item.split('=', 1)
        assert key not in env
        env[key] = value
    flags = {key:env[key] for key in ('VLLM_GLM53_EP_PREFILL_LOCAL', 'VLLM_B12X_EP_WARM_COMPACT',
                                    'VLLM_B12X_EP_ZERO_WEIGHT_MICRO', 'VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE')}
    assert flags == detail['flags'] == {key:('1' if key == 'VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE' else '0') for key in flags}
    assert detail['source'] == dict(manifest_sha256=sha(deployed_manifest), mounts=mounts)
    safe = {key:detail[key] for key in ('node','id','image','started_at','capture_started_at',
        'capture_finished_at','running_start_config_stable','environment_sha256','flags','endpoint','source','log_source')}
    safe['topology'] = {key:topology[key] for key in ('schema','source','enabled','nnodes','node_rank',
        'tensor_parallel_size','command_sha256','prelude_sha256','serve_argv_sha256','serve_argv_without_ep_sha256')}
    safe['private_inspect_hashes'] = dict(before_stored=sha(before_pack), before_original=sha(before_raw),
                                        after_stored=sha(after_pack), after_original=sha(after_raw))
    summary['nodes'][node] = safe
    for suffix in ('serving.log.gz','docker.log.gz','manifest.tsv.gz'):
        name = node+'.'+suffix
        raw = (private/name).read_bytes()
        meta = identity['files'][name]
        assert sha(raw) == meta['stored_sha256'] and sha(gzip.decompress(raw)) == meta['original_sha256']
        if suffix == 'manifest.tsv.gz':
            assert gzip.decompress(raw) == deployed_manifest
        save('B1-identity/'+name, raw, str(private/name),
             original_sha256=meta['original_sha256'], original_bytes=meta['original_bytes'])
record('B1-identity/allowlisted-summary.json', summary, 'validated private before/after B1 snapshots; no raw Config Env/Cmd archived')
save('B1-identity/launch-parser.py', parser_bytes, str(private/'launch-parser.py'))

stream_root = Path('/tmp/glm53-onepass9-streams')
event_bytes = (stream_root/'events.jsonl').read_bytes()
events = [json.loads(line) for line in event_bytes.splitlines()]
assert events[-1]['kind'] == 'observer_finished'
ends, chunks = {}, {}
for event in events:
    if event['kind'] != 'chunk':
        continue
    key = event['node'], event['channel']
    assert event['offset'] == ends.get(key, 0)
    ends[key] = event['offset']+event['bytes']
    chunks.setdefault(key, []).append(event)
raw_streams = {}
for (node, channel), end in ends.items():
    path = stream_root/(node+'.'+channel+'.raw')
    raw = path.read_bytes()
    assert len(raw) == end
    for event in chunks[node, channel]:
        assert sha(raw[event['offset']:event['offset']+event['bytes']]) == event['sha256']
    raw_streams[node, channel] = raw
    packed('streams/'+path.name+'.gz', raw, str(path))
packed('streams/events.jsonl.gz', event_bytes, str(stream_root/'events.jsonl'))
pid = int((stream_root/'observer.pid').read_text())
try:
    os.kill(pid, 0)
except ProcessLookupError:
    alive = False
else:
    alive = True
assert not alive, 'owned observer has not closed'
record('streams/closure.json', dict(observer_pid=pid, alive=alive,
    events=[e for e in events if e['kind'] in ('stream_end','observer_stop','observer_finished')],
    scope='Every original chunk hash/offset checked; streams span B1 and candidate startup.'), 'closed passive observer')

failures = {}
for node in NODES:
    path = Path('/tmp')/('glm53-onepass9-'+node+'.stdout.raw.selftest-fail.json')
    raw = path.read_bytes()
    receipt = json.loads(raw)
    assert receipt['verdict'] == 'FAIL'
    stream = raw_streams[node, 'stdout']
    matches = []
    for line in stream.splitlines(keepends=True):
        marker = b'[ep-local-selftest] FAIL '
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].decode()
        observed, _ = json.JSONDecoder().raw_decode(payload)
        if observed == receipt:
            matches.append(dict(line_sha256=sha(line), stream_byte_offset=stream.index(line)))
    assert len(matches) == 1, (node, 'failure receipt not uniquely present in original stream')
    save('failures/'+node+'.json', raw, str(path), raw_stream_match=matches[0])
    failed = [case for case in receipt['cases'] if case.get('verdict') != 'PASS']
    assert len(failed) == 1
    case = failed[0]
    failures[node] = dict(error=receipt['error'], case=case['case'], phase=case['phase'],
        candidate_first_failure=case.get('candidate_first_failure'),
        weights_dtype=case['inputs'][0]['weights']['dtype'],
        raw_stream_match=matches[0])
record('failure-summary.json', failures, 'unchanged four failure receipts; summaries do not waive any FAIL')
save('collect-evidence.py', Path(__file__).read_bytes(), 'this read-only collector')
record('originals.json', originals.copy(), 'original and stored SHA256 provenance')
print(json.dumps(dict(archive=str(OUT), state=remote['fleet_after']['state'],
                     measurement_rows=[row['name'] for row in rows], failures=len(failures), files=len(originals))))
