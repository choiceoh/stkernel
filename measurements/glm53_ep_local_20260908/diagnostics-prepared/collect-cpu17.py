#!/usr/bin/env python3
"""Archive a successful, source-matched CPU17 job; never runs a test or GPU.

Prepared only. --revision is the actual frozen CPU17 source SHA. Reads the
remote job twice around transfer, refuses every mismatch and existing output,
and requires unchanged CPU16 kernel sources and compiled artifacts. A rejected
transfer stays in *.collecting with its report; it is never an accepted proof.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
HOST = 'choiceoh@srv2'
SOURCE = '/home/choiceoh/stkernel-ep-local-0908-17'
JOB = '/tmp/glm53-ep-local-compile0908-17-head'
BASE = '111fff02f66f3e07a8332da1117afb75b161f0d6'
CPU16 = 'measurements/glm53_ep_local_20260908/cpu16'
CPU16_SHA = '70c06279f97f2cb951606ae64be65038c957a9274bb3885d2371dc998a16578c'
CAPSULE = '/tmp/glm53-bindings-capsule-cpu0908-2/capsule'
MANIFEST = 'b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
LIMIT = 128 * 2**20


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name):
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and str(p) == name and name != '.', 'unsafe relative path: '+name)
    return name


def command(revision):
    return ['bash','/home/choiceoh/stkernel/bench/fleet.sh','run','--cpu','eplocalcpu0908v17head','6',
        'CPU17 bounded numerical failure diagnostics; unchanged kernel, capsule13.0.3 no devices 4g2CPU','--',
        'python3','-B',SOURCE+'/probes/run_glm53_ep_local_cpu_compile.py',
        '--image',IMAGE,'--arm','local','--output',JOB+'/local',
        '--capsule-root',CAPSULE,'--manifest-sha256',MANIFEST]


# This remote code only reads files, hashes, imports pure validators, and calls
# read-only git. It never invokes the compiler, Docker, fleet, or CUDA imports.
REMOTE = r'''
import base64,hashlib,io,json,os,stat,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
C=json.loads(base64.b64decode(sys.argv[1])); source=Path(C['source']); job=Path(C['job'])
def need(value,message):
    if not value: raise RuntimeError(message)
def sha(data): return hashlib.sha256(data).hexdigest()
def safe(name):
    p=PurePosixPath(name)
    need(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in name and str(p)==name and name!='.', 'unsafe path: '+name)
    return name
def read(path):
    for p in (path,*path.parents): need(not p.is_symlink(), 'symlink input: '+str(p))
    st=path.stat(); need(stat.S_ISREG(st.st_mode) and st.st_size<=64*2**20, 'invalid/oversized file: '+str(path))
    raw=path.read_bytes(); end=path.stat()
    need((st.st_ino,st.st_size,st.st_mtime_ns)==(end.st_ino,end.st_size,end.st_mtime_ns) and len(raw)==st.st_size, 'changing input')
    return raw,dict(bytes=len(raw),sha256=sha(raw),source=str(path),mtime_ns=st.st_mtime_ns)
def tree(root):
    need(root.is_dir() and not root.is_symlink(), 'regular input directory required')
    paths=[]
    for parent,dirs,names in os.walk(root,followlinks=False):
        for name in dirs: need(not (Path(parent)/name).is_symlink(), 'symlink directory')
        paths.extend(Path(parent)/name for name in names)
    need(len(paths)<=4096, 'too many files')
    return sorted(paths)
def frozen():
    revision=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'status','--porcelain'],text=True).strip()
    need(revision==C['revision'] and not dirty, 'CPU17 frozen source differs or is dirty')
    need(not (source/'.git/objects/info/alternates').exists(), 'CPU17 unexpectedly has shared alternates')
frozen()
submission=json.loads(read(job/'submission.json')[0]); completed=json.loads(read(job/'exit.json')[0])
need(completed.get('complete') is True and completed.get('returncode')==0 and 'error' not in completed and 'ended' in completed,
     'CPU17 normal --cpu job has not completed successfully')
need(all(completed.get(k)==v for k,v in submission.items()), 'CPU17 completion does not preserve submission')
need(submission['revision']==C['revision'] and submission['source']==str(source)
     and submission['scheduler']=='/home/choiceoh/stkernel' and submission['command']==C['command'], 'CPU17 normal command/source differs')
need(submission['created']<=completed['started']<=completed['ended'], 'CPU17 terminal times invalid')
sys.dont_write_bytecode=True; sys.path.insert(0,str(source/'probes'))
from glm53_ep_local_evidence import validate_compile_evidence,CONTRACT_PATHS
from glm53_ep_bindings_capsule import validate_capsule
proof=validate_compile_evidence(source,job/'local/result.json')
need(proof['contracts']['tests_run']==146 and len(proof['contracts']['files'])==29
     and set(proof['contracts']['files'])==set(CONTRACT_PATHS) and len(proof['mounted_sources'])==13
     and len(proof['remap_compilation'])==24, 'CPU17 coverage differs from required 146/29/13/24')
need(proof['binding_runtime']['capsule_manifest_sha256']==C['manifest'], 'CPU17 binding runtime differs')
need(not any(name=='torch' or name=='cuda' or name.startswith('cuda.') for name in sys.modules), 'archive validator unexpectedly imported accelerator modules')
capsule=Path(C['capsule']); validated_capsule=validate_capsule(capsule,C['manifest'])
files={}; metadata={}; inputs={}; total=0
def add(name,path):
    global total
    safe(name); need(name not in files, 'duplicate output name')
    raw,info=read(path); total+=len(raw); need(total<=C['limit'], 'archive exceeds 128 MiB')
    files[name]=raw; metadata[name]=info; inputs[name]=path
originals=tree(job)
for path in originals: add(path.relative_to(job).as_posix(),path)
need(json.loads(files['local/result.json'])==proof, 'validated CPU17 receipt changed before snapshot')
selected={'build/glm53/manifest.tsv',*CONTRACT_PATHS}
for line in (source/'build/glm53/manifest.tsv').read_text().splitlines():
    if line and not line.startswith('#'): selected.add('build/glm53/'+safe(line.split('\t')[0]))
for rel in sorted(selected): add('frozen-source/'+safe(rel),source/rel)
add('runtime/capsule-manifest.json',capsule/'capsule-manifest.json')
need(tree(job)==originals, 'CPU17 job file set changed')
for name,path in inputs.items(): need(read(path)[0]==files[name], 'file changed before transfer: '+name)
need(validate_capsule(capsule,C['manifest'])==validated_capsule, 'capsule changed during snapshot')
frozen()
identity=dict(source=str(source),job=str(job),revision=C['revision'],clean=True,captured_epoch=time.time(),
    compile_receipt_validated=True,tests=146,mounted_files=13,contract_files=29,remap_compilations=24,
    capsule_manifest_sha256=C['manifest'],capsule_files_checked=len(validated_capsule['files']),
    original_job_files=len(originals),files=metadata,scope='Read-only archive validation; no device or compiler execution')
if C['mode']=='verify':
    need(metadata==C['expected_files'], 'remote source/job/runtime changed after transfer')
    print(json.dumps(dict(verified_after_transfer=True,files=len(metadata),revision=C['revision'],checked_epoch=time.time())))
else:
    need(C['mode']=='fetch', 'unsupported archive mode')
    files['source-identity.json']=(json.dumps(identity,indent=2)+'\n').encode()
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
        for name,raw in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.mtime=0
            archive.addfile(item,io.BytesIO(raw))
'''


def remote(mode, revision, *, expected_files=None):
    config = dict(mode=mode, revision=revision, source=SOURCE, job=JOB, capsule=CAPSULE,
                  manifest=MANIFEST, command=command(revision), limit=LIMIT, expected_files=expected_files)
    encoded = base64.b64encode(json.dumps(config).encode()).decode()
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,
                             shlex.join(['python3','-B','-c',REMOTE,encoded])],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    require(result.returncode == 0, 'CPU17 read-only '+mode+' refused: '+result.stderr.decode(errors='replace')[-2200:])
    require(len(result.stdout) <= LIMIT+8*2**20, 'oversized transfer')
    return result.stdout


def unpack(payload, staging):
    raw, records, stored_names, total = {}, [], set(), 0
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
        for entry in archive:
            name=safe(entry.name)
            require(entry.isfile() and name not in raw and entry.size <= 64*2**20, 'invalid tar member')
            data=archive.extractfile(entry).read(); total+=len(data)
            require(len(data)==entry.size and total<=LIMIT, 'invalid archive size')
            raw[name]=data
            compress=(Path(name).suffix in ('.log','.ptx','.cubin')
                      or name.startswith('frozen-source/') and name.endswith('.py'))
            target=name+'.gz' if compress else name
            require(target not in stored_names, 'stored path collision'); stored_names.add(target)
            stream=io.BytesIO()
            if compress:
                with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0,compresslevel=9) as out: out.write(data)
            stored=stream.getvalue() if compress else data
            require((gzip.decompress(stored) if compress else stored)==data, 'compression mismatch')
            p=staging/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(stored)
            records.append(dict(original_path=name,stored_path=target,original_bytes=len(data),stored_bytes=len(stored),
                                original_sha256=sha(data),stored_sha256=sha(stored)))
    identity=json.loads(raw['source-identity.json'])
    require(set(identity['files'])==set(raw)-{'source-identity.json'}, 'snapshot file set mismatch')
    for name,info in identity['files'].items():
        require(sha(raw[name])==info['sha256'] and len(raw[name])==info['bytes'], 'transfer hash mismatch: '+name)
    (staging/'archive-manifest.json').write_text(json.dumps(dict(host=HOST,records=records),indent=2)+'\n')
    return raw,identity


def git_bytes(*args):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),*args],stderr=subprocess.PIPE)


def compare_baseline(raw, revision):
    proof=json.loads(raw['local/result.json'])
    baseline_raw=(ROOT/CPU16/'local/result.json').read_bytes()
    require(sha(baseline_raw)==CPU16_SHA, 'CPU16 baseline receipt differs')
    baseline=json.loads(baseline_raw); differences=[]; artifact_records=[]
    for key in ('sources','mounted_sources','cache_key','m','binding_runtime'):
        if proof.get(key)!=baseline.get(key): differences.append('CPU16 differs: '+key)
    # Current local validators are pure. Their source bytes must first match the
    # tested frozen set, so a later local source edit cannot bless stale proof.
    for rel,want in proof['contracts']['files'].items():
        require(sha((ROOT/safe(rel)).read_bytes())==want==sha(raw['frozen-source/'+rel]), 'current/frozen contract differs: '+rel)
    sys.dont_write_bytecode=True; sys.path.insert(0,str(ROOT/'probes'))
    from glm53_ep_local_evidence import CONTRACT_PATHS
    require(set(CONTRACT_PATHS)==set(proof['contracts']['files']), 'current contract set differs')
    # Validation runs against the transferred receipt with the already matched
    # current source. No runtime imports, CUDA API or compilation are performed.
    for rel in sorted(n[len('frozen-source/'):] for n in raw if n.startswith('frozen-source/build/glm53/')):
        require(raw['frozen-source/'+rel]==(ROOT/rel).read_bytes(), 'current composed source differs: '+rel)
        if raw['frozen-source/'+rel]!=git_bytes('show',BASE+':'+rel): differences.append('CPU16 composed source differs: '+rel)
    require(not any(n=='torch' or n=='cuda' or n.startswith('cuda.') for n in sys.modules), 'collector imported accelerator modules')
    def archived_baseline(rel):
        rel=safe(rel); p=ROOT/CPU16/rel
        if p.exists(): return p.read_bytes()
        return gzip.decompress(p.with_name(p.name+'.gz').read_bytes())
    for kind in ('artifacts','resources'):
        entries=proof[kind]; old={row['file']:row for row in baseline[kind]}
        if len(entries)!=len(old) or {row['file'] for row in entries}!=set(old): differences.append('CPU16 '+kind+' file set differs')
        for row in entries:
            name='local/'+safe(row['file']); current=raw[name]
            require(sha(current)==row['sha256'] and len(current)==row['bytes'], 'CPU17 artifact differs from receipt: '+name)
            prior=old.get(row['file'])
            same=False
            if prior is not None:
                original=archived_baseline(name)
                require(sha(original)==prior['sha256'] and len(original)==prior['bytes'], 'CPU16 archived artifact corrupted: '+name)
                same=current==original and row==prior
            if not same: differences.append('CPU16 '+kind+' differs: '+row['file'])
            artifact_records.append(dict(file=name,sha256=sha(current),cpu16_equal=same))
            if kind=='resources':
                log_name=name.removesuffix('.cubin')+'.resources.log'
                require(raw[log_name].decode()==row['resources'], 'CPU17 resource stdout differs from receipt')
                log_equal=False
                if prior is not None:
                    old_log=archived_baseline(log_name)
                    require(old_log.decode()==prior['resources'], 'CPU16 resource stdout differs from pinned receipt')
                    log_equal=raw[log_name]==old_log
                if not log_equal: differences.append('CPU16 resource stdout differs: '+log_name)
                artifact_records.append(dict(file=log_name,sha256=sha(raw[log_name]),cpu16_equal=log_equal))
    old_remap={row['label']:row for row in baseline['remap_compilation']}
    require(len(old_remap)==24, 'CPU16 remap baseline incomplete')
    for row in proof['remap_compilation']:
        label=safe(row['label']); old=old_remap.get(label)
        if row!=old: differences.append('CPU16 remap descriptor differs: '+label)
        for extension in ('ptx','cubin'):
            name='local/remap/'+label+'/kernel.'+extension; current=raw[name]
            require(sha(current)==row[extension+'_sha256'], 'CPU17 remap artifact differs from receipt: '+name)
            same=False
            if old is not None:
                original=archived_baseline(name)
                require(sha(original)==old[extension+'_sha256'], 'CPU16 remap archive corrupted: '+name)
                same=current==original
            if not same: differences.append('CPU16 remap artifact differs: '+name)
            artifact_records.append(dict(file=name,sha256=sha(current),cpu16_equal=same))
    return dict(verdict='MATCH' if not differences else 'REJECT_DIFFERENCE', accepted_compile_proof=not differences,
                actual_cpu17_revision=revision,cpu16_revision=BASE,cpu16_receipt_sha256=CPU16_SHA,
                differences=differences,artifacts=artifact_records,
                scope='Exact source/PTX/cubin/resource/remap equality; no runtime numerical or speed claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',required=True,help='Full actual CPU17 frozen source SHA')
    args=parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}',args.revision) is not None, 'full revision required')
    out=ROOT/'measurements/glm53_ep_local_20260908/cpu17'; staging=out.with_name(out.name+'.collecting')
    require(not out.exists() and not staging.exists(), 'output/staging already exists; inspect without overwriting')
    require(not git_bytes('status','--porcelain').strip(), 'current local source must be clean')
    git_bytes('merge-base','--is-ancestor',BASE,args.revision)
    payload=remote('fetch',args.revision)
    staging.mkdir(parents=True,exist_ok=False)
    try:
        raw,identity=unpack(payload,staging)
        comparison=compare_baseline(raw,args.revision)
        (staging/'cpu16-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        require(comparison['accepted_compile_proof'], 'CPU16 source/artifact difference; see cpu16-comparison.json, not accepted')
        from glm53_ep_local_evidence import validate_compile_evidence
        checked=validate_compile_evidence(ROOT,staging/'local/result.json')
        require(checked['contracts']['tests_run']==146 and checked['cuda_initialized'] is False, 'local receipt validation differs')
        post=json.loads(remote('verify',args.revision,expected_files=identity['files']))
        (staging/'post-transfer-verification.json').write_text(json.dumps(post,indent=2)+'\n')
        # A local edit during network transfer also invalidates current-source admission.
        validate_compile_evidence(ROOT,staging/'local/result.json')
        (staging/'README.md').write_text('CPU17 actual normal --cpu PASS: 146 tests, 29 contracts, 13 mounted sources, 24 remap variants; CUDA uninitialized.\n\n'
            'CPU16 kernel/composed sources, PTX, cubin, resource and remap artifacts match exactly. This is compile proof, not GPU numerical or performance acceptance. The original GPU v5 failure remains unresolved.\n\n'
            'Original job files plus frozen source and capsule manifest were checked before and after transfer. archive-manifest.json preserves original/stored SHA256 and sizes; logs/PTX/cubins and frozen Python source snapshots use deterministic gzip without content normalization.\n')
        (staging/'collect.py').write_bytes(Path(__file__).read_bytes())
        files=sorted(p for p in staging.rglob('*') if p.is_file())
        (staging/'SHA256SUMS').write_text(''.join(sha(p.read_bytes())+'  '+p.relative_to(staging).as_posix()+'\n' for p in files))
        for line in (staging/'SHA256SUMS').read_text().splitlines():
            want,name=line.split('  ',1); require(sha((staging/safe(name)).read_bytes())==want,'stored artifact hash changed')
        require(not out.exists(),'output appeared during transfer')
        staging.rename(out)
        print(json.dumps(dict(destination=str(out),verdict='PASS_COMPILE_ONLY',revision=args.revision,
                             original_job_files=identity['original_job_files'],cpu16_artifacts_equal=True)))
    except BaseException as exc:
        (staging/'VERIFICATION_ERROR.txt').write_text(repr(exc)+'\n')
        raise


if __name__=='__main__':
    main()
