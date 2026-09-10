#!/usr/bin/env python3
"""Read-only collection of terminal full-GPU v5, including a failed suite.

Preparation only: no action on import. Run with --confirmed-terminal only after
the normal fleet supervisor has finished restoration/handoff and released.
No Docker, device, service, queue, install, fetch or checkout operations occur.
"""
import argparse
import ast
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
HOST = 'choiceoh@srv2'
SOURCE = '/home/choiceoh/stkernel-ep-local-gpu-0908-5b'
REVISION = 'd53fd44f2b8a69869cddc9df10fa4cbb21bc56d9'
JOB = '/tmp/glm53-ep-local-gpu-0908-5'
SESSION = 'eplocal0908v5'
CPU = 'measurements/glm53_ep_local_20260908/cpu16'
PROOF = CPU + '/local/result.json'
PROOF_SHA = '70c06279f97f2cb951606ae64be65038c957a9274bb3885d2371dc998a16578c'
CAPSULE = '/tmp/glm53-bindings-capsule-cpu0908-2/capsule'
MANIFEST = 'b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
IMAGE = 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211'
LIMIT = 256 * 2**20
CASES = ('balanced4096', 'balanced6912', 'balanced8192', 'concentrated6912',
         'remote4096', 'duplicate4096', 'zeros4097', 'balanced16384')
EXPECTED = [('remap', None)] + [(case, None) for case in CASES] + [
    (case, tool) for tool in ('memcheck', 'racecheck')
    for case in ('remap', 'balanced4096', 'remote4096', 'zeros4097')]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.parts and '..' not in p.parts
            and '\\' not in name and str(p) == name and name != '.', 'unsafe path: ' + name)
    return name


# This embedded script is exclusively filesystem reads and read-only git.
REMOTE = r'''
import hashlib, io, json, os, stat, subprocess, sys, tarfile, time
from pathlib import Path, PurePosixPath
def need(value, reason):
    if not value: raise RuntimeError(reason)
def digest(data): return hashlib.sha256(data).hexdigest()
source=Path(C['source']); job=Path(C['job']); fleet=Path('/home/choiceoh/glm53-logs/fleet')
def regular(path):
    need(path.is_absolute(), 'absolute input required')
    for p in (path, *path.parents): need(not p.is_symlink(), 'symlink input: '+str(p))
    st=path.stat(); need(stat.S_ISREG(st.st_mode), 'non-regular file: '+str(path))
    need(st.st_size <= 64*2**20, 'file exceeds 64 MiB: '+str(path))
    return st
def blob(path):
    before=regular(path); data=path.read_bytes(); after=regular(path)
    need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
         and len(data)==before.st_size, 'input changed during read: '+str(path))
    return data,dict(bytes=len(data),sha256=digest(data),mtime_ns=before.st_mtime_ns,source=str(path))
def tree(root):
    need(root.is_dir() and not root.is_symlink(), 'regular directory required: '+str(root))
    result=[]
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs: need(not (Path(parent)/name).is_symlink(), 'symlink directory refused')
        for name in names: result.append(Path(parent)/name)
    need(len(result)<=4096, 'too many input files')
    return sorted(result)
completion=json.loads(blob(job/'capture/completion.json')[0])
outer=json.loads(blob(job/'exit.json')[0])
need('ended' in completion and 'ended' in outer and type(outer.get('exit_code')) is int,
     'outer normal-fleet job is not terminal')
need(completion['source_revision']==outer['revision']==C['revision'] and outer['session']==C['session'], 'terminal identity differs')
need(completion['ended']<=outer['ended'], 'outer exit precedes payload completion')
need((job/'capture/restored.json').is_file() and completion.get('restored_original') is True,
     'exact incoming restoration receipt missing')
fleet_data={name:blob(fleet/name) for name in ('log','events.log','ledger.tsv','lifecycle.jsonl','queue')}
fleet_data['holder']=blob(fleet/'holder') if (fleet/'holder').exists() else (b'',dict(bytes=0,sha256=digest(b''),source=str(fleet/'holder'),absent=True))
if (fleet/'restore-debt.json').exists(): fleet_data['restore-debt.json']=blob(fleet/'restore-debt.json')
for name in ('log','events.log'):
    lines=[x for x in fleet_data[name][0].decode().splitlines() if len(x.split())>=3 and x.split()[1:3]==['release',C['session']]]
    need(len(lines)==1, 'normal release missing/ambiguous: '+name)
need(not any(x.split('|')[0]==C['session'] for x in fleet_data['holder'][0].decode().splitlines()), 'session still holds fleet')
need(not any(len(x.split('|'))>1 and x.split('|')[1]==C['session'] for x in fleet_data['queue'][0].decode().splitlines()), 'session still queued')
events=[json.loads(x) for x in fleet_data['lifecycle.jsonl'][0].decode().splitlines() if x.strip()]
events=[x for x in events if x.get('session')==C['session']]
payload=[x for x in events if x['event']=='payload-finished']
need(len(payload)==1 and payload[0]['rc']==completion['exit_code'], 'normal supervisor payload receipt missing/different')
finished=[x for x in events if x['event']=='restore-finished' and x.get('rc')==0 or x['event']=='handoff-accepted']
need(finished and payload[0]['t']<=finished[-1]['t']<=outer['ended'], 'normal supervisor restoration/handoff not finished')
if 'restore-debt.json' in fleet_data:
    debt=json.loads(fleet_data['restore-debt.json'][0])
    need(not debt or debt.get('owner',{}).get('session')!=C['session'], 'session still owns restoration debt')
def git(*args): return subprocess.check_output(['git','--no-optional-locks','-C',str(source),*args],text=True).strip()
need(git('rev-parse','HEAD')==C['revision'] and not git('status','--porcelain'), 'frozen source changed')
files={}; metadata={}; paths={}; total=0
def add(name,path=None,value=None):
    global total
    need(name not in files, 'duplicate archive path')
    data,info=blob(path) if path is not None else value
    total+=len(data); need(total<=C['limit'], 'archive exceeds 256 MiB')
    files[name]=data; metadata[name]=info
    if path is not None: paths[name]=path
job_paths=tree(job)
for p in job_paths: add('job/'+p.relative_to(job).as_posix(),p)
proof=json.loads(files['job/cpu-evidence.json'])
need(digest(files['job/cpu-evidence.json'])==C['proof_sha'], 'CPU16 copied proof changed')
source_paths={C['proof'],'build/glm53/manifest.tsv',*proof['contracts']['files']}
for line in (source/'build/glm53/manifest.tsv').read_text().splitlines():
    if line and not line.startswith('#'): source_paths.add('build/glm53/'+line.split('\t')[0])
# The committed CPU16 archive includes the original compile logs/PTX/cubins and
# submission/exit, so the proof remains independently inspectable after cleanup.
for p in tree(source/C['cpu']): source_paths.add(p.relative_to(source).as_posix())
for rel in sorted(source_paths):
    p=PurePosixPath(rel)
    need(not p.is_absolute() and '..' not in p.parts and '\\' not in rel, 'unsafe frozen source path')
    add('source/'+rel,source/rel)
need(files['source/'+C['proof']]==files['job/cpu-evidence.json'], 'frozen CPU16 proof differs')
cap=Path(C['capsule']); manifest_data,manifest_info=blob(cap/'capsule-manifest.json')
need(digest(manifest_data)==C['manifest'], 'capsule external manifest differs')
manifest=json.loads(manifest_data)
cap_paths=tree(cap)
need({p.relative_to(cap).as_posix() for p in cap_paths}==set(manifest['files'])|{'capsule-manifest.json'}, 'capsule file set differs')
for rel,want in manifest['files'].items():
    data,info=blob(cap/rel)
    need(info['sha256']==want['sha256'] and info['bytes']==want['size'], 'capsule file changed: '+rel)
add('runtime/capsule-manifest.json',value=(manifest_data,manifest_info))
for name in ('receipt.json','result.json','container.log','base-distributions.json','distribution-resolution.json'):
    add('runtime/cpu2/'+name,cap.parent/name)
for name,value in fleet_data.items(): add('fleet-state/'+name,value=value)
# A second bounded read verifies all immutable archived inputs, including the
# complete terminal job. Mutable scheduler files remain an explicit snapshot.
need(tree(job)==job_paths, 'terminal job file set changed')
for name,path in paths.items(): need(blob(path)[0]==files[name], 'input changed before serialization: '+name)
need(git('rev-parse','HEAD')==C['revision'] and not git('status','--porcelain'), 'source changed during collection')
identity=dict(captured_at_unix=time.time(),source=str(source),job=str(job),revision=C['revision'],
              source_clean=True,files=metadata,lifecycle_events=events,
              capsule_files_checked=len(manifest['files']),scope='Terminal snapshot; no live service equality check after handoff')
files['source_identity.json']=(json.dumps(identity,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for name,data in sorted(files.items()):
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
        archive.addfile(info,io.BytesIO(data))
'''


def fetch():
    config = dict(source=SOURCE, revision=REVISION, job=JOB, session=SESSION,
                  proof=PROOF, proof_sha=PROOF_SHA, cpu=CPU, capsule=CAPSULE,
                  manifest=MANIFEST, limit=LIMIT)
    script = 'C = ' + repr(config) + '\n' + REMOTE
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             HOST, 'python3', '-B', '-'], input=script.encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    require(result.returncode == 0, 'Read-only collection refused: ' + result.stderr.decode(errors='replace')[-2500:])
    require(len(result.stdout) <= LIMIT + 8*2**20, 'archive transport too large')
    return result.stdout


def unpack(payload, staging):
    raw_files, records, targets = {}, [], set()
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as archive:
        for member in archive:
            name = safe(member.name)
            require(member.isfile() and name not in raw_files and member.size <= 64*2**20, 'unsafe tar entry')
            raw = archive.extractfile(member).read()
            require(len(raw) == member.size and sum(map(len, raw_files.values()))+len(raw) <= LIMIT, 'invalid tar size')
            raw_files[name] = raw
            compress = Path(name).suffix in ('.log', '.ptx', '.cubin') or name.startswith('fleet-state/')
            target = name + '.gz' if compress else name
            require(target not in targets, 'stored path collision: ' + target)
            targets.add(target)
            stream = io.BytesIO()
            if compress:
                with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=9) as writer:
                    writer.write(raw)
            stored = stream.getvalue() if compress else raw
            require((gzip.decompress(stored) if compress else stored) == raw, 'compression mismatch')
            p = staging / target; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(stored)
            records.append(dict(original_path=name, stored_path=target, original_bytes=len(raw),
                                stored_bytes=len(stored), original_sha256=sha(raw), stored_sha256=sha(stored)))
    identity = json.loads(raw_files['source_identity.json'])
    require(set(identity['files']) == set(raw_files)-{'source_identity.json'}, 'snapshot file set differs')
    for name, info in identity['files'].items():
        require(info['sha256'] == sha(raw_files[name]) and info['bytes'] == len(raw_files[name]), 'snapshot hash differs: '+name)
    (staging/'archive_manifest.json').write_text(json.dumps(dict(host=HOST, records=records), indent=2)+'\n')
    return raw_files, identity


def summarize(raw, identity):
    read = lambda name: json.loads(raw[name])
    done, outer, submission = (read('job/'+name) for name in ('capture/completion.json', 'exit.json', 'submission.json'))
    require(all(x == REVISION for x in (done['source_revision'], outer['revision'], submission['revision'], identity['revision'])), 'revision mismatch')
    require(submission['session'] == outer['session'] == SESSION and submission['source'] == SOURCE, 'job identity mismatch')
    require(all(outer.get(k) == v for k, v in submission.items()), 'outer submission fields changed')
    require(outer['complete'] is (outer['exit_code'] == 0) and 'driver_error' not in outer, 'driver terminal state invalid')
    require(outer['exit_code'] == done['exit_code'], 'normal fleet and payload exits differ')
    command = ['bash', '/home/choiceoh/stkernel/bench/fleet.sh', 'run', '--gpu', SESSION, '45',
               'CPU16-matched full MoE remap/numerics/streams/sanitizers; capsule13.0.3 and exact incoming restore', '--',
               'python3', '-B', SOURCE+'/probes/run_glm53_ep_local_offline.py', '--revision', REVISION,
               '--out', JOB+'/capture', '--capsule-root', CAPSULE, '--manifest-sha256', MANIFEST]
    require(submission['command'] == command and sha(raw['job/driver.py']) == submission['driver_sha256'], 'normal command/driver differs')
    require(sha(raw['source/'+PROOF]) == sha(raw['job/cpu-evidence.json']) == PROOF_SHA, 'CPU16 proof mismatch')
    cpu_state = submission['cpu_state']
    require(cpu_state == dict(revision='111fff02f66f3e07a8332da1117afb75b161f0d6', result_sha256=PROOF_SHA,
        submission_sha256='38dcba5f3eba960c1a1ad792052060d7fcd7e2aa36818a2f747ee2e6272304d1',
        exit_sha256='eb4a8b4d2dfbbf4e5791623707414770b2e265e93f786907157a648c1dc39460'), 'CPU16 original job identity differs')
    for filename, key in (('submission.json','submission_sha256'), ('exit.json','exit_sha256')):
        require(sha(raw['source/'+CPU+'/'+filename]) == cpu_state[key], 'CPU16 original receipt changed')
    cpu = read('source/'+PROOF); contracts = cpu['contracts']; runtime = cpu['binding_runtime']
    require(cpu['verdict'] == 'PASS' and cpu['phase'] == 'complete' and cpu['binding_runtime_rechecked'] is True
            and cpu['cuda_initialized'] is False and 'error' not in cpu and 'binding_runtime_recheck_error' not in cpu, 'CPU16 incomplete')
    require(contracts['tests_run'] == 134 and all(contracts[k] == 0 for k in ('errors','failures','skips'))
            and len(contracts['files']) == 27 and len(cpu['mounted_sources']) == 13 and len(cpu['remap_compilation']) == 24, 'CPU16 coverage mismatch')
    for name, want in contracts['files'].items(): require(sha(raw['source/'+safe(name)]) == want, 'contract changed: '+name)
    mounted = {Path(p).name: want for p, want in cpu['mounted_sources'].items()}
    require(len(mounted) == 13, 'mount name collision')
    for name, want in mounted.items(): require(sha(raw['source/build/glm53/'+safe(name)]) == want, 'mounted source changed: '+name)
    require(done['binding_runtime'] == runtime == submission['cpu_summary']['binding_runtime']
            and runtime['capsule_manifest_sha256'] == done['capsule_manifest_sha256'] == submission['capsule_manifest_sha256'] == MANIFEST
            and sha(raw['runtime/capsule-manifest.json']) == MANIFEST, 'runtime/capsule identity mismatch')
    require('binding_runtime_recheck_error' not in done, 'outer runtime postcheck failed')
    cpu2 = read('runtime/cpu2/receipt.json')
    require(sha(raw['runtime/cpu2/receipt.json']) == 'd85e9203ac41dbb265db57da0ee6793c7fa3372013bf276662b64a65ee6c42f8'
            and cpu2['verdict'] == 'PASS' and cpu2['exit_code'] == 0 and cpu2['image'] == IMAGE
            and cpu2['capsule_manifest_sha256'] == MANIFEST
            and sha(raw['runtime/cpu2/result.json']) == cpu2['result_sha256']
            and sha(raw['runtime/cpu2/container.log']) == cpu2['log_sha256'], 'CPU2 runtime receipt identity differs')
    # Exact original state is distinct from the later public restore managed by fleet.
    before, stopped, restored = (read('job/capture/'+n+'.json') for n in ('before','stopped','restored'))
    require(len(before) == 4 and set(before) == set(stopped) == set(restored) and all(before.values()), 'incoming set incomplete')
    require(all(v['image'] == IMAGE and v['auto_remove'] is False and v['overlays'] for v in before.values()), 'incoming image/source identity missing')
    running = {v['running'] for v in before.values()}; require(len(running) == 1, 'mixed incoming state')
    immutable = lambda item: {k:v for k,v in item.items() if k not in ('running','started')}
    for node, original in before.items():
        require(immutable(original) == immutable(stopped[node]) == immutable(restored[node])
                and stopped[node]['running'] is False and restored[node]['running'] == original['running'], 'original identity/state restore mismatch: '+node)
    exact_all_fields = before == restored
    if running == {False}:
        require(before == stopped == restored == read('job/capture/stopped-restored.json'), 'stopped incoming state differs')
    cells = done['cells']; actual = [(c['case'], c['sanitizer']) for c in cells]
    require(actual == EXPECTED[:len(actual)] and len(actual) <= len(EXPECTED), 'cell sequence differs')
    reports = []
    for index, (case, tool) in enumerate(EXPECTED):
        label = (tool+'-' if tool else '')+case
        report = dict(label=label, case=case, sanitizer=tool, status='NOT_RUN', timing={})
        if index >= len(cells): reports.append(report); continue
        cell = cells[index]; evidence = read('job/capture/'+label+'.json') if 'job/capture/'+label+'.json' in raw else {}
        report.update(status='FAIL', exit_code=cell.get('exit_code'), probe_verdict=evidence.get('verdict', 'NO_JSON'),
                      phase=evidence.get('phase'), error=evidence.get('error'), binding_runtime_rechecked=evidence.get('binding_runtime_rechecked'),
                      controls=evidence.get('controls', []), candidate=evidence.get('candidate', []))
        require(cell['started'] <= cell['ended'] <= done['ended'], 'cell timestamps invalid')
        if evidence:
            require(evidence.get('performance_acceptance') is False, 'unexpected performance acceptance')
            if 'binding_runtime' in evidence: require(evidence['binding_runtime'] == runtime, 'cell runtime mismatch')
            if 'provenance' in evidence: require(evidence['provenance'] == mounted, 'cell source mismatch')
        passed = evidence.get('verdict') == 'PASS' and cell.get('exit_code') == 0
        if passed:
            require(evidence.get('phase') == 'complete' and evidence.get('binding_runtime_rechecked') is True
                    and evidence.get('binding_runtime') == runtime and cell.get('binding_runtime') == runtime
                    and 'error' not in evidence and 'binding_runtime_recheck_error' not in evidence, 'incomplete PASS receipt')
            if case == 'remap':
                checks = evidence['checks']; variants = {c['label'] for c in cpu['remap_compilation']}
                require(len(checks) == 24 and {c['label'] for c in checks} == variants
                        and all(c['changed_storage'] and c['rows'] == 4097 for c in checks), 'remap checks incomplete')
                report['exact_byte_checks'] = 24
            else:
                controls = [v for group in report['controls'] for v in group]
                require(len(controls) == 6 and len(report['candidate']) == 4, 'numeric comparisons incomplete')
                require(all(v['bad_rows'] == 0 and v['max_row_relative_l2'] <= .02 and v['max_row_relative_abs'] <= .04 for v in controls), 'stock control not clean')
                require(all(v['bad_rows'] == 0 and all(math.isfinite(n) for n in v.values()) for v in report['candidate']), 'candidate numerics not clean')
                require(evidence['provenance'] == mounted and evidence['sanitize'] == bool(tool)
                        and evidence['timing_scope'] == 'EP remap plus MoE wrapper', 'probe scope differs')
            report['status'] = 'PASS'
        if tool:
            log = raw.get('job/capture/'+label+'.log', b''); text = log.decode(errors='replace')
            errors = re.findall(r'^\s*=+\s*(ERROR SUMMARY:.*)$', text, re.M)
            races = re.findall(r'^\s*=+\s*(RACECHECK SUMMARY:.*)$', text, re.M)
            summaries = errors if tool == 'memcheck' else races
            clean = (bool(summaries) and not re.search(r'^\s*=+\s*(?:ERROR|FATAL)\s*:',text,re.I|re.M)
                     and all(x.strip() == 'ERROR SUMMARY: 0 errors' for x in errors)
                     and all(x.strip() == 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' for x in races))
            expected_summary = dict(tool=tool, verdict='PASS', summaries=summaries, log_sha256=sha(log))
            report['sanitizer_summary'] = dict(clean=clean, error_lines=errors, race_lines=races, log_sha256=sha(log))
            if not clean or cell.get('sanitizer_summary') != expected_summary: report['status'] = 'FAIL'
        for metric, arms in evidence.get('timing', {}).items():
            if metric not in ('wall_ms','device_ms'): continue
            require(not tool and case != 'remap' and set(arms) == {'compact','local'}, 'unexpected timing arms')
            require(all(len(v) == 8 and all(math.isfinite(n) and n > 0 for n in v) for v in arms.values()), 'invalid timing samples')
            baseline, candidate = (statistics.median(arms[arm]) for arm in ('compact','local'))
            report['timing'][metric] = dict(compact_median_ms=baseline, local_median_ms=candidate,
                speedup_pct=100*(baseline/candidate-1), latency_reduction_pct=100*(1-candidate/baseline), samples_per_arm=8, calls_per_sample=3)
        # A failed compare raises before appending; preserve its exact dict too.
        match = re.fullmatch(r'AssertionError\((\{.*\})\)', evidence.get('error',''))
        if match:
            try: report['failed_comparison'] = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError): pass
        reports.append(report)
    passed_count = sum(r['status'] == 'PASS' for r in reports)
    require(done['exit_code'] != 0 or passed_count == len(EXPECTED), 'outer success without every cell PASS')
    return dict(session=SESSION, revision=REVISION, verdict='PASS_COMPONENT_GATES' if done['exit_code'] == 0 else 'FAIL',
        performance_acceptance=False, inner_exit_code=done['exit_code'], outer_exit_code=outer['exit_code'], error=done.get('error'),
        completed_at=done['ended'], normal_fleet_returned_at=outer['ended'], expected_cells=17, observed_cells=len(cells),
        passed_cells=passed_count, failed_cells=sum(r['status']=='FAIL' for r in reports), unrun_cells=17-len(cells), cells=reports,
        source_validation=dict(cpu16_tests=134, contracts=27, mounted_sources=13, remap_compile_variants=24,
                               proof_sha256=PROOF_SHA, capsule_manifest_sha256=MANIFEST, capsule_files_checked=identity['capsule_files_checked']),
        restoration=dict(incoming_running=next(iter(running)), immutable_identity_and_running_equal=True,
                         all_original_fields_equal=exact_all_fields, original_container_ids={n:v['id'] for n,v in before.items()},
                         normal_supervisor_events=identity['lifecycle_events'], scope='Exact original snapshots before normal fleet finish; later public containers/holders can legitimately differ'),
        sanitizer_preflight=read('job/capture/sanitizer-preflight.json'),
        limitations=['Synthetic single-GB10 EP remap plus MoE wrapper; compact baseline, E72/H4096/N2048.',
                     'No shared expert, TP4/EP4 transport, full-model TTFT, serving quality or decode acceptance.',
                     'Nondefault-stream execution is checked; this suite does not establish CUDA graph replay.',
                     'Timing does not override numerical or sanitizer failure. Unrun cells remain unverified.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed-terminal', action='store_true')
    args = parser.parse_args()
    if not args.confirmed_terminal: parser.error('--confirmed-terminal required; terminal/release checks are independent')
    out = ROOT/'measurements/glm53_ep_local_20260908/gpu5-completed'
    staging = out.with_name(out.name+'.collecting')
    require(not out.exists() and not staging.exists(), 'archive/staging already exists; no overwrite')
    payload = fetch()  # No local archive is created before remote terminal guards.
    staging.mkdir(parents=True, exist_ok=False)
    try:
        raw, identity = unpack(payload, staging)
        summary = summarize(raw, identity)
        (staging/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        lines = ['# Full MoE GPU v5', '',
            f"{summary['verdict']}: {summary['passed_cells']} passed, {summary['failed_cells']} failed, {summary['unrun_cells']} not run. Inner/outer exit: {summary['inner_exit_code']}/{summary['outer_exit_code']}.", '',
            '| Cell | Status | Phase | Compact / local wall ms |', '|---|---|---|---|']
        for cell in summary['cells']:
            timing = cell['timing'].get('wall_ms')
            measured = f"{timing['compact_median_ms']:.4f} / {timing['local_median_ms']:.4f}" if timing else 'not measured'
            lines.append(f"| {cell['label']} | {cell['status']} | {cell.get('phase') or '—'} | {measured} |")
        lines += ['', 'Failure: '+str(summary['error']), '',
            'Original immutable identity and running state match archived restoration. Incoming running='+str(summary['restoration']['incoming_running'])+'. Normal supervisor restore/handoff and release are separately preserved in fleet-state/. No live equality is required after normal handoff.', '',
            'Source '+REVISION+'; CPU16 proof '+PROOF_SHA+'; capsule manifest '+MANIFEST+'.', '',
            'All original job files, frozen composed/contract sources, CPU16 archive, CPU2 runtime receipts, and scheduler snapshot are preserved. Capsule binaries are hashed at collection but not copied. archive_manifest.json records original/stored sizes and SHA256; logs/PTX/cubins use deterministic gzip. Summary extraction is not a test rerun.', '', *summary['limitations']]
        (staging/'README.md').write_text('\n'.join(lines)+'\n')
        (staging/'collect.py').write_bytes(Path(__file__).read_bytes())
        files = sorted(p for p in staging.rglob('*') if p.is_file())
        (staging/'SHA256SUMS').write_text(''.join(sha(p.read_bytes())+'  '+p.relative_to(staging).as_posix()+'\n' for p in files))
        for line in (staging/'SHA256SUMS').read_text().splitlines():
            want, name = line.split('  ',1); require(sha((staging/safe(name)).read_bytes()) == want, 'final hash mismatch')
        require(not out.exists(), 'output appeared during collection')
        staging.rename(out)
        print(json.dumps(dict(archive=str(out), verdict=summary['verdict'], passed=summary['passed_cells'],
                              failed=summary['failed_cells'], unrun=summary['unrun_cells']), indent=2))
    except BaseException as exc:
        (staging/'VERIFICATION_ERROR.txt').write_text(repr(exc)+'\n')
        raise


if __name__ == '__main__':
    main()
