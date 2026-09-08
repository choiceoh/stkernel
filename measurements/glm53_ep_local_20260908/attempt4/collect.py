#!/usr/bin/env python3
"""Archive a terminal, normally released GLM EP attempt; never invokes a GPU.

Preparation is read-only. Execution requires --confirmed-terminal, refuses an
existing output, and independently requires completion.json, outer exit.json
and the session's normal release records. Integrity failures retain only a
staging directory and exit nonzero instead of publishing a verified archive.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import tarfile
from zoneinfo import ZoneInfo

DEFAULT_ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
CASES = ('balanced4096','balanced6912','balanced8192','concentrated6912',
         'remote4096','duplicate4096','zeros4097','balanced16384')
EXPECTED = [('remap', None)] + [(case, None) for case in CASES] + [
    (case, tool) for tool in ('memcheck','racecheck')
    for case in ('remap','balanced4096','remote4096','zeros4097')]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or '\\' in name:
        raise ValueError('unsafe archive path: '+name)
    return str(path)


def kst(timestamp):
    return datetime.fromtimestamp(timestamp, ZoneInfo('Asia/Seoul')).isoformat()


def fetch(args):
    config = dict(job=args.job, source=args.source, revision=args.revision,
                  session=args.session, cpu=args.cpu)
    remote = 'CONFIG = '+repr(config)+'\n'+r'''
from pathlib import Path, PurePosixPath
import hashlib, io, json, subprocess, sys, tarfile, time
job=Path(CONFIG['job']); source=Path(CONFIG['source'])
assert job.is_absolute() and source.is_absolute()
completion=json.loads((job/'capture/completion.json').read_text())
outer=json.loads((job/'exit.json').read_text())
assert 'ended' in completion and 'ended' in outer and 'exit_code' in outer, 'job is not terminal'
assert completion['source_revision']==outer['revision']==CONFIG['revision'], 'revision mismatch'
assert outer['session']==CONFIG['session'], 'session mismatch'
fleet=Path('/home/choiceoh/glm53-logs/fleet')
fleet_data={name:(fleet/name).read_bytes() for name in ('log','events.log','ledger.tsv','holder','queue')}
for name in ('log','events.log'):
 assert any(('release '+CONFIG['session']) in line and line.split()[1]=='release' and line.split()[2]==CONFIG['session'] for line in fleet_data[name].decode().splitlines()), 'normal release missing: '+name
for name in ('holder','queue'):
 assert CONFIG['session'] not in fleet_data[name].decode(), 'session still held/queued'
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'status','--porcelain'],text=True)
assert revision==CONFIG['revision'] and not status, 'frozen source changed'
files={}
for p in sorted(job.rglob('*')):
 if p.is_symlink(): raise RuntimeError('symlink refused: '+str(p))
 if p.is_file(): files['job/'+str(p.relative_to(job))]=p.read_bytes()
receipt_rel='measurements/glm53_ep_local_20260908/'+CONFIG['cpu']+'/local/result.json'
receipt=json.loads((source/receipt_rel).read_text())
paths={'build/glm53/manifest.tsv',receipt_rel,*receipt['contracts']['files']}
for line in (source/'build/glm53/manifest.tsv').read_text().splitlines():
 if not line or line.startswith('#'): continue
 filename,*_=line.split('\t'); paths.add('build/glm53/'+filename)
for rel in sorted(paths):
 pure=PurePosixPath(rel)
 if pure.is_absolute() or '..' in pure.parts or '\\' in rel: raise RuntimeError('unsafe source path '+rel)
 p=source/rel
 if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()): raise RuntimeError('unsafe source '+rel)
 files['source/'+rel]=p.read_bytes()
for name,data in fleet_data.items():files['fleet-state/'+name]=data
assert sum(map(len,files.values())) < 256*1024*1024, 'archive exceeds 256 MiB; inspect before raising bound'
identity=dict(captured_at_unix=time.time(), source=str(source), job=str(job), revision=revision,
 status_porcelain=status, fleet_source=str(fleet), files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()})
files['source_identity.json']=(json.dumps(identity,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,data in files.items():
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
  archive.addfile(info,io.BytesIO(data))
'''
    fetched = subprocess.run(['ssh','-o','BatchMode=yes',args.host,'python3','-'],
                             input=remote.encode(), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=120)
    if fetched.returncode:
        raise RuntimeError('read-only remote archive refused: '+fetched.stderr.decode())
    return fetched.stdout


def unpack(payload, staging, host):
    records=[]; seen=set()
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as archive:
        for member in archive:
            name=safe_path(member.name)
            if name in seen or not member.isfile():
                raise ValueError('duplicate/non-regular archive member: '+name)
            seen.add(name)
            raw=archive.extractfile(member).read()
            target=staging/name
            compress=target.suffix=='.log' or name.startswith('fleet-state/')
            if compress:target=target.with_name(target.name+'.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            data=gzip.compress(raw,mtime=0) if compress else raw
            target.write_bytes(data)
            assert (gzip.decompress(data) if compress else data)==raw
            records.append(dict(remote_archive_path=name,local_path=str(target.relative_to(staging)),
                                raw_bytes=len(raw),raw_sha256=digest(raw),
                                stored_sha256=digest(data),compressed=compress))
    identity=json.loads((staging/'source_identity.json').read_text())
    for entry in records:
        want=identity['files'].get(entry['remote_archive_path'])
        if want is not None:
            assert want['sha256']==entry['raw_sha256'] and want['bytes']==entry['raw_bytes'],entry
    assert set(identity['files'])==seen-{'source_identity.json'}, 'archive file set mismatch'
    (staging/'archive_manifest.json').write_text(json.dumps(dict(remote_host=host,records=records),indent=2)+'\n')


def verify_and_summarize(root,args):
    read=lambda rel:json.loads((root/rel).read_text())
    raw=lambda rel:gzip.decompress((root/(rel+'.gz')).read_bytes())
    identity=read('source_identity.json');done=read('job/capture/completion.json')
    outer=read('job/exit.json');submission=read('job/submission.json')
    assert all(value==args.revision for value in (identity['revision'],done['source_revision'],outer['revision'],submission['revision']))
    assert outer['session']==submission['session']==args.session and identity['status_porcelain']==''
    assert done['exit_code']==outer['exit_code'], 'inner/outer exit mismatch'
    receipt_rel='source/measurements/glm53_ep_local_20260908/'+args.cpu+'/local/result.json'
    cpu=read(receipt_rel)
    existing=args.root/'measurements/glm53_ep_local_20260908'/args.cpu/'local/result.json'
    assert digest((root/receipt_rel).read_bytes())==digest(existing.read_bytes())
    assert cpu['arm']=='local' and cpu['cuda_initialized'] is False
    assert cpu['contracts']['tests_run']==37 and all(cpu['contracts'][k]==0 for k in ('errors','failures','skips'))
    assert len(cpu['remap_compilation'])==24
    variants={entry['label'] for entry in cpu['remap_compilation']}
    assert len(variants)==24
    for path,want in cpu['contracts']['files'].items():assert digest((root/'source'/safe_path(path)).read_bytes())==want,path
    mounted={Path(path).name:value for path,value in cpu['mounted_sources'].items()}
    assert len(mounted)==len(cpu['mounted_sources'])
    for path,want in mounted.items():assert digest((root/'source/build/glm53'/safe_path(path)).read_bytes())==want,path
    cells=done['cells'];actual=[(entry['case'],entry['sanitizer']) for entry in cells]
    assert actual==EXPECTED[:len(actual)] and len(actual)<=17,'unexpected cell ordering'
    if done['exit_code']==0:assert actual==EXPECTED,'success missing required cells'
    results=[]
    for entry in cells:
        case,tool=entry['case'],entry['sanitizer'];label=(tool+'-' if tool else '')+case
        evidence_path=root/'job/capture'/(label+'.json')
        evidence=json.loads(evidence_path.read_text()) if evidence_path.exists() else None
        result=dict(label=label,case=case,sanitizer=tool,exit_code=entry.get('exit_code'),
                    started_kst=kst(entry['started']),ended_kst=kst(entry['ended']),
                    verdict=evidence.get('verdict') if evidence else 'NO_PROBE_JSON',timing={})
        if evidence is not None:
            result['phase']=evidence.get('phase'); result['error']=evidence.get('error')
            assert evidence.get('performance_acceptance') is False,label
            if 'provenance' in evidence:assert evidence['provenance']==mounted,label
            if evidence.get('verdict')=='PASS':
                assert evidence.get('phase')=='complete',label
                result['probe_numerics_verdict']='PASS'
                if not tool:assert entry.get('exit_code')==0,label
                elif entry.get('exit_code')!=0:result['verdict']='FAIL_SANITIZER_EXIT'
                if case=='remap':
                    checks=evidence['checks']
                    assert len(checks)==24 and {check['label'] for check in checks}==variants,label
                    assert all(check['changed_storage'] and check['rows']==4097 for check in checks),label
                    result['exact_byte_variants_passed']=24
                else:
                    assert evidence['provenance']==mounted and evidence['timing_scope']=='EP remap plus MoE wrapper',label
                    assert evidence['sanitize']==bool(tool),label
                    controls=[item for group in evidence['controls'] for item in group]
                    candidates=evidence['candidate']
                    assert len(controls)==6 and len(candidates)==4,label
                    assert all(item['bad_rows']==0 and item['max_row_relative_l2']<=.02 and item['max_row_relative_abs']<=.04 for item in controls),label
                    assert all(item['bad_rows']==0 and all(math.isfinite(v) for v in item.values()) for item in candidates),label
                    result.update(rows=evidence['rows'],routing=evidence['routing'],controls_checked=6,
                                  candidate_comparisons=4,source_hashes_match_cpu7=True,
                                  controls_max_l2=max(item['max_row_relative_l2'] for item in controls),
                                  controls_max_peak=max(item['max_row_relative_abs'] for item in controls),
                                  candidate_max_l2=max(item['max_row_relative_l2'] for item in candidates),
                                  candidate_max_peak=max(item['max_row_relative_abs'] for item in candidates))
                    for metric in ('wall_ms','device_ms'):
                        if metric not in evidence['timing']:continue
                        arms=evidence['timing'][metric]
                        assert len(arms['compact'])==len(arms['local'])==8,label
                        assert all(v>0 and math.isfinite(v) for values in arms.values() for v in values),label
                        baseline,candidate=(statistics.median(arms[arm]) for arm in ('compact','local'))
                        result['timing'][metric]=dict(baseline_compact_median=baseline,
                            candidate_local_median=candidate,speedup_ratio=baseline/candidate,
                            speedup_pct=100*(baseline/candidate-1),latency_reduction_pct=100*(1-candidate/baseline),
                            samples_per_arm=8,calls_per_sample=3)
                    if result['timing']:
                        assert not tool and evidence['routing'] in ('balanced','concentrated'),label
                        assert math.isclose(result['timing']['wall_ms']['speedup_pct'],evidence['timing']['wall_speedup_pct']),label
                if tool and 'sanitizer_summary' in entry:
                    log=raw('job/capture/'+label+'.log');text=log.decode()
                    summary=entry['sanitizer_summary']
                    assert summary['tool']==tool and summary['verdict']=='PASS' and summary['log_sha256']==digest(log),label
                    assert not re.search(r'^\s*=+\s*(?:ERROR|FATAL)\s*:',text,re.I|re.M),label
                    errors=re.findall(r'^\s*=+\s*(ERROR SUMMARY:.*)$',text,re.M)
                    races=re.findall(r'^\s*=+\s*(RACECHECK SUMMARY:.*)$',text,re.M)
                    assert all(line.strip()=='ERROR SUMMARY: 0 errors' for line in errors),label
                    assert all(line.strip()=='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' for line in races),label
                    required=errors if tool=='memcheck' else races
                    assert required and required==summary['summaries'],label
                    result['sanitizer_summary']=summary
                elif tool and entry.get('exit_code')==0:
                    result['verdict']='INCOMPLETE_SANITIZER_SUMMARY'
        if done['exit_code']==0:
            assert result['verdict']=='PASS',label
            if tool:assert 'sanitizer_summary' in result,label
        results.append(result)
    before_path=root/'job/capture/before.json'
    recovery=dict(exact_original_restored=False,public_refresh_performed=False)
    if before_path.exists():
        before=json.loads(before_path.read_text())
        if all(value is not None for value in before.values()):
            original_ids={node:value['id'] for node,value in before.items()}
            recovery['original_container_ids']=original_ids
            restored_path=root/'job/capture/restored.json'
            if restored_path.exists():
                restored=json.loads(restored_path.read_text())
                immutable=lambda state:{k:v for k,v in state.items() if k not in ('running','started')}
                assert set(before)==set(restored) and len(before)==4,'node set mismatch'
                for node,original in before.items():
                    assert immutable(original)==immutable(restored[node]) and restored[node]['running'],node
                assert done.get('restored_original') is True
                recovery.update(exact_original_restored=True,original_endpoint_port=before['local']['port'],
                    health_evidence='restored.json is emitted after archived wait_restore verifies original identities/running and original endpoint /health=200; no separate HTTP transcript')
            else:recovery['restore_error']='original containers were present but restored.json is absent'
        elif all(value is None for value in before.values()):recovery['incoming_service']='absent'
        else:recovery['incoming_service']='partial; no exact-restoration claim'
    else:recovery['incoming_service']='no service snapshot; failure may precede inventory'
    if (root/'job/capture/public-restored.json').exists():
        public=read('job/capture/public-restored.json')
        assert len(public)==4 and all(value is not None and value['running'] for value in public.values()) and public['local']['port']==8000
        recovery['public_refresh_performed']=True
    recovery['public_restore_result']=done.get('public_restore')
    fleet={name:raw('fleet-state/'+name).decode() for name in ('log','events.log','holder','queue')}
    release={name:[line for line in fleet[name].splitlines() if len(line.split())>=3 and line.split()[1]=='release' and line.split()[2]==args.session] for name in ('log','events.log')}
    assert len(release['log'])==len(release['events.log'])==1,'release evidence absent/ambiguous'
    assert args.session not in fleet['holder'] and args.session not in fleet['queue']
    recovery.update(fleet_release_log=release['log'][0],fleet_release_event=release['events.log'][0],session_absent_from_later_holder_and_queue=True)
    preflight_path=root/'job/capture/sanitizer-preflight.json'
    preflight=json.loads(preflight_path.read_text()) if preflight_path.exists() else None
    if done['exit_code']==0:
        assert preflight and preflight['verdict']=='PASS' and not preflight['cuda_devices_exposed'] and preflight['exit_code']==0
        assert recovery['exact_original_restored'] or recovery['public_refresh_performed'] or done.get('public_restore')=='handed to queued boot per fleet policy'
    plain=sum(result['verdict']=='PASS' and result['case']!='remap' and not result['sanitizer'] for result in results)
    return dict(session=args.session,revision=args.revision,cpu_receipt=args.cpu,
        verdict='PASS component gates' if done['exit_code']==0 else 'PARTIAL_OR_FAILED terminal attempt',
        performance_acceptance=False,inner_exit_code=done['exit_code'],outer_exit_code=outer['exit_code'],
        error=done.get('error'),completion_at_kst=kst(done['ended']),outer_exit_at_kst=kst(outer['ended']),
        expected_cells=17,observed_cells=len(cells),plain_moe_cases_passed=plain,
        timed_cases=sum(bool(result['timing']) for result in results),cells=results,recovery=recovery,
        sanitizer_preflight=preflight,source_validation=dict(frozen_git_clean=True,
            cpu_receipt_matches_existing_archive=True,contract_files_matched=len(cpu['contracts']['files']),
            mounted_sources_matched=len(mounted),cpu_tests=37,triton_specializations=24),
        limitations=['Synthetic single-GB10 E72 component; baseline is the existing EP compact top-k1 wrapper with 8192-token pair-slice limit, not production TP4 E288/I512.',
            'Timings include each arm route-remap plus MoE wrapper, with changed input/routes/scales and fixed storage; they exclude shared expert, TP4/EP4 transport, attention/norm and full-model TTFT.',
            'Eight alternating paired samples per arm, three calls per sample; timed only for balanced/concentrated fixtures.',
            'Not directly interchangeable with attempt2 timings, which excluded remap.',
            'No full-model answer quality, decode regression, serving capacity or production performance acceptance; defaults remain off.'])


def publish(staging,out,summary):
    (staging/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Expert-local prefill '+summary['session'], '',
        summary['verdict']+'. '+str(summary['observed_cells'])+'/'+str(summary['expected_cells'])+' cells observed; '+str(summary['plain_moe_cases_passed'])+' plain MoE fixtures passed. Overall exit '+str(summary['inner_exit_code'])+'.', '',
        'Frozen source: '+summary['revision']+'. The matching '+summary['cpu_receipt']+' receipt, source files, job/capture artifacts and independent fleet release records are archived. Exact source and byte checks were performed before publication.', '',
        '| Fixture | Compact wall median (ms) | Local wall median (ms) | Component speedup | Compact device median (ms) | Local device median (ms) |',
        '|---|---:|---:|---:|---:|---:|']
    for cell in summary['cells']:
        if cell['sanitizer'] or cell['case']=='remap':continue
        if cell['timing']:
            w=cell['timing']['wall_ms'];d=cell['timing']['device_ms']
            lines.append(f"| {cell['case']} | {w['baseline_compact_median']:.3f} | {w['candidate_local_median']:.3f} | {w['speedup_ratio']:.3f}x | {d['baseline_compact_median']:.3f} | {d['candidate_local_median']:.3f} |")
        else:lines.append('| '+cell['case']+' | not timed | not timed | '+cell['verdict']+' | not timed | not timed |')
    lines+=['', 'Timing scope: EP remap plus MoE wrapper on one GB10. Both arms use the same source/runtime and inputs. The component baseline is existing EP compact, not production TP4. These measurements exclude shared expert, transport and full-model prefill; attempt2 excluded remap and is not an interchangeable baseline.', '',
        'Stage results are listed in summary.json, including exact remap variant counts, numerical controls and independently revalidated tool-specific sanitizer summaries. A sanitizer probe JSON marked PASS alone is not a sanitizer verdict.', '',
        'Recovery: '+json.dumps(summary['recovery'],sort_keys=True)+'.', '',
        'Job completion: '+summary['completion_at_kst']+'; outer exit: '+summary['outer_exit_at_kst']+'. GPU stage completion, exact original recovery, optional public refresh and normal fleet release are reported separately.', '',
        'Raw files are under job/; source/ retains frozen composed sources, manifest, CPU receipt and receipt-bound test/probe sources. fleet-state/ holds a separate scheduler snapshot. Logs use deterministic gzip. archive_manifest.json stores raw/stored SHA256 and sizes; SHA256SUMS covers every generated and archived file. collect.py records this collection procedure.', '',
        'Full-model TTFT, production TP4/EP4 performance, output quality, decode and serving memory/capacity remain separate gates. performance_acceptance is false; no default promotion.']
    if summary['error']:lines+=['','Terminal failure: '+summary['error']]
    (staging/'README.md').write_text('\n'.join(lines)+'\n')
    shutil.copyfile(__file__,staging/'collect.py')
    files=sorted(path for path in staging.rglob('*') if path.is_file() and path.name!='SHA256SUMS')
    (staging/'SHA256SUMS').write_text(''.join(digest(path.read_bytes())+'  '+str(path.relative_to(staging))+'\n' for path in files))
    for line in (staging/'SHA256SUMS').read_text().splitlines():
        want,name=line.split('  ',1);assert digest((staging/safe_path(name)).read_bytes())==want
    staging.rename(out)
    print(json.dumps(dict(archive=str(out),files=len(files)+1,verdict=summary['verdict'],plain_moe_pass=summary['plain_moe_cases_passed'],cells=summary['observed_cells'],release=summary['recovery']['fleet_release_log']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed-terminal',action='store_true',help='operator/root has confirmed completed/released; script independently verifies')
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--host',default='choiceoh@srv2')
    parser.add_argument('--session',default='eplocal0908v4')
    parser.add_argument('--revision',default='71e804e7aa6b29d6ddf4577809a5fa5e05a999e6')
    parser.add_argument('--source',default='/home/choiceoh/stkernel-ep-local-gpu-0908-4')
    parser.add_argument('--job',default='/tmp/glm53-ep-local-gpu-0908-4')
    parser.add_argument('--cpu',default='cpu7')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if not args.confirmed_terminal:parser.error('--confirmed-terminal required; do not collect a running attempt')
    if not re.fullmatch('[0-9a-f]{40}',args.revision):parser.error('full revision required')
    if not re.fullmatch('[a-zA-Z0-9_-]+',args.session):parser.error('invalid session')
    if not re.fullmatch('cpu[0-9]+',args.cpu):parser.error('invalid CPU receipt name')
    out=args.out or args.root/'measurements/glm53_ep_local_20260908/attempt4'
    staging=out.with_name(out.name+'.collecting')
    if out.exists() or staging.exists():parser.error('output/staging exists; inspect without overwriting')
    payload=fetch(args) # Read-only terminal/release checks precede local archive creation.
    staging.mkdir(parents=True)
    try:
        unpack(payload,staging,args.host)
        summary=verify_and_summarize(staging,args)
        publish(staging,out,summary)
    except BaseException as exc:
        (staging/'VERIFICATION_ERROR.txt').write_text(repr(exc)+'\n')
        raise

if __name__=='__main__':main()
