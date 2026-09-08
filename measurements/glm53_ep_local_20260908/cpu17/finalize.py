#!/usr/bin/env python3
"""Adjudicate only the known CPU17 debug-mtime rejection; preparation only.

The original cpu17.collecting directory and its rejection remain untouched.
This script is not a compiler/GPU run. It preserves every original cubin and
admits only the separately pinned strict ELF checker, never stripped binaries.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import stat
import sys

ROOT = Path('/Users/ost/.worktrees/stkernel2/glm53-prefill-moe-pipeline-20260908')
COLLECTOR = Path('/tmp/glm53_fetch_cpu17_owned_scheduler.py').resolve(strict=True).resolve()
COLLECTOR_SHA = 'eb4f48e9b333fcf6688bfa645e94959dd5b554d02db9b202dfe1561d34c05fac'
CHECKER = Path('/tmp/glm53-cpu17-remap-elf/inspect.py')
CHECKER_SHA = 'f724709672dadae85a7f2800c41fb63d98a0658ce085145b70c89ef1ba983e85'
REVISION = '123fbb01211eaf8cec46fd0965dd9320d02a375b'
MEASUREMENTS = ROOT/'measurements/glm53_ep_local_20260908'
OLD_MTIME = 1788872668
NEW_MTIME = 1788880342
ROUTE_SOURCE = 'frozen-source/build/glm53/glm53_ep_route_remap.py'
LIMIT = 128*2**20


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name):
    p=PurePosixPath(name)
    need(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in name
         and str(p)==name and name!='.', 'unsafe archive path: '+name)
    return name


def regular_bytes(path):
    for p in (path,*path.parents):
        need(not p.is_symlink(), 'symlink refused: '+str(p))
    before=path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size<=64*2**20, 'invalid or oversized file')
    raw=path.read_bytes(); after=path.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
         and len(raw)==before.st_size, 'input changed during read')
    return raw


def snapshot(root):
    need(root.is_dir() and not root.is_symlink(), 'existing regular rejected archive required')
    files={}; total=0
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'symlink in rejected archive')
        if path.is_dir(): continue
        raw=regular_bytes(path); total+=len(raw)
        need(total<=LIMIT and len(files)<4096, 'archive bounds exceeded')
        files[safe(path.relative_to(root).as_posix())]=raw
    return files


def load_originals(files):
    manifest=json.loads(files['archive-manifest.json'])
    need(manifest['host']=='choiceoh@srv2', 'archive host differs')
    raw={}; stored_names=set(); total=0
    for row in manifest['records']:
        original=safe(row['original_path']); stored=safe(row['stored_path'])
        need(original not in raw and stored not in stored_names, 'duplicate manifest entry')
        stored_names.add(stored); data=files[stored]
        need(len(data)==row['stored_bytes'] and sha(data)==row['stored_sha256'], 'stored hash/size mismatch: '+stored)
        need(stored in (original,original+'.gz'), 'unexpected archive transformation')
        need(row['original_bytes']<=64*2**20, 'oversized original file')
        original_data=gzip.decompress(data) if stored==original+'.gz' else data
        total+=len(original_data); need(total<=LIMIT, 'decompressed archive exceeds limit')
        need(len(original_data)==row['original_bytes'] and sha(original_data)==row['original_sha256'], 'original hash/size mismatch: '+original)
        raw[original]=original_data
    need(set(files)==stored_names|{'archive-manifest.json','cpu16-comparison.json','VERIFICATION_ERROR.txt'}, 'unexpected rejected archive file set')
    identity=json.loads(raw['source-identity.json'])
    need(set(identity['files'])==set(raw)-{'source-identity.json'}, 'identity file set differs')
    for name,info in identity['files'].items():
        need(sha(raw[name])==info['sha256'] and len(raw[name])==info['bytes'], 'identity file hash mismatch')
    return raw,identity


def import_pinned(name,path,want):
    need(len(want)==64 and all(c in '0123456789abcdef' for c in want), 'exact checker SHA256 required')
    need(sha(regular_bytes(path))==want, 'pinned helper source changed: '+str(path))
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    need(sha(regular_bytes(path))==want, 'helper changed while importing')
    return module


def adjudicate(collector,checker,files,raw,identity):
    need(identity['revision']==REVISION and identity['source']==collector.SOURCE
         and identity['job']==collector.JOB and identity['clean'] is True, 'frozen job identity differs')
    need((identity['tests'],identity['contract_files'],identity['mounted_files'],identity['remap_compilations'])==(146,29,13,24), 'CPU17 counts differ')
    proof=json.loads(raw['local/result.json'])
    baseline_bytes=regular_bytes(MEASUREMENTS/'cpu16/local/result.json')
    need(sha(baseline_bytes)==collector.CPU16_SHA, 'CPU16 proof changed')
    baseline=json.loads(baseline_bytes)
    old={row['label']:row for row in baseline['remap_compilation']}
    rows=proof['remap_compilation']
    need(len(rows)==len(old)==24 and {row['label'] for row in rows}==set(old), 'remap variants differ')
    expected=[]
    for row in rows:
        label=safe(row['label'])
        expected += ['CPU16 remap descriptor differs: '+label,
                     'CPU16 remap artifact differs: local/remap/'+label+'/kernel.cubin']
        strip=lambda entry:{k:v for k,v in entry.items() if k!='cubin_sha256'}
        need(strip(row)==strip(old[label]) and row['cubin_sha256']!=old[label]['cubin_sha256'], 'descriptor differs beyond cubin SHA')
    comparison=collector.compare_baseline(raw,REVISION)
    need(comparison==json.loads(files['cpu16-comparison.json']), 'recomputed original rejection differs')
    need(comparison['verdict']=='REJECT_DIFFERENCE' and comparison['accepted_compile_proof'] is False
         and comparison['differences']==expected and len(expected)==48, 'rejection is not exactly the known 48 entries')
    need(files['VERIFICATION_ERROR.txt'].decode().strip()=="ValueError('CPU16 source/artifact difference; see cpu16-comparison.json, not accepted')", 'original rejection error differs')
    route=identity['files'][ROUTE_SOURCE]
    need(route['mtime_ns']//10**9==NEW_MTIME and route['source']==collector.SOURCE+'/build/glm53/glm53_ep_route_remap.py', 'candidate source mtime provenance differs')
    need(route['sha256']==sha(raw[ROUTE_SOURCE])==proof['mounted_sources'][
        '/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/glm53_ep_route_remap.py'], 'route source bytes differ')
    reports=[]
    for row in rows:
        label=row['label']; name='local/remap/'+label+'/kernel.cubin'
        original=gzip.decompress(regular_bytes(MEASUREMENTS/'cpu16'/(name+'.gz')))
        candidate=raw[name]
        need(sha(original)==old[label]['cubin_sha256'] and sha(candidate)==row['cubin_sha256'], 'remap cubin source hash mismatch')
        report=checker.compare(original,candidate,expected_new_mtime=NEW_MTIME)
        need(all(report[key] is True for key in ('all_non_mtime_bytes_exact','all_section_headers_exact',
            'all_program_headers_exact','executable_sections_exact','all_alloc_sections_exact'))
             and report['cpu16_mtime']==OLD_MTIME and report['cpu17_mtime']==NEW_MTIME
             and len(report['raw_differences'])==4, 'strict debug-mtime checker did not establish the exact expected difference')
        reports.append(dict(label=label,report=report))
    from glm53_ep_local_evidence import validate_compile_evidence
    checked=validate_compile_evidence(ROOT,MEASUREMENTS/'cpu17.collecting/local/result.json')
    need(checked==proof and checked['contracts']['tests_run']==146 and checked['cuda_initialized'] is False, 'current-source CPU17 proof differs')
    return dict(schema=1,verdict='PASS_COMPILE_IDENTITY_DEBUG_MTIME_ONLY',accepted_compile_proof=True,
        original_rejection_preserved=True,original_comparison_sha256=sha(files['cpu16-comparison.json']),
        full_remap_cubins_byte_equal=False,executable_and_all_other_bytes_equal=True,
        cpu16_source_mtime_seconds=OLD_MTIME,cpu17_source_mtime_seconds=NEW_MTIME,
        cpu16_source_mtime_stat_independently_verified=False,
        candidate_source_metadata=route,remap_cubins=reports,
        gpu_numerical_acceptance=False,performance_acceptance=False,
        scope='Strict debug timestamp adjudication only; actual original full cubin bytes are retained. GPU v5 FAIL remains unresolved.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checker',type=Path,default=CHECKER)
    parser.add_argument('--checker-sha256',default=CHECKER_SHA)
    args=parser.parse_args()
    args.checker=args.checker.resolve(strict=True)
    args.checker=args.checker.resolve(strict=True)
    need(args.checker_sha256==CHECKER_SHA, 'only the independently reviewed checker is admitted')
    original=MEASUREMENTS/'cpu17.collecting'; finalizing=MEASUREMENTS/'cpu17.finalizing'; destination=MEASUREMENTS/'cpu17'
    need(not finalizing.exists() and not destination.exists(), 'finalizing/output already exists; no overwrite')
    files=snapshot(original); raw,identity=load_originals(files)
    collector=import_pinned('cpu17_owned_collector',COLLECTOR,COLLECTOR_SHA)
    checker=import_pinned('cpu17_debug_mtime_checker',args.checker,args.checker_sha256)
    need(collector.git_bytes('rev-parse','HEAD').decode().strip()==REVISION, 'local HEAD is not actual CPU17 revision')
    report=adjudicate(collector,checker,files,raw,identity)
    report['helpers']=dict(collector_sha256=COLLECTOR_SHA,checker_sha256=args.checker_sha256)
    before=json.loads(collector.remote('verify',REVISION,expected_files=identity['files']))
    need(before['verified_after_transfer'] is True, 'remote original identity check failed before copy')
    finalizing.mkdir(exist_ok=False)
    try:
        for name,data in files.items():
            target=finalizing/name; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(data)
        need(snapshot(finalizing)==files and snapshot(original)==files, 'copy or preserved original changed')
        after=json.loads(collector.remote('verify',REVISION,expected_files=identity['files']))
        need(after['verified_after_transfer'] is True, 'remote original identity changed after copy')
        from glm53_ep_local_evidence import validate_compile_evidence
        validate_compile_evidence(ROOT,finalizing/'local/result.json')
        need(sha(regular_bytes(COLLECTOR))==COLLECTOR_SHA and sha(regular_bytes(args.checker))==args.checker_sha256, 'adjudication helper changed')
        (finalizing/'debug-mtime-adjudication.json').write_text(json.dumps(report,indent=2)+'\n')
        (finalizing/'finalization-remote-verification.json').write_text(json.dumps(dict(before_copy=before,after_copy=after),indent=2)+'\n')
        (finalizing/'collect.py').write_bytes(regular_bytes(COLLECTOR))
        (finalizing/'debug-mtime-checker.py').write_bytes(regular_bytes(args.checker))
        (finalizing/'finalize.py').write_bytes(regular_bytes(Path(__file__).resolve()))
        (finalizing/'README.md').write_text('CPU17 actual normal --cpu PASS: 146 tests, 29 contracts, 13 mounted files, 24 remap variants; CUDA uninitialized.\n\n'
            'The original exact-byte comparison REJECT and VERIFICATION_ERROR.txt are preserved unchanged, both here and in cpu17.collecting. All CuTe artifacts and remap PTX are byte-identical to CPU16. The 24 remap full cubins are NOT byte-identical: the strict pinned checker accepts only the recorded source-mtime debug fields; executable and all other bytes match.\n\n'
            'Original cubin bytes were not stripped or normalized. Original/stored SHA and sizes remain in archive-manifest.json. Remote originals, current source contracts, and the pinned capsule were checked before and after copying. This is compile identity adjudication, not GPU numerical or speed acceptance; GPU v5 FAIL remains unresolved.\n')
        sums=[]
        for name,data in snapshot(finalizing).items(): sums.append(sha(data)+'  '+name+'\n')
        (finalizing/'SHA256SUMS').write_text(''.join(sums))
        for line in (finalizing/'SHA256SUMS').read_text().splitlines():
            want,name=line.split('  ',1); need(sha(regular_bytes(finalizing/safe(name)))==want, 'final stored hash differs')
        need(snapshot(original)==files and not destination.exists(), 'original changed or output appeared')
        finalizing.rename(destination)
        print(json.dumps(dict(destination=str(destination),verdict=report['verdict'],full_cubins_byte_equal=False,
                             gpu_numerical_acceptance=False,performance_acceptance=False)))
    except BaseException as exc:
        (finalizing/'FINALIZATION_ERROR.txt').write_text(repr(exc)+'\n')
        raise


if __name__=='__main__':
    main()
