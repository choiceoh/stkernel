#!/usr/bin/env python3
"""Read-only continuity check of four captured canary JSON receipts.

Example:
  python3 /tmp/glm53_onepass10_compare_canary.py \
    --old-dir measurements/glm53_ep_local_20260908/onepass9-startup-failed/failures \
    --new-dir /tmp/onepass10-canary-receipts \
    --old-source measurements/glm53_ep_local_20260908/onepass9-startup-failed/source/mounted-hashes.json \
    --new-source /tmp/onepass10-frozen-mounted-hashes.json

Source arguments are independently verified absolute runtime path -> SHA256
JSON maps, not an expected map inferred from the receipts under comparison.
This compares source bytes; receipts contain no Git revision, image digest,
or authenticated host identity. The caller must bind those separately.
Exit 0: continuity + all four bounded canaries PASS. Exit 1: invalid/mismatch.
Exit 2: continuity matches, but at least one new canary explicitly failed.
No network, CUDA import, waiting, source write, or archive mutation occurs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

NODES = ('local', '10.10.10.1', '10.10.10.3', '10.10.10.4')
WEIGHTS = ('w13', 'w2', 'sf13', 'sf2')
INPUTS = ('X', 'ids', 'weights', 'fc1_input', 'fc1_alpha', 'fc2_input', 'fc2_alpha')
CASES = ('concentrated6912', 'balanced4096', 'remote4096', 'duplicate4096',
         'zeros4097', 'balanced8192', 'short6')
BASE = '/usr/local/lib/python3.12/dist-packages/'
FI = BASE + 'flashinfer/fused_moe/cute_dsl/blackwell_sm12x/'
SOURCES = dict(dispatch=FI+'moe_dispatch.py', local=FI+'moe_dynamic_ep_local.py',
    selftest=FI+'glm53_ep_local_selftest.py', remap=FI+'glm53_ep_route_remap.py',
    micro=FI+'moe_micro_kernel.py', stock=FI+'_moe_dynamic/gated.py',
    wrapper=BASE+'vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py')
STOCK_SHA = '993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445'
HEX = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def read_json(path):
    require(path.is_file(), 'missing receipt/source file: '+str(path))
    require(path.stat().st_size <= 4*1024*1024, 'oversized JSON: '+str(path))
    raw = path.read_bytes()
    obj = json.loads(raw, object_pairs_hook=reject_duplicates,
                     parse_constant=lambda token: (_ for _ in ()).throw(ValueError('nonfinite JSON: '+token)))
    require(isinstance(obj, dict), 'JSON root is not an object: '+str(path))
    return obj, hashlib.sha256(raw).hexdigest()


def expected_source(path):
    obj, sha = read_json(path)
    require(all(isinstance(k,str) and k.startswith('/') and isinstance(v,str) and HEX.fullmatch(v)
                for k,v in obj.items()), 'source JSON must map absolute runtime paths to SHA256')
    expected = {}
    for role, name in SOURCES.items():
        expected_hash = STOCK_SHA if role == 'stock' else obj.get(name)
        require(isinstance(expected_hash,str) and HEX.fullmatch(expected_hash), 'expected source missing: '+role)
        if role == 'stock' and name in obj:
            require(obj[name] == STOCK_SHA, 'provided stock source differs from pinned inherited kernel')
        expected[role] = dict(path=name,sha256=expected_hash)
    return expected, sha


def identity(tensor, field):
    require(isinstance(tensor,dict), 'tensor identity missing: '+field)
    require(isinstance(tensor.get('dtype'),str) and tensor['dtype'].startswith('torch.'), 'tensor dtype missing: '+field)
    require(isinstance(tensor.get('shape'),list) and tensor['shape'] and all(
        type(v) is int and v > 0 for v in tensor['shape']), 'invalid tensor shape: '+field)
    require(isinstance(tensor.get('sha256'),str) and HEX.fullmatch(tensor['sha256']), 'tensor SHA missing: '+field)
    require(type(tensor.get('data_ptr')) is int and tensor['data_ptr'] > 0, 'tensor pointer missing: '+field)
    return {k:tensor[k] for k in ('dtype','shape','sha256')}


def validate(receipt, source, *, old):
    require(receipt.get('schema') == 1 and receipt.get('seed') == 905308, 'wrong canary schema/seed')
    require(receipt.get('verdict') in (('FAIL',) if old else ('PASS','FAIL')), 'missing/unsupported terminal verdict')
    require(type(receipt.get('pid')) is int and receipt['pid'] > 0, 'missing process identity')
    require(isinstance(receipt.get('device'),str) and receipt['device'].startswith('cuda:'), 'missing CUDA device identity')
    for key in ('started_at','completed_at'):
        require(type(receipt.get(key)) in (int,float) and math.isfinite(receipt[key]), 'missing terminal timestamp: '+key)
    require(receipt['completed_at'] >= receipt['started_at'], 'completion precedes start')
    require(receipt.get('provenance',{}).get('source') == source, 'exact source role/path/hash mismatch')
    versions = receipt.get('provenance',{}).get('versions')
    require(isinstance(versions,dict) and set(versions) == {'torch','cuda.bindings','flashinfer'}, 'runtime identity missing')
    for name, data in versions.items():
        require(isinstance(data,dict) and isinstance(data.get('version'),str), 'runtime version missing: '+name)
        require(isinstance(data.get('sha256'),str) and HEX.fullmatch(data['sha256']), 'runtime source SHA missing: '+name)
    require(set(receipt.get('weights',{})) == set(WEIGHTS), 'synthetic weight identity set differs')
    weights = {name:identity(receipt['weights'][name],name) for name in WEIGHTS}
    cases = receipt.get('cases')
    require(isinstance(cases,list) and cases and cases[0].get('case') == CASES[0], 'first case is not concentrated6912')
    inputs = cases[0].get('inputs')
    require(isinstance(inputs,list) and len(inputs) == 2, 'initial/changed identities are incomplete')
    normalized = []
    for phase, record in enumerate(inputs):
        require(isinstance(record,dict) and set(record) == set(INPUTS), 'input tensor identity set differs')
        normalized.append({name:identity(record[name],str(phase)+':'+name) for name in INPUTS})
    for name in INPUTS:
        require(inputs[0][name]['data_ptr'] == inputs[1][name]['data_ptr'], 'changed fixture replaced storage: '+name)
    aliases = [[a,b] for i,a in enumerate(INPUTS) for b in INPUTS[i+1:]
               if inputs[0][a]['data_ptr'] == inputs[0][b]['data_ptr']]
    if receipt['verdict'] == 'PASS':
        require('error' not in receipt and 'cleanup_error' not in receipt, 'PASS receipt retains a failure')
        for key in ('scratch_cache_restored','caller_scale_storage_unchanged','compiled_kernel_cache_retained'):
            require(receipt.get(key) is True, 'PASS receipt missing cleanup guarantee: '+key)
        require([case.get('case') for case in cases] == list(CASES), 'PASS receipt omitted/reordered cases')
        for case in cases:
            require(case.get('verdict') == 'PASS' and case.get('phase') == 'complete', 'PASS receipt contains incomplete case')
            require(len(case.get('candidate',[])) == 6 and len(case.get('controls',[])) == 2,
                    'PASS case omitted fixed candidate/control calls')
            results = list(case['candidate'])
            for group in case['controls']:
                require(isinstance(group,list) and len(group)==3, 'PASS case omitted stock controls')
                results.extend(group)
            require(all(isinstance(result,dict) and result.get('bad_rows') == 0 for result in results),
                    'PASS case retained numerical failure')
    else:
        require(isinstance(receipt.get('error'),str) or isinstance(receipt.get('cleanup_error'),str), 'FAIL receipt has no error')
    return dict(weights=weights,inputs=normalized,aliases=aliases,versions=versions,device=receipt['device'])


def differences(old, new, prefix=''):
    if type(old) is not type(new):
        yield dict(field=prefix,old_type=type(old).__name__,new_type=type(new).__name__)
    elif isinstance(old,dict):
        for key in sorted(set(old)|set(new)):
            field = prefix+'.'+str(key) if prefix else str(key)
            if key not in old or key not in new:
                yield dict(field=field,old_present=key in old,new_present=key in new)
            else:
                yield from differences(old[key],new[key],field)
    elif isinstance(old,list):
        if len(old)!=len(new):
            yield dict(field=prefix,old_length=len(old),new_length=len(new))
        for i,(a,b) in enumerate(zip(old,new)):
            yield from differences(a,b,prefix+'['+str(i)+']')
    elif old != new:
        yield dict(field=prefix,old=old,new=new)


def compare(args):
    old_source,old_sha=expected_source(args.old_source)
    new_source,new_sha=expected_source(args.new_source)
    report=dict(schema=1,scope='same-input bounded startup canary; no performance/sanitizer/adoption acceptance',
        old_source_manifest_sha256=old_sha,new_source_manifest_sha256=new_sha,
        source_revisions_verified=False,host_binding='caller-provided receipt filenames',nodes={},
        performance_acceptance=False,full_sanitizer_acceptance=False,adoption_acceptance=False)
    valid=True; all_pass=True
    for host in NODES:
        item={};report['nodes'][host]=item
        try:
            old,old_hash=read_json(args.old_dir/(host+'.json'))
            new,new_hash=read_json(args.new_dir/(host+'.json'))
            item.update(old_receipt_sha256=old_hash,new_receipt_sha256=new_hash,
                        old_verdict=old.get('verdict'),new_verdict=new.get('verdict'))
            before=validate(old,old_source,old=True);after=validate(new,new_source,old=False)
            delta=list(differences(before,after))
            item.update(identity_match=not delta,difference_count=len(delta),differences=delta[:32],
                        within_boot_storage_reused=True)
            valid &= not delta
            all_pass &= new['verdict']=='PASS'
            if new['verdict']=='FAIL':
                item['failure_summary']=str(new.get('error',new.get('cleanup_error')))[:240]
        except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
            item.update(identity_match=False,error=str(exc)[:300]);valid=False;all_pass=False
    report.update(identity_match=valid,all_four_new_canaries_pass=valid and all_pass,
        verdict='MATCH_AND_CANARY_PASS' if valid and all_pass else 'MATCH_BUT_CANARY_FAIL' if valid else 'REJECT')
    return report,0 if valid and all_pass else 2 if valid else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('old-dir','new-dir','old-source','new-source'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    try:report,code=compare(args)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        report=dict(verdict='REJECT',identity_match=False,all_four_new_canaries_pass=False,error=str(exc)[:300]);code=1
    print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
    return code


if __name__ == '__main__':sys.exit(main())
