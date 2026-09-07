"""All-row INT8/FP8 comparison on the same actual MoE partials; not acceptance."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from glm53_moe_m64_fp8_diagnostic import CASES, SEED_OFFSETS, TRIALS, plan, order, pair_summary, row_metrics
from glm53_prefill_int8_check import check_packet, codec_cases

MARKER='MOE_M64_INT8_DIAGNOSTIC_COMPLETE'
ARMS=('baseline','repeat','control','candidate')
PHASES=('fp8','int8','partial')


@contextmanager
def rs_mode(h,int8):
    previous=h._RS_INT8
    h._RS_INT8=int8
    try:yield
    finally:h._RS_INT8=previous


def source_hashes():
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
        'glm53_moe_m64_int8_diagnostic.py','glm53_prefill_int8_check.py','glm53_moe_m64_fp8_diagnostic.py')}


def quality(torch,value,reference):
    a,b=value.float(),reference.float()
    l2=(a-b).norm(dim=1)/b.norm(dim=1).clamp_min(1e-6)
    peak=(a-b).abs().amax(dim=1)/b.abs().amax(dim=1).clamp_min(1e-6)
    return dict(rows=a.shape[0],finite=bool(torch.isfinite(a).all() & torch.isfinite(b).all()),
        median_l2=float(l2.median()),max_l2=float(l2.max()),median_peak=float(peak.median()),max_peak=float(peak.max()))


def completion(records,preflight,provenance):
    if [(r['rows'],r['skew'],r['seed'],r['trial']) for r in records]!=plan():
        raise ValueError('exact 72-trial all-row coverage required')
    if [r['rank'] for r in preflight['codec']]!=list(range(4)) or [r['rank'] for r in preflight['short']]!=list(range(4)):
        raise ValueError('all four codec and short-control ranks required')
    expected=[(n,c) for n in (128,129,130,131,4095,4096,4097,8192) for c in ('zero','random','ties','extreme')]
    for rank in preflight['codec']:
        if [(r['rows'],r['case']) for r in rank['cases']]!=expected or any(
            r['bad_bytes'] or not r['cpu_reference'] or not r['finite'] or not r['source_unchanged'] for r in rank['cases']):
            raise ValueError('CPU codec fidelity cases incomplete or failed')
    for rank in preflight['short']:
        if [(r['rows'],r['gather_equal'],r['reduce_equal']) for r in rank['cases']]!=[(2128,True,True),(4095,True,True)]:
            raise ValueError('short BF16 identity cases incomplete or failed')
    groups={}
    for record in records:
        rows=record['rows']
        if tuple(record['order'])!=order(record['trial']):raise ValueError('alternating order required')
        for phase in PHASES:
            pairs=record[phase]
            if [r['rank'] for r in pairs]!=list(range(4)) or any(r['rows']!=(rows if phase=='partial' else rows//4) for r in pairs):
                raise ValueError('all rows and ranks required in each phase')
            key=(rows,record['skew'],record['seed'],phase)
            g=groups.setdefault(key,dict(rows=rows,skew=key[1],seed=key[2],phase=phase,candidate_bad=0,control_bad=0,finite=True))
            for r in pairs:
                g['candidate_bad']+=r['candidate_bad'];g['control_bad']+=r['control_bad'];g['finite']&=r['finite']
        if [r['rank'] for r in record['checks']]!=list(range(4)):
            raise ValueError('all four packet/reference ranks required')
        for rank in record['checks']:
            if set(rank['arms'])!=set(ARMS):raise ValueError('all four arms require fidelity evidence')
            for r in rank['arms'].values():
                if r['bad_bytes'] or not r['source_unchanged'] or not r['finite'] or not r['output_reference_equal'] or not r['gather_unchanged'] or not r['capture_unchanged']:
                    raise ValueError('packet/reference/source fidelity failed')
                if r['rows']!=rows or r['cpu_reference']!=(record['trial']==0):
                    raise ValueError('first-trial CPU and every-trial full-row references required')
        if [r['rank'] for r in record['quality']]!=list(range(4)):
            raise ValueError('all four quality ranks required')
        for rank in record['quality']:
            if set(rank['arms'])!=set(ARMS):raise ValueError('all arms require quantization evidence')
            for arm in rank['arms'].values():
                if set(arm)!={'fp8','int8'} or any(v['rows']!=rows//4 or not v['finite'] for v in arm.values()):
                    raise ValueError('finite full-row quantization evidence required')
    return dict(verdict=MARKER,trials=len(records),serving_gate=False,numerical_acceptance=False,
        thresholds=dict(l2=.02,peak=.04,repeat_multiplier=3),groups=list(groups.values()),
        preflight=preflight,provenance=provenance,source_hashes=source_hashes(),
        population='all rows in the fixed synthetic cases; no model quality or speed claim')


def run(*,torch,h,rank,provenance,reports,require,case_factory):
    require(not h._RS_INT8,'original FP8 fixture must begin with INT8 disabled')
    hashes=source_hashes();require(all(r==hashes for r in reports(hashes)),'diagnostic source differs across ranks')
    preflight=dict(codec=reports(codec_cases(torch,h,rank)))
    short=[]
    for rows in (2128,4095):
        generator=torch.Generator(device='cuda').manual_seed(9271+rows+rank)
        x=torch.randn((rows,4096),generator=generator,device='cuda',dtype=torch.bfloat16)
        shard=h.prefill_shard(x)
        with patch.object(h,'_reduce_scatter_int8',side_effect=AssertionError('short call entered INT8')):
            with rs_mode(h,False):a=h.prefill_all_gather(shard,num_tokens=rows);b=h.prefill_reduce_scatter(x)
            with rs_mode(h,True):c=h.prefill_all_gather(shard,num_tokens=rows);d=h.prefill_reduce_scatter(x)
        short.append(dict(rows=rows,gather_equal=torch.equal(a.view(torch.int16),c.view(torch.int16)),
            reduce_equal=torch.equal(b.view(torch.int16),d.view(torch.int16))))
    preflight['short']=reports(dict(rank=rank,cases=short))
    if rank==0:print(json.dumps(dict(kind='MOE_M64_INT8_PREFLIGHT',**preflight)),flush=True)
    require(all(not c['bad_bytes'] and c['source_unchanged'] and c['finite'] for r in preflight['codec'] for c in r['cases']),
            'INT8 codec does not match CPU reference')
    require(all(c['gather_equal'] and c['reduce_equal'] for r in preflight['short'] for c in r['cases']),
            'short BF16 behavior changed')
    records=[]
    for rows,skew in CASES:
        for offset in SEED_OFFSETS:
            seed=9211+rows+offset;call,_,unchanged=case_factory(rows,skew,seed)
            for trial in range(TRIALS):
                captures=dict(baseline=call(False),repeat=call(False))
                captures.update({arm:call(arm=='candidate') for arm in order(trial)})
                record=dict(kind='MOE_M64_INT8_DIAGNOSTIC_TRIAL',rows=rows,skew=skew,seed=seed,trial=trial,order=list(order(trial)))
                int8,checks,quantization={},{},{}
                for arm in ARMS:
                    c=captures[arm];partial=c['partial']
                    with rs_mode(h,True):int8[arm]=h.prefill_reduce_scatter(partial)
                    check,decoded=check_packet(torch,h,partial,cpu_reference=trial==0)
                    comm=h._check(partial)
                    reference=torch.empty_like(c['output'],dtype=torch.float32)
                    comm.reduce_scatter(reference,decoded)
                    check.update(output_reference_equal=torch.equal(int8[arm].view(torch.int16),reference.to(torch.bfloat16).view(torch.int16)),
                        gather_unchanged=c['gather_unchanged'],capture_unchanged=c['source_unchanged'])
                    checks[arm]=check
                    exact=torch.empty_like(reference);comm.reduce_scatter(exact,partial.float())
                    quantization[arm]=dict(fp8=quality(torch,c['output'],exact),int8=quality(torch,int8[arm],exact))
                record['checks']=reports(dict(rank=rank,arms=checks))
                record['quality']=reports(dict(rank=rank,arms=quantization))
                for phase,values in (('fp8',{a:c['output'] for a,c in captures.items()}),('int8',int8),
                                     ('partial',{a:c['partial'] for a,c in captures.items()})):
                    record[phase]=reports(pair_summary(row_metrics(torch,values['control'],values['baseline'],values['repeat']),
                        row_metrics(torch,values['candidate'],values['baseline'],values['repeat']),rank=rank,
                        row_offset=0 if phase=='partial' else rank*(rows//4)))
                records.append(record)
                if rank==0:print(json.dumps(record,allow_nan=False),flush=True)
                require(all(not c['bad_bytes'] and c['source_unchanged'] and c['finite'] and c['output_reference_equal']
                    and c['gather_unchanged'] and c['capture_unchanged'] for r in record['checks'] for c in r['arms'].values()),
                    'INT8 packet/reference fidelity failed')
                require(unchanged() and not h._RS_INT8,'input or original codec setting changed')
    result=completion(records,preflight,provenance)
    if rank==0:print(json.dumps(result,allow_nan=False),flush=True)
    return result
