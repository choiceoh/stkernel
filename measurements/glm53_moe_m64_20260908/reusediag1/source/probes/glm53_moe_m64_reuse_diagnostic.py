"""Bounded local M64 reuse comparison; collection never approves serving."""
import hashlib
import json
from pathlib import Path

from glm53_moe_m64_fp8_diagnostic import order, pair_summary
from glm53_moe_m64_fp8_trace import decode_arrays, encode_arrays, metrics

MARKER='MOE_M64_REUSE_DIAGNOSTIC_COMPLETE'
CASES=tuple((rows,skew) for rows in (6144,6912,8192) for skew in (False,True))
PHASES=('original','changed')
TRIALS=4
MAX_TRACE_ROWS=128


def plan():
    return [(rows,skew,phase,trial) for rows,skew in CASES
            for phase in PHASES for trial in range(TRIALS)]


def completion(records,provenance):
    if [(r['rows'],r['skew'],r['phase'],r['trial']) for r in records]!=plan():
        raise ValueError('missing, duplicate or reordered local reuse trials')
    groups={}
    for record in records:
        pair=record['comparison']
        if tuple(record['order'])!=order(record['trial']) or pair['rows']!=record['rows']:
            raise ValueError('wrong order or row coverage')
        if not record['input_unchanged'] or not record['retained_unchanged']:
            raise ValueError('input or retained output was modified')
        failures=pair['failures']
        if [r['row'] for r in failures]!=record['trace_rows']:
            raise ValueError('missing failure trace rows')
        if len(failures)>MAX_TRACE_ROWS:
            raise ValueError('failure payload budget exceeded; no rows may be dropped')
        for key in ('control','candidate'):
            if sum(bool(r[key+'_bad']) for r in failures)!=pair[key+'_bad']:
                raise ValueError('failure count disagrees with retained rows')
        key=(record['rows'],record['skew'],record['phase'])
        group=groups.setdefault(key,dict(rows=key[0],skew=key[1],phase=key[2],
            control_bad=0,candidate_bad=0,candidate_only=0,control_only=0,both_bad=0,finite=True))
        for field in ('control_bad','candidate_bad','candidate_only','control_only','both_bad'):
            group[field]+=pair[field]
        group['finite'] &= pair['finite']
    return dict(verdict=MARKER,serving_gate=False,numerical_acceptance=False,
        trials=len(records),thresholds=dict(l2=.02,peak=.04,repeat_multiplier=3),
        grouping='descriptive row-trial counts; reused trials are not independent experiments',
        groups=list(groups.values()),provenance=provenance,
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def verify_log(log,source):
    records=[json.loads(l) for l in log.splitlines() if l.startswith('{')]
    trials=[r for r in records if r.get('kind')=='MOE_M64_REUSE_DIAGNOSTIC_TRIAL']
    finals=[r for r in records if r.get('verdict')==MARKER]
    if len(finals)!=1:raise ValueError('one diagnostic completion required')
    provenance={}
    for line in (Path(source)/'build/glm53/manifest.tsv').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name=line.split('\t')[0]
        provenance[name]=hashlib.sha256((Path(source)/'build/glm53'/name).read_bytes()).hexdigest()
    if completion(trials,provenance)!=finals[0]:raise ValueError('source or completion mismatch')
    for record in trials:
        arrays=decode_arrays(record['payload']);rows=record['trace_rows']
        if set(arrays)!={'row_ids','input','baseline','repeat','control','candidate'}:
            raise ValueError('missing trace arrays')
        if arrays['row_ids'].tolist()!=rows:raise ValueError('wrong payload row IDs')
        for name in ('input','baseline','repeat','control','candidate'):
            if arrays[name].shape!=(len(rows),4096) or str(arrays[name].dtype)!='int16':
                raise ValueError('wrong BF16 failure payload shape/type')
    return finals[0]


def run(*,torch,case_factory,provenance):
    records=[]
    for rows,skew in CASES:
        x,call=case_factory(rows,skew)
        expected=x.clone()
        retained=[]
        for phase in PHASES:
            if phase=='changed':
                x.mul_(-.75);expected.mul_(-.75)
                trash=[torch.empty_like(x) for _ in range(3)];del trash
            # Keep this pair fixed for the repeated calls, as in the normal
            # sanitizer gate. Independent stock controls expose repeat noise.
            baseline=call(False);repeat=call(False)
            for trial in range(TRIALS):
                values={arm:call(arm=='candidate') for arm in order(trial)}
                pair=pair_summary(metrics(torch,values['control'],baseline,repeat),
                    metrics(torch,values['candidate'],baseline,repeat),rank=0)
                row_ids=[r['row'] for r in pair['failures']]
                if len(row_ids)>MAX_TRACE_ROWS:
                    raise RuntimeError('failure payload budget exceeded; refusing truncated evidence')
                index=torch.tensor(row_ids,device=x.device,dtype=torch.long)
                arrays={arm:value[index].view(torch.int16).cpu().numpy()
                    for arm,value in dict(input=x,baseline=baseline,repeat=repeat,**values).items()}
                arrays['row_ids']=index.cpu().numpy()
                payload=encode_arrays(arrays)
                record=dict(kind='MOE_M64_REUSE_DIAGNOSTIC_TRIAL',rows=rows,skew=skew,
                    seed=9211+rows,phase=phase,trial=trial,order=list(order(trial)),comparison=pair,
                    trace_rows=row_ids,payload=payload,input_unchanged=torch.equal(x,expected),
                    retained_unchanged=all(torch.equal(a,b) for a,b in retained))
                if not record['input_unchanged'] or not record['retained_unchanged']:
                    raise AssertionError('input or retained output changed')
                records.append(record)
                print(json.dumps(record,allow_nan=False),flush=True)
                if trial==0:retained.append((values['candidate'],values['candidate'].clone()))
        torch.cuda.synchronize()
    result=completion(records,provenance)
    print(json.dumps(result,allow_nan=False),flush=True)
    return result
