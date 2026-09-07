"""Verify intact complete/partial local diagnostic evidence without acceptance."""
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/probes'))
import glm53_moe_m64_reuse_diagnostic as d
from glm53_sanitizer_report import validate_canaries


def read(path):
    return gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()).decode() if not path.exists() else path.read_text()


def analyze():
    source=json.loads((ROOT/'source-manifest.json').read_text())
    result=dict(source_revision=source['revision'],serving_gate=False,numerical_acceptance=False,modes=[])
    result['detectors']=validate_canaries(read(ROOT/'raw/detectors.log'),ROOT/'source/probes')
    for mode in ('memcheck',):
        text=read(ROOT/f'raw/{mode}.log')
        records=[json.loads(l) for l in text.splitlines() if l.startswith('{')]
        trials=[r for r in records if r.get('kind')=='MOE_M64_REUSE_DIAGNOSTIC_TRIAL']
        finals=[r for r in records if r.get('verdict')==d.MARKER]
        assert [(r['rows'],r['skew'],r['phase'],r['trial']) for r in trials]==d.plan()[:len(trials)]
        groups={}
        for record in trials:
            pair=record['comparison'];rows=record['trace_rows']
            assert pair['rows']==record['rows'] and tuple(record['order'])==d.order(record['trial'])
            assert record['input_unchanged'] and record['retained_unchanged']
            assert rows==[r['row'] for r in pair['failures']]
            arrays=d.decode_arrays(record['payload'])
            assert arrays['row_ids'].tolist()==rows and len(rows)<=d.MAX_TRACE_ROWS
            for arm in ('input','baseline','repeat','control','candidate'):
                assert arrays[arm].shape==(len(rows),4096) and arrays[arm].dtype==np.int16
            for arm in ('control','candidate'):
                assert pair[arm+'_bad']==sum(r[arm+'_bad'] for r in pair['failures'])
            # Independently check raw failed-row norms from BF16 bits. This
            # verifies logged math; original pass/fail flags remain unchanged.
            fp={k:(v.astype(np.uint16).astype(np.uint32)<<16).view(np.float32)
                for k,v in arrays.items() if k!='row_ids'}
            for i,failure in enumerate(pair['failures']):
                b=fp['baseline'][i];repeat=fp['repeat'][i]
                for arm in ('control','candidate'):
                    a=fp[arm][i];v=failure[arm]
                    expected=dict(raw_error_l2=float(np.linalg.norm(a-b)),raw_error_peak=float(np.max(np.abs(a-b))),
                        raw_noise_l2=float(np.linalg.norm(repeat-b)),raw_noise_peak=float(np.max(np.abs(repeat-b))),
                        reference_norm=float(max(np.linalg.norm(b),1e-6)),reference_peak=float(max(np.max(np.abs(b)),1e-6)))
                    assert all(math.isclose(v[k],value,rel_tol=1e-5,abs_tol=1e-6) for k,value in expected.items())
            key=(record['rows'],record['skew'],record['phase'])
            group=groups.setdefault(key,dict(rows=key[0],skew=key[1],phase=key[2],trials=0,control_bad=0,candidate_bad=0))
            group['trials']+=1
            for arm in ('control','candidate'):group[arm+'_bad']+=pair[arm+'_bad']
        complete=len(trials)==len(d.plan())
        if complete:assert finals==[d.completion(trials,source['provenance'])]
        else:assert not finals
        result['modes'].append(dict(mode=mode,trials=len(trials),complete=complete,
            row_trials_per_arm=sum(r['rows'] for r in trials),groups=list(groups.values()),
            candidate_bad=sum(r['comparison']['candidate_bad'] for r in trials),
            control_bad=sum(r['comparison']['control_bad'] for r in trials),
            failure_payloads_verified=True,reported_sanitizer_errors=0 if 'ERROR SUMMARY: 0 errors' in text else None))
    before=json.loads((ROOT/'job/evidence/before.json').read_text())
    after=json.loads((ROOT/'job/evidence/restored.json').read_text())
    identity=lambda s:{k:v for k,v in s.items() if k not in ('running','started')}
    assert set(before)==set(after) and all(identity(before[n])==identity(after[n]) and after[n]['running'] for n in before)
    completion=json.loads((ROOT/'job/evidence/completion.json').read_text())
    assert completion['restored_original'] is True and completion['exit_code']==1
    result['exact_incoming_recovery']=True
    result['outer']=json.loads((ROOT/'job/completion.json').read_text())
    process=json.loads((ROOT/'raw/memcheck-process.json').read_text())
    assert process['exit_code']==15 and process['final_container']['state']['ExitCode']==15
    assert process['final_container']['state']['OOMKilled'] is False
    assert process['observed_memory_peak_bytes']==2902802432 and not process['sampler_errors']
    assert process['last_resource']['memory']['memory.max']=='17179869184'
    events=dict(l.split() for l in process['last_resource']['memory']['memory.events'].splitlines())
    assert events['max']==events['oom']==events['oom_kill']=='0'
    result['process']={k:process[k] for k in ('exit_code','samples','observed_memory_peak_bytes','min_host_mem_available_bytes','last_resource','final_container','sampler_errors')}
    memory=[r for r in records if r.get('kind')=='MOE_M64_REUSE_MEMORY']
    expected=[(rows,skew,stage) for rows,skew in d.CASES for stage in ('before_case','case_complete','case_released')]
    assert [(r['rows'],r['skew'],r['stage']) for r in memory]==expected[:16]
    assert all(0<=r['allocated_bytes']<=r['reserved_bytes'] for r in memory)
    assert memory[-1]['allocated_bytes']==97876293632
    assert memory[-1]['cuda_free_bytes']==14269620224
    result['case_memory']=memory
    result['driver_allocation_observations']=json.loads((ROOT/'job/kernel-allocation-observations.json').read_text())
    assert result['driver_allocation_observations']['exit_code']==0
    result['memcheck_termination_cause']='Unresolved. Live CUDA allocations accumulated despite completed-fixture release; no container cgroup OOM. The retained kernel-journal query contains only the two earlier allocation failures, no new failure for this run.'
    result['racecheck']='not reached'
    return result


if __name__=='__main__':
    result=analyze()
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verified=True,modes=[{k:v[k] for k in ('mode','trials','complete','row_trials_per_arm','candidate_bad','control_bad')} for v in result['modes']],
                         exact_incoming_recovery=result['exact_incoming_recovery'],outer=result['outer']),indent=2))
