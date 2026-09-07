"""Reconstruct every traced packet/sum on CPU; never changes acceptance."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from glm53_moe_m64_fp8_trace import ARMS, PHASES, decode_arrays, verify_logs


def analyze(directory):
    root=Path(directory)
    completed=verify_logs(root)
    trials=[]; payloads={}
    for rank in range(4):
        for line in (root/f'fp8-v3-rank-{rank}.log').read_text().splitlines():
            if not line.startswith('{'):continue
            record=json.loads(line)
            key=tuple(record.get(k) for k in ('rows','skew','seed','trial'))
            if record.get('kind')=='MOE_M64_FP8_TRACE_TRIAL':trials.append(record)
            if record.get('kind')=='MOE_M64_FP8_TRACE_ROWS':payloads[key+(rank,)]=record
    bit_equal=lambda a,b:torch.equal(a.view(torch.int32),b.view(torch.int32))
    groups={}; failures=[]; verified=collections.Counter()
    for trial in trials:
        key=tuple(trial[k] for k in ('rows','skew','seed','trial'))
        rows,skew,seed,iteration=key
        arrays=[decode_arrays(payloads[key+(rank,)]) for rank in range(4)]
        ids=trial['row_ids']; n=len(ids)
        for rank,values in enumerate(arrays):
            np.testing.assert_array_equal(values['row_ids'],ids)
            owned=[r for r in ids if rank*(rows//4)<=r<(rank+1)*(rows//4)]
            np.testing.assert_array_equal(values['owned_row_ids'],owned)
            expected={'row_ids','owned_row_ids'} | {a+s for a in ARMS for s in
                ('_partial_bits','_fp8_bytes','_scales','_sum32','_output_bits','_bf16_bits')}
            assert set(values)==expected
            for arm in ARMS:
                for suffix,shape,dtype in (('_partial_bits',(n,4096),np.int16),
                        ('_fp8_bytes',(n,4096),np.uint8),('_scales',(n,2),np.float32),
                        ('_sum32',(len(owned),4096),np.float32),
                        ('_output_bits',(len(owned),4096),np.int16),
                        ('_bf16_bits',(len(owned),4096),np.int16)):
                    assert values[arm+suffix].shape==shape and values[arm+suffix].dtype==dtype
        parts={}; qs={}; scales={}; decoded={}; sums={}; exact_sums={}; outputs={}
        for arm in ARMS:
            parts[arm]=[torch.from_numpy(a[arm+'_partial_bits']).view(torch.bfloat16).float() for a in arrays]
            qs[arm]=[torch.from_numpy(a[arm+'_fp8_bytes']) for a in arrays]
            scales[arm]=[torch.from_numpy(a[arm+'_scales']) for a in arrays]
            decoded[arm]=[q.view(torch.float8_e4m3fn).float()*s.repeat_interleave(2048,dim=1)
                          for q,s in zip(qs[arm],scales[arm])]
            sums[arm]=torch.zeros((n,4096),dtype=torch.float32)
            exact_sums[arm]=torch.zeros_like(sums[arm])
            outputs[arm]=torch.empty_like(sums[arm])
            for rank in range(4):
                sums[arm]+=decoded[arm][rank];exact_sums[arm]+=parts[arm][rank]
                part=parts[arm][rank].reshape(n,2,2048)
                wanted_scales=torch.exp2(torch.ceil(torch.log2(part.abs().amax(dim=2).clamp_min(1e-30)/448)))
                wanted_q=(part/wanted_scales[:,:,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n,4096)
                if not bit_equal(wanted_scales,scales[arm][rank]) or not torch.equal(wanted_q,qs[arm][rank]):
                    raise ValueError(f'CPU pack reconstruction differs: {key} rank={rank} arm={arm}')
                verified['rank_arm_packet_sets']+=1
            for rank,values in enumerate(arrays):
                positions=[ids.index(int(r)) for r in values['owned_row_ids']]
                out=torch.from_numpy(values[arm+'_output_bits']).view(torch.bfloat16).float()
                outputs[arm][positions]=out
                gpu_sum=torch.from_numpy(values[arm+'_sum32'])
                if not bit_equal(sums[arm][positions],gpu_sum):
                    raise ValueError(f'CPU source-order sum differs: {key} rank={rank} arm={arm}')
                verified['destination_arm_sum_sets']+=1
        for phase in PHASES:
            group_key=(rows,skew,seed,phase)
            group=groups.setdefault(group_key,dict(rows=rows,skew=skew,seed=seed,phase=phase,
                candidate_bad=0,control_bad=0,candidate_failed_trials=0,control_failed_trials=0,
                candidate_peak_raw_exceeds=0,control_peak_raw_exceeds=0))
            for arm in ('candidate','control'):
                count=sum(p[arm+'_bad'] for p in trial[phase])
                group[arm+'_bad']+=count;group[arm+'_failed_trials']+=count>0
                for pair in trial[phase]:
                    for failure in pair['failures']:
                        if not failure[arm+'_bad']:continue
                        value=failure[arm]
                        raw_exceeds=value['raw_error_peak']>max(.04*value['reference_peak'],3*value['raw_noise_peak'])
                        group[arm+'_peak_raw_exceeds']+=raw_exceeds
                        if phase!='transport':continue
                        index=ids.index(failure['row'])
                        component=int((outputs[arm][index]-outputs['baseline'][index]).abs().argmax())
                        item=dict(rows=rows,skew=skew,seed=seed,trial=iteration,arm=arm,
                            row=failure['row'],rank=pair['rank'],component=component,metrics=value,
                            raw_peak_exceeds=raw_exceeds,
                            changed_scale_blocks=sum(int((s[index]!=b[index]).sum())
                                for s,b in zip(scales[arm],scales['baseline'])),
                            rank_partial_at_component=[float(p[index,component]) for p in parts[arm]],
                            baseline_partial_at_component=[float(p[index,component]) for p in parts['baseline']],
                            rank_fp8_byte_at_component=[int(q[index,component]) for q in qs[arm]],
                            baseline_fp8_byte_at_component=[int(q[index,component]) for q in qs['baseline']],
                            rank_scales_at_component=[float(s[index,component//2048]) for s in scales[arm]],
                            baseline_scales_at_component=[float(s[index,component//2048]) for s in scales['baseline']],
                            exact_partial_sum_delta=float(exact_sums[arm][index,component]-exact_sums['baseline'][index,component]),
                            decoded_fp8_sum_delta=float(sums[arm][index,component]-sums['baseline'][index,component]),
                            stored_bf16_delta=float(outputs[arm][index,component]-outputs['baseline'][index,component]))
                        failures.append(item)
    return dict(diagnostic=completed,verified=dict(verified),groups=list(groups.values()),failures=failures,
        note='Raw-numerator comparisons diagnose normalized boundaries only; original failures are retained. No tolerance or acceptance change.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('output');args=ap.parse_args()
    result=analyze(args.directory)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=result['verified'],replay_all_equal=result['diagnostic']['replay_all_equal'],
        groups=result['groups'],failures=len(result['failures']))))
