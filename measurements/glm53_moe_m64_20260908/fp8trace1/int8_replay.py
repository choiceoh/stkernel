"""CPU-only INT8 replay on previously selected failed rows, not acceptance."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from glm53_moe_m64_fp8_trace import ARMS, decode_arrays, verify_logs
from glm53_moe_m64_fp8_diagnostic import pair_summary, row_metrics


def quantize(partial):
    blocks=partial.reshape(-1,2,2048)
    scales=torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(dim=2).clamp_min(1e-30)/127)))
    quantized=torch.round(blocks/scales[:,:,None]).clamp(-127,127).to(torch.int8)
    return (quantized.float()*scales[:,:,None]).reshape(-1,4096)


def replay(directory):
    root=Path(directory);verify_logs(root)
    trials=[];payloads={}
    for rank in range(4):
        for line in (root/f'fp8-v3-rank-{rank}.log').read_text().splitlines():
            if not line.startswith('{'):continue
            r=json.loads(line);key=tuple(r.get(k) for k in ('rows','skew','seed','trial'))
            if r.get('kind')=='MOE_M64_FP8_TRACE_TRIAL':trials.append(r)
            if r.get('kind')=='MOE_M64_FP8_TRACE_ROWS':payloads[key+(rank,)]=r
    groups={};quality={}
    for trial in trials:
        key=tuple(trial[k] for k in ('rows','skew','seed','trial'));rows,skew,seed,iteration=key
        ids=trial['row_ids']
        if not ids:continue
        arrays=[decode_arrays(payloads[key+(rank,)]) for rank in range(4)]
        outputs={transport:{} for transport in ('fp8','int8')}
        for arm in ARMS:
            parts=[torch.from_numpy(a[arm+'_partial_bits']).view(torch.bfloat16).float() for a in arrays]
            exact=torch.zeros_like(parts[0]);int8=torch.zeros_like(exact);fp8=torch.empty_like(exact)
            for part in parts:exact+=part;int8+=quantize(part)
            int8=int8.to(torch.bfloat16).float()
            for a in arrays:
                positions=[ids.index(int(r)) for r in a['owned_row_ids']]
                fp8[positions]=torch.from_numpy(a[arm+'_output_bits']).view(torch.bfloat16).float()
            for transport,value in (('fp8',fp8),('int8',int8)):
                outputs[transport][arm]=value
                norm=exact.norm(dim=1).clamp_min(1e-6);peak=exact.abs().amax(dim=1).clamp_min(1e-6)
                errors=(value-exact)
                q=quality.setdefault((rows,skew,transport),dict(l2=[],peak=[]))
                q['l2'].extend((errors.norm(dim=1)/norm).tolist())
                q['peak'].extend((errors.abs().amax(dim=1)/peak).tolist())
        for transport,values in outputs.items():
            pair=pair_summary(row_metrics(torch,values['control'],values['baseline'],values['repeat']),
                row_metrics(torch,values['candidate'],values['baseline'],values['repeat']),rank=0)
            g=groups.setdefault((rows,skew,seed,transport),dict(rows=rows,skew=skew,seed=seed,transport=transport,
                selected_row_trials=0,selected_trials=0,candidate_bad=0,control_bad=0))
            g['selected_row_trials']+=len(ids);g['selected_trials']+=1
            g['candidate_bad']+=pair['candidate_bad'];g['control_bad']+=pair['control_bad']
    return dict(serving_gate=False,numerical_acceptance=False,gpu_execution=False,
        population='Only the union of rows previously failing FP8/partial/BF16 checks; selection-biased and incomplete.',
        recipe='per-2048 block power-of-two ceil(amax/127), round-to-nearest-even, symmetric [-127,127]; FP32 source-order sum then BF16',
        wire='Hypothetical same one-byte values plus existing FP32 block scales; no runtime or speed measurement.',
        groups=list(groups.values()),quality=[dict(rows=k[0],skew=k[1],transport=k[2],
            selected_arm_row_trials=len(v['l2']),median_l2=statistics.median(v['l2']),max_l2=max(v['l2']),
            median_peak=statistics.median(v['peak']),max_peak=max(v['peak'])) for k,v in quality.items()])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('output');args=ap.parse_args()
    result=replay(args.directory)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
