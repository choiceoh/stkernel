"""Jointly fit fixed-budget residual factors with balanced Korean/English data.

Sparse weights stay frozen. BF16 factors train through the complete expert,
including its nonlinear activation and FP4 straight-through rounding. A new
checkpoint must not worsen either language group's validation error relative
to the starting factors. No final holdout enters fitting or selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_sparse_block_reconstruct import FP4StraightThrough, teacher
from probes.engine_sparse_calibrate import split_rows
from probes.engine_sparse_nvfp4 import dequant
from probes.engine_sparse_nvfp4_prune import read_experts
from probes.engine_sparse_recovery import metrics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class JointResidual(torch.nn.Module):
    def __init__(self, sparse, factors):
        super().__init__()
        for name in ('w13','w2'):
            self.register_buffer(name, sparse[name].detach().clone())
            for side, value in zip(('B','A'), factors[name]):
                self.register_parameter(name+'_'+side, torch.nn.Parameter(value.float().clone()))

    def residual(self, x, name):
        a, b = (getattr(self,name+'_'+s).to(torch.bfloat16) for s in ('A','B'))
        return F.linear(F.linear(x.to(torch.bfloat16),a),b).float()

    def forward(self, hidden):
        q = FP4StraightThrough.apply
        up, gate = (q(hidden.float()) @ self.w13.T + self.residual(hidden,'w13')).chunk(2,-1)
        mid = swiglu_clamped(gate,up,10.)
        return q(mid.float()) @ self.w2.T + self.residual(mid,'w2')

    def export(self):
        return {n:tuple(getattr(self,n+'_'+s).detach().to(torch.bfloat16).contiguous()
                        for s in ('B','A')) for n in ('w13','w2')}


@torch.no_grad()
def evaluate(model, x, target, coefficient, groups):
    output = torch.cat([model(x[start:start+128]) for start in range(0,len(x),128)])
    return {name:metrics(output[idx]*coefficient[idx,None],target[idx]*coefficient[idx,None])
            for name,idx in groups.items()}


def eligible_score(current, initial):
    if any(current[k]['relative_l2'] > initial[k]['relative_l2'] for k in initial):
        return float('inf')
    return sum((current[k]['relative_l2']/max(initial[k]['relative_l2'],1e-12))**2 for k in initial)/len(initial)


def fit(model, x, target, coefficient, groups, steps, seed):
    initial = evaluate(model,x['validation'],target['validation'],coefficient['validation'],groups['validation'])
    best, best_step, best_score = {k:v.clone() for k,v in model.state_dict().items()},0,1.
    initial_parameters = {n:p.detach().clone() for n,p in model.named_parameters()}
    rates = [max(p.detach().square().mean().sqrt().item()*.01,1e-7) for p in model.parameters()]
    optimizer = torch.optim.Adam([dict(params=[p],lr=lr) for p,lr in zip(model.parameters(),rates)])
    generator = torch.Generator(device=x['train'].device).manual_seed(seed)
    denominators = {k:(target['train'][idx]*coefficient['train'][idx,None]).square().mean().clamp_min(1e-12)
                    for k,idx in groups['train'].items()}
    history = [dict(step=0,validation=initial,score=1.)]
    for step in range(1,steps+1):
        batches = [idx[torch.randint(len(idx),(64,),device=idx.device,generator=generator)]
                   for idx in groups['train'].values()]
        indices = torch.cat(batches)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x['train'][indices])
        delta = prediction-target['train'][indices]
        routed = delta*coefficient['train'][indices,None]
        loss = sum(routed[i*64:(i+1)*64].square().mean()/denominators[k]
                   for i,k in enumerate(groups['train']))/len(batches)
        row_scale = target['train'][indices].square().mean(-1,keepdim=True).clamp_min(1e-8)
        loss = loss + .05*(delta.square()/row_scale).mean()
        drift = sum((p-initial_parameters[n]).square().mean()/initial_parameters[n].square().mean().clamp_min(1e-12)
                    for n,p in model.named_parameters())
        loss = loss + 1e-4*drift
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite joint residual loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('missing or nonfinite factor gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        multiplier = min(step/10,1)*(.1+.9*(1+math.cos(math.pi*step/steps))/2)
        for group,rate in zip(optimizer.param_groups,rates): group['lr']=rate*multiplier
        optimizer.step()
        if step%40 == 0 or step==steps:
            validation = evaluate(model,x['validation'],target['validation'],coefficient['validation'],groups['validation'])
            score = eligible_score(validation,initial)
            history.append(dict(step=step,training_loss=loss.item(),validation=validation,
                                eligible=math.isfinite(score),score=score if math.isfinite(score) else None))
            if score < best_score:
                best,best_step,best_score = {k:v.detach().clone() for k,v in model.state_dict().items()},step,score
    model.load_state_dict(best)
    return dict(best_step=best_step,initial_validation=initial,history=history,
                steps=steps,batch_per_language=64,initial_learning_rates=rates,
                objective='equal language-group routed relative MSE + .05 row-normalized MSE + 1e-4 factor drift',
                selection='neither language group may worsen from initialization; minimize mean squared relative error ratios')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--recovery',type=Path,required=True)
    ap.add_argument('--residual',type=Path,required=True)
    ap.add_argument('--rank',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=400)
    args=ap.parse_args()
    if not 40<=args.steps<=1000: ap.error('steps must be 40..1000')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((4<<30)/torch.cuda.get_device_properties(0).total_memory)
    info=json.loads(args.capture.with_suffix('.json').read_text())
    recovery=json.loads(args.recovery.read_text());residual=json.loads(args.residual.read_text())
    if file_hash(args.capture)!=recovery['capture_sha256'] or residual['recovery_sha256']!=file_hash(args.recovery):
        raise ValueError('calibration artifacts differ')
    payload=torch.load(args.capture,weights_only=True,map_location='cpu')
    packed=torch.load(args.recovery.with_suffix('.weights.pt'),weights_only=True,map_location='cpu')
    factors=torch.load(args.residual.with_suffix('.weights.pt'),weights_only=True,map_location='cpu')
    fixture=Path(__file__).parent/'fixtures/sparse_calibration_prompts.json'
    korean_ids={p['id'] for p in json.loads(fixture.read_text()) if any('\uac00'<=c<='\ud7a3' for c in p['text'])}
    korean_prompts=torch.tensor([i for i,p in enumerate(info['prompts']) if p['id'] in korean_ids])
    is_korean=torch.isin(payload['prompt_index'],korean_prompts)
    report=dict(scope=__doc__,source_sha256=file_hash(__file__),capture_sha256=recovery['capture_sha256'],
                recovery_sha256=file_hash(args.recovery),previous_residual_sha256=file_hash(args.residual),
                language_fixture_sha256=file_hash(fixture),production_adopted=False,cases=[])
    exports={}
    for case in recovery['cases']:
        expert=case['expert'];indices,counts=split_rows(payload,info,expert,recovery['training_cap'])
        x={s:payload['x'][idx].cuda() for s,idx in indices.items() if s!='test'}
        coefficients={s:(payload['coefficient'][indices[s]]*(payload['selected'][indices[s]]==expert)).sum(-1).cuda() for s in x}
        groups={s:{'ko':is_korean[indices[s]].nonzero().flatten().cuda(),
                   'en':(~is_korean[indices[s]]).nonzero().flatten().cuda()} for s in x}
        if any(len(idx)<4 for group in groups.values() for idx in group.values()):
            raise ValueError('insufficient routed coverage for both language groups')
        sparse,original,initial={},{},{}
        for name in ('w13','w2'):
            raw,sf,_=read_experts(args.rank,3,name,[expert]);original[name]=dequant(raw,sf)[0]
            sparse[name]=dequant(*(packed[f'e{expert}.{name}.{suffix}'].cuda() for suffix in ('packed','sf')))
            initial[name]=tuple(factors[f'e{expert}.{name}.{side}'].cuda() for side in ('B','A'))
        targets={s:teacher(hidden,original) for s,hidden in x.items()}
        model=JointResidual(sparse,initial)
        print(json.dumps(dict(stage='start',expert=expert,groups={s:{k:len(v) for k,v in group.items()} for s,group in groups.items()})),flush=True)
        fitting=fit(model,x,targets,coefficients,groups,args.steps,seed=1901+expert)
        exported=model.export();restored=JointResidual(sparse,exported)
        with torch.no_grad():
            if not torch.equal(model(x['validation'][:8]),restored(x['validation'][:8])):
                raise AssertionError('joint residual export differs from training forward')
        for name, pair in exported.items():
            for side,tensor in zip(('B','A'),pair):
                if tensor.shape!=factors[f'e{expert}.{name}.{side}'].shape:raise AssertionError('factor shape changed')
                exports[f'e{expert}.{name}.{side}']=tensor.cpu()
        row=dict(expert=expert,counts=counts,fitting=fitting,export_forward_exact=True)
        report['cases'].append(row)
        print(json.dumps(dict(stage='done',expert=expert,best_step=fitting['best_step'],
                              validation=evaluate(model,x['validation'],targets['validation'],coefficients['validation'],groups['validation']))),flush=True)
    report['factor_bytes']=sum(t.numel()*t.element_size() for t in exports.values())
    assert report['factor_bytes']==sum(t.numel()*t.element_size() for t in factors.values())
    report['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated()
    torch.save(exports,args.out.with_suffix('.weights.pt'))
    report['export_sha256']=file_hash(args.out.with_suffix('.weights.pt'))
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
