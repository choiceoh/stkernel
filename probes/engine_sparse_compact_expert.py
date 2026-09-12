"""Prune whole hidden-channel groups and refit a smaller dense NVFP4 expert.

Retain 16 of 32 K16 groups (512 -> 256 intermediate channels). Selected first
projection rows and scales are copied exactly, preserving their nonlinear
features. Refit/requantize only the down projection. This is an alternative
compression architecture, not a use of the sparse MMA instruction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from engine.modules.nvfp4_sf import swizzle_sf
from engine.profiles.glm53.lanes import served, swiglu_clamped
from probes.engine_sparse_calibrate import inputs16, split_rows
from probes.engine_sparse_nvfp4 import dequant
from probes.engine_sparse_nvfp4_prune import read_experts
from probes.engine_sparse_recovery import _encode_scaled, hessian, metrics
from probes.engine_sparse_sequential_reconstruct import anchored_target


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantize16_weight(weight):
    """Choose max/4 or max/6 E4M3 scale by exact block reconstruction SSE."""
    blocks = weight.reshape(weight.shape[0], -1, 16)
    candidates = []
    for divisor in (6., 4.):
        sf = (blocks.abs().amax(-1)/divisor).clamp(max=448).to(torch.float8_e4m3fn)
        value, codes = _encode_scaled(blocks, sf.float()[..., None])
        candidates.append((value, codes, sf))
    choose = (candidates[1][0]-blocks).square().sum(-1) < (candidates[0][0]-blocks).square().sum(-1)
    codes = torch.where(choose[..., None], candidates[1][1], candidates[0][1]).flatten(-2)
    sf = torch.where(choose, candidates[1][2].float(), candidates[0][2].float()).to(torch.float8_e4m3fn)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, sf.view(torch.uint8).contiguous()


def select_groups(weight, moment, mode):
    """Group saliency from output energy or conditional OBS deletion cost."""
    if weight.shape[1] != 512 or moment.shape != (512, 512):
        raise ValueError('this bounded pilot requires 512 intermediate channels')
    if mode == 'conditional':
        reg = moment.clone()
        reg.diagonal().add_(.1 * reg.diagonal().mean().clamp_min(1e-12))
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(reg))
    elif mode != 'energy':
        raise ValueError('unknown group saliency')
    scores = []
    for start in range(0, 512, 16):
        sl = slice(start, start+16)
        metric = torch.linalg.inv(inverse[sl, sl]) if mode == 'conditional' else moment[sl, sl]
        gram = weight[:, sl].T @ weight[:, sl]
        scores.append((gram*metric.T).sum())
    groups = torch.stack(scores).argsort(descending=True, stable=True)[:16].sort().values
    return (groups[:, None]*16 + torch.arange(16, device=weight.device)).flatten()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--final-capture', type=Path, required=True)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((4 << 30)/torch.cuda.get_device_properties(0).total_memory)
    info = json.loads(args.capture.with_suffix('.json').read_text())
    recovery = json.loads(args.recovery.read_text())
    if file_hash(args.capture) != recovery['capture_sha256']:
        raise ValueError('calibration capture identity mismatch')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    report = dict(scope=__doc__, source_sha256=file_hash(__file__), capture_sha256=recovery['capture_sha256'],
                  production_adopted=False, intermediate_before=512, intermediate_after=256,
                  sparse_mma=False, selection='minimum validation whole-expert relative L2',
                  final_outputs_unread_until_all_experts_selected=True, cases=[])
    exports = {}
    lane = served()
    for case in recovery['cases']:
        expert = case['expert']
        indices, counts = split_rows(payload, info, expert, recovery['training_cap'])
        hidden = {s: payload['x'][idx].cuda() for s, idx in indices.items() if s != 'test'}
        p13, s13, _ = read_experts(args.rank, 3, 'w13', [expert])
        p2, s2, _ = read_experts(args.rank, 3, 'w2', [expert])
        p13, s13, p2, s2 = p13[0], s13[0], p2[0], s2[0]
        w13, w2 = dequant(p13, s13), dequant(p2, s2)
        mid, target = {}, {}
        for split, x in hidden.items():
            up, gate = (inputs16(x) @ w13.T).chunk(2, -1)
            mid[split] = inputs16(swiglu_clamped(gate, up, 10.))
            target[split] = mid[split] @ w2.T
        moment = hessian(mid['train'])
        candidates, scores = {}, {}
        for mode in ('energy', 'conditional'):
            chosen = select_groups(w2, moment, mode)
            compact = {s: x[:, chosen].contiguous() for s, x in mid.items()}
            original_subset = w2[:, chosen]
            for damping in (None, .01, .1, 1.):
                key = f'{mode}_{damping}'
                if damping is None:
                    packed = p2.reshape(4096, 32, 8)[:, chosen[::16]//16].reshape(4096, 128).contiguous()
                    sf = s2[:, chosen[::16]//16].contiguous()
                else:
                    weight = anchored_target(original_subset, compact['train'], target['train'], damping)
                    packed, sf = quantize16_weight(weight)
                decoded = dequant(packed, sf)
                candidates[key] = (chosen, packed, sf)
                scores[key] = {s: metrics(compact[s] @ decoded.T, target[s]) for s in ('train','validation')}
        key = min(scores, key=lambda k: (scores[k]['validation']['relative_l2'], k))
        chosen, packed, sf = candidates[key]
        rows = torch.cat((chosen, chosen+512))
        compact13, compact_s13 = p13[rows].contiguous(), s13[rows].contiguous()
        # Check exact first-projection row preservation before exercising b12x.
        assert torch.equal(dequant(compact13, compact_s13), w13[rows])
        sample = hidden['validation'][:8]
        up, gate = (inputs16(sample) @ dequant(compact13, compact_s13).T).chunk(2, -1)
        oracle = inputs16(swiglu_clamped(gate, up, 10.)) @ dequant(packed, sf).T
        actual = lane.moe(sample, torch.zeros(len(sample),1,device='cuda',dtype=torch.int32),
                          torch.ones(len(sample),1,device='cuda'), compact13[None],
                          swizzle_sf(compact_s13)[None], packed[None], swizzle_sf(sf)[None], 10.)
        native = metrics(actual, oracle)
        if not native['finite'] or native['relative_l2'] > .02:
            raise AssertionError(('compact b12x disagrees with reciprocal oracle', native))
        for name, value in [('w13.packed', compact13), ('w13.sf',compact_s13), ('w2.packed',packed), ('w2.sf',sf)]:
            exports[f'e{expert}.{name}'] = value.cpu()
        row = dict(expert=expert, counts=counts, selected=key, channels=chosen.cpu().tolist(),
                   candidates=scores, first_projection_preserved_exactly=True, native_correctness=native)
        report['cases'].append(row)
        print(json.dumps(dict(stage='selected',expert=expert,choice=key,
                              validation=scores[key]['validation'],native=native)),flush=True)
    torch.save(exports, args.out.with_suffix('.weights.pt'))
    report['export_sha256'] = file_hash(args.out.with_suffix('.weights.pt'))
    report['weight_and_scale_bytes'] = sum(t.numel()*t.element_size() for t in exports.values())
    # Only now read fresh holdout inputs. No fitting/selection below this point.
    final_info = json.loads(args.final_capture.with_suffix('.json').read_text())
    if file_hash(args.final_capture) != final_info['tensor_file_sha256']:
        raise ValueError('final capture identity mismatch')
    if {p['id'] for p in info['prompts']} & {p['id'] for p in final_info['prompts']}:
        raise ValueError('overlapping final prompt IDs')
    if {tuple(p['token_ids']) for p in info['prompts']} & {tuple(p['token_ids']) for p in final_info['prompts']}:
        raise ValueError('overlapping final token sequences')
    final = torch.load(args.final_capture, weights_only=True, map_location='cpu')
    totals = {n:torch.zeros(len(final['x']),4096,device='cuda') for n in ('original','compact')}
    for row in report['cases']:
        expert = row['expert']
        selected = final['selected'] == expert
        index = selected.any(-1).nonzero().flatten()
        x = final['x'][index].cuda()
        coefficient = (final['coefficient'][index]*selected[index]).sum(-1).cuda()
        original, compact = {}, {}
        for name in ('w13','w2'):
            raw, sf, _ = read_experts(args.rank,3,name,[expert])
            original[name] = dequant(raw,sf)[0]
            compact[name] = dequant(*(exports[f'e{expert}.{name}.{s}'].cuda() for s in ('packed','sf')))
        outputs = {}
        for variant, weights in [('original',original),('compact',compact)]:
            up, gate = (inputs16(x) @ weights['w13'].T).chunk(2,-1)
            outputs[variant] = inputs16(swiglu_clamped(gate,up,10.)) @ weights['w2'].T
            totals[variant].index_add_(0,index.cuda(),outputs[variant]*coefficient[:,None])
        row['final'] = metrics(outputs['compact'],outputs['original'])
    active = torch.isin(final['selected'],torch.tensor(recovery['experts'])).any(-1).cuda()
    report['final_capture_sha256'] = file_hash(args.final_capture)
    report['final_weighted_selected_expert_sum'] = metrics(totals['compact'][active],totals['original'][active])
    report['per_prompt'] = []
    for i,prompt in enumerate(final_info['prompts']):
        mask = (final['prompt_index']==i).cuda() & active
        if mask.any():
            report['per_prompt'].append(dict(id=prompt['id'],metrics=metrics(totals['compact'][mask],totals['original'][mask])))
    report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='final',weighted=report['final_weighted_selected_expert_sum'])),flush=True)


if __name__ == '__main__':
    main()
