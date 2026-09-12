"""Joint two-projection reconstruction with fixed pair masks and FP4 STE.

Fit the whole expert's output, including SwiGLU and K32 activation rounding.
Only retained weights are trained; exported weights use the existing sparse
kernel with no residual GEMMs. This is a small calibration-only QAT experiment,
not full-model training or evidence of language-model quality recovery.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from engine.profiles.glm53.lanes import swiglu_clamped
from probes.engine_sparse_calibrate import inputs16, inputs32, split_rows
from probes.engine_sparse_nvfp4 import Library, Projection, dequant, qualify, validate_sparse
from probes.engine_sparse_nvfp4_prune import quantize32, read_experts
from probes.engine_sparse_recovery import _encode_scaled, metrics


class FP4StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        packed, scales = quantize32(value)
        return dequant(packed, scales)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class Float8ScaleStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.to(torch.float8_e4m3fn).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class FP4ValueStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value.abs() <= 6)
        return _encode_scaled(value, torch.ones((), device=value.device))[0]

    @staticmethod
    def backward(ctx, gradient):
        (inside,) = ctx.saved_tensors
        return gradient * inside


def legal_mask(packed):
    validate_sparse(packed)
    # A retained pair can learn either scalar; absent pairs remain exactly zero.
    return (packed & 0x77).ne(0).repeat_interleave(2, -1)


class SparseExpert(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        for name in ('w13', 'w2'):
            packed, scale = weights[name]
            decoded = dequant(packed, scale)
            self.register_parameter(name, torch.nn.Parameter(decoded.clone()))
            self.register_buffer(f'{name}_mask', legal_mask(packed))
            self.register_buffer(f'{name}_initial', decoded.clone())
            # Preserve initial exported scales, including subnormal scales for
            # which recomputing max/6 is not an idempotent representation.
            log_scale = scale.view(torch.float8_e4m3fn).float().clamp_min(1/512).log()
            self.register_parameter(f'{name}_log_scale', torch.nn.Parameter(log_scale.clone()))
            self.register_buffer(f'{name}_initial_log_scale', log_scale.clone())

    def quantized_weight(self, name):
        weight = getattr(self, name) * getattr(self, name+'_mask')
        scale = Float8ScaleStraightThrough.apply(getattr(self, name+'_log_scale').exp().clamp(1/512, 448))
        blocks = weight.reshape(*scale.shape, 32)
        value = FP4ValueStraightThrough.apply(blocks * scale.reciprocal()[..., None])
        return (value * scale[..., None]).reshape_as(weight)

    def forward(self, hidden):
        q = FP4StraightThrough.apply
        w13 = self.quantized_weight('w13')
        w2 = self.quantized_weight('w2')
        first = q(hidden.float()) @ w13.T
        up, gate = first.chunk(2, -1)
        middle = swiglu_clamped(gate, up, 10.)
        return q(middle.float()) @ w2.T

    def regularization(self):
        return sum(((getattr(self, n)-getattr(self, n+'_initial')) * getattr(self, n+'_mask')).square().mean()
                   / getattr(self, n+'_initial').square().mean().clamp_min(1e-12)
                   + .25*(getattr(self, n+'_log_scale')-getattr(self, n+'_initial_log_scale')).square().mean()
                   for n in ('w13', 'w2'))

    @torch.no_grad()
    def export(self):
        result = {}
        for name in ('w13', 'w2'):
            weight = getattr(self, name) * getattr(self, name+'_mask')
            scale = getattr(self, name+'_log_scale').exp().clamp(1/512, 448).to(torch.float8_e4m3fn)
            _, codes = _encode_scaled(weight.reshape(*scale.shape, 32), scale.float()[..., None])
            codes = codes.flatten(-2)
            packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
            sf = scale.view(torch.uint8).contiguous()
            validate_sparse(packed)
            result[name] = (packed, sf)
        return result


@torch.no_grad()
def teacher(hidden, weights):
    first = inputs16(hidden) @ weights['w13'].T
    up, gate = first.chunk(2, -1)
    return inputs16(swiglu_clamped(gate, up, 10.)) @ weights['w2'].T


@torch.no_grad()
def evaluate(model, inputs, target):
    return metrics(torch.cat([model(inputs[i:i+128]) for i in range(0, len(inputs), 128)]), target)


def fit(model, inputs, targets, *, steps, batch, seed, lr_relative=.01, regularization=.01):
    if steps < 1 or batch < 1:
        raise ValueError('positive steps and batch required')
    parameters = [getattr(model, n) for n in ('w13', 'w2', 'w13_log_scale', 'w2_log_scale')]
    rates = [max(p.detach().square().mean().sqrt().item() * lr_relative, 1e-8) for p in parameters[:2]] + [.005, .005]
    optimizer = torch.optim.Adam([dict(params=[p], lr=lr) for p, lr in zip(parameters, rates)])
    generator = torch.Generator(device=inputs['train'].device).manual_seed(seed)
    floor = targets['train'].square().mean().sqrt() * .1
    row_scale = targets['train'].square().mean(-1, keepdim=True).sqrt().clamp_min(floor)
    best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial = evaluate(model, inputs['validation'], targets['validation'])
    best_metric, best_step = initial['relative_l2'], 0
    history = [dict(step=0, validation=initial)]
    started = time.monotonic()
    for step in range(1, steps+1):
        indices = torch.randint(len(inputs['train']), (min(batch, len(inputs['train'])),),
                                device=inputs['train'].device, generator=generator)
        factor = min(step/10, 1) * (.1+.9*(1+math.cos(math.pi*step/steps))/2)
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate * factor
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs['train'][indices])
        reconstruction = ((prediction-targets['train'][indices])/row_scale[indices]).square().mean()
        loss = reconstruction + regularization * model.regularization()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite reconstruction loss')
        loss.backward()
        for name in ('w13', 'w2'):
            grad, mask = getattr(model, name).grad, getattr(model, name+'_mask')
            if not torch.isfinite(grad).all() or grad[~mask].count_nonzero():
                raise RuntimeError('nonfinite gradient or pruned pair received a gradient')
            if not torch.isfinite(getattr(model, name+'_log_scale').grad).all():
                raise RuntimeError('nonfinite scale gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 10.)
        optimizer.step()
        if step % 20 == 0 or step == steps:
            validation = evaluate(model, inputs['validation'], targets['validation'])
            row = dict(step=step, training_batch_loss=reconstruction.item(),
                       validation=validation, seconds=time.monotonic()-started)
            history.append(row)
            if validation['relative_l2'] < best_metric:
                best_metric, best_step = validation['relative_l2'], step
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(json.dumps(dict(stage='block_fit', **row)), flush=True)
    model.load_state_dict(best)
    return dict(initial_validation=initial, best_step=best_step, history=history,
                lr_relative=lr_relative, actual_initial_learning_rates=rates,
                regularization=regularization, steps=steps, batch=batch,
                objective='per-row RMS-normalized expert output MSE + masked weight/scale drift penalty',
                weight_scales='learned log scales; E4M3 rounding in every forward',
                selection='lowest validation whole-expert relative L2; initial state is a candidate')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--rank', type=Path, required=True)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--experts', type=int, default=4)
    args = ap.parse_args()
    if not 20 <= args.steps <= 1000 or not 1 <= args.batch <= 128 or not 1 <= args.experts <= 4:
        ap.error('bounded reconstruction: steps 20..1000, batch 1..128, experts 1..4')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((3 << 30)/torch.cuda.get_device_properties(0).total_memory)
    info = json.loads(args.capture.with_suffix('.json').read_text())
    recovery = json.loads(args.recovery.read_text())
    if hashlib.sha256(args.capture.read_bytes()).hexdigest() != recovery['capture_sha256']:
        raise ValueError('capture identity mismatch')
    payload = torch.load(args.capture, weights_only=True, map_location='cpu')
    weights = torch.load(args.recovery.with_suffix('.weights.pt'), weights_only=True, map_location='cpu')
    library = Library(args.library)
    report = dict(scope=__doc__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  recovery_sha256=hashlib.sha256(args.recovery.read_bytes()).hexdigest(),
                  capture_sha256=recovery['capture_sha256'], production_adopted=False, cases=[])
    exports = {}
    for case in recovery['cases'][:args.experts]:
        if 'skipped' in case:
            continue
        expert = case['expert']
        indices, counts = split_rows(payload, info, expert, recovery['training_cap'])
        inputs = {s: payload['x'][idx].cuda() for s, idx in indices.items()}
        original, initial = {}, {}
        for name in ('w13', 'w2'):
            raw, sf, _ = read_experts(args.rank, 3, name, [expert])
            original[name] = dequant(raw, sf)[0]
            initial[name] = tuple(weights[f'e{expert}.{name}.{suffix}'].cuda() for suffix in ('packed', 'sf'))
        targets = {s: teacher(x, original) for s, x in inputs.items()}
        model = SparseExpert(initial)
        print(json.dumps(dict(stage='expert_start', expert=expert, counts=counts)), flush=True)
        fitting = fit(model, inputs, targets, steps=args.steps, batch=args.batch, seed=713+expert)
        row = dict(expert=expert, counts=counts, fitting=fitting,
                   expert_chain={s:evaluate(model, inputs[s], targets[s]) for s in inputs})
        exported = model.export()
        restored = SparseExpert(exported)
        with torch.no_grad():
            if not torch.equal(model(inputs['test'][:8]), restored(inputs['test'][:8])):
                raise AssertionError('exported sparse block does not reproduce training forward')
        row['export_forward_exact'] = True
        row['kernel_correctness'] = {}
        for name, (pk, sc) in exported.items():
            exports[f'e{expert}.{name}.packed'] = pk.cpu()
            exports[f'e{expert}.{name}.sf'] = sc.cpu()
            with torch.no_grad():
                qinputs = inputs['test'][:6] if name == 'w13' else swiglu_clamped(
                    (inputs32(inputs['test'][:6]) @ dequant(*exported['w13']).T).chunk(2,-1)[1],
                    (inputs32(inputs['test'][:6]) @ dequant(*exported['w13']).T).chunk(2,-1)[0], 10.)
                xp, xs = quantize32(qinputs)
                projection = Projection(library, pk[None], xp[None], sc[None], xs[None])
                try:
                    row['kernel_correctness'][name] = qualify(projection)
                finally:
                    projection.close()
        report['cases'].append(row)
        report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        torch.save(exports, args.out.with_suffix('.weights.pt'))
        print(json.dumps(dict(stage='expert_done', expert=expert, best_step=fitting['best_step'],
                              test=row['expert_chain']['test'])), flush=True)


if __name__ == '__main__':
    main()
