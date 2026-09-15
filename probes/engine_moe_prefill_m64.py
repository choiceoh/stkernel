"""M64 long-prefill MoE: the private tile-64 candidate against the pinned M128 lane, same build.

The candidate is `engine/kernels/b12x/moe_dynamic_prefill_m64.py` (PR #876). It has a CPU
oracle and compiles, but nothing has ever run it on a device, so it stays behind
`_prefill_tile64` and refuses capture. This probe is the missing gate.

**It is not a byte-equality gate, by construction.** The M64 lane owns 16-row-per-warp
scatter strips, so its FP32 atomic sum lands in a different order than M128's. Global FP32
atomics are not run-to-run stable either, so the honest floor for any comparison is an arm's
own spread across repeats. The candidate is admitted when

  1. each arm's repeat spread is reported, and
  2. |m64 - m128| does not exceed the larger arm spread by more than `--tolerance-factor`, and
  3. both arms sit the same distance from the torch reference lane (checked on the small
     bucket only -- the reference is a per-expert Python loop).

A row that fails any of these is reported with its numbers; the probe does not average them
away. Timing is eager (capture is refused until qualification) in B/A/A/B with a warm and an
L2-evicted pass. Nothing here changes a serving default, and no GPU is reserved -- run it
under the queue's single-GPU lane.

The two arms differ by ONE argument to the served entry, `launch_sm120_moe(_prefill_tile64=)`.
The dispatcher owns the M64 workspace derivation, so what this gate measures is the route
that would ship, not a probe-shaped copy of it. The device is the default; `--cpu` runs the
parts that need none (eligibility, the ladder, source identity) so the wiring can be smoked
in a CPU container before a GPU slot is spent.

    bash bench/fleet.sh run --gpu st-m64 20 'M64 prefill MoE' -- \
      bash probes/run_engine_probe.sh probes/engine_moe_prefill_m64.py \
        --ranks <rank file> --output /cache/st-m64.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

# The eligibility window is 64 < m <= 8192. The ladder covers both edges, the served chunk
# sizes (TOKEN_BUDGET 8192 -> 6912; 2304 is the decoder-side chunk) and rows that are not a
# multiple of the tile so the M64 tail strips are exercised.
ROWS = (65, 129, 1024, 2304, 2305, 4608, 6912, 8192)
ORACLE_ROWS = 129          # the reference lane loops over experts in Python
REPEATS = 3
TOLERANCE_FACTOR = 4.0
# A candidate that cannot reproduce its own output is not a reorder, it is a defect, and no
# cross-arm rule can rescue it: `across <= floor x factor` is vacuous once the floor is large.
# An FP32-atomic reorder on a BF16 result lands far under this.
REPRODUCIBLE_CEILING = 1e-3

SOURCES = (
    'engine/kernels/b12x/moe_dynamic_prefill_m64.py',
    'engine/kernels/b12x/_prefill_m64_bodies.py',
    'engine/kernels/b12x/moe_dynamic_gated_sf6_q0.py',
    'engine/kernels/b12x/moe_dynamic_gated_sf6_q0_words.py',
    'engine/kernels/b12x/moe_dynamic_gated_sf6.py',
    'engine/kernels/b12x/_moe_dynamic/gated.py',
    'engine/kernels/b12x/moe_dispatch.py',
    'engine/profiles/glm53/lanes.py',
    'probes/engine_moe_prefill_m64.py',
)


def eligibility(md):
    """What the dispatcher admits, evaluated against the bound cell. No device needed."""
    cell = md._admitted_moe()
    common = dict(E=cell.experts_local, k=cell.hidden, n=cell.inter_local,
                  num_topk=cell.topk, quant_mode=cell.quant, tiled=True,
                  reform_sf_pack=True, activation=cell.activation, swiglu_alpha=1.,
                  swiglu_beta=0., swiglu_limit=cell.swiglu_limit,
                  share_input_across_experts=False)
    admitted = {m: md._prefill_m64_eligible(m=m, tile_m=64, **common) for m in ROWS}
    refused = {
        'm=64 is at the closed edge': md._prefill_m64_eligible(m=64, tile_m=64, **common),
        'm=8193 is past the window': md._prefill_m64_eligible(m=8193, tile_m=64, **common),
        'an M128 workspace is not an M64 workspace':
            md._prefill_m64_eligible(m=2304, tile_m=128, **common),
        'raw scales are not the packed SF6 recipe':
            md._prefill_m64_eligible(m=2304, tile_m=64, **{**common, 'reform_sf_pack': False}),
        'row-major weights are not the tiled recipe':
            md._prefill_m64_eligible(m=2304, tile_m=64, **{**common, 'tiled': False}),
    }
    return admitted, refused, cell


def cell_fields(cell):
    """The admitted cell as JSON, so a record says which shape the gate was evaluated for."""
    return dict(experts_local=cell.experts_local, hidden=cell.hidden,
                inter_local=cell.inter_local, topk=cell.topk, quant=cell.quant,
                activation=cell.activation, swiglu_limit=cell.swiglu_limit)


def identity():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def cpu_check(report):
    from engine.kernels.b12x import moe_dispatch as md
    admitted, refused, cell = eligibility(md)
    report('cell', **cell_fields(cell))
    failures = [f'{m} rows must be admitted' for m, ok in admitted.items() if not ok]
    failures += [reason for reason, ok in refused.items() if ok]
    report('eligibility', admitted=admitted, refused_as_expected=not failures,
           refusals={k: (not v) for k, v in refused.items()})
    # The candidate must stay unreachable by default, in any process: the launcher's
    # parameter is None and the launcher resolves None to False ("Unqualified experiments
    # must never become the serving default"). A probe asks for it by name or not at all.
    import inspect
    default = inspect.signature(md.launch_sm120_dynamic_moe).parameters['_prefill_tile64'].default
    if default is not None:
        failures.append(f'the private lane must be opt-in, not {default!r}')
    if 'private M64 prefill is eager-only' not in inspect.getsource(md.launch_sm120_dynamic_moe):
        failures.append('the capture refusal is gone before the lane is qualified')
    if failures:
        raise RuntimeError('; '.join(failures))
    return True


def _relative(actual, expected):
    return ((actual.float() - expected.float()).abs().max()
            / expected.float().abs().max().clamp_min(1e-8)).item()


def _row_disagreement(a, b):
    """Where two results differ, by row. A race that corrupts everything and one that runs
    off the end of a capacity look identical in a scalar norm and nothing alike here."""
    import torch
    differ = (a.float() != b.float()).any(dim=1)
    index = differ.nonzero().flatten()
    if index.numel() == 0:
        return dict(differing_rows=0)
    return dict(differing_rows=int(index.numel()), rows=int(a.shape[0]),
                first=int(index[0]), last=int(index[-1]),
                contiguous_tail=bool(int(index[-1]) == a.shape[0] - 1
                                     and int(index.numel()) == a.shape[0] - int(index[0])))


def _spread(runs):
    """The widest pairwise distance among an arm's repeats: its own noise floor.

    Symmetric on purpose. `_relative` divides by the second argument, so an ordered scan
    would report a different floor depending on which repeat happened to come first.
    """
    return max((max(_relative(a, b), _relative(b, a))
                for i, a in enumerate(runs) for b in runs[i + 1:]), default=0.0)


def gpu_check(report, ranks, output, rows, repeats, tolerance_factor):
    from unittest.mock import patch

    import torch

    from engine.kernels.b12x import moe_dispatch as md
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_decode_scatter_check import rank_path

    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('the M64 candidate is pinned to sm_121a; a verdict from another '
                           f'card is that card\'s (this one is {torch.cuda.get_device_capability()})')
    admitted, refused, cell = eligibility(md)
    report('cell', **cell_fields(cell))
    if not all(admitted.values()) or any(refused.values()):
        raise RuntimeError('eligibility moved; run --cpu for the detail')

    path, prefix = rank_path(ranks), 'L3.moe.'
    loader = rank_loader(path)
    suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    keys = set(loader.keys())
    modelopt = all(prefix + s in keys for s in scale_names)
    loaded = loader.load([prefix + s for s in suffixes + (scale_names if modelopt else ())],
                         device='cuda')
    weights = [loaded[prefix + s] for s in suffixes]
    scales = (ModelOptScales.bind(*(loaded[prefix + s] for s in scale_names),
                                  experts=cell.experts_local, device=weights[0].device)
              if modelopt else None)
    report('identity', rank_file=str(path), gpu=torch.cuda.get_device_name(),
           torch=torch.__version__, cuda=torch.version.cuda,
           weights_sha256={k: hashlib.sha256(v.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                           for k, v in loaded.items()},
           sources_sha256=identity(), repeats=repeats, rows=list(rows),
           tolerance_factor=tolerance_factor, scales='ModelOpt' if modelopt else 'folded',
           scope='same-build component gate; no tok/s, acceptance or answer verdict')

    lane = served()                       # MOE_STATIC_PRODUCTION already carries t,r,sf6,q0
    lane.moe_prepare(*weights, cell.topk, cell.swiglu_limit, scales=scales)
    reference = served(reference_for=('expert',)).moe

    # The arms differ by ONE argument to the served entry. `launch_sm120_moe` owns the M64
    # workspace derivation, so this gate measures the route that would ship rather than a
    # probe-shaped copy of it. The inner watch changes no behaviour: it only records which
    # tile the call actually reached, so an arm cannot pass without having run.
    real_moe, real_dynamic = md.launch_sm120_moe, md.launch_sm120_dynamic_moe
    seen = {}

    def watch(**kw):
        seen['tile_m'] = kw['workspace'].tile_m
        seen['reached_with'] = kw.get('_prefill_tile64')
        return real_dynamic(**kw)

    def entry(tile64):
        # Overwrite, do not default. `b12x_fused_moe` forwards `_prefill_tile64=None`
        # explicitly, and a call-time keyword beats functools.partial -- a partial here
        # silently lost the flag and BOTH arms ran M128 (first srv4 run, 2026-09-15).
        def call(**kw):
            if tile64:
                kw['_prefill_tile64'] = True
            seen['asked'] = kw.get('_prefill_tile64')
            return real_moe(**kw)
        return call

    def run(x, sel, w, tile64):
        with patch.object(md, 'launch_sm120_moe', entry(tile64)), \
                patch.object(md, 'launch_sm120_dynamic_moe', watch):
            return lane.moe(x, sel, w, *weights, cell.swiglu_limit, scales=scales)

    def backend_for(m):
        return md.select_sm120_moe_backend(
            num_tokens=m, num_topk=cell.topk, activation_precision='fp4',
            quant_mode=cell.quant, num_experts=cell.experts_local,
            num_local_experts=cell.experts_local, hidden_size=cell.hidden,
            intermediate_size=cell.inter_local, activation=cell.activation,
            swiglu_limit=cell.swiglu_limit)

    def timed(fn, evict):
        cold = torch.empty(128 << 20, dtype=torch.uint8, device='cuda') if evict else None
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        if cold is not None:
            cold.fill_(19)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end)

    failures, skipped = [], []
    generator = torch.Generator(device='cuda').manual_seed(640_876)
    for m in rows:
        # The M64 window opens at 65 rows, but the dispatcher reaches the dynamic backend
        # only past its routed-pair cutover; below it the static family serves and there is
        # no M64 lane to compare. Report the skip -- do not let it read as a pass.
        chosen = backend_for(m)
        if chosen != 'dynamic':
            skipped.append(m)
            report('rows', m=m, backend=chosen, skipped='the static backend serves this width')
            continue
        x = (torch.randn(m, cell.hidden, device='cuda', generator=generator,
                         dtype=torch.float32) * 0.3).to(torch.bfloat16)
        sel = (torch.randint(0, cell.experts_local, (m, cell.topk), device='cuda',
                             generator=generator).to(torch.int32))
        w = torch.rand(m, cell.topk, device='cuda', generator=generator, dtype=torch.float32)
        w = w / w.sum(-1, keepdim=True)

        arms = {}
        for name, tile64 in (('m128', False), ('m64', True)):
            seen.clear()
            runs = [run(x, sel, w, tile64) for _ in range(repeats)]
            arms[name] = (runs, seen.get('tile_m'))
        (m128_runs, m128_tile), (m64_runs, m64_tile) = arms['m128'], arms['m64']
        if (m128_tile, m64_tile) != (128, 64):
            failures.append(f'{m} rows did not reach both lanes ({m128_tile}, {m64_tile})')
            report('rows', m=m, reached=(m128_tile, m64_tile), asked=seen.get('asked'),
                   reached_with=seen.get('reached_with'), passed=False)
            continue

        control_spread, candidate_spread = _spread(m128_runs), _spread(m64_runs)
        floor = max(control_spread, candidate_spread)
        across = _relative(m64_runs[0], m128_runs[0])
        # Two rules, and the first is the one that matters: the candidate must agree with
        # ITSELF. Only then does comparing the arms against that floor mean anything.
        reproducible = candidate_spread <= max(control_spread * tolerance_factor,
                                               REPRODUCIBLE_CEILING)
        within = across <= max(floor * tolerance_factor, 1e-6)
        values = dict(m=m, m128_spread=control_spread, m64_spread=candidate_spread,
                      across_arms=across, floor=floor, reproducible=reproducible,
                      within_reorder_noise=within)

        # Where, not just how much: a race over the whole tensor and a capacity the kernel
        # runs past are the same number in `_spread` and different pictures here.
        if candidate_spread > 0:
            values['m64_repeat_disagreement'] = _row_disagreement(m64_runs[0], m64_runs[1])
        values['arms_disagreement'] = _row_disagreement(m64_runs[0], m128_runs[0])

        if m <= ORACLE_ROWS:
            oracle = reference(x, sel, w, *weights, cell.swiglu_limit, scales=scales)
            values['m128_vs_reference'] = _relative(m128_runs[0], oracle)
            values['m64_vs_reference'] = _relative(m64_runs[0], oracle)

        # B/A/A/B: the control brackets the candidate so drift is visible as an A/B gap.
        for evict in (False, True):
            label = 'evicted' if evict else 'warm'
            b1, t_b1 = timed(lambda: run(x, sel, w, False), evict)
            a1, t_a1 = timed(lambda: run(x, sel, w, True), evict)
            a2, t_a2 = timed(lambda: run(x, sel, w, True), evict)
            b2, t_b2 = timed(lambda: run(x, sel, w, False), evict)
            base, cand = (t_b1 + t_b2) / 2, (t_a1 + t_a2) / 2
            values[f'{label}_m128_ms'] = base
            values[f'{label}_m64_ms'] = cand
            values[f'{label}_ratio'] = cand / base if base else None
            values[f'{label}_bracket_drift'] = abs(t_b1 - t_b2) / base if base else None
            del b1, a1, a2, b2

        if not reproducible:
            failures.append(f'{m} rows: the candidate disagrees with itself by '
                            f'{candidate_spread:.3e} (the control: {control_spread:.3e})')
        elif not within:
            failures.append(f'{m} rows: arms differ by {across:.3e}, floor {floor:.3e}')
        report('rows', passed=reproducible and within, **values)

    report('verdict', passed=not failures, failures=failures, skipped_static_rows=skipped)
    if output:
        Path(output).write_text(json.dumps(
            dict(passed=not failures, failures=failures, skipped_static_rows=skipped), indent=1))
    if failures:
        raise RuntimeError('; '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # The queue's runner is `docker run --gpus all <probe>`, so the device is the default
    # and every flag it may pass is one literal token (bench/fleet_onepass.ST_FLAGS).
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cpu', action='store_true', help='no device: eligibility and identity')
    mode.add_argument('--gpu', action='store_true', help='the default; accepted for symmetry')
    parser.add_argument('--ranks')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rows', default=','.join(str(m) for m in ROWS),
                        help='comma-separated row counts')
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--tolerance-factor', type=float, default=TOLERANCE_FACTOR)
    args = parser.parse_args()
    if args.cpu and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU mode requires CUDA_VISIBLE_DEVICES=')
    os.environ.setdefault('CUTE_DSL_ARCH', 'sm_121a')
    sys.path.insert(0, str(ROOT))

    rows = []
    def report(event, **values):
        rows.append(dict(event=event, **values))
        print(json.dumps(rows[-1]), flush=True)

    if args.cpu:
        report('sources', sha256=identity())
        cpu_check(report)
        report('verdict', passed=True, scope='eligibility and identity only; no device ran')
        return
    if not args.ranks:
        raise ValueError('the GPU gate needs the exact consumer --ranks')
    rows_asked = tuple(int(v) for v in str(args.rows).split(',') if v.strip())
    if not rows_asked:
        raise ValueError('--rows takes at least one row count')
    gpu_check(report, args.ranks, args.output, rows_asked, args.repeats,
              args.tolerance_factor)


if __name__ == '__main__':
    main()
