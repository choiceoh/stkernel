"""Same-build gate for C=2 packed mHC: 16 verify rows read the BF16 coefficient packs 8 rows read.

Numerics first, at zero tolerance against the FP32 coefficients of the same build, then B/A/A/B component timing.
No model boot and no transport: a packet descriptor names four local rank tensors. Kernel evidence, not a consumer
speed claim (D17). `m14_diagnosis` replays the 2026-09-13 mhc_batch failure from its own seeds.
"""
import hashlib
import json
from pathlib import Path

import torch

from probes.engine_decode_dsa_inputs import timings
from probes.engine_decode_fusions import _capture

ROOT = Path(__file__).resolve().parents[1]
SCALARS = (1e-5, 1e-6, 2., 20)       # the served config: rms_norm_eps, hc_eps, post_mult, hc_sinkhorn_iters
FILES = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/mhc.py', 'engine/kernels/dense/__init__.py',
         'probes/engine_mhc_c2_packed.py', 'probes/engine_kernel_check.py', 'tests/test_engine_direct_mhc_cuda.py')
FIELDS = ('residual', 'post', 'comb', 'layer_input')
BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


class Arm:
    """One native form under the owner's own argument construction (tensors, workspace, scalars): swaps the weight
    pointer between the owner's FP32 tensor and its BF16 pack and fixes the launch flags."""

    def __init__(self, owner, *, packed, consumer=None):
        self.ext, self.packed, self.consumer = owner.ext, packed, consumer
        self.pointers = {}
        for fp32, pack in owner.weights.values():
            pair = (fp32.data_ptr(), None if pack is None else pack.data_ptr())
            self.pointers[pair[0]] = pair
            if pack is not None:
                self.pointers[pair[1]] = pair

    def _weight(self, ptrs):
        ptrs = list(ptrs)
        fp32, pack = self.pointers[ptrs[4]]
        if self.packed and pack is None:
            raise RuntimeError('this arm needs a lossless BF16 pack')
        ptrs[4] = pack if self.packed else fp32
        return ptrs

    def run_mhc(self, ptrs, scalars, ints, bf16_fn, ar_consumer):
        consumer = ar_consumer if self.consumer is None else self.consumer
        return self.ext.run_mhc(self._weight(ptrs), scalars, ints, self.packed, consumer)

    def run_mhc_packets(self, ptrs, scalars, ints, packets, bf16_fn):
        return self.ext.run_mhc_packets(self._weight(ptrs), scalars, ints, packets, self.packed)


class Record:
    """The owner's own dispatch, recorded: (rows, packed pointer?, bf16_fn, ar_consumer)."""

    def __init__(self, owner):
        self.ext, self.calls = owner.ext, []
        self.packs = {pack.data_ptr() for _, pack in owner.weights.values() if pack is not None}

    def run_mhc(self, ptrs, scalars, ints, bf16_fn, ar_consumer):
        self.calls.append((int(ints[0]), ptrs[4] in self.packs, bool(bf16_fn), bool(ar_consumer)))
        return self.ext.run_mhc(ptrs, scalars, ints, bf16_fn, ar_consumer)

    def run_mhc_packets(self, ptrs, scalars, ints, packets, bf16_fn):
        self.calls.append((int(ints[0]), ptrs[4] in self.packs, bool(bf16_fn), False))
        return self.ext.run_mhc_packets(ptrs, scalars, ints, packets, bf16_fn)


def served_dispatch(rows, *, packets, packed_rows):
    """What engine/kernels/dense/mhc.MHC launches for lossless coefficients, stated independently of it."""
    packed = rows <= packed_rows
    return (rows, packed, packed, packed and not packets)


class Rows:
    """Changing inputs behind fixed pointers: two banks of four rank tensors (the descriptor is rebound between them),
    their rank-ordered BF16 sum for the ordinary consumer, and the carried residual and mixes."""
    # A local-first fold, or a sum that skips the BF16 boundary, changes these columns (tests/test_engine_direct_mhc_cuda).
    CANARY = ((2.**24, 1.), (-2.**24, 2.**-8), (1., 2.**-9), (1., 0.))

    def __init__(self, rows):
        new = lambda *shape, dtype=torch.bfloat16: torch.empty(*shape, device='cuda', dtype=dtype)
        self.rows = rows
        self.banks = [[new(rows, 4096) for _ in range(4)] for _ in range(2)]
        self.addresses = [torch.tensor([t.data_ptr() for t in bank], device='cuda', dtype=torch.int64)
                          for bank in self.banks]
        self.descriptor = self.addresses[0].clone()
        self.meta = new(rows, 4096)       # the packet call's shape carrier; the kernel must not read it
        self.x = new(rows, 4096)
        self.res = new(rows, 4, 4096)
        self.post = new(rows, 4, 1, dtype=torch.float32)
        self.comb = new(rows, 4, 4, dtype=torch.float32)
        self.fill(0, 1.)

    def fill(self, step, scale):
        bank = self.banks[step % 2]
        for rank, value in enumerate(bank):
            value.normal_().mul_(scale)
            value[:, 0] = self.CANARY[rank][0]
            value[:, 1] = self.CANARY[rank][1]
        self.descriptor.copy_(self.addresses[step % 2])
        total = bank[0].float() + bank[1].float()
        total = (total + bank[2].float()) + bank[3].float()
        self.x.copy_(total.bfloat16())
        self.meta.fill_(float('nan'))
        self.res.normal_().mul_(scale)
        self.post.uniform_(0., 2.)
        self.comb.uniform_(0., 1.)


def call(owner, ext, keys, coefficients, inputs, packets):
    previous = owner.ext
    if ext is not None:
        owner.ext = ext
    try:
        outputs = []
        for key in keys:
            scale, base, norm = coefficients[key]
            x = inputs.meta if packets else inputs.x
            outputs.append(owner(key, x, inputs.res, inputs.post, inputs.comb, scale, base, norm, *SCALARS,
                                 packets=inputs.descriptor if packets else None))
        return outputs
    finally:
        owner.ext = previous


def difference(actual, expected):
    """Bitwise mismatches, sized in torch.testing's terms (float64 promotion, mismatched elements only)."""
    differ = actual.reshape(-1).view(BITS[actual.dtype]) != expected.reshape(-1).view(BITS[expected.dtype])
    count = int(differ.sum().item())
    if not count:
        return dict(mismatched=0, total=actual.numel())
    a, b = actual.double().flatten(), expected.double().flatten()
    delta = (a - b).abs()[differ]
    relative = delta / b.abs()[differ]
    return dict(mismatched=count, total=actual.numel(), max_absolute=float(delta.max().item()),
                max_relative=float(relative.max().item()), first_index=int(differ.nonzero()[0].item()))


def compare(expected, actual, label):
    for key_index, (want, got) in enumerate(zip(expected, actual)):
        for field, a, b in zip(FIELDS, got, want):
            if a.dtype != b.dtype or not torch.equal(a.reshape(-1).view(BITS[a.dtype]), b.reshape(-1).view(BITS[b.dtype])):
                raise AssertionError(f'{label}: call {key_index} {field} differs: {difference(a, b)}')


def m14_diagnosis(report):
    """The 2026-09-13 mhc_batch lane (source 2b3a74b9) sent MHC's [output, hidden, stream] pack to
    run_mhc(bf16_fn=True, ar_consumer=False), whose persistent BF16 kernel reads [output, stream, hidden]. Replay its
    first failing comparison from the same seeds, then give that kernel the layout it reads."""
    from engine.kernels.dense.mhc import MHC
    torch.manual_seed(91715)
    fn = (torch.randn(24, 16384, device='cuda') * .006).bfloat16().float()   # the lane's first of 89 packs
    torch.manual_seed(91613)                                                   # tests.test_engine_mhc_single.inputs
    torch.randn(24, 16384, device='cuda')
    rows = 14
    x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
    res = torch.randn(rows, 4, 4096, device='cuda', dtype=torch.bfloat16)
    post = torch.rand(rows, 4, 1, device='cuda')
    comb = torch.rand(rows, 4, 4, device='cuda')
    scale = torch.tensor([.2, .3, .4], device='cuda')
    base = torch.randn(24, device='cuda') * .1
    norm = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
    owner = MHC({'m14': fn})
    fp32, vector = owner.weights['m14']
    scalar = fn.bfloat16()
    assert vector is not None and tuple(vector.shape) == (24, 4096, 4) and tuple(scalar.shape) == (24, 16384)

    class Storage:
        def __init__(self, ext, weight, bf16_fn):
            self.ext, self.weight, self.bf16_fn = ext, weight, bf16_fn

        def run_mhc(self, ptrs, scalars, ints, bf16_fn, ar_consumer):
            ptrs = list(ptrs)
            ptrs[4] = self.weight.data_ptr()
            return self.ext.run_mhc(ptrs, scalars, ints, self.bf16_fn, False)

    def run(weight, bf16_fn):
        previous = owner.ext
        owner.ext = Storage(previous, weight, bf16_fn)
        try:
            result = owner('m14', x, res, post, comb, scale, base, norm, *SCALARS)
            torch.cuda.synchronize()
            return [t.clone() for t in result]
        finally:
            owner.ext = previous

    # The lane's two replays before the failure: scale 0 (every coefficient multiplies zero) and scale 0.001.
    for magnitude in (0., .001):
        for value in (x, res, post, comb):
            value.normal_().mul_(magnitude)
        reference = run(fp32, False)
        for layout, weight in (('vector pack, the lane', vector), ('scalar pack', scalar)):
            got = run(weight, True)
            report('m14_diagnosis', rows=rows, input_scale=magnitude, layout=layout,
                   kernel='mk_mhc_bf16_kernel (persistent grid, reads [output, stream, hidden])',
                   fields={f: difference(a, b) for f, a, b in zip(FIELDS, got, reference)},
                   recorded_failure=dict(field='post', mismatched=56, total=56, max_absolute=0.000341951847076416,
                                         max_relative=0.0003536288859322667) if magnitude else None)
    # The same persistent kernel with the layout it reads, at wider verify widths.
    for rows in (14, 16, 28):
        x, res = (torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16),
                  torch.randn(rows, 4, 4096, device='cuda', dtype=torch.bfloat16))
        post, comb = torch.rand(rows, 4, 1, device='cuda'), torch.rand(rows, 4, 4, device='cuda')
        for magnitude in (.001, 1.):
            x.normal_().mul_(magnitude)
            res.normal_().mul_(magnitude)
            reference = run(fp32, False)
            got = run(scalar, True)
            report('m14_scalar_layout', rows=rows, input_scale=magnitude,
                   fields={f: difference(a, b) for f, a, b in zip(FIELDS, got, reference)})


def load(ranks):
    from probes.engine_decode_scatter_check import rank_path
    from engine.profiles.glm53.weights import rank_loader
    path = rank_path(ranks)
    loader = rank_loader(path)
    layers = sorted(int(k.split('.')[0][1:]) for k in loader.keys() if k.endswith('.hc.attn_fn'))
    keys = [f'L{layer}.hc.{side}_fn' for layer in layers for side in ('attn', 'ffn')]
    extra = [f'L{layer}.hc.{side}_{part}' for layer in layers for side in ('attn', 'ffn') for part in ('scale', 'base')]
    norms = [f'L{layer}.{name}' for layer in layers for name in ('in_norm', 'post_norm')]
    loaded = loader.load(keys + extra + norms, device='cuda')
    coefficients = {}
    for key in keys:
        layer, side = key.split('.')[0], key.split('.')[2].removesuffix('_fn')
        coefficients[key] = (loaded[f'{layer}.hc.{side}_scale'], loaded[f'{layer}.hc.{side}_base'],
                             loaded[f'{layer}.{"in_norm" if side == "attn" else "post_norm"}'])
    digest = hashlib.sha256()
    for key in keys:
        digest.update(loaded[key].cpu().numpy().tobytes())
    return str(path), keys, {key: loaded[key].contiguous() for key in keys}, coefficients, digest.hexdigest()


def arms(candidate, control, rows, *, packets):
    """(name, owner, adapter, expected dispatch): the FP32 control first, then every candidate of that family."""
    served = [('control', control, Record(control), served_dispatch(rows, packets=packets, packed_rows=8)),
              ('serving', candidate, Record(candidate),
               served_dispatch(rows, packets=packets, packed_rows=candidate.PACKED_ROWS))]
    if packets:
        return [('fp32', candidate, Arm(candidate, packed=False), None),
                ('packed', candidate, Arm(candidate, packed=True), None)] + served
    family = [('grid_fp32', candidate, Arm(candidate, packed=False, consumer=False), None)]
    if rows <= 16:
        family += [('consumer_fp32', candidate, Arm(candidate, packed=False, consumer=True), None),
                   ('consumer', candidate, Arm(candidate, packed=True, consumer=True), None)]
    return family + served


def capture(owners, keys, coefficients, inputs, *, packets):
    candidate, control = owners
    family = arms(candidate, control, inputs.rows, packets=packets)
    graphs, outputs = {}, {}
    try:
        for name, owner, adapter, expected in family:
            graph, output = _capture(lambda: call(owner, adapter, keys, coefficients, inputs, packets))
            graphs[name], outputs[name] = graph, output
            if expected is not None and set(adapter.calls) != {expected}:
                raise AssertionError(f'{name} dispatch at {inputs.rows} rows: {sorted(set(adapter.calls))} != {expected}')
    except BaseException:
        for graph in graphs.values():
            graph.reset()
        raise
    return [name for name, *_ in family], graphs, outputs


def exact(report, owners, keys, coefficients, inputs, *, packets, scales, label):
    names, graphs, outputs = capture(owners, keys, coefficients, inputs, packets=packets)
    try:
        for step, magnitude in enumerate(scales):
            inputs.fill(step, magnitude)
            for order in (names, names[::-1]):
                for output in outputs.values():
                    for values in output:
                        for tensor in values:
                            tensor.fill_(float('nan'))
                for name in order:
                    graphs[name].replay()
                torch.cuda.synchronize()
                reference = outputs[names[0]]
                for values in reference:
                    if not all(t.isfinite().all().item() for t in values):
                        raise AssertionError(f'{label}: non-finite FP32 reference at scale {magnitude}')
                for name in names[1:]:
                    compare(reference, outputs[name], f'{label} {name} vs {names[0]} x{magnitude}')
        report('mhc_packed_exact', family='packets' if packets else 'ordinary', rows=inputs.rows, calls=len(keys),
               arms=names, scales=list(scales), replay_orders='forward/reverse', descriptor_rebound=packets,
               poisoned_outputs=True, bitwise=True, reference=names[0], label=label)
        return names, graphs, outputs
    except BaseException:
        for graph in graphs.values():
            graph.reset()
        raise


def refusals(report, owner, coefficients, key):
    """The native entry refuses what it has no kernel for before any launch, and admits the widest served width."""
    cases = []
    for rows, message in ((17, 'AR consumer requires'), (16, None)):
        inputs = Rows(rows)
        try:
            call(owner, Arm(owner, packed=True, consumer=True), [key], coefficients, inputs, False)
            torch.cuda.synchronize()
            refused = None
        except RuntimeError as error:
            refused = str(error).splitlines()[0]
        if (refused is None) != (message is None) or (message is not None and message not in refused):
            raise AssertionError(f'AR consumer at {rows} rows: expected refusal {message!r}, got {refused!r}')
        cases.append(dict(rows=rows, family='ordinary', consumer=True, refused=refused))
    report('mhc_packed_refusals', cases=cases)


def check(report, ranks=None, *, timing=True):
    import unittest
    from engine.kernels.dense.mhc import MHC
    # The served dispatch across families first: packets against the ordinary consumer on the rank-ordered sum, both
    # coefficient storages, rows 1/7/8/16/28/64 (tests/test_engine_direct_mhc_cuda).
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_engine_direct_mhc_cuda')
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    if not run.wasSuccessful() or run.skipped:
        raise RuntimeError('direct MHC dispatch gate failed or skipped')
    report('direct_mhc_dispatch', tests=run.testsRun, skipped=0, rows=[1, 7, 8, 16, 28, 64])
    origin, keys, weights, coefficients, digest = load(ranks)
    candidate, control = MHC(weights), MHC(weights, packed_rows=8)
    lossy = [key for key, (_, pack) in candidate.weights.items() if pack is None]
    report('mhc_packed_weights', source=origin, keys=len(keys), fn_sha256=digest, lossless_bf16=len(keys) - len(lossy),
           lossy=lossy)
    if lossy:
        raise RuntimeError('real mHC coefficients are expected to be BF16-origin; the packed path would not serve')
    owners = (candidate, control)
    refusals(report, candidate, coefficients, keys[1])
    # Breadth: every row count a verify step can take through each consumer, six real packs per launch graph.
    sample = keys[1::15][:6]
    for packets, widths in ((True, (1, 2, 7, 8, 9, 12, 15, 16, 17, 32)), (False, (1, 2, 7, 8, 9, 12, 15, 16, 17))):
        for rows in widths:
            _, graphs, _ = exact(report, owners, sample, coefficients, Rows(rows), packets=packets,
                                 scales=(0., .001, 1., 32., .5), label=f'breadth {rows}')
            for graph in graphs.values():
                graph.reset()
    # Depth at the served widths, then timing on the same graphs: 84 packet consumers and 5 ordinary ones per step.
    chain, ordinary = keys[1:85], keys[85:90]
    for rows in (8, 16):
        for packets, calls, pairs in (
                (True, chain, (('fp32', 'packed'), ('control', 'serving'))),
                (False, ordinary, (('grid_fp32', 'consumer'), ('grid_fp32', 'consumer_fp32'),
                                   ('consumer_fp32', 'consumer'), ('control', 'serving')))):
            inputs = Rows(rows)
            names, graphs, outputs = exact(report, owners, calls, coefficients, inputs, packets=packets,
                                           scales=(.001, 1., 1.), label=f'depth {rows}')
            try:
                if timing:
                    inputs.fill(0, 1.)
                    for bracket in range(2):
                        for base, candidate_arm in pairs:
                            timings(report, f'{"packets" if packets else "ordinary"}:{base}->{candidate_arm}', rows,
                                    [graphs[base], graphs[candidate_arm]], calls=len(calls), bracket=bracket,
                                    real_packs=True)
            finally:
                for graph in graphs.values():
                    graph.reset()
    # Last: a diagnosis, not a gate. It reports and never raises on a mismatch.
    m14_diagnosis(report)


def main(ranks=None, output=None):
    rows = []

    def report(event, **values):
        row = dict(event=event, **values)
        rows.append(row)
        print(json.dumps(row), flush=True)

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('requires an admitted GB10')
    report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES})
    try:
        check(report, ranks)
        report('complete', status='PASS', consumer_metrics_measured=False)
    finally:
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(rows, indent=1) + '\n')
