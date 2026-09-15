"""C1 static MHC tails against the same packed kernel's work queue.

The arithmetic and per-token chunk arrivals stay intact. Static C1 owners
remove the shared tail and exit tickets. Check all fields bitwise, replay
mixed row counts, and time native calls with external events inside graphs.
"""
import hashlib
import json
from pathlib import Path

import torch

from probes.engine_decode_fusions import _capture
from probes.engine_mhc_c2_packed import Rows, call, compare, load


class TailArm:
    """Change tail ownership while retaining the owner's shape and weight dispatch."""

    def __init__(self, owner, mode):
        self.ext, self.mode = owner.ext, mode

    def run_mhc(self, ptrs, scalars, ints, bf16_fn, ar_consumer):
        return self.ext.run_mhc(ptrs, scalars, ints, bf16_fn, ar_consumer, self.mode)

    def run_mhc_packets(self, ptrs, scalars, ints, packets, bf16_fn):
        return self.ext.run_mhc_packets(ptrs, scalars, ints, packets, bf16_fn, self.mode)


def captured_times(report, calls, *, layers, brackets, captures=2, **meta):
    flush = torch.empty(128 << 20, device='cuda', dtype=torch.uint8)
    for cache in ('warm', 'evicted'):
        repeats = max(1, 32 // layers) if cache == 'warm' else 1
        for capture in range(captures):
            graphs, events = {}, {}
            order = (0, 1) if capture % 2 == 0 else (1, 0)
            try:
                for arm in order:
                    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
                    def timed(arm=arm, start=start, end=end):
                        start.record()
                        for _ in range(repeats):
                            calls[arm]()
                        end.record()
                    graphs[arm], _ = _capture(timed)
                    events[arm] = start, end
                samples = []
                for _ in range(brackets):
                    for arm in (0, 1, 1, 0):
                        values = []
                        for _ in range(16):
                            if cache == 'evicted':
                                flush.zero_()
                            graphs[arm].replay()
                            start, end = events[arm]
                            end.synchronize()
                            values.append(start.elapsed_time(end) * 1000 / repeats)
                        samples.append(dict(arm=arm, us=sum(values) / len(values)))
                control = [s['us'] for s in samples if s['arm'] == 0]
                candidate = [s['us'] for s in samples if s['arm'] == 1]
                report('timing', cache=cache, capture=capture, capture_order=order,
                       layers=layers, repeats=repeats, samples=samples,
                       control_us=sum(control)/len(control), candidate_us=sum(candidate)/len(candidate),
                       change_pct=100*(sum(candidate)/sum(control)-1), **meta)
            finally:
                for graph in graphs.values():
                    graph.reset()


def exact(report, owner, keys, coefficients, inputs, packets):
    graphs, outputs = {}, {}
    try:
        for mode in (0, 1, -1):
            adapter = TailArm(owner, mode)
            graphs[mode], outputs[mode] = _capture(
                lambda adapter=adapter: call(owner, adapter, keys, coefficients, inputs, packets))
        for step, scale in enumerate((0., .001, 1., 30.)):
            inputs.fill(step, scale)
            for order in ((0, 1, -1), (-1, 1, 0)):
                for output in outputs.values():
                    for values in output:
                        for tensor in values:
                            tensor.fill_(float('nan'))
                for mode in order:
                    graphs[mode].replay()
                torch.cuda.synchronize()
                if not all(t.isfinite().all().item() for values in outputs[0] for t in values):
                    raise AssertionError('nonfinite dynamic reference')
                compare(outputs[0], outputs[1], 'static tails')
                compare(outputs[0], outputs[-1], 'automatic tails')
        report('exact', rows=8, layers=len(keys), packets=packets, bitwise=True,
               magnitudes=4, replay_orders=2, poisoned_outputs=True, descriptor_rebound=packets)
    finally:
        for graph in graphs.values():
            graph.reset()


def transitions(report, owner, key, coefficients, packets):
    graphs, outputs = {}, {}
    inputs = {m: Rows(m) for m in (1, 7, 8, 16, 32, 64)}
    try:
        for m, values in inputs.items():
            for mode in (0, -1):
                adapter = TailArm(owner, mode)
                graphs[m, mode], outputs[m, mode] = _capture(
                    lambda adapter=adapter, values=values: call(owner, adapter, [key], coefficients, values, packets))
        schedule = ((8, -1), (16, -1), (8, 0), (1, -1), (8, -1), (32, -1),
                    (7, -1), (16, 0), (8, -1), (64, -1), (8, -1))
        for step in range(32):
            for values in inputs.values():
                values.fill(step, .001 if step % 2 else 1.)
            for m, mode in schedule:
                graphs[m, mode].replay()
            # Compare every shape after the mixed schedule has exercised the
            # shared arrivals and the legacy launch's tail/exit rearming.
            for m in inputs:
                graphs[m, 0].replay()
                graphs[m, -1].replay()
            torch.cuda.synchronize()
            for m in inputs:
                compare(outputs[m, 0], outputs[m, -1], f'mixed replay rows={m}')
        for m in (1, 7, 16, 32, 64):
            try:
                call(owner, TailArm(owner, 1), [key], coefficients, inputs[m], packets)
            except RuntimeError as exc:
                if 'packed C1 consumer' not in str(exc):
                    raise
            else:
                raise AssertionError(f'forced static tails accepted {m} rows')
        report('transitions', packets=packets, rows=list(inputs), iterations=32,
               schedule=schedule, bitwise=True, forced_shape_refusals=True)
    finally:
        for graph in graphs.values():
            graph.reset()


def full_precision(report, coefficients, key, inputs):
    """Eight rows without a lossless BF16 pack must retain their FP32 consumer."""
    from engine.kernels.dense.mhc import MHC
    fn = torch.randn(24, 16384, device='cuda') * .006
    owner = MHC({key: fn})
    if owner.weights[key][1] is not None:
        raise AssertionError('the FP32 fallback test needs coefficients that cannot be packed losslessly')
    for packets in (False, True):
        graphs, outputs = {}, {}
        try:
            for mode in (0, -1):
                adapter = TailArm(owner, mode)
                graphs[mode], outputs[mode] = _capture(
                    lambda adapter=adapter: call(owner, adapter, [key], coefficients, inputs, packets))
            for step in range(4):
                inputs.fill(step, 1.)
                for output in outputs.values():
                    for values in output:
                        for tensor in values:
                            tensor.fill_(float('nan'))
                for mode in (0, -1):
                    graphs[mode].replay()
                torch.cuda.synchronize()
                if not all(t.isfinite().all().item() for values in outputs[0] for t in values):
                    raise AssertionError('nonfinite FP32 fallback reference')
                compare(outputs[0], outputs[-1], 'FP32 coefficient fallback')
            try:
                call(owner, TailArm(owner, 1), [key], coefficients, inputs, packets)
            except RuntimeError as exc:
                if 'packed C1 consumer' not in str(exc):
                    raise
            else:
                raise AssertionError('forced static tails accepted unpackable coefficients')
            report('fp32_fallback', rows=8, packets=packets, bitwise=True, forced_precision_refusal=True)
        finally:
            for graph in graphs.values():
                graph.reset()


def check(report, ranks=None, *, samples=4, timing=True):
    from engine.kernels.dense.mhc import MHC
    torch.manual_seed(915)
    if ranks:
        origin, keys, weights, coefficients, digest = load(ranks)
        keys = keys[1:]  # all 89 carried boundaries; the first attention has no predecessor
    else:
        keys = [f'cell{i}' for i in range(4)]
        weights = {key: (torch.randn(24, 16384, device='cuda')*.006).bfloat16().float() for key in keys}
        coefficients = {key: (torch.tensor([.2, .3, .4], device='cuda'), torch.randn(24, device='cuda')*.1,
                              torch.randn(4096, device='cuda', dtype=torch.bfloat16)) for key in keys}
        origin, digest = 'synthetic BF16-origin coefficients', None
    owner = MHC(weights)
    if any(pack is None for _, pack in owner.weights.values()):
        raise AssertionError('every coefficient must have a lossless BF16 pack')
    report('weights', source=origin, keys=keys, fn_sha256=digest, lossless_bf16=True)
    inputs = Rows(8)
    for packets in (False, True):
        exact(report, owner, keys, coefficients, inputs, packets)
        transitions(report, owner, keys[0], coefficients, packets)
    full_precision(report, coefficients, keys[0], inputs)
    if timing:
        inputs.fill(0, 1.)
        for packets in (False, True):
            for scope, chosen in (('single', keys[:1]), ('chain', keys)):
                calls = {mode: lambda mode=mode, chosen=chosen, packets=packets:
                         call(owner, TailArm(owner, mode), chosen, coefficients, inputs, packets) for mode in (0, 1)}
                captured_times(report, calls, layers=len(chosen), brackets=samples, scope=scope, packets=packets)


def main(ranks=None, *, samples=None, output=None):
    root = Path(__file__).resolve().parents[1]
    sink = open(output, 'w') if output else None
    def report(event, **values):
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        if sink:
            sink.write(line+'\n'); sink.flush()
    try:
        report('identity', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               source_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                              ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/mhc.py',
                               'probes/engine_mhc_c1_tails.py', 'probes/engine_mhc_c2_packed.py')},
               scope='one GB10, local rank packets; not TP4 transport or consumer performance')
        check(report, ranks, samples=int(samples) if samples else 4)
        report('complete', status='PASS', consumer_metrics_measured=False,
               peak_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if sink:
            sink.close()
