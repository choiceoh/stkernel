"""Qualify the decode fastpaths missed by K=7, in one admitted short hold.

The serving allowlists stay unchanged. The probe-only overrides expire with
engine_decode_bundle on 2026-09-16. Promote only cells that win both cache
regimes after their numerical gate; this is not a serving speed verdict.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import time
import unittest
from unittest.mock import patch


def verification_rows(spec_k):
    """This experiment covers exactly four K=7 verification widths."""
    if type(spec_k) is not int or spec_k != 7:
        raise ValueError('the K=7 fastpath probe requires spec_k=7')
    return tuple(width * (spec_k + 1) for width in range(1, 5))


def w4_projection_keys(loader):
    """Match the rank's executed dense layout; packed MLPs do not use W4A8."""
    keys = sorted(loader.keys())
    suffixes = ('.kda.in_proj', '.kda.o_proj', '.mla.qkv_a', '.mla.q_b',
                '.mla.o_proj', '.idx.wq_b')
    selected = []
    for suffix in suffixes:
        found = [key for key in keys if key.endswith(suffix)]
        if not found:
            raise RuntimeError(f'the rank lacks the W4 projection family {suffix}')
        selected.append(found[0])
    plain = [next((key for key in keys if key.endswith(suffix)), None)
             for suffix in ('.mlp.gate_up', '.mlp.down')]
    packed = [next((key for key in keys if key.endswith(suffix)), None)
              for suffix in ('.mlp.w13', '.mlp.w2')]
    if all(plain) and not any(packed):
        return selected + plain, False
    if all(packed) and not any(plain):
        return selected, True
    raise RuntimeError('the rank must declare a complete, unambiguous BF16 or packed dense MLP')


@contextmanager
def candidate_projections(module, rows):
    """Bind the proposed capture cells only inside this probe process."""
    with patch.object(module, 'DECODE_ROWS', tuple(sorted(set(module.DECODE_ROWS) | set(rows)))):
        yield


def c1_check(report, ranks):
    import gc
    import torch
    from engine.kernels.dense import DenseLinear, W4Pack, extension
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_decode_batch import timing
    from probes.engine_decode_fusions import _capture
    from probes.engine_decode_scatter_check import rank_path

    path = rank_path(ranks)
    loader, ext = rank_loader(path), extension()
    before, mode, state = ext.gemm_input_mode(), ext.gemm_input_cta_mode(), ext.probe_state()
    all_keys, packed_mlp = w4_projection_keys(loader)
    suffixes = ('.kda.in_proj', '.mla.o_proj', '.mlp.gate_up')
    keys = [key for key in all_keys if key.endswith(suffixes)]
    covered = set()
    try:
        ext.set_gemm2(0)
        ext.set_input_cta(4)
        for key in keys:
            weight = loader.load([key], device='cuda')[key]
            layer = DenseLinear(weight, prefill=False)
            if len(layer.packs) != 1:
                raise RuntimeError(f'{key}: the probe requires a single prepared W4 pack')
            p = layer.packs[0]
            ext.set_gemm_input(1)
            baseline_plan = ext.gemm_input_plan(8, p.rows, p.cols, False, False)
            ext.set_gemm_input(2)
            candidate_plan = ext.gemm_input_plan(8, p.rows, p.cols, False, False)
            if baseline_plan[0] or not candidate_plan[0]:
                raise RuntimeError(f'{key}: M8 did not change from ordinary to input reuse')
            covered.add((p.rows, p.cols))
            digest = hashlib.sha256()
            for value in (p.data, p.scale, p.rowscale):
                digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
            packs = [W4Pack(p.data.clone(), p.scale.clone(), p.rowscale.clone(), p.rows, p.cols)
                     for _ in range(4)]
            parent = torch.full((8, p.cols + 8), float('nan'), device='cuda', dtype=torch.bfloat16)
            x = parent[:, 4:4+p.cols]
            x.normal_()

            def run(enabled):
                ext.set_gemm_input(2 if enabled else 1)
                outputs = []
                for pack in packs:
                    out = torch.empty(8, pack.rows, device=x.device, dtype=x.dtype)
                    ext.run_gemm(x, pack.data, pack.scale, out, pack.rows,
                                 1., 0, pack.rowscale.data_ptr(), 0, 0, 0)
                    outputs.append(out)
                return outputs

            graphs, outputs = [], []
            base, candidate = lambda: run(False), lambda: run(True)
            try:
                for fn in (base, candidate):
                    graph, output = _capture(fn)
                    graphs.append(graph); outputs.append(output)
                for magnitude in (0., .001, 1., 50., 1.):
                    x.normal_().mul_(magnitude)
                    for order in ((0, 1), (1, 0)):
                        for arm in order:
                            for value in outputs[arm]:
                                value.fill_(float('nan'))
                            graphs[arm].replay()
                        for got, want in zip(outputs[1], outputs[0]):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                report('decode_k7_numerics', candidate='c1_input', rows=8, key=key,
                       n=p.rows, k=p.cols, exact=True, poisoned_replay=True, parent_stride=x.stride(0),
                       rank_file=str(path), packed_weight_sha256=digest.hexdigest())
                timing(report, 'c1_input', 8, *graphs, (base, candidate), key=key,
                       n=p.rows, k=p.cols, packs=len(packs), baseline_plan=baseline_plan,
                       candidate_plan=candidate_plan, packed_weight_sha256=digest.hexdigest())
            finally:
                for graph in graphs:
                    graph.reset()
            del packs, outputs, layer, weight
            gc.collect()
        expected = {(6416, 4096), (4096, 4096)}
        if not packed_mlp:
            expected.add((6144, 4096))
        if covered != expected:
            raise RuntimeError(f'the actual rank missed a C1 fastpath family: {covered}')
    finally:
        ext.set_gemm_input(before)
        ext.set_input_cta(mode)
        ext.restore_probe_state(state)


def check(report, ranks):
    import torch
    from engine.kernels import decode_projection
    from engine.profiles.glm53 import facts
    from probes.engine_decode_batch import indexer_check, wide_check
    from probes.engine_decode_projection import paired_check
    from probes.engine_decode_bundle import require_current_probe

    require_current_probe()
    rows = verification_rows(facts.SPEC_K)
    torch.cuda.set_per_process_memory_fraction((6 << 30) / torch.cuda.get_device_properties(0).total_memory)
    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/__init__.py',
             'engine/kernels/decode_projection.py', 'engine/profiles/glm53/net.py',
             'engine/profiles/glm53/facts.py', 'tests/test_engine_decode_seven.py',
             'probes/engine_decode_k7.py', 'probes/engine_decode_batch.py',
             'probes/engine_decode_projection.py', 'probes/engine_decode_fusions.py')
    report('decode_k7_identity', spec_k=facts.SPEC_K, rows=rows, torch=torch.__version__,
           cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
           source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
           serving_defaults_changed=False,
           scope='same-pack projection components; no model boot, state arithmetic, or acceptance claim')
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_engine_decode_seven.SevenRowDenseTests')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 3:
        raise RuntimeError('C1 input-reuse graph/stride gate failed or skipped')
    report('decode_k7_dense_gate', tests=result.testsRun, passed=True)
    c1_check(report, ranks)
    wide_check(report, ranks, row_cases=rows[1:], timing_rows=rows[1:])
    with candidate_projections(decode_projection, rows):
        paired_check(report, ranks, row_cases=rows, run_tests=False)
        indexer_check(report, ranks, row_cases=rows)
    report('decode_k7_complete', passed=True, seconds=time.monotonic()-started,
           max_allocated_bytes=torch.cuda.max_memory_allocated())
