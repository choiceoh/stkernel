"""Same-pack decode candidates, all explicitly off in serving.

Each lane qualifies changed-input graph replay before timing complete consumers
over distinct weight packs. None of these timings is engine throughput.
"""
import copy
from unittest.mock import patch

import torch


def dense_call(x, pack, *, pack_rows=8, short_input=False):
    from engine.kernels.dense import extension
    output = torch.empty((x.shape[0], pack.rows), dtype=x.dtype, device=x.device)
    extension().run_gemm(x, pack.data, pack.scale, output, pack.rows, 1., 0,
                         pack.rowscale.data_ptr(), 0, 0, 0,
                         pack_rows=pack_rows, short_input=short_input)
    return output


def clone_pack(pack):
    from engine.kernels.dense import W4Pack
    return W4Pack(pack.data.clone(), pack.scale.clone(), pack.rowscale.clone(),
                  pack.rows, pack.cols, pack.calibrated)


def timings(graphs, labels, *, calls, report, **fields):
    from probes.engine_decode_fusions import _time
    for index, label in enumerate(labels[1:], 1):
        samples = []
        for _ in range(2):
            for arm, which in (('B', 0), ('A', index), ('A', index), ('B', 0)):
                samples.append(dict(arm=arm, us=_time(graphs[which], iterations=64)*1000/calls))
        report('decode_batch_timing', candidate=label, samples=samples, calls=calls,
               scope='same-source complete component over distinct packs; not engine throughput',
               **fields)


def dense_check(report, lane):
    from engine.kernels.dense import DenseLinear, extension
    from probes.engine_decode_fusions import _capture
    ext = extension()
    before, mode, state = ext.gemm_input_mode(), ext.gemm_input_cta_mode(), ext.probe_state()
    short = lane == 'short_gemm'
    shapes = [(4096, 2048)] if short else [(6416, 4096), (4096, 4096), (6144, 4096)]
    variants = [('served', {})]
    if short:
        variants += [('short-input', dict(short_input=True))]
    variants += [(f'pack-{rows}', dict(pack_rows=rows, short_input=short)) for rows in (4, 2)]
    torch.manual_seed(91616)
    try:
        ext.set_gemm_input(1); ext.set_input_cta(4); ext.set_gemm2(0)
        # Qualify every selected shape before any timing, including the
        # wider parent row that exposed the previous stride bug.
        for n, k in shapes:
            layer = DenseLinear((torch.randn(n, k, device='cuda')*.02).bfloat16(), prefill=False)
            weight = layer.packs[0]
            for rows in (6, 7):
                parent = torch.full((rows, 5*k+8), float('nan'), dtype=torch.bfloat16, device='cuda')
                x = parent[:, 2*k:3*k]
                x.normal_()
                graphs, outputs = [], []
                try:
                    for _, options in variants:
                        graph, out = _capture(lambda: dense_call(x, weight, **options))
                        graphs.append(graph); outputs.append(out)
                    for scale in (0., .001, 1., 15., 1.):
                        x.normal_().mul_(scale)
                        for order in (range(len(graphs)), reversed(range(len(graphs)))):
                            for index in order:
                                graphs[index].replay()
                            for out in outputs[1:]:
                                torch.testing.assert_close(out, outputs[0], rtol=0, atol=0)
                    report('decode_batch_numerics', lane_name=lane, rows=rows, n=n, k=k,
                           variants=[label for label, _ in variants], exact=True,
                           input_stride=x.stride(0), original_plan=ext.gemm2_plan(rows, n, k))
                finally:
                    for graph in graphs:
                        graph.reset()
        for n, k in shapes:
            layer = DenseLinear((torch.randn(n, k, device='cuda')*.02).bfloat16(), prefill=False)
            packs = [clone_pack(layer.packs[0]) for _ in range(8)]
            for rows in (6, 7):
                x = torch.randn(rows, k, dtype=torch.bfloat16, device='cuda')
                graphs = []
                try:
                    for _, options in variants:
                        graph, _ = _capture(lambda: [dense_call(x, weight, **options) for weight in packs])
                        graphs.append(graph)
                    timings(graphs, [label for label, _ in variants], calls=len(packs),
                            report=report, lane_name=lane, rows=rows, n=n, k=k)
                finally:
                    for graph in graphs:
                        graph.reset()
        report('decode_batch_complete', lane_name=lane, passed=True, gpu=torch.cuda.get_device_name())
    finally:
        ext.set_gemm_input(before); ext.set_input_cta(mode); ext.restore_probe_state(state)


class DirectDown:
    def __init__(self, extension):
        self.extension = extension

    def run_smlp2(self, *args):
        return self.extension.run_smlp2(*args, direct_down=True)


def shared_check(report):
    from engine.kernels.dense import extension
    from probes.engine_decode_fusions import _capture
    from tests.test_engine_shared_mlp import SharedMLPTests
    ext = extension()
    state = ext.probe_state()
    try:
        ext.set_gemm2(0)
        gu, down, owner = SharedMLPTests.layers()
        _, _, other = SharedMLPTests.layers(seed=133)
        for rows in (1, 6, 7, 8):
            x = torch.randn(rows, 4096, dtype=torch.bfloat16, device='cuda')
            graphs, outputs = [], []
            try:
                for direct in (False, True):
                    adapter = DirectDown(ext) if direct else ext
                    with patch('engine.kernels.dense.shared_mlp.extension', return_value=adapter):
                        graph, out = _capture(lambda: (owner(x), other(x), owner(x)))
                    graphs.append(graph); outputs.append(out)
                for scale in (0., .001, 1., 15., 1.):
                    x.normal_().mul_(scale)
                    for order in ((0, 1), (1, 0)):
                        for index in order:
                            graphs[index].replay()
                        for actual, expected in zip(outputs[1], outputs[0]):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(outputs[1][0], outputs[1][2], rtol=0, atol=0)
                report('decode_batch_numerics', lane_name='shared_direct', rows=rows,
                       exact=True, scratch_rearmed=True, original_down_plan=ext.gemm2_plan(rows, 4096, 512))
            finally:
                for graph in graphs:
                    graph.reset()
        # Independent packs exceed L2; the measured unit includes gate-up,
        # SwiGLU publication and down, including their dependency waits.
        owners = []
        for _ in range(16):
            clone = copy.copy(owner)
            clone.gate_up, clone.down = copy.copy(gu), copy.copy(down)
            clone.gate_up.packs = tuple(clone_pack(p) for p in gu.packs)
            clone.down.packs = tuple(clone_pack(p) for p in down.packs)
            owners.append(clone)
        for rows in (6, 7):
            x = torch.randn(rows, 4096, dtype=torch.bfloat16, device='cuda')
            graphs = []
            try:
                for direct in (False, True):
                    adapter = DirectDown(ext) if direct else ext
                    with patch('engine.kernels.dense.shared_mlp.extension', return_value=adapter):
                        graph, _ = _capture(lambda: [layer(x) for layer in owners])
                    graphs.append(graph)
                timings(graphs, ['served', 'direct-down'], calls=len(owners), report=report,
                        lane_name='shared_direct', rows=rows, n=4096, k=512)
            finally:
                for graph in graphs:
                    graph.reset()
        report('decode_batch_complete', lane_name='shared_direct', passed=True,
               gpu=torch.cuda.get_device_name())
    finally:
        ext.restore_probe_state(state)
