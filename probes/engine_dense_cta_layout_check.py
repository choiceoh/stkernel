"""Same-weight layout correctness and graph replay; no timing or fleet admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu:
        ap.error('explicit --gpu required')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from engine.kernels.dense import build, pack_w4, repack_w4, producer_pack_nbytes, DenseLinear
    from types import SimpleNamespace
    prop = torch.cuda.get_device_properties(0)
    ext = build((prop.major, prop.minor))
    torch.manual_seed(917)
    old = pack_w4((torch.randn(6416, 4096, device='cuda') * .02).bfloat16())
    new = repack_w4(old)
    report = dict(status='RUNNING', device=prop.name, torch=torch.__version__, cuda=torch.version.cuda,
                  scope='same synthetic weight layout, exact outputs and graph replay; performance unmeasured', cells=[],
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                 ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/__init__.py',
                                  'probes/engine_dense_cta_layout_check.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        # Match production's source retirement, including the separate FP8 reader.
        layer = DenseLinear.__new__(DenseLinear)
        layer.packs = (new,)
        fp8 = (torch.randn(6528, 4096, device='cuda').to(torch.float8_e4m3fn),
               torch.ones(51, 32, device='cuda'))
        layer.fp8 = SimpleNamespace(weight=fp8)
        backing = torch.full((6416 * 4096 * 2 + 512,), 83, device='cuda', dtype=torch.uint8)
        arena = backing[256:-256]
        layer.consume_weight(arena)
        relocated = layer.packs[0]
        for before, after in zip((new.data, new.scale, new.rowscale, *fp8),
                                 (relocated.data, relocated.scale, relocated.rowscale, *layer.fp8.weight)):
            assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
            assert arena.data_ptr() <= after.data_ptr() < arena.data_ptr() + arena.numel()
        assert bool((backing[:256] == 83).all() and (backing[-256:] == 83).all())
        new = relocated
        report['arena_relocation'] = dict(bit_exact=True, fp8_unchanged=True, guard_bytes_intact=True,
                                           resident_extra_bytes=0)
        for m in (1, 6, 7, 8, 14, 16, 21, 24, 28, 32):
            # Keep a wider parent stride, as real fused-projection consumers do.
            x = torch.randn(m, 4104, device='cuda', dtype=torch.bfloat16)[:, 4:4100]
            modes = ('plain', 'bound') if m in (8, 16, 24, 32) else ('plain',)
            if m == 8:
                modes += ('producer',)
            if m in (21, 28):
                modes += ('wide',)
            for mode in modes:
                pk = torch.empty(producer_pack_nbytes(8, 4096), device='cuda', dtype=torch.uint8) if mode == 'producer' else None
                outs = [torch.empty(m, 6416, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
                def run(p, out):
                    if mode in ('plain', 'wide'):
                        call = ext.run_gemm_wide_input if mode == 'wide' else ext.run_gemm
                        call(x, p.data, p.scale, out, 6416, 1., 0, p.rowscale.data_ptr(), 0, 0, 0)
                    else:
                        if pk is not None:
                            ext.run_input_pack(x, pk)
                        ext.run_gemm_bound_input(x, p.data, p.scale, out, 6416, p.rowscale.data_ptr(), None, None, producer_pack=pk)
                for scale in (0., .1, 1., 4.):
                    x.normal_().mul_(scale)
                    for p, out in zip((old, new), outs):
                        run(p, out)
                    torch.testing.assert_close(*outs, rtol=0, atol=0)
                    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
                graphs = []
                for p, out in zip((old, new), outs):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run(p, out)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        run(p, out)
                    graphs.append(graph)
                for _ in range(3):
                    x.normal_()
                    graphs[1].replay(); graphs[0].replay()
                    torch.testing.assert_close(*outs, rtol=0, atol=0)
                    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
                for graph in graphs:
                    graph.reset()
                cell = dict(rows=m, mode=mode, bit_exact=True, graph_replays=3)
                report['cells'].append(cell); save(); print(json.dumps(cell), flush=True)
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'], report['error'] = 'FAIL', repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
