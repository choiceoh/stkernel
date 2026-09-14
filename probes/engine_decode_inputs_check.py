"""CPU execution of the actual input/draw kernels, or SM121 compilation without a CUDA context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def interpreter():
    import torch
    import triton
    from engine.base import draws
    from engine.modules.draft_inputs import build
    from engine.kernels.decode_inputs import _draft_inputs, _step_block
    torch.set_num_threads(1)
    records = []
    for n in (1, 2, 3, 4):
        anchors = torch.tensor([0, 9, 154879, 8, -1, 7, (1 << 63)-1, 6])[::2][:n]
        positions = torch.tensor([0, 9, 128*1024, 8, (1 << 63)-3, 7, -(1 << 63), 6])[::2][:n]
        for k in (0, 1, 5, 7, 31):
            t, guard = k+1, -987654321
            storage = torch.full((2*n*t + 32,), guard, dtype=torch.int64)
            ids, pos = storage[16:-16].view(2, n*t).unbind(0)
            _draft_inputs[(n,)](anchors, positions, ids, pos, anchors.stride(0), positions.stride(0),
                                t, 154879, True, 0, triton.next_power_of_2(t), num_warps=1)
            a, p = build(anchors, positions, k, 154879)
            assert torch.equal(ids, a) and torch.equal(pos, p)
            assert (storage[:16] == guard).all() and (storage[-16:] == guard).all()
            records.append(dict(kernel='draft_inputs', rows=n, k=k, exact=True))
    anchors = torch.tensor([42])
    for position in (0, 128*1024, (1 << 63)-4):
        ids, pos = torch.empty(8, dtype=torch.int64), torch.empty(8, dtype=torch.int64)
        _draft_inputs[(1,)](anchors, anchors, ids, pos, 1, 0, 8, 154879, False, position, 8, num_warps=1)
        a, p = build(anchors, position, 7, 154879)
        assert torch.equal(ids, a) and torch.equal(pos, p)
        records.append(dict(kernel='draft_inputs', rows=1, k=7, host_position=position, exact=True))
    nonces = torch.tensor([-(1 << 63), 9, -1, 8, 0, 7, (1 << 63)-1, 6])[::2]
    gens = torch.tensor([(1 << 63)-1, 9, 0, 8, -1, 7, 128*1024, 6])[::2]
    for seed in (0, -1, 19, (1 << 80)+37):
        for k in (0, 1, 5, 7, 31):
            storage = torch.full((4*(2*k+1) + 32,), float('nan'))
            actual = storage[16:-16].view(4, 2*k+1)
            _step_block[(4,)](nonces, gens, actual, nonces.stride(0), gens.stride(0), draws.mix(seed), k,
                              triton.next_power_of_2(2*k+1), num_warps=1, enable_fp_fusion=False)
            expected = draws.step_block(seed, nonces, gens, k)
            assert torch.equal(actual, expected), (seed, k, actual, expected)
            for row in range(4):
                key = draws.row_key(seed, int(nonces[row]), int(gens[row]))
                assert actual[row].tolist() == [draws.uniform(key, purpose, at) for purpose, at in draws.step_layout(k)]
            assert storage[:16].isnan().all() and storage[-16:].isnan().all()
            records.append(dict(kernel='step_block', rows=4, k=k, seed=seed, exact=True))
    assert not torch.cuda.is_initialized()
    return records


def compile_native():
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.base.draws import mix
    from engine.kernels.decode_inputs import _draft_inputs, _step_block
    records = []
    for t in (1, 8, 32):
        for device in (False, True):
            c = dict(AS=2, PS=3, T=t, MASK=154879, DEVICE_POSITION=device,
                     POSITION=0 if device else (1 << 63)-4, B=triton.next_power_of_2(t))
            src = ASTSource(_draft_inputs, dict(A='*i64', P='*i64', IDS='*i64', POS='*i64'), constexprs=c)
            kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=1))
            records.append(dict(kernel='draft_inputs', tokens=t, device_position=device,
                                shared_bytes=kernel.metadata.shared, cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    for k in (0, 1, 5, 7, 31):
        src = ASTSource(_step_block, dict(NONCE='*i64', GEN='*i64', OUT='*fp32'),
                        constexprs=dict(NS=2, GS=3, SEED_KEY=mix(19), K=k, B=triton.next_power_of_2(2*k+1)))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                                options=dict(num_warps=1, enable_fp_fusion=False))
        records.append(dict(kernel='step_block', k=k, shared_bytes=kernel.metadata.shared,
                            cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    assert not torch.cuda.is_initialized()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('interpreter', 'compile'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('requires CUDA and NVIDIA devices hidden')
    if (os.environ.get('TRITON_INTERPRET') == '1') != (args.mode == 'interpreter'):
        raise RuntimeError('TRITON_INTERPRET=1 is required only for interpreter mode')
    cases = interpreter() if args.mode == 'interpreter' else compile_native()
    report = dict(status='PASS', mode=args.mode, gpu_used=False, cases=cases,
                  scope='CPU bit identity or native compilation; GPU replay and serving speed pending',
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                 ('engine/kernels/decode_inputs.py', 'engine/modules/draft_inputs.py',
                                  'engine/base/draws.py', 'probes/engine_decode_inputs_check.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
