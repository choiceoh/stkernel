"""CPU bit-copy execution, source operation inventory or device-free SM121 embedding compilation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def interpret():
    import torch
    import triton
    from engine.kernels.token_embedding import _lookup
    from engine.modules.token_embedding import lookup
    records = []
    for rows, hidden in ((1, 127), (8, 128), (16, 257), (24, 4096), (32, 4096), (65, 129), (257, 127)):
        pitch = hidden + 8
        raw = (torch.arange(17*pitch) * 257).to(torch.int16).view(17, pitch)
        weight = raw[:, 4:4+hidden]
        before = raw.clone()
        for rank in range(4):
            start = rank*38720
            held_ids = torch.zeros(rows*2, dtype=torch.int64)
            ids = held_ids[::2]
            ids.copy_(start + torch.arange(rows) % 21 - 2)
            if rows > 1: ids[-1] = -(1 << 63)
            storage = torch.full((rows*hidden+32,), 12345, dtype=torch.int16)
            out = storage[16:-16].view(rows, hidden)
            # The actual kernel reads/stores uint16. Raw integer tensors avoid the
            # interpreter's BF16-to-FP32 input adaptation and exercise those same bytes.
            _lookup[(rows, triton.cdiv(hidden, 256))](ids, weight, out, ids.stride(0), pitch, 17, hidden,
                                                     start, 256, num_warps=4)
            expected = lookup(ids, weight.view(torch.bfloat16), start).view(torch.int16)
            assert torch.equal(out, expected)
            assert torch.equal(raw, before)
            assert (storage[:16] == 12345).all() and (storage[-16:] == 12345).all()
            records.append(dict(rows=rows, hidden=hidden, rank=rank, weight_stride=pitch, exact=True))
    return records


def compile_native():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.token_embedding import _lookup
    records = []
    for hidden, stride, vocab in ((127, 135, 17), (257, 265, 17), (4096, 4096, 38720), (4096, 4104, 38720)):
        for rank in (0, 3):
            src = ASTSource(_lookup, dict(IDS='*i64', W='*bf16', OUT='*bf16'),
                            constexprs=dict(IS=2, WS=stride, V=vocab, H=hidden, START=rank*vocab, B=256))
            kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
            records.append(dict(hidden=hidden, weight_stride=stride, vocab=vocab, rank=rank,
                                shared_bytes=kernel.metadata.shared,
                                cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    return records


def inventory():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from engine.modules.token_embedding import lookup
    class Inventory(TorchDispatchMode):
        def __init__(self): self.ops = Counter()
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if all(result.alias_info is None for result in func._schema.returns): self.ops[str(func)] += 1
            return func(*args, **(kwargs or {}))
    weight = torch.zeros(17, 4096, dtype=torch.bfloat16)
    records = []
    for rows in (8, 32, 2304):
        ids = torch.arange(rows)
        with Inventory() as inv: lookup(ids, weight, 17)
        records.append(dict(rows=rows, hidden=4096, reference_materializing_tensor_ops=sum(inv.ops.values()),
                            operations=dict(sorted(inv.ops.items())), candidate_kernel_launches=1))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('interpreter', 'compile', 'inventory'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('requires CUDA and NVIDIA devices hidden')
    if (os.environ.get('TRITON_INTERPRET') == '1') != (args.mode == 'interpreter'):
        raise RuntimeError('TRITON_INTERPRET=1 is required only for interpreter mode')
    import torch
    torch.set_num_threads(1)
    cases = {'interpreter': interpret, 'compile': compile_native, 'inventory': inventory}[args.mode]()
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', mode=args.mode, gpu_used=False, cases=cases,
                  scope='CPU byte identity, native compile or tensor-op inventory; GPU execution and speed pending',
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                 ('engine/kernels/token_embedding.py', 'engine/modules/token_embedding.py',
                                  'engine/profiles/glm53/net.py', 'probes/engine_token_embedding_check.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
