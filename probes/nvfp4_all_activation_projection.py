"""CPU activation/projection ablation on real weights and synthetic BF16 inputs.

The reference keeps the SAME dequantized checkpoint weights and BF16 SiLU output,
but omits FP4 activation quantization. This isolates activation error; it is not
error against the original unquantized model, native CUDA arithmetic, or prompts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import torch
from engine.modules.moe import dequant_nvfp4
from engine.modules.nvfp4_sf import unswizzle_sf
from probes.nvfp4_scale_search_review import compare_block


class Rank:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('rb')
        size, = struct.unpack('<Q', self.file.read(8))
        header = self.file.read(size)
        self.header = json.loads(header)
        self.header_sha256 = hashlib.sha256(header).hexdigest()
        self.start = 8 + size

    def expert(self, name, index):
        meta = self.header[name]
        lo, hi = meta['data_offsets']
        count = meta['shape'][0]
        stride = (hi-lo)//count
        assert stride*count == hi-lo and 0 <= index < count
        self.file.seek(self.start + lo + index*stride)
        raw = bytearray(self.file.read(stride))
        assert len(raw) == stride
        dtype = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'F32': torch.float32}[meta['dtype']]
        return torch.frombuffer(raw, dtype=dtype).reshape(meta['shape'][1:]), hashlib.sha256(raw).hexdigest()


def quant_pair(x, scale):
    pairs = [compare_block(v, float(scale), radius=1) for v in x.reshape(-1, 16).tolist()]
    assert all(b['sse'] <= a['sse'] for a, b in pairs)
    return tuple(torch.tensor([pair[i]['restored'] for pair in pairs]).reshape_as(x) for i in (0, 1))


def activate(fc1):
    # Rank files are explicitly up|gate, as consumed by b12x, not gate|up.
    up, gate = fc1.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16().float()


def down(x, w):
    # Preserve the reference's BF16 round trip for each K128 contribution.
    return sum((x[:, k:k+128] @ w[:, k:k+128].T).bfloat16().float()
               for k in range(0, x.shape[1], 128))


def sse(x, reference):
    return float((x.double()-reference.double()).square().sum())


def cell(rank, layer, expert, dense, samples):
    prefix = f'L{layer}.' + ('mlp.' if dense else 'moe.')
    tensors, hashes = {}, {}
    for name in ('w13', 'w13_sf', 'w2', 'w2_sf'):
        tensors[name], hashes[name] = rank.expert(prefix+name, expert)
    scales = {}
    for name in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale'):
        if prefix+name in rank.header:
            scales[name], hashes[name] = rank.expert(prefix+name, expert)
        else:
            scales[name] = torch.tensor(1.)
    n, k = tensors['w2'].shape[1]*2, tensors['w13'].shape[1]*2
    w13 = dequant_nvfp4(tensors['w13'], unswizzle_sf(tensors['w13_sf'], 2*n, k//16), scales['w13_alpha'])
    w2 = dequant_nvfp4(tensors['w2'], unswizzle_sf(tensors['w2_sf'], k, n//16), scales['w2_alpha'])
    seed = 917 + layer*1000 + expert
    torch.manual_seed(seed)
    x = torch.randn(samples, k).bfloat16().float()
    x0, x1 = quant_pair(x, scales['a13_scale'])
    fc1_ref = x @ w13.T
    fc1_0, fc1_1 = x0 @ w13.T, x1 @ w13.T
    a_ref, a0, a1 = (activate(v) for v in (fc1_ref, fc1_0, fc1_1))
    # FC2 local ablation uses identical activation values in both arms.
    a00, a01 = quant_pair(a0, scales['a2_scale'])
    _, a11 = quant_pair(a1, scales['a2_scale'])
    local_ref = down(a0, w2)
    full_ref = down(a_ref, w2)
    y00, y01, y11 = (down(v, w2) for v in (a00, a01, a11))
    # Existing ss1 already searches routed FC2, but does not touch dense MLPs.
    current = y00 if dense else y01
    comparisons = {
        'fc1_activation': [sse(x0, x), sse(x1, x)],
        'fc1_projection': [sse(fc1_0, fc1_ref), sse(fc1_1, fc1_ref)],
        'fc2_activation_isolated': [sse(a00, a0), sse(a01, a0)],
        'fc2_projection_isolated': [sse(y00, local_ref), sse(y01, local_ref)],
        'full_mlp_current_ss1_to_as1': [sse(current, full_ref), sse(y11, full_ref)],
        'full_mlp_no_search_to_as1': [sse(y00, full_ref), sse(y11, full_ref)],
    }
    return dict(layer=layer, expert=expert, kind='modelopt_dense' if dense else 'routed',
                seed=seed, samples=samples, shape=[k,n], input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                weight_sha256=hashes, comparisons=comparisons)


def run(args):
    torch.set_num_threads(2)
    rows, ranks = [], []
    for path, dense in ((args.routed_ranks, False), (args.dense_ranks, True)):
        if path is None:
            continue
        rank = Rank(path)
        ranks.append(dict(path=str(path), header_sha256=rank.header_sha256,
                          metadata=rank.header.get('__metadata__', {})))
        cells = [(l, 0) for l in (0,1,2)] if dense else [(l,e) for l in (3,20,40) for e in (0,73,287)]
        try:
            for layer, expert in cells:
                row = cell(rank, layer, expert, dense, args.samples)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            rank.file.close()
    totals = {}
    for kind in sorted({r['kind'] for r in rows}):
        group = [r for r in rows if r['kind']==kind]
        totals[kind] = {}
        for metric in group[0]['comparisons']:
            old, new = [math.fsum(r['comparisons'][metric][i] for r in group) for i in (0,1)]
            totals[kind][metric] = dict(baseline_sse=old, candidate_sse=new,
                reduction_pct=100*(1-new/old) if old else None,
                improved_cells=sum(r['comparisons'][metric][1] < r['comparisons'][metric][0] for r in group),
                total_cells=len(group))
    result = dict(scope=__doc__, gpu_used=False, torch=torch.__version__, ranks=ranks,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows, totals=totals)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(totals=totals)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--routed-ranks', type=Path, required=True)
    ap.add_argument('--dense-ranks', type=Path)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--output', type=Path, required=True)
    run(ap.parse_args())
