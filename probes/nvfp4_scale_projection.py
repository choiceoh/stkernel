"""CPU real-weight projection study; synthetic inputs, not answer-quality proof.

Read only individual experts from the exact rank file. Keep FC1 input
quantization, both weight matrices and activation rounding fixed, then compare
FC2 output error against the same unquantized activation. No GPU is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import torch
from engine.modules.moe import dequant_nvfp4, quant_nvfp4_act, dequant_nvfp4_act
from engine.modules.nvfp4_sf import unswizzle_sf
from probes.nvfp4_scale_search_review import compare_block


def run(ranks, output):
    torch.set_num_threads(2)
    torch.manual_seed(917)
    with Path(ranks).open('rb') as f:
        size, = struct.unpack('<Q', f.read(8))
        header_bytes = f.read(size)
        header = json.loads(header_bytes)
        start = 8 + size
        def expert(name, index):
            meta = header[name]
            lo, hi = meta['data_offsets']
            count = meta['shape'][0]
            stride = (hi - lo) // count
            assert stride * count == hi - lo
            f.seek(start + lo + index * stride)
            data = bytearray(f.read(stride))
            assert len(data) == stride
            dtype = {'U8': torch.uint8, 'F8_E4M3': torch.float8_e4m3fn, 'F32': torch.float32}[meta['dtype']]
            value = torch.frombuffer(data, dtype=dtype).reshape(meta['shape'][1:])
            return value, hashlib.sha256(data).hexdigest()
        rows = []
        for layer in (3, 20, 40):
            for index in (0, 73, 287):
                p = f'L{layer}.moe.'
                tensors, hashes = {}, {}
                for name in ('w13', 'w13_sf', 'w2', 'w2_sf'):
                    tensors[name], hashes[name] = expert(p + name, index)
                scales = {}
                for name in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale'):
                    scales[name] = expert(p + name, index)[0] if p + name in header else torch.tensor(1.)
                w13 = dequant_nvfp4(tensors['w13'], unswizzle_sf(tensors['w13_sf'], 1024, 256), scales['w13_alpha'])
                w2 = dequant_nvfp4(tensors['w2'], unswizzle_sf(tensors['w2_sf'], 4096, 32), scales['w2_alpha'])
                x = torch.randn(16, 4096).bfloat16()
                packed, sf = quant_nvfp4_act(x, scales['a13_scale'])
                xq = dequant_nvfp4_act(packed, sf, scales['a13_scale'])
                gate, up = (xq @ w13.T).chunk(2, dim=-1)
                activation = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16().float()
                assert bool(torch.isfinite(activation).all())
                reference = activation @ w2.T
                data = activation.reshape(-1, 16).tolist()
                results = {}
                for radius in (0, 1, 2):
                    restored = torch.tensor([compare_block(v, float(scales['a2_scale']), radius)[1]['restored']
                                             for v in data]).reshape_as(activation)
                    got = restored @ w2.T
                    # Also retain the existing per-128 down-projection BF16 round trip.
                    sliced = sum((restored[:, k:k+128] @ w2[:, k:k+128].T).bfloat16().float()
                                 for k in range(0, 512, 128))
                    results[str(radius)] = dict(
                        activation_sse=float((restored.double() - activation.double()).square().sum()),
                        projection_sse=float((got.double() - reference.double()).square().sum()),
                        sliced_projection_sse=float((sliced.double() - reference.double()).square().sum()))
                row = dict(layer=layer, expert=index, weight_sha256=hashes, results=results)
                rows.append(row)
                print(json.dumps(row), flush=True)
    totals = {str(r): {key: math.fsum(row['results'][str(r)][key] for row in rows)
                      for key in ('activation_sse', 'projection_sse', 'sliced_projection_sse')}
              for r in (0, 1, 2)}
    result = dict(scope='CPU real rank weights, synthetic BF16 input; not native kernel or answer quality',
                  torch=torch.__version__, ranks=str(ranks), rank_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows, totals=totals)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(totals=totals)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    run(args.ranks, args.output)
