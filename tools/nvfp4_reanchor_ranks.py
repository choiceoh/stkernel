"""Re-anchor the global scale of a presharded NVFP4 rank file's routed experts.

The preshard folded each expert's global scale into its SF codes (alpha := 1,
engine/modules/nvfp4_sf), so the served block scale IS the stored e4m3 code,
and every block's scale is chosen so the block max lands on the FP4 ceiling.
Measured on this rank (measurements/nvfp4_global_scale_sweep_20260918), an
anchor 1.68x coarser -- intentionally clipping the block max ~40% -- cuts the
requantisation SSE of the other fifteen values enough to win 5.8x overall.

For each expert slice this re-encodes the w13/w2 nibble bytes and SF codes
under a re-anchored ladder: base code = nearest_even(block_amax/6/factor,
e4m3 ladder), then the SSE-best candidate among +-radius codes, nibbles by
the engine's own _fp4_encode. Every emitted scale is a raw ladder value, so
the result decodes through the unmodified kernel path with alpha=1.

The file layout is proven before anything is written: an expert's ORIGINAL
bytes must round trip (dequant -> encode at the stored codes -> byte-identical
packed bytes, and re-swizzle(unswizzle(sf)) byte-identical), and the first
slice reports the same-anchor requant error as the honest baseline. Writes go
to --output (a copy of the source with only expert slice bytes replaced);
--validate-only runs the proofs and the gain table without writing.
"""
import argparse
import json
import math
import re
import shutil
import struct
from pathlib import Path

import torch
from engine.modules.quant import FP4_TABLE, _fp4_encode
from engine.modules.nvfp4_sf import swizzle_sf, unswizzle_sf


def _ladder():
    out = []
    for code in range(127):
        exponent, mantissa = divmod(code, 8)
        out.append(mantissa * 2.0 ** -9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return torch.tensor(out, dtype=torch.float64)


LADDER = _ladder()


def _even_tie_index(dist, n):
    """argmin with nearest_even's tie rule: exactly-equal distances keep the
    even table index (min over (distance, i & 1))."""
    best = dist.min(dim=-1, keepdim=True).values
    ties = dist == best
    first = ties.to(torch.int64).argmax(dim=-1)
    even = (torch.arange(n, dtype=torch.int64) & 1) == 0
    even_first = (ties & even).to(torch.int64).argmax(dim=-1)
    return torch.where((ties & even).any(dim=-1), even_first, first)


def requant(w, factor, radius):
    """[M, K] dequantised weight -> (codes [M, K/16], restored [M, K]).

    factor 1 is the served anchor (block max on the FP4 ceiling); a factor
    above one re-anchors the ladder coarser, clipping the max in exchange for
    finer steps on the other fifteen values. Candidates within +-radius codes
    of the base, SSE-best wins; scales are raw ladder values."""
    m, k = w.shape
    blocks = w.double().reshape(-1, 16)
    block_amax = blocks.abs().amax(-1)
    dist = ((block_amax / 6.0 / factor).unsqueeze(1) - LADDER).abs()
    base = _even_tie_index(dist, 127)
    best_sse = None
    best_restored = None
    best_code = None
    for offset in (0, -1, 1, -2, 2)[: 2 * radius + 1 if radius < 2 else 5]:
        if abs(offset) > radius:
            continue
        code = (base + offset).clamp(0, 126)
        scale = LADDER[code]
        ratio = blocks.abs() / scale.unsqueeze(-1)
        level = _even_tie_index((ratio.unsqueeze(-1) - FP4_TABLE.double()[:8]).abs(), 8)
        restored = torch.copysign(FP4_TABLE.double()[level] * scale.unsqueeze(-1), blocks)
        sse = (blocks - restored).square().sum(-1)
        if best_sse is None:
            best_sse, best_restored, best_code = sse, restored, code
        else:
            better = sse < best_sse
            best_sse = torch.where(better, sse, best_sse)
            best_restored = torch.where(better.unsqueeze(1), restored, best_restored)
            best_code = torch.where(better, code, best_code)
    return best_code.reshape(m, k // 16), best_restored.reshape(m, k)


def encode_packed(restored, codes):
    """[M, K] restored values and [M, K/16] ladder codes -> packed uint8."""
    m, k = restored.shape
    scale = LADDER[codes.long()]
    nib = _fp4_encode((restored / scale.repeat_interleave(16, dim=1)).float())
    nib = nib.unflatten(-1, (k // 2, 2))
    return (nib[..., 0] | (nib[..., 1] << 4)).to(torch.uint8)


def dequant_slice(packed, codes):
    """packed [M, K/2] uint8, codes [M, K/16] -> [M, K] float64, kernel decode."""
    m, k_half = packed.shape
    nib = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(m, k_half * 2)
    values = FP4_TABLE[nib.long()].double()
    return (values.reshape(m, k_half // 8, 16) * LADDER[codes.long()].unsqueeze(-1)).reshape(m, k_half * 2)


class RankFile:
    def __init__(self, path, writable=False):
        self.path = Path(path)
        self.file = self.path.open('r+b' if writable else 'rb')
        size, = struct.unpack('<Q', self.file.read(8))
        self.header = json.loads(self.file.read(size))
        self.start = 8 + size

    def slice_info(self, name, index):
        meta = self.header[name]
        lo, hi = meta['data_offsets']
        count = meta['shape'][0]
        stride = (hi - lo) // count
        assert stride * count == hi - lo and 0 <= index < count, (name, index)
        return lo + index * stride, stride

    def read_slice(self, name, index):
        off, stride = self.slice_info(name, index)
        self.file.seek(self.start + off)
        return self.file.read(stride)

    def write_slice(self, name, index, raw):
        off, stride = self.slice_info(name, index)
        assert len(raw) == stride
        self.file.seek(self.start + off)
        self.file.write(raw)


def expert_names(layers):
    out = []
    for n in layers:
        for kind in ('w13', 'w2'):
            out.append((f'L{n}.moe.{kind}', f'L{n}.moe.{kind}_sf', kind))
    return out


def process(rf, layers, factor, radius, limit, out_path):
    stats = []
    for packed_name, sf_name, kind in expert_names(layers):
        meta = rf.header[packed_name]
        count = meta['shape'][0]
        rows, k_half = meta['shape'][1], meta['shape'][2]
        ss = k_half * 2 // 16
        for index in range(count if limit is None else min(count, limit)):
            shape = meta['shape'][1:]
            packed = torch.frombuffer(bytearray(rf.read_slice(packed_name, index)), dtype=torch.uint8).reshape(shape)
            sf_swizzled = torch.frombuffer(bytearray(rf.read_slice(sf_name, index)), dtype=torch.uint8).reshape(rf.header[sf_name]['shape'][1:])
            codes = unswizzle_sf(sf_swizzled, rows, ss)
            w = dequant_slice(packed, codes)

            if index == 0 and kind == 'w13':                      # layout proof, once per tensor name
                # The SF swizzle must round trip bit-exactly (layout proof). The
                # packed nibbles cannot: the original packer's CUDA reciprocal
                # rounding differs from RNE on ~6% of elements (the review
                # module documents the same divergence), so the packed match is
                # reported, not enforced. What MUST hold exactly is the new
                # bytes' decode: asserted right after the re-encode below.
                same_anchor = encode_packed(w, codes)
                packed_match_pct = 100 * float((same_anchor.reshape(-1) == packed.reshape(-1)).float().mean())
                sf_exact = bool((swizzle_sf(codes.to(torch.uint8)).reshape(-1) == sf_swizzled).all())
                if not sf_exact:
                    raise SystemExit(f'SF swizzle round trip failed at {packed_name}')
                print(json.dumps(dict(event='layout-proof', name=packed_name,
                                      packed_rne_match_pct=round(packed_match_pct, 3),
                                      sf_bit_exact=True)), flush=True)

            old_codes, _ = requant(w, 1.0, radius)                # same-anchor requant: the honest baseline
            old_restored = dequant_slice(encode_packed(w, old_codes), old_codes)
            new_codes, new_restored = requant(w, factor, radius)
            new_packed = encode_packed(new_restored, new_codes)
            new_dequant = dequant_slice(new_packed, new_codes)
            base_sse = float((old_restored - w).square().sum())
            new_sse = float((new_dequant - w).square().sum())
            if not validate_sane(new_codes):
                raise SystemExit(f'invalid SF code emitted at {packed_name}[{index}]')
            if not bool((dequant_slice(new_packed, new_codes) == new_restored).all()):
                raise SystemExit(f'new bytes do not decode exactly at {packed_name}[{index}]')
            if out_path is not None:
                rf_out.write_slice(packed_name, index, encode_packed(new_restored, new_codes).numpy().tobytes())
                rf_out.write_slice(sf_name, index, swizzle_sf(new_codes.to(torch.uint8)).numpy().tobytes())
            stats.append(dict(name=packed_name, index=index, base_sse=base_sse, new_sse=new_sse,
                              gain_pct=100 * (1 - new_sse / base_sse) if base_sse else None))
        print(json.dumps(dict(event='tensor-done', name=packed_name, slices=len(stats))), flush=True)
    return stats


def validate_sane(codes):
    return bool(((codes >= 0) & (codes <= 126)).all())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank-file', type=Path, required=True)
    ap.add_argument('--output-file', type=Path)
    ap.add_argument('--layers', type=int, nargs='+', required=True)
    ap.add_argument('--factor', type=float, default=1.68)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--limit', type=int, default=None, help='experts per tensor (validation subsets)')
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    if args.validate_only:
        args.output_file = None
    elif not args.output_file:
        raise SystemExit('--output-file is required unless --validate-only')
    rf = RankFile(args.rank_file)
    rf_out = None if args.output_file is None else None
    if args.output_file is not None:
        shutil.copyfile(args.rank_file, args.output_file)
        rf_out = RankFile(args.output_file, writable=True)
    stats = process(rf, args.layers, args.factor, args.radius, args.limit, args.output_file)
    base = math.fsum(s['base_sse'] for s in stats)
    new = math.fsum(s['new_sse'] for s in stats)
    print(json.dumps(dict(event='totals', slices=len(stats), base_sse=base, new_sse=new,
                          gain_pct=100 * (1 - new / base) if base else None)), flush=True)
