"""CPU feasibility study of NVFP4 activation scale search; no serving imports.

Compare the usual round_e4m3(amax / (6 * global_scale)) against that code
and its four neighbours. Both arms use the same E2M1 RNE encoder and group
size 16. This is an independent mathematical reference, not a bit-exact
emulation of the served CUDA reciprocal/rounding instructions. It measures
activation reconstruction error, NOT model quality or GPU speed.

Optional --input JSON: {"blocks": [[16 finite values], ...],
"global_scales": [positive finite scale per block]}. Supplying captured
activations avoids treating the synthetic fixtures as model measurements.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import random
import struct

FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
OFFSETS = (0, -1, 1, -2, 2)


def decode_scale(code: int) -> float:
    if not 0 <= code <= 126:
        raise ValueError('scale code must be a nonnegative finite E4M3 code')
    exponent, mantissa = divmod(code, 8)
    return mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0**(exponent - 7)


SCALES = tuple(decode_scale(i) for i in range(127))


def nearest_even(value: float, table: tuple[float, ...]) -> int:
    """Saturating nearest rounding, breaking midpoint ties by the code LSB."""
    hi = bisect.bisect_left(table, value)
    if hi == 0:
        return 0
    if hi == len(table):
        return hi - 1
    lo = hi - 1
    return min((lo, hi), key=lambda i: (abs(table[i] - value), i & 1))


def at_scale(values: tuple[float, ...], code: int, gs: float) -> dict:
    scale = SCALES[code] * gs
    nibbles = tuple((nearest_even(abs(v) / scale, FP4) if scale else 0)
                    | (8 if math.copysign(1.0, v) < 0 else 0) for v in values)
    restored = tuple(math.copysign(FP4[q & 7] * scale, -1.0 if q & 8 else 1.0)
                     for q in nibbles)
    errors = tuple(abs(a - b) for a, b in zip(values, restored))
    return dict(code=code, nibbles=nibbles, restored=restored,
                sse=math.fsum(e * e for e in errors), max_abs=max(errors))


def compare_block(values, global_scale=1.0, radius=2) -> tuple[dict, dict]:
    values = tuple(float(v) for v in values)
    gs = float(global_scale)
    if radius not in (0, 1, 2):
        raise ValueError('search radius must be 0, 1 or 2 scale codes')
    if len(values) != 16 or not all(math.isfinite(v) for v in values):
        raise ValueError('a block must contain exactly 16 finite values')
    if not math.isfinite(gs) or gs <= 0:
        raise ValueError('global scale must be finite and strictly positive')
    normalised = max(abs(v) for v in values) / gs
    if not math.isfinite(normalised) or any(not math.isfinite(s * gs) for s in SCALES):
        raise ValueError('values/global scale exceed this finite reference domain')
    base_code = nearest_even(normalised / 6.0, SCALES)
    base = at_scale(values, base_code, gs)
    best = base
    for offset in OFFSETS[1:]:
        if abs(offset) > radius:
            continue
        code = base_code + offset
        if 0 <= code <= 126:
            candidate = at_scale(values, code, gs)
            if candidate['sse'] < best['sse']:
                best = candidate
    return base, best


def bf16(value: float) -> float:
    bits, = struct.unpack('<I', struct.pack('<f', value))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def fixtures(count: int, seed: int):
    rng = random.Random(seed)
    for kind in ('normal', 'silu_product', 'clamped_silu_product', 'one_outlier'):
        blocks, scales = [], []
        for _ in range(count):
            amplitude = 2.0 ** rng.randint(-6, 6)
            gs = amplitude * 2.0 ** rng.randint(-3, 3)
            if kind == 'normal':
                values = [rng.gauss(0, 1) for _ in range(16)]
            elif kind == 'one_outlier':
                values = [rng.gauss(0, 0.2) for _ in range(16)]
                values[rng.randrange(16)] *= 32
            else:
                values = []
                for _ in range(16):
                    gate, up = rng.gauss(0, 3), rng.gauss(0, 3)
                    if kind == 'clamped_silu_product':
                        gate, up = min(gate, 10), min(max(up, -10), 10)
                    values.append(gate / (1 + math.exp(-gate)) * up)
            blocks.append([bf16(v * amplitude) for v in values])
            scales.append(gs)
        yield kind, blocks, scales


def analyse(blocks, scales) -> dict:
    if not blocks or len(blocks) != len(scales):
        raise ValueError('nonempty blocks and equally many global scales are required')
    base_sse = new_sse = narrow_sse = energy = 0.0
    improved = worse_sse = worse_max = 0
    hist = {str(k): 0 for k in OFFSETS}
    for values, gs in zip(blocks, scales):
        base, candidate = compare_block(values, gs)
        _, narrow = compare_block(values, gs, radius=1)
        base_sse += base['sse']
        new_sse += candidate['sse']
        narrow_sse += narrow['sse']
        energy += math.fsum(float(v) ** 2 for v in values)
        improved += candidate['sse'] < base['sse']
        worse_sse += candidate['sse'] > base['sse']
        worse_max += candidate['max_abs'] > base['max_abs']
        hist[str(candidate['code'] - base['code'])] += 1
    payload = json.dumps(dict(blocks=blocks, global_scales=scales), separators=(',', ':'), allow_nan=False)
    return dict(blocks=len(blocks), input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                baseline_nmse=base_sse / energy if energy else 0,
                search_nmse=new_sse / energy if energy else 0,
                mse_reduction_pct=100 * (1 - new_sse / base_sse) if base_sse else 0,
                three_candidate_nmse=narrow_sse / energy if energy else 0,
                three_candidate_mse_reduction_pct=100 * (1 - narrow_sse / base_sse) if base_sse else 0,
                three_candidate_fraction_of_gain=(base_sse - narrow_sse) / (base_sse - new_sse)
                    if base_sse != new_sse else 1.0,
                improved_blocks=improved, worsened_sse_blocks=worse_sse,
                worsened_max_abs_blocks=worse_max, selected_offset_counts=hist)


def torch_check() -> dict:
    """Independent CPU checks of E4M3 decoding/rounding and E2M1 ties."""
    import torch
    all_codes = torch.arange(127, dtype=torch.uint8)
    actual = all_codes.view(torch.float8_e4m3fn).float().tolist()
    assert actual == list(SCALES)
    midpoints = [(a + b) / 2 for a, b in zip(SCALES, SCALES[1:])]
    got = torch.tensor(midpoints).to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    assert got == [nearest_even(v, SCALES) for v in midpoints]
    from engine.modules.quant import _fp4_encode
    mids = [(a + b) / 2 for a, b in zip(FP4, FP4[1:])]
    got = _fp4_encode(torch.tensor(mids)).tolist()
    assert got == [nearest_even(v, FP4) for v in mids]
    return dict(device='cpu', torch=torch.__version__, scale_codes=127,
                scale_midpoints=126, fp4_midpoints=7, passed=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--blocks', type=int, default=4096, help='synthetic blocks per distribution')
    ap.add_argument('--seed', type=int, default=917)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--torch-check', action='store_true')
    args = ap.parse_args()
    if args.blocks < 1:
        ap.error('--blocks must be positive')
    if args.input:
        data = json.loads(args.input.read_text())
        sets = [('provided', data['blocks'], data['global_scales'])]
    else:
        sets = fixtures(args.blocks, args.seed)
    result = dict(scope='CPU mathematical reference; no model quality or timing claim',
                  input_kind='provided' if args.input else 'synthetic_bf16', seed=args.seed,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  results={name: analyse(blocks, scales) for name, blocks, scales in sets})
    if args.torch_check:
        result['torch_check'] = torch_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
