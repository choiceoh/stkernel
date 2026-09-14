"""Interpret native packed-weight readers and W4A8 activation quantization on CPU."""
import argparse
import json
import os
from pathlib import Path

if os.environ.get("TRITON_INTERPRET") != "1":
    raise RuntimeError("set TRITON_INTERPRET=1; this is a CPU-only address probe")

import torch
import triton
import triton.language as tl

from engine.kernels.tile_dataflow import _scale, _weight, _w4a8_weight, _w4a8_scaled_input
from engine.modules.expert_layout import row_major_expert, W13_K_IN_BYTES, W2_K_IN_BYTES
from tests.test_engine_nvfp4_dataflow import packed_weights
from tests.test_engine_w4a8_dataflow import pack
from engine.kernels.dense.packing import mk_w4_dequant, _mk_quant_x_ref


@triton.jit
def _addresses(W, S, O, OS, ROWS: tl.constexpr, K: tl.constexpr,
               TILE: tl.constexpr, SF6: tl.constexpr, FC2: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0)*B + tl.arange(0, B)
    values = _weight(W, idx//(K//2), idx % (K//2), ROWS, K, TILE)
    tl.store(O+idx, values, idx < ROWS*K//2)
    scale = _scale(S, idx//(K//16), idx % (K//16), ROWS, K, SF6, FC2)
    tl.store(OS+idx, scale.to(tl.uint8, bitcast=True), idx < ROWS*K//16)


@triton.jit
def _w4a8_addresses(W, S, O, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0)*B + tl.arange(0, B)
    w = _w4a8_weight(W, S, idx//K, idx%K, N, K)
    tl.store(O+idx, w.to(tl.uint8, bitcast=True), idx < N*K)


@triton.jit
def _w4a8_activation(X, O, OS, M: tl.constexpr, K: tl.constexpr):
    r = tl.arange(0, 32)
    c = tl.program_id(0)*128 + tl.arange(0, 128)
    x = tl.load(X+r[:, None]*K+c[None, :], r[:, None] < M, other=0)
    q, s = _w4a8_scaled_input(x)
    tl.store(O+r[:, None]*K+c[None, :], q, r[:, None] < M)
    tl.store(OS+r*(K//128)+tl.program_id(0), s, r < M)


def main(output):
    torch.set_num_threads(1)
    records = []
    for tiled, sf6 in ((False, False), (True, False), (True, True)):
        weights = packed_weights(tiled=tiled, sf6=sf6)
        for second in (False, True):
            w, s = (weights.w2, weights.sf2) if second else (weights.w13, weights.sf13)
            rows, k = w.shape[1], w.shape[2]*2
            tile = weights.tile2 if second else weights.tile13
            actual = torch.empty((rows, k//2), dtype=torch.uint8)
            actual_sf = torch.empty((rows, k//16), dtype=torch.uint8)
            _addresses[(triton.cdiv(actual.numel(), 256),)](w, s, actual, actual_sf, rows, k, tile, sf6, second, 256)
            expected = row_major_expert(w, 0, W2_K_IN_BYTES if second else W13_K_IN_BYTES)
            assert torch.equal(actual, expected), "native packed-weight address mismatch"
            assert torch.equal(actual_sf, weights.raw_scales_cpu(second=second).view(torch.uint8)), "native scale address mismatch"
            records.append(dict(tiled=tiled, sf6=sf6, plane="fc2" if second else "fc1",
                                packed_bytes=actual.numel(), scale_bytes=actual_sf.numel(), exact=True))
    for n, k in ((384, 256), (256, 384), (129, 128)):
        p = pack(n, k)
        actual = torch.empty(n, k, dtype=torch.uint8)
        _w4a8_addresses[(triton.cdiv(n*k, 256),)](p.data, p.scale, actual, n, k, 256)
        expected = mk_w4_dequant(p.data, p.scale, n).to(torch.float8_e4m3fn).view(torch.uint8)
        # Sign of zero is immaterial to MMA; the native reader retains it.
        assert torch.equal(actual & 127, expected & 127), "native W4A8 expanded-weight magnitude mismatch"
        assert torch.equal(actual[expected & 127 != 0], expected[expected & 127 != 0]), "native W4A8 weight sign mismatch"
        records.append(dict(plane="w4a8-expanded", rows=n, columns=k, exact=True))
    torch.manual_seed(308)
    x = torch.randn(17, 384).bfloat16()
    x[0, :128] = 0
    x[1, :128] *= 1.e-28
    x[2, :128] *= 1.e28
    q, s = torch.empty_like(x, dtype=torch.float32), torch.empty(17, 3)
    _w4a8_activation[(3,)](x, q, s, 17, 384)
    # Triton 3.8's interpreter rounds FP8 halfway cases upward, unlike its
    # generated cvt.rn. Test actual scaling arithmetic here; use torch's RTNE
    # conversion, offline PTX checks, and the optional native device gate.
    actual = (q.to(torch.float8_e4m3fn).float().view(17, 3, 128)*s[:, :, None]).view_as(x)
    assert torch.equal(actual, _mk_quant_x_ref(x)), "native FP8 activation quantization changed"
    records.append(dict(plane="w4a8-activation-scaling", rows=17, columns=384, exact=True,
                        fp8_conversion="torch CPU RTNE; device conversion not executed"))
    assert not torch.cuda.is_initialized()
    report = dict(scope="actual Triton address functions interpreted on CPU", gpu_used=False,
                  cuda_initialized=False, variants=records)
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(records)} native reader/quantization variants match independent CPU references")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
