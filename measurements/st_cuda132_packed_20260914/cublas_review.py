"""CPU feasibility proof: preserve ST's FP8 scales in the cuBLASLt MX layout.

This neither calls cuBLAS nor estimates GEMM latency. The shape budgets count
payload bytes, not cache transactions or bandwidth. No activation is requantized.
"""
import argparse
import json
import math
from pathlib import Path
import random


def exponent_byte(value):
    mantissa, exponent = math.frexp(value)
    code = exponent + 126
    if mantissa != .5 or not 0 <= code < 255:
        raise ValueError('MX scale must be a positive representable power of two')
    return code


def scale_offset(row, group128, groups):
    # One 32-bit store repeats a group-128 exponent over its four MX groups.
    return ((row // 128 * groups + group128) * 512
            + row % 32 * 16 + row % 128 // 32 * 4)


def activation_layout(scales):
    rows, groups = len(scales), len(scales[0])
    result = bytearray([127]) * (((rows + 127) // 128) * groups * 512)
    for row, values in enumerate(scales):
        for group, value in enumerate(values):
            at = scale_offset(row, group, groups)
            result[at:at+4] = bytes([exponent_byte(value)]) * 4
    return result


def weight_layout(scales):
    # ST's weight scale is shared by a whole 128x128 block. Every exponent
    # in the corresponding cuBLAS 128x4 scale tile is therefore identical.
    return b''.join(bytes([exponent_byte(value)]) * 512 for row in scales for value in row)


def layout_proof():
    for exponent in range(-127, 128):
        value = math.ldexp(1., exponent)
        assert math.ldexp(1., exponent_byte(value) - 127) == value
    for value in (0., -1., 1.5, math.inf, math.nan, math.ldexp(1., -128)):
        try:
            exponent_byte(value)
        except ValueError:
            pass
        else:
            raise AssertionError(('invalid scale accepted', value))
    rng = random.Random(132)
    rows, groups = 257, 9
    scales = [[math.ldexp(1., rng.randrange(-127, 128)) for _ in range(groups)] for _ in range(rows)]
    packed = activation_layout(scales)
    weights = scales[:3]
    packed_weights = weight_layout(weights)
    seen = set()
    for row in range(384):
        for inner in range(groups * 4):
            # NVIDIA's independent 128x4 tile formula, then decode E8M0.
            at = (row // 128 * groups + inner // 4) * 512
            at += (row % 32) * 16 + ((row % 128) // 32) * 4 + inner % 4
            assert at not in seen
            seen.add(at)
            expected = scales[row][inner // 4] if row < rows else 1.
            assert math.ldexp(1., packed[at] - 127) == expected
            assert math.ldexp(1., packed_weights[at] - 127) == weights[row // 128][inner // 4]
    assert seen == set(range(len(packed)))
    return dict(status='PASS', exponent_values=255, activation_rows=rows,
                padded_rows=384, groups128=groups, checked_scale_bytes=len(seen),
                scope='CPU scale-value/layout proof; no GPU numerics or performance')


def budget(m, n, k):
    assert min(m, n, k) > 0 and n % 128 == k % 128 == 0
    weight_old = n * k + n * k // 4096
    weight_mx = n * k + n * k // 32
    scales_old = m * k // 32
    scales_mx = ((m + 127) // 128 * 128) * k // 32
    return dict(m=m, n=n, k=k, fp8_weight_bytes=weight_old,
                mxfp8_weight_bytes=weight_mx,
                extra_resident_weight_scale_bytes=weight_mx-weight_old,
                activation_scale_bytes=scales_old, mx_activation_scale_bytes=scales_mx,
                extra_activation_scale_bytes=scales_mx-scales_old,
                w4_weight_bytes=n*k//2+n*k//16+4*n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(gpu_used=False, proof=layout_proof(),
                  budgets=[budget(m, n, k) for m in (1, 4, 7, 8, 28, 32, 1024, 8192, 32256)
                           for n, k in ((6144,4096), (6528,4096), (4096,4096),
                                        (4096,2048), (1024,4096), (4096,512),
                                        (38784,4096), (4096,20480))])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['proof']))
