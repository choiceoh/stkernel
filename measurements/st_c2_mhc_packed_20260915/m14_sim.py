"""CPU estimate: how far does mHC post move when the [output, hidden, stream] pack is read as [output, stream, hidden]?

Distributions of the 2026-09-13 mhc_batch lane at its first failing replay (rows 14, input scale 0.001). Not the lane's
CUDA random values; the GPU diagnosis replays those.
"""
import json
import sys

import torch

sys.path.insert(0, '/repo')
from engine.modules.hyper_connection import mhc_pre, mhc_post

torch.manual_seed(20260915)
rows, trials, maxima = 14, 200, []
scale, eps = torch.tensor([.2, .3, .4]), (1e-5, 1e-6, 1e-6, 2., 20)
for trial in range(trials):
    fn = (torch.randn(24, 16384) * .006).bfloat16().float()
    permuted = fn.reshape(24, 4, 4096).transpose(1, 2).contiguous().reshape(24, 16384)
    base = torch.randn(24) * .1
    x = (torch.randn(rows, 4096) * .001).bfloat16()
    res = (torch.randn(rows, 4, 4096) * .001).bfloat16()
    post = torch.randn(rows, 4, 1) * .001
    comb = torch.randn(rows, 4, 4) * .001
    rc = mhc_post(x, res, post, comb)
    want = mhc_pre(rc, fn, scale, base, *eps)[0]
    got = mhc_pre(rc, permuted, scale, base, *eps)[0]
    maxima.append((got.double() - want.double()).abs().max().item())
maxima.sort()
recorded = 0.000341951847076416
print(json.dumps(dict(rows=rows, input_scale=.001, trials=trials, median=maxima[trials // 2],
                      p10=maxima[trials // 10], p90=maxima[9 * trials // 10], max=maxima[-1],
                      recorded=recorded, recorded_quantile=sum(m <= recorded for m in maxima) / trials)))
