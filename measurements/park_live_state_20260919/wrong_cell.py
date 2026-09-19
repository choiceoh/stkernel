"""Sensitivity of tests/test_engine_park_live_state.KdaRingTests: the real KDA ring kernel on a ring resumed with the
live cell gives the parked ring's step; resumed with the wrong cell, or none, it does not.

From the repo root (stk-test):
    TRITON_INTERPRET=1 python3 measurements/park_live_state_20260919/wrong_cell.py
"""
import sys
from dataclasses import replace

sys.path.insert(0, ".")
import torch  # noqa: E402

from engine.base import kernel_shape as ks  # noqa: E402
from engine.base.kernel_shape import MEASURED, LinearAttention  # noqa: E402
from engine.kernels.kda.ring import recurrent_kda_ring  # noqa: E402
from tests.test_engine_kernel_glue import kda_kernels  # noqa: E402

torch.manual_seed(1)
h, kd, cells, t, context = 2, 16, 8, 1, 13
ks.reset()
ks.bind(replace(MEASURED, linear=LinearAttention(heads=h, v_heads=h, k_dim=kd, v_dim=kd, conv=4)))
g = lambda *shape: torch.randn(*shape)  # noqa: E731
q, k, v, raw, beta, a_log, bias = g(1, t, h, kd), g(1, t, h, kd), g(1, t, h, kd), g(1, t, h, kd), g(1, t, h), g(h) * .2, g(h * kd) * .1
parked = torch.randn(3, cells, h, kd, kd) * .1
live = (context - 1) % cells
for name, keep in (("the live cell", live), ("the cell after it", context % cells), ("no cell", None)):
    resumed = torch.zeros_like(parked)
    if keep is not None:
        resumed[1, keep] = parked[1, keep]
    with kda_kernels():
        want = recurrent_kda_ring(q, k, v, raw, beta, a_log, bias, parked.clone(), 1, context, -5.0)
        got = recurrent_kda_ring(q, k, v, raw, beta, a_log, bias, resumed, 1, context, -5.0)
    print(f"resumed with {name}: output equal = {torch.equal(got, want)}")
