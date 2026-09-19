"""GatedDeltaNet's gate computed inside the ring kernel (engine/QWEN38_CARRY.md K1).

A captured Qwen3.8 decode step computed each GDN layer's decay and beta in one launch (engine/kernels/gdn.gates: an fp32
decay, a copy of the raw beta logits) and handed them to the ring kernel (kda/ring.recurrent_decay_ring_rows), which
reads the decay through a stride-0 channel axis and sigmoids beta itself. kda/ring.recurrent_gdn_ring(_rows) reads a and
b -- the in_proj columns -- through their strides and computes the same fp32 decay inside the recurrence's launch
(HEAD_GATE in kda/fused_recurrent.py): one launch fewer on each of the 36 GDN layers of a step. The launch's outputs and
ring writes are the two launches' bytes. Both GDN entries round sigmoid to the projection dtype, matching prefill
and the model before the FP32 recurrence; per-channel KDA keeps its existing FP32 sigmoid.

Under TRITON_INTERPRET=1 the inputs are FP32 (the interpreter does not round BF16 as a GPU does) and libdevice's log1p,
which the interpreter cannot call, is numpy's in both launches; on a GPU the same cases run in BF16 at Qwen3.8's per-rank
cell (probes/engine_qwen38_cells runs them).

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_gdn_ring_gate
"""
import ast
import contextlib
import importlib.util
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())


@contextlib.contextmanager
def ring_kernels():
    """tests/test_engine_kernel_glue.kda_kernels, and under the interpreter libdevice's log1p as numpy's in both the
    gates launch and the ring kernel (the interpreter has no extern libdevice call)."""
    from tests.test_engine_kernel_glue import kda_kernels
    with contextlib.ExitStack() as stack:
        stack.enter_context(kda_kernels())
        if INTERPRET:
            import numpy as np
            import triton.language as tl
            from triton.runtime.interpreter import TensorHandle
            from engine.kernels import gdn
            from engine.kernels.kda import fused_recurrent

            def log1p(x):
                return tl.core.tensor(TensorHandle(np.log1p(x.handle.data), x.handle.dtype), x.type)
            for module in (gdn, fused_recurrent):
                stack.enter_context(patch.object(module, "libdevice", SimpleNamespace(log1p=log1p)))
            stack.enter_context(np.errstate(over="ignore", invalid="ignore"))   # numpy computes the branch tl.where drops
        yield


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class GdnRingGateTests(unittest.TestCase):
    """The ring kernel with GDN's gate against the gates launch followed by the decay entry, byte for byte."""

    def setUp(self):
        from engine.base import kernel_shape as ks
        from tests.test_engine_kernel_glue import HEAD_DECAY
        from engine.base.kernel_shape import LinearAttention
        torch.manual_seed(20260917)
        ks.reset()
        self.addCleanup(ks.reset)
        # the interpreter is slow: small widths there, Qwen3.8's per-rank cell on a GPU
        self.h, self.hv, self.kd = (2, 4, 16) if INTERPRET else (4, 12, 128)
        ks.bind(replace(HEAD_DECAY, linear=LinearAttention(heads=self.h, v_heads=self.hv, k_dim=self.kd, v_dim=self.kd,
                                                          conv=4, decay="head")))
        self.dtype = torch.float32 if INTERPRET else torch.bfloat16

    def inputs(self, n):
        """q/k/v [1, n, ...] and the in_proj row [n, qkv | z | b | a] with its b and a columns as views, as net._gdn_rows
        splits it; A_log and dt_bias fp32 [HV]. The a logits reach softplus's threshold 20 on a few heads."""
        g = lambda *shape: torch.randn(*shape, device=DEVICE, dtype=self.dtype)
        qkv = (2 * self.h + self.hv) * self.kd
        proj = torch.randn(n, qkv + self.hv * self.kd + 2 * self.hv, device=DEVICE) * 3
        proj[:, -1] += 21.0                                                # one head's a above the threshold
        proj = proj.to(self.dtype)
        _, _, b, a = proj.split([qkv, self.hv * self.kd, self.hv, self.hv], dim=-1)
        A_log = torch.randn(self.hv, device=DEVICE) * .5
        dt_bias = torch.randn(self.hv, device=DEVICE)
        return g(1, n, self.h, self.kd), g(1, n, self.h, self.kd), g(1, n, self.hv, self.kd), a, b, A_log, dt_bias

    def ring(self, cells=6):
        return torch.randn(3, cells, self.hv, self.kd, self.kd, device=DEVICE) * .1

    def test_the_gate_in_the_ring_is_the_gates_launch_then_the_decay_entry(self):
        from engine.kernels import gdn
        from engine.kernels.kda.ring import recurrent_decay_ring, recurrent_gdn_ring
        for t in (1, 2):
            for packed in (False, True):
                q, k, v, a, b, A_log, dt_bias = self.inputs(t)
                if packed:                                             # contiguous inputs: the unstrided loaders
                    a, b = a.contiguous(), b.contiguous()
                for slot, context in ((0, 0), (1, 1), (2, 8)):
                    with self.subTest(tokens=t, packed=packed, slot=slot, context=context):
                        ring = self.ring()
                        expected = ring.clone()
                        with ring_kernels():
                            decay, beta = gdn.gates(a, b, A_log, dt_bias, sigmoid_beta=False)
                            want = recurrent_decay_ring(q, k, v, decay[None], beta[None], expected, slot, context)
                            got = recurrent_gdn_ring(q, k, v, a[None], b[None], A_log, dt_bias, ring, slot, context)
                        self.assertTrue(torch.equal(got, want))
                        self.assertTrue(torch.equal(ring, expected))

    def test_the_rows_fold_is_the_decay_rows_entry_and_the_one_row_launches(self):
        from engine.kernels import gdn
        from engine.kernels.kda.ring import recurrent_decay_ring_rows, recurrent_gdn_ring, recurrent_gdn_ring_rows
        rows, t = 2, 2
        q, k, v, a, b, A_log, dt_bias = self.inputs(rows * t)
        ring = self.ring()
        decay_ring, one = ring.clone(), ring.clone()
        slots = torch.tensor([2, 0], device=DEVICE, dtype=torch.int32)
        contexts = torch.tensor([5, 0], device=DEVICE, dtype=torch.int32)
        with ring_kernels():
            folded = recurrent_gdn_ring_rows(q, k, v, a[None], b[None], A_log, dt_bias, ring, slots, contexts)
            decay, beta = gdn.gates(a, b, A_log, dt_bias, sigmoid_beta=False)
            via_decay = recurrent_decay_ring_rows(q, k, v, decay[None], beta[None], decay_ring, slots, contexts)
            parts = [recurrent_gdn_ring(*(x[:, i * t:(i + 1) * t] for x in (q, k, v, a[None], b[None])), A_log, dt_bias,
                                        one, int(slots[i]), int(contexts[i])) for i in range(rows)]
        self.assertTrue(torch.equal(folded, via_decay) and torch.equal(ring, decay_ring))
        self.assertTrue(torch.equal(folded, torch.cat(parts, 1)) and torch.equal(ring, one))

    def test_the_ring_is_the_recurrence_on_gdn_s_decay(self):
        from engine.kernels.kda.ring import recurrent_gdn_ring
        from engine.modules.linear_attention import gated_delta_rule, gdn_decay
        t, cells = 2, 6
        q, k, v, a, b, A_log, dt_bias = self.inputs(t)
        ring = self.ring(cells)
        initial = ring[1, 4][None].clone()                                 # context 5 reads cell 4
        with ring_kernels():
            out = recurrent_gdn_ring(q, k, v, a[None], b[None], A_log, dt_bias, ring, 1, 5)
        group = self.hv // self.h
        oracle, final = gated_delta_rule(q.repeat_interleave(group, 2), k.repeat_interleave(group, 2), v,
                                         gdn_decay(a[None], A_log, dt_bias), torch.sigmoid(b[None]), initial,
                                         scale=self.kd ** -0.5, qk_l2norm=True)
        rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30))
        tolerance = 1e-5 if INTERPRET else 1e-2
        self.assertLessEqual(rel(out, oracle), tolerance)
        self.assertLessEqual(rel(ring[1, (5 + t - 1) % cells], final[0]), tolerance)

    def test_bf16_beta_matches_prefill_and_the_model_at_every_saved_position(self):
        """The model sigmoids the BF16 projection BEFORE widening it for the FP32 recurrence. Checking the
        saved states, rather than just a BF16 output at a 1% tolerance, exposes a decode-only gate change."""
        from engine.kernels.kda.ring import recurrent_gdn_ring, recurrent_decay_ring
        from engine.modules.linear_attention import gated_delta_rule, gdn_decay
        from tests.test_engine_qwen38_kernels import served_kernels
        t = 2
        q = torch.full((1, t, self.h, self.kd), .25, dtype=torch.bfloat16, device=DEVICE)
        k = q.clone()
        v = torch.full((1, t, self.hv, self.kd), .75, dtype=torch.bfloat16, device=DEVICE)
        a = torch.zeros(1, t, self.hv, dtype=torch.bfloat16, device=DEVICE)
        b = torch.full_like(a, 1.1)
        A_log = torch.zeros(self.hv, device=DEVICE)
        dt_bias = torch.zeros_like(A_log)
        decay = gdn_decay(a, A_log, dt_bias)
        group = self.hv // self.h
        want, _, states = gated_delta_rule(q.repeat_interleave(group, 2), k.repeat_interleave(group, 2), v,
                                           decay, b.sigmoid(), all_states=True)
        for kind in ("fused", "decay"):
            ring = torch.zeros(1, 4, self.hv, self.kd, self.kd, device=DEVICE)
            with self.subTest(kind=kind), served_kernels(), ring_kernels():
                got = (recurrent_gdn_ring(q, k, v, a, b, A_log, dt_bias, ring, 0, 0) if kind == "fused" else
                       recurrent_decay_ring(q, k, v, decay, b, ring, 0, 0))
            torch.testing.assert_close(ring[0, :t], states[0], rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@unittest.skipUnless(torch is not None and TRITON, "requires torch and the reference lane's Triton imports")
class GdnReferenceGateTests(unittest.TestCase):
    def test_chunk_and_ring_use_the_same_beta_values(self):
        from engine.profiles.qwen38.lanes import reference
        lane = reference()
        q = torch.full((1, 2, 1, 16), .25, dtype=torch.bfloat16)
        v = torch.full((1, 2, 2, 16), .75, dtype=torch.bfloat16)
        a = torch.zeros(2, 2, dtype=torch.bfloat16)
        b = torch.full_like(a, 1.1)
        A_log = torch.zeros(2)
        dt_bias = torch.zeros(2)
        decay, beta = lane.gdn_gates(a, b, A_log, dt_bias, sigmoid_beta=True)
        want, state = lane.gdn_chunk(q, q, v, decay[None], beta[None], None)
        ring = torch.zeros(1, 4, 2, 16, 16)
        got = lane.gdn_ring(q, q, v, a[None], b[None], A_log, dt_bias, ring, 0, 0)
        torch.testing.assert_close(ring[0, 1], state[0], rtol=0, atol=0)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@unittest.skipUnless(torch is not None and TRITON, "the KDA modules import Triton")
class GdnRingRefusalTests(unittest.TestCase):
    """The GDN entries refuse before any launch: a per-channel (KDA) cell, a decay projection that is not per value head,
    gate parameters that are not contiguous FP32 [HV]."""

    def setUp(self):
        from engine.base import kernel_shape as ks
        ks.reset()
        self.addCleanup(ks.reset)

    def test_a_kda_cell_cannot_take_gdn_s_gate_nor_a_gdn_cell_kda_s(self):
        from engine.base import kernel_shape as ks
        from engine.base.kernel_shape import MEASURED
        from engine.kernels.kda import ring
        from tests.test_engine_kernel_glue import HEAD_DECAY
        ks.bind(MEASURED)                                                  # GLM-5.3: KDA's per-channel gate
        with self.assertRaisesRegex(ValueError, "GatedDeltaNet"):
            ring._check_cell(head_gate=True)
        ring._check_cell(False)
        ks.reset()
        ks.bind(HEAD_DECAY)
        with self.assertRaisesRegex(ValueError, "recurrent_gdn_ring"):
            ring._check_cell(False)
        ring._check_cell(head_gate=True)

    def test_the_entries_take_gdn_s_projection_and_parameters(self):
        from engine.base import kernel_shape as ks
        from engine.kernels.kda import ring
        from tests.test_engine_kernel_glue import HEAD_DECAY
        ks.bind(HEAD_DECAY)
        t, h, hv, kd = 2, 4, 12, 128
        q, k, v = torch.zeros(1, t, h, kd), torch.zeros(1, t, h, kd), torch.zeros(1, t, hv, kd)
        a, b = torch.zeros(1, t, hv), torch.zeros(1, t, hv)
        A_log, dt_bias = torch.zeros(hv), torch.zeros(hv)
        states = torch.zeros(2, 2, hv, kd, kd)
        with self.assertRaisesRegex(ValueError, "per value head"):
            ring.recurrent_gdn_ring(q, k, v, torch.zeros(1, t, hv, kd), b, A_log, dt_bias, states, 0, 0)
        with self.assertRaisesRegex(ValueError, "CUDA"):                  # CPU tensors stop at the launch contract
            ring.recurrent_gdn_ring(q, k, v, a, b, A_log, dt_bias, states, 0, 0)
        with patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
            for bad in (A_log.half(), torch.zeros(hv + 1), torch.zeros(2 * hv)[::2]):
                with self.subTest(parameter=tuple(bad.shape)), self.assertRaisesRegex(ValueError, "A_log and dt_bias"):
                    ring.recurrent_gdn_ring(q, k, v, a, b, bad, dt_bias, states, 0, 0)
        with self.assertRaisesRegex(ValueError, "slot and one CUDA context"):
            ring.recurrent_gdn_ring_rows(q, k, v, a, b, A_log, dt_bias, states, 0, 0)
        with self.assertRaisesRegex(ValueError, "not both"):
            ring._recurrent(q, k, v, a, b, A_log, dt_bias, states, 0, 0, None, decay=True, head_gate=True)


class ServedLaneTests(unittest.TestCase):
    """Qwen3.8 binds the GDN entries, and a captured step's GDN layer issues no gates launch."""

    def test_the_served_ring_lanes_are_the_gdn_entries(self):
        source = (ROOT / "engine/profiles/qwen38/lanes.py").read_text()
        self.assertIn("from engine.kernels.kda.ring import recurrent_gdn_ring, recurrent_gdn_ring_rows", source)
        self.assertNotIn("recurrent_decay_ring", source)

    def test_a_captured_gdn_layer_launches_no_gates_and_an_eager_decode_step_neither(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        net = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Qwen38Net")
        body = {f.name: ast.get_source_segment(source, f) for f in net.body if isinstance(f, ast.FunctionDef)}
        self.assertNotIn("gdn_gates", body["_gdn_rows"])
        self.assertIn('lanes.gdn_ring_rows(q, k, v, a[None], b[None], p[n + "A_log"], p[n + "dt_bias"]', body["_gdn_rows"])
        # the eager layer computes the gates only for a step with a chunk (a segment longer than the ring)
        self.assertIn("(None, None) if all(s.length <= wr for s in step.segments) else", body["_gdn"])
        self.assertIn('lanes.gdn_ring(q, k, v, a[sl][None], b[sl][None], p[n + "A_log"], p[n + "dt_bias"]', body["_gdn"])

    @unittest.skipUnless(torch is not None, "requires torch")
    def test_the_reference_ring_lane_computes_gdn_s_decay(self):
        from engine.modules.linear_attention import gated_delta_rule, gdn_decay
        from engine.profiles.qwen38 import lanes
        ref = lanes.reference()
        torch.manual_seed(7)
        t, h, hv, kd = 2, 2, 4, 8
        q, k, v = torch.randn(1, t, h, kd), torch.randn(1, t, h, kd), torch.randn(1, t, hv, kd)
        a, b = torch.randn(1, t, hv) * 3, torch.randn(1, t, hv)
        A_log, dt_bias = torch.randn(hv) * .5, torch.randn(hv)
        ring = torch.randn(2, 2, hv, kd, kd) * .1
        initial = ring[1, 0][None].clone()
        out = ref.gdn_ring(q, k, v, a, b, A_log, dt_bias, ring, 1, 1)
        oracle, final = gated_delta_rule(q.repeat_interleave(2, 2), k.repeat_interleave(2, 2), v, gdn_decay(a, A_log, dt_bias),
                                         torch.sigmoid(b), initial, scale=kd ** -0.5, qk_l2norm=True)
        self.assertTrue(torch.allclose(out, oracle, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(ring[1, 0], final[0], rtol=1e-5, atol=1e-6))   # context 1 + t - 1 = 2 -> cell 0


if __name__ == "__main__":
    unittest.main()
