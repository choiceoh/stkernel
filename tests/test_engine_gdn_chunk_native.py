"""The chunk pipeline reads GDN's per-head decay and its grouped q, k as they are (engine/QWEN38_CARRY.md K3).

`kda/chunk_decay.chunk_kda_with_decay` used to build the forms the pipeline's kernels addressed: the chunk-summed decay
expanded to [B,T,HV,K] fp32 and the normalised q, k repeated from H to HV heads -- 189 MiB and 2 x 94.5 MiB a GDN layer
of a 32,256-token chunk at Qwen3.8's per-rank cell. The four kernels that read them (the two K K^T kernels, w/u/kg,
the output; and the state recurrence's gate) now take `G_HEAD` -- the gate is [T,HV], one value a head, stride 0 along
the key channels -- and `QG` -- value head i reads key head i // QG. The values a program loads are the same values, so
the output, the final state and the states at the marks are byte for byte what the widened forms give; with G_HEAD
false and QG 1 every offset is the one KDA's per-channel gate has always been read at.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_gdn_chunk_native
"""
import importlib.util
import os
import unittest
from dataclasses import replace

from engine.base import kernel_shape as ks
from engine.base.kernel_shape import LinearAttention

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class NativeChunkTests(unittest.TestCase):
    def setUp(self):
        from tests.test_engine_kernel_glue import HEAD_DECAY          # the GDN-shaped kernel shape those tests bind
        torch.manual_seed(615811)
        ks.reset()
        self.addCleanup(ks.reset)
        # the interpreter is slow: small widths there, Qwen3.8's per-rank cell on a GPU
        self.h, self.hv, self.kd = (2, 4, 16) if INTERPRET else (4, 12, 128)
        self.dtype = torch.float32 if INTERPRET else torch.bfloat16
        ks.bind(replace(HEAD_DECAY, linear=LinearAttention(heads=self.h, v_heads=self.hv, k_dim=self.kd, v_dim=self.kd,
                                                          conv=4, decay="head")))

    def kernels(self):
        from tests.test_engine_kernel_glue import kda_kernels
        return kda_kernels()

    def step(self, t, heads=None):
        heads = self.h if heads is None else heads
        g = lambda *shape: torch.randn(*shape, device=DEVICE, dtype=self.dtype)
        decay = -torch.nn.functional.softplus(torch.randn(1, t, self.hv, device=DEVICE)) * .1
        beta = torch.sigmoid(torch.randn(1, t, self.hv, device=DEVICE))
        return g(1, t, heads, self.kd), g(1, t, heads, self.kd), g(1, t, self.hv, self.kd), decay, beta

    def run_entry(self, q, k, v, decay, beta, *, bounds=True, **kw):
        from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
        from engine.kernels.kda.index import single_sequence_bounds
        # without `out` the pipeline writes its output over v's storage (as the fused entry does): hand it a copy; an
        # `out` must be a CUDA tensor, which the interpreter's are not
        return chunk_kda_with_decay(q, k, v.clone(), decay, beta, use_qk_l2norm_in_kernel=True,
                                    cu_seqlens=single_sequence_bounds(q.shape[1], q.device) if bounds else None,
                                    out=None if INTERPRET else torch.empty_like(v), **kw)

    def test_the_native_forms_are_the_widened_forms_byte_for_byte(self):
        t = 130 if INTERPRET else 300                                   # three 64-token chunks, the last one short
        q, k, v, decay, beta = self.step(t)
        state0 = torch.randn(1, self.hv, self.kd, self.kd, device=DEVICE) * .1
        for name, kw in (("a fresh sequence, its marks", dict(output_final_state=True, states_at=[1, 2])),
                         ("a continued sequence", dict(output_final_state=True, initial_state=state0)),
                         ("one batch row, no bounds", dict(output_final_state=True, bounds=False))):
            with self.subTest(case=name), self.kernels():
                native = self.run_entry(q, k, v, decay, beta, **kw)
                widened = self.run_entry(q, k, v, decay, beta, widen=True, **kw)
                self.assertEqual(len(native), len(widened))
                for ours, theirs in zip(native, widened):
                    self.assertTrue(torch.equal(ours, theirs))

    def test_it_is_still_the_recurrence(self):
        from engine.modules.linear_attention import gated_delta_rule
        t = 130 if INTERPRET else 300
        q, k, v, decay, beta = self.step(t)
        group = self.hv // self.h
        with self.kernels():
            o, state, marks = self.run_entry(q, k, v, decay, beta, output_final_state=True, states_at=[1])
        oracle = lambda n: gated_delta_rule(q[:, :n].repeat_interleave(group, 2), k[:, :n].repeat_interleave(group, 2),
                                            v[:, :n], decay[:, :n], beta[:, :n], None, scale=self.kd ** -0.5,
                                            qk_l2norm=True)
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
        tolerance = 1e-5 if INTERPRET else 1e-2
        want, final = oracle(t)
        self.assertLessEqual(rel(o, want), tolerance)
        self.assertLessEqual(rel(state.transpose(-1, -2), final), tolerance)
        self.assertLessEqual(rel(marks[0].transpose(-1, -2), oracle(64)[1][0]), tolerance)

    def test_a_per_channel_gate_over_equal_heads_is_read_where_it_always_was(self):
        """KDA's form through the same kernels (G_HEAD false, QG 1): a per-channel decay, as many key heads as value
        heads -- the recurrence, and the same bytes whether or not `widen` is asked for (there is nothing to widen)."""
        from engine.modules.linear_attention import gated_delta_rule
        t = 130 if INTERPRET else 300
        q, k, v, decay, beta = self.step(t, heads=self.hv)
        wide = (decay.unsqueeze(-1) * (1 + torch.rand(1, t, self.hv, self.kd, device=DEVICE))).contiguous()
        with self.kernels():
            o, state = self.run_entry(q, k, v, wide, beta, output_final_state=True)
            again, _ = self.run_entry(q, k, v, wide, beta, output_final_state=True, widen=True)
        want, final = gated_delta_rule(q, k, v, wide, beta, None, scale=self.kd ** -0.5, qk_l2norm=True,
                                       decay_per_channel=True)
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
        self.assertTrue(torch.equal(o, again))
        self.assertLessEqual(rel(o, want), 1e-5 if INTERPRET else 1e-2)
        self.assertLessEqual(rel(state.transpose(-1, -2), final), 1e-5 if INTERPRET else 1e-2)


@unittest.skipUnless(torch is not None, "requires torch")
class AdapterTests(unittest.TestCase):
    def test_the_entry_builds_neither_widened_form_unless_asked(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "engine/kernels/kda/chunk_decay.py").read_text(encoding="utf-8")
        self.assertIn("if widen and hv != h:", source)
        self.assertIn("if widen and g.ndim == 3:", source)
        self.assertEqual(source.count("repeat_interleave("), 2)
        self.assertEqual(source.count(".expand(b, t, hv, kd)"), 1)


if __name__ == "__main__":
    unittest.main()
