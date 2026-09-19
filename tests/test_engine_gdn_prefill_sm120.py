"""Qwen3.8's long prefill GDN on FlashInfer's SM120 kernel (engine/kernels/gdn_prefill_sm120.py; sm_121a intake U13).

The kernel is the image's; what the engine owns is the call around it: q/k normalised first (the image's build ignores
`use_qk_l2norm_in_kernel`, flashinfer#5255), the forget gate as exp(log decay), the state's two layouts, and the
boundary states as checkpoints every N tokens picked out by chunk index. On any CPU those are held against a stand-in
kernel that records what it was handed and writes each checkpoint's index into it; on a GB10 (the single-GPU lane's
probes/engine_qwen38_cells.GLUE_CASES) the lane is held to the served KDA chunk kernel, a boundary state included,
within `BAND`, and the boot's `qualify` runs as the boot runs it.

    wsl: ~/.cache/stk-engine-cpu/bin/python -m unittest tests.test_engine_gdn_prefill_sm120 -v
"""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = None
if importlib.util.find_spec("torch"):
    import torch

GPU = (torch is not None and torch.cuda.is_available() and importlib.util.find_spec("flashinfer") is not None
       and torch.cuda.get_device_capability()[0] == 12)
HK, HV, D = 4, 12, 128                                # Qwen3.8's per-rank cell


class StandIn:
    """flashinfer.gdn_prefill.chunk_gated_delta_rule's surface: records the call, writes checkpoint j as the value j + 1
    and the output state as -1, returns (o, state) the way the kernel does."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        t, hv, d = kw["v"].shape
        if kw.get("state_checkpoints") is not None:
            for j in range(kw["state_checkpoints"].shape[0]):
                kw["state_checkpoints"][j].fill_(j + 1)
        kw["output_state"].fill_(-1)
        return torch.zeros(t, hv, d, dtype=kw["v"].dtype), kw["output_state"]


def inputs(t, *, state=True):
    gen = torch.Generator().manual_seed(t)
    q = torch.randn(1, t, HK, D, generator=gen).bfloat16()
    k = torch.randn(1, t, HK, D, generator=gen).bfloat16()
    v = torch.randn(1, t, HV, D, generator=gen).bfloat16()
    decay = -torch.rand(1, t, HV, generator=gen)
    beta = torch.rand(1, t, HV, generator=gen).bfloat16()
    state0 = torch.randn(1, HV, D, D, generator=gen) if state else None
    return q, k, v, decay, beta, state0


@unittest.skipUnless(torch is not None, "torch required")
class CallTests(unittest.TestCase):
    def call(self, t=1024, states_at=None, state=True):
        from engine.kernels import gdn_prefill_sm120
        kernel = StandIn()
        module = types.ModuleType("flashinfer.gdn_prefill")
        module.chunk_gated_delta_rule = kernel
        package = types.ModuleType("flashinfer")
        package.gdn_prefill = module
        args = inputs(t, state=state)
        with mock.patch.dict(sys.modules, {"flashinfer": package, "flashinfer.gdn_prefill": module}), \
                mock.patch.object(gdn_prefill_sm120, "_normalized", side_effect=lambda q, k: (q.contiguous(), k.contiguous())):
            out = gdn_prefill_sm120.chunk(*args, states_at=states_at)
        return kernel.calls[0], out, args

    def test_what_the_kernel_is_handed(self):
        kw, (o, state), (q, k, v, decay, beta, state0) = self.call()
        self.assertFalse(kw["use_qk_l2norm_in_kernel"], "normalised here: the image's build ignores the flag")
        self.assertEqual(tuple(kw["q"].shape), (1024, HK, D))
        self.assertTrue(torch.equal(kw["g"], decay[0].exp()), "the forget gate as alpha")
        self.assertEqual(kw["beta"].dtype, torch.float32)
        self.assertEqual(kw["cu_seqlens"].tolist(), [0, 1024])
        self.assertTrue(torch.equal(kw["initial_state"], state0.transpose(-1, -2)), "the kernel keeps [HV, V, K]")
        self.assertEqual(kw["scale"], D ** -0.5)
        self.assertNotIn("state_checkpoints", kw)
        self.assertEqual((tuple(o.shape), tuple(state.shape)), ((1, 1024, HV, D), (1, HV, D, D)))

    def test_a_fresh_sequence_starts_from_zero(self):
        kw, _, _ = self.call(state=False)
        self.assertEqual(float(kw["initial_state"].abs().sum()), 0.0)

    def test_boundaries_are_checkpoints_a_block_apart_picked_by_chunk(self):
        # marks at 768 and 1536 tokens (a block of 768 is 12 chunks of 64): a checkpoint every 768, the first two
        kw, (o, state, states), _ = self.call(t=2048, states_at=[12, 24])
        self.assertEqual(kw["checkpoint_every_n_tokens"], 768)
        self.assertEqual(tuple(kw["state_checkpoints"].shape), (2, HV, D, D))
        self.assertEqual(kw["checkpoint_cu_starts"].tolist(), [0, 2])
        self.assertIs(kw["use_cp"], False, "the CP path takes no checkpoints")
        self.assertEqual([float(s[0, 0, 0]) for s in states], [1.0, 2.0], "checkpoint j is the state after (j+1)N")

    def test_marks_that_share_no_block_fall_to_the_largest_common_interval(self):
        from engine.kernels import gdn_prefill_sm120
        self.assertEqual(gdn_prefill_sm120.checkpoint_every([12, 20]), 256)           # 768 and 1280 tokens
        kw, (_, _, states), _ = self.call(t=1280 + 64, states_at=[12, 20])
        self.assertEqual([float(s[0, 0, 0]) for s in states], [3.0, 5.0])
        with self.assertRaises(ValueError):
            gdn_prefill_sm120.checkpoint_every([0, 4])

    def test_a_segment_it_does_not_take_is_refused_before_any_launch(self):
        from engine.kernels import gdn_prefill_sm120
        self.assertFalse(gdn_prefill_sm120.admits(1023, D))
        self.assertFalse(gdn_prefill_sm120.admits(4096, 64))
        with self.assertRaisesRegex(ValueError, "the caller chooses by `admits`"):
            gdn_prefill_sm120.chunk(*inputs(512))


@unittest.skipUnless(torch is not None, "torch required")
class ServedTests(unittest.TestCase):
    """On by the operator's decision of 2026-09-19 (fleet unmeasured): a long prefill segment takes the lane, the rest
    keep the served chunk kernel, and a boot can decline it from the launcher to the net."""

    def net(self, *, lane=True, on=True):
        from engine.profiles.qwen38.net import Qwen38Net
        stand_in = Qwen38Net.__new__(Qwen38Net)
        stand_in.F = SimpleNamespace(k_dim=D)
        stand_in.lanes = SimpleNamespace(gdn_chunk_long=object() if lane else None)
        stand_in.gdn_flashinfer = on
        return stand_in

    def test_a_long_segment_takes_it_and_nothing_else_does(self):
        from engine.profiles.qwen38.net import Qwen38Net
        self.assertTrue(Qwen38Net._gdn_long(self.net(), 1024))
        self.assertFalse(Qwen38Net._gdn_long(self.net(), 1023))
        self.assertFalse(Qwen38Net._gdn_long(self.net(on=False), 8192))
        self.assertFalse(Qwen38Net._gdn_long(self.net(lane=False), 8192))
        self.assertFalse(Qwen38Net._gdn_long(SimpleNamespace(lanes=SimpleNamespace(), F=SimpleNamespace(k_dim=D)), 8192),
                         "a stand-in that never made the choice")

    def test_it_is_on_unless_a_boot_declines(self):
        import inspect
        from engine.profiles.qwen38 import lanes
        from engine.profiles.qwen38.net import Qwen38Net
        self.assertIs(inspect.signature(Qwen38Net.__init__).parameters["gdn_flashinfer"].default, True)
        self.assertIs(inspect.signature(lanes.qualify).parameters["gdn_flashinfer"].default, True)
        self.assertIn("gdn_chunk_long=on_main(gdn_prefill_sm120.chunk)",
                      (ROOT / "engine/profiles/qwen38/lanes.py").read_text(encoding="utf-8"))
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--no-gdn-flashinfer", action="store_true",', fleet)
        self.assertIn("gdn_flashinfer=not a.no_gdn_flashinfer", fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('case "${ST_GDN_FLASHINFER:-1}" in', launcher)
        self.assertIn('0) GDN_ARG="--no-gdn-flashinfer" ;;', launcher)
        self.assertIn("$UNION_ARG $GDN_ARG'", launcher)
        self.assertIn("`ST_GDN_FLASHINFER=1`", (ROOT / "engine/SERVING_DEFAULTS.md").read_text(encoding="utf-8"))

    def test_the_net_refuses_a_choice_that_is_not_a_boolean(self):
        from engine.profiles.qwen38.net import Qwen38Net
        with self.assertRaisesRegex(ValueError, "declared boolean"):
            Qwen38Net(SimpleNamespace(), SimpleNamespace(world_size=4, rank=0), None, gdn_flashinfer=1)


@unittest.skipUnless(GPU, "a GB10 with the image's FlashInfer")
class GdnPrefillOnTheGpuTests(unittest.TestCase):
    """The lane against the served chunk kernel on a GB10: what the boot's `qualify` holds, and a longer segment with
    block-boundary states."""

    def test_the_boot_qualification_passes(self):
        from engine.kernels import gdn_prefill_sm120
        held = gdn_prefill_sm120.qualify(torch.device("cuda"), k_heads=HK, v_heads=HV, dim=D)
        self.assertTrue(all(held[name] <= gdn_prefill_sm120.BAND for name in ("o", "state", "states")), held)

    def test_a_prompt_chunk_with_block_boundaries(self):
        from engine.kernels import gdn_prefill_sm120
        from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
        from engine.kernels.kda.index import single_sequence_bounds
        t, marks = 4096, [12, 24, 36, 48]                                             # 768-token blocks
        q, k, v, decay, beta, state0 = (x.cuda() if x is not None else None for x in inputs(t))
        decay = decay.float() * 0.1
        o, state, states = gdn_prefill_sm120.chunk(q, k, v, decay, beta, state0, states_at=marks)
        ref_o, ref_state, ref_states = chunk_kda_with_decay(
            q, k, v, decay, beta, scale=D ** -0.5, initial_state=state0.transpose(-1, -2).contiguous(),
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=single_sequence_bounds(t, q.device),
            out=torch.empty_like(v), states_at=marks)

        def error(got, want):
            return float((got.float() - want.float()).abs().max() / want.float().abs().max())
        for name, got, want in (("o", o, ref_o), ("state", state, ref_state.transpose(-1, -2)),
                                ("states", states, ref_states.transpose(-1, -2))):
            with self.subTest(name):
                self.assertTrue(bool(torch.isfinite(got).all()))
                self.assertLessEqual(error(got, want), gdn_prefill_sm120.BAND)


if __name__ == "__main__":
    unittest.main()
