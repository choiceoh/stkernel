"""probes/engine_qwen38_kda.py on the CPU (engine/QWEN38_CARRY.md C3): the case table is Qwen3.8's per-rank cell, the BV
hooks are inert until set and come back unset, a forced tile reaches both recurrent launches (a stand-in kernel object
records the constexpr and runs nothing), the exact gate's arithmetic and reports, and -- under TRITON_INTERPRET=1 -- the
probe's gates end to end on the real kernels.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_kda
    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_kda
"""
from contextlib import contextmanager
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KERNELS = torch is not None and TRITON


def probe():
    import probes.engine_qwen38_kda as module
    return module


@contextmanager
def cuda_view():
    """The launchers' CUDA-only argument checks see CPU tensors as the device's; nothing launches on them here."""
    with patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
        yield


def recorder(launches):
    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **meta: launches.append((tuple(grid), meta["BK"], meta["BV"]))
    return Kernel()


@unittest.skipUnless(torch is not None, "requires torch")
class CaseTable(unittest.TestCase):
    def test_the_probe_imports_without_a_gpu_or_the_kernel_package(self):
        code = "import sys, probes.engine_qwen38_kda as p; print(callable(p.run), 'engine.kernels' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["True", "False"])

    def test_the_case_table_is_qwen38_s_per_rank_cell(self):
        from engine.profiles.qwen38 import facts, shapes
        from tests.test_engine_kernel_shape import QWEN38_TEXT_CONFIG
        p = probe()
        shape = shapes.kernel_shape(QWEN38_TEXT_CONFIG)
        linear = shape.linear
        self.assertEqual((p.K_HEADS, p.V_HEADS, p.DIM, p.DIM), (linear.heads, linear.v_heads, linear.k_dim, linear.v_dim))
        self.assertEqual((p.K_HEADS, p.V_HEADS, p.DIM), (4, 12, 128))
        self.assertEqual(linear.decay, "head")                              # GDN's per-head decay, not KDA's gate
        self.assertEqual((p.SPEC_K, p.SPEC_K), (shape.spec_k, facts.SPEC_K))
        self.assertEqual(p.RING_CELLS, facts.SPEC_K + 1)                   # net.rec_ring
        self.assertEqual(facts.GDN_STATE_DTYPE, "fp32")                    # the probe's ring and states
        self.assertEqual(p.QWEN38, p.Cell(linear.heads, linear.v_heads, linear.k_dim, facts.SPEC_K + 1))
        self.assertEqual(p.QWEN38.qkv, 2 * linear.heads * linear.k_dim + linear.v_heads * linear.v_dim)
        self.assertEqual(p.QWEN38.qkv, 2560)
        self.assertEqual(p.QWEN38.state_bytes, 12 * 128 * 128 * 4)
        self.assertEqual(p.RING_CASES, tuple(itertools.product((1, 2, 4), range(1, facts.SPEC_K + 2))))
        self.assertTrue(all(1 <= tokens <= p.RING_CELLS for _, tokens in p.RING_CASES))
        self.assertEqual(p.RING_EAGER_TOKENS, tuple(range(1, p.RING_CELLS + 1)))       # net._gdn: s.length <= rec_ring
        self.assertEqual(max(rows for rows, _ in p.RING_CASES), p.MAX_ROWS)
        self.assertEqual((p.RECURRENT_TOKENS, p.BVS, p.PREFILL_TOKENS), ((1, 2, 4), (8, 16, 32), (128, 1024, 8192)))
        self.assertTrue(all(tokens % 64 == 0 for tokens in p.PREFILL_TOKENS))          # whole FLA chunks
        # today's rule at this cell: not GLM-5.3's 16/16 x 128 branch, so min(next_power_of_2(V), 8)
        self.assertNotEqual((linear.heads, linear.v_heads), (16, 16))
        self.assertEqual(p.RULE_BV, min(1 << (linear.v_dim - 1).bit_length(), 8))
        self.assertEqual(p.GDN_LAYERS, 36)

    def test_the_field_holds_a_step_s_gdn_layers(self):
        config = ROOT / "probes" / "qwen38_config.json"
        if not config.exists():
            self.skipTest("probes/qwen38_config.json (the checkpoint's config, PR #1089) is not on this base")
        from engine.profiles.qwen38 import facts
        F = facts.architecture(json.loads(config.read_text()))
        p = probe()
        self.assertEqual(len(F.gdn_layers), p.GDN_LAYERS)
        self.assertEqual((F.k_heads_local, F.v_heads_local, F.k_dim, F.v_dim, F.qkv_local, F.spec_k + 1),
                         (p.K_HEADS, p.V_HEADS, p.DIM, p.DIM, p.QWEN38.qkv, p.RING_CELLS))

    def test_the_inputs_are_the_views_the_net_hands_the_lanes(self):
        from engine.kernels.linear_decay import per_channel
        from engine.profiles.qwen38.net import Qwen38Net
        p = probe()
        F = SimpleNamespace(k_heads_local=p.K_HEADS, k_dim=p.DIM, v_heads_local=p.V_HEADS, v_dim=p.DIM)
        for n in (1, 2, 8):
            with self.subTest(rows_times_tokens=n):
                i = p.served_inputs(p.QWEN38, n, torch.device("cpu"))
                self.assertEqual((i.y.shape, i.y.dtype), ((n, 2560), torch.bfloat16))
                for got, want in zip((i.q, i.k, i.v), Qwen38Net._heads(SimpleNamespace(F=F), i.y, n)):
                    self.assertEqual((got.shape, got.stride(), got.storage_offset()),
                                     (want.shape, want.stride(), want.storage_offset()))
                    self.assertEqual(got.untyped_storage().data_ptr(), i.y.untyped_storage().data_ptr())
                self.assertEqual((i.decay.shape, i.decay.dtype, i.beta.shape, i.beta.dtype),
                                 ((1, n, 12), torch.float32, (1, n, 12), torch.bfloat16))
                self.assertEqual(per_channel(i.decay, p.DIM).stride()[-1], 0)       # the launchers' stride-0 channels
                # the ring arms' a and b: net._gdn_rows' split of the in_proj row, so the kernel compiles the served strides
                self.assertEqual((i.proj.shape, i.proj.dtype), ((n, 4120), torch.bfloat16))
                qkv, z, b, a = i.proj.split([p.QWEN38.qkv, 12 * 128, 12, 12], dim=-1)
                for got, want in ((i.a, a[None]), (i.b, b[None])):
                    self.assertEqual((got.shape, got.stride(), got.storage_offset()),
                                     (want.shape, want.stride(), want.storage_offset()))
                self.assertEqual((i.A_log.shape, i.A_log.dtype, i.dt_bias.shape, i.dt_bias.dtype, i.A_log.is_contiguous()),
                                 ((12,), torch.float32, (12,), torch.float32, True))
                if n > 1:
                    self.assertFalse(i.q.is_contiguous())                          # a column slice of the conv row
                    self.assertFalse(i.a.is_contiguous())
        a, b = p.served_inputs(p.QWEN38, 4, torch.device("cpu")), p.served_inputs(p.QWEN38, 4, torch.device("cpu"))
        p.fill(a, 7)
        p.fill(b, 7)
        for x, y in ((a.y, b.y), (a.decay, b.decay), (a.beta, b.beta), (a.proj, b.proj)):
            self.assertTrue(torch.equal(x, y))                                    # a seed writes the same bytes
        p.fill(b, 8)
        self.assertFalse(torch.equal(a.y, b.y))
        self.assertTrue(bool((a.decay < 0).all()) and bool((a.decay.exp() > 1e-4).all()))
        from engine.modules.linear_attention import gdn_decay                     # the functional arms' decay is GDN's
        self.assertTrue(torch.equal(a.decay, gdn_decay(a.a, a.A_log, a.dt_bias)))
        self.assertTrue(torch.equal(a.beta, a.b))


@unittest.skipUnless(KERNELS, "requires torch and triton (the kernel modules import it)")
class Hooks(unittest.TestCase):
    def setUp(self):
        from engine.base import kernel_shape as ks
        from engine.kernels.kda import kda, ring
        from tests.test_engine_kernel_glue import HEAD_DECAY
        self.ring, self.kda = ring, kda
        ks.reset()
        self.addCleanup(ks.reset)
        ks.bind(HEAD_DECAY)                                                 # the GDN ring entries refuse a KDA cell
        self.kernels = ring.fused_recurrent_gated_delta_rule_fwd_kernel, kda.fused_recurrent_gated_delta_rule_fwd_kernel
        self.addCleanup(self.restored)

    def restored(self):
        """Nothing a test does leaks: the hooks unset and the real kernel objects in place."""
        leaked = (self.ring._BV_OVERRIDE, self.kda._BV_OVERRIDE)
        self.ring._BV_OVERRIDE = self.kda._BV_OVERRIDE = None
        self.assertEqual(leaked, (None, None))
        self.assertIs(self.ring.fused_recurrent_gated_delta_rule_fwd_kernel, self.kernels[0])
        self.assertIs(self.kda.fused_recurrent_gated_delta_rule_fwd_kernel, self.kernels[1])

    def hooks(self):
        return self.ring._BV_OVERRIDE, self.kda._BV_OVERRIDE

    def ring_launch(self, rows, tokens, h, hv, dim, eager=False):
        p = probe()
        field = torch.zeros(rows + 1, max(tokens, 2), hv, dim, dim)
        return p.ring_case(p.Cell(h, hv, dim, max(tokens, 2)), rows, tokens, field, eager=eager).launch

    def recurrent_launch(self, tokens, h, hv, dim):
        p = probe()
        return p.recurrent_case(p.Cell(h, hv, dim), tokens, torch.device("cpu")).launch

    def recorded(self, module, launch, bv):
        """(grid, BK, BV) of the one launch `launch` makes with both hooks at `bv`, through a stand-in kernel object."""
        launches = []
        with patch.object(module, "fused_recurrent_gated_delta_rule_fwd_kernel", recorder(launches)), cuda_view():
            self.ring._BV_OVERRIDE = self.kda._BV_OVERRIDE = bv
            try:
                launch()
            finally:
                self.ring._BV_OVERRIDE = self.kda._BV_OVERRIDE = None
        self.assertEqual(len(launches), 1)
        return launches[0]

    def test_the_hooks_default_to_none_and_the_probe_restores_them(self):
        p = probe()
        self.assertEqual(self.hooks(), (None, None))
        with p.forced_bv(16):
            self.assertEqual(self.hooks(), (16, 16))
            with p.forced_bv(32):
                self.assertEqual(self.hooks(), (32, 32))
            self.assertEqual(self.hooks(), (16, 16))
        self.assertEqual(self.hooks(), (None, None))
        with self.assertRaises(KeyError), p.forced_bv(8):
            raise KeyError("a launch that fails")
        self.assertEqual(self.hooks(), (None, None))
        self.assertEqual(p.at_bv(16, self.hooks), (16, 16))
        self.assertEqual(self.hooks(), (None, None))

    def test_a_forced_tile_reaches_the_ring_launch_and_none_keeps_the_rule(self):
        p = probe()
        for rows, tokens in p.RING_CASES:
            launch = self.ring_launch(rows, tokens, p.K_HEADS, p.V_HEADS, p.DIM)
            for bv, want in ((None, 8), (8, 8), (16, 16), (32, 32)):
                with self.subTest(rows=rows, tokens=tokens, bv=bv):
                    self.assertEqual(self.recorded(self.ring, launch, bv), ((1, 128 // want, rows * 12), 128, want))
        for tokens in p.RING_EAGER_TOKENS:                      # the one-row entry: host slot and context
            launch = self.ring_launch(1, tokens, p.K_HEADS, p.V_HEADS, p.DIM, eager=True)
            for bv, want in ((None, 8), (16, 16), (32, 32)):
                with self.subTest(eager_tokens=tokens, bv=bv):
                    self.assertEqual(self.recorded(self.ring, launch, bv), ((1, 128 // want, 12), 128, want))
        for tokens, rule in ((6, 16), (7, 8)):                  # GLM-5.3's cell keeps its branch until a tile is forced
            launch = self.ring_launch(1, tokens, 16, 16, 128)
            with self.subTest(glm_tokens=tokens):
                self.assertEqual(self.recorded(self.ring, launch, None)[2], rule)
                self.assertEqual(self.recorded(self.ring, launch, 8)[2], 8)
                self.assertEqual(self.recorded(self.ring, launch, 32)[2], 32)
        with self.assertRaisesRegex(ValueError, "one row when eager"):
            self.ring_launch(2, 1, p.K_HEADS, p.V_HEADS, p.DIM, eager=True)

    def test_a_forced_tile_reaches_the_functional_launch_and_none_keeps_the_rule(self):
        p = probe()
        for tokens in p.RECURRENT_TOKENS:
            launch = self.recurrent_launch(tokens, p.K_HEADS, p.V_HEADS, p.DIM)
            for bv, want in ((None, 8), (8, 8), (16, 16), (32, 32)):
                with self.subTest(tokens=tokens, bv=bv):
                    self.assertEqual(self.recorded(self.kda, launch, bv), ((1, 128 // want, 12), 128, want))
        for tokens, rule in ((6, 16), (7, 8)):
            launch = self.recurrent_launch(tokens, 16, 16, 128)
            with self.subTest(glm_tokens=tokens):
                self.assertEqual(self.recorded(self.kda, launch, None)[2], rule)
                self.assertEqual(self.recorded(self.kda, launch, 32)[2], 32)

    def test_a_tile_the_launch_cannot_take_is_refused_before_it(self):
        p = probe()
        for module, launch in ((self.ring, self.ring_launch(2, 2, p.K_HEADS, p.V_HEADS, p.DIM)),
                               (self.kda, self.recurrent_launch(2, p.K_HEADS, p.V_HEADS, p.DIM))):
            for bad in (0, -8, 12, 48, 256, True, 8.0, "16"):
                with self.subTest(module=module.__name__, bv=bad):
                    launches = []
                    with patch.object(module, "fused_recurrent_gated_delta_rule_fwd_kernel", recorder(launches)), \
                            cuda_view(), p.forced_bv(bad), self.assertRaisesRegex(ValueError, "_BV_OVERRIDE"):
                        launch()
                    self.assertEqual(launches, [])
                    self.assertEqual(self.hooks(), (None, None))

    def test_the_probe_records_the_launch_it_would_make_and_restores_the_kernels(self):
        p = probe()
        rows4 = self.ring_launch(4, 2, p.K_HEADS, p.V_HEADS, p.DIM)
        with cuda_view():
            self.assertEqual(p.launcher_tile(rows4), dict(grid=(1, 16, 48), BK=128, BV=8))
            self.assertEqual(p.launcher_tile(rows4, 32), dict(grid=(1, 4, 48), BK=128, BV=32))
            self.assertEqual(p.launcher_tile(self.recurrent_launch(4, p.K_HEADS, p.V_HEADS, p.DIM), 16),
                             dict(grid=(1, 8, 12), BK=128, BV=16))
            with self.assertRaises(ValueError):
                p.launcher_tile(rows4, 256)
            with self.assertRaisesRegex(RuntimeError, "one recurrent launch"):
                p.launcher_tile(lambda: None)
        self.restored()
        events = []
        with cuda_view():
            p.check_launcher(lambda event, **values: events.append((event, values)), "ring", rows4, rows=4, tokens=2)
        self.assertEqual(events, [("launcher", dict(entry="ring", rows=4, tokens=2, rule_bv=8, rule_grid=[1, 16, 48],
                                                    forced={"8": 8, "16": 16, "32": 32}))])
        glm = self.ring_launch(1, 2, 16, 16, 128)                     # the rule passes 16 there, not this cell's 8
        with cuda_view(), self.assertRaisesRegex(RuntimeError, "launcher passes BV 16"):
            p.check_launcher(lambda *args, **values: None, "ring", glm, rows=1, tokens=2)


@unittest.skipUnless(KERNELS, "requires torch and triton (the probe's hooks live in the kernel modules)")
class Gate(unittest.TestCase):
    """The exact gate's arithmetic and reports on stand-ins whose bytes follow the hook."""

    def setUp(self):
        from engine.kernels.kda import ring
        self.ring = ring

    def tile(self):
        return self.ring._BV_OVERRIDE

    def test_first_difference_compares_bytes(self):
        p = probe()
        x = torch.tensor([1.0, -2.0, float("nan"), 0.0], dtype=torch.bfloat16)
        self.assertIsNone(p.first_difference(x, x.clone()))                               # NaN-safe
        y = x.clone()
        y[1] = -2.5
        self.assertEqual(p.first_difference(y, x), dict(elements=1, of=4, first=1, max_abs=0.5, nonfinite=0))
        z = x.clone()
        z[3] = -0.0
        self.assertEqual(p.first_difference(z, x)["first"], 3)                           # -0.0 is not 0.0
        w = x.clone()
        w[0], w[2] = float("inf"), 3.0
        self.assertEqual(p.first_difference(w, x), dict(elements=2, of=4, first=0, max_abs=None, nonfinite=2))
        wide = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()                   # strided views compare too
        self.assertIsNone(p.first_difference(wide, wide.contiguous()))
        self.assertEqual(p.first_difference(x.float(), x)["dtype"], "torch.float32")
        self.assertEqual(p.first_difference(x[:3], x)["want_shape"], [4])

    def test_compare_records_names_the_first_step_and_tensor(self):
        p = probe()
        a = [(("output", torch.zeros(2)), ("ring", torch.zeros(3))), (("output", torch.ones(2)), ("ring", torch.ones(3)))]
        b = [tuple((name, t.clone()) for name, t in step) for step in a]
        self.assertIsNone(p.compare_records(a, b))
        b[1][1][1][2] = 5
        self.assertEqual(p.compare_records(a, b), dict(step=1, tensor="ring", elements=1, of=3, first=2, max_abs=4.0,
                                                       nonfinite=0))
        self.assertEqual(p.compare_records(a[:1], b), dict(steps=1, want_steps=2))

    def test_the_ring_chain_opens_accepts_and_rolls_back(self):
        p = probe()
        for rows, tokens in p.RING_CASES:
            with self.subTest(rows=rows, tokens=tokens):
                chain = p.ring_chain(rows, tokens)
                self.assertEqual(len(chain), 4)
                for slots, contexts in chain:
                    self.assertEqual(sorted(slots), list(range(rows)))
                    self.assertTrue(all(c >= 0 for c in contexts) and len(contexts) == rows)
                firsts = [contexts for _, contexts in chain]
                self.assertEqual(firsts[0][0], 0)                                          # a sequence opens
                if rows > 1:
                    self.assertEqual({c % 2 for c in firsts[0][1:]}, {0, 1} if rows > 2 else {0})
                self.assertEqual([b[0] - a[0] for a, b in zip(firsts, firsts[1:])], [tokens, 1, tokens])
                # the rollback step reads the ring cell the step before wrote for its first token
                cell_read = (firsts[2][0] - 1) % p.RING_CELLS
                self.assertEqual(cell_read, firsts[1][0] % p.RING_CELLS)

    def fake_launch(self, value, perturb=(), refuse=()):
        def launch():
            bv = self.tile()
            if bv in refuse:
                raise RuntimeError(f"no tile {bv}")
            out = value().clone()
            if bv in perturb:
                out.view(-1)[1] += 1
            return out
        return launch

    def test_the_eager_gate_holds_every_tile_to_the_rule(self):
        p = probe()
        steps = lambda run: [(("output", run()),), (("output", run() * 2),)]
        rule, arms = p.eager_gate(steps, self.fake_launch(lambda: torch.arange(4.0), perturb=(16,), refuse=(32,)))
        self.assertTrue(torch.equal(rule[1][0][1], torch.arange(4.0) * 2))
        self.assertIsNone(arms[8])
        self.assertEqual(arms[16], dict(form="eager", step=0, tensor="output", elements=1, of=4, first=1, max_abs=1.0,
                                        nonfinite=0))
        self.assertEqual(arms[32], dict(form="eager", refused="RuntimeError: no tile 32"))
        with self.assertRaisesRegex(RuntimeError, "BV 8 forced does not hold"):
            p.eager_gate(steps, self.fake_launch(lambda: torch.arange(4.0), perturb=(8,)))
        with self.assertRaisesRegex(RuntimeError, "no tile 8"):                         # the rule's own tile never passes as a result
            p.eager_gate(steps, self.fake_launch(lambda: torch.arange(4.0), refuse=(8,)))
        self.assertIsNone(self.tile())

    def test_the_ring_steps_start_every_arm_from_the_same_bytes(self):
        """A stand-in that adds to the addressed slots: without the restore every arm after the first would differ."""
        p = probe()
        cell = p.Cell(1, 2, 4, 2)
        field = torch.randn(4, 2, 2, 4, 4)
        case = p.ring_case(cell, 2, 2, field, dtype=torch.float32)
        before = case.initial.clone()

        def launch():
            for slot, context in zip(case.slots.tolist(), case.contexts.tolist()):
                case.field[slot, context % 2] += case.inputs.y.sum()
            if self.tile() == 16:
                case.field[0, 1, 1, 2, 3] += 1
            return case.inputs.v.clone()
        rule, arms = p.eager_gate(lambda run: p.ring_steps(run, case), launch)
        p.ring_rule_check(case, rule)
        self.assertEqual((arms[8], arms[32]), (None, None))
        self.assertEqual((arms[16]["step"], arms[16]["tensor"], arms[16]["elements"]), (0, "ring", 1))
        self.assertTrue(torch.equal(case.initial, before))
        self.assertEqual([len(step) for step in rule], [2] * 4)

    def test_the_rule_check_refuses_a_vacuous_or_stray_ring(self):
        p = probe()
        case = SimpleNamespace(rows=2, tokens=2, initial=torch.zeros(3, 2, 1, 2, 2))
        out = torch.zeros(2)

        def record(region):
            return (("output", out), ("ring", region))
        written = torch.zeros(3, 2, 1, 2, 2)
        written[:2] = 1
        p.ring_rule_check(case, [record(written)])
        stray = written.clone()
        stray[2, 0, 0, 0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "wrote slot 2"):
            p.ring_rule_check(case, [record(stray)])
        idle = written.clone()
        idle[1] = 0
        with self.assertRaisesRegex(RuntimeError, "slot 1 was not written"):
            p.ring_rule_check(case, [record(idle)])
        with self.assertRaisesRegex(RuntimeError, "slot 0 was not written"):
            p.ring_rule_check(case, [record(written), record(written.clone())])       # a step that changed nothing
        broken = written.clone()
        broken[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            p.ring_rule_check(case, [record(broken)])
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            p.recurrent_rule_check(SimpleNamespace(tokens=2), [(("output", out), ("states", broken))])

    def test_the_replay_gate_reads_the_graph_s_own_output(self):
        p = probe()
        output = torch.zeros(3)
        values = iter([torch.tensor([1.0, 2, 3]), torch.tensor([4.0, 5, 6]), torch.tensor([1.0, 2, 3]),
                       torch.tensor([4.0, 5, 7])])
        graph = SimpleNamespace(replay=lambda: output.copy_(next(values)))
        steps = lambda run: [(("output", run().clone()),) for _ in range(2)]
        rule = [(("output", torch.tensor([1.0, 2, 3])),), (("output", torch.tensor([4.0, 5, 6])),)]
        self.assertIsNone(p.replay_gate(steps, graph, output, rule))
        self.assertEqual(p.replay_gate(steps, graph, output, rule)["step"], 1)

    def test_a_case_reports_each_tile_and_returns_the_exact_ones(self):
        """gate_case without a stream (the CPU dry run): launcher, then eager verdicts per tile."""
        p = probe()
        events = []
        value = torch.arange(4.0)

        def launch():
            kernel = self.ring.fused_recurrent_gated_delta_rule_fwd_kernel
            if isinstance(kernel, p.LaunchRecorder):                        # the launcher check: record the tile only
                bv = self.tile() or p.RULE_BV
                kernel[(1, 128 // bv, 12)](BK=128, BV=bv)
                return None
            return self.fake_launch(lambda: value, perturb=(16,), refuse=(32,))()
        case = SimpleNamespace(launch=launch)
        checked = []
        exact, graphs = p.gate_case(lambda event, **values: events.append(dict(event=event, **values)), "ring", case,
                                    lambda run, case: [(("output", run()),)], None,
                                    lambda case, rule: checked.append(len(rule)), rows=1, tokens=2)
        self.assertEqual((exact, graphs, checked), ({8: True, 16: False, 32: False}, {}, [1]))
        self.assertEqual([(e["event"], e.get("bv")) for e in events],
                         [("launcher", None), ("exact", 8), ("inexact", 16), ("refused", 32)])
        self.assertEqual(events[2]["form"], "eager")
        self.assertTrue(all(e["rows"] == 1 and e["tokens"] == 2 and e["entry"] == "ring" for e in events))
        self.assertIsNone(self.tile())

    def test_the_verdict_names_the_fastest_exact_tile(self):
        p = probe()
        exact = {8: True, 16: True, 32: False}
        timings = {8: dict(cold_us=10.0, warm_us=5.0), 16: dict(cold_us=8.0, warm_us=6.0), 32: dict(cold_us=1.0, warm_us=1.0)}
        self.assertEqual(p.verdict(exact, timings), dict(
            exact=[8, 16], inexact=[32], fastest_cold=16, fastest_warm=8,
            cold_over_rule={"8": 1.0, "16": 0.8}, warm_over_rule={"8": 1.0, "16": 1.2}))
        tie = {8: dict(cold_us=4.0, warm_us=4.0), 32: dict(cold_us=4.0, warm_us=3.0)}
        got = p.verdict({8: True, 16: False, 32: True}, tie)
        self.assertEqual((got["fastest_cold"], got["fastest_warm"]), (8, 32))           # a tie keeps the narrower tile
        self.assertEqual(p.verdict({8: True}, {}), dict(exact=[8], inexact=[]))

    def test_the_timing_arithmetic(self):
        p = probe()
        self.assertEqual(p.ring_bytes(p.QWEN38, 2, 2), 2 * 3 * 12 * 128 * 128 * 4)     # each row: read one, write two
        self.assertEqual(p.recurrent_bytes(p.QWEN38, 4), 5 * 12 * 128 * 128 * 4)
        row = p.summary([3.0, 1.0, 2.0], [2.0, 4.0, 6.0], 2**20)
        self.assertEqual(row, dict(cold_us=2.0, warm_us=4.0, cold_min_us=1.0, warm_min_us=2.0, samples=3, mib=1.0,
                                   cold_gbps=round(2**20 / 2.0 / 1e3, 2), warm_gbps=round(2**20 / 4.0 / 1e3, 2)))


@unittest.skipUnless(KERNELS and INTERPRET, "runs the kernels under TRITON_INTERPRET=1")
class Interpreted(unittest.TestCase):
    """The probe's gates end to end on the real kernels under Triton's CPU interpreter, fp32 (the interpreter has no
    bf16). Its per-row reductions do not depend on how many value rows a tile holds, so every tile must come out exact
    here: an inexact verdict would be the probe's own bug (an unrestored ring, a stale input), not a finding."""

    def setUp(self):
        from engine.base import kernel_shape as ks
        from tests.test_engine_kernel_glue import HEAD_DECAY
        ks.reset()
        self.addCleanup(ks.reset)
        ks.bind(HEAD_DECAY)                                                 # the GDN ring entries refuse a KDA cell
        torch.manual_seed(20260917)

    def gate(self, entry, case, steps, rule_check, **key):
        from engine.kernels.kda import kda, ring
        from tests.test_engine_gdn_ring_gate import ring_kernels             # kda_kernels and libdevice's log1p
        p = probe()
        events = []
        with ring_kernels():
            exact, graphs = p.gate_case(lambda event, **values: events.append(dict(event=event, **values)), entry, case,
                                        steps, None, rule_check, **key)
        self.assertEqual((ring._BV_OVERRIDE, kda._BV_OVERRIDE), (None, None))
        self.assertEqual(graphs, {})
        self.assertEqual(events[0]["event"], "launcher")
        self.assertEqual(events[0]["rule_bv"], p.RULE_BV)
        return exact, events

    def test_the_ring_gate_on_the_kernel(self):
        p = probe()
        small, small_field = p.Cell(2, 4, 32, 2), torch.randn(5, 2, 4, 32, 32) * .5
        for cell, field, cases in ((small, small_field, ((1, 1), (1, 2), (2, 1), (4, 2))),
                                   (p.QWEN38, torch.randn(2, 2, 12, 128, 128) * .5, ((1, 2),))):
            for rows, tokens in cases:
                with self.subTest(cell=cell, rows=rows, tokens=tokens):
                    case = p.ring_case(cell, rows, tokens, field, dtype=torch.float32)
                    exact, _ = self.gate("ring", case, p.ring_steps, p.ring_rule_check, rows=rows, tokens=tokens)
                    self.assertEqual(exact, {8: True, 16: True, 32: True})
        for tokens in p.RING_EAGER_TOKENS:
            with self.subTest(eager_tokens=tokens):
                case = p.ring_case(small, 1, tokens, small_field, dtype=torch.float32, eager=True)
                exact, events = self.gate("ring_eager", case, p.ring_steps, p.ring_rule_check, rows=1, tokens=tokens)
                self.assertEqual(exact, {8: True, 16: True, 32: True})
                self.assertEqual({e["event"] for e in events[1:]}, {"exact"})

    def test_the_recurrent_gate_on_the_kernel(self):
        p = probe()
        for tokens in p.RECURRENT_TOKENS:
            with self.subTest(tokens=tokens):
                case = p.recurrent_case(p.Cell(2, 4, 32), tokens, torch.device("cpu"), dtype=torch.float32)
                exact, _ = self.gate("recurrent", case, p.recurrent_steps, p.recurrent_rule_check, tokens=tokens)
                self.assertEqual(exact, {8: True, 16: True, 32: True})

    def test_the_prefill_call_is_the_lane_s(self):
        from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
        from tests.test_engine_kernel_glue import kda_kernels
        p = probe()
        cell = p.Cell(2, 4, 32)
        args = p.prefill_args(cell, 70, torch.device("cpu"), dtype=torch.float32, out=False)
        self.assertEqual((args["initial_state"].shape, args["initial_state"].is_contiguous()), ((1, 4, 32, 32), True))
        self.assertTrue(bool((args["beta"] > 0).all() and (args["beta"] < 1).all()))
        with kda_kernels():
            o, state = chunk_kda_with_decay(**args)
        self.assertEqual((o.shape, state.shape), ((1, 70, 4, 32), (1, 4, 32, 32)))
        self.assertTrue(bool(torch.isfinite(o).all()) and bool(torch.isfinite(state).all()))


class LaneRoutingTests(unittest.TestCase):
    def test_kernel_check_routes_the_lane_to_the_probe(self):
        text = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_kda'", text)
        self.assertIn("from probes.engine_qwen38_kda import run as qwen38_kda", text)


if __name__ == "__main__":
    unittest.main()
