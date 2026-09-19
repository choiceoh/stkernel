"""The Qwen3.8 MoE EP-cell probe (probes/engine_qwen38_moe.py) on the CPU: it imports without a GPU, its cases are
Qwen3.8's, what it says of the dispatcher holds in the dispatcher's own source, the probe hooks default to None and come
back after a pin, its routes are top-10 of 512 on 128 local experts, and its oracle is expert_gemm's dataflow."""
import __future__
import ast
import importlib.util
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "engine/kernels/b12x/moe_dispatch.py"
PROBE = ROOT / "probes/engine_qwen38_moe.py"
LANES = ROOT / "engine/profiles/qwen38/lanes.py"
HAVE_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAVE_TORCH, "torch is not installed")


def probe():
    return importlib.import_module("probes.engine_qwen38_moe")


def qwen_shape():
    from tests.test_engine_kernel_shape import qwen_shape as shape
    return shape()


def dispatch_module(names, **env):
    """The named top-level functions and assignments of moe_dispatch.py as a module, without cutlass or flashinfer
    (the functions' globals are the module's, so a pin set on it is what they read)."""
    tree = ast.parse(DISPATCH.read_text())
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            hit = {t.id for t in targets if isinstance(t, ast.Name)} & set(names)
            if hit:
                body.append(node)
                found |= hit
    missing = set(names) - found
    if missing:
        raise AssertionError(f"moe_dispatch.py no longer defines {sorted(missing)}")
    module = types.ModuleType("moe_dispatch_under_test")
    module.__dict__.update(env)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DISPATCH), "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), module.__dict__)
    return module


SELECTORS = {"_select_moe_mma_tiler_mn", "_select_micro_mma_tiler_mn", "_lookup_mac_ladder", "_select_micro_mac",
             "_MICRO_MAC_LADDER", "_MICRO_TILE_M_OVERRIDE", "_MICRO_MAC_OVERRIDE", "_select_dynamic_tile_m",
             "_DYNAMIC_TILE_M_OVERRIDE", "_LEVEL_TILE_M", "_b12x_ep_zero_weight_micro_expert_id",
             "_B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS", "_B12X_EP_ZERO_WEIGHT_MICRO_TOKENS", "_B12X_EP_ZERO_WEIGHT_MICRO_TOPK",
             "_B12X_EP_ZERO_WEIGHT_MICRO_K", "_B12X_EP_ZERO_WEIGHT_MICRO_N", "_B12X_EP_ZERO_WEIGHT_MICRO_SWIGLU_LIMIT",
             "_EP_ZERO_WEIGHT_MICRO_CELL", "_MICRO_MAX_TOKENS", "_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK",
             "_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT", "_DIRECT_MICRO_MAX_N"}


def selectors(sms=48):
    import torch
    moe = qwen_shape().moe
    return dispatch_module(SELECTORS, torch=torch, get_num_sm=lambda *_: sms, _admitted_moe=lambda: moe,
                           is_gated_activation=lambda a: a in {"silu", "gelu_tanh", "swigluoai_uninterleave"})


def micro_tile(md, m, c):
    """The micro tile of an m-token captured launch; the workspace is the 8-token step's (captured largest first)."""
    return md._select_micro_mma_tiler_mn(
        state_E=c.local, weight_E=c.local, m=m, k=c.hidden, n=c.inter, num_topk=c.topk,
        skip_zero_weight_expert_id=c.local, quant_mode="nvfp4", activation="silu", swiglu_alpha=1.0, swiglu_beta=0.0,
        swiglu_limit=None, max_rows=8 * c.topk)


def sentinel(md, m, c):
    return md._b12x_ep_zero_weight_micro_expert_id(
        enabled=False, state_E=c.local, weight_E=c.local, num_tokens=m, k=c.hidden, n=c.inter, num_topk=c.topk,
        activation_precision="fp4", quant_mode="nvfp4", activation="silu", swiglu_limit=None, forced_backend=None)


class ProbeModuleTests(unittest.TestCase):
    @needs_torch
    def test_imports_without_a_gpu(self):
        p = probe()
        self.assertTrue(callable(p.run))
        self.assertEqual(p.HOOKS, ("_MICRO_TILE_M_OVERRIDE", "_MICRO_MAC_OVERRIDE", "_DYNAMIC_TILE_M_OVERRIDE"))
        self.assertIn("pack_stage_bytes", p.__doc__)
        self.assertIn("--lanes qwen38_moe", p.__doc__)

    @needs_torch
    def test_the_shape_is_the_wizards_qwen38(self):
        from tests.test_engine_kernel_shape import QWEN38_TEXT_CONFIG
        p = probe()
        self.assertEqual(p.TEXT_CONFIG, QWEN38_TEXT_CONFIG)
        self.assertEqual(p.kernel_shape(), qwen_shape())
        c = p.cell_of(p.kernel_shape())
        self.assertEqual((c.experts, c.local, c.hidden, c.inter, c.topk, c.first_expert, c.spec_k),
                         (512, 128, 2560, 640, 10, 0, 3))      # the served draft count, from the wizard's shape
        self.assertEqual(p.cell_of(p.kernel_shape(), rank=3).first_expert, 384)

    @needs_torch
    def test_cases_are_qwen38s_steps_and_tiles(self):
        from engine.base.kernel_shape import MEASURED
        p = probe()
        c = p.cell_of(p.kernel_shape())
        # 1..8 captured rows of K + 1 tokens at K=1 and K=3: micro to 8, the static shapes above it
        self.assertEqual(p.decode_tokens(c), (2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32))
        self.assertEqual(p.decode_order(p.decode_tokens(c), 8), [8, 6, 4, 2, 10, 12, 14, 16, 20, 24, 28, 32])
        self.assertEqual((p.PREFILL_CHECKS, p.PREFILL_TIMINGS), ((16, 64, 128, 1024, 4096), (1024, 4096, 8192)))
        self.assertEqual(p.DYNAMIC_TILES, (16, 32, 64, 128))
        self.assertTrue(set(p.MICRO_TILES) <= {32, 64, 128} and 64 in p.MICRO_TILES)
        sms = MEASURED.device.sms
        self.assertTrue(all(1 <= mac <= sms for mac in p.MICRO_MACS) and sms in p.MICRO_MACS)
        self.assertEqual(p.LAYERS, 3)


class DispatcherReadingTests(unittest.TestCase):
    """What the probe's docstring says the dispatcher does with Qwen3.8's cell, held against its source."""

    @needs_torch
    def test_every_decode_launch_takes_micro_at_m64_and_mac_48(self):
        p, md = probe(), selectors()
        c = p.cell_of(p.kernel_shape())
        md._EP_ZERO_WEIGHT_MICRO_CELL = True                        # lanes.served() configures it
        for m in p.decode_tokens(c):
            with self.subTest(tokens=m):
                if m > md._MICRO_MAX_TOKENS:
                    # above the micro cap no sentinel is engaged: the launch is the static kernel's
                    # (tests/test_engine_moe_backend_prefill.py holds the selector to it)
                    self.assertIsNone(sentinel(md, m, c))
                    continue
                self.assertEqual(sentinel(md, m, c), c.local)
                self.assertEqual(micro_tile(md, m, c), (64, 128))
                self.assertEqual(md._select_micro_mac(m * c.topk, c.inter, 48, md._MICRO_MAC_LADDER), 48)
        # the sentinel, not the cutover, takes 6 and 8 tokens to micro; one token and nine are outside the skip
        self.assertLess(md._MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK, 6 * c.topk)
        self.assertIsNone(sentinel(md, 1, c))
        self.assertIsNone(sentinel(md, 9, c))
        # every ladder rung is above GB10's 48 SMs, and 40..80 routed rows have none
        self.assertTrue(all(mac > 48 for _, mac in md._MICRO_MAC_LADDER))
        self.assertIsNone(md._lookup_mac_ladder(md._MICRO_MAC_LADDER, 40))
        self.assertLess(md._DIRECT_MICRO_MAX_N, c.inter)
        self.assertEqual(md._STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT, p.STATIC_CUTOVER_PAIRS)
        md._EP_ZERO_WEIGHT_MICRO_CELL = False
        self.assertIsNone(sentinel(md, 2, c))

    @needs_torch
    def test_the_lanes_launch_misses_the_tp_gate_and_sums_decode_in_fp32(self):
        """lanes.served() passes the rank's 128 experts as num_experts, so the admitted-geometry gate (512 of 128) never
        opens for it: no static v2 lane, no _GLM53_B12X_* ladder. The top-10 decode launch accumulates its routes in FP32
        (the bound cell's scatter plane); compact one-route pairs now retain that FP32 sum too."""
        p = probe()
        c = p.cell_of(p.kernel_shape())
        moe = qwen_shape().moe
        md = dispatch_module({"_is_admitted_tp_geometry", "_glm_tp_scatter_shape", "_glm_tp_scatter_fp32"},
                             _admitted_moe=lambda: moe)
        lane = dict(num_local_experts=c.local, hidden_size=c.hidden, intermediate_size=c.inter, num_topk=c.topk,
                    quant_mode="nvfp4", activation="silu", swiglu_limit=None)
        self.assertFalse(md._is_admitted_tp_geometry(num_experts=c.local, **lane))
        self.assertTrue(md._is_admitted_tp_geometry(num_experts=c.experts, **lane))
        fp32 = lambda topk: md._glm_tp_scatter_fp32(state_E=c.local, weight_E=c.local, k=c.hidden, n=c.inter,
                                                    num_topk=topk, quant_mode="nvfp4", activation="silu",
                                                    swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=None)
        self.assertTrue(fp32(c.topk))
        self.assertTrue(fp32(1))

    @needs_torch
    def test_prefill_tiles_follow_rows_per_expert(self):
        p, md = probe(), selectors()
        c = p.cell_of(p.kernel_shape())
        pairs = lambda tokens: tokens * c.topk * c.local // c.experts          # this rank's share of the routes
        self.assertLessEqual(pairs(128), p.STATIC_CUTOVER_PAIRS)                # the check at 128 is a static launch
        self.assertEqual([md._select_dynamic_tile_m(pairs(n), c.local, "silu") for n in (1024, 2048, 4096, 8192)],
                         [32, 32, 64, 128])                                    # 20, 40, 80, 160 rows an expert
        for tokens in p.PREFILL_TIMINGS:
            self.assertGreater(pairs(tokens), p.STATIC_CUTOVER_PAIRS)
            self.assertIn(md._select_dynamic_tile_m(pairs(tokens), c.local, "silu"), p.DYNAMIC_TILES)

    @needs_torch
    def test_launch_records_read_the_dispatchers_cache_keys(self):
        import torch
        p = probe()
        keys = dispatch_module({"_micro_kernel_cache_key", "_static_kernel_cache_key", "_dynamic_kernel_cache_key"})
        common = dict(quant_mode="nvfp4", topk_ids_dtype=torch.int32, input_scales_are_reciprocal=False, fast_math=True,
                      activation="silu", swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=None)
        micro = keys._micro_kernel_cache_key(state_E=128, weight_E=128, m=6, k=2560, n=640, num_topk=10, max_rows=80,
                                             mac=24, mma_tiler_mn=(32, 128), share_input_across_experts=False,
                                             share_expert_scales=False, single_token=False, skip_zero_weight_expert_id=128,
                                             scatter_fp32=True, **common)
        self.assertEqual(p.kernel_fields("micro", micro),
                         dict(kernel="micro", m=6, max_rows=80, mac=24, tile=[32, 128], skip=128))
        static = keys._static_kernel_cache_key(activation_precision="fp4", state_E=128, weight_E=128, m=321, k=2560,
                                               n=640, num_topk=1, max_rows=321, mac=48, mma_tiler_mn=(128, 128), **common)
        self.assertEqual(p.kernel_fields("static", static), dict(kernel="static", m=321, max_rows=321, mac=48,
                                                                 tile=[128, 128]))
        dynamic = keys._dynamic_kernel_cache_key(activation_precision="fp4", E=128, k=2560, n=640, num_topk=1, mac=48,
                                                 mma_tiler_mn=(16, 128), share_input_across_experts=False, **common)
        self.assertEqual(p.kernel_fields("dynamic", dynamic), dict(kernel="dynamic", mac=48, tile=[16, 128]))
        self.assertEqual(p.kernel_fields("direct_micro", None), dict(kernel="direct_micro"))


class HookTests(unittest.TestCase):
    def test_hooks_are_declared_none_in_the_dispatcher(self):
        tree = ast.parse(DISPATCH.read_text())
        declared = {node.target.id: node.value for node in tree.body
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        for name in ("_MICRO_TILE_M_OVERRIDE", "_MICRO_MAC_OVERRIDE", "_DYNAMIC_TILE_M_OVERRIDE"):
            with self.subTest(hook=name):
                self.assertIn(name, declared)
                self.assertIsInstance(declared[name], ast.Constant)
                self.assertIsNone(declared[name].value)

    def test_the_launch_path_reads_the_hooks(self):
        tree = ast.parse(DISPATCH.read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        called = lambda fn: {n.func.id for n in ast.walk(functions[fn]) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name)}
        self.assertIn("_select_micro_mac", called("launch_sm120_static_moe"))
        self.assertIn("_select_micro_mma_tiler_mn", called("_get_micro_kernel"))
        self.assertIn("_select_dynamic_tile_m", called("_dynamic_workspace_tile_m"))
        launch = ast.unparse(functions["launch_sm120_static_moe"])
        self.assertNotIn("micro_work_tiles", launch)           # the MAC arithmetic lives in the helper alone
        read = lambda fn: {n.id for n in ast.walk(functions[fn]) if isinstance(n, ast.Name)}
        self.assertIn("_MICRO_TILE_M_OVERRIDE", read("_select_micro_mma_tiler_mn"))
        self.assertIn("_MICRO_MAC_OVERRIDE", read("_select_micro_mac"))

    @needs_torch
    def test_the_mac_helper_is_the_launchers_old_arithmetic(self):
        md = selectors()
        for ladder in (md._MICRO_MAC_LADDER, ((4, 30), (64, 12))):
            for routed_rows in range(1, 101):
                for n in (128, 512, 640, 2048):
                    for base in (8, 32, 48, 128):
                        work = max(1, routed_rows * max(1, (n + 128 - 1) // 128))
                        old = min(md._lookup_mac_ladder(ladder, routed_rows) or base, work, base)
                        self.assertEqual(md._select_micro_mac(routed_rows, n, base, ladder), old)

    @needs_torch
    def test_pins_take_validate_and_restore(self):
        p, md = probe(), selectors()
        c = p.cell_of(p.kernel_shape())
        for name in p.HOOKS:
            self.assertIsNone(getattr(md, name))
        with p.pinned(md, _MICRO_TILE_M_OVERRIDE=32, _MICRO_MAC_OVERRIDE=24, _DYNAMIC_TILE_M_OVERRIDE=128):
            self.assertEqual(micro_tile(md, 2, c), (32, 128))
            self.assertEqual(md._select_micro_mac(20, 640, 48, md._MICRO_MAC_LADDER), 24)
            self.assertEqual(md._select_dynamic_tile_m(2560, 128, "silu"), 128)
        for name in p.HOOKS:
            self.assertIsNone(getattr(md, name))
        self.assertEqual(micro_tile(md, 2, c), (64, 128))
        for tile in (16, 48, 256):
            with self.subTest(tile=tile), p.pinned(md, _MICRO_TILE_M_OVERRIDE=tile), self.assertRaises(ValueError):
                micro_tile(md, 2, c)
        for mac in (0, 49, 24.0, True):
            with self.subTest(mac=mac), p.pinned(md, _MICRO_MAC_OVERRIDE=mac), self.assertRaises(ValueError):
                md._select_micro_mac(20, 640, 48, md._MICRO_MAC_LADDER)
        with self.assertRaises(RuntimeError):
            with p.pinned(md, _MICRO_TILE_M_OVERRIDE=128):
                raise RuntimeError("the block dies")
        self.assertIsNone(md._MICRO_TILE_M_OVERRIDE)
        with self.assertRaises(AttributeError):
            with p.pinned(md, _MICRO_TILE_M_OVERRIDE=32, _NOT_A_HOOK=1):
                pass
        self.assertIsNone(md._MICRO_TILE_M_OVERRIDE)


@needs_torch
class RouteTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.p = probe()
        self.c = self.p.cell_of(self.p.kernel_shape())

    def test_routes_are_top10_of_512_with_three_quarters_on_other_ranks(self):
        torch, p, c = self.torch, self.p, self.c
        from engine.profiles.qwen38.lanes import local_routes
        tokens = 4096
        ids, w = p.routes(tokens, c, torch.Generator().manual_seed(7), "cpu")
        self.assertEqual((ids.shape, ids.dtype, w.shape, w.dtype), ((tokens, 10), torch.int32, (tokens, 10), torch.float32))
        self.assertTrue(bool(((ids >= 0) & (ids < 512)).all()))
        self.assertTrue(bool((ids.sort(dim=1).values.diff(dim=1) > 0).all()))          # ten distinct experts a token
        self.assertTrue(torch.allclose(w.sum(dim=1), torch.ones(tokens), atol=1e-5) and bool((w > 0).all()))
        local, lw = local_routes(ids, w, c.first_expert, c.local, c.local)
        sentinel = local == c.local
        self.assertAlmostEqual(float(sentinel.float().mean()), 0.75, delta=0.01)
        self.assertTrue(bool((lw[sentinel] == 0).all()) and bool((lw[~sentinel] == w[~sentinel]).all()))
        stats = p.route_stats(ids, c)
        self.assertAlmostEqual(stats["sentinel_fraction"], float(sentinel.float().mean()), places=3)
        self.assertEqual(stats["local_pairs"], int((~sentinel).sum()))
        counts = torch.bincount(local[~sentinel].long(), minlength=c.local).float()
        rate = c.topk / c.experts                                                      # an expert's share of a token
        self.assertAlmostEqual(float(counts.mean()) / (tokens * rate), 1.0, delta=0.03)
        self.assertAlmostEqual(float(counts.var()) / (tokens * rate * (1 - rate)), 1.0, delta=0.3)
        self.assertGreater(int(counts.min()), 0)
        self.assertEqual(stats["active_local_experts"], c.local)

    def test_foreign_routes_are_every_other_ranks(self):
        torch, p = self.torch, self.p
        for rank in (0, 2):
            c = p.cell_of(p.kernel_shape(), rank=rank)
            ids, _ = p.routes(8, c, torch.Generator().manual_seed(rank), "cpu")
            foreign = p.foreign_routes(ids, c)
            self.assertEqual((foreign.shape, foreign.dtype), (ids.shape, torch.int32))
            mine, _ = p.local_mask(foreign, c)
            self.assertFalse(bool(mine.any()))
            self.assertTrue(bool((foreign.sort(dim=1).values.diff(dim=1) > 0).all()))

    def test_pairs_are_the_eager_steps_dispatch(self):
        torch, p, c = self.torch, self.p, self.c
        x, ids, w = p.pattern(64, c, torch.Generator().manual_seed(3), "cpu", min_local=120)
        token, xp, idp, wp = p.pairs_of(x, ids, w, c)
        mine, shifted = p.local_mask(ids, c)
        self.assertEqual(token.numel(), int(mine.sum()))
        self.assertGreaterEqual(token.numel(), 120)
        self.assertTrue(bool((token.diff() >= 0).all()))
        self.assertTrue(torch.equal(xp, x[token]))
        self.assertEqual((idp.shape, idp.dtype, wp.shape, wp.dtype), ((token.numel(), 1), torch.int32,
                                                                       (token.numel(), 1), torch.float32))
        self.assertTrue(torch.equal(idp[:, 0].long(), shifted[mine]) and torch.equal(wp[:, 0], w[mine]))


@needs_torch
class OracleTests(unittest.TestCase):
    """A small cell on the CPU: rank 1 of 16 experts, 4 a rank, top-4, hidden and intermediate 64."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch, cls.p = torch, probe()
        cls.c = cls.p.Cell(experts=16, local=4, hidden=64, inter=64, topk=4, first_expert=4, spec_k=1)
        cls.experts = cls.p.build_experts(cls.c, torch.Generator().manual_seed(11), "cpu")
        cls.x, cls.ids, cls.w = cls.p.pattern(24, cls.c, torch.Generator().manual_seed(12), "cpu", min_local=12)

    def test_experts_are_the_rank_file_layout(self):
        torch, c, e = self.torch, self.c, self.experts
        self.assertEqual((e.w13.shape, e.w13.dtype), ((c.local, 2 * c.inter, c.hidden // 2), torch.uint8))
        self.assertEqual((e.w2.shape, e.w2.dtype), ((c.local, c.hidden, c.inter // 2), torch.uint8))
        # specs._routed_specs: scales [E, 2I * H/16] and [E, H * I/16], tile-interleaved at 128 x 4
        self.assertEqual((e.w13_sf.shape, e.w13_sf.dtype), ((c.local, 128 * 4), torch.float8_e4m3fn))
        self.assertEqual((e.w2_sf.shape, e.w2_sf.dtype), ((c.local, 128 * 4), torch.float8_e4m3fn))
        s = e.scales
        for value in (s.weight13, s.input13, s.weight2, s.input2):
            self.assertEqual((value.shape, value.dtype), ((c.local,), torch.float32))
            self.assertTrue(bool((value > 0).all()))
        self.assertTrue(torch.equal(s.alpha13, s.weight13 * s.input13))
        self.assertGreater(len(set(s.weight13.tolist())), 1)                            # the experts differ
        up, gate, down = e.dense(1)
        self.assertEqual((up.shape, gate.shape, down.shape), ((c.inter, c.hidden), (c.inter, c.hidden), (c.hidden, c.inter)))
        self.assertTrue(all(bool(torch.isfinite(t).all()) and float(t.abs().max()) > 0 for t in (up, gate, down)))

    def test_the_oracle_is_the_reference_lane_on_this_ranks_pairs(self):
        """With exact division as its quantiser the oracle is lanes.reference()'s moe (engine/modules/moe.expert_gemm)
        over this rank's pairs, byte for byte: on the GPU only the quantiser is swapped for the kernels' reciprocal one."""
        from engine.modules import moe
        torch, p, c, e = self.torch, self.p, self.c, self.experts
        got = p.oracle(self.x, self.ids, self.w, e, c, moe.quant_nvfp4_act)
        want = p.reference_partial(self.x, self.ids, self.w, e, c, _reference())
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertGreater(int(torch.count_nonzero(got)), self.x.numel() // 2)
        self.assertTrue(torch.equal(got, want))

    def test_the_oracle_rounds_the_swiglu_output_before_fc2(self):
        """The kernels store silu(gate) * up as BF16 (their sC stage) and quantise that for FC2; quantising the FP32 value
        instead picks other FP4 bytes at rounding thresholds -- percent-level, the size of the 2% gate."""
        source = PROBE.read_text()
        self.assertIn("gemm((silu(gate) * up).bfloat16().float(), down_w, s.input2[e])", source)
        kernel = (ROOT / "engine/kernels/b12x/moe_micro_kernel.py").read_text()
        self.assertIn("acc_vec = acc_vec.to(cutlass.BFloat16)", kernel)
        self.assertIn("cvt.rn.satfinite.bf16x2.f32", kernel)

    def test_routes_to_other_ranks_contribute_nothing(self):
        from engine.modules import moe
        torch, p, c, e = self.torch, self.p, self.c, self.experts
        quant = moe.quant_nvfp4_act
        foreign = p.foreign_routes(self.ids, c)
        self.assertEqual(int(torch.count_nonzero(p.oracle(self.x, foreign, self.w, e, c, quant))), 0)
        mine, _ = p.local_mask(self.ids, c)
        self.assertTrue(torch.equal(p.oracle(self.x, self.ids, self.w * mine, e, c, quant),
                                    p.oracle(self.x, self.ids, self.w, e, c, quant)))
        self.assertEqual(int(torch.count_nonzero(p.reference_partial(self.x, foreign, self.w, e, c, _reference()))), 0)

    def test_the_reference_lane_on_pairs_is_the_reference_lane(self):
        """reference_partial dispatches the pairs one route a row; the reference lane over the whole routes visits the
        same pairs, so the two differ only by where BF16 rounds a route's product (before or after the sum)."""
        torch, p, c, e = self.torch, self.p, self.c, self.experts
        pairs = p.reference_partial(self.x, self.ids, self.w, e, c, _reference())
        whole = _reference().moe(self.x, self.ids, self.w, e.w13, e.w13_sf, e.w2, e.w2_sf, scales=e.scales,
                                 first_expert=c.first_expert)
        self.assertEqual(pairs.dtype, torch.bfloat16)
        self.assertLessEqual(p.relative(pairs, whole), 0.02)
        touched = lambda out: int(torch.count_nonzero(out.float().abs().sum(dim=1)))
        self.assertEqual(touched(pairs), touched(whole))

    def test_the_bucketed_quantiser_is_the_plain_one(self):
        from engine.modules import moe
        torch, p = self.torch, self.p
        scale = self.experts.scales.input13[0]
        for rows in (1, 5, 8, 13):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 64, generator=torch.Generator().manual_seed(rows)) * 0.5
                packed, sf = p.bucketed(moe.quant_nvfp4_act)(x, scale)
                want_packed, want_sf = moe.quant_nvfp4_act(x, scale)
                self.assertEqual((packed.shape, sf.shape), (want_packed.shape, want_sf.shape))
                self.assertTrue(torch.equal(packed.view(torch.uint8), want_packed.view(torch.uint8)))
                self.assertTrue(torch.equal(sf.view(torch.uint8), want_sf.view(torch.uint8)))


_REFERENCE = []


def _reference():
    if not _REFERENCE:
        from engine.profiles.qwen38 import lanes
        _REFERENCE.append(lanes.reference())
    return _REFERENCE[0]


@needs_torch
class ComparisonTests(unittest.TestCase):
    def test_relative_and_bf16_distance(self):
        import torch
        p = probe()
        b = torch.tensor([1.0, -2.0, 4.0])
        self.assertAlmostEqual(p.relative(torch.tensor([1.0, -2.0, 4.08]), b), 0.02, places=6)
        x = torch.tensor([1.0, -1.5, 0.0, 3.0], dtype=torch.bfloat16)
        up = (x.view(torch.int16) + torch.tensor([1, 1, 0, 2], dtype=torch.int16)).view(torch.bfloat16)
        self.assertEqual(p.bf16_ulps(x, x), 0)
        self.assertEqual(p.bf16_ulps(up[:2], x[:2]), 1)                     # one step up in magnitude, either sign
        self.assertEqual(p.bf16_ulps(up, x), 2)
        self.assertEqual(p.bf16_ulps(torch.tensor([-0.0]), torch.tensor([0.0])), 0)
        # stable: element by element, one adjacent BF16 value or 0.1% of the largest magnitude
        b = torch.tensor([64.0, 1e-3, -1e-3], dtype=torch.bfloat16)
        step = lambda t, i, k: torch.cat([t[:i], (t[i:i + 1].view(torch.int16) + k).view(torch.bfloat16), t[i + 1:]])
        flipped = torch.tensor([0.0, 0.0, 2e-3], dtype=torch.bfloat16)
        one_up_and_a_sign = step(b, 0, 1) + flipped                  # 0.78% of the largest, and thousands of values
        self.assertTrue(p.stable(one_up_and_a_sign, b))
        self.assertGreater(p.relative(one_up_and_a_sign, b), 0.001)   # each measure alone says otherwise
        self.assertGreater(p.bf16_ulps(one_up_and_a_sign, b), 1)
        self.assertFalse(p.stable(step(b, 0, 2), b))                   # two values at the largest: 1.6%
        self.assertTrue(p.stable(step(b, 1, 40), b))                   # far in values, 0.06 of 64 in magnitude
        self.assertFalse(p.stable(b + torch.tensor([0.0, 0.07, 0.0], dtype=torch.bfloat16), b))
        self.assertTrue(p.stable(b, b))

    def test_the_fastest_exact_choice(self):
        p = probe()
        row = lambda tokens, tile, cold, exact=True: dict(tokens=tokens, tile_m=tile, cold_us=cold, exact=exact)
        rows = [row(1024, 16, 9.0, exact=False), row(1024, 32, 10.0), row(1024, 64, 12.0),
                row(4096, 32, 40.0), row(4096, 64, 30.0), row(4096, 128, 20.0)]
        self.assertEqual(p.fastest(rows[:3])["tile_m"], 32)
        self.assertIsNone(p.fastest(rows[:1]))
        # 32: (1 + 2) / 2 = 1.5; 64: (1.2 + 1.5) / 2 = 1.35; 128 is not timed at 1024
        self.assertEqual(p.overall_tile(rows), 64)
        self.assertIsNone(p.overall_tile([]))
        self.assertEqual(p.samples_summary([3.0, 1.0, 2.0], [5.0, 4.0, 6.0]),
                         dict(cold_us=2.0, warm_us=5.0, cold_min_us=1.0, warm_min_us=4.0, samples=3))


class DispatchMirrorTests(unittest.TestCase):
    def test_the_timed_call_is_the_lanes_b12x_call(self):
        """probes/engine_qwen38_moe.dispatch times the call lanes.served()'s moe makes: the same keywords, the same
        constants (the tensors are the same roles under the probe's names)."""
        def call(path, function):
            tree = ast.parse(path.read_text())
            owner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
            return next(n for n in ast.walk(owner)
                        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "b12x_fused_moe")
        lane, timed = call(LANES, "served"), call(PROBE, "dispatch")
        names = lambda c: {k.arg for k in c.keywords}
        constants = lambda c: {k.arg: k.value.value for k in c.keywords if isinstance(k.value, ast.Constant)}
        self.assertEqual(names(timed), names(lane))
        self.assertEqual(constants(timed), constants(lane))
        self.assertEqual(constants(lane)["activation"], "silu")


class RepeatDiagnosisTests(unittest.TestCase):
    """The first lane run (measurements/qwen38_lane_20260917) stopped at the eager prefill check: two identical calls
    parted on one element by 0.0039. The check now records the parting with where it happened and goes on."""

    @needs_torch
    def test_the_diagnosis_names_the_call_or_the_sum(self):
        import torch
        from unittest import mock
        p = probe()
        c = p.cell_of(p.kernel_shape())
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(6, 8, generator=gen).bfloat16()
        token = torch.tensor([0, 0, 2, 3, 5])
        pairs = (token, x.index_select(0, token), torch.zeros(5, 1, dtype=torch.int32), torch.ones(5, 1))
        outputs = torch.randn(5, 8, generator=gen).bfloat16()
        calls = iter([outputs.clone(), outputs.clone(), outputs.clone(), outputs.clone()])
        with mock.patch.object(p, "pairs_of", return_value=pairs), \
                mock.patch.object(p, "dispatch", side_effect=lambda *a, **k: next(calls)):
            steady = p.repeat_diagnosis(x, None, None, None, None, c)
        self.assertEqual((steady["call"]["rows"], steady["index_add"]["rows"]), (0, 0))
        self.assertEqual((steady["pairs"], steady["tokens_with_several_routes"]), (5, 1))
        parted = outputs.clone()
        parted[2, 3] = parted[2, 3] + 0.25
        calls = iter([outputs.clone(), parted, outputs.clone(), outputs.clone()])
        with mock.patch.object(p, "pairs_of", return_value=pairs), \
                mock.patch.object(p, "dispatch", side_effect=lambda *a, **k: next(calls)):
            found = p.repeat_diagnosis(x, None, None, None, None, c)
        self.assertEqual((found["call"]["rows"], found["call"]["first_rows"]), (1, [2]))
        self.assertGreater(found["call"]["ulps"], 1)
        self.assertEqual(found["index_add"]["rows"], 0)                  # one call's outputs summed alike every time

    def test_a_parted_repeat_is_recorded_not_raised(self):
        source = PROBE.read_text()
        check = source[source.index("    def prefill_check("):source.index("    def prefill_sweep(")]
        self.assertNotIn("eager repeats differ", check)
        self.assertIn('row["unstable"] = not row["repeat_stable"]', check)
        self.assertIn("repeat_diagnosis(", check)
        self.assertIn("prefill_unstable=self.unstable", source)


class LaneRoutingTests(unittest.TestCase):
    def test_kernel_check_routes_the_lane_to_the_probe(self):
        text = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_moe'", text)
        self.assertIn("from probes.engine_qwen38_moe import run as qwen38_moe", text)


if __name__ == "__main__":
    unittest.main()
