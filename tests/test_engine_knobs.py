"""CPU checks for the D11 knob sunset (2026-09-12): the kernel package reads no
environment; the profile declares the two remaining axes (STK_moe_static,
STK_mla_prefill) and applies them once, before anything binds or arms."""
import datetime
import importlib
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class KnobDeclarationTests(unittest.TestCase):
    def _declared(self, env):
        from engine.base.config import Config, ConfigError, Fact, Knob
        from engine.profiles.glm53 import lanes
        # the profile's declaration, spelled here without importing boot (torch/CUDA-heavy imports)
        knobs = [Knob("moe_static", lanes.MOE_STATIC_STOCK, datetime.date(2026, 9, 30), "b12x static lane", "stock"),
                 Knob("mla_prefill", "stock", datetime.date(2026, 9, 30), "MLA prefill", "stock")]
        return Config([Fact("world", 4, "facts.TP")], knobs, env=env, today=datetime.date(2026, 9, 12)), ConfigError

    def test_defaults_are_the_judged_stock_paths(self):
        cfg, _ = self._declared({})
        self.assertEqual((cfg["moe_static"], cfg["mla_prefill"]), ("stock", "stock"))

    def test_env_selects_the_production_candidate_and_undeclared_dies(self):
        cfg, ConfigError = self._declared({"STK_moe_static": "t,r,sf6,q0", "STK_mla_prefill": "pair"})
        self.assertEqual((cfg["moe_static"], cfg["mla_prefill"]), ("t,r,sf6,q0", "pair"))
        self.assertEqual(cfg.overridden, ["mla_prefill", "moe_static"])
        with self.assertRaises(ConfigError):
            self._declared({"STK_B12X_STATIC_V2": "t"})

    def test_boot_declares_exactly_these_knobs(self):
        src = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        self.assertIn('Knob("moe_static"', src)
        self.assertIn('Knob("mla_prefill"', src)
        self.assertEqual(src.count("Knob("), 2)
        self.assertIn('lane_tables.served(moe_static=cfg["moe_static"], mla_prefill=cfg["mla_prefill"])', src)


class MoeStaticSpecTests(unittest.TestCase):
    def test_parse_moe_static(self):
        from engine.profiles.glm53.lanes import MOE_STATIC_PRODUCTION, parse_moe_static
        self.assertEqual(parse_moe_static("stock"), (None, False))
        self.assertEqual(parse_moe_static("0"), (None, False))
        self.assertEqual(parse_moe_static(MOE_STATIC_PRODUCTION), ("t,r,sf6", False))
        self.assertEqual(parse_moe_static("t,r,sf6,q0"), ("t,r,sf6", True))
        with self.assertRaisesRegex(ValueError, "q0 needs"):
            parse_moe_static("u,q0")

    def test_dispatcher_spec_is_applied_once_and_refused_after_views_exist(self):
        try:
            md = importlib.import_module("engine.kernels.b12x.moe_dispatch")
        except ImportError as exc:                      # CuTe DSL is an image dependency
            self.skipTest(f"b12x dispatcher unavailable here: {exc}")
        self.assertIsNone(md._GLM53_B12X_STATIC_V2)      # import-time default: the stock static kernel
        self.assertFalse(md._TP_SF6_Q0_ENABLED)
        cfg = md.configure_static_v2("t,r,sf6")
        try:
            self.assertTrue(cfg["tiled"] and cfg["decode_reform"] and cfg["reform_sf_pack"])
            md.configure_tp_sf6_q0(True)
            self.assertTrue(md._TP_SF6_Q0_ENABLED)
            md._WEIGHT_CACHE["sentinel"] = ()
            with self.assertRaisesRegex(RuntimeError, "already exist"):
                md.configure_static_v2(None)
            self.assertIs(md.configure_static_v2("t,r,sf6"), cfg)   # same spec: a no-op, never a rebuild
        finally:
            md._WEIGHT_CACHE.pop("sentinel", None)
            md.configure_tp_sf6_q0(False)
            md.configure_static_v2(None)
        with self.assertRaisesRegex(ValueError, "q0 needs"):
            md.configure_tp_sf6_q0(True)
        with self.assertRaises(ValueError):
            md.configure_static_v2("1")                  # the sunset v2 lane token is rejected, not remapped

    def test_tile_major_relayout_reads_back_row_major(self):
        try:
            md = importlib.import_module("engine.kernels.b12x.moe_dispatch")
        except ImportError as exc:
            self.skipTest(f"b12x dispatcher unavailable here: {exc}")
        from engine.modules.expert_layout import W13_K_IN_BYTES, W2_K_IN_BYTES, row_major_expert
        torch.manual_seed(1)
        w13 = torch.randint(0, 256, (2, 1024, 2048), dtype=torch.uint8)   # GLM TP4: [E, 2I, H/2]
        w2 = torch.randint(0, 256, (2, 4096, 256), dtype=torch.uint8)     # [E, H, I/2]
        keep13, keep2 = w13.clone(), w2.clone()
        md.tile_expert_weights_inplace(w13, w2)
        self.assertEqual(getattr(w13, md._TILE_MAJOR_ATTR), "plain")
        self.assertFalse(torch.equal(w13, keep13))                          # the bytes moved ...
        for e in range(2):                                                 # ... and the reference view undoes it exactly
            self.assertTrue(torch.equal(row_major_expert(w13, e, W13_K_IN_BYTES), keep13[e]))
            self.assertTrue(torch.equal(row_major_expert(w2, e, W2_K_IN_BYTES), keep2[e]))
        md.tile_expert_weights_inplace(w13, w2)                            # idempotent: a second call is a no-op
        self.assertTrue(torch.equal(row_major_expert(w13, 1, W13_K_IN_BYTES), keep13[1]))


class MlaPrefillModeTests(unittest.TestCase):
    def test_configure_prefill_modes(self):
        from engine.kernels import mla as mk
        self.assertEqual((mk.ENABLE_MLA_PREFILL32, mk.ENABLE_MLA_PREFILL_PAIR, mk.MLA_PREFILL_GROUP), (False, False, 2))
        try:
            mk.configure_prefill("pair4")
            self.assertEqual((mk.ENABLE_MLA_PREFILL32, mk.ENABLE_MLA_PREFILL_PAIR, mk.MLA_PREFILL_GROUP), (False, True, 4))
            mk.configure_prefill("tile32")
            self.assertEqual((mk.ENABLE_MLA_PREFILL32, mk.ENABLE_MLA_PREFILL_PAIR, mk.MLA_PREFILL_GROUP), (True, False, 2))
            with self.assertRaises(ValueError):
                mk.configure_prefill("4")
            mk._ARMED["mla"] = True
            with self.assertRaisesRegex(RuntimeError, "already armed"):
                mk.configure_prefill("pair")
            mk.configure_prefill("tile32")               # the armed mode itself stays selectable
        finally:
            mk._ARMED["mla"] = False
            mk.configure_prefill("stock")


class ProbeHookTests(unittest.TestCase):
    """What left the environment but stayed as an explicit probe instrument."""

    def test_dynamic_tile_m_override_is_validated(self):
        try:
            md = importlib.import_module("engine.kernels.b12x.moe_dispatch")
        except ImportError as exc:
            self.skipTest(f"b12x dispatcher unavailable here: {exc}")
        self.assertIsNone(md._DYNAMIC_TILE_M_OVERRIDE)
        try:
            md._DYNAMIC_TILE_M_OVERRIDE = 32
            self.assertEqual(md._select_dynamic_tile_m(4096, 288, "swigluoai_uninterleave"), 32)
            md._DYNAMIC_TILE_M_OVERRIDE = 48
            with self.assertRaises(ValueError):
                md._select_dynamic_tile_m(4096, 288, "swigluoai_uninterleave")
        finally:
            md._DYNAMIC_TILE_M_OVERRIDE = None

    def test_mla_probe_instruments_are_arguments_not_env(self):
        import inspect
        from engine.kernels import mla as mk
        self.assertFalse(mk.PAIR_STATS)
        self.assertIn("forced", inspect.signature(mk.mla_splits).parameters)
        self.assertEqual(mk.mla_splits(8, forced=3), 1)            # no extension built here: the rule is inert
        params = inspect.signature(mk.mla_decode).parameters
        self.assertTrue(params["splits"].kind is inspect.Parameter.KEYWORD_ONLY and params["probe"].default == 0)
        cu = (ROOT / "engine/kernels/mla/glm53_megakernel.cu").read_text()
        self.assertIn("a.probe = ints.size() > 3 ? (int)ints[3] : 0;", cu)

    def test_mhc_pass_hook_is_fixed_before_the_kernels_compile(self):
        import engine.kernels as kernels
        self.assertIsNone(kernels.MHC_PASSES)
        name = "engine.kernels.mhc.tilelang_kernels"
        loaded = sys.modules.pop(name, None)
        try:
            kernels.configure_mhc_passes(True, False)
            self.assertEqual(kernels.MHC_PASSES, (True, False))
            sys.modules[name] = types.ModuleType(name)
            with self.assertRaisesRegex(RuntimeError, "already compiled"):
                kernels.configure_mhc_passes(False, False)
        finally:
            sys.modules.pop(name, None)
            if loaded is not None:
                sys.modules[name] = loaded
            kernels.MHC_PASSES = None
        src = (ROOT / "engine/kernels/mhc/tilelang_kernels.py").read_text()
        self.assertIn("_DENEB_MHC_PASSES = _kernels_pkg.MHC_PASSES", src)


class ProbeRunnerTests(unittest.TestCase):
    def test_probe_runner_and_launcher_forward_declared_knobs(self):
        self.assertIn("compgen -v STK_", (ROOT / "probes/run_engine_probe.sh").read_text())
        self.assertIn("compgen -v STK_", (ROOT / "launchers/start-st-glm53.sh").read_text())


if __name__ == "__main__":
    unittest.main()
