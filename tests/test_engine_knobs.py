"""Native production defaults are fixed; only unqualified experiments expire."""
import datetime
import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class KnobDeclarationTests(unittest.TestCase):
    def _declared(self, env, *, production=False, today=datetime.date(2026, 9, 12)):
        from engine.profiles.glm53 import boot
        from engine.base.config import Config
        args = types.SimpleNamespace(production=production, ckpt_meta="/meta", ranks="/ranks", kv_gib=8.73, port=8000)
        with patch.object(boot, "Config", side_effect=lambda facts, knobs: Config(facts, knobs, env=env, today=today)):
            return boot.declared(args, 4)

    def test_production_remains_restartable_after_experiment_expiry(self):
        cfg = self._declared({}, production=True, today=datetime.date(2040, 1, 1))
        self.assertFalse(cfg.knobs)
        self.assertEqual([cfg[k] for k in ("moe_static", "mla_prefill", "context_ceiling", "lanes", "decode_eager", "execution")],
                         ["t,r,sf6,q0", "tile32", 0, "served", 0, "native"])

    def test_adopted_execution_cannot_be_changed_by_stale_bisect_environment(self):
        from engine.base.config import ConfigError
        for production in (False, True):
            for name, value in (("execution", "stock"), ("moe_static", "stock"),
                                ("lanes", "reference"), ("decode_eager", "1")):
                with self.subTest(production=production, knob=name), self.assertRaises(ConfigError):
                    self._declared({"STK_"+name:value}, production=production)

    def test_only_declared_precision_mla_context_and_execution_experiments_remain(self):
        cfg = self._declared({"STK_mla_prefill":"stock", "STK_context_ceiling":"131072"})
        self.assertEqual(set(cfg.knobs), {"mla_prefill", "context_ceiling", "kda_state_dtype",
                                          "execution_overlap", "early_observe", "prefill_tiles", "direct_mhc", "prefill_project_tiles",
                                          "nvme_mapped_staging", "decode_iterations"})
        self.assertEqual((cfg["mla_prefill"], cfg["context_ceiling"]), ("stock", 131072))
        self.assertEqual((cfg["execution"], cfg["moe_static"]), ("native", "t,r,sf6,q0"))
        from engine.base.config import ConfigError
        with self.assertRaises(ConfigError):
            self._declared({"STK_mla_prefill":"stock"}, production=True)

    def test_gb10_serving_defaults_are_on_and_production_refuses_overrides(self):
        from engine.base.config import ConfigError
        for production in (False, True):
            cfg = self._declared({}, production=production)
            self.assertEqual([cfg[k] for k in ("execution_overlap", "early_observe", "prefill_tiles", "direct_mhc", "prefill_project_tiles", "nvme_mapped_staging", "decode_iterations")], [0, 0, 1, 1, 1, 1, 4])
            self.assertEqual(cfg["kda_state_dtype"], "fp32")
        rollback = {"direct_mhc": 0, "prefill_project_tiles": 0,
                    "nvme_mapped_staging": 0, "decode_iterations": 1}
        cfg = self._declared({"STK_"+key: str(value) for key, value in rollback.items()})
        self.assertEqual({key: cfg[key] for key in rollback}, rollback)
        for key in ("execution_overlap", "early_observe", "prefill_tiles", "direct_mhc", "prefill_project_tiles", "nvme_mapped_staging", "decode_iterations"):
            with self.subTest(key=key), self.assertRaises(ConfigError):
                self._declared({"STK_"+key:"1"}, production=True)


class MoeStaticSpecTests(unittest.TestCase):
    def test_parse_moe_static(self):
        from engine.profiles.glm53.lanes import MOE_STATIC_PRODUCTION, parse_moe_static
        self.assertEqual(parse_moe_static("stock"), (None, False))
        self.assertEqual(parse_moe_static("0"), (None, False))
        self.assertEqual(parse_moe_static(MOE_STATIC_PRODUCTION), ("t,r,sf6", True))
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

    def test_the_timing_cells_reach_the_served_recipe_but_never_serving(self):
        """xs/xa drop a TMA issue at compile time, which is the only way to ask which of the served recipe's
        boxes holds its fixed cost. They used to be excluded from `r` and `sf6` and so could not be pointed
        at what production runs (45차 §23 조사 17차)."""
        import importlib
        try:
            md = importlib.import_module("engine.kernels.b12x.moe_dispatch")
        except ImportError as exc:
            self.skipTest(f"b12x dispatcher unavailable here: {exc}")
        for cells, skip_sf, skip_a in (("xs", True, False), ("xa", False, True), ("xs,xa", True, True)):
            cfg = md._parse_glm53_static_v2(f"t,r,sf6,{cells}", probe=True)
            self.assertEqual((cfg["skip_sf"], cfg["skip_a"]), (skip_sf, skip_a), cells)
            self.assertTrue(cfg["tiled"] and cfg["decode_reform"] and cfg["reform_sf_pack"], cells)
            with self.assertRaisesRegex(ValueError, "probe-only"):
                md._parse_glm53_static_v2(f"t,r,sf6,{cells}", probe=False)
        for cells in ("q", "v"):                          # the cells that do change the scale path stay out of r
            with self.assertRaisesRegex(ValueError, "r requires t"):
                md._parse_glm53_static_v2(f"t,r,sf6,{cells}", probe=True)

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
        self.assertEqual(mk.PREFILL_MODES, ("stock", "tile32"), "the union candidates were measured and retired")
        self.assertFalse(mk.ENABLE_MLA_PREFILL32)
        try:
            mk.configure_prefill("stock")
            self.assertFalse(mk.ENABLE_MLA_PREFILL32)
            mk.configure_prefill("tile32")
            self.assertTrue(mk.ENABLE_MLA_PREFILL32)
            for gone in ("pair", "pair4"):
                with self.assertRaises(ValueError):
                    mk.configure_prefill(gone)
            with self.assertRaises(ValueError):
                mk.configure_prefill("4")
            mk._ARMED["mla"] = True
            with self.assertRaisesRegex(RuntimeError, "already armed"):
                mk.configure_prefill("stock")
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
