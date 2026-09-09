"""CPU-only latch and two-level cache isolation for same-build SF6 A/B."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_moe"
DISPATCH = ast.parse((MODULE / "moe_dispatch.py").read_text())


def definitions(tree, names, env):
    nodes = [copy.deepcopy(node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "actual-sf6-unpack-controls", "exec"), env)
    return env


def cache_namespace():
    return definitions(DISPATCH, {"_disk_kernel_name", "_static_kernel_cache_key",
        "_static_v2_cache_key", "_static_v2_decode_config", "_dynamic_kernel_cache_key",
        "_get_static_kernel_v2", "_get_dynamic_kernel"}, dict(
        hashlib=hashlib, torch=SimpleNamespace(int32="i32", int64="i64", device=lambda value:value),
        _LEVEL_TILE_M=128, _MAX_SHARED_INPUT_TOPK=32,
        _normalize_activation_precision=lambda value:value,
        _normalize_quant_mode=lambda value, precision:value,
        _sf_params_for_quant_mode=lambda mode:(16, "sf8"),
        get_num_sm=lambda device:48, get_max_active_clusters=lambda size:48,
        _level_tile_n=lambda precision:128, _DYNAMIC_MAC_LADDER=(),
        _GLM53_B12X_DYNAMIC_MAC_LADDER=None, _lookup_mac_ladder=lambda ladder, rows:None,
        _GLM53_B12X_PREFILL_REUSE=False, _GLM53_B12X_PREFILL_FC1_N128=False,
        is_gated_activation=lambda value:True, _STATIC_V2_KERNEL_CACHE={},
        _DYNAMIC_KERNEL_CACHE={}))


def static_fields(m=6):
    return dict(activation_precision="fp4", quant_mode="nvfp4", state_E=288,
        weight_E=288, m=m, k=4096, n=512, num_topk=8, max_rows=128, mac=48,
        mma_tiler_mn=(16 if m <= 8 else 32, 128), topk_ids_dtype="i32",
        input_scales_are_reciprocal=False, fast_math=True, activation="silu",
        swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=None)


def dynamic_fields():
    return dict(activation_precision="fp4", quant_mode="nvfp4", E=288,
        k=4096, n=512, num_topk=8, mac=48, mma_tiler_mn=(128,128),
        topk_ids_dtype="i32", input_scales_are_reciprocal=False, fast_math=True,
        activation="silu", swiglu_alpha=1.702, swiglu_beta=1.0,
        swiglu_limit=None, share_input_across_experts=False, tiled=True,
        reform_sf_pack=True)


def config():
    return dict(tile_m=32, fc1=2, fc2=2, a_rows=32, stamps=False,
                tiled=True, decode_reform=True, reform_sf_pack=True)


class UnpackControls(unittest.TestCase):
    def test_default_and_exact_values_are_latched_once(self):
        common = ast.parse((MODULE / "moe_static_common.py").read_text())
        assignment = next(node for node in common.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_SF6_UNPACK_U8X4" for t in node.targets))
        for raw, expected in ((None, True), ("1", True), ("0", False)):
            values = {} if raw is None else {"VLLM_GLM53_SF6_UNPACK_U8X4": raw}
            env = definitions(common, {"_parse_sf6_unpack_u8x4"},
                              dict(os=SimpleNamespace(environ=values)))
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), "actual-latch", "exec"), env)
            self.assertIs(env["_SF6_UNPACK_U8X4"], expected)
            values["VLLM_GLM53_SF6_UNPACK_U8X4"] = "0" if expected else "1"
            self.assertIs(env["_SF6_UNPACK_U8X4"], expected)
        for raw in ("", "true", "false", "off", "01", "2", " 1", "0 "):
            with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                env["_parse_sf6_unpack_u8x4"](raw)

    def test_static_sf6_and_legacy_q_keys_separate_both_artifact_levels(self):
        ns = cache_namespace()
        for m in (2, 6, 8, 16):
            for cfg in (config(), dict(config(), reform_sf_pack=False, sf_pack=True)):
                effective = ns["_static_v2_decode_config"](cfg, m)
                keys = [ns["_static_v2_cache_key"](effective, sf6_unpack_u8x4=mode,
                                                   **static_fields(m)) for mode in (False, True)]
                self.assertNotEqual(*keys)
                self.assertNotEqual(*(ns["_disk_kernel_name"]("same-static-shape", key) for key in keys))
                self.assertEqual(len(dict(zip(keys, ("scalar", "vector")))), 2)
        raw = dict(config(), reform_sf_pack=False)
        self.assertEqual(ns["_static_v2_cache_key"](raw, sf6_unpack_u8x4=False, **static_fields()),
                         ns["_static_v2_cache_key"](raw, sf6_unpack_u8x4=True, **static_fields()))

    def test_dynamic_sf6_keys_separate_both_artifact_levels_only_when_used(self):
        ns = cache_namespace()
        for sf6 in (False, True):
            fields = dict(dynamic_fields(), reform_sf_pack=sf6)
            keys = [ns["_dynamic_kernel_cache_key"](**fields, sf6_unpack_u8x4=mode)
                    for mode in (False, True)]
            names = [ns["_disk_kernel_name"]("same-dynamic-shape", key) for key in keys]
            self.assertEqual(keys[0] == keys[1], not sf6)
            self.assertEqual(names[0] == names[1], not sf6)

    def test_actual_getters_retrieve_only_the_current_latched_specialization(self):
        ns = cache_namespace()
        for mode in (False, True):
            ns["_SF6_UNPACK_U8X4"] = mode
            for cached_mode in (False, True):
                key = ns["_static_v2_cache_key"](config(), sf6_unpack_u8x4=cached_mode,
                                                  **static_fields())
                ns["_STATIC_V2_KERNEL_CACHE"][key] = ("static", cached_mode)
                key = ns["_dynamic_kernel_cache_key"](**dynamic_fields(), sf6_unpack_u8x4=cached_mode)
                ns["_DYNAMIC_KERNEL_CACHE"][key] = ("dynamic", cached_mode)
            self.assertEqual(ns["_get_static_kernel_v2"](288, 288, 6, 4096, 512, 8, 128,
                                                        config=config()), ("static", mode))
            self.assertEqual(ns["_get_dynamic_kernel"](288, 3456, 4096, 512, 8, 27648,
                                tiled=True, reform_sf_pack=True), ("dynamic", mode))

    def test_compiler_objects_and_persistent_names_receive_the_same_mode(self):
        for name in ("_get_static_kernel_v2", "_get_dynamic_kernel"):
            fn = next(n for n in DISPATCH.body if isinstance(n, ast.FunctionDef) and n.name == name)
            # The getter must pass its one module latch to BOTH the key and
            # the compiler object, not merely rename one cache level.
            mode_values = [kw.value for n in ast.walk(fn) if isinstance(n, ast.Call)
                           for kw in n.keywords if kw.arg == "sf6_unpack_u8x4"]
            self.assertEqual(len(mode_values), 2)
            self.assertTrue(all(isinstance(v, ast.Name) and v.id == "_SF6_UNPACK_U8X4"
                                for v in mode_values))
            build = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "build_and_load_cute_dsl_kernel")
            self.assertEqual(ast.unparse(build.args[1]), "artifact_name")
            artifacts = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "artifact_name" for t in n.targets)]
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(ast.unparse(artifacts[0].value.args[-1]), "cache_key")


if __name__ == "__main__":
    unittest.main()
