"""Exact EP micro tile admission and construction; no accelerator imports.

The actual dispatcher and constructor run with tensor/compiler I/O replaced.
This checks the CPU call contract, not CuTe lowering or device numerics.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "overlay/modules/glm53_moe/moe_dispatch.py"
MICRO = ROOT / "overlay/modules/glm53_moe/moe_micro_kernel.py"
EXACT = dict(state_E=72, weight_E=72, m=8, k=4096, n=2048, num_topk=8,
             skip_zero_weight_expert_id=72, quant_mode="nvfp4",
             activation="swigluoai_uninterleave", swiglu_alpha=1.0,
             swiglu_beta=0.0, swiglu_limit=10.0)


def load_actual_functions(path, names, namespace):
    selected = [copy.deepcopy(node) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("missing actual-source function")
    module = ast.Module(body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
                             *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def load_actual_constructor(namespace):
    source_class = next(node for node in ast.parse(MICRO.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == "MoEMicroKernel")
    init = copy.deepcopy(next(node for node in source_class.body
                              if isinstance(node, ast.FunctionDef) and node.name == "__init__"))
    setup = copy.deepcopy(next(node for node in source_class.body
                               if isinstance(node, ast.FunctionDef) and node.name == "_setup_attributes"))
    # Preserve the source prefix through MMA integer geometry. The imported
    # layout backend is supplied by the harness; later SMEM lowering needs the
    # real CuTe compiler and is deliberately outside this CPU oracle.
    end = next(i for i, node in enumerate(setup.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "num_k_blocks"
                       for t in node.targets))
    setup.body = [node for node in setup.body[:end + 1]
                  if not isinstance(node, (ast.Import, ast.ImportFrom))]
    cls = ast.ClassDef(name="MoEMicroKernel", bases=[], keywords=[],
                       body=[init, setup], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
                             cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MICRO), "exec"), namespace)


class Harness:
    def __init__(self):
        self.compiles = []
        cutlass = SimpleNamespace(**{name: name for name in (
            "Float4E2M1FN", "BFloat16", "Float32", "Int32", "Int64", "Uint8")})
        cute = SimpleNamespace(
            AddressSpace=SimpleNamespace(gmem="gmem"),
            runtime=SimpleNamespace(
                make_fake_compact_tensor=lambda dtype, shape, **kw: SimpleNamespace(
                    dtype=dtype, shape=shape, **kw),
                make_fake_stream=lambda **kw: SimpleNamespace(**kw)),
            compile=self.compile,
            make_layout=lambda shape: shape,
            make_tiled_mma=lambda op, layout, **kw: (op, layout, kw),
            make_mma_atom=lambda op: op,
            nvgpu=SimpleNamespace(warp=SimpleNamespace(
                MmaMXF4Op=lambda *args: args, MmaMXF4NVF4Op=lambda *args: args)),
        )
        self.ns = dict(
            torch=SimpleNamespace(int32="int32", device=lambda name: name),
            cutlass=cutlass, cute=cute,
            get_num_sm=lambda _: 48, get_max_active_clusters=lambda _: 48,
            _normalize_quant_mode=lambda mode: mode,
            _sf_params_for_quant_mode=lambda mode: (16, "e4m3"),
            _MICRO_KERNEL_CACHE={}, _CUTE_DSL_MODULE="actual-source-test",
            _align_up=lambda value, alignment: (value + alignment - 1) // alignment * alignment,
            make_ptr=lambda *args, **kw: SimpleNamespace(args=args, **kw),
            build_and_load_cute_dsl_kernel=lambda module, name, build, **kw: build(),
            _disk_kernel_name=lambda prefix, key: (prefix, key),
            _kernel_source_files=lambda: (), DenseGemmKernel=object,
            is_gated_activation=lambda activation: activation in ("silu", "swigluoai_uninterleave"),
            utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda _: 101376),
            pipeline=SimpleNamespace(NamedBarrier=lambda **kw: SimpleNamespace(**kw)),
            sm120_utils=SimpleNamespace(get_permutation_mnk=lambda *args: args),
        )
        load_actual_constructor(self.ns)
        load_actual_functions(DISPATCH, {"_select_moe_mma_tiler_mn", "_select_micro_mma_tiler_mn",
                                         "_ep_micro_scatter_fp32", "_ep_micro_direct_scatter", "_micro_kernel_cache_key",
                                         "_get_micro_kernel"}, self.ns)

    def compile(self, kernel, *args, **kwargs):
        self.compiles.append((kernel, args, kwargs))
        return object()

    def select(self, **changes):
        return self.ns["_select_micro_mma_tiler_mn"](**(EXACT | changes))

    def get(self, **changes):
        return self.ns["_get_micro_kernel"](**(EXACT | dict(max_rows=64, mac_override=48) | changes))


class EPMicroTileTests(unittest.TestCase):
    def test_only_exact_geometry_overrides_default(self):
        harness = Harness()
        fallback_calls = []
        def fallback(rows, n):
            fallback_calls.append((rows, n))
            return (128, 256)
        harness.ns["_select_moe_mma_tiler_mn"] = fallback
        self.assertEqual(harness.select(), (32, 128))
        self.assertEqual(fallback_calls, [])
        alternatives = {
            "state_E": (71, 73, 288), "weight_E": (71, 73, 288),
            "m": (1, 6, 7, 9, 16, 32), "k": (2048, 4095, 4097, 8192),
            "n": (512, 1024, 2047, 2049), "num_topk": (1, 7, 9),
            "skip_zero_weight_expert_id": (None, -1, 0, 71, 73),
            "quant_mode": ("mxfp4", "nvfp4_alt"),
            "activation": ("silu", "relu2", "gelu_tanh", "swiglu"),
            "swiglu_alpha": (0.0, 1.702, float("nan")),
            "swiglu_beta": (1.0, -1.0, float("nan")),
            "swiglu_limit": (None, 0.0, 9.0, 11.0, float("nan")),
        }
        for name, values in alternatives.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    self.assertEqual(harness.select(**{name: value}), (128, 256))
                    geom = EXACT | {name: value}
                    self.assertEqual(fallback_calls[-1], (geom["m"] * geom["num_topk"], geom["n"]))

    def test_existing_tp_and_non_sentinel_tiles_unchanged(self):
        harness = Harness()
        for changes in (dict(skip_zero_weight_expert_id=None),
                        dict(state_E=288, weight_E=288, n=512, skip_zero_weight_expert_id=None),
                        dict(m=6), dict(m=16), dict(num_topk=1)):
            geom = EXACT | changes
            with self.subTest(changes=changes):
                expected = harness.ns["_select_moe_mma_tiler_mn"](geom["m"] * geom["num_topk"], geom["n"])
                self.assertEqual(harness.select(**changes), expected)
        self.assertEqual(harness.select(skip_zero_weight_expert_id=None), (64, 128))

    def test_actual_compiler_receives_m32_and_fixed_fake_geometry(self):
        harness = Harness()
        result = harness.get()
        kernel, args, options = harness.compiles[0]
        self.assertEqual(kernel.tile_shape_mnk, (32, 128, 128))
        self.assertEqual(kernel.output_tile_count_n, 16)
        self.assertEqual(kernel.skip_zero_weight_expert_id, 72)
        self.assertEqual((kernel.activation, kernel.swiglu_alpha, kernel.swiglu_beta, kernel.swiglu_limit),
                         ("swigluoai_uninterleave", 1.0, 0.0, 10.0))
        self.assertEqual(args[0].shape, (8, 4096))
        self.assertEqual(args[1].shape, (64,))
        self.assertEqual(args[3].shape, (64, 4096, 72))
        self.assertEqual(args[9].shape, (4096, 4096, 72))
        self.assertEqual(args[11].shape, (4096, 2048, 72))
        self.assertEqual(args[-2], 48)
        self.assertEqual(options, dict(options="--opt-level 2 --enable-tvm-ffi"))
        key = next(iter(harness.ns["_MICRO_KERNEL_CACHE"]))
        self.assertEqual(key[2:10], (72, 72, 8, 4096, 2048, 8, 64, 48))
        self.assertEqual(key[10], (32, 128))
        self.assertEqual(key[17], 72)
        self.assertEqual(harness.get(), result)
        self.assertEqual(len(harness.compiles), 1)

    def test_existing_m64_cache_entry_cannot_satisfy_m32(self):
        harness = Harness()
        harness.get()
        key = next(iter(harness.ns["_MICRO_KERNEL_CACHE"]))
        old_key = key[:10] + ((64, 128),) + key[11:]
        old_result = (object(), 48)
        harness.ns["_MICRO_KERNEL_CACHE"] = {old_key: old_result}
        actual = harness.get()
        self.assertNotEqual(actual, old_result)
        self.assertEqual(len(harness.compiles), 2)
        self.assertEqual(set(harness.ns["_MICRO_KERNEL_CACHE"]), {key, old_key})

    def test_nonmatching_call_still_constructs_m64(self):
        harness = Harness()
        harness.get(skip_zero_weight_expert_id=None)
        kernel = harness.compiles[0][0]
        self.assertEqual(kernel.tile_shape_mnk, (64, 128, 128))
        self.assertIsNone(kernel.skip_zero_weight_expert_id)
        key = next(iter(harness.ns["_MICRO_KERNEL_CACHE"]))
        self.assertEqual(key[10], (64, 128))

    def test_m32_retains_valid_mma_and_physical_scale_geometry(self):
        harness = Harness()
        harness.get()
        kernel = harness.compiles[0][0]
        kernel.a_dtype, kernel.sf_dtype = "fp4", "e4m3"
        kernel._setup_attributes(4096)
        self.assertEqual((kernel.num_m_tiles, kernel.num_n_tiles, kernel.num_k_blocks), (1, 8, 2))
        self.assertEqual(kernel.tiled_mma[1], (2, 2, 1))
        self.assertEqual((kernel.sa_tile_shape_mk, kernel.sfa_tile_shape_mk), ((128, 128), (128, 128)))
        self.assertEqual((kernel.sa_tiles_per_block, kernel.sfa_tiles_per_block), (4, 4))
        self.assertEqual(kernel.epi_tile, (32, 128))
        self.assertEqual(kernel.threads_per_cta, 160)
        for rows in range(65):
            # Independent ceil reference: the new task count covers every
            # admitted row exactly once, including 33..64 row boundary cases.
            tiles = len(range(0, rows, kernel.tile_shape_mnk[0]))
            self.assertEqual(tiles, rows // 32 + int(rows % 32 != 0))
            covered = [tile * 32 + row for tile in range(tiles) for row in range(32)
                       if tile * 32 + row < rows]
            self.assertEqual(covered, list(range(rows)))


if __name__ == "__main__":
    unittest.main()
