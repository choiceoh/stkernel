"""CPU source oracles for compact startup coverage; no accelerator imports."""
import ast
import copy
import math
import os
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "overlay/modules/glm53_moe/flashinfer_b12x_moe.py"
DISPATCH = ROOT / "overlay/modules/glm53_moe/moe_dispatch.py"


def load_functions(path, names, namespace, *, compiler_prefixes=()):
    tree = ast.parse(path.read_text())
    selected = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            if node.name in compiler_prefixes:
                end = next(i for i, statement in enumerate(node.body)
                           if isinstance(statement, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "cache_key"
                                   for t in statement.targets))
                node.body = node.body[:end + 1] + [ast.Return(ast.Name("cache_key", ast.Load()))]
            selected.append(node)
    if {n.name for n in selected} != set(names):
        raise AssertionError("missing actual-source function")
    module = ast.Module(body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
                             *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class DynamicWorkspace:
    def __init__(self, rows, tile_m):
        self.routed_rows_capacity = rows
        self.max_rows = rows + 71 * tile_m
        self.tile_m = tile_m


class OtherWorkspace:
    pass


class FakeTensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", storage=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.storage = storage if storage is not None else object()

    def stride(self):
        return tuple(math.prod(self.shape[i + 1:]) for i in range(len(self.shape)))

    def numel(self):
        return math.prod(self.shape)

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.start is None and index.step is None
        return FakeTensor((index.stop, *self.shape[1:]), self.dtype, self.device, self.storage)


class Harness:
    """Actual planner, selectors, workspace cache and compiler-key prefixes.

    Only GPU allocation/launch, hardware discovery and unrelated TP/v2 gates
    are replaced. The independent key oracle executes each real compiler's
    source through its cache-key assignment, without compiling a kernel.
    """
    def __init__(self):
        self.events, self.allocations, self.logs = [], [], []
        self.fail_launch = None
        self.fail_sync = False
        self.omit_key = None
        self.cutover = 640
        self.torch = SimpleNamespace(
            int32="int32", float32="fp32", bfloat16="bf16",
            device=lambda value: value, zeros=self.zeros,
            cuda=SimpleNamespace(synchronize=self.synchronize),
        )
        self.ns = dict(
            torch=self.torch, os=os, Lock=Lock,
            logger=SimpleNamespace(info=self.log), B12X_EP_COMPACT_PAIR_ALIGN=64,
            _B12X_EP_COMPACT_WARMED=set(), _B12X_EP_COMPACT_WARM_LOCK=Lock(),
        )
        load_functions(WRAPPER, {
            "b12x_ep_compact_pair_count", "b12x_ep_compact_warmup_buckets",
            "_b12x_ep_compact_warmup_plan", "_b12x_ep_compact_warmup_ready",
            "_b12x_ep_compact_warmup_execute", "_b12x_ep_warm_tensor_signature",
            "_warm_compact_shapes",
        }, self.ns)
        self.dispatch = ModuleType("fake_dispatch")
        dns = self.dispatch.__dict__
        dns.update(
            torch=self.torch, _LEVEL_TILE_M=128, _LEVEL_TILE_N=128,
            _MAX_SHARED_INPUT_TOPK=32, _WORKSPACE_CACHE={},
            _STATIC_KERNEL_CACHE={}, _DYNAMIC_KERNEL_CACHE={},
            _FORCED_BACKEND=None, _GLM53_B12X_FORCE_BACKEND=None,
            _GLM53_B12X_STATIC_CUTOVER_PAIRS=None,
            _GLM53_B12X_STATIC_MAC_LADDER=None, _GLM53_B12X_DYNAMIC_MAC_LADDER=None,
            _GLM53_B12X_PREFILL_REUSE=False, _GLM53_B12X_PREFILL_FC1_N128=False,
            _STATIC_MAC_LADDER=((128, 130), (640, 188)),
            _DYNAMIC_MAC_LADDER=((640, 188), (1024, 147)),
            _get_static_compact_cutover_pairs=lambda _: self.cutover,
            _normalize_quant_mode=lambda value, *args: value or "nvfp4",
            _normalize_activation_precision=lambda value: value,
            _activation_precision_from_quant_mode=lambda _: "fp4",
            _is_w4a16=lambda _: False, is_gated_activation=lambda _: True,
            _sf_params_for_quant_mode=lambda _: (16, None),
            _ep_local_prefill_kernel=lambda **_: None,
            _static_v2_config_for=lambda **_: None,
            _is_glm53_b12x_tp_geometry=lambda **_: False,
            get_max_active_clusters=lambda _: 48, get_num_sm=lambda _: 48,
            Sm120DynamicMoEWorkspace=DynamicWorkspace,
            Sm120W4A16MoEWorkspace=OtherWorkspace,
            allocate_sm120_moe_workspace=self.workspace,
        )
        load_functions(DISPATCH, {
            "_select_dynamic_tile_m", "_level_tile_n", "_lookup_mac_ladder",
            "_effective_glm53_mac_ladder", "_effective_glm53_static_cutover",
            "_effective_glm53_forced_backend", "select_sm120_moe_backend",
            "_get_cached_workspace", "_dynamic_kernel_cache_key", "_static_kernel_cache_key",
            "_get_dynamic_kernel", "_get_static_kernel",
        }, dns, compiler_prefixes={"_get_dynamic_kernel", "_get_static_kernel"})
        parent = ModuleType("flashinfer.fused_moe.cute_dsl.blackwell_sm12x")
        parent.moe_dispatch = self.dispatch
        fused = ModuleType("flashinfer.fused_moe")
        fused.b12x_fused_moe = self.launch
        self.modules = {"flashinfer.fused_moe": fused,
                        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x": parent}
        self.expert = SimpleNamespace(
            max_num_tokens=8192, topk=8, num_local_experts=72,
            global_num_experts=288, hidden_dim=4096,
            intermediate_size_per_partition=2048, _kernel_num_experts=72,
            _activation_str="swigluoai_uninterleave", _swiglu_alpha=1.0,
            _swiglu_beta=0.0, _swiglu_limit=10.0,
            _warm_activation_dtype=lambda: "bf16",
            w1_sf_mma=FakeTensor((72, 4096, 256)),
            w2_sf_mma=FakeTensor((72, 4096, 128)),
            g1_alphas=FakeTensor((72,), "fp32"),
            g2_alphas=FakeTensor((72,), "fp32"),
            _fc2_input_scale=FakeTensor((72,), "fp32"),
        )
        self.layer = SimpleNamespace(w13_weight=FakeTensor((72, 4096, 2048)),
                                     w2_weight=FakeTensor((72, 4096, 1024)))

    def workspace(self, **kw):
        rows = kw["routed_rows"]
        if kw["backend"] == "dynamic":
            tile = self.dispatch._select_dynamic_tile_m(rows, kw["state_E"], kw["activation"])
            return DynamicWorkspace(rows, tile)
        return SimpleNamespace(max_rows=rows)

    def plan(self):
        e = self.expert
        rows = self.ns["b12x_ep_compact_warmup_buckets"](
            e.max_num_tokens, e.topk, e.num_local_experts, e.global_num_experts)
        return self.ns["_b12x_ep_compact_warmup_plan"](
            self.dispatch, rows, device="cuda:0", kernel_e=e._kernel_num_experts,
            hidden=e.hidden_dim, intermediate=e.intermediate_size_per_partition,
            activation=e._activation_str, alpha=e._swiglu_alpha, beta=e._swiglu_beta,
            limit=e._swiglu_limit, input_gs_shared=e.g1_alphas.numel() == 1)

    def compiler_key(self, rows):
        d, e = self.dispatch, self.expert
        geometry = dict(num_topk=1, num_experts=72, num_local_experts=72,
                        hidden_size=4096, intermediate_size=2048,
                        activation=e._activation_str, swiglu_limit=10.0)
        backend = d.select_sm120_moe_backend(num_tokens=rows, **geometry)
        workspace = d._get_cached_workspace(
            backend=backend, state_E=72, weight_E=72, routed_rows=rows, k=4096,
            n=2048, num_topk=1, device="cuda:0", activation=e._activation_str)
        args = dict(topk_ids_dtype="int32", activation=e._activation_str,
                    swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0)
        if backend == "dynamic":
            key = d._get_dynamic_kernel(72, rows, 4096, 2048, 1, workspace.max_rows,
                                        tile_m=workspace.tile_m,
                                        share_input_across_experts=e.g1_alphas.numel() == 1,
                                        **args)
        else:
            mac = min(d._lookup_mac_ladder(d._STATIC_MAC_LADDER, rows) or 48, 48)
            key = d._get_static_kernel(72, 72, rows, 4096, 2048, 1, workspace.max_rows,
                                       mac_override=mac, **args)
        return backend, key

    def zeros(self, shape, **kw):
        tensor = FakeTensor(shape, **kw)
        self.allocations.append(tensor)
        return tensor

    def launch(self, **kw):
        rows = kw["x"].shape[0]
        self.events.append(("launch", rows))
        if self.fail_launch == rows:
            raise RuntimeError("launch failed")
        assert kw["w1_weight"] is self.layer.w13_weight
        assert kw["w2_weight"] is self.layer.w2_weight
        assert kw["w1_alpha"] is self.expert.g1_alphas
        assert kw["fc2_input_scale"] is self.expert._fc2_input_scale
        backend, key = self.compiler_key(rows)
        if rows != self.omit_key:
            getattr(self.dispatch, f"_{backend.upper()}_KERNEL_CACHE")[key] = object()

    def synchronize(self, device):
        self.events.append(("sync", device))
        if self.fail_sync:
            raise RuntimeError("sync failed")

    def log(self, fmt, *args):
        self.events.append(("complete",))
        self.logs.append(fmt % args)

    def run(self, enabled="1"):
        with patch.dict("sys.modules", self.modules), patch.dict(os.environ, {"VLLM_B12X_EP_WARM_COMPACT": enabled}):
            self.ns["_warm_compact_shapes"](self.expert, self.layer)


class CoverageTests(unittest.TestCase):
    def test_every_padded_and_sliced_local_count_is_covered(self):
        h = Harness()
        buckets = h.ns["b12x_ep_compact_warmup_buckets"]
        for tokens, topk, local, total in ((8192, 8, 72, 288), (65, 3, 2, 8), (128, 1, 1, 2)):
            actual = set()
            for count in range(1, tokens * min(topk, local) + 1):
                padded = math.ceil(count / 64) * 64
                actual.update(min(tokens, padded - start) for start in range(0, padded, tokens))
            self.assertEqual(set(buckets(tokens, topk, local, total)), actual)
        self.assertEqual(buckets(8192, 8, 72, 288), tuple(range(8192, 0, -64)))

    def test_all_reachable_rows_match_real_compiler_prefix_keys(self):
        h = Harness()
        plan = h.plan()
        self.assertEqual([row for row, _, _ in plan],
                         [8192, 6848, 3392, 1024, 640, 576, 512, 448, 384, 320, 256, 192, 128, 64])
        ready = {(backend, key) for _, backend, key in plan}
        for rows in range(64, 8193, 64):
            self.assertIn(h.compiler_key(rows), ready, rows)
        static = [key for _, backend, key in plan if backend == "static"]
        self.assertEqual({key[9] for key in static}, {640})
        self.assertEqual(len(static), 10)

    def test_actual_cutover_ladders_scalar_scales_and_existing_capacity(self):
        h = Harness()
        h.cutover = 128
        h.dispatch._DYNAMIC_MAC_LADDER = ((1024, 7), (4096, 13))
        h.expert.g1_alphas = FakeTensor((1,), "fp32")
        # A prior functional allocation may be bigger than the cutover.
        h.dispatch._get_cached_workspace(backend="static", state_E=72, weight_E=72,
            routed_rows=896, k=4096, n=2048, num_topk=1, device="cuda:0",
            activation=h.expert._activation_str)
        plan = h.plan()
        expected = {(backend, key) for _, backend, key in plan}
        for rows in range(64, 8193, 64):
            self.assertIn(h.compiler_key(rows), expected, rows)
        self.assertEqual({key[9] for _, backend, key in plan if backend == "static"}, {896})
        self.assertEqual({key[7] for _, backend, key in plan if backend == "dynamic"}, {7, 13, 48})
        self.assertTrue(all(key[16] for _, backend, key in plan if backend == "dynamic"))

    def test_unbounded_or_unsupported_preparation_fails_closed(self):
        h = Harness()
        h.expert.max_num_tokens = 32768
        with self.assertRaisesRegex(ValueError, "16384"):
            h.run()
        h.expert.max_num_tokens = 65
        with self.assertRaisesRegex(RuntimeError, "64-aligned"):
            h.run()
        h.expert.max_num_tokens = 8192
        h.cutover = 8192
        with self.assertRaisesRegex(RuntimeError, "64-specialization"):
            h.run()
        self.assertFalse(h.logs)
        self.assertFalse(h.allocations)


class OrchestrationTests(unittest.TestCase):
    def test_default_off_does_no_preparation(self):
        h = Harness()
        h.run("0")
        self.assertFalse(h.events)
        self.assertFalse(h.allocations)
        self.assertFalse(h.dispatch._WORKSPACE_CACHE)

    def test_all_calls_sync_and_cache_keys_precede_completion_then_deduplicate(self):
        h = Harness()
        h.run()
        self.assertEqual(len(h.events), 16)  # 14 launches, synchronize, marker.
        self.assertEqual(h.events[-2:], [("sync", "cuda:0"), ("complete",)])
        self.assertIn("launch_rows=128 specializations=14 static=10 dynamic=4 required=14 ready=14", h.logs[0])
        self.assertEqual([t.shape for t in h.allocations],
                         [(8192, 4096), (8192, 1), (8192, 1), (8192, 4096)])
        # Another layer's weights have different addresses/values, same codegen.
        h.layer = SimpleNamespace(w13_weight=FakeTensor((72, 4096, 2048)),
                                  w2_weight=FakeTensor((72, 4096, 1024)))
        h.run()
        self.assertEqual(len(h.events), 16)
        self.assertEqual(len(h.allocations), 4)

    def test_launch_sync_or_missing_key_never_marks_complete(self):
        for kind in ("launch", "sync", "missing"):
            with self.subTest(kind=kind):
                h = Harness()
                h.fail_launch = 3392 if kind == "launch" else None
                h.fail_sync = kind == "sync"
                h.omit_key = 192 if kind == "missing" else None
                with self.assertRaisesRegex(RuntimeError, "failed|keys are missing"):
                    h.run()
                self.assertFalse(h.logs)
                self.assertFalse(h.ns["_B12X_EP_COMPACT_WARMED"])

    def test_dedup_does_not_hide_evicted_key_or_new_scale_signature(self):
        h = Harness()
        h.run()
        h.dispatch._DYNAMIC_KERNEL_CACHE.clear()
        h.run()
        self.assertEqual(len(h.logs), 2)
        h.expert.g1_alphas = FakeTensor((1,), "fp32")
        h.run()
        self.assertEqual(len(h.logs), 3)
        self.assertEqual(len(h.ns["_B12X_EP_COMPACT_WARMED"]), 2)


if __name__ == "__main__":
    unittest.main()
