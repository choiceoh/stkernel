#!/usr/bin/env python3
"""Execute real dispatch gates/cache keys without importing CUDA libraries."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/modules/glm53_moe/moe_dispatch.py"
TREE = ast.parse(SOURCE.read_text())


class DispatchTests(unittest.TestCase):
    def eligible(self, **changes):
        calls = []
        ns = dict(_GLM53_B12X_PREFILL_STREAM_FC2=True, m=6912, E=288,
                  k=4096, n=512, num_topk=8, activation_precision="fp4",
                  quant_mode="nvfp4", mma_tiler_mn=(128, 128),
                  activation="swigluoai_uninterleave", swiglu_alpha=1.,
                  swiglu_beta=0., swiglu_limit=10.,
                  torch=SimpleNamespace(cuda=SimpleNamespace(
                      get_device_capability=lambda: (12, 1))),
                  _prefill_stream_stock_contract_matches=lambda: calls.append(1) or True)
        ns.update(changes)
        function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_get_dynamic_kernel")
        gate = next(n for n in function.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "prefill_stream"
                            for t in n.targets))
        exec(compile(ast.Module(body=[gate], type_ignores=[]), str(SOURCE), "exec"), ns)
        return ns["prefill_stream"], calls

    def test_executed_chunk_boundaries(self):
        for m in (4096, 6912, 8192):
            self.assertEqual(self.eligible(m=m), (True, [1]))
        for m in (1, 2048, 2593, 4095, 8193, 32768):
            self.assertEqual(self.eligible(m=m), (False, []))

    def test_unsupported_contract_does_not_import_candidate(self):
        for field, value in dict(E=72, k=2048, n=2048, num_topk=4,
                                 activation_precision="bf16", quant_mode="mxfp4",
                                 mma_tiler_mn=(64, 128), activation="silu",
                                 swiglu_alpha=1.1, swiglu_beta=.1, swiglu_limit=None,
                                 _GLM53_B12X_PREFILL_STREAM_FC2=False).items():
            with self.subTest(field=field):
                self.assertEqual(self.eligible(**{field: value}), (False, []))
        self.assertFalse(self.eligible(torch=SimpleNamespace(cuda=SimpleNamespace(
            get_device_capability=lambda: (12, 0))))[0])
        self.assertFalse(self.eligible(_prefill_stream_stock_contract_matches=lambda: False)[0])

    def test_keys_separate_layouts_and_candidates_preserve_stock(self):
        func = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_dynamic_kernel_cache_key")
        ns = {}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), func], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
        key = ns["_dynamic_kernel_cache_key"]
        args = dict(activation_precision="fp4", quant_mode="nvfp4", E=288,
                    k=4096, n=512, num_topk=8, mac=48, mma_tiler_mn=(128,128),
                    topk_ids_dtype="int32", input_scales_are_reciprocal=False,
                    fast_math=True, activation="swigluoai_uninterleave",
                    swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
                    share_input_across_experts=True)
        stock = key(**args)
        self.assertEqual(stock, ("dynamic", *args.values(), False))
        variants = [key(**args, tiled=tiled, **flag)
                    for tiled in (False, True)
                    for flag in ({}, {"prefill_reuse": True},
                                 {"prefill_fc1_n128": True}, {"prefill_stream_fc2": True})]
        self.assertEqual(len(set(variants)), 8)
        self.assertEqual(key(**args, prefill_stream_fc2=False), stock)

    def test_compile_marker_is_not_serving_proof(self):
        spec = importlib.util.spec_from_file_location("moe_proof_test", ROOT / "bench/proof.py")
        proof = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proof)
        knob = "VLLM_GLM53_B12X_PREFILL_STREAM_FC2"
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "boot.log"
            log.write_text("[b12x prefill stream] ENGAGED\n[b12x prefill stream] COMPILED\n")
            self.assertIs(proof.check([knob], str(log))["proof"][knob], False)
            log.write_text("[b12x prefill stream] LAUNCHED m=6912 tiled=True\n")
            self.assertIs(proof.check([knob], str(log))["proof"][knob], True)

    def test_dynamic_marker_requires_actual_candidate_launch(self):
        function = copy.deepcopy(next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                                      and n.name == "launch_sm120_dynamic_moe"))
        start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == "compiled")
        function.body = function.body[start:]
        function.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
        function.returns = None
        module = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                         str(SOURCE), "exec")
        for candidate in (False, True):
            events = []
            compiled = lambda *a: events.append("launch")
            key = ("dynamic", "glm53_prefill_stream_fc2_v1" if candidate else "stock")
            ns = dict(compiled=compiled, runtime_args=(), _GLM53_B12X_PREFILL_STREAM_FC2=True,
                      _prefill_stream_launch_announce=[False], _DYNAMIC_KERNEL_CACHE={key: (compiled, 48)},
                      num_tokens=6912, weights=SimpleNamespace(tiled=True), scatter_output="output",
                      logging=SimpleNamespace(getLogger=lambda _: SimpleNamespace(
                          warning=lambda *a: events.append("proof"))))
            exec(module, ns)
            self.assertEqual(ns["launch_sm120_dynamic_moe"](), "output")
            self.assertEqual(events, ["launch", "proof"] if candidate else ["launch"])


if __name__ == "__main__":
    unittest.main()
