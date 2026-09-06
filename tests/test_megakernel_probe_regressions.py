#!/usr/bin/env python3
"""CPU regressions for fail-closed baseline selection and balanced sampling.

Run: python3 tests/test_megakernel_probe_regressions.py
The CUDA/numerical gates themselves run in mk_smlp2_concurrent_probe.py on
GB10; these tests need neither torch nor the serving image.
"""
from __future__ import annotations

import ast
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import math
import os
from pathlib import Path
import statistics
import sys
import types
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO / "probes/mk_smlp2_concurrent_probe.py"
spec = importlib.util.spec_from_file_location("smlp_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

# The existing all-segment probe imports torch at module load. Extract the
# actual baseline selector so this check remains executable on a CPU-only
# developer machine instead of silently skipping the regression.
tree = ast.parse((REPO / "probes/megakernel_glm53_bench.py").read_text())
selector = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "native_smlp_activation")
namespace = {}
exec(compile(ast.Module(body=[selector], type_ignores=[]), str(PROBE_PATH), "exec"), namespace)
native_activation = namespace["native_smlp_activation"]
ACT_MODULE = "vllm.model_executor.layers.activation"


class BaselineSelection(unittest.TestCase):
    def module(self, cls):
        module = types.ModuleType(ACT_MODULE)
        module.SiluAndMulWithClamp = cls
        return patch.dict(sys.modules, {ACT_MODULE: module})

    def test_uses_cuda_method_even_if_generic_dispatch_is_native_torch(self):
        class Activation:
            def __init__(self, *, swiglu_limit):
                self.limit = swiglu_limit

            def __call__(self, _):
                raise AssertionError("generic/Torch dispatch must not be timed")

            def forward_cuda(self, value):
                return (value, self.limit)

        with self.module(Activation):
            forward, identity = native_activation(10.0)
            self.assertEqual(forward("input"), ("input", 10.0))
            self.assertIn("Activation.forward_cuda", identity)

    def test_missing_cuda_method_fails_instead_of_torch_fallback(self):
        class TorchOnly:
            def __init__(self, **kwargs):
                pass

            def __call__(self, value):
                return value

        with self.module(TorchOnly), self.assertRaisesRegex(RuntimeError, "baseline unavailable"):
            native_activation(10.0)

    def test_image_constructor_mismatch_is_an_explicit_failure(self):
        class ChangedAPI:
            def __init__(self, unsupported_argument):
                pass

        with self.module(ChangedAPI), self.assertRaisesRegex(RuntimeError, "forward_cuda is required"):
            native_activation(10.0)

    def test_native_runtime_error_propagates(self):
        class BrokenCUDA:
            def __init__(self, **kwargs):
                pass

            def forward_cuda(self, _):
                raise RuntimeError("CUDA symbol missing")

        with self.module(BrokenCUDA):
            forward, _ = native_activation(10.0)
            with self.assertRaisesRegex(RuntimeError, "CUDA symbol missing"):
                forward("input")


class NumericalGate(unittest.TestCase):
    def test_nonfinite_relative_error_fails_even_if_comparison_said_pass(self):
        for rel in (math.nan, math.inf, -math.inf):
            with self.subTest(rel=rel), self.assertRaises(RuntimeError):
                probe._require_gate("captured output", True, True, rel)

    def test_nonfinite_tensor_or_failed_tolerance_fails(self):
        for finite, ok in ((False, True), (True, False)):
            with self.subTest(finite=finite, ok=ok), self.assertRaises(RuntimeError):
                probe._require_gate("captured output", finite, ok, 0.0)

    def test_finite_matching_output_passes(self):
        probe._require_gate("captured output", True, True, 1e-7)


class TimingOrder(unittest.TestCase):
    def test_opposite_orders_cancel_linear_drift_between_identical_arms(self):
        calls = []

        def drifting_timer(name):
            calls.append(name)
            return 100.0 + len(calls)

        results = probe._balanced_samples(("stock", "fused", "moe"), drifting_timer, 2)
        self.assertEqual(calls, ["stock", "fused", "moe", "moe", "fused", "stock",
                                 "moe", "fused", "stock", "stock", "fused", "moe"])
        means = []
        for arm in results.values():
            self.assertEqual(len(arm["forward"]), 2)
            self.assertEqual(len(arm["reverse"]), 2)
            means.append(statistics.mean(arm["forward"] + arm["reverse"]))
        self.assertEqual(len(set(means)), 1)

    def test_invalid_event_time_aborts_the_sample_set(self):
        for invalid in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "invalid timing"):
                probe._balanced_samples(("stock", "fused"), lambda _: invalid, 1)

    def test_invalid_experiment_parameters_fail(self):
        for names, rounds in (((), 1), (("stock",), 0), (("stock", "stock"), 1)):
            with self.subTest(names=names, rounds=rounds), self.assertRaises(ValueError):
                probe._balanced_samples(names, lambda _: 1.0, rounds)


class CapturedOutputGate(unittest.TestCase):
    """Execute the real capture gate with a graph that mutates held outputs."""

    def capture(self, values, *, with_moe=False):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return Tensor(self.value)

        tensor = Tensor(1.0)
        values = iter(values)

        class Graph:
            def replay(self):
                tensor.value = next(values)

        def capture(a, b, order):
            a()
            b()
            return Graph()

        def finite(value):
            return math.isfinite(value.value)

        def check_mlp(label, got):
            rel = abs(got.value - 1.0)
            probe._require_gate(label, finite(got), rel <= 2e-3, rel)

        # Only extract the nested gate, not a rewritten approximation of it.
        tree = ast.parse(PROBE_PATH.read_text())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "capture_case")
        cases = {}
        scope = dict(_capture=capture, moe=lambda: tensor,
                     torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None)),
                     args=types.SimpleNamespace(replays=3), cases=cases,
                     check_mlp=check_mlp, finite=finite, moe_reference=Tensor(1.0),
                     relative=lambda got, ref: abs(got.value - ref.value),
                     _require_gate=probe._require_gate)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(PROBE_PATH), "exec"), scope)
        run = lambda: scope["capture_case"]("candidate", None if with_moe else lambda: tensor,
                                             with_moe=with_moe)
        return run, cases

    def test_valid_graph_enters_timing_cases_only_after_replay_checks(self):
        run, cases = self.capture((1.0, 1.0, 1.0))
        self.assertEqual(cases, {})
        run()
        self.assertIn("candidate", cases)

    def test_corrupt_captured_output_rejected_despite_valid_eager_value(self):
        run, cases = self.capture((math.nan, 1.0, 1.0))
        with self.assertRaisesRegex(RuntimeError, "numerics/replay failed"):
            run()
        self.assertEqual(cases, {})

    def test_replay_drift_rejected_even_inside_numerical_tolerance(self):
        run, cases = self.capture((1.0, 1.00001, 1.0))
        with self.assertRaisesRegex(RuntimeError, "graph replay"):
            run()
        self.assertEqual(cases, {})

    def test_concurrent_moe_corruption_rejected(self):
        run, cases = self.capture((2.0, 1.0, 1.0), with_moe=True)
        with self.assertRaisesRegex(RuntimeError, "moe graph"):
            run()
        self.assertEqual(cases, {})


class BootGateOption(unittest.TestCase):
    def extract(self, name, scope):
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(PROBE_PATH), "exec"), scope)
        return scope[name]

    def test_selected_gates_deduplicate_gemm_and_exact(self):
        calls = []
        mk = types.SimpleNamespace(
            _selftest_gemm=lambda: calls.append("gemm") or True,
            _selftest_mhc=lambda: calls.append("mhc") or True)
        run = self.extract("probe_boot_gates", {})
        with redirect_stdout(io.StringIO()):
            self.assertTrue(run(mk, ["exact", "gemm", "mhc"]))
        self.assertEqual(calls, ["gemm", "mhc"])

    def test_false_nonfinite_or_exception_stops_later_gates(self):
        def broken():
            raise RuntimeError("CUDA gate broke")

        run = self.extract("probe_boot_gates", {})
        for first in (lambda: False, lambda: math.nan, broken):
            calls = []
            mk = types.SimpleNamespace(_selftest_gemm=first,
                                       _selftest_mhc=lambda: calls.append("mhc") or True)
            with redirect_stdout(io.StringIO()):
                self.assertFalse(run(mk, ["gemm", "mhc"]))
            self.assertEqual(calls, [])

    def run_main(self, flags, boot_result):
        calls = []
        mk = types.SimpleNamespace(SINKHORN_SERVED=20, _build=lambda: types.SimpleNamespace(
            probe_device=lambda: (12, 1, 48, 101376)))
        layers = types.ModuleType("vllm.model_executor.layers")
        layers.glm53_megakernel = mk
        scope = dict(argparse=argparse, os=os, GEMM_SHAPES=[], _arm_env=lambda segs: None,
                     torch=types.SimpleNamespace(cuda=types.SimpleNamespace(init=lambda: None)),
                     probe_boot_gates=lambda *_: calls.append("boot") or boot_result,
                     probe_exact=lambda *_: calls.append("exact") or True)
        main = self.extract("main", scope)
        argv = ["probe", "--segments", "exact", "--iters", "1", *flags]
        with patch.dict(sys.modules, {layers.__name__: layers}), patch.object(sys, "argv", argv), \
                redirect_stdout(io.StringIO()):
            status = main()
        return status, calls

    def test_parser_default_does_not_add_boot_work(self):
        self.assertEqual(self.run_main([], False), (0, ["exact"]))

    def test_parser_flag_runs_boot_gates_before_probe(self):
        self.assertEqual(self.run_main(["--boot-gates"], True), (0, ["boot", "exact"]))

    def test_failed_boot_gate_never_reaches_timed_probe(self):
        self.assertEqual(self.run_main(["--boot-gates"], False), (1, ["boot"]))


if __name__ == "__main__":
    unittest.main()
