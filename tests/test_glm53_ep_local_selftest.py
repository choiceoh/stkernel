"""CPU startup-canary contracts; no Torch/CUDA imports or GPU execution."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import logging
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/modules/glm53_moe/glm53_ep_local_selftest.py"
spec = importlib.util.spec_from_file_location("ep_local_selftest_cpu", SOURCE)
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def runtime():
    cuda = SimpleNamespace(synchronize=Mock(), memory_allocated=lambda _: 10,
        memory_reserved=lambda _: 20, max_memory_allocated=lambda _: 30,
        max_memory_reserved=lambda _: 40)
    md = SimpleNamespace(_WORKSPACE_CACHE={"prior": object()}, _WEIGHT_CACHE={},
                         _MICRO_KERNEL_CACHE={"compiled": object()})
    return SimpleNamespace(cuda=cuda), md, object(), "cuda:0", {"source": "fixed"}


class StateTests(unittest.TestCase):
    def setUp(self):
        canary._STATES.clear()
        self.rt = runtime()
        self.wrapper = SimpleNamespace()

    def test_warning_logger_still_publishes_complete_pass_once_after_sync(self):
        events = []
        output = io.StringIO()
        previous_level = canary._LOG.level
        self.addCleanup(canary._LOG.setLevel, previous_level)
        canary._LOG.setLevel(logging.WARNING)
        self.assertFalse(canary._LOG.isEnabledFor(logging.INFO))
        self.rt[0].cuda.synchronize.side_effect = lambda _: events.append("sync")
        with redirect_stdout(output), patch("builtins.print", wraps=print) as publish, patch.object(
                canary, "_runtime", return_value=self.rt), patch.object(
                canary, "_run", side_effect=lambda *args: events.append("run")) as run:
            first = canary.ensure_ep_local_selftest(self.wrapper, device="cuda:0")
            first_output = output.getvalue()
            second = canary.ensure_ep_local_selftest(self.wrapper, device="cuda:0")
            self.assertEqual(output.getvalue(), first_output)
        self.assertIs(first, second)
        self.assertEqual(events, ["run", "sync"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first["verdict"], "PASS")
        self.assertFalse(first["performance_acceptance"])
        marker = "[ep-local-selftest] PASS "
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertTrue(first_output.startswith(marker))
        self.assertEqual(json.loads(first_output[len(marker):]), first)
        publish.assert_called_once_with(marker + json.dumps(first, sort_keys=True), flush=True)

    def test_first_failure_and_secondary_cleanup_are_sticky(self):
        error = AssertionError("CANDIDATE_NUMERICS_FAIL original")
        output = io.StringIO()
        self.rt[0].cuda.synchronize.side_effect = RuntimeError("cleanup failed")
        with redirect_stdout(output), patch.object(canary, "_runtime", return_value=self.rt), patch.object(
                canary, "_run", side_effect=error) as run, patch.object(canary._LOG, "error") as log_error:
            with self.assertRaisesRegex(RuntimeError, "readiness refused") as caught:
                canary.ensure_ep_local_selftest(self.wrapper, device="cuda:0")
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                canary.ensure_ep_local_selftest(self.wrapper, device="cuda:0")
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(run.call_count, 1)
        receipt = next(iter(canary._STATES.values()))
        self.assertIn("CANDIDATE_NUMERICS_FAIL original", receipt["error"])
        self.assertIn("cleanup failed", receipt["cleanup_error"])
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(receipt["verdict"], "FAIL")
        log_error.assert_called_once_with("[ep-local-selftest] FAIL %s", json.dumps(receipt, sort_keys=True))

    def test_reentrant_call_fails_instead_of_arming_running_test(self):
        with patch.object(canary, "_runtime", return_value=self.rt), patch.object(
                canary, "_run", side_effect=lambda *a: canary.ensure_ep_local_selftest(
                    self.wrapper, device="cuda:0")), patch.object(canary._LOG, "error"):
            with self.assertRaisesRegex(RuntimeError, "readiness refused"):
                canary.ensure_ep_local_selftest(self.wrapper, device="cuda:0")
        self.assertEqual(next(iter(canary._STATES.values()))["verdict"], "FAIL")

    def test_only_scratch_cache_changes_restored_compiles_retained(self):
        md = self.rt[1]
        old = md._WORKSPACE_CACHE["prior"]
        before = canary._cache_snapshot(md)
        md._WORKSPACE_CACHE.update(prior=object(), added=object())
        md._WEIGHT_CACHE["synthetic"] = object()
        md._MICRO_KERNEL_CACHE["new-kernel"] = object()
        report = canary._restore_scratch_caches(md, before)
        self.assertEqual(md._WORKSPACE_CACHE, {"prior": old})
        self.assertEqual(md._WEIGHT_CACHE, {})
        self.assertIn("new-kernel", md._MICRO_KERNEL_CACHE)
        self.assertEqual(report["_WORKSPACE_CACHE"], 2)

    def test_inference_tensor_descriptor_snapshot_is_read_only(self):
        class InferenceTensor:
            shape = (72,)
            def data_ptr(self): return 123
            @property
            def _version(self):
                raise RuntimeError("Inference tensors do not track version counter")
        value = InferenceTensor()
        wrapper = SimpleNamespace(g1_alphas=value, g2_alphas=value)
        state = canary._caller_state(wrapper)
        self.assertEqual(state["g1_alphas"][2], "inference-no-version-counter")
        self.assertEqual(state["g1_alphas"], state["g2_alphas"])


class ContractTests(unittest.TestCase):
    def test_existing_numerical_and_failure_capture_contracts_are_exact(self):
        actual = functions(SOURCE)
        for name, expected in functions(ROOT / "probes/glm53_ep_local_check.py").items():
            if name in ("row_errors", "check_control", "compare"):
                self.assertEqual(actual[name], expected, name)
        for name, expected in functions(ROOT / "probes/glm53_ep_numerics_diagnostics.py").items():
            self.assertEqual(actual[name], expected, name)
        self.assertEqual((canary.ROW_L2_FLOOR, canary.ROW_PEAK_FLOOR), (.02, .04))
        self.assertEqual((canary.MAX_BAD_ROWS, canary.MAX_COLUMNS), (8, 8))

    def test_fixed_coverage_and_rng_calls_do_not_mutate_global_rng(self):
        self.assertEqual(canary.SEED, 905308)
        self.assertEqual(canary.CASES, (("concentrated6912", 6912, "concentrated"),
            ("balanced4096", 4096, "balanced"), ("remote4096", 4096, "remote"),
            ("duplicate4096", 4096, "duplicate"), ("zeros4097", 4097, "zeros"),
            ("balanced8192", 8192, "balanced")))
        tree = ast.parse(SOURCE.read_text())
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("torch.manual_seed", calls)
        self.assertNotIn("torch.cuda.manual_seed", calls)
        self.assertNotIn("torch.cuda.reset_peak_memory_stats", calls)
        self.assertNotIn("md.clear_sm120_moe_caches", calls)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("torch.rand", "torch.randn", "torch.randint"):
                self.assertIn("generator", [k.arg for k in node.keywords])
        self.assertNotIn("from probes", SOURCE.read_text())

    def test_run_is_fixed_order_and_first_failure_prevents_later_cases(self):
        rt = runtime()
        cells = []
        def cell(*args):
            cells.append(args[-2])
            if len(cells) == 2:
                raise AssertionError("first numerical failure")
        receipt = {}
        with patch.object(canary, "_weights", return_value=(1, 2, 3, 4)), patch.object(
                canary, "_tensor_identity", return_value={}), patch.object(canary, "_case", side_effect=cell):
            with self.assertRaisesRegex(AssertionError, "first numerical failure"):
                canary._run(object(), *rt[:4], receipt)
        self.assertEqual(cells, list(canary.CASES[:2]))
        self.assertEqual(len(receipt["cases"]), 2)
        self.assertTrue(all("duration_s" in case for case in receipt["cases"]))

    def test_m32_candidate_and_m64_control_actual_keys_are_required(self):
        fp32 = "glm53_ep_micro_scatter_fp32_v1"
        direct = "glm53_ep_micro_direct_scatter_v1"
        shared = "glm53_ep_micro_shared_fc1_a_v1"
        def key(topk, capacity, tile, sentinel, tags):
            result = ["fp4", "nvfp4", 72, 72, 8, 4096, 2048, topk, capacity, 48,
                      tile, "int32", False, True, "swigluoai_uninterleave", 1., 0., sentinel]
            return tuple(result) + tags
        good = key(8, 64, (32, 128), 72, (fp32, direct, shared))
        control = key(1, 8, (64, 128), None, (fp32,))
        md = SimpleNamespace(_MICRO_KERNEL_CACHE={good: object(), control: object()})
        result = canary._micro_keys(md)
        self.assertEqual(result["candidate"], [repr(good)])
        self.assertEqual(json.loads(json.dumps(result))["control"], [repr(control)])
        for bad in ({control: object()}, {good: object()},
                    {key(8, 64, (64, 128), 72, (fp32, direct, shared)): object(), control: object()},
                    {good[:-1]: object(), control: object()},
                    {good[:-3] + (direct, fp32, shared): object(), control: object()},
                    {good[:-3] + (direct, shared): object(), control: object()},
                    {good[:-3] + (fp32, shared): object(), control: object()},
                    {good[:-1] + (shared + "_wrong",): object(), control: object()},
                    {good: object(), control[:-1]: object()},
                    {good: object(), control + (direct,): object()}):
            md._MICRO_KERNEL_CACHE = bad
            with self.assertRaises(AssertionError):
                canary._micro_keys(md)


class Q0Tests(unittest.TestCase):
    def records(self, *, reverse=False, poison_padding=0, mutate=False, bad_token=False):
        # Two meaningful rows in one expert's M128 atom. Physical allocation
        # order changes, while logical token/weight/input bytes stay identical.
        order = [1, 0] if reverse else [0, 1]
        packed = bytearray([poison_padding] * (128*2048))
        scale = bytearray([poison_padding] * 32768)
        tokens, weights = [0]*128, [0.]*128
        for physical, token in enumerate(order):
            tokens[physical], weights[physical] = token, .25 + token*.25
            packed[physical*2048:(physical+1)*2048] = bytes([10+token])*2048
            base = (physical%32)*16 + ((physical//32)%4)*4
            for sf in range(256): scale[base+(sf//4)*512+sf%4] = (sf+token)%256
        if mutate: packed[0] ^= 1
        if bad_token: tokens[0] = 3
        expected = [(0, token, struct.pack("<f", .25+token*.25).hex()) for token in range(2)]
        return canary._q0_records([2]+[0]*71, [0]*73, tokens, weights, bytes(packed), bytes(scale),
                                   rows=2, expected=expected)

    def test_canonical_q0_ignores_allocation_order_and_unwritten_padding(self):
        self.assertEqual(self.records(), self.records(reverse=True, poison_padding=255))
        canary._compare_q0(self.records(), self.records(reverse=True))

    def test_valid_packed_byte_drift_and_bad_route_fail_closed(self):
        with self.assertRaisesRegex(AssertionError, "Q0_REPLAY_BYTES_CHANGED"):
            canary._compare_q0(self.records(), self.records(mutate=True))
        with self.assertRaisesRegex(AssertionError, "token map escaped"):
            self.records(bad_token=True)
        with self.assertRaisesRegex(AssertionError, "Q0_REPLAY_BYTES_CHANGED"):
            canary._compare_q0(self.records(), self.records()[:1])


if __name__ == "__main__":
    unittest.main()
