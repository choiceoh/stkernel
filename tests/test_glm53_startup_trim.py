"""Execute the real startup helper with fake allocators; no GPU imports."""
import ast
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_runtime/gpu_worker.py"


class Harness:
    def __init__(self, *, flag="1", fail=None, trim_return=1, libc=True,
                 missing_proc=False, metric_failure=False):
        self.events = []
        self.logs = []
        self.live = {name: object() for name in ("weights", "kv", "graphs", "workspace")}
        self.worker = SimpleNamespace(rank=2, device="cuda:2", **self.live)
        self.flag = flag
        self.trim = SimpleNamespace(argtypes=None, restype=None)

        def stage(name, value=None):
            self.events.append(name)
            if fail == name:
                raise RuntimeError("failure at " + name)
            return value

        def metric(name):
            self.events.append(name)
            if metric_failure:
                raise RuntimeError("metric unavailable")
            after = "empty_cache" in self.events
            return 4096 if name == "allocated" else (8192 if after else 16384)

        def proc(path):
            self.events.append(path)
            if missing_proc:
                raise OSError("proc unavailable")
            if path == "/proc/meminfo":
                return io.StringIO("MemTotal: 32768 kB\nMemAvailable: 12000 kB\n")
            if path == "/proc/self/status":
                return io.StringIO("Name: worker\nVmRSS: 8000 kB\n")
            raise AssertionError(path)

        class Trim:
            argtypes = None
            restype = None
            def __call__(inner, padding):
                assert padding == 0
                assert inner.argtypes == ["size_t"] and inner.restype == "c_int"
                return stage("malloc_trim", trim_return)

        self.trim = Trim()
        def load_libc(handle):
            self.events.append("load_libc")
            assert handle is None
            if not libc:
                return SimpleNamespace()
            return SimpleNamespace(gnu_get_libc_version=object(), malloc_trim=self.trim)

        self.ctypes = SimpleNamespace(CDLL=load_libc, c_size_t="size_t", c_int="c_int")
        ns = dict(os=SimpleNamespace(environ={"VLLM_GLM53_STARTUP_TRIM": flag}),
                  gc=SimpleNamespace(collect=lambda: stage("gc_collect", 7)),
                  torch=SimpleNamespace(
                      cuda=SimpleNamespace(memory_allocated=lambda d: metric("allocated"),
                                           memory_reserved=lambda d: metric("reserved"),
                                           synchronize=lambda d: stage("synchronize")),
                      accelerator=SimpleNamespace(empty_cache=lambda: stage("empty_cache"))),
                  time=SimpleNamespace(time=lambda: 123.), open=proc,
                  logger=SimpleNamespace(warning=lambda fmt, raw: self.logs.append((fmt, json.loads(raw)))))
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_glm53_startup_trim")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)], type_ignores=[])),
                     str(SOURCE), "exec"), ns)
        self.function = ns[fn.name]

    def call(self):
        with patch.dict("sys.modules", {"ctypes": self.ctypes}):
            return self.function(self.worker)


class StartupTrimTests(unittest.TestCase):
    def test_disabled_flag_performs_no_diagnostics_or_imports(self):
        for flag in ("0", "", "true", "01"):
            h = Harness(flag=flag)
            self.assertIsNone(h.call())
            self.assertEqual(h.events, [])
            self.assertEqual(h.logs, [])
            self.assertFalse(hasattr(h.worker, "_glm53_startup_trim_receipt"))
        fn = next(n for n in ast.parse(SOURCE.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_glm53_startup_trim")
        imports = [i for i,n in enumerate(fn.body) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertTrue(imports)
        self.assertTrue(all(i > 2 for i in imports))

    def test_real_stage_order_metrics_and_live_ownership(self):
        h = Harness()
        result = h.call()
        operations = [e for e in h.events if e in ("synchronize", "gc_collect", "empty_cache", "malloc_trim")]
        self.assertEqual(operations, ["synchronize", "gc_collect", "empty_cache", "malloc_trim"])
        self.assertEqual(result["verdict"], "COMPLETE")
        self.assertEqual(result["before"], dict(allocated=4096, reserved=16384,
                                             mem_available=12000*1024, vm_rss=8000*1024))
        self.assertEqual(result["after"]["allocated"], 4096)
        self.assertEqual(result["after"]["reserved"], 8192)
        self.assertEqual(result["stages"][-1]["returned"], 1)
        self.assertEqual(result["stages"][1]["collected"], 7)
        self.assertEqual(h.logs, [("[glm53-startup-trim] %s", result)])
        for name, reference in h.live.items():
            self.assertIs(getattr(h.worker, name), reference)
        self.assertEqual(set(vars(h.worker)), {"rank", "device", "_glm53_startup_trim_receipt", *h.live})

    def test_success_is_one_time_even_when_hook_is_called_again(self):
        h = Harness()
        result = h.call()
        events = list(h.events)
        self.assertIs(h.call(), result)
        self.assertEqual(h.events, events)
        self.assertEqual(len(h.logs), 1)

    def test_gpu_or_collection_failure_stops_later_stages_and_remains_failed(self):
        sequence = ["synchronize", "gc_collect", "empty_cache", "malloc_trim"]
        for failure in sequence[:3]:
            with self.subTest(failure=failure):
                h = Harness(fail=failure)
                with self.assertRaisesRegex(RuntimeError, "failure at " + failure):
                    h.call()
                result = h.worker._glm53_startup_trim_receipt
                self.assertEqual((result["verdict"], result["failed_stage"]), ("FAIL", failure))
                self.assertIn("after", result)
                self.assertEqual(len(h.logs), 1)
                self.assertEqual([e for e in h.events if e in sequence], sequence[:sequence.index(failure)+1])
                events = list(h.events)
                with self.assertRaisesRegex(RuntimeError, "previously failed"):
                    h.call()
                self.assertEqual(h.events, events)

    def test_optional_libc_and_zero_return_do_not_claim_reclamation(self):
        h = Harness(libc=False)
        result = h.call()
        self.assertEqual(result["verdict"], "PARTIAL")
        self.assertEqual(result["stages"][-1]["status"], "UNAVAILABLE")
        self.assertNotIn("malloc_trim", h.events)
        events = list(h.events)
        self.assertIs(h.call(), result)
        self.assertEqual(h.events, events)
        result = Harness(trim_return=0).call()
        self.assertEqual(result["verdict"], "COMPLETE")
        self.assertEqual(result["stages"][-1]["returned"], 0)
        self.assertNotIn("reclaimed", result)
        h = Harness(trim_return=7)
        with self.assertRaisesRegex(RuntimeError, "unexpected status"):
            h.call()
        self.assertEqual(h.worker._glm53_startup_trim_receipt["verdict"], "FAIL")

    def test_missing_metrics_are_null_and_partial_never_fabricated_zero(self):
        result = Harness(missing_proc=True, metric_failure=True).call()
        self.assertEqual(result["verdict"], "PARTIAL")
        self.assertEqual(result["before"], dict(allocated=None, reserved=None, mem_available=None, vm_rss=None))
        self.assertEqual(result["before"], result["after"])
        self.assertEqual(len(result["measurement_errors"]), 8)

    def test_hook_only_runs_after_last_warmup_before_gc_freeze(self):
        tree = ast.parse(SOURCE.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_glm53_startup_trim"]
        self.assertEqual(len(calls), 1)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "compile_or_warm_up_model")
        by_name = {ast.unparse(n.func): n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)}
        self.assertGreater(calls[0].lineno, by_name["trigger_inductor_lazy_init"])
        self.assertGreater(calls[0].lineno, by_name["warmup_kernels"])
        self.assertLess(calls[0].lineno, by_name["freeze_gc_heap"])
        self.assertEqual(ast.unparse(calls[0]), "_glm53_startup_trim(self)")


class StartupProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = SOURCE.parents[3] / "bench/proof.py"
        spec = importlib.util.spec_from_file_location("glm53_trim_proof_test", path)
        cls.proof = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.proof)

    def receipt(self, **options):
        h = Harness(**options)
        h.worker.rank = 0
        return h.call()

    def trim_log(self, record):
        return "(Worker_TP0) WARNING [glm53-startup-trim] " + json.dumps(record)

    def q0_log(self, rows=8192):
        return ("[tp-sf6-q0-selftest] PASS " + json.dumps(dict(verdict="PASS", phase="complete"))
                + "\n[tp-sf6-q0] LAUNCHED E288/H4096/I512/top8 T=" + str(rows))

    def test_actual_complete_helper_receipt_proves_even_zero_reclamation(self):
        for returned in (0, 1):
            record = self.receipt(trim_return=returned)
            self.assertTrue(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", self.trim_log(record)))
        self.assertFalse(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", "STARTUP_TRIM=1 armed"))
        self.assertFalse(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", self.trim_log(self.receipt(libc=False))))

    def test_trim_rejects_nonterminal_wrong_rank_missing_and_failure_fields(self):
        for key, value in (("verdict", "RUNNING"), ("verdict", "PARTIAL"), ("verdict", "FAIL"),
                           ("rank", 1), ("rank", False), ("schema", True), ("error", "failure"),
                           ("cleanup_error", None), ("measurement_errors", [dict(error="missing")]),
                           ("started_at", -1), ("completed_at", 122)):
            record = self.receipt(); record[key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", self.trim_log(record)))
        for key in ("before", "after", "stages", "completed_at", "measurement_errors"):
            record = self.receipt(); del record[key]
            self.assertFalse(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", self.trim_log(record)))

    def test_trim_rejects_incomplete_stage_or_unmeasured_bytes(self):
        mutations = [lambda r: r["stages"].reverse(), lambda r: r["stages"].pop(),
                     lambda r: r["stages"][2].update(status="PARTIAL"),
                     lambda r: r["stages"][1].update(error="failed"),
                     lambda r: r["stages"][3].update(returned=True)]
        for phase in ("before", "after"):
            for field in ("allocated", "reserved", "mem_available", "vm_rss"):
                for value in (None, -1, True, 0.5):
                    mutations.append(lambda r, p=phase, f=field, v=value: r[p].update({f:v}))
        for mutate in mutations:
            record = self.receipt(); mutate(record)
            self.assertFalse(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", self.trim_log(record)))

    def test_tp_q0_requires_actual_launch_and_complete_canary_without_fail(self):
        for rows in (4096, 6912, 8192):
            self.assertTrue(self.proof._startup_proof("VLLM_GLM53_TP_SF6_Q0", self.q0_log(rows)))
        for log in (self.q0_log(4095), self.q0_log(8193), self.q0_log().split("\n")[0],
                    self.q0_log().split("\n")[1], self.q0_log().replace('"complete"', '"initial"'),
                    self.q0_log().replace('"PASS"', '"RUNNING"'),
                    self.q0_log()+"\n[tp-sf6-q0-selftest] FAIL {}"):
            self.assertFalse(self.proof._startup_proof("VLLM_GLM53_TP_SF6_Q0", log))

    def test_malformed_duplicate_and_nonfinite_json_never_proves(self):
        for knob, prefix in (("VLLM_GLM53_STARTUP_TRIM", "[glm53-startup-trim] "),
                             ("VLLM_GLM53_TP_SF6_Q0", "[tp-sf6-q0-selftest] PASS ")):
            for payload in ('{', '[]', '{"verdict":"FAIL","verdict":"PASS"}',
                            '{"verdict":"PASS","phase":"complete","x":NaN}',
                            '{"verdict":"PASS","phase":"complete","x":1e999}'):
                log = prefix+payload+"\n[tp-sf6-q0] LAUNCHED E288/H4096/I512/top8 T=8192"
                self.assertFalse(self.proof._startup_proof(knob, log))
        record = self.trim_log(self.receipt())
        self.assertFalse(self.proof._startup_proof("VLLM_GLM53_STARTUP_TRIM", record+"\n"+record))

    def test_public_check_keeps_each_active_knob_strict_and_no_fixed_marker_waiver(self):
        import tempfile
        knobs = ["VLLM_GLM53_STARTUP_TRIM", "VLLM_GLM53_TP_SF6_Q0"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"head.log"
            path.write_text(self.trim_log(self.receipt())+"\n"+self.q0_log())
            result = self.proof.check(knobs, str(path), {k:("armed", "fixture") for k in knobs})
            self.assertEqual(result["proof"], {k:True for k in knobs})
            self.assertEqual(result["proof_ok"], "2/2")
            path.write_text(self.trim_log(self.receipt())+"\narmed")
            result = self.proof.check(knobs, str(path), {k:("armed", "fixture") for k in knobs})
            self.assertEqual(result["proof"], {knobs[0]:True, knobs[1]:False})
            self.assertEqual(result["proof_ok"], "1/2")


if __name__ == "__main__":
    unittest.main()
