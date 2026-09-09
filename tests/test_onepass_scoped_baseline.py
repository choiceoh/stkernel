"""CPU evidence contracts for explicit shared-control onepass comparisons."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import baseline
import judge
import proof

SKIP = "VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE"
LOCAL = "VLLM_GLM53_EP_PREFILL_LOCAL"
WARM = "VLLM_B12X_EP_WARM_COMPACT"
SKIPPED = ("[glm53-graph-profile] skipped unused estimate rank=0; "
           "model/MM profile and real graph warmup retained")
CAPTURED = "Graph capturing finished in 7 secs, took 0.86 GiB"
WARMED = ("[b12x EP compact warmup] COMPLETE device=cuda:0 launch_rows=128 "
          "specializations=14 static=10 dynamic=4 required=14 ready=14 "
          "representatives=8192,6848,3392,1024,640,576,512,448,384,320,256,192,128,64")


def record(name="B1", knobs=None):
    return dict(name=name, session="fixture", overlay="abc123", git="abcdef01", harness=40,
                doc_lang="en", thinking=True, workload={"ctx": [2000, 32000, 128000]},
                endpoint={"completion": "http://127.0.0.1:18000/v1/chat/completions"},
                boot_id=("a" if name == "B1" else "b") * 64 + "|2026-09-09T00:00:00Z",
                knobs={SKIP: "1"} if knobs is None else knobs,
                proof={SKIP: True, LOCAL: True, WARM: True}, quality={"ok": 9, "total": 9},
                korean={"dirty": 0, "n": 5}, decode={"windows_med": 20},
                prefill=[{"ctx": 32000, "cold_s": 10.0, "tok": 32545}])


class StartupProofTests(unittest.TestCase):
    def check(self, knobs, log):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boot.log"
            path.write_text(log)
            return proof.check(knobs, str(path))

    def test_skip_requires_skipped_estimate_and_completed_real_capture(self):
        self.assertTrue(self.check([SKIP], SKIPPED + "\n" + CAPTURED)["proof"][SKIP])
        for log in ("armed", SKIPPED, CAPTURED, CAPTURED + SKIPPED,
                    SKIPPED.replace("rank=0", "rank=1") + CAPTURED,
                    SKIPPED + CAPTURED + "\nProfiling CUDA graph memory (FULL): 25%"):
            with self.subTest(log=log):
                self.assertFalse(self.check([SKIP], log)["proof"][SKIP])

    def test_compact_requires_entire_coherent_completed_plan(self):
        self.assertTrue(self.check([WARM], WARMED)["proof"][WARM])
        for change in (WARMED.replace("COMPLETE", "STARTED"), WARMED.replace("device=cuda:0", "device=cpu"),
                       WARMED.replace("ready=14", "ready=13"),
                       WARMED.replace("required=14", "required=15"), WARMED.replace("static=10", "static=9"),
                       WARMED.replace("128,64", "128,128"), WARMED.replace("128,64", "128,0"),
                       WARMED.replace("launch_rows=128", "launch_rows=1"),
                       WARMED + "\n" + WARMED.replace("ready=14", "ready=13"),
                       WARMED + "\n[b12x EP compact warmup] FAILED"):
            with self.subTest(log=change):
                self.assertFalse(self.check([WARM], change)["proof"][WARM])

    def test_unknown_marker_stays_unknown_and_cli_fails(self):
        self.assertIsNone(self.check(["VLLM_UNREGISTERED"], CAPTURED)["proof"]["VLLM_UNREGISTERED"])
        with patch.object(sys, "argv", ["proof.py", "--knobs", "VLLM_UNREGISTERED", "--log", "/missing"]), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(proof.main(), 1)

    def test_real_onepass3_logs_recognize_shared_proof_without_rewriting_archives(self):
        archive = ROOT / "measurements/glm53_ep_local_20260908/onepass3-completed"
        raw = (archive / "onepass.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertTrue(judge.unproved(rows[1]))
        for row, name in zip(rows, ("boot-B1.log.gz", "boot-A.log.gz")):
            # Simulate the next invocation in memory, including its new warm
            # proof schema. No historical record is admitted or rewritten.
            row["session"] = "new-session-fixture"
            log = gzip.decompress((archive / name).read_bytes()).decode()
            if row["name"] == "EPONEPASS3A":
                row["knobs"][WARM] = "1"
                log = WARMED + "\n" + log
            evidence = self.check(list(row["knobs"]), log)
            row.update(proof=evidence["proof"], proof_ok=evidence["proof_ok"])
            self.assertFalse(judge.unproved(row))
        with patch.dict(os.environ, {"ONEPASS_BASELINE_KNOBS": json.dumps({SKIP: "1"}),
                                     "FLEET_SESSION": rows[1]["session"]}):
            self.assertEqual(judge.baselines_on(rows, rows[1])[0], [rows[0]])
            result = judge.judge(rows[1], rows[0], rows, {"metric": "prefill_ttft", "ctx": 32000})
            self.assertEqual(result["status"], "incomplete")
            self.assertLess(result["delta"], 0)
        self.assertEqual((archive / "onepass.jsonl").read_bytes(), raw)


class ScopedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"ONEPASS_BASELINE_KNOBS": json.dumps({SKIP: "1"}),
                                                    "FLEET_SESSION": "fixture"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.b = record()
        self.a = record("A", {SKIP: "1", LOCAL: "1", WARM: "1"})

    def test_scope_does_not_change_production_defaults_or_records(self):
        original = copy.deepcopy(self.b)
        self.assertEqual(baseline.is_baseline(self.b), (False, "knobs"))
        self.assertTrue(baseline.is_baseline(record(knobs={}))[0])
        self.assertEqual(judge.baselines_on([self.b, self.a, record(knobs={})], self.a)[0], [self.b])
        self.assertEqual(self.b, original)

    def test_invalid_json_values_duplicates_and_missing_session_fail(self):
        for value in ("[]", "{}", "null", '{"VLLM_X":"1","VLLM_X":"2"}',
                      '{"VLLM_X":1}', '{"VLLM_X":""}', '{"OTHER":"1"}', "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                baseline.comparison_scope({"ONEPASS_BASELINE_KNOBS": value, "FLEET_SESSION": "fixture"})
        with self.assertRaises(ValueError):
            baseline.comparison_scope({"ONEPASS_BASELINE_KNOBS": json.dumps({SKIP: "1"})})

    def test_exact_knobs_no_silent_dropping_or_candidate_as_baseline(self):
        for knobs in ({}, {SKIP: "0"}, {SKIP: "1", "VLLM_OTHER": "0"}, self.a["knobs"]):
            with self.subTest(knobs=knobs):
                self.assertEqual(judge.baselines_on([record(knobs=knobs)], self.a)[0], [])

    def test_unrelated_context_runtime_and_missing_identity_are_excluded(self):
        for key, value in (("session", "other"), ("overlay", "other"), ("git", "other"),
                           ("harness", 41), ("workload", {"ctx": [2000]}), ("endpoint", {"completion": "other"}),
                           ("runtime", {"image": "other"}), ("doc_lang", "ko"), ("thinking", False),
                           ("boot_id", None), ("boot_id", "not-a-container")):
            other = copy.deepcopy(self.b); other[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(judge.baselines_on([other], self.a)[0], [])
        self.b["runtime"] = self.a["runtime"] = {"image": "same", "capacity": 1056}
        self.assertEqual(judge.baselines_on([self.b], self.a)[0], [self.b])
        self.a["runtime"] = {"image": "same", "capacity": 415}
        self.assertEqual(judge.baselines_on([self.b], self.a)[0], [])

    def test_candidate_and_baseline_unproved_or_contradictory_proofs_are_invalid(self):
        for side in ("candidate", "baseline"):
            for value in (None, False, 1, "true"):
                b, a = copy.deepcopy(self.b), copy.deepcopy(self.a)
                (a if side == "candidate" else b)["proof"][SKIP] = value
                with self.subTest(side=side, value=value):
                    result = judge.judge(a, b, [b, a])
                    self.assertEqual(result["status"], "invalid")
                    self.assertIn("UNPROVED", result["verdict"])
                    if side == "baseline":
                        self.assertFalse(baseline.comparison_baseline(b, a, scope=baseline.comparison_scope()))

    def test_scope_preserves_noise_gate_and_does_not_borrow_cross_session_floor(self):
        b2 = record("B2"); b2["decode"]["windows_med"] = 21
        self.a["boot_id"] = "c" * 64 + "|2026-09-09T00:00:00Z"
        extras = [record("OLD") for _ in range(3)]
        for row in extras:
            row.update(session="other", overlay="older")
        result = judge.judge(self.a, self.b, [self.b, b2, self.a] + extras,
                             {"metric": "prefill_ttft", "ctx": 32000})
        self.assertEqual(result["floor_n"], 2)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["baseline_scope"]["knobs"], {SKIP: "1"})

    def test_count_for_final_baseline_counts_distinct_proved_scoped_boots(self):
        b2 = record("B2"); duplicate = copy.deepcopy(self.b)
        bad = record("BAD"); bad["proof"][SKIP] = None
        with patch.object(baseline, "load", return_value=[self.b, duplicate, self.a, bad, b2]), \
                patch.object(sys, "argv", ["baseline.py", "--count-for", "B2"]), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(baseline.main(), 0)
            self.assertEqual(stdout.getvalue().strip(), "2")


if __name__ == "__main__":
    unittest.main()
