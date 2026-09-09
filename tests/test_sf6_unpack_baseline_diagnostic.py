"""Offline independent-reservation diagnostics with retained A and synthetic B."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "measurements/glm53_sf6_unpack_baseline_20260909/analyze_baseline.py"
spec = importlib.util.spec_from_file_location("sf6_baseline_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
ARCHIVE = ROOT / "measurements/glm53_sf6_unpack_onepass_20260909/final"


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


class IndependentBaselineDiagnostic(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.candidate, self.baseline = root / "candidate", root / "baseline"
        shutil.copytree(ARCHIVE, self.candidate)
        self.baseline.mkdir()
        a, _ = diagnostic.one_record(self.candidate)
        self.name = "synthetic-scalarB"
        self.record = deepcopy(a)
        self.record.update(name=self.name, git=diagnostic.REVISIONS["baseline"][:7],
                           session="synthetic-scalar-session", cold_compile=False,
                           boot_id="baseline-" + a["boot_id"], knobs={})
        self.write_record()
        (self.baseline / "campaign.exit").write_text("0\n")
        (self.baseline / "campaign.log").write_text("[fixed2K rep2] cjk_mixed=2\nHalvorsen博士\n")
        shutil.copyfile(self.candidate / (a["name"] + ".memory.jsonl"),
                        self.baseline / (self.name + ".memory.jsonl"))
        (self.baseline / "baseline-source.commit").write_text(diagnostic.REVISIONS["baseline"] + "\n")
        shutil.copyfile(self.candidate / ("expected-" + a["name"] + ".json"),
                        self.baseline / "baseline-expected.json")
        self.state = dict(schema=diagnostic.BASELINE_SCHEMA, name=self.name, mode="baseline",
            sf6_direct=False, sf6_unpack=True, source_commit=diagnostic.REVISIONS["baseline"],
            session=self.record["session"], collection_status="COMPLETE", runtime_validation="PASS", errors=[], phases={})
        for prefix in ("prepared", "runtime"):
            directory = self.baseline / ("baseline-" + prefix)
            directory.mkdir()
            receipt = diagnostic.audit.read_json(self.candidate / ("observed-" + prefix + "-" + a["name"] + ".json"))
            receipt.update(schema=diagnostic.BASELINE_SCHEMA, name=self.name, mode="baseline",
                           source_commit=diagnostic.REVISIONS["baseline"], collection_status="COMPLETE",
                           collection_errors=[], runtime_validation="PASS", validation_errors=[])
            for field in ("head_before", "head_after"):
                receipt[field]["boot_id"] = self.record["boot_id"]
                receipt[field]["knobs"][diagnostic.proof.SF6_UNPACK_KNOB] = "0"
            files = []
            for host in diagnostic.audit.HOSTS:
                stem = prefix + "-" + a["name"] + "-" + host
                report = diagnostic.audit.read_json(self.candidate / (stem + ".json"))
                raw = (self.candidate / (stem + ".log")).read_bytes().decode(errors="replace").replace("u8x4=1", "u8x4=0")
                raw = re.sub(r"(artifact=[A-Za-z0-9_]+_)([0-9a-f]{16})", lambda m:
                             m[1] + hashlib.sha256(("scalar-" + m[2]).encode()).hexdigest()[:16], raw).encode()
                report.update(mode="baseline", boot_id="baseline-" + report["boot_id"],
                              log_sha256=diagnostic.sha(raw), markers=diagnostic.proof.parse_markers(raw.decode(), sf6_unpack=True))
                report["knobs"][diagnostic.proof.SF6_UNPACK_KNOB] = "0"
                write(directory / (host + ".json"), report)
                (directory / (host + ".log")).write_bytes(raw)
                files.extend(host + suffix for suffix in (".json", ".log"))
            if prefix == "runtime":
                raw = (self.baseline / "records.raw.jsonl").read_bytes().rstrip(b"\n")
                (directory / "record.raw.json").write_bytes(raw)
                receipt["record_sha256"] = diagnostic.sha(raw)
                files.append("record.raw.json")
            receipt["artifacts_sha256"] = {name: diagnostic.sha((directory / name).read_bytes()) for name in files}
            write(directory / "receipt.json", receipt)
            self.state["phases"][prefix] = dict(path=directory.name, collection_status="COMPLETE",
                receipt_sha256=diagnostic.sha((directory / "receipt.json").read_bytes()))
        write(self.baseline / "baseline-observer.json", self.state)

    def write_record(self):
        write(self.baseline / "records.raw.jsonl", self.record)

    def summarize(self):
        return diagnostic.summarize(self.candidate, self.baseline)

    def reseal(self, prefix):
        directory = self.baseline / ("baseline-" + prefix)
        receipt = diagnostic.audit.read_json(directory / "receipt.json")
        receipt["artifacts_sha256"] = {name: diagnostic.sha((directory / name).read_bytes())
                                      for name in receipt["artifacts_sha256"]}
        write(directory / "receipt.json", receipt)
        self.state["phases"][prefix]["receipt_sha256"] = diagnostic.sha((directory / "receipt.json").read_bytes())
        write(self.baseline / "baseline-observer.json", self.state)

    def test_complete_diagnostic_retains_both_failures_despite_baseline_exit0(self):
        result = self.summarize()
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["diagnostic_checks_passed"])
        self.assertFalse(result["valid"])
        self.assertIsNone(result["comparison"])
        self.assertEqual(result["raw_speed_delta"]["step_s_change_pct"], 0)
        self.assertAlmostEqual(result["raw_speed_delta"]["candidate_step_s"], 20.21523095264694)
        self.assertEqual(result["output_hash_matches"]["equal_count"], 8)
        self.assertTrue(result["output_hash_matches"]["fixed2k_rep2_same"])
        for mode in ("candidate", "baseline"):
            self.assertEqual(result["arms"][mode]["korean"]["dirty"], 1)
            self.assertEqual(result["arms"][mode]["record_validation_errors"], ["Korean corruption/coverage gate failed"])
        self.assertEqual(result["arms"]["baseline"]["campaign_returncode"], 0)
        self.assertFalse(result["arms"]["baseline"]["cold_compile"])
        self.assertTrue(result["arms"]["candidate"]["cold_compile"])
        self.assertTrue(any("Halvorsen博士" in line["text"] for line in result["arms"]["baseline"]["korean_scanner_excerpt"]))
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(diagnostic.main([str(self.candidate), str(self.baseline)]), 0)
        self.assertIsNone(json.loads(stream.getvalue())["comparison"])

    def test_numeric_recomputation_rejects_bad_pooled_even_after_quality_failure(self):
        self.record["decode"]["fixed_pooled_step_s"] += 1
        self.write_record()
        result = self.summarize()
        self.assertIsNone(result["raw_speed_delta"])
        self.assertTrue(any(error["stage"] == "baseline timing" for error in result["errors"]))

    def test_raw_delta_uses_pooled_rate_and_reciprocal_time(self):
        decode = self.record["decode"]
        for window in decode["fixed_intervals"]:
            window["steps"] *= .9
        decode["windows"] = [value * .9 for value in decode["windows"]]
        decode["windows_med"] *= .9
        decode["fixed_pooled_step_s"] *= .9
        self.write_record()
        result = self.summarize()
        delta = result["raw_speed_delta"]
        self.assertAlmostEqual(delta["step_s_change_pct"], 100 / 9)
        self.assertAlmostEqual(delta["ms_per_step_reduction_pct"], 10)
        self.assertAlmostEqual(delta["saved_ms_per_step"], delta["baseline_ms_per_step"] - delta["candidate_ms_per_step"])
        # This changed record deliberately no longer matches the retained seal.
        self.assertFalse(delta["evidence_checks_passed"])
        self.assertFalse(result["valid"])
        self.assertIsNone(result["comparison"])

    def test_duplicate_record_cannot_select_a_baseline(self):
        with (self.baseline / "records.raw.jsonl").open("ab") as stream:
            stream.write((self.baseline / "records.raw.jsonl").read_bytes())
        result = self.summarize()
        self.assertIsNone(result["raw_speed_delta"])
        self.assertTrue(any(error["stage"] == "baseline record" for error in result["errors"]))

    def test_changed_raw_log_and_changed_marker_even_when_resealed_fail(self):
        path = self.baseline / "baseline-runtime/srv3.log"
        path.write_bytes(path.read_bytes() + b"tampered\n")
        self.assertTrue(any(error["stage"] == "baseline runtime" for error in self.summarize()["errors"]))
        self.reseal("runtime")
        result = self.summarize()
        self.assertTrue(any("log SHA/parser" in error["error"] for error in result["errors"]))
        self.assertFalse(result["raw_speed_delta"]["evidence_checks_passed"])

    def test_actual_video_zero_and_source_mismatch_are_not_exempted(self):
        path = self.baseline / "baseline-runtime/srv4.json"
        report = diagnostic.audit.read_json(path)
        index = report["serving_argv"].index("--limit-mm-per-prompt") + 1
        report["serving_argv"][index] = '{"image":4,"video":0}'
        write(path, report)
        self.reseal("runtime")
        self.assertTrue(any("multimedia CLI" in error["error"] for error in self.summarize()["errors"]))
        (self.baseline / "baseline-source.commit").write_text("0" * 40 + "\n")
        self.assertTrue(any("unexpected source" in error["error"] for error in self.summarize()["errors"]))

    def test_excess_traffic_stays_error_with_raw_arithmetic_only(self):
        self.record["traffic"]["after"]["finished"] += 1
        self.write_record()
        result = self.summarize()
        self.assertTrue(any(error["stage"] == "baseline onepass" and "traffic" in error["error"] for error in result["errors"]))
        self.assertIsNotNone(result["raw_speed_delta"])
        self.assertFalse(result["raw_speed_delta"]["evidence_checks_passed"])
        self.assertIsNone(result["comparison"])


if __name__ == "__main__":
    unittest.main()
