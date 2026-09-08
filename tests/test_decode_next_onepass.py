"""CPU audits of actual onepass schema and portable retained GPU admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import analyze_decode_next_onepass as analyze
from tests import test_ar_consumer_reuse as ar_fixture
from tests import test_decode_transport_gpu as transport_fixture
from tests import test_decode_sf6_gpu_runner as sf_fixture
from tests import test_decode_next_runtime as runtime_fixture

ARCHIVE = ROOT / "measurements/glm53_decode_reform_20260908/serving"


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def seal(out, receipt, names):
    receipt["artifacts_sha256"] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in names}
    write(out / "admission.json", receipt)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.record = analyze.read_jsonl(ARCHIVE / "records.raw.jsonl")[0]
        self.channels = analyze.read_jsonl(ARCHIVE / ("channels-" + self.record["name"] + ".jsonl"))

    def test_actual_archived_onepass_schema_and_primary(self):
        result = analyze.validate_record(self.record, self.channels)
        self.assertAlmostEqual(result["ms_per_step"], 45.29763408172011)
        self.assertEqual(result["quality"], {"ok": 18, "total": 18})
        self.assertTrue(result["ordered_stream_hashes_verified"])

    def test_partial_traffic_quality_korean_and_arithmetic_fail(self):
        changes = (
            lambda r: r["requests"][-1].update(completion_tokens=2047),
            lambda r: r["requests"].pop(),
            lambda r: r["decode"].update(fixed_pooled_step_s=1000),
            lambda r: r["decode"]["fixed_intervals"][1].update(start=0),
            lambda r: r["decode"].update(windows_med=None),
            lambda r: r["traffic"]["after"].update(finished=9),
            lambda r: r["traffic"]["samples"][0].update(running=2),
            lambda r: r["quality"].update(ok=17),
            lambda r: r["korean"].update(dirty=1),
            lambda r: r.update(evidence_issues=["counter reset"]),
            lambda r: r["prefill"][0].update(warm_s=1),
        )
        for change in changes:
            record = deepcopy(self.record)
            change(record)
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze.validate_record(record, self.channels)

    def test_ordered_stream_hash_mismatch_fails(self):
        channels = deepcopy(self.channels)
        channels[0], channels[1] = channels[1], channels[0]
        with self.assertRaisesRegex(ValueError, "ordered stream"):
            analyze.validate_record(self.record, channels)
        channels = deepcopy(self.channels)
        channels[-1]["output_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ordered stream"):
            analyze.validate_record(self.record, channels)

    def test_cold_compile_difference_does_not_become_prefill_win(self):
        baseline = analyze.validate_record(self.record, self.channels)
        candidate = deepcopy(baseline)
        candidate["cold_compile"] = True
        result = analyze.comparison(candidate, baseline)
        self.assertEqual(result["saved_ms_per_step"], 0)
        for row in result["prefill"]:
            self.assertFalse(row["cold_comparable"])
            self.assertIsNone(row["cold_ttft_change_pct"])
            self.assertIn("incomparable", row["cold_note"])
        self.assertIn("candidate_warm_s", result["prefill"][0])
        self.assertFalse(result["prefill"][1]["warm_independent"])
        self.assertNotIn("candidate_warm_s", result["prefill"][1])

    def test_canonical_record_keeps_checks_without_independent_stream_claim(self):
        result = analyze.validate_record(self.record, canonical=True)
        self.assertAlmostEqual(result["ms_per_step"], 45.29763408172011)
        self.assertFalse(result["independent_stream_hashes_verified"])
        self.assertFalse(result["ordered_stream_hashes_verified"])
        with self.assertRaisesRegex(ValueError, "stream/request count"):
            analyze.validate_record(self.record)
        with self.assertRaisesRegex(ValueError, "injected stream"):
            analyze.validate_record(self.record, self.channels, canonical=True)
        for mutate in (lambda r: r["quality"].update(ok=17),
                       lambda r: r["requests"][-1].update(completion_tokens=2047),
                       lambda r: r["requests"][0].update(output_sha256="missing"),
                       lambda r: r["decode"].update(fixed_pooled_step_s=1000),
                       lambda r: r["traffic"]["after"].update(finished=9)):
            record = deepcopy(self.record)
            mutate(record)
            with self.assertRaises(ValueError):
                analyze.validate_record(record, canonical=True)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ar_fixture.StageReuse()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.out = self.fixture.base / "campaign"
        self.out.mkdir()
        self.source = analyze.transport_evidence.source()
        self.revision = self.source["revision"]
        self.source["harness_sha256"] = {"probes/decode_transport_gpu_probe.py": "e" * 64}
        self.expected = {"/pkg/" + Path(name).name: value for name, value in self.source["source_sha256"].items() if name.startswith("overlay/")}
        self.expected.update({"/pkg/" + name: value for name, value in sf_fixture.source()["kernels_sha256"].items()})
        (self.out / "source.commit").write_text(self.revision + "\n")
        records = analyze.read_jsonl(ARCHIVE / "records.raw.jsonl")
        for name, mode, record in zip(("A", "B"), ("candidate", "baseline"), records):
            old_name = record["name"]
            record.update(name=name, git=self.revision[:7], boot_id="srv2|" + mode)
            shutil.copyfile(ARCHIVE / ("channels-" + old_name + ".jsonl"), self.out / ("channels-" + name + ".jsonl"))
            (self.out / ("arm-" + name + ".exit")).write_text("0\n")
            write(self.out / ("expected-" + name + ".json"), self.expected)
            memory = dict(elapsed_s=0, minimum_kib=10 * 1024**2, issues=[],
                nodes={host: dict(total_kib=128 * 1024**2, available_kib=16 * 1024**2) for host in analyze.MEMORY_HOSTS})
            write(self.out / ("memory-" + name + ".jsonl"), memory)
            for host in analyze.HOSTS:
                proof = runtime_fixture.report(mode, host)
                proof["source_sha256"] = self.expected
                for prefix in ("prepared", "runtime"):
                    write(self.out / (prefix + "-" + name + "-" + host + ".json"), proof)
        self.records = records
        self.write_records()
        self.transport()
        self.sf6()

    def write_records(self):
        (self.out / "records.raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def transport(self):
        out = self.out / "transport-gpu"
        out.mkdir()
        completed, names = [], ["source.json", "runtime.json"]
        write(out / "source.json", self.source)
        write(out / "runtime.json", self.fixture.runtime)
        for rank in range(4):
            name = "probe-rank" + str(rank)
            completed.append(self.fixture.rank_evidence(out, name))
            report = analyze.read_json(out / (name + ".json"))
            report.update(schema=analyze.TRANSPORT_SCHEMA, samples=[], transport=transport_fixture.transport_proof(),
                          wrapper_sha256="e" * 64)
            write(out / (name + ".json"), report)
            names.extend(name + suffix for suffix in (".json", ".container.json", ".log"))
        seal(out, dict(schema=analyze.TRANSPORT_SCHEMA, status="PASS", image=analyze.runtime_proof.IMAGE,
                       flags=analyze.FLAGS, selected_stages=["probe"], completed=completed), names)

    def sf6(self):
        out = self.out / "sf6-gpu"
        out.mkdir()
        source = sf_fixture.source()
        source["revision"] = self.revision
        write(out / "source.json", source)
        write(out / "result.json", sf_fixture.report())
        write(out / "container.json", sf_fixture.container("cid", "owner"))
        write(out / "samples.json", [dict(available_bytes=25 * 1024**3)])
        (out / "probe.log").write_text("REFORM_SF6_CORRECTNESS_PASS\n")
        seal(out, dict(schema=analyze.sf6.SCHEMA, status="PASS", image=analyze.runtime_proof.IMAGE,
                       returncode=0, issues=[], source_commit=self.revision, container_id="cid", owner_token="owner"),
             ("source.json", "result.json", "probe.log", "container.json", "samples.json"))

    def summary(self):
        return analyze.summarize(self.out, "A", "B")

    def canonical_evidence(self):
        for name in ("A", "B"):
            (self.out / ("channels-" + name + ".jsonl")).unlink()
            (self.out / ("arm-" + name + ".exit")).unlink()
            (self.out / ("memory-" + name + ".jsonl")).rename(self.out / (name + ".memory.jsonl"))
        for directory in ("transport-gpu", "sf6-gpu"):
            shutil.rmtree(self.out / directory)
        (self.out / "campaign.exit").write_text("0\n")
        record_hashes = {json.loads(line)["name"]: hashlib.sha256(line).hexdigest()
                         for line in (self.out / "records.raw.jsonl").read_bytes().splitlines()}
        for name in ("A", "B"):
            for prefix in ("prepared", "runtime"):
                files = []
                for host in analyze.HOSTS:
                    stem = prefix + "-" + name + "-" + host
                    (self.out / (stem + ".log")).write_text("observed startup log\n")
                    files.extend(stem + suffix for suffix in (".json", ".log"))
                write(self.out / ("observed-" + prefix + "-" + name + ".json"), dict(status="PASS", errors=[],
                    artifacts_sha256={filename: hashlib.sha256((self.out / filename).read_bytes()).hexdigest() for filename in files}))
        write(self.out / "observer.json", dict(schema=1, status="PASS", source_commit=self.revision,
            candidate="A", baseline="B", errors=[], arms={name: dict(mode=mode, status="PASS",
                head_boot_id="srv2|" + mode, errors=[], record_sha256=record_hashes[name],
                before_receipt="observed-prepared-" + name + ".json", after_receipt="observed-runtime-" + name + ".json")
                for name, mode in (("A", "candidate"), ("B", "baseline"))}))

    def canonical_summary(self):
        return analyze.summarize(self.out, "A", "B", canonical=True)

    def test_complete_copied_evidence_is_portable_without_current_source(self):
        moved = self.fixture.base / "copied-evidence"
        shutil.copytree(self.out, moved)
        with patch.object(analyze.transport_evidence, "source", side_effect=AssertionError("current source read")), \
             patch.object(analyze.sf6, "source_identity", side_effect=AssertionError("current source read")):
            result = analyze.summarize(moved, "A", "B")
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["campaign_cleanup"], "pending")
        self.assertAlmostEqual(result["comparison"]["saved_ms_per_step"], .7261450327238617)
        self.assertIn("no statistical significance", result["note"])

    def test_arm_or_cleanup_failure_invalidates_existing_numbers(self):
        (self.out / "arm-A.exit").write_text("2\n")
        result = self.summary()
        self.assertFalse(result["valid"])
        self.assertIsNone(result["comparison"])
        self.assertEqual(len(result["per_boot"]), 2)
        self.assertTrue(any(e["stage"] == "A arm exit" for e in result["errors"]))
        (self.out / "arm-A.exit").write_text("0\n")
        (self.out / "campaign.exit").write_text("1\n")
        self.assertTrue(any(e["stage"] == "campaign exit/cleanup" for e in self.summary()["errors"]))

    def test_partial_record_and_missing_rank_fail_readably(self):
        self.records.pop()
        self.write_records()
        result = self.summary()
        self.assertFalse(result["valid"])
        self.assertIn("exactly two", result["errors"][0]["error"])

    def test_runtime_mode_drift_or_memory_failure_is_invalid(self):
        path = self.out / "runtime-B-srv3.json"
        value = analyze.read_json(path)
        value["knobs"]["VLLM_GLM53_AR_COMPACT_CTA"] = "1"
        write(path, value)
        self.assertTrue(any(e["stage"] == "B srv3 runtime" for e in self.summary()["errors"]))
        memory = analyze.read_json(self.out / "memory-A.jsonl")
        memory["nodes"]["local"]["available_kib"] = 1
        write(self.out / "memory-A.jsonl", memory)
        self.assertTrue(any(e["stage"] == "A memory" for e in self.summary()["errors"]))

    def test_gpu_seal_missing_rank_or_old_flag_proof_cannot_admit(self):
        out = self.out / "transport-gpu"
        receipt = analyze.read_json(out / "admission.json")
        receipt["completed"].pop()
        write(out / "admission.json", receipt)
        self.assertTrue(any(e["stage"] == "transport GPU gate" for e in self.summary()["errors"]))
        (self.out / "sf6-gpu/probe.log").write_text("changed")
        self.assertTrue(any(e["stage"] == "SF6 GPU gate" for e in self.summary()["errors"]))

    def test_changed_ordered_requests_and_same_boot_fail(self):
        path = self.out / "channels-B.jsonl"
        channels = analyze.read_jsonl(path)
        self.records[1]["requests"][0]["request_sha256"] = "f" * 64
        channels[0]["request_sha256"] = "f" * 64
        channels[0]["timing"]["request_sha256"] = "f" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in channels))
        self.write_records()
        self.assertTrue(any(e["stage"] == "matched ordered requests" for e in self.summary()["errors"]))
        self.records[1]["boot_id"] = self.records[0]["boot_id"]
        self.write_records()
        self.assertIn("distinct", self.summary()["errors"][0]["error"])

    def test_canonical_complete_has_measured_coverage_without_fake_gpu_pass(self):
        self.canonical_evidence()
        result = self.canonical_summary()
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["mode"], "canonical")
        self.assertEqual(result["dedicated_gpu_correctness"], "not run under onepass-only policy")
        self.assertFalse(result["independent_stream_hashes_verified"])
        self.assertEqual(result["gpu"], {})
        self.assertEqual(result["coverage"], dict(measured_onepass_valid=True, startup_selftests_verified=True,
            independent_stream_hashes_verified=False, dedicated_gpu_correctness_verified=False))
        self.assertTrue(all(not row["independent_stream_hashes_verified"] for row in result["per_boot"]))
        self.assertIsNotNone(result["comparison"])
        self.assertFalse(self.summary()["valid"], "strict default must still require its independent gates")

    def test_canonical_missing_final_exit_is_pending_not_valid(self):
        self.canonical_evidence()
        (self.out / "campaign.exit").unlink()
        result = self.canonical_summary()
        self.assertEqual(result["status"], "PENDING", result["errors"])
        self.assertFalse(result["valid"])
        self.assertIsNone(result["comparison"])
        self.assertTrue(result["pending"])
        (self.out / "campaign.exit").write_text("7\n")
        self.assertEqual(self.canonical_summary()["status"], "INVALID")

    def test_canonical_missing_runtime_and_quality_failure_remain_invalid(self):
        self.canonical_evidence()
        (self.out / "prepared-A-srv4.json").unlink()
        self.records[0]["quality"]["ok"] = 17
        self.write_records()
        result = self.canonical_summary()
        self.assertEqual(result["status"], "INVALID")
        self.assertIsNone(result["comparison"])
        self.assertFalse(result["coverage"]["measured_onepass_valid"])
        self.assertFalse(result["coverage"]["startup_selftests_verified"])
        self.assertTrue(any(row["stage"] == "A srv4 runtime" for row in result["errors"]))
        self.assertTrue(any("18/18" in row["error"] for row in result["errors"]))

    def test_canonical_observer_completion_is_mandatory(self):
        self.canonical_evidence()
        path = self.out / "observer.json"
        correct = analyze.read_json(path)
        for mutate in (lambda r: r.update(status="RUNNING"), lambda r: r.update(errors=["missed snapshot"]),
                       lambda r: r["arms"]["A"].update(head_boot_id="another boot"),
                       lambda r: r["arms"]["A"].update(record_sha256="0" * 64),
                       lambda r: r["arms"].pop("B"), lambda r: r.update(source_commit="0" * 40)):
            observer = deepcopy(correct)
            mutate(observer)
            write(path, observer)
            result = self.canonical_summary()
            self.assertFalse(result["valid"])
            self.assertIsNone(result["comparison"])
            self.assertTrue(any(error["stage"] == "passive observer completion" for error in result["errors"]))
        path.unlink()
        self.assertFalse(self.canonical_summary()["valid"])

    def test_canonical_changed_observer_artifact_does_not_pass(self):
        self.canonical_evidence()
        path = self.out / "runtime-B-srv2.log"
        path.write_text("changed retained log\n")
        result = self.canonical_summary()
        self.assertFalse(result["valid"])
        self.assertTrue(any("sealed artifact" in row["error"] for row in result["errors"]))


if __name__ == "__main__":
    unittest.main()
