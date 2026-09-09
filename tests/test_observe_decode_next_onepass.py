"""CPU-only fake ledger, container and SSH tests for passive observation."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "tests"))
import test_decode_next_runtime as fixtures

SPEC = importlib.util.spec_from_file_location("observer", ROOT / "probes/observe_decode_next_onepass.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)

REVISION = "a" * 40


class FakeReader:
    def __init__(self, mode="candidate", *, sf6_direct=False, sf6_unpack=False, mhc_active=False):
        self.mode, self.owner, self.rank_calls = mode, True, 0
        self.sf6_direct = sf6_direct
        self.sf6_unpack = sf6_unpack
        self.mhc_active = mhc_active
        self.suffix = ""
        self.on_ranks = None

    def owned(self):
        return self.owner

    def healthy(self):
        return True

    def head(self):
        factory = fixtures.unpack_report if self.sf6_unpack else fixtures.direct_report if self.sf6_direct else fixtures.report
        return dict(boot_id="srv2|" + self.mode + self.suffix, running=True,
                    image=observer.proof.IMAGE, knobs=factory(self.mode, "srv2")["knobs"])

    def ranks(self, mode, expected):
        self.rank_calls += 1
        if self.on_ranks:
            self.on_ranks()
        values = {}
        for host in observer.HOSTS:
            factory = fixtures.unpack_report if self.sf6_unpack else fixtures.direct_report if self.sf6_direct else fixtures.report
            report = factory(mode, host, mhc_active=self.mhc_active) if self.sf6_unpack else factory(mode, host)
            report["boot_id"] += self.suffix
            candidate_log = fixtures.SF6_DIRECT_LOG if self.sf6_direct else fixtures.CANDIDATE_LOG
            log = (fixtures.unpack_log(mode, mhc_active=self.mhc_active) if self.sf6_unpack else
                   fixtures.COMMON_LOG + (candidate_log if mode == "candidate" else "")).encode()
            report["log_sha256"] = observer.digest(log)
            values[host] = dict(report=report, log_b64=base64.b64encode(log).decode(), error="")
        return values


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.reader = FakeReader()
        self.agent = observer.Observer(self.out, "runA", "runB", REVISION,
                                       fixtures.MANIFEST, self.reader, clock=lambda: 100.0)

    def tearDown(self):
        self.agent.executor.shutdown(wait=True)
        self.tmp.cleanup()

    def record(self, name, boot=None):
        row = dict(name=name, boot_id=boot or self.reader.head()["boot_id"])
        raw = json.dumps(row).encode()
        with (self.out / "records.raw.jsonl").open("ab") as stream:
            stream.write(raw + b"\n")
        return observer.digest(raw)

    def test_two_arms_bind_four_ranks_original_records_and_logs(self):
        for name, mode in (("runA", "candidate"), ("runB", "baseline")):
            self.reader.mode = mode
            self.agent.snapshot(name, "prepared")
            sha = self.record(name)
            if mode == "baseline":
                self.reader.owner = False  # Final B survives canonical release.
            self.agent.snapshot(name, "runtime")
            arm = self.agent.state["arms"][name]
            self.assertEqual(arm["status"], "PASS")
            self.assertEqual(arm["record_sha256"], sha)
            receipt = json.loads((self.out / arm["after_receipt"]).read_text())
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(len(receipt["artifacts_sha256"]), 8)
            for filename, digest in receipt["artifacts_sha256"].items():
                self.assertEqual(observer.digest((self.out / filename).read_bytes()), digest)
            for host in observer.HOSTS:
                self.assertEqual((self.out / f"boot-{name}-{host}.log").read_bytes(),
                                 (self.out / f"runtime-{name}-{host}.log").read_bytes())
        self.assertTrue(self.agent.step())
        self.assertEqual(self.agent.state["status"], "PASS")
        import analyze_decode_next_onepass as analysis
        records = {name: value["record"] for name, value in self.agent.records().items()}
        checked = analysis.validate_observer(self.out, "runA", "runB", REVISION, records)
        self.assertEqual(checked["status"], "PASS")
        self.assertFalse((self.out / "campaign.exit").exists())

    def test_transition_keeps_failed_receipt_and_never_promotes_other_boot(self):
        self.agent.snapshot("runA", "prepared")
        self.record("runA")
        self.reader.mode = "baseline"
        self.agent.snapshot("runA", "runtime")
        self.assertEqual(self.agent.state["arms"]["runA"]["status"], "FAIL")
        self.assertFalse((self.out / "runtime-runA-srv2.json").exists())
        failures = [json.loads(path.read_text()) for path in (self.out / "attempts").glob("*/receipt.json")]
        self.assertEqual(sum(value["status"] == "FAIL" for value in failures), 1)
        failed = next(value for value in failures if value["status"] == "FAIL")
        self.assertTrue(any("head" in error for error in failed["errors"]))

    def test_record_arriving_during_prepared_read_is_not_prepared_success(self):
        self.reader.on_ranks = lambda: self.record("runA")
        self.agent.snapshot("runA", "prepared")
        self.assertFalse((self.out / "observed-prepared-runA.json").exists())
        receipt = json.loads(next((self.out / "attempts").glob("*/receipt.json")).read_text())
        self.assertEqual(receipt["status"], "FAIL")
        self.assertTrue(receipt["record_present_after"])

    def test_resume_preserves_valid_boot_and_rejects_changed_source_or_evidence(self):
        self.agent.snapshot("runA", "prepared")
        original = (self.out / "prepared-runA-srv2.json").read_bytes()
        resumed = observer.Observer(self.out, "runA", "runB", REVISION, fixtures.MANIFEST, self.reader)
        try:
            self.assertIsNotNone(resumed.prepared("runA"))
            self.reader.suffix = "different"
            resumed.snapshot("runA", "prepared")
            self.assertEqual((self.out / "prepared-runA-srv2.json").read_bytes(), original)
            self.assertTrue(resumed.state["arms"]["runA"]["failed_attempts"])
        finally:
            resumed.executor.shutdown(wait=True)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            observer.Observer(self.out, "runA", "runB", "b" * 40, fixtures.MANIFEST, self.reader)
        (self.out / "prepared-runA-srv2.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            observer.Observer(self.out, "runA", "runB", REVISION, fixtures.MANIFEST, self.reader)

    def test_foreign_baseline_and_foreign_candidate_never_collect(self):
        self.reader.owner = False
        for mode in ("baseline", "candidate"):
            self.reader.mode = mode
            self.assertFalse(self.agent.step())
        self.assertEqual(self.reader.rank_calls, 0)
        # Even an owned default boot cannot be B before A's record exists.
        self.reader.owner, self.reader.mode = True, "baseline"
        self.assertFalse(self.agent.step())
        self.assertEqual(self.reader.rank_calls, 0)

    def test_release_during_snapshot_fails_and_no_foreign_runtime_collection(self):
        self.reader.on_ranks = lambda: setattr(self.reader, "owner", False)
        self.agent.snapshot("runA", "prepared")
        self.assertFalse((self.out / "observed-prepared-runA.json").exists())
        calls = self.reader.rank_calls
        self.record("runA")
        self.agent.snapshot("runA", "runtime")
        self.assertEqual(self.reader.rank_calls, calls)
        self.assertEqual(self.agent.state["arms"]["runA"]["status"], "FAIL")

    def test_runtime_release_during_collection_keeps_latched_boot_valid(self):
        self.agent.snapshot("runA", "prepared")
        self.record("runA")
        self.reader.on_ranks = lambda: setattr(self.reader, "owner", False)
        self.agent.snapshot("runA", "runtime")
        self.assertEqual(self.agent.state["arms"]["runA"]["status"], "PASS")
        receipt = json.loads((self.out / "observed-runtime-runA.json").read_text())
        self.assertTrue(receipt["owned_before"])
        self.assertFalse(receipt["owned_after"])
        self.assertTrue(receipt["runtime_bound_before"])

    def test_runtime_after_release_new_boot_never_collects_or_passes(self):
        self.agent.snapshot("runA", "prepared")
        self.record("runA")
        self.reader.owner, self.reader.suffix = False, "new-boot"
        calls = self.reader.rank_calls
        self.agent.snapshot("runA", "runtime")
        self.assertEqual(self.reader.rank_calls, calls)
        self.assertEqual(self.agent.state["arms"]["runA"]["status"], "FAIL")
        self.assertFalse((self.out / "observed-runtime-runA.json").exists())

    def test_after_release_still_rejects_mounted_source_drift(self):
        self.agent.snapshot("runA", "prepared")
        self.record("runA")
        self.reader.owner = False
        original = self.reader.ranks
        def changed(mode, expected):
            ranks = original(mode, expected)
            ranks["srv3"]["report"]["source_sha256"] = {"/pkg/changed.py": "f" * 64}
            return ranks
        with patch.object(self.reader, "ranks", side_effect=changed):
            self.agent.snapshot("runA", "runtime")
        self.assertEqual(self.agent.state["arms"]["runA"]["status"], "FAIL")
        self.assertFalse((self.out / "observed-runtime-runA.json").exists())

    def test_runtime_latch_rejects_head_image_configuration_or_missing_record(self):
        self.agent.snapshot("runA", "prepared")
        self.record("runA")
        record = self.agent.records()["runA"]
        prepared = self.agent.prepared("runA")
        self.assertEqual(observer.runtime_binding_errors("candidate", fixtures.MANIFEST,
                         self.reader.head(), record, prepared), [])
        for field, value in (("image", "other-image"), ("knobs", {}), ("running", False)):
            head = dict(self.reader.head(), **{field: value})
            self.assertTrue(observer.runtime_binding_errors("candidate", fixtures.MANIFEST,
                            head, record, prepared), field)
        self.assertTrue(observer.runtime_binding_errors("candidate", fixtures.MANIFEST,
                        self.reader.head(), None, prepared))

    def test_partial_record_is_ignored_and_duplicate_complete_records_fail(self):
        path = self.out / "records.raw.jsonl"
        raw = b'{"name":"runA","boot_id":"actual"}'
        path.write_bytes(raw)
        self.assertEqual(observer.read_records(path, {"runA"}), {})
        path.write_bytes(raw + b"\r\n")
        self.assertEqual(observer.read_records(path, {"runA"})["runA"]["sha256"], observer.digest(raw))
        path.write_bytes((raw + b"\n") * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            observer.read_records(path, {"runA"})

    def test_polling_continues_while_async_rank_snapshot_is_running(self):
        from concurrent.futures import Future
        pending = Future()
        self.agent.pending = pending
        with patch.object(self.reader, "head", wraps=self.reader.head) as head:
            for _ in range(3):
                self.assertFalse(self.agent.step())
            self.assertEqual(head.call_count, 3)
        self.assertEqual(self.reader.rank_calls, 0)

    def test_terminal_ticket_failure_is_read_only_and_ticket_bound(self):
        directory = self.out / "fleet"
        (directory / "pending").mkdir(parents=True)
        path = directory / "pending" / (observer.digest(b"session") + ".json")
        value = dict(ticket="ticket", state="paused")
        path.write_text(json.dumps(value))
        self.assertIsNone(observer.terminal_failure(directory, "session", "ticket"))
        self.assertIn("ticket changed", observer.terminal_failure(directory, "session", "new"))
        value.update(state="finished", outcome="failed", returncode=1)
        path.write_text(json.dumps(value))
        before = path.read_bytes()
        self.assertIn("terminated", observer.terminal_failure(directory, "session", "ticket"))
        self.assertEqual(path.read_bytes(), before)
        for success in (dict(state="finished", returncode=0),
                        dict(state="finished", outcome="succeeded")):
            path.write_text(json.dumps(dict(ticket="ticket", **success)))
            self.assertIsNone(observer.terminal_failure(directory, "session", "ticket"))
        path.write_text(json.dumps(dict(ticket="ticket", state="cancelled", returncode=0)))
        self.assertIn("cancelled", observer.terminal_failure(directory, "session", "ticket"))


class ReaderTests(unittest.TestCase):
    def test_source_manifest_rejects_stale_composed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            (repo / "profiles").mkdir()
            (repo / "profiles/glm53.env").write_text("MODULES='one'\nTARGET_PREFIX=/pkg/\n")
            module = repo / "overlay/modules/one"
            module.mkdir(parents=True)
            (module / "manifest.tsv").write_text("file.py\tfile.py\tabsent\n")
            (module / "file.py").write_text("x=1\n")
            build = repo / "build/glm53"
            build.mkdir(parents=True)
            (build / "manifest.tsv").write_text("file.py\t/pkg/file.py\tabsent\n")
            (build / "file.py").write_text("x=1\n")
            with patch.object(observer.onepass_deploy, "source_revision", return_value=REVISION):
                revision, manifest = observer.frozen_manifest(repo)
                self.assertEqual(revision, REVISION)
                self.assertEqual(manifest, {"/pkg/file.py": observer.digest(b"x=1\n")})
                (build / "file.py").write_text("stale")
                with self.assertRaisesRegex(ValueError, "stale composed"):
                    observer.frozen_manifest(repo)

    def test_reader_uses_ssh_stdin_read_script_and_only_health_http(self):
        reader = observer.Reader(ROOT, 8000, "run", Path("/unavailable"))
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, json.dumps({"report": {}, "error": "fixture"}), "")
        with patch.object(observer.subprocess, "run", side_effect=run):
            reader.rank("srv1", "candidate", fixtures.MANIFEST)
        command, kwargs = calls[0]
        self.assertEqual(command[-1], "python3 -")
        self.assertEqual(command[0], "ssh")
        self.assertIn("collect_report", kwargs["input"])
        self.assertNotIn("docker stop", kwargs["input"])
        self.assertNotIn("import torch", kwargs["input"])
        self.assertFalse(reader.owned())
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.status = 200
        with patch.object(observer, "urlopen", return_value=response) as get:
            self.assertTrue(reader.healthy())
        get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=2)

    def test_embedded_collector_binds_variant_for_collect_and_validation(self):
        for direct, unpack in ((False, False), (True, False), (False, True)):
            reader = observer.Reader(ROOT, 18000 if unpack else 8000, "run", Path("/unavailable"),
                                     sf6_direct=direct, sf6_unpack=unpack)
            result = subprocess.CompletedProcess([], 0, '{"report": {}, "error": "fixture"}', "")
            with patch.object(observer.subprocess, "run", return_value=result) as run:
                reader.rank("srv1", "candidate", fixtures.MANIFEST)
            script = run.call_args.kwargs["input"]
            self.assertIn("\nSF6_DIRECT=" + repr(direct) + "\n", script)
            self.assertIn("\nSF6_UNPACK=" + repr(unpack) + "\n", script)
            self.assertIn('ns["collect_report"](MODE, EXPECTED, sf6_direct=SF6_DIRECT, sf6_unpack=SF6_UNPACK)', script)
            self.assertIn('ns["validate_report"](report, EXPECTED, sf6_direct=SF6_DIRECT, sf6_unpack=SF6_UNPACK)', script)
            self.assertIn('ns["parse_markers"](raw.decode(errors="replace"), sf6_unpack=SF6_UNPACK)', script)


class DirectObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.reader = FakeReader(sf6_direct=True)
        self.agent = self.make_observer()

    def make_observer(self, *, sf6_direct=True, reader=None):
        value = observer.Observer(self.out, "runA", "runB", REVISION, fixtures.MANIFEST,
                                  reader or self.reader, sf6_direct=sf6_direct, clock=lambda: 100.)
        self.addCleanup(value.executor.shutdown, wait=True)
        return value

    def record(self, name):
        row = dict(name=name, boot_id=self.reader.head()["boot_id"])
        with (self.out / "records.raw.jsonl").open("ab") as stream:
            stream.write(json.dumps(row).encode() + b"\n")

    def test_direct_pair_binds_state_receipts_and_restart(self):
        for name, mode in (("runA", "candidate"), ("runB", "baseline")):
            self.reader.mode = mode
            self.agent.snapshot(name, "prepared")
            self.record(name)
            self.agent.snapshot(name, "runtime")
            self.assertEqual(self.agent.state["arms"][name]["status"], "PASS")
            for phase in ("prepared", "runtime"):
                receipt = json.loads((self.out / f"observed-{phase}-{name}.json").read_text())
                self.assertIs(receipt["sf6_direct"], True)
        self.assertTrue(self.agent.step())
        self.assertEqual(self.agent.state["status"], "PASS")
        self.assertIs(self.agent.state["sf6_direct"], True)
        resumed = self.make_observer()
        self.assertEqual(resumed.state["status"], "PASS")
        self.assertEqual(resumed.prepared("runA"), self.agent.prepared("runA"))
        with self.assertRaisesRegex(ValueError, "different SF6 variant"):
            self.make_observer(sf6_direct=False, reader=FakeReader())

    def test_direct_preparation_rejects_legacy_transport_candidate(self):
        self.agent.reader = FakeReader()
        self.assertIsNone(observer.mode_from_metadata(self.agent.reader.head(), sf6_direct=True))
        self.agent.snapshot("runA", "prepared")
        self.assertFalse((self.out / "observed-prepared-runA.json").exists())
        receipt = json.loads(next((self.out / "attempts").glob("*/receipt.json")).read_text())
        self.assertIs(receipt["sf6_direct"], True)
        self.assertEqual(receipt["status"], "FAIL")

    def test_runtime_rejects_one_rank_variant_drift(self):
        self.agent.snapshot("runA", "prepared")
        self.record("runA")
        original = self.reader.ranks
        def mixed(mode, expected):
            values = original(mode, expected)
            values["srv4"]["report"]["sf6_direct"] = False
            return values
        with patch.object(self.reader, "ranks", side_effect=mixed):
            self.agent.snapshot("runA", "runtime")
        self.assertEqual(self.agent.state["arms"]["runA"]["status"], "FAIL")
        self.assertFalse((self.out / "observed-runtime-runA.json").exists())

    def test_restart_refuses_missing_or_changed_direct_receipt_identity(self):
        self.agent.snapshot("runA", "prepared")
        path = self.out / "observed-prepared-runA.json"
        original = json.loads(path.read_text())
        for value in (False, None, 1, "true"):
            receipt = dict(original)
            if value is None:
                receipt.pop("sf6_direct")
            else:
                receipt["sf6_direct"] = value
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "different SF6 variant"):
                self.make_observer()
        path.write_text(json.dumps(original))
        self.assertIsNotNone(self.make_observer().prepared("runA"))

    def test_legacy_restart_allows_absent_false_identity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = FakeReader()
            original = observer.Observer(directory, "runA", "runB", REVISION, fixtures.MANIFEST, reader)
            self.addCleanup(original.executor.shutdown, wait=True)
            original.snapshot("runA", "prepared")
            for filename in ("observer.json", "observed-prepared-runA.json"):
                path = Path(directory) / filename
                value = json.loads(path.read_text())
                self.assertIs(value.pop("sf6_direct"), False)
                path.write_text(json.dumps(value))
            resumed = observer.Observer(directory, "runA", "runB", REVISION, fixtures.MANIFEST, reader)
            self.addCleanup(resumed.executor.shutdown, wait=True)
            self.assertIs(resumed.state["sf6_direct"], False)
            self.assertIsNotNone(resumed.prepared("runA"))
            with self.assertRaisesRegex(ValueError, "different SF6 variant"):
                observer.Observer(directory, "runA", "runB", REVISION, fixtures.MANIFEST,
                                  FakeReader(sf6_direct=True), sf6_direct=True)

    def test_reader_variant_mismatch_is_rejected_before_writing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "reader and observer SF6 variant"):
                observer.Observer(Path(directory) / "out", "runA", "runB", REVISION,
                                  fixtures.MANIFEST, FakeReader(), sf6_direct=True)
            self.assertFalse((Path(directory) / "out").exists())

    def test_cli_forwards_direct_flag_to_reader_observer_and_ready(self):
        import io
        from unittest.mock import MagicMock
        for options, direct in (([], False), (["--sf6-direct"], True)):
            fake = MagicMock()
            fake.step.return_value = True
            fake.state = {"status": "PASS"}
            output = io.StringIO()
            with patch.object(observer, "frozen_manifest", return_value=(REVISION, fixtures.MANIFEST)), \
                 patch.object(observer, "Reader") as reader_class, \
                 patch.object(observer, "Observer", return_value=fake) as observer_class, \
                 patch.object(observer.sys, "stdout", output):
                self.assertEqual(observer.main(["--out", str(self.out), "--candidate", "runA",
                    "--baseline", "runB", "--session", "fixture-session", *options]), 0)
            self.assertIs(reader_class.call_args.kwargs["sf6_direct"], direct)
            self.assertIs(observer_class.call_args.kwargs["sf6_direct"], direct)
            ready = json.loads(output.getvalue().split("READY ", 1)[1])
            self.assertIs(ready["sf6_direct"], direct)


class UnpackObserverTests(unittest.TestCase):
    def make_observer(self, out, reader):
        value = observer.Observer(out, "runA", "runB", REVISION, fixtures.MANIFEST,
                                  reader, sf6_unpack=True, clock=lambda: 100.)
        self.addCleanup(value.executor.shutdown, wait=True)
        return value

    def test_unpack_two_arms_keep_receipts_and_actual_state(self):
        import analyze_decode_next_onepass as analysis
        for active in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                reader = FakeReader(sf6_unpack=True, mhc_active=active)
                agent = self.make_observer(out, reader)
                for name, mode in (("runA", "candidate"), ("runB", "baseline")):
                    reader.mode = mode
                    self.assertEqual(observer.mode_from_metadata(reader.head(), sf6_unpack=True), mode)
                    agent.snapshot(name, "prepared")
                    row = dict(name=name, boot_id=reader.head()["boot_id"])
                    with (out / "records.raw.jsonl").open("ab") as stream:
                        stream.write(json.dumps(row).encode() + b"\n")
                    if mode == "baseline":
                        reader.owner = False
                    agent.snapshot(name, "runtime")
                    self.assertEqual(agent.state["arms"][name]["status"], "PASS")
                    for phase in ("prepared", "runtime"):
                        receipt = json.loads((out / f"observed-{phase}-{name}.json").read_text())
                        self.assertIs(receipt["sf6_unpack"], True)
                        self.assertIs(receipt["sf6_direct"], False)
                self.assertTrue(agent.step())
                self.assertEqual(agent.state["status"], "PASS")
                records = {name: item["record"] for name, item in agent.records().items()}
                self.assertEqual(analysis.validate_observer(out, "runA", "runB", REVISION,
                    records, sf6_unpack=True)["status"], "PASS")
                self.assertEqual(self.make_observer(out, reader).state["status"], "PASS")

    def test_unpack_missing_variant_or_actual_state_never_becomes_prepared(self):
        for field in ("sf6_unpack", "mhc_consumer_failures"):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                reader = FakeReader(sf6_unpack=True)
                original = reader.ranks
                def missing(mode, expected):
                    ranks = original(mode, expected)
                    report = ranks["srv4"]["report"]
                    if field == "sf6_unpack":
                        report.pop(field)
                    else:
                        report["markers"][field] = []
                    return ranks
                reader.ranks = missing
                agent = self.make_observer(out, reader)
                agent.snapshot("runA", "prepared")
                self.assertFalse((out / "observed-prepared-runA.json").exists())

    def test_unpack_restart_refuses_missing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            reader = FakeReader(sf6_unpack=True)
            agent = self.make_observer(out, reader)
            agent.snapshot("runA", "prepared")
            path = out / "observed-prepared-runA.json"
            receipt = json.loads(path.read_text())
            receipt.pop("sf6_unpack")
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "different SF6 variant"):
                self.make_observer(out, reader)

    def test_unpack_cli_binds_reader_observer_ready_and_rejects_other_port(self):
        import io
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.step.return_value = True
        fake.state = {"status": "PASS"}
        output = io.StringIO()
        argv = ["--out", "/unused", "--candidate", "runA", "--baseline", "runB",
                "--session", "fixture-session", "--sf6-unpack", "--port", "18000"]
        with patch.object(observer, "frozen_manifest", return_value=(REVISION, fixtures.MANIFEST)), \
             patch.object(observer, "Reader") as reader_class, \
             patch.object(observer, "Observer", return_value=fake) as observer_class, \
             patch.object(observer.sys, "stdout", output):
            self.assertEqual(observer.main(argv), 0)
        self.assertIs(reader_class.call_args.kwargs["sf6_unpack"], True)
        self.assertIs(observer_class.call_args.kwargs["sf6_unpack"], True)
        self.assertIs(json.loads(output.getvalue().split("READY ", 1)[1])["sf6_unpack"], True)
        for changed in (argv[:-1] + ["8000"], argv + ["--sf6-direct"]):
            with self.assertRaises(SystemExit), patch.object(observer.sys, "stderr", io.StringIO()):
                observer.main(changed)


if __name__ == "__main__":
    unittest.main()
