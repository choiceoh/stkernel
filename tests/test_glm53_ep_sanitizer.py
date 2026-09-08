"""No-device tests of sanitizer admission, receipts, and false-success rejection."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import glm53_ep_sanitizer as sanitizer
import run_glm53_ep_local_offline as runner

IMAGE = "sha256:" + "a" * 64
IDENTITY = dict(directory="/host/sanitizer", executable="/host/sanitizer/compute-sanitizer",
                sha256=sanitizer.BINARY_SHA256)


class PreflightTests(unittest.TestCase):
    def test_pinned_version_runs_in_bounded_no_device_container_and_records_identity(self):
        checked = subprocess.CompletedProcess([], 0, "Compute Sanitizer version " + sanitizer.VERSION, "")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "receipt.json"
            with patch.object(sanitizer, "identity", return_value=IDENTITY), \
                    patch.object(sanitizer.subprocess, "run", return_value=checked) as run:
                result = sanitizer.preflight(IMAGE, output)
            args = run.call_args.args[0]
            for value in ("--runtime=runc", "--network=none", "--memory=256m",
                          "--memory-swap=256m", "--cpus=1", "--pids-limit=64",
                          "NVIDIA_VISIBLE_DEVICES=void", "CUDA_VISIBLE_DEVICES=", IMAGE):
                self.assertIn(value, args)
            self.assertNotIn("--gpus", args)
            self.assertIn("type=bind,source=/host/sanitizer,target=/opt/glm-ep-sanitizer,readonly", args)
            self.assertIn("assert not list(Path('/dev').glob('nvidia*'))", args[-1])
            self.assertIn("timeout=15", args[-1])
            self.assertEqual(run.call_args.kwargs["timeout"], 45)
            self.assertEqual(result["identity"], IDENTITY)
            self.assertEqual(result["verdict"], "PASS")
            self.assertFalse(result["cuda_devices_exposed"])
            self.assertEqual(json.loads(output.read_text()), result)

    def test_missing_executable_and_wrong_hash_fail_before_container(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            binary = directory / "compute-sanitizer"
            with patch.object(sanitizer, "HOST_DIR", directory), \
                    patch.object(sanitizer.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    sanitizer.preflight(IMAGE, directory / "missing.json")
                binary.write_bytes(b"not the pinned binary")
                binary.chmod(0o755)
                with self.assertRaisesRegex(RuntimeError, "hash changed"):
                    sanitizer.preflight(IMAGE, directory / "hash.json")
            run.assert_not_called()
            self.assertEqual(json.loads((directory / "hash.json").read_text())["verdict"], "FAIL")

    def test_nonzero_exit_or_missing_version_cannot_pass(self):
        for code, stdout in ((127, "Compute Sanitizer version " + sanitizer.VERSION),
                             (0, ""), (0, "Compute Sanitizer version 2025.2.0.0")):
            with self.subTest(code=code, stdout=stdout), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "receipt.json"
                checked = subprocess.CompletedProcess([], code, stdout, "")
                with patch.object(sanitizer, "identity", return_value=IDENTITY), \
                        patch.object(sanitizer.subprocess, "run", return_value=checked):
                    with self.assertRaises(RuntimeError):
                        sanitizer.preflight(IMAGE, output)
                failed = json.loads(output.read_text())
                self.assertEqual(failed["verdict"], "FAIL")
                self.assertIsNone(failed["cuda_devices_exposed"])

    def test_outer_timeout_removes_only_its_preflight_container(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "receipt.json"
            with patch.object(sanitizer, "identity", return_value=IDENTITY), \
                    patch.object(sanitizer.subprocess, "run", side_effect=[
                        subprocess.TimeoutExpired("docker run", 45),
                        subprocess.CompletedProcess([], 0, "", "")]) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    sanitizer.preflight(IMAGE, output)
            launch, cleanup = [call.args[0] for call in run.call_args_list]
            self.assertEqual(cleanup, ["docker", "rm", "-f", launch[launch.index("--name") + 1]])
            self.assertEqual(json.loads(output.read_text())["verdict"], "FAIL")

    def test_changed_executable_cannot_reuse_successful_preflight(self):
        receipt = dict(verdict="PASS", version=sanitizer.VERSION, identity=IDENTITY)
        with patch.object(sanitizer, "identity", return_value=dict(IDENTITY, directory="/changed")):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                sanitizer.mount_args(receipt)


class SummaryTests(unittest.TestCase):
    ERROR_ZERO = "========= ERROR SUMMARY: 0 errors\n"
    RACE_ZERO = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"

    def test_tool_specific_clean_summaries_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "sanitizer.log"
            for tool, text in (("memcheck", self.ERROR_ZERO), ("racecheck", self.RACE_ZERO)):
                with self.subTest(tool=tool):
                    log.write_text("Probe PASS\n" + text)
                    result = sanitizer.validate_summary(log, tool)
                    self.assertEqual(result["verdict"], "PASS")
                    self.assertEqual(result["tool"], tool)
                    self.assertEqual(len(result["log_sha256"]), 64)

    def test_absent_errors_ignored_errors_and_warnings_are_rejected(self):
        cases = (("memcheck", "Probe PASS\n"), ("racecheck", self.ERROR_ZERO),
                 ("memcheck", self.RACE_ZERO),
                 ("memcheck", "========= ERROR SUMMARY: 1 errors\n" + self.ERROR_ZERO),
                 ("memcheck", "========= ERROR SUMMARY: 0 errors (1 errors ignored)\n"),
                 ("memcheck", "========= Error: failed to launch application\n" + self.ERROR_ZERO),
                 ("racecheck", "========= RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)\n"),
                 ("racecheck", self.RACE_ZERO + "========= ERROR SUMMARY: 1 errors\n"))
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "sanitizer.log"
            for tool, text in cases:
                with self.subTest(tool=tool, text=text):
                    log.write_text(text)
                    with self.assertRaises(RuntimeError):
                        sanitizer.validate_summary(log, tool)


class RunnerOrderTests(unittest.TestCase):
    def test_failed_preflight_never_reaches_service_inventory_or_pause(self):
        with tempfile.TemporaryDirectory() as folder:
            capsule = Path(folder).resolve() / "capsule"
            output = Path(folder).resolve() / "capture"
            args = ["runner", "--revision", "a" * 40, "--out", str(output),
                    "--capsule-root", str(capsule),
                    "--manifest-sha256", runner.capsule_runtime.CAPSULE_SHA256]
            proof = {"binding_runtime": runner.capsule_runtime.expected_runtime_receipt()}
            with patch.object(sys, "argv", args), patch.object(runner.signal, "signal"), \
                    patch.object(runner.lifecycle, "check_holder"), \
                    patch.object(runner.lifecycle, "pinned"), \
                    patch.object(runner.capsule_runtime, "validate_capsule_input", return_value=capsule) as capsule_check, \
                    patch.object(runner, "validate_compile_evidence", return_value=proof) as compile_check, \
                    patch.object(runner.sanitizer_support, "preflight", side_effect=RuntimeError("missing tool")) as preflight, \
                    patch.object(runner, "resources") as resources, \
                    patch.object(runner.lifecycle, "snapshot") as snapshot, \
                    patch.object(runner.lifecycle, "with_paused") as pause, \
                    patch.object(runner.subprocess, "run") as process, \
                    patch.object(runner.subprocess, "check_output") as process_output, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 1)
            # Reach the intended failure after capsule/CPU admission, rather
            # than accidentally passing on an earlier argument or proof error.
            compile_check.assert_called_once_with(ROOT, ROOT / runner.CPU_EVIDENCE)
            preflight.assert_called_once_with(runner.lifecycle.IMAGE, output / "sanitizer-preflight.json")
            self.assertEqual(capsule_check.call_count, 2)  # admission and final integrity check
            completion = json.loads((output / "completion.json").read_text())
            self.assertEqual(completion["error"], "RuntimeError('missing tool')")
            self.assertEqual(completion["binding_runtime"], proof["binding_runtime"])
            self.assertEqual(completion["cells"], [])
            resources.assert_not_called()
            snapshot.assert_not_called()
            pause.assert_not_called()
            process.assert_not_called()
            process_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
