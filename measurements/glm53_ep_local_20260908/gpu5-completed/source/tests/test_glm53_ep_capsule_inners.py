"""CPU-only admission/failure tests of the three real inner entry points.

Accelerator imports are trapped; fake operations stand in for compilation and
GPU work. These tests do not validate numerics, sanitizer behavior or speed.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import builtins
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import glm53_ep_capsule_runtime as runtime_helper
import glm53_ep_local_check as local
import glm53_ep_local_compile_check as compiler
import glm53_ep_local_evidence as evidence_helper
import glm53_ep_route_remap_check as remap


class AcceleratorImport(RuntimeError):
    pass


class InnerCapsuleTests(unittest.TestCase):
    modules = {"local": local, "remap": remap, "cpu": compiler}

    def test_inners_can_be_loaded_as_packages_without_probes_on_path(self):
        # Compile evidence imports this module as probes.<name> in host
        # preparers. A prior top-level import must not hide a broken import.
        for name in ("glm53_ep_route_remap_check", "glm53_ep_local_check", "glm53_ep_local_compile_check"):
            with self.subTest(name=name):
                spec = importlib.util.spec_from_file_location(
                    "probes._capsule_inner_package_test", ROOT / ("probes/" + name + ".py"))
                module = importlib.util.module_from_spec(spec)
                with (patch.object(sys, "path", [str(ROOT)] +
                                   [entry for entry in sys.path if entry != str(ROOT / "probes")]),
                      patch.dict(sys.modules, {key: None for key in (
                          "glm53_ep_capsule_runtime", "glm53_ep_local_evidence", "glm53_ep_route_remap_check")})):
                    spec.loader.exec_module(module)
                self.assertTrue(callable(module.main))

    def argv(self, kind, output):
        args = ["inner", "--output", str(output),
                "--capsule-root", runtime_helper.CAPSULE_MOUNT,
                "--manifest-sha256", runtime_helper.CAPSULE_SHA256]
        return args + (["--arm", "local"] if kind == "cpu" else
                       ["--compile-evidence", "compile.json"] +
                       (["--case", "balanced4096"] if kind == "local" else []))

    def invoke(self, kind, *, compile_runtime="matching", first_error=None,
               operation_error=None, post_runtime="matching", post_error=None,
               real_work=False, devices=()):
        module = self.modules[kind]
        receipt = runtime_helper.expected_runtime_receipt()
        events = []
        verification_calls = []

        def verify(root, sha):
            self.assertEqual((str(root), sha),
                             (runtime_helper.CAPSULE_MOUNT, runtime_helper.CAPSULE_SHA256))
            verification_calls.append((root, sha))
            events.append("verify")
            if len(verification_calls) == 1:
                if first_error is not None:
                    raise first_error
                return deepcopy(receipt)
            self.assertEqual(len(verification_calls), 2)
            if post_error is not None:
                raise post_error
            return deepcopy(receipt if post_runtime == "matching" else post_runtime)

        def compiled(*_):
            events.append("compile-evidence")
            value = deepcopy(receipt if compile_runtime == "matching" else compile_runtime)
            result = dict(sources={}, mounted_sources={})
            if value is not None:
                result["binding_runtime"] = value
            return result

        def work(*args):
            result = args[0] if kind == "remap" else args[1]
            self.assertEqual(result["binding_runtime"], receipt)
            events.append("operation")
            result.update(phase="fixture-operation", partial_checks=["first completed check"])
            if operation_error is not None:
                raise operation_error

        original_import = builtins.__import__

        def no_accelerator_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "triton", "flashinfer", "vllm"}:
                events.append("import:" + name)
                raise AcceleratorImport("accelerator import trapped: " + name)
            return original_import(name, *args, **kwargs)

        original_read = Path.read_text
        original_glob = Path.glob

        def read_text(path, *args, **kwargs):
            if str(path) == "/repo/build/glm53/manifest.tsv":
                return ""
            return original_read(path, *args, **kwargs)

        def glob(path, pattern):
            if str(path) == "/dev" and pattern == "nvidia*":
                return iter(devices)
            return original_glob(path, pattern)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / ("compile" if kind == "cpu" else "result.json")
            result_path = output / "result.json" if kind == "cpu" else output
            stream = io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", self.argv(kind, output)))
                stack.enter_context(patch.object(module, "verify_runtime", side_effect=verify))
                owner = local if kind == "local" else evidence_helper
                stack.enter_context(patch.object(owner, "validate_compile_evidence", side_effect=compiled))
                stack.enter_context(patch.dict(local.os.environ))
                stack.enter_context(patch.object(Path, "read_text", read_text))
                stack.enter_context(patch.object(Path, "glob", glob))
                stack.enter_context(patch.object(builtins, "__import__", no_accelerator_import))
                stack.enter_context(redirect_stdout(stream))
                if not real_work:
                    name = {"cpu": "compile_arm", "local": "_run_case", "remap": "verify_gpu"}[kind]
                    stack.enter_context(patch.object(module, name, side_effect=work))
                error = None
                try:
                    module.main()
                except BaseException as exc:
                    error = exc
            result = json.loads(result_path.read_text())
        return error, result, events, stream.getvalue()

    def test_all_inner_clis_require_both_capsule_arguments(self):
        for kind, module in self.modules.items():
            for missing in ("--capsule-root", "--manifest-sha256"):
                with self.subTest(kind=kind, missing=missing):
                    argv = self.argv(kind, Path("unused"))
                    index = argv.index(missing)
                    del argv[index:index + 2]
                    with (patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()),
                          patch.object(module, "verify_runtime") as verify,
                          self.assertRaises(SystemExit) as raised):
                        module.main()
                    self.assertEqual(raised.exception.code, 2)
                    verify.assert_not_called()

    def test_invalid_runtime_stops_before_accelerator_import_or_operation(self):
        for kind in self.modules:
            with self.subTest(kind=kind):
                error, result, events, text = self.invoke(
                    kind, first_error=ValueError("bad capsule identity"), real_work=True)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(events, ["verify"])
                self.assertEqual((result["verdict"], result["phase"]), ("FAIL", "binding-runtime"))
                self.assertNotIn("binding_runtime", result)
                self.assertNotIn("PASS", text)

    def test_cpu_device_guard_still_precedes_runtime_and_compilation(self):
        error, result, events, _ = self.invoke("cpu", devices=[Path("/dev/nvidia0")])
        self.assertIsInstance(error, AssertionError)
        self.assertEqual(events, [])
        self.assertEqual((result["verdict"], result["phase"]), ("FAIL", "cpu-device-guard"))

    def test_gpu_rejects_missing_and_different_compile_runtime_before_import(self):
        changed = runtime_helper.expected_runtime_receipt()
        changed["pathfinder"]["sha256"] = "0" * 64
        extra = dict(runtime_helper.expected_runtime_receipt(), unexpected=True)
        for kind in ("local", "remap"):
            for receipt in (None, changed, extra):
                with self.subTest(kind=kind, receipt=receipt):
                    error, result, events, _ = self.invoke(kind, compile_runtime=receipt, real_work=True)
                    self.assertIsInstance(error, ValueError)
                    self.assertIn("runtimes differ", str(error))
                    self.assertEqual(events, ["verify", "compile-evidence"])
                    self.assertEqual((result["verdict"], result["phase"]), ("FAIL", "source-binding"))

    def test_real_operation_imports_are_after_admission_and_failure_is_rechecked(self):
        for kind in self.modules:
            with self.subTest(kind=kind):
                error, result, events, _ = self.invoke(kind, real_work=True)
                self.assertIsInstance(error, AcceleratorImport)
                prefix = ["verify"] + ([] if kind == "cpu" else ["compile-evidence"])
                self.assertEqual(events, prefix + ["import:torch", "verify"])
                self.assertEqual(result["verdict"], "FAIL")
                self.assertTrue(result["binding_runtime_rechecked"])

    def test_success_requires_matching_checks_before_and_after_operation(self):
        for kind in self.modules:
            with self.subTest(kind=kind):
                error, result, events, _ = self.invoke(kind)
                self.assertIsNone(error)
                prefix = ["verify"] + ([] if kind == "cpu" else ["compile-evidence"])
                self.assertEqual(events, prefix + ["operation", "verify"])
                self.assertEqual((result["verdict"], result["phase"]), ("PASS", "complete"))
                self.assertEqual(result["binding_runtime"], runtime_helper.expected_runtime_receipt())
                self.assertTrue(result["binding_runtime_rechecked"])
                if kind != "cpu":
                    self.assertFalse(result["performance_acceptance"])

    def test_post_operation_identity_change_prevents_pass_and_keeps_partial_results(self):
        altered = runtime_helper.expected_runtime_receipt()
        altered["capsule_manifest_sha256"] = "f" * 64
        for kind in self.modules:
            with self.subTest(kind=kind):
                error, result, events, text = self.invoke(kind, post_runtime=altered)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual((result["verdict"], result["phase"]), ("FAIL", "binding-runtime-recheck"))
                self.assertEqual(result["partial_checks"], ["first completed check"])
                self.assertNotIn("binding_runtime_rechecked", result)
                self.assertIn("binding runtime changed", result["binding_runtime_recheck_error"])
                self.assertEqual(events[-2:], ["operation", "verify"])
                self.assertNotIn("PASS", text)

    def test_failed_operation_keeps_primary_error_phase_and_records_secondary_error(self):
        for kind in self.modules:
            for secondary in (None, ValueError("capsule changed after operation")):
                with self.subTest(kind=kind, secondary=secondary):
                    primary = RuntimeError("original operation failure")
                    error, result, events, text = self.invoke(
                        kind, operation_error=primary, post_error=secondary)
                    self.assertIs(error, primary)
                    self.assertEqual((result["verdict"], result["phase"]), ("FAIL", "fixture-operation"))
                    self.assertEqual(result["error"], repr(primary))
                    self.assertEqual(result["partial_checks"], ["first completed check"])
                    self.assertEqual(events[-2:], ["operation", "verify"])
                    if secondary is None:
                        self.assertTrue(result["binding_runtime_rechecked"])
                    else:
                        self.assertEqual(result["binding_runtime_recheck_error"], repr(secondary))
                    self.assertNotIn("PASS", text)

    def test_postcheck_validation_exception_also_prevents_pass(self):
        for kind in self.modules:
            with self.subTest(kind=kind):
                error, result, _, text = self.invoke(kind, post_error=ValueError("mutated capsule file"))
                self.assertIsInstance(error, ValueError)
                self.assertEqual((result["verdict"], result["phase"]), ("FAIL", "binding-runtime-recheck"))
                self.assertEqual(result["partial_checks"], ["first completed check"])
                self.assertNotIn("PASS", text)


if __name__ == "__main__":
    unittest.main()
