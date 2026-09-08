"""Mocked admission/launch/recovery contracts; no Docker or CUDA execution."""
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import glm53_ep_capsule_runtime as runtime
import run_glm53_ep_local_cpu_compile as cpu
import run_glm53_ep_local_offline as gpu

SHA = runtime.CAPSULE_SHA256


@contextmanager
def cpu_fixture():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        parent = Path(directory).resolve()
        root, capsule, output = parent / "source", parent / "capsule", parent / "output"
        (root / "build/glm53").mkdir(parents=True)
        (root / "build/glm53/manifest.tsv").write_text("moe.py\t/installed/flashinfer/moe.py\tunused\n")
        capsule.mkdir()
        args = ["cpu", "--image", "sha256:" + "a" * 64, "--arm", "local", "--output", str(output),
                "--capsule-root", str(capsule), "--manifest-sha256", SHA]
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return "MemAvailable: 12582912 kB\n"
            return read_text(path, *args, **kwargs)

        stack.enter_context(patch.object(cpu, "__file__", str(root / "probes/run_glm53_ep_local_cpu_compile.py")))
        stack.enter_context(patch.object(sys, "argv", args))
        stack.enter_context(patch.object(Path, "read_text", read))
        capsule_check = stack.enter_context(patch.object(runtime, "validate_capsule_input", return_value=capsule))
        launch = stack.enter_context(patch.object(cpu.subprocess, "call", return_value=0))
        result = stack.enter_context(patch.object(cpu, "validate_result", return_value={}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        yield SimpleNamespace(root=root, capsule=capsule, output=output, args=args,
                              capsule_check=capsule_check, launch=launch, result=result)


@contextmanager
def gpu_fixture():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        parent = Path(directory).resolve()
        root, capsule, output = parent / "source", parent / "capsule", parent / "output"
        (root / "build/glm53").mkdir(parents=True)
        (root / "build/glm53/manifest.tsv").write_text("moe.py\t/installed/flashinfer/moe.py\tunused\n")
        capsule.mkdir()
        args = ["gpu", "--revision", "a" * 40, "--out", str(output),
                "--capsule-root", str(capsule), "--manifest-sha256", SHA]
        h = SimpleNamespace(root=root, capsule=capsule, output=output, args=args, events=[], commands=[],
                            proof_hook=None, capsule_hook=None, cell_runtime=runtime.expected_runtime_receipt(),
                            docker_exit=0, cell_status={})

        def capsule_check(path, sha):
            h.events.append("capsule")
            if sha != SHA:
                raise ValueError("wrong capsule SHA")
            if h.capsule_hook:
                h.capsule_hook()
            return capsule

        def proof(source, path):
            h.events.append("proof")
            if h.proof_hook:
                h.proof_hook()
            return {"binding_runtime": runtime.expected_runtime_receipt()}

        before = {node: {"running": False, "port": 8000} for node in gpu.lifecycle.NODES}

        def snapshot():
            h.events.append("inventory")
            return before

        def paused(states, run, save, *, before_restore):
            h.events.append("pause")
            try:
                run()
            finally:
                before_restore()
                save("restored.json", states)
                h.events.append("restored")

        def process(command, **kwargs):
            if command[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(command, 1, "", "")
            if command[:2] != ["docker", "run"]:
                raise AssertionError("unexpected mocked process: " + repr(command))
            h.events.append("docker")
            h.commands.append(command)
            name = Path(command[command.index("--output") + 1]).name
            evidence = dict(verdict="PASS", phase="complete", binding_runtime_rechecked=True,
                            performance_acceptance=False, binding_runtime=h.cell_runtime)
            evidence.update(h.cell_status)
            (output / name).write_text(json.dumps(evidence))
            return subprocess.CompletedProcess(command, h.docker_exit)

        stack.enter_context(patch.object(gpu, "__file__", str(root / "probes/run_glm53_ep_local_offline.py")))
        stack.enter_context(patch.object(sys, "argv", args))
        stack.enter_context(patch.dict(gpu.os.environ, FLEET_SESSION="capsule-wrapper-unit"))
        stack.enter_context(patch.object(gpu.signal, "signal"))
        stack.enter_context(patch.object(gpu.lifecycle, "check_holder"))
        stack.enter_context(patch.object(gpu.lifecycle, "pinned"))
        h.capsule_check = stack.enter_context(patch.object(runtime, "validate_capsule_input", side_effect=capsule_check))
        h.proof = stack.enter_context(patch.object(gpu, "validate_compile_evidence", side_effect=proof))
        h.inventory = stack.enter_context(patch.object(gpu.lifecycle, "snapshot", side_effect=snapshot))
        stack.enter_context(patch.object(gpu.lifecycle, "validate_before", return_value="stopped"))
        h.pause = stack.enter_context(patch.object(gpu.lifecycle, "with_paused", side_effect=paused))
        stack.enter_context(patch.object(gpu, "resources", return_value={}))
        h.preflight = stack.enter_context(patch.object(gpu.sanitizer_support, "preflight", return_value={"verdict": "PASS"}))
        stack.enter_context(patch.object(gpu.sanitizer_support, "mount_args", return_value=["--mount", "sanitizer,readonly"]))
        stack.enter_context(patch.object(gpu.sanitizer_support, "command", side_effect=lambda tool: ["compute-sanitizer", "--tool", tool]))
        stack.enter_context(patch.object(gpu.sanitizer_support, "validate_summary", return_value={"verdict": "PASS"}))
        stack.enter_context(patch.object(gpu.subprocess, "run", side_effect=process))
        stack.enter_context(patch.object(gpu.subprocess, "check_output", return_value=""))
        stack.enter_context(patch.object(gpu, "CASES", ("balanced4096",)))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        yield h


class CpuWrapperTests(unittest.TestCase):
    def test_capsule_arguments_are_required_on_both_outer_clis(self):
        for module, args in ((cpu, ["cpu", "--image", "sha256:" + "a" * 64, "--arm", "local", "--output", "/tmp/out"]),
                             (gpu, ["gpu", "--revision", "a" * 40, "--out", "/tmp/out"])):
            with self.subTest(module=module.__name__), patch.object(sys, "argv", args), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.main()

    def test_cpu_command_uses_readonly_capsule_fixed_inner_path_and_existing_limits(self):
        with cpu_fixture() as h:
            self.assertEqual(cpu.main(), 0)
            command = h.launch.call_args.args[0]
            for value in ("--runtime=runc", "--network=none", "--memory=4g", "--memory-swap=4g",
                          "--cpus=2", "NVIDIA_VISIBLE_DEVICES=void", "-B", "PYTHONNOUSERSITE=1",
                          "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + runtime.CAPSULE_MOUNT):
                self.assertIn(value, command)
            self.assertFalse(any(arg.startswith(("--gpus", "--device")) for arg in command))
            self.assertIn(f"type=bind,source={h.capsule},target={runtime.CAPSULE_MOUNT},readonly", command)
            self.assertEqual(command[command.index("--capsule-root") + 1], runtime.CAPSULE_MOUNT)
            self.assertEqual(command[command.index("--manifest-sha256") + 1], SHA)
            h.result.assert_called_once_with(h.root, h.output / "result.json", "local")
            self.assertEqual(h.capsule_check.call_count, 3)  # input, Docker args, postcheck

    def test_invalid_capsule_is_rejected_before_output_or_docker(self):
        with cpu_fixture() as h:
            h.capsule_check.side_effect = ValueError("capsule changed")
            with self.assertRaisesRegex(ValueError, "capsule changed"):
                cpu.main()
            self.assertFalse(h.output.exists())
            h.launch.assert_not_called()

    def test_output_cannot_overlap_source_or_capsule_in_either_direction(self):
        for factory, module, option in ((cpu_fixture, cpu, "--output"), (gpu_fixture, gpu, "--out")):
            for relation in ("inside_source", "inside_capsule", "ancestor"):
                with self.subTest(module=module.__name__, relation=relation), factory() as h:
                    target = {"inside_source": h.root / "new-output", "inside_capsule": h.capsule / "new-output",
                              "ancestor": h.root.parent}[relation]
                    h.args[h.args.index(option) + 1] = str(target)
                    with self.assertRaises(SystemExit):
                        module.main()
                    if relation != "ancestor":
                        self.assertFalse(target.exists())
                    if module is cpu:
                        h.launch.assert_not_called()
                    else:
                        h.inventory.assert_not_called()
                        self.assertEqual(h.commands, [])

    def test_cpu_postcheck_failure_cannot_mask_docker_failure_or_exception(self):
        for outcome in (42, RuntimeError("original Docker error")):
            with self.subTest(outcome=repr(outcome)), cpu_fixture() as h:
                h.capsule_check.side_effect = [h.capsule, h.capsule, ValueError("postcheck changed")]
                if isinstance(outcome, Exception):
                    h.launch.side_effect = outcome
                    with self.assertRaisesRegex(RuntimeError, "original Docker error"):
                        cpu.main()
                else:
                    h.launch.return_value = outcome
                    self.assertEqual(cpu.main(), outcome)
                h.result.assert_not_called()
        with cpu_fixture() as h:
            h.capsule_check.side_effect = [h.capsule, h.capsule, ValueError("postcheck changed")]
            with self.assertRaisesRegex(ValueError, "postcheck changed"):
                cpu.main()
            h.result.assert_not_called()

    def test_stock_result_requires_capsule_receipt_and_matching_contract_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, source, result = root / "contract.py", root / "moe.py", root / "result.json"
            contract.write_text("tested source")
            source.write_text("compiled source")
            evidence = dict(arm="stock", verdict="PASS", phase="complete", binding_runtime_rechecked=True,
                cuda_initialized=False, cache_key=["stock"], artifacts=["ptx"], resources=["cubin"],
                binding_runtime=runtime.expected_runtime_receipt(),
                contracts=dict(tests_run=1, failures=0, errors=0, skips=0, files={"contract.py": cpu.digest(contract)}),
                remap_compilation=[dict(label="remap", ptx_sha256="ptx", cubin_sha256="cubin")],
                mounted_sources={"/installed/moe.py": cpu.digest(source)}, sources={"/installed/moe.py": cpu.digest(source)})
            result.write_text(json.dumps(evidence))
            with patch.object(cpu, "CONTRACT_PATHS", ("contract.py",)), \
                    patch.object(cpu, "compile_cases", return_value=[{"label": "remap"}]), \
                    patch.object(cpu, "mounted_sources", return_value={"/installed/moe.py": source}):
                cpu.validate_result(root, result, "stock")
                for mutation in ("runtime", "arm", "contract", "mount", "failed_recheck", "unfinished", "missing_recheck", "error"):
                    altered = copy.deepcopy(evidence)
                    if mutation == "runtime":
                        altered.pop("binding_runtime")
                    elif mutation == "arm":
                        altered["arm"] = "local"
                    elif mutation == "contract":
                        altered["contracts"]["files"]["contract.py"] = "stale"
                    elif mutation == "mount":
                        altered["mounted_sources"]["/installed/moe.py"] = "stale"
                    elif mutation == "failed_recheck":
                        altered.update(verdict="FAIL", binding_runtime_rechecked=False,
                                       binding_runtime_recheck_error="changed after compilation")
                    elif mutation == "unfinished":
                        altered["phase"] = "compiled"
                    elif mutation == "missing_recheck":
                        altered.pop("binding_runtime_rechecked")
                    else:
                        altered["error"] = "late failure despite artifacts and contracts"
                    result.write_text(json.dumps(altered))
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        cpu.validate_result(root, result, "stock")


class GpuWrapperTests(unittest.TestCase):
    def test_every_remap_moe_and_sanitizer_cell_uses_cpu15_capsule_identity(self):
        with gpu_fixture() as h:
            self.assertEqual(gpu.main(), 0)
            self.assertEqual(len(h.commands), 10)
            self.assertIn("cpu15/local/result.json", str(gpu.CPU_EVIDENCE))
            self.assertLess(h.events.index("proof"), h.events.index("inventory"))
            self.assertGreater(h.proof.call_count, len(h.commands))
            for command in h.commands:
                self.assertIn(f"type=bind,source={h.capsule},target={runtime.CAPSULE_MOUNT},readonly", command)
                self.assertEqual(command[command.index("python3") + 1], "-B")
                self.assertEqual(command[command.index("--capsule-root") + 1], runtime.CAPSULE_MOUNT)
                self.assertEqual(command[command.index("--manifest-sha256") + 1], SHA)
                self.assertEqual(command[command.index("--compile-evidence") + 1], "/repo/" + str(gpu.CPU_EVIDENCE))
                self.assertIn("--memory=12g", command)
                self.assertIn("--cpus=4", command)
            self.assertEqual(sum("compute-sanitizer" in command for command in h.commands), 8)
            completed = json.loads((h.output / "completion.json").read_text())
            self.assertEqual(completed["binding_runtime"], runtime.expected_runtime_receipt())
            self.assertTrue(completed["restored_original"])
            self.assertTrue(all(cell["binding_runtime"] == completed["binding_runtime"] for cell in completed["cells"]))

    def test_stale_cpu_source_or_runtime_fails_before_inventory_and_pause(self):
        for stale in (ValueError("CPU source changed"), None):
            with self.subTest(stale=repr(stale)), gpu_fixture() as h:
                if stale:
                    h.proof.side_effect = stale
                else:
                    h.proof.side_effect = None
                    h.proof.return_value = {"binding_runtime": {"version": "13.3.1"}}
                self.assertEqual(gpu.main(), 1)
                h.inventory.assert_not_called()
                h.pause.assert_not_called()
                h.preflight.assert_not_called()
                self.assertEqual(h.commands, [])

    def test_changed_capsule_before_first_cell_restores_without_launching(self):
        with gpu_fixture() as h:
            def changed():
                if "pause" in h.events:
                    raise ValueError("capsule changed during admission")
            h.capsule_hook = changed
            self.assertEqual(gpu.main(), 1)
            self.assertEqual(h.commands, [])
            self.assertIn("restored", h.events)
            completed = json.loads((h.output / "completion.json").read_text())
            self.assertTrue(completed["restored_original"])
            self.assertIn("capsule changed", completed["error"])

    def test_cell_runtime_mismatch_stops_followups_and_restores(self):
        with gpu_fixture() as h:
            h.cell_runtime = copy.deepcopy(h.cell_runtime)
            h.cell_runtime["binding_identity"]["version"] = "13.3.1"
            self.assertEqual(gpu.main(), 1)
            self.assertEqual(len(h.commands), 1)
            self.assertIn("restored", h.events)
            completed = json.loads((h.output / "completion.json").read_text())
            self.assertTrue(completed["restored_original"])
            self.assertIn("runtime identity", completed["error"])

    def test_complete_looking_cell_with_failed_or_missing_recheck_cannot_pass(self):
        for status in ({"binding_runtime_rechecked": False}, {"phase": "compiled"},
                       {"error": "late failure"}, {"binding_runtime_recheck_error": "changed file"}):
            with self.subTest(status=status), gpu_fixture() as h:
                h.cell_status = status
                self.assertEqual(gpu.main(), 1)
                self.assertEqual(len(h.commands), 1)
                completed = json.loads((h.output / "completion.json").read_text())
                self.assertIn("valid component-only evidence", completed["error"])
                self.assertTrue(completed["restored_original"])

    def test_final_source_recheck_cannot_report_success_after_all_cells(self):
        with gpu_fixture() as h:
            def changed():
                if "restored" in h.events:
                    raise ValueError("CPU source changed after cells")
            h.proof_hook = changed
            self.assertEqual(gpu.main(), 1)
            self.assertEqual(len(h.commands), 10)
            completed = json.loads((h.output / "completion.json").read_text())
            self.assertIn("after cells", completed["error"])
            self.assertTrue(completed["restored_original"])

    def test_capsule_final_failure_does_not_hide_original_gpu_exit(self):
        with gpu_fixture() as h:
            h.docker_exit = 42
            def changed():
                if "restored" in h.events:
                    raise ValueError("capsule changed after GPU failure")
            h.capsule_hook = changed
            self.assertEqual(gpu.main(), 1)
            completed = json.loads((h.output / "completion.json").read_text())
            self.assertEqual(completed["cells"][0]["exit_code"], 42)
            self.assertIn("exit 42", completed["error"])
            self.assertIn("capsule changed", completed["capsule_postcheck_error"])
            self.assertTrue(completed["restored_original"])


if __name__ == "__main__":
    unittest.main()
