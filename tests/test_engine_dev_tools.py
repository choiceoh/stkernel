"""Agent tool discovery and invocation contracts, without GPUs, installation or network."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dev  # noqa: E402 -- exercise the checkout's scripts without installing a package
from dev_catalog import BY_ID, TOOLS, Tool  # noqa: E402


class CatalogTests(unittest.TestCase):
    def test_every_adapter_and_document_still_exists(self):
        self.assertEqual(dev.registry_errors(), [])
        for tool in TOOLS:
            with self.subTest(tool=tool.id):
                spec = dev.describe(tool.id)
                self.assertEqual(spec["input_schema"]["properties"]["arguments"]["items"], {"type": "string"})
                self.assertTrue(spec["examples"])

    def test_deleted_target_is_a_registry_failure(self):
        missing = Tool("deleted", "no longer exists", ("{python}", "tools/not-here.py"))
        with mock.patch.object(dev, "TOOLS", (*TOOLS, missing)), mock.patch.dict(dev.BY_ID, {missing.id: missing}):
            self.assertTrue(any("missing target" in error for error in dev.registry_errors()))

    def test_new_script_is_discoverable_without_registration_or_execution(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(dev, "ROOT", Path(tmp)):
            path = Path(tmp) / "tools/new_tool.py"
            path.parent.mkdir()
            path.write_text('"""new capability not yet registered"""\nraise RuntimeError("must not import")\n'
                            'if __name__ == "__main__": pass\n')
            rows = dev.discover()
            self.assertEqual([row["id"] for row in rows], ["tools/new_tool.py"])
            self.assertFalse(rows[0]["managed"])
            self.assertIn("new capability", dev.describe(rows[0]["id"])["source_help"])

    def test_unknown_or_outside_paths_cannot_be_described(self):
        with self.assertRaises(dev.ToolError):
            dev.describe("/etc/passwd")

    def test_cli_json_discovery_works_from_another_directory(self):
        result = subprocess.run([sys.executable, str(ROOT / "dev")], cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["repo"], str(ROOT))
        self.assertIn("fleet.submit", [row["id"] for row in payload["tools"]])

    def test_invalid_arguments_still_return_one_json_object(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = dev.main(["search", "--limit", "0"])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "blocked")


class EnvironmentTests(unittest.TestCase):
    def test_explicit_missing_python_does_not_fall_back(self):
        with mock.patch.dict(os.environ, {"ST_DEV_PYTHON": "/no/such/python"}):
            with self.assertRaises(dev.ToolError):
                dev.python_runtime()

    def test_checkout_python_is_selected_without_shell_startup(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(dev, "ROOT", Path(tmp)), \
                mock.patch.dict(os.environ, {}, clear=True):
            candidate = Path(tmp) / ".venv/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.symlink_to(sys.executable)
            self.assertEqual(dev.python_runtime(), (str(candidate), "checkout .venv"))
            self.assertEqual(dev.child_environment()["PATH"].split(os.pathsep)[0], str(candidate.parent))

    def test_missing_packages_block_before_test_execution(self):
        inventory = {"packages": [{"name": "numpy", "installed": None}]}
        with mock.patch.object(dev, "package_inventory", return_value=inventory), \
                mock.patch.object(dev.subprocess, "Popen") as spawn:
            with self.assertRaises(dev.ToolError):
                dev.execute(BY_ID["check"], [])
            spawn.assert_not_called()

    def test_dry_run_setup_never_installs_or_imports(self):
        with mock.patch.object(dev.platform, "system", return_value="Linux"), \
                mock.patch.object(dev.subprocess, "Popen") as spawn:
            result, rc = dev.execute(BY_ID["env.setup"], [], dry_run=True)
            self.assertEqual(rc, 0)
            self.assertEqual(result["status"], "planned")
            self.assertIn("tools/devenv/node.sh", " ".join(result["argv"]))
            spawn.assert_not_called()

    def test_gpu_work_cannot_use_the_cpu_adapter(self):
        for identifier in ("check", "regress"):
            with self.assertRaises(dev.ToolError):
                dev.command_for(BY_ID[identifier], ["--gpu"])


class ExecutionTests(unittest.TestCase):
    @contextlib.contextmanager
    def script(self, body, identifier="fixture"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "command.py"
            path.write_text(body)
            yield Tool(identifier, "test fixture", (sys.executable, str(path))), Path(tmp)

    def test_arguments_are_literal_and_exit_code_is_preserved(self):
        args = ["two words", "$(touch should-not-exist)", "semi;colon", 'a"b', "--", "--flag"]
        with self.script("import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(7)") as (tool, _):
            result, rc = dev.execute(tool, args)
        self.assertEqual(rc, 7)
        self.assertEqual(result["process_exit_code"], 7)
        self.assertEqual(result["data"], args)
        self.assertEqual(result["status"], "failed")

    def test_output_is_bounded_and_truncation_is_explicit(self):
        with self.script("print('x' * 100000)") as (tool, _):
            result, rc = dev.execute(tool, [], limit=1024)
        self.assertEqual(rc, 0)
        self.assertLessEqual(len(result["stdout"]), 1024)
        self.assertTrue(result["truncated"]["stdout"])
        self.assertNotIn("data", result)

    def test_timeout_terminates_child_processes_too(self):
        body = """import subprocess, sys, time
subprocess.Popen([sys.executable, '-c', 'import pathlib,time,sys; time.sleep(.5); pathlib.Path(sys.argv[1]).touch()', sys.argv[1]])
time.sleep(30)
"""
        with self.script(body) as (tool, tmp):
            marker = tmp / "child-survived"
            result, rc = dev.execute(tool, [str(marker)], timeout=0.2)
            self.assertEqual(rc, 124)
            self.assertEqual(result["status"], "timeout")
            time.sleep(0.6)
            self.assertFalse(marker.exists())

    def test_zero_exit_is_not_a_complete_test_verdict(self):
        for summary in ("4 files, 5 tests: 3 ok, 0 failed, 1 cannot run",
                        "4 files, 5 tests: 3 ok, 0 failed, 0 cannot run, 1 skipped",
                        "0 files, 0 tests: 0 ok, 0 failed, 0 cannot run"):
            with self.subTest(summary=summary), self.script("print(" + repr(summary) + ")", "check") as (tool, _):
                result, rc = dev.execute(tool, [])
                self.assertEqual(rc, 3)
                self.assertEqual(result["process_exit_code"], 0)
                self.assertEqual(result["validation"], "incomplete")

    def test_a_complete_cpu_verdict_remains_distinct_from_gpu_evidence(self):
        with self.script("print('4 files, 9 tests: 4 ok, 0 failed, 0 cannot run')", "check") as (tool, _):
            result, rc = dev.execute(tool, [])
        self.assertEqual(rc, 0)
        self.assertEqual(result["validation"], "passed")
        self.assertEqual(result["test_counts"]["tests"], 9)

    def test_remote_arguments_cannot_become_shell_commands(self):
        args = ["agent", "/controller/with space/spec.json", "$(touch not-a-command)", "a'b"]
        with mock.patch.object(dev.socket, "gethostname", return_value="mac"), \
                mock.patch.dict(os.environ, {"FLEET_CONTROLLER": "srv2"}):
            command, _ = dev.command_for(BY_ID["fleet.submit"], args)
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(shlex.split(command[-1]), ["cd", "$HOME/stkernel", "&&", "exec", "env",
                                                  "REPO=$HOME/stkernel", "bash", "$HOME/glm53-logs/fleet.sh", "submit", *args])
        self.assertNotIn("FLEET_ALLOW_LOCAL", " ".join(command))

    def test_remote_timeout_does_not_claim_to_cancel_a_ticket(self):
        with self.script("import time; time.sleep(30)") as (tool, _), \
                mock.patch.object(dev, "command_for", return_value=([*tool.command], os.environ.copy())):
            result, rc = dev.execute(BY_ID["fleet.submit"], [], timeout=0.1)
        self.assertEqual(rc, 124)
        self.assertIn("may still exist", result["note"])

    def test_remote_source_override_is_quoted_and_absolute(self):
        with mock.patch.object(dev.socket, "gethostname", return_value="mac"), \
                mock.patch.dict(os.environ, {"FLEET_CONTROLLER": "srv2", "ST_DEV_FLEET_REPO": "/repo with space;literal"}):
            command, _ = dev.command_for(BY_ID["fleet.status"], [])
            tokens = shlex.split(command[-1])
            self.assertEqual(tokens[1], "/repo with space;literal")
            self.assertIn("REPO=/repo with space;literal", tokens)
        with mock.patch.dict(os.environ, {"ST_DEV_FLEET_REPO": "relative"}):
            with self.assertRaises(dev.ToolError):
                dev.command_for(BY_ID["fleet.status"], [])

    def test_queue_helper_failure_is_not_a_successful_observation(self):
        with self.script("print(\"python3: can't open file '/missing/helper.py'\")") as (tool, _), \
                mock.patch.object(dev, "command_for", return_value=([*tool.command], os.environ.copy())), \
                mock.patch.object(dev.socket, "gethostname", return_value="mac"):
            result, rc = dev.execute(BY_ID["fleet.status"], [])
        self.assertEqual(rc, 3)
        self.assertEqual(result["process_exit_code"], 0)
        self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
