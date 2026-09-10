"""Exercise startup memory sizing without running any real reclaim commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "launchers/memfree-preflight.sh"
NODES = ["10.10.10.2", "10.10.10.1", "10.10.10.3", "10.10.10.4"]

# The real script is run with /bin/bash. Its local `bash -c` and remote `ssh`
# are intercepted here, so even a failing test cannot drop caches or kill a
# daemon. Events let us check the all-node join without timing assertions.
FAKE_COMMAND = r'''
import json, os, pathlib, shutil, subprocess, sys, time

root = pathlib.Path(os.environ["PREFLIGHT_TEST_ROOT"])
config = json.loads(os.environ["PREFLIGHT_TEST_CONFIG"])
command = pathlib.Path(sys.argv[0]).name

def event(kind, node=""):
    with (root / "events").open("a") as stream:
        stream.write(f"{kind} {node}\n")

if command in ("sync", "sudo", "sleep", "awk", "pkill", "killall"):
    node = os.environ.get("PREFLIGHT_TEST_NODE", "")
    if command in ("pkill", "killall"):
        event("FORBIDDEN", node)
        raise SystemExit(99)
    if command == "awk":
        args = sys.argv[1:]
        if args[-1:] == ["/proc/meminfo"]:
            event("meminfo", node)
            fixture = root / ("meminfo-" + node)
            fixture.write_text(config.get("meminfo", {}).get(
                node, "MemFree: 109051904 kB\nAnonPages: 2097152 kB\n"))
            args[-1] = str(fixture)
        raise SystemExit(subprocess.call(["/usr/bin/awk", *args]))
    if command == "sudo":
        assert sys.argv[1:] == ["-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], sys.argv
    if command == "sleep":
        assert sys.argv[1:] == ["2"], sys.argv
    event(command, node)
    raise SystemExit(config.get("payload_fail", {}).get(command, 0)
                     if node == NODES[0] else 0)

if command == "hostname":
    print("10.10.10.2" if config.get("local", True) else "192.0.2.1")
    raise SystemExit(0)
if command == "rm":
    # Only the script's private temp directory may be removed by this mock.
    target = pathlib.Path(sys.argv[-1]).resolve()
    assert target.parent == (root / "tmp").resolve(), target
    event("cleanup")
    shutil.rmtree(target)
    raise SystemExit(0)
if command == "bash":
    assert sys.argv[1] == "-c", sys.argv
    node = "10.10.10.2"
    payload = sys.argv[2]
else:
    assert command == "ssh", command
    node = next(arg.split("@", 1)[1] for arg in sys.argv[1:]
                if arg.startswith("choiceoh@"))
    payload = sys.stdin.read()
assert "sudo -n sh" in payload and "sleep 2" in payload, payload
assert payload.index("\nsync\n") < payload.index("sudo -n sh") < payload.index("sleep 2") < payload.index("awk ")
(root / ("payload-" + node)).write_text(payload)
event("start", node)
(root / ("started-" + node)).touch()
if config.get("barrier"):
    deadline = time.monotonic() + 3
    while not all((root / ("started-" + n)).exists() for n in config["nodes"]):
        if time.monotonic() >= deadline:
            event("barrier-timeout", node)
            raise SystemExit(97)
        time.sleep(0.005)
time.sleep(config.get("delays", {}).get(node, 0))
if config.get("execute_payload"):
    env = os.environ.copy()
    env["PREFLIGHT_TEST_NODE"] = node
    result = subprocess.run(["/bin/bash", "-c", payload], env=env)
    event("finish", node)
    raise SystemExit(result.returncode)
event("finish", node)
if node in config.get("raw_output", {}):
    sys.stdout.write(config["raw_output"][node])
elif node not in config.get("empty", []):
    print(*config["memory"][node])
raise SystemExit(config.get("exit_codes", {}).get(node, 0))
'''


class MemfreePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "tmp").mkdir()
        for command in ("hostname", "bash", "ssh", "rm", "sync", "sudo",
                        "sleep", "awk", "pkill", "killall"):
            path = self.bin / command
            path.write_text(f"#!{sys.executable}\nNODES = {NODES!r}\n" + FAKE_COMMAND)
            path.chmod(0o755)

    def run_preflight(self, *args: str, config: dict | None = None,
                      script: Path = SCRIPT,
                      terminate_after_started: bool = False,
                      ) -> subprocess.CompletedProcess:
        settings = {
            "nodes": NODES,
            "memory": {
                NODES[0]: [110.0, 2.0], NODES[1]: [104.0, 3.0],
                NODES[2]: [108.0, 2.5], NODES[3]: [112.0, 2.0],
            },
        }
        settings.update(config or {})
        (self.root / "events").unlink(missing_ok=True)
        for node in NODES:
            (self.root / ("started-" + node)).unlink(missing_ok=True)
        env = os.environ.copy()
        env.update({
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "TMPDIR": str(self.root / "tmp"),
            "PREFLIGHT_TEST_ROOT": str(self.root),
            "PREFLIGHT_TEST_CONFIG": json.dumps(settings),
            "BOOT_COST": "11", "TOTAL_GIB": "119.69",
        })
        command = ["/bin/bash", str(script), *args]
        if not terminate_after_started:
            return subprocess.run(command, env=env, text=True,
                                  capture_output=True, timeout=10)
        with subprocess.Popen(command, env=env, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            deadline = time.monotonic() + 3
            while not all((self.root / ("started-" + n)).exists()
                          for n in settings["nodes"]):
                if time.monotonic() >= deadline or process.poll() is not None:
                    process.kill()
                    self.fail("not all node probes started before TERM")
                time.sleep(0.005)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10)
            return subprocess.CompletedProcess(command, process.returncode,
                                               stdout, stderr)

    def events(self) -> list[str]:
        return (self.root / "events").read_text().splitlines()

    def assert_joined_before_cleanup(self) -> None:
        events = self.events()
        cleanup = events.index("cleanup ")
        for node in NODES:
            self.assertLess(events.index(f"finish {node}"), cleanup)
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_probes_overlap_and_report_in_node_order(self) -> None:
        result = self.run_preflight(config={
            "barrier": True,
            "delays": {NODES[0]: 0.12, NODES[1]: 0.08, NODES[2]: 0.04},
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        # min(110,104,108,112) - 11 boot - 10 margin, divided by 119.69.
        self.assertEqual(result.stdout, "0.69\n")
        events = self.events()
        starts = [events.index(f"start {node}") for node in NODES]
        finishes = [events.index(f"finish {node}") for node in NODES]
        self.assertLess(max(starts), min(finishes))
        report_positions = [result.stderr.index(node) for node in NODES]
        self.assertEqual(report_positions, sorted(report_positions))
        self.assert_joined_before_cleanup()

    def test_failed_node_waits_for_all_other_probes_before_refusing(self) -> None:
        result = self.run_preflight(config={
            "exit_codes": {NODES[0]: 23},
            "delays": {node: 0.15 for node in NODES[1:]},
        })
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(f"{NODES[0]}  UNREACHABLE", result.stderr)
        self.assert_joined_before_cleanup()

    def test_empty_remote_result_refuses_after_all_nodes_finish(self) -> None:
        result = self.run_preflight(config={"empty": [NODES[1]]})
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(f"{NODES[1]}  UNREACHABLE", result.stderr)
        self.assert_joined_before_cleanup()

    def test_term_waits_for_active_probes_before_returning_signal_status(self) -> None:
        result = self.run_preflight(terminate_after_started=True, config={
            "barrier": True, "delays": {node: 0.5 for node in NODES},
        })
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assert_joined_before_cleanup()

    def test_export_with_explicit_margin_and_remote_head(self) -> None:
        result = self.run_preflight("--export", "12.5", *NODES,
                                    config={"local": False})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "export GMU_SAFE=0.67\n")
        self.assertIn("-margin=12.5", result.stderr)
        self.assert_joined_before_cleanup()

    def test_address_first_keeps_first_node_and_refuses_low_memory(self) -> None:
        memory = {node: [110, 2] for node in NODES}
        memory[NODES[0]] = [30, 2]
        result = self.run_preflight(*NODES, config={"memory": memory})
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(f"start {NODES[0]}", self.events())
        self.assertIn("refusing to size this low", result.stderr)
        self.assert_joined_before_cleanup()

    def test_emitted_node_script_reclaims_without_terminating_processes(self) -> None:
        result = self.run_preflight(config={"execute_payload": True})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0.69\n")
        events = self.events()
        self.assertNotIn("FORBIDDEN", "\n".join(events))
        for node in NODES:
            payload = (self.root / ("payload-" + node)).read_text()
            self.assertNotRegex(payload, r"\b(?:pkill|killall|kill)\b")
            ordered = [events.index(f"{stage} {node}")
                       for stage in ("sync", "sudo", "sleep", "meminfo", "finish")]
            self.assertEqual(ordered, sorted(ordered))
        self.assert_joined_before_cleanup()

    def test_emitted_script_reclaim_failure_prevents_memory_measurement(self) -> None:
        for stage in ("sync", "sudo", "sleep"):
            with self.subTest(stage=stage):
                result = self.run_preflight(config={
                    "execute_payload": True, "payload_fail": {stage: 23}})
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn(f"meminfo {NODES[0]}", self.events())
                self.assertIn("probe/reclaim failed", result.stderr)
                self.assert_joined_before_cleanup()

    def test_emitted_script_rejects_missing_malformed_or_duplicate_meminfo(self) -> None:
        malformed = (
            "", "MemFree: 109051904 kB\n", "AnonPages: 2097152 kB\n",
            "MemFree: nope kB\nAnonPages: 2097152 kB\n",
            "MemFree: 109051904 kB\nAnonPages: NaN kB\n",
            "MemFree: -1 kB\nAnonPages: 2097152 kB\n",
            "MemFree: 109051904 bytes\nAnonPages: 2097152 kB\n",
            "MemFree: 109051904 kB extra\nAnonPages: 2097152 kB\n",
            "MemFree: 109051904 kB\nMemFree: 109051904 kB\nAnonPages: 2097152 kB\n",
            "MemFree: 109051904 kB\nAnonPages: 2097152 kB\nAnonPages: 2097152 kB\n",
        )
        for text in malformed:
            with self.subTest(meminfo=text):
                result = self.run_preflight(config={
                    "execute_payload": True, "meminfo": {NODES[1]: text}})
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_rejects_malformed_remote_output_even_after_success_exit(self) -> None:
        for text in ("104\n", "104 nope\n", "NaN 2\n", "104 -1\n",
                     "104 2 extra\n", "104 2\n105 2\n", "1e999 2\n"):
            with self.subTest(output=text):
                result = self.run_preflight(config={"raw_output": {NODES[2]: text}})
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("INVALID MEMORY", result.stderr)

    def test_gmu_is_floored_without_rounding_free_memory_up(self) -> None:
        # 104.3 - 21 divided by 119.69 is about .6959646: nearest .70
        # would exceed the measured budget. Preserve the full input budget.
        memory = {node: [104.3, 0] for node in NODES}
        result = self.run_preflight(config={"memory": memory})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0.69\n")
        self.assertLessEqual(float(result.stdout), (104.3 - 11 - 10) / 119.69)
        # Exercise real emitted awk with a value that the old %.1f readout
        # rounded across the next hundredth-of-GMU boundary.
        free_kib = int(104.77 * 1048576)
        meminfo = {node: f"MemFree: {free_kib} kB\nAnonPages: 0 kB\n"
                   for node in NODES}
        result = self.run_preflight(config={"execute_payload": True, "meminfo": meminfo})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0.69\n")
        self.assertLessEqual(float(result.stdout), (free_kib / 1048576 - 21) / 119.69)

    def test_truncated_gmu_below_floor_is_rejected(self) -> None:
        # Nearest rounding would falsely admit .40 from an upper bound < .40.
        memory = {node: [68.4, 0] for node in NODES}
        result = self.run_preflight(config={"memory": memory})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("GMU 0.39", result.stderr)


if __name__ == "__main__":
    unittest.main()
