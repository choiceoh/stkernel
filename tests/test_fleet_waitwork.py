"""큐 대기 중 자동 CPU 작업(fleet_waitwork)의 계획·생명주기 — GPU 도 플릿도 없다.

`fleet.sh wait` 가 부르는 경계: plan 은 순수 함수, start/note/stop 은 임시 디렉터리에서
실제 프로세스로 왕복한다. 기본 작업(step_sim 최근 기록 재검증)의 --last 선택은
test_step_tools 가 담당한다.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import fleet_waitwork as ww                                  # noqa: E402


class PlanTests(unittest.TestCase):
    def test_off_and_custom_come_from_env(self):
        self.assertEqual(ww.plan("R", "J", {"FLEET_WAIT_WORK": "off"})["kind"], "off")
        self.assertEqual(ww.plan("R", "J", {"FLEET_WAIT_WORK": "0"})["kind"], "off")
        custom = ww.plan("R", "J", {"FLEET_WAIT_WORK": "make check"})
        self.assertEqual(custom["kind"], "custom")
        self.assertEqual(custom["argv"][:2], ["bash", "-c"])
        self.assertIn("make check", custom["why"])

    def test_default_is_sim_only_when_the_ledger_has_records(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "bracket-onepass.jsonl"
            empty.touch()
            self.assertEqual(ww.plan("R", str(empty), {})["kind"], "none")
            self.assertEqual(ww.plan("R", str(Path(d) / "nope.jsonl"), {})["kind"], "none")
            empty.write_text('{"name": "A"}\n', encoding="utf-8")
            sim = ww.plan("R", str(empty), {})
            self.assertEqual(sim["kind"], "sim")
            self.assertIn("step_sim.py", sim["argv"][1])
            self.assertEqual(sim["argv"][2:], ["--against", str(empty), "--last", "3", "--no-calib"])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fleet = self.root / "fleet"
        self.fleet.mkdir()
        self.logd = self.root / "logs"

    def run_cli(self, *args, extra_env=None):
        env = dict(os.environ, **(extra_env or {}))
        return subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]
                                                   / "bench" / "fleet_waitwork.py"), *args],
                              capture_output=True, text=True, timeout=30, env=env)

    def test_custom_command_roundtrip_finishes_and_notes_summary(self):
        # 시작도 CLI 경계로: fleet.sh 이 부르는 그 모양(자식은 start 프로세스가 떠나보내고
        # launchd 로 입양된다 — in-process 로 돌리면 좀비가 _alive 에 걸린다)
        outs = self.run_cli("start", "--session", "s1", "--repo", str(self.root),
                            "--jsonl", str(self.root / "none.jsonl"),
                            "--logd", str(self.logd), "--fleet-dir", str(self.fleet),
                            extra_env={"FLEET_WAIT_WORK": "printf 'line\\n-- 예측 행 최대 잔여: TPOT +5.0%\\n'"})
        self.assertEqual(outs.returncode, 0)
        self.assertIn("wait work:", outs.stdout)
        marker = ww.read_marker(self.fleet, "s1")
        self.assertEqual(marker["kind"], "custom")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and ww._alive(marker["pid"]):
            time.sleep(0.05)
        outs = self.run_cli("note", str(self.fleet), "s1")
        self.assertEqual(outs.returncode, 0)
        self.assertIn("끝났다", outs.stdout)
        self.assertIn("예측 행 최대 잔여", outs.stdout)          # 요약 줄이 한 줄로 요약된다
        self.assertIn("wait-sim", outs.stdout)
        self.assertFalse(ww.marker_path(self.fleet, "s1").exists())
        self.assertTrue(Path(marker["out"]).exists())           # 산출물은 남는다
        self.assertIn("예측 행 최대 잔여", Path(marker["out"]).read_text())

    def test_stop_kills_still_running_work_and_clears_marker(self):
        outs = self.run_cli("start", "--session", "s2", "--repo", str(self.root),
                            "--jsonl", str(self.root / "none.jsonl"),
                            "--logd", str(self.logd), "--fleet-dir", str(self.fleet),
                            extra_env={"FLEET_WAIT_WORK": "sleep 30"})
        self.assertEqual(outs.returncode, 0)
        marker = ww.read_marker(self.fleet, "s2")
        self.assertTrue(ww._alive(marker["pid"]))
        stop = self.run_cli("stop", str(self.fleet), "s2")
        self.assertEqual(stop.returncode, 0)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and ww._alive(marker["pid"]):
            time.sleep(0.05)
        self.assertFalse(ww._alive(marker["pid"]))
        self.assertFalse(ww.marker_path(self.fleet, "s2").exists())

    def test_note_without_work_says_so(self):
        outs = self.run_cli("note", str(self.fleet), "ghost")
        self.assertEqual(outs.returncode, 0)
        self.assertIn("없었다", outs.stdout)

    def test_start_without_records_prints_one_line_and_no_marker(self):
        empty = self.root / "empty.jsonl"
        empty.touch()
        outs = self.run_cli("start", "--session", "s3", "--repo", str(self.root),
                            "--jsonl", str(empty), "--logd", str(self.logd),
                            "--fleet-dir", str(self.fleet))
        self.assertEqual(outs.returncode, 0)
        self.assertIn("없다", outs.stdout)
        self.assertFalse(ww.marker_path(self.fleet, "s3").exists())


if __name__ == "__main__":
    unittest.main()
