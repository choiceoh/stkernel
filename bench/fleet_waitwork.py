#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""큐에서 기다리는 동안 돌아가는 CPU 작업 — 대기가 작업 시간이 되게.

운영자: "gpu 없이 할수 있는 작업 같으면 병렬로". 네 박스를 기다리는 티켓은 컨트롤러에
앉아 있고 컨트롤러의 CPU 는 한가하다. `fleet.sh wait` 가 이 모듈의 `start` 를 부르면
대기 작업이 자동으로 시작된다 — 기본은 **step_sim 재검증**: onepass 원장(bracket-onepass.jsonl)의
최근 기록들로 비용 상수를 다시 폴딩해 예측 표를 다시 낸다. GO 가 와도 끝나지 않았으면
두고 간다(컨트롤러 CPU, 플릿와 무관); 대기가 포기하는 끝남(TIMEOUT·실패)에서는 `stop` 이
끊는다. `FLEET_WAIT_WORK` 로 다른 명령을 줄 수 있고 `off` 면 끈다.

    fleet_waitwork.py start --session s --repo R --jsonl J --logd L --fleet-dir D
    fleet_waitwork.py note  D s    # GO 에 한 줄: 계속 돈다 / 끝났다(+요약) — 산출물 경로 포함
    fleet_waitwork.py stop  D s    # 포기하는 끝남: 돌고 있으면 끊고 표식을 지운다

산출물은 <logd>/wait-sim/wait-<session>-<ts>.txt. 표식은 <fleet-dir>/.waitwork.<session>
(pid·산출물·명령) — note/stop 이 읽고 지운다.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def plan(repo: str, jsonl: str, env=None) -> dict:
    """무엇을 돌릴지 — 순수 함수(스폰 없음). 테스트가 여기를 직접 본다.

    off    FLEET_WAIT_WORK=off/0: 대기 작업을 끈다
    custom FLEET_WAIT_WORK='<cmd>': 그 명령(bash -c)
    sim    기본: step_sim --against <jsonl> --last 3(최근 onepass 기록 재검증)
    none   재검증할 기록이 아직 없다(원장 비었음) — 시작도 안 한다
    """
    env = dict(env if env is not None else os.environ)
    override = env.get("FLEET_WAIT_WORK", "").strip()
    if override.lower() in ("off", "0"):
        return {"kind": "off", "argv": None, "why": "FLEET_WAIT_WORK=off"}
    if override:
        return {"kind": "custom", "argv": ["bash", "-c", override], "why": f"FLEET_WAIT_WORK: {override}"}
    try:
        has_records = any(l.strip() for l in open(jsonl, encoding="utf-8"))
    except OSError:
        has_records = False
    if not has_records:
        return {"kind": "none", "argv": None, "why": "재검증할 onepass 기록이 아직 없다"}
    return {"kind": "sim", "argv": [sys.executable, str(Path(repo) / "bench" / "step_sim.py"),
                                    "--against", jsonl, "--last", "3", "--no-calib"],
            "why": "step_sim 재검증(최근 onepass 기록 3개, CPU)"}


def marker_path(directory, session: str) -> Path:
    return Path(directory) / f".waitwork.{session}"


def read_marker(directory, session: str) -> dict:
    try:
        return json.loads(marker_path(directory, session).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def start(session: str, repo: str, jsonl: str, logd: str, directory: str, env=None) -> int:
    what = plan(repo, jsonl, env)
    if what["argv"] is None:
        print(f"wait work: 없다 — {what['why']}")
        return 0
    out_dir = Path(logd) / "wait-sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"wait-{session}-{stamp}.txt"
    child = subprocess.Popen(what["argv"], stdout=open(out, "w"), stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=repo,
                             env=dict(env if env is not None else os.environ))
    marker_path(directory, session).write_text(
        json.dumps({"pid": child.pid, "out": str(out), "kind": what["kind"], "why": what["why"],
                    "started_at": time.time()}), encoding="utf-8")
    print(f"wait work: {what['why']} → {out}")
    return 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _summary(out: str) -> str:
    """산출물의 마지막 요약 줄(예측 행 최대 잔여) — 없으면 줄 수만."""
    try:
        lines = [l for l in Path(out).read_text(encoding="utf-8").splitlines() if l.strip()]
        for l in reversed(lines):
            if l.startswith("-- 예측 행"):
                return l
        return f"{len(lines)}줄" if lines else "빈 출력"
    except OSError:
        return "출력 없음"


def note(directory, session: str) -> int:
    """GO 에 호출: 한 줄로 말하고 표식을 지운다(산출물 파일은 남는다)."""
    m = read_marker(directory, session)
    marker_path(directory, session).unlink(missing_ok=True)
    if not m:
        print("wait work: 없었다")
        return 0
    if _alive(int(m["pid"])):
        print(f"wait work: 계속 돈다(컨트롤러 CPU, 플릿와 무관) → {m['out']}")
    else:
        print(f"wait work: 끝났다 — {_summary(m['out'])} → {m['out']}")
    return 0


def stop(directory, session: str) -> int:
    """포기하는 끝남(TIMEOUT·실패)에 호출: 돌고 있으면 끊는다."""
    m = read_marker(directory, session)
    marker_path(directory, session).unlink(missing_ok=True)
    if m and _alive(int(m["pid"])):
        try:
            os.kill(int(m["pid"]), signal.SIGTERM)
        except OSError:
            pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="큐 대기 중 자동 CPU 작업 (step_sim 재검증 기본)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start", help="대기 작업 시작(계획이 없으면 한 줄만 찍는다)")
    st.add_argument("--session", required=True)
    st.add_argument("--repo", required=True)
    st.add_argument("--jsonl", required=True)
    st.add_argument("--logd", required=True)
    st.add_argument("--fleet-dir", required=True, dest="fleet_dir")
    nt = sub.add_parser("note", help="GO: 상태 한 줄")
    nt.add_argument("directory")
    nt.add_argument("session")
    sp = sub.add_parser("stop", help="포기 끝남: 끊는다")
    sp.add_argument("directory")
    sp.add_argument("session")
    args = ap.parse_args()
    if args.cmd == "start":
        return start(args.session, args.repo, args.jsonl, args.logd, args.fleet_dir)
    if args.cmd == "note":
        return note(args.directory, args.session)
    return stop(args.directory, args.session)


if __name__ == "__main__":
    sys.exit(main())
