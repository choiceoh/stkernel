"""Resume this frozen session once, after other sessions and queued work finish.

This is a waiter for the launcher's documented session window, not a fleet
queue entry. It never requests handover from a session or passes queued work.
All boot/stop/lease actions remain in collect_window.sh. A failed run is retained
and is not retried automatically; no calibration pack is promoted by this tool.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def available(lease_path, queue_path):
    try:
        lease = json.loads(lease_path.read_text()) if lease_path.exists() else {}
        queue = queue_path.read_text() if queue_path.exists() else ""
    except (OSError, ValueError):
        return False, "unreadable fleet state"
    if not isinstance(lease, dict):
        return False, "unreadable fleet state"
    if queue.strip():
        return False, "canonical queue has waiting work"
    if lease.get("yield_to"):
        return False, "another requester awaits handover"
    if lease and lease.get("kind") != "production":
        return False, "held by " + str(lease.get("owner", "unknown"))
    return True, "free or production quiet handover available"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-hours", type=float, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "waiter.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = Path("/home/choiceoh/glm53-logs/st-fleet.lock")
    queue = lease.parent / "fleet/queue"
    child, observer = None, None
    started = time.time()
    def report(state, **details):
        value = dict(state=state, t=time.time(), started=started, pid=os.getpid(),
                     source_sha=args.sha, source_tree=str(args.tree), **details)
        tmp = args.out / "waiter-status.tmp"
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(args.out / "waiter-status.json")
    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()  # Its EXIT trap stops only its own serving window.
            child.wait()
        report("cancelled", signal=signum)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous = None
    while time.time() - started < args.deadline_hours * 3600:
        ready, reason = available(lease, queue)
        report("waiting", reason=reason)
        if reason != previous:
            print(reason, flush=True)
            previous = reason
        if ready:
            break
        time.sleep(30)
    else:
        report("expired", reason="no available session window before deadline")
        return 3
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.tree, text=True).strip()
    if actual != args.sha:
        report("refused", reason="frozen checkout changed", actual_sha=actual)
        return 4
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.tree, text=True).strip():
        report("refused", reason="frozen checkout is dirty")
        return 4
    driver = args.tree / "measurements/qwen38_gptq_20260919/collect_window.sh"
    with (args.out / "consumer-window.log").open("x") as log:
        child = subprocess.Popen(["bash", str(driver), "serve"], cwd=args.tree,
                                 stdout=log, stderr=subprocess.STDOUT)
        report("running", child_pid=child.pid)
        while child.poll() is None:
            try:
                owned = json.loads(lease.read_text()).get("owner") == "session/q38gptq-0919"
            except (OSError, ValueError):
                owned = False
            if owned and observer is None:
                observer = subprocess.Popen(["bash", str(driver.with_name("observe_fleet.sh")),
                                             str(args.tree), str(lease.parent / "qwen38-gptq-20260919/serving/occupancy")],
                                            stdout=log, stderr=subprocess.STDOUT)
            time.sleep(5)
    if observer is not None:
        try:
            observer.wait(timeout=30)
        except subprocess.TimeoutExpired:
            observer.terminate()  # Read-only observer only; never a serving process.
            observer.wait()
    report("finished" if child.returncode == 0 else "failed", returncode=child.returncode,
           note="See serving/onepass.jsonl and each run's rc; driver success does not imply quality passed.")
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
