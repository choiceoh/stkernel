#!/usr/bin/env python3
"""Watch UMA headroom while an owned onepass client runs.

On low or unreadable memory, terminate only the client launched here. Closing
its HTTP connection cancels its request; the fleet arm handles failed-leg
recovery. This never signals serving workers or changes the OOM policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def parse_memory(text):
    fields = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ("MemTotal:", "MemAvailable:"):
            if len(parts) != 3 or parts[2] != "kB":
                raise ValueError("invalid memory units")
            fields[parts[0][:-1]] = int(parts[1])
    total, available = fields.get("MemTotal"), fields.get("MemAvailable")
    if total is None or available is None or not 0 <= available <= total or total <= 0:
        raise ValueError("valid MemTotal and MemAvailable are required")
    return {"total_kib": total, "available_kib": available}


def sample_hosts(hosts):
    def read(host):
        try:
            if host == "local":
                raw = Path("/proc/meminfo").read_text()
            else:
                raw = subprocess.check_output(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=2",
                     host, "cat /proc/meminfo"], text=True,
                    stderr=subprocess.PIPE, timeout=3,
                )
            return host, parse_memory(raw)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return host, {"error": str(exc)}
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        return dict(pool.map(read, hosts))


def stop_client(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run_guarded(command, sample, report, minimum_kib, interval=1.0, spawn=subprocess.Popen):
    child = None
    start = time.monotonic()
    try:
        while True:
            if child is not None and child.poll() is not None:
                return child.returncode
            nodes = sample()
            issues = []
            if not nodes:
                issues.append("no host memory samples")
            for host, state in nodes.items():
                if state.get("error"):
                    issues.append(f"{host}: memory unavailable: {state['error']}")
                elif state["available_kib"] < minimum_kib:
                    issues.append(f"{host}: {state['available_kib']} KiB available < {minimum_kib}")
            report.write(json.dumps({"t": time.time(), "elapsed_s": time.monotonic() - start,
                "minimum_kib": minimum_kib, "nodes": nodes, "issues": issues}) + "\n")
            report.flush()
            if issues:
                print("REFUSED onepass memory: " + "; ".join(issues), file=sys.stderr, flush=True)
                return 3
            if child is None:
                child = spawn(command)
            try:
                return child.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if child is not None:
            stop_client(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default="local,choiceoh@10.10.10.1,choiceoh@10.10.10.3,choiceoh@10.10.10.4")
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-gib", type=int, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    hosts = args.hosts.split(",")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.minimum_gib < 10 or not all(hosts) or len(set(hosts)) != len(hosts):
        parser.error("command, distinct hosts and at least 10 GiB memory reserve are required")
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as report:
        return run_guarded(command, lambda: sample_hosts(hosts), report,
                           args.minimum_gib * 1024**2)


if __name__ == "__main__":
    raise SystemExit(main())
