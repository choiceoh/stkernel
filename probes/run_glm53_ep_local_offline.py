#!/usr/bin/env python3
"""Normal fleet boot hold: isolated EP correctness/sanitizers and exact restore."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import glm53_probe_lifecycle as lifecycle
import glm53_ep_sanitizer as sanitizer_support
from glm53_ep_local_evidence import validate_compile_evidence

CASES = ("balanced4096", "balanced6912", "balanced8192", "concentrated6912",
         "remote4096", "duplicate4096", "zeros4097", "balanced16384")
CPU_EVIDENCE = Path("measurements/glm53_ep_local_20260908/cpu9/local/result.json")


def resources(require_memory):
    results = {}
    for node in lifecycle.NODES:
        results[node] = lifecycle.remote(node, """import json,shutil,pathlib
available=next(int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
print(json.dumps(dict(available_kib=available,disk_free_bytes=shutil.disk_usage('/home/choiceoh').free)))
""")
    if any(r["disk_free_bytes"] < 128*2**30 for r in results.values()):
        raise RuntimeError("128 GiB per-node disk reserve required")
    if require_memory and any(r["available_kib"] < 16*1024**2 for r in results.values()):
        raise RuntimeError("16 GiB per-node offline memory reserve required: "+json.dumps(results))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        ap.error("full frozen revision required")
    root = Path(__file__).resolve().parents[1]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(started=time.time(), source_revision=args.revision,
                  exit_code=1, performance_acceptance=False, cells=[])
    owned = set()
    sanitizer_receipt = None

    def save(name, data):
        (args.out/name).write_text(json.dumps(data, indent=2)+"\n")

    def interrupted(signum, frame):
        raise InterruptedError("termination requested")

    signal.signal(signal.SIGTERM, interrupted)

    def cleanup():
        lifecycle.check_holder()
        for name in sorted(owned):
            inspected = subprocess.run(["docker", "inspect", name], text=True, capture_output=True)
            if inspected.returncode:
                # Distinguish an absent owned container from Docker failure.
                names = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
                if name in names:
                    raise RuntimeError("owned container inspect failed: "+name)
                continue
            c = json.loads(inspected.stdout)[0]
            if (c["Image"] != lifecycle.IMAGE
                    or c["Config"].get("Labels", {}).get("glm53.ep-local.session") != os.environ["FLEET_SESSION"]):
                raise RuntimeError("owned probe container identity changed")
            subprocess.run(["docker", "rm", "-f", c["Id"]], check=True, timeout=45,
                           stdout=subprocess.DEVNULL)

    def cell(case, sanitizer=None):
        lifecycle.check_holder()
        lifecycle.pinned(str(root), args.revision)
        label = (sanitizer+"-" if sanitizer else "")+case
        save("resources-"+label+".json", resources(True))
        states = lifecycle.snapshot()
        if any(state is not None and state["running"] for state in states.values()):
            raise RuntimeError("GLM serving must be stopped on every node before a GPU probe")
        name = "ep-local-"+os.environ["FLEET_SESSION"]+"-"+label
        names = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
        if name in names:
            raise RuntimeError("probe container name already exists")
        command = ["docker", "run", "--rm", "--name", name,
                   "--label", "glm53.ep-local.session="+os.environ["FLEET_SESSION"],
                   "--gpus", "all", "--network=none", "--memory=12g", "--memory-swap=12g",
                   "--cpus=4", "--pids-limit=256", "-e", "MAX_JOBS=1",
                   "--entrypoint=/usr/bin/env", "-v", f"{root}:/repo:ro",
                   "-v", f"{args.out}:/evidence"]
        for line in (root/"build/glm53/manifest.tsv").read_text().splitlines():
            filename, target, *_ = line.split("\t")
            if "/flashinfer/" in target or filename == "flashinfer_b12x_moe.py":
                command += ["-v", f"{root/'build/glm53'/filename}:{target}:ro"]
        if sanitizer:
            command += sanitizer_support.mount_args(sanitizer_receipt)
        command += [lifecycle.IMAGE]
        if sanitizer:
            command += sanitizer_support.command(sanitizer)
        probe = ("glm53_ep_route_remap_check.py" if case == "remap" else "glm53_ep_local_check.py")
        command += ["python3", "/repo/probes/"+probe]
        if case != "remap":
            command += ["--case", case]
        command += ["--compile-evidence", "/repo/"+str(CPU_EVIDENCE),
                    "--output", "/evidence/"+label+".json"]
        if sanitizer and case != "remap":
            command += ["--sanitize"]
        entry = dict(case=case, sanitizer=sanitizer, started=time.time(), command=command)
        result["cells"].append(entry)
        owned.add(name)
        print("START "+label, flush=True)
        try:
            with (args.out/(label+".log")).open("x") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            entry["exit_code"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"GPU cell {label} failed with exit {process.returncode}")
            if sanitizer:
                entry["sanitizer_summary"] = sanitizer_support.validate_summary(
                    args.out/(label+".log"), sanitizer)
            evidence = json.loads((args.out/(label+".json")).read_text())
            if evidence.get("verdict") != "PASS" or evidence.get("performance_acceptance") is not False:
                raise RuntimeError("GPU cell did not produce valid component-only evidence")
        finally:
            cleanup()
            entry["ended"] = time.time()
            save("progress.json", result)
        print("PASS "+label, flush=True)

    def run():
        cell("remap")
        for case in CASES:
            cell(case)
        for sanitizer in ("memcheck", "racecheck"):
            for case in ("remap", "balanced4096", "remote4096", "zeros4097"):
                cell(case, sanitizer)

    try:
        lifecycle.check_holder()
        lifecycle.pinned(str(root), args.revision)
        # Reject stale evidence before pausing the incoming service.
        validate_compile_evidence(root, root/CPU_EVIDENCE)
        sanitizer_receipt = sanitizer_support.preflight(
            lifecycle.IMAGE, args.out/"sanitizer-preflight.json")
        save("resources-before.json", resources(False))
        before = lifecycle.snapshot()
        save("before.json", before)
        mode = lifecycle.validate_before(before)
        result["incoming_mode"] = mode
        if mode in ("present", "stopped"):
            if mode == "present":
                lifecycle.idle(before["local"]["port"])
            lifecycle.with_paused(before, run, save, before_restore=cleanup)
            result["restored_original"] = True
            if mode == "present" and before["local"]["port"] != 8000:
                lifecycle.restore_public(args.out, save, result)
        else:
            try:
                run()
            finally:
                previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    cleanup()
                    lifecycle.restore_public(args.out, save, result)
                finally:
                    signal.signal(signal.SIGTERM, previous)
        result["exit_code"] = 0
    except BaseException as exc:
        result["error"] = repr(exc)
    finally:
        result["ended"] = time.time()
        # A failed cell can still have completed exact original recovery.
        if (args.out/"restored.json").exists():
            result["restored_original"] = True
        save("completion.json", result)
        print(json.dumps(result), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
