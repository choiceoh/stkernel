"""Pinned Compute Sanitizer identity, no-device preflight, and strict summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

HOST_DIR = Path("/usr/local/cuda/compute-sanitizer")
CONTAINER_DIR = "/opt/glm-ep-sanitizer"
BINARY_SHA256 = "7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b"
VERSION = "2025.3.1.0"


def identity():
    directory = HOST_DIR.resolve(strict=True)
    binary = directory / "compute-sanitizer"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("host Compute Sanitizer executable is missing or not executable")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != BINARY_SHA256:
        raise RuntimeError("host Compute Sanitizer executable hash changed: " + digest)
    return dict(directory=str(directory), executable=str(binary), sha256=digest)


def assert_identity(receipt):
    current = identity()
    if (receipt is None or receipt.get("verdict") != "PASS"
            or receipt.get("identity") != current or receipt.get("version") != VERSION):
        raise RuntimeError("Compute Sanitizer differs from the successful no-device preflight")
    return current


def mount_args(receipt):
    current = assert_identity(receipt)
    return ["--mount", f"type=bind,source={current['directory']},target={CONTAINER_DIR},readonly"]


def command(tool):
    if tool not in ("memcheck", "racecheck"):
        raise ValueError("only memcheck and racecheck are supported")
    return [CONTAINER_DIR + "/compute-sanitizer", "--tool", tool, "--error-exitcode=86"]


def preflight(image, output):
    """Check the mounted executable inside the serving image before any pause.

    Explicit runc and void NVIDIA visibility exclude the NVIDIA runtime. The
    inner process also rejects device nodes, and bounds --version to 15 seconds.
    """
    report = dict(started=time.time(), verdict="FAIL", version=VERSION,
                  cuda_devices_exposed=None)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("an immutable local image ID is required")
        report["identity"] = identity()
        name = "ep-sanitizer-preflight-" + uuid.uuid4().hex
        inner = (
            "from pathlib import Path; import subprocess; "
            "assert not list(Path('/dev').glob('nvidia*')), 'GPU device exposed'; "
            "p=subprocess.run(['/opt/glm-ep-sanitizer/compute-sanitizer','--version'], "
            "text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15); "
            "print(p.stdout,end=''); raise SystemExit(p.returncode)"
        )
        directory = report["identity"]["directory"]
        invocation = ["docker", "run", "--rm", "--name", name, "--runtime=runc",
                      "--network=none", "--memory=256m", "--memory-swap=256m",
                      "--cpus=1", "--pids-limit=64", "--stop-timeout=5",
                      "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=",
                      "--entrypoint=python3", "--mount",
                      f"type=bind,source={directory},target={CONTAINER_DIR},readonly",
                      image, "-c", inner]
        report["command"] = invocation
        try:
            checked = subprocess.run(invocation, text=True, capture_output=True, timeout=45)
        except BaseException:
            # Only this uniquely named, bounded preflight container is ours.
            subprocess.run(["docker", "rm", "-f", name], text=True,
                           capture_output=True, timeout=15)
            raise
        report.update(exit_code=checked.returncode, stdout=checked.stdout, stderr=checked.stderr)
        if checked.returncode:
            raise RuntimeError(f"Compute Sanitizer no-device preflight exited {checked.returncode}")
        text = checked.stdout + checked.stderr
        if not re.search(r"\bversion\s+" + re.escape(VERSION) + r"\b", text, re.I):
            raise RuntimeError("Compute Sanitizer version output is missing or changed")
        if identity() != report["identity"]:
            raise RuntimeError("Compute Sanitizer changed during preflight")
        report.update(verdict="PASS", cuda_devices_exposed=False)
        return report
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["ended"] = time.time()
        output.write_text(json.dumps(report, indent=2) + "\n")


def validate_summary(log, tool):
    """Exit zero is insufficient: require the instrumented tool's clean summary."""
    command(tool)
    text = Path(log).read_text()
    if re.search(r"^\s*=+\s*(?:ERROR|FATAL)\s*:", text, re.I | re.M):
        raise RuntimeError("Compute Sanitizer reported a tool error")
    error_lines = re.findall(r"^\s*=+\s*(ERROR SUMMARY:.*)$", text, re.M)
    race_lines = re.findall(r"^\s*=+\s*(RACECHECK SUMMARY:.*)$", text, re.M)
    if any(line.strip() != "ERROR SUMMARY: 0 errors" for line in error_lines):
        raise RuntimeError("Compute Sanitizer reported errors or an unrecognized error summary")
    clean_race = "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
    if any(line.strip() != clean_race for line in race_lines):
        raise RuntimeError("Compute Sanitizer reported races or an unrecognized race summary")
    required = error_lines if tool == "memcheck" else race_lines
    if not required:
        raise RuntimeError("Compute Sanitizer clean " + tool + " summary is missing")
    return dict(tool=tool, verdict="PASS", summaries=required,
                log_sha256=hashlib.sha256(Path(log).read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    preflight(args.image, args.output)


if __name__ == "__main__":
    main()
