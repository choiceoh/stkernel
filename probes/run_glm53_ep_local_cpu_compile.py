#!/usr/bin/env python3
"""Resource-capped CPU-only container for the large MoE compile check.

No GPU/device/runtime override or arbitrary command is accepted. This can
run under fleet run --cpu on a worker with sufficient available host RAM.
The inner check asserts that device nodes and an initialized context are absent.
"""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--arm", choices=("stock", "local"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        ap.error("image must be an immutable local sha256 ID")
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 12 * 1024 * 1024:
        ap.error("CPU compile needs at least 12 GiB available; no serving memory is reclaimed")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "--runtime=runc", "--network=none",
               "--memory=4g", "--memory-swap=4g", "--cpus=2", "--pids-limit=128",
               "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "MAX_JOBS=1",
               "--entrypoint=python3", "-v", f"{root}:/repo:ro",
               "-v", f"{output}:/evidence"]
    for line in (root / "build/glm53/manifest.tsv").read_text().splitlines():
        name, target, *_ = line.split("\t")
        # Mount every actual MoE source from the composed manifest.
        if "/flashinfer/" in target or name == "flashinfer_b12x_moe.py":
            command += ["-v", f"{root / 'build/glm53' / name}:{target}:ro"]
    command += [args.image, "/repo/probes/glm53_ep_local_compile_check.py",
                "--arm", args.arm, "--output", "/evidence"]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
