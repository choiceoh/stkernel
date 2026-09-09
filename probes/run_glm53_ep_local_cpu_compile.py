#!/usr/bin/env python3
"""Resource-capped CPU-only container for the large MoE compile check.

No GPU/device/runtime override or arbitrary command is accepted. This can
run under fleet run --cpu on a worker with sufficient available host RAM.
The inner check asserts that device nodes and an initialized context are absent.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import glm53_ep_capsule_runtime as capsule_runtime
from glm53_ep_local_evidence import (CONTRACT_PATHS, compile_cases, digest, mounted_sources,
                                    validate_compile_evidence)


def validate_result(root, path, arm):
    evidence = json.loads(Path(path).read_text())
    if (evidence.get("verdict") != "PASS" or evidence.get("phase") != "complete"
            or evidence.get("binding_runtime_rechecked") is not True
            or "error" in evidence or "binding_runtime_recheck_error" in evidence
            or evidence.get("arm") != arm or evidence.get("cuda_initialized") is not False
            or not evidence.get("cache_key") or not evidence.get("artifacts") or not evidence.get("resources")):
        raise ValueError("matching no-device compile result required")
    capsule_runtime.validate_runtime_receipt(evidence.get("binding_runtime"))
    if arm == "local":
        return validate_compile_evidence(root, path)
    if "glm53_ep_prefill_local_fp32_v2" in evidence["cache_key"]:
        raise ValueError("stock compile result contains the local dispatch tag")
    contracts = evidence.get("contracts", {})
    if (contracts.get("tests_run", 0) <= 0 or contracts.get("failures") != 0
            or contracts.get("errors") != 0 or contracts.get("skips") != 0):
        raise ValueError("passing pinned CPU contracts without skips required")
    for relative in CONTRACT_PATHS:
        if contracts.get("files", {}).get(relative) != digest(root / relative):
            raise ValueError("CPU-tested contract source changed: " + relative)
    remap = evidence.get("remap_compilation", [])
    if ({row.get("label") for row in remap} != {row["label"] for row in compile_cases()}
            or len(remap) != len(compile_cases())
            or not all(row.get("ptx_sha256") and row.get("cubin_sha256") for row in remap)):
        raise ValueError("complete no-device remap compilation proof required")
    mounted = mounted_sources(root)
    if set(evidence.get("mounted_sources", {})) != set(mounted):
        raise ValueError("compiled and mounted MoE file sets differ")
    for target, source in mounted.items():
        if evidence["mounted_sources"][target] != digest(source):
            raise ValueError("compiled overlay source changed: " + source.name)
        if target in evidence.get("sources", {}) and evidence["sources"][target] != digest(source):
            raise ValueError("kernel and mount source receipts disagree: " + source.name)
    return evidence


def postcheck_capsule(path, sha, *, prior_failure):
    try:
        capsule_runtime.validate_capsule_input(path, sha)
    except Exception as exc:
        if not prior_failure:
            raise
        # Preserve the original Docker exit/exception while retaining the
        # additional integrity failure in the captured outer log.
        print("CAPSULE POSTCHECK FAILED: " + repr(exc), file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--arm", choices=("stock", "local"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--capsule-root", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        ap.error("image must be an immutable local sha256 ID")
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 12 * 1024 * 1024:
        ap.error("CPU compile needs at least 12 GiB available; no serving memory is reclaimed")
    capsule = capsule_runtime.validate_capsule_input(args.capsule_root, args.manifest_sha256)
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (root, capsule)):
        ap.error("output must be separate from the capsule and frozen source")
    if (output / "result.json").exists():
        ap.error("a fresh compile result path is required")
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
    command += capsule_runtime.docker_capsule_args(capsule, args.manifest_sha256)
    command += [args.image, "-B", "/repo/probes/glm53_ep_local_compile_check.py",
                "--arm", args.arm, "--output", "/evidence",
                "--capsule-root", capsule_runtime.CAPSULE_MOUNT,
                "--manifest-sha256", args.manifest_sha256]
    status = None
    try:
        status = subprocess.call(command)
    finally:
        postcheck_capsule(capsule, args.manifest_sha256, prior_failure=status != 0)
    if status:
        return status
    validate_result(root, output / "result.json", args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
