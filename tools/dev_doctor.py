#!/usr/bin/env python3
"""Report whether this checkout is ready for the common CPU/GPU workflows."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PACKAGES = ("numpy", "safetensors")
OPTIONAL_PACKAGES = ("torch", "triton")


@dataclass
class Finding:
    name: str
    status: str
    detail: str
    required: bool = False


def run_git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_finding(name: str, required: bool) -> Finding:
    if importlib.util.find_spec(name) is None:
        status = "FAIL" if required else "WARN"
        detail = "not installed"
        return Finding(name, status, detail, required)
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "version unavailable")
    except Exception as exc:  # pragma: no cover - depends on local native wheels
        status = "FAIL" if required else "WARN"
        return Finding(name, status, f"import failed: {type(exc).__name__}", required)
    return Finding(name, "OK", str(version), required)


def collect_findings(require_gpu: bool, strict: bool) -> list[Finding]:
    findings = [
        Finding(
            "python",
            "OK" if sys.version_info >= (3, 11) else "FAIL",
            platform.python_version(),
            True,
        ),
        Finding(
            "checkout",
            "OK" if (ROOT / "engine").is_dir() and (ROOT / "tests").is_dir() else "FAIL",
            str(ROOT),
            True,
        ),
    ]

    branch = run_git("branch", "--show-current") or "detached"
    dirty = run_git("status", "--porcelain")
    dirty_count = len(dirty.splitlines()) if dirty else 0
    findings.append(
        Finding(
            "git",
            "WARN" if dirty_count else "OK",
            f"branch={branch}, changes={dirty_count}",
        )
    )
    findings.extend(package_finding(name, required=strict) for name in BASELINE_PACKAGES)
    findings.extend(package_finding(name, required=False) for name in OPTIONAL_PACKAGES)

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        findings.append(
            Finding(
                "cuda",
                "FAIL" if require_gpu else "WARN",
                "torch is unavailable; GPU tests cannot run",
                require_gpu,
            )
        )
        return findings

    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        if not available:
            findings.append(
                Finding(
                    "cuda",
                    "FAIL" if require_gpu else "WARN",
                    f"unavailable; torch_cuda={torch.version.cuda or 'none'}",
                    require_gpu,
                )
            )
            return findings
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        findings.append(
            Finding(
                "cuda",
                "OK",
                f"torch_cuda={torch.version.cuda or 'none'}, devices={'; '.join(names)}",
                require_gpu,
            )
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        findings.append(
            Finding(
                "cuda",
                "FAIL" if require_gpu else "WARN",
                f"probe failed: {type(exc).__name__}",
                require_gpu,
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when baseline packages or required environment checks are missing",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="require an importable torch CUDA runtime and at least one device",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable findings")
    args = parser.parse_args()

    findings = collect_findings(require_gpu=args.gpu, strict=args.strict)
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        for finding in findings:
            print(f"[{finding.status:<4}] {finding.name:<10} {finding.detail}")

    if args.strict:
        failures = [finding for finding in findings if finding.status == "FAIL"]
        if failures:
            print(f"\nstrict doctor: {len(failures)} blocking finding(s)", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
