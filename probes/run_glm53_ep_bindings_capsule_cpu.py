#!/usr/bin/env python3
"""Run the fixed capsule import check in a resource-capped no-device container.

Use the normal fleet --cpu path. No image, device/runtime, command, or serving
override is accepted. An existing 12 GiB host-memory guard remains unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from glm53_ep_bindings_capsule import PINNED_WHEELS
from glm53_ep_bindings_capsule_check import IMAGE, SCOPE, file_hash, source_hashes


def host_available_kib():
    return next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def command(root, wheels, output, name):
    paths = [Path(path).resolve() for path in (root, wheels, output)]
    if any(any(char in str(path) for char in ",\n\r\0") for path in paths):
        raise ValueError("bind paths must not contain mount separators or control characters")
    root, wheels, output = paths
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (root, wheels)):
        raise ValueError("output must be separate from the read-only source and wheel directories")
    return ["docker", "run", "--rm", "--name", name, "--pull=never", "--runtime=runc", "--network=none",
            "--memory=4g", "--memory-swap=4g", "--cpus=2", "--pids-limit=128",
            "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint=python3",
            "--mount", f"type=bind,source={root},target=/repo,readonly",
            "--mount", f"type=bind,source={wheels},target=/wheels,readonly",
            "--mount", f"type=bind,source={output},target=/evidence",
            IMAGE, "-B", "/repo/probes/glm53_ep_bindings_capsule_check.py"]


def run(wheels, output):
    available = host_available_kib()
    if available < 12 * 1024 * 1024:
        raise RuntimeError("CPU check needs at least 12 GiB available; no serving memory is reclaimed")
    root = Path(__file__).resolve().parents[1]
    wheels, output = Path(wheels).resolve(), Path(output).resolve()
    wheel_hashes = {}
    for pin in PINNED_WHEELS.values():
        path = wheels / pin["filename"]
        if path.is_symlink() or not path.is_file() or file_hash(path) != pin["sha256"]:
            raise RuntimeError("missing or changed pinned wheel: " + pin["filename"])
        wheel_hashes[pin["filename"]] = pin["sha256"]
    name = "ep-bindings-capsule-cpu-" + uuid.uuid4().hex
    argv = command(root, wheels, output, name)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=1, verdict="FAIL", scope=SCOPE, image=IMAGE, command=argv,
                   started=time.time(), host_mem_available_kib=available,
                   sources=source_hashes(root), wheels=wheel_hashes)
    try:
        with (output / "container.log").open("x") as log:
            try:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            except subprocess.TimeoutExpired:
                cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30)
                receipt["timeout_cleanup_exit_code"] = cleanup.returncode
                raise
        receipt["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("capsule CPU container failed")
        result_path = output / "result.json"
        inner = json.loads(result_path.read_text())
        if (inner.get("verdict") != "PASS" or inner.get("image") != IMAGE
                or inner.get("sources") != receipt["sources"] or inner.get("scope") != SCOPE
                or inner.get("exposed_device_nodes") != []
                or inner.get("probe_cuda_api_calls") is not False
                or inner.get("cuda_context_queried") is not False or inner.get("torch_imported") is not False
                or inner.get("dependencies", {}).get("compatible") is not True
                or not inner.get("imports")):
            raise RuntimeError("missing or mismatched capsule CPU result")
        if receipt["sources"] != source_hashes(root):
            raise RuntimeError("probe source changed during the check")
        receipt["result_sha256"] = file_hash(result_path)
        receipt["capsule_manifest_sha256"] = inner["capsule_manifest_sha256"]
        receipt["verdict"] = "PASS"
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt["ended"] = time.time()
        log = output / "container.log"
        if log.is_file():
            receipt["log_sha256"] = file_hash(log)
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.wheels, args.output)
    print(json.dumps(dict(verdict=result["verdict"], image=IMAGE, output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
