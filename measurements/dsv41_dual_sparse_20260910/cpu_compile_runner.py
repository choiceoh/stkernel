#!/usr/bin/env python3
"""Normal fleet CPU payload: compile dual-pool sparse attention without GPU devices.

Use a clean frozen revision and an already-present image. No model mount,
network, image pull, inference, CUDA context, or service change is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

IMAGE = "sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211"
INPUTS = (
    "overlay/modules/dsv41_model/dsv41_indexer.py",
    "overlay/modules/dsv41_model/dsv41_indexer_triton.py",
    "overlay/modules/dsv41_model/dsv41_layers.py",
    "overlay/modules/dsv41_model/dsv41_packed_index.py",
    "overlay/modules/dsv41_model/dsv41_packed_index_triton.py",
    "overlay/modules/dsv41_model/dsv41_packed_reference_adapter.py",
    "overlay/modules/dsv41_model/dsv41_reference_adapter.py",
    "overlay/modules/dsv41_model/dsv41_dual_sparse.py",
    "overlay/modules/dsv41_model/dsv41_dual_sparse_triton.py",
    "overlay/modules/dsv41_model/dsv41_dual_sparse_reference_adapter.py",
    "probes/dsv41_dual_sparse_diff.py",
    "probes/dsv41_packed_index_diff.py",
    "tests/test_dsv41_dual_sparse.py",
    "tests/test_dsv41_dual_sparse_reference_adapter.py",
    "measurements/dsv41_dual_sparse_20260910/cpu_compile_runner.py",
)

INNER = r'''
import hashlib, json, os, pathlib, sys
root, out = pathlib.Path('/repo'), pathlib.Path('/evidence')
assert os.environ['NVIDIA_VISIBLE_DEVICES'] == 'void'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
devices = [str(p) for pattern in ('nvidia*', 'dri/*', 'kfd')
           for p in pathlib.Path('/dev').glob(pattern)]
assert not devices, devices
import torch, triton
assert not torch.cuda.is_initialized()
sys.path.insert(0, str(root / 'overlay/modules/dsv41_model'))
from dsv41_dual_sparse_triton import offline_compile
compiled = offline_compile(out / 'compiled')
assert len(compiled) == 4, 'expected H8/H16/H32/H64 attention specializations'
assert not torch.cuda.is_initialized()
assert not any(k == 'vllm' or k.startswith('vllm.') for k in sys.modules)
report = dict(schema=1, passed=True, compiled=compiled,
              torch_version=torch.__version__, triton_version=triton.__version__,
              torch_cuda_version=torch.version.cuda, devices=devices,
              cuda_initialized=False, gpu_numerics=False,
              gpu_performance=False, model_equivalence=False)
(out / 'aot.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(argv):
    return subprocess.check_output(argv, text=True).strip()


def source_receipt(source, revision):
    actual = text(["git", "-C", str(source), "rev-parse", "HEAD"])
    if actual != revision or text(["git", "-C", str(source), "status", "--porcelain"]):
        raise RuntimeError("source must be clean at the frozen full revision")
    files = []
    for relative in INPUTS:
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/nonregular source: {relative}")
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path)))
    return dict(revision=actual, clean=True, files=files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        ap.error("full frozen revision required")
    source, out = args.source.resolve(strict=True), args.output.resolve()
    if out.is_relative_to(source):
        ap.error("evidence directory must be outside the source tree")
    before = source_receipt(source, args.revision)
    if sha(Path(__file__)) != sha(source / INPUTS[-1]):
        raise RuntimeError("runner differs from frozen source")
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if available < 12 * 1024**3:
        raise RuntimeError(f"need 12 GiB host MemAvailable before compile; got {available}")
    if text(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]) != IMAGE:
        raise RuntimeError("unexpected local image")
    out.mkdir(parents=True, exist_ok=False)
    (out / "inner.py").write_text(INNER)
    name = "dsv41-dual-sparse-cpu-" + str(os.getpid())
    env = {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "TRITON_CACHE_DIR": "/evidence/cache"}
    cmd = ["docker", "run", "--pull=never", "--rm", "--runtime=runc", "--network=none",
           "--name", name, "--cpuset-cpus=14-15", "--memory=6g", "--memory-swap=6g"]
    for key, value in env.items():
        cmd += ["-e", key + "=" + value]
    cmd += ["--mount", f"type=bind,src={source},dst=/repo,readonly",
            "--mount", f"type=bind,src={out},dst=/evidence",
            "--workdir", "/repo", "--entrypoint", "python3", IMAGE, "-B", "/evidence/inner.py"]
    report = dict(schema=1, passed=False, image=IMAGE, source_before=before,
                  mem_available_bytes=available, command=cmd, started_at=time.time(),
                  wrapper_sha256=sha(Path(__file__)), inner_sha256=sha(out / "inner.py"),
                  gpu_numerics=False, gpu_performance=False, model_equivalence=False)
    try:
        with (out / "compile.log").open("x") as log:
            report["returncode"] = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=600).returncode
        if report["returncode"] != 0:
            raise RuntimeError(f"offline compile failed: exit {report['returncode']}; see compile.log")
        receipt = json.loads((out / "aot.json").read_text())
        if not receipt["passed"] or receipt["cuda_initialized"]:
            raise RuntimeError("invalid offline compile receipt")
        report["source_after"] = source_receipt(source, args.revision)
        if before != report["source_after"]:
            raise RuntimeError("source changed during compile")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        subprocess.run(["docker", "stop", "-t", "1", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=10)
        report["finished_at"] = time.time()
        (out / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
        files = [p for p in sorted(out.rglob("*")) if p.is_file()]
        (out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"passed": True, "output": str(out), "execution_sha256": sha(out / "execution.json")}))


if __name__ == "__main__":
    main()
