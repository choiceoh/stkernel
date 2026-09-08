#!/usr/bin/env python3
"""Compile actual large-prefill dispatcher arms without a CUDA context.

Run through run_glm53_ep_local_cpu_compile.py. Fake pointers/streams are compiler
arguments only. PTX and assembler resource diagnostics are CPU evidence;
neither validates device numerics, barriers, or throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("stock", "local"), required=True)
    ap.add_argument("--m", type=int, default=6912)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUTE_DSL_ARCH="sm_121a", CUTE_DSL_KEEP="ptx,cubin",
                      CUTE_DSL_DUMP_DIR=str(args.output),
                      CUTE_DSL_CACHE_DIR=str(args.output / "cache"),
                      CUTE_DSL_DISABLE_FILE_CACHING="1",
                      CUTE_DSL_COMPILER_OPT="ptx-options=-v",
                      VLLM_GLM53_B12X_PREFILL_REUSE="0",
                      VLLM_GLM53_B12X_PREFILL_FC1_N128="0",
                      VLLM_GLM53_EP_PREFILL_LOCAL=str(int(args.arm == "local")),
                      VLLM_GLM53_B12X_STATIC_V2="0")
    assert not list(Path("/dev").glob("nvidia*")), "CPU container exposes GPU devices"
    import torch
    assert not torch.cuda.is_initialized()
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_capability = lambda *a, **kw: (12, 1)
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    md.get_num_sm = lambda *a: 48
    md.get_max_active_clusters = lambda *a: 48
    # Build from this process's mounted sources; never load an on-disk .so.
    md.build_and_load_cute_dsl_kernel = lambda module, name, build, **kw: build()
    t0 = time.monotonic()
    md._get_dynamic_kernel(
        72, args.m, 4096, 2048, 8, args.m,
        activation="swigluoai_uninterleave", swiglu_alpha=1.0,
        swiglu_beta=0.0, swiglu_limit=10.0, tiled=False)
    keys = list(md._DYNAMIC_KERNEL_CACHE)
    assert len(keys) == 1, keys
    tag = "glm53_ep_prefill_local_v1"
    assert (keys[0][-1] == tag) if args.arm == "local" else tag not in keys[0]
    assert not torch.cuda.is_initialized(), "compile created a CUDA context"
    artifacts = []
    for ptx in sorted(args.output.rglob("*.ptx")):
        artifacts.append({"file": ptx.name, "bytes": ptx.stat().st_size,
                          "sha256": hashlib.sha256(ptx.read_bytes()).hexdigest()})
    resources = []
    for cubin in sorted(args.output.rglob("*.cubin")):
        # Inspect the DSL's compiled binary itself. The image's stand-alone
        # CUDA 13.0 ptxas cannot reassemble the DSL compiler's PTX 9.3.
        result = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin)],
                                text=True, capture_output=True)
        text = result.stdout + result.stderr
        cubin.with_suffix(".resources.log").write_text(text)
        print(text, flush=True)
        result.check_returncode()
        resources.append({"file": cubin.name, "bytes": cubin.stat().st_size,
                          "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
                          "resources": text})
    assert resources, "no compiled cubin resource evidence"
    assert artifacts, "no generated PTX: cached builds are not compile proof"
    evidence = dict(arm=args.arm, m=args.m,
                    elapsed_s=time.monotonic() - t0, cache_key=keys[0],
                    cuda_initialized=torch.cuda.is_initialized(), artifacts=artifacts, resources=resources,
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in map(Path, (*md._kernel_source_files(), str(Path(md.__file__).with_name("moe_dynamic_ep_local.py"))))})
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    print(f"COMPILE PASS arm={args.arm} m={args.m}", flush=True)


if __name__ == "__main__":
    main()
