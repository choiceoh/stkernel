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
import unittest
import time

if __package__:
    from .glm53_ep_local_evidence import CONTRACT_PATHS, CPU_TEST_MODULES, digest, mounted_sources
    from .glm53_ep_route_remap_check import compile_remap
    from .glm53_ep_capsule_runtime import verify_runtime
else:
    from glm53_ep_local_evidence import CONTRACT_PATHS, CPU_TEST_MODULES, digest, mounted_sources
    from glm53_ep_route_remap_check import compile_remap
    from glm53_ep_capsule_runtime import verify_runtime


def compile_arm(args, evidence):
    os.environ.update(CUTE_DSL_ARCH="sm_121a", CUTE_DSL_KEEP="ptx,cubin",
                      CUTE_DSL_DUMP_DIR=str(args.output),
                      CUTE_DSL_CACHE_DIR=str(args.output / "cache"),
                      CUTE_DSL_DISABLE_FILE_CACHING="1",
                      CUTE_DSL_COMPILER_OPT="ptx-options=-v",
                      VLLM_GLM53_B12X_PREFILL_REUSE="0",
                      VLLM_GLM53_B12X_PREFILL_FC1_N128="0",
                      VLLM_GLM53_EP_PREFILL_LOCAL=str(int(args.arm == "local")),
                      VLLM_GLM53_B12X_STATIC_V2="0")
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
    tag = "glm53_ep_prefill_local_fp32_v2"
    assert (keys[0][-1] == tag) if args.arm == "local" else tag not in keys[0]
    assert not torch.cuda.is_initialized(), "compile created a CUDA context"
    evidence["phase"] = "compiler-artifacts"
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
    evidence.update(artifacts=artifacts, resources=resources)
    evidence["phase"] = "remap-compilation"
    remap_compilation = compile_remap(args.output)
    evidence["remap_compilation"] = remap_compilation
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name)
        for name in CPU_TEST_MODULES)
    evidence["phase"] = "cpu-contracts"
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    contracts = dict(tests_run=tested.testsRun, failures=len(tested.failures),
                     errors=len(tested.errors), skips=len(tested.skipped),
                     files={name: digest(root / name) for name in CONTRACT_PATHS})
    evidence["contracts"] = contracts
    assert tested.wasSuccessful() and not tested.skipped, contracts
    assert not torch.cuda.is_initialized(), "CPU contracts created a CUDA context"
    mounted = {}
    for target, source in mounted_sources(root).items():
        assert digest(source) == digest(target), target
        mounted[target] = digest(target)
    evidence.update(arm=args.arm, m=args.m,
                    elapsed_s=time.monotonic() - t0, cache_key=keys[0],
                    cuda_initialized=torch.cuda.is_initialized(), artifacts=artifacts, resources=resources,
                    contracts=contracts, mounted_sources=mounted, remap_compilation=remap_compilation,
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in map(Path, (*md._kernel_source_files(), str(Path(md.__file__).with_name("moe_dynamic_ep_local.py"))))})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("stock", "local"), required=True)
    ap.add_argument("--m", type=int, default=6912)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--capsule-root", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = dict(arm=args.arm, m=args.m, verdict="RUNNING", phase="cpu-device-guard")
    try:
        assert not list(Path("/dev").glob("nvidia*")), "CPU container exposes GPU devices"
        evidence["phase"] = "binding-runtime"
        runtime = verify_runtime(args.capsule_root, args.manifest_sha256)
        evidence["binding_runtime"] = runtime
        operation_error = None
        try:
            evidence["phase"] = "compile"
            compile_arm(args, evidence)
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            if operation_error is None:
                evidence["phase"] = "binding-runtime-recheck"
            try:
                if verify_runtime(args.capsule_root, args.manifest_sha256) != runtime:
                    raise RuntimeError("binding runtime changed during CPU compilation")
                evidence["binding_runtime_rechecked"] = True
            except BaseException as exc:
                evidence["binding_runtime_recheck_error"] = repr(exc)
                if operation_error is None:
                    raise
        evidence.update(verdict="PASS", phase="complete")
        print(f"COMPILE PASS arm={args.arm} m={args.m}", flush=True)
    except BaseException as exc:
        evidence.update(verdict="FAIL", error=repr(exc))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
        print(json.dumps(evidence, default=str), flush=True)


if __name__ == "__main__":
    main()
