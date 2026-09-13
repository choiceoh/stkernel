"""GB10 execution-plan gate; compile-only never creates a CUDA context.

GPU use belongs to bench/fleet.sh. This focused gate is not onepass proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def compile_dense():
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    source = ROOT / "engine/kernels/dense/kernels.cu"
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a",
             "-DMK_GRID_DEF=96", "-DMK_MHC_GRID_DEF=144", "-DMK_NBUF2_DEF=3",
             "-DMK_FP8_PACK2_DEF=1", "-DMK_GEMM_TRANSPOSE_M8_DEF=1",
             "-DMK_GEMM_COMPACT_M8_DEF=1", "-DMK_M8_FASTPATH_DEF=1"]
    key, directory, staged = prepare_sources(Path.home()/".cache/st/dense", [source],
                                              (flags, torch.__version__, torch.version.cuda))
    ext = load(name="st_dense_"+key, sources=list(staged), extra_cuda_cflags=flags,
               build_directory=str(directory), verbose=False)
    if torch.cuda.is_initialized():
        raise RuntimeError("compile-only initialized CUDA")
    return {"module": ext.__name__, "workspace_bytes": ext.gemm_workspace_elements()*4,
            "cuda_initialized": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("/cache/execution-plan.json"))
    args = ap.parse_args()
    start = time.monotonic()
    if args.compile_only:
        result = compile_dense()
        scope = "SM121a compilation only; no numerical or performance verdict"
    else:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("the numerical gate requires an admitted GB10")
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_engine_execution_plan_cuda")
        run = unittest.TextTestRunner(verbosity=2).run(suite)
        if not run.wasSuccessful() or run.skipped:
            raise RuntimeError("execution-plan numerical gate failed or skipped")
        result = {"tests": run.testsRun, "skipped": len(run.skipped)}
        scope = "private W4 concurrent graph correctness; no serving performance verdict"
    files = ("engine/profiles/glm53/execution.py", "engine/kernels/dense/kernels.cu",
             "engine/profiles/glm53/decode_graphs.py", "engine/profiles/glm53/pipeline.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": scope, "result": result, "seconds": time.monotonic()-start,
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}, indent=2)+"\n")


if __name__ == "__main__":
    main()
