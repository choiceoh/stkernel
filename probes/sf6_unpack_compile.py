#!/usr/bin/env python3
"""CPU-only assembly comparison of SF6 scalar and four-byte expansion.

The scalar reference is the inner loop of MoEStaticKernelV4._sf_expand_stage
at baseline c24494aa. Both arms use identical nonconstant global input loads
and output stores, including one raw-base load per thread. Only the u8x4 arm
broadcasts that base once before calling the actual production helper.

This is isolated arithmetic compile evidence for 1/4/8 output words per
thread, not a full-kernel instruction count, GPU correctness check or timing.
There are no tensor allocations on CUDA and compiled kernels are never run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _save_assembly(compiled, directory, stem):
    artifacts = {}
    for kind, expected_type in (("ptx", str), ("cubin", bytes), ("sass", str)):
        value = getattr(compiled, f"__{kind}__")
        if not isinstance(value, expected_type) or not value:
            raise AssertionError((stem, kind, "missing or wrong assembly type"))
        if isinstance(value, str):
            if not value.strip():
                raise AssertionError((stem, kind, "empty assembly text"))
            value = value.encode("utf-8")
        path = directory / f"{stem}.{kind}"
        with path.open("xb") as output:
            output.write(value)
        artifacts[kind] = {"path": path.name, "sha256": _sha256(value),
                           "bytes": len(value)}
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        parser.error("--out already exists; use a fresh output directory")
    compiler_dir, cache_dir = out.parent / "compiler", out.parent / "cache"
    # No earlier invocation's cache or dump may supply assembly evidence.
    compiler_dir.mkdir(exist_ok=False)
    cache_dir.mkdir(exist_ok=False)
    os.environ["CUTE_DSL_ARCH"] = "sm_121a"
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin,sass"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(compiler_dir)
    os.environ["CUTE_DSL_CACHE_DIR"] = str(cache_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))
    report = {"schema": "sf6-unpack-compile-v1", "status": "FAIL", "mode": "cpu",
              "evidence": "isolated-unpack-compile-only", "cuda_initialized": None,
              "baseline_commit": "c24494aa", "arch": "sm_121a",
              "compile_options": "--opt-level 2 --enable-tvm-ffi",
              "base_input": "one nonconstant raw-base load per thread in both arms",
              "words_per_thread": [1, 4, 8], "cases": [],
              "source_sha256": {Path(__file__).name: _sha256(Path(__file__).read_bytes())}}
    torch = None
    try:
        import torch
        assert not torch.cuda.is_initialized(), "CUDA initialized before CPU compile"
        # Import-time architecture selection only; compilation receives fake
        # tensors and a fake stream, never a CUDA allocation or real stream.
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda *args, **kwargs: (12, 1)
        import cutlass
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_common
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_common import _sf6_unpack_u8x4

        source = Path(moe_static_common.__file__).resolve()
        report["source_sha256"][source.name] = _sha256(source.read_bytes())
        report["production_helper"] = f"{moe_static_common.__name__}._sf6_unpack_u8x4"

        def compile_arm(words, vector):
            @cute.kernel
            def unpack(low: cute.Tensor, high: cute.Tensor, bases: cute.Tensor,
                       output: cute.Tensor):
                tid, _, _ = cute.arch.thread_idx()
                # Keep packed source-word reuse identical to production:
                # each low word supplies 8 bytes and each high word 16 bytes.
                a = []
                for w in range((words + 1) // 2):
                    a.append(low[tid, w])
                b = []
                for w in range((words + 3) // 4):
                    b.append(high[tid, w])
                base = bases[tid] & cutlass.Int32(0xFF)
                if cutlass.const_expr(vector):
                    base_word = base * cutlass.Int32(0x01010101)
                for j in range(words):
                    if cutlass.const_expr(vector):
                        low4 = a[j >> 1] >> cutlass.Int32(16 * (j & 1))
                        high4 = b[j >> 2] >> cutlass.Int32(8 * (j & 3))
                        word = _sf6_unpack_u8x4(low4, high4, base_word)
                    else:
                        # Verbatim scalar arithmetic from baseline c24494aa,
                        # with Int32 qualified to avoid an import alias.
                        word = cutlass.Int32(0)
                        for m in range(4):
                            i = 4 * j + m
                            nib = (a[i >> 3] >> cutlass.Int32(8 * ((i >> 1) & 3) + 4 * (i & 1))) & cutlass.Int32(0xF)
                            hi = (b[i >> 4] >> cutlass.Int32(8 * ((i >> 2) & 3) + 2 * (i & 3))) & cutlass.Int32(0x3)
                            val = (base + nib + (hi << cutlass.Int32(4))) & cutlass.Int32(0xFF)
                            word = word | (val << cutlass.Int32(8 * m))
                    output[tid, j] = word

            @cute.jit
            def entry(low: cute.Tensor, high: cute.Tensor, bases: cute.Tensor,
                      output: cute.Tensor, stream: cuda.CUstream):
                unpack(low, high, bases, output).launch(
                    grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

            tensors = [cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, shape, stride_order=tuple(reversed(range(len(shape)))),
                assumed_align=16) for shape in
                ((128, (words + 1) // 2), (128, (words + 3) // 4), (128,), (128, words))]
            return cute.compile(entry, *tensors,
                                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                                options=report["compile_options"])

        for words in report["words_per_thread"]:
            for arm in ("scalar", "u8x4"):
                compiled = compile_arm(words, arm == "u8x4")
                artifacts = _save_assembly(compiled, out.parent, f"{arm}-words{words}")
                report["cases"].append({"words": words, "arm": arm, "artifacts": artifacts})
                assert not torch.cuda.is_initialized(), "CPU compile initialized CUDA"
        assert len(report["cases"]) == 6, "incomplete compile matrix"
        report["status"] = "PASS"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if torch is not None:
            report["cuda_initialized"] = bool(torch.cuda.is_initialized())
            if report["cuda_initialized"]:
                report["status"] = "FAIL"
                report["cuda_error"] = "CPU compile initialized CUDA"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "mode": "cpu", "out": str(out),
                      "cases": len(report["cases"]),
                      "cuda_initialized": report["cuda_initialized"]}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
