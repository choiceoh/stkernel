"""Compile the tree KDA and persistent MLP for SM121, without a CUDA context."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from engine.kernels.kda.tree import _prepare, _verify, _materialize, _conv
from engine.kernels.tile_dataflow import _workers
from engine.kernels.w4a8_pipeline import _gate_up, _down


def compile_variants(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("offline probe must not inherit an initialized CUDA context")
    variants = [
        ("kda-conv", _conv, {**{p: "*bf16" for p in ("X", "W", "OUT", "HISTORY")}, "PATH": "*i32"},
            dict(C=6144, TAPS=4, B=256)),
        ("kda-conv-fp32-weights", _conv, {**{p: "*bf16" for p in ("X", "OUT", "HISTORY")}, "W": "*fp32", "PATH": "*i32"},
            dict(C=6144, TAPS=4, B=256)),
        ("kda-prepare", _prepare, {**{p: "*bf16" for p in ("Q", "K", "G", "B")},
            **{p: "*fp32" for p in ("A", "BIAS", "QF", "KF", "DEC", "BET")}},
            dict(H=16, D=128, LOWER=-5., BD=128)),
        ("kda-commit", _materialize, {**{p: "*fp32" for p in ("INITIAL", "KEY", "DEC", "UPDATE", "OUT")},
                                     "PATH": "*i32"}, dict(COUNT=8, H=16, K=128, V=128, B=1024))]
    for nodes, depth, k, v in ((8, 8, 128, 128), (16, 5, 128, 128), (31, 5, 128, 128), (6, 4, 7, 5)):
        for carry in (False, True):
            variants.append((f"kda-tree-n{nodes}-d{k}-v{v}-carry{int(carry)}", _verify,
                {**{p: "*fp32" for p in ("Q", "K", "DEC", "BET", "INITIAL", "UPDATE")},
                 **{p: "*bf16" for p in ("VEC", "OUT")}, **{p: "*i32" for p in ("PATH", "DEPTH", "ORDER", "PARENT")}},
                dict(N=nodes, H=16, KDIM=k, VDIM=v, WIDTH=depth, BK=triton.next_power_of_2(k), BV=16, CARRY=carry)))
    for rows, hidden, intermediate in ((1, 4096, 1024), (4, 4096, 1024), (16, 4096, 1024), (17, 95, 33)):
        variants.append((f"mlp-m{rows}-h{hidden}-i{intermediate}", _workers,
            {**{p: "*bf16" for p in ("X", "WG", "WD", "U", "OUT")}, "PART": "*fp32", "CONTROL": "*i32"},
            dict(M=rows, H=hidden, I=intermediate, P=triton.cdiv(intermediate, 32), O=triton.cdiv(hidden, 32),
                 BM=max(16, triton.next_power_of_2(rows)), B=32, LIMIT=10., SPINS=1000000,
                 S1=None, S2=None, US=None, A1=None, A2=None, ALPHA1=None, ALPHA2=None,
                 NV4=False, TILE13=0, TILE2=0, SF6=False, W4A8=False)))
    for rows, tiled, sf6 in ((1, False, False), (4, True, False), (16, True, True)):
        variants.append((f"nvfp4-m{rows}-tile{int(tiled)}-sf6{int(sf6)}", _workers,
            {**{p: "*u8" for p in ("WG", "WD", "U", "S1", "S2")},
             **{p: "*bf16" for p in ("X", "OUT")}, "US": "*u8", "CONTROL": "*i32",
             **{p: "*fp32" for p in ("PART", "A1", "A2", "ALPHA1", "ALPHA2")}},
            dict(M=rows, H=4096, I=3072, P=48, O=64, BM=16, B=64, LIMIT=10., SPINS=1000000,
                 NV4=True, TILE13=256 if tiled else 0, TILE2=64 if tiled else 0, SF6=sf6, W4A8=False)))
    for rows in (1, 4, 16, 32):
        for name, fn in (("gate-up", _gate_up), ("down", _down)):
            signature = {**{p: "*u8" for p in ("U", "W")}, "S": "*i8", "RS": "*fp32", "US": "*fp32"}
            signature["X" if fn is _gate_up else "OUT"] = "*bf16"
            constants = dict(M=rows, H=4096, I=3072, BM=max(16, rows))
            if fn is _gate_up:
                constants["LIMIT"] = 10.
            variants.append((f"w4a8-pipeline-{name}-m{rows}", fn, signature, constants))
        variants.append((f"w4a8-m{rows}", _workers,
            {**{p: "*u8" for p in ("WG", "WD")}, **{p: "*i8" for p in ("S1", "S2")},
             **{p: "*bf16" for p in ("X", "OUT")}, "U": "*u8", "CONTROL": "*i32",
             **{p: "*fp32" for p in ("PART", "US", "ALPHA1", "ALPHA2")}},
            dict(M=rows, H=4096, I=3072, P=24, O=32, BM=max(16, rows), B=128, LIMIT=10., SPINS=1000000,
                 A1=None, A2=None, NV4=False, TILE13=0, TILE2=0, SF6=False, W4A8=True)))
    report = {"scope": "offline SM121 compilation only", "gpu_used": False,
              "torch": torch.__version__, "triton": triton.__version__, "variants": [],
              "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                  ("engine/kernels/kda/tree.py", "engine/kernels/tile_dataflow.py", "engine/kernels/w4a8_pipeline.py")}}
    for name, fn, signature, constants in variants:
        print("compile " + name, flush=True)
        attrs = {(fn.arg_names.index(p),): [("tt.divisibility", 16)] for p in signature}
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
            target=GPUTarget("cuda", 121, 32),
            options=dict(num_warps=4, num_stages=1,
                         enable_fp_fusion=not (constants.get("W4A8", False) or fn in (_conv, _gate_up, _down))))
        ptx, cubin = kernel.asm["ptx"], kernel.asm["cubin"]
        (output / (name + ".ptx")).write_text(ptx)
        (output / (name + ".cubin")).write_bytes(cubin)
        if fn is _workers and ("gpu.acquire" not in ptx or "gpu.release" not in ptx or "nanosleep" not in ptx):
            raise AssertionError("dataflow publication/polling operations did not reach device code")
        if constants.get("NV4") and ("kind::mxf4nvf4" not in ptx or "e2m1.e2m1" not in ptx):
            raise AssertionError("NVFP4 binding must compile to native block-scaled FP4 MMA")
        if (constants.get("W4A8") or fn in (_gate_up, _down)) and ("e4m3.e4m3" not in ptx or "e2m1.e2m1" in ptx):
            raise AssertionError("W4A8 must retain FP8 MMA after on-chip W4 expansion")
        if fn in (_gate_up, _down) and any(op in ptx for op in ("atom.", "nanosleep", "ld.local", "st.local")):
            raise AssertionError("staged W4A8 must not introduce polling, atomics or register spills")
        report["variants"].append({"name": name, "constants": constants,
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(), "shared_bytes": kernel.metadata.shared})
    report["cuda_initialized"] = torch.cuda.is_initialized()
    if report["cuda_initialized"]:
        raise AssertionError("compilation initialized CUDA")
    (output / "compile.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_variants(args.output)
