"""Compile the production Gram shapes for SM121 with no CUDA device."""
import json
from pathlib import Path
import sys

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.kernels.dense.calibration_gram import _observe, _gram, _advance

rows = []
for m in (1, 7, 28):
    for fn, signature, constants in (
        (_observe, dict(X="*bf16", Mask="*i8", Buffer="*fp32", Cursor="*i32", Armed="*fp32", Count="*fp32", Peaks="*fp32"),
         dict(K=20480, M=m, SX=20480, MASKED=True, STAGE=True, BM=triton.next_power_of_2(m), BC=128)),
        (_gram, dict(Buffer="*fp32", H="*fp32", Cursor="*i32", Armed="*fp32"),
         dict(K=20480, M=m, CAPACITY=283, FORCE=False, PROGRAMS=96, THRESHOLD=256, B=32, R=32)),
        (_advance, dict(Cursor="*i32", Armed="*fp32"), dict(M=m, FORCE=False, THRESHOLD=256)),
    ):
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                target=GPUTarget("cuda", 121, 32), options={"num_warps": 4})
        rows.append(dict(kernel=fn.__name__, rows=m, shared=kernel.metadata.shared))
kernel = triton.compile(ASTSource(_gram, dict(Buffer="*fp32", H="*fp32", Cursor="*i32", Armed="*fp32"),
                                 constexprs=dict(K=20480, M=0, CAPACITY=283, FORCE=True,
                                                 PROGRAMS=96, THRESHOLD=256, B=32, R=32)),
                         target=GPUTarget("cuda", 121, 32), options={"num_warps": 4})
rows.append(dict(kernel="flush", rows=0, shared=kernel.metadata.shared))
from engine.kernels.glm_pointwise import _activation, _scores, _weights, _layernorm
for fn, sig, const in (
    (_activation, dict(G="*bf16", U="*bf16", O="*bf16"), dict(SG=2560, SU=2560, D=1280, LIMIT=10., B=512)),
    (_scores, dict(X="*fp32", Bias="*fp32", O="*fp32"), dict(E=288, B=512)),
    (_weights, dict(X="*fp32", Sel="*i64", Ids="*i32", W="*fp32"), dict(E=288, K=8, SCALE=2.5, B=8)),
    (_layernorm, dict(X="*bf16", W="*fp32", Bias="*fp32", O="*bf16"), dict(SX=256, D=128, EPS=1e-6, B=128)),
    (_observe, dict(X="*bf16", Armed="*fp32", Count="*fp32", Peaks="*fp32"),
     dict(Mask=None, Buffer=None, Cursor=None, K=95, M=7, SX=128, MASKED=False, STAGE=False, BM=8, BC=128)),
):
    kernel = triton.compile(ASTSource(fn, sig, constexprs=const), target=GPUTarget("cuda", 121, 32),
                            options={"num_warps": 4, "enable_fp_fusion": False})
    rows.append(dict(kernel=fn.__name__, shared=kernel.metadata.shared))
assert not torch.cuda.is_initialized()
for m in (1, 7, 28):
    kernel = triton.compile(ASTSource(_gram, dict(Buffer="*bf16", H="*fp32", Cursor="*i32", Armed="*fp32"),
        constexprs=dict(K=20480, M=m, CAPACITY=283, FORCE=False, PROGRAMS=96, THRESHOLD=256, B=32, R=32)),
        target=GPUTarget("cuda", 121, 32), options={"num_warps": 4})
    rows.append(dict(kernel="bf16_gram", rows=m, shared=kernel.metadata.shared))
kernel = triton.compile(ASTSource(_observe,
    dict(X="*bf16", Mask="*i8", Buffer="*bf16", Cursor="*i32", Armed="*fp32", Count="*fp32", Peaks="*fp32"),
    constexprs=dict(K=20480, M=7, SX=20480, MASKED=True, STAGE=True, BM=8, BC=128)),
    target=GPUTarget("cuda", 121, 32), options={"num_warps": 4, "enable_fp_fusion": False})
rows.append(dict(kernel="bf16_observe", rows=7, shared=kernel.metadata.shared))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status="PASS", gpu_used=False, variants=rows)))
