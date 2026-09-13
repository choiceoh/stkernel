"""Compile both KDA state policies and deferred commit for SM121 without a GPU."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from engine.kernels.kda.deferred import _commit, _commit_layers
from engine.kernels.kda.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--commit-only", action="store_true")
    mode.add_argument("--compact-only", action="store_true")
    mode.add_argument("--serving-only", action="store_true",
                      help="compact recurrence/commit plus the real cache's boundary publication kernel")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = []
    for t in (1, 6, 7, 8):
        bv = 16 if t <= 6 else 8
        for deferred in (False, True):
            signature = {p:"*bf16" for p in ("q", "k", "v", "g", "beta", "o")}
            signature.update({p:"*fp32" for p in ("h0", "ht", "a_log", "g_bias")})
            signature.update(N="i64", T="i64", ring_slot="*i64", ring_context="*i64")
            constants = dict(cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
                             scale=128**-.5, B=1, H=16, HV=16, K=128, V=128, BK=128, BV=bv,
                             stride_init_state_token=16*128*128, stride_final_state_token=16*128*128,
                             stride_indices_seq=1, stride_indices_tok=1,
                             USE_INITIAL_STATE=True, INPLACE_FINAL_STATE=False, IS_BETA_HEADWISE=False,
                             USE_QK_L2NORM_IN_KERNEL=True, IS_VARLEN=False, IS_CONTINUOUS_BATCHING=False,
                             IS_SPEC_DECODING=False, IS_KDA=True, SIGMOID_BETA=True, COMPUTE_GATE=True,
                             SAFE_GATE=True, LOWER_BOUND=-5., STATE_KV=True, INPUT_STRIDES=None,
                             RING_SIZE=8, RING_SLOT_STRIDE=8*16*128*128, RING_DEVICE_INDICES=True,
                             RING_INDEX_STRIDE=0, DEFERRED_STATE=deferred)
            for p in ("deferred_keys", "deferred_decay", "deferred_updates"):
                if deferred:
                    signature[p] = "*fp32"
                else:
                    constants[p] = None
            variants.append((f"verify-t{t}-deferred{int(deferred)}",
                             fused_recurrent_gated_delta_rule_fwd_kernel.fn, signature, constants))
        variants.append((f"commit-t{t}", _commit,
                         {**{p:"*fp32" for p in ("KEY", "DECAY", "UPDATE", "RING")},
                          **{p:"*i64" for p in ("SLOT", "CONTEXT", "COUNT")}},
                         dict(T=t, H=16, K=128, V=128, R=8, SLOT_STRIDE=8*16*128*128,
                              BK=128, BV=bv, BLOCK=768)))
    # Same row addressing and strided projection views used by C=1/C=4.
    for rows in (1, 4):
        for t in (1, 7, 8):
            for tiled, hoist, cells in ((False, False, 1024), (True, False, 1024),
                                       (True, True, 1024), (True, True, 2048), (True, True, 4096)):
                width = max(7, t)
                variants.append((f"batch-commit-c{rows}-t{t}-tiled{int(tiled)}-hoist{int(hoist)}-b{cells}", _commit_layers,
                             {**{p: "*fp32" for p in ("KEY", "DECAY", "UPDATE", "RING")},
                              **{p: "*i64" for p in ("OFFSETS", "SLOT", "CONTEXT", "COUNT")}},
                             dict(T=t, ROWS=rows, H=16, K=128, V=128, R=width,
                                  SLOT_STRIDE=34*(width*16*128*128+64)+64, BLOCK=768, B=cells,
                                  TILED=tiled, HOIST_FINAL=hoist, OFFSET_ALIGNMENT=4 if hoist else 1,
                                  BOUNDARY_OFFSETS=None, COMPACT=False)))
                if tiled and hoist and cells == 1024:
                    name, fn, signature, constants = variants[-1]
                    variants.append((name+"-scalar", fn, signature, {**constants, "OFFSET_ALIGNMENT": 1}))
                if tiled and hoist and cells == 4096:
                    name, fn, signature, constants = variants[-1]
                    variants.append((name+"-w8", fn, signature, constants))
    # Compile the real batched verifier, including per-row factor offsets.
    _, fn, signature, constants = next(v for v in variants if v[0] == "verify-t7-deferred1")
    signature, constants = dict(signature), dict(constants)
    constants.update(RING_INDEX_STRIDE=1, RING_SIZE=7,
                     INPUT_STRIDES=((6144, 128, 1),)*3+((2048, 128, 1), (6416, 1)))
    variants.append(("verify-batched-strided", fn, signature, constants))
    # The compact ABI changes physical addressing while retaining the exact
    # recurrence tile. Compile both layouts without initializing CUDA.
    for t in (1, 8):
        _, fn, signature, constants = next(v for v in variants if v[0] == f"verify-t{t}-deferred1")
        for compact in (False, True):
            physical = 2 if compact else 8
            variants.append((f"compact-verify-t{t}-abi{int(compact)}", fn, dict(signature),
                             {**constants, "RING_SIZE": 1 if compact else 8,
                              "RING_SLOT_STRIDE": 34*(physical*16*128*128+64)+64,
                              "RING_INDEX_STRIDE": 1,
                              "INPUT_STRIDES": ((6144, 384, 1),)*3+((2048, 128, 1), (48, 1))}))
    for rows in (1, 4):
        _, fn, signature, constants = next(v for v in variants
            if v[0] == f"batch-commit-c{rows}-t8-tiled1-hoist1-b2048")
        variants.append((f"compact-commit-c{rows}-abi0", fn, dict(signature), dict(constants)))
        signature, constants = dict(signature), dict(constants)
        signature.update(BOUNDARY_OFFSETS="*i64")
        del constants["BOUNDARY_OFFSETS"]
        constants.update(COMPACT=True, R=1, SLOT_STRIDE=34*(2*16*128*128+64)+64)
        variants.append((f"compact-commit-c{rows}-abi1", fn, signature, constants))
    if args.serving_only:
        from dataclasses import replace
        from engine.kernels.state import _stage_conv
        from engine.profiles.glm53.caches import layout, stage_layout
        from tests.test_engine_glm53 import tiny_facts
        for compact in (False, True):
            f = replace(tiny_facts(), layers=45, kinds=("kda",)*34+("dsa",)*11,
                        spec_k=7, block=768, kda_heads=64, kda_dim=128,
                        kda_state_layout="committed_boundary" if compact else "ring")
            slot_bytes = layout(f, range(45)).slot_bytes
            signature = {p: "*bf16" for p in ("RING", "STAGE")}
            signature.update({p: "*i64" for p in ("ROFF", "SOFF", "SLOT", "BEFORE", "COUNT")})
            constants = dict(BLOCK_TOKENS=f.block, WIDTH=f.conv-1+f.spec_k, TAPS=f.conv-1,
                             CHANNELS=3*f.kda_heads_local*f.kda_dim, RS=slot_bytes//2,
                             SS=stage_layout(f, range(45))[0]//2, BLOCK=256,
                             MS=slot_bytes//8 if compact else 0, COMPACT=compact)
            if compact:
                signature["META"] = "*i64"
            else:
                constants["META"] = None
            variants.append((f"compact-stage-abi{int(compact)}", _stage_conv, signature, constants))
    if args.commit_only:
        variants = [v for v in variants if v[1] is _commit_layers]
    elif args.compact_only or args.serving_only:
        variants = [v for v in variants if v[0].startswith("compact-")]
    root = Path(__file__).resolve().parents[1]
    files = ("engine/kernels/kda/deferred.py", "engine/kernels/kda/ring.py",
             "engine/kernels/kda/fused_recurrent.py", "probes/engine_kda_deferred_compile.py",
             "engine/kernels/state.py", "engine/profiles/glm53/caches.py")
    report = dict(scope="SM121 compilation only", gpu_used=False, torch=torch.__version__,
                  triton=triton.__version__, variants=[],
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    for name, fn, signature, constants in variants:
        print("compile " + name, flush=True)
        # Match the pointer specialization of the actual Batch allocation.
        # Without this, offline compilation hides vector loads/stores that
        # runtime JIT can emit, making register/layout comparisons misleading.
        attrs = {(fn.arg_names.index(p),): [("tt.divisibility", 16)] for p in signature}
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants,
                                         attrs=attrs if fn is _commit_layers else None),
                                target=GPUTarget("cuda", 121, 32),
                                options=dict(num_warps=8 if name.endswith("-w8") else 4, num_stages=1)
                                if fn is _commit_layers else (
                                    dict(num_warps=4, num_stages=3) if name.startswith("compact-stage-")
                                    else dict(num_warps=1, num_stages=3)))
        (args.output/(name+".ptx")).write_text(kernel.asm["ptx"])
        (args.output/(name+".ttgir")).write_text(kernel.asm["ttgir"])
        cubin = args.output/(name+".cubin")
        cubin.write_bytes(kernel.asm["cubin"])
        resources = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(cubin)],
                                   check=True, text=True, capture_output=True).stdout
        report["variants"].append(dict(name=name, shared_bytes=kernel.metadata.shared,
                                       resources=resources,
                                       cubin_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest()))
    assert not torch.cuda.is_initialized()
    report["status"] = "PASS"
    (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
