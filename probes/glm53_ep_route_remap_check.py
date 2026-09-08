#!/usr/bin/env python3
"""Exact route-remap oracle and explicit-target no-device Triton compilation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

if __package__:
    from .glm53_ep_capsule_runtime import verify_runtime
else:
    from glm53_ep_capsule_runtime import verify_runtime


def compile_cases():
    cases = []
    for kind in ("mapped", "empty", "offset"):
        for ids, weight, mapping in itertools.product(
            ("i32", "i64"), ("fp32", "fp16", "bf16"),
            ("i32", "i64") if kind == "mapped" else ("i32",),
        ):
            cases.append(dict(
                label="-".join((kind, ids, weight, mapping)), kind=kind,
                signature=dict(IDS="*"+ids, WEIGHTS="*"+weight,
                               EXPERT_MAP="*"+mapping, OUT_IDS="*i32",
                               OUT_WEIGHTS="*"+weight, N_PAIRS="i32",
                               LOCAL_OFFSET="i32"),
                constants=dict(MAP_LEN=288 if kind == "mapped" else 0,
                               HAS_MAP=kind != "offset", BLOCK=256)))
    return cases


def compile_remap(output):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import glm53_ep_route_remap as helper
    results = []
    for case in compile_cases():
        kernel = triton.compile(
            ASTSource(helper._remap_ep_local_kernel, case["signature"],
                      constexprs=case["constants"]),
            target=GPUTarget("cuda", 121, 32), options={"num_warps": 4})
        folder = Path(output)/"remap"/case["label"]
        folder.mkdir(parents=True, exist_ok=False)
        ptx = kernel.asm["ptx"].encode()
        cubin = kernel.asm["cubin"]
        (folder/"kernel.ptx").write_bytes(ptx)
        (folder/"kernel.cubin").write_bytes(cubin)
        results.append(dict(
            **case, hash=kernel.hash, shared_bytes=kernel.metadata.shared,
            ptx_sha256=hashlib.sha256(ptx).hexdigest(),
            cubin_sha256=hashlib.sha256(cubin).hexdigest()))
    return results


def exact(actual, expected, label):
    import torch
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not torch.equal(actual.contiguous().view(torch.uint8),
                               expected.contiguous().view(torch.uint8))):
        raise AssertionError("remap bytes differ: "+label)


def verify_gpu(result):
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import glm53_ep_route_remap as helper
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import remap_b12x_ep_tensors
    assert torch.cuda.get_device_capability() == (12, 1)
    rows = 4097  # Non-multiple of BLOCK, exercising masked tail writes.
    id_pattern = torch.tensor(
        [-1, -2, 0, 5, 9, 11, 71, 72, 287, 288, 2**31, 2**32+5, -2**32+5],
        device="cuda", dtype=torch.int64)
    result["checks"] = []
    for case in compile_cases():
        label = case["label"]
        result["phase"] = label
        signature = case["signature"]
        ids = id_pattern.repeat((rows*8+12)//13)[:rows*8].reshape(rows, 8).to(
            torch.int64 if signature["IDS"] == "*i64" else torch.int32)
        weight_dtype = {"*fp32":torch.float32, "*fp16":torch.float16,
                        "*bf16":torch.bfloat16}[signature["WEIGHTS"]]
        # Construct raw storage without a float conversion that could quiet
        # a signaling NaN or discard its payload before either arm sees it.
        bits = {
            torch.float32: [0, 0x80000000, 0x7fc01234, 0x7fa12345, 0x3f800000, 0x7f800000],
            torch.float16: [0, 0x8000, 0x7e12, 0x7c01, 0x3c00, 0x7c00],
            torch.bfloat16: [0, 0x8000, 0x7fc1, 0x7f81, 0x3f80, 0x7f80],
        }[weight_dtype]
        width = 32 if weight_dtype == torch.float32 else 16
        signed = [v-(1 << width) if v >= (1 << (width-1)) else v for v in bits]
        storage = torch.tensor(signed, device="cuda",
                               dtype=torch.int32 if width == 32 else torch.int16)
        weights = storage.repeat((rows*8+5)//6)[:rows*8].reshape(rows, 8).view(weight_dtype)
        mapping = None
        if case["kind"] != "offset":
            dtype = torch.int64 if signature["EXPERT_MAP"] == "*i64" else torch.int32
            mapping = torch.arange(case["constants"]["MAP_LEN"], device="cuda", dtype=dtype)
            if mapping.numel():
                mapping[72:] = -1
                mapping[9], mapping[11] = 72, 73  # Preserve legacy map semantics.
        out_ids = torch.empty_like(ids, dtype=torch.int32)
        out_weights = torch.empty_like(weights)
        kwargs = dict(num_local_experts=72, local_expert_offset=72,
                      expert_map=mapping)
        # Reuse the same storage with changed IDs/map and poisoned outputs.
        for changed in (False, True):
            if changed:
                ids.copy_(ids.flip(0))
                if mapping is not None and mapping.numel():
                    mapping[:8].fill_(-1)
            expected = remap_b12x_ep_tensors(ids, weights, **kwargs)
            out_ids.fill_(-777)
            out_weights.fill_(float("nan"))
            if not helper.try_remap_ep_local(
                ids, weights, out_ids=out_ids, out_scales=out_weights, **kwargs):
                raise AssertionError("remap unexpectedly rejected "+label)
            exact(out_ids, expected[0], label+"-ids")
            exact(out_weights, expected[1], label+"-weights")
        result["checks"].append(dict(label=label, rows=rows, changed_storage=True))
    torch.cuda.synchronize()


def run_check(args, result):
    if __package__:
        from .glm53_ep_local_evidence import validate_compile_evidence
    else:
        from glm53_ep_local_evidence import validate_compile_evidence
    root = Path(__file__).resolve().parents[1]
    result["phase"] = "binding-runtime"
    runtime = verify_runtime(args.capsule_root, args.manifest_sha256)
    result["binding_runtime"] = runtime
    result["phase"] = "source-binding"
    evidence = validate_compile_evidence(root, args.compile_evidence)
    if evidence.get("binding_runtime") != runtime:
        raise ValueError("CPU compile and GPU binding runtimes differ")
    for target, want in evidence["mounted_sources"].items():
        assert hashlib.sha256(Path(target).read_bytes()).hexdigest() == want, target
    operation_error = None
    try:
        verify_gpu(result)
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if operation_error is None:
            result["phase"] = "binding-runtime-recheck"
        try:
            if verify_runtime(args.capsule_root, args.manifest_sha256) != runtime:
                raise RuntimeError("binding runtime changed during remap probe")
            result["binding_runtime_rechecked"] = True
        except BaseException as exc:
            result["binding_runtime_recheck_error"] = repr(exc)
            if operation_error is None:
                raise
    result.update(verdict="PASS", phase="complete", performance_acceptance=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile-evidence", type=Path, required=True)
    ap.add_argument("--capsule-root", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = dict(started=time.time(), phase="binding-runtime", verdict="FAIL",
                  performance_acceptance=False)
    try:
        run_check(args, result)
    except BaseException as exc:
        result.update(verdict="FAIL", error=repr(exc))
        raise
    finally:
        result["ended"] = time.time()
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
