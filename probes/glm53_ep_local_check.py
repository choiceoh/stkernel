#!/usr/bin/env python3
"""Same-image E72 full-token versus the actual legacy compact EP method.

One case per fresh container. This is eager component validation, not graph,
TP4 transport, model quality, or serving TTFT acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

if __package__:
    from .glm53_ep_local_evidence import validate_compile_evidence
    from .glm53_ep_capsule_runtime import verify_runtime
    from .glm53_ep_numerics_diagnostics import capture_failure
else:
    from glm53_ep_local_evidence import validate_compile_evidence
    from glm53_ep_capsule_runtime import verify_runtime
    from glm53_ep_numerics_diagnostics import capture_failure

CASES = {
    "balanced4096": (4096, "balanced"),
    "balanced6912": (6912, "balanced"),
    "balanced8192": (8192, "balanced"),
    "concentrated6912": (6912, "concentrated"),
    "remote4096": (4096, "remote"),
    "duplicate4096": (4096, "duplicate"),
    "zeros4097": (4097, "zeros"),
    "balanced16384": (16384, "balanced"),
}


ROW_L2_FLOOR = .02
ROW_PEAK_FLOOR = .04


def row_errors(actual, reference):
    import torch
    assert actual.shape == reference.shape and actual.ndim == 2, "output shape mismatch"
    a, b = actual.float(), reference.float()
    assert all(bool(torch.isfinite(t).all()) for t in (a, b)), "nonfinite output"
    delta = a - b
    return (delta.norm(dim=1) / b.norm(dim=1).clamp_min(1e-6),
            delta.abs().amax(dim=1) / b.abs().amax(dim=1).clamp_min(1e-6))


def check_control(baseline, repeat):
    """Stock variability itself must stay within the fixed numerical floors."""
    l2, peak = row_errors(repeat, baseline)
    bad = (l2 > ROW_L2_FLOOR) | (peak > ROW_PEAK_FLOOR)
    result = dict(bad_rows=int(bad.sum()), max_row_relative_l2=float(l2.max()),
                  max_row_relative_abs=float(peak.max()))
    assert not result["bad_rows"], dict(verdict="UNSTABLE_STOCK_CONTROL", **result)
    return result


def compare(candidate, baseline, repeat, *, failure_context=None):
    import torch
    # A noisy reference must never authorize arbitrarily noisy candidates.
    check_control(baseline, repeat)
    error, max_error = row_errors(candidate, baseline)
    noise, max_noise = row_errors(repeat, baseline)
    bad = ((error > torch.maximum(3*noise, torch.full_like(noise, ROW_L2_FLOOR)))
           | (max_error > torch.maximum(3*max_noise, torch.full_like(max_noise, ROW_PEAK_FLOOR))))
    result = dict(bad_rows=int(bad.sum()), max_row_relative_l2=float(error.max()),
                  max_row_relative_abs=float(max_error.max()),
                  stock_max_row_relative_l2=float(noise.max()),
                  stock_max_row_relative_abs=float(max_noise.max()))
    if result["bad_rows"] and failure_context is not None:
        # The original failure is permanent even if diagnostic copying fails.
        # No host tensor copies or additional comparisons occur on success.
        sink = failure_context["result"]
        sink["verdict"] = "FAIL"
        if "candidate_first_failure" not in sink:
            sink["candidate_first_failure"] = dict(
                verdict="CANDIDATE_NUMERICS_FAIL", phase=sink.get("phase"), **result)
            try:
                sink["candidate_failure_diagnostics"] = capture_failure(
                    candidate, baseline, repeat, failure_context["third"],
                    bad=bad, error=error, peak=max_error, noise=noise, peak_noise=max_noise,
                    l2_limits=torch.maximum(3*noise, torch.full_like(noise, ROW_L2_FLOOR)),
                    peak_limits=torch.maximum(3*max_noise, torch.full_like(max_noise, ROW_PEAK_FLOOR)),
                    total_bad_rows=result["bad_rows"],
                    route_ids=failure_context["route_ids"], route_weights=failure_context["route_weights"],
                    expert_map=failure_context["expert_map"], scales=failure_context["scales"],
                    inputs=failure_context["inputs"])
            except BaseException as exc:
                sink["candidate_failure_diagnostics_error"] = repr(exc)
    assert not result["bad_rows"], dict(verdict="CANDIDATE_NUMERICS_FAIL", **result)
    return result


def expert_set():
    import torch
    from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
    e, h, n = 72, 4096, 2048
    gen = torch.Generator().manual_seed(905308)
    w13 = torch.randint(0, 256, (e, 2*n, h//2), dtype=torch.uint8, generator=gen).cuda()
    w2 = torch.randint(0, 256, (e, h, n//2), dtype=torch.uint8, generator=gen).cuda()
    s13 = (torch.rand(e, 2*n, h//16, generator=gen)*.05+.01).to(torch.float8_e4m3fn).cuda()
    s2 = (torch.rand(e, h, n//16, generator=gen)*.05+.01).to(torch.float8_e4m3fn).cuda()
    sf13 = flashinfer_convert_sf_to_mma_layout(
        s13.reshape(e*2*n, h//16), m=2*n, k=h, num_groups=e)
    sf2 = flashinfer_convert_sf_to_mma_layout(
        s2.reshape(e*h, n//16), m=h, k=n, num_groups=e)
    return w13, sf13, w2, sf2


def routing(rows, kind, changed=False):
    import torch
    token = torch.arange(rows, device="cuda")[:, None]
    slot = torch.arange(8, device="cuda")[None, :]
    if kind == "remote" and not changed:
        global_ids = (72 + (token + slot) % 216).expand(rows, 8)
    elif kind == "concentrated":
        global_ids = ((slot + (3 if changed else 0)) % 72).expand(rows, 8)
    elif kind == "duplicate":
        global_ids = torch.full((rows, 8), 6 if changed else 5, device="cuda")
    else:
        global_ids = (token*37 + slot*31 + (137 if changed else 0)) % 288
    weights = torch.rand(rows, 8, device="cuda")
    weights /= weights.sum(dim=1, keepdim=True)
    ids = global_ids.to(torch.int32).contiguous()
    if kind == "zeros":
        weights[:, ::2] = 0
    return ids, weights


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--compile-evidence", type=Path, required=True)
    ap.add_argument("--capsule-root", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def run_case(args, result):
    # Bind the imported package and CPU receipt before importing accelerator
    # consumers: even a device-capability query is later than this admission.
    result["phase"] = "binding-runtime"
    runtime = verify_runtime(args.capsule_root, args.manifest_sha256)
    result["binding_runtime"] = runtime
    result["phase"] = "source-binding"
    compiled = validate_compile_evidence(Path("/repo"), args.compile_evidence)
    if compiled.get("binding_runtime") != runtime:
        raise ValueError("CPU compile and GPU binding runtimes differ")
    for filename, want in compiled["sources"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == want, filename
    provenance = {}
    for line in Path("/repo/build/glm53/manifest.tsv").read_text().splitlines():
        name, target, *_ = line.split("\t")
        if "/flashinfer/" in target or name == "flashinfer_b12x_moe.py":
            want = hashlib.sha256((Path("/repo/build/glm53")/name).read_bytes()).hexdigest()
            have = hashlib.sha256(Path(target).read_bytes()).hexdigest()
            assert have == want, (name, have, want)
            provenance[name] = have
    operation_error = None
    try:
        result["phase"] = "prepare"
        _run_case(args, result, provenance)
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if operation_error is None:
            result["phase"] = "binding-runtime-recheck"
        try:
            if verify_runtime(args.capsule_root, args.manifest_sha256) != runtime:
                raise RuntimeError("binding runtime changed during GPU probe")
            result["binding_runtime_rechecked"] = True
        except BaseException as exc:
            result["binding_runtime_recheck_error"] = repr(exc)
            if operation_error is None:
                raise
    result.update(verdict="PASS", phase="complete")
    print("EP_LOCAL_GPU PASS; full-model TTFT and TP4 remain separate gates", flush=True)


def _run_case(args, result, provenance):
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import FlashInferB12xExperts
    assert torch.cuda.get_device_capability() == (12, 1)
    assert md._GLM53_EP_PREFILL_LOCAL
    torch.manual_seed(905308)
    w13, sf13, w2, sf2 = expert_set()
    # Deliberately different expert scales exercise the generic scale branch.
    scales = torch.linspace(.8, 1.2, 72, device="cuda")
    wrapper = SimpleNamespace(
        num_local_experts=72, _kernel_num_experts=72, max_num_tokens=8192,
        _activation_str="swigluoai_uninterleave", _swiglu_alpha=1.,
        _swiglu_beta=0., _swiglu_limit=10., w1_sf_mma=sf13, w2_sf_mma=sf2,
        g1_alphas=scales, g2_alphas=scales.flip(0).contiguous(),
        _fc2_input_scale=scales)
    rows, kind = CASES[args.case]
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)*.5
    ids, weights = routing(rows, kind)
    expert_map = torch.arange(288, dtype=torch.int32, device="cuda")
    expert_map[72:] = -1
    # Keep the legacy pair-slice limit at 8192, as before. Remap scratch is
    # sized to this fixture, including the synthetic 16384-token boundary.
    for name, dtype in (
        ("_ep_ids", torch.int32), ("_ep_scales", weights.dtype),
        ("_ep_long", torch.int64), ("_ep_mapped", expert_map.dtype),
        ("_ep_remote", torch.bool), ("_ep_tmp_a", torch.bool), ("_ep_tmp_b", torch.bool),
    ):
        setattr(wrapper, name, torch.empty((rows, 8), device="cuda", dtype=dtype))
    wrapper.local_expert_offset = 0
    out = torch.empty_like(x)
    side = torch.cuda.Stream()

    def call(candidate):
        mapped_ids, mapped_weights = FlashInferB12xExperts._remap_ep_tensors(
            wrapper, ids, weights, expert_map, fuse_local_prefill=bool(candidate))
        method = (FlashInferB12xExperts._apply_ep_local_prefill if candidate
                  else FlashInferB12xExperts._apply_ep_compact)
        got = method(wrapper, out, x, w13, w2, mapped_ids, mapped_weights)
        assert got is out

    def eager(candidate, nondefault=False):
        out.fill_(float("nan"))
        if nondefault:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                call(candidate)
            torch.cuda.current_stream().wait_stream(side)
        else:
            call(candidate)
        torch.cuda.synchronize()
        return out.clone()

    result.update(rows=rows, routing=kind, legacy_max_num_tokens=wrapper.max_num_tokens,
                  remap_capacity_tokens=rows, timing_scope="EP remap plus MoE wrapper",
                  provenance=provenance, controls=[], candidate=[], timing={})
    # Initial, changed input/routes at the same addresses, then poisoned-output
    # nondefault-stream replay. Every arm uses the same original tensors.
    for changed in (False, True):
        if changed:
            x.mul_(-.75)
            # The per-CTA scale cache must be rebuilt on every launch.
            old_scale_ptr = scales.data_ptr()
            scales.mul_(1.125)
            assert scales.data_ptr() == old_scale_ptr
            new_ids, new_weights = routing(rows, kind, changed=True)
            ids.copy_(new_ids); weights.copy_(new_weights)
        result["phase"] = "changed-control" if changed else "initial-control"
        b, b2, b3 = eager(False), eager(False), eager(False)
        result["controls"].append([check_control(b, b2), check_control(b, b3), check_control(b2, b3)])
        result["phase"] = "changed-candidate" if changed else "initial-candidate"
        failure_context = dict(result=result, third=b3, inputs=x, route_ids=ids, route_weights=weights,
                               expert_map=expert_map, scales=dict(
                                   fc1_input=wrapper.g1_alphas, fc1_alpha=wrapper.g1_alphas,
                                   fc2_input=wrapper._fc2_input_scale, fc2_alpha=wrapper.g2_alphas))
        result["candidate"].append(compare(eager(True), b, b2, failure_context=failure_context))
        result["phase"] += "-nondefault-stream"
        result["candidate"].append(compare(eager(True, nondefault=True), b, b2,
                                           failure_context=failure_context))
        if kind == "remote" and not changed:
            assert bool((b == 0).all()), "empty-local control must be exactly zero"
            assert bool((out == 0).all()), "empty-local candidate must be exactly zero"
    keys = list(md._DYNAMIC_KERNEL_CACHE)
    assert any(key[-1] == "glm53_ep_prefill_local_fp32_v2" for key in keys), keys
    result["cache_keys"] = keys
    if not args.sanitize and kind in ("balanced", "concentrated"):
        result["phase"] = "paired-timing"
        wall, device = [[], []], [[], []]
        for iteration in range(8):
            for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                call(bool(arm)); torch.cuda.synchronize()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                t0 = time.perf_counter()
                start.record()
                for _ in range(3):
                    call(bool(arm))
                end.record(); end.synchronize()
                wall[arm].append((time.perf_counter()-t0)*1000/3)
                device[arm].append(start.elapsed_time(end)/3)
        result["timing"] = dict(
            wall_ms=dict(compact=wall[0], local=wall[1]),
            device_ms=dict(compact=device[0], local=device[1]),
            wall_speedup_pct=100*(statistics.median(wall[0])/statistics.median(wall[1])-1))
    result.update(device=torch.cuda.get_device_name(),
                  max_allocated_bytes=torch.cuda.max_memory_allocated(),
                  max_reserved_bytes=torch.cuda.max_memory_reserved(),
                  performance_acceptance=False)


def main():
    args = parse_args()
    os.environ.update(
        VLLM_GLM53_EP_PREFILL_LOCAL="1",
        VLLM_GLM53_B12X_PREFILL_REUSE="0",
        VLLM_GLM53_B12X_PREFILL_FC1_N128="0",
        VLLM_GLM53_B12X_STATIC_V2="t,r",
    )

    result = dict(case=args.case, sanitize=args.sanitize, verdict="RUNNING",
                  phase="prepare", performance_acceptance=False)
    try:
        run_case(args, result)
    except BaseException as exc:
        result.update(verdict="FAIL", error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, default=str)+"\n")
        print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
