#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Register-Q MLA: compile-only, numerics/replay, and paired kernel timing.

Run with probes/run_mk_probe.sh in the fleet lane. --compile-only is CPU-only
and requires MK_PROBE_NO_GPU=1. --sanitize omits large timing cases; invoke it
under compute-sanitizer. Kernel timing is diagnostic, not a serving verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--sanitize", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))
    import torch
    import torch.utils.cpp_extension as ce

    root = Path(__file__).resolve().parents[1]
    src = root / "overlay/modules/glm53_megakernel/glm53_megakernel.py"
    spec = importlib.util.spec_from_file_location("mk_prefill32", src)
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    print("source_sha256", hashlib.sha256(src.with_suffix(".cu").read_bytes()).hexdigest(), flush=True)
    if args.compile_only:
        assert os.environ.get("MK_PROBE_NO_GPU") == "1", "compile-only requires a container without GPUs"
        # The architecture is explicit in _build; no device query/context is
        # needed. Verbose/resource flags add diagnostics, not different math.
        original_load = ce.load
        def resource_load(**kw):
            kw["extra_cuda_cflags"] += ["-Xptxas=-v", "-Xptxas=--warn-on-spills"]
            kw["verbose"] = True
            return original_load(**kw)
        ce.load = resource_load
        mk._build()
        assert not torch.cuda.is_initialized(), "compile check initialized CUDA"
        print("COMPILE PASS; GPU numerics/performance not tested", flush=True)
        return

    assert torch.cuda.get_device_capability() == (12, 1)
    mk._build()
    torch.manual_seed(917)
    D, H, NS = 512, 16, (8192 if args.sanitize else 300000)
    cache = (torch.randn(NS, D, device="cuda") * .4).to(torch.float8_e4m3fn)
    cases = [(128, 1), (129, 17), (131, 31), (128, 32), (129, 33),
             (128, 2048), (131, 2176)]
    if not args.sanitize:
        cases += [(2593, 2048), (4143, 2048), (6912, 2048), (8192, 2048)]
    rows = []
    for ci, (T, W) in enumerate(cases):
        q = (torch.randn(T, H, D, device="cuda") * (.3, 1., 3.)[ci % 3]).to(torch.bfloat16)
        slots = torch.randint(0, NS, (T, W), dtype=torch.int32, device="cuda")
        lens = torch.randint(0, W + 1, (T,), dtype=torch.int32, device="cuda")
        lens[0] = 0; slots[0].fill_(-1)
        lens[1] = W; slots[1].fill_(0)  # duplicate keys are repeated softmax terms
        lens[2] = W
        # Real prefill is full-length except the early causal rows. Retain
        # the edge fixtures above, then time mostly full W for large cases.
        if T > 131:
            lens[3:].fill_(W)
        sm, scale = D ** -.5, (.125, .7, 2.)[ci % 3]
        baseline, candidate = torch.empty_like(q), torch.empty_like(q)
        mk.ENABLE_MLA_PREFILL32 = False
        mk.ENABLE_MLA_PREFILL_PAIR = False
        def base():
            return mk.mla_decode(q, cache.view(torch.uint8), slots, lens, sm, scale, baseline)
        def cand():
            return mk._mla_prefill32(q, cache.view(torch.uint8), slots, lens, sm, scale, candidate)
        base(); cand()
        sub = min(T, 131)
        ref = mk.mla_decode_ref(q[:sub], cache, slots[:sub], lens[:sub], sm, scale)
        torch.cuda.synchronize()
        rel = mk._rel_err(candidate[:sub].float(), ref.float())
        base_rel = mk._rel_err(baseline[:sub].float(), ref.float())
        full_delta = mk._rel_err(candidate.float(), baseline.float())
        # A global norm can hide one broken row in a large chunk. Check
        # every row as well, including all rows outside the FP32 subsample.
        def worst_row(got, expected):
            g, r = got.float().flatten(1), expected.float().flatten(1)
            return ((g - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-6)).max().item()
        max_row_ref = worst_row(candidate[:sub], ref)
        max_row_delta = worst_row(candidate, baseline)
        assert torch.isfinite(candidate).all().item()
        assert torch.count_nonzero(candidate[0]).item() == 0
        assert rel <= .02 and full_delta <= .02, (T, W, rel, full_delta)
        assert max_row_ref <= .02 and max_row_delta <= .03, (T, W, max_row_ref, max_row_delta)
        snapshot = candidate.clone()
        cand(); torch.cuda.synchronize()
        assert torch.equal(snapshot, candidate), "same-input replay changed output"
        # Graph pointers stay fixed while device lengths and slots change.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cand()
        lens[2] = min(W, 7)
        slots[2].fill_(NS - 1)
        graph.replay(); torch.cuda.synchronize()
        graph_result = candidate.clone()
        cand(); torch.cuda.synchronize()
        assert torch.equal(graph_result, candidate), "graph ignored changed device input"
        ref2 = mk.mla_decode_ref(q[2:3], cache, slots[2:3], lens[2:3], sm, scale)
        assert mk._rel_err(candidate[2:3].float(), ref2.float()) <= .02
        result = dict(T=T, W=W, rel=rel, base_rel=base_rel, full_delta=full_delta,
                      max_row_ref=max_row_ref, max_row_delta=max_row_delta,
                      replay=True, graph_changed_inputs=True)
        if not args.sanitize:
            times = {"base": [], "candidate": []}
            for fn in (base, cand):
                for _ in range(2): fn()
            torch.cuda.synchronize()
            for order in (("base", base), ("candidate", cand),
                          ("candidate", cand), ("base", base)) * 3:
                name, fn = order
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(3): fn()
                b.record(); b.synchronize()
                times[name].append(a.elapsed_time(b) / 3)
            result["timing_ms"] = times
            result["kernel_speedup"] = statistics.median(times["base"]) / statistics.median(times["candidate"])
        rows.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"verdict": "PASS", "cases": len(rows), "serving_speedup": None}), flush=True)


if __name__ == "__main__":
    main()
