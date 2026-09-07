#!/usr/bin/env python3
"""Same-build MoE prefill candidate/control, graph replay and paired timing.

The control is the served tiled stock dynamic kernel, using the same packed
weights, scales, workspace and inputs. Every output row is checked against
stock and its repeated-run BF16 atomic-scatter noise. This probe cannot prove
end-to-end prefill/TTFT or text quality; a serving bracket is still required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.update(VLLM_GLM53_B12X_STATIC_V2="t", VLLM_GLM53_B12X_PREFILL_REUSE="0",
                  VLLM_GLM53_B12X_PREFILL_FC1_N128="0",
                  VLLM_GLM53_B12X_PREFILL_STREAM_FC2="1")
sys.path.insert(0, "/repo/probes")


def compare(candidate, baseline, repeat):
    import torch
    a, b, r = (t.float() for t in (candidate, baseline, repeat))
    assert all(bool(torch.isfinite(t).all()) for t in (a, b, r)), "nonfinite output"
    # Use each row's own magnitude; a large row cannot hide corruption of
    # another row. The floor covers BF16 atomic addition reordering.
    norm = b.norm(dim=1).clamp_min(1e-6)
    error = (a - b).norm(dim=1) / norm
    noise = (r - b).norm(dim=1) / norm
    limits = torch.maximum(3 * noise, torch.full_like(noise, .02))
    bad = error > limits
    peak = b.abs().amax(dim=1).clamp_min(1e-6)
    max_error = (a - b).abs().amax(dim=1) / peak
    max_noise = (r - b).abs().amax(dim=1) / peak
    bad |= max_error > torch.maximum(3 * max_noise, torch.full_like(max_noise, .04))
    result = dict(bad_rows=int(bad.sum()), max_row_relative_l2=float(error.max()),
                  max_row_relative_abs=float(max_error.max()),
                  stock_max_row_relative_l2=float(noise.max()))
    assert not result["bad_rows"], result
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from b12x_static_probe import expert_set
    assert torch.cuda.get_device_capability() == (12, 1)
    md._STATIC_V2_OVERRIDE = md._parse_glm53_static_v2("t", probe=True)
    provenance = {}
    for line in Path("/repo/build/glm53/manifest.tsv").read_text().splitlines():
        name, target, *_ = line.split("\t")
        if "/flashinfer/" in target:
            want = hashlib.sha256((Path("/repo/build/glm53") / name).read_bytes()).hexdigest()
            have = hashlib.sha256(Path(target).read_bytes()).hexdigest()
            assert want == have, (name, want, have)
            provenance[name] = have
    assert Path(md.__file__).resolve() == Path(next(
        l.split("\t")[1] for l in Path("/repo/build/glm53/manifest.tsv").read_text().splitlines()
        if l.startswith("moe_dispatch.py\t"))).resolve()
    torch.manual_seed(905307)
    weights = expert_set(torch.Generator().manual_seed(905307))
    w13, sf13, w2, sf2 = weights
    md.tile_expert_weights_inplace(w13, w2)
    wrapper = B12xMoEWrapper(
        num_experts=288, top_k=8, hidden_size=4096, intermediate_size=512,
        use_cuda_graph=True, max_num_tokens=8192, num_local_experts=288,
        activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0.,
        swiglu_limit=10.)
    ones = torch.ones(288, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    rows = []
    cases = [(4096, "balanced"), (6912, "skew")]
    if not args.sanitize:
        cases += [(6912, "balanced"), (8192, "balanced"), (8192, "skew"), (2593, "balanced")]
    for m, routing in cases:
        x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16) * .5
        ids = ((torch.arange(m * 8, device="cuda").reshape(m, 8) %
                (288 if routing == "balanced" else 8))).to(torch.int32)
        scales = torch.rand(m, 8, device="cuda")
        scales /= scales.sum(dim=1, keepdim=True)
        out = torch.empty_like(x)

        def call(candidate):
            md._GLM53_B12X_PREFILL_STREAM_FC2 = candidate
            wrapper.run(x, w13, sf13, w2, sf2, ids, scales,
                        w1_alpha=ones, w2_alpha=ones, fc2_input_scale=ones, out=out)

        def eager(candidate):
            out.fill_(float("nan"))
            call(candidate)
            torch.cuda.synchronize()
            return out.clone()

        b, b2, a = eager(False), eager(False), eager(True)
        row = dict(m=m, routing=routing, eager=compare(a, b, b2))
        for candidate in (False, True):
            # This is a probe assertion, not a production override: selected
            # cache keys prove short chunks declined and large chunks engaged.
            md._DYNAMIC_KERNEL_CACHE.clear()
            call(candidate)
            assert len(md._DYNAMIC_KERNEL_CACHE) == 1
            key = next(iter(md._DYNAMIC_KERNEL_CACHE))
            assert (key[-1] == "glm53_prefill_stream_fc2_v1") == (candidate and m >= 4096), key
        graphs = []
        for candidate in (False, True):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call(candidate)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call(candidate)
            graphs.append(graph)
        # Change both activations and routes at the captured addresses.
        # Keep the wrapper, tensors, compiled kernels and graphs alive together.
        x.mul_(-.75)
        ids.add_(137).remainder_(288)
        b, b2 = eager(False), eager(False)
        for label, graph in zip(("stock_graph", "candidate_graph"), graphs):
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            row[label] = compare(out, b, b2)
        if not args.sanitize and m >= 4096:
            timing = [[], []]
            for iteration in range(8):
                for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    for _ in range(2):
                        graphs[arm].replay()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(5):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    timing[arm].append(start.elapsed_time(end) / 5)
            row["timing_ms"] = dict(stock=timing[0], stream=timing[1])
            row["median_speedup_pct"] = 100 * (statistics.median(timing[0]) /
                                               statistics.median(timing[1]) - 1)
        print(json.dumps(row), flush=True)
        rows.append(row)
        del graphs, graph
    result = dict(device=torch.cuda.get_device_name(), sanitize=args.sanitize,
                  provenance=provenance, rows=rows, verdict="PASS")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("GPU_NUMERICS PASS; serving prefill/TTFT remains a separate gate", flush=True)


if __name__ == "__main__":
    main()
