#!/usr/bin/env python3
"""KDA pure-prefill output-copy removal: correctness and paired GPU timing.

Run only in an idle fleet window, in a fresh container with composed sources:
  bash probes/run_mk_probe.sh probes/kda_prefill_direct_out.py | tee /tmp/kda-direct.log

Both arms include the same KDA computation. Stock copies its result into the
layer buffer; direct writes there. Eager timing excludes input reset; graph
timing includes the same reset in both arms only when v is contiguous. Neither includes gated RMSNorm,
projections, state gather/scatter, or a model forward: these are kernel-chain
measurements, not end-to-end prefill throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics

import torch
import triton

from kda_prefill_bench import H, K, V, build_pkg, cfg_repr, collect_autotuners, make_inputs


def sequence_cases(lengths):
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must be positive")
    return [(length,) for length in lengths] + [(1, 63, 64, 65, 127)]


def paired_summary(samples):
    """Each row is A,B,B,A; report per-round paired median ratios."""
    stock = [statistics.mean((row[0], row[3])) for row in samples]
    direct = [statistics.mean((row[1], row[2])) for row in samples]
    return {
        "stock_us": statistics.median(stock),
        "direct_us": statistics.median(direct),
        "speedup_pct": statistics.median(
            100 * (a / b - 1) for a, b in zip(stock, direct)
        ),
        "abba_us": samples,
    }


def exact(got, expected, label):
    if not torch.isfinite(got).all().item():
        raise AssertionError(f"{label}: nonfinite output")
    if not torch.equal(got.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8)):
        delta = (got.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label}: not bit-exact (max_abs={delta})")


def fixture(lengths, layout, device="cuda"):
    total = sum(lengths)
    inp = make_inputs(total, device)
    ends = [0]
    for length in lengths:
        ends.append(ends[-1] + length)
    inp["cu_seqlens"] = torch.tensor(ends, dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(71)
    inp["initial_state"] = torch.randn(
        len(lengths), H, K, V, generator=gen, device=device, dtype=torch.float32
    )
    # Fresh state alongside an existing prefix's state in the packed case.
    if len(lengths) > 1:
        inp["initial_state"][0].zero_()
    if layout in ("conv", "qkv"):
        # Conv returns [channels, T], transposed to [T, 3*H*D] and split.
        # A .contiguous() in the entry therefore copies v for BOTH arms.
        qkv = torch.empty(3 * H * K, total, device=device, dtype=torch.bfloat16).t()
        if layout == "qkv":
            qkv = qkv.contiguous()
        for name, view in zip(("q", "k", "v"), qkv.split(H * K, dim=-1)):
            view = view.reshape(1, total, H, K)
            view.copy_(inp[name])
            inp[name] = view
    return inp


def run_case(entry, lengths, layout, reps, graph_replays):
    inp = fixture(lengths, layout)
    stock_mutates_v = inp["v"].is_contiguous()
    original = {name: value.clone() for name, value in inp.items()
                if isinstance(value, torch.Tensor)}
    total = sum(lengths)
    buffers = {arm: torch.full((1, total + 7, H, V), 42,
                              device="cuda", dtype=torch.bfloat16)
               for arm in ("stock", "direct")}
    destinations = {arm: buf[:, :total] for arm, buf in buffers.items()}

    def reset():
        # The contiguous stock path overwrites v. Restore it before EVERY
        # invocation, so repeated calls never feed previous output as input.
        if stock_mutates_v:
            inp["v"].copy_(original["v"])

    def call(arm):
        output, state = entry(
            inp["q"], inp["k"], inp["v"], raw_g=inp["raw_g"],
            beta=inp["beta"], A_log=inp["A_log"], g_bias=inp["g_bias"],
            scale=inp["scale"], initial_state=inp["initial_state"],
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=inp["cu_seqlens"], safe_gate=True,
            lower_bound=inp["lower_bound"],
            out=destinations[arm] if arm == "direct" else None,
        )
        if state is None:
            raise AssertionError("missing final state")
        if arm == "stock":
            destinations[arm].copy_(output)
        elif output is not destinations[arm]:
            raise AssertionError("entry ignored the explicit output buffer")
        return destinations[arm], state

    def check_inputs(arm):
        for name, reference in original.items():
            if name != "v" or arm == "direct" or not stock_mutates_v:
                exact(inp[name], reference, f"{arm} input {name}")
        exact(buffers[arm][:, total:], torch.full_like(buffers[arm][:, total:], 42),
              f"{arm} padded destination")

    # Prewarm both pointer/layout variants before either arm is timed.
    for arm in ("stock", "direct", "direct", "stock"):
        reset()
        call(arm)
    torch.cuda.synchronize()
    reset()
    ref = tuple(value.clone() for value in call("stock"))
    check_inputs("stock")
    reset()
    got = call("direct")
    for name, value, reference in zip(("output", "state"), got, ref):
        exact(value, reference, f"eager {name}")
    check_inputs("direct")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_eager(arm):
        reset()
        start.record()
        result = call(arm)
        end.record()
        end.synchronize()
        # Keep the returned final state alive through the end event.
        del result
        return start.elapsed_time(end) * 1000

    eager = [[time_eager(arm) for arm in ("stock", "direct", "direct", "stock")]
             for _ in range(reps)]
    graphs, graph_outputs = {}, {}
    for arm in ("stock", "direct"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            reset()
            graph_outputs[arm] = call(arm)
        graphs[arm] = graph
        for _ in range(3):
            for value in graph_outputs[arm]:
                value.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for name, value, reference in zip(("output", "state"), graph_outputs[arm], ref):
                exact(value, reference, f"graph {arm} {name}")
            check_inputs(arm)

    def time_graph(arm):
        start.record()
        for _ in range(graph_replays):
            graphs[arm].replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / graph_replays

    graph = [[time_graph(arm) for arm in ("stock", "direct", "direct", "stock")]
             for _ in range(reps)]
    return {"lengths": lengths, "layout": layout, "exact_output_state": True,
            "v_stride": list(inp["v"].stride()),
            "graph_resets_v": stock_mutates_v,
            "copy_bytes_removed": total * H * V * 2,
            "eager_excludes_reset": paired_summary(eager),
            "graph_includes_reset": paired_summary(graph)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="1,63,64,65,127,128,129,512,2048,4096,8192")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--graph-replays", type=int, default=5)
    parser.add_argument("--order", choices=("long-first", "short-first"), default="long-first",
                        help="first shape seeds shared autotuners; compare both orders in fresh containers")
    parser.add_argument("--json", type=Path,
                        help="optional JSON path inside the container; use a writable host bind to retain it")
    args = parser.parse_args()
    if args.reps < 1 or args.graph_replays < 1:
        parser.error("reps and graph-replays must be positive")
    cases = sequence_cases([int(value) for value in args.lengths.split(",")])
    cases.sort(key=sum, reverse=args.order == "long-first")
    torch.manual_seed(0)
    modules = build_pkg()
    entry = modules["kda"].chunk_kda_with_fused_gate
    # Older entry accepts **kwargs and would silently ignore out.
    if "out" not in inspect.signature(entry).parameters:
        raise RuntimeError("probe requires composed kda.py with explicit out support")
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "triton": triton.__version__, "reps": args.reps,
              "graph_replays": args.graph_replays,
              "order": args.order, "prewarm_lengths": cases[0],
              "sources": {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for name, module in modules.items()}, "cases": []}
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    for lengths in cases:
        for layout in ("contiguous", "conv", "qkv"):
            row = run_case(entry, lengths, layout, args.reps, args.graph_replays)
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
    report["autotuners"] = {
        name: {repr(key): cfg_repr(config) for key, config in tuner.cache.items()}
        for name, tuner in collect_autotuners(modules).items()
    }
    print(json.dumps({"autotuners": report["autotuners"]}), flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: exact output/state, input integrity, padded destination and graph replay")


if __name__ == "__main__":
    main()
