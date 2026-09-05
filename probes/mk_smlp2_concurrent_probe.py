#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SMLP2 versus the v2 GEMM/native-activation chain in CUDA graphs.

Capture each shared-expert arm alone and beside the served b12x MoE fixture
in both stream issue orders. Reuse mk_gemm_concurrent_probe's fork/join and
moe_decode_stream_probe's serving wrapper, geometry and expert packer. Both
arms consume identical activations and W4 packs. A pure-Torch mathematical
reference is used only for correctness, never as the timed stock baseline.

Every captured output is checked before timing for finite values, numerical
agreement and repeated-replay stability. Timings alternate forward/reverse
case order, with a cold-L2 preparation outside every measured graph replay;
the shared expert's small pack must not become an L2-resident alone baseline.

The exposed difference is a per-layer synthetic measurement. It excludes
the router/top-k lead-in and the rest of the serving step; x42 is a projection,
not a measured serving gain or a quality/promotion gate.

Run only in an approved idle fleet window, in a fresh scratch container:
    bash probes/run_mk_probe.sh probes/mk_smlp2_concurrent_probe.py --rounds 10
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys


def _require_gate(label: str, finite: bool, ok: bool, rel: float) -> None:
    """A failed/nonfinite gate must never reach the timer."""
    if not finite or not math.isfinite(rel) or not ok:
        raise RuntimeError(f"{label}: numerics/replay failed "
                           f"(finite={finite}, rel={rel:.3e})")


def _balanced_samples(names, measure, rounds: int):
    """Opposite measurement orders each round; all cases get equal samples."""
    names = tuple(names)
    if rounds < 1 or not names or len(set(names)) != len(names):
        raise ValueError("positive rounds and distinct nonempty cases required")
    samples = {name: {"forward": [], "reverse": []} for name in names}
    for round_id in range(rounds):
        directions = ("forward", "reverse") if round_id % 2 == 0 else ("reverse", "forward")
        for direction in directions:
            order = names if direction == "forward" else tuple(reversed(names))
            for name in order:
                value = measure(name)
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"{name}: invalid timing {value}")
                samples[name][direction].append(value)
    return samples


def _run(args) -> int:
    os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1")
    os.environ.setdefault("VLLM_GLM53_MK_GEMM", "1")
    os.environ.setdefault("VLLM_GLM53_MK_PDL", "1")
    sys.path.insert(0, os.environ.get("MK_PKG_PATH",
                                    "/usr/local/lib/python3.12/dist-packages"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    from vllm.model_executor.layers import glm53_megakernel as mk
    from megakernel_glm53_bench import _l2_flush, native_smlp_activation
    from mk_gemm_concurrent_probe import _capture, MOE_LAYERS, SHARED_K, SHARED_N
    from moe_decode_stream_probe import DEV, E, HID, T, _routing, expert_set, served_wrapper

    torch.cuda.init()
    torch.manual_seed(args.seed)
    ext = mk._build()
    device = tuple(ext.probe_device())
    if device[:3] != (12, 1, 48):
        raise RuntimeError(f"not a GB10: {device}")
    if not hasattr(ext, "run_smlp2") or not hasattr(mk, "_smlp2_call"):
        raise RuntimeError("SMLP2 binding is unavailable")
    activation, identity = native_smlp_activation(10.0)
    print(f"device={torch.cuda.get_device_name()} T={T} U={args.moe_experts} "
          f"PDL={os.environ['VLLM_GLM53_MK_PDL']}")
    print(f"baseline: v2 GEMM -> {identity} -> v2 GEMM (native CUDA; no fallback)")
    # Explicitly select the same v2 rule for the baseline and the fused arm.
    # The finally restores every override, including lazy environment state.
    state = ext.probe_state()
    ext.set_gemm2(1, 0, 0)
    try:
        print(f"v2 plans: up={list(ext.gemm2_plan(T, SHARED_N, HID))} "
              f"down={list(ext.gemm2_plan(T, HID, SHARED_K))}")
        x = torch.randn(T, HID, dtype=torch.bfloat16, device=DEV) * 0.5
        up_pack = mk.build_mk_weight_w4(
            torch.randn(SHARED_N, HID, dtype=torch.bfloat16, device=DEV) * 0.05)
        down_pack = mk.build_mk_weight_w4(
            torch.randn(HID, SHARED_K, dtype=torch.bfloat16, device=DEV) * 0.05)

        def chain():
            up = mk._gemm_call(x, up_pack, SHARED_N, bg=True)
            return mk._gemm_call(activation(up), down_pack, HID, bg=True)

        def fused():
            return mk._smlp2_call(x, up_pack, down_pack, SHARED_N, SHARED_K, HID, 10.0)

        reference = mk._smlp_ref(x, up_pack, down_pack, SHARED_N, SHARED_K, HID, 10.0)

        def finite(t):
            return bool(torch.isfinite(t).all())

        def relative(got, ref):
            diff = (got.float() - ref.float()).norm()
            norm = ref.float().norm()
            return float(diff / norm) if norm > 0 else float(diff)

        def check_mlp(label, got):
            good, rel, _ = mk._smlp_gate(got, reference)
            _require_gate(label, finite(got) and finite(reference), good, rel)

        # Warm and check native dispatch before allocating/capturing any graph.
        for label, fn in (("v2 chain eager", chain), ("smlp2 eager", fused)):
            initial = fn().clone()
            check_mlp(label, initial)
            for _ in range(args.replays):
                got = fn()
                check_mlp(label, got)
                rel = relative(got, initial)
                _require_gate(label + " replay", finite(got), rel <= 1e-6, rel)

        wrapper = served_wrapper()
        w13, sf13, w2, sf2 = expert_set(torch.Generator().manual_seed(args.seed + 1))
        scales = torch.ones(E, dtype=torch.float32, device=DEV)
        ids, weights = _routing(args.moe_experts)
        out = torch.empty_like(x)

        def moe():
            wrapper.run(x, w13, sf13, w2, sf2, ids, weights, w1_alpha=scales,
                        w2_alpha=scales, fc2_input_scale=scales, out=out)
            return out

        moe_reference = moe().clone()
        if not finite(moe_reference):
            raise RuntimeError("served MoE baseline produced nonfinite output")

        cases = {}

        def capture_case(name, mlp=None, with_moe=False, order="moe-first"):
            # Keep the final capture's tensors alive and inspect the graph's
            # outputs, rather than re-running eager Python as a graph verdict.
            outputs = {}

            def a():
                if with_moe:
                    outputs["moe"] = moe()

            def b():
                if mlp is not None:
                    outputs["mlp"] = mlp()

            graph = _capture(a, b, order)
            snapshots = {}
            for replay_id in range(args.replays):
                graph.replay()
                torch.cuda.synchronize()
                for kind, got in outputs.items():
                    label = f"{name} {kind} graph"
                    if kind == "mlp":
                        check_mlp(label, got)
                    else:
                        rel = relative(got, moe_reference)
                        _require_gate(label, finite(got), rel <= 1e-6, rel)
                    if replay_id == 0:
                        snapshots[kind] = got.clone()
                    else:
                        rel = relative(got, snapshots[kind])
                        _require_gate(label + " replay", finite(got), rel <= 1e-6, rel)
            cases[name] = (graph, outputs)

        capture_case("moe alone", with_moe=True)
        capture_case("chain alone", chain)
        capture_case("smlp2 alone", fused)
        for order in ("moe-first", "pair-first"):
            capture_case(f"chain {order}", chain, True, order)
            capture_case(f"smlp2 {order}", fused, True, order)
        print(f"numerics: finite + mathematical/native comparison + "
              f"{args.replays} graph replays PASS for all {len(cases)} cases")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def measure(name):
            _l2_flush(hot=(x,))
            start.record()
            cases[name][0].replay()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1e3

        samples = _balanced_samples(cases, measure, args.rounds)
        medians = {}
        print(f"timing: {2 * args.rounds} cold-weight graph samples/case; "
              "paired forward/reverse order; us/layer")
        print(f"{'case':<24}{'median':>10}{'forward':>10}{'reverse':>10}{'min':>10}{'max':>10}")
        for name, directions in samples.items():
            forward, reverse = directions["forward"], directions["reverse"]
            values = forward + reverse
            medians[name] = statistics.median(values)
            print(f"{name:<24}{medians[name]:>10.2f}{statistics.median(forward):>10.2f}"
                  f"{statistics.median(reverse):>10.2f}{min(values):>10.2f}{max(values):>10.2f}")
        for order in ("moe-first", "pair-first"):
            baseline = medians[f"chain {order}"] - medians["moe alone"]
            candidate = medians[f"smlp2 {order}"] - medians["moe alone"]
            saved = baseline - candidate
            print(f"{order}: exposed chain={baseline:.2f} smlp2={candidate:.2f} "
                  f"saved={saved:.2f} us/layer; x{MOE_LAYERS}="
                  f"{saved * MOE_LAYERS / 1e3:.3f} ms (projection, not serving gain)")
    finally:
        ext.restore_probe_state(state)
    print("VERDICT: PASS (finite/numerics/replay; serving speed and quality untested)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10,
                        help="paired forward/reverse rounds (2 samples/case/round)")
    parser.add_argument("--replays", type=int, default=5,
                        help="correctness replays per graph, before timing (>= 2)")
    parser.add_argument("--moe-experts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.replays < 2 or not 8 <= args.moe_experts <= 64:
        parser.error("rounds >= 1, replays >= 2 and 8 <= moe-experts <= 64 required")
    try:
        return _run(args)
    except Exception as exc:
        print(f"VERDICT: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
