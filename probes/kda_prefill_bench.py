#!/usr/bin/env python3
"""KDA prefill chunk-kernel launch-config sweep (GB10, sm_121a).

GLM-5.3 imports its production prefill entry from
``vllm.third_party.flash_linear_attention.ops.kda``. The chunk path reaches
five Autotuners there (KKT inter/intra, recompute, output, gate+cumsum) plus
``ops.chunk_delta_h``. Decode uses the separate fused-recurrent entry and is
not part of this sweep. This harness patches exactly those six config lists
in-process and times production ``chunk_kda_with_fused_gate`` with final-state
writeback enabled (coordinate descent, one kernel at a time, others at stock).

Run in a FRESH GPU container (never docker-exec CUDA in the serving one):
  docker run --rm --gpus all --entrypoint python3 \
    -v /home/choiceoh/stkernel-c2:/repo:ro glm53:v13-b12x \
    /repo/probes/kda_prefill_bench.py [--T 8192] [--iters 30]

Winners are kernel-time only: adoption = a config-table edit in a
glm53_kda takeover (env-gated), then a bracket boot (prefill tok/s +
quality 9/9 + Korean 0/16). Outputs of different BT/BK configs may differ
in reduce order -- each candidate is value-checked against the stock
output and flagged (not silently dropped) at rel err > 1e-2.
"""
import argparse
import sys
import traceback

import torch
import triton

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

# GLM-5.3 KDA per-rank prefill shapes (TP=4): linear_num_heads=64 -> 16
# local, linear_head_dim=128, chunk T = max_num_batched_tokens.
H, K, V = 16, 128, 128


def build_pkg():
    from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
        chunk_delta_h,
        kda,
    )
    return {"kda": kda, "chunk_delta_h": chunk_delta_h}


def collect_autotuners(mods):
    expected = {
        "kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
        "kda.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
        "kda.recompute_w_u_fwd_kernel",
        "kda.chunk_gla_fwd_kernel_o",
        "kda.kda_gate_cumsum_fwd_kernel",
        "chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    }
    out = {}
    for name in sorted(expected):
        module_name, attr = name.split(".", 1)
        fn = getattr(mods[module_name], attr)
        # @triton.heuristics is the outer decorator on all six live kernels;
        # its public object wraps the Autotuner in .fn.
        seen = set()
        while not isinstance(fn, triton.runtime.Autotuner):
            if id(fn) in seen or not hasattr(fn, "fn"):
                break
            seen.add(id(fn))
            fn = fn.fn
        if not isinstance(fn, triton.runtime.Autotuner):
            raise TypeError(f"production kernel is no longer an Autotuner: {name}")
        out[name] = fn
    if set(out) != expected:
        raise AssertionError(f"core-six inventory drift: {sorted(out)}")
    return out


def make_inputs(T, device):
    g = torch.Generator(device=device).manual_seed(0)
    dt = torch.bfloat16

    def r(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device,
                           dtype=torch.float32).mul(scale).to(dt)

    q = r(1, T, H, K, 0.8)
    k = r(1, T, H, K, 0.8)
    v = r(1, T, H, V, 0.8)
    raw_g = r(1, T, H, K, 0.3)          # f_b_proj output (raw gate)
    raw_beta = torch.rand(1, T, H, generator=g, device=device,
                          dtype=torch.float32).to(dt)
    # Production GLM applies its compiled fp32 sigmoid before generic FLA KDA.
    beta = raw_beta.float().sigmoid()
    A_log = torch.randn(H, generator=g, device=device,
                        dtype=torch.float32).mul(0.1)
    dt_bias = torch.randn(H * K, generator=g, device=device,
                          dtype=torch.float32).mul(0.1)
    initial_state = torch.randn(1, H, K, V, generator=g, device=device,
                                dtype=torch.float32)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    lower_bound = -5.0
    scale = K ** -0.5
    return dict(q=q, k=k, v=v, raw_g=raw_g, beta=beta, A_log=A_log,
                g_bias=dt_bias, scale=scale, initial_state=initial_state,
                lower_bound=lower_bound, cu_seqlens=cu_seqlens)


def run_entry(pkg, inp):
    o, state = pkg["kda"].chunk_kda_with_fused_gate(
        inp["q"].clone(),
        inp["k"].clone(),
        inp["v"].clone(),
        raw_g=inp["raw_g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        g_bias=inp["g_bias"],
        scale=inp["scale"],
        initial_state=inp["initial_state"].clone(),
        output_final_state=True,
        lower_bound=inp["lower_bound"],
        safe_gate=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"],
    )
    if state is None:
        raise AssertionError("production prefill must write final recurrent state")
    return o, state


def bench_us(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def rel_err(a, b):
    d = (a.float() - b.float()).norm()
    den = b.float().norm()
    return float(d / den) if den > 0 else float("inf")


def output_state_rel_err(got, ref):
    return max(rel_err(got[0], ref[0]), rel_err(got[1], ref[1]))


def cfg_repr(cfg):
    kw = dict(cfg.kwargs)
    return (f"{kw if kw else {}} warps={cfg.num_warps} "
            f"stages={cfg.num_stages}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    mods = build_pkg()
    tuners = collect_autotuners(mods)
    print(f"T={args.T} H={H} K={K} V={V}  autotuned kernels: {len(tuners)}")
    for name, fn in tuners.items():
        print(f"  {name}: {len(fn.configs)} configs")

    inp = make_inputs(args.T, "cuda")
    pkg = mods

    def timed():
        return bench_us(lambda: run_entry(pkg, inp), args.iters)

    # ---- stock baseline (autotune picks per its own benchmark) ----
    ref = run_entry(pkg, inp)
    torch.cuda.synchronize()
    base = timed()
    print(f"\nstock(autotune) baseline: {base:.1f}us")

    # ---- coordinate descent: one kernel at a time ----
    winners = {}
    for name, fn in sorted(tuners.items()):
        stock = list(fn.configs)
        results = []
        for cfg in stock:
            fn.configs = [cfg]
            fn.cache.clear()
            try:
                out = run_entry(pkg, inp)
                torch.cuda.synchronize()
                err = output_state_rel_err(out, ref)
                t = bench_us(lambda: run_entry(pkg, inp), args.iters)
                results.append((t, cfg, err))
                print(f"  {name}: {cfg_repr(cfg):<44} {t:9.1f}us"
                      f"{'  !rel=%.2e' % err if err > 1e-2 else ''}",
                      flush=True)
            except Exception as ex:  # noqa: BLE001 -- keep the sweep alive
                print(f"  {name}: {cfg_repr(cfg):<44} FAILED {ex!r}",
                      flush=True)
        fn.configs = stock
        fn.cache.clear()
        ok = [r for r in results if r[2] <= 1e-2]
        if ok:
            t, cfg, _ = min(ok, key=lambda x: x[0])
            winners[name] = (cfg, t)
            fn.configs = [cfg]
            fn.cache.clear()
            print(f"  -> {name} winner {cfg_repr(cfg)} {t:.1f}us "
                  f"(stock-base {base:.1f}us)\n", flush=True)

    # ---- all winners together ----
    for name, (cfg, _) in winners.items():
        tuners[name].configs = [cfg]
        tuners[name].cache.clear()
    try:
        out = run_entry(pkg, inp)
        torch.cuda.synchronize()
        err = output_state_rel_err(out, ref)
        t = timed()
        print(f"\ncombined winners: {t:.1f}us vs stock {base:.1f}us "
              f"({100 * (base - t) / base:+.1f}%)  rel_err={err:.2e}")
        print("\nsuggested takeover: pin the winning configs per kernel in a "
              "glm53_kda config-table edit (env-gated), then bracket.")
    except Exception as ex:  # noqa: BLE001
        print(f"combined run failed: {ex!r}")


if __name__ == "__main__":
    main()
