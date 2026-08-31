#!/usr/bin/env python3
"""KDA prefill chunk-kernel launch-config sweep (GB10, sm_121a).

The active KDA prefill kernels are the vendored triton package
``vllm/models/kimi_k3/nvidia/ops/third_party/kda/`` (chunk.py,
chunk_intra.py, chunk_intra_token_parallel.py, fused_recurrent.py). Their
``triton.autotune`` cache keys carry H/K/V/BT/BK/BV but explicitly NOT T
(``do_not_specialize=["T"]``) -- so ONE chosen (warps, stages, BK/BV) serves
both decode T=8 and prefill T=8192, decided by whichever regime autotuned
first. This harness re-sweeps each kernel's config list at the PREFILL shape
by patching the Autotuner configs in-process and timing the real
``chunk_kda_with_fused_gate`` entry (coordinate descent, one kernel at a
time, others at stock).

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
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (  # noqa: E402
        chunk,
        chunk_intra,
        chunk_intra_token_parallel,
        fused_recurrent,
    )
    return {
        "chunk": chunk,
        "chunk_intra": chunk_intra,
        "chunk_intra_token_parallel": chunk_intra_token_parallel,
        "fused_recurrent": fused_recurrent,
    }


def collect_autotuners(mods):
    out = {}
    for mname, mod in mods.items():
        for attr in dir(mod):
            fn = getattr(mod, attr)
            if isinstance(fn, triton.runtime.Autotuner):
                out[f"{mname}.{attr}"] = fn
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
    A_log = torch.randn(H, generator=g, device=device,
                        dtype=torch.float32).mul(0.1)
    dt_bias = torch.randn(H * K, generator=g, device=device,
                          dtype=torch.float32).mul(0.1)
    initial_state = torch.randn(1, H, K, V, generator=g, device=device,
                                dtype=torch.float32)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    lower_bound = -5.0
    scale = K ** -0.5
    return dict(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log,
                g_bias=dt_bias, scale=scale, initial_state=initial_state,
                lower_bound=lower_bound, cu_seqlens=cu_seqlens)


def run_entry(pkg, inp, output_final_state=False):
    o, state = pkg["chunk"].chunk_kda_with_fused_gate(
        inp["q"].clone(),
        inp["k"].clone(),
        inp["v"].clone(),
        raw_g=inp["raw_g"],
        raw_beta=inp["raw_beta"],
        A_log=inp["A_log"],
        g_bias=inp["g_bias"],
        scale=inp["scale"],
        initial_state=inp["initial_state"].clone(),
        output_final_state=output_final_state,
        lower_bound=inp["lower_bound"],
        cu_seqlens=inp["cu_seqlens"],
    )
    return o


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


def cfg_repr(cfg):
    kw = dict(cfg.kwargs)
    return (f"{kw or '{{}}} warps={cfg.num_warps} "
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
    ref = run_entry(pkg, inp, output_final_state=False)
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
                err = rel_err(out, ref)
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
        err = rel_err(out, ref)
        t = timed()
        print(f"\ncombined winners: {t:.1f}us vs stock {base:.1f}us "
              f"({100 * (base - t) / base:+.1f}%)  rel_err={err:.2e}")
        print("\nsuggested takeover: pin the winning configs per kernel in a "
              "glm53_kda config-table edit (env-gated), then bracket.")
    except Exception as ex:  # noqa: BLE001
        print(f"combined run failed: {ex!r}")


if __name__ == "__main__":
    main()
