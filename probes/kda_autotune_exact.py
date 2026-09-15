"""KDA prefill autotune on one GB10: what a boot pays for it, and whether its choice can change a bit.

The GLM-5.3 prefill chunk lane (profiles/glm53/lanes.served kda_chunk -> kernels/kda.chunk_kda_with_fused_gate)
reaches Triton Autotuners that re-benchmark in every process: nothing keeps their choice on disk. Two questions:

  exact   For every Autotuner the lane reaches, reduce its config list to one config at a time (every other
          Autotuner at its first config) and run the lane on production-shaped inputs; compare every output byte
          (output, final state, marked states) with the all-first-config run. A kernel whose configs all agree
          bit for bit can keep a choice across boots without changing a served number.
  cost    In fresh processes, the lane's first call with the stock lists (compile + benchmark + run) and its
          second call, per boot case -- once with this probe's Triton cache as found and once more (warm); and a
          third process with TRITON_CACHE_AUTOTUNING=1 twice, the second of which reads the first's choices.

Cases are the boot's (adapter._warmup_prefill_memory and the gate's continuation pass): 128 tokens, no state, no
marks; 1,024 tokens with a mark at 768 (one prefix snapshot); 1,024 tokens from an fp32 state with the mark; and the
gate's 32,256-token passes, from no state and from an fp32 state with a mark every 768 tokens.
Per-rank shapes: 16 heads, head dim 128. A_log, dt_bias and the lower bound are layer 0's from rank 3's file.

usage: kda_autotune_exact.py --output /cache/kda-autotune.json   (the queue admits --output only)
"""
import argparse
import json
import os
import subprocess
import sys
import time

H, D = 16, 128
RANK_FILE = "/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors"
CONFIG = "/home/choiceoh/models/st-glm53-nvidia-tp4-9391/config.json"
CASES = {"128": (128, False, ()), "1024m": (1024, False, (12,)), "1024sm": (1024, True, (12,)),
         # the gate's widths: 32,256 tokens from no state (the first pass's key, tuned at 128) and from an fp32 state
         # with a snapshot mark every 768 tokens (the far pass's key)
         "32256": (32256, False, ()), "32256sm": (32256, True, tuple(range(12, 504, 12)))}


def layer0_params(device):
    import torch
    from safetensors import safe_open
    with safe_open(RANK_FILE, framework="pt", device="cpu") as f:
        A_log, dt_bias = f.get_tensor("L0.kda.A_log"), f.get_tensor("L0.kda.dt_bias")
    t = json.load(open(CONFIG))
    bound = t.get("text_config", t)["linear_lower_bound"]
    return A_log.to(device), dt_bias.to(device), float(bound)


def inputs(case, device, seed=0):
    """Tensors laid out as net._kda hands them to the lane: q/k/v are views of one conv output, beta a view of the
    fused projection, g the pair's contiguous [N, H, D]."""
    import torch
    T, with_state, marks = CASES[case]
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    y = torch.randn(T, 3 * H * D, generator=g, device=device).to(bf16)
    q, k, v = (t.reshape(1, T, H, D) for t in y.split(H * D, dim=-1))
    proj = torch.randn(T, 3 * H * D + H + D + D, generator=g, device=device).to(bf16)
    beta = proj.split([3 * H * D, H, D, D], dim=-1)[1][None]
    raw_g = torch.randn(T, H, D, generator=g, device=device).to(bf16)[None]
    state0 = (torch.randn(1, H, D, D, generator=g, device=device) * 0.05) if with_state else None
    return dict(q=q, k=k, v=v, g=raw_g, beta=beta, state0=state0, marks=list(marks))


def lane(x, A_log, dt_bias, bound):
    """profiles/glm53/lanes.served's kda_chunk, argument for argument."""
    import torch
    from engine.kernels.kda import chunk_kda_with_fused_gate
    from engine.kernels.kda.index import single_sequence_bounds
    q, v = x["q"], x["v"]
    out = torch.empty_like(v)
    result = chunk_kda_with_fused_gate(
        q=q, k=x["k"], v=v, raw_g=x["g"], beta=torch.sigmoid(x["beta"].float()), A_log=A_log.view(1, 1, -1, 1),
        g_bias=dt_bias, initial_state=x["state0"].transpose(-1, -2).contiguous() if x["state0"] is not None else None,
        output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=single_sequence_bounds(q.shape[1], q.device),
        safe_gate=True, lower_bound=bound, out=out, states_at=x["marks"] or None)
    torch.cuda.synchronize()
    return tuple(t.detach().clone() for t in result)


def autotuners():
    """{qualified name: Autotuner} over the KDA modules, unwrapping heuristics wrappers."""
    import triton
    from engine.kernels.kda import chunk_delta_h, cumsum, kda, l2norm, solve_tril
    found = {}
    for module in (kda, chunk_delta_h, solve_tril, cumsum, l2norm):
        for name, obj in vars(module).items():
            seen, fn = set(), obj
            while fn is not None and not isinstance(fn, triton.runtime.Autotuner) and id(fn) not in seen:
                seen.add(id(fn))
                fn = getattr(fn, "fn", None)
            if isinstance(fn, triton.runtime.Autotuner):
                found[f"{module.__name__.rsplit('.', 1)[-1]}.{name}"] = fn
    return found


def exact(cases):
    """Per case: the stock autotuners pick (recorded), every tuner is pinned to its pick for the reference run, then
    each reached tuner walks its whole config list with the others pinned. A config the device cannot launch
    (OutOfResources: the SMs' shared memory) is one the stock benchmark scores infinite and never picks."""
    import torch
    from triton.runtime.errors import OutOfResources
    device = torch.device("cuda")
    A_log, dt_bias, bound = layer0_params(device)
    tuners = autotuners()
    stock = {name: list(t.configs) for name, t in tuners.items()}
    calls = {name: 0 for name in tuners}
    for name, t in tuners.items():
        original = t.run

        def counted(*a, _name=name, _run=original, **kw):
            calls[_name] += 1
            return _run(*a, **kw)
        t.run = counted
    report = {}
    try:
        for case in cases:
            x = inputs(case, device)
            for name, t in tuners.items():
                t.configs, t.cache = stock[name], {}
            for n in calls:
                calls[n] = 0
            lane(x, A_log, dt_bias, bound)                          # the stock autotuners pick, as a boot does
            reached = sorted(n for n, c in calls.items() if c)
            picks = {name: {str(key): str(config) for key, config in tuners[name].cache.items()} for name in reached}
            chosen = {name: next(iter(tuners[name].cache.values()), stock[name][0]) for name in reached}
            for name, t in tuners.items():
                t.configs, t.cache = [chosen.get(name, stock[name][0])], {}
            ref = lane(x, A_log, dt_bias, bound)
            rows = {}
            for name in reached:
                verdicts = []
                for config in stock[name]:
                    tuners[name].configs, tuners[name].cache = [config], {}
                    try:
                        got = lane(x, A_log, dt_bias, bound)
                    except OutOfResources as exc:
                        verdicts.append(dict(config=str(config), launchable=False, reason=str(exc)[:160]))
                        continue
                    same = all(torch.equal(a.view(torch.uint8) if a.dtype.is_floating_point else a,
                                           b.view(torch.uint8) if b.dtype.is_floating_point else b)
                               for a, b in zip(got, ref))
                    diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(got, ref))
                    verdicts.append(dict(config=str(config), launchable=True, bit_exact=same, max_abs_diff=diff))
                tuners[name].configs, tuners[name].cache = [chosen[name]], {}
                launchable = [v for v in verdicts if v["launchable"]]
                rows[name] = dict(configs=len(stock[name]), launchable=len(launchable), chosen=str(chosen[name]),
                                  picks=picks[name], bit_exact_all_launchable=all(v["bit_exact"] for v in launchable),
                                  differing=[v for v in launchable if not v["bit_exact"]],
                                  unlaunchable=[v["config"] for v in verdicts if not v["launchable"]])
            report[case] = dict(reached=reached, outputs=[list(t.shape) for t in ref], kernels=rows)
    finally:
        for name, t in tuners.items():
            t.configs, t.cache = stock[name], {}
    return report


def cost_child(cases):
    """One fresh process: the lane's first and second call per case, stock config lists."""
    import torch
    device = torch.device("cuda")
    start = time.perf_counter()
    import engine.kernels.kda  # noqa: F401
    imported = time.perf_counter() - start
    A_log, dt_bias, bound = layer0_params(device)
    out = dict(import_s=round(imported, 3))
    for case in cases:
        x = inputs(case, device)
        t0 = time.perf_counter()
        lane(x, A_log, dt_bias, bound)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        lane(x, A_log, dt_bias, bound)
        second = time.perf_counter() - t0
        out[case] = dict(first_s=round(first, 3), second_s=round(second, 3))
    return out


def cost(cases):
    rows = []
    for label, env in (("stock, cache as found", {}), ("stock, warm", {}),
                       ("autotune cache on, first", {"TRITON_CACHE_AUTOTUNING": "1"}),
                       ("autotune cache on, second", {"TRITON_CACHE_AUTOTUNING": "1"})):
        proc = subprocess.run([sys.executable, __file__, "--child", ",".join(cases)], capture_output=True, text=True,
                              env={**os.environ, **env}, timeout=3600)
        line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None)
        rows.append(dict(label=label, rc=proc.returncode, result=json.loads(line) if line else None,
                         stderr_tail=proc.stderr[-800:] if proc.returncode else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        print(json.dumps(cost_child(a.child.split(","))))
        return 0
    cases = list(CASES)
    import torch
    import triton
    result = dict(torch=torch.__version__, triton=triton.__version__, device=torch.cuda.get_device_name(),
                  triton_cache=os.environ.get("TRITON_CACHE_DIR"))
    def publish():
        text = json.dumps(result, indent=1)
        if a.output:
            with open(a.output, "w") as f:
                f.write(text)
        return text
    result["cost"] = cost(cases)                      # fresh processes first: this one's tuners are then untouched
    publish()
    try:
        result["exact"] = exact(cases)
    except Exception as exc:                          # noqa: BLE001 -- keep the cost rows, name the failure
        import traceback
        result["exact_error"] = traceback.format_exc()[-2000:]
        print(publish())
        raise
    print(publish())
    return 0


if __name__ == "__main__":
    sys.exit(main())
