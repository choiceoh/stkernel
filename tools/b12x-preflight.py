#!/usr/bin/env python3
"""Can this model take the b12x MoE kernel at a given TP size?

b12x is the only MoE path on GB10 that reaches the FP4 tensor cores -- marlin
and the bf16 fallbacks dequantize first -- so the fleet default is to run it and
the interesting question is when it cannot be run. Two conditions decide that,
and both were established by measurement rather than by reading:

1. Expert parallelism is refused outright. flashinfer's entry point raises
   NotImplementedError("b12x_fused_moe does not yet support Expert Parallelism")
   when num_local_experts != num_experts. vLLM's `_supports_parallel_config ->
   not use_ep` is not a wiring gap, it is that refusal restated. So b12x means
   EP off: experts stay whole and TP shards the intermediate dimension.

2. With EP off, the rank-local gate+up rows must be a multiple of 128. The FP4
   block-scale swizzle rounds scale rows up to 128, and a gated MoE gives
   2 * intermediate_per_partition of them. When that is not already aligned the
   weights need padding, and padding is where this turns dangerous: wiring the
   TRTLLM branch's align helper into the CUTLASS/B12X branch makes the model
   boot, log "Padding intermediate size from 160 to 192", pass HEALTH-OK -- and
   answer "1" to "안녕하세요". marlin's justification for padding ("the padded
   region never reaches the output") is a marlin-layout argument and does not
   carry to FlashInfer's swizzle. Measured on Qwen3.8-Flash-Next at TP=4:
   640/4 = 160 -> 320 gate+up rows -> would need 384. Do not retry.

So: 2 * (moe_intermediate_size / tp) % 128 == 0, i.e. the per-rank intermediate
must be a multiple of 64.

The image agrees, in the one place it could:
mxfp4_round_up_hidden_size_and_intermediate_size returns the shapes unchanged
for B12X while rounding them up to 128/256 for MARLIN, DEEPGEMM and TRTLLM.
B12X is the backend that does not pad, so the model has to arrive aligned.

Usage:
  b12x-preflight.py <model_dir> [tp=4]
  b12x-preflight.py --scan <models_root> [tp=4]
"""
import json
import os
import sys

ALIGN = 128  # swizzle_blockscale rounds scale rows to a multiple of this


def _first(cfg, *keys):
    """Look in the config and in the nested text/language sub-config.

    Multimodal-style configs (Qwen4Exp, Glm5Next, Step3p7) keep the MoE shape
    under text_config, so a top-level-only lookup silently reports "not an MoE"
    for exactly the models this check exists to judge.
    """
    scopes = [cfg]
    for nest in ("text_config", "language_config", "llm_config"):
        sub = cfg.get(nest)
        if isinstance(sub, dict):
            scopes.append(sub)
    for scope in scopes:
        for k in keys:
            v = scope.get(k)
            if v is not None:
                return v
    return None


def inspect(model_dir: str, tp: int) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        return {"name": os.path.basename(model_dir.rstrip("/")), "moe": False,
                "reason": "no config.json"}
    try:
        cfg = json.load(open(path))
    except Exception as exc:  # noqa: BLE001 -- a broken config is a finding
        return {"name": os.path.basename(model_dir.rstrip("/")), "moe": False,
                "reason": f"unreadable config.json: {exc}"}

    name = os.path.basename(model_dir.rstrip("/"))
    inter = _first(cfg, "moe_intermediate_size", "intermediate_size_moe")
    experts = _first(cfg, "n_routed_experts", "num_experts", "moe_num_experts",
                 "num_local_experts")
    if inter is None or experts is None:
        return {"name": name, "moe": False, "reason": "not an MoE config"}

    per_rank = inter / tp
    rows = 2 * per_rank
    aligned = per_rank == int(per_rank) and int(rows) % ALIGN == 0
    out = {
        "name": name,
        "moe": True,
        "experts": experts,
        "moe_intermediate": inter,
        "per_rank": per_rank,
        "gate_up_rows": rows,
        "b12x": aligned,
    }
    if not aligned:
        need = ((int(rows) + ALIGN - 1) // ALIGN) * ALIGN
        out["reason"] = (
            f"gate+up rows {int(rows)} not a multiple of {ALIGN} "
            f"(would need {need}); padding corrupts this branch"
        )
    return out


def main() -> int:
    args = [a for a in sys.argv[1:]]
    scan = False
    if args and args[0] == "--scan":
        scan = True
        args = args[1:]
    if not args:
        print(__doc__.strip().splitlines()[-3].strip(), file=sys.stderr)
        return 2
    root = args[0]
    tp = int(args[1]) if len(args) > 1 else 4

    dirs = (sorted(os.path.join(root, d) for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
            if scan else [root])

    rows = [inspect(d, tp) for d in dirs]
    moe = [r for r in rows if r["moe"]]
    if not moe:
        print(f"no MoE model found (tp={tp})")
        return 1

    width = max(len(r["name"]) for r in moe)
    print(f"{'model':<{width}}  experts  moe_int  /tp={tp}  gate+up  b12x")
    for r in sorted(moe, key=lambda r: (not r["b12x"], r["name"])):
        mark = "yes" if r["b12x"] else "NO"
        print(f"{r['name']:<{width}}  {r['experts']:>7}  {r['moe_intermediate']:>7}  "
              f"{r['per_rank']:>7g}  {r['gate_up_rows']:>7g}  {mark}")
        if not r["b12x"]:
            print(f"{'':<{width}}    ^ {r['reason']}")
    blocked = [r for r in moe if not r["b12x"]]
    print(f"\n{len(moe) - len(blocked)}/{len(moe)} can run b12x at TP={tp}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
