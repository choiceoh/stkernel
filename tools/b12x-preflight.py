#!/usr/bin/env python3
"""Can this model take the b12x MoE kernel at a given TP / EP size?

b12x is the only MoE path on GB10 that reaches the FP4 tensor cores -- marlin
and the bf16 fallbacks dequantize first -- so the fleet default is to run it and
the interesting question is when it cannot be run.

The fused kernel still refuses ``num_local != num_experts`` at its own
entry (flashinfer #3383: weight_E vs state_E, then illegal address). The
overlay does not lift that. It makes EP look like a smaller non-EP MoE:
global top-k ids are remapped onto the local shard, remote slots land on a
dummy expert at scale 0, and vLLM's EP all-reduce combines the ranks.
``--enable-expert-parallel`` is what selects that path.

Two alignment questions remain, both measured:

1. EP off (TP shards intermediate): rank-local gate+up rows must be a
   multiple of 128. The FP4 block-scale swizzle rounds scale rows up to 128,
   and a gated MoE gives 2 * intermediate_per_partition of them. Padding is
   where this turns dangerous: wiring the TRTLLM branch's align helper into
   the CUTLASS/B12X branch makes the model boot, log "Padding intermediate
   size from 160 to 192", pass HEALTH-OK -- and answer "1" to "안녕하세요".
   marlin's justification for padding ("the padded region never reaches the
   output") is a marlin-layout argument and does not carry to FlashInfer's
   swizzle. Measured on Qwen3.8-Flash-Next at TP=4: 640/4 = 160 -> 320
   gate+up rows -> would need 384. Do not retry.

2. EP on (MoE TP becomes 1, experts shard): the *full* intermediate is what
   has to align, and the expert count must divide the EP size. Qwen3.8's
   640 is 1280 gate+up rows -- a multiple of 128 -- so EP=4 is the path that
   reaches b12x for that model. GLM-5.3's 2048 aligns either way.

So: EP off needs 2 * (moe_intermediate_size / tp) % 128 == 0.
    EP on  needs moe_intermediate_size % 128 == 0 and experts % ep == 0
    (the wrapper tiles intermediate at 128; a 64-aligned shape still pads).

The image agrees, in the one place it could:
mxfp4_round_up_hidden_size_and_intermediate_size returns the shapes unchanged
for B12X while rounding them up to 128/256 for MARLIN, DEEPGEMM and TRTLLM.
B12X is the backend that does not pad, so the model has to arrive aligned.

Usage:
  b12x-preflight.py <model_dir> [tp=4]
  b12x-preflight.py --scan <models_root> [tp=4]
  b12x-preflight.py --scan <models_root> 4 --ep 4
"""
import json
import os
import sys

ALIGN = 128  # swizzle_blockscale rounds scale rows to a multiple of this


def gate_up_aligned(intermediate, partition: int) -> bool:
    """True when 2 * (intermediate / partition) is an integer multiple of 128."""
    per_rank = intermediate / partition
    rows = 2 * per_rank
    return per_rank == int(per_rank) and int(rows) % ALIGN == 0


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


def inspect(model_dir: str, tp: int, ep: int | None = None) -> dict:
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

    ep = tp if ep is None else ep
    per_rank = inter / tp
    rows = 2 * per_rank
    aligned_tp = gate_up_aligned(inter, tp)
    experts_ok = experts == int(experts) and int(experts) % ep == 0
    # EP uses the full intermediate (MoE TP=1). The wrapper pads to a 128
    # tile when intermediate % 128 != 0 — same corruption as the TP pad.
    aligned_ep = (
        inter == int(inter) and int(inter) % ALIGN == 0 and experts_ok
    )
    out = {
        "name": name,
        "moe": True,
        "experts": experts,
        "moe_intermediate": inter,
        "per_rank": per_rank,
        "gate_up_rows": rows,
        "ep": ep,
        "b12x": aligned_tp,
        "b12x_ep": aligned_ep,
    }
    if not aligned_tp:
        need = ((int(rows) + ALIGN - 1) // ALIGN) * ALIGN
        out["reason"] = (
            f"gate+up rows {int(rows)} not a multiple of {ALIGN} "
            f"(would need {need}); padding corrupts this branch"
        )
    if not aligned_ep:
        if not experts_ok:
            out["reason_ep"] = (
                f"experts {experts} not divisible by ep={ep}"
            )
        else:
            need = ((int(inter) + ALIGN - 1) // ALIGN) * ALIGN
            out["reason_ep"] = (
                f"intermediate {inter} not a multiple of {ALIGN} "
                f"(would need {need}); wrapper tile pad corrupts this branch"
            )
    return out


def main() -> int:
    args = [a for a in sys.argv[1:]]
    scan = False
    ep = None
    if args and args[0] == "--scan":
        scan = True
        args = args[1:]
    if "--ep" in args:
        i = args.index("--ep")
        if i + 1 >= len(args):
            print("b12x-preflight.py: --ep needs a size", file=sys.stderr)
            return 2
        ep = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__.strip().splitlines()[-3].strip(), file=sys.stderr)
        return 2
    root = args[0]
    tp = int(args[1]) if len(args) > 1 else 4
    if ep is None:
        ep = tp

    dirs = (sorted(os.path.join(root, d) for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
            if scan else [root])

    rows = [inspect(d, tp, ep) for d in dirs]
    moe = [r for r in rows if r["moe"]]
    if not moe:
        print(f"no MoE model found (tp={tp})")
        return 1

    width = max(len(r["name"]) for r in moe)
    print(f"{'model':<{width}}  experts  moe_int  /tp={tp}  gate+up  b12x-tp  b12x-ep{ep}")
    for r in sorted(moe, key=lambda r: (not r["b12x"] and not r["b12x_ep"], r["name"])):
        mark_tp = "yes" if r["b12x"] else "NO"
        mark_ep = "yes" if r["b12x_ep"] else "NO"
        print(f"{r['name']:<{width}}  {r['experts']:>7}  {r['moe_intermediate']:>7}  "
              f"{r['per_rank']:>7g}  {r['gate_up_rows']:>7g}  {mark_tp:>7}  {mark_ep}")
        if not r["b12x"]:
            print(f"{'':<{width}}    ^ tp: {r['reason']}")
        if not r["b12x_ep"]:
            print(f"{'':<{width}}    ^ ep: {r['reason_ep']}")
    blocked_tp = [r for r in moe if not r["b12x"]]
    blocked_ep = [r for r in moe if not r["b12x_ep"]]
    print(f"\n{len(moe) - len(blocked_tp)}/{len(moe)} can run b12x at TP={tp} (EP off).")
    print(f"{len(moe) - len(blocked_ep)}/{len(moe)} can run b12x at EP={ep} (MoE TP=1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
