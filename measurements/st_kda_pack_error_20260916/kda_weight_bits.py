"""Does a bf16 tensor actually carry bf16 information, or is it an upcast fp8 one?

A weight that was quantised to e4m3 upstream and then widened into a bf16 container keeps
only 3 significant mantissa bits, so its bf16 mantissa has at least 4 trailing zeros. In a
tensor that was really trained and stored in bf16 the low mantissa bits are incidental, so
the fraction with >=b trailing zeros halves with every b: 0.500 0.250 0.125 0.062 0.031.
An upcast tensor instead sits on a plateau through b=4 and only then starts halving.

This matters because specs.py declaring a weight `BF` says where the bytes go, not what is
in them. The KDA projections of st-glm53-nvidia-tp4-9391 read as a plateau: modelopt's NVFP4
`ignore` list excludes every layer's `self_attn*`, so the upstream release's fp8 weights were
passed through untouched. Only the 272 gate rows (beta 16, fa 128, ga 128) are true bf16.

Always read a control from the SAME file -- `embed`, `head` and any `*.in_norm` are true
bf16 there -- so that a surprising plateau is a fact about the tensor and not about the test.

    python3 kda_weight_bits.py ~/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
        L1.kda.in_proj L1.kda.o_proj L20.kda.o_proj L1.in_norm embed head

    python3 kda_weight_bits.py ~/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
        --rows L1.kda.in_proj          # the in_proj row groups separately
"""
import argparse

import torch
from safetensors import safe_open

# 3*Hl*D + Hl + 2*D (specs.py:125), split by tree_decode.py:169.
ROW_GROUPS = (("q/k/v", 0, 6144), ("beta", 6144, 6160), ("fa", 6160, 6288), ("ga", 6288, 6416))
TRUE_BF16 = "0.500 0.250 0.125 0.062 0.031"


def trailing_zero_fractions(t, bits=5):
    """Fraction of nonzero elements whose bf16 mantissa has >=b trailing zero bits, b in 1..bits."""
    u = t.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    mantissa, nonzero = u & 0x7F, (u & 0x7FFF) != 0
    if int(nonzero.sum()) == 0:
        return None
    return [float(((mantissa & ((1 << b) - 1)) == 0)[nonzero].float().mean()) for b in range(1, bits + 1)]


def sample(t, limit):
    """`limit` elements spread across the whole tensor -- a prefix would only read its first rows,
    and in in_proj the first rows are all q/k/v."""
    flat = t.reshape(-1)
    stride = max(1, flat.numel() // limit)
    return flat[::stride][:limit]


def report(name, t, limit):
    if t.dtype != torch.bfloat16:
        print(f"   {name:<34} {str(t.dtype)} {tuple(t.shape)} -- not bf16, skipped")
        return
    fr = trailing_zero_fractions(sample(t, limit))
    if fr is None:
        print(f"   {name:<34} all-zero")
        return
    # a plateau is an upcast: >=4 trailing zeros on most elements means 3 significant bits
    verdict = "fp8 upcast" if fr[3] > 0.5 else "true bf16"
    print(f"   {name:<34} {' '.join(f'{x:.3f}' for x in fr)}   {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--rows", action="store_true", help="split each key by the in_proj row groups")
    ap.add_argument("--limit", type=int, default=1 << 22, help="elements sampled per tensor")
    args = ap.parse_args(argv)

    print(f"fraction of nonzero weights with >=1..>=5 trailing zero mantissa bits")
    print(f"   {'':<34} {TRUE_BF16}   <- true bf16")
    with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
        for key in args.keys:
            try:
                t = f.get_tensor(key)
            except Exception:
                print(f"   {key:<34} MISSING")
                continue
            if args.rows and t.ndim == 2 and t.shape[0] == ROW_GROUPS[-1][2]:
                for label, a, b in ROW_GROUPS:
                    report(f"{key} [{label}]", t[a:b], args.limit)
            else:
                report(key, t, args.limit)


if __name__ == "__main__":
    main()
