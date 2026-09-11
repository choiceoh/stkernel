"""Qwen3.8-Flash-Next's per-sequence state and per-token caches (profile)."""
from __future__ import annotations

from engine.base.caches import GIB, Cache, total_bytes, max_seq
from engine.profiles.qwen38.plan import text_config, state_bytes


def caches(tp: int = 4) -> "list[Cache]":
    c = text_config()
    per_seq, kv_tok, idx_tok = state_bytes(c, tp)
    n_lin = c["layer_types"].count("linear_attention"); n_full = len(c["layer_types"]) - n_lin
    conv_dim = c["linear_key_head_dim"] * c["linear_num_key_heads"] * 2 + c["linear_value_head_dim"] * c["linear_num_value_heads"]
    conv = n_lin * (conv_dim // tp) * (c["linear_conv_kernel_dim"] - 1) * 2
    return [
        Cache("gdn conv state", n_lin, 0.0, conv, 0.0, f"[{conv_dim // tp}, {c['linear_conv_kernel_dim'] - 1}] bf16 per GDN layer; a slot, not a page"),
        Cache("gdn recurrent state", n_lin, 0.0, per_seq - conv, 0.0,
              f"[{c['linear_num_value_heads'] // tp}, {c['linear_value_head_dim']}, {c['linear_key_head_dim']}] fp32 (mamba_ssm_dtype) per GDN layer"),
        Cache("full-attn kv", n_full, kv_tok, 0.0, 0.0,
              f"1 kv head per rank (2 < TP {tp}, replicated) x {c['head_dim']} x k,v x bf16 per QSA layer -- paged"),
        Cache("qsa compressed keys", n_full, idx_tok, 0.0, 0.0,
              f"[T/{c['indexer_compress_ratio']}, {c['indexer_head_dim']}] bf16 per QSA layer"),
    ]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--kv-gib", type=float, default=40.0); a = ap.parse_args()
    cs = caches(); cfg = text_config(); S = 131072
    for k in cs:
        at = k.per_batch_per_token * S + k.per_batch + k.per_token * S
        print(f"  {k.name:22s} {k.blocks:>3} layers  {at / GIB:8.4f} GiB @ B=1 S=128K   {k.note}")
    print(f"  total @ B=1 S=128K: {total_bytes(cs, 1, S) / GIB:.3f} GiB")
    print(f"  {a.kv_gib:.0f} GiB buys:")
    for b in (1, 8, 32, 128):
        print(f"    concurrency {b:>3}: {min(max_seq(cs, a.kv_gib, b), cfg['max_position_embeddings']):>9,} tok")
