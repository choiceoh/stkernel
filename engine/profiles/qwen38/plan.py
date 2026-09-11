"""Qwen3.8-Flash-Next on this fleet: where every tensor goes, what a rank
holds, and what a sequence costs. Nothing here allocates.

The placement rules are read off the vLLM model files that ship in
`vllm/vllm-openai:qwen38-flash-next` (image d464f3b466fa) and the KV/state
rules off HF's modeling file -- each pinned by sha256 so a vendor change fails
here instead of drifting into a checkpoint that loads and computes garbage
(tp_plan.py held DSv4.1's convert.py to the same contract).

    nvidia/model.py                  d900cd6fcacba18f  parallel classes per module
    gdn/qwen_gdn_linear_attn.py      81b4dcd095249237  GDN: Merged/Column/RowParallel, ba_proj replicated at TP>=2
    common/ple.py                    ab0d4075367c4a85  PLE table: one TP vocabulary range per rank
    mamba/mamba_utils.py             e168adae4ac9a951  gated_delta_net_state_shape
    HF modeling_qwen4_exp.py         77fec77d87f2a0eb  the oracle (transformers 5.16.1)

Two things the numbers say that the config does not:

  KV is not the constraint.  Twelve full-attention layers, one kv head per
    rank, 256 wide, bf16: 12 KiB per token. The 36 GDN layers carry a
    recurrent state that does not grow with context (27.5 MiB per sequence).
    A 40 GiB KV budget reaches the model's 262,144 ceiling at concurrency 8.

  PLE is the D1 line.  47.68 GiB of n-gram table, vocab-parallel over TP, is
    11.92 GiB on every rank -- the second-largest resident item after the
    experts, and the one the vLLM profile spends two knobs on
    (PLE_CPU_OFFLOAD, FORCE_FP8_EMBED). On unified memory "host RAM" is the
    same pool, so the offload knob buys nothing here; the shard does.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

GIB = 1 << 30
CKPT = Path("/home/choiceoh/models/qwen38-flash-next-nvfp4")
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
PINS = {
    "vllm/models/qwen3_8_flash_next/nvidia/model.py": "d900cd6fcacba18f",
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": "81b4dcd095249237",
    "vllm/models/qwen3_8_flash_next/common/ple.py": "ab0d4075367c4a85",
    "vllm/model_executor/layers/mamba/mamba_utils.py": "e168adae4ac9a951",
}
HF_PIN = ("transformers/models/qwen4_exp/modeling_qwen4_exp.py", "77fec77d87f2a0eb")

# (label, matcher, divisor at TP=4 [0 = dropped], source)
RULES = [
    ("routed experts (EP)", lambda n: ".mlp.experts." in n and "shared" not in n, 4,
     "TEP=4: an expert lives whole on one rank"),
    ("PLE n-gram table (vocab-parallel)", lambda n: ".ple.ple_embedding" in n, 4,
     "common/ple.py compute_ple_shard_overlap: one TP vocabulary range"),
    ("PLE other", lambda n: ".ple." in n, 1, "replicated, small"),
    ("vision (dropped)", lambda n: ".visual." in n, 0, "this fleet serves text only"),
    ("embed / lm_head (vocab-parallel)",
     lambda n: n.endswith(("embed_tokens.weight", "lm_head.weight")), 4,
     "VocabParallelEmbedding / ParallelLMHead"),
    ("GDN in/out proj + conv (TP by heads)",
     lambda n: ".linear_attn." in n and "ba_proj" not in n
     and any(s in n for s in ("in_proj", "out_proj", "conv1d", "A_log", "dt_bias", "norm")),
     4, "qwen_gdn_linear_attn.py: ColumnParallel conv1d, RowParallel out_proj, divide(num_v_heads, tp)"),
    ("GDN ba_proj (replicated)", lambda n: ".linear_attn." in n and "ba_proj" in n, 1,
     "qwen_gdn_linear_attn.py:590 -- [num_v_heads]*2 layout does not split at TP>=2"),
    ("GDN rest", lambda n: ".linear_attn." in n, 1, "replicated"),
    ("full-attn q/o proj (TP)",
     lambda n: ".self_attn." in n and any(s in n for s in ("q_proj", "o_proj")), 4,
     "QKVParallelLinear / RowParallelLinear, 24 heads / 4"),
    ("full-attn k/v proj (replicated)",
     lambda n: ".self_attn." in n and any(s in n for s in ("k_proj", "v_proj")), 1,
     "2 kv heads < TP 4: vLLM replicates kv heads"),
    ("indexer + attn rest (replicated)", lambda n: ".self_attn." in n, 1,
     "indexer_kv_heads=1; norms"),
    ("shared expert (TP)", lambda n: "shared_expert." in n, 4, "Merged/RowParallel"),
    ("router / shared gate (replicated)", lambda n: ".mlp.gate" in n or "shared_expert_gate" in n, 1,
     "Gate is replicated"),
    ("mtp (replicated)", lambda n: n.startswith("mtp."), 1, "MTP head, one copy per rank"),
    ("norms / hyper-connections / other (replicated)", lambda n: True, 1, "replicated"),
]


def census(ckpt: "str | Path" = CKPT) -> "list[tuple[str, float, float, int, str]]":
    """(label, checkpoint GiB, per-rank GiB, tensors, source) per rule."""
    ckpt = Path(ckpt)
    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    by, cnt = {}, {}
    for shard in sorted(set(weight_map.values())):
        with (ckpt / shard).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for name, entry in header.items():
            nbytes = entry["data_offsets"][1] - entry["data_offsets"][0]
            for label, match, _div, _src in RULES:
                if match(name):
                    by[label] = by.get(label, 0) + nbytes
                    cnt[label] = cnt.get(label, 0) + 1
                    break
    out = []
    for label, _m, div, src in RULES:
        b = by.get(label, 0)
        if b:
            out.append((label, b / GIB, (b / div / GIB) if div else 0.0, cnt[label], src))
    return out


def resident_gib(ckpt: "str | Path" = CKPT) -> float:
    return sum(r[2] for r in census(ckpt))


def text_config(ckpt: "str | Path" = CKPT) -> dict:
    return json.loads((Path(ckpt) / "config.json").read_text())["text_config"]


def state_bytes(cfg: dict, tp: int = 4) -> "tuple[int, int, int]":
    """(GDN state per sequence, full-attn KV per token, indexer cache per token)."""
    types = cfg["layer_types"]
    n_lin, n_full = types.count("linear_attention"), types.count("full_attention")
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv, conv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"]
    conv_dim = dk * nk * 2 + dv * nv                          # mamba_utils.py:259
    conv_state = conv_dim // tp * (conv - 1) * 2               # bf16
    recurrent = (nv // tp) * dv * dk * 4                       # mamba_ssm_dtype float32
    per_seq = n_lin * (conv_state + recurrent)
    kv_heads = max(1, cfg["num_key_value_heads"] // tp)
    kv_tok = n_full * 2 * kv_heads * cfg["head_dim"] * 2       # k and v, bf16
    idx_tok = int(n_full * cfg["indexer_kv_heads"] * cfg["indexer_head_dim"] * 2
                  / cfg["indexer_compress_ratio"])
    return per_seq, kv_tok, idx_tok


def max_context(kv_gib: float, concurrency: int, cfg: dict, tp: int = 4) -> int:
    per_seq, kv_tok, idx_tok = state_bytes(cfg, tp)
    tokens = (kv_gib * GIB - concurrency * per_seq) / (concurrency * (kv_tok + idx_tok))
    return int(min(max(tokens, 0), cfg["max_position_embeddings"]))


def verify_pins(image: str = IMAGE) -> "list[tuple[str, bool]]":
    """Re-extract the pinned files from the image and compare digests."""
    import subprocess, tempfile
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    out = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for rel, want in PINS.items():
                dst = Path(tmp) / Path(rel).name
                subprocess.run(["docker", "cp", f"{cid}:/usr/local/lib/python3.12/dist-packages/{rel}", dst],
                               check=True, capture_output=True)
                got = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
                out.append((rel, got == want))
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)
    return out


def report(ckpt: "str | Path" = CKPT, kv_gib: "float | None" = None) -> str:
    rows = census(ckpt)
    cfg = text_config(ckpt)
    width = max(len(r[0]) for r in rows)
    out = [f"  {'placement':<{width}}  {'ckpt GiB':>9}  {'per rank':>9}  tensors"]
    for label, tot, mine, n, _src in rows:
        out.append(f"  {label:<{width}}  {tot:>9.2f}  {mine:>9.2f}  {n:>8,}")
    resident = sum(r[2] for r in rows)
    out.append(f"  {'resident per rank':<{width}}  {'':>9}  {resident:>9.2f}")
    per_seq, kv_tok, idx_tok = state_bytes(cfg)
    out.append("")
    out.append(f"  GDN state {per_seq / 2**20:.1f} MiB / sequence (context-independent); "
               f"full-attn KV {kv_tok / 1024:.0f} KiB + indexer {idx_tok / 1024:.2f} KiB per token")
    if kv_gib:
        out.append(f"  {kv_gib:.0f} GiB of KV buys (ceiling {cfg['max_position_embeddings']:,}):")
        for b in (1, 8, 32, 128):
            out.append(f"    concurrency {b:>3}: {max_context(kv_gib, b, cfg):>9,} tok")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--kv-gib", type=float, default=40.0)
    ap.add_argument("--verify-pins", action="store_true")
    a = ap.parse_args()
    print(report(a.ckpt, a.kv_gib))
    if a.verify_pins:
        for rel, ok in verify_pins():
            print(f"  {'PIN OK ' if ok else 'PIN DRIFT'} {rel}")
