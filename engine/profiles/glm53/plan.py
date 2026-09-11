"""GLM-5.3-Flash on this fleet (profile): placement, budget, state -- and a
reality check, because this is the one model whose vLLM path is live.

Every constant here was measured ON GLM in the 39th/40th campaigns, so this
profile is the only one whose budget lines are all `ledger` or `read`: the
runtime floor (5.54), load scratch above weights (59.17 - 50.4 = 8.77),
the max-shape activation profile (+9.17), and the KV that resulted
(8.73 GiB at 63.52 consumed). `check()` puts the placement arithmetic next
to vLLM's own report of 50.4 GiB of weights per rank: if the rules below
are wrong, that comparison says so before anything is built on them.

Placement sources are this repo's own overlay (overlay/modules/glm53_model/,
which is the served model file) and the launcher (TP=4, ENABLE_EP=0,
--block-size 2304, kv fp8_e4m3, SPEC_K=5 with the DFlash2 drafter).
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

GIB = 1 << 30
CKPT = Path("/home/choiceoh/models/glm53-redhat-nvfp4")
TP = 4

def _comp(n: str) -> str:
    """The component right after `self_attn.` -- a TOKEN, not a substring.
    Substring rules bit once: "f_" matched "sel**f_**attn", so every attention
    tensor was excluded from the MLA rule and 4.6 GiB was counted replicated."""
    m = re.search(r"\.self_attn\.([A-Za-z0-9_]+)", n)
    return m.group(1) if m else ""

KDA_TP = {"q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj",
          "q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias", "o_norm", "o_proj"}
DRAFTER = Path("/home/choiceoh/models/GLM-5.3-Flash-DFlash2")   # launcher DRAFT_HOST_PATH, DRAFT_TP=1
MLA_TP = {"q_b_proj", "kv_b_proj", "o_proj"}
MLA_REP = {"q_a_proj", "q_a_layernorm", "kv_a_proj_with_mqa", "kv_a_layernorm"}

# (label, matcher on the stripped name, divisor at TP=4 [0 = dropped], source)
RULES = [
    ("routed experts (TP on the intermediate dim)", lambda n: ".mlp.experts." in n, TP, "launcher ENABLE_EP=0 (EP is EXP-1, an experiment): every rank holds a quarter of every expert -- same bytes as EP, different kernel geometry"),
    ("vision (dropped)", lambda n: n.startswith("visual."), 0, "text only (memory: vision hangs)"),
    ("embed / lm_head (vocab-parallel)", lambda n: n.endswith(("embed_tokens.weight", "lm_head.weight")), TP, "VocabParallelEmbedding / ParallelLMHead"),
    ("indexer (replicated)", lambda n: ".indexer." in n, 1, "index_n_heads 32, kpool compress; served replicated"),
    ("MLA q_b / kv_b / o_proj (TP by heads)", lambda n: _comp(n) in MLA_TP and not re.search(r"layers\.(0|1|2|4|5|6|8|9|10|1[2-4]|1[6-8]|2[0-2]|2[4-6]|2[89]|30|3[2-4]|3[6-8]|4[0-2]|44)\.self_attn\.o_proj", n), TP, "glm5next MLA: 64 heads / 4"),
    ("MLA q_a / kv_a (replicated)", lambda n: _comp(n) in MLA_REP, 1, "low-rank down projections, replicated"),
    ("KDA projections + conv + o_proj (TP by heads)", lambda n: _comp(n) in KDA_TP, TP, "kda.py: 64 heads x 128 / 4"),
    ("attn rest (replicated)", lambda n: ".self_attn." in n, 1, "anything left: should be ~0"),
    ("dense MLP layers 0-2 (TP)", lambda n: re.match(r"layers\.[0-2]\.mlp\.(gate|up|down)_proj", n) is not None, TP, "first_k_dense_replace=3"),
    ("shared expert (TP)", lambda n: "shared_experts" in n, TP, "Merged/RowParallel"),
    ("router gate (replicated)", lambda n: ".mlp.gate." in n, 1, "gate + e_score_correction_bias"),
    ("MTP layer 45 non-expert (replicated)", lambda n: n.startswith("layers.45."), 1, "eh_proj/enorm/hnorm; its experts fell into the EP rule above"),
    ("norms / hc / other (replicated)", lambda n: True, 1, "replicated"),
]


def census(ckpt: "str | Path" = CKPT):
    ckpt = Path(ckpt)
    wm = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    by, cnt = {}, {}
    for shard in sorted(set(wm.values())):
        with (ckpt / shard).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for name, e in h.items():
            b = e["data_offsets"][1] - e["data_offsets"][0]
            s = name.removeprefix("model.").removeprefix("language_model.")
            for label, match, _d, _src in RULES:
                if match(s):
                    by[label] = by.get(label, 0) + b; cnt[label] = cnt.get(label, 0) + 1; break
    return [(l, by.get(l, 0) / GIB, (by.get(l, 0) / d / GIB) if d else 0.0, cnt.get(l, 0), src)
            for l, _m, d, src in RULES if by.get(l)]


def drafter_gib(path: "str | Path" = DRAFTER) -> float:
    """DFlash2, served at DRAFT_TP=1: a full copy on every rank."""
    import glob
    total = 0
    for f in glob.glob(str(Path(path) / "*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n)); h.pop("__metadata__", None)
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for v in h.values())
    return total / GIB


def resident_gib(ckpt=CKPT) -> float:
    return sum(r[2] for r in census(ckpt)) + drafter_gib()


def text_config(ckpt=CKPT) -> dict:
    c = json.loads((Path(ckpt) / "config.json").read_text()); return c.get("text_config", c)


def state_bytes(cfg: dict, tp: int = TP, kv_bytes: int = 1, spec_k: int = 5):
    """(KDA state per sequence, MLA KV per token [fp8 -> 1 B], indexer cache per token).

    With DFlash2 verifying K=5 drafts a step, a sequence keeps K+1 recurrent
    states (one per draft position, so a rejection rolls back by index) and a
    conv window of K + kernel-1 inputs -- the served kda_state_shape(num_spec)
    and the engine's position rings alike. 34.8 MiB/seq was the K=0 number."""
    la = cfg["linear_attn_config"]; heads, hd, k = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
    n_kda = len(la["kda_layers"]); n_full = len(la["full_attn_layers"])
    conv = (3 * heads * hd // tp) * (k - 1 + spec_k) * 2        # q|k|v conv ring, bf16
    recurrent = (spec_k + 1) * (heads // tp) * hd * hd * 4      # fp32, one state per draft position
    tail = (cfg["index_kpool"] - 1 + spec_k) * 2 * cfg["index_head_dim"] * 2  # retain history across rejected drafts
    per_seq = n_kda * (conv + recurrent) + n_full * tail
    kv_tok = n_full * cfg["kv_lora_rank"] * kv_bytes            # MLA latent, nope-only (rope dim 0)
    idx_tok = n_full * cfg["index_head_dim"] * 2 // 4           # kpool compress 4 (index_kpool_compress), bf16
    return per_seq, kv_tok, idx_tok


def check() -> str:
    """The placement rules against vLLM's own number for the same model.

    "Model loading took 50.4 GiB" is a memory delta across load, so it also
    holds what load DERIVES from the checkpoint and keeps: the fp8 fold of
    dense projections (FP8_DENSE=1) and the megakernel W4 packs (180 of them).
    Those are not in any index file; the residual below is their size, and
    it is reported as a residual, not folded into a rule to make zero."""
    weights = resident_gib()
    vllm_reported = 50.4                                          # 40th boot table
    gap = vllm_reported - weights
    return (f"  placement {weights - drafter_gib():.2f} + drafter {drafter_gib():.2f} = {weights:.2f} GiB per rank; "
            f"vLLM reported {vllm_reported} -> residual {gap:+.2f} GiB ({gap / vllm_reported:+.1%}): "
            "load-derived buffers (fp8 fold, MK W4 packs) -- to measure, not to assume")


def report(kv_gib: float = 8.73) -> str:
    rows = census(); w = max(len(r[0]) for r in rows)
    out = [f"  {'placement':<{w}}  {'ckpt GiB':>9}  {'per rank':>9}  tensors"]
    for l, tot, mine, n, _ in rows:
        out.append(f"  {l:<{w}}  {tot:>9.2f}  {mine:>9.2f}  {n:>8,}")
    out.append(f"  {'DFlash2 drafter (replicated, DRAFT_TP=1)':<{w}}  {drafter_gib():>9.2f}  {drafter_gib():>9.2f}")
    out.append(f"  {'resident per rank':<{w}}  {'':>9}  {resident_gib():>9.2f}")
    out.append(check())
    cfg = text_config(); per_seq, kv_tok, idx_tok = state_bytes(cfg)
    out.append(f"\n  KDA state {per_seq / 2**20:.1f} MiB / sequence; MLA KV {kv_tok / 1024:.2f} KiB + indexer {idx_tok / 1024:.2f} KiB per token (fp8 KV)")
    out.append(f"  {kv_gib:.2f} GiB of KV (the 40th boot's actual) buys:")
    for b in (1, 4, 8, 32):
        toks = (kv_gib * GIB - b * per_seq) / (b * (kv_tok + idx_tok))
        out.append(f"    concurrency {b:>3}: {int(min(max(toks, 0), cfg['max_position_embeddings'])):>9,} tok")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
