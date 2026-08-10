#!/usr/bin/env python3
"""Domain-refit the DSpark Markov head (W1/W2) from a serving-domain corpus.

The DSpark sequential stage (arXiv:2607.05147) biases draft logits with a
first-order transition term B[prev, next] = W1[prev] . W2[next], jointly
trained with the backbone (rank 256). This tool builds a smoothed-PMI bigram
correction from a corpus that matches the actual serving traffic (e.g. Korean
agent prose, where measured acceptance is lowest), blends it with the shipped
low-rank matrix, and re-factorizes to the checkpoint rank:

    M = (1 - blend) * W1_ship @ W2_ship^T + blend * clip(PMI_corpus)
    W1', W2'  =  rank-R randomized SVD factors of M   (M ~= W1' @ W2'^T)

Output is a .pt payload consumed by the VLLM_DSPARK_MARKOV_SIDELOAD launcher
knob (MARKOV_SIDELOAD=/home/choiceoh/models/<name>.pt — must live under the
models mount so every node sees it). Proposal-q only: the seeded-Gumbel
verification keeps the target distribution invariant regardless of these
weights, so the ONLY metric is bench-dec acceptance (+ 9/9 as hygiene).

Ceiling caveat (ledger): MARKOV_SCALE 0.7-1.6 was acceptance-flat, so the
head is scale-insensitive around the trained point; a refit moves the
DIRECTION of the bias, not its scale, but expectations should stay modest.
Upgrade path if PMI-blend shows signal: distill B against target-model
logits from serving traces instead of corpus counts (same sideload contract).

Run on a fleet node inside a THROWAWAY container (never docker exec into the
serving container — second CUDA context stalls TP collectives; this tool is
CPU-only but the rule is absolute):

    docker run --rm -v /home/choiceoh/models:/home/choiceoh/models \
      -v ~/stkernel/tools:/tools --entrypoint python3 \
      aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6 \
      /tools/markov_refit.py \
        --tokenizer /home/choiceoh/models/DeepSeek-V4-Flash-0731 \
        --base /home/choiceoh/models/DeepSeek-V4-Flash-0731 \
        --corpus /home/choiceoh/models/refit-corpus/*.txt \
        --out /home/choiceoh/models/markov-refit-v1.pt
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer dir (the model checkpoint dir)")
    p.add_argument("--corpus", nargs="+", required=True,
                   help=".txt (raw) or .jsonl files (keys: completion|text)")
    p.add_argument("--out", required=True, help="output payload .pt path")
    p.add_argument("--base", default=None,
                   help="model dir with the shipped mtp markov weights "
                        "(safetensors); omit for a pure-PMI fit (needs --rank)")
    p.add_argument("--rank", type=int, default=None,
                   help="factor rank; default = shipped W1 rank")
    p.add_argument("--blend", type=float, default=0.5,
                   help="PMI weight in [0,1]; 0 = shipped only (default 0.5)")
    p.add_argument("--min-count", type=int, default=4,
                   help="keep bigrams seen at least this often (default 4)")
    p.add_argument("--clip", type=float, default=3.0,
                   help="clip |PMI| to this bound (shipped biases are ~+-3)")
    p.add_argument("--smooth-k", type=float, default=1.0,
                   help="additive smoothing pseudo-count (default 1.0)")
    p.add_argument("--max-tokens", type=int, default=200_000_000,
                   help="corpus token cap")
    p.add_argument("--oversample", type=int, default=16,
                   help="randomized-SVD oversampling columns")
    p.add_argument("--power-iters", type=int, default=2,
                   help="randomized-SVD power iterations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def iter_corpus_texts(paths: list[str]):
    expanded: list[str] = []
    for raw in paths:
        hits = sorted(glob.glob(raw))
        if not hits:
            sys.exit(f"corpus path matched nothing: {raw}")
        expanded.extend(hits)
    for path in expanded:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = row.get("completion") or row.get("text") or ""
                    if text:
                        yield path, text
        else:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            if text.strip():
                yield path, text


def count_bigrams(tokenizer, args) -> tuple[Counter, Counter, int]:
    pair_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    total = 0
    for path, text in iter_corpus_texts(args.corpus):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        take = min(len(ids), args.max_tokens - total)
        if take < 2:
            break
        ids = ids[:take]
        unigram_counts.update(ids)
        pair_counts.update(zip(ids, ids[1:]))
        total += take
        print(f"  counted {path}: +{take} tokens (total {total})")
        if total >= args.max_tokens:
            break
    if total < 10_000:
        sys.exit(f"corpus too small for a refit: {total} tokens (< 10k)")
    return pair_counts, unigram_counts, total


def find_shipped_markov(base_dir: str):
    """Load the shipped mtp markov W1/W2 (highest stage wins, as in vLLM)."""
    import torch
    from safetensors import safe_open

    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        shards = {}
        for key, shard in weight_map.items():
            if key.endswith(("markov_head.markov_w1.weight",
                             "markov_head.markov_w2.weight")):
                shards[key] = os.path.join(base_dir, shard)
    else:
        shards = {}
        for shard in sorted(glob.glob(os.path.join(base_dir, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(("markov_head.markov_w1.weight",
                                     "markov_head.markov_w2.weight")):
                        shards[key] = shard
    if not shards:
        sys.exit(f"no markov_head weights found under {base_dir}")

    def stage_of(key: str) -> int:
        head = key.split(".", 2)
        return int(head[1]) if head[0] == "mtp" and head[1].isdigit() else -1

    best = max(stage_of(k) for k in shards)
    picked = {k: v for k, v in shards.items() if stage_of(k) == best}
    out = {}
    for key, shard in picked.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            out[key.rsplit(".", 2)[-2]] = f.get_tensor(key).float()
    w1 = out.get("markov_w1")
    w2 = out.get("markov_w2")
    if w1 is None or w2 is None:
        sys.exit(f"markov stage mtp.{best} is missing w1 or w2")
    print(f"  shipped markov from mtp.{best}: W1 {tuple(w1.shape)}, "
          f"W2 {tuple(w2.shape)}")
    return w1, w2


def build_pmi_sparse(pair_counts, unigram_counts, total, vocab_size, args):
    """clip(log P(next|prev) - log P(next)) as sparse COO (+ its transpose)."""
    import torch

    k = args.smooth_k
    rows, cols, vals = [], [], []
    kept = 0
    for (prev, nxt), c_pn in pair_counts.items():
        if c_pn < args.min_count:
            continue
        if not (0 <= prev < vocab_size and 0 <= nxt < vocab_size):
            continue
        c_p = unigram_counts[prev]
        p_next = unigram_counts[nxt] / total
        pmi = math.log((c_pn + k * p_next) / (c_p + k)) - math.log(p_next)
        pmi = max(-args.clip, min(args.clip, pmi))
        rows.append(prev)
        cols.append(nxt)
        vals.append(pmi)
        kept += 1
    if kept == 0:
        sys.exit("no bigram survived --min-count; corpus too small/diverse")
    print(f"  PMI matrix: {kept} entries "
          f"(of {len(pair_counts)} distinct pairs)")
    idx = torch.tensor([rows, cols], dtype=torch.int64)
    val = torch.tensor(vals, dtype=torch.float32)
    shape = (vocab_size, vocab_size)
    s = torch.sparse_coo_tensor(idx, val, shape).coalesce()
    s_t = torch.sparse_coo_tensor(idx.flip(0), val, shape).coalesce()
    return s, s_t, kept


def randomized_factor(matvec, rmatvec, vocab_size, rank, args):
    """Rank-R factorization of the implicit [V, V] operator M ~= L @ R^T."""
    import torch

    gen = torch.Generator().manual_seed(args.seed)
    sketch = rank + args.oversample
    omega = torch.randn(vocab_size, sketch, generator=gen)
    y = matvec(omega)
    for _ in range(args.power_iters):
        y = matvec(rmatvec(y))
    q, _ = torch.linalg.qr(y)  # [V, sketch]
    b = rmatvec(q).T  # [sketch, V]
    u_b, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_b
    root = torch.sqrt(s[:rank])
    w1 = u[:, :rank] * root  # [V, R]
    w2 = vh[:rank].T * root  # [V, R]
    energy = float((s[:rank].sum() / s.sum()).item()) if s.sum() > 0 else 0.0
    return w1.contiguous(), w2.contiguous(), energy


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        sys.exit(f"--blend must be in [0,1], got {args.blend}")

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    # DSV4 tokenizer files may report the base vocab; trust the larger of the
    # two so ids at the padded tail never index out of range.
    vocab_size = max(vocab_size, len(tokenizer))
    print(f"tokenizer vocab: {vocab_size}")

    w1_ship = w2_ship = None
    if args.base:
        w1_ship, w2_ship = find_shipped_markov(args.base)
        vocab_size = max(vocab_size, w1_ship.shape[0])
        rank = args.rank or w1_ship.shape[1]
        if (w1_ship.shape != (vocab_size, w1_ship.shape[1])
                or w2_ship.shape != w1_ship.shape):
            sys.exit(f"shipped markov shapes disagree: {tuple(w1_ship.shape)} "
                     f"vs {tuple(w2_ship.shape)} (vocab {vocab_size})")
    else:
        rank = args.rank
        if not rank:
            sys.exit("--rank is required without --base")
        if args.blend < 1.0:
            print("WARN: no --base; forcing blend=1.0 (pure PMI). The shipped "
                  "head was trained jointly with the backbone — full "
                  "replacement is expected to regress; prefer --base.")
            args.blend = 1.0

    print("counting corpus bigrams ...")
    pair_counts, unigram_counts, total = count_bigrams(tokenizer, args)
    s, s_t, kept = build_pmi_sparse(
        pair_counts, unigram_counts, total, vocab_size, args
    )

    lam = args.blend

    def matvec(x: "torch.Tensor") -> "torch.Tensor":
        out = torch.sparse.mm(s, x) * lam
        if w1_ship is not None and lam < 1.0:
            out += (1.0 - lam) * (w1_ship @ (w2_ship.T @ x))
        return out

    def rmatvec(x: "torch.Tensor") -> "torch.Tensor":
        out = torch.sparse.mm(s_t, x) * lam
        if w1_ship is not None and lam < 1.0:
            out += (1.0 - lam) * (w2_ship @ (w1_ship.T @ x))
        return out

    print(f"factorizing blended matrix to rank {rank} ...")
    w1_new, w2_new, energy = randomized_factor(
        matvec, rmatvec, vocab_size, rank, args
    )
    print(f"  captured spectral energy (sketch-relative): {energy:.3f}")

    # Spot-check reconstruction on the heaviest corpus bigrams (recompute the
    # exact blended target from the counts — same formula as build_pmi_sparse).
    top_pairs = [pair for pair, c in pair_counts.most_common(512)
                 if c >= args.min_count][:256]
    if top_pairs:
        k = args.smooth_k
        exact = []
        for prev, nxt in top_pairs:
            p_next = unigram_counts[nxt] / total
            pmi = math.log(
                (pair_counts[(prev, nxt)] + k * p_next)
                / (unigram_counts[prev] + k)
            ) - math.log(p_next)
            val = lam * max(-args.clip, min(args.clip, pmi))
            if w1_ship is not None and lam < 1.0:
                val += (1.0 - lam) * float(w1_ship[prev] @ w2_ship[nxt])
            exact.append(val)
        prev_idx = torch.tensor([p for p, _ in top_pairs])
        next_idx = torch.tensor([n for _, n in top_pairs])
        approx = (w1_new[prev_idx] * w2_new[next_idx]).sum(-1)
        exact_t = torch.tensor(exact)
        err = float((approx - exact_t).abs().mean())
        print(f"  top-bigram reconstruction MAE: {err:.4f} "
              f"(mean |target| {float(exact_t.abs().mean()):.4f})")

    corpus_meta = []
    for raw in args.corpus:
        for path in sorted(glob.glob(raw)):
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            corpus_meta.append({"path": path, "sha256": digest})

    payload = {
        "markov_w1": w1_new.float(),
        "markov_w2": w2_new.float(),
        "meta": {
            "tool": "markov_refit.py",
            "blend": lam,
            "rank": rank,
            "clip": args.clip,
            "smooth_k": args.smooth_k,
            "min_count": args.min_count,
            "corpus_tokens": total,
            "kept_bigrams": kept,
            "spectral_energy": energy,
            "corpus": corpus_meta,
            "base": args.base,
            "seed": args.seed,
        },
    }
    torch.save(payload, args.out)
    with open(args.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print(f"wrote {args.out}\n  sha256={digest}\n"
          f"arm with: MARKOV_SIDELOAD={args.out} bash launchers/start-hy4-tp4.sh")


if __name__ == "__main__":
    main()
