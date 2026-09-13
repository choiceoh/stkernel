"""DSv4.1-Flash's shape constraints (profile).
"""
from __future__ import annotations

import json
from math import gcd
from pathlib import Path

from engine.base.shapes import Constraint, chunk_for as base_chunk_for


def constraints(repo: "str | Path") -> "list[Constraint]":
    cfg = json.loads((Path(repo) / "config.json").read_text())["text_config"]
    ratios = [r for r in cfg["compress_ratios"] if r > 1]
    group = 1
    for r in ratios:
        group = group * r // gcd(group, r)
    block = cfg["candidate_block_size"]
    return [
        Constraint(
            "compressor group", group, "config.compress_ratios",
            "a chunk that ends mid-group leaves the tail in Compressor.kv_state "
            "and the next chunk pools across a boundary the reference never does"),
        Constraint(
            "candidate block", block, "config.candidate_block_size",
            "candidate selection scores whole blocks of compressed positions; "
            "a partial block at the end is scored against padding"),
        Constraint(
            "chunk alignment", group * block, "the two above",
            "the raw-token multiple that leaves neither a partial compressor "
            "group nor a partial candidate block"),
        Constraint(
            "sliding window", cfg["sliding_window"], "config.sliding_window",
            "prefill seeds the ring from the LAST `window` tokens with a "
            "rotated two-part copy; a chunk shorter than the window seeds "
            "fewer slots than a decode step will read (dsv41_window.py)"),
        Constraint(
            "index topk", cfg["index_topk"], "config.index_topk",
            "the indexer keeps this many compressed positions per query"),
        Constraint(
            "candidate domain", block * cfg["candidate_topk_blocks"],
            "candidate_block_size x candidate_topk_blocks",
            "compressed positions the candidate stage narrows to before top-k"),
        Constraint(
            "topk dtype", "int32", "dsv41_sparse_contract.py",
            "the TileLang kernel declares T.Tensor[(b, m, topk), INT32]"),
        Constraint(
            "topk sentinel", -1, "dsv41_sparse_contract.py",
            "-1 and ONLY -1; any other negative value is an index to the "
            "kernel, and its gather has no upper bound at all"),
        Constraint(
            "packed index tile", 262144, "dsv41_packed_index.py::_MAX_TILE_VALUES",
            "the packed E2M1 score path restores BF16 in tiles of this many "
            "values; a larger tile is an unbounded temporary"),
    ]


def kernel_shape(cfg: dict, tp: int = 4, spec_k: "int | None" = None) -> "KernelShape":
    """DeepSeek-V4.1-Flash's kernel shape (engine/base/kernel_shape) from its text config.

    Per rank at TP=4 the way placement.py lays it out: query heads split by heads over ONE shared
    `head_dim`-wide compressed key per position (num_key_value_heads 1 -- the MQA-over-a-latent form
    the ST MLA lane serves, so `kind` "mla"); routed experts whole and one to a rank (EP: `experts // tp`
    local, `inter_local` the model's); no linear attention (`linear` None: the KDA lanes do not apply);
    the CED indexer at `index_n_heads` x `index_head_dim` over compressor pools of the largest compress
    ratio; `spec_k` the MTP head's `num_nextn_predict_layers`. The experts are FP4 e2m1 in groups of 32
    per row with E8M0 scales -- the MXFP4 weight layout -- multiplied by FP8 activations
    (engine/modules/quant.fp4_gemm), so quant "mxfp4-a8", not the b12x lane's NVFP4 group-16: the
    admission table refuses that lane for this model, by name. DSv4.1 is outside the engine's scope (CHARTER D5); this derivation
    exists so the wizard can judge a second real checkpoint.
    """
    from engine.base.kernel_shape import Attention, Comm, Indexer, KernelShape, MoE
    hidden = cfg["hidden_size"]
    ratios = [r for r in cfg["compress_ratios"] if r > 0]
    dense = cfg.get("intermediate_size") or 0
    return KernelShape(
        comm=Comm(world=tp, hidden=hidden), hidden=hidden, hc=cfg["hc_mult"], tp=tp,
        attention=Attention(kind="mla", heads=cfg["num_attention_heads"] // tp, head_dim=cfg["head_dim"],
                            kv_heads=max(1, cfg["num_key_value_heads"] // tp),
                            sink=True),                        # modules/sparse_attention.sparse_attn: attn_sink
        linear=None,
        indexer=Indexer(heads=cfg["index_n_heads"], head_dim=cfg["index_head_dim"],
                        pool=max(ratios) if ratios else 1, topk=cfg["index_topk"], compress="ced"),
        moe=MoE(experts=cfg["n_routed_experts"], experts_local=cfg["n_routed_experts"] // tp, hidden=hidden,
                inter=cfg["moe_intermediate_size"], inter_local=cfg["moe_intermediate_size"],
                topk=cfg["num_experts_per_tok"], quant="mxfp4-a8", activation=cfg.get("hidden_act", "silu"),
                swiglu_limit=cfg.get("swiglu_limit"), dense_inter_local=dense // tp),
        spec_k=cfg["num_nextn_predict_layers"] if spec_k is None else spec_k,
        hc_variant="split_sinkhorn")                           # modules/hyper_connection.hc_split_sinkhorn


def kernel_shape_of(ckpt: "str | Path", tp: int = 4) -> "KernelShape":
    """The shape wizard's entry (engine/base/kernel_shape.derive_for): the checkpoint's text config, derived."""
    return kernel_shape(json.loads((Path(ckpt) / "config.json").read_text())["text_config"], tp)


def chunk_for(repo: "str | Path", token_budget: int, draft_slots: int = 0) -> int:
    """The largest legal prefill chunk inside a token budget.

    This is DSv4.1's version of GLM's floor((MAX_BATCHED - K) / 2304) * 2304.
    Written as one function so that the answer is never three side effects deep
    again -- if a chunk is not what you expected, print this.
    """
    align = next(c.value for c in constraints(repo) if c.name == "chunk alignment")
    return base_chunk_for(align, token_budget, draft_slots)


def report(repo: "str | Path", budgets=(2048, 4096, 8192, 9216, 16384)) -> str:
    cs = constraints(repo)
    width = max(len(c.name) for c in cs)
    out = ["  constraints -- a shape that breaks one of these computes the wrong",
           "  thing or reads out of bounds. None of them is a preference.", ""]
    for c in cs:
        out.append(f"  {c.name:<{width}}  {str(c.value):>8}   [{c.source}]")
        out.append(f"  {'':<{width}}            {c.bites}")
    align = next(c.value for c in cs if c.name == "chunk alignment")
    out.append("")
    out.append(f"  legal prefill chunks (alignment {align}):")
    out.append(f"    {'token budget':>13}  {'no drafter':>11}  {'with 3 MTP slots':>18}")
    for b in budgets:
        out.append(f"    {b:>13}  {chunk_for(repo, b):>11,}  "
                   f"{chunk_for(repo, b, 3):>18,}")
    return "\n".join(out)


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="/home/choiceoh/models/DeepSeek-V4.1-Flash")
    args = parser.parse_args(argv)
    print(report(args.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
