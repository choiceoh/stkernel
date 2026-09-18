"""Hold the batched QSA selection (modules/attention.QSA.allowed) to the per-query form it replaced, mask for
mask, and time the two.

The batched path pools, normalises, rotates and scores every block the step can reach ONCE; the per-query form
did all of it again for each of the step's `t` queries, through modules/sparse_indexer.qsa_select (which stays as
the reference the transformers oracle is held to). Nothing about the SELECTION moves: each query's top-k still
runs over its own prefix of the scores, the vector qsa_select would have scored.

`_was` below is the previous implementation, verbatim, so the two are timed as whole `allowed` calls -- the
projection and the state write included. Runs anywhere torch does: no kernel, no image.

What the times are NOT: evidence about the engine (CHARTER D17). The served lane selects in engine/kernels/qsa
and never calls this module; these are the reference's own costs -- what an oracle test, a CPU composition or a
probe judging a ported lane against the reference pays per QSA layer."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.base.composition import State
from engine.modules.attention import QSA, Attention, Query
from engine.modules.norm import rmsnorm_unit_offset
from engine.modules.rotary import apply_rope, rope_tables
from engine.modules.sparse_indexer import qsa_select


def _was(sel: QSA, at: Query):
    """QSA.allowed before the batching: one qsa_select per query, each rebuilding every block key."""
    cos_all, sin_all = at.tables
    t, eps = at.t, at.feature.eps
    iq, ik = torch.split(torch.nn.functional.linear(at.xs, at.w("index_qk")),
                         [sel.index_heads * sel.index_head_dim, sel.index_head_dim], dim=-1)
    iq = apply_rope(rmsnorm_unit_offset(iq.reshape(t, sel.index_heads, sel.index_head_dim),
                                        at.w("index_q_norm"), eps), cos_all[at.ctx:], sin_all[at.ctx:])
    at.state.put_rows(at.layer, "qsa_raw_keys", at.seq, ik)
    raw_keys = at.state.rows(at.layer, "qsa_raw_keys", at.seq, at.total)
    allowed = torch.zeros(t, at.total, dtype=torch.bool, device=at.xs.device)
    for j in range(t):
        allowed[j, qsa_select(iq[j], raw_keys, at.ctx + j, sel.ratio, sel.budget // sel.ratio, cos_all, sin_all,
                              at.w("index_k_norm"), eps)] = True
    return allowed


def _feature(hidden: int, index_heads: int, index_head_dim: int, budget: int, ratio: int, eps: float,
             dev: str, dtype):
    g = torch.Generator(device=dev).manual_seed(0)
    w = {"index_qk": torch.randn(index_heads * index_head_dim + index_head_dim, hidden, generator=g,
                                 device=dev, dtype=dtype) * 0.05,
         "index_q_norm": torch.randn(index_head_dim, generator=g, device=dev, dtype=dtype) * 0.1,
         "index_k_norm": torch.randn(index_head_dim, generator=g, device=dev, dtype=dtype) * 0.1}
    return Attention(form="gqa", heads=4, kv_heads=4, head_dim=index_head_dim, rotary_dim=index_head_dim,
                     theta=10000.0, eps=eps, qk_norm="rms_unit_offset", gate="channel",
                     select=QSA(index_heads=index_heads, index_head_dim=index_head_dim, budget=budget, ratio=ratio),
                     weights=lambda layer, name: w[name], dtype=str(dtype).replace("torch.", ""))


def _time(fn, dev: str, budget_ms: float = 120.0) -> float:
    """Best of three rounds, each sized to `budget_ms` -- a t=1 step runs in microseconds, and this box's GPU
    also carries a desktop, so a fixed small repeat count measures the neighbours."""
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    fn(); sync()
    start = time.perf_counter(); fn(); sync()
    once = max((time.perf_counter() - start) * 1e3, 1e-4)
    repeat = max(3, min(2000, int(budget_ms / once)))
    best = float("inf")
    for _ in range(3):
        sync(); start = time.perf_counter()
        for _ in range(repeat):
            fn()
        sync()
        best = min(best, (time.perf_counter() - start) / repeat * 1e3)
    return best


def case(t: int, ctx: int, *, dev: str, dtype=torch.float32, index_heads: int = 2, index_head_dim: int = 32,
         budget: int = 2048, ratio: int = 4, hidden: int = 256, eps: float = 1e-6, timed: bool = True) -> bool:
    total = ctx + t
    feature = _feature(hidden, index_heads, index_head_dim, budget, ratio, eps, dev, dtype)
    sel = feature.select
    g = torch.Generator(device=dev).manual_seed(7)
    xs = torch.randn(t, hidden, generator=g, device=dev, dtype=dtype) * 0.5
    before = torch.randn(ctx, index_head_dim, generator=g, device=dev, dtype=dtype) if ctx else None
    cos, sin = rope_tables(torch.arange(total, device=dev), index_head_dim, 10000.0, dtype=dtype)

    def fresh():
        """A step whose sequence already holds `ctx` rows -- the tensors are made once, above."""
        state = State()
        if before is not None:
            state.put_rows(0, "qsa_raw_keys", 0, before)
        return Query(feature=feature, layer=0, seq=0, xs=xs, ctx=ctx, total=total, state=state,
                     tables=(cos, sin), q_resid=None)

    want, got = _was(sel, fresh()), sel.allowed(fresh())
    same = bool(torch.equal(got, want))
    kept = int(want.sum(1).float().mean())
    shape = f"  t={t:>5} ctx={ctx:>6} blocks={total // ratio:>5} budget={budget:>4} kept/query~{kept:>5}"
    verdict = "mask identical" if same else "MASKS DIFFER"
    if timed:
        was_ms = _time(lambda: _was(sel, fresh()), dev)
        now_ms = _time(lambda: sel.allowed(fresh()), dev)
        print(f"{shape}  per-query {was_ms:8.2f} ms -> batched {now_ms:7.2f} ms  "
              f"({was_ms / max(now_ms, 1e-9):5.1f}x)  {verdict}")
    else:
        print(f"{shape}  {str(dtype).replace('torch.', '')}  {verdict}")
    if not same:
        print(f"    first differing queries: {(got != want).any(1).nonzero().flatten()[:4].tolist()}")
    return same


CASES = ((1, 0), (7, 0), (40, 0), (1, 4095), (8, 4088), (512, 0), (2048, 0), (1, 16383), (4096, 0))


def main() -> int:
    ok = True
    # both devices: a t=1 step on the GPU is launch-bound, so the CPU run is what says whether the batched path
    # does less WORK or merely fewer launches
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        print(f"QSA selection, batched against the per-query form ({dev}):")
        for t, ctx in CASES:
            ok &= case(t, ctx, dev=dev)
        # the oracle's geometry (tests/test_engine_composition: budget 8 over blocks of 4, so a 40-token prefill keeps
        # 2 blocks of 10) and its chunked continuation, in both activation dtypes -- bf16 is where the casts bite
        for dtype in (torch.float32, torch.bfloat16):
            for t, ctx in ((40, 0), (17, 23), (1, 39), (333, 1000)):
                ok &= case(t, ctx, dev=dev, dtype=dtype, budget=8, timed=False)
    print("  OK" if ok else "  FAILED: the batched path does not select what the per-query form selects")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
