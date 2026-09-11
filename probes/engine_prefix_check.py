"""Prefix reuse on real GLM layers: the second prompt that shares two chunks with the first must
generate the same tokens as it would without the cache, from one third of the prefill.

Runs the engine's local composition (four ranks as threads, layers 0-4, reference lanes -- the cache
is a runtime property, not a kernel one): prompt A = two prefill chunks + a tail; seq 0 prefills all
of it (checkpoints at both chunk boundaries); seq 1 = the same two chunks + a different tail adopts
the second boundary and prefills only its tail; seq 2 = seq 1's prompt with the cache cleared is the
oracle. Judged: seq 1 == seq 2 token for token (greedy), seq 1 started at 2 chunks, and its prefill
took a fraction of seq 2's.

    bash probes/run_engine_probe.sh probes/engine_prefix_check.py [--layers 0-4] [--tail 300]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                                     # noqa: E402

from engine.base.comm import LocalTP                             # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.profiles.glm53 import boot, facts, lanes as lane_tables   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT))
    ap.add_argument("--drafter-dir", default=str(boot.drafter_mod.DRAFTER))
    ap.add_argument("--kv-gib", type=float, default=boot.KV_GIB)
    ap.add_argument("--tail", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=8)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    print(f"  box: {facts.check_box()}")
    F = facts.load(a.ckpt_meta)
    chunk = boot.chunk_for(F.block, boot.TOKEN_BUDGET, 0)
    g = torch.Generator().manual_seed(11)
    shared = torch.randint(100, 20000, (2 * chunk,), generator=g).tolist()
    tail_a = torch.randint(100, 20000, (a.tail,), generator=g).tolist()
    tail_b = torch.randint(100, 20000, (a.tail,), generator=g).tolist()
    prompt_a, prompt_b = shared + tail_a, shared + tail_b
    out = {}

    def rank_main(comm):
        rec = Recorder(f"rank{comm.rank}")
        _, net, caches, engine, runner = boot.build(comm, layers, lane_tables.reference(), a.ranks, a.kv_gib, boot.MAX_SEQS, False, rec,
                                                    max_new=a.max_new, ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir)

        def run(seq, ids):
            engine.add(seq, ids, max_new=a.max_new)
            runner.submit(seq, len(ids), now=0.0, ids=ids)
            computed = runner.state.computed[seq]
            t0 = time.perf_counter()
            while seq in runner.state.waiting or seq in runner.state.running:
                if runner.step(now=0.0) is None:
                    break
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            tokens = list(engine.generated(seq))
            runner.evict(seq) if seq in runner.idle else None
            engine.forget(seq)
            return computed, secs, tokens

        r = {}
        r["a"] = run(0, prompt_a)
        r["b_cached"] = run(1, prompt_b)
        runner.prefix.clear()
        r["b_plain"] = run(2, prompt_b)
        r["cache"] = {"hits": runner.prefix.hits, "misses": runner.prefix.misses, "evictions": runner.prefix.evictions,
                      "snapshot_MiB": round(caches.snapshot_bytes / 2**20, 1)}
        r["free_blocks"] = runner.kv.available == runner.kv.num_blocks
        return r

    tp = LocalTP(facts.TP)
    outs = tp.run(rank_main)
    r0 = outs[0]
    same_ranks = all(o["b_cached"][2] == r0["b_cached"][2] and o["b_plain"][2] == r0["b_plain"][2] for o in outs)
    ca, cb, cp = r0["a"], r0["b_cached"], r0["b_plain"]
    print(f"  layers {a.layers}, chunk {chunk}: prompt A {len(prompt_a)} tokens, prompt B shares {2 * chunk} with it, tail {a.tail}")
    print(f"    A (no cache):     computed at submit {ca[0]:>6}, {ca[1]:.2f} s, tokens {ca[2]}")
    print(f"    B with the cache: computed at submit {cb[0]:>6}, {cb[1]:.2f} s, tokens {cb[2]}")
    print(f"    B without:        computed at submit {cp[0]:>6}, {cp[1]:.2f} s, tokens {cp[2]}")
    print(f"    cache {r0['cache']}; all blocks reclaimable at the end: {r0['free_blocks']}; four ranks agree: {same_ranks}")
    ok = cb[0] == 2 * chunk and cp[0] == 0 and cb[2] == cp[2] and len(cb[2]) == a.max_new and same_ranks and r0["free_blocks"]
    print("\n  " + ("PASS: the cached prefix yields the tokens the full prefill yields" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
