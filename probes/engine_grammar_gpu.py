#!/usr/bin/env python3
"""The grammar mask on a real device: the kernel, the transfer, and what they cost (45차 §31).

One GPU, no model, no fleet -- everything here is the part of structured output the CPU suite cannot reach:

  1  `Grammars.warm` pays xgrammar's Triton JIT, the way boot does. If the kernel cannot build, this dies here.
  2  `prepare` fills a step's masks into pinned staging and crosses once, non_blocking, behind an event.
  3  `apply` writes -inf from the packed words. Judged against the same mask expanded on the host: the sets
     must be identical, or the kernel is not doing what the CPU path did.
  4  the cost, against the path this replaced (expand to a vocabulary of bool, then masked_fill).

  usage (inside the ST image, one GPU):
    bash probes/run_engine_probe.sh probes/engine_grammar_gpu.py --ckpt /home/choiceoh/models/glm53-redhat-nvfp4
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time


def main() -> int:
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, "/repo")
    from engine.base.grammar import Grammars, available

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/home/choiceoh/models/glm53-redhat-nvfp4")
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if not available():
        print("xgrammar is not installed in this image: structured output is not served here")
        return 1
    if not torch.cuda.is_available():
        print("this probe needs a device")
        return 1
    device = torch.device("cuda")
    import json
    from pathlib import Path
    ends = json.loads((Path(a.ckpt) / "generation_config.json").read_text())["eos_token_id"]
    ends = ends if isinstance(ends, list) else [ends]

    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    g = Grammars(tok, a.vocab, stop_token_ids=ends)
    print(f"tokenizer + TokenizerInfo: {time.perf_counter() - t:.1f}s   vocab {a.vocab}  words {g.words}"
          f"   stop tokens {ends}")

    t = time.perf_counter()
    g.warm(device)
    torch.cuda.synchronize()
    print(f"1  warm (Triton JIT + one apply): {time.perf_counter() - t:.2f}s")

    matchers = [g.matcher({"type": "json_object"}, a.k + 2) for _ in range(a.rows)]
    opener = tok.encode("{", add_special_tokens=False)[0]
    for m in matchers:
        m.advance([opener])                                   # inside an object: the masks are real ones
    drafts = tok.encode('"a": 1}', add_special_tokens=False)[: a.k]
    rows = [(i, matchers[i], list(drafts)) for i in range(a.rows)]

    t = time.perf_counter()
    masks = g.prepare(rows, device)
    host = time.perf_counter() - t
    torch.cuda.synchronize()
    live = [masks.live(i, a.k + 1) for i in range(a.rows)]
    print(f"2  prepare: {host * 1e3:.3f} ms on the host, one transfer, live positions {live}")

    # 3 -- the kernel against the host's own expansion of the same words
    bad = 0
    for i in range(a.rows):
        n = live[i]
        logits = torch.zeros(n, a.vocab, device=device)
        masks.apply(i, logits)
        off = masks.filled[i][0]
        words = g.landing[off: off + n].cpu()
        shift = torch.arange(32, dtype=torch.int32)
        want = words.unsqueeze(-1).bitwise_right_shift(shift).bitwise_and_(1).reshape(n, -1)[:, : a.vocab].ne(0)
        got = ~torch.isinf(logits.cpu())
        if not torch.equal(want, got):
            bad += int((want != got).sum())
        if i == 0:
            allowed = want[0].nonzero().flatten().tolist()
            print(f"3  row 0 position 0 allows {len(allowed)} ids, e.g. {[tok.convert_ids_to_tokens(x) for x in allowed[:6]]}")
    print(f"3  kernel vs the host's expansion: {'IDENTICAL' if not bad else f'{bad} DISAGREEING ids -- FAIL'}")

    # 4 -- what it costs, against the path it replaced
    def timed(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return min(ts), statistics.median(ts)

    n = live[0]
    logits = torch.zeros(n, a.vocab, device=device)
    off = masks.filled[0][0]
    shift = torch.arange(32, dtype=torch.int32, device=device)

    def kernel():
        masks.apply(0, logits)

    def expand():
        words = g.landing[off: off + n]
        bits = words.unsqueeze(-1).bitwise_right_shift(shift).bitwise_and_(1)
        m = bits.ne(0).reshape(n, -1)[:, : a.vocab]
        for j in range(n):
            logits[j] = logits[j].masked_fill(~m[j], float("-inf"))

    k_min, k_med = timed(kernel)
    e_min, e_med = timed(expand)
    print(f"4  one row of {n} positions: kernel {k_min:.3f}/{k_med:.3f} ms   bool expansion + masked_fill "
          f"{e_min:.3f}/{e_med:.3f} ms   (min/median)")

    t = time.perf_counter()
    for _ in range(20):
        g.prepare(rows, device)
    torch.cuda.synchronize()
    print(f"5  20 steps of prepare (fill + crossing, event-guarded): {(time.perf_counter() - t) * 1e3 / 20:.3f} ms/step")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
