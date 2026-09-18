#!/usr/bin/env python3
"""The draw contract on the real image: host ints, tensor ints and the fused step kernel must agree.

Every uniform in this engine is an input, computed as a hash of the row key and a
(purpose, position) word (engine/base/draws.py). Three implementations produce it --
the host in Python integers, the generic int64 tensor arithmetic, and the Triton
`decode_inputs.step_block` the captured drafter graph actually runs -- and only tests
pin them together, offline. A runtime disagreement (a different image, a recompiled
or fused kernel) would be silent: the tokens would simply differ.

This checks them against each other on the machine it runs on, for the qualified
geometry, and refuses on the first bit that differs.

    bash bench/fleet.sh run --gpu draw-contract 5 'Check draw implementations' -- \
        bash probes/run_engine_probe.sh probes/engine_draw_contract.py --k 7
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.base import draws                                                    # noqa: E402

PURPOSES = (draws.DRAFT, draws.PICK, draws.VERIFY, draws.FRESH, draws.RICH)
NAMES = {draws.DRAFT: "DRAFT", draws.PICK: "PICK", draws.VERIFY: "VERIFY",
         draws.FRESH: "FRESH", draws.RICH: "RICH"}


def host_words(seed, nonces, generations, k):
    out = {}
    for row, (nonce, generation) in enumerate(zip(nonces, generations)):
        key = draws.row_key(seed, nonce, generation)
        for purpose in PURPOSES:
            for position in range(k + 1):
                out[(row, purpose, position)] = draws.uniform(key, purpose, position)
    return out


def compare(label, got, want):
    """Bit-for-bit: a float32 that differs in the last bit is a different number."""
    bad = [key for key, value in want.items() if float(got[key]) != float(value)]
    print(f"  {label}: {len(want) - len(bad)}/{len(want)} equal"
          f"{'' if not bad else f' -- first difference at {bad[0]}: {got[bad[0]]!r} vs {want[bad[0]]!r}'}")
    return not bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=7, help="speculative width (7 is the qualified cell)")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("  no CUDA device: the device half cannot run here")
        return 2
    nonces = torch.arange(args.rows, dtype=torch.int64, device="cuda")
    generations = torch.arange(100, 100 + args.rows, dtype=torch.int64, device="cuda")
    want = host_words(args.seed, nonces.tolist(), generations.tolist(), args.k)

    # the fused block the captured drafter graph runs: [rows, 2k+1] in step_layout order
    block = draws.step_block(args.seed, nonces, generations, args.k).cpu().tolist()
    fused = {(row, purpose, position): block[row][index]
             for row in range(args.rows)
             for index, (purpose, position) in enumerate(draws.step_layout(args.k))}
    fused_want = {(row, purpose, position): want[(row, purpose, position)]
                  for row in range(args.rows)
                  for purpose, position in draws.step_layout(args.k)}
    ok = compare("host vs step_block (fused kernel)", fused, fused_want)

    # the generic int64 tensor path, for every purpose and k+1 positions
    keys = draws.row_keys(args.seed, nonces, generations)
    for purpose in PURPOSES:
        device = draws.uniform_tensor(keys, purpose, args.k + 1).cpu().tolist()
        got = {(row, purpose, position): device[row][position]
               for row in range(args.rows) for position in range(args.k + 1)}
        purpose_want = {key: value for key, value in want.items() if key[1] == purpose}
        ok = compare(f"host vs uniform_tensor ({NAMES[purpose]})", got, purpose_want) and ok

    # and the words themselves: a collision would make a draw unattributable
    words = {draws.word(purpose, position) for purpose in PURPOSES for position in range(args.k + 1)}
    distinct = len(words) == len(PURPOSES) * (args.k + 1)
    print(f"  words distinct: {distinct} ({len(words)} of {len(PURPOSES) * (args.k + 1)})")
    if args.out:
        Path(args.out).write_text(f"k={args.k} rows={args.rows} seed={args.seed} ok={ok and distinct}\n")
    print("PASS" if ok and distinct else "FAIL")
    return 0 if ok and distinct else 1


if __name__ == "__main__":
    raise SystemExit(main())
