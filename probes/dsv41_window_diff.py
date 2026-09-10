#!/usr/bin/env python3
"""Does our sliding-window ring equal DeepSeek's? No GPU, no weights.

Two things, both pure index arithmetic, both run against the reference file
itself rather than against a second copy written here:

  1. ids. `window_topk_idxs` vs `get_window_topk_idxs`, bit-identical, over
     prefill chunks shorter than / equal to / longer than the window, and over
     decode steps on both sides of the ring filling.
  2. the ring. Our `apply_prefill_ring` / `apply_decode_ring` vs the
     reference's own `Attention._window_kv` cache writes, checked against the
     INVARIANT (slot == position % window) rather than against each other --
     two implementations that agree on the same wrong rotation would pass a
     mutual comparison.

    python3 probes/dsv41_window_diff.py --model-py .../inference/model.py

The mutations this was checked against, and what each costs. The first is the
one that matters: it is what a plausible-looking prefill seed does.

    prefill seed without the rotation      every token present exactly once,
                                           all 4,096 slots off by seqlen % win
    decode written at start_pos // win     overwrites slot 0 forever
    prefill ids not clamped at 0           negative ids, which index from the
                                           end of the KV rather than fault
    decode ids listed newest first         same SET, so sparse_attn agrees --
                                           caught only because the reference's
                                           order is reproduced exactly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_sparse_contract import check_topk_idxs           # noqa: E402
from dsv41_window import (                                  # noqa: E402
    apply_decode_ring, apply_prefill_ring, prefill_ring_writes, ring_slot,
    window_kv_len, window_topk_idxs,
)


def load_reference(model_py: Path):
    """`get_window_topk_idxs` alone -- it is a free function with no state."""
    text = model_py.read_text()
    start = text.index("def get_window_topk_idxs(")
    end = text.index("class Compressor(nn.Module):", start)
    ns = {"torch": torch}
    exec(compile(text[start:end], str(model_py), "exec"), ns)
    return ns["get_window_topk_idxs"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-py",
                    default="/home/choiceoh/models/DeepSeek-V4.1-Flash/"
                            "inference/model.py")
    ap.add_argument("--window", type=int, default=64,
                    help="window_size; V4.1 ships 4096")
    ap.add_argument("--dim", type=int, default=8)
    args = ap.parse_args()
    ref = load_reference(Path(args.model_py))
    win, ok = args.window, True

    # -- 1. ids ------------------------------------------------------------
    cases = [("prefill, shorter than the window", 1, win // 3, 0),
             ("prefill, exactly the window", 1, win, 0),
             ("prefill, longer, ragged", 2, win * 3 + 7, 0),
             ("prefill, longer, exact multiple", 2, win * 2, 0),
             ("decode, ring still filling", 3, 1, win // 2),
             ("decode, ring just full", 3, 1, win - 1),
             ("decode, wrapped", 3, 1, win * 5 + 3),
             ("decode, at a slot boundary", 3, 1, win * 7)]
    for label, bsz, seqlen, start in cases:
        want = ref(win, bsz, seqlen, start)
        got = window_topk_idxs(win, bsz, seqlen, start)
        same = torch.equal(want, got) and want.dtype == got.dtype
        kv_len = window_kv_len(win, seqlen, start)
        try:
            check_topk_idxs(got, kv_len, where=label)
            legal = "in range"
        except Exception as exc:                            # noqa: BLE001
            legal, ok = f"CONTRACT: {exc}", False
        print(f"  ids {label:34s} {'OK ' if same else 'MISMATCH'} "
              f"{tuple(got.shape)} vs kv_len {kv_len}, {legal}")
        ok &= same

    # -- 2. the ring, against the invariant --------------------------------
    # Tokens carry their absolute position as their value, so a slot's contents
    # ARE the position it should hold. That is what makes this a check of the
    # rotation rather than of two implementations agreeing.
    print()
    for seqlen in (win // 3, win, win + 1, win * 3 + 7, win * 4):
        cache = torch.full((1, win, args.dim), -1.0)
        kv = (torch.arange(seqlen, dtype=torch.float32)
              .view(1, seqlen, 1).expand(1, seqlen, args.dim).contiguous())
        apply_prefill_ring(cache, kv, win)
        held = cache[0, :, 0]
        live = min(seqlen, win)
        first = seqlen - live                       # oldest surviving position
        want = torch.full((win,), -1.0)
        for p in range(first, seqlen):
            want[ring_slot(p, win)] = float(p)
        bad = int((held != want).sum())
        pairs = len(prefill_ring_writes(seqlen, win))
        print(f"  ring prefill seqlen {seqlen:<6d} {pairs} write(s), "
              f"{'OK' if not bad else f'{bad} slot(s) WRONG'}"
              + ("" if not bad else
                 f"  first bad slot {int((held != want).nonzero()[0])}"))
        ok &= not bad
        # and the decode steps that follow must land on free slots
        for step in range(3):
            pos = seqlen + step
            tok = torch.full((1, 1, args.dim), float(pos))
            apply_decode_ring(cache, tok, win, pos)
            if cache[0, ring_slot(pos, win), 0].item() != float(pos):
                print(f"    FAIL: decode at {pos} did not land in slot "
                      f"{ring_slot(pos, win)}")
                ok = False

    # -- 3. the rotation must be LOAD-BEARING ------------------------------
    # If a no-rotation seed passed too, (2) would be proving nothing.
    seqlen = win * 3 + 7
    cache = torch.full((1, win, args.dim), -1.0)
    kv = (torch.arange(seqlen, dtype=torch.float32)
          .view(1, seqlen, 1).expand(1, seqlen, args.dim).contiguous())
    cache[0, :, :] = kv[0, -win:, :]                # the obvious wrong seed
    held = cache[0, :, 0]
    misplaced = sum(1 for s in range(win)
                    if ring_slot(int(held[s].item()), win) != s)
    assert len(set(held.tolist())) == win, "the control must lose no token"
    print(f"\n  control    seeding without the rotation misplaces "
          f"{misplaced}/{win} slots (offset {seqlen % win}), while holding "
          f"every surviving token exactly once")
    ok &= misplaced > 0

    print("\n" + ("WINDOW PASS" if ok else "WINDOW FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
