"""What the indexer could read on this incident's own lengths, through the engine's own functions.

The incident's prompt is 50,005 tokens and `index_kpool` is 4, so decode step g sees a sequence length
of 50,005 + g. The appended tail covers `[pool_len * pool_size, seq)` -- EMPTY whenever `seq % 4 == 0`
-- and `index_kpool_always_select_tail` exists to guarantee that the tokens that just arrived are
attended anyway. `tail_pin_pools` is that guarantee: it names the pool that just completed so
`pin_pools_in_logits` can raise it above the row's own maximum before the top-k. On the incident's
boot no code path called either of them (see the evidence receipt), so on those steps the newest four
tokens had to win the top-k on relevance like any other pool.

This prints, for the first decode steps, which pool the pin names and what the selection can read when
the top-k keeps or drops it. Run it where `engine` imports and torch is installed:

    python3 indexer_recency_audit.py --repo . --prompt-tokens 50005
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

KPOOL, TOPK, PROMPT_TOKENS = 4, 2048, 50005


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='.', help='checkout root (the engine package lives under it)')
    ap.add_argument('--prompt-tokens', type=int, default=PROMPT_TOKENS)
    ap.add_argument('--kpool', type=int, default=KPOOL)
    ap.add_argument('--topk', type=int, default=TOPK)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--out', default=None, help='also write the receipt as JSON here')
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from engine.modules.sparse_indexer import select_with_tail, tail_pin_pools

    width = a.topk // a.kpool                       # pools one query may read
    steps = list(range(a.steps + 1))
    seqs = torch.tensor([a.prompt_tokens + g for g in steps], dtype=torch.int32)
    pin = tail_pin_pools(seqs, a.kpool).tolist()
    armed = [g for g, p in zip(steps, pin) if p >= 0]
    assert armed, 'the pin never arms on these lengths: the trigger arithmetic below has moved'
    assert armed[0] == (a.kpool - a.prompt_tokens) % a.kpool, (
        'the first armed step is not the first whole-pool length', armed[0])

    def readable(seq, keep_newest):
        """The token ids the sparse attention may read at `seq`, through the engine's own expansion."""
        newest = seq // a.kpool - 1
        first = newest if keep_newest else newest - 1
        pools = [first] + list(range(0, width - 1))
        out = select_with_tail(torch.tensor([pools], dtype=torch.int32),
                               torch.tensor([seq], dtype=torch.int32), a.kpool)[0].tolist()
        return {token for token in out if token >= 0}

    rows = []
    print(f'prompt {a.prompt_tokens} tokens, kpool {a.kpool}, topk {a.topk} ({width} pools a query)')
    print('step   seq    appended tail  pin    newest pool       kept  dropped')
    for g, p in zip(steps, pin):
        seq = a.prompt_tokens + g
        start = (seq // a.kpool - 1) * a.kpool
        newest = set(range(start, start + a.kpool))
        kept = len(newest & readable(seq, True))
        dropped = len(newest & readable(seq, False))
        tail = seq - (seq // a.kpool) * a.kpool          # tokens the appended tail carries
        rows.append(dict(step=g, seq=seq, appended_tail_tokens=tail, pin=p, newest_pool=start,
                         kept=kept, dropped=dropped))
        print(f'{g:4d}  {seq:7d}  {tail:>13}  {p if p >= 0 else "-":>5}   '
              f'{start:6d}..{start + a.kpool - 1:<6d}  {kept}/{a.kpool}  {dropped}/{a.kpool}')
    whole = [r for r in rows if r['pin'] >= 0]
    print(f'\nappended tail is EMPTY on the {len(whole)} whole-pool steps of these {len(rows)}; '
          f'the pin names the newest pool on exactly those steps (first at generation {armed[0]}, '
          f'seq {a.prompt_tokens + armed[0]}). With the top-k dropping that pool, `dropped` is 0/{a.kpool}: '
          'no path in the pre-fix engine could put it back.')
    if a.out:
        receipt = dict(prompt_tokens=a.prompt_tokens, kpool=a.kpool, topk=a.topk, pools_per_query=width,
                       pin_armed_generations=armed, first_trigger_seq=a.prompt_tokens + armed[0], rows=rows)
        with open(a.out, 'w') as handle:
            json.dump(receipt, handle, indent=2)
        print('WROTE', a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
