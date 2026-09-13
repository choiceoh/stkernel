"""Greedy first-rejection attribution, over the candidates actually proposed.

Only compact results cross the existing decode readback. No vocabulary gather,
new collective, random draw or target-logit modification is introduced.
"""
from collections import Counter
import torch

REASONS = ('all_accepted', 'candidate_miss', 'selector_miss', 'output_boundary', 'policy_modified')


def classify(picks, drafts, support, alive, remaining, ends, temps):
    """Return [accepted-prefix length, reason code] per row; -1 skips inactive/sampled rows.

    EOS and output limits bound which target picks are observable. A mismatch
    after that boundary cannot be blamed on the drafter. All tensor operations
    are device-side and may be captured inside the bounded decode body.
    """
    n, k = drafts.shape
    at = torch.arange(k, device=drafts.device).view(1, k)
    first = torch.where(picks[:, :k] != drafts, at, k).amin(1)
    chosen = picks[:, :k].gather(1, first.clamp_max(k - 1).view(n, 1)).squeeze(1)
    candidates = support.gather(1, first.clamp_max(k - 1).view(n, 1, 1).expand(n, 1, support.shape[2])).squeeze(1)
    covered = (candidates == chosen[:, None]).any(1)
    cause = torch.where(first == k, 0, torch.where(covered, 2, 1))
    end_at = torch.arange(k + 1, device=drafts.device).view(1, k + 1)
    is_end = (picks[:, :, None] == ends[:, None, :]).any(-1)
    visible = torch.minimum(remaining, torch.where(is_end, end_at + 1, k + 1).amin(1))
    cause = torch.where(visible <= first, 3, cause)
    valid = alive & (remaining > 0) & (temps <= 0)
    return torch.stack((first, torch.where(valid, cause, -1)), 1).to(torch.int64)


class DraftDiagnostics:
    def __init__(self, field, k, candidates):
        self.field, self.k = field, k
        self.support = torch.empty(field.shape[0], k, candidates, device=field.device, dtype=torch.int64)
        self.counts = Counter()
        self.sink = None
        self.selector_trace = None
        self.trace_every = 0
        self.trace_steps = Counter()
        self.trace_digest = None
        self.trace_rank = 0

    def enable_selector_trace(self, every, digest, rank=0):
        self.trace_every, self.trace_digest = every, digest
        self.trace_rank = rank
        self.selector_trace = tuple(torch.empty_like(self.support, dtype=torch.float32) for _ in range(2))

    def slot(self, ring):
        stride = self.field[0].numel() * self.field.element_size()
        delta = ring.data_ptr() - self.field.data_ptr()
        if delta < 0 or delta % stride or delta // stride >= self.field.shape[0]:
            raise ValueError('diagnostic ring must belong to its serving field')
        return ring.new_full((1,), delta // stride, dtype=torch.int64)

    def classify(self, picks, b):
        return classify(picks, b['drafts'], self.support.index_select(0, b['real_slot']),
                        b['alive'], b['limit'] - b['generated'], b['ends'], b['temps'])

    def note(self, seqs, contexts, results):
        for seq, context, (prefix, code) in zip(seqs, contexts, results):
            if code < 0:
                continue
            reason = REASONS[code]
            self.counts[reason, prefix] += 1
            if self.sink is not None:
                self.sink(kind='draft_rejection', operation='greedy_first_rejection', phase='decode',
                          seq=int(seq), context=int(context), accepted_prefix=int(prefix),
                          reason=reason, draft_width=self.k)

    def note_sync(self, seq, context, slot, accepted, new, remaining, ends, *, policy_modified=False, trace_eligible=True):
        # agree_walk broadcasts rank zero's actual predecessor path. Other
        # ranks' private paths and constrained target labels cannot fit it.
        if (self.selector_trace is not None and self.sink is not None and self.trace_rank == 0
                and not policy_modified and trace_eligible):
            count = min(self.k, accepted + 1, remaining,
                        next((i + 1 for i, token in enumerate(new) if token in ends), len(new)))
            step = self.trace_steps[seq]
            self.trace_steps[seq] += 1
            if count > 0 and step % self.trace_every == 0:
                # Synchronous only: each slot still holds THIS proposal. Rows
                # after the first mismatch are never treated as teacher labels.
                self.sink(kind='draft_selector', operation='selector_calibration', phase='decode',
                    seq=int(seq), context=int(context), tuning=self.trace_digest, policy_modified=bool(policy_modified),
                    target=list(map(int, new[:count])), candidates=self.support[slot, :count].tolist(),
                    unary=self.selector_trace[0][slot, :count].tolist(),
                    edge=self.selector_trace[1][slot, :count].tolist(), draft_width=self.k)
        if policy_modified:
            code, prefix = 4, -1
        else:
            prefix = accepted
            visible = min(remaining, next((i + 1 for i, token in enumerate(new) if token in ends), len(new)))
            if visible <= accepted:
                code = 3
            elif accepted == self.k:
                code = 0
            else:
                candidates = self.support[slot, accepted].tolist()
                code = 2 if new[accepted] in candidates else 1
        self.note([seq], [context], [[prefix, code]])

    def snapshot(self):
        return [dict(reason=reason, accepted_prefix=prefix, count=count)
                for (reason, prefix), count in sorted(self.counts.items())]

    def close(self):
        # The support is outside the arena; stop holding it after graph teardown.
        self.support = self.field = self.sink = self.selector_trace = None
        self.trace_steps.clear()
