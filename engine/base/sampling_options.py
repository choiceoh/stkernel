"""Sampling policy and committed token counts carried alongside device decode rows.

Only requests that rewrite logits or return logprobs allocate this state. Plain
greedy decoding keeps its existing vocabulary-parallel sampler and burst graph.
The CPU form uses process_logits as its reference; CUDA processes every position
of a row in one launch, with hypothetical draft prefixes and no host readback.
"""
from __future__ import annotations

import torch


def _upload(value, device):
    """Admission must not synchronize a surviving row's queued device work."""
    if torch.device(device).type == "cuda":
        return value.pin_memory().to(device, non_blocking=True)
    return value.to(device)


def needs_device_policy(options, minimum=0, generated=0):
    return (bool(options.get("presence_penalty") or options.get("frequency_penalty"))
            or options.get("repetition_penalty", 1) not in (None, 1)
            or bool(options.get("logit_bias"))
            or options.get("logprobs") is not None or minimum > generated)


class SamplingState:
    TENSORS = ("seen", "counts", "bias", "penalties", "minimum")

    def __init__(self, options, seen, counts, minimum):
        self.options = [dict(o) for o in options]
        self.seen, self.counts = seen, counts
        n, self.vocab = counts.shape
        dev = counts.device
        bias_rows = torch.zeros(n, self.vocab, dtype=torch.float32)
        for row, opts in enumerate(options):
            bias = opts.get("logit_bias") or {}
            if bias:
                ids = torch.tensor(list(bias), dtype=torch.int64)
                bias_rows[row].index_add_(0, ids, torch.tensor(list(bias.values()), dtype=torch.float32))
        self.bias = _upload(bias_rows, dev)
        self.penalties = _upload(torch.tensor([[o.get("repetition_penalty") or 1., o.get("presence_penalty") or 0.,
                                        o.get("frequency_penalty") or 0.] for o in options],
                                       dtype=torch.float32).reshape(n, 3), dev)
        self.minimum = _upload(torch.as_tensor(minimum, dtype=torch.int64), dev)
        self.logprobs = [o.get("logprobs") for o in options]

    @classmethod
    def from_histories(cls, options, tokens, prompt_lengths, minimum, vocab, device):
        from engine.base.sampler import History
        history = History(vocab, "cpu")
        rows = [history.of(i, ids, prompt) for i, (ids, prompt) in enumerate(zip(tokens, prompt_lengths))]
        return cls(options, _upload(torch.stack([r[0] for r in rows]), device),
                   _upload(torch.stack([r[1] for r in rows]), device), minimum)

    def select(self, indices, order):
        out = object.__new__(type(self))
        out.vocab = self.vocab
        out.options = [self.options[i] for i in order]
        out.logprobs = [self.logprobs[i] for i in order]
        for name in self.TENSORS:
            setattr(out, name, getattr(self, name).index_select(0, indices))
        return out

    def join(self, other):
        if self.vocab != other.vocab:
            raise ValueError("sampling rows must share a vocabulary")
        out = object.__new__(type(self))
        out.vocab = self.vocab
        out.options, out.logprobs = self.options + other.options, self.logprobs + other.logprobs
        for name in self.TENSORS:
            setattr(out, name, torch.cat((getattr(self, name), getattr(other, name)), 0))
        return out

    def process(self, logits, drafts, generated, ends, *, start=0, decodable=None, out=None, forces=None):
        n = self.counts.shape[0]
        if (not n or logits.ndim != 2 or logits.shape[0] % n or drafts.ndim != 2 or drafts.shape[0] != n
                or logits.shape[0] // n > drafts.shape[1] + 1
                or generated.shape != (n,) or ends.ndim != 2 or ends.shape[0] != n
                or start < 0 or start + logits.shape[1] > self.vocab):
            raise ValueError("sampling policy, history, draft and logits rows must agree")
        if out is None:
            out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        if out.shape != logits.shape or out.dtype != torch.float32 or out.device != logits.device:
            raise ValueError("processed logits require a matching FP32 output")
        if forces is not None and forces.shape != (logits.shape[0],):
            raise ValueError("forced tokens require one entry per logits position")
        tensors = (drafts, generated, ends, self.counts, self.seen, self.bias, self.penalties, self.minimum)
        if any(x.device != logits.device for x in tensors) or (forces is not None and forces.device != logits.device):
            raise ValueError("sampling inputs must share a device")
        if logits.is_cuda:
            from engine.kernels.common.sampling_options import process
            return process(logits, drafts, generated, ends, self, out, start, decodable, forces)
        from engine.base.sampler import process_logits
        t = logits.shape[0] // n
        for row, opts in enumerate(self.options):
            # Immutable seen bits cover the initial history; counts cover every
            # output committed since, including steps the host has not read yet.
            seen = self.seen[row] | (self.counts[row] > 0)
            for pos in range(t):
                forbid = ends[row][ends[row] >= 0] if generated[row] + pos < self.minimum[row] else None
                force = None if forces is None or forces[row*t+pos] < 0 else int(forces[row*t+pos])
                # The reference processes the global row before slicing a shard.
                full = torch.full((self.vocab,), -torch.inf, device=logits.device)
                full[start:start+logits.shape[1]] = logits[row*t+pos]
                value = process_logits(full, opts, seen, self.counts[row], drafts[row, :pos].tolist(),
                                       decodable, forbid=forbid, force=force)
                out[row*t+pos].copy_(value[start:start+logits.shape[1]])
        return out

    def commit(self, tokens, count):
        """Only the clipped, accepted output updates history. Rejected drafts never do."""
        if tokens.is_cuda:
            from engine.kernels.common.sampling_options import commit
            commit(tokens, count, self.counts)
        else:
            for row in range(tokens.shape[0]):
                ids = tokens[row, :int(count[row])]
                self.counts[row].index_add_(0, ids, torch.ones(len(ids), device=ids.device))

    def logprob_packet(self, processed, picks):
        """Compact device results; host publication waits for the ordinary outcome event."""
        if not any(k is not None for k in self.logprobs):
            return {}
        n, t = picks.shape
        lp = torch.log_softmax(processed.float(), dim=-1).view(n, t, -1)
        k = min(max(want or 0 for want in self.logprobs), lp.shape[-1])
        values, ids = lp.topk(k, dim=-1)
        return dict(logprob=lp.gather(2, picks.unsqueeze(2)).squeeze(2),
                    top_logprobs=values, top_ids=ids)


def warm_sampling_options(logits, vocab, positions, start, decodable):
    """Compile the shard/full-width and forced-token variants before serving.

    Return FP32 outputs for the profile to warm its vocabulary collective too.
    Grammar live spans and stop-set lengths are runtime values of the same kernel.
    This runs during boot, when synchronizing for qualification is intentional.
    """
    if not logits.is_cuda:
        return []
    n, device = logits.shape[0] // positions, logits.device
    opts = [{"repetition_penalty": 1.2, "presence_penalty": .4, "frequency_penalty": -.2,
             "logit_bias": {0: 1.}, "logprobs": min(20, vocab)} for _ in range(n)]
    state = SamplingState.from_histories(opts, [[] for _ in range(n)], [0]*n, [2]*n, vocab, device)
    drafts = torch.zeros(n, positions-1, dtype=torch.int64, device=device)
    generated = torch.zeros(n, dtype=torch.int64, device=device)
    ends = torch.zeros(n, 1, dtype=torch.int64, device=device)
    forces = torch.full((logits.shape[0],), -1, dtype=torch.int64, device=device)
    outputs = []
    for width, offset in dict.fromkeys(((logits.shape[1], start), (vocab, 0))):
        raw = torch.zeros(logits.shape[0], width, dtype=logits.dtype, device=device)
        transformed = state.process(raw, drafts, generated, ends, start=offset, decodable=decodable)
        forced = state.process(raw, drafts, generated, ends, start=offset, decodable=decodable, forces=forces)
        torch.testing.assert_close(transformed, forced, rtol=0, atol=0)
        outputs.append((offset, transformed))
        if width == vocab:
            # FP32 sampler is the input type after options; normal boot warms the head's dtype.
            from engine.base.sampler import rows
            m = logits.shape[0]
            picks = rows(transformed, torch.ones(m, device=device),
                         torch.zeros(m, dtype=torch.int32, device=device), torch.ones(m, device=device),
                         torch.full((m,), .5, device=device), decodable)
            probs = torch.empty_like(transformed)
            for uniform in (None, torch.full((m,), .5, device=device)):
                rows(transformed, torch.ones(m, device=device), torch.zeros(m, dtype=torch.int32, device=device),
                     torch.ones(m, device=device), uniform, decodable, probs)
            state.logprob_packet(transformed, picks.view(n, positions))
            state.commit(picks.view(n, positions), torch.ones(n, dtype=torch.int64, device=device))
            if not bool((state.counts.sum(1) == 1).all()):
                raise ValueError("device sampling history failed boot qualification")
    return outputs
