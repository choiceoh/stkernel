"""Known target hard boundaries, carried as a small captured proposal input.

Rich/near-boundary rows already use synchronous decoding. No grammar matcher,
history scan per candidate, target rewrite or extra model pass is introduced.
"""
import torch

MAX_ENDS = 32
WIDTH = MAX_ENDS + 3                  # minimum remaining, force after, force id, unique end ids
EMPTY = (0, 0, -1) + (-1,) * MAX_ENDS


def for_request(engine, seq):
    if not getattr(engine.drafter, 'request_boundaries', False):
        return None
    opts = engine.options.get(seq, {})
    if seq in engine.matchers or opts.get('grammar') is not None:
        return None                  # keep grammar's precedence; never guess its future state
    generated = engine._generated_count(seq)
    remaining = max(0, engine.min_new.get(seq, 0) - generated)
    ends = sorted(set(engine.ends.get(seq, engine.eos))) if remaining else []
    if len(ends) > MAX_ENDS:
        return None                  # optimization is bounded; target still enforces every end id
    force, after = -1, 0
    budget = opts.get('reasoning_budget')
    if (budget is not None and engine.thinking.get(seq, False)
            and generated + engine.drafter.k - 1 >= budget):
        end = opts['reasoning_end']
        valid = engine.decodable if engine.decodable is not None else engine.F.vocab
        if 0 <= end < valid and end not in engine.tokens[seq][engine.prompt_len[seq]:]:
            force, after = end, max(0, budget - generated)
    if not remaining and force < 0:
        return None
    return (remaining, after, force, *ends, *([-1] * (MAX_ENDS - len(ends))))


def tensor(packet, device):
    if isinstance(packet, torch.Tensor):
        if packet.shape != (WIDTH,) or packet.dtype != torch.int64 or packet.device != torch.device(device):
            raise ValueError('draft boundary must be an int64 packet on the proposal device')
        return packet
    if packet is None:
        packet = EMPTY
    if len(packet) != WIDTH or any(type(x) is not int for x in packet):
        raise ValueError('draft boundary has the wrong packet geometry')
    return torch.tensor(packet, dtype=torch.int64, device=device)


def mask_ends(logits, start, packet):
    """Mask a few EOS columns BEFORE local top-k, so valid replacements can enter."""
    if packet is None:
        return logits
    if logits.is_cuda:
        from engine.kernels.draft_boundary import mask_ends as launch
        launch(logits, start, packet)
    else:
        ids = packet[3:] - start
        ids = ids[(ids >= 0) & (ids < logits.shape[1])]
        logits[:min(logits.shape[0], int(packet[0])), ids] = -float('inf')
    return logits


def sampled(unary, cand, anchor, proj, pred, succ, alphas, temperature, uniforms, packet):
    """A constrained sampled walk, with the exact sparse distribution it draws.

    Compute only the actual predecessor's edge, including a forced token that
    was absent from top-k. Forced insertion preserves unique candidate ids.
    """
    from engine.base.sampler import _inverse_cdf
    token = anchor.reshape(())
    active = (packet[2] >= 0) & (packet[2] < pred.shape[0])
    chosen, supports, masses = [], [], []
    for step in range(len(cand)):
        ids = cand[step]
        edge = (succ[ids].float() * (pred[token].float() * proj[step])[None, :]).sum(-1)
        scores = unary[step] + alphas[step] * edge
        blocked = (step < packet[0]) & (packet[3:] == packet[2]).any()
        force = active & (step >= packet[1]) & ~blocked
        matches = ids == packet[2]
        index = matches.long().argmax()
        # If absent, index is zero; if present, do not create a duplicate id.
        ids = ids.clone().scatter(0, index.reshape(1), torch.where(force, packet[2], ids[index]).reshape(1))
        p = torch.softmax(scores / max(temperature, 1e-5), -1)
        forced = torch.zeros_like(p).scatter(0, index.reshape(1), 1.)
        p = torch.where(force, forced, p)
        pick = _inverse_cdf(p.reshape(1, -1), uniforms[step:step+1])[0]
        token = ids[pick]
        active = active & (token != packet[2])
        chosen.append(token)
        supports.append(ids)
        masses.append(p)
    return torch.stack(chosen), torch.stack(supports), torch.stack(masses)
