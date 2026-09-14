"""One rank-authoritative draft walk, including its actual sampling distribution, and one verdict on it."""
import torch


def agree_walk(comm, drafts, candidates=None, probabilities=None):
    """Keep every TP rank's next target input and rejection denominator identical.

    Vocabulary candidates are gathered, but a drafter's final hidden projection
    is local. Independently quantized replicas need not choose the same walk.
    The greedy case broadcasts token IDs directly. Sampled walks carry their
    compact support and the exact FP32 probability bits in one integer packet;
    a vocabulary-sized distribution never crosses the network.
    """
    if (candidates is None) != (probabilities is None):
        raise ValueError('sampled draft agreement needs support and probabilities together')
    if comm.world_size == 1:
        return drafts if candidates is None else (drafts, candidates, probabilities)
    if drafts.dtype != torch.int64:
        raise TypeError('draft agreement requires int64 token IDs')
    if candidates is None:
        return comm.broadcast_tensor(drafts)
    if (candidates.dtype != torch.int64 or probabilities.dtype != torch.float32
            or candidates.shape != probabilities.shape or candidates.shape[:-1] != drafts.shape):
        raise ValueError('sampled walk requires int64 support and FP32 probabilities for every draft position')
    tokens, support = drafts.numel(), candidates.numel()
    packet = torch.cat((drafts.reshape(-1), candidates.reshape(-1),
                        probabilities.contiguous().view(torch.int32).reshape(-1).to(torch.int64)))
    comm.broadcast_tensor(packet)
    return (packet[:tokens].view_as(drafts),
            packet[tokens:tokens + support].view_as(candidates),
            packet[tokens + support:].to(torch.int32).view(torch.float32).view_as(probabilities))


def agree_verdict(comm, accepted, tokens):
    """Rank zero's verification of a sampled block on every rank: how many drafts it kept and what it committed.

    `agree_walk` gives the ranks one draft and one draft distribution, and the target's arrives through an
    all-gather, but the verdict over them is still arithmetic each rank does alone -- and the correction draw's
    cumsum (base/sampler._inverse_cdf) is not bit-stable. torch lists CUDA float cumsum as nondeterministic, and
    on the GB10 fleet it is exactly where the correction draw runs alone: one [1, 154880] row summed 300 times
    gave 296 different results, while [2, 4 or 8, 154880] blocks and the same row under
    `use_deterministic_algorithms` gave one each; at the same uniforms 1 draw in 1,500 named a different token.
    So one sequence at temperature 1 split the ranks, and the host readback check (#848) stopped serving with a
    CollectiveDivergence 351, 1,353 and 14,470 steps in (2026-09-14), while four-row stress never did. Greedy
    verification never needed this: its argmax meets in a MAX collective. accepted [n] and tokens [n, K+1] int64;
    returns contiguous tensors holding rank zero's.
    """
    if comm.world_size == 1:
        return accepted, tokens
    if accepted.dtype != torch.int64 or tokens.dtype != torch.int64 or tokens.dim() != 2 \
            or accepted.shape != tokens.shape[:1]:
        raise ValueError('verdict agreement wants int64 accepted [n] and tokens [n, K+1]')
    n, t = tokens.shape
    # tokens first: both slices of the one flat packet stay contiguous, which the commit kernel's strides assume
    packet = torch.cat((tokens.reshape(-1), accepted))
    comm.broadcast_tensor(packet)
    return packet[n * t:], packet[: n * t].view(n, t)
