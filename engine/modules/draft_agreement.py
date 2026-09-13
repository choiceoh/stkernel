"""One rank-authoritative draft walk, including its actual sampling distribution."""
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
