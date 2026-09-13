"""Exact vocabulary-parallel greedy selection with one int64 MAX per token.

Order FP32 logits in the high word; the low word breaks ties in favor of
the smallest global token id, just like argmax on the gathered vocabulary.
BF16/FP16 logits convert exactly to FP32. NaNs win argmax, signed zeros tie,
and ranks with no decodable tokens contribute the minimum int64 sentinel.
No random draws and no full-vocabulary communication are required.
"""
import torch


def argmax(local_logits, comm, start: int, decodable: int | None = None):
    if local_logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("vocabulary argmax requires BF16, FP16 or FP32 logits")
    width = local_logits.shape[-1]
    if start < 0 or start + width >= 2**31 or (decodable is not None and decodable <= 0):
        raise ValueError("vocabulary ids must fit nonnegative int32 and contain a valid token")
    valid = width if decodable is None else max(0, min(width, decodable - start))
    if local_logits.is_cuda and local_logits.ndim == 2:
        from engine.kernels.common.vocab_candidates import argmax_key
        key = argmax_key(local_logits, start, valid)
    elif valid:
        value, index = local_logits[..., :valid].float().max(dim=-1)
        value = torch.where(value == 0, torch.zeros_like(value), value)
        value = torch.where(torch.isnan(value), torch.full_like(value, float("nan")), value)
        bits = value.contiguous().view(torch.int32).to(torch.int64)
        ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
        key = (ordered << 32) | (0xffffffff - (index + start))
    else:
        key = torch.full(local_logits.shape[:-1], -(2**63), dtype=torch.int64, device=local_logits.device)
    comm.all_reduce_max(key)
    return 0xffffffff - (key & 0xffffffff)


class CandidateBuffer:
    """A drafter-owned dense merge with only the previously touched columns cleared.

    Allocate before capture; every use on the serving stream completes its
    top-k before the next use. Inactive rows keep their own last packet, so a
    shrink followed by growth cannot leave stale candidates in the vocabulary.
    """
    def __init__(self, rows, vocab, count, device):
        self.dense = torch.full((rows, vocab), float('-inf'), dtype=torch.float32, device=device)
        self.previous = torch.full((rows, count), -(2**63), dtype=torch.int64, device=device)

    def restore(self, gathered, vocab):
        if (gathered.ndim != 2 or gathered.dtype != torch.int64 or gathered.device != self.dense.device
                or gathered.shape[0] > self.dense.shape[0] or gathered.shape[1] != self.previous.shape[1]
                or vocab != self.dense.shape[1] or not gathered.is_contiguous()):
            raise ValueError('candidate buffer requires its declared row, vocabulary and packet geometry')
        from engine.kernels.common.vocab_candidates import restore_reuse
        return restore_reuse(gathered, self.dense, self.previous)


def topk(local_logits, comm, start: int, k: int, decodable: int | None = None, *, workspace=None):
    """Exchange k packed candidates per rank, then retain CUDA topk ordering.

    Each int64 carries an ordered FP32 score and its global token id. Local
    cutoff ties keep the first vocabulary ids, matching the pinned CUDA radix
    selection. Restore candidates at their original vocabulary positions before
    calling topk: selecting directly from the small packet changes tie order
    and can change the drafter's subsequent greedy walk.

    The local selection is over the keys directly (kernels/vocab_candidates.select): they are unique, so the k
    largest are one set, and `sorted=False` says this step does not decide the order. torch's dense topk over
    the restored candidates still does, which is what keeps the tie order pinned (45차 §87).

    This saves network traffic, not the final dense selection workspace. CUDA
    tie equivalence must be rechecked when changing the pinned PyTorch runtime.
    CPU topk has a different, unspecified tie policy; only untied equivalence
    is promised there. NaNs retain their ordering, not their payload bits.
    """
    if local_logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("vocabulary topk requires BF16, FP16 or FP32 logits")
    width = local_logits.shape[-1]
    vocab = width * comm.world_size
    if (width <= 0 or type(k) is not int or not 0 < k <= vocab or
            start < 0 or start % width or start + width > vocab or vocab >= 2**31 or
            (decodable is not None and decodable <= 0)):
        raise ValueError("topk requires equal vocabulary shards, valid k and nonnegative int32 ids")
    valid = width if decodable is None else max(0, min(width, decodable - start))
    sentinel = -(2**63)
    local_k = min(k, valid)
    fused = local_logits.is_cuda and local_logits.ndim == 2
    if local_k:
        if fused:
            # the k largest of a rank's shard, as a set: `sorted=False` says the order here is not the answer,
            # and the merge below decides that. The keys are unique, so the set is one (kernels/vocab_candidates)
            from engine.kernels.common.vocab_candidates import select_logits
            packet = select_logits(local_logits, start, valid, local_k)
            key = None
        else:
            value = local_logits[..., :valid].float().contiguous()
            bits = value.view(torch.int32).to(torch.int64)
            ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
            ordered = torch.where(torch.isnan(value), 0x7fffffff, ordered)
            ids = start + torch.arange(valid, device=value.device, dtype=torch.int64)
            key = (ordered << 32) | (0xffffffff - ids)
        if key is not None:
            packet = key.topk(local_k, dim=-1, sorted=False).values
    else:
        packet = torch.empty((*local_logits.shape[:-1], 0), dtype=torch.int64, device=local_logits.device)
    if local_k < k:
        padding = torch.full((*packet.shape[:-1], k-local_k), sentinel,
                             dtype=torch.int64, device=packet.device)
        packet = torch.cat((packet, padding), dim=-1)
    gathered = comm.all_gather(packet, dim=-1)
    if fused:
        from engine.kernels.common.vocab_candidates import restore
        dense = restore(gathered, vocab) if workspace is None else workspace.restore(gathered, vocab)
        return dense.topk(k, dim=-1)
    ids = 0xffffffff - (gathered & 0xffffffff)
    ordered = gathered >> 32
    bits = torch.where(ordered < 0, ordered ^ 0x7fffffff, ordered).to(torch.int32)
    values = bits.contiguous().view(torch.float32)
    # Give each padded entry a distinct extra column. No duplicate scatter
    # destinations, dynamic boolean indexing, host readback or graph branch.
    padding_ids = vocab + torch.arange(gathered.shape[-1], device=ids.device)
    ids = torch.where(gathered == sentinel, padding_ids, ids)
    dense = torch.full((*values.shape[:-1], vocab+gathered.shape[-1]), float('-inf'),
                       dtype=torch.float32, device=values.device)
    dense.scatter_(-1, ids, values)
    return dense[..., :vocab].topk(k, dim=-1)
