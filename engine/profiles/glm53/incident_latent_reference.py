"""Private controls: FP32 torch sparse MLA over native FP8 or original BF16 KV.

The paired reference arms use identical attention arithmetic. They preserve
the native projections, pool selection, absorption and every other layer.
Only isolated eager requests with an uncached prefix are supported.
"""
from contextlib import contextmanager

import torch


class LatentReference:
    def __init__(self, identity, *, bf16, capacity=65536, chunk=32):
        self.identity, self.bf16 = identity, bf16
        self.capacity, self.chunk = capacity, chunk
        self.values, self.ends = {}, {}

    def write(self, layer, values, step):
        if getattr(step, 'captured', False) or len(step.segments) != 1:
            raise ValueError('latent reference requires one eager sequence')
        s = step.segments[0]
        if s.start != 0 or len(values) != s.length or values.dtype != torch.bfloat16:
            raise ValueError('latent reference expects the entire normalized BF16 segment')
        end = s.ctx + s.length
        if s.ctx != self.ends.get(layer, 0) or end > self.capacity:
            raise ValueError('latent sidecar requires an uncached contiguous prefix within capacity')
        if layer not in self.values:
            self.values[layer] = torch.empty((self.capacity, values.shape[-1]),
                                             dtype=values.dtype, device=values.device)
        self.values[layer][s.ctx:end].copy_(values)
        self.ends[layer] = end

    def logical_slots(self, layer, slots, valid, step, caches):
        block_row, block, stride, offset = caches.token_map(layer, step.segments[0].seq)
        active = torch.arange(slots.shape[1], device=slots.device)[None, :] < valid[:, None]
        physical = torch.where(active, slots, 0).long()
        if block_row is None:
            logical = physical
        else:
            count = (self.ends[layer] + block - 1) // block
            blocks = block_row[:count].long()
            inverse = torch.full((caches.latent(layer).shape[0] // stride,), -1,
                                 dtype=torch.int64, device=slots.device)
            inverse[blocks] = torch.arange(count, device=slots.device)
            within = physical % stride - offset
            if bool(((within < 0) | (within >= block)).masked_fill(~active, False).any()):
                raise ValueError('selected physical slot belongs to a different layer')
            logical = inverse[physical // stride] * block + within
        if bool(((logical < 0) | (logical >= self.ends[layer])).masked_fill(~active, False).any()):
            raise ValueError('selected physical slot is outside the written latent sidecar')
        return logical.masked_fill(~active, 0), active

    def context(self, layer, q, latent, slots, valid, step, caches, scale):
        logical, active = self.logical_slots(layer, slots, valid, step, caches)
        result = torch.empty_like(q)
        for start in range(0, len(q), self.chunk):
            end = min(start + self.chunk, len(q))
            mask = active[start:end]
            if self.bf16:
                rows = self.values[layer][logical[start:end]].float()
            else:
                # Read the actual cache writer's FP8 bytes, not a re-encoding
                # of the sidecar which could hide a physical addressing bug.
                physical = slots[start:end].long().masked_fill(~mask, 0)
                rows = latent[physical].float()
            rows.masked_fill_(~mask[:, :, None], 0)
            scores = torch.bmm(q[start:end].float(), rows.transpose(1, 2)) * scale
            scores.masked_fill_(~mask[:, None, :], float('-inf'))
            probabilities = torch.softmax(scores, dim=-1)
            probabilities.masked_fill_(~mask.any(dim=-1)[:, None, None], 0)
            result[start:end] = torch.bmm(probabilities, rows).to(q.dtype)
        return result


@contextmanager
def control(net, identity, *, bf16):
    cached = getattr(net, '_incident_latent_store', None)
    if cached is None or cached.identity != identity:
        cached = LatentReference(identity, bf16=bf16)
        net._incident_latent_store = cached
    previous = getattr(net, 'incident_latent_reference', None)
    net.incident_latent_reference = cached
    try:
        yield
    finally:
        net.incident_latent_reference = previous
