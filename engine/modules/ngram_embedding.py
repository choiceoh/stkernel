"""Hashed n-gram embeddings injected into residual streams (module): Qwen3.8's per-layer embedding (PLE).

A token's n-grams (orders 2..ngram_size, the tokens before it within the same EOS-delimited segment) are hashed into
`heads_per_ngram` tables each -- a table's size is a distinct prime above `ngram_vocab_size_base`, its multipliers come
from splitmix64 of a seed -- and the looked-up rows are concatenated. The injection projects them to a value [H] and a
key per stream [hc, H]; each normalised stream gates the value by its key (sqrt-sign of the scaled dot product, then a
sigmoid); a dilated depthwise conv (kernel `conv`, dilation ngram_size, silu) over the normalised gated values adds
local context, and the sum is added to the streams (transformers qwen4_exp Qwen4ExpTextNGramEmbedding and
Qwen4ExpTextPLELayer, op for op).

Per sequence it carries the last ngram_size-1 token ids (EOS before the first token) and the conv's last
(conv-1)*ngram_size inputs. The table itself is the model's largest (47.68 GiB for Qwen3.8, fp8 in the checkpoint);
where its rows come from -- memory or NVMe (modules/lookup_table) -- is the `table` callable's business.
"""
from __future__ import annotations

import math

import torch

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def multipliers(unigram_vocab: int, ngram_size: int, table_index: int, seed: int) -> torch.Tensor:
    """[ngram_size] odd int64 multipliers of one PLE layer's hashes."""
    half = max(1, (((1 << 63) - 1) // max(unigram_vocab, 1)) // 2)
    base = seed + PRIME_1 * table_index
    return torch.tensor([2 * (splitmix64((base + SPLITMIX_GAMMA * (i + 1)) & MASK64) % half) + 1
                         for i in range(ngram_size)], dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def head_tables(ngram_size: int, heads_per_ngram: int, base: int, table_index: int) -> "tuple[torch.Tensor, torch.Tensor, int]":
    """(sizes [heads], offsets [heads], total rows) of one PLE layer's hash heads: head h of layer i is the
    (i*heads + h + 1)-th prime above base - 1."""
    heads = (ngram_size - 1) * heads_per_ngram
    sizes, offsets, total = [], [], 0
    for h in range(heads):
        size = _nth_prime_after(base - 1, table_index * heads + h + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return torch.tensor(sizes, dtype=torch.long), torch.tensor(offsets, dtype=torch.long), total


def shift_right_ignore_eos(ids: torch.Tensor, shift: int, eos: int) -> torch.Tensor:
    """ids [L] int64 -> the token `shift` places back in the same EOS-delimited segment, EOS where there is none."""
    if shift == 0:
        return ids
    ids = ids[None]
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device, dtype=torch.long)
    eos_positions = torch.where(ids == eos, positions, -1)
    inclusive = torch.cummax(eos_positions, dim=1).values
    previous = torch.cat([eos_positions.new_full((1, 1), -1), inclusive[:, :-1]], dim=1)
    in_segment = positions.unsqueeze(0) - (previous + 1)
    source = positions - shift
    shifted = ids.gather(dim=1, index=source.clamp_min(0).unsqueeze(0))
    valid = (in_segment >= shift) & (source.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, ids.new_full((), eos))[0]


def ngram_rows(history: torch.Tensor, count: int, mult: torch.Tensor, sizes: torch.Tensor, offsets: torch.Tensor,
               ngram_size: int, heads_per_ngram: int, eos: int) -> torch.Tensor:
    """The table rows of the last `count` tokens of `history` [L] (the carried context then the new tokens):
    int64 [count, (ngram_size-1)*heads_per_ngram]."""
    shifted = [shift_right_ignore_eos(history, s, eos) for s in range(ngram_size)]
    blocks = []
    for order in range(2, ngram_size + 1):
        lo = (order - 2) * heads_per_ngram
        mixed = shifted[0] * mult[0]
        for p in range(1, order):
            mixed = torch.bitwise_xor(mixed, shifted[p] * mult[p])
        rows = torch.remainder(mixed.unsqueeze(-1), sizes[lo:lo + heads_per_ngram].view(1, -1))
        blocks.append(rows + offsets[lo:lo + heads_per_ngram].view(1, -1))
    return torch.cat(blocks, dim=-1)[-count:]


class NGramInjection:
    """PLE as a residual injection (engine/base/composition.Feature): (layer, streams [N, hc*H]) -> addend [N, hc*H].

    `table_index(layer)`: the layer's position in the model's PLE layer list (it seeds the hashes and picks the primes).
    `weights(layer, name)`: key_proj, value_proj, norm_key, norm_query, norm_conv, conv1d ([hc*H, K] or [hc*H, 1, K]).
    `table(layer, rows)`: rows int64 [..., heads] -> [..., heads, head_width] -- `ple_embedding.ngram_embedding` rows."""

    def __init__(self, *, hidden: int, hc: int, ngram_size: int, heads_per_ngram: int, unigram_vocab: int,
                 ngram_vocab_base: int, seed: int, eos: int, conv: int, eps: float, table_index, weights, table,
                 dtype: str = "bfloat16"):
        self.hidden, self.hc, self.ngram_size, self.heads_per_ngram = hidden, hc, ngram_size, heads_per_ngram
        self.unigram_vocab, self.ngram_vocab_base, self.seed, self.eos = unigram_vocab, ngram_vocab_base, seed, eos
        self.conv, self.eps, self.table_index, self.weights, self.table = conv, eps, table_index, weights, table
        self.dtype = dtype
        self._hashes = {}

    def hashes(self, layer: int):
        if layer not in self._hashes:
            index = self.table_index(layer)
            sizes, offsets, _ = head_tables(self.ngram_size, self.heads_per_ngram, self.ngram_vocab_base, index)
            self._hashes[layer] = (multipliers(self.unigram_vocab, self.ngram_size, index, self.seed), sizes, offsets)
        return self._hashes[layer]

    def __call__(self, layer, h, step, state):
        from engine.modules.causal_conv import causal_conv1d
        from engine.modules.norm import rmsnorm_unit_offset
        linear = torch.nn.functional.linear
        w = lambda name: self.weights(layer, name)
        mult, sizes, offsets = self.hashes(layer)
        context = self.ngram_size - 1
        out = torch.empty_like(h)
        for s in step.segments:
            ids = step.ids[s.start:s.start + s.length]
            carried = state.get(layer, "ngram_context", s.seq)
            if carried is None:
                carried = ids.new_full((context,), self.eos)
            history = torch.cat([carried, ids])
            rows = ngram_rows(history, ids.numel(), mult.to(ids.device), sizes.to(ids.device), offsets.to(ids.device),
                              self.ngram_size, self.heads_per_ngram, self.eos)
            embeddings = self.table(layer, rows).flatten(-2)
            key = rmsnorm_unit_offset(linear(embeddings, w("key_proj")), w("norm_key"), self.eps, group=self.hidden)
            value = linear(embeddings, w("value_proj"))
            hs = h[s.start:s.start + s.length]
            query = rmsnorm_unit_offset(hs, w("norm_query"), self.eps, group=self.hidden)
            gate = (key.unflatten(-1, (self.hc, self.hidden)) * query.unflatten(-1, (self.hc, self.hidden))).sum(
                dim=-1, keepdim=True) / math.sqrt(self.hidden)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gated = torch.sigmoid(gate) * value.unsqueeze(-2)
            normed = rmsnorm_unit_offset(gated.flatten(-2), w("norm_conv"), self.eps, group=self.hidden)
            conv_w = w("conv1d")
            local, conv_state = causal_conv1d(normed, conv_w.reshape(conv_w.shape[0], -1), None,
                                              state.get(layer, "ngram_conv", s.seq), "silu", dilation=self.ngram_size)
            out[s.start:s.start + s.length] = gated.flatten(-2) + local
            state.put(layer, "ngram_context", s.seq, history[-context:])
            state.put(layer, "ngram_conv", s.seq, conv_state)
        return out

    def cache_specs(self, layers):
        from engine.base.cache_spec import SlotSpec, _ITEMSIZE
        span = (self.conv - 1) * self.ngram_size
        return [SlotSpec("ple token context", len(layers), (self.ngram_size - 1) * 8, "[ngram_size-1] int64 token ids",
                         key="ngram_context", dtype="int64", shape=(self.ngram_size - 1,)),
                SlotSpec("ple conv state", len(layers), self.hc * self.hidden * span * _ITEMSIZE[self.dtype],
                         f"[hc*H, (kernel-1)*ngram_size] {self.dtype}: the dilated conv's last inputs",
                         key="ngram_conv", dtype=self.dtype, shape=(self.hc * self.hidden, span))]
