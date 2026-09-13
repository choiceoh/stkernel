"""Hashed n-gram memory injected into residual streams (module): the family Qwen3.8's per-layer embedding (PLE) and
DeepSeek-V4.1's engram share -- one hash, one gate-and-write, and the axes the two differ on.

A token's n-grams (orders 2..ngram_size, the tokens before it) are hashed into `heads` prime-sized bucket ranges per
order; the looked-up rows, concatenated, are projected to one key per residual stream and a shared value; each stream
gates the value by how well it matches its key (a normalised dot product, a signed square root, a sigmoid), and the
gated values are written into the streams. Around that:

  the hash (`NGramHash`)
    window      the lookback, newest first, stops -- and every farther slot takes the `pad` key -- at the start of the
                sequence (a DEAD slot, which is what the carried context holds before the first token), at a dead token
                (DeepSeek-V4.1's image spans), and with `eos` set, at an EOS token one or more places back (Qwen3.8's
                segments: a token's n-grams never reach into the previous document; the EOS itself still belongs to
                the segment it ends)
    token_map   None (keys are token ids: Qwen3.8) | a table id -> key where tokens that normalise alike collapse
                (DeepSeek-V4.1: `normalized_token_map`, 129,280 tokens -> 99,092 keys)
    multipliers one odd int64 per lookback, bounded so key * multiplier cannot overflow: splitmix64 of a seed and the
                layer's table index (`NGramHash.splitmix`, Qwen3.8) | numpy's default_rng(10007 * model layer id)
                (`NGramHash.rng`, DeepSeek-V4.1)
    buckets     for both: the consecutive primes at or above `base`, table t's order o head h taking the
                (t * (n-1) * heads + o * heads + h + 1)-th; rows = rolling XOR of key * multiplier over the order's
                lookbacks, modulo its prime, plus the running offset of the ranges before it

  the gate and the write (`NGramInjection`)
    projections key [hc*H] and value [H] as two matrices (Qwen3.8's key_proj/value_proj) or one split (DeepSeek's wkv)
    norm_order  "separate": key and query RMS-normalised per stream with their weights, rounded to the activation
                dtype, then the dot (transformers qwen4_exp Qwen4ExpTextPLELayer) | "joint": the stream, the product
                of the two weights and the key multiplied in fp32, times the product of the two rsqrt terms (the vendor
                DeepSeek-V4.1 `Engram.forward`)
    norm_offset the norm weights enter as (1 + w) (Qwen3.8) | as w (DeepSeek-V4.1)
    signed_sqrt what sqrt(|dot|) takes as its sign: "sign" (x.sign(): zero at an exact zero, Qwen3.8) | "copysign"
                (DeepSeek-V4.1)
    conv        0 (DeepSeek-V4.1 writes the gated values) | a kernel: the gated values, RMS-normalised per stream, go
                through a depthwise causal conv dilated by ngram_size with silu, added to them (Qwen3.8)

Per sequence it carries the last ngram_size-1 token ids (DEAD before the first token) and, with a conv, the conv's last
(kernel-1)*ngram_size inputs. The tables are the models' largest tensors (Qwen3.8 47.68 GiB fp8; DeepSeek-V4.1 two
[384M x 256] fp8 tables): where their rows come from -- memory or NVMe (modules/lookup_table) -- and how they are
dequantised (`block_fp8_rows` for DeepSeek's per-32 e8m0 scales; the Qwen3.8 loader's scalar scale) is the `table`
callable's business.

Held on the CPU (tests/test_engine_ngram_family.py): the PLE variant to transformers qwen4_exp (the layer and its
hash ids; the whole model by tests/test_engine_composition.py), the engram variant to the vendor DeepSeek-V4.1
inference code (engram.py NgramHashState / EngramLayout / compute_hash_multipliers / build_compressed_token_map and
model.py ParallelEngramEmbedding / Engram, sha-pinned in profiles/dsv41/caches.py), bit for bit where the arithmetic
is the same.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

DEAD = -1                                   # a slot with no token: before the sequence, or a dead (image) token
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007


# ---------------------------------------------------------------------------------------------------------------------
# the hash
# ---------------------------------------------------------------------------------------------------------------------

def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def multipliers(unigram_vocab: int, ngram_size: int, table_index: int, seed: int) -> torch.Tensor:
    """[ngram_size] odd int64 multipliers of one PLE layer's hashes (Qwen3.8: splitmix64 of seed + 10007 * table)."""
    half = max(1, (((1 << 63) - 1) // max(unigram_vocab, 1)) // 2)
    base = seed + PRIME_1 * table_index
    return torch.tensor([2 * (splitmix64((base + SPLITMIX_GAMMA * (i + 1)) & MASK64) % half) + 1
                         for i in range(ngram_size)], dtype=torch.long)


def multipliers_rng(layer_id: int, ngram_size: int, vocab: int) -> torch.Tensor:
    """[ngram_size] odd int64 multipliers of one engram layer (DeepSeek-V4.1 compute_hash_multipliers: numpy's
    default_rng(10007 * the model layer id), bounded by the key vocabulary)."""
    import numpy as np
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    values = np.random.default_rng(10007 * layer_id).integers(low=0, high=bound, size=(ngram_size,), dtype=np.int64)
    return torch.tensor(values * 2 + 1)


_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin (exact below 3.3e24, far past any table size)."""
    if value < 2:
        return False
    for p in _WITNESSES:
        if value % p == 0:
            return value == p
    d, s = value - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _WITNESSES:
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


@functools.lru_cache(maxsize=64)
def consecutive_primes(base: int, count: int) -> "tuple[int, ...]":
    """The first `count` primes at or above `base`, in order."""
    out, candidate = [], base
    while len(out) < count:
        if _is_prime(candidate):
            out.append(candidate)
        candidate += 1
    return tuple(out)


def head_tables(ngram_size: int, heads_per_ngram: int, base: int, table_index: int) -> "tuple[torch.Tensor, torch.Tensor, int]":
    """(sizes [heads], offsets [heads], total rows) of one table's bucket ranges: head h of table t is the
    (t * heads + h + 1)-th prime at or above base (Qwen3.8's `_find_nth_prime_after(base - 1, ...)`; DeepSeek-V4.1's
    `find_next_prime` walk with its seen set gives the same sequence)."""
    heads = (ngram_size - 1) * heads_per_ngram
    sizes = list(consecutive_primes(base, (table_index + 1) * heads)[table_index * heads:])
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    return torch.tensor(sizes, dtype=torch.long), torch.tensor(offsets, dtype=torch.long), total


def ngram_windows(history: torch.Tensor, count: int, ngram_size: int, eos: "int | None" = None):
    """The lookbacks of the last `count` tokens of `history` [L] (the carried context, then the new tokens; DEAD for
    no token): (tokens [count, n] newest first, blocked [count, n]). A slot is blocked from the first lookback that
    falls before the history, lands on DEAD, or (with `eos`) lands on EOS one or more places back -- and every farther
    slot with it."""
    length = history.numel()
    positions = torch.arange(length - count, length, device=history.device)
    blocked = torch.zeros(count, dtype=torch.bool, device=history.device)
    tokens, blocks = [], []
    for shift in range(ngram_size):
        at = positions - shift
        source = history[at.clamp_min(0)]
        stop = (at < 0) | (source == DEAD)
        if eos is not None and shift > 0:
            stop = stop | (source == eos)
        blocked = blocked | stop
        tokens.append(source)
        blocks.append(blocked)
    return torch.stack(tokens, dim=-1), torch.stack(blocks, dim=-1)


@dataclass(frozen=True)
class NGramHash:
    """One table's hash: the window rule, the keys, the multipliers and the bucket ranges (the module docstring)."""
    ngram_size: int
    heads: int                                  # per n-gram order
    multipliers: torch.Tensor                   # [ngram_size] int64
    sizes: torch.Tensor                         # [(ngram_size-1) * heads] int64 primes
    offsets: torch.Tensor                       # [(ngram_size-1) * heads] int64
    pad: int                                    # the key a blocked slot takes
    eos: "int | None" = None
    token_map: "torch.Tensor | None" = None     # [vocab] int64 id -> key

    @property
    def total(self) -> int:
        return int(self.sizes.sum())

    @classmethod
    def splitmix(cls, *, ngram_size: int, heads: int, unigram_vocab: int, base: int, table_index: int, seed: int,
                 eos: int) -> "NGramHash":
        """Qwen3.8's PLE: keys are token ids, EOS closes a segment and pads it."""
        sizes, offsets, _ = head_tables(ngram_size, heads, base, table_index)
        return cls(ngram_size, heads, multipliers(unigram_vocab, ngram_size, table_index, seed), sizes, offsets,
                   pad=eos, eos=eos)

    @classmethod
    def rng(cls, *, ngram_size: int, heads: int, base: int, table_index: int, layer_id: int, vocab: int,
            token_map: "torch.Tensor | None", pad: int) -> "NGramHash":
        """DeepSeek-V4.1's engram: keys through the normalised token map (`vocab` keys), no document boundary, `pad`
        the key of the pad token."""
        if token_map is not None and int(token_map.max()) + 1 != vocab:
            raise ValueError(f"the token map has {int(token_map.max()) + 1} keys, the hash is bounded by {vocab}")
        sizes, offsets, _ = head_tables(ngram_size, heads, base, table_index)
        return cls(ngram_size, heads, multipliers_rng(layer_id, ngram_size, vocab), sizes, offsets, pad=pad,
                   token_map=token_map)

    def rows(self, history: torch.Tensor, count: int, dead: "torch.Tensor | None" = None) -> torch.Tensor:
        """int64 [count, (ngram_size-1) * heads]: the table rows of the last `count` tokens of `history` [L] (ids;
        DEAD for no token; `dead` [L] marks tokens that take no part in an n-gram)."""
        if dead is not None:
            history = torch.where(dead, torch.full_like(history, DEAD), history)
        tokens, blocked = ngram_windows(history, count, self.ngram_size, self.eos)
        keys = tokens if self.token_map is None else self.token_map.to(tokens.device)[tokens.clamp_min(0)]
        keys = torch.where(blocked, torch.full_like(keys, self.pad), keys)
        products = keys * self.multipliers.to(keys.device)
        sizes, offsets = self.sizes.to(keys.device), self.offsets.to(keys.device)
        rolling, out = products[:, 0], []
        for i in range(1, self.ngram_size):
            rolling = torch.bitwise_xor(rolling, products[:, i])
            lo, hi = (i - 1) * self.heads, i * self.heads
            out.append(torch.remainder(rolling[:, None], sizes[lo:hi][None]) + offsets[lo:hi][None])
        return torch.cat(out, dim=-1)


def normalized_token_map(tokenizer) -> "tuple[list[int], int]":
    """(id -> key, number of keys): tokens that normalise alike -- NFKC, NFD, accents stripped, lowercased, runs of
    whitespace to one space, stripped (a token that is exactly one space kept) -- share a key; a token that decodes to a
    partial UTF-8 byte is keyed by its raw form. DeepSeek-V4.1's `build_compressed_token_map` (inference/engram.py,
    MIT), step for step: keys are numbered in the order they first appear."""
    from tokenizers import Regex, normalizers
    sentinel = "\ue000"                   # a private-use char: a token that is exactly one space survives Strip()
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    backend = tokenizer.backend_tokenizer
    keys: dict = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = keys.setdefault(key, len(keys))
    return lookup, len(keys)


def block_fp8_rows(weight: torch.Tensor, scale: torch.Tensor, rows: torch.Tensor, block: int) -> torch.Tensor:
    """Rows of a table stored as e4m3 values with one e8m0 scale per `block` elements, dequantised to bf16 --
    DeepSeek-V4.1's ParallelEngramEmbedding at world 1 (the vendor casts to bf16 here, so the engram's value path is
    bf16 whatever the activations are)."""
    values = torch.nn.functional.embedding(rows, weight)
    scales = torch.nn.functional.embedding(rows, scale)
    values = values.float().unflatten(-1, (-1, block)) * scales.float().unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------------
# the gate and the write
# ---------------------------------------------------------------------------------------------------------------------

VARIANTS = {
    "ple": dict(norm_order="separate", norm_offset=True, signed_sqrt="sign"),          # Qwen3.8 (+ its conv kernel)
    "engram": dict(norm_order="joint", norm_offset=False, signed_sqrt="copysign", conv=0),   # DeepSeek-V4.1
}

SCHEMES = {
    "qwen4_exp": {"key": "key_proj", "value": "value_proj", "q_norm": "norm_query", "k_norm": "norm_key",
                  "conv_norm": "norm_conv", "conv": "conv1d"},
    "dsv41": {"kv": "wkv", "q_norm": "q_weight", "k_norm": "k_weight"},
}


def named(scheme: str, source):
    """canonical name -> tensor over `source(checkpoint name)`; KeyError for a name the scheme or the checkpoint lacks."""
    table = SCHEMES[scheme]

    def get(name: str) -> torch.Tensor:
        if name not in table:
            raise KeyError(name)
        return source(table[name])
    return get


class NGramInjection:
    """Hashed n-gram memory as a residual injection (engine/base/composition.Feature): (layer, streams [N, hc*H]) ->
    addend [N, hc*H].

    `hash(layer)` -> the layer's NGramHash. `weights(layer, name)`: key and value, or kv; q_norm, k_norm ([hc*H] or
    [hc, H]); with a conv, conv_norm and conv ([hc*H, K] or [hc*H, 1, K]). `table(layer, rows)`: rows int64
    [..., heads] -> [..., heads, head_width], dequantised."""

    def __init__(self, *, hidden: int, hc: int, ngram_size: int, eps: float, hash, weights, table, conv: int = 0,
                 norm_order: str = "separate", norm_offset: bool = True, signed_sqrt: str = "sign",
                 dtype: str = "bfloat16"):
        if norm_order not in ("separate", "joint"):
            raise ValueError(f"norm_order is 'separate' or 'joint', not {norm_order!r}")
        if signed_sqrt not in ("sign", "copysign"):
            raise ValueError(f"signed_sqrt is 'sign' or 'copysign', not {signed_sqrt!r}")
        if ngram_size < 2 or hc < 1 or conv < 0 or conv == 1:
            raise ValueError("n-gram memory: ngram_size >= 2, hc >= 1, a conv kernel of 0 or >= 2")
        self.hidden, self.hc, self.ngram_size, self.eps, self.conv = hidden, hc, ngram_size, eps, conv
        self.hash, self.weights, self.table, self.dtype = hash, weights, table, dtype
        self.norm_order, self.norm_offset, self.signed_sqrt = norm_order, norm_offset, signed_sqrt
        self._hashes = {}

    def hashes(self, layer: int) -> NGramHash:
        if layer not in self._hashes:
            made = self.hash(layer)
            if made.ngram_size != self.ngram_size:
                raise ValueError(f"layer {layer}: a {made.ngram_size}-gram hash under a {self.ngram_size}-gram memory")
            self._hashes[layer] = made
        return self._hashes[layer]

    # -- pieces ------------------------------------------------------------------------------------------------------
    def _norm(self, x, weight):
        """RMS norm per stream (groups of H) with the weight as (1 + w) or w, rounded to x's dtype."""
        from engine.modules.norm import rmsnorm_unit_offset
        if self.norm_offset:
            return rmsnorm_unit_offset(x, weight.reshape(-1), self.eps, group=self.hidden)
        xf = x.float().unflatten(-1, (-1, self.hidden))
        normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        return weight.reshape(-1) * normed.to(x.dtype)

    def _sqrt(self, dot):
        root = dot.abs().clamp_min(1e-6).sqrt()
        return root * dot.sign() if self.signed_sqrt == "sign" else torch.copysign(root, dot)

    def _gated(self, h, embeddings, w):
        """[N, hc, H]: each stream's gate times the shared value."""
        linear = torch.nn.functional.linear
        hc, H = self.hc, self.hidden
        try:
            key, value = linear(embeddings, w("kv")).split([hc * H, H], dim=-1)
        except KeyError:
            key, value = linear(embeddings, w("key")), linear(embeddings, w("value"))
        if self.norm_order == "separate":
            key = self._norm(key, w("k_norm")).unflatten(-1, (hc, H))
            query = self._norm(h, w("q_norm")).unflatten(-1, (hc, H))
            gate = self._sqrt((key * query).sum(dim=-1, keepdim=True) / math.sqrt(H))
            return torch.sigmoid(gate) * value.unsqueeze(-2)
        key = key.float().unflatten(-1, (hc, H))
        q_w, k_w = w("q_norm").float().reshape(hc, H), w("k_norm").float().reshape(hc, H)
        if self.norm_offset:
            q_w, k_w = 1.0 + q_w, 1.0 + k_w
        weight = q_w * k_w
        hf = h.float().unflatten(-1, (hc, H))
        rstd = torch.rsqrt(hf.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hf * weight * key).sum(-1) * rstd * H ** -0.5
        return torch.sigmoid(self._sqrt(dot)).unsqueeze(-1) * value.float().unsqueeze(-2)

    # -- the feature --------------------------------------------------------------------------------------------------
    def __call__(self, layer, h, step, state):
        from engine.base.composition import put_state
        from engine.modules.causal_conv import causal_conv1d, conv_states
        w = lambda name: self.weights(layer, name)
        made = self.hashes(layer)
        context = self.ngram_size - 1
        out = torch.empty_like(h)
        for s in step.segments:
            ids = step.ids[s.start:s.start + s.length]
            carried = state.get(layer, "ngram_context", s.seq)
            if carried is None:
                carried = ids.new_full((context,), DEAD)
            history = torch.cat([carried, ids])
            rows = made.rows(history, ids.numel())
            embeddings = self.table(layer, rows).flatten(-2)
            gated = self._gated(h[s.start:s.start + s.length], embeddings, w).flatten(-2)
            if self.conv:
                normed = self._norm(gated, w("conv_norm"))
                conv_w = w("conv")
                held_conv = state.get(layer, "ngram_conv", s.seq)
                span = (self.conv - 1) * self.ngram_size
                local, conv_state = causal_conv1d(normed, conv_w.reshape(conv_w.shape[0], -1), None,
                                                  held_conv, "silu", dilation=self.ngram_size)
                gated = gated + local
                put_state(state, layer, "ngram_conv", s, conv_state,
                          conv_states(normed, held_conv, span) if s.verify else None)
            out[s.start:s.start + s.length] = gated
            put_state(state, layer, "ngram_context", s, history[-context:],
                      torch.stack([history[j + 1:j + 1 + context] for j in range(s.length)]) if s.verify else None)
        return out

    def cache_specs(self, layers):
        from engine.base.cache_spec import SlotSpec, _ITEMSIZE
        specs = [SlotSpec("ngram token context", len(layers), (self.ngram_size - 1) * 8,
                          "[ngram_size-1] int64 token ids (DEAD before the first token)",
                          key="ngram_context", dtype="int64", shape=(self.ngram_size - 1,))]
        if self.conv:
            span = (self.conv - 1) * self.ngram_size
            specs.append(SlotSpec("ngram conv state", len(layers), self.hc * self.hidden * span * _ITEMSIZE[self.dtype],
                                  f"[hc*H, (kernel-1)*ngram_size] {self.dtype}: the dilated conv's last inputs",
                                  key="ngram_conv", dtype=self.dtype, shape=(self.hc * self.hidden, span)))
        return specs
