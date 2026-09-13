"""Stateless uniforms: every draw is a function of what it is for, never of what came before it
(45차, 2026-09-13; the open suspect of §97's seven rows against six).

The engine used to hold one Philox stream per rank, seeded alike at boot, and advance it a
data-dependent number of times: `rand(count)` for a rich row's live grammar span, `rand(K)` for
a draft walk, `rand(n, K)` then `rand(n)` for a verification, a `multinomial` per position inside
a captured graph. The stream position at step t was therefore a function of the whole history of
the process -- of every span every matcher computed and every path every step took -- and one
difference anywhere, one ULP in a probability that moved a span by one, put two ranks on two
streams for as long as the fleet stood. Nothing compared them; the tokens simply differed.

Here a uniform is a hash. A ROW KEY is what the draws of one row at one point hang off -- the
seed, the row's admission nonce (or a request's own seed alone), and how many tokens the row had
generated when the step began -- and a draw is that key mixed with a WORD naming its purpose
(draft walk, target pick, verification, the correction draw, the rich sampler) and its position.
Every input is replicated: the nonce is the admission count every rank makes in the same order,
the generation count rides the step. So the same draw on four ranks is the same number whatever
came before, on the host or on the device, eager or captured, and replaying a step needs its key,
not the process's history (D12).

Purpose and position share one word so that everything a decode step draws for a row -- K for the
walk, K for the verification, one for the correction -- is ONE mix over [rows, 2K+1] on the
device (`step_block`), computed inside the captured drafter graph: the chain pays a copy for the
key's inputs, not a hash's worth of kernel launches.

The hash is splitmix64's finalizer, written twice -- in Python integers for the host and in int64
tensor arithmetic for the device -- and pinned to agree bit for bit: two's-complement wraparound
multiplication is what torch does, so only the right shifts need care (arithmetic on a signed
int64: the mask makes them logical). A uniform is the top 53 bits over 2^53, rounded to float32
once, the same rounding in both.
"""
from __future__ import annotations

import struct

MASK = (1 << 64) - 1
C1, C2, C3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB

# what a draw is for; a purpose's draws are numbered by position within the step
DRAFT, PICK, VERIFY, FRESH, RICH = 1, 2, 3, 4, 5


def mix(x: int) -> int:
    """splitmix64's finalizer over a 64-bit word (unsigned arithmetic)."""
    x = (x + C1) & MASK
    x = ((x ^ (x >> 30)) * C2) & MASK
    x = ((x ^ (x >> 27)) * C3) & MASK
    return x ^ (x >> 31)


def row_key(seed: int, nonce: int, generation: int) -> int:
    """The word a row's draws hang off at one generation count."""
    h = mix(int(seed) & MASK)
    h = mix(h ^ (int(nonce) & MASK))
    return mix(h ^ (int(generation) & MASK))


def word(purpose: int, position: int) -> int:
    """Purpose in the high half, position in the low half: distinct for every (purpose, position)."""
    if not 0 <= int(position) < (1 << 32) or not 0 < int(purpose) < (1 << 16):
        raise ValueError("a draw's purpose is 1..65535 and its position 0..2^32-1")
    return (int(purpose) << 32) | int(position)


def _float32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def uniform(key: int, purpose: int, position: int) -> float:
    """One uniform in [0, 1) as the float32 the device produces."""
    z = mix((int(key) ^ word(purpose, position)) & MASK)
    return _float32((z >> 11) * 2.0 ** -53)


def uniforms(key: int, purpose: int, count: int, start: int = 0) -> "list[float]":
    return [uniform(key, purpose, start + i) for i in range(count)]


def step_layout(k: int) -> "list[tuple[int, int]]":
    """What a decode step draws for one sampled row, in block order: the walk's K, the verification's K,
    the correction's one. `step_block` computes exactly these; the host path asks for the same words."""
    return [(DRAFT, i) for i in range(k)] + [(VERIFY, i) for i in range(k)] + [(FRESH, 0)]


# ---- the same on tensors ------------------------------------------------------------------------------------
def _signed(c: int) -> int:
    c &= MASK
    return c - (1 << 64) if c >= (1 << 63) else c


def _lsr(t, bits: int):
    """Logical shift right of an int64 tensor: torch shifts arithmetically, the mask drops the sign fill."""
    return (t >> bits) & ((1 << (64 - bits)) - 1)


def mix_tensor(t):
    """`mix` over an int64 tensor, bit for bit."""
    t = t + _signed(C1)
    t = (t ^ _lsr(t, 30)) * _signed(C2)
    t = (t ^ _lsr(t, 27)) * _signed(C3)
    return t ^ _lsr(t, 31)


def row_keys(seed: int, nonces, generations):
    """`row_key` for every row: `nonces` and `generations` are int64 tensors [n]. The seed's own mix is a
    host constant, so the device does two mixes, not three."""
    h = mix_tensor(nonces ^ _signed(mix(int(seed) & MASK)))
    return mix_tensor(h ^ generations)


def _to_uniform(z):
    import torch
    return (_lsr(z, 11).to(torch.float64) * 2.0 ** -53).to(torch.float32)


def uniform_tensor(keys, purpose: int, count: int, start: int = 0):
    """[n, count] float32: row i's positions start..start+count-1 of `purpose` under keys[i]."""
    import torch
    positions = torch.arange(start, start + count, device=keys.device, dtype=torch.int64) + (int(purpose) << 32)
    return _to_uniform(mix_tensor(keys.view(-1, 1) ^ positions.view(1, -1)))


def step_block(seed: int, nonces, generations, k: int):
    """[n, 2k+1] float32 in `step_layout(k)` order -- one mix over the block, and nothing here moves a
    tensor from the host, so a captured graph may compute it (the positions come from `arange`)."""
    import torch
    keys = row_keys(seed, nonces, generations)
    at = torch.arange(2 * k + 1, device=keys.device, dtype=torch.int64)
    words = torch.where(at < k, at + (DRAFT << 32),
                        torch.where(at < 2 * k, at - k + (VERIFY << 32), torch.full_like(at, FRESH << 32)))
    return _to_uniform(mix_tensor(keys.view(-1, 1) ^ words.view(1, -1)))


def _selfcheck() -> None:
    import torch
    k = row_key(0, 7, 12)
    host = [uniform(k, p, i) for p, i in step_layout(3)]
    dev = step_block(0, torch.tensor([7]), torch.tensor([12]), 3)[0].tolist()
    assert host == dev, (host, dev)
    assert all(0.0 <= u < 1.0 for u in host)
    distinct = {uniform(row_key(0, 7, g), p, i) for g in range(4) for p in (DRAFT, PICK, VERIFY, FRESH, RICH) for i in range(6)}
    assert len(distinct) == 120
    print("  draws: host and device agree bit for bit; distinct by generation, purpose and position OK")


if __name__ == "__main__":
    _selfcheck()
