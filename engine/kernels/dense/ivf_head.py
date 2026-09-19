"""An FP8 vocabulary head's argmax from a fraction of its rows: an inverted-file index over them (kernels, dense).

A drafter only needs the head's argmax, and a speculative step verifies whatever it drafts -- a wrong draft costs
acceptance, never output. Qwen3.8's MTP chain reads the rank's whole head (62,080 x 2,560 FP8, 159 MB) three times a
K=3 step for three argmaxes, 2.1 ms of the draft graph's 3.97 (q38draft-0919a). This reads a few MB instead:

    build     once, at boot: the rows k-means clustered (balanced: no cluster over `cap` rows), each cluster's centroid
              in BF16 and its member row ids
    argmax    the rows scored against the centroids (skinny_gemv), the `probes` best clusters a row, and every member
              row of those scored exactly as the head scores it -- the row's block-128 FP8 quantisation (fp8.quantize)
              against the weight's e4m3 rows and scales, rounded to the head's BF16 -- into the vocabulary argmax key
              (vocab_candidates: `(ordered score << 32) | (0xffffffff - id)`, ties to the lower id), one int64 atomic
              max a row. The keys a rank returns are the full head's kind, so the ranks' all-reduce max is unchanged.

Where the true argmax row is in a probed cluster the answer is the full head's (up to the order of an FP32 sum); where
it is not, the draft is the best probed row. With every cluster probed it is the full head's argmax (the test).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
KEY_MIN = -(1 << 63)


@dataclass
class IVFHead:
    """The index over one rank's head rows. `weight`: the head's (e4m3 [N, K], fp32 [N/128, K/128]) as FP8Linear holds
    it; `centroids` [C, K] BF16; `members` [C, cap] int32 row ids, -1 past a cluster's size; `probes` clusters a row."""
    weight: tuple
    centroids: torch.Tensor
    members: torch.Tensor
    probes: int

    @property
    def clusters(self) -> int:
        return self.centroids.shape[0]

    @property
    def cap(self) -> int:
        return self.members.shape[1]

    def read_bytes(self) -> int:
        """Bytes one argmax row reads at most: the centroids and `probes` full clusters of FP8 rows."""
        return self.centroids.numel() * 2 + self.probes * self.cap * self.centroids.shape[1]


@triton.jit
def _ivf_score(XQ, XS, WQ, WS, PROBED, MEMBERS, OUT, START, VALID, K: tl.constexpr, P: tl.constexpr,
               CAP: tl.constexpr, CAP2: tl.constexpr):
    r, p = tl.program_id(0), tl.program_id(1)
    c = tl.load(PROBED + r * P + p)
    j = tl.arange(0, CAP2)
    row = tl.load(MEMBERS + c * CAP + j, mask=j < CAP, other=-1)
    live = (row >= 0) & (row < VALID)
    safe = tl.where(live, row, 0)
    acc = tl.zeros((CAP2,), dtype=tl.float32)
    for kb in range(K // 128):
        ks = kb * 128 + tl.arange(0, 128)
        x = tl.load(XQ + r * K + ks).to(tl.float32)
        w = tl.load(WQ + safe[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
        w = w.to(tl.float32)
        s = tl.load(WS + (safe // 128) * (K // 128) + kb, mask=live, other=0.0)
        xs = tl.load(XS + r * (K // 128) + kb)
        acc += tl.sum(w * x[None, :], 1) * s * xs
    value = acc.to(tl.bfloat16).to(tl.float32)                     # the head's logits are BF16
    value = tl.where(value == 0, 0.0, value)                        # vocab_candidates' canonical key
    bits = value.to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.where(bits < 0, bits ^ 0x7fffffff, bits)
    ordered = tl.where(value != value, 0x7fc00000, ordered)
    key = (ordered << 32) | (0xffffffff - (START + row.to(tl.int64)))
    key = tl.where(live, key, -9223372036854775808)
    tl.atomic_max(OUT + r, tl.max(key, 0))


def _dequantized(weight, start: int, stop: int) -> torch.Tensor:
    """Rows [start, stop) of the FP8 head as FP32: each e4m3 value times its 128 x 128 block's scale."""
    wq, ws = weight
    scale = ws.repeat_interleave(128, 0)[start:stop].repeat_interleave(128, 1)[:, :wq.shape[1]]
    return wq[start:stop].float() * scale


def build(weight, *, clusters: int, probes: int, rows: "int | None" = None, iters: int = 6, slack: float = 1.15,
          candidates: int = 8, chunk: int = 8192, seed: int = 0) -> IVFHead:
    """Cluster the first `rows` rows of `weight` (default all) into `clusters` of at most ceil(slack * rows / clusters)
    rows: k-means in FP32 on the dequantised rows, then each row, most confident first, into the nearest of its
    `candidates` nearest centroids that has room (else the nearest with room), and every centroid the mean of its
    members. Deterministic for a seed."""
    wq, ws = weight
    n = wq.shape[0] if rows is None else rows
    if not 1 <= probes <= clusters <= n:
        raise ValueError(f"ivf_head: 1 <= probes ({probes}) <= clusters ({clusters}) <= rows ({n})")
    device = wq.device
    # the rows in BF16 (a rank's 62,080 x 2,560 is 318 MB; FP32 would be twice), distances as BF16 products, sums FP32
    w = torch.cat([_dequantized(weight, a, min(a + chunk, n)).to(torch.bfloat16) for a in range(0, n, chunk)])

    def distances(a, centroids):
        c = centroids.to(torch.bfloat16)
        return (-2.0 * (w[a:a + chunk] @ c.t()).float() + centroids.square().sum(1))

    def means(assign, centroids):
        sums = torch.zeros(clusters, w.shape[1], dtype=torch.float32, device=device)
        for a in range(0, n, chunk):
            sums.index_add_(0, assign[a:a + chunk], w[a:a + chunk].float())
        return sums, torch.bincount(assign, minlength=clusters).float()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    centroids = w[torch.randperm(n, generator=gen)[:clusters].to(device)].float()
    for _ in range(iters):
        assign = torch.cat([distances(a, centroids).argmin(1) for a in range(0, n, chunk)])
        sums, counts = means(assign, centroids)
        empty = counts == 0
        centroids = torch.where(empty[:, None], centroids, sums / counts.clamp_min(1)[:, None])
        if bool(empty.any()):                                           # an empty cluster restarts at a random row
            refill = torch.randint(0, n, (int(empty.sum()),), generator=gen).to(device)
            centroids[empty] = w[refill].float()
    cap = -(-int(slack * n) // clusters)
    dist, near = [], []
    for a in range(0, n, chunk):
        top = distances(a, centroids).topk(candidates, dim=1, largest=False)
        dist.append(top.values)
        near.append(top.indices)
    dist, near = torch.cat(dist).cpu(), torch.cat(near).cpu()
    order = torch.argsort(dist[:, 0], stable=True).tolist()
    room = [cap] * clusters
    owner = [-1] * n
    for i in order:
        for c in near[i].tolist():
            if room[c]:
                owner[i], room[c] = c, room[c] - 1
                break
    left = [i for i in order if owner[i] < 0]
    if left:                                                            # the nearest centroid with room, over all
        full = torch.tensor([r == 0 for r in room])
        for i in left:
            d = (-2.0 * (w[i].float() @ centroids.t()) + centroids.square().sum(1)).cpu()
            d[full] = float("inf")
            c = int(d.argmin())
            owner[i], room[c] = c, room[c] - 1
            full[c] = room[c] == 0
    owner_t = torch.tensor(owner, device=device)
    members = torch.full((clusters, cap), -1, dtype=torch.int32)
    fill = [0] * clusters
    for i, c in enumerate(owner):
        members[c, fill[c]] = i
        fill[c] += 1
    sums, counts = means(owner_t, centroids)
    return IVFHead(weight, (sums / counts.clamp_min(1)[:, None]).to(torch.bfloat16).contiguous(), members.to(device),
                   probes)


def argmax_key(head: IVFHead, h: torch.Tensor, start: int, valid: int) -> torch.Tensor:
    """h [R <= 16, K] BF16 -> the vocabulary argmax key [R] int64 over the probed rows below `valid` (rank-local ids +
    `start`), the kind vocab.argmax all-reduces."""
    from engine.kernels.common import skinny_gemv
    from .fp8 import quantize
    rows, k = h.shape
    if not 1 <= rows <= MAX_ROWS or h.dtype != torch.bfloat16 or k != head.centroids.shape[1]:
        raise ValueError(f"ivf_head takes 1..{MAX_ROWS} BF16 rows of the head's width: {tuple(h.shape)}")
    h = h.contiguous()
    scores = skinny_gemv.gemv(h, head.centroids, (16, 256, 1, 4, 3)) if h.is_cuda else h.float() @ head.centroids.float().t()
    probed = scores.float().topk(head.probes, dim=1, sorted=False).indices.to(torch.int32).contiguous()
    q, s = quantize(h) if h.is_cuda else _quantize_cpu(h)
    out = torch.full((rows,), KEY_MIN, dtype=torch.int64, device=h.device)
    wq, ws = head.weight
    _ivf_score[(rows, head.probes)](q, s, wq, ws, probed, head.members, out, start, valid, K=k, P=head.probes,
                                    CAP=head.cap, CAP2=triton.next_power_of_2(head.cap), num_warps=4)
    return out


def _quantize_cpu(x: torch.Tensor):
    """fp8.quantize's arithmetic in torch, for the Triton interpreter: block-128 amax, a power-of-two scale."""
    m, k = x.shape
    g = x.float().view(m, k // 128, 128)
    amax = g.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (g / scale[..., None]).view(m, k).to(torch.float8_e4m3fn), scale.contiguous()


def exact_key(weight, h: torch.Tensor, start: int, valid: int) -> torch.Tensor:
    """The same arithmetic over every row below `valid`, in torch (the reference `argmax_key` meets with every cluster
    probed) -> [R] int64."""
    from .fp8 import quantize
    q, s = quantize(h) if h.is_cuda else _quantize_cpu(h)
    wq, ws = weight
    k = h.shape[1]
    x = q.float().view(-1, k // 128, 128)
    w = wq[:valid].float().view(valid, k // 128, 128)
    part = torch.einsum("rbk,nbk->rnb", x, w)                               # [R, N, K/128] block partial sums
    scale = s[:, None, :] * ws.repeat_interleave(128, 0)[:valid][None, :, :]
    value = (part * scale).sum(-1).to(torch.bfloat16).float()
    value = torch.where(value == 0, torch.zeros_like(value), value)
    bits = value.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
    ids = torch.arange(valid, device=h.device, dtype=torch.int64) + start
    return ((ordered << 32) | (0xffffffff - ids)).amax(1)


__all__ = ["IVFHead", "MAX_ROWS", "build", "argmax_key", "exact_key"]
