"""Ancestor-only KDA verification, with FP32 factors and accepted-path materialization.

Each CTA owns a head/value tile for the whole tree. An ancestor update is
therefore produced and consumed by the same CTA; no cross-CTA spin barrier
or speculative full-state array is required. A separate preparation launch
publishes normalized keys and channelwise decay to all value tiles.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from engine.modules.tree_kda import Factors, Topology


@triton.jit
def _conv_sum(X, W, HISTORY, PATH, node, c, C: tl.constexpr, TAPS: tl.constexpr,
              context=0, RING: tl.constexpr = 0):
    acc = tl.full(c.shape, 0., tl.float32)
    for tap in tl.static_range(TAPS):
        source = tl.load(PATH+node*TAPS+tap)
        raw = tl.load(X+source*C+c, (source >= 0) & (c < C), other=0).to(tl.float32)
        if RING:
            position = context+source
            history = tl.load(HISTORY+c*RING+tl.maximum(position, 0) % RING,
                              (source < 0) & (position >= 0) & (c < C), other=0).to(tl.float32)
        else:
            history = tl.load(HISTORY+c*(TAPS-1)+(TAPS-1+source),
                              (source < 0) & (c < C), other=0).to(tl.float32)
        weight = tl.load(W+c*TAPS+tap, c < C, other=0).to(tl.float32)
        acc += (raw+history)*weight
    return acc


@triton.jit
def _conv(X, W, HISTORY, PATH, OUT, C: tl.constexpr, TAPS: tl.constexpr, B: tl.constexpr,
          context=0, RING: tl.constexpr = 0):
    node, block = tl.program_id(0), tl.program_id(1)
    c = block*B + tl.arange(0, B)
    acc = _conv_sum(X, W, HISTORY, PATH, node, c, C, TAPS, context, RING)
    value = tl.div_rn(acc, 1.+libdevice.exp(-acc))
    tl.store(OUT+node*C+c, value, c < C)


def conv(raw, weight, history, topology, *, context=None):
    if (not raw.is_cuda or topology.device != raw.device or topology.taps != weight.shape[1]
            or len(topology.tree.tokens) != len(raw) or weight.shape[0] != raw.shape[1]
            or weight.device != raw.device or history.device != raw.device):
        raise ValueError("native tree convolution needs matching topology and operands")
    out = torch.empty_like(raw)
    _conv[(len(raw), triton.cdiv(raw.shape[1], 256))](raw.contiguous(), weight.contiguous(), history.contiguous(),
        topology.conv, out, raw.shape[1], weight.shape[1], 256,
        context=0 if context is None else context, RING=0 if context is None else history.shape[1],
        num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _prepare(Q, K, G, B, A, BIAS, QF, KF, DEC, BET,
             H: tl.constexpr, D: tl.constexpr, LOWER: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    head = row % H
    q = tl.load(Q + row * D + d, d < D, other=0).to(tl.float32)
    k = tl.load(K + row * D + d, d < D, other=0).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * D ** -.5
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    raw = tl.load(G + row * D + d, d < D, other=0).to(tl.float32)
    bias = tl.load(BIAS + head * D + d, d < D, other=0).to(tl.float32)
    a = tl.exp(tl.load(A + head).to(tl.float32))
    decay = tl.exp(LOWER / (1 + tl.exp(-a * (raw + bias))))
    beta = 1 / (1 + tl.exp(-tl.load(B + row).to(tl.float32)))
    tl.store(QF + row * D + d, q, d < D)
    tl.store(KF + row * D + d, k, d < D)
    tl.store(DEC + row * D + d, decay, d < D)
    tl.store(BET + row, beta)


@triton.jit
def _verify(Q, K, VEC, DEC, BET, INITIAL, PATH, DEPTH, UPDATE, OUT, ORDER, PARENT,
            N: tl.constexpr, H: tl.constexpr, KDIM: tl.constexpr, VDIM: tl.constexpr,
            WIDTH: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, CARRY: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    kk, vv = tl.arange(0, BK), tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < VDIM) & (kk[None, :] < KDIM)
    initial = tl.load(INITIAL + head*KDIM*VDIM + kk[None, :]*VDIM + vv[:, None], mask, other=0)
    state, previous = initial, -1
    for index in range(N):
        node = tl.load(ORDER+index) if CARRY else index
        reset = tl.load(PARENT+node) != previous if CARRY else True
        if reset:
            state = initial
            depth = tl.load(DEPTH + node)
            for level in range(depth):
                ancestor = tl.load(PATH + node * WIDTH + level)
                row = ancestor * H + head
                key = tl.load(K + row*KDIM + kk, kk < KDIM, other=0)
                decay = tl.load(DEC + row*KDIM + kk, kk < KDIM, other=0)
                update = tl.load(UPDATE + row*VDIM + vv, vv < VDIM, other=0)
                state = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [state, decay[None, :]],
                                                   dtype=tl.float32, is_pure=True, pack=1)
                state = tl.fma(key[None, :], update[:, None], state)
        row = node * H + head
        key = tl.load(K + row*KDIM + kk, kk < KDIM, other=0)
        decay = tl.load(DEC + row*KDIM + kk, kk < KDIM, other=0)
        state = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [state, decay[None, :]],
                                           dtype=tl.float32, is_pure=True, pack=1)
        value = tl.load(VEC + row*VDIM + vv, vv < VDIM, other=0).to(tl.float32)
        update = (value - tl.sum(state * key[None, :], 1)) * tl.load(BET + row)
        state = tl.fma(key[None, :], update[:, None], state)
        query = tl.load(Q + row*KDIM + kk, kk < KDIM, other=0)
        output = tl.sum(state * query[None, :], 1)
        tl.store(UPDATE + row*VDIM + vv, update, vv < VDIM)
        tl.store(OUT + row*VDIM + vv, output, vv < VDIM)
        previous = node
        # Subsequent nodes may read this CTA's preceding value updates.
        tl.debug_barrier()


@triton.jit
def _materialize(INITIAL, KEY, DEC, UPDATE, PATH, OUT,
                 COUNT: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, B: tl.constexpr):
    cell = tl.program_id(0) * B + tl.arange(0, B)
    mask = cell < H*K*V
    head, key, value = cell // (K*V), cell // V % K, cell % V
    state = tl.load(INITIAL + cell, mask, other=0)
    for i in range(COUNT):
        node = tl.load(PATH + i)
        row = node * H + head
        k = tl.load(KEY + row*K + key, mask, other=0)
        decay = tl.load(DEC + row*K + key, mask, other=0)
        update = tl.load(UPDATE + row*V + value, mask, other=0)
        state = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [state, decay],
                                           dtype=tl.float32, is_pure=True, pack=1)
        state = tl.fma(k, update, state)
    tl.store(OUT + cell, state, mask)


class NativeFactors(Factors):
    def state(self, node):
        path = self.tree.path(node)
        indices = torch.tensor(path, dtype=torch.int32, device=self.initial.device)
        output = torch.empty_like(self.initial)
        h, k, v = output.shape
        _materialize[(triton.cdiv(output.numel(), 1024),)](
            self.initial, self.key, self.decay, self.update, indices, output, len(path), h, k, v, 1024)
        return output


def verify(tree, q, k, v, g, beta, a, bias, initial, lower_bound, *, topology=None, carry=True):
    n, h, d = q.shape
    vd = v.shape[-1]
    if not q.is_cuda or d > 256 or vd > 256:
        raise ValueError("native KDA tree requires CUDA and head dimensions <= 256")
    topology = topology or Topology(tree, q.device)
    if topology.tree != tree or topology.device != q.device:
        raise ValueError("KDA topology must belong to this tree and device")
    qf, kf, decay = (torch.empty_like(q, dtype=torch.float32) for _ in range(3))
    bet = torch.empty_like(beta, dtype=torch.float32)
    initial = initial.contiguous().clone()
    update, out = torch.empty_like(v, dtype=torch.float32), torch.empty_like(v)
    _prepare[(n*h,)](q.contiguous(), k.contiguous(), g.contiguous(), beta.contiguous(), a.contiguous(), bias.contiguous(),
                     qf, kf, decay, bet, h, d, lower_bound, triton.next_power_of_2(d), num_warps=4)
    _verify[(h, triton.cdiv(vd, 16))](qf, kf, v.contiguous(), decay, bet, initial,
        topology.paths, topology.depths, update, out, topology.order, topology.parents,
        n, h, d, vd, topology.width, triton.next_power_of_2(d), 16, carry, num_warps=4)
    return out, NativeFactors(tree, initial, kf, decay, update)
