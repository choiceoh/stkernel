"""DFlash2, the fleet's drafter, on the ST engine (profile).

GLM-5.3 is served with SPEC_K=5 drafts a step from GLM-5.3-Flash-DFlash2:
a 5-layer Qwen3-shaped block drafter (hidden 4096, 32 q / 8 kv heads of
128, q/k norms, rope theta 1e4, sliding window 2048, NON-causal inside the
block) that reads the target's hidden states at layers 5,14,24,33,42
(concatenated, `fc` -> 4096, `hidden_norm`) as its attention CONTEXT --
projected once per verified token to K/V for all five layers, rope'd, kept
-- and, per step, runs one block of [anchor token, K mask tokens] against
that context to emit K drafts at once. Two DFlash2 additions: a grouped
depthwise conv (2 taps, groups of 16) around attention and MLP whose
coefficients the token itself projects, and a path selector: top-16
candidates per position, edge scores from predecessor/successor codebooks
under a 256-d projection of the hidden state, and a greedy walk from the
anchor. The walk at temperature 0 is what the served speculator's
`_selector_walk_kernel` does; the Gumbel branch is not ported (the engine
samples the TARGET; drafts only need to be good guesses).

Replicated: every rank holds the whole drafter (2.18 GiB, DRAFT_TP=1 as
served) and computes identical drafts from identical inputs; the only
collectives are the target's vocab-parallel embed and head, which the
drafter borrows (its checkpoint ships neither).

Two calls, from the engine (engine.py):
    observe(slot, positions, aux)    the target's aux hidden of newly VERIFIED tokens -> context K/V into the slot's ring
    propose(anchor, position, slot)  K draft ids for the block starting at `position` (the anchor's)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as Fn

from engine.base.params import Spec
from engine.profiles.glm53.facts import SPEC_K, TP

DRAFTER = Path("/home/choiceoh/models/GLM-5.3-Flash-DFlash2")
BF16, F32 = torch.bfloat16, torch.float32


@dataclass(frozen=True)
class DrafterFacts:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    rms_eps: float
    rope_theta: float
    window: int
    block: int                      # training block: 1 anchor + up to block-1 masks
    mask_id: int
    conv_taps: int
    conv_group: int
    sel_rank: int
    sel_top_k: int
    target_layers: tuple            # target layer ids whose hidden states feed fc (1-based as the config counts them)
    k: int                          # drafts per step: SPEC_K

    @property
    def aux_layers(self) -> "list[int]":
        """The target's layer indices whose OUTPUT is taken: the served model
        keeps `hidden after layer idx` when idx + 1 is in target_layer_ids."""
        return [t - 1 for t in self.target_layers]


def load(path: "str | Path" = DRAFTER) -> DrafterFacts:
    c = json.loads((Path(path) / "config.json").read_text())
    d = c["dflash_config"]
    f = DrafterFacts(c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
                     c["intermediate_size"], c["rms_norm_eps"], c["rope_parameters"]["rope_theta"], c["sliding_window"],
                     d["block_size"], d["mask_token_id"], d["conv_kernel_size"], d["conv_group_size"], d["selector_rank"],
                     d["selector_top_k"], tuple(d["target_layer_ids"]), SPEC_K)
    assert c["model_type"] == "qwen3" and c["architectures"] == ["DFlash2DraftModel"] and not c["is_causal"]
    assert all(t == "sliding_attention" for t in c["layer_types"]) and c["use_sliding_window"] and c["rope_parameters"]["rope_type"] == "default"
    assert c["hidden_size"] == 4096 and c["num_target_layers"] == 45 and c["vocab_size"] == 154880 and not c["tie_word_embeddings"]
    assert f.k <= f.block - 1 and f.hidden % f.conv_group == 0 and f.heads % f.kv_heads == 0
    return f


def specs(F: DrafterFacts) -> "list[Spec]":
    """The checkpoint's 81 tensors, whole, under their own names."""
    H, I, D = F.hidden, F.inter, F.head_dim
    out = [
        Spec("fc.weight", (H, H * len(F.target_layers)), BF16), Spec("hidden_norm.weight", (H,), BF16), Spec("norm.weight", (H,), BF16),
        Spec("candidate_selector.hidden_projection.weight", (F.sel_rank, H), BF16),
        Spec("candidate_selector.predecessor_codebook", (154880, F.sel_rank), BF16),
        Spec("candidate_selector.successor_codebook", (154880, F.sel_rank), BF16),
    ]
    G = H // F.conv_group
    for L in range(F.layers):
        p = f"layers.{L}."
        out += [
            Spec(p + "input_layernorm.weight", (H,), BF16), Spec(p + "post_attention_layernorm.weight", (H,), BF16),
            Spec(p + "self_attn.q_proj.weight", (F.heads * D, H), BF16), Spec(p + "self_attn.k_proj.weight", (F.kv_heads * D, H), BF16),
            Spec(p + "self_attn.v_proj.weight", (F.kv_heads * D, H), BF16), Spec(p + "self_attn.o_proj.weight", (H, F.heads * D), BF16),
            Spec(p + "self_attn.q_norm.weight", (D,), BF16), Spec(p + "self_attn.k_norm.weight", (D,), BF16),
            Spec(p + "mlp.gate_proj.weight", (I, H), BF16), Spec(p + "mlp.up_proj.weight", (I, H), BF16), Spec(p + "mlp.down_proj.weight", (H, I), BF16),
            Spec(p + "attention_conv.base_kernel", (2, F.conv_taps, H), BF16), Spec(p + "attention_conv.kernel_projection.weight", (2 * F.conv_taps * G, H), BF16),
            Spec(p + "mlp_conv.base_kernel", (2, F.conv_taps, H), BF16), Spec(p + "mlp_conv.kernel_projection.weight", (2 * F.conv_taps * G, H), BF16),
        ]
    return out


def rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float):
    """Neox-style rotary on [N, heads, D] at absolute `positions` [N]."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=F32) / d))
    ang = positions.float()[:, None] * inv[None, :]                                 # [N, D/2]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., : d // 2].float(), x[..., d // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class Drafter:
    def __init__(self, F: DrafterFacts, target, decodable: int):
        """`target` is the Glm53Net (embed/head are borrowed); `decodable` masks ids the tokenizer cannot decode."""
        self.F, self.target, self.decodable = F, target, decodable
        self.k = F.k
        self.p = None
        self.decode_graphs = None

    def capture_decode(self, caches, memory=None):
        from engine.profiles.glm53.decode_graphs import DrafterDecodeGraphs
        self.decode_graphs = DrafterDecodeGraphs(self, caches, memory=memory)

    def observe_decode(self, ring, positions, aux):
        if self.decode_graphs is None:
            raise RuntimeError("drafter decode context graphs were not captured")
        self.decode_graphs.observe(ring, positions, aux)

    @property
    def aux_layers(self) -> "list[int]":
        return self.F.aux_layers

    def specs(self):
        return specs(self.F)

    def bind(self, views: dict) -> None:
        from engine.base.params import bind
        self.p = bind(self.specs(), views)

    # -- context: verified tokens' target states -> K/V rings -------------------------------
    def observe(self, ring: torch.Tensor, positions: torch.Tensor, aux: torch.Tensor) -> None:
        """ring [L, 2, window, kv_heads, D] bf16 (a slot's); positions [n]; aux [n, 5*4096] target states."""
        F, p = self.F, self.p
        if positions.numel() == 0:
            return
        # A long prefill can wrap the ring several times. Scatter each cell
        # once, retaining the newest window; duplicate CUDA indices have no
        # defined last-writer order.
        positions, aux = positions[-F.window:], aux[-F.window:]
        c = rmsnorm(Fn.linear(aux, p["fc.weight"]), p["hidden_norm.weight"], F.rms_eps)          # context states, normed once for every layer
        idx = positions % F.window
        for L in range(F.layers):
            q = f"layers.{L}.self_attn."
            k = Fn.linear(c, p[q + "k_proj.weight"]).view(-1, F.kv_heads, F.head_dim)
            k = rope(rmsnorm(k, p[q + "k_norm.weight"], F.rms_eps), positions, F.rope_theta)
            v = Fn.linear(c, p[q + "v_proj.weight"]).view(-1, F.kv_heads, F.head_dim)
            ring[L, 0, idx] = k
            ring[L, 1, idx] = v

    def observe_masked(self, ring: torch.Tensor, positions: torch.Tensor, aux: torch.Tensor, valid: torch.Tensor) -> None:
        """`observe` for a decode step running ahead of the host (45차 §23 B3): the step's K+1 positions are given, but
        only the first `valid` (a device scalar: the tokens the device committed) may enter the ring -- the rest keep
        the cells they would have overwritten. Blend, then scatter: no host read, no duplicate writer."""
        F, p = self.F, self.p
        t = positions.numel()
        c = rmsnorm(Fn.linear(aux, p["fc.weight"]), p["hidden_norm.weight"], F.rms_eps)
        idx = positions % F.window
        keep = (torch.arange(t, device=positions.device) < valid).view(t, 1, 1)
        for L in range(F.layers):
            q = f"layers.{L}.self_attn."
            k = Fn.linear(c, p[q + "k_proj.weight"]).view(-1, F.kv_heads, F.head_dim)
            k = rope(rmsnorm(k, p[q + "k_norm.weight"], F.rms_eps), positions, F.rope_theta)
            v = Fn.linear(c, p[q + "v_proj.weight"]).view(-1, F.kv_heads, F.head_dim)
            ring[L, 0, idx] = torch.where(keep, k, ring[L, 0, idx])
            ring[L, 1, idx] = torch.where(keep, v, ring[L, 1, idx])

    # -- the block ------------------------------------------------------------------------------
    def _conv(self, x, delta, base, tap_valid):
        F = self.F
        G = F.hidden // F.conv_group
        blocks = x.unflatten(-1, (G, F.conv_group))                                    # [B, G, 16]
        coeff = base.view(1, F.conv_taps, G, F.conv_group) + delta.unsqueeze(-1)            # [B, taps, G, 16]
        out = coeff[:, 0] * blocks
        for tap in range(1, F.conv_taps):
            shifted = Fn.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coeff[:, tap] * shifted * tap_valid[:, tap].view(-1, 1, 1)
        return out.flatten(-2)

    def _attn(self, L: int, x: torch.Tensor, positions: torch.Tensor, ring: torch.Tensor, ctx_len: int) -> torch.Tensor:
        F, p = self.F, self.p
        q = f"layers.{L}.self_attn."
        B = x.shape[0]
        qh = rope(rmsnorm(Fn.linear(x, p[q + "q_proj.weight"]).view(B, F.heads, F.head_dim), p[q + "q_norm.weight"], F.rms_eps), positions, F.rope_theta)
        kh = rope(rmsnorm(Fn.linear(x, p[q + "k_proj.weight"]).view(B, F.kv_heads, F.head_dim), p[q + "k_norm.weight"], F.rms_eps), positions, F.rope_theta)
        vh = Fn.linear(x, p[q + "v_proj.weight"]).view(B, F.kv_heads, F.head_dim)
        # the context window: the last min(ctx, window) verified positions, then the block itself (non-causal)
        # A fixed window keeps GEMM/reduction geometry identical in eager and
        # captured execution, including the first 2048 positions.
        cpos = ctx_len + torch.arange(-F.window, 0, device=x.device)
        kc, vc = ring[L, 0, cpos % F.window], ring[L, 1, cpos % F.window]                    # [n_ctx, kv, D]
        kc = kc.masked_fill((cpos < 0)[:, None, None], 0)
        vc = vc.masked_fill((cpos < 0)[:, None, None], 0)
        k_all, v_all = torch.cat([kc, kh]), torch.cat([vc, vh])                                 # [n_ctx + B, kv, D]
        rep = F.heads // F.kv_heads
        k_all, v_all = k_all.repeat_interleave(rep, dim=1), v_all.repeat_interleave(rep, dim=1)
        scores = torch.einsum("bhd,nhd->bhn", qh.float(), k_all.float()) * F.head_dim ** -0.5
        valid = torch.cat([cpos >= 0, torch.ones(B, device=x.device, dtype=torch.bool)])
        scores = scores.masked_fill(~valid[None, None, :], float("-inf"))
        o = torch.einsum("bhn,nhd->bhd", torch.softmax(scores, dim=-1), v_all.float()).to(x.dtype)
        return Fn.linear(o.reshape(B, F.heads * F.head_dim), p[q + "o_proj.weight"])

    def block(self, ids: torch.Tensor, positions: torch.Tensor, ring: torch.Tensor, ctx_len: int) -> torch.Tensor:
        """One block through the five layers; returns the final hidden [B, hidden]."""
        F, p = self.F, self.p
        B = ids.shape[0]
        tap_valid = torch.arange(B, device=ids.device)[:, None] >= torch.arange(F.conv_taps, device=ids.device)[None, :]
        x = self.target.embed(ids)
        res = None
        for L in range(F.layers):
            q = f"layers.{L}."
            if res is None:
                res, h = x, rmsnorm(x, p[q + "input_layernorm.weight"], F.rms_eps)
            else:
                res = res + x
                h = rmsnorm(res, p[q + "input_layernorm.weight"], F.rms_eps)
            coeff = Fn.linear(h, p[q + "attention_conv.kernel_projection.weight"]).reshape(B, 2, F.conv_taps, -1)
            h = self._conv(h, coeff[:, 0], p[q + "attention_conv.base_kernel"][0], tap_valid)
            h = self._attn(L, h, positions, ring, ctx_len)
            h = self._conv(h, coeff[:, 1], p[q + "attention_conv.base_kernel"][1], tap_valid)
            res = res + h
            h = rmsnorm(res, p[q + "post_attention_layernorm.weight"], F.rms_eps)
            coeff = Fn.linear(h, p[q + "mlp_conv.kernel_projection.weight"]).reshape(B, 2, F.conv_taps, -1)
            h = self._conv(h, coeff[:, 0], p[q + "mlp_conv.base_kernel"][0], tap_valid)
            h = Fn.linear(Fn.silu(Fn.linear(h, p[q + "mlp.gate_proj.weight"])) * Fn.linear(h, p[q + "mlp.up_proj.weight"]), p[q + "mlp.down_proj.weight"])
            x = self._conv(h, coeff[:, 1], p[q + "mlp_conv.base_kernel"][1], tap_valid)
        return rmsnorm(res + x, p["norm.weight"], F.rms_eps)

    def propose(self, anchor: int, position: int, ring: torch.Tensor) -> "list[int]":
        """K drafts for the block [anchor at `position`, K masks after it]; the ring holds the context up to position-1."""
        if self.decode_graphs is not None:
            return self.decode_graphs.propose(anchor, position, ring).tolist()
        anchor = torch.full((1,), anchor, dtype=torch.int64, device=ring.device)
        return self.propose_tensor(anchor, position, ring).tolist()

    def propose_tensor(self, anchor: torch.Tensor, position, ring: torch.Tensor) -> torch.Tensor:
        """The same greedy walk, with every selection remaining on device."""
        F, p = self.F, self.p
        K = self.k
        dev = ring.device
        ids = torch.cat([anchor.reshape(1), torch.full((K,), F.mask_id, dtype=torch.int64, device=dev)])
        positions = position + torch.arange(K + 1, device=dev)
        h = self.block(ids, positions, ring, position)[1:]                                   # the K mask positions
        from engine.modules.vocab import topk
        unary, cand = topk(self.target.head_local(h), self.target.comm,
                           self.target.rank * self.target.vp, F.sel_top_k, self.decodable)  # [K, 16]
        proj = Fn.linear(h, p["candidate_selector.hidden_projection.weight"]).float()        # [K, 256]
        pred_ids = torch.cat([anchor.reshape(1, 1).expand(1, F.sel_top_k), cand[:-1]])   # [K, 16]
        pred = p["candidate_selector.predecessor_codebook"][pred_ids].float()                # [K, 16, 256]
        succ = p["candidate_selector.successor_codebook"][cand].float()                      # [K, 16, 256]
        scores = unary[:, None, :] + torch.einsum("kpr,kcr->kpc", pred * proj[:, None, :], succ)   # [K, prev, cur]
        out, prev = [], torch.zeros(1, device=dev, dtype=torch.int64)
        for s in range(K):                                                                  # greedy walk, as the served kernel at temperature 0
            prev = scores[s].index_select(0, prev).argmax(-1)
            out.append(cand[s].index_select(0, prev))
        return torch.cat(out)

    def propose_sampled(self, anchor: int, position: int, ring: torch.Tensor, temperature: float, generator,
                        vocab: int) -> "tuple[list[int], torch.Tensor]":
        """The same walk drawn at `temperature` instead of argmax (production's DRAFT_SAMPLE=probabilistic): returns the K
        draft ids and the distribution each was drawn from, [K, vocab] fp32 (zero outside the 16 candidates) -- what
        rejection sampling divides by (base/sampler.speculative_pick). Eager, for stochastic rows only."""
        F, p = self.F, self.p
        K = self.k
        dev = ring.device
        anchor_t = torch.full((1,), anchor, dtype=torch.int64, device=dev)
        ids = torch.cat([anchor_t, torch.full((K,), F.mask_id, dtype=torch.int64, device=dev)])
        positions = position + torch.arange(K + 1, device=dev)
        h = self.block(ids, positions, ring, position)[1:]
        from engine.modules.vocab import topk
        unary, cand = topk(self.target.head_local(h), self.target.comm, self.target.rank * self.target.vp, F.sel_top_k, self.decodable)
        proj = Fn.linear(h, p["candidate_selector.hidden_projection.weight"]).float()
        pred_ids = torch.cat([anchor_t.reshape(1, 1).expand(1, F.sel_top_k), cand[:-1]])
        pred = p["candidate_selector.predecessor_codebook"][pred_ids].float()
        succ = p["candidate_selector.successor_codebook"][cand].float()
        scores = unary[:, None, :] + torch.einsum("kpr,kcr->kpc", pred * proj[:, None, :], succ)
        drafts, dists = self._walk(scores, cand, temperature, generator, vocab, K, dev)
        return drafts.tolist(), dists                                   # one crossing for the walk, not two a step

    def _walk(self, scores, cand, temperature: float, generator, vocab: int, K: int, dev):
        """The K-step candidate walk drawn at `temperature`: (draft ids [K], distributions [K, vocab] fp32).

        Each step picks from the sixteen candidates the last one opened, so the walk itself cannot be
        batched -- but its uniforms can be drawn in one call, and over sixteen candidates the
        cumulative walk is the whole of a draw. `multinomial` per step was five kernels and, when the
        caller wanted host ints, ten crossings a row a step.
        """
        u = torch.rand(K, generator=generator, device=dev)
        dists = torch.zeros(K, vocab, device=dev, dtype=torch.float32)
        drafts = []
        prev = torch.zeros(1, dtype=torch.int64, device=dev)
        for s in range(K):
            probs = torch.softmax(scores[s].index_select(0, prev)[0].float() / max(temperature, 1e-5), dim=-1)
            walk = probs.cumsum(0)
            pick = torch.searchsorted(walk.contiguous(), (u[s] * walk[-1]).reshape(1), right=True) \
                .clamp_max(probs.numel() - 1)
            dists[s].index_add_(0, cand[s], probs)                      # duplicates (if any) add up
            drafts.append(cand[s].index_select(0, pick))
            prev = pick
        return torch.cat(drafts), dists


    def propose_sampled_tensor(self, anchor: torch.Tensor, position, ring: torch.Tensor, temperature: float, generator,
                               vocab: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """`propose_sampled` with every pick a tensor (45차 §23 B3: a row ahead of the host cannot read its drafts back).
        anchor [1] int64 on device; position a device scalar or int. Returns (drafts [K], dists [K, vocab] fp32)."""
        F, p = self.F, self.p
        K = self.k
        dev = ring.device
        ids = torch.cat([anchor.reshape(1), torch.full((K,), F.mask_id, dtype=torch.int64, device=dev)])
        positions = position + torch.arange(K + 1, device=dev)
        h = self.block(ids, positions, ring, position)[1:]
        from engine.modules.vocab import topk
        unary, cand = topk(self.target.head_local(h), self.target.comm, self.target.rank * self.target.vp, F.sel_top_k, self.decodable)
        proj = Fn.linear(h, p["candidate_selector.hidden_projection.weight"]).float()
        pred_ids = torch.cat([anchor.reshape(1, 1).expand(1, F.sel_top_k), cand[:-1]])
        pred = p["candidate_selector.predecessor_codebook"][pred_ids].float()
        succ = p["candidate_selector.successor_codebook"][cand].float()
        scores = unary[:, None, :] + torch.einsum("kpr,kcr->kpc", pred * proj[:, None, :], succ)
        return self._walk(scores, cand, temperature, generator, vocab, K, dev)


def ring_bytes(F: DrafterFacts) -> int:
    return F.layers * 2 * F.window * F.kv_heads * F.head_dim * 2


def _selfcheck() -> None:
    F = load()
    assert (F.layers, F.heads, F.kv_heads, F.head_dim, F.window, F.block, F.k) == (5, 32, 8, 128, 2048, 8, 5)
    assert F.aux_layers == [4, 13, 23, 32, 41] and F.sel_top_k == 16 and F.mask_id == 154856
    sp = specs(F)
    from engine.base.params import total_bytes
    import struct
    with (DRAFTER / "model.safetensors").open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n)); h.pop("__metadata__", None)
    assert {s.name for s in sp} == set(h), (set(h) - {s.name for s in sp}, {s.name for s in sp} - set(h))
    assert all(tuple(h[s.name]["shape"]) == s.shape for s in sp)
    print(f"  drafter: DFlash2 facts asserted, {len(sp)} tensors = the checkpoint's ({total_bytes(sp) / 2**30:.2f} GiB, replicated), "
          f"context ring {ring_bytes(F) / 2**20:.0f} MiB per slot, K={F.k} OK")


if __name__ == "__main__":
    _selfcheck()
