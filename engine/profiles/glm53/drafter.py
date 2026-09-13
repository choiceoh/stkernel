"""DFlash2, the fleet's drafter, on the ST engine (profile).

GLM-5.3 is served with SPEC_K=6 drafts a step from GLM-5.3-Flash-DFlash2:
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

The loader reads the whole 2.18 GiB drafter as temporary preparation inputs.
Native preparation reserves only the live packed readers and shards its
attention heads and MLP across TP4; row-parallel outputs are reduced.
It also borrows the target's vocab-parallel embed and head (the drafter
checkpoint ships neither).

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
from contextlib import nullcontext
from torch.nn.attention import SDPBackend, sdpa_kernel

from engine.base.params import Spec
from engine.kernels.draft_conv import tap_mix
from engine.kernels.draft_select import walk_scores
from engine.kernels.swiglu import swiglu
from engine.kernels.norm_rope import add_norm, norm, norm_rope, warm as warm_rotary
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


# The torch forms of the two normalisations. The block runs the fused kernels (kernels/norm_rope) -- one launch
# where these are six and twenty-one -- and these stay as the reference they are judged against, which the graph
# probe's FP64 oracle and the retention test read directly.
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


STORE_PREFIX = "DFlash2Qwen3ForCausalLM/model."      # the pack store's namespace for the drafter (kernels/dense/store)


def store_name(name: str) -> str:
    """The pack store's name of a prepared dense weight: what its packs and calibration blobs are filed under."""
    module = name.removesuffix(".weight").replace("self_attn.qkv", "self_attn.qkv_proj").replace("mlp.gate_up", "mlp.gate_up_proj")
    return STORE_PREFIX + module


def dense_shapes(F: DrafterFacts, world: int) -> "dict[str, tuple[int, int]]":
    """[rows, cols] of every dense weight `prepare_fast` packs at this TP, by its drafter name -- what a boot asks
    the pack store about before anything is loaded."""
    H, I, D = F.hidden, F.inter, F.head_dim
    G = H // F.conv_group
    heads, kv, inter = F.heads // world, F.kv_heads // world, I // world
    out = {"fc.weight": (H, H * len(F.target_layers))}
    for L in range(F.layers):
        n = f"layers.{L}."
        out.update({n + "self_attn.qkv": ((heads + 2 * kv) * D, H), n + "self_attn.o_proj.weight": (H, heads * D),
                    n + "mlp.gate_up": (2 * inter, H), n + "mlp.down_proj.weight": (H, inter),
                    n + "attention_conv.kernel_projection.weight": (2 * F.conv_taps * G, H),
                    n + "mlp_conv.kernel_projection.weight": (2 * F.conv_taps * G, H)})
    return out


class Drafter:
    def __init__(self, F: DrafterFacts, target, decodable: int):
        """`target` is the Glm53Net (embed/head are borrowed); `decodable` masks ids the tokenizer cannot decode."""
        self.F, self.target, self.decodable = F, target, decodable
        self.k = F.k
        self.p = None
        self.decode_graphs = None
        self.dense = {}
        self.fast_attention = False
        self.local_heads, self.local_kv_heads = F.heads, F.kv_heads
        self.context_kv = None
        self.max_block_rows = None

    def capture_decode(self, caches, memory=None, generator=None, vocab=None):
        from engine.profiles.glm53.decode_graphs import DrafterDecodeGraphs
        # The rotary table is a constant of the model; built here it belongs to the arena, not to whichever graph
        # happened to run first and would free it on close (kernels/norm_rope.warm).
        warm_rotary(caches.device, self.F.head_dim, self.F.rope_theta)
        self.decode_graphs = DrafterDecodeGraphs(self, caches, memory=memory, generator=generator, vocab=vocab)

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

    def smoothing_plan(self, amax_of) -> dict:
        """The channel smoothing of the block's two norm outputs (kernels/dense/smoothing): input_layernorm feeds
        attention_conv's kernel projection and, through the grouped conv (per channel, so the factor passes),
        q/k/v; post_attention_layernorm feeds mlp_conv's kernel projection and gate/up. The norms are divided in
        place; the readers' smoothed weights come back as {weight key: tensor} with {norm key: s_eff} -- the
        CONTEXT projection keeps the unsmoothed k/v (its input, the fc's normed hidden, is not divided)."""
        from engine.kernels.dense.smoothing import fold, scales, smooth_weight
        F, p = self.F, self.p
        weights, factors = {}, {}
        for L in range(F.layers):
            n = f"layers.{L}."
            for norm, dense_name, readers in ((n + "input_layernorm.weight", n + "self_attn.qkv",
                                               [n + "attention_conv.kernel_projection.weight"] + [n + f"self_attn.{s}_proj.weight" for s in ("q", "k", "v")]),
                                              (n + "post_attention_layernorm.weight", n + "mlp.gate_up",
                                               [n + "mlp_conv.kernel_projection.weight"] + [n + f"mlp.{s}_proj.weight" for s in ("gate", "up")])):
                amax = amax_of(store_name(dense_name))
                if amax is None or any(p.get(k) is None for k in readers + [norm]):
                    continue
                s_eff = fold(p[norm], scales(amax, [p[k] for k in readers]))
                factors[norm] = s_eff
                for k in readers:
                    weights[k] = smooth_weight(p[k], s_eff)
        return weights, factors

    def prepare_fast(self, store=None, *, consume_weights=False, max_seqs=None, compact_into=None):
        """TP-shard dense compute and bind calibrated packs before capture.

        Only this rank's heads/MLP shard are read by decode; row-parallel
        outputs join through the target comm. Production explicitly retires
        source BF16 regions into packs after every source consumer is prepared.
        """
        from engine.kernels.dense import DenseLinear
        from .drafter_storage import block_rows, needs_fp8, compact
        if compact_into is not None and (consume_weights or max_seqs is None):
            raise ValueError('compact drafter storage needs a sequence capacity and independent source weights')
        F, p, comm = self.F, self.p, self.target.comm
        self.max_block_rows = block_rows(F, max_seqs) if max_seqs is not None else None
        if F.heads % comm.world_size or F.kv_heads % comm.world_size or F.inter % comm.world_size:
            raise ValueError("DFlash dimensions must split over the target communicator")
        self.local_heads, self.local_kv_heads = F.heads//comm.world_size, F.kv_heads//comm.world_size
        def shard(w, dim):
            return w.chunk(comm.world_size,dim=dim)[comm.rank].contiguous()
        smoothed, factors = self.smoothing_plan(store.amax) if store is not None else ({}, {})
        sw = lambda key: smoothed.get(key, p[key])                       # the block path's weight, smoothed when its norm was folded
        weights, smooth = {"fc.weight": p["fc.weight"]}, {}
        context = []
        for L in range(F.layers):
            n = f"layers.{L}."
            weights[n+"self_attn.qkv"] = torch.cat([shard(sw(n+"self_attn."+s+"_proj.weight"),0) for s in ("q", "k", "v")])
            weights[n+"mlp.gate_up"] = torch.cat([shard(sw(n+"mlp."+s+"_proj.weight"),0) for s in ("gate", "up")])
            for key in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
                weights[n+key] = shard(p[n+key],1)
            for key in ("attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight"):
                weights[n+key] = sw(n+key)
            smooth[n+"self_attn.qkv"] = smooth[n+"attention_conv.kernel_projection.weight"] = factors.get(n + "input_layernorm.weight")
            smooth[n+"mlp.gate_up"] = smooth[n+"mlp_conv.kernel_projection.weight"] = factors.get(n + "post_attention_layernorm.weight")
            context.extend(shard(p[n+"self_attn."+s+"_proj.weight"],0) for s in ("k", "v"))   # unsmoothed: its input is not divided
        self.dense = {}
        for name, w in weights.items():
            self.dense[name] = DenseLinear(w,store=store,name=store_name(name),smooth=smooth.get(name),
                                          prefill=needs_fp8(F, max_seqs, name))
        self.context_kv = torch.cat(context)
        self.context_norm = torch.stack([p[f"layers.{L}.self_attn.k_norm.weight"] for L in range(F.layers)])
        if compact_into is not None:
            compact(self, compact_into, max_seqs)
        if consume_weights:
            for name, layer in self.dense.items():
                source = (name.replace("self_attn.qkv","self_attn.q_proj.weight")
                          .replace("mlp.gate_up","mlp.gate_proj.weight"))
                layer.consume_weight(p[source])
                p[source]=None
            # All query/gate/up projections now have explicit packed readers;
            # context KV was merged above, before any source was retired.
            for L in range(F.layers):
                for suffix in ("self_attn.k_proj.weight","self_attn.v_proj.weight","mlp.up_proj.weight"):
                    p[f"layers.{L}."+suffix]=None
        self.fast_attention = True

    def linear(self, x, name, mask=None):
        """x through the named weight: the prepared dense pack when there is one, else the bf16 source. `mask` [rows]
        tells a calibrating pack which rows are real (kernels/dense/calibration); the product covers every row."""
        layer = self.dense.get(name)
        if layer is None:
            return Fn.linear(x, self.p[name])
        if mask is not None and getattr(layer, "observer", None) is not None:   # only a calibrating pack reads the mask
            return layer(x, mask)
        return layer(x)

    # -- context: verified tokens' target states -> K/V rings -------------------------------
    def observe(self, ring: torch.Tensor, positions: torch.Tensor, aux: torch.Tensor) -> None:
        """ring [L, 2, window, kv_heads, D] bf16 (a slot's); positions [n]; aux [n, 5*4096] target states."""
        self._observe(ring, positions, aux)

    def observe_masked(self, ring, positions, aux, valid):
        """Commit a device-counted accepted prefix using the same TP context projection."""
        self._observe(ring, positions, aux, valid)

    def _observe(self, ring, positions, aux, valid=None):
        F, p = self.F, self.p
        if positions.numel() == 0:
            return
        # A long prefill can wrap the ring several times. Scatter each cell
        # once, retaining the newest window; duplicate CUDA indices have no
        # defined last-writer order.
        positions, aux = positions[-F.window:], aux[-F.window:]
        keep = (torch.arange(len(positions), device=positions.device) < valid) if valid is not None else None
        c = norm(self.linear(aux, "fc.weight", keep), p["hidden_norm.weight"], F.rms_eps)          # context states, normed once for every layer
        idx = positions % F.window
        context = (Fn.linear(c,self.context_kv).reshape(-1,F.layers,2,self.local_kv_heads,F.head_dim)
                   if self.context_kv is not None else None)
        for L in range(F.layers):
            q = f"layers.{L}.self_attn."
            k = context[:,L,0] if context is not None else self.linear(c, q + "k_proj.weight").view(-1, F.kv_heads, F.head_dim)
            k = norm_rope(k, p[q + "k_norm.weight"], F.rms_eps, positions, F.rope_theta)
            v = context[:,L,1] if context is not None else self.linear(c, q + "v_proj.weight").view(-1, F.kv_heads, F.head_dim)
            if isinstance(ring, tuple):
                from engine.kernels.draft_attention import write_draft_kv
                write_draft_kv(ring[0],ring[1],L,positions,k,v,valid=valid)
            else:
                if valid is not None:
                    keep = (torch.arange(len(positions), device=positions.device) < valid)[:, None, None]
                    k = torch.where(keep, k, ring[L, 0, idx, :self.local_kv_heads])
                    v = torch.where(keep, v, ring[L, 1, idx, :self.local_kv_heads])
                ring[L, 0, idx, :self.local_kv_heads] = k
                ring[L, 1, idx, :self.local_kv_heads] = v

    # -- the block ------------------------------------------------------------------------------
    def _conv(self, x, delta, base):
        """The grouped causal tap mix over one block's rows (kernels/draft_conv)."""
        return tap_mix(x, delta, base, self.F.conv_group)

    def _attn(self, L: int, x: torch.Tensor, positions: torch.Tensor, ring: torch.Tensor, ctx_len: int) -> torch.Tensor:
        F, p = self.F, self.p
        q = f"layers.{L}.self_attn."
        B = x.shape[0]
        heads, kv_heads = self.local_heads, self.local_kv_heads
        if self.fast_attention:
            q0, k0, v0 = self.linear(x, q+"qkv").split((heads*F.head_dim, kv_heads*F.head_dim, kv_heads*F.head_dim), -1)
        else:
            q0, k0, v0 = (Fn.linear(x, p[q+s+"_proj.weight"]) for s in ("q", "k", "v"))
        qh = norm_rope(q0.reshape(B, heads, F.head_dim), p[q + "q_norm.weight"], F.rms_eps, positions, F.rope_theta)
        kh = norm_rope(k0.reshape(B, kv_heads, F.head_dim), p[q + "k_norm.weight"], F.rms_eps, positions, F.rope_theta)
        vh = v0.reshape(B, kv_heads, F.head_dim)
        if self.fast_attention:
            from engine.kernels.draft_attention import draft_attention
            if isinstance(ring, tuple):
                o = draft_attention(qh.contiguous(),kh.contiguous(),vh.contiguous(),ring[0],ctx_len,slot=ring[1],layer=L)
            else:
                o = draft_attention(qh.contiguous(), kh.contiguous(), vh.contiguous(), ring[L], ctx_len)
            return self.target.comm.all_reduce(self.linear(o.reshape(B, heads*F.head_dim), q+"o_proj.weight"))
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
        return self.linear(o.reshape(B, F.heads * F.head_dim), q + "o_proj.weight")

    def block(self, ids: torch.Tensor, positions: torch.Tensor, ring: torch.Tensor, ctx_len: int) -> torch.Tensor:
        """One block through the five layers; returns the final hidden [B, hidden]."""
        self._check_block_rows(ids.numel())
        F, p = self.F, self.p
        B = ids.shape[0]
        x = self.target.embed(ids)
        res = None
        for L in range(F.layers):
            q = f"layers.{L}."
            if res is None:
                res, h = x, norm(x, p[q + "input_layernorm.weight"], F.rms_eps)
            else:
                res, h = add_norm(res, x, p[q + "input_layernorm.weight"], F.rms_eps)
            coeff = self.linear(h, q + "attention_conv.kernel_projection.weight").reshape(B, 2, F.conv_taps, -1)
            h = self._conv(h, coeff[:, 0], p[q + "attention_conv.base_kernel"][0])
            h = self._attn(L, h, positions, ring, ctx_len)
            h = self._conv(h, coeff[:, 1], p[q + "attention_conv.base_kernel"][1])
            res, h = add_norm(res, h, p[q + "post_attention_layernorm.weight"], F.rms_eps)
            coeff = self.linear(h, q + "mlp_conv.kernel_projection.weight").reshape(B, 2, F.conv_taps, -1)
            h = self._conv(h, coeff[:, 0], p[q + "mlp_conv.base_kernel"][0])
            if self.fast_attention:
                h = swiglu(self.linear(h, q+"mlp.gate_up"))
            else:
                gate, up = (Fn.linear(h, p[q+"mlp."+s+"_proj.weight"]) for s in ("gate", "up"))
                h = swiglu(torch.cat([gate, up], -1))
            h = self.linear(h, q + "mlp.down_proj.weight")
            if self.fast_attention:
                h = self.target.comm.all_reduce(h)
            x = self._conv(h, coeff[:, 1], p[q + "mlp_conv.base_kernel"][1])
        return add_norm(res, x, p["norm.weight"], F.rms_eps)[1]

    # -- every row of a step at once (45차 §23 GPU 판정 4차) --------------------------------------------
    # The pipeline used to replay the one-row graphs once per row: each replay read the whole drafter (2.03 GiB of
    # GEMM weights, replicated on every rank) and copied the row's 40 MiB ring three times. Batched over the rows the
    # weights are read once a step; the rings are never copied -- the observe writes its cells in place and the
    # attention reads every slot's ring where it lies, one fused call over the whole field per layer.
    def _project_context(self, positions, aux, valid, *, observe=True):
        """The shared context projection; calibration sees only committed rows."""
        F, p = self.F, self.p
        n, t = positions.shape
        keep = (torch.arange(t, device=positions.device) < valid.view(n, 1)).reshape(n * t)
        projected = self.linear(aux, "fc.weight", keep) if observe else self.dense["fc.weight"](aux, observe=False)
        c = norm(projected, p["hidden_norm.weight"], F.rms_eps)
        return Fn.linear(c, self.context_kv).reshape(n, t, F.layers, 2, self.local_kv_heads, F.head_dim)

    def observe_kv(self, positions: torch.Tensor, aux: torch.Tensor, valid: torch.Tensor):
        """observe's compute phase (fast path): every position's drafter K/V, per layer, from `aux` and
        `positions` alone. It does NOT read the slot or the accepted count -- `valid` only forms the mask a
        CALIBRATING pack sums through, a no-op once the packs are calibrated -- so on a serving boot this can
        run while the target's last layers, the sampler and the commit are still deciding how many of these
        positions survive. Returns [(k, v)] over the layers, each [n, t, kv, D]."""
        F, p = self.F, self.p
        n, t = positions.shape
        context = self._project_context(positions, aux, valid)
        out = []
        for L in range(F.layers):
            q = f"layers.{L}.self_attn."
            k = norm_rope(context[:, :, L, 0].reshape(n * t, self.local_kv_heads, F.head_dim),
                          p[q + "k_norm.weight"], F.rms_eps, positions.reshape(-1), F.rope_theta)
            out.append((k.reshape(n, t, self.local_kv_heads, F.head_dim), context[:, :, L, 1]))
        return out

    def observe_write(self, field: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor, kv, valid) -> None:
        """observe's write phase (fast path): the computed K/V into the slots' rings, `valid` of the t positions
        a row. One launch a layer (draft_attention.write_draft_kv_rows). This is the only part that needs the
        accepted count, so it is the only part that waits for the commit."""
        from engine.kernels.draft_attention import write_draft_kv_rows
        for L, (k, v) in enumerate(kv):
            write_draft_kv_rows(field, slots, L, positions, k, v, valid=valid)

    def observe_prepared(self, field, slots, positions, context, valid, aux):
        """Commit an early context projection using the ordinary fused writer.

        `valid` is the final retained count after EOS/limit trimming, not raw
        draft acceptance. Rejected and null rows never reach the live ring.
        """
        from engine.kernels.draft_observe import write_context
        projection = self.dense["fc.weight"]
        if projection.observer is not None:
            keep = (torch.arange(positions.shape[1], device=positions.device) < valid[:, None]).flatten()
            projection.observer(aux.reshape(-1, projection.cols), keep)
        write_context(field, slots, positions, context, self.context_norm, valid, self.F.rms_eps, self.F.rope_theta)

    def observe_rows(self, field: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor, aux: torch.Tensor,
                     valid: torch.Tensor) -> None:
        """`observe_masked` for every row at once, straight into the slots' rings. field [S, L, 2, cells, kv, D] is
        the whole draft field; slots [n]; positions [n, t]; aux [n*t, A] in row order; valid [n] (device counts)."""
        F, p = self.F, self.p
        if self.fast_attention:
            from engine.kernels.draft_observe import write_context
            context = self._project_context(positions, aux, valid)
            write_context(field, slots, positions, context, self.context_norm, valid, F.rms_eps, F.rope_theta)
            return
        n, t = positions.shape
        keep = (torch.arange(t, device=positions.device) < valid.view(n, 1)).reshape(n * t)
        c = norm(self.linear(aux, "fc.weight", keep), p["hidden_norm.weight"], F.rms_eps)
        flat = positions.reshape(-1)
        idx = positions % F.window
        rows = slots.view(n, 1)
        keep = keep.view(n, t, 1, 1)
        for L in range(F.layers):
            q = f"layers.{L}.self_attn."
            k = Fn.linear(c, p[q + "k_proj.weight"]).view(-1, F.kv_heads, F.head_dim)
            k = norm_rope(k, p[q + "k_norm.weight"], F.rms_eps, flat, F.rope_theta).view(n, t, F.kv_heads, F.head_dim)
            v = Fn.linear(c, p[q + "v_proj.weight"]).view(n, t, F.kv_heads, F.head_dim)
            field[rows, L, 0, idx] = torch.where(keep, k, field[rows, L, 0, idx])
            field[rows, L, 1, idx] = torch.where(keep, v, field[rows, L, 1, idx])

    def _conv_rows(self, x, delta, base, t: int):
        """`_conv` over the step's blocks of t rows: the taps look back inside a block, never into the one before."""
        return tap_mix(x, delta, base, self.F.conv_group, block=t)

    def _attn_rows(self, L: int, x: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor, ctx: torch.Tensor,
                   field: torch.Tensor, n: int, t: int, rows_ok=None) -> torch.Tensor:
        """Each block against its own slot's ring, as one fused attention over the whole field. The rows' queries and
        block keys go to their slots (the block's keys into the ring's scratch tail, cells window..window+t); the ring
        is read in storage order -- its keys are rope'd at their positions, so the order of keys is immaterial -- and
        the cells a slot has not written yet (context shorter than the window) are masked. The heads sharing a kv
        head are laid out as more queries against it, so no head is repeated in memory."""
        F, p = self.F, self.p
        q = f"layers.{L}.self_attn."
        if self.fast_attention:
            from engine.kernels.draft_attention import attend_rows
            heads, kv, D = self.local_heads, self.local_kv_heads, F.head_dim
            q0, k0, v0 = self.linear(x, q + "qkv", rows_ok).split((heads*D, kv*D, kv*D), -1)
            qh = norm_rope(q0.reshape(n*t, heads, D), p[q + "q_norm.weight"], F.rms_eps, positions, F.rope_theta)
            kh = norm_rope(k0.reshape(n*t, kv, D), p[q + "k_norm.weight"], F.rms_eps, positions, F.rope_theta)
            vh = v0.reshape(n*t, kv, D).contiguous()
            # GEMMs cover all rows once, and so does the attention: each row reads its own device-selected slot
            # at its own context length, so the rows are a grid dimension and there is nothing to concatenate.
            out = attend_rows(qh.view(n, t, heads, D), kh.view(n, t, kv, D), vh.view(n, t, kv, D),
                              field, ctx, slot=slots, layer=L)
            return self.target.comm.all_reduce(self.linear(out.reshape(n*t, heads*D), q + "o_proj.weight", rows_ok))
        S, W, kv, D, rep = field.shape[0], F.window, F.kv_heads, F.head_dim, F.heads // F.kv_heads
        qh = norm_rope(Fn.linear(x, p[q + "q_proj.weight"]).view(n * t, F.heads, D), p[q + "q_norm.weight"], F.rms_eps, positions, F.rope_theta)
        kh = norm_rope(Fn.linear(x, p[q + "k_proj.weight"]).view(n * t, kv, D), p[q + "k_norm.weight"], F.rms_eps, positions, F.rope_theta)
        vh = Fn.linear(x, p[q + "v_proj.weight"]).view(n * t, kv, D)
        field[slots, L, 0, W:W + t] = kh.view(n, t, kv, D)
        field[slots, L, 1, W:W + t] = vh.view(n, t, kv, D)
        q_rows = qh.view(n, t, kv, rep, D).permute(0, 2, 3, 1, 4).reshape(n, kv, rep * t, D)
        q_all = torch.zeros(S, kv, rep * t, D, dtype=qh.dtype, device=qh.device).index_copy_(0, slots, q_rows)
        length = torch.zeros(S, dtype=ctx.dtype, device=ctx.device).index_copy_(0, slots, ctx)
        cells = torch.arange(W + t, device=ctx.device)
        mask = ((cells < length.clamp_max(W).view(S, 1)) | (cells >= W)).view(S, 1, 1, W + t)
        keys, values = field[:, L, 0, :W + t].transpose(1, 2), field[:, L, 1, :W + t].transpose(1, 2)   # [S, kv, W+t, D], views
        fused = sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) if x.is_cuda else nullcontext()
        with fused:                                                                            # D3: no math fallback on CUDA
            o = Fn.scaled_dot_product_attention(q_all, keys, values, attn_mask=mask, scale=D ** -0.5)
        o = o.index_select(0, slots).view(n, kv, rep, t, D).permute(0, 3, 1, 2, 4).reshape(n * t, F.heads * D)
        return Fn.linear(o, p[q + "o_proj.weight"])

    def block_rows(self, ids: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor, ctx: torch.Tensor,
                   field: torch.Tensor, n: int, t: int, alive=None) -> torch.Tensor:
        """`block` for n blocks of t rows at once: ids/positions [n*t] in row order, slots/ctx [n]; `alive` [n] marks the
        rows that are real (a calibration run leaves the others out of its sums)."""
        self._check_block_rows(n * t)
        F, p = self.F, self.p
        rows_ok = alive.repeat_interleave(t) if alive is not None else None
        x = self.target.embed(ids)
        res = None
        for L in range(F.layers):
            q = f"layers.{L}."
            if res is None:
                res, h = x, norm(x, p[q + "input_layernorm.weight"], F.rms_eps)
            else:
                res, h = add_norm(res, x, p[q + "input_layernorm.weight"], F.rms_eps)
            coeff = self.linear(h, q + "attention_conv.kernel_projection.weight", rows_ok).reshape(n * t, 2, F.conv_taps, -1)
            h = self._conv_rows(h, coeff[:, 0], p[q + "attention_conv.base_kernel"][0], t)
            h = self._attn_rows(L, h, positions, slots, ctx, field, n, t, rows_ok)
            h = self._conv_rows(h, coeff[:, 1], p[q + "attention_conv.base_kernel"][1], t)
            res, h = add_norm(res, h, p[q + "post_attention_layernorm.weight"], F.rms_eps)
            coeff = self.linear(h, q + "mlp_conv.kernel_projection.weight", rows_ok).reshape(n * t, 2, F.conv_taps, -1)
            h = self._conv_rows(h, coeff[:, 0], p[q + "mlp_conv.base_kernel"][0], t)
            if self.fast_attention:
                h = swiglu(self.linear(h, q + "mlp.gate_up", rows_ok))
            else:
                gate, up = (self.linear(h, q + "mlp." + name + "_proj.weight") for name in ("gate", "up"))
                h = swiglu(torch.cat([gate, up], -1))
            h = self.linear(h, q + "mlp.down_proj.weight", rows_ok)
            if self.fast_attention:
                h = self.target.comm.all_reduce(h)
            x = self._conv_rows(h, coeff[:, 1], p[q + "mlp_conv.base_kernel"][1], t)
        return add_norm(res, x, p["norm.weight"], F.rms_eps)[1]

    def _check_block_rows(self, rows):
        if self.max_block_rows is not None and rows > self.max_block_rows:
            raise ValueError(f'drafter block has {rows} rows, above prepared capacity {self.max_block_rows}')

    def propose_rows(self, field: torch.Tensor, slots: torch.Tensor, anchors: torch.Tensor, positions: torch.Tensor,
                     temps: "torch.Tensor | None" = None, generator=None, vocab: "int | None" = None, alive=None):
        """Every row's K drafts at once: anchors [n], positions [n] (each row's context: the anchor's position), slots [n],
        all on the device. Greedy walk, [n, K]; with `temps` [n] the sampled walk at each row's temperature (rows at 0
        stay greedy), plus the candidates each pick was drawn from and their mass -- [n, K, sel_top_k] each, which
        is the whole distribution: the walk puts nothing anywhere else."""
        F, p = self.F, self.p
        K = self.k
        t = K + 1
        n = anchors.numel()
        dev = anchors.device
        ids = torch.cat([anchors.view(n, 1), torch.full((n, K), F.mask_id, dtype=torch.int64, device=dev)], 1).reshape(-1)
        pos = (positions.view(n, 1) + torch.arange(t, device=dev)).reshape(-1)
        h = self.block_rows(ids, pos, slots, positions, field, n, t, alive).view(n, t, -1)[:, 1:].reshape(n * K, -1)
        from engine.modules.vocab import topk
        unary, cand = topk(self.target.head_local(h), self.target.comm, self.target.rank * self.target.vp, F.sel_top_k, self.decodable)
        unary, cand = unary.view(n, K, F.sel_top_k), cand.view(n, K, F.sel_top_k)
        proj = self.linear(h, "candidate_selector.hidden_projection.weight").float().view(n, K, -1)
        if temps is None:
            # the scores never exist: a step reads one codebook row against this step's candidates
            return walk_scores(unary, cand, anchors, proj, p["candidate_selector.predecessor_codebook"],
                               p["candidate_selector.successor_codebook"])
        pred_ids = torch.cat([anchors.view(n, 1, 1).expand(n, 1, F.sel_top_k), cand[:, :-1]], 1)         # [n, K, 16]
        pred = p["candidate_selector.predecessor_codebook"][pred_ids].float()                              # [n, K, 16, 256]
        succ = p["candidate_selector.successor_codebook"][cand].float()
        scores = unary[:, :, None, :] + torch.einsum("nkpr,nkcr->nkpc", pred * proj[:, :, None, :], succ)   # [n, K, prev, cur]
        rows = torch.arange(n, device=dev)
        prev = torch.zeros(n, dtype=torch.int64, device=dev)
        # The walk puts mass on `sel_top_k` candidates a position and nothing else. Handing that back as
        # [n, K, vocab] meant allocating and zeroing 12.4 MiB every decode step (n=4, K=5, V=154,880) to carry
        # 320 numbers, and the verifier then read it twice. The candidates and their mass are the same fact.
        qprob = torch.zeros(n, K, F.sel_top_k, dtype=torch.float32, device=dev)
        qcand = torch.zeros(n, K, F.sel_top_k, dtype=torch.int64, device=dev)
        # The sampled walk stays a loop: its draw is the engine's generator, and moving that into a kernel
        # would put rank agreement and D12's replay in there with it.
        out = []
        for s in range(K):
            sel = scores[rows, s, prev]                                                                    # [n, 16]
            best = sel.argmax(-1)
            stochastic = (temps > 0).view(n, 1)
            probs = torch.softmax(sel / temps.clamp_min(1e-5).view(n, 1), dim=-1)
            probs = torch.where(stochastic, probs, torch.zeros_like(probs).scatter_(1, best.view(n, 1), 1.0))
            pick = torch.where(stochastic.view(n), torch.multinomial(probs, 1, generator=generator).view(n), best)
            qcand[:, s], qprob[:, s] = cand[:, s], probs
            out.append(cand[rows, s, pick])
            prev = pick
        return torch.stack(out, 1), qcand, qprob

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
        dev = anchor.device
        ids = torch.cat([anchor.reshape(1), torch.full((K,), F.mask_id, dtype=torch.int64, device=dev)])
        positions = position + torch.arange(K + 1, device=dev)
        h = self.block(ids, positions, ring, position)[1:]                                   # the K mask positions
        from engine.modules.vocab import topk
        unary, cand = topk(self.target.head_local(h), self.target.comm,
                           self.target.rank * self.target.vp, F.sel_top_k, self.decodable)  # [K, 16]
        proj = Fn.linear(h, p["candidate_selector.hidden_projection.weight"]).float()        # [K, 256]
        return walk_scores(unary.unsqueeze(0), cand.unsqueeze(0), anchor.reshape(1), proj.unsqueeze(0),
                           p["candidate_selector.predecessor_codebook"],
                           p["candidate_selector.successor_codebook"]).reshape(K)   # the served kernel at temperature 0

    def propose_sampled(self, anchor: int, position: int, ring: torch.Tensor, temperature: float, generator,
                        vocab: int) -> "tuple[list[int], torch.Tensor]":
        """The same walk drawn at `temperature` instead of argmax (production's DRAFT_SAMPLE=probabilistic): returns the K
        draft ids and the distribution each was drawn from, [K, vocab] fp32 (zero outside the 16 candidates).

        The distribution handed back is the one the pick was actually drawn from, not one recomputed later: the accept
        test divides by it (base/sampler.block_verify) and is only unbiased if the two are the same. Zero outside the
        candidates is not an approximation either -- the residual keeps every token the candidates left out, so the
        truncation costs acceptance and nothing else.

        The walk itself is `propose_sampled_tensor`, and the ids cross to the host once at the end rather than twice a
        position, which on a five-wide draft was ten synchronizations a row a step.
        """
        drafts, dists = self.propose_sampled_tensor(
            torch.full((1,), anchor, dtype=torch.int64, device=ring.device),
            position, ring, temperature, generator, vocab)
        return drafts.tolist(), dists

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
        # Each step picks from the sixteen candidates the last one opened, so the walk cannot be batched --
        # but its uniforms can be drawn in one call, and over sixteen candidates the cumulative walk is the
        # whole of a draw. `multinomial` was a kernel a position to do that.
        u = torch.rand(K, generator=generator, device=dev)
        drafts, dists = [], torch.zeros(K, vocab, device=dev, dtype=torch.float32)
        prev = torch.zeros(1, dtype=torch.int64, device=dev)
        for s in range(K):
            probs = torch.softmax(scores[s].index_select(0, prev)[0].float() / max(temperature, 1e-5), dim=-1)   # over the 16 candidates
            walk = probs.cumsum(0)
            pick = torch.searchsorted(walk.contiguous(), (u[s] * walk[-1]).reshape(1), right=True) \
                .clamp_max(probs.numel() - 1)
            dists[s].index_add_(0, cand[s], probs)
            drafts.append(cand[s].index_select(0, pick))
            prev = pick
        return torch.cat(drafts), dists


def ring_cells(F: DrafterFacts) -> int:
    """A slot's ring per layer and half: the window's cells, then a scratch tail of block-width cells where the batched
    attention parks a block's own keys so the fused kernel reads context and block as one contiguous run (45차 §23).
    The training block bounds a proposal's width (k + 1 <= block, asserted in `load`)."""
    return F.window + F.block


def ring_bytes(F: DrafterFacts) -> int:
    return F.layers * 2 * ring_cells(F) * F.kv_heads * F.head_dim * 2


def _selfcheck() -> None:
    F = load()
    assert (F.layers, F.heads, F.kv_heads, F.head_dim, F.window, F.block) == (5, 32, 8, 128, 2048, 8)
    assert F.k == SPEC_K and F.k <= F.block - 1, "the draft width is the profile's, and a block holds it"
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
