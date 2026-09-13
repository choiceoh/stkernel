"""Multi-token prediction heads (module): the drafter all seven models ship -- a small head that reads the target's
state at a position and the token after it, and predicts the token after that -- as compositions over the same
features, and the engine/base/composed Drafter that runs them.

A head is a `Composition` (engine/base/composition): its layers are the target's own features at an `offset` past the
target's layers (their rows live in the same blocks, `store_for(also=...)`), its head is usually the target's; what
makes it a draft head is `fuse`, how it opens from (the next token's embedding, the target's state):

    fuse_streams   Qwen3.8 (vLLM qwen3_8_flash_next nvidia/mtp.py, "residual_linear_shared"): the target's state is
                   its multi-stream residual BEFORE the final mixer [N, hc*H]; RMS-normalised (1 + w) over all hc*H
                   jointly, projected per stream by one shared matrix, and the next token's embedding -- normalised (1 + w)
                   and projected -- added to every stream. The head's layer is a full Qwen3.8 layer (QSA + MoE under the
                   gated streams, no PLE); its final mixer collapses the streams for the shared lm_head.
    fuse_concat    DeepSeek-V3's form (GLM-5.3's MTP, and the configs of Kimi K3, Ling-3.0, MiniMax-M3): project
                   [norm(embedding) ; norm(state)] (or state first: Inkling) to H, the embedding zeroed at position 0.
                   Not held to an oracle here.

The chain (`MTPDrafter`): after the target keeps positions ctx..ctx+L-1, the head runs over the same positions with the
tokens after them (`observe`) -- its rows at those positions are real -- and its prediction at the last is draft 1.
Draft i+1 runs the head at the next position with draft i and the head's own pre-mix state from the step before,
head layer i % n (Qwen3.8 has one layer, reused; vLLM `spec_step_idx % num_mtp_layers`). Those chain rows are
provisional: the drafter's lane steps back to the target's context after proposing, and the next observe overwrites
them. Drafts are argmax (the engine samples the target; a draft only has to be a good guess). vLLM reuses a Qwen3.8
head's step-0 sparse selection on later chain steps; this runs the selection every step.

The head's state is a lane of the target's store (PositionStore.lane): the same blocks and open sequences, contexts of
its own. A head whose features keep per-sequence values would need verify rings of its own and is refused.
"""
from __future__ import annotations

import torch

from engine.base.composition import Step
from engine.modules.norm import rmsnorm, rmsnorm_unit_offset


def fuse_streams(embeddings: torch.Tensor, given: torch.Tensor, weights, *, hc: int, hidden: int, eps: float) -> torch.Tensor:
    """Qwen3.8's MTP fuse -> [N, hc*H]. `weights(name)`: embed_norm [H], embed_proj [H, H], hidden_norm [hc*H],
    hidden_proj [H, H]."""
    linear = torch.nn.functional.linear
    n = given.shape[0]
    embedded = linear(rmsnorm_unit_offset(embeddings, weights("embed_norm"), eps), weights("embed_proj"))
    streams = rmsnorm_unit_offset(given.reshape(n, hc * hidden), weights("hidden_norm"), eps).view(n, hc, hidden)
    streams = linear(streams, weights("hidden_proj"))
    return (embedded.unsqueeze(-2) + streams).flatten(-2)


def fuse_concat(embeddings: torch.Tensor, given: torch.Tensor, weights, *, eps: float, positions: torch.Tensor,
                norm: str = "rms", state_first: bool = False, zero_first: bool = True) -> torch.Tensor:
    """DeepSeek-V3's MTP fuse -> [N, H]: proj([norm_e(embedding) ; norm_h(state)]). `weights(name)`: embed_norm,
    hidden_norm [H], proj [H, 2H]. Not held to an oracle."""
    normalise = rmsnorm_unit_offset if norm == "rms_unit_offset" else rmsnorm
    embedded = normalise(embeddings, weights("embed_norm"), eps)
    if zero_first:
        embedded = embedded.masked_fill((positions == 0)[:, None], 0)
    state = normalise(given, weights("hidden_norm"), eps)
    parts = [state, embedded] if state_first else [embedded, state]
    return torch.nn.functional.linear(torch.cat(parts, dim=-1), weights("proj"))


class MTPDrafter:
    """engine/base/composed's Drafter over MTP heads (the module docstring). `heads`: one Composition per head layer,
    each with a `fuse`; `store`: the target's PositionStore (the heads' specs were laid out with it); `k`: drafts a
    step; `vocab`: the ids a draft may take."""

    def __init__(self, heads, store, *, k: int, vocab: int):
        if not heads or type(k) is not int or k < 1:
            raise ValueError("an MTP drafter needs at least one head and k >= 1")
        for head in heads:
            if head.fuse is None:
                raise ValueError("an MTP head opens from the target's state: it needs a fuse")
            if head.cache_specs()[1]:
                raise ValueError("an MTP head that keeps per-sequence values would need verify rings of its own")
        self.heads, self.k, self.vocab = list(heads), k, vocab
        self.lane = store.lane()
        self.carry: dict = {}                       # seq -> (the head's pre-mix state at the last kept position, its logits)

    def observe(self, seq: int, ctx: int, next_ids, hidden: torch.Tensor) -> None:
        if not len(next_ids):
            return
        self.lane.place(seq, ctx)
        ids = torch.tensor(list(next_ids), dtype=torch.int64, device=hidden.device)
        logits, state = self.heads[0].forward(Step.of([(seq, ctx, ids)]), self.lane, hidden=True, given=hidden)
        self.carry[seq] = (state[-1:].clone(), logits[0])

    def propose(self, seqs) -> "list[list[int]]":
        out = []
        for seq in seqs:
            held = self.carry.get(seq)
            if held is None or seq not in self.lane.contexts:
                out.append([])
                continue
            state, logits = held
            base = self.lane.contexts[seq]
            drafts = [int(logits[:self.vocab].argmax())]
            for i in range(1, self.k):
                head = self.heads[i % len(self.heads)]
                token = torch.tensor([drafts[-1]], dtype=torch.int64, device=state.device)
                step_logits, state = head.forward(Step.of([(seq, base + i - 1, token)]), self.lane, hidden=True, given=state)
                drafts.append(int(step_logits[0, :self.vocab].argmax()))
            self.lane.place(seq, base)                   # the chain's rows are provisional: the next observe rewrites them
            out.append(drafts)
        return out

    def forget(self, seq: int) -> None:
        self.carry.pop(seq, None)
        self.lane.contexts.pop(seq, None)


__all__ = ["fuse_streams", "fuse_concat", "MTPDrafter"]
