"""Structured output (base): a grammar per request, a matcher per row, one bitmask for the whole step -- xgrammar.

OpenAI `response_format` json_object / json_schema become a compiled grammar (cached per spec); every rank builds
its own matcher for a row and advances it with the same committed tokens, so the masks agree without a message.
With drafts, the masks for the K+1 positions are produced by walking the drafts through a fork of the matcher:
a draft the grammar refuses ends the row's live positions -- the target's pick at that position cannot be the
draft, so acceptance stops there, the same acceptance rule as everywhere, nothing special-cased.

Three things are the step's, not the row's, and that is what makes this cheap:

* **One buffer.** The step's masks are filled into a single pinned bitmask and cross to the device in one
  transfer, issued before the forward is launched (`prepare`) -- the walk needs only the matcher's committed
  state and this step's drafts, both known by then, so the host's fill and its copy ride under the device's
  work instead of landing between the forward and the pick.
* **No reordering.** vLLM fills compactly, then sorts the rows back into the order of the batch's logits, because
  one kernel call has to cover the whole batch and the speculative offsets have moved every row's positions. We
  fill compactly too and hand each row *its own slice* to the kernel, so there is no order to restore -- and a
  row whose grammar refused a draft packs only the positions that are still alive, not the dead ones behind it.
* **The kernel.** `xgrammar.apply_token_bitmask_inplace` writes -inf straight from the packed words -- one
  launch for a row's positions, one int32 read per 32 tokens. Expanding a bitmask into a vocabulary of `bool`
  and then `masked_fill`-ing it costs five launches and ~2 MB of traffic per position instead.
"""
from __future__ import annotations

import json
import threading


def available() -> bool:
    try:
        import xgrammar  # noqa: F401
        return True
    except ImportError:
        return False


class Grammars:
    """Compiled grammars keyed by spec, over one tokenizer (the checkpoint's, as a transformers tokenizer),
    and the one bitmask every row of a step is filled into."""

    def __init__(self, hf_tokenizer, vocab_size: int):
        import xgrammar as xgr
        self.xgr = xgr
        info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer, vocab_size=vocab_size)
        self.vocab_size = vocab_size
        self.compiler = xgr.GrammarCompiler(info)
        self.words = int(xgr.allocate_token_bitmask(1, vocab_size).shape[-1])
        self._cache = {}
        self._lock = threading.Lock()
        self.staging = None                      # pinned [step positions, words]: xgrammar fills on the host
        self.landing = None                      # device [step positions, words]: one transfer for the whole step
        self.crossed = None                      # the event that says the last transfer has finished reading staging

    def compile(self, spec: dict):
        key = json.dumps(spec, sort_keys=True)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        if spec["type"] == "json_object":
            compiled = self.compiler.compile_builtin_json_grammar()
        elif spec["type"] == "json_schema":
            compiled = self.compiler.compile_json_schema(spec["schema"])
        else:
            raise ValueError(f"unknown grammar spec {spec['type']!r}")
        with self._lock:
            self._cache[key] = compiled
        return compiled

    def matcher(self, spec: dict, max_rollback: int):
        return Matcher(self, self.compile(spec), max_rollback)

    def _buffers(self, positions: int, device):
        """The two sides of one transfer, kept between steps and grown only upwards. The caller has already
        waited for the last transfer: growing frees the staging it was reading from."""
        import torch
        want = torch.device(device)
        pinned = want.type == "cuda"                       # pinning is meaningless without a device to send to
        if self.staging is None or self.staging.shape[0] < positions or self.staging.is_pinned() != pinned:
            self.staging = torch.empty(positions, self.words, dtype=torch.int32, pin_memory=pinned)
        if self.landing is None or self.landing.shape[0] < positions or self.landing.device != want:
            self.landing = torch.empty(positions, self.words, dtype=torch.int32, device=want)
            self.crossed = torch.cuda.Event() if pinned else None
        return self.staging, self.landing

    def prepare(self, rows, device) -> "StepMasks":
        """This step's masks for every row that carries a grammar. `rows` is (key, matcher, drafts).

        Call this before the forward is launched. Everything here is host work and one host-to-device copy, and
        none of it depends on the forward: the device covers it.
        """
        if self.crossed is not None:
            self.crossed.synchronize()        # the last transfer is done reading the staging we are about to rewrite
        staging, landing = self._buffers(sum(len(d) + 1 for _, _, d in rows), device)   # (or free, if this step is the widest yet)
        filled, at = {}, 0
        for key, matcher, drafts in rows:
            live, needed = matcher.fill(staging, at, drafts)
            filled[key] = (at, live, needed)
            at += live                        # a row that died early packs only what is alive: the step carries no holes
        landing[:at].copy_(staging[:at], non_blocking=self.crossed is not None)   # one crossing for the step
        if self.crossed is not None:
            self.crossed.record()
        return StepMasks(self, landing, filled)

    def warm(self, device) -> None:
        """Pay the mask kernel's compile at boot, not inside the first structured request: xgrammar's CUDA backend
        is a Triton kernel, and its JIT is a second that would otherwise land in a served step."""
        import torch
        staging, landing = self._buffers(1, device)
        staging[0].fill_(-1)                                                   # every token allowed: a no-op mask
        landing[:1].copy_(staging[:1])
        self.xgr.apply_token_bitmask_inplace(torch.zeros(1, self.vocab_size, device=device), landing[:1],
                                             vocab_size=self.vocab_size)


class StepMasks:
    """One step's masks, on the device, in the row order the step's logits already have."""

    def __init__(self, grammars: Grammars, landing, filled: dict):
        self.g, self.landing, self.filled = grammars, landing, filled

    def has(self, key) -> bool:
        return key in self.filled

    def live(self, key, positions: int) -> int:
        """How many of the row's positions the grammar can reach: all of them for a row without one, and up to
        the first draft the grammar refuses for a row with one. The positions behind that draft are dead -- they
        are never gathered, processed or picked. (vLLM writes -1 over such a draft so that its device-side
        rejection sampler drops it; this is the same fact, one step earlier, where it can still save the work.)"""
        row = self.filled.get(key)
        return positions if row is None else min(positions, row[1])

    def apply(self, key, logits) -> None:
        """Mask `logits` [n, V] in place for the row's first n positions -- one launch, one int32 read per 32
        tokens. The row owns a contiguous slice of the step's bitmask, so the kernel is handed that slice and
        needs no index list; there is no way for one row's mask to land on another row's logits."""
        row = self.filled.get(key)
        if row is None or not row[2]:                      # the grammar refuses nothing here: there is no mask to apply
            return
        off = row[0]
        self.g.xgr.apply_token_bitmask_inplace(logits, self.landing[off: off + logits.shape[0]],
                                               vocab_size=self.g.vocab_size)


class Matcher:
    """One row's grammar state: fill the step's positions, then advance by what was committed."""

    def __init__(self, grammars: Grammars, compiled, max_rollback: int):
        self.g = grammars
        self.m = grammars.xgr.GrammarMatcher(compiled, max_rollback_tokens=max_rollback)

    def fill(self, bitmask, at: int, drafts: "list[int]") -> "tuple[int, bool]":
        """Fill rows `at` .. `at + len(drafts)` of `bitmask` with this row's mask for each of the step's
        positions: position i assumes drafts[:i] were accepted. Returns (live positions, whether any of them
        refuses anything).

        A draft the grammar refuses ends the row there. A terminated grammar allows only its stop tokens (the
        mask xgrammar fills there). The walk is taken back before returning: only committed tokens advance a
        matcher, and that is `advance`.

        No `reset_token_bitmask` first -- `fill_next_token_bitmask` writes the whole row it is given, verified
        against a row left dirty, so resetting would only be a second 19 KB memset per position (as expensive
        as the fill itself, measured: 1.6 us against 1.3 us at GLM-5.3's vocabulary).
        """
        walked = live = 0
        needed = False
        try:
            for i in range(len(drafts) + 1):
                # `fill_next_token_bitmask` answers whether the mask refuses anything at all; older builds
                # answer nothing, and then every position is taken as needing the mask.
                needed |= self.m.fill_next_token_bitmask(bitmask, at + i) is not False
                live += 1
                if i < len(drafts):
                    if self.m.is_terminated() or not self.m.accept_token(drafts[i]):
                        break
                    walked += 1
        finally:
            if walked:
                self.m.rollback(walked)
        return live, needed

    def advance(self, tokens: "list[int]") -> None:
        for t in tokens:
            if self.m.is_terminated():
                return
            if not self.m.accept_token(t):
                raise ValueError(f"committed token {t} is outside the grammar (the mask should have refused it)")

    @property
    def terminated(self) -> bool:
        return bool(self.m.is_terminated())
