"""Structured output (base): a grammar per request, a matcher per row, a token mask per step -- xgrammar.

OpenAI `response_format` json_object / json_schema become a compiled grammar (cached per spec); every rank builds
its own matcher for a row and advances it with the same committed tokens, so the masks agree without a message.
With drafts, the masks for the K+1 positions are produced by walking the drafts through a fork of the matcher:
a draft the grammar refuses is masked out of the target's pick at that position, so the target's pick differs
from the draft and acceptance stops there -- the same acceptance rule as everywhere, nothing special-cased.
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
    """Compiled grammars keyed by spec, over one tokenizer (the checkpoint's, as a transformers tokenizer)."""

    def __init__(self, hf_tokenizer, vocab_size: int):
        import xgrammar as xgr
        self.xgr = xgr
        info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer, vocab_size=vocab_size)
        self.vocab_size = vocab_size
        self.compiler = xgr.GrammarCompiler(info)
        self._cache = {}
        self._lock = threading.Lock()

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


class Matcher:
    """One row's grammar state: masks for a step's positions, then advance by what was committed."""

    def __init__(self, grammars: Grammars, compiled, max_rollback: int):
        self.g = grammars
        self.m = grammars.xgr.GrammarMatcher(compiled, max_rollback_tokens=max_rollback)
        self.bitmask = grammars.xgr.allocate_token_bitmask(1, grammars.vocab_size)
        self.words = int(self.bitmask.shape[-1])
        self.staging = None                      # pinned [positions, words]: xgrammar fills on the host
        self.landing = None                      # device [positions, words]: one transfer for the whole step

    def _buffers(self, positions: int, device):
        """The two sides of one transfer, kept between steps.

        xgrammar fills its bitmask on the host, so every position used to cross to the device
        on its own -- a pageable copy, which synchronizes, once per draft position per row.
        Staging in pinned memory makes the crossing a direct transfer, and filling every
        position before crossing makes it one transfer instead of `positions` of them.
        """
        import torch
        if self.staging is None or self.staging.shape[0] < positions:
            pinned = str(device).startswith("cuda")        # pinning is meaningless without a device to send to
            self.staging = torch.empty(positions, self.words, dtype=torch.int32, pin_memory=pinned)
        want = torch.device(device)
        if self.landing is None or self.landing.shape[0] < positions or self.landing.device != want:
            self.landing = torch.empty(positions, self.words, dtype=torch.int32, device=want)
        return self.staging, self.landing

    def masks(self, drafts: "list[int]", device) -> "list[torch.Tensor | None]":
        """Allowed-token masks [V] bool for positions 0..len(drafts): position i assumes drafts[:i] were accepted.
        A draft the grammar refuses ends the list (later positions are never reached); a terminated grammar
        allows only its stop tokens (the mask xgrammar fills there)."""
        import torch
        from engine.base.constants import iota
        positions = len(drafts) + 1
        staging, landing = self._buffers(positions, device)
        walked = filled = 0
        try:
            for i in range(positions):
                self.g.xgr.reset_token_bitmask(self.bitmask)
                self.m.fill_next_token_bitmask(self.bitmask)
                staging[i].copy_(self.bitmask[0])
                filled += 1
                if i < len(drafts):
                    if self.m.is_terminated() or not self.m.accept_token(drafts[i]):
                        break
                    walked += 1
        finally:
            if walked:
                self.m.rollback(walked)
        # One crossing for the step. It blocks, deliberately: a non-blocking copy would let the
        # next step overwrite the staging while this one's transfer was still reading it.
        landing[:filled].copy_(staging[:filled])
        shift = iota(32, device, torch.int32)
        bits = landing[:filled].unsqueeze(-1).bitwise_right_shift(shift).bitwise_and_(1)
        return list(bits.ne(0).reshape(filled, -1)[:, : self.g.vocab_size])

    def advance(self, tokens: "list[int]") -> None:
        for t in tokens:
            if self.m.is_terminated():
                return
            if not self.m.accept_token(t):
                raise ValueError(f"committed token {t} is outside the grammar (the mask should have refused it)")

    @property
    def terminated(self) -> bool:
        return bool(self.m.is_terminated())
