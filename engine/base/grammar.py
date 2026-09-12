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

    def masks(self, drafts: "list[int]", device) -> "list[torch.Tensor | None]":
        """Allowed-token masks [V] bool for positions 0..len(drafts): position i assumes drafts[:i] were accepted.
        A draft the grammar refuses ends the list (later positions are never reached); a terminated grammar
        allows only its stop tokens (the mask xgrammar fills there)."""
        import torch
        out = []
        walked = 0
        try:
            for i in range(len(drafts) + 1):
                self.g.xgr.reset_token_bitmask(self.bitmask)
                self.m.fill_next_token_bitmask(self.bitmask)
                bits = self.bitmask[0].to(device)
                mask = ((bits.unsqueeze(-1) >> torch.arange(32, device=device, dtype=torch.int32)) & 1).bool().reshape(-1)[: self.g.vocab_size]
                out.append(mask)
                if i < len(drafts):
                    if self.m.is_terminated() or not self.m.accept_token(drafts[i]):
                        break
                    walked += 1
        finally:
            if walked:
                self.m.rollback(walked)
        return out

    def advance(self, tokens: "list[int]") -> None:
        for t in tokens:
            if self.m.is_terminated():
                return
            if not self.m.accept_token(t):
                raise ValueError(f"committed token {t} is outside the grammar (the mask should have refused it)")

    @property
    def terminated(self) -> bool:
        return bool(self.m.is_terminated())
