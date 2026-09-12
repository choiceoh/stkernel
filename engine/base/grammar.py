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

Two things the grammar must not do, both learned from SGLang's `constrained/` (45차 §32):

* **It must not start inside the model's reasoning.** A row whose prompt ends inside a think block generates
  its reasoning first, and a JSON grammar forbids every token of it -- including the block's own end token, so
  the block never closes and the whole answer lands in `reasoning_content` with `content` empty. `after` holds
  the token the grammar waits for; until it is committed the row is unconstrained and the matcher does not move.
* **It must not take the engine down.** A schema xgrammar cannot build (a regex backreference, a `$ref` that
  goes nowhere) raises from its C++ layer, and that exception, raised where a row is admitted, would abort every
  live request on every rank. `compile` turns those into `ValueError`, which the door answers as a bad request.

Compiling is also not free -- a wide regex measured 283 ms -- so it happens on a thread and is waited for at the
first mask, which for a thinking row is after the reasoning and for any row is after its prompt.
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

    def __init__(self, hf_tokenizer, vocab_size: int, stop_token_ids=None):
        import xgrammar as xgr
        from concurrent.futures import ThreadPoolExecutor
        self.xgr = xgr
        # The engine's end tokens are the authority on where a generation may end, so they are the grammar's
        # stop tokens too. Left to itself xgrammar takes the tokenizer's single `eos_token`, and a model whose
        # generation config ends on something else would finish its JSON on a token the engine does not stop
        # at -- the grammar then allows nothing but that token, and the row runs to its limit repeating it.
        info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer, vocab_size=vocab_size,
                                                  stop_token_ids=sorted(stop_token_ids) if stop_token_ids else None)
        self.vocab_size = vocab_size
        self.compiler = xgr.GrammarCompiler(info)
        self.words = int(xgr.allocate_token_bitmask(1, vocab_size).shape[-1])
        self._cache = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grammar")
        self.staging = None                      # pinned [step positions, words]: xgrammar fills on the host
        self.landing = None                      # device [step positions, words]: one transfer for the whole step
        self.crossed = None                      # the event that says the last transfer has finished reading staging

    def _build(self, spec: dict):
        """The compile itself. Everything it can fail at is the request's schema, so those failures are the
        door's to answer (`ValueError` -> 400) and not the engine's to die of (D3 is about the engine)."""
        try:
            if spec["type"] == "json_object":
                return self.compiler.compile_builtin_json_grammar()
            if spec["type"] == "json_schema":
                return self.compiler.compile_json_schema(spec["schema"])
            if spec["type"] == "ebnf":
                # A grammar the profile wrote, for a wire format that is not JSON -- the tool-call
                # shape the chat template teaches (45차 §45). The door builds it; this compiles it.
                return self.compiler.compile_grammar(self.xgr.Grammar.from_ebnf(spec["grammar"]))
        except (RuntimeError, TypeError, ValueError, UnicodeError) as exc:
            raise ValueError(f"the grammar cannot be compiled: {exc}") from exc
        raise ValueError(f"unknown grammar spec {spec['type']!r}")

    def compile(self, spec: dict):
        """A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by key, so a second
        row asking for the same schema waits on the same work instead of doing it again."""
        key = json.dumps(spec, sort_keys=True)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            started = self._pool.submit(self._build, spec)
            self._cache[key] = started
            return started

    def resolve(self, handle):
        """The compiled grammar, waiting for its thread if it is still running (and raising what it raised)."""
        if hasattr(handle, "result"):
            handle = handle.result()
        return handle

    def ready(self, spec: dict):
        """Compile now and answer for it -- what the door asks before a request is admitted, so that a schema
        xgrammar refuses is a 400 and not a step that raises on four ranks at once."""
        self.resolve(self.compile(spec))

    def matcher(self, spec: dict, max_rollback: int, after: "int | None" = None):
        return Matcher(self, self.compile(spec), max_rollback, after)

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

    def qualify(self, device) -> None:
        """Prove the mask on this device before the door opens (D3), and pay xgrammar's Triton JIT here rather
        than inside the first structured request.

        A boot that cannot mask cannot serve `response_format`, and it should say so here and not in an answer.
        So it is asked for a real mask -- what the builtin JSON grammar allows at its first position -- and the
        kernel's verdict on the device is compared with the same packed words expanded on the host. The mask
        must also be a proper subset: all-true or all-false at a JSON start means the tokenizer and the head
        disagree about the vocabulary, which every later mask would inherit silently.
        """
        import torch
        masks = self.prepare([(0, self.matcher({"type": "json_object"}, 1), [])], device)
        at, live, needed = masks.filled[0]
        logits = torch.zeros(1, self.vocab_size, device=device)
        masks.apply(0, logits)
        shift = torch.arange(32, dtype=torch.int32)
        words = self.landing[at: at + live].cpu().unsqueeze(-1)
        want = words.bitwise_right_shift(shift).bitwise_and_(1).reshape(live, -1)[:, : self.vocab_size].ne(0)
        got = ~torch.isinf(logits.cpu())
        if not needed or not torch.equal(want, got):
            raise RuntimeError(f"the grammar mask kernel disagrees with the bitmask it was given on "
                               f"{int((want != got).sum())} of {self.vocab_size} ids (needed={needed})")
        if not bool(want.any()) or bool(want.all()):
            raise RuntimeError(f"the grammar allows {int(want.sum())} of {self.vocab_size} ids at a JSON start: "
                               "the tokenizer and the head do not agree on the vocabulary")


class StepMasks:
    """One step's masks, on the device: a contiguous slice per row, in the order the rows were filled."""

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
    """One row's grammar state: fill the step's positions, then advance by what was committed.

    Dormant until `after` is committed when the row asked for it (the reasoning end): the model writes its
    reasoning unconstrained, and the grammar takes over at the first token of the answer.
    """

    def __init__(self, grammars: Grammars, handle, max_rollback: int, after: "int | None" = None):
        self.g = grammars
        self.handle, self.max_rollback = handle, max_rollback
        self.m = None                            # built at the first mask: the compile may still be running
        self.after = after
        self.armed = after is None

    @property
    def matcher(self):
        if self.m is None:
            self.m = self.g.xgr.GrammarMatcher(self.g.resolve(self.handle), max_rollback_tokens=self.max_rollback)
        return self.m

    def fill(self, bitmask, at: int, drafts: "list[int]") -> "tuple[int, bool]":
        """Fill rows `at` .. `at + len(drafts)` of `bitmask` with this row's mask for each of the step's
        positions: position i assumes drafts[:i] were accepted. Returns (live positions, whether any of them
        refuses anything).

        A draft the grammar refuses ends the row there. Accepting a stop token ends the walk after that token's
        position: xgrammar cannot fill another mask after termination. The walk is taken back before returning: only committed tokens advance a
        matcher, and that is `advance`.

        While the row is dormant nothing is written and nothing is refused, so a step of pure reasoning costs
        the grammar nothing at all. A step where the drafts cross `after` is the one case that has to write:
        the positions before the crossing are opened by hand, because the kernel applies the row's slice whole.

        No `reset_token_bitmask` first -- `fill_next_token_bitmask` writes the whole row it is given, verified
        against a row left dirty, so resetting would only be a second 19 KB memset per position (as expensive
        as the fill itself, measured: 1.6 us against 1.3 us at GLM-5.3's vocabulary).
        """
        dormant = not self.armed                 # the walk may arm it; only a committed token really does
        walked = live = 0
        first = None                             # the first position the grammar constrains
        needed = False
        try:
            for i in range(len(drafts) + 1):
                if self.armed:
                    if first is None:
                        first = i
                    # `fill_next_token_bitmask` answers whether the mask refuses anything at all; older builds
                    # answer nothing, and then every position is taken as needing the mask.
                    needed |= self.matcher.fill_next_token_bitmask(bitmask, at + i) is not False
                live += 1
                if i < len(drafts):
                    if self.armed:
                        if self.matcher.is_terminated() or not self.matcher.accept_token(drafts[i]):
                            break
                        walked += 1
                        if self.matcher.is_terminated():
                            break               # the stop token has a mask; positions after it are dead
                    elif drafts[i] == self.after:
                        self.armed = True        # the reasoning ends here: the answer's first token is the next one
        finally:
            if walked:
                self.matcher.rollback(walked)
            if dormant:
                self.armed = False               # `advance` arms it, on a token that was really committed
        if needed and first:
            bitmask[at: at + first].fill_(-1)    # the dormant positions the kernel will sweep with the rest: open them
        return live, needed

    def advance(self, tokens: "list[int]") -> None:
        for t in tokens:
            if not self.armed:
                self.armed = t == self.after
                continue
            if self.matcher.is_terminated():
                return
            if not self.matcher.accept_token(t):
                raise ValueError(f"committed token {t} is outside the grammar (the mask should have refused it)")
