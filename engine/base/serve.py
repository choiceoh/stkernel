"""Bounded request admission and the rank-0 HTTP door.

Public request IDs are independent of the runner's reusable KV rows. Every
rank receives the same FIFO, admits only requests whose entire decode horizon
fits the declared budget, and recycles rows after copying out results.
Kernel failures remain fatal; waiting HTTP clients receive an error on exit.

With a tier, a conversation is its first request's id and lives on NVMe
between turns: a finished turn is parked (blocks, state slot, host record)
and its row and slot are free once the write is done, a continuation resumes
into whichever row is free once the read is done. Both run on the tier's
thread; the step loop only asks whether they are done, and every rank agrees
on that answer before acting (the ranks stay in lockstep). Retained
conversations are bounded by the disk, not by rows; when the tier is full
the least recently parked conversation is forgotten.

The door speaks two dialects: the engine's own (`POST /v1/completions` with
ids or a prompt, and `conversation` for a further turn on retained caches)
and the OpenAI chat one every client and bench here already speaks
(`POST /v1/chat/completions`, streamed as SSE or not, `GET /v1/models`,
`GET /metrics` in the bench's counter names, `GET /health`). Chat needs a
`chat` renderer (messages -> prompt text; the profile supplies the
checkpoint's template) and a tokenizer. A `reasoning_end` token id splits the
generation into `reasoning_content` and `content`, the way the served model
writes them. Streaming is by token: the step loop hands each request's new
tokens to its queue and the HTTP thread turns them into text deltas, holding
back a partial multi-byte character until its next token completes it.
"""
from __future__ import annotations

import base64
import heapq
import json
import math
import queue
import re
import select
import socket
import threading
import time
import unicodedata
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine.base import prefix as prefix_cache
from engine.base.kv_tier import TierFull

MEDIA_FETCH_TIMEOUT_S = {"image": 5.0, "video": 30.0}           # vLLM's VLLM_IMAGE_FETCH_TIMEOUT / VLLM_VIDEO_FETCH_TIMEOUT defaults
MEDIA_MAX_BYTES = {"image": 64 << 20, "video": 512 << 20}        # a door-side ceiling on what one part may carry (vLLM has none)


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_SAMPLING_RANGES = {"presence_penalty": (-2.0, 2.0), "frequency_penalty": (-2.0, 2.0)}

def _grammar_rows(grammars):
    """What the compiled-grammar cache is holding. Empty when this boot serves no structured output."""
    cache = getattr(grammars, "_cache", None)
    if grammars is None or cache is None:
        return ()
    return (("gauge", "st:grammar_cache_entries", "compiled grammars held", len(cache)),
            ("gauge", "st:grammar_cache_limit", "the declared ceiling on that", grammars.KEPT),
            ("counter", "st:grammar_compiles_total", "grammars compiled, cache misses included", grammars.compiles),
            ("counter", "st:grammar_cache_hits_total", "requests that reused a compiled grammar", grammars.cache_hits),
            ("counter", "st:grammar_cache_evictions_total", "compiled grammars dropped at the ceiling",
             grammars.cache_evictions))


class PromptTokens:
    """Tokenize a continuing turn by its new tail instead of from the top.

    An agent sends its whole conversation every turn, so the door tokenizes the same hundred
    thousand characters again to add twenty. Measured on this checkpoint: **239 ms** to
    re-tokenize a 106K-token conversation against **0.05 ms** for the 29 tokens that were
    actually new. While a turn is dominated by prefill that hides; once the prefix cache is
    doing its job -- which for agent traffic is the normal case -- the model barely works on
    such a turn and that 239 ms becomes its floor (45차 §61).

    Splicing two token streams is sound only where no merge can cross the cut. `tokenizers`
    matches ADDED tokens before the BPE model, so an added token is exactly such a cut, and a
    rendered chat prompt ends with one by construction (the generation prompt). The rule is
    therefore read off the tokenizer's own added-token set rather than guessed from a template,
    and a base that does not end on one is simply not spliced against. Verified both ways: at
    `<|assistant|>` the splice is identical to a full pass; cut mid-word it is not (47 tokens
    against 45), which is why the check is not optional.

    Bounded by entries and by characters, because a door that remembers every prompt is a leak
    wearing a cache's clothes.
    """

    def __init__(self, tok, keep: int = 8, max_chars: int = 8 << 20):
        self.tok, self.keep, self.max_chars = tok, keep, max_chars
        self.splice_ids = set()
        try:                                            # tokenizers >= 0.20; without it, nothing splices
            self.splice_ids = set(tok.get_added_tokens_decoder() or {})
        except Exception:                               # noqa: BLE001 -- an absent API is not an error here
            pass
        self.entries: "list[tuple[str, list]]" = []     # (rendered text, its ids), most recent last
        self.chars = 0
        self.spliced = self.full = self.chars_saved = 0
        self._lock = threading.Lock()

    def _base_for(self, text: str):
        with self._lock:
            for base, ids in reversed(self.entries):
                if len(base) < len(text) and ids and ids[-1] in self.splice_ids and text.startswith(base):
                    return base, ids
        return None, None

    def _remember(self, text: str, ids) -> None:
        with self._lock:
            self.entries.append((text, ids))
            self.chars += len(text)
            while len(self.entries) > self.keep or (self.chars > self.max_chars and len(self.entries) > 1):
                gone, _ = self.entries.pop(0)
                self.chars -= len(gone)

    def encode(self, text: str) -> list:
        """This prompt's ids, spliced onto a remembered prefix when one is a legal cut."""
        base, ids = self._base_for(text)
        if base is not None:
            out = list(ids) + self.tok.encode(text[len(base):], add_special_tokens=False).ids
            self.spliced += 1
            self.chars_saved += len(base)
        else:
            out = self.tok.encode(text, add_special_tokens=False).ids
            self.full += 1
        self._remember(text, out)
        return out


EFFORT_RUNGS = {"low": "low", "medium": "high", "high": "high", "max": "max"}
"""OpenAI's rungs onto GLM-5.3's two, mapped on purpose instead of by falling through.

The template reads `reasoning_effort in ['low', 'high']` and turns EVERYTHING ELSE into 'max'.
So an ordinary OpenAI `"medium"` silently buys the deepest setting there is -- the opposite of
what the caller asked for. Refusing it was wrong the other way: `medium` is a standard value of
the API this door claims to speak, and the Deneb gateway sends it whenever its thinking budget
lands between 4K and 10K tokens. That 400 does not fail over either, because wormhole
deliberately does not treat a request-shape 4xx as transient -- it goes straight back to the
caller as a dead turn (45차 §59).

`medium` therefore maps to `high`: the order survives (low <= medium <= high <= max) and nothing
buys `max` by accident. A caller who wants a real ceiling has `reasoning_budget`, which counts
tokens instead of naming a rung.
"""


def cache_key(req: dict) -> "str | None":
    """The caller's own string for the cache, under either name. `cache_salt` is vLLM's; `prompt_cache_key` is the
    OpenAI field an agent's SDK already sends, so accepting it is the difference between an agent getting tenant
    isolation for free and not knowing the engine has any. Ours is stronger than OpenAI's hint -- it is folded into
    the boundary chain, so two keys can never read each other's prefixes -- and the cost is the same one OpenAI warns
    about: a key that changes every call shares nothing. Both names at once is a contradiction, not a default (D3)."""
    salt, key = req.get("cache_salt"), req.get("prompt_cache_key")
    if salt is not None and key is not None and salt != key:
        raise RequestError("cache_salt and prompt_cache_key are the same field under two names: send one")
    return salt if salt is not None else key


def sampling_options(req: dict, defaults: "dict | None" = None) -> "tuple[float, dict]":
    """(temperature, options) of an OpenAI-dialect request, validated the way the served model's door spells them.

    `defaults` are the checkpoint's generation_config values (temperature 1.0 for GLM-5.3: what vLLM applies when a
    request says nothing). Options beyond temperature travel to the engine, which enforces them (base/serve never
    drops a field silently -- an option the engine cannot serve is refused at submit, D3)."""
    defaults = defaults or {}
    temperature = req.get("temperature", defaults.get("temperature", 0.0))
    if temperature is None:
        temperature = defaults.get("temperature", 0.0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise RequestError("temperature must be finite and nonnegative")
    options = {}
    top_p = req.get("top_p", defaults.get("top_p"))
    if top_p is not None:
        if type(top_p) not in (int, float) or not 0 < top_p <= 1:
            raise RequestError("top_p must be in (0, 1]")
        if top_p < 1:
            options["top_p"] = float(top_p)
    top_k = req.get("top_k", defaults.get("top_k"))
    if top_k is not None and top_k != -1 and top_k != 0:
        if type(top_k) is not int or top_k < 1:
            raise RequestError("top_k must be a positive integer (or -1 for all)")
        options["top_k"] = top_k
    for key, (lo, hi) in _SAMPLING_RANGES.items():
        v = req.get(key)
        if v is not None and v != 0:
            if type(v) not in (int, float) or not lo <= v <= hi:
                raise RequestError(f"{key} must be between {lo} and {hi}")
            options[key] = float(v)
    rp = req.get("repetition_penalty", defaults.get("repetition_penalty"))
    if rp is not None and rp != 1:
        if type(rp) not in (int, float) or not 0 < rp <= 2:
            raise RequestError("repetition_penalty must be in (0, 2]")
        options["repetition_penalty"] = float(rp)
    seed = req.get("seed")
    if seed is not None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise RequestError("seed must be a nonnegative integer")
        options["seed"] = seed
    bias = req.get("logit_bias")
    if bias:
        if not isinstance(bias, dict):
            raise RequestError("logit_bias must be an object of token id -> bias")
        out = {}
        for k, v in bias.items():
            try:
                tid = int(k)
            except (TypeError, ValueError):
                raise RequestError("logit_bias keys must be token ids") from None
            if type(v) not in (int, float) or not -100 <= v <= 100:
                raise RequestError("logit_bias values must be between -100 and 100")
            out[tid] = float(v)
        options["logit_bias"] = out
    stop_ids = req.get("stop_token_ids")
    if stop_ids:
        if not isinstance(stop_ids, list) or any(type(t) is not int or t < 0 for t in stop_ids):
            raise RequestError("stop_token_ids must be a list of token ids")
        options["stop_token_ids"] = list(stop_ids)
    return float(temperature), options


# What a chat answer may be when the request does not say. It is a length in CHARACTERS, not
# in tokens, because a token is not the same amount of answer in every language: on this
# checkpoint it buys 5.8 characters of English and 1.3 of Korean, so one number in tokens is
# four different answers in four languages -- and the shortest of them was Korean, cut at 334
# characters where English got 1,480 (45차 §38). vLLM and SGLang have no cap at all; ours
# stays, because admission reserves a request's whole horizon and never preempts (D3), but it
# is priced in the unit a reader counts.
DEFAULT_ANSWER_CHARS = 1500
DEFAULT_ANSWER_TOKENS = (256, 2048)          # never below what the old token budget bought, never past this


def written_text(messages) -> str:
    """What the person actually wrote in the last turn they wrote, template scaffolding aside.

    The rendered prompt would be cheaper -- it is already tokenized -- but it carries the
    template's own English, and a long system block would read a short Korean question as
    English. The last user turn is the one the answer follows.
    """
    for m in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content
                           if isinstance(part, dict) and part.get("type") == "text")
    return ""


def answer_budget(tok, text: str) -> int:
    """How many tokens `DEFAULT_ANSWER_CHARS` characters cost in the language this was written in.

    The text's own tokens-per-character is the whole measurement: 0.77 for Korean, 0.73 for
    Japanese, 0.54 for Chinese, 0.17 for English on this checkpoint -- and it reads that way
    off six characters of prompt, so a one-line question is enough to price its own answer.
    """
    floor, ceiling = DEFAULT_ANSWER_TOKENS
    if tok is None or not text:
        return floor
    try:
        try:
            n = len(tok.encode(text, add_special_tokens=False).ids)
        except TypeError:                                   # a tokenizer without the switch
            n = len(tok.encode(text).ids)
    except Exception:                                       # noqa: BLE001 -- a budget never fails a request
        return floor
    return max(floor, min(ceiling, round(DEFAULT_ANSWER_CHARS * n / len(text))))


# UTF-8's lead byte 0xED covers U+D000-U+D7FF, and its continuations are cut short by the
# surrogate hole above U+D7FF. xgrammar compiles a character-class range into byte branches and
# loses a range that ENDS inside that branch, keeping only its exact endpoint: `[가-힣]` admits
# 이 and 힣 and refuses 타 파 하 한 해 호 후 희 흥 -- a sixth of the Hangul block, and the sixth
# Korean uses most (45차 §41). Hangul is the only common script that straddles it.
_ED_BRANCH = 0xD000
_ED_BRANCH_END = 0xD7FF          # a range that reaches past this ends on a whole branch and compiles


def _class_item(pattern: str, i: int) -> "tuple[int | None, int]":
    """(code point, characters it spelled) for the class item at `i`; None where it is a set."""
    c = pattern[i]
    if c != "\\" or i + 1 >= len(pattern):
        return ord(c), 1
    kind = pattern[i + 1]
    width = {"u": 4, "x": 2, "U": 8}.get(kind)
    if width and i + 2 + width <= len(pattern):
        try:
            return int(pattern[i + 2:i + 2 + width], 16), 2 + width
        except ValueError:
            return None, 2
    return None, 2                                   # \d, \w, \\ ... not an endpoint to reason about


def split_surrogate_branch(pattern: str) -> str:
    """A class range that ends inside the 0xED branch, written as two that do not.

    `[가-힣]` becomes `[가-\uCFFF\uD000-힣]`, the same set of characters -- U+CFFF and U+D000 are
    neighbours -- and one xgrammar compiles correctly. Only that one shape is touched: a range
    reaching past U+D7FF ends on a whole branch and already compiles, so it is left alone, as is
    anything this cannot read confidently. A pattern is the caller's.
    """
    if "-" not in pattern or "[" not in pattern:
        return pattern
    out, i, n, inside = [], 0, len(pattern), False
    while i < n:
        c = pattern[i]
        if not inside:
            if c == "\\" and i + 1 < n:
                out.append(pattern[i:i + 2]); i += 2
            else:
                inside = c == "["
                out.append(c); i += 1
            continue
        if c == "]":
            inside = False
            out.append(c); i += 1
            continue
        low, width = _class_item(pattern, i)
        after = i + width
        if low is not None and after + 1 < n and pattern[after] == "-" and pattern[after + 1] != "]":
            high, hwidth = _class_item(pattern, after + 1)
            if high is not None and low < _ED_BRANCH <= high <= _ED_BRANCH_END:
                out.append(pattern[i:after] + "-\uCFFF\uD000-" + pattern[after + 1:after + 1 + hwidth])
                i = after + 1 + hwidth
                continue
        out.append(pattern[i:after]); i = after
    return "".join(out)


def repair_patterns(node):
    """Every `pattern` in a JSON schema, with `split_surrogate_branch` applied."""
    if isinstance(node, dict):
        return {k: (split_surrogate_branch(v) if k == "pattern" and isinstance(v, str) else repair_patterns(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [repair_patterns(v) for v in node]
    return node


def device_memory_rows() -> "list[tuple]":
    """What the box has left, as a scrape can see it (45차 §50).

    The OOM study says the thing to watch is `memory_reserved`, not what is live: on this
    machine the caching allocator maps new pages on churn rather than reusing freed ones, so the
    gap between reserved and allocated is memory lost, not memory cached. And it says the period
    has to be a step, not a scrape -- a boot went from 26 GiB free to 5 in four seconds.

    A scrape every fifteen seconds cannot see a four-second cliff, so the peak is reported
    beside the current value: torch keeps `max_memory_reserved` for nothing, so the high-water
    mark between two scrapes costs no per-step work at all.

    And `MemAvailable`, because that is the number earlyoom actually acts on -- it took this
    engine at 3.64% of the box today (45차 §48) -- and nothing the engine exported could have
    shown anyone that it was coming. vLLM does not export any of these either; its
    `gpu_cache_usage_perc` counts blocks, not bytes.
    """
    rows = []
    try:
        from engine.base.runtime_memory import host_available_bytes
        rows.append(("gauge", "st:host_memory_available_bytes",
                     "MemAvailable: what the box has left, and what earlyoom decides on",
                     host_available_bytes()))
    except Exception:                                   # noqa: BLE001 -- /metrics answers with what it has
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            rows += [("gauge", "st:device_memory_reserved_bytes",
                      "what the allocator holds: on this machine the gap to allocated is lost, not cached",
                      torch.cuda.memory_reserved()),
                     ("gauge", "st:device_memory_reserved_peak_bytes",
                      "the high-water mark of that, which is what a scrape between two cliffs would miss",
                      torch.cuda.max_memory_reserved()),
                     ("gauge", "st:device_memory_allocated_bytes", "the part of it that is live tensors",
                      torch.cuda.memory_allocated()),
                     ("gauge", "st:device_memory_free_bytes", "what the driver says is left", free),
                     ("gauge", "st:device_memory_total_bytes", "what the driver says there is", total)]
    except Exception:                                   # noqa: BLE001
        pass
    return rows


def media_parts(messages) -> "list[tuple[str, str]]":
    """(kind, url) of every image_url / video_url part, in the order the chat template renders them. Text parts
    pass through; any other part type is refused here rather than silently dropped by the template."""
    out = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                raise RequestError("content parts must be objects with a type")
            kind = part["type"]
            if kind == "text":
                if not isinstance(part.get("text"), str):
                    raise RequestError("a text part needs a text string")
                continue
            if kind in ("image_url", "video_url"):
                ref = part.get(kind)
                url = ref.get("url") if isinstance(ref, dict) else ref
                if not isinstance(url, str) or not url:
                    raise RequestError(f"a {kind} part needs a url")
                out.append((kind[:-4], url))
                continue
            raise RequestError(f"content part type {kind!r} is not served (text, image_url, video_url are)")
    return out


def fetch_media(kind: str, url: str) -> bytes:
    """The bytes behind a data: URL or an http(s) URL, within the kind's timeout and size ceiling."""
    limit = MEDIA_MAX_BYTES[kind]
    if url.startswith("data:"):
        head, sep, payload = url.partition(",")
        if not sep or not head.endswith(";base64"):
            raise RequestError(f"{kind}: only base64 data URLs are served")
        if len(payload) > limit * 4 // 3 + 4:
            raise RequestError(f"{kind}: larger than the served {limit >> 20} MiB")
        try:
            return base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as exc:
            raise RequestError(f"{kind}: data URL is not valid base64") from exc
    if url.startswith("http://") or url.startswith("https://"):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "st-engine"}),
                                        timeout=MEDIA_FETCH_TIMEOUT_S[kind]) as r:
                data = r.read(limit + 1)
        except Exception as exc:                                          # noqa: BLE001 -- the fetch's verdict, whatever it was
            raise RequestError(f"{kind}: could not fetch {url!r}: {exc}") from exc
        if len(data) > limit:
            raise RequestError(f"{kind}: larger than the served {limit >> 20} MiB")
        return data
    raise RequestError(f"{kind}: url must be a data: or http(s) URL")


def stop_token_ids_for(stops, tok) -> "list[int]":
    """The stop strings this model can emit as one token, for the engine to end on itself.

    The text scan stays and is still the authority: the same string can arrive as several
    tokens, or straddle a boundary, and only the scan sees that. This lets the common case
    -- a marker that is its own token -- end the row one step earlier and without the door
    having to cancel it afterwards. A token whose text is not exactly the stop string (a
    special token renders as nothing here) is not eligible.
    """
    ids = []
    for stop in stops:
        try:
            encoded = tok.encode(stop, add_special_tokens=False).ids
        except TypeError:                                   # a tokenizer without the switch
            encoded = tok.encode(stop).ids
        if len(encoded) == 1 and tok.decode(encoded) == stop:
            ids.append(encoded[0])
    return ids


def prompt_switches(req: dict) -> "tuple[bool, bool]":
    """`add_generation_prompt` and `continue_final_message`, as OpenAI names them.

    They are opposites -- open a new assistant turn, or resume inside the one already
    there -- so both true is refused rather than left to the template to resolve.
    """
    opening = req.get("add_generation_prompt", True)
    resuming = req.get("continue_final_message", False)
    if not isinstance(opening, bool) or not isinstance(resuming, bool):
        raise RequestError("add_generation_prompt and continue_final_message must be booleans")
    if opening and resuming:
        raise RequestError("add_generation_prompt and continue_final_message cannot both be true")
    return opening, resuming


def nfc(text: str) -> str:
    """`text` in Unicode's composed form, which is the form the model was trained on.

    The same Korean is two different strings to a tokenizer that does not normalise -- and this
    checkpoint's does not (`"normalizer": null`). A syllable written whole (NFC, U+AC00-U+D7A3)
    is one or two tokens; the same syllable written as its jamo (NFD, U+1100-U+11FF) is six or
    eight, and macOS filenames, some IMEs and a lot of extracted text arrive that way. Measured
    here: an NFD Korean prompt costs 6.4x the tokens of the identical NFC one -- 7,721 against
    1,202 -- and hashes differently, so it misses the prefix cache the earlier turn filled.

    NFC and NFD are the same text by Unicode's own definition (canonical equivalence, UAX #15),
    and NFC is what the web recommends for interchange, so this is not a rewrite of what the
    caller sent. It costs 2.5 us on a 1,600-character Korean prompt and 0.1 us on English,
    where it does nothing at all.
    """
    return text if unicodedata.is_normalized("NFC", text) else unicodedata.normalize("NFC", text)


def stop_strings(req: dict) -> "list[str]":
    stop = req.get("stop")
    stop = [stop] if isinstance(stop, str) else (stop or [])
    if not isinstance(stop, list) or any(not isinstance(x, str) or not x for x in stop):
        raise RequestError("stop must be a nonempty string or a list of them")
    # The scan compares these against text the model wrote, which is composed. A stop string in
    # the decomposed form would never match the answer it names.
    return [nfc(x) for x in stop]


# What an answer keeps for itself however long the model thinks. A thinking model can spend the
# whole limit inside the block -- writing a draft, counting its characters, redrafting -- and come
# back with nothing outside it, which is in this stack's own record (29차: 답 0 자). The block is
# bounded so that an answer is always possible, and the bound is a share of the limit rather than
# a number, because the limit is itself a share of a length now (45차 §38, §46).
ANSWER_SHARE = 4                # the answer keeps at least a quarter of the limit ...
ANSWER_FLOOR = 128              # ... and never fewer tokens than this


def reasoning_budget(req: dict, max_tokens: int) -> "int | None":
    """How many tokens this answer's reasoning may take, or None for as many as it likes.

    `reasoning_budget` in the request overrides it; -1 asks for no bound at all, which is what
    llama.cpp's flag of the same name means.
    """
    asked = req.get("reasoning_budget")
    if asked is not None:
        if type(asked) is not int or asked < -1:
            raise RequestError("reasoning_budget must be -1 or a nonnegative integer")
        return None if asked < 0 else asked
    return max(1, max_tokens - max(ANSWER_FLOOR, max_tokens // ANSWER_SHARE))


def response_format_grammar(req: dict) -> "dict | None":
    """OpenAI response_format -> the engine's grammar spec (enforced by the engine's grammar sampler)."""
    fmt = req.get("response_format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or not isinstance(fmt.get("type"), str):
        raise RequestError("response_format must be an object with a type")
    kind = fmt["type"]
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        spec = fmt.get("json_schema")
        schema = spec.get("schema") if isinstance(spec, dict) else None
        if not isinstance(schema, dict):
            raise RequestError("response_format.json_schema.schema must be an object")
        try:
            return {"type": "json_schema", "schema": json.dumps(repair_patterns(schema), sort_keys=True)}
        except (TypeError, ValueError) as exc:
            raise RequestError(f"json_schema is not JSON: {exc}") from exc
    raise RequestError("response_format.type must be text, json_object or json_schema")


def partial_suffix(text: str, needles) -> int:
    """How many trailing characters of `text` could still turn into one of `needles`.

    Showing them would be a mistake that cannot be taken back: a client that reads
    "STO" has read it, and the cut that arrives with the next token comes too late.
    """
    keep = 0
    for needle in needles:
        # Every flush asks this and almost every answer is no, so settle the no in one test: if
        # `text` ends with `needle[:cut]` for any cut below the whole needle, its last character
        # is one of `needle[:-1]`. Without this the loop builds and compares a prefix per length
        # per needle per flush, which measured as the door's largest cost per streamed step.
        if not text or text[-1] not in needle[:-1]:
            continue
        for cut in range(min(len(needle) - 1, len(text)), keep, -1):
            if text.endswith(needle[:cut]):
                keep = cut
                break
    return keep


# Streamed text the door had to repair, by what went wrong: `/metrics` reports it, because a
# repair is the one thing here that changes what a client reads and leaves no other trace.
# Each Server keeps its own (`Server.detok_repairs`) so a scrape names the door that did it;
# this one belongs to whatever has no door -- `token_spans`, and the tests' bare streams.
DETOK_REPAIRS = {"invalid_token_id": 0, "invalid_prefix": 0, "stalled": 0}


def new_repairs() -> dict:
    return dict.fromkeys(DETOK_REPAIRS, 0)

# tokenizers raises this one untyped, so the message is the only way to tell it from a real bug.
# https://github.com/huggingface/tokenizers `DecodeStreamError::InvalidPrefix`
_INVALID_PREFIX = "Invalid prefix encountered"

# How many settled tokens to decode with a held-back tail so it says what it says in place.
# SGLang keeps the same five (`INIT_INCREMENTAL_DETOKENIZATION_OFFSET`).
_CONTEXT_TOKENS = 5

# How far a held-back tail may reach before it is said as it is. This is NOT a bound on how
# long a character may wait: a step commits several tokens at once, so a tail one byte short of
# a character is already `k` tokens long, and two such steps are twice that -- counting four
# bytes' worth of tokens cut Korean answers apart, one U+FFFD per syllable, because Hangul is
# three bytes and the guard fired before the third arrived. It only exists so the decode a held
# step repeats cannot grow without end, so it is eight of the widest speculative steps.
_STALL_TOKENS = 64


def decode_stream(tok, skip_special_tokens: bool = True, ids=None):
    """`tokenizers`' Rust DecodeStream for this tokenizer, or None when that is not what it is.

    The library's own default here is False where `Tokenizer.decode`'s is True, so the flag is
    passed rather than left out: the two paths below must render the same text.

    `ids` primes the stream -- it takes them as already said, without producing their text. Only
    the repair below needs that, and only tokenizers >= 0.22 has it; older ones start empty and
    the repair costs a token's text instead of none.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import DecodeStream
    except ImportError:                                  # base imports without the library
        return None
    if not isinstance(tok, Tokenizer):                   # a wrapper, or one of the tests' fakes
        return None
    if ids:
        try:
            return DecodeStream(ids=list(ids), skip_special_tokens=skip_special_tokens)
        except TypeError:
            pass
    return DecodeStream(skip_special_tokens=skip_special_tokens)


def _decode(tok, ids, repairs=DETOK_REPAIRS) -> str:
    """`tok.decode(ids)`, with an id that is not a token id dropped instead of raised.

    `Tokenizer.decode` raises OverflowError on a negative or oversized id and TypeError on one
    that is not an integer. On the HTTP thread that ends the answer mid-stream and the client
    never learns why, so a bad id costs its own text and nothing else.
    """
    try:
        return tok.decode(ids)
    except (OverflowError, TypeError):
        repairs["invalid_token_id"] += 1
        keep = []
        for tid in ids:
            try:
                tok.decode([tid])
            except (OverflowError, TypeError):
                continue
            keep.append(tid)
        return tok.decode(keep)


def _byte_level_chars() -> "list[str]":
    """GPT-2's byte-to-character table, which a byte-level BPE vocabulary is written in."""
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    table, spare = {}, 0
    for b in range(256):
        table[b] = chr(b) if b in printable else chr(256 + spare)
        spare += b not in printable
    return [table[b] for b in range(256)]


_BYTE_OF_CHAR = {c: b for b, c in enumerate(_byte_level_chars())}


def token_bytes(tok, tid: int, text: str) -> "list[int]":
    """A token's own bytes, for OpenAI's `bytes` field.

    That field exists for exactly one reason, which the spec states: a character can be spread
    over several tokens, and a client rejoins it from the bytes. So a token that is half a
    character is the case the field is for -- and `decode([id])` cannot answer it, because a
    Python string cannot hold half a character. It renders U+FFFD, whose bytes say nothing.
    On this checkpoint that is 32.1% of the tokens of Korean, against 0.0% of English.

    So when the text came back with a replacement character, read the vocabulary entry
    instead: a byte-level entry spells its bytes in GPT-2's table, a byte-fallback one spells
    one byte as `<0xNN>`. Anything else keeps the old answer. vLLM and SGLang both return the
    replacement character's bytes here.
    """
    if "\ufffd" not in text:
        return list(text.encode())
    piece = tok.id_to_token(tid) if hasattr(tok, "id_to_token") else None
    if isinstance(piece, str):
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            try:
                return [int(piece[3:5], 16)]
            except ValueError:
                pass
        elif piece and all(c in _BYTE_OF_CHAR for c in piece):
            return [_BYTE_OF_CHAR[c] for c in piece]
    return list(text.encode())


def token_spans(tok, ids, start: int = 0) -> "tuple[list[str], list[int]]":
    """What each token added to the text, and where that lands in it.

    `text_offset` indexes into the answer, so a token's entry has to be what that token added
    -- not what it says decoded on its own. The two differ wherever a character is spread over
    tokens: each half renders U+FFFD, one character wide, and the offsets walk off the text.
    Measured on this checkpoint, a Korean answer drifted 44 characters and English none
    (45차 §38). So grow the text a token at a time, the way the door streams it, and take the
    growth: the halves add nothing and the token that finishes the character adds it whole.
    """
    stream, said, pos = _Stream(tok), "", start
    tokens, offsets = [], []
    for tid in ids:
        stream.extend((tid,))
        grown = stream.decoded(False)
        added, said = grown[len(said):], grown
        tokens.append(added)
        offsets.append(pos)
        pos += len(added)
    if tokens:
        tokens[-1] += stream.decoded(True)[len(said):]      # whatever never became a character
    return tokens, offsets


def _whole(text: str) -> str:
    """`text` without the trailing bytes that are not a character yet.

    A code point split across tokens renders as U+FFFD at the very end, one per byte still
    missing, so what comes before is whole. A U+FFFD the model really wrote is trimmed here
    too and arrives with whatever follows it, which costs nothing.
    """
    i = len(text)
    while i and text[i - 1] == "\ufffd":
        i -= 1
    return text[:i]


class _Stream:
    """One channel's text, extended as its tokens arrive.

    Neither obvious way works. Decoding one token alone is wrong -- a token's rendering
    depends on its neighbours, because a character can be spread over several byte pieces
    and because a piece carries its leading space only when something precedes it. Decoding
    the whole answer again for every token is right but quadratic: measured on this
    checkpoint's tokenizer, 3.06 s of CPU for a 4,096-token answer against 0.011 s for a
    two-token window whose already-accounted part is subtracted, which is what vLLM's prefix
    and read offsets do.

    `tokenizers` has that window in Rust (`DecodeStream`), and a checkpoint's own tokenizer
    is the Rust one, so that is the path it takes: one call a step -- the step's whole batch
    of tokens at once -- instead of two decodes and the Python arithmetic around them.
    Anything else (every fake in the tests) keeps the window below. Same algorithm, same
    text, and `tests/test_engine_serve.py` pins the two against each other.

    A step whose text ends mid-character does not hold that whole step back. `text` is what
    is settled, `_ahead` is the whole characters in front of it that are shown but not
    settled, and `decoded` is the two together -- which only grows, so nothing is said twice.
    It matters where a character is not one byte: at six accepted tokens a step, holding the
    step back showed the client nothing on 37.2% of the steps of Korean and 43.5% of CJK,
    against 0.0% of English, which is why it went unseen (45차 §32). SGLang has this and vLLM
    does not, though SGLang cuts at the last space (HF's TextStreamer heuristic, whose own
    comment says Hangul is not covered) where this cuts at the character.
    """

    __slots__ = ("tok", "ids", "text", "_ahead", "_repairs", "_rust", "_stream", "_fed",
                 "_holding", "_prefix", "_read")

    def __init__(self, tok, repairs=None):
        self.tok = tok
        self._repairs = DETOK_REPAIRS if repairs is None else repairs
        self.ids = []                       # this channel's tokens, in order
        self.text = ""                      # what they say, settled
        self._ahead = ""                    # whole characters shown in front of it, not settled
        self._stream = decode_stream(tok)   # None where the window below is the path
        self._rust = self._stream is not None   # decided once: the two keep different bookkeeping,
                                                # and changing path mid-answer would say it all twice
        self._fed = 0                       # ids already handed to the Rust stream
        self._holding = 0                   # trailing ids it took and has not turned into text
        self._prefix = self._read = 0       # the Python window, as offsets into `ids`

    def extend(self, ids) -> None:
        self.ids.extend(ids)

    def decoded(self, final: bool) -> str:
        """This channel's text, extended by whatever the newest tokens added."""
        if not self._rust:
            self._window(final)
            return self.text + self._ahead
        if self._fed < len(self.ids):
            pending = self.ids[self._fed:]
            self._fed = len(self.ids)
            grown, self._holding = self._step(pending)
            if grown:
                self.text += grown          # which begins with whatever `_ahead` was showing
                self._ahead = ""
        # A held-back tail is a character waiting for its rest. A tail past the bound is not
        # waiting -- it is a run of lone bytes nothing will complete -- and holding it shows the
        # client nothing while the decode that repeats grows with the run. Both end the same
        # way: say what the tail says, replacement characters and all.
        if self._holding:
            shown, settle = self._in_place(len(self.ids) - self._holding)
            if final or self._holding > _STALL_TOKENS:
                if not final:
                    self._repairs["stalled"] += 1
                self.text += settle                      # which begins with `shown`, by construction
                self._ahead = ""
                self._holding = 0
                self._stream = decode_stream(self.tok)   # its prefix names text already shown
            else:
                self._ahead = shown
        return self.text + self._ahead

    def _in_place(self, cut: int) -> "tuple[str, str]":
        """What `self.ids[cut:]` says where it sits: (the whole characters of it, all of it).

        The stream says "not yet" about its whole tail, but a step can end mid-character and
        have finished several characters before that. So decode the tail where it sits -- a few
        settled tokens in front of it, subtracted off again, because a piece carries its leading
        space only when something precedes it.

        Both halves come from the same decode, and that is the invariant every caller needs:
        what is settled always begins with what was shown, so nothing shown is taken back. A
        decoder that rewrites its settled part when an incomplete byte follows it (byte fallback
        turns the whole run into U+FFFD) fails the prefix test; then nothing is shown early and
        the tail is settled on its own, which is what this did before there was anything early.
        """
        context = self.ids[max(0, cut - _CONTEXT_TOKENS):cut]
        before = _decode(self.tok, context, self._repairs) if context else ""
        grown = _decode(self.tok, context + self.ids[cut:], self._repairs)
        if not grown.startswith(before):
            return "", _decode(self.tok, self.ids[cut:], self._repairs)
        settle = grown[len(before):]
        return _whole(settle), settle

    def _step(self, ids) -> "tuple[str, int]":
        """One `step`, and the two ways it is known to fail where people are watching.

        Both are vLLM's, hit in production there and repaired there (vllm-project/vllm#21951
        and #17448). vLLM steps one token at a time and so repairs one token at a time; carrying
        the step's whole batch would normally make a repair coarser, and does not here, because
        the argument conversion refuses the batch before the stream is touched.

        Returns the text produced and how many trailing ids the stream is then holding back.
        """
        try:
            text = self._stream.step(self.tok, ids) or ""
            return text, 0 if text else self._holding + len(ids)
        except (OverflowError, TypeError):
            # Not a token id at all: out of the range the Rust side takes, or not an integer.
            # The argument conversion fails before any of the batch is taken, so the rest of
            # the step is still good -- replay it one at a time and lose only the bad one.
            self._repairs["invalid_token_id"] += 1
            text, holding = "", self._holding
            for tid in ids:
                try:
                    piece = self._stream.step(self.tok, tid) or ""
                except (OverflowError, TypeError):
                    continue
                holding = 0 if piece else holding + 1
                text += piece
            return text, holding
        except Exception as exc:                     # noqa: BLE001 -- tokenizers raises it untyped
            if not str(exc).startswith(_INVALID_PREFIX):
                raise
            # The decoder rewrote text it had already produced, so the stream's own prefix no
            # longer names what it holds and every later step raises the same way. The state is
            # gone; the text is not, and neither are the ids: say what everything the old stream
            # had not accounted for says, and prime a new stream with exactly those so it carries
            # on with the right prefix. vLLM drops that token's text here; we do not.
            self._repairs["invalid_prefix"] += 1
            cut = len(self.ids) - self._holding - len(ids)
            _, settle = self._in_place(cut)
            self._ahead = ""                             # `settle` accounts for it, empty or not
            self._stream = decode_stream(self.tok, ids=self.ids[cut:])
            return settle, 0

    def _window(self, final: bool) -> None:
        """The same window in Python, for a tokenizer that is not the Rust one."""
        ids = self.ids
        if self._read >= len(ids):
            return
        before = _decode(self.tok, ids[self._prefix:self._read], self._repairs) if self._read > self._prefix else ""
        grown = _decode(self.tok, ids[self._prefix:], self._repairs)
        new = grown[len(before):]
        if not new:
            return
        # a trailing replacement character is a code point waiting for its rest: show what is
        # whole and wait, unless nothing more is coming or the wait has stopped being one
        stalled = len(ids) - self._read > _STALL_TOKENS
        if final or stalled or not new.endswith("\ufffd"):
            if stalled and not final and new.endswith("\ufffd"):
                self._repairs["stalled"] += 1
            self.text += new
            self._ahead = ""
            self._prefix, self._read = self._read, len(ids)
        else:
            self._ahead = _whole(new) if grown.startswith(before) else ""


class _Choice:
    """One generation inside an OpenAI response: its request, its token queue and the per-channel text it has
    shown so far. Chat answers split at the reasoning-end token into reasoning_content / content; a content that
    reaches a stop string ends there; complete <tool_call> blocks become tool_calls (streamed as they complete)."""

    def __init__(self, index: int, request: int, event, q, *, tok, stop, reasoning: bool, tool_parser=None,
                 want_logprobs: "int | None" = None, min_new: int = 0, repairs=None, tool_stream=None):
        self.index, self.request, self.event, self.q = index, request, event, q
        self.tok, self.stop, self.reasoning, self.tool_parser = tok, list(stop), reasoning, tool_parser
        self.tool_stream = tool_stream               # text -> [(name, arguments so far, closed)] or None
        self.want_logprobs = want_logprobs
        self.min_new = min_new
        self._stop_from = 0          # a stop string may not START below the floor: min_tokens means at least
                                     # that many, and a stop the model happens to write early cannot undo it
        self._scanned = 0            # how much of the content channel the stop scan has already read
        self._stop_span = max((len(s) for s in self.stop), default=1) - 1   # how far back a new one can reach
        self.streams = {"reasoning_content": _Stream(tok, repairs), "content": _Stream(tok, repairs)}
        self.shown = {"reasoning_content": 0, "content": 0}
        self.text = {"reasoning_content": "", "content": ""}
        self.logprobs = []                           # per generated token: (id, logprob, [(id, logprob), ...])
        self.total = 0
        self.finish = None
        self.done = False
        self.error = None
        self.tool_calls = []
        self._tool_seen = 0
        self._tool_done = []                         # per call: has `</tool_call>` arrived

    def feed(self, tokens, logprobs, reasoning_end) -> None:
        """The step's new tokens, into the channel they belong to -- as one batch, not one at a
        time, because a batch is one call into the decode stream instead of one per token."""
        if not self.reasoning and logprobs is None:      # an answer past its reasoning, which is most steps
            self.total += len(tokens)
            self.streams["content"].extend(tokens)
            return
        channel = "reasoning_content" if self.reasoning else "content"
        batch = []
        for i, t in enumerate(tokens):
            self.total += 1
            if logprobs is not None and i < len(logprobs):
                self.logprobs.append(logprobs[i])
            if self.reasoning and t == reasoning_end:
                self.streams[channel].extend(batch)          # the split lands inside this step
                batch, channel, self.reasoning = [], "content", False
                continue
            batch.append(t)
        self.streams[channel].extend(batch)

    def flush(self, final: bool = False) -> "list[dict]":
        """Decode each channel; what is new becomes a delta. Returns the deltas in order."""
        deltas = []
        for channel, stream in self.streams.items():
            decoded = stream.decoded(final)
            if channel == "content":
                if self.stop:
                    if self.total <= self.min_new:
                        self._stop_from = len(decoded)
                    # Scan the tail the last flush could not have seen whole, not the answer: a stop
                    # that lies entirely below `_scanned` was already looked for, and the floor only
                    # ever rises. Searching from 0 every step made the cost of a step grow with the
                    # answer, which is the shape you cannot fix later with a faster decode.
                    floor = max(self._stop_from, self._scanned - self._stop_span)
                    self._scanned = len(decoded)
                    hits = [i for i in (decoded.find(x, floor) for x in self.stop) if i >= 0]
                    cut = min(hits) if hits else -1
                    if cut >= 0:
                        decoded = decoded[:cut]
                        self.finish = "stop"
                    elif not final and self.total > self.min_new:    # a tail that could still become one waits
                        held_back = partial_suffix(decoded, self.stop)
                        decoded = decoded[:len(decoded) - held_back] if held_back else decoded
                if self.tool_parser is not None:
                    if not final:                                   # a partial "<tool_call>" prefix waits for the rest
                        held_back = partial_suffix(decoded, ("<tool_call>",))
                        decoded = decoded[:len(decoded) - held_back] if held_back else decoded
                    start = decoded.find("<tool_call>")
                    if start >= 0:
                        deltas.extend(self._tool_deltas(decoded))
                        decoded = decoded[:start]
            delta = decoded[self.shown[channel]:]
            if delta:
                deltas.append({channel: delta})
                self.shown[channel] = len(decoded)
            self.text[channel] = decoded[:self.shown[channel]]
        return deltas

    def tool_calls_done(self) -> "list[dict]":
        """The calls that closed. One the answer was cut off inside is not a call the caller can
        make, so it is not in the body and it does not name the finish -- its fragments went out,
        because they had already been read, and the reason stays `length`."""
        return [call for call, done in zip(self.tool_calls, self._tool_done) if done]

    def _tool_deltas(self, decoded: str) -> "list[dict]":
        """OpenAI tool-call deltas for whatever of the calls has arrived.

        The first delta of a call carries its id, type and name with empty arguments; every one
        after carries the next fragment of `arguments`. That is the shape the OpenAI streaming
        API defines, and the reason it exists is this format: the arguments are most of a call,
        so waiting for `</tool_call>` was waiting for nearly the whole answer -- six seconds on a
        thousand Korean characters (45차 §44). vLLM, SGLang and llama.cpp all stream them.

        `partial` returns each call's arguments cut off at the last thing the model has actually
        written, and that text only grows, so a fragment is what is new since the last flush.
        Without a partial parser this falls back to what it always did: the whole call, once.
        """
        out = []
        if self.tool_stream is None:
            blocks = _TOOL_CALL.findall(decoded)
            for body in blocks[self._tool_seen:]:
                for name, args in (self.tool_parser(f"<tool_call>{body}</tool_call>") or []):
                    call = {"index": len(self.tool_calls), "id": f"call_{self.request}_{len(self.tool_calls)}",
                            "type": "function", "function": {"name": name, "arguments": args}}
                    self.tool_calls.append(call)
                    self._tool_done.append(True)
                    out.append({"tool_calls": [call]})
            self._tool_seen = len(blocks)
            return out
        for i, (name, args, done) in enumerate(self.tool_stream(decoded)):
            if i == len(self.tool_calls):
                self.tool_calls.append({"index": i, "id": f"call_{self.request}_{i}", "type": "function",
                                        "function": {"name": name, "arguments": ""}})
                self._tool_done.append(False)
                out.append({"tool_calls": [{"index": i, "id": self.tool_calls[i]["id"], "type": "function",
                                            "function": {"name": name, "arguments": ""}}]})
            sent = len(self.tool_calls[i]["function"]["arguments"])
            if len(args) > sent:
                out.append({"tool_calls": [{"index": i, "function": {"arguments": args[sent:]}}]})
                self.tool_calls[i]["function"]["arguments"] = args
            self._tool_done[i] = done
        return out

    def finish_reason(self) -> str:
        if any(self._tool_done):
            return "tool_calls"
        return self.finish or "length"

    def logprobs_payload(self, offset: int = 0) -> "dict | None":
        """OpenAI chat `logprobs.content`: the shown tokens' log-probabilities (content channel, chosen + top)."""
        if self.want_logprobs is None:
            return None
        rows = []
        seen = {}

        def text_of(tid):
            """One decode per distinct id. The payload asks for each token's text and its bytes,
            and the top-k of one position repeat across the next, so this was two calls per entry
            over the whole answer at once -- a stall in front of the last chunk."""
            if tid not in seen:
                seen[tid] = self.tok.decode([tid])
            return seen[tid]

        bytes_seen = {}

        def bytes_of(tid):
            if tid not in bytes_seen:
                bytes_seen[tid] = token_bytes(self.tok, tid, text_of(tid))
            return bytes_seen[tid]

        for tid, lp, top in self.logprobs[offset:]:
            rows.append({"token": text_of(tid), "logprob": lp, "bytes": bytes_of(tid),
                         "top_logprobs": [{"token": text_of(i), "logprob": v, "bytes": bytes_of(i)}
                                          for i, v in top[: self.want_logprobs]]})
        return {"content": rows}



class _Histogram:
    """A Prometheus histogram over fixed bounds: cumulative buckets, sum, count.

    Observations come from the step loop and the exposition from the HTTP thread.
    Under the GIL each increment and each read is atomic, so a scrape can land between
    two of them and see a bucket and the sum one observation apart -- the ordinary
    error of any scrape, not a corrupt series. `rows` keeps the buckets monotone even
    then, which is the one property a histogram may not break.
    """

    __slots__ = ("bounds", "counts", "total", "sum")

    def __init__(self, bounds):
        self.bounds = tuple(bounds)
        self.counts = [0] * len(self.bounds)
        self.total = 0
        self.sum = 0.0

    def observe(self, seconds: float) -> None:
        if not seconds >= 0:                                     # never a negative or NaN sample
            return
        for i, bound in enumerate(self.bounds):
            if seconds <= bound:
                self.counts[i] += 1
        self.total += 1
        self.sum += seconds

    def rows(self, name: str):
        running = 0
        for bound, count in zip(self.bounds, self.counts):
            running = max(running, count)
            yield f'{name}_bucket{{engine="st",le="{bound}"}}', running
        yield f'{name}_bucket{{engine="st",le="+Inf"}}', max(running, self.total)
        yield f'{name}_sum{{engine="st"}}', round(self.sum, 6)
        yield f'{name}_count{{engine="st"}}', self.total


# vLLM's own bucket bounds, so a scraper or dashboard built for the engine this one
# replaces keeps reading the same series. They also resolve this engine's measured
# range: a 214.7 ms prefill step lands among the TTFT bounds around 0.25, and a 46 ms
# inter-token interval between 0.025 and 0.05.
_TTFT_BOUNDS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0)
_ITL_BOUNDS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5)
_E2E_BOUNDS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0,
               50.0, 60.0, 120.0, 240.0, 480.0)
# Step wall time, host-observed. Both kinds end on a readback (the sampled ids), so this
# is the whole step, not a launch time. The bounds straddle what this engine measured
# offline -- a 46 ms decode step and a 214.7 ms prefill chunk -- finely enough that a
# regression moves a bucket, which is what a scrape can see and a CUPTI trace cannot.
_STEP_BOUNDS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25,
                0.3, 0.4, 0.5, 0.75, 1.0, 2.0)


class Server:
    @staticmethod
    def _agree_on_parked(comm, parked) -> None:
        """Every rank's tier must hold the same conversations before the first request (D3).

        The tier is per rank and per node, so one node's leftovers are invisible to the others:
        that rank starts numbering conversations after them, hands the same turn a different row,
        and the collectives then mix two different states -- the answer is garbage and nothing
        raises until a retire hits a key that rank already parked. 45th 21: srv4 still carried a
        local run's seq-0/seq-1, rank 3 died with "conversation 0 is already parked" and the other
        three spun at 96% GPU in the next all-reduce. Disagreement kills the boot on every rank.
        """
        if int(getattr(comm, "world_size", 1)) <= 1:
            return
        import torch
        checksum = 0
        for key in parked:
            checksum = (checksum * 1000003 + int(key) + 1) % (1 << 40)
        device = "cuda" if torch.cuda.is_available() else "cpu"      # the same channel _votes uses
        mine = torch.tensor([len(parked), int(parked[-1]) if parked else -1, checksum],
                            dtype=torch.int64, device=device)
        highest = comm.all_reduce_max(mine.clone())
        disagree = torch.tensor([0 if bool(torch.equal(highest, mine)) else 1], dtype=torch.int64, device=device)
        if int(comm.all_reduce_max(disagree).item()):
            raise RuntimeError(
                f"the ranks' NVMe tiers hold different conversations: this rank has {len(parked)} "
                f"{parked[:8]}{'...' if len(parked) > 8 else ''}, the fleet's highest is "
                f"{[int(x) for x in highest.tolist()]} (count, last key, checksum). Clear "
                f"glm53-logs/st-tier on every node, or fan the same tier out -- a boot cannot start "
                f"with the ranks numbering conversations differently.")

    def __init__(self, engine, runner, comm, port: int = 8000, tokenizer=None,
                 host: str = "0.0.0.0", max_pending: int = 64, chat=None, model_name: str = "st",
                 reasoning_end: "int | None" = None, request_timeout_s: float = 3600.0, tool_parser=None,
                 generation: "dict | None" = None, max_choices: int = 4, vision=None, tool_stream=None,
                 tool_grammar=None, tool_call_start: "int | None" = None,
                 lease: "dict | None" = None, latency_root=None):
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if type(request_timeout_s) not in (int, float) or not request_timeout_s > 0:
            raise ValueError("request_timeout_s must be positive")
        if runner.slot_of:
            raise ValueError("the server requires an idle runner")
        if reasoning_end is not None and (type(reasoning_end) is not int or reasoning_end < 0):
            raise ValueError("reasoning_end must be a token id")
        self.engine, self.runner, self.comm = engine, runner, comm
        from engine.base.latency import Recorder
        from pathlib import Path
        self.latency = Recorder(comm.rank, latency_root or Path.home() / 'glm53-logs' / 'onepass-latency')
        runner.latency = self.latency
        self.latency_replies = {}
        self.latency_boot_id = uuid.uuid4().hex
        self.port, self.host, self.tok = port, host, tokenizer
        # D3 is about kernels, but its rule holds here too: a path that is taken silently is a
        # path nobody checks. /metrics says which detokenizer served, so a scrape settles it.
        self.rust_detok = tokenizer is not None and decode_stream(tokenizer) is not None
        self._prompt_tokens = None                 # built from `tok` on first use (see the property)
        self.detok_repairs = new_repairs()         # this door's, so a scrape names who repaired
        self.chat, self.model_name, self.reasoning_end = chat, model_name, reasoning_end
        self.tool_parser = tool_parser             # text -> [(name, arguments json)] or None (the profile knows the model's format)
        self.tool_stream = tool_stream             # the same format, read while it is still arriving (streamed deltas)
        self.tool_grammar = tool_grammar           # tools -> an EBNF grammar for calls of them, or None
        self.tool_call_start = tool_call_start     # the token that opens a call: where that grammar arms
        self.vision = vision                       # the profile's door half for pictures (prepare / expand / limits), or None: text only
        self.generation = dict(generation or {})   # the checkpoint's generation_config defaults (temperature ...) a request may omit
        self.max_choices = int(max_choices)        # n / best_of ceiling: one row each, never more than the decode width
        self._stop_ids = {}                        # request id -> stop_token_ids: an end by one of them is finish_reason "stop"
        self.max_pending = max_pending
        self.max_context = int(getattr(engine, "max_context", 2**31 - 1))   # the model's trained positions; the door refuses beyond
        self.request_timeout_s = float(request_timeout_s)
        self.clock = time.monotonic                # injectable for tests
        self.booted = int(time.time())             # wall clock, for the model card's `created` (OpenAI's field)
        self._cancels = set()                      # (request id, reason) asked by HTTP threads / the timeout scan; rank 0 broadcasts them
        self._deadline = {}                        # request id -> clock() by which it must have finished (rank 0)
        self.cancelled = 0
        self.timed_out = 0                         # the subset of `cancelled` the deadline scan took
        # The fleet lease (engine/base/fleet_lease): `{'owner':..., 'path':...}` on rank 0, or
        # None when nothing reserved this fleet. Two things ride on it -- what this engine is
        # doing, published for whoever is waiting, and the request to hand the fleet over.
        self.lease = dict(lease) if lease else None
        self.draining = None                       # the requester we are handing the fleet to
        self.drained = False                       # every conversation parked; the loop may end
        self.handed_over = None                    # {'to':..., 'parked': n, 'lost': n} once it happens
        self._lease_seen = 0.0                     # last poll, so a step is not a file stat
        self.lease_poll_s = 2.0
        self.steps_prefill = self.steps_decode = 0   # D9: a step is one kind or the other, never both
        # Latency is owed from the request's arrival, not from the step that served it.
        # Rank 0 admits, answers and serves /metrics, so only rank 0 keeps these.
        self._arrived = {}                         # request id -> clock() at admission
        self._token_at = {}                        # row -> clock() of its last delivered token
        self.ttft = _Histogram(_TTFT_BOUNDS)
        self.step_seconds = {"prefill": _Histogram(_STEP_BOUNDS), "decode": _Histogram(_STEP_BOUNDS)}
        self._step_began = None                    # clock() at the top of the step being timed
        self.itl = _Histogram(_ITL_BOUNDS)         # per output token: a step's seconds divided by what it produced
        self.step_gap = _Histogram(_ITL_BOUNDS)    # per step: how bursty the stream is. Under speculation the two
                                                   # differ by exactly the mean acceptance length, and one alone
                                                   # cannot tell "steps got slower" from "tokens got cheaper"
        self.queued = _Histogram(_E2E_BOUNDS)      # arrival to the first step that carried this request
        self.inference = _Histogram(_E2E_BOUNDS)   # that step to the last token: the half queueing does not explain
        self.by_reason = {}
        self.e2e = _Histogram(_E2E_BOUNDS)
        self._streams = {}                         # request id -> queue of ("tokens", ids) | ("end", finish) | ("error", text)
        self._wake = threading.Event()             # set whenever a stream queue gains an item, so a drain can wait
                                                   # on it instead of polling; a 20 ms poll put 20 ms of jitter on
                                                   # every streamed token, a fifth of this engine's inter-token time
        self._sent = {}                            # row -> generated tokens already handed to its stream
        self.prompt_tokens_total = self.generation_tokens_total = 0
        # Tokens are what the engine spends; characters are what the reader gets, and on this
        # checkpoint one token is 5.8 characters of English and 1.3 of Korean. A deployment with
        # only the token counter reads its own throughput 4.4x too kindly (45차 §40).
        self.generation_characters_total = 0
        self.arrivals = queue.Queue()
        self.pending, self.results = {}, {}
        # conversation ids are request ids; parked conversations from an earlier boot keep theirs
        parked = sorted(runner.parked_keys())
        self._agree_on_parked(comm, parked)
        if getattr(runner, "load_prefix_tier", None) is not None:
            runner.load_prefix_tier()                                    # boundaries an earlier boot left on the prefix tier
            self._agree_on_parked(comm, runner.prefix_tier_keys())        # ... which every rank must hold alike (45차 §23 A)
        self.next_seq, self.served = 1 + max(parked, default=-1), 0
        self.alive = True
        self._lock = threading.Lock()
        self._waiting = deque()                    # request id, tokens, limit, temperature, promised blocks
        self._active = {}                          # reusable row -> (request id, promised blocks)
        self._admitted = {}                        # request id -> clock() when a row began stepping it
        self._cached = {}                          # request id -> prompt tokens the row already held when it was admitted:
        # a cached boundary, a boundary read back from the tier, or a retained conversation's history. An agent resends its
        # whole transcript every step, so this is the number that says whether the way it builds a prompt is working, and
        # `usage.prompt_tokens_details.cached_tokens` is where every OpenAI client already looks for it.
        self._conversations, self._conversation_of = {}, {}   # resident (idle or live) conversations <-> rows
        self._tenant_of = {}                                  # conversation -> its tenant salt: only that tenant continues it.
        # Parked conversations outlive the process; after a restart theirs is unknown, and an unknown tenant matches
        # nobody but the unsalted caller -- a miss and a fresh prompt (whose prefix chain still hits), never a leak.
        self._idle_order = {}                      # resident idle rows, least recently completed turn first (no tier)
        self._retiring = {}                        # row -> conversation: its park is on the tier's thread (D10: no step waits on it)
        self._resuming = {}                        # row -> (conversation, request, ids, limit, temperature, promised): its resume is in flight
        self._restoring = {}                       # row -> the request whose prefix is being read back from the prefix tier (45차 §23 A)
        self._deferred = set()                     # requests waiting for a running prefill to cache the prefix they share (B)
        self.controls = queue.Queue()              # rank 0's cache controls (pin / unpin / reset), broadcast with the arrivals (C)
        # A decode step runs inside a captured graph, so its stages cannot be timed with CUDA events from
        # python -- `Event.record` is not capturable, and a mark placed inside `forward` would time the
        # CAPTURE. What does see through a replay is CUPTI: torch's profiler reports the kernels a replay
        # runs. `/v1/engine/profile` turns it on for a few decode steps and hands back what they were made of.
        self._profiling = None                     # {"left": n, "prof": profiler} while a run is in flight
        self.profile_table = None                  # the last run's kernels, biggest device time first
        self.prefix_resets = 0                     # how many times an operator threw the prefix cache away
        # Which way a prompt found its KV. D16's conversation tier can only serve a prompt that
        # EXTENDS a retained history exactly; the prefix cache serves one that merely shares
        # whole blocks. A client whose prompt diverges near its end -- which is what Deneb's
        # wire-only tail injection produces on purpose, to keep the byte prefix stable -- takes
        # the second path always and the first never. This census is how that stops being an
        # argument (45차 §68).
        self.reuse_paths = {"continuation": 0, "prefix_or_cold": 0}
        self.reasoning_shapes = {}                  # (thinking, effort) -> chat requests: see note_reasoning
        self._free_rows = list(range(min(runner.kv.max_seqs, runner.c.max_running, runner.slots.available)))
        if not self._free_rows:
            raise ValueError("the server needs at least one request row and state slot")

    def room_for(self, prompt_tokens: int) -> int:
        """The largest `max_new` this prompt could still be given, both limits together.

        A request reserves its whole horizon at admission and is never preempted (D3), so a
        default that does not fit is a refusal the caller did not ask for. Only defaults are
        clamped by this -- a number the caller wrote is still answered with a refusal.
        """
        kv = self.runner.kv
        draft = self.runner.c.draft_slots
        by_blocks = min(kv.num_blocks, kv.max_blocks_per_seq) * kv.block_size
        return max(1, min(self.max_context, by_blocks) - prompt_tokens + 1 - draft)

    def submit(self, ids, max_new: int, temperature: float, conversation: "int | None" = None, stream: bool = False,
               min_new: int = 0, options: "dict | None" = None, continue_history: bool = False, media=None,
               cache_salt: "str | None" = None):
        """Validate and enqueue on rank 0 without acquiring any model resources.
        `stream`: the request also gets a token queue (see `_streams`). `min_new`: no end token before this many.
        `options`: the request's sampling/behaviour options beyond temperature (the engine validates them).
        `continue_history`: the OpenAI path re-sends a whole chat every turn -- when a retained conversation's history
        (prompt + what it generated) is a proper prefix of `ids`, continue it with the new suffix instead of
        prefilling everything again (45차 §23 B1). The hint is taken here, on rank 0; admission re-checks it and
        falls back to a fresh prompt if the conversation left in between.
        `media`: the pictures standing at placeholder runs inside `ids` (the profile's door built them: kind, digest,
        positions, canvas, grid); they ride to every rank with the request and are encoded there (45차 §23 A7).
        `cache_salt`: vLLM's field of the same name -- a tenant's own string, folded into the first block of the
        boundary chain, so one tenant's prompts can never adopt (or be told about) another's cached prefix. It is
        not a secret and not authentication: it separates namespaces, which is all a prefix cache can promise."""
        if self.comm.rank != 0:
            raise RequestError("requests must enter on rank 0")
        if self.draining is not None:
            # Handing the fleet over: what is here finishes and is parked, nothing new joins.
            # Refused before any state exists -- a rejected request that left a pending entry
            # behind would keep the engine from ever going quiet, and so from ever letting go.
            raise RequestError("the engine is handing the fleet over; retry shortly", 503)
        options = dict(options or {})
        if options and hasattr(self.engine, "validate_options"):
            try:
                self.engine.validate_options(options)
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
        if options and hasattr(self.engine, "prepare_options"):
            try:
                self.engine.prepare_options(options)     # a grammar is built here, where failing is an answer
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
        if cache_salt is not None and (not isinstance(cache_salt, str) or not 0 < len(cache_salt) <= 256):
            raise RequestError("cache_salt / prompt_cache_key must be a string of 1 to 256 characters")
        salt = prefix_cache.tenant_salt(cache_salt) if cache_salt else None
        media = list(media or [])
        for m in media:
            if (not isinstance(m, dict) or not isinstance(m.get("kind"), str) or not isinstance(m.get("digest"), str)
                    or not isinstance(m.get("positions"), list) or not m["positions"]
                    or any(type(p) is not int or not 0 <= p < len(ids) for p in m["positions"])):
                raise RequestError("media records need a kind, a digest and placeholder positions inside the prompt")
        if media and not hasattr(self.engine, "media_marks"):
            raise RequestError("images are not served by this engine")
        hint = None
        if continue_history and conversation is None and self.runner.keep_idle:
            hint = self._continuation(ids, media, salt)
        self.reuse_paths["continuation" if (hint is not None or conversation is not None) else "prefix_or_cold"] += 1
        if conversation is not None:
            if type(conversation) is not int or conversation < 0:
                raise RequestError("conversation must be a nonnegative integer")
            if not self.runner.keep_idle:
                raise RequestError("this server does not retain conversations", 409)
            if self._tenant_of.get(conversation) != salt:
                # Conversation ids are small integers, so naming one is not proof of anything: it has to be the
                # tenant that started it. The same answer as an unknown one, which is also what it is to this caller.
                raise RequestError("conversation is unknown, live or evicted", 409)
        if not isinstance(ids, (list, tuple)) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise RequestError("ids must be a nonempty list of nonnegative token integers")
        if type(max_new) is not int or max_new <= 0:
            raise RequestError("max_tokens must be a positive integer")
        if type(min_new) is not int or not 0 <= min_new <= max_new:
            raise RequestError("min_tokens must be an integer between 0 and max_tokens")
        if type(temperature) not in (int, float):
            raise RequestError("temperature must be finite and nonnegative")
        try:
            temperature = float(temperature)
        except OverflowError as exc:
            raise RequestError("temperature must be finite and nonnegative") from exc
        if not math.isfinite(temperature) or temperature < 0:
            raise RequestError("temperature must be finite and nonnegative")
        try:
            self.engine.validate(ids, max_new, temperature)
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        horizon = len(ids) + max_new - 1 + (self.runner.c.draft_slots if max_new > 1 else 0)
        blocks = self.runner.kv.blocks_for(horizon)
        # The numbers, not just the verdict. A request reserves its whole horizon, so a caller
        # who is refused needs to know which half to cut -- and the caller who meets this first
        # is writing in a language that costs more tokens a character (45차 §40).
        room = min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq) * self.runner.kv.block_size
        needs = f"needs {horizon} ({len(ids)} for the prompt, {max_new} to generate)"
        if horizon >= 2**31 or blocks > min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq):
            raise RequestError(f"the KV pool holds {room} tokens for one request; this one {needs}")
        if horizon > self.max_context:
            raise RequestError(f"this model serves {self.max_context} tokens of context; this request {needs}")
        with self._lock:
            if not self.alive:
                raise RequestError("engine is stopping", 503)
            if len(self.pending) + len(self.results) >= self.max_pending:
                raise RequestError("request queue is full", 503)
            request = self.next_seq
            self.next_seq += 1
            event = threading.Event()
            self.pending[request] = event
            if stream:
                self._streams[request] = queue.Queue()
            now = self.clock()
            self._deadline[request] = now + self.request_timeout_s
            self._arrived[request] = now                       # latency is owed from here, not from the step that serves it
            self.prompt_tokens_total += len(ids)
            if options.get("stop_token_ids"):
                self._stop_ids[request] = set(options["stop_token_ids"])
            tier = chain = None
            prefix = getattr(self.runner, "prefix", None)
            if conversation is None and prefix is not None:
                # the prompt's boundary chain, hashed once here and carried to every rank with the request: admission's
                # lookups, the dedup lookahead, the tier candidate and the runner's submit all read it, none recompute it
                salts = ([salt] if salt else []) + [(m["positions"][0], bytes.fromhex(m["digest"])) for m in media]
                chain = prefix.chain(ids, salts)
                if hint is None and getattr(self.runner, "prefix_tier", None) is not None:
                    found = prefix.tier_lookup_chain(chain, len(ids), prefix.peek_chain(chain, len(ids)))   # rank 0's view; every rank votes
                    if found is not None:
                        tier = (found[0], found[1].hex())
            if conversation is None and self.runner.keep_idle:
                if len(self._tenant_of) > 4 * self.max_pending:
                    live = set(self._conversations) | set(self.runner.parked_keys())
                    self._tenant_of = {k: v for k, v in self._tenant_of.items() if k in live}
                self._tenant_of[request] = salt                # the conversation this turn may become belongs to that tenant
            self.arrivals.put((request, list(ids), max_new, float(temperature), blocks, conversation, min_new, options, hint,
                               media, tier, chain, salt))
        return request, event

    def note_reasoning(self, kwargs: dict) -> None:
        """Count the reasoning shape this chat request will be RENDERED with, not the one it was sent with.

        45차 §81 measured why a conversation is or is not continuable, and it turns on exactly this: with
        thinking off the template writes `<think></think>` and the next turn re-renders that assistant turn
        identically, so the history stays a prefix and D16 continues it. With thinking on the model writes a
        reasoning span the next render only reproduces if the client echoes `reasoning_content` back -- and
        the layer that turns reasoning on (wormhole's effort.go, from an Ares decision or a caller's
        high/max) is not the layer that decided whether to echo it (Deneb, from its own config, already
        sent). So the same fleet serves both shapes and nothing recorded which.

        `effort` is what the TEMPLATE will read, which is why absent is its own label and not folded into
        max: absent means nobody said, and the template turns that into max on its own.
        """
        thinking = kwargs.get("thinking")
        shape = ("on" if thinking is None or thinking else "off",
                 str(kwargs.get("reasoning_effort", "absent")))
        self.reasoning_shapes[shape] = self.reasoning_shapes.get(shape, 0) + 1

    def _continuation(self, ids, media=(), salt=None) -> "tuple[int, int] | None":
        """(conversation, prefix length) of the retained conversation whose history is the longest proper prefix of
        `ids`: a resident idle row, or a parked one (its record carries the tokens). None if nothing matches.
        The pictures must match too: the same placeholder run with another picture is another prompt, and so does the
        tenant salt: a conversation another tenant left behind is not this one's to continue."""
        best = None
        n = len(ids)
        marks = sorted((m["positions"][0], m["digest"]) for m in media)
        ends = set(getattr(self.engine, "eos", None) or ())
        def consider(key, history, history_marks):
            nonlocal best
            if self._tenant_of.get(key) != salt:
                return                                            # another tenant's turn, or one this boot cannot vouch for
            m = len(history)
            if m <= 1 or m - 1 >= n or (ids[m - 1] != history[m - 1] and ids[m - 2] != history[m - 2]):
                return                                            # the cheap test first: no list compare for the many that cannot match
            history = list(history)
            if (0 < m < n and (best is None or m > best[1]) and ids[:m] == history
                    and [(p, d) for p, d in marks if p < m] == sorted((int(p), str(d)) for p, d in history_marks)):
                best = (key, m, False)
            # the history ended with an end token the template does not render back (<|endoftext|> after an answer,
            # where the next turn renders <|user|>): the caches stand before that token, which was sampled but never fed
            elif (m > 1 and m - 1 < n and history[-1] in ends and (best is None or m - 1 > best[1]) and ids[:m - 1] == history[:-1]
                    and [(p, d) for p, d in marks if p < m - 1] == sorted((int(p), str(d)) for p, d in history_marks)):
                best = (key, m - 1, True)
        view = getattr(self.engine, "history_ref", None) or getattr(self.engine, "history", None)
        for row in list(self._idle_order):
            key = self._conversation_of.get(row)
            if key is not None and view is not None:
                consider(key, view(row),                                  # read, compared, never mutated
                         self.engine.media_marks(row) if hasattr(self.engine, "media_marks") else [])
        for key in self.runner.parked_keys():
            # The digest is three numbers; the token list behind it is 3.8 MiB of Python ints for
            # a 100K-token conversation, and reading one per parked conversation per request kept
            # 1.04 GiB resident at this fleet's 280 (45차 §62). Reject on the digest, read on a hit.
            digest = self.runner.parked_digest(key)
            if digest is None:
                continue
            m = digest["tokens"]
            if m <= 1 or m - 1 >= n or (ids[m - 1] != digest["last"] and ids[m - 2] != digest["prev"]):
                continue
            record = self.runner.parked_record(key)
            if record is not None and "tokens" in record:
                consider(key, record["tokens"], [(r[2], r[1]) for r in record.get("media", [])])
        return best

    @staticmethod
    def _media_after(media, prefix: int):
        """The pictures standing past `prefix` with their positions re-based on it; None if one straddles the cut."""
        out = []
        for m in media:
            pos = m["positions"]
            if pos[-1] < prefix:
                continue
            if pos[0] < prefix:
                return None
            out.append(dict(m, positions=[p - prefix for p in pos]))
        return out

    def cancel(self, request: int, reason: str = "client closed") -> None:
        """Ask the loop to drop `request` wherever it is (waiting, prefilling, decoding); every rank
        applies it in the same iteration. Reasons: "client closed", "timeout", "stop"."""
        with self._lock:
            self._cancels.add((int(request), str(reason)))

    def _expire(self) -> None:
        """Rank 0: requests past their deadline are cancelled as timeouts."""
        now = self.clock()
        for request, deadline in list(self._deadline.items()):
            if now > deadline:
                self._cancels.add((request, "timeout"))

    def _drain_cancels(self):
        with self._lock:
            out = sorted(self._cancels)
            self._cancels.clear()
        return out

    def _cancel(self, request: int, reason: str) -> None:
        """Applied on every rank: the request leaves the FIFO or its row, and rank 0 answers the client."""
        for i, entry in enumerate(self._waiting):
            if entry[0] == request:
                del self._waiting[i]
                self._deferred.discard(request)
                break
        else:
            row = next((row for row, (req, _) in self._active.items() if req == request), None)
            if row is None:
                resuming = next((r for r, e in self._resuming.items() if e["request"] == request), None)
                if resuming is not None and self._resuming[resuming]["cancelled"] is None:
                    self._resuming[resuming]["cancelled"] = reason        # its conversation parks again once the read lands
                    self.cancelled += 1
                    self.timed_out += reason == "timeout"
                    self._deadline.pop(request, None)
                    self._arrived.pop(request, None)
                    self._admitted.pop(request, None)
                    self._cached.pop(request, None)
                    self._answer(request, RequestError(f"request cancelled: {reason}", 504 if reason == "timeout" else 499))
                    return
                restoring = next((r for r, e in self._restoring.items() if e["request"] == request), None)
                if restoring is not None and self._restoring[restoring]["cancelled"] is None:
                    self._restoring[restoring]["cancelled"] = reason      # the read lands, the boundary stays cached, the row goes back
                    self.cancelled += 1
                    self.timed_out += reason == "timeout"
                    self._deadline.pop(request, None)
                    self._arrived.pop(request, None)
                    self._admitted.pop(request, None)
                    self._cached.pop(request, None)
                    self._answer(request, RequestError(f"request cancelled: {reason}", 504 if reason == "timeout" else 499))
                    return
                self._deadline.pop(request, None)
                self._arrived.pop(request, None)
                self._admitted.pop(request, None)
                self._cached.pop(request, None)
                return                                     # finished already (or unknown): nothing to drop
            self._active.pop(row)
            self._sent.pop(row, None)
            self._token_at.pop(row, None)
            try:
                self.runner.cancel(row)
            finally:
                self.engine.forget(row)
                if self.runner.keep_idle:
                    self._conversations.pop(self._conversation_of.pop(row, None), None)
                    self._idle_order.pop(row, None)
                heapq.heappush(self._free_rows, row)
        self.cancelled += 1
        self.timed_out += reason == "timeout"
        self._deadline.pop(request, None)
        self._arrived.pop(request, None)
        self._admitted.pop(request, None)
        self._cached.pop(request, None)
        status = 504 if reason == "timeout" else 499
        self._answer(request, RequestError(f"request cancelled: {reason}", status))

    def take_result(self, request):
        with self._lock:
            result = self.results.pop(request)
        if isinstance(result, RequestError):
            raise result
        return result

    def finish_reason(self, out, request=None) -> str:
        """OpenAI's word for how a generation ended: at one of the model's end tokens (or the request's stop_token_ids),
        or at the limit."""
        ends = set(getattr(self.engine, "eos", ())) | self._stop_ids.get(request, set())
        return "stop" if out and out[-1] in ends else "length"

    def split(self, out):
        """(reasoning ids, content ids): what came before `reasoning_end` and after it (the token itself
        is neither). Without an end token everything generated so far is still reasoning."""
        out = list(out)
        if self.reasoning_end is None:
            return [], out
        if self.reasoning_end in out:
            i = out.index(self.reasoning_end)
            return out[:i], out[i + 1:]
        return out, []

    def _answer(self, request, result):
        if self.comm.rank == 0:
            with self._lock:
                event = self.pending.pop(request, None)
                if event is not None:
                    self.results[request] = result
                    if not isinstance(result, RequestError):
                        self.generation_tokens_total += len(result)
                    event.set()
            stream = self._streams.get(request)
            if stream is not None:
                stream.put(("error", str(result)) if isinstance(result, RequestError) else ("end", self.finish_reason(result, request)))
                self._wake.set()
            self._stop_ids.pop(request, None)

    def _drain(self):
        out = []
        while True:
            try:
                out.append(self.arrivals.get_nowait())
            except queue.Empty:
                return out

    def _drain_controls(self):
        out = []
        while True:
            try:
                out.append(self.controls.get_nowait())
            except queue.Empty:
                return out

    def model_card(self) -> dict:
        """What this engine serves, in the shape a router can read without being told.

        `/v1/models` used to answer three fields, so everything downstream had to be configured
        by hand -- and a hand-written capability drifts. SparkFleet probes exactly this endpoint
        to decide a backend is a routable chat model, and wormhole turns its inventory into
        routes (`gateway-go/cmd/wormhole/fleet.go`), so this is the one place the engine can
        state what it can do and have it arrive.

        `max_model_len` carries vLLM's field name on purpose: anything that already reads a vLLM
        `/v1/models` gets the served ceiling for free. Everything under `capabilities` is read
        off what this boot actually bound -- a vision tower that is present, a tool parser the
        profile supplied, grammars that compiled, a drafter with a k -- never a constant, so it
        cannot say yes to something this process cannot do (45차 §56).
        """
        engine = self.engine
        drafter = getattr(engine, "drafter", None)
        return {
            "id": self.model_name, "object": "model", "owned_by": "st", "root": self.model_name,
            "created": self.booted,
            "max_model_len": int(getattr(engine, "max_context", 0)) or None,
            "capabilities": {
                "vision": self.vision is not None,
                "tools": self.tool_parser is not None,
                "tool_grammar": self.tool_grammar is not None,
                "structured_output": bool(getattr(engine, "grammars", None)),
                "reasoning": self.reasoning_end is not None,
                "streaming": True,
                "speculative_tokens": int(getattr(drafter, "k", 0) or 0),
                "prefix_cache": getattr(self.runner, "prefix", None) is not None,
                "conversation_tier": getattr(self.runner, "tiered", None) is not None,
                "max_concurrent_requests": int(self.runner.c.max_running),
            },
        }

    @property
    def prompt_tokens(self) -> "PromptTokens | None":
        """The splice cache for whatever tokenizer this server actually has.

        A property and not a constructor field because the tokenizer can arrive afterwards --
        a boot binds one, a test swaps one in -- and a cache built against a tokenizer the door
        no longer uses would splice one vocabulary's ids onto another's.
        """
        if self.tok is None:
            return None
        cache = self._prompt_tokens
        if cache is None or cache.tok is not self.tok:
            cache = self._prompt_tokens = PromptTokens(self.tok)
        return cache

    def check_model(self, name) -> str:
        """The model this request is for, or 404 if this engine does not serve it.

        Until now the door took whatever name arrived and **echoed it back** in the response
        while answering with the model it actually has. That is worse than refusing: a router
        entry left pointing here under an old name gets a correct-looking answer labelled as
        something else, and everything downstream that keys on the name -- Deneb's per-model
        usage, its metered gate, its `ProfileFor` sampling rules -- believes the label. The
        fleet has that exact wiring today: wormhole's `deepseek-v4-flash` and `dsv4-nothink`
        entries still point at this head with `upstreamModel: deepseek-v4-flash`, and Deneb's
        own notes record what the same class of stale entry cost the last time nobody noticed
        (1,346 billed calls over twelve days).

        So it answers the way every OpenAI-compatible server does, vLLM included
        (`entrypoints/openai/models/serving.check_model`): an absent or empty name is this
        engine's own, and a name it does not serve is a 404 carrying that sentence verbatim,
        so a client matching on the message keeps working (45차 §73).
        """
        if not isinstance(name, str) or not name:
            return self.model_name
        if name == self.model_name:
            return name
        raise RequestError(f"The model `{name}` does not exist.", 404)

    def fleet_status(self) -> "dict | None":
        """Who holds the fleet, whether it has been asked to let go, and how the handover went.

        The engine knows all three -- it writes them into `~/st-fleet.lock` -- and until now the
        only way to read them was to ssh to rank 0 and cat that file. Everything that watches
        this engine already reaches the door: the supervisor, wormhole's probe, SparkFleet, an
        operator with curl. So the lifecycle belongs on the door too (45차 §60).

        None when this boot holds no lease, so a bare `--local` run's status is unchanged and
        nobody has to special-case a field that means nothing there.
        """
        if not self.lease:
            return None
        return {"owner": self.lease.get("owner"), "path": self.lease.get("path"),
                "draining": self.draining, "handed_over": self.handed_over}

    def catalog(self) -> "tuple[dict, int]":
        """`/v1/models`, and whether this engine is routable right now.

        This is the endpoint the control plane actually asks. Wormhole re-probes it every 60 s
        for `max_model_len` (`router_discovery.probeMaxModelLen`), and SparkFleet sets a
        service's model id from it -- which is what makes a backend routable at all
        (`wormhole/fleet.go`: `if !sv.OK || sv.Model == ""` skips it). Nothing in that chain
        reads `/health`, so a handover has to be visible HERE or it is not visible.

        While draining the catalog is empty and the status is 503: a prober that checks the code
        and one that checks for a model id reach the same conclusion, and the route is dropped
        before a caller pays a failed hop into an engine that is already refusing (45차 §56).
        """
        if not self.alive:
            return {"object": "list", "data": [], "status": "stopping"}, 503
        if self.draining is not None:
            return {"object": "list", "data": [], "status": "draining",
                    "handing_over_to": self.draining}, 503
        return {"object": "list", "data": [self.model_card()]}, 200

    def readiness(self) -> "tuple[dict, int]":
        """(body, HTTP status) for `/health`: serving, handing over, or stopping.

        A handover used to look healthy: `alive` stays true through the drain -- that is the
        point, the rows already here finish and are parked -- while every NEW request is already
        refused with 503. So this said `ok` about an engine that was accepting nothing.

        Nothing in the Deneb chain reads this endpoint (wormhole probes `/v1/models`, the
        supervisor generates a real chat), so this is honesty rather than a route change; the
        one that moves a route is `catalog` above. It is here because an operator with `curl`
        asks `/health` first, and a state nobody can name is a state nobody watches.
        """
        if not self.alive:
            return {"status": "stopping"}, 503
        if self.draining is not None:
            return {"status": "draining", "handing_over_to": self.draining,
                    "running": len(self.runner.state.running),
                    "waiting": len(self.runner.state.waiting) + len(self._waiting)}, 503
        return {"status": "ok"}, 200

    PROFILE_MAX_STEPS = 32                         # a run holds every kernel of every step it covers

    def _begin_profile(self, steps: int) -> None:
        """Profile the next `steps` DECODE steps on this rank. Prefill is not offered: one chunk is seconds of
        kernels, and what is unaccounted for is the decode step."""
        import torch
        if self._profiling is not None:
            return                                 # a run is already in flight; the second ask joins nothing
        steps = max(1, min(int(steps), self.PROFILE_MAX_STEPS))
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False)
        prof.__enter__()
        self._profiling = {"left": steps, "prof": prof, "steps": steps}

    def _end_profile(self) -> None:
        run = self._profiling
        self._profiling = None
        run["prof"].__exit__(None, None, None)
        rows = []
        for event in run["prof"].key_averages():
            total = getattr(event, "self_device_time_total", 0.0) or 0.0
            if total > 0:
                rows.append({"kernel": event.key, "calls": event.count,
                             "us_total": round(total, 1), "us_per_step": round(total / run["steps"], 1)})
        rows.sort(key=lambda r: -r["us_total"])
        device = sum(r["us_total"] for r in rows)
        self.profile_table = {"steps": run["steps"], "rank": self.comm.rank,
                              "device_us_per_step": round(device / run["steps"], 1),
                              "kernels": rows[:120]}

    def _control(self, control) -> None:
        """A cache control, applied on every rank in the same iteration (the caches must stay identical)."""
        kind, payload = control
        if kind == 'latency':
            operation, token = payload.get('op'), payload.get('token')
            try:
                if operation == 'begin':
                    if self._active or self._waiting or self.runner.inflight or self._profiling is not None:
                        raise ValueError('latency recording requires idle serving and no other profiler')
                    mine = self.latency.begin(token, payload.get('diagnostic', False), payload.get('concurrency', 1))
                elif operation in ('end', 'abort'):
                    if operation == 'end' and (self._active or self._waiting or self.runner.inflight):
                        raise ValueError('latency recording still has active requests')
                    if operation == 'abort' and self.latency.active and self.latency.active['token'] == token:
                        self.latency.active['errors'].append('client aborted recording')
                    clock = getattr(getattr(self.engine, 'pipeline', None), 'clock', None)
                    mine = self.latency.finish(token, clock)
                elif operation == 'artifact':
                    mine = self.latency.artifact(token, payload.get('file'), payload.get('offset', 0)) if payload.get('rank') == self.comm.rank else {'rank': self.comm.rank}
                else:
                    raise ValueError('unknown latency operation')
            except Exception as exc:
                mine = dict(rank=self.comm.rank, token=token, error=str(exc), complete=False)
            ranks = self.comm.gather_objects(mine) if getattr(self.comm, 'world_size', 1) > 1 else [mine]
            if operation == 'begin' and any(r.get('error') for r in ranks):
                if self.latency.active and self.latency.active['token'] == token:
                    self.latency.active['errors'].append('another rank rejected begin')
                    self.latency.finish(token)
            if self.comm.rank == 0:
                waiting = self.latency_replies.get(payload['_control_id'])
                if waiting is not None:
                    waiting['reply'] = dict(schema=1, ranks=ranks, boot_id=self.latency_boot_id)
                    waiting['event'].set()
            return
        if kind == "profile":
            if self.latency.active:
                return
            self._begin_profile(int(payload))
            return
        if kind == "calibration":                          # every rank files its own sums (the sharded projections differ per rank)
            file = getattr(self.engine, "file_calibration", None)
            if file is not None:
                file(payload)
            return
        prefix = getattr(self.runner, "prefix", None)
        if prefix is None:
            return
        if kind == "pin":
            prefix.pin(bytes.fromhex(h) for h in payload)
        elif kind == "unpin":
            prefix.unpin_all()
        elif kind == "reset":                              # every rank forgets the same boundaries in the same step
            report = self.runner.reset_prefix()
            self.prefix_resets += 1
            if self.comm.rank == 0:
                print(f"  prefix reset: {report['entries']} boundaries and {report['faded']} faded, "
                      f"{report['tier_forgotten']} off the tier"
                      + (f"; {report['kept_spilling']} kept (being written)" if report["kept_spilling"] else ""),
                      flush=True)
    def _admit_clock(self, request) -> None:
        """The moment a row began stepping this request: queue time ends here, inference time starts."""
        if request in self._admitted:
            return                                  # a continuation reuses its row; the first admission owns the clock
        now = self.clock()
        self._admitted[request] = now
        arrived = self._arrived.get(request)
        if arrived is not None:
            self.queued.observe(now - arrived)
            self.latency.row(kind='request', operation='queue', phase='admission', request_id=request,
                             duration_us=(now - arrived) * 1e6)

    def _note_cached(self, request, row) -> None:
        """Prompt tokens this row did not prefill, read where every admission path has already agreed on it: the
        scheduler's `computed` is set to the adopted prefix by `runner.submit` (a cached or restored boundary) and to
        the held history by `runner.extend` (a retained conversation). One number, three paths, no predicate."""
        if len(self._cached) > 4 * self.max_pending:
            live = set(self.pending)
            self._cached = {k: v for k, v in self._cached.items() if k in live}
        self._cached[request] = int(self.runner.state.computed.get(row, 0))
        self.latency.row(kind='request', operation='admit', phase='admission', request_id=request,
                         row=row, cached_tokens=self._cached[request])

    def cached_tokens(self, *requests) -> int:
        """Prompt tokens these requests did not have to prefill. Taken, not read: one answer asks once."""
        return sum(self._cached.pop(int(r), 0) for r in requests)

    def _evict_idle(self, exclude=None):
        """No tier: a resident idle conversation makes room by ending."""
        row = next((s for s in self._idle_order if s != exclude), None)
        if row is None:
            return False
        try:
            self.runner.evict(row)
        finally:
            self.engine.forget(row)
        self._idle_order.pop(row)
        self._conversations.pop(self._conversation_of.pop(row))
        heapq.heappush(self._free_rows, row)
        return True

    def _reorder_waiting(self) -> None:
        """A request that yielded to a running prefill goes first once the boundary it waited for is cached (45차 §23 F):
        it adopts what it waited for, and the ones behind it did not wait."""
        if not self._deferred or len(self._waiting) < 2:
            return
        prefix = self.runner.prefix
        for i, entry in enumerate(self._waiting):
            if i and entry[0] in self._deferred:
                ids, chain = entry[1], entry[11]
                if chain is None or self.runner.shared_ahead(ids, (), prefix.peek_chain(chain, len(ids)), chain=chain) is None:
                    del self._waiting[i]
                    self._waiting.appendleft(entry)
                    return

    def _admit(self):
        self._reorder_waiting()
        spun = 0
        while self._waiting:
            (request, ids, limit, temperature, promised, conversation, min_new, options, hint, media, tier, chain,
             salt) = self._waiting[0]
            row = None
            resident = held = 0
            parked = False
            drop = False
            if conversation is None and hint is not None:
                key, prefix, drop = hint
                row_ = self._conversations.get(key)
                rest = self._media_after(media, prefix)
                if rest is None:
                    self._waiting[0] = (request, ids, limit, temperature, promised, None, min_new, options, None, media, tier, chain, salt)   # a picture straddles the cut
                    continue
                if (row_ is not None and row_ in self.runner.idle) or (row_ is None and self.runner.is_parked(key)):
                    conversation, ids, media = key, ids[prefix:], rest     # continue the retained conversation with the new turn
                elif row_ is not None or key in self._retiring.values() or any(e["conversation"] == key for e in self._resuming.values()):
                    break                                         # it is mid-park/resume or live: decide next step
                else:
                    self._waiting[0] = (request, ids, limit, temperature, promised, None, min_new, options, None, media, tier, chain, salt)   # gone: fresh prompt
                    continue
            salts = ([salt] if salt else []) + [(m["positions"][0], bytes.fromhex(m["digest"])) for m in media]
            if conversation is None and hint is None and getattr(self.runner, "prefix", None) is not None:
                if chain is None:
                    chain = self.runner.prefix.chain(ids, salts)  # a request that arrived without one (a continuation that fell back)
                # the same prompt is being prefilled right now: wait for its boundary rather than compute it beside it (B)
                above = self.runner.prefix.peek_chain(chain, len(ids))
                ahead = self.runner.shared_ahead(ids, salts, above, chain=chain)
                if ahead is not None:
                    if request not in self._deferred:
                        self._deferred.add(request)
                        self.runner.dedup_waits += 1
                    if len(self._waiting) > 1 and spun < len(self._waiting):
                        self._waiting.rotate(-1)                  # let the ones behind it go; it comes back around
                        spun += 1
                        continue
                    break
                self._deferred.discard(request)
                if tier is not None and tier[0] <= above:
                    tier = None                                   # memory already gives as much: nothing to read
            if conversation is not None:
                row = self._conversations.get(conversation)
                if row is None and (conversation in self._retiring.values()
                                    or any(e["conversation"] == conversation for e in self._resuming.values())):
                    break                                     # its park/resume is still on the tier's thread: next step
                parked = row is None and self.runner.is_parked(conversation)
                if (row is None and not parked) or (row is not None and row not in self.runner.idle):
                    self._waiting.popleft()
                    self._answer(request, RequestError("conversation is unknown, live or evicted", 409))
                    continue
                if parked:
                    record = self.runner.parked_record(conversation)
                    end = record["context"] + record["pending"] + len(ids)
                    held = self.runner.parked_blocks(conversation)
                else:
                    end = self.engine.context(row) + self.engine.extension_tokens(row, ids)
                    resident = held = self.runner.kv.blocks_for(self.runner.kv.tokens[row])
                horizon = end + limit - 1 + (self.runner.c.draft_slots if limit > 1 else 0)
                promised = self.runner.kv.blocks_for(horizon)
                room = min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq) * self.runner.kv.block_size
                needs = f"needs {horizon} ({end} for the conversation so far, {limit} to generate)"
                if horizon >= 2**31 or promised > min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq):
                    self._waiting.popleft()
                    self._answer(request, RequestError(f"the KV pool holds {room} tokens for one request; this turn {needs}"))
                    continue
                if horizon > self.max_context:
                    self._waiting.popleft()
                    self._answer(request, RequestError(f"this model serves {self.max_context} tokens of context; this turn {needs}"))
                    continue
                promised = max(promised, held)       # rejected-draft reservations may exceed the new turn
            if row is None and not self._free_rows:
                if self._evict_idle():
                    continue
                break
            # Future decode growth already belongs to admitted requests even
            # though the block pool acquires those blocks only when written.
            future = (sum(b - self.runner.kv.blocks_for(self.runner.kv.tokens[r]) for r, (_, b) in self._active.items())
                      + sum(e["promised"] - self.runner.kv.blocks_for(self.runner.kv.tokens[r]) for r, e in self._resuming.items())
                      + sum(e["promised"] - self.runner.kv.blocks_for(self.runner.kv.tokens[r]) for r, e in self._restoring.items()))
            if promised > self.runner.kv.available - future + resident:
                if self._evict_idle(exclude=row):
                    continue
                break
            if conversation is None and tier is not None:
                # a longer boundary is on the prefix tier: read it into the row, admit when every rank has it (A). The budget
                # above already holds this request's blocks; the restore takes the first of them now.
                tokens, h = int(tier[0]), bytes.fromhex(tier[1])
                have = h in self.runner.prefix.tier_keys and not self.runner.prefix.has(h)
                world = int(getattr(self.comm, "world_size", 1) or 1)
                if self._votes([have])[0] == world:
                    row = heapq.heappop(self._free_rows)
                    try:
                        self.runner.restore_begin(row, h, tokens)
                    except Exception:                             # noqa: BLE001 -- no snapshot / no blocks / no tier: prefill it instead
                        heapq.heappush(self._free_rows, row)
                        self._waiting[0] = (request, ids, limit, temperature, promised, None, min_new, options, None, media, None, chain, salt)
                        continue
                    self._restoring[row] = dict(request=request, ids=ids, limit=limit, temperature=temperature, promised=promised,
                                                min_new=min_new, options=options, media=media, chain=chain, salt=salt,
                                                cancelled=None)
                    self._waiting.popleft()
                    continue
                self._waiting[0] = (request, ids, limit, temperature, promised, None, min_new, options, None, media, None, chain, salt)
                continue
            if conversation is None:
                row = heapq.heappop(self._free_rows)
                try:
                    self.engine.add(row, ids, max_new=limit, temperature=temperature, **({"min_new": min_new} if min_new else {}),
                                    **({"options": options} if options else {}), **({"media": media} if media else {}))
                    self.runner.submit(row, len(ids), ids=ids, salts=salts, chain=chain)
                except BaseException:
                    self.engine.forget(row)
                    heapq.heappush(self._free_rows, row)
                    raise
                if self.runner.keep_idle:
                    self._conversations[request] = row
                    self._conversation_of[row] = request
            else:
                if parked:
                    row = heapq.heappop(self._free_rows)
                    try:
                        self.runner.resume_begin(row, key=conversation)   # blocks + slot back from NVMe on the tier's thread
                    except BaseException:
                        heapq.heappush(self._free_rows, row)              # the disk copy survives
                        raise
                    self._resuming[row] = dict(conversation=conversation, request=request, ids=ids, limit=limit,
                                               temperature=temperature, promised=promised, min_new=min_new, options=options,
                                               media=media, drop=drop, cancelled=None)
                    self._waiting.popleft()
                    continue                                              # admitted when every rank's read is done (_settle)
                self._idle_order.pop(row)
                tokens = self.engine.extend(row, ids, max_new=limit, temperature=temperature, **({"min_new": min_new} if min_new else {}),
                                            **({"options": options} if options else {}), **({"media": media} if media else {}),
                                            **({"drop_unfed": True} if drop else {}))
                self.runner.extend(row, tokens)
            self._waiting.popleft()
            self._active[row] = (request, promised)
            self._note_cached(request, row)
            self._admit_clock(request)

    def _votes(self, flags) -> "list[int]":
        """How many ranks say yes to each flag. Every rank must call this with the same flags in the
        same order (the transfers are submitted in lockstep); a single rank answers itself."""
        world = int(getattr(self.comm, "world_size", 1) or 1)
        if world <= 1 or not flags:
            return [int(bool(f)) for f in flags]
        if hasattr(self.comm, "all_reduce_host"):                       # the control group: no device work, no stream wait
            return self.comm.all_reduce_host([int(bool(f)) for f in flags])
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        votes = torch.tensor([int(bool(f)) for f in flags], dtype=torch.int32, device=device)
        return [int(v) for v in self.comm.all_reduce(votes).tolist()]

    def _settle(self):
        """Finish the transfers every rank agrees are done. Nothing here waits on the disk (D10):
        a park or resume that is still writing/reading is simply looked at again next step.
        Outcomes are agreed too, so the ranks keep the same set of conversations."""
        rows = self.runner.transfers()
        if not rows:
            return
        world = int(getattr(self.comm, "world_size", 1) or 1)
        done = self._votes([self.runner.transfer_done(r) for r in rows])
        outcomes = []                                         # (row, ok, full)
        restored = {}                                         # row -> (tokens, snap) a prefix restore landed with
        for row, votes in zip(rows, done):
            if votes < world:
                continue
            ok, full = True, False
            try:
                if row in self._retiring:
                    self.runner.park_finish(row)
                elif row in self._restoring:
                    restored[row] = self.runner.restore_finish(row)
                else:
                    self.runner.resume_finish(row)
            except TierFull:
                ok, full = False, True
            except Exception:                                 # noqa: BLE001 -- a failed transfer is the conversation's loss, not the engine's
                ok = False
            outcomes.append((row, ok, full))
        if not outcomes:
            return
        agreed = self._votes([ok for _, ok, _ in outcomes] + [full for _, _, full in outcomes])
        for (row, ok, full), all_ok, all_full in zip(outcomes, agreed[:len(outcomes)], agreed[len(outcomes):]):
            if row in self._retiring:
                conversation = self._retiring.pop(row)
                if all_ok == world:                           # parked everywhere: the row is free
                    heapq.heappush(self._free_rows, row)
                elif all_full == world and self.runner.forget_oldest_parked() is not None:
                    self.runner.park_begin(row, key=conversation)   # room was made on every rank: write again
                    self._retiring[row] = conversation
                else:                                         # dropped everywhere: forget where it landed, evict where it stayed
                    if ok:
                        self.runner.forget_parked(conversation)
                    else:
                        self.runner.evict(row)
                        self.engine.forget(row)
                    heapq.heappush(self._free_rows, row)
            elif row in self._restoring:
                e = self._restoring.pop(row)
                request = e["request"]
                if all_ok == world and e["cancelled"] is None:   # the boundary is back in memory on every rank: the prompt starts after it
                    tokens, snap = restored[row]
                    try:
                        self.engine.add(row, e["ids"], max_new=e["limit"], temperature=e["temperature"],
                                        **({"min_new": e["min_new"]} if e["min_new"] else {}),
                                        **({"options": e["options"]} if e.get("options") else {}),
                                        **({"media": e["media"]} if e.get("media") else {}))
                        self.runner.submit(row, len(e["ids"]), ids=e["ids"],
                                           salts=[(m["positions"][0], bytes.fromhex(m["digest"])) for m in e["media"]] if e.get("media") else (),
                                           prepared=(tokens, snap), chain=e.get("chain"))
                    except BaseException:
                        self.engine.forget(row)
                        self.runner.restore_undo(row)
                        heapq.heappush(self._free_rows, row)
                        raise
                    if self.runner.keep_idle:
                        self._conversations[request] = row
                        self._conversation_of[row] = request
                    self._active[row] = (request, e["promised"])
                    self._note_cached(request, row)
                else:
                    if ok:
                        self.runner.restore_undo(row)             # landed here but not everywhere, or nobody wants it: the row goes back
                    heapq.heappush(self._free_rows, row)
                    if e["cancelled"] is None:                    # prefill it the plain way, ahead of the queue
                        self._waiting.appendleft((request, e["ids"], e["limit"], e["temperature"], e["promised"], None, e["min_new"],
                                                  e["options"], None, e["media"], None, e.get("chain"), e.get("salt")))
            else:
                e = self._resuming.pop(row)
                conversation, request = e["conversation"], e["request"]
                if all_ok == world and e["cancelled"] is None:   # resident everywhere: the turn proceeds
                    tokens = self.engine.extend(row, e["ids"], max_new=e["limit"], temperature=e["temperature"],
                                                **({"min_new": e["min_new"]} if e["min_new"] else {}),
                                                **({"options": e["options"]} if e.get("options") else {}),
                                                **({"media": e["media"]} if e.get("media") else {}),
                                                **({"drop_unfed": True} if e.get("drop") else {}))
                    self.runner.extend(row, tokens)
                    self._conversations[conversation] = row
                    self._conversation_of[row] = conversation
                    self._active[row] = (request, e["promised"])
                    self._note_cached(request, row)
                    self._admit_clock(request)
                elif all_ok == world:                         # the client left while its conversation was coming back: park it again
                    self._conversations[conversation] = row
                    self._conversation_of[row] = conversation
                    self._retire(row)
                else:                                         # a rank could not read it back: the conversation is gone everywhere
                    if ok:
                        self.runner.evict(row)
                        self.engine.forget(row)
                    else:
                        self.runner.forget_parked(conversation)
                    heapq.heappush(self._free_rows, row)
                    if e["cancelled"] is None:
                        self._answer(request, RequestError("conversation could not be restored from the tier", 503))

    def _yield_asked(self) -> "str | None":
        """Rank 0: has anyone asked for the fleet? Polled, and it publishes while it looks.

        This is the half of a handover a queue cannot do on its own. A queue can put a
        session at the front of the line; only the engine can finish the conversations it
        is holding and put them where they survive the next boot (D16).
        """
        if self.draining is not None:
            return self.draining
        if not self.lease or self.comm.rank != 0:
            return None
        now = self.clock()
        if now - self._lease_seen < self.lease_poll_s:
            return None
        self._lease_seen = now
        from engine.base import fleet_lease
        try:
            record = fleet_lease.read(self.lease["path"])
            if not record or record.get("owner") != self.lease["owner"]:
                return None                       # not our lease any more: nothing to answer
            fleet_lease.publish(self.lease["owner"], path=self.lease["path"],
                                running=len(self.runner.state.running),
                                waiting=len(self.runner.state.waiting) + len(self._waiting),
                                served=self.served, steps=self.runner.steps)
            asked = fleet_lease.yield_requested(record)
        except Exception:                         # noqa: BLE001 -- the lease never takes serving down
            return None
        return asked.get("requester") if asked else None

    def _quiet(self) -> bool:
        """Nothing left to finish: no request anywhere, and no transfer on the tier's thread."""
        return not (self.runner.state.running or self.runner.state.waiting or self._waiting
                    or self.pending or self._active or self._retiring or self._resuming
                    or self._restoring)

    def _hand_over(self) -> None:
        """Park what is still resident, let the lease go, and end the loop.

        A finished turn is already parked by `_retire` when a tier is configured, so what
        is left here is the rows that stayed resident. They are parked under their
        conversation key, which is exactly what the next holder resumes by -- so the
        fleet changes hands without anyone losing their context.
        """
        parked = lost = 0
        if self.runner.tiered is not None:
            for row, conversation in list(self._conversation_of.items()):
                self._conversations.pop(conversation, None)
                self._conversation_of.pop(row, None)
                self._idle_order.pop(row, None)
                try:
                    self.runner.park(row, key=conversation)
                    parked += 1
                except Exception:                 # noqa: BLE001 -- one lost turn is not a lost handover
                    lost += 1                     # ... but it is never a silent one
        self.handed_over = {"to": self.draining, "parked": parked, "lost": lost}
        if lost:
            print(f"  handover to {self.draining}: {parked} conversations parked, {lost} LOST", flush=True)
        if self.lease and self.comm.rank == 0:
            from engine.base import fleet_lease
            try:
                # Say what happened before letting go: the next holder reads this file.
                fleet_lease.publish(self.lease["owner"], path=self.lease["path"],
                                    phase="handed over", parked=parked, lost=lost)
                # Hand it OVER, not back: the record becomes the requester's in one step, so
                # there is no moment where the fleet reads as free and the production supervisor
                # relaunches into the window this drain was for (its own comment names that
                # race). What the request named -- kind, pid, host -- is what the requester is.
                asked = fleet_lease.yield_requested(fleet_lease.read(self.lease["path"])) or {}
                to = asked.get("requester") or self.draining
                if to:
                    fleet_lease.transfer(self.lease["owner"], to, path=self.lease["path"],
                                         kind=asked.get("kind") or "session", pid=int(asked.get("pid") or 0),
                                         host=asked.get("host") or "", note=asked.get("reason") or "",
                                         est_minutes=int(asked.get("est_minutes") or 0))
                else:
                    fleet_lease.release(self.lease["owner"], path=self.lease["path"])
            except Exception:                     # noqa: BLE001
                pass
        self.alive = False                        # `once` returns False and `loop` ends

    def _retire(self, row):
        """A finished turn leaves its row: parked with a tier, resident idle without, released otherwise."""
        if not self.runner.keep_idle:
            self.engine.forget(row)
            heapq.heappush(self._free_rows, row)
            return
        if self.runner.tiered is None:
            self._idle_order[row] = None
            return
        conversation = self._conversation_of.pop(row)
        self._conversations.pop(conversation)
        self.runner.park_begin(row, key=conversation)         # the write runs on the tier's thread; the row frees in _settle
        self._retiring[row] = conversation

    def _abort(self):
        self.alive = False
        # Preserve the original engine exception; cleanup must wake clients
        # even if a model's close hook also fails. Parked conversations are
        # on disk and stay there (D16: they outlive this process).
        error = None
        try:
            self.runner.settle()
        except BaseException as exc:                          # noqa: BLE001
            error = exc
        for row in list(self.runner.slot_of):
            try:
                try:
                    self.runner.cancel(row)
                finally:
                    self.engine.forget(row)
            except BaseException as exc:
                error = error or exc
        self._active.clear()
        self._waiting.clear()
        self._retiring.clear()
        self._resuming.clear()
        self._idle_order.clear()
        self._conversations.clear()
        self._conversation_of.clear()
        if error is not None:
            raise error

    def _fail_pending(self):
        with self._lock:
            self.alive = False
            for request, event in self.pending.items():
                self.results[request] = RequestError("engine stopped before completing the request", 503)
                event.set()
            self.pending.clear()
            self._drain()
        for stream in list(self._streams.values()):
            stream.put(("error", "engine stopped before completing the request"))
        self._wake.set()
        self._streams.clear()
        self._sent.clear()
        self._deadline.clear()

    def metrics(self) -> str:
        """Prometheus text: the bench's vLLM dialect, plus what an operator needs to see.

        The first block is the contract with bench/window_metrics.py and bench/bracket.py --
        those names and their meanings do not move, so the same onepass gate judges this
        engine and the one it replaces. The rest answers the three questions the counters
        alone could not: how long a request waits (latency histograms, from admission, not
        from the step that served it), how close the caches are to full, and whether the
        reuse that was built is being hit.
        """
        engine, runner = self.engine, self.runner
        kv, slots = runner.kv, runner.slots
        # Occupancy is what a row (or a tier read in flight) holds. A block a boundary still
        # claims is NOT held: it waits in the free list under that boundary's name and leaves
        # the moment a reservation wants one (base/kv.py), exactly as vLLM's free queue holds
        # its cached blocks. How much of the free space is reuse is the split below.
        free_blocks = kv.available
        used_blocks = kv.num_blocks - free_blocks
        rows = [
            ("counter", "vllm:request_success_total", "requests answered", self.served),
            ("gauge", "vllm:num_requests_running", "requests in the model's step", len(runner.state.running)),
            ("gauge", "vllm:num_requests_waiting", "admitted or queued, not yet stepping",
             len(runner.state.waiting) + len(self._waiting) + len(self.pending) - len(self._active)),
            ("counter", "vllm:prompt_tokens_total", "prompt tokens admitted", self.prompt_tokens_total),
            ("counter", "vllm:generation_tokens_total", "tokens generated", self.generation_tokens_total),
            ("counter", "st:generation_characters_total", "characters those tokens spelled, as the client read them",
             self.generation_characters_total),
            ("counter", "vllm:spec_decode_num_accepted_tokens_total", "drafts the target confirmed",
             getattr(engine, "accepted_total", 0)),
            ("counter", "vllm:spec_decode_num_draft_tokens_total", "tokens the drafter proposed",
             getattr(engine, "drafted_total", 0)),
        ]
        if hasattr(engine, "drafts_total"):
            rows.append(("counter", "vllm:spec_decode_num_drafts_total", "proposal rounds", engine.drafts_total))
        # bench/bracket._StepWindows samples this vLLM histogram count as "engine steps" for its decode
        # windows (steps/s): the runner's step counter is the same quantity here (a prefill chunk or a
        # decode step each). The split below is the same total, by kind (D9).
        rows += [
            ("counter", "vllm:iteration_tokens_total_count", "model steps", runner.steps),
            ("gauge", "vllm:gpu_cache_usage_perc", "KV blocks held, as a fraction of the pool",
             round(used_blocks / kv.num_blocks, 6) if kv.num_blocks else 0.0),
            ("counter", "st:steps_prefill_total", "steps that were a prefill chunk", self.steps_prefill),
            ("counter", "st:steps_decode_total", "steps that were a decode", self.steps_decode),
            ("counter", "st:async_decode_steps_total", "decode steps launched ahead of host readback", runner.async_steps),
            ("counter", "st:sync_drain_steps_total", "readbacks forced by prefill or synchronous decode", runner.sync_drain_steps),
            ("counter", "st:decode_row_steps_total", "sum of submitted row counts across decode steps",
             sum(n * count for n, count in enumerate(runner.decode_batches))),
            ("gauge", "st:decode_batch_capacity", "maximum resident decode rows", runner.c.max_running),
            ("counter", "st:requests_cancelled_total", "requests cancelled, for any reason", self.cancelled),
            ("counter", "st:requests_timed_out_total", "the subset the deadline scan took", self.timed_out),
            ("gauge", "st:handing_over", "1 while the fleet is being handed to another session",
             int(self.draining is not None)),
            # `_quiet` is the engine's own answer to "is there anything left to finish", and it is
            # the one a handover waits for. It counts more than the two request gauges do -- a tier
            # transfer on its own thread is not a request and stops nothing from reporting zero --
            # so anything deciding it may take this engine down has to read this and not those.
            ("gauge", "st:quiet", "1 when no request and no tier transfer is outstanding",
             int(self._quiet())),
            ("counter", "st:handover_conversations_parked", "turns parked for the next holder",
             (self.handed_over or {}).get("parked", 0)),
            ("counter", "st:handover_conversations_lost", "turns a handover could not park",
             (self.handed_over or {}).get("lost", 0)),
            ("gauge", "st:kv_blocks_total", f"blocks of {kv.block_size} tokens in the pool", kv.num_blocks),
            ("gauge", "st:kv_blocks_used", "blocks a row holds (or a tier transfer in flight)", used_blocks),
            ("gauge", "st:kv_rows_in_use", "pool rows with tokens", kv.rows_in_use),
            ("gauge", "st:state_slots_total", "recurrent/conv state slots (slot 0 is the null slot)",
             slots.num_slots - 1),
            ("gauge", "st:kv_blocks_free", "blocks no row holds: free now, whoever remembers them", free_blocks),
            ("gauge", "st:kv_blocks_anonymous", "free blocks no boundary remembers: spent before any reuse is", kv.anonymous),
            ("gauge", "st:kv_blocks_cached", "free blocks a cached boundary holds: the last thing a reservation spends",
             kv.cached),
            ("gauge", "st:kv_blocks_faded", "free blocks held by a boundary whose snapshot is gone", kv.faded),
            ("gauge", "st:state_slots_free", "state slots a new request could take", slots.available),
            *device_memory_rows(),
            *_grammar_rows(getattr(self.engine, "grammars", None)),
            ("counter", "st:prompt_tokenize_spliced_total",
             "chat prompts tokenized as a tail onto a remembered prefix", getattr(self.prompt_tokens, "spliced", 0)),
            ("counter", "st:prompt_tokenize_full_total",
             "chat prompts tokenized from the first character", getattr(self.prompt_tokens, "full", 0)),
            ("counter", "st:prompt_tokenize_chars_saved_total",
             "prompt characters a splice did not have to tokenize again",
             getattr(self.prompt_tokens, "chars_saved", 0)),
            ("gauge", "st:detokenizer_rust_stream",
             "1 when streamed text is decoded through tokenizers' Rust DecodeStream", int(self.rust_detok)),
        ]
        prefix = getattr(runner, "prefix", None)
        if prefix is not None:
            rows += [
                ("counter", "vllm:prefix_cache_queries_total", "prompts looked up in the prefix cache",
                 prefix.hits + prefix.misses),
                ("counter", "vllm:prefix_cache_hits_total", "lookups that reused a cached prefix", prefix.hits),
                ("counter", "st:prefix_cache_evictions_total", "cached prefixes dropped", prefix.evictions),
                ("counter", "st:prefix_resets_total", "times an operator threw the whole prefix cache away", self.prefix_resets),
                ("gauge", "st:prefix_cache_reclaimable_blocks", "free blocks a boundary holds: reuse a reservation can spend",
                 prefix.reclaimable()),
                ("counter", "st:prefix_reused_tokens_total", "prompt tokens served from a cached boundary (memory or tier)",
                 getattr(runner, "reused_tokens", 0)),
                ("gauge", "st:prefix_entries", "boundaries in memory", len(prefix.entries)),
                ("gauge", "st:prefix_faded_entries", "boundaries that kept their blocks after giving up a snapshot",
                 len(prefix.faded)),
                ("counter", "st:prefix_cache_fades_total", "boundaries that gave a snapshot away and kept their blocks",
                 prefix.fades),
                ("gauge", "st:prefix_pinned_entries", "boundaries an operator pinned", sum(1 for e in prefix.entries.values() if e.pinned)),
                ("gauge", "st:prefix_tier_entries", "boundaries the prefix tier holds", len(prefix.tier_keys)),
                ("counter", "st:prefix_tier_spills_total", "boundaries written to the prefix tier", getattr(runner, "prefix_spills", 0)),
                ("counter", "st:prefix_tier_restores_total", "boundaries read back from the prefix tier", getattr(runner, "prefix_restores", 0)),
                ("counter", "st:prefix_dedup_waits_total", "requests that waited for a running prefill's boundary instead of computing it",
                 getattr(runner, "dedup_waits", 0)),
                ("gauge", "st:prefix_snapshots_free", "snapshot slots no boundary holds: what the next block boundary can take",
                 len(prefix.free_snaps)),
                ("counter", "st:prefix_snapshot_denials_total", "block boundaries left uncached because no snapshot could be freed",
                 prefix.snapshot_denials),
                ("counter", "st:prefix_snapshot_self_evicts_total",
                 "checkpoints a prefill displaced to make room for its own later ones: state copies computed and thrown away",
                 getattr(runner, "snapshot_self_evicts", 0)),
            ]
        # The prefix tier's OWN bytes, keyed on that tier and not on the cache above it. The pair
        # under `tiered` below is the CONVERSATION tier, so everything the boundary tier moved was
        # invisible: a live fleet showed 42 spills and 2 restores against `st:tier_bytes_read_total`
        # 0, and the one thing that decides the snapshot pool's size -- what a faded boundary costs
        # to read back -- could not be read from the scrape at all (2026-09-12).
        boundary = getattr(getattr(runner, "prefix_tier", None), "tier", None)
        moved_out, moved_in = getattr(boundary, "bytes_written", None), getattr(boundary, "bytes_read", None)
        if moved_out is not None and moved_in is not None:
            rows += [("counter", "st:prefix_tier_bytes_written_total",
                      "bytes of boundary state written to the prefix tier", moved_out),
                     ("counter", "st:prefix_tier_bytes_read_total",
                      "bytes of boundary state read back from the prefix tier", moved_in)]
        tiered = getattr(runner, "tiered", None)
        if tiered is not None:
            tier = getattr(tiered, "tier", None)
            rows.append(("gauge", "st:conversations_parked", "conversations resident on the NVMe tier",
                         len(tiered.parked)))
            # The byte counters are NvmeTier's, not the tier contract's: a tier without them
            # simply has no series here. /metrics answers with what it has; it never raises.
            written, read = getattr(tier, "bytes_written", None), getattr(tier, "bytes_read", None)
            if written is not None and read is not None:
                rows += [("counter", "st:tier_bytes_written_total", "bytes parked to the tier", written),
                         ("counter", "st:tier_bytes_read_total", "bytes resumed from the tier", read)]
        # --- what no vLLM counter can answer, because no vLLM has these parts ---
        labelled = []                                   # (name, type, help, [(label text, value)])
        shapes = getattr(engine, "decode_shape_counts", None)
        if shapes:
            # The scheduler's real batch: how many sequences a decode step actually carried.
            by_seqs = {}
            for (seqs, _), count in shapes.items():
                by_seqs[seqs] = by_seqs.get(seqs, 0) + count
            labelled.append(("st:decode_steps_by_sequences_total", "counter",
                             "decode steps that carried this many sequences: the batch the scheduler filled",
                             [(f'sequences="{n}"', v) for n, v in sorted(by_seqs.items())]))
            # The captured graph each step ran. A bucket that never appears is a graph captured
            # at every boot for a request that does not arrive (STK_context_ceiling prices it).
            by_bucket = {}
            for (_, capacity), count in shapes.items():
                by_bucket[capacity] = by_bucket.get(capacity, 0) + count
            labelled.append(("st:decode_capacity_bucket_total", "counter",
                             "decode steps by the context-capacity bucket whose graph served them",
                             [(f'capacity="{c}"', v) for c, v in sorted(by_bucket.items())]))
        if any(self.detok_repairs.values()):
            # Zero in every healthy run, so the series only exists once something went wrong.
            labelled.append(("st:detokenizer_repairs_total", "counter",
                             "streamed text the door had to repair, by what went wrong",
                             [(f'reason="{reason}"', count) for reason, count in sorted(self.detok_repairs.items()) if count]))
        if self.reasoning_shapes:
            # The other half of st:reuse_path_total: that series says whether a prompt continued a
            # conversation, this one says whether it COULD (45차 §81).
            labelled.append(("st:reasoning_shape_total", "counter",
                             "chat requests by the reasoning shape the template rendered them with",
                             [(f'thinking="{t}",effort="{e}"', count)
                              for (t, e), count in sorted(self.reasoning_shapes.items())]))
        if any(self.reuse_paths.values()):
            labelled.append(("st:reuse_path_total", "counter",
                             "prompts by how they found their KV: a conversation they extend, or blocks they share",
                             [(f'path="{path}"', count) for path, count in sorted(self.reuse_paths.items())]))
        if self.by_reason:
            labelled.append(("vllm:request_success_by_reason_total", "counter",
                             "requests answered, by why they stopped",
                             [(f'finished_reason="{reason}"', count) for reason, count in sorted(self.by_reason.items())]))
        positions = getattr(engine, "ceiling_positions", 0)
        if positions:
            # Acceptance has three ceilings; these say which one to lift next (base/sampler.draft_ceilings).
            rows.extend([
                ("counter", "st:spec_draft_positions_sampled_total", "draft positions behind the two masses below", positions),
                ("counter", "st:spec_reachable_mass_total",
                 "sum over those of sum_x min(target, draft): the most any verification rule could accept",
                 round(engine.reachable_mass, 6)),
                ("counter", "st:spec_candidate_mass_total",
                 "sum over those of the target mass the drafter's candidates cover at all",
                 round(engine.covered_mass, 6)),
            ])
        clock = getattr(getattr(engine, "pipeline", None), "clock", None)
        if clock is not None and clock.totals:
            # Where a decode step's DEVICE time goes, sampled one step in 64 and read a round late so nothing
            # waits (base/stage_clock). The two step kinds are already counted; this is what a step is made of.
            labelled.append(("st:decode_stage_seconds_total", "counter",
                             "device time inside a decode step, by stage, over the sampled steps",
                             [(f'stage="{k}"', round(v, 6)) for k, v in sorted(clock.totals.items())]))
            rows.append(("counter", "st:decode_stage_samples_total",
                         "decode steps whose stages were timed", clock.samples))
        # The gauge's own health, always, even when it never produced a position. It is off the
        # answer path by construction now; this is how anyone finds out it stopped measuring.
        failures = getattr(engine, "ceiling_failures", None)
        if failures is not None:
            rows.append(("counter", "st:spec_ceiling_samples_failed_total",
                         "acceptance-ceiling samples that raised and were dropped rather than ending the step",
                         failures))
            if getattr(engine, "ceilings_off", False):
                rows.append(("gauge", "st:spec_ceiling_sampling_off",
                             "1 when the ceiling gauge disarmed itself after repeated failures", 1))
        exits = getattr(engine, "chain_exits", None)
        if exits:
            # st:sync_drain_steps_total says how often the pipeline was emptied. This says by what, which is the
            # half an operator can act on: the batch runs ahead together or not at all, so at max_seqs 4 one
            # request asking for logprobs appears here as the whole step's reason.
            labelled.append(("st:decode_chain_exits_total", "counter",
                             "decode steps the device-side chain refused, by what refused them",
                             [(f'reason="{k}"', v) for k, v in sorted(exits.items())]))
        accepted = getattr(engine, "accepted_per_step", None)
        if accepted:
            # Acceptance as a shape, not a mean: a run that is bimodal at 0 and k wants a
            # different k than one that tails off, and the totals above cannot tell them apart.
            labelled.append(("st:spec_accepted_per_step_total", "counter",
                             "decode segments that committed this many drafted tokens",
                             [(f'accepted="{i}"', v) for i, v in enumerate(accepted) if v or i <= 1]))
        out = []
        for kind, name, help_text, value in rows:
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} {kind}\n")
            out.append(f'{name}{{engine="st"}} {max(0, value)}\n')
        for name, kind, help_text, series in labelled:
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} {kind}\n")
            out.extend(f'{name}{{engine="st",{label}}} {max(0, value)}\n' for label, value in series)
        info = getattr(engine, "lane_info", None)
        if info:
            # Armed is not served (45차 §17): this says which lanes and kernel cells this
            # process bound, so a scrape settles it instead of a boot log nobody kept.
            labels = ",".join(f'{k}="{str(v)[:64]}"' for k, v in sorted(info.items()))
            out.append("# HELP st:lane_info the lanes and kernel cells this process actually bound\n"
                       "# TYPE st:lane_info gauge\n")
            out.append(f'st:lane_info{{engine="st",{labels}}} 1\n')
        for name, help_text, histogram in (
                ("vllm:time_to_first_token_seconds", "arrival to first token, queueing included", self.ttft),
                ("vllm:time_per_output_token_seconds", "a step's seconds divided by the tokens it produced", self.itl),
                ("vllm:inter_token_latency_seconds", "seconds between steps that carried tokens", self.step_gap),
                ("vllm:request_queue_time_seconds", "arrival to the first step that carried it", self.queued),
                ("vllm:request_inference_time_seconds", "that step to the last token", self.inference),
                ("vllm:e2e_request_latency_seconds", "admission to the answer", self.e2e)):
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} histogram\n")
            out.extend(f"{series} {value}\n" for series, value in histogram.rows(name))
        # One step, host-observed end to end: both kinds finish on a readback of the sampled
        # ids, so this is the step and not its launch. The floor of the decode series is the
        # host cost I1 exists to bound, without a trace.
        out.append("# HELP st:step_seconds one model step, host-observed end to end\n"
                   "# TYPE st:step_seconds histogram\n")
        for kind, histogram in self.step_seconds.items():
            for series, value in histogram.rows("st:step_seconds"):
                head, _, tail = series.partition('{engine="st"')
                out.append(f'{head}{{engine="st",kind="{kind}"{tail} {value}\n')
        return "".join(out)

    def once(self) -> bool:
        """One ordered broadcast, bounded admission and homogeneous model step."""
        try:
            if self.comm.rank == 0:
                self._expire()
                run = self.latency.active
                if (run and not self._active and not self._waiting and not self.runner.inflight and self.arrivals.empty()
                        and time.monotonic() - run.get('last_row_at', run['started']) > 120):
                    self.controls.put(('latency', dict(op='abort', token=run['token'], _control_id='idle-expiry')))
            alive, arrivals, cancels, controls, draining = self.comm.broadcast_object(
                (self.alive, self._drain(), self._drain_cancels(), self._drain_controls(),
                 self._yield_asked()) if self.comm.rank == 0 else None)
            if draining is not None and self.draining is None:
                self.draining = draining          # every rank stops admitting on the same step
            if not alive:
                self._abort()
                self._fail_pending()
                return False
            self._waiting.extend(arrivals)
            for request, reason in cancels:
                self._cancel(request, reason)
            for control in controls:
                self._control(control)
            self._settle()
            self._admit()
            began = self.clock()
            step = self.runner.step()
            housekeeping = getattr(self.engine, "housekeeping", None)      # a model's own after-step chores (rare device reads)
            if housekeeping is not None and step is not None:
                housekeeping(self.runner.steps)
            now = self.clock()
            if step is not None:                                    # D9: one kind or the other
                kind = "prefill" if step.kind == "prefill" else "decode"   # base/scheduler.PREFILL
                if kind == "prefill":
                    self.steps_prefill += 1
                else:
                    self.steps_decode += 1
                self.step_seconds[kind].observe(now - began)
                if self._profiling is not None and kind == "decode":
                    self._profiling["left"] -= 1
                    if self._profiling["left"] <= 0:
                        self._end_profile()
            # Every live row's new tokens, whether or not it streams: the first one is this
            # request's time to first token, each later one an inter-token interval. A step
            # that lands several (the drafter's accepted run) shares its elapsed time across
            # them, which is how the vLLM counters these names belong to define it.
            for row, (request, _) in self._active.items():
                sent = self._sent.get(row, 0)
                count = self.engine.generated_count(row)     # the row's whole output is never copied to count it
                fresh = count - sent
                if fresh > 0:
                    last = self._token_at.get(row)
                    if last is None:
                        arrived = self._arrived.get(request)
                        if arrived is not None:
                            self.ttft.observe(now - arrived)
                        fresh -= 1                                  # the first token is the TTFT, not an interval
                        last = now
                    if fresh > 0:
                        self.step_gap.observe(now - last)     # one sample per step, however many tokens it carried
                        each = (now - last) / fresh
                        for _ in range(fresh):
                            self.itl.observe(each)
                    self._token_at[row] = now
                    stream = self._streams.get(request)
                    if stream is not None:                          # rank 0: hand it the new tokens
                        lp = getattr(self.engine, "logprobs", None)
                        entries = lp(row) if lp is not None else None
                        stream.put(("tokens", (self.engine.generated_since(row, sent), list(entries[sent:]) if entries else None)))
                        self._wake.set()
                    self._sent[row] = count
            if self.draining is not None and not self.drained and self._quiet():
                self.drained = True
                self._hand_over()
            live = set(self.runner.state.running) | set(self.runner.state.waiting)
            for row in list(self._active):
                if row not in live:
                    request, _ = self._active.pop(row)
                    self._sent.pop(row, None)
                    self._token_at.pop(row, None)
                    arrived = self._arrived.pop(request, None)
                    if arrived is not None:
                        self.e2e.observe(now - arrived)
                    result = list(self.engine.generated(row))
                    admitted = self._admitted.pop(request, None)
                    if admitted is not None:
                        self.inference.observe(now - admitted)
                    self._retire(row)
                    self.served += 1
                    reason = self.finish_reason(result, request)
                    self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
                    self._deadline.pop(request, None)
                    self._answer(request, result)
            return step is not None
        except BaseException:
            try:
                self._abort()
            except BaseException:
                pass                                  # re-raise the original engine/transport failure
            finally:
                self._fail_pending()
            raise

    def _serve_http(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, payload):
                # ensure_ascii=False: JSON is UTF-8 by definition (RFC 8259), and escaping puts a
                # Korean character on the wire as six ASCII bytes instead of its three. A Korean
                # answer's body was 1.83x the size it needed to be (45차 §40).
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    catalog, code = server.catalog()
                    self.reply(code, catalog)
                elif self.path == "/metrics":
                    body = server.metrics().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/v1/engine/latency":
                    self.reply(200, dict(schema=1, boot_id=server.latency_boot_id,
                        ranks=getattr(server.comm, 'world_size', 1),
                        policy='fresh-prefix; observed preparation; profiler-off measurement; separate diagnostic replay'))
                elif self.path == "/health":
                    status, code = server.readiness()
                    self.reply(code, status)
                elif self.path == "/v1/engine/profile":
                    table = server.profile_table
                    self.reply(200 if table else 404,
                               table or {"error": {"message": "no profile has been run: POST /v1/engine/profile"}})
                else:
                    status = {"engine": "ST", "model": server.model_name, "running": list(server.runner.state.running),
                              "waiting": list(server.runner.state.waiting), "queued": len(server._waiting),
                              "parked": len(server.runner.parked_keys()),
                              "parking": len(server._retiring), "resuming": len(server._resuming),
                              "steps": server.runner.steps, "served": server.served}
                    fleet = server.fleet_status()
                    if fleet is not None:
                        status["fleet"] = fleet
                    self.reply(200, status)

            def body(self):
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 4 << 20:
                    raise RequestError("request body must contain 1 to 4194304 bytes", 413)
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise RequestError("request must be a JSON object")
                return req

            def sse(self, payload):
                self.wfile.write(b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()

            def gone(self) -> bool:
                """The client hung up: its socket reads EOF without our having sent anything."""
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
                except (OSError, ValueError):
                    return True

            def wait_result(self, request, event):
                """Block for a whole answer, cancelling the request if the client leaves first."""
                while not event.wait(0.25):
                    if self.gone():
                        server.cancel(request, "client closed")
                        event.wait()
                        break
                return server.take_result(request)

            # ---- the OpenAI dialect ------------------------------------------------------------------------------
            def choices_for(self, ids, count, max_new, temperature, options, stop, *, reasoning, tool_parser=None,
                            want_logprobs=None, min_new=0, continue_history=False, media=None, cache_salt=None):
                """Submit `count` generations of one prompt; each is a _Choice fed by its own token queue.
                With a seed, choice i draws from seed + i so the n answers differ but stay reproducible."""
                choices = []
                for i in range(count):
                    opts = dict(options)
                    if count > 1 and "seed" in opts:
                        opts["seed"] = opts["seed"] + i
                    request, event = server.submit(ids, max_new, temperature, stream=True, min_new=min_new,
                                                   options=opts, continue_history=continue_history, media=media,
                                                   cache_salt=cache_salt)
                    choices.append(_Choice(len(choices), request, event, server._streams[request], tok=server.tok, stop=stop,
                                           reasoning=reasoning, tool_parser=tool_parser, want_logprobs=want_logprobs,
                                           min_new=min_new, repairs=server.detok_repairs,
                                           tool_stream=server.tool_stream))
                return choices

            def run_choices(self, choices, on_delta) -> bool:
                """Drive every choice's queue until all have ended; `on_delta(choice, deltas)` receives each flush.
                False when the client left (every live generation is cancelled)."""
                live = {c.request: c for c in choices}

                def retire(c):
                    """A choice leaves the loop, however it leaves. The characters it showed are
                    counted here and only here, so an answer the client hung up on is counted the
                    same as one that finished -- that traffic is exactly what a ratio against
                    `vllm:generation_tokens_total` is read for."""
                    server.generation_characters_total += sum(len(t) for t in c.text.values())
                    live.pop(c.request, None)

                while live:
                    # Cleared before the queues are read, so an item that arrives during the
                    # read leaves the flag set and the wait below returns at once.
                    server._wake.clear()
                    progressed = False
                    for c in list(live.values()):
                        try:
                            kind, payload = c.q.get_nowait()
                        except queue.Empty:
                            continue
                        progressed = True
                        if kind == "tokens":
                            ids, lps = payload if isinstance(payload, tuple) else (payload, None)
                            c.feed(ids, lps, server.reasoning_end)
                            deltas = c.flush()
                            if deltas:
                                on_delta(c, deltas)
                            if c.finish == "stop":
                                server.cancel(c.request, "stop")            # the loop drops the row; the answer is complete here
                                c.done = True
                                retire(c)
                        elif kind == "end":
                            deltas = c.flush(final=True)
                            if deltas:
                                on_delta(c, deltas)
                            c.finish = c.finish or payload
                            c.done = True
                            retire(c)
                        else:
                            c.error = payload
                            c.done = True
                            retire(c)
                    if not progressed:
                        if self.gone():
                            for c in list(live.values()):
                                server.cancel(c.request, "client closed")
                                retire(c)
                            return False
                        server._wake.wait(0.02)      # the timeout is the disconnect check's period
                return True

            def release(self, choices):
                for c in choices:
                    server._streams.pop(c.request, None)
                    c.event.wait()
                    try:
                        server.take_result(c.request)
                    except RequestError:
                        pass

            def chat(self, req):
                """OpenAI chat completions over the engine: template -> ids -> n generations; the tokens come back
                through each request's queue whether the reply streams or not, so `stop` strings and a client that
                hangs up end the generation early in both modes."""
                if server.chat is None or server.tok is None:
                    raise RequestError("this server has no chat template", 404)
                messages = req.get("messages")
                if (not isinstance(messages, list) or not messages
                        or any(not isinstance(m, dict) or not isinstance(m.get("role"), str)
                               or not (m.get("content") is None or isinstance(m.get("content"), (str, list)))
                               for m in messages)):
                    raise RequestError("messages must be a nonempty list of {role, content} objects")
                kwargs = req.get("chat_template_kwargs") or {}
                if not isinstance(kwargs, dict):
                    raise RequestError("chat_template_kwargs must be an object")
                kwargs = dict(kwargs)
                # the production middleware's contract (glm53_chat.py): thinking/enable_thinking agree, and the
                # top-level reasoning_effort reaches the template (which otherwise defaults to max)
                if "thinking" in kwargs and "enable_thinking" in kwargs and kwargs["thinking"] != kwargs["enable_thinking"]:
                    raise RequestError("thinking and enable_thinking must agree")
                if "enable_thinking" in kwargs and "thinking" not in kwargs:
                    kwargs["thinking"] = kwargs["enable_thinking"]
                effort = req.get("reasoning_effort")
                if effort is None and "reasoning_effort" in kwargs:
                    effort = kwargs["reasoning_effort"]       # a caller that only spoke to the template
                if effort is not None:
                    if effort not in EFFORT_RUNGS:
                        raise RequestError("reasoning_effort must be low, medium, high, or max")
                    if kwargs.get("reasoning_effort", effort) != effort:
                        raise RequestError("top-level and template reasoning_effort must agree")
                    kwargs["reasoning_effort"] = EFFORT_RUNGS[effort]
                server.note_reasoning(kwargs)             # the shape the template will render (45차 §81)
                options_stream = req.get("stream_options")
                if options_stream is not None and not isinstance(options_stream, dict):
                    raise RequestError("stream_options must be an object")
                n = req.get("n", 1)
                if n is None:
                    n = 1
                if type(n) is not int or not 1 <= n <= server.max_choices:
                    raise RequestError(f"n must be an integer between 1 and {server.max_choices}")
                want_logprobs = None
                if req.get("logprobs"):
                    top = req.get("top_logprobs", 0) or 0
                    if type(top) is not int or not 0 <= top <= 20:
                        raise RequestError("top_logprobs must be an integer between 0 and 20")
                    want_logprobs = top
                stop = stop_strings(req)
                tools = req.get("tools")
                choice = req.get("tool_choice")
                if choice == "none":
                    tools = None
                elif choice not in (None, "auto"):
                    raise RequestError("tool_choice: only auto and none are served (required/named calls are not enforced)")
                if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
                    raise RequestError("tools must be a list of objects")
                min_tokens = req.get("min_tokens", 0) or 0
                if type(min_tokens) is not int or min_tokens < 0:
                    raise RequestError("min_tokens must be a nonnegative integer")
                max_tokens = req.get("max_tokens")
                if max_tokens is None:
                    max_tokens = req.get("max_completion_tokens")
                defaulted = max_tokens is None
                if defaulted:
                    max_tokens = answer_budget(server.tok, nfc(written_text(messages)))
                stream = bool(req.get("stream", False))
                include_usage = bool(options_stream and options_stream.get("include_usage"))
                model = server.check_model(req.get("model"))
                temperature, options = sampling_options(req, server.generation)
                derived = stop_token_ids_for(stop, server.tok) if (stop and server.tok is not None) else []
                if derived:
                    options["stop_token_ids"] = sorted(set(options.get("stop_token_ids") or []) | set(derived))
                if want_logprobs is not None:
                    options["logprobs"] = want_logprobs
                grammar = response_format_grammar(req)
                if grammar is not None:
                    options["grammar"] = grammar
                elif tools and server.tool_grammar is not None and server.tool_call_start is not None:
                    # Nothing held a tool call to the tools that were declared: a call could name a
                    # tool nobody offered, or an argument it does not take, and the caller would be
                    # handed something it cannot make. The grammar arms at `<tool_call>` and not
                    # before, so the answer's prose is free -- llama.cpp's lazy trigger, and our
                    # `grammar_after` is exactly that (45차 §45).
                    ebnf = server.tool_grammar(tools)
                    if ebnf is not None:
                        options["grammar"] = {"type": "ebnf", "grammar": ebnf}
                        options["grammar_after"] = server.tool_call_start
                parts = media_parts(messages)                            # (kind, url) in the order the template will emit them
                items = []
                if parts:
                    if server.vision is None:
                        raise RequestError("images and videos are not served by this deployment")
                    counts = {}
                    for kind, url in parts:
                        counts[kind] = counts.get(kind, 0) + 1
                        limit = server.vision.limits.get(kind, 0)
                        if counts[kind] > limit:
                            raise RequestError(f"at most {limit} {kind}(s) per request are served" if limit else f"{kind} is not served")
                    for kind, url in parts:
                        try:
                            items.append(server.vision.prepare(kind, fetch_media(kind, url)))
                        except ValueError as exc:
                            raise RequestError(f"{kind}: {exc}") from exc
                try:
                    template_start = time.perf_counter()
                    opening, resuming = prompt_switches(req)
                    prompt = server.chat(messages, dict(kwargs, tools=tools) if tools else kwargs,
                                         generation_prompt=opening, continue_final=resuming)
                except Exception as exc:                                  # noqa: BLE001 -- the template's verdict on these messages
                    raise RequestError(f"chat template rejected the request: {exc}") from exc
                template_us = (time.perf_counter() - template_start) * 1e6
                tokenize_start = time.perf_counter()
                ids = server.prompt_tokens.encode(nfc(prompt))   # a continuing turn pays for its tail only
                tokenize_us = (time.perf_counter() - tokenize_start) * 1e6
                media = None
                if items:
                    try:
                        ids, media = server.vision.expand(ids, items)   # one placeholder per part -> the runs the model sees
                    except ValueError as exc:
                        raise RequestError(str(exc)) from exc
                # thinking off: the template already closed the think block (the rendered prompt ends with the reasoning-end
                # token), so everything generated is content -- otherwise a whole answer lands in reasoning_content
                # (45차 §22: the gateway's -low route asks thinkingMode off and reads content)
                reasoning = server.reasoning_end is not None and not (ids and ids[-1] == server.reasoning_end)
                if reasoning and "grammar" in options and "grammar_after" not in options:
                    # The answer starts inside a think block, and a grammar that started here would forbid the
                    # reasoning -- including the block's own end token, so the block would never close and the
                    # whole answer would come back as reasoning_content with content empty. It waits instead.
                    options["grammar_after"] = server.reasoning_end
                if defaulted:
                    # Only ever back down to what the old token budget was: a prompt the old
                    # default could not fit is still refused, in the same words, rather than
                    # quietly answered in one token.
                    max_tokens = min(max_tokens, max(DEFAULT_ANSWER_TOKENS[0], server.room_for(len(ids))))
                if reasoning:
                    budget = reasoning_budget(req, max_tokens)
                    if budget is not None:
                        options["reasoning_budget"] = budget
                        options["reasoning_end"] = server.reasoning_end
                choices = self.choices_for(ids, n, max_tokens, temperature, options, stop, reasoning=reasoning,
                                           tool_parser=server.tool_parser, want_logprobs=want_logprobs, min_new=min_tokens,
                                           continue_history=True, media=media, cache_salt=cache_key(req))
                head = {"id": f"chatcmpl-{choices[0].request}", "created": int(time.time()), "model": model}
                for c in choices:
                    server.latency.row(kind='request', operation='template', phase='http',
                        request_id=c.request, duration_us=template_us)
                    server.latency.row(kind='request', operation='tokenize', phase='http',
                        request_id=c.request, duration_us=tokenize_us)

                def chunk(index, delta=None, finish=None, usage=None, logprobs=None):
                    payload = {**head, "object": "chat.completion.chunk",
                               "choices": [] if usage is not None else
                               [{"index": index, "delta": delta or {}, "finish_reason": finish,
                                 **({"logprobs": logprobs} if logprobs is not None else {})}],
                               **({"usage": usage} if usage is not None else {})}
                    self.sse(payload)

                try:
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for c in choices:
                            chunk(c.index, {"role": "assistant", "content": ""})

                    def on_delta(c, deltas):
                        if stream:
                            for d in deltas:
                                chunk(c.index, d)

                    if not self.run_choices(choices, on_delta):
                        return
                    errors = [c.error for c in choices if c.error]
                    if errors:
                        if stream:
                            self.sse({"error": {"message": errors[0], "type": "engine"}})
                        else:
                            self.reply(503, {"error": errors[0]})
                        return
                    usage = {"prompt_tokens": len(ids), "completion_tokens": sum(c.total for c in choices),
                             "total_tokens": len(ids) + sum(c.total for c in choices),
                             "prompt_tokens_details": {"cached_tokens": server.cached_tokens(*(c.request for c in choices))},
                             "completion_tokens_details": {"reasoning_tokens": sum(len(c.streams["reasoning_content"].ids) for c in choices)}}
                    if stream:
                        for c in choices:
                            chunk(c.index, None, finish=c.finish_reason(), logprobs=c.logprobs_payload())
                        if include_usage:
                            chunk(0, usage=usage)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        out = []
                        for c in choices:
                            message = {"role": "assistant", "content": c.text["content"] or None}
                            if c.streams["reasoning_content"].ids:
                                message["reasoning_content"] = c.text["reasoning_content"]
                            calls = c.tool_calls_done()
                            if calls:
                                message["tool_calls"] = calls
                            entry = {"index": c.index, "message": message, "finish_reason": c.finish_reason()}
                            if c.want_logprobs is not None:
                                entry["logprobs"] = c.logprobs_payload()
                            out.append(entry)
                        self.reply(200, {**head, "object": "chat.completion", "choices": out, "usage": usage})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    for c in choices:
                        server.cancel(c.request, "client closed")
                finally:
                    self.release(choices)

            def completions(self, req):
                """OpenAI (legacy) completions: prompt text or ids, n choices, echo, logprobs; no template, no reasoning split."""
                if server.tok is None:
                    raise RequestError("no tokenizer: use /v1/engine/completions with ids")
                prompt = req.get("prompt", "")
                if isinstance(prompt, str):
                    prompts = [server.tok.encode(nfc(prompt), add_special_tokens=True).ids]
                elif isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt):
                    prompts = [list(prompt)]
                elif isinstance(prompt, list) and prompt and all(isinstance(p, str) for p in prompt):
                    prompts = [server.tok.encode(nfc(p), add_special_tokens=True).ids for p in prompt]
                elif isinstance(prompt, list) and prompt and all(isinstance(p, list) and p and all(type(t) is int for t in p) for p in prompt):
                    prompts = [list(p) for p in prompt]
                else:
                    raise RequestError("prompt must be a string, a list of strings, token ids, or lists of token ids")
                if any(not p for p in prompts):
                    raise RequestError("prompt must not be empty")
                n = req.get("n", 1) or 1
                if type(n) is not int or not 1 <= n <= server.max_choices:
                    raise RequestError(f"n must be an integer between 1 and {server.max_choices}")
                best_of = req.get("best_of")
                if best_of is not None and (type(best_of) is not int or best_of < n or best_of > server.max_choices):
                    raise RequestError(f"best_of must be an integer between n and {server.max_choices}")
                if req.get("suffix"):
                    raise RequestError("suffix (insertion) is not served")
                want_logprobs = req.get("logprobs")
                if want_logprobs is not None and (type(want_logprobs) is not int or not 0 <= want_logprobs <= 20):
                    raise RequestError("logprobs must be an integer between 0 and 20")
                echo = bool(req.get("echo", False))
                stop = stop_strings(req)
                max_tokens = req.get("max_tokens", 16)
                stream = bool(req.get("stream", False))
                options_stream = req.get("stream_options")
                include_usage = bool(isinstance(options_stream, dict) and options_stream.get("include_usage"))
                model = server.check_model(req.get("model"))
                temperature, options = sampling_options(req, server.generation)
                derived = stop_token_ids_for(stop, server.tok) if (stop and server.tok is not None) else []
                if derived:
                    options["stop_token_ids"] = sorted(set(options.get("stop_token_ids") or []) | set(derived))
                count = best_of or n
                if want_logprobs is not None or best_of:
                    options["logprobs"] = want_logprobs if want_logprobs is not None else 0
                lp_want = want_logprobs if want_logprobs is not None else (0 if best_of else None)
                choices, prompt_of = [], {}
                for ids in prompts:
                    group = self.choices_for(ids, count, max_tokens, temperature, options, stop, reasoning=False,
                                             want_logprobs=lp_want, cache_salt=cache_key(req))
                    for c in group:
                        c.index = len(choices)
                        prompt_of[c.request] = ids
                        choices.append(c)
                head = {"id": f"cmpl-{choices[0].request}", "created": int(time.time()), "model": model}

                def legacy_logprobs(c, ids_prompt):
                    if want_logprobs is None:
                        return None
                    start = len(server.tok.decode(ids_prompt)) if echo else 0
                    tokens, offsets = token_spans(server.tok, [tid for tid, _, _ in c.logprobs], start)
                    lps = [lp for _, lp, _ in c.logprobs]
                    def top_map(top):
                        """OpenAI's legacy shape is a dict keyed by the token's text, and in Korean
                        several of a position's candidates are halves of a character -- every one of
                        them U+FFFD. A dict comprehension would keep the last, which is the least
                        likely of them; keep the likeliest instead."""
                        out = {}
                        for i, v in top[:want_logprobs]:
                            text = server.tok.decode([i])
                            if v > out.get(text, float("-inf")):
                                out[text] = v
                        return out

                    tops = [top_map(top) if want_logprobs else None for _, _, top in c.logprobs]
                    return {"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops, "text_offset": offsets}

                def chunk(index, text=None, finish=None, usage=None):
                    self.sse({**head, "object": "text_completion",
                              "choices": [] if usage is not None else [{"index": index, "text": text or "", "logprobs": None, "finish_reason": finish}],
                              **({"usage": usage} if usage is not None else {})})

                try:
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        if echo:
                            for c in choices:
                                chunk(c.index, server.tok.decode(prompt_of[c.request]))

                    def on_delta(c, deltas):
                        if stream:
                            for d in deltas:
                                if "content" in d:
                                    chunk(c.index, d["content"])

                    if not self.run_choices(choices, on_delta):
                        return
                    errors = [c.error for c in choices if c.error]
                    if errors:
                        if stream:
                            self.sse({"error": {"message": errors[0], "type": "engine"}})
                        else:
                            self.reply(503, {"error": errors[0]})
                        return
                    kept = choices
                    if best_of and best_of > n:                           # the n best of best_of by mean token log-probability
                        by_prompt = {}
                        for c in choices:
                            by_prompt.setdefault(c.request in prompt_of and tuple(prompt_of[c.request]), []).append(c)
                        kept = []
                        for group in by_prompt.values():
                            group.sort(key=lambda c: -(sum(lp for _, lp, _ in c.logprobs) / max(1, len(c.logprobs))))
                            kept += group[:n]
                        for i, c in enumerate(kept):
                            c.index = i
                    usage = {"prompt_tokens": sum(len(p) for p in prompts), "completion_tokens": sum(c.total for c in choices),
                             "total_tokens": sum(len(p) for p in prompts) + sum(c.total for c in choices),
                             "prompt_tokens_details": {"cached_tokens": server.cached_tokens(*(c.request for c in choices))}}
                    if stream:
                        for c in kept:
                            chunk(c.index, None, finish=c.finish or "length")
                        if include_usage:
                            chunk(0, usage=usage)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        out = []
                        for c in kept:
                            text = (server.tok.decode(prompt_of[c.request]) if echo else "") + c.text["content"]
                            out.append({"index": c.index, "text": text, "logprobs": legacy_logprobs(c, prompt_of[c.request]),
                                        "finish_reason": c.finish or "length"})
                        self.reply(200, {**head, "object": "text_completion", "choices": out, "usage": usage})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    for c in choices:
                        server.cancel(c.request, "client closed")
                finally:
                    self.release(choices)

            def tokenize(self, req):
                if server.tok is None:
                    raise RequestError("no tokenizer", 404)
                if "messages" in req:
                    if server.chat is None:
                        raise RequestError("this server has no chat template", 404)
                    kwargs = req.get("chat_template_kwargs") or {}
                    tools = req.get("tools")
                    try:
                        opening, resuming = prompt_switches(req)
                        prompt = server.chat(req["messages"], dict(kwargs, tools=tools) if tools else dict(kwargs),
                                             generation_prompt=opening, continue_final=resuming)
                    except Exception as exc:                              # noqa: BLE001
                        raise RequestError(f"chat template rejected the request: {exc}") from exc
                    add_special = bool(req.get("add_special_tokens", False))
                else:
                    prompt = req.get("prompt")
                    if not isinstance(prompt, str):
                        raise RequestError("prompt must be text")
                    add_special = bool(req.get("add_special_tokens", True))
                # Composed, because that is what generation will tokenise -- but this endpoint is
                # asked "how many tokens is THIS string", so when the answer is about a different
                # one it says so and hands back the string the ids belong to.
                composed = nfc(prompt)
                ids = server.tok.encode(composed, add_special_tokens=add_special).ids
                out = {"count": len(ids), "max_model_len": server.max_context, "tokens": ids}
                if composed != prompt:
                    out["normalized"] = "NFC"
                    out["prompt"] = composed
                self.reply(200, out)

            def detokenize(self, req):
                if server.tok is None:
                    raise RequestError("no tokenizer", 404)
                tokens = req.get("tokens")
                if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                    raise RequestError("tokens must be a list of token ids")
                self.reply(200, {"prompt": server.tok.decode(tokens)})

            def engine_completions(self, req):
                """The engine's own dialect: ids or a raw prompt, `conversation` continues a retained one."""
                ids = req.get("ids")
                if ids is None:
                    if server.tok is None:
                        raise RequestError("no tokenizer: send ids")
                    prompt = req.get("prompt", "")
                    if not isinstance(prompt, str):
                        raise RequestError("prompt must be text")
                    ids = server.tok.encode(nfc(prompt)).ids
                t0 = time.perf_counter()
                conversation = req.get("conversation")
                temperature, options = sampling_options(req, {})
                request, event = server.submit(ids, req.get("max_tokens", 64), temperature, conversation, options=options,
                                               cache_salt=cache_key(req))
                out = self.wait_result(request, event)
                text = server.tok.decode(out) if server.tok is not None else None
                conversation = (request if conversation is None else conversation) if server.runner.keep_idle else None
                self.reply(200, {"seq": request, "conversation": conversation, "ids": out, "text": text, "prompt_tokens": len(ids),
                                 "cached_tokens": server.cached_tokens(request),
                                 "completion_tokens": len(out), "seconds": round(time.perf_counter() - t0, 3)})

            def prefix_warm(self, req):
                """Compute a prompt's prefix so its boundaries are cached before anyone asks (45차 §23 C): `messages` (through
                the chat template) or `prompt` / `ids`; one token is generated and discarded. `pin: true` keeps the
                boundaries out of eviction until `/v1/prefix/unpin`. `cache_salt` warms that tenant's own chain."""
                if getattr(server.runner, "prefix", None) is None:
                    raise RequestError("this server has no prefix cache", 404)
                if req.get("messages") is not None:
                    if server.chat is None or server.tok is None:
                        raise RequestError("this server has no chat template", 404)
                    kwargs = req.get("chat_template_kwargs") or {}
                    if not isinstance(kwargs, dict):
                        raise RequestError("chat_template_kwargs must be an object")
                    try:
                        prompt = server.chat(req["messages"], dict(kwargs))
                    except Exception as exc:                          # noqa: BLE001
                        raise RequestError(f"chat template rejected the request: {exc}") from exc
                    ids = server.tok.encode(nfc(prompt), add_special_tokens=False).ids
                elif isinstance(req.get("prompt"), str):
                    if server.tok is None:
                        raise RequestError("this server has no tokenizer", 404)
                    ids = server.tok.encode(nfc(req["prompt"]), add_special_tokens=False).ids
                elif isinstance(req.get("ids"), list):
                    ids = req["ids"]
                else:
                    raise RequestError("warm needs messages, a prompt or ids")
                request, event = server.submit(ids, 1, 0.0, cache_salt=cache_key(req))
                if not event.wait(server.request_timeout_s):
                    server.cancel(request, "timeout")
                    raise RequestError("warm timed out", 504)
                server.take_result(request)
                prefix = server.runner.prefix
                salt = cache_key(req)
                chain = prefix.chain(ids, [prefix_cache.tenant_salt(salt)] if salt else ())
                cached = sorted(t for t, h in chain.items() if prefix.has(h))
                if req.get("pin"):
                    server.controls.put(("pin", [chain[t].hex() for t in cached]))
                self.reply(200, {"tokens": len(ids), "boundaries": cached, "pinned": bool(req.get("pin"))})

            def prefix_unpin(self, req):
                if getattr(server.runner, "prefix", None) is None:
                    raise RequestError("this server has no prefix cache", 404)
                server.controls.put(("unpin", None))
                self.reply(200, {"ok": True})

            def prefix_reset(self, req):
                """Throw the whole prefix cache away (vLLM's `/reset_prefix_cache`).

                For the case nothing in the engine can see: the prompt's MEANING changed under an
                unchanged prefix -- a tool list, a retrieved document, an edited system template --
                so the token ids still hash the same and every boundary still matches. `unpin` only
                releases an operator's pin; this forgets the boundaries themselves, on every rank in
                the same step, and deletes their copies off the prefix tier.
                """
                if getattr(server.runner, "prefix", None) is None:
                    raise RequestError("this server has no prefix cache", 404)
                server.controls.put(("reset", None))
                self.reply(200, {"ok": True})

            def calibration(self, req):
                """File this boot's calibration sums now (kernels/dense/calibration): between two steps, on every rank's
                loop thread. A boot whose packs were all calibrated has nothing to file."""
                calibration = getattr(server.engine, "calibration", None)
                if calibration is None:
                    raise RequestError("this boot is not calibrating: every pack it serves was already calibrated", 404)
                server.controls.put(("calibration", req.get("root") if isinstance(req.get("root"), str) else None))
                self.reply(200, {"ok": True, "status": calibration.status(), "rows": {k: float(v) for k, v in calibration.rows.items()}})

            def profile(self, req):
                """Profile the next few decode steps and say what their kernels were.

                A decode step replays a captured graph, so nothing inside it can be timed with CUDA events --
                the stage clock can only wrap the replay whole (base/stage_clock). CUPTI sees through it. The
                answer is this rank's; the ranks are symmetric, and rank 0 is the one the door speaks for."""
                steps = req.get("steps", 8)
                if not isinstance(steps, int) or not 1 <= steps <= server.PROFILE_MAX_STEPS:
                    raise RequestError(f"steps must be 1..{server.PROFILE_MAX_STEPS} decode steps", 400)
                server.controls.put(("profile", steps))
                self.reply(202, {"ok": True, "steps": steps,
                                 "read": "GET /v1/engine/profile once those steps have run"})

            def latency_control(self, req):
                import ipaddress
                if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                    raise RequestError('latency controls require a loopback client', 403)
                ident = uuid.uuid4().hex
                waiting = {'event': threading.Event()}
                server.latency_replies[ident] = waiting
                server.controls.put(('latency', dict(req, _control_id=ident)))
                try:
                    if not waiting['event'].wait(60):
                        raise RequestError('latency control timed out; inspect server recording before retrying', 504)
                    result = waiting['reply']
                    self.reply(409 if any(r.get('error') for r in result['ranks']) else 200, result)
                finally:
                    server.latency_replies.pop(ident, None)

            def do_POST(self):
                try:
                    active = server.latency.active
                    if active and self.path != '/v1/engine/latency' and self.headers.get('X-ST-Latency-Token') != active['token']:
                        raise RequestError('server reserved by a latency recording', 409)
                    routes = {"/v1/chat/completions": self.chat, "/v1/completions": self.completions,
                              "/v1/engine/completions": self.engine_completions, "/tokenize": self.tokenize,
                              "/detokenize": self.detokenize, "/v1/prefix/warm": self.prefix_warm, "/v1/prefix/unpin": self.prefix_unpin,
                              "/v1/prefix/reset": self.prefix_reset,
                              "/v1/engine/calibration": self.calibration, "/v1/engine/profile": self.profile,
                              "/v1/engine/latency": self.latency_control}
                    handler = routes.get(self.path)
                    if handler is None:
                        raise RequestError("unknown endpoint", 404)
                    handler(self.body())
                except RequestError as exc:
                    self.reply(exc.status, {"error": str(exc)})
                except (ValueError, TypeError, UnicodeError) as exc:
                    self.reply(400, {"error": str(exc)})

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
        return httpd

    def loop(self, idle_sleep: float = 0.002):
        httpd = self._serve_http() if self.comm.rank == 0 else None
        try:
            while True:                              # rank 0 broadcasts the stop before leaving
                ran = self.once()
                if not self.alive:
                    break
                if not ran:
                    time.sleep(idle_sleep)
        finally:
            self._fail_pending()
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()
