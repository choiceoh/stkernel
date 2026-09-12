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
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine.base.kv_tier import TierFull

MEDIA_FETCH_TIMEOUT_S = {"image": 5.0, "video": 30.0}           # vLLM's VLLM_IMAGE_FETCH_TIMEOUT / VLLM_VIDEO_FETCH_TIMEOUT defaults
MEDIA_MAX_BYTES = {"image": 64 << 20, "video": 512 << 20}        # a door-side ceiling on what one part may carry (vLLM has none)


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_SAMPLING_RANGES = {"presence_penalty": (-2.0, 2.0), "frequency_penalty": (-2.0, 2.0)}


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


def stop_strings(req: dict) -> "list[str]":
    stop = req.get("stop")
    stop = [stop] if isinstance(stop, str) else (stop or [])
    if not isinstance(stop, list) or any(not isinstance(x, str) or not x for x in stop):
        raise RequestError("stop must be a nonempty string or a list of them")
    return stop


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
            return {"type": "json_schema", "schema": json.dumps(schema, sort_keys=True)}
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
        for cut in range(min(len(needle) - 1, len(text)), keep, -1):
            if text.endswith(needle[:cut]):
                keep = cut
                break
    return keep


class _Choice:
    """One generation inside an OpenAI response: its request, its token queue and the per-channel text it has
    shown so far. Chat answers split at the reasoning-end token into reasoning_content / content; a content that
    reaches a stop string ends there; complete <tool_call> blocks become tool_calls (streamed as they complete)."""

    def __init__(self, index: int, request: int, event, q, *, tok, stop, reasoning: bool, tool_parser=None,
                 want_logprobs: "int | None" = None):
        self.index, self.request, self.event, self.q = index, request, event, q
        self.tok, self.stop, self.reasoning, self.tool_parser = tok, list(stop), reasoning, tool_parser
        self.want_logprobs = want_logprobs
        self.held = {"reasoning_content": [], "content": []}
        self.shown = {"reasoning_content": 0, "content": 0}
        self.text = {"reasoning_content": "", "content": ""}
        self.acc = {"reasoning_content": "", "content": ""}      # decoded so far, grown a window at a time
        self.window = {"reasoning_content": [0, 0], "content": [0, 0]}   # into `held`: shown up to, decoded up to
        self.logprobs = []                           # per generated token: (id, logprob, [(id, logprob), ...])
        self.total = 0
        self.finish = None
        self.done = False
        self.error = None
        self.tool_calls = []
        self._tool_seen = 0

    def feed(self, tokens, logprobs, reasoning_end) -> None:
        for i, t in enumerate(tokens):
            self.total += 1
            if logprobs is not None and i < len(logprobs):
                self.logprobs.append(logprobs[i])
            if self.reasoning and t == reasoning_end:
                self.reasoning = False
                continue
            self.held["reasoning_content" if self.reasoning else "content"].append(t)

    def _decoded(self, channel: str, final: bool) -> str:
        """This channel's text, extended by whatever the newest tokens added.

        Neither obvious way works. Decoding one token alone is wrong -- a token's
        rendering depends on its neighbours, because a character can be spread over
        several byte pieces and because a piece carries its leading space only when
        something precedes it. Decoding the whole answer again for every token is
        right but quadratic: measured on this checkpoint's tokenizer, 3.06 s of CPU
        for a 4,096-token answer against 0.011 s for this. So decode a two-token
        window and subtract the part of it already accounted for, which is what
        vLLM's prefix and read offsets do.
        """
        ids = self.held[channel]
        prefix, read = self.window[channel]
        if read < len(ids):
            before = self.tok.decode(ids[prefix:read]) if read > prefix else ""
            grown = self.tok.decode(ids[prefix:])
            # a trailing replacement character is half a code point: wait for the rest,
            # unless nothing more is coming
            if len(grown) > len(before) and (final or not grown.endswith("\ufffd")):
                self.acc[channel] += grown[len(before):]
                self.window[channel] = [read, len(ids)]
        return self.acc[channel]

    def flush(self, final: bool = False) -> "list[dict]":
        """Decode each channel; what is new becomes a delta. Returns the deltas in order."""
        deltas = []
        for channel in self.held:
            decoded = self._decoded(channel, final)
            if channel == "content":
                if self.stop:
                    cut = min((decoded.find(x) for x in self.stop if x in decoded), default=-1)
                    if cut >= 0:
                        decoded = decoded[:cut]
                        self.finish = "stop"
                    elif not final:                                 # a tail that could still become one waits
                        held_back = partial_suffix(decoded, self.stop)
                        decoded = decoded[:len(decoded) - held_back] if held_back else decoded
                if self.tool_parser is not None:
                    if not final:                                   # a partial "<tool_call>" prefix waits for the rest
                        held_back = partial_suffix(decoded, ("<tool_call>",))
                        decoded = decoded[:len(decoded) - held_back] if held_back else decoded
                    start = decoded.find("<tool_call>")
                    if start >= 0:
                        blocks = _TOOL_CALL.findall(decoded)
                        for body in blocks[self._tool_seen:]:
                            for name, args in (self.tool_parser(f"<tool_call>{body}</tool_call>") or []):
                                i = len(self.tool_calls)
                                call = {"index": i, "id": f"call_{self.request}_{i}", "type": "function",
                                        "function": {"name": name, "arguments": args}}
                                self.tool_calls.append(call)
                                deltas.append({"tool_calls": [call]})
                        self._tool_seen = len(blocks)
                        decoded = decoded[:start]
            delta = decoded[self.shown[channel]:]
            if delta:
                deltas.append({channel: delta})
                self.shown[channel] = len(decoded)
            self.text[channel] = decoded[:self.shown[channel]]
        return deltas

    def finish_reason(self) -> str:
        if self.tool_calls:
            return "tool_calls"
        return self.finish or "length"

    def logprobs_payload(self, offset: int = 0) -> "dict | None":
        """OpenAI chat `logprobs.content`: the shown tokens' log-probabilities (content channel, chosen + top)."""
        if self.want_logprobs is None:
            return None
        rows = []
        for tid, lp, top in self.logprobs[offset:]:
            token = self.tok.decode([tid])
            rows.append({"token": token, "logprob": lp, "bytes": list(token.encode()),
                         "top_logprobs": [{"token": self.tok.decode([i]), "logprob": v, "bytes": list(self.tok.decode([i]).encode())}
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
                 generation: "dict | None" = None, max_choices: int = 4, vision=None):
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if type(request_timeout_s) not in (int, float) or not request_timeout_s > 0:
            raise ValueError("request_timeout_s must be positive")
        if runner.slot_of:
            raise ValueError("the server requires an idle runner")
        if reasoning_end is not None and (type(reasoning_end) is not int or reasoning_end < 0):
            raise ValueError("reasoning_end must be a token id")
        self.engine, self.runner, self.comm = engine, runner, comm
        self.port, self.host, self.tok = port, host, tokenizer
        self.chat, self.model_name, self.reasoning_end = chat, model_name, reasoning_end
        self.tool_parser = tool_parser             # text -> [(name, arguments json)] or None (the profile knows the model's format)
        self.vision = vision                       # the profile's door half for pictures (prepare / expand / limits), or None: text only
        self.generation = dict(generation or {})   # the checkpoint's generation_config defaults (temperature ...) a request may omit
        self.max_choices = int(max_choices)        # n / best_of ceiling: one row each, never more than the decode width
        self._stop_ids = {}                        # request id -> stop_token_ids: an end by one of them is finish_reason "stop"
        self.max_pending = max_pending
        self.max_context = int(getattr(engine, "max_context", 2**31 - 1))   # the model's trained positions; the door refuses beyond
        self.request_timeout_s = float(request_timeout_s)
        self.clock = time.monotonic                # injectable for tests
        self._cancels = set()                      # (request id, reason) asked by HTTP threads / the timeout scan; rank 0 broadcasts them
        self._deadline = {}                        # request id -> clock() by which it must have finished (rank 0)
        self.cancelled = 0
        self.timed_out = 0                         # the subset of `cancelled` the deadline scan took
        self.steps_prefill = self.steps_decode = 0   # D9: a step is one kind or the other, never both
        # Latency is owed from the request's arrival, not from the step that served it.
        # Rank 0 admits, answers and serves /metrics, so only rank 0 keeps these.
        self._arrived = {}                         # request id -> clock() at admission
        self._token_at = {}                        # row -> clock() of its last delivered token
        self.ttft = _Histogram(_TTFT_BOUNDS)
        self.step_seconds = {"prefill": _Histogram(_STEP_BOUNDS), "decode": _Histogram(_STEP_BOUNDS)}
        self._step_began = None                    # clock() at the top of the step being timed
        self.itl = _Histogram(_ITL_BOUNDS)
        self.e2e = _Histogram(_E2E_BOUNDS)
        self._streams = {}                         # request id -> queue of ("tokens", ids) | ("end", finish) | ("error", text)
        self._wake = threading.Event()             # set whenever a stream queue gains an item, so a drain can wait
                                                   # on it instead of polling; a 20 ms poll put 20 ms of jitter on
                                                   # every streamed token, a fifth of this engine's inter-token time
        self._sent = {}                            # row -> generated tokens already handed to its stream
        self.prompt_tokens_total = self.generation_tokens_total = 0
        self.arrivals = queue.Queue()
        self.pending, self.results = {}, {}
        # conversation ids are request ids; parked conversations from an earlier boot keep theirs
        parked = sorted(runner.parked_keys())
        self._agree_on_parked(comm, parked)
        self.next_seq, self.served = 1 + max(parked, default=-1), 0
        self.alive = True
        self._lock = threading.Lock()
        self._waiting = deque()                    # request id, tokens, limit, temperature, promised blocks
        self._active = {}                          # reusable row -> (request id, promised blocks)
        self._conversations, self._conversation_of = {}, {}   # resident (idle or live) conversations <-> rows
        self._idle_order = {}                      # resident idle rows, least recently completed turn first (no tier)
        self._retiring = {}                        # row -> conversation: its park is on the tier's thread (D10: no step waits on it)
        self._resuming = {}                        # row -> (conversation, request, ids, limit, temperature, promised): its resume is in flight
        self._free_rows = list(range(min(runner.kv.max_seqs, runner.c.max_running, runner.slots.available)))
        if not self._free_rows:
            raise ValueError("the server needs at least one request row and state slot")

    def submit(self, ids, max_new: int, temperature: float, conversation: "int | None" = None, stream: bool = False,
               min_new: int = 0, options: "dict | None" = None, continue_history: bool = False, media=None):
        """Validate and enqueue on rank 0 without acquiring any model resources.
        `stream`: the request also gets a token queue (see `_streams`). `min_new`: no end token before this many.
        `options`: the request's sampling/behaviour options beyond temperature (the engine validates them).
        `continue_history`: the OpenAI path re-sends a whole chat every turn -- when a retained conversation's history
        (prompt + what it generated) is a proper prefix of `ids`, continue it with the new suffix instead of
        prefilling everything again (45차 §23 B1). The hint is taken here, on rank 0; admission re-checks it and
        falls back to a fresh prompt if the conversation left in between.
        `media`: the pictures standing at placeholder runs inside `ids` (the profile's door built them: kind, digest,
        positions, canvas, grid); they ride to every rank with the request and are encoded there (45차 §23 A7)."""
        if self.comm.rank != 0:
            raise RequestError("requests must enter on rank 0")
        options = dict(options or {})
        if options and hasattr(self.engine, "validate_options"):
            try:
                self.engine.validate_options(options)
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
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
            hint = self._continuation(ids, media)
        if conversation is not None:
            if type(conversation) is not int or conversation < 0:
                raise RequestError("conversation must be a nonnegative integer")
            if not self.runner.keep_idle:
                raise RequestError("this server does not retain conversations", 409)
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
        if horizon >= 2**31 or blocks > min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq):
            raise RequestError("prompt and generation limit exceed the KV capacity")
        if horizon > self.max_context:
            raise RequestError("prompt and generation limit exceed the model's context")
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
            self.arrivals.put((request, list(ids), max_new, float(temperature), blocks, conversation, min_new, options, hint, media))
        return request, event

    def _continuation(self, ids, media=()) -> "tuple[int, int] | None":
        """(conversation, prefix length) of the retained conversation whose history is the longest proper prefix of
        `ids`: a resident idle row, or a parked one (its record carries the tokens). None if nothing matches.
        The pictures must match too: the same placeholder run with another picture is another prompt."""
        best = None
        n = len(ids)
        marks = sorted((m["positions"][0], m["digest"]) for m in media)
        def consider(key, history, history_marks):
            nonlocal best
            m = len(history)
            if (0 < m < n and (best is None or m > best[1]) and ids[:m] == list(history)
                    and [(p, d) for p, d in marks if p < m] == sorted((int(p), str(d)) for p, d in history_marks)):
                best = (key, m)
        for row in list(self._idle_order):
            key = self._conversation_of.get(row)
            if key is not None and hasattr(self.engine, "history"):
                consider(key, self.engine.history(row),
                         self.engine.media_marks(row) if hasattr(self.engine, "media_marks") else [])
        for key in self.runner.parked_keys():
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
                    self._answer(request, RequestError(f"request cancelled: {reason}", 504 if reason == "timeout" else 499))
                    return
                self._deadline.pop(request, None)
                self._arrived.pop(request, None)
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

    def _admit(self):
        while self._waiting:
            request, ids, limit, temperature, promised, conversation, min_new, options, hint, media = self._waiting[0]
            row = None
            resident = held = 0
            parked = False
            if conversation is None and hint is not None:
                key, prefix = hint
                row_ = self._conversations.get(key)
                rest = self._media_after(media, prefix)
                if rest is None:
                    self._waiting[0] = (request, ids, limit, temperature, promised, None, min_new, options, None, media)   # a picture straddles the cut
                    continue
                if (row_ is not None and row_ in self.runner.idle) or (row_ is None and self.runner.is_parked(key)):
                    conversation, ids, media = key, ids[prefix:], rest     # continue the retained conversation with the new turn
                elif row_ is not None or key in self._retiring.values() or any(e["conversation"] == key for e in self._resuming.values()):
                    break                                         # it is mid-park/resume or live: decide next step
                else:
                    self._waiting[0] = (request, ids, limit, temperature, promised, None, min_new, options, None, media)   # gone: fresh prompt
                    continue
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
                if horizon >= 2**31 or promised > min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq):
                    self._waiting.popleft()
                    self._answer(request, RequestError("conversation and generation limit exceed the KV capacity"))
                    continue
                if horizon > self.max_context:
                    self._waiting.popleft()
                    self._answer(request, RequestError("conversation and generation limit exceed the model's context"))
                    continue
                promised = max(promised, held)       # rejected-draft reservations may exceed the new turn
            if row is None and not self._free_rows:
                if self._evict_idle():
                    continue
                break
            # Future decode growth already belongs to admitted requests even
            # though the block pool acquires those blocks only when written.
            future = (sum(b - self.runner.kv.blocks_for(self.runner.kv.tokens[r]) for r, (_, b) in self._active.items())
                      + sum(e["promised"] - self.runner.kv.blocks_for(self.runner.kv.tokens[r]) for r, e in self._resuming.items()))
            if promised > self.runner.kv.available - future + resident:
                if self._evict_idle(exclude=row):
                    continue
                break
            if conversation is None:
                row = heapq.heappop(self._free_rows)
                try:
                    self.engine.add(row, ids, max_new=limit, temperature=temperature, **({"min_new": min_new} if min_new else {}),
                                    **({"options": options} if options else {}), **({"media": media} if media else {}))
                    self.runner.submit(row, len(ids), ids=ids,
                                       salts=[(m["positions"][0], bytes.fromhex(m["digest"])) for m in media] if media else ())
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
                                               media=media, cancelled=None)
                    self._waiting.popleft()
                    continue                                              # admitted when every rank's read is done (_settle)
                self._idle_order.pop(row)
                tokens = self.engine.extend(row, ids, max_new=limit, temperature=temperature, **({"min_new": min_new} if min_new else {}),
                                            **({"options": options} if options else {}), **({"media": media} if media else {}))
                self.runner.extend(row, tokens)
            self._waiting.popleft()
            self._active[row] = (request, promised)

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
        for row, votes in zip(rows, done):
            if votes < world:
                continue
            ok, full = True, False
            try:
                if row in self._retiring:
                    self.runner.park_finish(row)
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
            else:
                e = self._resuming.pop(row)
                conversation, request = e["conversation"], e["request"]
                if all_ok == world and e["cancelled"] is None:   # resident everywhere: the turn proceeds
                    tokens = self.engine.extend(row, e["ids"], max_new=e["limit"], temperature=e["temperature"],
                                                **({"min_new": e["min_new"]} if e["min_new"] else {}),
                                                **({"options": e["options"]} if e.get("options") else {}),
                                                **({"media": e["media"]} if e.get("media") else {}))
                    self.runner.extend(row, tokens)
                    self._conversations[conversation] = row
                    self._conversation_of[row] = conversation
                    self._active[row] = (request, e["promised"])
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
        # Raw occupancy, not kv.available(): that one adds the blocks the prefix cache would
        # give back on demand, which is a separate series below. A block pinned by the cache
        # is held, and the saturation signal has to say so.
        free_blocks = len(kv.free)
        used_blocks = kv.num_blocks - free_blocks
        rows = [
            ("counter", "vllm:request_success_total", "requests answered", self.served),
            ("gauge", "vllm:num_requests_running", "requests in the model's step", len(runner.state.running)),
            ("gauge", "vllm:num_requests_waiting", "admitted or queued, not yet stepping",
             len(runner.state.waiting) + len(self._waiting) + len(self.pending) - len(self._active)),
            ("counter", "vllm:prompt_tokens_total", "prompt tokens admitted", self.prompt_tokens_total),
            ("counter", "vllm:generation_tokens_total", "tokens generated", self.generation_tokens_total),
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
            ("counter", "st:requests_cancelled_total", "requests cancelled, for any reason", self.cancelled),
            ("counter", "st:requests_timed_out_total", "the subset the deadline scan took", self.timed_out),
            ("gauge", "st:kv_blocks_total", f"blocks of {kv.block_size} tokens in the pool", kv.num_blocks),
            ("gauge", "st:kv_blocks_used", "blocks held by a row or pinned by the cache", used_blocks),
            ("gauge", "st:kv_rows_in_use", "pool rows with tokens", kv.rows_in_use),
            ("gauge", "st:state_slots_total", "recurrent/conv state slots (slot 0 is the null slot)",
             slots.num_slots - 1),
            ("gauge", "st:kv_blocks_free", "blocks no row or pin holds", free_blocks),
            ("gauge", "st:state_slots_free", "state slots a new request could take", slots.available),
        ]
        prefix = getattr(runner, "prefix", None)
        if prefix is not None:
            rows += [
                ("counter", "vllm:prefix_cache_queries_total", "prompts looked up in the prefix cache",
                 prefix.hits + prefix.misses),
                ("counter", "vllm:prefix_cache_hits_total", "lookups that reused a cached prefix", prefix.hits),
                ("counter", "st:prefix_cache_evictions_total", "cached prefixes dropped", prefix.evictions),
                ("gauge", "st:prefix_cache_reclaimable_blocks", "blocks only the prefix cache holds",
                 prefix.reclaimable()),
            ]
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
                ("vllm:time_to_first_token_seconds", "admission to first token", self.ttft),
                ("vllm:time_per_output_token_seconds", "interval between consecutive tokens", self.itl),
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
            alive, arrivals, cancels = self.comm.broadcast_object(
                (self.alive, self._drain(), self._drain_cancels()) if self.comm.rank == 0 else None)
            if not alive:
                self._abort()
                self._fail_pending()
                return False
            self._waiting.extend(arrivals)
            for request, reason in cancels:
                self._cancel(request, reason)
            self._settle()
            self._admit()
            began = self.clock()
            step = self.runner.step()
            now = self.clock()
            if step is not None:                                    # D9: one kind or the other
                kind = "prefill" if step.kind == "prefill" else "decode"   # base/scheduler.PREFILL
                if kind == "prefill":
                    self.steps_prefill += 1
                else:
                    self.steps_decode += 1
                self.step_seconds[kind].observe(now - began)
            # Every live row's new tokens, whether or not it streams: the first one is this
            # request's time to first token, each later one an inter-token interval. A step
            # that lands several (the drafter's accepted run) shares its elapsed time across
            # them, which is how the vLLM counters these names belong to define it.
            for row, (request, _) in self._active.items():
                generated = self.engine.generated(row)
                sent = self._sent.get(row, 0)
                fresh = len(generated) - sent
                if fresh > 0:
                    last = self._token_at.get(row)
                    if last is None:
                        arrived = self._arrived.get(request)
                        if arrived is not None:
                            self.ttft.observe(now - arrived)
                        fresh -= 1                                  # the first token is the TTFT, not an interval
                        last = now
                    if fresh > 0:
                        each = (now - last) / fresh
                        for _ in range(fresh):
                            self.itl.observe(each)
                    self._token_at[row] = now
                    stream = self._streams.get(request)
                    if stream is not None:                          # rank 0: hand it the new tokens
                        lp = getattr(self.engine, "logprobs", None)
                        entries = lp(row) if lp is not None else None
                        stream.put(("tokens", (list(generated[sent:]), list(entries[sent:]) if entries else None)))
                        self._wake.set()
                    self._sent[row] = len(generated)
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
                    self._retire(row)
                    self.served += 1
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
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply(200, {"object": "list", "data": [{"id": server.model_name, "object": "model", "owned_by": "st"}]})
                elif self.path == "/metrics":
                    body = server.metrics().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/health":
                    self.reply(200 if server.alive else 503, {"status": "ok" if server.alive else "stopping"})
                else:
                    self.reply(200, {"engine": "ST", "model": server.model_name, "running": list(server.runner.state.running),
                                     "waiting": list(server.runner.state.waiting), "queued": len(server._waiting),
                                     "parked": len(server.runner.parked_keys()),
                                     "parking": len(server._retiring), "resuming": len(server._resuming),
                                     "steps": server.runner.steps, "served": server.served})

            def body(self):
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 4 << 20:
                    raise RequestError("request body must contain 1 to 4194304 bytes", 413)
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise RequestError("request must be a JSON object")
                return req

            def sse(self, payload):
                self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
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
                            want_logprobs=None, min_new=0, continue_history=False, media=None):
                """Submit `count` generations of one prompt; each is a _Choice fed by its own token queue.
                With a seed, choice i draws from seed + i so the n answers differ but stay reproducible."""
                choices = []
                for i in range(count):
                    opts = dict(options)
                    if count > 1 and "seed" in opts:
                        opts["seed"] = opts["seed"] + i
                    request, event = server.submit(ids, max_new, temperature, stream=True, min_new=min_new,
                                                   options=opts, continue_history=continue_history, media=media)
                    choices.append(_Choice(len(choices), request, event, server._streams[request], tok=server.tok, stop=stop,
                                           reasoning=reasoning, tool_parser=tool_parser, want_logprobs=want_logprobs))
                return choices

            def run_choices(self, choices, on_delta) -> bool:
                """Drive every choice's queue until all have ended; `on_delta(choice, deltas)` receives each flush.
                False when the client left (every live generation is cancelled)."""
                live = {c.request: c for c in choices}
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
                                live.pop(c.request)
                        elif kind == "end":
                            deltas = c.flush(final=True)
                            if deltas:
                                on_delta(c, deltas)
                            c.finish = c.finish or payload
                            c.done = True
                            live.pop(c.request)
                        else:
                            c.error = payload
                            c.done = True
                            live.pop(c.request)
                    if not progressed:
                        if self.gone():
                            for c in live.values():
                                server.cancel(c.request, "client closed")
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
                if effort is not None:
                    if effort not in ("low", "high", "max"):
                        raise RequestError("reasoning_effort must be low, high, or max")
                    if kwargs.get("reasoning_effort", effort) != effort:
                        raise RequestError("top-level and template reasoning_effort must agree")
                    kwargs["reasoning_effort"] = effort
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
                    max_tokens = req.get("max_completion_tokens", 256)
                stream = bool(req.get("stream", False))
                include_usage = bool(options_stream and options_stream.get("include_usage"))
                model = req.get("model") if isinstance(req.get("model"), str) and req.get("model") else server.model_name
                temperature, options = sampling_options(req, server.generation)
                if want_logprobs is not None:
                    options["logprobs"] = want_logprobs
                grammar = response_format_grammar(req)
                if grammar is not None:
                    options["grammar"] = grammar
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
                    opening, resuming = prompt_switches(req)
                    prompt = server.chat(messages, dict(kwargs, tools=tools) if tools else kwargs,
                                         generation_prompt=opening, continue_final=resuming)
                except Exception as exc:                                  # noqa: BLE001 -- the template's verdict on these messages
                    raise RequestError(f"chat template rejected the request: {exc}") from exc
                ids = server.tok.encode(prompt, add_special_tokens=False).ids
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
                choices = self.choices_for(ids, n, max_tokens, temperature, options, stop, reasoning=reasoning,
                                           tool_parser=server.tool_parser, want_logprobs=want_logprobs, min_new=min_tokens,
                                           continue_history=True, media=media)
                head = {"id": f"chatcmpl-{choices[0].request}", "created": int(time.time()), "model": model}

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
                             "completion_tokens_details": {"reasoning_tokens": sum(len(c.held["reasoning_content"]) for c in choices)}}
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
                            if c.held["reasoning_content"]:
                                message["reasoning_content"] = c.text["reasoning_content"]
                            if c.tool_calls:
                                message["tool_calls"] = c.tool_calls
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
                    prompts = [server.tok.encode(prompt, add_special_tokens=True).ids]
                elif isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt):
                    prompts = [list(prompt)]
                elif isinstance(prompt, list) and prompt and all(isinstance(p, str) for p in prompt):
                    prompts = [server.tok.encode(p, add_special_tokens=True).ids for p in prompt]
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
                model = req.get("model") if isinstance(req.get("model"), str) and req.get("model") else server.model_name
                temperature, options = sampling_options(req, server.generation)
                count = best_of or n
                if want_logprobs is not None or best_of:
                    options["logprobs"] = want_logprobs if want_logprobs is not None else 0
                lp_want = want_logprobs if want_logprobs is not None else (0 if best_of else None)
                choices, prompt_of = [], {}
                for ids in prompts:
                    group = self.choices_for(ids, count, max_tokens, temperature, options, stop, reasoning=False,
                                             want_logprobs=lp_want)
                    for c in group:
                        c.index = len(choices)
                        prompt_of[c.request] = ids
                        choices.append(c)
                head = {"id": f"cmpl-{choices[0].request}", "created": int(time.time()), "model": model}

                def legacy_logprobs(c, ids_prompt):
                    if want_logprobs is None:
                        return None
                    tokens, lps, tops, offsets = [], [], [], []
                    pos = len(server.tok.decode(ids_prompt)) if echo else 0
                    for tid, lp, top in c.logprobs:
                        text = server.tok.decode([tid])
                        tokens.append(text); lps.append(lp); offsets.append(pos); pos += len(text)
                        tops.append({server.tok.decode([i]): v for i, v in top[:want_logprobs]} if want_logprobs else None)
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
                             "total_tokens": sum(len(p) for p in prompts) + sum(c.total for c in choices)}
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
                ids = server.tok.encode(prompt, add_special_tokens=add_special).ids
                self.reply(200, {"count": len(ids), "max_model_len": server.max_context, "tokens": ids})

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
                    ids = server.tok.encode(prompt).ids
                t0 = time.perf_counter()
                conversation = req.get("conversation")
                temperature, options = sampling_options(req, {})
                request, event = server.submit(ids, req.get("max_tokens", 64), temperature, conversation, options=options)
                out = self.wait_result(request, event)
                text = server.tok.decode(out) if server.tok is not None else None
                conversation = (request if conversation is None else conversation) if server.runner.keep_idle else None
                self.reply(200, {"seq": request, "conversation": conversation, "ids": out, "text": text, "prompt_tokens": len(ids),
                                 "completion_tokens": len(out), "seconds": round(time.perf_counter() - t0, 3)})

            def do_POST(self):
                try:
                    routes = {"/v1/chat/completions": self.chat, "/v1/completions": self.completions,
                              "/v1/engine/completions": self.engine_completions, "/tokenize": self.tokenize,
                              "/detokenize": self.detokenize}
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
