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

import heapq
import json
import math
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine.base.kv_tier import TierFull


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Server:
    def __init__(self, engine, runner, comm, port: int = 8000, tokenizer=None,
                 host: str = "0.0.0.0", max_pending: int = 64, chat=None, model_name: str = "st",
                 reasoning_end: "int | None" = None):
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if runner.slot_of:
            raise ValueError("the server requires an idle runner")
        if reasoning_end is not None and (type(reasoning_end) is not int or reasoning_end < 0):
            raise ValueError("reasoning_end must be a token id")
        self.engine, self.runner, self.comm = engine, runner, comm
        self.port, self.host, self.tok = port, host, tokenizer
        self.chat, self.model_name, self.reasoning_end = chat, model_name, reasoning_end
        self.max_pending = max_pending
        self.max_context = int(getattr(engine, "max_context", 2**31 - 1))   # the model's trained positions; the door refuses beyond
        self._streams = {}                         # request id -> queue of ("tokens", ids) | ("end", finish) | ("error", text)
        self._sent = {}                            # row -> generated tokens already handed to its stream
        self.prompt_tokens_total = self.generation_tokens_total = 0
        self.arrivals = queue.Queue()
        self.pending, self.results = {}, {}
        # conversation ids are request ids; parked conversations from an earlier boot keep theirs
        self.next_seq, self.served = 1 + max(runner.parked_keys(), default=-1), 0
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

    def submit(self, ids, max_new: int, temperature: float, conversation: "int | None" = None, stream: bool = False):
        """Validate and enqueue on rank 0 without acquiring any model resources.
        `stream`: the request also gets a token queue (see `_streams`)."""
        if self.comm.rank != 0:
            raise RequestError("requests must enter on rank 0")
        if conversation is not None:
            if type(conversation) is not int or conversation < 0:
                raise RequestError("conversation must be a nonnegative integer")
            if not self.runner.keep_idle:
                raise RequestError("this server does not retain conversations", 409)
        if not isinstance(ids, (list, tuple)) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise RequestError("ids must be a nonempty list of nonnegative token integers")
        if type(max_new) is not int or max_new <= 0:
            raise RequestError("max_tokens must be a positive integer")
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
            self.prompt_tokens_total += len(ids)
            self.arrivals.put((request, list(ids), max_new, float(temperature), blocks, conversation))
        return request, event

    def take_result(self, request):
        with self._lock:
            result = self.results.pop(request)
        if isinstance(result, RequestError):
            raise result
        return result

    def finish_reason(self, out) -> str:
        """OpenAI's word for how a generation ended: at one of the model's end tokens, or at the limit."""
        return "stop" if out and out[-1] in getattr(self.engine, "eos", ()) else "length"

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
                stream.put(("error", str(result)) if isinstance(result, RequestError) else ("end", self.finish_reason(result)))

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
            request, ids, limit, temperature, promised, conversation = self._waiting[0]
            row = None
            resident = held = 0
            parked = False
            if conversation is not None:
                row = self._conversations.get(conversation)
                if row is None and (conversation in self._retiring.values()
                                    or any(c == conversation for c, *_ in self._resuming.values())):
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
                      + sum(b - self.runner.kv.blocks_for(self.runner.kv.tokens[r]) for r, (*_, b) in self._resuming.items()))
            if promised > self.runner.kv.available - future + resident:
                if self._evict_idle(exclude=row):
                    continue
                break
            if conversation is None:
                row = heapq.heappop(self._free_rows)
                try:
                    self.engine.add(row, ids, max_new=limit, temperature=temperature)
                    self.runner.submit(row, len(ids))
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
                    self._resuming[row] = (conversation, request, ids, limit, temperature, promised)
                    self._waiting.popleft()
                    continue                                              # admitted when every rank's read is done (_settle)
                self._idle_order.pop(row)
                tokens = self.engine.extend(row, ids, max_new=limit, temperature=temperature)
                self.runner.extend(row, tokens)
            self._waiting.popleft()
            self._active[row] = (request, promised)

    def _votes(self, flags) -> "list[int]":
        """How many ranks say yes to each flag. Every rank must call this with the same flags in the
        same order (the transfers are submitted in lockstep); a single rank answers itself."""
        world = int(getattr(self.comm, "world_size", 1) or 1)
        if world <= 1 or not flags:
            return [int(bool(f)) for f in flags]
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
                conversation, request, ids, limit, temperature, promised = self._resuming.pop(row)
                if all_ok == world:                           # resident everywhere: the turn proceeds
                    tokens = self.engine.extend(row, ids, max_new=limit, temperature=temperature)
                    self.runner.extend(row, tokens)
                    self._conversations[conversation] = row
                    self._conversation_of[row] = conversation
                    self._active[row] = (request, promised)
                else:                                         # a rank could not read it back: the conversation is gone everywhere
                    if ok:
                        self.runner.evict(row)
                        self.engine.forget(row)
                    else:
                        self.runner.forget_parked(conversation)
                    heapq.heappush(self._free_rows, row)
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
        self._streams.clear()
        self._sent.clear()

    def metrics(self) -> str:
        """Prometheus text in the names bench/window_metrics.py and bench/bracket.py read (the bench's
        dialect, kept so the same onepass gate judges this engine and the one it replaces)."""
        engine = self.engine
        rows = [("vllm:request_success_total", self.served),
                ("vllm:num_requests_running", len(self.runner.state.running)),
                ("vllm:num_requests_waiting", len(self.runner.state.waiting) + len(self._waiting) + len(self.pending) - len(self._active)),
                ("vllm:prompt_tokens_total", self.prompt_tokens_total),
                ("vllm:generation_tokens_total", self.generation_tokens_total),
                ("vllm:spec_decode_num_accepted_tokens_total", getattr(engine, "accepted_total", 0)),
                ("vllm:spec_decode_num_draft_tokens_total", getattr(engine, "drafted_total", 0))]
        if hasattr(engine, "drafts_total"):
            rows.append(("vllm:spec_decode_num_drafts_total", engine.drafts_total))
        return "".join(f'{name}{{engine="st"}} {max(0, value)}\n' for name, value in rows)

    def once(self) -> bool:
        """One ordered broadcast, bounded admission and homogeneous model step."""
        try:
            alive, arrivals = self.comm.broadcast_object(
                (self.alive, self._drain()) if self.comm.rank == 0 else None)
            if not alive:
                self._abort()
                self._fail_pending()
                return False
            self._waiting.extend(arrivals)
            self._settle()
            self._admit()
            step = self.runner.step()
            if self._streams:                                       # rank 0: hand each streaming request its new tokens
                for row, (request, _) in self._active.items():
                    stream = self._streams.get(request)
                    if stream is None:
                        continue
                    generated = self.engine.generated(row)
                    sent = self._sent.get(row, 0)
                    if len(generated) > sent:
                        stream.put(("tokens", list(generated[sent:])))
                        self._sent[row] = len(generated)
            live = set(self.runner.state.running) | set(self.runner.state.waiting)
            for row in list(self._active):
                if row not in live:
                    request, _ = self._active.pop(row)
                    self._sent.pop(row, None)
                    result = list(self.engine.generated(row))
                    self._retire(row)
                    self.served += 1
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

            def chat(self, req):
                """OpenAI chat completions over the engine: template -> ids -> submit; streamed by token or whole."""
                if server.chat is None or server.tok is None:
                    raise RequestError("this server has no chat template", 404)
                messages = req.get("messages")
                if (not isinstance(messages, list) or not messages
                        or any(not isinstance(m, dict) or not isinstance(m.get("role"), str)
                               or not isinstance(m.get("content"), str) for m in messages)):
                    raise RequestError("messages must be a nonempty list of {role, content} objects")
                kwargs = req.get("chat_template_kwargs") or {}
                if not isinstance(kwargs, dict):
                    raise RequestError("chat_template_kwargs must be an object")
                options = req.get("stream_options")
                if options is not None and not isinstance(options, dict):
                    raise RequestError("stream_options must be an object")
                stream = bool(req.get("stream", False))
                include_usage = bool(options and options.get("include_usage"))
                model = req.get("model") if isinstance(req.get("model"), str) and req.get("model") else server.model_name
                ids = server.tok.encode(server.chat(messages, kwargs), add_special_tokens=False).ids
                request, event = server.submit(ids, req.get("max_tokens", 256), req.get("temperature", 0.0), stream=stream)
                head = {"id": f"chatcmpl-{request}", "created": int(time.time()), "model": model}
                if not stream:
                    event.wait()
                    out = server.take_result(request)
                    reasoning, content = server.split(out)
                    message = {"role": "assistant", "content": server.tok.decode(content)}
                    if reasoning:
                        message["reasoning_content"] = server.tok.decode(reasoning)
                    self.reply(200, {**head, "object": "chat.completion",
                                     "choices": [{"index": 0, "message": message, "finish_reason": server.finish_reason(out)}],
                                     "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def chunk(delta=None, finish=None, usage=None):
                    self.sse({**head, "object": "chat.completion.chunk",
                              "choices": [] if usage is not None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
                              **({"usage": usage} if usage is not None else {})})

                chunk({"role": "assistant", "content": ""})
                held = {"reasoning_content": [], "content": []}         # ids per channel
                shown = {"reasoning_content": 0, "content": 0}          # characters already sent per channel
                reasoning = server.reasoning_end is not None
                total = 0

                def flush(final=False):
                    for channel, chan_ids in held.items():
                        text = server.tok.decode(chan_ids)
                        delta = text[shown[channel]:]
                        if delta and (final or not delta.endswith("\ufffd")):   # a partial character waits for its next token
                            chunk({channel: delta})
                            shown[channel] = len(text)

                stream_q = server._streams[request]
                try:
                    while True:
                        kind, payload = stream_q.get()
                        if kind == "tokens":
                            for t in payload:
                                total += 1
                                if reasoning and t == server.reasoning_end:
                                    reasoning = False
                                    continue
                                held["reasoning_content" if reasoning else "content"].append(t)
                            flush()
                        elif kind == "end":
                            flush(final=True)
                            chunk(finish=payload)
                            if include_usage:
                                chunk(usage={"prompt_tokens": len(ids), "completion_tokens": total, "total_tokens": len(ids) + total})
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                            break
                        else:
                            self.sse({"error": {"message": payload, "type": "engine"}})
                            break
                finally:
                    server._streams.pop(request, None)
                    event.wait()
                    try:
                        server.take_result(request)
                    except RequestError:
                        pass

            def do_POST(self):
                try:
                    if self.path == "/v1/chat/completions":
                        self.chat(self.body())
                        return
                    if self.path != "/v1/completions":
                        raise RequestError("unknown endpoint", 404)
                    req = self.body()
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
                    request, event = server.submit(ids, req.get("max_tokens", 64), req.get("temperature", 0.0), conversation)
                    event.wait()
                    out = server.take_result(request)
                    text = server.tok.decode(out) if server.tok is not None else None
                    conversation = (request if conversation is None else conversation) if server.runner.keep_idle else None
                    self.reply(200, {"seq": request, "conversation": conversation, "ids": out, "text": text, "prompt_tokens": len(ids),
                                     "completion_tokens": len(out), "seconds": round(time.perf_counter() - t0, 3)})
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
