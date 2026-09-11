"""Bounded request admission and the rank-0 HTTP door.

Public request IDs are independent of the runner's reusable KV rows. Every
rank receives the same FIFO, admits only requests whose entire decode horizon
fits the declared budget, and recycles rows after copying out results.
Kernel failures remain fatal; waiting HTTP clients receive an error on exit.

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
import select
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Server:
    def __init__(self, engine, runner, comm, port: int = 8000, tokenizer=None,
                 host: str = "0.0.0.0", max_pending: int = 64, chat=None, model_name: str = "st",
                 reasoning_end: "int | None" = None, request_timeout_s: float = 3600.0, tool_parser=None):
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
        self.max_pending = max_pending
        self.request_timeout_s = float(request_timeout_s)
        self.clock = time.monotonic                # injectable for tests
        self._cancels = set()                      # (request id, reason) asked by HTTP threads / the timeout scan; rank 0 broadcasts them
        self._deadline = {}                        # request id -> clock() by which it must have finished (rank 0)
        self.cancelled = 0
        self._streams = {}                         # request id -> queue of ("tokens", ids) | ("end", finish) | ("error", text)
        self._sent = {}                            # row -> generated tokens already handed to its stream
        self.prompt_tokens_total = self.generation_tokens_total = 0
        self.arrivals = queue.Queue()
        self.pending, self.results = {}, {}
        self.next_seq, self.served = 0, 0
        self.alive = True
        self._lock = threading.Lock()
        self._waiting = deque()                    # request id, tokens, limit, temperature, promised blocks
        self._active = {}                          # reusable row -> (request id, promised blocks)
        self._conversations, self._conversation_of = {}, {}
        self._idle_order = {}                      # least recently completed turn first
        self._free_rows = list(range(min(runner.kv.max_seqs, runner.c.max_running, runner.slots.available)))
        if not self._free_rows:
            raise ValueError("the server needs at least one request row and state slot")

    def submit(self, ids, max_new: int, temperature: float, conversation: "int | None" = None, stream: bool = False,
               min_new: int = 0):
        """Validate and enqueue on rank 0 without acquiring any model resources.
        `stream`: the request also gets a token queue (see `_streams`). `min_new`: no end token before this many."""
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
            self._deadline[request] = self.clock() + self.request_timeout_s
            self.prompt_tokens_total += len(ids)
            self.arrivals.put((request, list(ids), max_new, float(temperature), blocks, conversation, min_new))
        return request, event

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
                self._deadline.pop(request, None)
                return                                     # finished already (or unknown): nothing to drop
            self._active.pop(row)
            self._sent.pop(row, None)
            try:
                self.runner.cancel(row)
            finally:
                self.engine.forget(row)
                if self.runner.keep_idle:
                    self._conversations.pop(self._conversation_of.pop(row, None), None)
                    self._idle_order.pop(row, None)
                heapq.heappush(self._free_rows, row)
        self.cancelled += 1
        self._deadline.pop(request, None)
        status = 504 if reason == "timeout" else 499
        self._answer(request, RequestError(f"request cancelled: {reason}", status))

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
            request, ids, limit, temperature, promised, conversation, min_new = self._waiting[0]
            row = None
            resident = 0
            if conversation is not None:
                row = self._conversations.get(conversation)
                if row is None or row not in self.runner.idle:
                    self._waiting.popleft()
                    self._answer(request, RequestError("conversation is unknown, live or evicted", 409))
                    continue
                end = self.engine.context(row) + self.engine.extension_tokens(row, ids)
                horizon = end + limit - 1 + (self.runner.c.draft_slots if limit > 1 else 0)
                promised = self.runner.kv.blocks_for(horizon)
                if horizon >= 2**31 or promised > min(self.runner.kv.num_blocks, self.runner.kv.max_blocks_per_seq):
                    self._waiting.popleft()
                    self._answer(request, RequestError("conversation and generation limit exceed the KV capacity"))
                    continue
                resident = self.runner.kv.blocks_for(self.runner.kv.tokens[row])
                held = resident
                if self.runner.tiered is not None and self.runner.tiered.is_parked(row):
                    held = self.runner.tiered.tier.index[str(row)]["blocks"]
                promised = max(promised, held)       # rejected-draft reservations may exceed the new turn
            elif not self._free_rows:
                if self._evict_idle():
                    continue
                break
            # Future decode growth already belongs to admitted requests even
            # though the block pool acquires those blocks only when written.
            future = sum(b - self.runner.kv.blocks_for(self.runner.kv.tokens[row])
                         for row, (_, b) in self._active.items())
            if promised > self.runner.kv.available - future + resident:
                if self._evict_idle(exclude=row):
                    continue
                break
            if conversation is None:
                row = heapq.heappop(self._free_rows)
                try:
                    self.engine.add(row, ids, max_new=limit, temperature=temperature, **({"min_new": min_new} if min_new else {}))
                    self.runner.submit(row, len(ids), ids=ids)
                except BaseException:
                    self.engine.forget(row)
                    heapq.heappush(self._free_rows, row)
                    raise
                if self.runner.keep_idle:
                    self._conversations[request] = row
                    self._conversation_of[row] = request
            else:
                if self.runner.tiered is not None and self.runner.tiered.is_parked(row):
                    self.runner.resume(row)
                tokens = self.engine.extend(row, ids, max_new=limit, temperature=temperature, **({"min_new": min_new} if min_new else {}))
                self.runner.extend(row, tokens)
                self._idle_order.pop(row)
            self._waiting.popleft()
            self._active[row] = (request, promised)

    def _abort(self):
        self.alive = False
        # Preserve the original engine exception; cleanup must wake clients
        # even if a model's close hook also fails.
        error = None
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
        self._deadline.clear()

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
        rows.append(("st:requests_cancelled_total", self.cancelled))
        return "".join(f'{name}{{engine="st"}} {max(0, value)}\n' for name, value in rows)

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
                    if self.runner.keep_idle:
                        if self.runner.tiered is not None:
                            self.runner.park(row)
                        self._idle_order[row] = None
                    else:
                        self.engine.forget(row)
                        heapq.heappush(self._free_rows, row)
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

            def chat(self, req):
                """OpenAI chat completions over the engine: template -> ids -> submit; the tokens come back
                through the request's queue whether the reply streams or not, so `stop` strings and a client
                that hangs up end the generation early in both modes."""
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
                options = req.get("stream_options")
                if options is not None and not isinstance(options, dict):
                    raise RequestError("stream_options must be an object")
                if req.get("n", 1) != 1:
                    raise RequestError("n must be 1: one generation per request")
                if req.get("logprobs"):
                    raise RequestError("logprobs are not served")
                stop = req.get("stop")
                stop = [stop] if isinstance(stop, str) else (stop or [])
                if not isinstance(stop, list) or len(stop) > 4 or any(not isinstance(x, str) or not x for x in stop):
                    raise RequestError("stop must be a nonempty string or up to four of them")
                tools = req.get("tools")
                if req.get("tool_choice") == "none":
                    tools = None
                if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
                    raise RequestError("tools must be a list of objects")
                min_tokens = req.get("min_tokens", 0) or 0
                if type(min_tokens) is not int or min_tokens < 0:
                    raise RequestError("min_tokens must be a nonnegative integer")
                stream = bool(req.get("stream", False))
                include_usage = bool(options and options.get("include_usage"))
                model = req.get("model") if isinstance(req.get("model"), str) and req.get("model") else server.model_name
                try:
                    prompt = server.chat(messages, dict(kwargs, tools=tools) if tools else kwargs)
                except Exception as exc:                                  # noqa: BLE001 -- the template's verdict on these messages
                    raise RequestError(f"chat template rejected the request: {exc}") from exc
                ids = server.tok.encode(prompt, add_special_tokens=False).ids
                request, event = server.submit(ids, req.get("max_tokens", 256), req.get("temperature", 0.0), stream=True,
                                               min_new=min_tokens)
                head = {"id": f"chatcmpl-{request}", "created": int(time.time()), "model": model}
                chunks_out = []                                          # SSE payloads, written now (stream) or never (whole)

                def chunk(delta=None, finish=None, usage=None):
                    payload = {**head, "object": "chat.completion.chunk",
                               "choices": [] if usage is not None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
                               **({"usage": usage} if usage is not None else {})}
                    if stream:
                        self.sse(payload)

                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    chunk({"role": "assistant", "content": ""})
                held = {"reasoning_content": [], "content": []}         # ids per channel
                shown = {"reasoning_content": 0, "content": 0}          # characters already sent per channel
                text = {"reasoning_content": "", "content": ""}         # decoded so far per channel
                reasoning = server.reasoning_end is not None
                total = 0
                finish = None

                def flush(final=False):
                    """Decode each channel, send what is new; a stop string ends the content channel."""
                    nonlocal finish
                    for channel, chan_ids in held.items():
                        decoded = server.tok.decode(chan_ids)
                        if channel == "content" and stop:
                            cut = min((decoded.find(x) for x in stop if x in decoded), default=-1)
                            if cut >= 0:
                                decoded = decoded[:cut]
                                finish = "stop"
                        delta = decoded[shown[channel]:]
                        if delta and (final or finish == "stop" or not delta.endswith("\ufffd")):   # a partial character waits
                            chunk({channel: delta})
                            shown[channel] = len(decoded)
                        text[channel] = decoded[:shown[channel]]
                    return finish == "stop"

                stream_q = server._streams[request]
                error = None
                try:
                    while True:
                        try:
                            kind, payload = stream_q.get(timeout=0.25)
                        except queue.Empty:
                            kind, payload = None, None
                        if self.gone():
                            server.cancel(request, "client closed")
                            return
                        if kind is None:
                            continue
                        if kind == "tokens":
                            for t in payload:
                                total += 1
                                if reasoning and t == server.reasoning_end:
                                    reasoning = False
                                    continue
                                held["reasoning_content" if reasoning else "content"].append(t)
                            if flush():
                                server.cancel(request, "stop")            # the loop drops the row; the answer is complete here
                                break
                        elif kind == "end":
                            flush(final=True)
                            finish = finish or payload
                            break
                        else:
                            error = payload
                            break
                    if error is not None:
                        if stream:
                            self.sse({"error": {"message": error, "type": "engine"}})
                        else:
                            self.reply(503, {"error": error})
                        return
                    calls = server.tool_parser(text["content"]) if server.tool_parser is not None and text["content"] else None
                    usage = {"prompt_tokens": len(ids), "completion_tokens": total, "total_tokens": len(ids) + total,
                             "completion_tokens_details": {"reasoning_tokens": len(held["reasoning_content"])}}
                    if calls:
                        finish = "tool_calls"
                        tool_calls = [{"id": f"call_{request}_{i}", "type": "function", "function": {"name": name, "arguments": args}}
                                      for i, (name, args) in enumerate(calls)]
                        content = text["content"][:text["content"].find("<tool_call>")] if "<tool_call>" in text["content"] else ""
                    else:
                        tool_calls, content = None, text["content"]
                    if stream:
                        chunk({"tool_calls": tool_calls} if tool_calls else None, finish=finish)
                        if include_usage:
                            chunk(usage=usage)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        message = {"role": "assistant", "content": content or None}
                        if held["reasoning_content"]:
                            message["reasoning_content"] = text["reasoning_content"]
                        if tool_calls:
                            message["tool_calls"] = tool_calls
                        self.reply(200, {**head, "object": "chat.completion",
                                         "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    server.cancel(request, "client closed")
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
                    out = self.wait_result(request, event)
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
