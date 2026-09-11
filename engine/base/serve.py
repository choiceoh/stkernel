"""Bounded request admission and the rank-0 HTTP door.

Public request IDs are independent of the runner's reusable KV rows. Every
rank receives the same FIFO, admits only requests whose entire decode horizon
fits the declared budget, and recycles rows after copying out results.
Kernel failures remain fatal; waiting HTTP clients receive an error on exit.
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


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Server:
    def __init__(self, engine, runner, comm, port: int = 8000, tokenizer=None,
                 host: str = "0.0.0.0", max_pending: int = 64):
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if runner.slot_of:
            raise ValueError("the server requires an idle runner")
        self.engine, self.runner, self.comm = engine, runner, comm
        self.port, self.host, self.tok = port, host, tokenizer
        self.max_pending = max_pending
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

    def submit(self, ids, max_new: int, temperature: float, conversation: "int | None" = None):
        """Validate and enqueue on rank 0 without acquiring any model resources."""
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
        with self._lock:
            if not self.alive:
                raise RequestError("engine is stopping", 503)
            if len(self.pending) + len(self.results) >= self.max_pending:
                raise RequestError("request queue is full", 503)
            request = self.next_seq
            self.next_seq += 1
            event = threading.Event()
            self.pending[request] = event
            self.arrivals.put((request, list(ids), max_new, float(temperature), blocks, conversation))
        return request, event

    def take_result(self, request):
        with self._lock:
            result = self.results.pop(request)
        if isinstance(result, RequestError):
            raise result
        return result

    def _answer(self, request, result):
        if self.comm.rank == 0:
            with self._lock:
                event = self.pending.pop(request, None)
                if event is not None:
                    self.results[request] = result
                    event.set()

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
            request, ids, limit, temperature, promised, conversation = self._waiting[0]
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
                if self.runner.tiered is not None and self.runner.tiered.is_parked(row):
                    self.runner.resume(row)
                tokens = self.engine.extend(row, ids, max_new=limit, temperature=temperature)
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
            self._admit()
            step = self.runner.step()
            live = set(self.runner.state.running) | set(self.runner.state.waiting)
            for row in list(self._active):
                if row not in live:
                    request, _ = self._active.pop(row)
                    result = list(self.engine.generated(row))
                    if self.runner.keep_idle:
                        if self.runner.tiered is not None:
                            self.runner.park(row)
                        self._idle_order[row] = None
                    else:
                        self.engine.forget(row)
                        heapq.heappush(self._free_rows, row)
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
                self.reply(200, {"engine": "ST", "running": list(server.runner.state.running),
                                 "waiting": list(server.runner.state.waiting), "queued": len(server._waiting),
                                 "steps": server.runner.steps, "served": server.served})

            def do_POST(self):
                try:
                    if self.path != "/v1/completions":
                        raise RequestError("unknown endpoint", 404)
                    n = int(self.headers.get("Content-Length", "0"))
                    if not 0 < n <= 4 << 20:
                        raise RequestError("request body must contain 1 to 4194304 bytes", 413)
                    req = json.loads(self.rfile.read(n))
                    if not isinstance(req, dict):
                        raise RequestError("request must be a JSON object")
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
