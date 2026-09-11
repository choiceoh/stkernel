"""The serve loop and its one door (base): requests arrive on rank 0, every
rank steps in lockstep, tokens come out where they came in.

    Server(engine, runner, comm, port).loop()

Rank 0 listens (a small HTTP server on its own thread) and puts arrivals on a
queue; each iteration of `loop` broadcasts the drained queue to every rank
(one collective, `comm.broadcast_object`), submits them, and runs one
scheduler step. Sampling is seeded and the logits are identical on every
rank (all-gathered), so the ranks never exchange tokens: they compute the
same ones. A request completes when its sequence leaves the runner; rank 0
then answers the waiting HTTP call.

The API is deliberately tiny -- POST /v1/completions with `prompt` (text,
when a tokenizer is given) or `ids`, `max_tokens`, `temperature` -- because
the point of this file is the loop, not the door.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Server:
    def __init__(self, engine, runner, comm, port: int = 8000, tokenizer=None, host: str = "0.0.0.0"):
        self.engine, self.runner, self.comm = engine, runner, comm
        self.port, self.host, self.tok = port, host, tokenizer
        self.arrivals = queue.Queue()             # rank 0: (seq, ids, max_new, temperature)
        self.pending = {}                         # rank 0: seq -> Event
        self.results = {}
        self.next_seq = 0
        self.alive = True
        self.served = 0

    # -- rank 0's door ---------------------------------------------------------------
    def submit(self, ids, max_new: int, temperature: float, conversation: "int | None" = None):
        """A new conversation, or one more turn on `conversation` (its caches are
        held idle, or parked on NVMe -- the loop resumes them; D16)."""
        if conversation is None:
            seq = self.next_seq; self.next_seq += 1
        else:
            seq = int(conversation)
        ev = threading.Event(); self.pending[seq] = ev
        self.arrivals.put((seq, list(ids), int(max_new), float(temperature), conversation is not None))
        return seq, ev

    def _serve_http(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"engine": "ST", "running": list(server.runner.state.running), "waiting": list(server.runner.state.waiting),
                                   "steps": server.runner.steps, "served": server.served}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                ids = req.get("ids")
                if ids is None:
                    if server.tok is None:
                        self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"no tokenizer: send ids"}'); return
                    ids = server.tok.encode(req.get("prompt", "")).ids
                t0 = time.perf_counter()
                seq, ev = server.submit(ids, req.get("max_tokens", 64), req.get("temperature", 0.0), req.get("conversation"))
                ev.wait()
                out = server.results.pop(seq)
                if isinstance(out, str):
                    self.send_response(409); self.end_headers(); self.wfile.write(json.dumps({"error": out}).encode()); return
                text = server.tok.decode(out) if server.tok is not None else None
                body = json.dumps({"seq": seq, "conversation": seq, "ids": out, "text": text, "prompt_tokens": len(ids),
                                   "completion_tokens": len(out), "seconds": round(time.perf_counter() - t0, 3)}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
        return httpd

    # -- every rank's loop ---------------------------------------------------------------
    def _drain(self):
        out = []
        while True:
            try:
                out.append(self.arrivals.get_nowait())
            except queue.Empty:
                return out

    def _admit(self, seq, ids, max_new, temperature, continues):
        r = self.runner
        if not continues:
            self.engine.add(seq, ids, max_new=max_new, temperature=temperature)
            r.submit(seq, len(ids))
            return None
        if seq not in r.idle:
            return f"conversation {seq} is not idle (unknown, live, or evicted)"
        if r.tiered is not None and r.tiered.is_parked(seq):
            r.resume(seq)                                          # NVMe -> fresh blocks, off the step path
        n = self.engine.extend(seq, ids, max_new=max_new, temperature=temperature)
        r.extend(seq, n)                                           # the pending token and the new prompt, from the model's context
        return None

    def once(self) -> bool:
        """One iteration: broadcast arrivals, admit, one step. Returns whether a step ran."""
        arrivals = self.comm.broadcast_object(self._drain() if self.comm.rank == 0 else None)
        for seq, ids, max_new, temperature, continues in arrivals:
            err = self._admit(seq, ids, max_new, temperature, continues)
            if err is not None:
                self.served += 1
                if self.comm.rank == 0 and seq in self.pending:
                    self.results[seq] = err; self.pending.pop(seq).set()
        before = set(self.runner.state.running) | set(self.runner.state.waiting)
        step = self.runner.step()
        if step is None:
            return False
        live = set(self.runner.state.running) | set(self.runner.state.waiting)
        for seq in before - live:
            self.served += 1
            if self.comm.rank == 0 and seq in self.pending:
                self.results[seq] = self.engine.generated(seq)                   # this turn's tokens (extend resets the prompt boundary)
                self.pending.pop(seq).set()
            if self.runner.keep_idle and self.runner.tiered is not None and seq in self.runner.idle:
                self.runner.park(seq)                                  # idle turns leave the arena at once; the conversation keeps its KV
        return True

    def loop(self, idle_sleep: float = 0.002):
        httpd = self._serve_http() if self.comm.rank == 0 else None
        try:
            while self.alive:
                if not self.once():
                    time.sleep(idle_sleep)              # every rank sleeps the same idle tick; the next broadcast realigns them
        finally:
            if httpd is not None:
                httpd.shutdown()
