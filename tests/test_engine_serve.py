"""Request lifecycle regressions using the real scheduler, pools and runner."""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import threading
import unittest
import urllib.error
import urllib.request

from engine.base.kv import BlockPool, SlotPool
from engine.base.record import Ring
from engine.base.runner import Runner, STEP_RECORD
from engine.base.scheduler import Contract
from engine.base.serve import RequestError, Server


class Engine:
    def __init__(self):
        self.tokens, self.ctx, self.limits, self.output = {}, {}, {}, {}
        self.opened = []
        self.fail_open = self.fail_decode = False

    def validate(self, ids, limit, temperature):
        if any(t >= 256 for t in ids):
            raise ValueError("token outside vocabulary")

    def add(self, seq, ids, max_new, temperature):
        self.tokens[seq], self.limits[seq], self.output[seq] = list(ids), max_new, []

    def open(self, seq, slot):
        self.ctx[seq] = 0
        self.opened.append(seq)
        if self.fail_open:
            raise RuntimeError("open failed")

    def close(self, seq):
        self.ctx.pop(seq, None)

    def forget(self, seq):
        for rows in (self.tokens, self.limits, self.output):
            rows.pop(seq, None)

    def horizon(self, seq):
        return self.ctx[seq] + 1

    def context(self, seq):
        return self.ctx[seq]

    def extension_tokens(self, seq, ids):
        return len(self.tokens[seq]) + len(self.output[seq]) + len(ids) - self.ctx[seq]

    def extend(self, seq, ids, max_new, temperature):
        n = self.extension_tokens(seq, ids)
        self.tokens[seq] += self.output[seq] + list(ids)
        self.output[seq] = []
        self.limits[seq] = max_new
        return n

    def prefill(self, seq, start, tokens, blocks, slot):
        self.ctx[seq] = start + tokens
        if self.ctx[seq] == len(self.tokens[seq]):
            self.output[seq].append(self.tokens[seq][-1])
        return len(self.output[seq]) == self.limits[seq]

    def decode(self, seqs, blocks, slots):
        if self.fail_decode:
            raise RuntimeError("kernel failed")
        for seq in seqs:
            self.ctx[seq] += 1
            self.output[seq].append(self.tokens[seq][-1])
        return [len(self.output[seq]) == self.limits[seq] for seq in seqs]

    def generated(self, seq):
        return self.output[seq]


class Comm:
    rank = 0
    def broadcast_object(self, obj):
        return obj


def server(*, rows=2, blocks=16, comm=None, max_pending=64, keep_idle=False, tiered=False):
    engine = Engine()
    runner = Runner(engine, Contract(4, 8, 0, 0, rows), BlockPool(blocks, 4, rows, blocks),
                    SlotPool(rows + 1), Ring(16, STEP_RECORD.size), keep_idle=keep_idle)
    if tiered:
        from test_engine_tier import MemoryTier, Storage
        from engine.base.tiered_kv import TieredKV
        runner.kv.attach_storage(Storage(blocks * 4), 4)
        runner.tiered = TieredKV(runner.kv, MemoryTier())
    return Server(engine, runner, comm or Comm(), host="127.0.0.1", port=0, max_pending=max_pending)


class ServeTests(unittest.TestCase):
    def drain(self, s, retained=False):
        for _ in range(500):
            if not s.once() and not s._waiting:
                break
        else:
            self.fail("server did not drain bounded requests")
        if retained:
            self.assertFalse(s._active)
            self.assertFalse(s.runner.state.running or s.runner.state.waiting)
            self.assertLessEqual(len(s.engine.tokens), s.runner.c.max_running)
            return
        self.assertFalse(s.engine.tokens)
        self.assertFalse(s.engine.ctx)
        self.assertFalse(s.runner.state.waiting)
        self.assertFalse(s.runner.state.running)
        self.assertFalse(s.runner.slot_of)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        self.assertEqual(s.runner.slots.available, s.runner.c.max_running)

    def test_continuation_reuses_context_but_has_an_independent_request_result(self):
        s = server(keep_idle=True)
        first, _ = s.submit([3], 2, 0)
        self.drain(s, retained=True)
        second, _ = s.submit([9], 3, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertNotEqual(first, second)
        self.assertEqual(s.take_result(first), [3, 3])
        self.assertEqual(s.take_result(second), [9, 9, 9])
        self.assertEqual(s.engine.opened, [0])
        self.assertEqual(s._conversations, {first: 0})

    def test_new_requests_evict_old_idle_conversations_without_exhausting_rows(self):
        s = server(keep_idle=True)
        jobs = [s.submit([i], 1, 0) for i in range(12)]
        self.drain(s, retained=True)
        self.assertEqual(set(s._conversations), {10, 11})
        for i, (request, _) in enumerate(jobs):
            self.assertEqual(s.take_result(request), [i])
        bad, event = s.submit([1], 1, 0, conversation=0)
        self.drain(s, retained=True)
        self.assertTrue(event.is_set())
        with self.assertRaises(RequestError) as error:
            s.take_result(bad)
        self.assertEqual(error.exception.status, 409)
        s.alive = False
        s.once()
        self.assertFalse(s.engine.tokens or s.runner.slot_of or s.runner.idle)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_parked_conversation_resumes_before_extension_and_returns_to_disk(self):
        s = server(keep_idle=True, tiered=True)
        first, _ = s.submit([3], 2, 0)
        self.drain(s, retained=True)
        row = s._conversations[first]
        self.assertTrue(s.runner.tiered.is_parked(row))
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        second, _ = s.submit([9], 2, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(second), [9, 9])
        self.assertTrue(s.runner.tiered.is_parked(row))
        self.assertEqual(s.engine.opened, [0])
        s.alive = False
        s.once()
        self.assertFalse(s.runner.tiered.tier.index or s.runner.tiered.parked)
        self.assertFalse(s.engine.tokens or s.runner.slot_of)

    def test_oversized_continuation_preserves_the_previous_idle_context(self):
        s = server(blocks=2, keep_idle=True)
        first, _ = s.submit([3] * 4, 2, 0)
        self.drain(s, retained=True)
        before = list(s.engine.tokens[0])
        bad, _ = s.submit([9] * 4, 2, 0, conversation=first)
        self.drain(s, retained=True)
        with self.assertRaises(RequestError):
            s.take_result(bad)
        self.assertEqual(s.engine.tokens[0], before)
        good, _ = s.submit([7], 1, 0, conversation=first)
        self.drain(s, retained=True)
        self.assertEqual(s.take_result(good), [7])

    def test_failed_resume_wakes_client_and_releases_parked_conversation_ownership(self):
        s = server(keep_idle=True, tiered=True)
        first, _ = s.submit([3], 2, 0)
        self.drain(s, retained=True)
        s.runner.tiered.tier.fail_promote = True
        request, event = s.submit([9], 2, 0, conversation=first)
        with self.assertRaisesRegex(OSError, 'read failed'):
            s.once()
        self.assertTrue(event.is_set())
        with self.assertRaises(RequestError):
            s.take_result(request)
        self.assertFalse(s.engine.tokens or s.runner.slot_of or s.runner.idle)
        self.assertFalse(s.runner.tiered.parked or s.runner.tiered.tier.index)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_busy_continuation_does_not_replace_the_active_turns_event(self):
        s = server(keep_idle=True)
        first, event = s.submit([3], 4, 0)
        s.once()
        bad, _ = s.submit([9], 1, 0, conversation=first)
        self.drain(s, retained=True)
        with self.assertRaises(RequestError) as error:
            s.take_result(bad)
        self.assertEqual(error.exception.status, 409)
        self.assertTrue(event.is_set())
        self.assertEqual(s.take_result(first), [3] * 4)

    def test_public_ids_outlive_rows_and_uncollected_results_keep_their_tokens(self):
        s = server()
        jobs = [s.submit([i], 1 + i % 3, 0) for i in range(40)]
        self.drain(s)
        self.assertEqual([r for r, _ in jobs], list(range(40)))
        self.assertTrue(all(ev.is_set() for _, ev in jobs))
        self.assertLess(max(s.engine.opened), 2)
        for i, _ in jobs:
            self.assertEqual(s.take_result(i), [i] * (1 + i % 3))
        self.assertFalse(s.pending or s.results)

    def test_admission_reserves_future_decode_growth_before_accepting_a_second_row(self):
        s = server(blocks=4)
        first, _ = s.submit([7] * 4, 13, 0)       # needs all four blocks eventually
        second, _ = s.submit([9] * 4, 13, 0)
        s.once()
        self.assertEqual(len(s._active), 1)
        self.assertEqual(len(s._waiting), 1)
        self.drain(s)
        self.assertEqual(s.take_result(first), [7] * 13)
        self.assertEqual(s.take_result(second), [9] * 13)

    def test_invalid_requests_never_enter_the_queue_or_model(self):
        s = server()
        cases = [([], 1, 0), ([True], 1, 0), ([1.0], 1, 0), ([256], 1, 0),
                 ([1], 0, 0), ([1], True, 0), ([1], 1, float('nan')),
                 ([1], 1, -1), ([1], 1, 10**1000), ([1], 1000, 0), ('text', 1, 0)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(RequestError):
                s.submit(*args)
        self.assertEqual(s.next_seq, 0)
        self.assertFalse(s.pending or s.engine.tokens)

    def test_backpressure_counts_results_until_the_caller_collects_them(self):
        s = server(max_pending=2)
        ids = [s.submit([i], 1, 0)[0] for i in range(2)]
        self.drain(s)
        with self.assertRaises(RequestError) as error:
            s.submit([3], 1, 0)
        self.assertEqual(error.exception.status, 503)
        s.take_result(ids[0])
        request, _ = s.submit([4], 1, 0)
        self.drain(s)
        self.assertEqual(s.take_result(request), [4])

    def test_concurrent_submit_allocates_unique_public_ids(self):
        s = server(max_pending=40)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            jobs = list(pool.map(lambda i: s.submit([i], 1, 0), range(40)))
        self.assertEqual(len({r for r, _ in jobs}), 40)
        self.drain(s)
        for i, (request, _) in enumerate(jobs):
            self.assertEqual(s.take_result(request), [i])

    def test_stop_cancels_partial_prefill_and_waiting_requests(self):
        s = server()
        jobs = [s.submit([3] * 16, 4, 0) for _ in range(4)]
        s.once()                                    # one request is midway through prefill
        self.assertIsNotNone(s.runner.state.in_prefill)
        s.alive = False
        self.assertFalse(s.once())
        self.assertIsNone(s.runner.state.in_prefill)
        self.assertFalse(s.engine.tokens or s.runner.slot_of)
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)
        for request, event in jobs:
            self.assertTrue(event.is_set())
            with self.assertRaises(RequestError):
                s.take_result(request)
        with self.assertRaises(RequestError):
            s.submit([1], 1, 0)

    def test_open_failure_wakes_all_clients_and_preserves_the_original_error(self):
        s = server()
        jobs = [s.submit([2], 3, 0) for _ in range(4)]
        s.engine.fail_open = True
        with self.assertRaisesRegex(RuntimeError, "open failed"):
            s.once()
        self.assertFalse(s.engine.tokens or s.runner.slot_of)
        self.assertTrue(all(event.is_set() for _, event in jobs))
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_kernel_failure_cleans_rows_and_wakes_pending_clients(self):
        s = server()
        jobs = [s.submit([2], 3, 0) for _ in range(4)]
        s.once()
        s.engine.fail_decode = True
        # The starvation valve may prefill the second request first.
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            for _ in range(3):
                s.once()
        self.assertFalse(s.engine.tokens or s.runner.slot_of)
        self.assertTrue(all(event.is_set() for _, event in jobs))
        self.assertEqual(s.runner.kv.available, s.runner.kv.num_blocks)

    def test_http_bad_json_and_invalid_requests_return_errors_then_valid_request_succeeds(self):
        s = server()
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/completions'
        def post(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=3) as response:
                return json.load(response)
        try:
            for body in (b'{', b'[]', b'{"ids":[],"max_tokens":1}', b'{"ids":[999],"max_tokens":1}'):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(body)
                self.assertEqual(error.exception.code, 400)
                self.assertIn('error', json.load(error.exception))
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(post, b'{"ids":[7],"max_tokens":2}')
                for _ in range(2000):
                    s.once()
                    if future.done():
                        break
                    threading.Event().wait(0.001)
                self.assertEqual(future.result(timeout=3)['ids'], [7, 7])
            self.assertFalse(s.pending or s.results)
        finally:
            httpd.shutdown()
            httpd.server_close()

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires PyTorch for LocalTP')
    def test_four_ranks_admit_reuse_and_stop_in_the_same_order(self):
        from engine.base.comm import LocalTP
        snapshots = []
        def rank_main(comm, _):
            s = server(comm=comm, keep_idle=True)
            if comm.rank == 0:
                jobs = [s.submit([i], 1 + i % 3, 0) for i in range(12)]
            for _ in range(60):
                s.once()
            if comm.rank == 0:
                snapshots.append([s.take_result(i) for i, _ in jobs])
                request, _ = s.submit([99], 2, 0, conversation=10)
            for _ in range(20):
                s.once()
            if comm.rank == 0:
                snapshots.append(s.take_result(request))
                s.alive = False
            s.once()
            return s.served, s.engine.opened, s.alive, s.runner.kv.available
        out = LocalTP(4).run(rank_main, None)
        self.assertTrue(all(row == out[0] for row in out))
        self.assertEqual(out[0][0], 13)
        self.assertFalse(out[0][2])
        self.assertEqual(snapshots[0], [[i] * (1 + i % 3) for i in range(12)])
        self.assertEqual(snapshots[1], [99, 99])


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class Tokenizer:
    """A vocabulary of 256: a character is its code point (below 256), a token decodes to its character."""
    def encode(self, text, add_special_tokens=True):
        return Encoded([ord(c) % 256 for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def chat_server(**kw):
    s = server(**kw)
    s.tok = Tokenizer()
    s.chat = lambda messages, kwargs: "".join(m["content"] for m in messages) + ("!" if kwargs.get("thinking") else "")
    s.model_name = "fake"
    return s


def drive(s, future, steps=2000):
    for _ in range(steps):
        s.once()
        if future.done():
            return future.result(timeout=3)
        threading.Event().wait(0.001)
    return future.result(timeout=3)


class ChatDoorTests(unittest.TestCase):
    """The OpenAI dialect over the fake engine (which repeats the prompt's last token)."""

    def test_models_metrics_and_health(self):
        s = chat_server()
        httpd = s._serve_http()
        base = f'http://127.0.0.1:{httpd.server_port}'
        try:
            models = json.load(urllib.request.urlopen(base + '/v1/models', timeout=3))
            self.assertEqual([m['id'] for m in models['data']], ['fake'])
            self.assertEqual(json.load(urllib.request.urlopen(base + '/health', timeout=3)), {'status': 'ok'})
            text = urllib.request.urlopen(base + '/metrics', timeout=3).read().decode()
            self.assertIn('vllm:request_success_total{engine="st"} 0\n', text)
            self.assertIn('vllm:num_requests_running{engine="st"} 0\n', text)
            self.assertIn('vllm:spec_decode_num_draft_tokens_total{engine="st"} 0\n', text)
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_chat_completion_whole(self):
        s = chat_server()
        s.engine.eos = {ord('b')}                    # 'b' ends a generation
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def post(body):
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=3) as r:
                return json.load(r)
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(post, {"model": "m", "messages": [{"role": "user", "content": "ab"}], "max_tokens": 3}))
            self.assertEqual(out['object'], 'chat.completion')
            # the fake engine runs to its limit regardless of eos; the door names the ending by the last token
            self.assertEqual(out['choices'][0]['message'], {'role': 'assistant', 'content': 'bbb'})
            self.assertEqual(out['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(out['usage'], {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5})
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = drive(s, pool.submit(post, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 3}))
            self.assertEqual(out['choices'][0]['message']['content'], 'yyy')
            self.assertEqual(out['choices'][0]['finish_reason'], 'length')
            for body in ({"messages": []}, {"messages": [{"role": "user"}]}, {"messages": "hi"}, {"messages": [{"role": "user", "content": "a"}], "chat_template_kwargs": 3}):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(body)
                self.assertEqual(error.exception.code, 400)
            self.assertFalse(s.pending or s.results or s._streams)
            self.assertIn('vllm:request_success_total{engine="st"} 2\n', s.metrics())
            self.assertIn('vllm:generation_tokens_total{engine="st"} 6\n', s.metrics())
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_chat_completion_streams_by_token_with_reasoning_split(self):
        s = chat_server()
        s.reasoning_end = ord('y')                  # the fake engine repeats the last prompt token: 'y'... so with
        httpd = s._serve_http()                     # reasoning_end = 'y' the first token closes the reasoning and the rest is content
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def stream(body):
            events = []
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode()), timeout=5) as r:
                self.assertEqual(r.headers['Content-Type'], 'text/event-stream')
                for raw in r:
                    line = raw.decode().strip()
                    if line.startswith('data:'):
                        events.append(line[5:].strip())
            return events
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                events = drive(s, pool.submit(stream, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 4, "stream": True,
                                                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": True}}))
            self.assertEqual(events[-1], '[DONE]')
            chunks = [json.loads(e) for e in events[:-1]]
            self.assertTrue(all(c['object'] == 'chat.completion.chunk' for c in chunks))
            deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
            self.assertEqual(deltas[0], {'role': 'assistant', 'content': ''})
            # the rendered prompt is "xy!" (thinking kwarg honoured), so the engine repeats '!': 4 tokens; the reasoning end
            # ('y') never comes, so all of it is still reasoning -- each token in its own chunk as the loop steps
            self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), '!!!!')
            self.assertEqual(''.join(d.get('content', '') for d in deltas), '')
            self.assertGreaterEqual(sum(1 for d in deltas if d.get('reasoning_content')), 2)
            self.assertEqual([c['choices'][0]['finish_reason'] for c in chunks if c['choices']][-1], 'length')
            self.assertEqual(chunks[-1]['usage'], {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7})
            self.assertFalse(s._streams or s._sent or s.pending or s.results)
            # reasoning: prompt "xy" repeats 'y' = reasoning_end -> the first token closes the (empty) reasoning, then content 'yyy'
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                events = drive(s, pool.submit(stream, {"messages": [{"role": "user", "content": "xy"}], "max_tokens": 4, "stream": True}))
            chunks = [json.loads(e) for e in events[:-1]]
            deltas = [c['choices'][0]['delta'] for c in chunks if c['choices']]
            self.assertEqual(''.join(d.get('content', '') for d in deltas), 'yyy')
            self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), '')
            self.assertEqual(s.split([1, ord('y'), 2, 3]), ([1], [2, 3]))
            self.assertEqual(s.split([1, 2]), ([1, 2], []))
        finally:
            httpd.shutdown(); httpd.server_close()

    def test_streaming_client_is_released_when_the_engine_dies(self):
        s = chat_server()
        httpd = s._serve_http()
        url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
        def stream():
            with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 5, "stream": True}).encode()), timeout=5) as r:
                return [l.decode().strip() for l in r if l.strip()]
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(stream)
                for _ in range(200):
                    if s._streams:
                        break
                    threading.Event().wait(0.001)
                s.once()
                s.engine.fail_decode = True
                with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                    for _ in range(3):
                        s.once()
                lines = future.result(timeout=5)
            self.assertTrue(any('"error"' in l for l in lines), lines)
            self.assertFalse(s._streams or s.pending)
        finally:
            httpd.shutdown(); httpd.server_close()


if __name__ == '__main__':
    unittest.main()
