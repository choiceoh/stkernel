"""Missing evidence, concurrent dispatch, per-channel timing and partial durability."""
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import onepass
from onepass_recording import CURRENT, Run, group, steady_errors


class EvidenceTests(unittest.TestCase):
    def test_real_http_recording_blocks_foreign_requests_and_persists_server_spans(self):
        from tests.test_engine_serve import chat_server
        from engine.base.latency import Recorder
        with TemporaryDirectory() as root:
            server = chat_server()
            server.latency = server.runner.latency = Recorder(0, Path(root) / 'server')
            httpd = server._serve_http()
            stop = threading.Event()
            def loop():
                while not stop.is_set():
                    server.once()
                    time.sleep(.001)
            thread = threading.Thread(target=loop)
            thread.start()
            try:
                url = f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions'
                run = Run({}, Path(root) / 'ledger.jsonl', url)
                self.assertTrue(run.supported)
                run.begin('measure-c1')
                foreign = urllib.request.Request(url, data=b'{}', headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(foreign)
                self.assertEqual(error.exception.code, 409)
                timing = {}
                text, *_ = onepass.ask_stream(url, 'fake', 'ab', 4, timing)
                self.assertTrue(text)
                report = run.end()
                self.assertTrue(report['ranks'][0]['complete'])
                operations = {r['operation'] for r in report['ranks'][0]['rows']}
                self.assertTrue({'template', 'tokenize', 'queue', 'runner', 'admit'} <= operations)
                self.assertEqual(timing['cached_tokens'], 0)
                run.finish()
            finally:
                CURRENT.set(None)
                stop.set()
                thread.join(5)
                if server.latency.active:
                    server.latency.finish(server.latency.active['token'])
                httpd.shutdown()
                httpd.server_close()

    def report(self, width=4):
        return {'ranks': [dict(rank=0, complete=True, diagnostic=False, preparation_changed=False,
            preparation_after={'observers': ['triton.runtime.jit._do_compile']},
            rows=[dict(kind='host_step', phase='decode', rows=list(range(width)))])]}

    def test_missing_compile_prefix_and_batch_evidence_fail_closed(self):
        req = dict(cached_tokens=0, completion_tokens=20, finish_reason='length')
        self.assertEqual(steady_errors(self.report(), [req] * 4, 4), [])
        self.assertTrue(steady_errors(self.report(1), [req] * 4, 4))
        self.assertTrue(steady_errors({}, [req] * 4, 4))
        self.assertTrue(steady_errors(self.report(), [dict(req, cached_tokens=None)] * 4, 4))
        report = self.report()
        report['ranks'][0]['preparation_changed'] = True
        self.assertTrue(steady_errors(report, [req] * 4, 4))

    def test_concurrent_clients_and_aggregate_denominator(self):
        lock, active, maximum = threading.Lock(), 0, 0
        def ask(url, model, content, limit, timing, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.02)
            timing.update(started_monotonic=10, ended_monotonic=14, completion_tokens=8)
            with lock: active -= 1
            return 'answer', 1, 2, 8, 'length'
        got = group(None, ask, 'unused', 'test', dict(ctx=2, question=0, content='same', max_tokens=8), 4)
        self.assertEqual(maximum, 4)
        self.assertEqual(got['aggregate_output_tok_s'], 8)

    def test_partial_run_keeps_completed_requests_and_unique_run_directories(self):
        with TemporaryDirectory() as root, patch('urllib.request.urlopen', side_effect=OSError('offline')):
            out = Path(root) / 'ledger.jsonl'
            a = Run({}, out, 'http://localhost/v1/chat/completions')
            a.begin('measure-c1')
            a.request({'completion_tokens': 3}, '한국어', [])
            a.finish(RuntimeError('connection lost'))
            CURRENT.set(None)
            b = Run({}, out, 'http://localhost/v1/chat/completions')
            b.finish()
            self.assertNotEqual(a.path, b.path)
            self.assertIn('한국어', (a.path / 'requests.jsonl').read_text())
            records = [json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual(records[0]['recording']['status'], 'incomplete')
            self.assertEqual(records[1]['recording']['status'], 'complete')

    def test_first_reasoning_and_content_are_separate_and_salt_is_outside_workload_hash(self):
        rows, requests = [], []
        run = SimpleNamespace(token='test', request=lambda *args: rows.append(args))
        frames = [{'choices': [{'delta': {'reasoning_content': '근거'}}]},
                  {'choices': [{'delta': {'content': '답'}}]},
                  {'id': 'chatcmpl-1', 'choices': [{'finish_reason': 'length'}],
                   'usage': {'prompt_tokens': 8, 'completion_tokens': 4, 'prompt_tokens_details': {'cached_tokens': 0}}}]
        raw = ''.join('data: ' + json.dumps(r) + '\n' for r in frames).encode()
        def open_(req, **kwargs):
            requests.append(req)
            return io.BytesIO(raw)
        token = CURRENT.set(run)
        try:
            for _ in range(2):
                with patch('urllib.request.urlopen', open_), patch.object(onepass.time, 'monotonic', side_effect=[10, 11, 13, 14]):
                    onepass.ask_stream('http://localhost/v1/chat/completions', 'm', 'same', 4, {})
        finally: CURRENT.reset(token)
        a, b = rows[0][0], rows[1][0]
        self.assertEqual(a['first_channels_s'], {'reasoning_content': 1, 'content': 3})
        self.assertEqual(a['cached_tokens'], 0)
        self.assertEqual(a['workload_sha256'], b['workload_sha256'])
        self.assertNotEqual(a['request_sha256'], b['request_sha256'])
        self.assertEqual(requests[0].get_header('X-st-latency-token'), 'test')


if __name__ == '__main__':
    unittest.main()
