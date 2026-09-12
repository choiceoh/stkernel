"""CPU proofs for attribution, recording ownership, and actual scheduler width."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from engine.base.graph_labels import assign
from engine.base.latency import Recorder
from engine.base.latency_trace import attribute, summarize, union_us
from engine.base.stage_clock import StageClock


class AttributionTests(unittest.TestCase):
    def test_nested_scopes_and_fork_join_graph(self):
        parents = {1: [], 2: [1], 3: [1], 4: [2, 3], 5: [4]}
        labels = assign(parents, [('layer', (), (5,), 1), ('attention', (1,), (4,), 2)])
        self.assertEqual(labels, {1: 'layer', 2: 'attention', 3: 'attention', 4: 'attention', 5: 'layer'})

    def test_eager_and_graph_replay_attribution_do_not_guess_from_kernel_name(self):
        events = [dict(ph='X', cat='user_annotation', name='st.op/L2/moe', ts=0, dur=10, tid=1),
                  dict(ph='X', cat='cuda_runtime', name='launch', ts=2, dur=1, tid=1, args={'External id': 7}),
                  dict(ph='X', cat='kernel', name='same', ts=20, dur=6, args={'External id': 7}),
                  dict(ph='X', cat='kernel', name='same', ts=23, dur=8, args={'graph node id': 19}),
                  dict(ph='X', cat='kernel', name='same', ts=40, dur=2, args={})]
        rows = attribute({'traceEvents': events}, {'19': 'L3/kda'})
        self.assertEqual([r['operation'] for r in rows], ['L2/moe', 'L3/kda', 'unmapped'])
        self.assertEqual(union_us([(r['start_us'], r['start_us'] + r['duration_us']) for r in rows]), 13)
        self.assertEqual(sum(r['duration_us'] for r in rows), 16)
        self.assertNotIn('p95_us', summarize(rows)[0])

    def test_delayed_stage_sample_keeps_its_original_request_context(self):
        class Event:
            def query(self): return True
            def elapsed_time(self, other): return 2
        clock, rows = StageClock(), []
        clock._pending = [('forward', Event(), Event())]
        clock._pending_context = {'request': 'old'}
        clock._pending_sink = lambda name, seconds, context: rows.append((name, seconds, context))
        clock.context = {'request': 'new'}
        clock._drain()
        self.assertEqual(rows, [('forward', .002, {'request': 'old'})])

    def test_unfinished_stage_round_is_not_overwritten_by_a_new_request(self):
        class Event:
            def query(self): return False
        clock = StageClock(every=1)
        clock._torch = object()
        clock._pending = [('forward', Event(), Event())]
        clock._pending_context = {'request': 'old'}
        clock.context = {'request': 'new'}
        self.assertFalse(clock.step())
        self.assertEqual(clock._pending_context, {'request': 'old'})


class RecordingTests(unittest.TestCase):
    def test_immutable_run_and_token_ownership(self):
        with TemporaryDirectory() as root:
            rec = Recorder(0, root)
            rec.begin('first')
            with self.assertRaises(ValueError): rec.begin('other')
            with self.assertRaises(ValueError): rec.finish('other')
            with rec.step('decode', [0], [10], 1): pass
            report = rec.finish('first')
            self.assertTrue(report['complete'])
            self.assertEqual(report['steps'], {'prefill': 0, 'decode': 1})
            self.assertEqual(len((Path(root) / 'first/rank-0/latency.jsonl').read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError): rec.begin('first')
            with self.assertRaises(ValueError): rec.artifact('first', '../manifest.json', 0)
            rec.begin('second')
            rec.finish('second')

    def test_preparation_change_is_preserved_not_called_warm(self):
        from engine.base import latency
        with TemporaryDirectory() as root:
            rec = Recorder(0, root)
            rec.begin('test')
            with patch.object(latency, '_PREPARATIONS', latency._PREPARATIONS + 1):
                self.assertTrue(rec.finish('test')['preparation_changed'])

    def test_server_control_and_four_real_scheduler_rows(self):
        from tests.test_engine_serve import server
        with TemporaryDirectory() as root:
            s = server(rows=4)
            s.latency = s.runner.latency = Recorder(0, root)
            def control(op, token='mine'):
                waiting = {'event': threading.Event()}
                s.latency_replies['control'] = waiting
                s._control(('latency', dict(op=op, token=token, concurrency=4, _control_id='control')))
                self.assertTrue(waiting['event'].is_set())
                return waiting['reply']['ranks'][0]
            self.assertEqual(control('begin')['status'], 'recording')
            for i in range(4): s.submit([10 + i] * 4, 6, 0.0)
            for _ in range(100):
                s.once()
                if not s._active and not s._waiting: break
            self.assertIn('error', control('end', 'other'))
            report = control('end')
            self.assertTrue(report['complete'])
            widths = [len(r['rows']) for r in report['rows'] if r['kind'] == 'host_step' and r['phase'] == 'decode']
            self.assertIn(4, widths)
            self.assertEqual(len([r for r in report['rows'] if r.get('operation') == 'admit']), 4)
            self.assertIsNone(s.latency.active)


if __name__ == '__main__':
    unittest.main()
