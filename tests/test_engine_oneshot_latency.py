"""Exercise the production latency sampler's event/graph lifecycle without CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LatencyTests(unittest.TestCase):
    def test_sampler_initializes_events_before_timing_and_waits_once_per_cell(self):
        trace, graphs, operations = [], [], []
        cuda = SimpleNamespace(now=0., completed=0., capturing=None)

        class Event:
            def __init__(self, *, enable_timing):
                self.assert_timing = enable_timing
                self.initialized = False
                self.at = None
                trace.append('event-create')

            def record(self):
                if not self.initialized:
                    trace.append('event-init')
                    self.initialized = True
                self.at = cuda.now

            def synchronize(self):
                trace.append('event-wait')
                cuda.completed = self.at

            def elapsed_time(self, other):
                assert self.assert_timing and other.assert_timing
                assert cuda.completed >= other.at
                return other.at - self.at

        class Graph:
            def __init__(self):
                assert all(graph.resets == 1 for graph in graphs)
                self.captured, self.replays, self.resets = 0, 0, 0
                graphs.append(self)

            def replay(self):
                trace.append('replay')
                self.replays += 1
                # Discard the warm replay's outlier; retained samples are 1..11 ms.
                cuda.now += 100. if self.replays == 1 else self.replays - 1

            def reset(self):
                self.resets += 1

        class Capture:
            def __init__(self, graph):
                self.graph = graph

            def __enter__(self):
                cuda.capturing = self.graph

            def __exit__(self, *args):
                cuda.capturing = None

        def sync():
            trace.append('warmup-wait')
            cuda.completed = cuda.now

        def op(name):
            def run(x):
                operations.append(name)
                if cuda.capturing is not None:
                    cuda.capturing.captured += 1
            return run

        cuda.Event, cuda.CUDAGraph, cuda.graph, cuda.synchronize = Event, Graph, Capture, sync
        torch = SimpleNamespace(cuda=cuda, bfloat16='bf16', int64='int64',
                                zeros=lambda shape, **kw: SimpleNamespace(
                                    numel=lambda: shape[0] * shape[1] if isinstance(shape, tuple) else shape))
        control = object()

        def barrier(*, group):
            self.assertIs(group, control)
            trace.append('barrier')

        tree = ast.parse((ROOT / 'engine/kernels/oneshot/__init__.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OneShot')
        cls.body = [n for n in cls.body if isinstance(n, ast.Assign)
                    or isinstance(n, ast.FunctionDef) and n.name == '_sample_latency']
        from engine.kernels.cells import ONESHOT_CONSUMER_MAX_ELEMENTS
        namespace = dict(torch=torch, dist=SimpleNamespace(barrier=barrier),
                         CONSUMER_MAX_ELEMENTS=ONESHOT_CONSUMER_MAX_ELEMENTS)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<production latency sampler>', 'exec'), namespace)
        sampler = namespace['OneShot']()
        sampler.hidden, sampler.control = 4096, control
        sampler.ext = SimpleNamespace(oneshot_ar_consumer=op('consumer'), oneshot_ar=op('ordinary'),
                                      oneshot_max_int64=op('max'))
        result = sampler._sample_latency()
        self.assertEqual(result, {name + suffix: value for name, _ in sampler.LATENCY_CELLS
                                  for suffix, value in (('', 375.), ('_p90', 625.))})
        # C=1's and C=2's sums take the consumer the serving dispatch takes; C=4's takes the ordinary kernel.
        self.assertEqual([name for name, _ in sampler.LATENCY_CELLS], ['sum_8rows', 'sum_16rows', 'sum_32rows', 'max_8keys'])
        self.assertEqual(operations, ['consumer'] * 38 + ['ordinary'] * 19 + ['max'] * 19)
        self.assertEqual(len(graphs), 4)
        for graph in graphs:
            self.assertEqual((graph.captured, graph.replays, graph.resets), (16, 12, 1))
        # Reuse 24 event handles across all cells. Their twelve replays run
        # without a host wait or allocation between individual samples.
        initialization = ['event-create'] * 24 + ['event-init'] * 24
        expected_cell = ['warmup-wait', 'barrier'] + ['replay'] * 12 + ['event-wait']
        self.assertEqual(trace, initialization + expected_cell * 4)


if __name__ == '__main__':
    unittest.main()
