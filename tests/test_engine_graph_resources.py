"""A captured raw workspace address must outlive eager cache replacement."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from engine.base.graphs import DecodeGraphs


class Workspace:
    def __init__(self, rows):
        self.rows = rows


class Capture:
    def __init__(self):
        self.address = None
        self.reset_done = False

    def replay(self):
        assert self.address() is not None, 'captured workspace was freed'

    def reset(self):
        self.replay()
        self.reset_done = True


class GraphResourceTests(unittest.TestCase):
    def setUp(self):
        self.current = None
        self.cache = {}
        self.refs = []
        self.captures = []
        stream = SimpleNamespace(wait_stream=lambda _: None)
        self.cuda = SimpleNamespace(
            graph_pool_handle=object, Stream=lambda: stream, current_stream=lambda: stream,
            stream=lambda _: nullcontext(), synchronize=lambda: None,
            CUDAGraph=self.new_capture, graph=self.capture)

    def new_capture(self):
        graph = Capture()
        self.captures.append(graph)
        return graph

    @contextmanager
    def capture(self, graph, **kwargs):
        self.current = graph
        try:
            yield
        finally:
            self.current = None

    def step(self, rows):
        if not self.cache or self.cache['moe'].rows < rows:
            self.cache['moe'] = Workspace(rows)
            self.refs.append(weakref.ref(self.cache['moe']))
        if self.current is not None:
            # Like a CUDA kernel node, this records an address without owning it.
            self.current.address = weakref.ref(self.cache['moe'])
        return rows

    def build(self, **kwargs):
        return DecodeGraphs(self.step, lambda rows: rows, [(6,), (12,), (24,)],
                            resources=lambda: tuple(self.cache.values()), **kwargs)

    def test_growth_after_capture_preserves_every_shape_until_reset(self):
        with patch('engine.base.graphs.torch.cuda', self.cuda):
            graphs = self.build()
            self.step(80)                    # a later eager prefill grows the cache again
            self.assertEqual(len(graphs.resources), 3)
            for shape in ((6,), (24,), (12,), (6,)):
                graphs.run(shape, lambda _: None)
            self.assertTrue(all(ref() is not None for ref in self.refs))
            graphs.close()
            self.assertTrue(all(g.reset_done for g in self.captures))
            self.assertTrue(all(ref() is None for ref in self.refs[:-1]))
            self.assertIsNotNone(self.refs[-1]())  # eager cache keeps its own latest owner
            self.assertFalse(graphs.resources)

    def test_shared_workspace_is_retained_once(self):
        with patch('engine.base.graphs.torch.cuda', self.cuda):
            self.step(80)
            graphs = self.build()
            self.assertEqual(len(graphs.resources), 1)
            self.cache.clear()
            for graph in graphs.graphs.values():
                graph.replay()
            graphs.close()
            self.assertIsNone(self.refs[0]())

    def test_failed_capture_resets_before_releasing_previous_owners(self):
        class FailingMemory:
            def checkpoint(_, phase):
                if phase == 'decode/(24,)/captured':
                    raise MemoryError('test budget')
        with patch('engine.base.graphs.torch.cuda', self.cuda):
            with self.assertRaisesRegex(MemoryError, 'test budget'):
                self.build(memory=FailingMemory())
            self.assertTrue(all(g.reset_done for g in self.captures))
            self.assertTrue(all(ref() is None for ref in self.refs[:-1]))


if __name__ == '__main__':
    unittest.main()
