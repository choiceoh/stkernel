"""A long component comparison must not cycle onto its live shared stream."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class Cuda:
    class StreamHandle:
        def __init__(self, device, index):
            self.device, self.index = device, index
        def __eq__(self, other):
            return (self.device, self.index) == (other.device, other.index)
        def wait_stream(self, other):
            assert self.device == other.device

    def __init__(self):
        self.device, self.allocated, self.current = 0, {}, {}
    def current_device(self):
        return self.device
    def current_stream(self):
        return self.current.setdefault(self.device, self.StreamHandle(self.device, 0))
    def Stream(self, device=None):
        device = self.device if device is None else device
        count = self.allocated.get(device, 0)
        self.allocated[device] = count+1
        return self.StreamHandle(device, count % 2+1)  # bounded native stream pool
    @contextmanager
    def stream(self, stream):
        parent = self.current_stream()
        self.current[self.device] = stream
        try:
            yield
        finally:
            self.current[self.device] = parent
    def CUDAGraph(self):
        return object()
    @contextmanager
    def graph(self, graph, stream=None):
        with self.stream(stream or self.Stream()):
            yield


class CaptureTests(unittest.TestCase):
    def test_repeated_cells_keep_shared_branch_distinct_on_each_device(self):
        # Run the production helper with a deliberately small pool. The old
        # fresh-stream-per-cell helper collides with the shared branch.
        path = Path(__file__).resolve().parents[1] / 'probes/engine_decode_fusions.py'
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        cuda = Cuda()
        ns = dict(torch=NS(cuda=cuda))
        exec(compile(tree, str(path), 'exec'), ns)
        shared, calls = {}, []
        for device in (0, 1, 0, 1):
            cuda.device = device
            if device not in shared:
                shared[device] = cuda.Stream()
            def work():
                parent = cuda.current_stream()
                self.assertEqual(parent.device, device)
                self.assertNotEqual(parent, shared[device], 'shared work forked onto its own parent')
                calls.append((device, parent.index))
                return 'output'
            for _ in range(64):
                _, value = ns['_capture'](work)
                self.assertEqual(value, 'output')
                self.assertEqual(cuda.current_stream().index, 0)
        self.assertEqual(len(calls), 512)  # warmup and capture execute each cell


if __name__ == '__main__':
    unittest.main()
