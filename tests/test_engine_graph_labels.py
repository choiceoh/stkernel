"""CPU ABI fixtures for graph attribution; no CUDA library or device is opened."""
import ctypes as C
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.base import graph_labels as labels

# Expected exported names/signatures from the CUDA runtime API, rather than a
# fallback library exposing every historical symbol at once.
ABI = {
    12000: ('cudaStreamGetCaptureInfo_v2', 'cudaGraphGetEdges', False),
    12030: ('cudaStreamGetCaptureInfo_v3', 'cudaGraphGetEdges_v2', True),
    12080: ('cudaStreamGetCaptureInfo_v3', 'cudaGraphGetEdges_v2', True),
    13000: ('cudaStreamGetCaptureInfo', 'cudaGraphGetEdges', True),
}


class LazyCaptureTests(unittest.TestCase):
    def test_unscoped_body_refreshes_the_initial_empty_frontier(self):
        from unittest.mock import Mock
        api = SimpleNamespace(frontier=Mock(side_effect=[(None, ()), (123, (7,))]),
                              topology=Mock(return_value=({7: "node7"}, {7: []})))
        with patch.object(labels, "_api", api), patch.object(labels, "ERRORS", []), patch.object(labels, "NODE_LABELS", {}):
            with labels.capture(42, "toy"):
                pass
            api.topology.assert_called_once_with(123)
            self.assertEqual(labels.ERRORS, [])
            self.assertEqual(labels.NODE_LABELS, {"node7": "toy/unmapped"})

    def test_empty_capture_does_not_query_a_null_graph(self):
        from unittest.mock import Mock
        api = SimpleNamespace(frontier=Mock(return_value=(None, ())), topology=Mock())
        with patch.object(labels, "_api", api), patch.object(labels, "ERRORS", []):
            with labels.capture(42, "empty"):
                pass
            api.topology.assert_not_called()
            self.assertEqual(labels.ERRORS, [])


class Function:
    def __init__(self, arity, fn):
        self.arity, self.fn = arity, fn

    def __call__(self, *args):
        assert len(args) == len(self.argtypes) == self.arity
        return self.fn(*args)


def set_value(pointer, kind, value):
    C.cast(pointer, C.POINTER(kind))[0] = value


def libraries(version, node_count=2, edge_count=1):
    capture_name, edges_name, modern = ABI[version]
    # Large handles catch accidental 32-bit pointer conversions.
    first, second, graph = (1 << 40) + 1, (1 << 40) + 2, (1 << 40) + 3
    deps = (C.c_void_p * 1)(second)
    metadata = (labels._EdgeData * 1)()
    metadata[0][0] = 1  # non-default edge data cannot be omitted from a query

    def runtime_version(out):
        set_value(out, C.c_int, version)
        return 0

    def frontier(stream, status, ident, out, dependencies, *tail):
        assert stream == 42
        set_value(status, C.c_int, 1)
        set_value(ident, C.c_ulonglong, 99)
        set_value(out, C.c_void_p, graph)
        set_value(dependencies, C.POINTER(C.c_void_p), C.cast(deps, C.POINTER(C.c_void_p)))
        if modern:
            assert tail[0] is not None, 'PDL frontier requires edge data'
            set_value(tail[0], C.POINTER(labels._EdgeData), C.cast(metadata, C.POINTER(labels._EdgeData)))
        set_value(tail[-1], C.c_size_t, 1)
        return 0

    def nodes(g, out, count):
        assert g == graph
        if out is not None:
            if not node_count:
                return 1
            for i, node in enumerate((first, second)[:node_count]):
                out[i] = node
        set_value(count, C.c_size_t, node_count)
        return 0

    def edges(g, start, end, *tail):
        assert g == graph
        if start is not None:
            if not edge_count:
                return 1
            start[0], end[0] = first, second
            if modern:
                assert tail[0] is not None, 'PDL topology requires edge data'
                tail[0][0][0] = 1
        set_value(tail[-1], C.c_size_t, edge_count)
        return 0

    def node_id(node, out):
        set_value(out, C.c_ulonglong, node + 100)
        return 0

    # Each fixture exposes only the symbols exported by that runtime ABI.
    rt = SimpleNamespace(cudaRuntimeGetVersion=Function(1, runtime_version),
                         cudaGraphGetNodes=Function(3, nodes))
    setattr(rt, capture_name, Function(7 if modern else 6, frontier))
    setattr(rt, edges_name, Function(5 if modern else 4, edges))
    return rt, SimpleNamespace(cuptiGetGraphNodeId=Function(2, node_id)), (first, second, graph)


class GraphLabelAbiTests(unittest.TestCase):
    def test_empty_and_edgeless_graphs_do_not_issue_zero_capacity_data_queries(self):
        for version in ABI:
            for count in (0, 1, 2):
                with self.subTest(version=version, nodes=count):
                    rt, cupti, (first, second, graph) = libraries(version, count, 0)
                    with patch.object(labels.ctypes.util, 'find_library', return_value='fixture'), \
                            patch.object(labels.C, 'CDLL', side_effect=[rt, cupti]):
                        api = labels.CUDA()
                    nodes = (first, second)[:count]
                    self.assertEqual(api.topology(graph),
                                     ({n: str(n+100) for n in nodes}, {n: [] for n in nodes}))

    def test_runtime_signatures_and_non_default_dependencies(self):
        for version in (12000, 12030, 12080, 13000):
            with self.subTest(version=version):
                rt, cupti, (first, second, graph) = libraries(version)
                with patch.object(labels.ctypes.util, 'find_library', return_value='fixture'), \
                        patch.object(labels.C, 'CDLL', side_effect=[rt, cupti]):
                    api = labels.CUDA()
                self.assertEqual(api.frontier(42), (graph, (second,)))
                self.assertEqual(api.topology(graph),
                                 ({first: str(first + 100), second: str(second + 100)},
                                  {first: [], second: [first]}))

    def test_unavailable_attribution_does_not_become_a_model_failure_context(self):
        original = RuntimeError('one-shot proxy stopped progressing')
        with patch.object(labels, '_api', None), patch.object(labels, 'ERRORS', []), \
                patch.object(labels, 'CUDA', side_effect=AttributeError('missing capture symbol')):
            with self.assertRaises(RuntimeError) as raised:
                with labels.capture(42, 'bounded/test'):
                    raise original
            self.assertIs(raised.exception, original)
            self.assertIsNone(original.__context__)
            self.assertEqual(labels.ERRORS, [dict(graph='bounded/test', error='missing capture symbol')])


class GraphLabelCudaTests(unittest.TestCase):
    def test_single_copy_graph_leaves_the_next_shape_launch_healthy(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires CUDA')
        if not torch.cuda.is_available():
            self.skipTest('requires admitted CUDA')
        from engine.base.graphs import DecodeGraphs
        from engine.kernels.bounded_graph import append_child
        for retained in (False, True):
            values = {n: torch.arange(n*2, device='cuda').view(n, 2) for n in (4, 3)}
            before = len(labels.ERRORS)
            graphs = DecodeGraphs(lambda x: x.clone(), lambda n, t: values[n], [(4, 2), (3, 2)],
                                  append_child=append_child if retained else None)
            try:
                for n in (4, 3):
                    actual = graphs.run((n, 2), lambda x: None)
                    torch.testing.assert_close(actual, values[n], rtol=0, atol=0)
                torch.ones(1, device='cuda').add_(1)
                torch.cuda.synchronize()
                self.assertEqual(labels.ERRORS[before:], [])
            finally:
                graphs.close()


if __name__ == '__main__':
    unittest.main()
