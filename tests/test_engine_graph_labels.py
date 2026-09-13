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


class Function:
    def __init__(self, arity, fn):
        self.arity, self.fn = arity, fn

    def __call__(self, *args):
        assert len(args) == len(self.argtypes) == self.arity
        return self.fn(*args)


def set_value(pointer, kind, value):
    C.cast(pointer, C.POINTER(kind))[0] = value


def libraries(version):
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
            out[0], out[1] = first, second
        set_value(count, C.c_size_t, 2)
        return 0

    def edges(g, start, end, *tail):
        assert g == graph
        if start is not None:
            start[0], end[0] = first, second
            if modern:
                assert tail[0] is not None, 'PDL topology requires edge data'
                tail[0][0][0] = 1
        set_value(tail[-1], C.c_size_t, 1)
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


if __name__ == '__main__':
    unittest.main()
