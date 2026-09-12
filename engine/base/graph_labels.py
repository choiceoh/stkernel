"""Name captured graph nodes without adding timing nodes or changing their work.

At a semantic boundary CUDA exposes the capture frontier. After the last op,
ancestors(end) - ancestors(begin) gives its nodes, including forked streams.
CUPTI's stable node IDs connect those nodes to later replay activity records.
Only CPU graph inspection occurs during capture. Missing support is explicit.
"""
import ctypes as C
import ctypes.util
from contextlib import contextmanager, nullcontext
from functools import wraps
from inspect import signature
from pathlib import Path
import threading

_local = threading.local()
NODE_LABELS = {}
ERRORS = []
CAPTURES = 0
_api = None


class CUDA:
    def __init__(self):
        def library(name):
            paths = [ctypes.util.find_library(name), f'lib{name}.so', f'lib{name}.so.13', f'lib{name}.so.12']
            import sys
            for root in sys.path:
                paths.extend(str(p) for p in Path(root).glob(f'nvidia/*/lib/lib{name}.so*'))
            paths.extend(str(p) for p in Path('/usr/local/cuda').glob(f'**/lib{name}.so*'))
            for path in paths:
                if path:
                    try:
                        return C.CDLL(path)
                    except OSError:
                        pass
            raise RuntimeError(f'{name} is unavailable for graph attribution')
        self.rt, self.cupti = library('cudart'), library('cupti')
        self.rt.cudaStreamGetCaptureInfo_v2.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_ulonglong),
            C.POINTER(C.c_void_p), C.POINTER(C.POINTER(C.c_void_p)), C.POINTER(C.c_size_t)]
        self.rt.cudaGraphGetNodes.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
        self.rt.cudaGraphGetEdges.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
        self.cupti.cuptiGetGraphNodeId.argtypes = [C.c_void_p, C.POINTER(C.c_ulonglong)]

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA graph inspection returned {code}')

    def frontier(self, stream):
        status, ident, graph = C.c_int(), C.c_ulonglong(), C.c_void_p()
        deps, n = C.POINTER(C.c_void_p)(), C.c_size_t()
        self.check(self.rt.cudaStreamGetCaptureInfo_v2(stream, C.byref(status), C.byref(ident),
                                                     C.byref(graph), C.byref(deps), C.byref(n)))
        if status.value != 1:
            raise RuntimeError('semantic boundary is outside an active CUDA capture')
        return graph.value, tuple(deps[i] for i in range(n.value))

    def topology(self, graph):
        n = C.c_size_t()
        self.check(self.rt.cudaGraphGetNodes(graph, None, C.byref(n)))
        if n.value > 50000:
            raise RuntimeError('graph exceeds 50000-node attribution bound')
        nodes = (C.c_void_p * n.value)()
        self.check(self.rt.cudaGraphGetNodes(graph, nodes, C.byref(n)))
        ids = {}
        for node in nodes:
            ident = C.c_ulonglong()
            self.check(self.cupti.cuptiGetGraphNodeId(node, C.byref(ident)))
            ids[node] = str(ident.value)
        n = C.c_size_t()
        self.check(self.rt.cudaGraphGetEdges(graph, None, None, C.byref(n)))
        a, b = (C.c_void_p * n.value)(), (C.c_void_p * n.value)()
        self.check(self.rt.cudaGraphGetEdges(graph, a, b, C.byref(n)))
        parents = {node: [] for node in ids}
        for source, target in zip(a, b):
            parents[target].append(source)
        return ids, parents


def assign(parents, spans):
    def ancestors(frontier):
        seen, todo = set(), list(frontier)
        while todo:
            node = todo.pop()
            if node not in seen:
                seen.add(node)
                todo.extend(parents.get(node, ()))
        return seen
    labels = {}
    # Broad scopes first; the most specific scope owns the final label.
    for name, start, end, depth in sorted(spans, key=lambda s: s[3]):
        for node in ancestors(end) - ancestors(start):
            labels[node] = name
    return labels


class Capture:
    def __init__(self, stream, label):
        self.stream, self.label = stream, label
        self.spans, self.stack, self.error = [], [], None

    def frontier(self):
        if self.error:
            return ()
        try:
            self.graph, deps = _api.frontier(self.stream)
            return deps
        except Exception as exc:
            self.error = str(exc)
            return ()

    def finish(self):
        try:
            if self.error:
                raise RuntimeError(self.error)
            ids, parents = _api.topology(self.graph)
            labels = assign(parents, self.spans)
            NODE_LABELS.update({ids[node]: labels.get(node, self.label + '/unmapped') for node in ids})
        except Exception as exc:
            if len(ERRORS) < 100:
                ERRORS.append(dict(graph=self.label, error=str(exc)))


@contextmanager
def capture(stream, label):
    global _api, CAPTURES
    CAPTURES += 1
    try:
        if _api is None:
            _api = CUDA()
    except Exception as exc:
        if not ERRORS:
            ERRORS.append(dict(graph=label, error=str(exc)))
        yield
        return
    held = getattr(_local, 'capture', None)
    cap = _local.capture = Capture(stream, label)
    cap.frontier()
    try:
        yield
    finally:
        cap.finish()
        _local.capture = held


@contextmanager
def scope(name):
    cap = getattr(_local, 'capture', None)
    profiling = getattr(_local, 'profiling', False)
    if cap:
        begin = cap.frontier()
        cap.stack.append(name)
        full, depth = cap.label + '/' + '/'.join(cap.stack), len(cap.stack)
    if profiling:
        import torch
        annotation = torch.profiler.record_function('st.op/' + name)
    else:
        annotation = nullcontext()
    try:
        with annotation:
            yield
    finally:
        if cap:
            cap.spans.append((full, begin, cap.frontier(), depth))
            cap.stack.pop()


def operation(name, *, layer_arg=None, name_arg=None):
    def decorate(fn):
        parameters = list(signature(fn).parameters)
        @wraps(fn)
        def call(*args, **kwargs):
            if not getattr(_local, 'capture', None) and not getattr(_local, 'profiling', False):
                return fn(*args, **kwargs)
            def argument(index):
                return args[index] if index < len(args) else kwargs.get(parameters[index], 'unknown')
            label = str(argument(name_arg)) if name_arg is not None else name
            if layer_arg is not None:
                label = f'L{argument(layer_arg)}/{label}'
            with scope(label):
                return fn(*args, **kwargs)
        return call
    return decorate


def profiling(active):
    _local.profiling = active
