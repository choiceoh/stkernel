"""Prepared MXFP8 cuBLASLt algorithms and stream-owned workspace generations."""
from functools import cache
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
import torch


@cache
def _build():
    from torch.utils.cpp_extension import CUDA_HOME, load
    from engine.kernels.common.native_cache import prepare_cuda_sources
    from engine.kernels.native_root import build_root
    source = Path(__file__).with_suffix('.cpp')
    flags, links = ['-O2', '-std=c++17'], ['-lcublasLt']
    headers = [Path(CUDA_HOME) / 'include' / name for name in ('cublasLt.h', 'cublas_api.h')]
    identity = (flags, links, torch.__version__, torch.version.cuda,
                [(str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in headers])
    key, directory, sources = prepare_cuda_sources(build_root('cublaslt'), [source], identity)
    return load(name='st_cublaslt_' + key, sources=list(sources), extra_cflags=flags,
                extra_ldflags=links, with_cuda=True, build_directory=str(directory), verbose=False)


WORKSPACE_LIMIT = 64 << 20  # search ceiling; only a winner's demand stays resident


@dataclass(frozen=True)
class Choice:
    index: int | None = None
    workspace: int = 0
    reason: str = 'deep_gemm'


def select_winner(brackets):
    """Require both A/B pairs to win by 2%; a noisy mean does not switch lanes.

    Rows are (index, workspace, B_before, A_first, A_second, B_after).
    All times include the same producer-to-BF16-output boundary.
    """
    eligible = []
    for index, workspace, b0, a0, a1, b1 in brackets:
        times = (b0, a0, a1, b1)
        if any(not isinstance(t, (float, int)) or not 0 < t < float('inf') for t in times):
            raise ValueError('timing brackets must contain positive finite times')
        if a0 < .98 * b0 and a1 < .98 * b1:
            eligible.append(((a0 + a1) / 2, workspace, index))
    if not eligible:
        return Choice(reason='no_consistent_gain')
    _, workspace, index = min(eligible)
    return Choice(index, workspace, 'measured_pipeline_gain')


def _measure(fn, pool, *, repeats=4):
    """Warm and time graph replay on the caller's non-default tuning stream."""
    fn()
    graph = torch.cuda.CUDAGraph()
    started = False
    try:
        graph.capture_begin(pool, capture_error_mode='thread_local')
        started = True
        fn()
        graph.capture_end()
        started = False
        graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        milliseconds = start.elapsed_time(end) / repeats
    except BaseException as error:
        from engine.base.graphs import cleanup_after_error
        if started:
            cleanup_after_error(error, graph.capture_end, 'tuning capture end')
        cleanup_after_error(error, graph.reset, 'tuning graph reset')
        raise
    else:
        graph.reset()
        return milliseconds


class Bank:
    """Geometry plans shared by layers; scratch is isolated by CUDA stream.

    Old scratch generations stay owned because a captured graph may still
    reference them. Growth is geometric, bounded by twice the largest demand
    per stream, rather than 64 MiB for every layer or every shape.
    """
    def __init__(self, device):
        from engine.base.kernel_shape import bound
        self.device = device
        self.native = _build()
        capability = tuple(bound().device.capability)
        self.context = self.native.Context(device, *capability)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.identity = dict(device=device, capability=capability, sms=sms, cublaslt=self.native.version(),
                             torch=torch.__version__, cuda=torch.version.cuda)
        self.plans, self.workspaces = {}, {}
        self.owners = []
        self.lock = threading.RLock()
        self.tuning_stream = torch.cuda.Stream(device=device)

    def workspace(self, count):
        if type(count) is not int or not 0 <= count <= WORKSPACE_LIMIT:
            raise ValueError('cuBLAS workspace exceeds the prepared search bound')
        stream = torch.cuda.current_stream(self.device).cuda_stream
        previous = self.workspaces.get(stream)
        if previous is None or previous.numel() < count:
            size = 0 if count == 0 else max(256, 1 << (count - 1).bit_length())
            if size > WORKSPACE_LIMIT:
                raise ValueError('cuBLAS workspace exceeds the prepared search bound')
            previous = torch.empty(size, device=torch.device('cuda', self.device), dtype=torch.uint8)
            self.workspaces[stream] = previous
            self.owners.append(previous)
        return previous

    def prepare(self, key, weight, mx_weight, producer):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f'cuBLAS shape {key} was not warmed before capture')
        if self.identity['capability'] != (12, 1) or self.identity['sms'] != 48:
            raise RuntimeError('algorithm timing is restricted to GB10; other cards are numerical checks only')
        with self.lock, torch.cuda.device(self.device):
            m, n, k, *_ = key
            geometry = m, n, k
            if geometry not in self.plans:
                self.plans[geometry] = self.native.Plan(self.context, m, n, k, WORKSPACE_LIMIT)
            plan = self.plans[geometry]
            candidates = plan.candidates()
            record = dict(self.identity, shape=(m, n, k), producer=key[3], candidates=len(candidates),
                          screened=[], brackets=[], scope='warm captured component pipeline; not serving throughput')
            if not candidates:
                choice = Choice(reason='no_supported_algorithm')
            else:
                calling = torch.cuda.current_stream(self.device)
                self.tuning_stream.wait_stream(calling)
                # Tuning alone borrows the search ceiling. Captured serving
                # graphs subsequently retain only the chosen algorithm's need.
                from engine.base.graphs import frozen_gc
                with torch.cuda.stream(self.tuning_stream), frozen_gc():
                    choice = self._search(plan, candidates, weight, mx_weight, producer, record)
                calling.wait_stream(self.tuning_stream)
            record['choice'] = dict(index=choice.index, workspace=choice.workspace, reason=choice.reason)
            return choice, record

    def _search(self, plan, candidates, weight, mx_weight, producer, record):
        from deep_gemm import fp8_gemm_nt
        from engine.kernels.deep_gemm import _initialize
        _initialize()
        q_base, s_base = producer(False, None)
        q_mx, s_mx = producer(True, None)
        from .mxfp8 import scale_bytes
        m, _, k = record['shape']
        for q, s, dtype, shape in ((q_base, s_base, torch.float32, (m, k//128)),
                                    (q_mx, s_mx, torch.uint8, (scale_bytes(m, k),))):
            if (q.shape != (m, k) or q.dtype != torch.float8_e4m3fn or q.device != weight[0].device
                    or s.shape != shape or s.dtype != dtype or s.device != q.device
                    or not q.is_contiguous() or not s.is_contiguous() or q.data_ptr() % 16 or s.data_ptr() % 16):
                raise ValueError('prepared producer does not match its declared geometry and scale ABI')
        if not torch.equal(q_base.view(torch.uint8), q_mx.view(torch.uint8)):
            raise RuntimeError('MX producer changed the existing FP8 activation bytes')
        shape = q_base.shape[0], weight[0].shape[0]
        baseline = torch.empty(shape, dtype=torch.bfloat16, device=q_base.device)
        candidate = torch.empty_like(baseline)
        scratch = torch.empty(max(c['workspace'] for c in candidates), device=q_base.device, dtype=torch.uint8)
        pool = torch.cuda.graph_pool_handle()

        def base():
            producer(False, (q_base, s_base))
            fp8_gemm_nt((q_base, s_base), weight, baseline)

        def trial(index):
            producer(True, (q_mx, s_mx))
            plan.run(index, q_mx, weight[0], s_mx, mx_weight, candidate, scratch)

        base()
        if not bool(torch.isfinite(baseline).all()):
            raise RuntimeError('FP8 baseline is nonfinite during algorithm preparation')
        ranked = []
        # All admitted candidates are bounded at 96. One captured screening
        # sample narrows to three; only these pay for the full paired bracket.
        for c in candidates:
            fn = lambda: trial(c['index'])
            milliseconds = _measure(fn, pool, repeats=2)
            numerics = torch.allclose(candidate, baseline, rtol=.01, atol=.001)
            record['screened'].append(dict(c, milliseconds=milliseconds, numerics=bool(numerics)))
            if numerics:
                ranked.append((milliseconds, c['workspace'], c['index']))
        for _, workspace, index in sorted(ranked)[:3]:
            fn = lambda: trial(index)
            record['brackets'].append((index, workspace, _measure(base, pool), _measure(fn, pool),
                                       _measure(fn, pool), _measure(base, pool)))
        return select_winner(record['brackets'])


@cache
def bank(device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('cuBLAS context must be prepared before capture')
    return Bank(device)


class PreparedProjection:
    """Explicitly prepared weight/shape/producer; calling it never searches.

    Preparation belongs outside model forwards and TP collectives. This class
    is exercised by the cuBLAS probe; serving cells require measured admission
    before replacing their existing DeepGEMM reader. There is no first-use
    timing or runtime exception fallback hidden in FP8Linear.
    """
    def __init__(self, weight, rows, producer, producer_key, *, mx_weight=None):
        from .mxfp8 import pack_weight_scales
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('cuBLAS projection must be prepared before capture')
        q, scales = weight
        if (q.ndim != 2 or min(q.shape) <= 0 or q.dtype != torch.float8_e4m3fn or not q.is_cuda
                or not q.is_contiguous() or q.shape[0] % 128 or q.shape[1] % 128
                or scales.device != q.device or scales.dtype != torch.float32 or not scales.is_contiguous()
                or scales.shape != (q.shape[0]//128, q.shape[1]//128)
                or q.data_ptr() % 16 or scales.data_ptr() % 16 or type(rows) is not int or rows <= 0):
            raise ValueError('cuBLAS preparation requires a resident aligned FP8 weight and positive rows')
        if not isinstance(producer_key, str) or not producer_key:
            raise ValueError('cuBLAS preparation requires a named producer ABI')
        self.weight, self.rows, self.producer_key = weight, rows, producer_key
        self.owner = bank(q.device.index)
        self.mx_weight = pack_weight_scales(scales, *q.shape) if mx_weight is None else mx_weight
        from .mxfp8 import scale_bytes
        if (self.mx_weight.dtype != torch.uint8 or self.mx_weight.device != q.device
                or self.mx_weight.shape != (scale_bytes(*q.shape),) or not self.mx_weight.is_contiguous()):
            raise ValueError('prepared MX weight scales do not match the bound weight')
        # Geometry plans can be shared, but a different weight must earn its
        # own numeric decision. Own these tensors for the decision's lifetime.
        self.key = rows, *q.shape, producer_key
        self.choice, self.record = self.owner.prepare(self.key, weight, self.mx_weight, producer)

    def __call__(self, producer, *, out=None):
        q_weight = self.weight[0]
        shape = self.rows, q_weight.shape[0]
        if out is None:
            out = torch.empty(shape, device=q_weight.device, dtype=torch.bfloat16)
        if (out.shape != shape or out.dtype != torch.bfloat16 or out.device != q_weight.device
                or not out.is_contiguous() or out.data_ptr() % 16):
            raise ValueError('prepared projection output must be aligned contiguous BF16 of its exact shape')
        if self.choice.index is None:
            from deep_gemm import fp8_gemm_nt
            from engine.kernels.deep_gemm import _initialize
            _initialize()
            q, scales = producer(False, None)
            if (q.shape != (self.rows, q_weight.shape[1]) or q.dtype != q_weight.dtype
                    or scales.shape != (self.rows, q_weight.shape[1] // 128)
                    or scales.dtype != torch.float32 or q.device != out.device or scales.device != out.device
                    or not q.is_contiguous() or not scales.is_contiguous()):
                raise ValueError('prepared projection producer changed its ABI')
            from .fp8 import require_disjoint
            for tensor in (q, scales, *self.weight):
                require_disjoint(out, tensor)
            fp8_gemm_nt((q, scales), self.weight, out)
        else:
            q, scales = producer(True, None)
            self.owner.plans[self.key[:3]].run(self.choice.index, q, q_weight, scales, self.mx_weight,
                                              out, self.owner.workspace(self.choice.workspace))
        return out
