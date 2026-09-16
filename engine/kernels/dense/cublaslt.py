"""Prepared MXFP8 cuBLASLt algorithms and stream-owned workspace generations."""
from functools import cache
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from weakref import WeakValueDictionary
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
GRAPH_UNROLL = 16  # keep Python replay submission gaps out of small-GEMM timing


class BF16Producer:
    """The same source for baseline and MX, with an explicitly bound fast path."""
    def __init__(self, source):
        self.source = source

    def __call__(self, mx, out, *, num_warps=4):
        from . import fp8, mxfp8
        if mx:
            return mxfp8.quantize(self.source, out=out, num_warps=num_warps)
        if num_warps != 4:
            raise ValueError('baseline producer requires 4 warps')
        return fp8.quantize(self.source, out=out)

    def bind(self, mx, *, out=None, num_warps=4):
        if mx:
            from .mxfp8 import bind_quantize
            return bind_quantize(self.source, out=out, num_warps=num_warps)
        if num_warps != 4:
            raise ValueError('baseline producer requires 4 warps')
        outputs = self(False, out)
        return lambda: self(False, outputs)


class PacketProducer:
    def __init__(self, source, local_rows, *, real_rows, routed=False):
        self.source, self.local_rows = source, local_rows
        self.real_rows, self.routed = real_rows, routed

    def __call__(self, mx, out, *, num_warps=4):
        from engine.kernels.prefill_collectives.consumer import quantize_gather
        return quantize_gather(self.source, self.local_rows, real_rows=self.real_rows,
                               routed=self.routed, mx=mx, out=out, num_warps=num_warps)

    def bind(self, mx, *, out=None, num_warps=4):
        if mx:
            from engine.kernels.prefill_collectives.consumer import bind_quantize_gather
            return bind_quantize_gather(self.source, self.local_rows,
                                        real_rows=self.real_rows, routed=self.routed, out=out, num_warps=num_warps)
        if num_warps != 4:
            raise ValueError('baseline producer requires 4 warps')
        outputs = self(False, out)
        return lambda: self(False, outputs)


def _bind_producer(producer, mx, *, out=None, num_warps=4):
    if isinstance(producer, (BF16Producer, PacketProducer)):
        return producer.bind(mx, out=out, num_warps=num_warps)
    if num_warps != 4:
        raise ValueError('custom producers do not expose a warp search')
    outputs = producer(mx, out)
    return lambda: producer(mx, outputs)


def _aligned(candidate, weight, activation, output):
    return all(tensor.data_ptr() % candidate['alignment_' + operand] == 0
               for operand, tensor in zip('abcd', (weight, activation, output, output)))


@dataclass(frozen=True)
class Choice:
    index: int | None = None
    workspace: int = 0
    reason: str = 'deep_gemm'
    producer_warps: int = 4
    split_parts: int = 1


def select_winner(brackets):
    """Require both A/B pairs to win by 2%; a noisy mean does not switch lanes.

    Rows are (index, workspace, producer_warps, B_before, A_first, A_second, B_after).
    All times include the same producer-to-BF16-output boundary.
    """
    eligible = []
    for index, workspace, warps, b0, a0, a1, b1 in brackets:
        if type(warps) is not int or warps not in (1, 2, 4):
            raise ValueError('invalid producer warp count in timing bracket')
        times = (b0, a0, a1, b1)
        if any(not isinstance(t, (float, int)) or not 0 < t < float('inf') for t in times):
            raise ValueError('timing brackets must contain positive finite times')
        if a0 < .98 * b0 and a1 < .98 * b1:
            eligible.append(((a0 + a1) / 2, workspace, index, warps))
    if not eligible:
        return Choice(reason='no_consistent_gain')
    _, workspace, index, warps = min(eligible)
    return Choice(index, workspace, 'measured_pipeline_gain', warps)


def _measure(fn, pool, *, repeats=4):
    """Time a chain of GPU operations, amortizing host graph-submission gaps."""
    fn()
    graph = torch.cuda.CUDAGraph()
    started = False
    try:
        graph.capture_begin(pool, capture_error_mode='thread_local')
        started = True
        for _ in range(GRAPH_UNROLL):
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
        milliseconds = start.elapsed_time(end) / (repeats * GRAPH_UNROLL)
    except BaseException as error:
        from engine.base.graphs import cleanup_after_error
        if started:
            cleanup_after_error(error, graph.capture_end, 'tuning capture end')
        cleanup_after_error(error, graph.reset, 'tuning graph reset')
        raise
    else:
        graph.reset()
        return milliseconds


def require_timing_target(identity, target='gb10'):
    if target == 'gb10' and identity['capability'] == (12, 1) and identity['sms'] == 48:
        return
    if target == 'sm120-probe' and identity['capability'] == (12, 0):
        return
    raise RuntimeError('algorithm timing is restricted to GB10 unless an SM120 probe is explicit')


class Bank:
    """Geometry plans shared by layers; scratch is isolated by CUDA stream.

    Old scratch generations stay owned because a captured graph may still
    reference them. Growth is geometric, bounded by twice the largest rounded allocation
    per stream, rather than 64 MiB for every layer or every shape.
    """
    def __init__(self, device, timing_target='gb10'):
        from engine.base.kernel_shape import bound
        self.device = device
        self.timing_target = timing_target
        self.native = _build()
        capability = tuple(bound().device.capability)
        self.context = self.native.Context(device, *capability)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.identity = dict(device=device, capability=capability, sms=sms, cublaslt=self.native.version(),
                             torch=torch.__version__, cuda=torch.version.cuda)
        self.plans, self.workspaces = {}, {}
        self.split_weights = WeakValueDictionary()
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
        target = getattr(self, 'timing_target', 'gb10')
        require_timing_target(self.identity, target)
        with self.lock, torch.cuda.device(self.device):
            m, n, k, *_ = key
            geometry = m, n, k
            if geometry not in self.plans:
                _, query_scales = producer(True, None)
                self.plans[geometry] = self.native.Plan(self.context, m, n, k, WORKSPACE_LIMIT,
                                                       query_scales, mx_weight)
            plan = self.plans[geometry]
            candidates = plan.candidates()
            record = dict(self.identity, shape=(m, n, k), producer=key[3], candidates=len(candidates),
                          timing_target=target,
                          search=plan.statistics(),
                          graph_unroll=GRAPH_UNROLL,
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
            record['choice'] = dict(index=choice.index, workspace=choice.workspace, reason=choice.reason,
                                    producer_warps=choice.producer_warps)
            return choice, record

    def _search(self, plan, candidates, weight, mx_weight, producer, record):
        from deep_gemm import fp8_gemm_nt
        from engine.kernels.deep_gemm import _initialize
        _initialize()
        base_producer, mx_producer = _bind_producer(producer, False), _bind_producer(producer, True)
        q_base, s_base = base_producer()
        q_mx, s_mx = mx_producer()
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
        # Each timing graph is reset before the next capture. Reusing a pool
        # token after its last graph dies trips the CUDA allocator's live-pool
        # assertion in Torch 2.13. Independent captures need no shared storage.
        pool = None

        def base():
            base_producer()
            fp8_gemm_nt((q_base, s_base), weight, baseline)

        def trial(index, produce=mx_producer):
            produce()
            plan.run(index, q_mx, weight[0], s_mx, mx_weight, candidate, scratch)

        base()
        if not bool(torch.isfinite(baseline).all()):
            raise RuntimeError('FP8 baseline is nonfinite during algorithm preparation')
        ranked = []
        # All admitted candidates are bounded at 192. One captured screening
        # sample narrows to three; only these pay for the full paired bracket.
        for c in candidates:
            if not _aligned(c, weight[0], q_mx, candidate):
                record['screened'].append(dict(c, milliseconds=None, numerics=False, status='alignment_mismatch'))
                continue
            fn = lambda: trial(c['index'])
            milliseconds = _measure(fn, pool, repeats=2)
            numerics = torch.allclose(candidate, baseline, rtol=.01, atol=.001)
            record['screened'].append(dict(c, milliseconds=milliseconds, numerics=bool(numerics)))
            if numerics:
                ranked.append((milliseconds, c['workspace'], c['index']))
        # Rank GEMMs with the existing four-warp producer, then measure the
        # full pipeline across producer layouts. A producer-only win need not
        # be a pipeline win: occupancy, stores and subsequent GEMM reads matter.
        finalists = sorted(ranked)[:3]
        warps = (4, 1, 2) if isinstance(producer, (BF16Producer, PacketProducer)) else (4,)
        record['producer_warps_searched'] = []
        record['bracket_columns'] = ['index', 'workspace', 'producer_warps', 'B_before', 'A_first', 'A_second', 'B_after']
        record['producer_checks'] = []
        reference_scales = s_mx.clone() if finalists else None
        for count in warps:
            if not finalists:
                break
            # Serial tuning reuses the same Q/scale addresses for all variants;
            # it does not retain three activation buffers on a large prefill.
            produce = (mx_producer if count == 4 else
                       _bind_producer(producer, True, out=(q_mx, s_mx), num_warps=count))
            q, s = produce()
            record['producer_warps_searched'].append(count)
            if q.data_ptr() != q_mx.data_ptr() or s.data_ptr() != s_mx.data_ptr():
                raise RuntimeError('producer search replaced its fixed storage')
            same = torch.equal(q_base.view(torch.uint8), q.view(torch.uint8))
            same_scales = torch.equal(reference_scales, s)
            record['producer_checks'].append(dict(warps=count, identical_fp8=bool(same), identical_scales=bool(same_scales)))
            if not same or not same_scales:
                raise RuntimeError('producer warp configuration changed the FP8 activation or scale bytes')
            for _, workspace, index in finalists:
                fn = lambda: trial(index, produce)
                samples = (_measure(base, pool), _measure(fn, pool), _measure(fn, pool), _measure(base, pool))
                if not torch.allclose(candidate, baseline, rtol=.01, atol=.001):
                    raise RuntimeError('producer/GEMM finalist failed its numerical gate')
                record['brackets'].append((index, workspace, count, *samples))
        return select_winner(record['brackets'])


@cache
def bank(device, timing_target='gb10'):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('cuBLAS context must be prepared before capture')
    return Bank(device, timing_target)


class PreparedProjection:
    """Explicitly prepared weight/shape/producer; calling it never searches.

    Preparation belongs outside model forwards and TP collectives. This class
    is exercised by the cuBLAS probe; serving cells require measured admission
    before replacing their existing DeepGEMM reader. There is no first-use
    timing or runtime exception fallback hidden in FP8Linear.
    """
    def __init__(self, weight, rows, producer, producer_key, *, mx_weight=None, timing_target='gb10'):
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
        self.owner = bank(q.device.index, timing_target)
        self.mx_weight = pack_weight_scales(scales, *q.shape) if mx_weight is None else mx_weight
        from .mxfp8 import scale_bytes
        if (self.mx_weight.dtype != torch.uint8 or self.mx_weight.device != q.device
                or self.mx_weight.shape != (scale_bytes(*q.shape),) or not self.mx_weight.is_contiguous()):
            raise ValueError('prepared MX weight scales do not match the bound weight')
        # Geometry plans can be shared, but a different weight must earn its
        # own numeric decision. Own these tensors for the decision's lifetime.
        self.key = rows, *q.shape, producer_key
        self.choice, self.record = self.owner.prepare(self.key, weight, self.mx_weight, producer)
        self.split = None
        from .cublaslt_split import prepare
        from engine.base.graphs import frozen_gc
        with self.owner.lock, torch.cuda.device(q.device):
            calling = torch.cuda.current_stream(q.device)
            self.owner.tuning_stream.wait_stream(calling)
            with torch.cuda.stream(self.owner.tuning_stream), frozen_gc():
                self.split, split_record = prepare(self, producer)
            calling.wait_stream(self.owner.tuning_stream)
        if split_record is not None:
            self.record['split_k'] = split_record
        if self.split is not None:
            self.record['direct_choice'] = self.record['choice']
            self.choice = Choice(self.split.index, self.split.workspace_bytes,
                                 'measured_split_pipeline_gain', 1, self.split.packed.parts)
            self.record['choice'] = vars(self.choice)

    def bind(self, producer, *, out=None, workspace=None):
        """Own fixed buffers and a private descriptor before capture/execution.

        Each binding owns its scratch by default, so independently replayed
        graphs cannot race through a warmup stream's shared workspace. A caller
        may supply shared scratch only when those bindings execute serially.
        Retain the binding for graph lifetime; storage addresses are immutable.
        """
        if getattr(self, 'split', None) is not None:
            return self.split.bind(producer, out=out, workspace=workspace)
        return BoundProjection(self, producer, out=out, workspace=workspace)

    def __call__(self, producer, *, out=None):
        if getattr(self, 'split', None) is not None:
            return self.bind(producer, out=out)()
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
            if isinstance(producer, (BF16Producer, PacketProducer)):
                q, scales = producer(True, None, num_warps=self.choice.producer_warps)
            else:
                if self.choice.producer_warps != 4:
                    raise ValueError('custom producers do not expose the selected warp configuration')
                q, scales = producer(True, None)
            self.owner.plans[self.key[:3]].run(self.choice.index, q, q_weight, scales, self.mx_weight,
                                              out, self.owner.workspace(self.choice.workspace))
        return out


class BoundProjection:
    """No allocation, shape search or descriptor mutation in repeated calls."""
    def __init__(self, prepared, producer, *, out=None, workspace=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('projection buffers must be bound before capture')
        self.prepared = prepared
        self.mx = prepared.choice.index is not None
        self.producer = _bind_producer(producer, self.mx, num_warps=prepared.choice.producer_warps)
        self.buffers = self.producer()
        weight = prepared.weight[0]
        shape = prepared.rows, weight.shape[0]
        self.out = torch.empty(shape, device=weight.device, dtype=torch.bfloat16) if out is None else out
        if (self.out.shape != shape or self.out.dtype != torch.bfloat16 or self.out.device != weight.device
                or not self.out.is_contiguous() or self.out.data_ptr() % 16):
            raise ValueError('bound projection requires its exact aligned BF16 output shape')
        q, scales = self.buffers
        from .mxfp8 import scale_bytes
        scale_shape = (scale_bytes(prepared.rows, weight.shape[1]),) if self.mx else (prepared.rows, weight.shape[1]//128)
        if (q.shape != (prepared.rows, weight.shape[1]) or q.dtype != torch.float8_e4m3fn
                or scales.shape != scale_shape or scales.dtype != (torch.uint8 if self.mx else torch.float32)
                or q.device != weight.device or scales.device != weight.device
                or not q.is_contiguous() or not scales.is_contiguous()):
            raise ValueError('bound producer does not match its prepared ABI')
        from .fp8 import require_disjoint
        for tensor in (*self.buffers, *prepared.weight, prepared.mx_weight):
            require_disjoint(self.out, tensor)
        if self.mx:
            self.workspace = (torch.empty(prepared.choice.workspace, device=weight.device, dtype=torch.uint8)
                              if workspace is None else workspace)
            self.native = prepared.owner.plans[prepared.key[:3]].bind(
                prepared.choice.index, q, weight, scales, prepared.mx_weight, self.out, self.workspace)
            self.matmul = self.native.run
        else:
            from deep_gemm import fp8_gemm_nt
            from engine.kernels.deep_gemm import _initialize
            _initialize()
            self.workspace = None
            self.matmul = lambda: fp8_gemm_nt(self.buffers, prepared.weight, self.out)

    def __call__(self):
        actual = self.producer()
        if len(actual) != 2 or any(a.data_ptr() != b.data_ptr() for a, b in zip(actual, self.buffers)):
            raise RuntimeError('bound producer replaced its fixed output storage')
        self.matmul()
        return self.out
