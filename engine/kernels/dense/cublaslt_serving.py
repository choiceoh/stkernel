"""Default head/FC reader: fixed cuBLAS algorithms, graph-owned activations.

Algorithms are declared at weight preparation. New eager row counts only create
and check layouts from that declaration; capture requires a warmed layout.
There is no GPU timing or DeepGEMM fallback in this reader.
"""
import torch
import triton
import triton.language as tl
from . import mxfp8
from .fp8 import require_disjoint


@triton.jit
def _activation_scales(S, MX, M: tl.constexpr, G: tl.constexpr):
    row = tl.program_id(0)*4 + tl.arange(0, 4)
    group = tl.program_id(1)
    scales = tl.load(S + row*G + group, row < M, 1.)
    mxfp8._publish(MX, scales, row, group, M, G, True, True)


def weight_nbytes(n, k, *, split_decode=False):
    direct = mxfp8.scale_bytes(n, k)
    return direct + (n*k + direct if split_decode else 0)


def _candidate(candidates):
    candidates = [c for c in candidates if c['workspace'] == 0 and c['split_k'] == 1
                  and c['reduction'] == 0 and max(c['alignment_'+v] for v in 'abcd') <= 16]
    if not candidates:
        raise RuntimeError('no supported zero-workspace cuBLAS serving algorithm')
    # Prefer the tested implementation; a different device may expose a
    # different valid heuristic seed. Either way the choice is fixed at boot.
    return min(candidates, key=lambda c: (not (c['id'] == 70 and c['tile'] == 20 and c['stages'] == 36), c['index']))


class Reader:
    def __init__(self, weight, *, split_decode=False, storage=None):
        from .cublaslt import _build
        from .cublaslt_split import PackedWeight
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('cuBLAS reader must be prepared before capture')
        q, scales = weight
        self.weight, self.split_decode = weight, split_decode
        n, k = q.shape
        self.n, self.k = n, k
        native = _build()
        self.context = native.Context(q.device.index, *torch.cuda.get_device_capability(q.device))
        self.mx_weight = mxfp8.pack_weight_scales(scales, n, k)
        self.split_weight = PackedWeight(weight, 5) if split_decode else None
        values = [self.mx_weight]
        if self.split_weight is not None:
            values.extend((self.split_weight.q, self.split_weight.scales))
        self.resident_bytes = weight_nbytes(n, k, split_decode=split_decode)
        if storage is not None:
            from engine.modules.packed_storage import consume
            values = list(consume(storage, values))
            self.mx_weight = values[0]
            if self.split_weight is not None:
                self.split_weight.q, self.split_weight.scales = values[1:]
        self.seeds, self.plans, self.executed = {}, {}, set()
        self.workspace = torch.empty(0, device=q.device, dtype=torch.uint8)
        for parts in ((1, 5) if split_decode else (1,)):
            kp = k//parts
            size = mxfp8.scale_bytes(8, kp)
            query = torch.full((size,) if parts == 1 else (parts, size), 127, device=q.device, dtype=torch.uint8)
            ws = self.mx_weight if parts == 1 else self.split_weight.scales
            plan = native.Plan(self.context, 8, n, kp, 0, query, ws, parts, parts != 1)
            selected = _candidate(plan.candidates())
            self.seeds[parts] = (plan, selected)
            self.plans[parts, 8] = (plan, selected['index'])

    def _plan(self, m, parts, scales):
        key = parts, m
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'cuBLAS rows {key} were not warmed before capture')
            seed, choice = self.seeds[parts]
            ws = self.mx_weight if parts == 1 else self.split_weight.scales
            self.plans[key] = (seed.with_rows(m, choice['index'], scales, ws), 0)
        return self.plans[key]

    def _out(self, m, out, *inputs):
        if out is None:
            return torch.empty((m, self.n), dtype=torch.bfloat16, device=self.weight[0].device)
        if (out.shape != (m, self.n) or out.dtype != torch.bfloat16 or out.device != self.weight[0].device
                or not out.is_contiguous() or out.data_ptr() % 16):
            raise ValueError('cuBLAS output requires its full padded aligned BF16 shape')
        for value in (*inputs, *self.weight, self.mx_weight):
            require_disjoint(out, value)
        if self.split_weight is not None:
            require_disjoint(out, self.split_weight.q)
            require_disjoint(out, self.split_weight.scales)
        return out

    def __call__(self, x, *, out=None, decode=False):
        if (x.ndim != 2 or x.shape[1] != self.k or x.shape[0] <= 0 or x.dtype != torch.bfloat16
                or x.device != self.weight[0].device or not x.is_contiguous()):
            raise ValueError('cuBLAS input requires contiguous BF16 [M,K]')
        m = x.shape[0]
        out = self._out(m, out, x)
        if decode and self.split_decode:
            from .cublaslt_split import _quantize, _reduce
            p, kp = 5, self.k//5
            q = torch.empty((p, m, kp), dtype=torch.float8_e4m3fn, device=x.device)
            s = torch.empty((p, mxfp8.scale_bytes(m, kp)), dtype=torch.uint8, device=x.device)
            partials = torch.empty((p, m, self.n), dtype=torch.float32, device=x.device)
            _quantize[(triton.cdiv(m, 4), self.k//128)](x, q, s.view(torch.int32), m, self.k, p,
                                                       bool(m % 128), num_warps=1)
            plan, index = self._plan(m, p, s)
            plan.run(index, q, self.split_weight.q, s, self.split_weight.scales, partials, self.workspace)
            _reduce[(triton.cdiv(m*self.n, 256),)](partials, out, m*self.n, p, num_warps=4)
            self.executed.add('split_decode')
        else:
            q, s = mxfp8.quantize(x, num_warps=1)
            plan, index = self._plan(m, 1, s)
            plan.run(index, q, self.weight[0], s, self.mx_weight, out, self.workspace)
            self.executed.add('direct')
        return out

    def project_quantized(self, q, scales, *, out=None):
        if (q.ndim != 2 or q.shape[1] != self.k or q.shape[0] <= 0 or q.dtype != torch.float8_e4m3fn
                or scales.shape != (q.shape[0], self.k//128) or scales.dtype != torch.float32
                or q.device != self.weight[0].device or scales.device != q.device
                or not q.is_contiguous() or not scales.is_contiguous()):
            raise ValueError('cuBLAS quantized input must match the block-128 FP8 recipe')
        m = q.shape[0]
        out = self._out(m, out, q, scales)
        mx = torch.empty(mxfp8.scale_bytes(m, self.k), device=q.device, dtype=torch.uint8)
        _activation_scales[(triton.cdiv(m, 4), self.k//128)](scales, mx.view(torch.int32), m, self.k//128, num_warps=1)
        plan, index = self._plan(m, 1, mx)
        plan.run(index, q, self.weight[0], mx, self.mx_weight, out, self.workspace)
        self.executed.add('direct')
        return out

    def report(self):
        return dict(backend='cublaslt', resident_bytes=self.resident_bytes, executed=sorted(self.executed),
                    algorithms={str(p): c for p, (_, c) in self.seeds.items()},
                    warmed_rows=sorted([p, m] for p, m in self.plans))
