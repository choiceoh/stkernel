"""Prepared K-partitioned MXFP8 GEMM: one producer, one batch, one FP32 sum.

The FC candidate is evaluated during explicit preparation. It does not change
FP8Linear dispatch and is never selected from a model forward.
"""
import torch
import triton
import triton.language as tl
from . import mxfp8


@triton.jit
def _quantize(X, Q, S, M: tl.constexpr, K: tl.constexpr, P: tl.constexpr):
    row = tl.program_id(0)*4 + tl.arange(0, 4)
    group = tl.program_id(1)
    part = group // (K//P//128)
    within = group % (K//P//128)
    col = group*128 + tl.arange(0, 128)
    x = tl.load(X + row[:, None]*K + col[None, :], row[:, None] < M, 0).to(tl.float32)
    scale, inverse = mxfp8._power2_scale(tl.maximum(tl.max(tl.abs(x), 1), 1e-4))
    tl.store(Q + part*M*(K//P) + row[:, None]*(K//P) + within*128 + tl.arange(0, 128)[None, :],
             (x*inverse[:, None]).to(tl.float8e4nv), row[:, None] < M)
    base = part*tl.cdiv(M, 128)*(K//P//128)*128
    # Reuse the one-warp scalar publisher; neutral padding belongs to binding.
    mxfp8._publish(S + base, scale, row, within, M, K//P//128, False, True)


@triton.jit
def _reduce(X, Y, SIZE: tl.constexpr, P: tl.constexpr):
    i = tl.program_id(0)*256 + tl.arange(0, 256)
    value = tl.full((256,), 0, tl.float32)
    for part in tl.static_range(P):
        value += tl.load(X + part*SIZE + i, i < SIZE, 0)
    tl.store(Y + i, value.to(tl.bfloat16), i < SIZE)


def _versions(weight):
    # Inference tensors intentionally have no version counter. They can be
    # repacked, but cannot participate in a mutation-aware shared cache.
    return tuple(None if t.is_inference() else t._version for t in weight)


class PackedWeight:
    """Retain original tensors and versions so shared repacks cannot go stale."""
    def __init__(self, weight, parts):
        q, scales = weight
        n, k = q.shape
        if type(parts) is not int or not 2 <= parts <= 16 or k % (parts*128):
            raise ValueError('split weights require 2..16 aligned K partitions')
        self.source = weight
        self.versions = _versions(weight)
        self.parts, self.shape = parts, (n, k)
        kp = k//parts
        self.q = q.reshape(n, parts, kp).permute(1, 0, 2).contiguous()
        source_scales = scales.reshape(n//128, parts, kp//128).permute(1, 0, 2).contiguous()
        self.scales = torch.stack([mxfp8.pack_weight_scales(source_scales[p], n, kp) for p in range(parts)])

    def require_current(self):
        if _versions(self.source) != self.versions:
            raise RuntimeError('split weight was modified after preparation')

    @property
    def nbytes(self):
        return self.q.numel() + self.scales.numel()


class SplitPlan:
    def __init__(self, owner, packed, rows, source):
        self.owner, self.packed, self.rows = owner, packed, rows
        self.index, self.workspace_bytes = 0, 0
        # Scale padding is neutral in both query and execution allocations.
        _, k = packed.shape
        query = torch.full((packed.parts, mxfp8.scale_bytes(rows, k//packed.parts)), 127,
                           dtype=torch.uint8, device=source.device)
        from .cublaslt import WORKSPACE_LIMIT
        self.native = owner.native.Plan(owner.context, rows, packed.shape[0], k//packed.parts,
                                        WORKSPACE_LIMIT, query, packed.scales, packed.parts, True)

    def bind(self, producer, *, out=None, workspace=None):
        return BoundSplit(self, producer, out=out, workspace=workspace)


class BoundSplit:
    """Private partials/workspace/descriptor; every replay updates real inputs."""
    def __init__(self, plan, producer, *, out=None, workspace=None):
        from .cublaslt import BF16Producer
        from .fp8 import require_disjoint
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('split projection must be bound before capture')
        if not isinstance(producer, BF16Producer):
            raise ValueError('split projection requires a BF16 producer')
        plan.packed.require_current()
        self.plan, self.source = plan, producer.source
        x = self.source
        n, k = plan.packed.shape
        m, p = plan.rows, plan.packed.parts
        if (x.shape != (m, k) or x.dtype != torch.bfloat16 or x.device != plan.packed.q.device
                or not x.is_contiguous() or x.data_ptr() % 16):
            raise ValueError('split producer does not match its prepared BF16 shape')
        self.out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device) if out is None else out
        if (self.out.shape != (m, n) or self.out.dtype != torch.bfloat16 or self.out.device != x.device
                or not self.out.is_contiguous() or self.out.data_ptr() % 16):
            raise ValueError('split projection requires its exact aligned BF16 output shape')
        self.workspace = (torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=x.device)
                          if workspace is None else workspace)
        for value in (x, *plan.packed.source, plan.packed.q, plan.packed.scales):
            require_disjoint(self.out, value)
            require_disjoint(self.workspace, value)
        require_disjoint(self.out, self.workspace)
        self.q = torch.empty((p, m, k//p), dtype=torch.float8_e4m3fn, device=x.device)
        self.scales = torch.full((p, mxfp8.scale_bytes(m, k//p)), 127, dtype=torch.uint8, device=x.device)
        self.words = self.scales.view(torch.int32)
        self.partials = torch.empty((p, m, n), dtype=torch.float32, device=x.device)
        self.native = plan.native.bind(plan.index, self.q, plan.packed.q, self.scales, plan.packed.scales,
                                       self.partials, self.workspace)
        self.geometry = m, n, k, p
        self()  # compile both fixed kernels before graph capture

    def __call__(self):
        m, n, k, p = self.geometry
        _quantize[(triton.cdiv(m, 4), k//128)](self.source, self.q, self.words, m, k, p, num_warps=1)
        self.native.run()
        _reduce[(triton.cdiv(m*n, 256),)](self.partials, self.out, m*n, p, num_warps=4)
        return self.out


def prepare(prepared, producer):
    """Compare the bounded FC candidate with the already selected whole-K path.

    Two complete brackets must each clear 2%. Screening never admits a lane.
    Repacked weights are weakly cached across row shapes, retained by winners.
    """
    from .cublaslt import BF16Producer, _measure, select_winner
    if not isinstance(producer, BF16Producer) or prepared.key[:3] not in ((8, 4096, 20480), (16, 4096, 20480)):
        return None, None
    owner = prepared.owner
    versions = _versions(prepared.weight)
    packed_key = (tuple((id(t), version) for t, version in zip(prepared.weight, versions)), 5)
    cacheable = all(version is not None for version in versions)
    packed = owner.split_weights.get(packed_key) if cacheable else None
    if packed is None:
        packed = PackedWeight(prepared.weight, 5)
        if cacheable:
            owner.split_weights[packed_key] = packed
    baseline = prepared.bind(producer)
    plan = SplitPlan(owner, packed, prepared.rows, producer.source)
    candidates = plan.native.candidates()
    record = dict(parts=5, search=plan.native.statistics(), screened=[], brackets=[],
                  weight_bytes=packed.nbytes, baseline_choice=vars(prepared.choice),
                  reason='no_supported_algorithm')
    baseline()
    ranked = []
    for candidate in candidates:
        plan.index, plan.workspace_bytes = candidate['index'], candidate['workspace']
        trial = plan.bind(producer)
        restored = trial.q.permute(1, 0, 2).reshape(prepared.rows, packed.shape[1])
        reference_q, reference_scales = producer(True, None)
        if not torch.equal(restored.view(torch.uint8), reference_q.view(torch.uint8)):
            raise RuntimeError('split producer changed the existing FP8 activation bytes')
        # Compare full MX metadata, including neutral padded rows, per K slice.
        expected = reference_scales.reshape(1, 5, -1).squeeze(0)
        if not torch.equal(trial.scales, expected):
            raise RuntimeError('split producer changed the existing MX activation scales')
        numeric = bool(torch.allclose(trial.out, baseline.out, rtol=.01, atol=.001))
        ms = _measure(trial, None, repeats=2) if numeric else None
        record['screened'].append(dict(candidate, numerics=numeric, milliseconds=ms))
        if numeric:
            ranked.append((ms, candidate['workspace'], candidate['index']))
        del trial, restored, reference_q, reference_scales
    eligible = []
    for _, workspace, index in sorted(ranked)[:3]:
        plan.index, plan.workspace_bytes = index, workspace
        trial = plan.bind(producer)
        brackets = []
        for _ in range(2):
            samples = [_measure(fn, None) for fn in (baseline, trial, trial, baseline)]
            if not torch.allclose(trial.out, baseline.out, rtol=.01, atol=.001):
                raise RuntimeError('split GEMM finalist failed its numerical gate')
            brackets.append((index, workspace, 1, *samples))
        record['brackets'].extend(brackets)
        if all(select_winner([row]).index is not None for row in brackets):
            eligible.append((sum(row[4]+row[5] for row in brackets)/4, workspace, index))
        del trial
    if not eligible:
        record['reason'] = 'no_consistent_gain' if candidates else record['reason']
        return None, record
    _, plan.workspace_bytes, plan.index = min(eligible)
    record['reason'] = 'measured_split_pipeline_gain'
    return plan, record
