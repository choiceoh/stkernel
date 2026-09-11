"""Tensor-parallel linear layers (module), drop-in for what glm53_model calls.

The served GLM model file constructs exactly these, with exactly these
kwargs (read off overlay/modules/glm53_model, 44th ledger):

    ColumnParallelLinear(in, out, bias=False, quant_config=None, prefix="", disable_tp=False)
    RowParallelLinear(in, out, bias=False, quant_config=None, prefix="")
    MergedColumnParallelLinear(in, [outs], bias=False, quant_config=None, prefix="", disable_tp=False,
                               replicated_shard_ids=(), tp_size=None)
    ReplicatedLinear(in, out, bias=False, quant_config=None, prefix="")

and calls them as `out, _ = layer(x)` -- so forward returns (out, bias).
That contract is what makes re-hosting the model file an import change
rather than a rewrite (CHARTER §5).

TP is a `comm` object handed in (base/comm.py), never a global: at world 1
every collective is the identity, which is how the self-check runs a
simulated TP=2 by slicing the same weights two ways and summing.

Weights are plain bf16 here. Quantised paths (NVFP4 experts, fp8 dense)
live in modules/quant and modules/moe; `quant_config` is accepted and
recorded so a profile can route a layer to them by prefix.
"""
from __future__ import annotations

import torch
from torch import nn


class _Identity:
    world_size, rank = 1, 0
    def all_reduce(self, t): return t
    def all_gather(self, t, dim=-1): return t


class _Base(nn.Module):
    def __init__(self, in_features, out_features, bias, quant_config, prefix, comm):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.quant_config, self.prefix = quant_config, prefix
        self.comm = comm or _Identity()
        self.tp = self.comm.world_size
        self.rank = self.comm.rank

    def _param(self, out, inn, bias):
        self.weight = nn.Parameter(torch.empty(out, inn, dtype=torch.get_default_dtype()), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out, dtype=torch.get_default_dtype()), requires_grad=False) if bias else None


class ReplicatedLinear(_Base):
    def __init__(self, in_features, out_features, bias=False, quant_config=None, prefix="", comm=None):
        super().__init__(in_features, out_features, bias, quant_config, prefix, comm)
        self._param(out_features, in_features, bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), self.bias

    def load(self, full_weight):                # the whole tensor, every rank
        self.weight.data.copy_(full_weight)


class ColumnParallelLinear(_Base):
    """Output dim split across ranks; input replicated; no collective."""

    def __init__(self, in_features, out_features, bias=False, quant_config=None, prefix="",
                 disable_tp=False, comm=None):
        super().__init__(in_features, out_features, bias, quant_config, prefix, comm)
        self.tp = 1 if disable_tp else self.tp
        if out_features % self.tp:
            raise ValueError(f"{prefix}: out {out_features} not divisible by TP {self.tp}")
        self.out_local = out_features // self.tp
        self._param(self.out_local, in_features, bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), self.bias

    def load(self, full_weight):                # [out, in] -> this rank's rows
        self.weight.data.copy_(full_weight.narrow(0, self.rank * self.out_local, self.out_local))


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Several column-parallel outputs fused into one GEMM, each shard split
    by TP -- except `replicated_shard_ids`, which every rank holds whole
    (GLM's KDA carries f_a/g_a that way)."""

    def __init__(self, in_features, output_sizes, bias=False, quant_config=None, prefix="",
                 disable_tp=False, replicated_shard_ids=(), tp_size=None, comm=None):
        self.output_sizes = list(output_sizes)
        self.replicated = set(replicated_shard_ids)
        tp = 1 if disable_tp else (comm or _Identity()).world_size
        self.local_sizes = [s if i in self.replicated else s // tp for i, s in enumerate(self.output_sizes)]
        _Base.__init__(self, in_features, sum(self.output_sizes), bias, quant_config, prefix, comm)
        self.tp = tp
        self.out_local = sum(self.local_sizes)
        self._param(self.out_local, in_features, bias)

    def load_shard(self, shard_id, full_shard):  # full_shard: [output_sizes[shard_id], in]
        off = sum(self.local_sizes[:shard_id]); n = self.local_sizes[shard_id]
        src = full_shard if shard_id in self.replicated else full_shard.narrow(0, self.rank * n, n)
        self.weight.data[off:off + n].copy_(src)


class RowParallelLinear(_Base):
    """Input dim split across ranks; partial sums all-reduced."""

    def __init__(self, in_features, out_features, bias=False, quant_config=None, prefix="",
                 input_is_parallel=True, reduce_results=True, comm=None):
        super().__init__(in_features, out_features, bias, quant_config, prefix, comm)
        if in_features % self.tp:
            raise ValueError(f"{prefix}: in {in_features} not divisible by TP {self.tp}")
        self.in_local = in_features // self.tp
        self.reduce_results = reduce_results
        self._param(out_features, self.in_local, bias)

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight)
        if self.reduce_results and self.tp > 1:
            out = self.comm.all_reduce(out)
        return out, self.bias

    def load(self, full_weight):                # [out, in] -> this rank's columns
        self.weight.data.copy_(full_weight.narrow(1, self.rank * self.in_local, self.in_local))


def _selfcheck() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)

    class FakeComm:                              # two ranks, partial sums added by hand
        def __init__(self, rank): self.world_size, self.rank = 2, rank
        def all_reduce(self, t): return t        # the check sums the two ranks itself
        def all_gather(self, t, dim=-1): return t

    x = torch.randn(4, 64, device=dev); W = torch.randn(96, 64, device=dev); W2 = torch.randn(64, 96, device=dev)
    ref = torch.nn.functional.linear(x, W); ref2 = torch.nn.functional.linear(ref, W2)
    with torch.device(dev):
        cols = [ColumnParallelLinear(64, 96, comm=FakeComm(r)) for r in (0, 1)]
        rows = [RowParallelLinear(96, 64, comm=FakeComm(r)) for r in (0, 1)]
    for c, r in zip(cols, rows): c.load(W); r.load(W2)
    y = torch.cat([c(x)[0] for c in cols], dim=-1)
    assert torch.allclose(y, ref, atol=1e-2, rtol=1e-2), "column-parallel halves concatenate to the full GEMM"
    z = sum(r(y.narrow(-1, i * 48, 48))[0] for i, r in enumerate(rows))
    assert torch.allclose(z.float(), ref2.float(), atol=2e-1, rtol=2e-2), "row-parallel partial sums add to the full GEMM"
    # merged with a replicated shard, GLM's KDA shape
    with torch.device(dev):
        m = [MergedColumnParallelLinear(64, [32, 32, 8], replicated_shard_ids=(2,), comm=FakeComm(r)) for r in (0, 1)]
    A, B, C = (torch.randn(n, 64, device=dev) for n in (32, 32, 8))
    for mm in m:
        mm.load_shard(0, A); mm.load_shard(1, B); mm.load_shard(2, C)
    assert m[0].weight.shape == (16 + 16 + 8, 64)
    o0, o1 = m[0](x)[0], m[1](x)[0]
    assert torch.allclose(torch.cat([o0[:, :16], o1[:, :16]], -1), torch.nn.functional.linear(x, A), atol=1e-2, rtol=1e-2)
    assert torch.allclose(o0[:, 32:], o1[:, 32:]) and torch.allclose(o0[:, 32:], torch.nn.functional.linear(x, C), atol=1e-2, rtol=1e-2)
    with torch.device(dev):
        rep = ReplicatedLinear(64, 96); rep.load(W)
    assert torch.allclose(rep(x)[0], ref, atol=1e-2, rtol=1e-2) and rep(x)[1] is None
    print("  linear: column/row/merged(+replicated shard)/replicated == full GEMM under simulated TP=2 OK")


if __name__ == "__main__":
    _selfcheck()
