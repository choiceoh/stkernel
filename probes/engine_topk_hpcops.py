"""HPC-Ops' sample-verified exact top-k against ST's own DSA selectors on GB10 -- component timing, never serving speed.

The paper map (docs/PAPER_MAP_20260917.html, 10절) named Tencent HPC-Ops' `topk_filtered` (#93, MIT,
arXiv:2609.08450) as an adoption candidate, then found the decode slot already taken: `st_dsa_select`
(#1010, #1025) reads a row once at 85% of the one-read floor on sm_120, so no exact selector can buy more than
~12 us a layer there. The open question is the long prefill, where `prefill_topk` (an SGLang radix derivative)
selects 512 of up to ~236K pools for 1024 query rows per pass and no ST record says how far it sits from the floor.

Arms, every one on the same fp32 logits and the same per-row horizon `ke` (columns >= ke[r] are invisible):
  decode  (rows 8/16, captured and replayed): st_dsa_select | hpcops | read floor
  prefill (rows 1024, eager):                 prefill_topk  | hpcops | read floor
The read floor is one `torch.amax` sweep over the visible logits' storage: a bound on any exact selector, which must
read every visible element at least once. Selections are compared as SETS with torch.topk over the masked row
(tie-free and plateau distributions reported apart; ties at the cut may be split differently by a different rule).
Only arms exact on the tie-free distributions are timed.

    bash bench/fleet.sh run --gpu topk-hpcops 15 'HPC-Ops top-k vs ST selectors' -- \
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes topk_hpcops
"""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

K = 512
VENDOR = Path(__file__).resolve().parent / 'vendor' / 'hpcops_topk'
DECODE = ((8, 8192), (8, 32768), (8, 131072), (16, 32768), (16, 131072), (16, 236032))
PREFILL = ((1024, 32768), (1024, 131072), (1024, 236032))
DISTRIBUTIONS = ('indexer', 'normal', 'plateau')

_HPC = None


def hpcops():
    """Build the vendored operator for sm_121a (its own CMake knows sm_90/100/103 only) and return torch.ops.hpc."""
    global _HPC
    if _HPC is None:
        from torch.utils.cpp_extension import load
        from engine.kernels.native_root import build_root
        build = build_root('probe-hpcops-topk')
        build.mkdir(parents=True, exist_ok=True)
        common = ['-O3', '-std=c++20', '-DHPC_TARGET_ARCH=121', '-DNDEBUG']
        load(name='probe_hpcops_topk',
             sources=[str(VENDOR / 'src/topk/entry.cc'), str(VENDOR / 'src/topk/topk_filtered.cu'),
                      str(VENDOR / 'src/utils/utils.cc')],
             extra_include_paths=[str(VENDOR)], extra_cflags=common,
             extra_cuda_cflags=common + ['--expt-relaxed-constexpr', '-gencode', 'arch=compute_121a,code=sm_121a'],
             build_directory=str(build), is_python_module=False, verbose=False)
        _HPC = torch.ops.hpc
    return _HPC


def logits_for(dist, rows, n, gen):
    if dist == 'indexer':
        # sum_h w_h relu(q.k) over a few heads: non-negative, a thin zero plateau, a long right tail
        heads = 8
        x = torch.zeros(rows, n, dtype=torch.float32, device='cuda')
        for _ in range(heads):
            x.add_(torch.randn(rows, n, generator=gen, device='cuda').clamp_min_(0) * torch.rand(rows, 1, generator=gen, device='cuda'))
        return x
    if dist == 'normal':
        return torch.randn(rows, n, generator=gen, device='cuda')
    # plateau: 64 levels, so the k-th value is tied with many columns
    return torch.randint(0, 64, (rows, n), generator=gen, device='cuda').float()


def horizons(kind, rows, n):
    if kind == 'decode':
        return (n - torch.arange(rows, dtype=torch.int32, device='cuda') * 3).clamp_min(K + 1).to(torch.int32)
    # prefill: the chunk's last 1024 query rows at a context of 4n tokens, pool 4
    start = 4 * n - rows
    return ((start + torch.arange(rows, device='cuda') + 1) // 4).clamp(K + 1, n).to(torch.int32)


def reference_sets(logits, ke):
    masked = logits.clone()
    cols = torch.arange(logits.shape[1], device=logits.device)
    masked.masked_fill_(cols[None, :] >= ke[:, None].long(), float('-inf'))
    return torch.topk(masked, K, dim=-1, sorted=False).indices.to(torch.int32)


def mismatched_rows(got, want):
    a = torch.sort(torch.where(got < 0, torch.iinfo(torch.int32).max, got), dim=-1).values
    b = torch.sort(want, dim=-1).values
    return int((a != b).any(-1).sum())


class Hpc:
    def __init__(self, rows, n):
        ops = hpcops()
        counters, workspace = ops.topk_filtered_workspace_size(rows, n)
        self.counters = torch.zeros(counters, dtype=torch.uint8, device='cuda')
        self.workspace = torch.empty(workspace, dtype=torch.uint8, device='cuda')
        self.valid = torch.tensor([rows], dtype=torch.int32, device='cuda')
        self.out = torch.empty(rows, K, dtype=torch.int32, device='cuda')
        self.ops = ops

    def __call__(self, logits, ke):
        return self.ops.topk_filtered(logits, ke, self.out, K, self.valid, self.counters, self.workspace)


def arms(kind, rows, n):
    from engine.kernels import decode_topk, prefill_topk
    hpc = Hpc(rows, n)
    floor_out = torch.empty(rows, dtype=torch.float32, device='cuda')

    def floor(logits, ke):
        return torch.amax(logits, dim=1, out=floor_out)
    if kind == 'decode':
        out = torch.empty(rows, K, dtype=torch.int32, device='cuda')
        own = ('st_dsa_select', lambda logits, ke: decode_topk.select(logits, ke, K, out))
    else:
        own = ('prefill_topk', lambda logits, ke: prefill_topk.select(logits, ke, K))
    return (own, ('hpcops', hpc), ('read_floor', floor))


def timed_eager(fn, logits, ke, repeats):
    for _ in range(3):
        fn(logits, ke)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(logits, ke)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples), min(samples)


def timed_graph(fn, logits, ke, repeats):
    from probes.engine_decode_fusions import _capture, _time
    graph, _ = _capture(lambda: fn(logits, ke))
    samples = [_time(graph, iterations=256) * 1e3 for _ in range(repeats)]
    return statistics.median(samples), min(samples)


def run(output=None):
    events = []

    def report(event, **values):
        row = dict(event=event, **values)
        events.append(row)
        print(json.dumps(row), flush=True)
        if output:
            Path(output).write_text(''.join(json.dumps(e) + '\n' for e in events))

    props = torch.cuda.get_device_properties(0)
    assert torch.cuda.get_device_capability() == (12, 1), 'requires GB10'
    report('device', name=props.name, torch=torch.__version__, cuda=torch.version.cuda,
           smem_per_block=getattr(props, 'shared_memory_per_block', None))
    gen = torch.Generator(device='cuda').manual_seed(20260917)
    for kind, shapes in (('decode', DECODE), ('prefill', PREFILL)):
        for rows, n in shapes:
            names = [name for name, _ in arms(kind, rows, n)]
            exact = {name: True for name in names if name != 'read_floor'}
            for dist in DISTRIBUTIONS:
                logits = logits_for(dist, rows, n, gen)
                ke = horizons(kind, rows, n)
                want = reference_sets(logits, ke)
                for name, fn in arms(kind, rows, n):
                    if name == 'read_floor':
                        continue
                    got = fn(logits, ke)
                    if got is None:
                        report('refused', kind=kind, rows=rows, n=n, arm=name)
                        exact[name] = False
                        continue
                    bad = mismatched_rows(got, want)
                    report('exactness', kind=kind, rows=rows, n=n, dist=dist, arm=name, mismatched_rows=bad)
                    if bad and dist != 'plateau':
                        exact[name] = False
                del want
            logits = logits_for('indexer', rows, n, gen)
            ke = horizons(kind, rows, n)
            visible_bytes = int(ke.long().sum()) * 4
            for name, fn in arms(kind, rows, n):
                if name != 'read_floor' and not exact.get(name, False):
                    report('skipped_inexact', kind=kind, rows=rows, n=n, arm=name)
                    continue
                if kind == 'decode':
                    median_us, best_us = timed_graph(fn, logits, ke, repeats=5)
                else:
                    median_us, best_us = timed_eager(fn, logits, ke, repeats=15)
                report('timing', kind=kind, rows=rows, n=n, arm=name, median_us=round(median_us, 2),
                       best_us=round(best_us, 2), visible_mib=round(visible_bytes / 2**20, 2),
                       gb_per_s=round(visible_bytes / 1e9 / (median_us / 1e6), 1))
            torch.cuda.empty_cache()
    return events


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else None)
