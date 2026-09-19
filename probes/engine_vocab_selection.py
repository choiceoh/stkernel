"""Exact candidate packets and one-warp versus four-warp component timings.

The merged measurement includes the unchanged dense CUDA top-k and simulated
peer packets on one GPU. It excludes model execution and network transport.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import triton

from engine.kernels.common import vocab_candidates as kernels
from engine.modules.vocab import CandidateBuffer, topk


def control(x, start, valid, k, *, encoded=False, warps=4, resources=None):
    """The pre-change launch geometry at every level of the same selection."""
    rows = x.shape[0]
    block = min(2048, max(16, triton.next_power_of_2(valid)))
    segs = triton.cdiv(valid, block)
    out = torch.empty((rows, segs*k), dtype=torch.int64, device=x.device)
    if encoded:
        compiled = kernels._select[(rows, segs)](
            x, out, valid, x.stride(0), out.stride(0), k, segs, block, num_warps=warps)
    else:
        compiled = kernels._select_logits[(rows, segs)](
            x, out, x.stride(0), x.stride(1), valid, start, k, segs, block, num_warps=warps)
    if resources is not None:
        resources.append(dict(block=block, segments=segs, warps=warps, encoded=encoded,
                              shared_bytes=compiled.metadata.shared, registers=compiled.n_regs,
                              spills=compiled.n_spills, ptx_bar_sync=compiled.asm['ptx'].count('bar.sync')))
    return out if segs == 1 else control(out, 0, out.shape[1], k, encoded=True,
                                        warps=warps, resources=resources)


def capture(call, copies):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(copies):
            out = call()
    return graph, out


def paired(graphs, copies, brackets=4, replays=32):
    samples = {name: [] for name in graphs}
    # Interleave B/A/A/B, with the graph containing multiple calls so host
    # launch gaps do not dominate a selection lasting a few microseconds.
    for _ in range(brackets):
        for name in ('control', 'candidate', 'candidate', 'control'):
            graph = graphs[name][0]
            for _ in range(4):
                graph.replay()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end)*1000/(replays*copies))
    return {name: dict(median_us=statistics.median(values), samples_us=values)
            for name, values in samples.items()}


@torch.inference_mode()
def run(output=None):
    torch.set_num_threads(2)
    # A model-free probe with an absolute 512 MiB allocator budget.
    torch.cuda.set_per_process_memory_fraction((512 << 20)/torch.cuda.mem_get_info()[1])
    source = ('engine/kernels/common/vocab_candidates.py', 'engine/modules/vocab.py',
              'probes/engine_vocab_selection.py', 'probes/engine_kernel_check.py',
              'tests/test_engine_vocab_topk.py', 'tests/test_engine_vocab_packet.py')
    root = Path(__file__).resolve().parents[1]
    report = dict(scope=__doc__, device=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(), host=platform.node(),
                  torch=torch.__version__, torch_git=torch.version.git_version,
                  cuda=torch.version.cuda, triton=triton.__version__, cases=[],
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source})
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ('tests.test_engine_vocab_topk', 'tests.test_engine_vocab_packet'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('candidate packet numerical/replay gates failed or skipped')
    report['tests'] = dict(run=result.testsRun, skipped=len(result.skipped), passed=True)
    generator = torch.Generator(device='cuda').manual_seed(915950)
    width, k, world, decodable = 38720, 16, 4, 153880
    copies = 16
    for rows in (1, 7, 14, 28):
        full = torch.randn(rows, width*world, generator=generator, device='cuda').bfloat16()
        local = full[:, :width]  # A strided row view, including the first-stage strides.
        peers = torch.cat([
            kernels.select_logits(full[:, rank*width:(rank+1)*width], rank*width,
                                  min(width, decodable-rank*width), k)
            for rank in range(1, world)], dim=-1)
        comm = SimpleNamespace(world_size=world,
                               all_gather=lambda packet, dim=-1: torch.cat((packet, peers), dim=dim))
        packet_calls = dict(control=lambda: control(local, 0, width, k),
                            candidate=lambda: kernels.select_logits(local, 0, width, k))
        packet_graphs = {name: capture(call, copies) for name, call in packet_calls.items()}
        merge_graphs = {}
        workspaces = {name: CandidateBuffer(rows, width*world, world*k, 'cuda') for name in packet_calls}
        for name, selection in (('control', control), ('candidate', kernels.select_logits)):
            with patch.object(kernels, 'select_logits', selection):
                merge_graphs[name] = capture(
                    lambda: topk(local, comm, 0, k, decodable, workspace=workspaces[name]), copies)
        resources = {}
        for warps in (4, 1):
            resources[warps] = []
            control(local, 0, width, k, warps=warps, resources=resources[warps])
        try:
            # Mutate captured inputs and move maxima across segments and
            # ranks. Cutoff ties exercise the pinned dense CUDA ordering.
            for trial in range(4):
                full.normal_(generator=generator)
                full[:, decodable:] = 1000
                if trial == 1:
                    full[:, :64] = 7
                elif trial == 2:
                    full[:, width-9:width+30] = 8
                elif trial == 3:
                    full[:, 2*width+500:2*width+540] = 9
                    local[:, :6] = torch.tensor([0., -0., float('nan'), -float('nan'),
                                                float('inf'), -float('inf')], device='cuda')
                peers.copy_(torch.cat([
                    kernels.select_logits(full[:, rank*width:(rank+1)*width], rank*width,
                                          min(width, decodable-rank*width), k)
                    for rank in range(1, world)], dim=-1))
                expected_packet = kernels.pack(local, 0, width).topk(k, dim=-1).values
                masked = full.float()
                masked[:, decodable:] = float('-inf')
                expected_merge = masked.topk(k, dim=-1)
                before = full.view(torch.int16).clone()
                for graph, actual in packet_graphs.values():
                    graph.replay()
                    torch.testing.assert_close(actual, expected_packet, rtol=0, atol=0)
                for graph, actual in merge_graphs.values():
                    graph.replay()
                    torch.testing.assert_close(actual.values, expected_merge.values, rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(actual.indices, expected_merge.indices, rtol=0, atol=0)
                if not torch.equal(full.view(torch.int16), before):
                    raise AssertionError('selection mutated its source')
            # Timings use finite random logits; exceptional-value fixtures
            # above judge correctness, not a representative score distribution.
            full.normal_(generator=generator)
            peers.copy_(torch.cat([
                kernels.select_logits(full[:, rank*width:(rank+1)*width], rank*width,
                                      min(width, decodable-rank*width), k)
                for rank in range(1, world)], dim=-1))
            case = dict(rows=rows, draft_k=7, candidates=k, shard_width=width,
                        active_production_shape=rows in (7, 14), graph_cases=4, exact=True,
                        resources=resources, graph_copies=copies,
                        local_packet=paired(packet_graphs, copies),
                        simulated_merge=paired(merge_graphs, copies))
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
        finally:
            for graph, _ in (*packet_graphs.values(), *merge_graphs.values()):
                graph.reset()
        del packet_graphs, merge_graphs, workspaces
        torch.cuda.empty_cache()
    report.update(passed=True, peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report


# -- the greedy pick's two launches (Qwen3.8 carry D5) ---------------------------------------------------------------------
ARGMAX_WARPS = (4, 2, 1)                       # Triton's default -- every launch before D5 -- first
# (shard width, rows): Qwen3.8's 248,320 tokens over four ranks at its decode ladder (C x (K+1) tokens: 2..32), and
# GLM-5.3's shard at the rows its verify step picks
ARGMAX_CASES = ((62080, (2, 4, 8, 16, 32)), (38720, (1, 7, 14, 28)))


def argmax_launches(x, start, valid, warps):
    """engine/kernels/common/vocab_candidates.argmax_key's two launches at `warps`."""
    rows = x.shape[0]
    parts = triton.cdiv(valid, 1024)
    partials = torch.empty((rows, parts), dtype=torch.int64, device=x.device)
    kernels._argmax_partials[(rows, parts)](x, partials, x.stride(0), x.stride(1), valid, start, parts, 1024,
                                            num_warps=warps)
    if parts == 1:
        return partials.view(rows)
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    kernels._argmax_finish[(rows,)](partials, out, parts, triton.next_power_of_2(parts), num_warps=warps)
    return out


def argmax_reference(x, start, valid):
    """The key engine/modules/vocab.argmax computes off the device, its CPU branch line for line and on the CPU (where
    tests/test_engine_vocab.py holds it to torch.argmax): zeros and NaNs canonical, the lowest id of equal scores."""
    value, index = x[..., :valid].float().cpu().max(dim=-1)
    value = torch.where(value == 0, torch.zeros_like(value), value)
    value = torch.where(torch.isnan(value), torch.full_like(value, float("nan")), value)
    bits = value.contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
    return ((ordered << 32) | (0xffffffff - (index + start))).to(x.device)


def argmax_timings(graphs, copies, brackets=4, replays=32):
    """{warps: median us a call}: every bracket runs the arms forward then backward."""
    samples = {name: [] for name in graphs}
    order = list(graphs)
    for _ in range(brackets):
        for name in order + order[::-1]:
            graph = graphs[name][0]
            for _ in range(4):
                graph.replay()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end)*1000/(replays*copies))
    return {name: dict(median_us=round(statistics.median(values), 3), min_us=round(min(values), 3),
                       samples=len(values)) for name, values in samples.items()}


@torch.inference_mode()
def argmax_run(output=None):
    """argmax_key's launches at 4, 2 and 1 warps: the same packet (integer keys, exact at any warps), then the time of
    a call inside a captured graph. A kernel component on one device; no engine speed is claimed from it."""
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction((512 << 20)/torch.cuda.mem_get_info()[1])
    report = dict(scope=argmax_run.__doc__, device=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(), host=platform.node(), torch=torch.__version__,
                  cuda=torch.version.cuda, triton=triton.__version__, default_warps=kernels.ARGMAX_WARPS, cases=[])
    generator = torch.Generator(device='cuda').manual_seed(919005)
    copies = 16
    for width, row_counts in ARGMAX_CASES:
        decodable = width - 1000                                             # the last rank's cut: padded ids past it
        for rows in row_counts:
            full = torch.randn(rows, 2*width, generator=generator, device='cuda').bfloat16()
            local = full[:, ::2]                                             # a strided shard, as a head's view can be
            start = 3*width
            graphs = {warps: capture(lambda warps=warps: argmax_launches(local, start, decodable, warps), copies)
                      for warps in ARGMAX_WARPS}
            graphs['served'] = capture(lambda: kernels.argmax_key(local, start, decodable), copies)
            try:
                for trial in range(4):
                    full.normal_(generator=generator)
                    local[:, decodable:] = 1000                              # never decodable, never chosen
                    if trial == 1:
                        local[:, 5:40] = 7                                   # a tie: the lowest id
                    elif trial == 2:
                        local[:, 1020:1030] = 8                              # a tie across two partials
                    elif trial == 3:
                        local[:, :6] = torch.tensor([0., -0., float('nan'), -float('nan'), float('inf'),
                                                     -float('inf')], device='cuda')
                    expected = argmax_reference(local, start, decodable)
                    for name, (graph, actual) in graphs.items():
                        graph.replay()
                        if not torch.equal(actual, expected):
                            raise AssertionError(f'argmax packet at {name} warps differs from the reference key')
                full.normal_(generator=generator)
                case = dict(rows=rows, shard_width=width, valid=decodable, parts=triton.cdiv(decodable, 1024),
                            exact=True, graph_copies=copies, timings={str(k): v for k, v in
                                                                      argmax_timings(graphs, copies).items()})
                base = case['timings']['4']['median_us']
                case['over_four_warps'] = {k: round(v['median_us']/base, 4) for k, v in case['timings'].items()}
                report['cases'].append(case)
                print(json.dumps(case), flush=True)
            finally:
                for graph, _ in graphs.values():
                    graph.reset()
            del graphs
            torch.cuda.empty_cache()
    report.update(passed=True, peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report
