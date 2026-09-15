"""Prefix-boundary staging: exact copies and same-input component timings.

The control is the two staging kernels and grids in main da1c326d. The probe
uses GLM's 34 KDA layers, 16 local heads, 128x128 recurrent cells, K=7 and
6144 conv channels. It does not load a model or use network communication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import statistics
import unittest

import torch
import triton
import triton.language as tl

from engine.kernels.state import _stage_kda


def equal_bits(left, right):
    # Comparing bytes would allocate a boolean for every byte in a large
    # stage. Integer words retain all payload bits with a smaller temporary.
    word = torch.int32 if left.element_size() == 4 else torch.int16
    return torch.equal(left.view(word), right.view(word))


# Unmodified arithmetic and launch bodies from da1c326d:engine/kernels/state.py;
# kept only in this probe so the before/after comparison uses one runtime.
@triton.jit
def _reference_rec(RING, STAGE, ROFF, SOFF, SLOT, BEFORE, COUNT, BLOCK_TOKENS: tl.constexpr,
                   CELLS: tl.constexpr, CELL: tl.constexpr, RS: tl.constexpr,
                   SS: tl.constexpr, BLOCK: tl.constexpr):
    i, L, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot, before, count = tl.load(SLOT + i), tl.load(BEFORE + i), tl.load(COUNT + i)
    after = before + count
    boundary = (after // BLOCK_TOKENS) * BLOCK_TOKENS
    crossed = (count > 0) & (boundary > before)
    cell = (boundary - 1) % CELLS
    col = c * BLOCK + tl.arange(0, BLOCK)
    mask = (col < CELL) & crossed
    value = tl.load(RING + slot * RS + tl.load(ROFF + L) + cell * CELL + col, mask, other=0.0)
    tl.store(STAGE + slot * SS + tl.load(SOFF + L) + col, value, mask)


@triton.jit
def _reference_conv(RING, STAGE, ROFF, SOFF, SLOT, BEFORE, COUNT, BLOCK_TOKENS: tl.constexpr,
                    WIDTH: tl.constexpr, TAPS: tl.constexpr, CHANNELS: tl.constexpr,
                    RS: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr):
    i, L, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot, before, count = tl.load(SLOT + i), tl.load(BEFORE + i), tl.load(COUNT + i)
    after = before + count
    boundary = (after // BLOCK_TOKENS) * BLOCK_TOKENS
    crossed = (count > 0) & (boundary > before)
    ch = c * BLOCK + tl.arange(0, BLOCK)
    mask = (ch < CHANNELS) & crossed
    for j in tl.static_range(TAPS):
        cell = (boundary - TAPS + j) % WIDTH
        value = tl.load(RING + slot * RS + tl.load(ROFF + L) + ch * WIDTH + cell, mask, other=0.0)
        tl.store(STAGE + slot * SS + tl.load(SOFF + L) + ch * TAPS + j, value, mask)


class Fixture:
    layers, slots, cell, cells, channels, width, taps, block = 34, 2, 16*128*128, 8, 6144, 10, 3, 768

    def __init__(self):
        self.rec = torch.empty((self.slots, self.layers, self.cells, self.cell), device='cuda').normal_()
        self.conv = torch.empty((self.slots, self.layers, self.channels, self.width),
                                device='cuda', dtype=torch.bfloat16).normal_()
        self.outputs = {name: (torch.empty((self.slots, self.layers, self.cell), device='cuda'),
                               torch.empty((self.slots, self.layers, self.channels, self.taps),
                                           device='cuda', dtype=torch.bfloat16))
                        for name in ('control', 'candidate')}
        offsets = torch.arange(self.layers, device='cuda', dtype=torch.int64)
        self.roff, self.rsoff = offsets*self.cells*self.cell, offsets*self.cell
        self.coff, self.csoff = offsets*self.channels*self.width, offsets*self.channels*self.taps
        self.slot = torch.tensor([0, 1], device='cuda')
        self.before = torch.tensor([766, 768], device='cuda')
        self.count = torch.tensor([1, 7], device='cuda')

    def launch(self, name, rows):
        rstage, cstage = self.outputs[name]
        if name == 'control':
            _reference_rec[(rows, self.layers, triton.cdiv(self.cell, 1024))](
                self.rec, rstage, self.roff, self.rsoff, self.slot, self.before, self.count,
                self.block, self.cells, self.cell, self.rec.stride(0), rstage.stride(0), 1024)
            _reference_conv[(rows, self.layers, triton.cdiv(self.channels, 256))](
                self.conv, cstage, self.coff, self.csoff, self.slot, self.before, self.count,
                self.block, self.width, self.taps, self.channels, self.conv.stride(0), cstage.stride(0), 256)
        else:
            return _stage_kda[(rows, self.layers, 8)](
                self.rec, self.conv, rstage, cstage, self.roff, self.rsoff, self.coff, self.csoff,
                self.slot, self.before, self.count, self.block, self.cells, self.cell,
                self.width, self.taps, self.channels, self.rec.stride(0), rstage.stride(0),
                self.conv.stride(0), cstage.stride(0), 1024, 256)

    def verify(self, rows, slots, before, counts, graphs):
        for dst, values in zip((self.slot, self.before, self.count), (slots, before, counts)):
            dst.copy_(torch.tensor(values))
        for name, outputs in self.outputs.items():
            for tensor in outputs:
                tensor.fill_(-256.)
            graphs[name].replay()
        for control, candidate in zip(self.outputs['control'], self.outputs['candidate']):
            if not equal_bits(control, candidate):
                raise AssertionError('staged bytes differ from the original kernels')
        # Independently check the crossing cell and tap indices, using Python
        # int64-sized contexts and tensor indexing rather than either kernel.
        rstage, cstage = self.outputs['candidate']
        crossed = set()
        for slot, ctx, count in zip(slots[:rows], before[:rows], counts[:rows]):
            boundary = ((ctx+count)//self.block)*self.block
            if count <= 0 or boundary <= ctx:
                continue
            crossed.add(slot)
            expected_rec = self.rec[slot, :, (boundary-1) % self.cells]
            indices = torch.tensor([(boundary-self.taps+j) % self.width for j in range(self.taps)],
                                   device='cuda')
            expected_conv = self.conv[slot].index_select(-1, indices)
            if not equal_bits(rstage[slot], expected_rec):
                raise AssertionError('wrong recurrent boundary cell')
            if not equal_bits(cstage[slot], expected_conv):
                raise AssertionError('wrong convolution boundary taps')
        for slot in set(range(self.slots))-crossed:
            if not (bool((rstage[slot] == -256.).all()) and bool((cstage[slot] == -256.).all())):
                raise AssertionError('non-crossing slot changed')


def capture(call, copies):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(copies):
            call()
    return graph


def digest(tensor):
    # Bound host/GPU temporary storage while checking every input byte.
    result = hashlib.sha256()
    view = tensor.view(torch.uint8).flatten()
    for offset in range(0, view.numel(), 8 << 20):
        result.update(view[offset:offset+(8 << 20)].cpu().numpy().tobytes())
    return result.hexdigest()


def paired(graphs, copies):
    samples = {name: [] for name in graphs}
    for _ in range(4):
        for name in ('control', 'candidate', 'candidate', 'control'):
            graph = graphs[name]
            for _ in range(3):
                graph.replay()
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            for _ in range(16):
                graph.replay()
            b.record()
            b.synchronize()
            samples[name].append(a.elapsed_time(b)*1000/(16*copies))
    return {name: dict(median_us=statistics.median(values), samples_us=values)
            for name, values in samples.items()}


@torch.inference_mode()
def run(output=None):
    torch.set_num_threads(2)
    free, total = torch.cuda.mem_get_info()
    if free < 1 << 30:
        raise RuntimeError('bounded staging probe needs 1 GiB free before starting')
    torch.cuda.set_per_process_memory_fraction((768 << 20)/total)
    torch.manual_seed(915950)
    source = ('engine/kernels/state.py', 'tests/test_engine_boundary_stage.py',
              'probes/engine_boundary_stage.py', 'probes/engine_kernel_check.py')
    root = Path(__file__).resolve().parents[1]
    report = dict(scope=__doc__, baseline='da1c326d', device=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(), host=platform.node(),
                  torch=torch.__version__, torch_git=torch.version.git_version,
                  cuda=torch.version.cuda, triton=triton.__version__, cases=[],
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source})
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_engine_boundary_stage')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('boundary staging/restore gates failed or skipped')
    report['tests'] = dict(run=result.testsRun, skipped=len(result.skipped), passed=True)
    torch.cuda.empty_cache()
    f = Fixture()
    before_hash = (digest(f.rec), digest(f.conv))
    large = ((1 << 40)//768)*768
    copies = 8
    for rows in (1, 2):
        graphs = {name: capture(lambda name=name: f.launch(name, rows), copies) for name in f.outputs}
        compiled = f.launch('candidate', rows)
        report['resources'] = dict(registers=compiled.n_regs, spills=compiled.n_spills,
                                   shared_bytes=compiled.metadata.shared)
        try:
            for mode, slots, before, counts in (
                    ('no_crossing', [0, 1], [766, 768], [1, 7]),
                    ('one_crossing', [0, 1], [767, 766], [1, 1]),
                    ('both_crossing', [0, 1], [767, 1534], [2, 7]),
                    ('reordered_wrap', [1, 0], [2302, 3071], [7, 1]),
                    ('large_context', [0, 1], [large-1, large+767], [1, 7]),
                    ('zero_counts', [0, 1], [767, large-1], [0, 0])):
                f.verify(rows, slots, before, counts, graphs)
                case = dict(rows=rows, mode=mode, slots=slots[:rows], before=before[:rows],
                            counts=counts[:rows], layers=f.layers, heads=16, key_dim=128,
                            value_dim=128, draft_k=7, prefix_block=f.block, graph_copies=copies,
                            exact=True, ctas=dict(control=rows*f.layers*(256+24), candidate=rows*f.layers*8))
                if mode in ('no_crossing', 'one_crossing', 'both_crossing'):
                    case['timing'] = paired(graphs, copies)
                report['cases'].append(case)
                print(json.dumps(case), flush=True)
        finally:
            for graph in graphs.values():
                graph.reset()
    if before_hash != (digest(f.rec), digest(f.conv)):
        raise AssertionError('staging changed source ring bytes')
    report.update(passed=True, source_unchanged=True, peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report
