"""Owned TP4 one-shot transport; all ranks prepare, connect and qualify together.

This transport serves aligned BF16 decode sums and small exact int64 MAX
and gather packets for vocabulary selection. Other shapes use NCCL. A failure cannot
change the selected collective on one rank.
Graph owners must be destroyed before close(), just like the NCCL group.
"""
import os
from pathlib import Path

import torch
import torch.distributed as dist

# The compiled cell, stated once in engine/kernels/cells.py: the element cap compiled in as MAXEL (64 rows of the
# measured hidden), the PDL consumer kernel's bound (1..32768 elements run its 12-CTA form), NPEER 3 (four ranks).
from engine.kernels.cells import (ONESHOT_CONSUMER_MAX_ELEMENTS as CONSUMER_MAX_ELEMENTS,
                                  ONESHOT_MAX_ELEMENTS as MAX_ELEMENTS, ONESHOT_WORLD as COMPILED_WORLD)


def _cell():
    """The bound kernel shape's collective geometry (world, hidden); the transport takes its
    row width from it instead of a literal, and refuses a world the kernel is not compiled for."""
    from engine.base.kernel_shape import bound
    comm = bound().comm
    if comm.world != COMPILED_WORLD:
        raise ValueError(f"one-shot is compiled for {COMPILED_WORLD} ranks; the bound kernel shape asks for {comm.world}")
    if comm.hidden > MAX_ELEMENTS or comm.hidden % 8:
        raise ValueError(f"one-shot rows must be a multiple of 8 elements within {MAX_ELEMENTS}, got hidden {comm.hidden}")
    return comm


class RankPackets:
    """One exchange, consumed immediately on its stream before another collective.

    The source stays alive while the descriptor references it. The same-stream
    GPU order is recorded in graphs; this host owner checks construction order.
    Peers cannot reach a ring overwrite without this rank's next collective.
    """
    def __init__(self, owner, source, descriptor):
        self.owner, self.source, self.descriptor = owner, source, descriptor
        self.stream = torch.cuda.current_stream(source.device).cuda_stream

    def consume(self, consumer):
        if self.owner.pending is not self:
            raise RuntimeError("rank packets are stale or already consumed")
        if torch.cuda.current_stream(self.source.device).cuda_stream != self.stream:
            raise RuntimeError("rank packets must be consumed on their exchange stream")
        try:
            result = consumer(self.source, self.descriptor)
        except BaseException:
            self.owner.packet_failed = True
            raise
        finally:
            self.owner.pending = None
        return result


def build():
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    root = Path(__file__).parent
    sources = [root/'dsv4_oneshot_ar.cu', root/'dsv4_oneshot_transport.h']
    flags = ['-O2', '-gencode', 'arch=compute_121a,code=sm_121a', f'-DMAXEL={MAX_ELEMENTS}']
    root = Path(os.environ.get('ST_ONESHOT_BUILD_ROOT', str(Path.home()/'.cache/st/oneshot')))
    ldflags = ['-libverbs']
    key, directory, staged = prepare_sources(root, sources, (flags, ldflags, torch.__version__, torch.version.cuda))
    return load(name='st_oneshot_'+key,sources=[staged[0]],extra_cuda_cflags=flags,
                extra_ldflags=ldflags,build_directory=str(directory),verbose=False)


class OneShot:
    def __init__(self, comm, addresses):
        cell = _cell()
        if comm.world_size != cell.world or len(addresses) != cell.world:
            raise ValueError(f'one-shot requires the explicit {cell.world}-rank address table')
        self.world, self.hidden = cell.world, cell.hidden
        self.ext = None
        self.control = dist.new_group(backend='gloo')
        self.closed = False
        self.pending = None
        self.packet_failed = False
        try:
            error = None
            try:
                self.ext = build()
                self.ext.init(comm.rank, comm.world_size, addresses[comm.rank])
            except Exception as exc:
                error = repr(exc)
            self.agree(error, 'local preparation')
            signatures = [None]*self.world
            signature = (self.ext.__name__, tuple(addresses), MAX_ELEMENTS, comm.world_size, self.hidden)
            dist.all_gather_object(signatures, signature, group=self.control)
            if any(other != signature for other in signatures):
                raise RuntimeError(f'one-shot binary or rank table differs: {signatures}')
            infos = [None]*self.world
            dist.all_gather_object(infos,self.ext.local_infos(),group=self.control)
            error = None
            try:
                self.ext.connect(infos)
            except Exception as exc:
                error = repr(exc)
            self.agree(error,'connection')
            for rows in (1,6,24,32,48,64):
                x = torch.full((rows,self.hidden),comm.rank+1,device='cuda',dtype=torch.bfloat16)
                ref = x.clone()
                dist.all_reduce(ref,group=comm.group)
                reducers = [self.ext.oneshot_ar]
                if rows <= 8:
                    reducers.append(self.ext.oneshot_ar_consumer)
                for reduce in reducers:
                    actual = reduce(x)
                    torch.cuda.synchronize()
                    self.agree(None if torch.equal(actual,ref) else 'sum differs from NCCL',f'{rows}-row self-test')
            self._check_sum_order(comm.rank)
            for rows in (1, 3, 6, 24, 48, 64):
                keys = torch.arange(rows, device='cuda', dtype=torch.int64) * (1 << 40) - comm.rank
                keys[0] = -(2**63) if comm.rank != 2 else 2**63 - 1
                reference = keys.clone()
                dist.all_reduce(reference, op=dist.ReduceOp.MAX, group=comm.group)
                self.ext.oneshot_max_int64(keys)
                torch.cuda.synchronize()
                self.agree(None if torch.equal(keys, reference) else 'MAX differs from NCCL',
                           f'{rows}-key int64 self-test')
            for keys in (1, 3, 96, 384, 512):
                packet = torch.arange(keys, device='cuda', dtype=torch.int64) + comm.rank * (1 << 40)
                packet[0] = -(2**63) + comm.rank
                expected = torch.empty((self.world, keys), device='cuda', dtype=torch.int64)
                dist.all_gather_into_tensor(expected.flatten(), packet, group=comm.group)
                actual = self.ext.oneshot_gather_int64(packet)
                torch.cuda.synchronize()
                self.agree(None if torch.equal(actual, expected) else 'gather differs from NCCL',
                           f'{keys}-key int64 gather self-test')
        except BaseException:
            self.close()
            raise

    def _check_sum_order(self, rank):
        # Every input and expected result is exactly representable in BF16.
        # Local-first FP32 summation produces [2, 2, 1, 1] for the first
        # column; positive-only fixtures cannot reveal this rank divergence.
        values = ((2.**24, 256., 1., -1.), (-2.**24, -256., 2., 1.),
                  (1., 2.**-16, 3., 2.**-24), (1., 2.**-16, 4., -2.**-24))
        row = torch.tensor(values[rank], device='cuda', dtype=torch.bfloat16).repeat(1024)
        expected_row = torch.tensor((2., 2.**-15, 10., 0.), device='cuda', dtype=torch.bfloat16).repeat(1024)
        for rows in (1, 7, 24, 64):
            x, expected = row.repeat(rows, 1), expected_row.repeat(rows, 1)
            reducers = [self.ext.oneshot_ar]
            if rows <= 8:
                reducers.append(self.ext.oneshot_ar_consumer)
            for reduce in reducers:
                actual = reduce(x)
                torch.cuda.synchronize()
                self.agree(None if torch.equal(actual, expected) else 'rank-ordered sum differs',
                           f'{rows}-row cancellation self-test')
                if rows != 7:
                    continue
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        actual = reduce(x)
                    for factor in (0., 2.**-8, 2.**8):
                        x.copy_(row.unsqueeze(0).expand_as(x) * factor)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.agree(None if torch.equal(actual, expected * factor) else 'captured rank-ordered sum differs',
                                   f'7-row cancellation replay at scale {factor:g}')
                finally:
                    graph.reset()
                x.copy_(row.unsqueeze(0).expand_as(x))

    def agree(self, error, stage):
        errors = [None]*self.world
        dist.all_gather_object(errors,error,group=self.control)
        if any(e is not None for e in errors):
            raise RuntimeError(f'one-shot {stage} failed: {errors}')

    @staticmethod
    def eligible(t):
        return (t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous()
                and 0 < t.numel() <= MAX_ELEMENTS and t.numel()%8 == 0 and t.data_ptr()%16 == 0)

    def reduce(self,t):
        self.assert_consumed()
        if self.closed or not self.eligible(t):
            raise ValueError('unavailable one-shot transport or unsupported tensor')
        if not self.ext.healthy():
            raise RuntimeError('one-shot proxy stopped progressing')
        return (self.ext.oneshot_ar_consumer if t.numel() <= CONSUMER_MAX_ELEMENTS else self.ext.oneshot_ar)(t)

    def assert_consumed(self):
        if self.pending is not None or self.packet_failed:
            raise RuntimeError('the previous rank-packet consumer did not complete')

    def exchange(self, t):
        self.assert_consumed()
        if self.closed or not self.eligible(t) or t.ndim != 2 or t.shape[1] != self.hidden:
            raise ValueError(f'rank packets require live TP{self.world} BF16 [1..{MAX_ELEMENTS // self.hidden},{self.hidden}]')
        if not self.ext.healthy():
            raise RuntimeError('one-shot proxy stopped progressing')
        self.pending = RankPackets(self, t, self.ext.oneshot_packets(t))
        return self.pending

    def produce(self, template, producer):
        """Reserve -> GEMM -> publish, on one stream. Template is shape metadata.

        The descriptor's local rank points to the reserved TX slot. Addresses
        are resolved on device each replay; no graph retains a fixed ring slot.
        """
        self.assert_consumed()
        if (self.closed or not self.eligible(template) or template.ndim != 2
                or template.shape[1] != self.hidden or template.shape[0] > 32):
            raise ValueError('direct producer requires live BF16 [1..32,4096] metadata')
        if not self.ext.healthy():
            raise RuntimeError('one-shot proxy stopped progressing')
        stream = torch.cuda.current_stream(template.device).cuda_stream
        self.pending = object()  # no collective is legal while the producer owns its slot
        try:
            reservation = self.ext.reserve_packets(template)
            producer(reservation)
            if torch.cuda.current_stream(template.device).cuda_stream != stream:
                raise RuntimeError('direct producer changed its reservation stream')
            descriptor = self.ext.publish_packets(template, reservation)
            self.pending = RankPackets(self, template, descriptor)
            return self.pending
        except BaseException:
            self.packet_failed = True  # a reserved slot cannot fall back to another collective
            self.pending = None
            raise

    @staticmethod
    def eligible_max(t):
        return (t.is_cuda and t.dtype == torch.int64 and t.is_contiguous()
                and 0 < t.numel() <= 64 and t.data_ptr() % 16 == 0)

    def reduce_max(self, t):
        self.assert_consumed()
        if self.closed or not self.eligible_max(t):
            raise ValueError('unavailable one-shot transport or unsupported MAX tensor')
        if not self.ext.healthy():
            raise RuntimeError('one-shot proxy stopped progressing')
        return self.ext.oneshot_max_int64(t)

    @staticmethod
    def eligible_gather(t):
        return (t.is_cuda and t.dtype == torch.int64 and t.is_contiguous()
                and 0 < t.numel() <= 512 and t.data_ptr() % 16 == 0)

    def gather(self, t):
        self.assert_consumed()
        if self.closed or not self.eligible_gather(t):
            raise ValueError('unavailable one-shot transport or unsupported integer gather')
        if not self.ext.healthy():
            raise RuntimeError('one-shot proxy stopped progressing')
        return self.ext.oneshot_gather_int64(t)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.ext is not None:
            self.ext.shutdown()
        dist.destroy_process_group(self.control)
