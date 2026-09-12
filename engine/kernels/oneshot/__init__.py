"""Owned TP4 one-shot transport; all ranks prepare, connect and qualify together.

This transport serves only aligned BF16 decode reductions. Other declared
shapes use NCCL. A failure cannot change the selected collective on one rank.
Graph owners must be destroyed before close(), just like the NCCL group.
"""
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

MAX_ELEMENTS = 131072


def build():
    from torch.utils.cpp_extension import load
    root = Path(__file__).parent
    sources = [root/'dsv4_oneshot_ar.cu', root/'dsv4_oneshot_transport.h']
    flags = ['-O2', '-gencode', 'arch=compute_121a,code=sm_121a', '-DMAXEL=131072']
    key = hashlib.sha256(b''.join(p.read_bytes() for p in sources)+repr((flags,torch.__version__,torch.version.cuda)).encode()).hexdigest()[:16]
    directory = Path(os.environ.get('ST_ONESHOT_BUILD_ROOT', str(Path.home()/'.cache/st/oneshot')))/key
    directory.mkdir(parents=True,exist_ok=True)
    return load(name='st_oneshot_'+key,sources=[str(sources[0])],extra_cuda_cflags=flags,
                extra_ldflags=['-libverbs'],build_directory=str(directory),verbose=False)


class OneShot:
    def __init__(self, comm, addresses):
        if comm.world_size != 4 or len(addresses) != 4:
            raise ValueError('one-shot requires the explicit four-rank address table')
        self.ext = None
        self.control = dist.new_group(backend='gloo')
        self.closed = False
        try:
            error = None
            try:
                self.ext = build()
                self.ext.init(comm.rank, comm.world_size, addresses[comm.rank])
            except Exception as exc:
                error = repr(exc)
            self.agree(error, 'local preparation')
            signatures = [None]*4
            signature = (self.ext.__name__, tuple(addresses), MAX_ELEMENTS, comm.world_size)
            dist.all_gather_object(signatures, signature, group=self.control)
            if any(other != signature for other in signatures):
                raise RuntimeError(f'one-shot binary or rank table differs: {signatures}')
            infos = [None]*4
            dist.all_gather_object(infos,self.ext.local_infos(),group=self.control)
            error = None
            try:
                self.ext.connect(infos)
            except Exception as exc:
                error = repr(exc)
            self.agree(error,'connection')
            for rows in (1,6,24,32):
                x = torch.full((rows,4096),comm.rank+1,device='cuda',dtype=torch.bfloat16)
                ref = x.clone()
                dist.all_reduce(ref,group=comm.group)
                reducers = [self.ext.oneshot_ar]
                if rows <= 8:
                    reducers.append(self.ext.oneshot_ar_consumer)
                for reduce in reducers:
                    actual = reduce(x)
                    torch.cuda.synchronize()
                    self.agree(None if torch.equal(actual,ref) else 'sum differs from NCCL',f'{rows}-row self-test')
        except BaseException:
            self.close()
            raise

    def agree(self, error, stage):
        errors = [None]*4
        dist.all_gather_object(errors,error,group=self.control)
        if any(e is not None for e in errors):
            raise RuntimeError(f'one-shot {stage} failed: {errors}')

    @staticmethod
    def eligible(t):
        return (t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous()
                and 0 < t.numel() <= MAX_ELEMENTS and t.numel()%8 == 0 and t.data_ptr()%16 == 0)

    def reduce(self,t):
        if self.closed or not self.eligible(t):
            raise ValueError('unavailable one-shot transport or unsupported tensor')
        if not self.ext.healthy():
            raise RuntimeError('one-shot proxy stopped progressing')
        return (self.ext.oneshot_ar_consumer if t.numel() <= 8*4096 else self.ext.oneshot_ar)(t)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.ext is not None:
            self.ext.shutdown()
        dist.destroy_process_group(self.control)
