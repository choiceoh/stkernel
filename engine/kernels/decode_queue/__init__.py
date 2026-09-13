"""Bounded, single-producer GB10 token publication with system-scope atomics.

Published iteration rows are immutable until the owning burst has retired.
CPU cancellation sets a separate atomic; every rank votes before another step.
"""
from functools import cache
from pathlib import Path


@cache
def build():
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_sources
    source = Path(__file__).with_name("queue.cu")
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"]
    key, directory, staged = prepare_sources(Path.home()/".cache/st/decode-queue", [source],
                                             (flags, torch.__version__, torch.version.cuda))
    return load(name="st_decode_queue_"+key, sources=list(staged), extra_cuda_cflags=flags,
                build_directory=str(directory), verbose=False)


class SharedDecodeQueue:
    def __init__(self, rows, tokens):
        import torch
        from engine.kernels.mapped_staging import allocate
        if type(rows) is not int or not 1 <= rows <= 4 or type(tokens) is not int or not 1 <= tokens <= 8:
            raise ValueError("shared decode queue requires 1..4 rows and 1..8 token positions")
        self.rows, self.tokens = rows, tokens
        self.stride = (tokens + 4 + 7) // 8 * 8
        size = 128 + 4 * rows * self.stride * 8
        self.host, self.device = allocate((size + 4095) // 4096 * 4096)
        self.values = self.host[128:size].view(torch.int64).view(4, rows, self.stride)
        self.ext = build()
        self.ext.check_device()
        self.begin()

    def begin(self):
        """The owner must have drained the previous burst before reusing rows."""
        self.ext.reset(self.host)

    def publish(self, result, index):
        self.ext.publish(self.device, result["tokens"], result["count"], result["done"],
                         result["accepted"], result["before"], index, self.rows, self.stride)

    def read_interrupt(self, into):
        self.ext.read_interrupt(self.device, into)

    def cancel(self):
        self.ext.cancel(self.host)

    def take(self, index, rows):
        if not 0 <= index < 4 or not 1 <= rows <= self.rows:
            raise ValueError("shared decode read exceeds its reserved rows")
        ready = self.ext.published(self.host)  # acquire before touching immutable data
        if not 0 <= ready <= 4:
            raise RuntimeError("invalid shared decode publication count")
        if index >= ready:
            return None
        values = self.values[index, :rows, :self.tokens+4].tolist()
        return dict(tokens=[v[:self.tokens] for v in values],
                    count=[v[self.tokens] for v in values],
                    done=[bool(v[self.tokens+1]) for v in values],
                    accepted=[v[self.tokens+2] for v in values],
                    before=[v[self.tokens+3] for v in values])
