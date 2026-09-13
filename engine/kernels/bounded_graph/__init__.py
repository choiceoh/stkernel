"""Experimental bounded conditional graph; callers own the iteration protocol."""
from functools import cache
from pathlib import Path


@cache
def build():
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    source = Path(__file__).with_name("loop.cu")
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"]
    key, directory, sources = prepare_sources(Path.home()/".cache/st/bounded-graph", [source],
                                               (flags, torch.__version__, torch.version.cuda))
    return load(name="st_bounded_graph_"+key, sources=list(sources), extra_cuda_cflags=flags,
                build_directory=str(directory), verbose=False)


def append_child(graph):
    """Embed a retained graph after the current stream's captured dependencies."""
    build().append_child(graph)


class BoundedGraph:
    """The body executes once, then until rank-agreed stop or the finite limit.

    The caller must authorize the first iteration, preserve outputs for every
    iteration (using count as the log index), and publish a rank-agreed stop in
    the body. Only deterministic bodies are supported: captured RNG offsets
    cannot be reused this way. Timings include the body and its stop agreement.
    """
    def __init__(self, body, count, stop, limit, *, owners=()):
        if type(limit) is not int or limit not in (1, 2, 4):
            raise ValueError("bounded graph supports 1, 2 or 4 iterations")
        self.body, self.owners = body, tuple(owners)
        self.count, self.stop = count, stop
        import torch
        self.timings = torch.empty(4, 2, dtype=torch.int64, device=count.device)
        self.native = build().BoundedGraph(body.raw_cuda_graph(), count, stop, limit,
                                          self.timings, (body, *self.owners))

    def replay(self):
        if self.native is None:
            raise RuntimeError("bounded graph is closed")
        self.native.replay()

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None
        self.body, self.owners = None, ()
