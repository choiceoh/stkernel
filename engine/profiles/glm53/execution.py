"""Experimental GB10 execution orders, selected before graph capture.

Compute kernels share one stream and their existing workspaces. Only the TP
reduction stream overlaps them: two C=4 groups execute in the same fixed order
on every rank. This is operation scheduling, not concurrent model replays.
Prefill remains a separate, bounded step; its tiles advance layer by layer.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPlan:
    overlap: bool = False
    early_observe: bool = False
    prefill_tiles: int = 1
    tile_rows: int = 9216
    direct_mhc: bool = False
    prefill_project_tiles: bool = False
    decode_iterations: int = 1

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.overlap, self.early_observe, self.direct_mhc, self.prefill_project_tiles)):
            raise ValueError("execution switches must be booleans")
        if self.direct_mhc and self.overlap:
            raise ValueError("direct MHC packets require unsplit same-stream collectives")
        if type(self.prefill_tiles) is not int or self.prefill_tiles not in (1, 2, 4):
            raise ValueError("prefill_tiles must be 1, 2 or 4")
        if type(self.decode_iterations) is not int or self.decode_iterations not in (1, 2, 4):
            raise ValueError("decode_iterations must be 1, 2 or 4")
        if type(self.tile_rows) is not int or self.tile_rows <= 0 or self.tile_rows % 2304:
            raise ValueError("prefill tile rows must be a positive multiple of 2304")

    @property
    def active(self):
        return (self.overlap or self.early_observe or self.prefill_tiles != 1 or self.direct_mhc
                or self.prefill_project_tiles or self.decode_iterations != 1)

    def groups(self, sequences):
        if sequences <= 0:
            raise ValueError("a decode group must contain a request")
        # Other captured widths keep their existing batch, including C=1.
        return ((0, 2), (2, 4)) if self.overlap and sequences == 4 else ((0, sequences),)

    def label(self):
        return (f"tp_overlap={int(self.overlap)},early_observe={int(self.early_observe)},"
                f"prefill_tiles={self.prefill_tiles},direct_mhc={int(self.direct_mhc)},"
                f"prefill_project_tiles={int(self.prefill_project_tiles)},decode_iterations={self.decode_iterations}")


@dataclass
class Carry:
    step: object
    caches: object
    x: object
    res: object
    sp: object = None
    post: object = None
    comb: object = None
    ready: object = None
    projection: object = None


def begin(net, step, caches):
    sp = net.prefill_transport if (not net.probe and len(step.segments) == 1
                                  and step.ids.numel() >= 128 and step.ids.numel() % net.comm.world_size == 0
                                  and not getattr(step, "captured", False)) else None
    x = net.embed(step.ids)
    for pos, rows in step.patches:
        x.index_copy_(0, pos, rows.to(x.dtype))
    if sp is not None:
        x = x.chunk(net.comm.world_size, dim=0)[net.rank]
    return Carry(step, caches, x, x[:, None, :].expand(-1, net.F.hc, net.F.hidden).contiguous(), sp)


def prepare(net, layer, carry, side):
    c = carry
    if c.post is None:
        c.post, c.comb, c.x = net._hc_pre(layer, c.res, side)
    else:
        c.res, c.post, c.comb, c.x = net._hc_post_pre(layer, c.x, c.res, c.post, c.comb, side)
    c.projection = None
    if c.sp is not None and side == "attn" and not net.F.is_dsa(layer) and c.sp.project_tiles:
        c.projection = c.sp.gather_project(c.x.contiguous(), lambda v: net.linear(v, f"L{layer}.kda.in_proj"))
    elif c.sp is not None:
        c.x = c.sp.all_gather(c.x.contiguous())


def local(net, layer, carry, side):
    c = carry
    identity = lambda x: x
    if side == "attn":
        if net.F.is_dsa(layer):
            return net._dsa(layer, c.x, c.step, c.caches, reduce=identity)
        return net._kda(layer, c.x, c.step, c.caches, reduce=identity, projection=c.projection)
    op = net._moe if net.F.is_moe(layer) else net._dense
    return op(layer, c.x, reduce=identity)


def auxiliary(net, carry):
    return net.lanes.mhc_post(carry.x, carry.res, carry.post, carry.comb).float().mean(1).to(carry.x.dtype)


def finish(net, carry):
    res = net.lanes.mhc_post(carry.x, carry.res, carry.post, carry.comb)
    h = net._norm(res.float().mean(1).to(carry.x.dtype), net.p["norm"], net.F.rms_eps)
    return net.comm.all_gather(h, dim=0) if carry.sp is not None else h


class SerialStreams:
    """Executable CPU oracle for the same ordering and ownership rules."""
    def wait(self, event):
        pass

    def reduce(self, comm, value):
        # The real path owns its input until the NIC/consumer finishes.
        return comm.all_reduce(value.clone()), None

    def join(self):
        pass


class CudaStreams:
    def __init__(self):
        import torch
        self.stream = torch.cuda.Stream()

    def wait(self, event):
        import torch
        if event is not None:
            torch.cuda.current_stream().wait_event(event)

    def reduce(self, comm, value):
        import torch
        # Do not let a following compute kernel reuse an external output while
        # the communication stream/NIC still reads it. Compute workspaces stay
        # serialized; each reduction owns this graph-pool allocation.
        owned = value.clone()
        produced = torch.cuda.Event()
        produced.record()
        owned.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(produced)
            result = comm.all_reduce(owned)
            done = torch.cuda.Event()
            done.record()
        result.record_stream(torch.cuda.current_stream())
        return result, done

    def join(self):
        import torch
        torch.cuda.current_stream().wait_stream(self.stream)


def decode_overlap(net, step, caches, plan, streams, aux_layers=(), aux_ready=None):
    """Capture one fixed C=4 graph with compute and ordered TP reductions.

    Sampling, RNG, commit and row lifecycle still run once for the full batch.
    Only target work is grouped; never independently replay shared graph pools.
    """
    import torch
    groups = plan.groups(len(step.segments))
    if len(groups) == 1:
        return net.forward(step, caches, aux_layers=aux_layers, aux_ready=aux_ready)
    carries = [begin(net, step.subset(a, b), caches.subset(a, b)) for a, b in groups]
    aux = {}
    features = None
    try:
        for layer in net.layers:
            for side in ("attn", "ffn"):
                for c in carries:
                    streams.wait(c.ready)
                    prepare(net, layer, c, side)
                    c.x, c.ready = streams.reduce(net.comm, local(net, layer, c, side))
            if layer in aux_layers:
                parts = []
                for c in carries:
                    streams.wait(c.ready)
                    parts.append(auxiliary(net, c))
                aux[layer] = torch.cat(parts, dim=0)
                if aux_ready is not None and layer == max(aux_layers):
                    features = torch.cat([aux[l] for l in aux_layers], dim=-1)
                    aux_ready(features)
        out = []
        for c in carries:
            streams.wait(c.ready)
            out.append(finish(net, c))
        h = torch.cat(out, dim=0)
        if aux_layers:
            return h, features if features is not None else torch.cat([aux[l] for l in aux_layers], dim=-1)
        return h
    finally:
        # Every auxiliary stream must rejoin the capturing stream, including
        # exceptions, before the caller can close the graph or reuse its pool.
        streams.join()


def prefill_steps(step, tile_rows):
    from engine.profiles.glm53.net import Step
    if len(step.segments) != 1 or getattr(step, "captured", False):
        raise ValueError("layer-major prefill requires one uncaptured prompt")
    s = step.segments[0]
    if s.start != 0 or s.length != step.ids.numel():
        raise ValueError("prefill segment must cover its ids")
    # Patches can cross tile boundaries. Slice both indices and embedding rows.
    for start in range(0, s.length, tile_rows):
        end = min(start + tile_rows, s.length)
        patches = []
        for positions, values in step.patches:
            mask = (positions >= start) & (positions < end)
            patches.append((positions[mask] - start, values[mask]))
        marks = tuple((m - start, snap) for m, snap in step.marks if start < m < end)
        yield Step.prefill(step.ids[start:end], s.ctx + start, s.seq, s.slot, tuple(patches), marks)


def prefill_layer_major(net, step, caches, plan, aux_layers=()):
    """Bounded window: same tile shapes/arithmetic, a different layer order.

    A layer's tiles always run in token order. KDA and indexer histories are
    per layer, so later layers may still start at the window's original context.
    Prefix marks are materialized during each layer, before its rings advance.
    """
    import torch
    if step.ids.numel() > plan.tile_rows * plan.prefill_tiles:
        raise ValueError("prefill exceeds the declared activation window")
    carries = [begin(net, tile, caches) for tile in prefill_steps(step, plan.tile_rows)]
    aux = {}
    end_marks = dict(step.marks)
    origin = step.segments[0].ctx
    for layer in net.layers:
        for side in ("attn", "ffn"):
            for c in carries:
                prepare(net, layer, c, side)
                reduce = c.sp.reduce_scatter if c.sp is not None else net.comm.all_reduce
                c.x = reduce(local(net, layer, c, side))
                s = c.step.segments[0]
                end = s.ctx + s.length
                if side == "attn" and not net.F.is_dsa(layer) and end - origin in end_marks:
                    conv, rec = caches.kda(layer, s.slot)
                    positions = torch.arange(end - (net.F.conv - 1), end, device=c.x.device)
                    taps = conv.index_select(1, positions % conv.shape[1]).T
                    caches.mark_kda(layer, end_marks[end - origin], rec[(end - 1) % rec.shape[0]], taps)
        if layer in aux_layers:
            parts = []
            for c in carries:
                a = auxiliary(net, c)
                parts.append(net.comm.all_gather(a, dim=0) if c.sp is not None else a)
            aux[layer] = torch.cat(parts, dim=0)
    h = torch.cat([finish(net, c) for c in carries], dim=0)
    return (h, torch.cat([aux[l] for l in aux_layers], dim=-1)) if aux_layers else h
