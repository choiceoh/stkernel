"""Decode-only TP4 exchanges with immediate native MHC consumers.

Auxiliary feature layers and the last FFN retain ordinary reduced tensors.
Every packet is consumed before the next collective; no communication side
stream or split batch may share this transport while the step runs.
"""
from .execution import begin, prepare, local, auxiliary, finish


def consume(net, layer, carry, side, packet):
    n, f, p = f"L{layer}.", net.F, net.p
    def mhc(source, descriptor):
        return net.mhc(n+f"hc.{side}_fn", source, carry.res, carry.post, carry.comb,
                       p[n+f"hc.{side}_scale"], p[n+f"hc.{side}_base"],
                       p[n+("in_norm" if side == "attn" else "post_norm")],
                       f.rms_eps, f.hc_eps, f.post_mult, f.sinkhorn, packets=descriptor)
    carry.res, carry.post, carry.comb, carry.x = packet.consume(mhc)


def exchange_local(net, layer, carry, side, *, moe_output=False):
    transport = net.comm.transport
    if moe_output and side == "ffn" and net.F.is_moe(layer):
        return net._moe(layer, carry.x, finalize=transport.exchange_moe)
    if not hasattr(transport, "produce") or (side == "ffn" and (net.F.is_moe(layer) or getattr(net, "modelopt", False))):
        return transport.exchange(local(net, layer, carry, side))
    def project(x, name):
        dense = net.dense.get(name)
        writer = getattr(dense, "slot_writer", lambda rows: None)(x.shape[0])
        if writer is None:
            return transport.exchange(net.linear(x, name))
        # The previous MHC activation has the result's shape. Native MHC
        # uses it only as metadata when the descriptor supplies all ranks.
        return transport.produce(carry.x, lambda address: writer(x, address))
    return local(net, layer, carry, side, project=project)


def decode_direct(net, step, caches, aux_layers=(), aux_ready=None, *, consumer=consume, contract=None,
                  moe_output=False):
    import torch
    transport = net.comm.transport
    if transport is None or not hasattr(transport, "exchange") or (consumer is consume and net.mhc is None):
        raise ValueError("direct MHC requires the native TP4 packet transport and MHC")
    if (net.probe or step.patches or step.marks or not 1 <= step.ids.numel() <= 64 or
            any(s.length > net.F.spec_k + 1 for s in step.segments)):
        raise ValueError("direct MHC requires 1..64 unpatched decode rows without probes")
    if any(layer not in net.layers for layer in aux_layers):
        raise ValueError("direct MHC features must name layers in this target")
    if moe_output and (not hasattr(transport, 'exchange_moe') or step.ids.numel() > 32):
        raise ValueError('MoE output qualification requires its packet consumer and at most 32 rows')
    transport.assert_consumed()
    c = begin(net, step, caches)
    packet, features = None, None
    aux = {}
    if contract is not None and aux_layers:
        features = torch.empty((c.x.shape[0], len(aux_layers) * net.F.hidden),
                               device=c.x.device, dtype=c.x.dtype)
    for layer in net.layers:
        if packet is None:
            prepare(net, layer, c, "attn")
        else:
            consumer(net, layer, c, "attn", packet)
            packet = None
        packet = exchange_local(net, layer, c, "attn")
        consumer(net, layer, c, "ffn", packet)
        packet = None
        if layer == net.layers[-1] or layer in aux_layers:
            if moe_output and net.F.is_moe(layer):
                from engine.kernels.moe_output import combine
                c.x = net.comm.all_reduce(net._moe(layer, c.x, finalize=combine))
            else:
                c.x = net.comm.all_reduce(local(net, layer, c, "ffn"))
        else:
            packet = exchange_local(net, layer, c, "ffn", moe_output=moe_output)
        if layer in aux_layers:
            if contract is None:
                aux[layer] = auxiliary(net, c)
            else:
                # Each feature has its final column address before capture.
                # Preserve caller order and repeated layers in truncated probes;
                # these are read-only consumers of the continuing residual carry.
                for i, requested in enumerate(aux_layers):
                    if requested == layer:
                        contract(c.x, c.res, c.post, c.comb,
                                 out=features[:, i * net.F.hidden:(i + 1) * net.F.hidden])
            if aux_ready is not None and layer == max(aux_layers):
                if features is None:
                    features = torch.cat([aux[l] for l in aux_layers], dim=-1)
                aux_ready(features)
    transport.assert_consumed()
    h = finish(net, c, contract=contract)
    if aux_layers:
        return h, features if features is not None else torch.cat([aux[l] for l in aux_layers], dim=-1)
    return h
