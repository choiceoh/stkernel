"""Private bounded actual-request operands. Never part of the release path."""
from contextlib import contextmanager
from pathlib import Path
import torch


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return [clone(v) for v in x]
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    return x


def cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, (tuple, list)):
        return [cpu(v) for v in x]
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    return x


@contextmanager
def capture(net, step):
    identity = dict(getattr(net, 'incident_step_identity', {}))
    root = getattr(net, 'incident_audit_root', None)
    enabled = (root is not None and identity.get('mode') == 5
               and identity.get('admission') <= 3
               and (identity.get('prefill') or identity.get('generation') in (1,124,125)))
    if not enabled:
        yield
        return
    assert len(step.segments) == 1 and not getattr(step, 'captured', False)
    events, saved = [], {}
    def replace(name, method):
        saved[name] = (name in net.__dict__, net.__dict__.get(name))
        setattr(net, name, method)
    def keep(kind, **data):
        events.append(dict(kind=kind, **clone(data)))
    def rows(t):
        total = step.ids.numel()
        local = (total + net.comm.world_size - 1) // net.comm.world_size
        if total >= 128 and net.prefill_transport is not None and t.shape[0] == local:
            at = min(local, total - net.rank * local) - 1
            return t[at:at+1].contiguous()
        return t[-1:].contiguous()

    pre = net._hc_pre
    def hc_pre(L, res, side):
        before = clone(rows(res))
        result = pre(L, res, side)
        keep('mhc_pre', layer=L, side=side, residual=before,
             norm=net.p[f'L{L}.'+('in_norm' if side=='attn' else 'post_norm')],
             actual=[rows(t) for t in result])
        return result
    replace('_hc_pre', hc_pre)
    post_pre = net._hc_post_pre
    def hc_post_pre(L, x, res, post, comb, side):
        before = clone([rows(t) for t in (x,res,post,comb)])
        result = post_pre(L,x,res,post,comb,side)
        keep('mhc_post_pre', layer=L, side=side, inputs=before,
             norm=net.p[f'L{L}.'+('in_norm' if side=='attn' else 'post_norm')],
             actual=[rows(t) for t in result])
        return result
    replace('_hc_post_pre', hc_post_pre)

    linear = net.linear
    def linear_call(x, name, **kwargs):
        before = clone(rows(x))
        result = linear(x,name,**kwargs)
        dense = net.dense.get(name)
        keep('linear', name=name, input=before, actual=rows(result),
             smooth=getattr(dense,'smooth',None), rows=x.shape[0])
        return result
    replace('linear',linear_call)

    absorb = net._mla_absorb
    def mla_absorb(L,x,weight,step,*,transpose=False):
        before=clone(rows(x))
        result=absorb(L,x,weight,step,transpose=transpose)
        keep('mla_absorb',layer=L,input=before,weight=weight,transpose=transpose,actual=rows(result))
        return result
    replace('_mla_absorb',mla_absorb)
    context=net._mla_context
    def mla_context(L,q,latent,slots,valid,step,caches):
        before=clone(rows(q))
        # Last real query only; retain exactly the rows it can read.
        chosen=slots[-1:].long().clamp(0,latent.shape[0]-1)
        selected=latent[chosen[0]].detach().clone()
        result=context(L,q,latent,slots,valid,step,caches)
        keep('mla_context',layer=L,input=before,latent=selected,valid=rows(valid),actual=rows(result))
        return result
    replace('_mla_context',mla_context)

    route=net.route
    def route_call(L,x):
        result=route(L,x)
        keep('route',layer=L,input=rows(x),actual=[rows(t) for t in result])
        return result
    replace('route',route_call)

    original_experts=dict(net._experts)
    original_packet_experts=dict(net._packet_experts)
    for L,fn in original_experts.items():
        def expert_call(x,ids,weights,*args,_L=L,_fn=fn,**kwargs):
            before=clone(rows(x))
            result=_fn(x,ids,weights,*args,**kwargs)
            if kwargs.get('finalize') is None:
                keep('expert',layer=_L,input=before,ids=rows(ids),weights=rows(weights),actual=rows(result),packet=False)
            return result
        net._experts[L]=expert_call
    for L,fn in original_packet_experts.items():
        def packet_call(batch,ids,weights,*args,_L=L,_fn=fn,**kwargs):
            g=batch.geometry
            src,local=divmod(g.rows-1,g.local_rows)
            start=src*g.stride+local*g.hidden
            values=batch.received[start:start+g.hidden].view(torch.float8_e4m3fn).float()
            start=src*g.stride+g.local_elements+local*(g.hidden//g.block)*4
            scales=batch.received[start:start+(g.hidden//g.block)*4].view(torch.float32)
            inp=(values*scales.repeat_interleave(g.block)).to(torch.bfloat16)[None]
            result=_fn(batch,ids,weights,*args,**kwargs)
            keep('expert',layer=_L,input=inp,ids=rows(ids),weights=rows(weights),actual=rows(result),packet=True)
            return result
        net._packet_experts[L]=packet_call
    try:
        yield
    finally:
        net._experts=original_experts
        net._packet_experts=original_packet_experts
        for name,(existed,value) in saved.items():
            if existed:
                setattr(net,name,value)
            else:
                delattr(net,name)
        directory=Path(root).parent/'incident-components'
        directory.mkdir(parents=True,exist_ok=True)
        s=step.segments[0]
        path=directory/f"rank{net.rank}-admit{identity['admission']}-gen{identity['generation']}-ctx{s.ctx}.pt"
        torch.save(cpu(dict(identity=identity,rank=net.rank,context=s.ctx,tokens=s.length,
                            events=events)),path)
