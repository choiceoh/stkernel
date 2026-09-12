import json
from dataclasses import replace
caches.reset();slot=caches.slots.take(0);caches.pool.reserve(0,6912)
step=Step.prefill(ids[:6912],0,0,slot);caches.prepare(step)
x=net.embed(step.ids).chunk(4,0)[comm.rank];res=x[:,None,:].expand(x.shape[0],net.F.hc,net.F.hidden).contiguous()
_,_,x=net._hc_pre(0,res,'attn');fixed_x=net.prefill_transport.all_gather(x.contiguous()).clone()
original_linear=net.linear;original_lanes=net.lanes
held=[];mode='all'
def keep(value):
    held.append(value)
    return value
def linear(x,name):
    value=original_linear(x,name)
    return keep((x,value))[1] if mode in ('all','linear') else value
net.linear=linear
wrappers={}
for name in ('conv_prefill','kda_chunk','kda_output_norm'):
    fn=getattr(original_lanes,name)
    def wrapped(*args,_fn=fn,_name=name,**kwargs):
        out=_fn(*args,**kwargs)
        return keep((args,kwargs,out))[2] if mode in ('all',_name) else out
    wrappers[name]=wrapped
net.lanes=replace(original_lanes,**wrappers)
reference=None
for mode in ('all','none','linear','conv_prefill','kda_chunk','kda_output_norm','none','all'):
    for attempt in range(2):
        output=net._kda(0,fixed_x,step,caches,net.prefill_transport.reduce_scatter)
        torch.cuda.synchronize()
        if reference is None:reference=output.clone()
        print(json.dumps(dict(rank=comm.rank,lifetime_mode=mode,attempt=attempt,diff=(output.float()-reference.float()).abs().max().item())),flush=True)
        held.clear()
net.linear=original_linear;net.lanes=original_lanes
caches.pool.release(0);caches.slots.give(slot)
del reference,fixed_x,output,x,res
