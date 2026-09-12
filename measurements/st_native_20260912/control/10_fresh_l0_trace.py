import json
from dataclasses import replace
original_linear=net.linear;original_lanes=net.lanes
trace=[]
def save(value,name):
    for i,v in enumerate(value if isinstance(value,tuple) else (value,)):
        if isinstance(v,torch.Tensor):trace.append((name+str(i),v.detach().clone()))
    return value
def linear(x,name):return save(original_linear(x,name),name)
net.linear=linear
wrappers={}
for name in ('conv_prefill','kda_chunk','kda_output_norm'):
    fn=getattr(original_lanes,name)
    def wrapped(*args,_fn=fn,_name=name,**kwargs):return save(_fn(*args,**kwargs),_name)
    wrappers[name]=wrapped
net.lanes=replace(original_lanes,**wrappers)
reference=None
for attempt in range(8):
    trace=[]
    caches.reset();slot=caches.slots.take(0);caches.pool.reserve(0,6912)
    step=Step.prefill(torch.tensor(ids[:6912].tolist(),dtype=torch.int64,device='cuda'),0,0,slot);caches.prepare(step)
    x=net.embed(step.ids).chunk(4,0)[comm.rank];res=x[:,None,:].expand(x.shape[0],net.F.hc,net.F.hidden).contiguous()
    _,_,x=net._hc_pre(0,res,'attn');x=net.prefill_transport.all_gather(x.contiguous())
    save(x,'input')
    y=net._kda(0,x,step,caches,net.prefill_transport.reduce_scatter);save(y,'reduce')
    torch.cuda.synchronize()
    if reference is None:reference=trace
    else:print(json.dumps(dict(rank=comm.rank,fresh_trace=attempt,differences=[dict(name=a[0],max_abs=(a[1].float()-b[1].float()).abs().max().item()) for a,b in zip(trace,reference)])),flush=True)
    caches.pool.release(0);caches.slots.give(slot)
net.linear=original_linear;net.lanes=original_lanes
trace.clear();reference.clear();del x,res,y
