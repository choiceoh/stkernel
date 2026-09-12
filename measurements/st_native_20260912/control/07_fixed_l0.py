import json
from dataclasses import replace
caches.reset();slot=caches.slots.take(0);caches.pool.reserve(0,6912)
step=Step.prefill(ids[:6912],0,0,slot);caches.prepare(step)
x=net.embed(step.ids).chunk(4,0)[comm.rank]
res=x[:,None,:].expand(x.shape[0],net.F.hc,net.F.hidden).contiguous()
_,_,x=net._hc_pre(0,res,'attn');fixed_x=net.prefill_transport.all_gather(x.contiguous()).clone()
original_linear=net.linear; original_lanes=net.lanes;original_reduce=net.prefill_transport.reduce_scatter
values=[]
def record(value,name):
    for i,v in enumerate(value if isinstance(value,tuple) else (value,)):
        if isinstance(v,torch.Tensor):values.append((name+str(i),v.detach().clone()))
    return value
def linear(x,name):return record(original_linear(x,name),name)
net.linear=linear
wrappers={}
for name in ('conv_prefill','kda_chunk','kda_output_norm'):
    fn=getattr(original_lanes,name)
    def wrapped(*args,_fn=fn,_name=name,**kwargs):return record(_fn(*args,**kwargs),_name)
    wrappers[name]=wrapped
net.lanes=replace(original_lanes,**wrappers)
def reduce(x):return record(original_reduce(x),'reduce')
reference=None
for attempt in range(8):
    values=[]
    y=net._kda(0,fixed_x,step,caches,reduce)
    torch.cuda.synchronize()
    if reference is None:reference=values
    else:
        differences=[dict(name=a[0],max_abs=(a[1].float()-b[1].float()).abs().max().item(),ref_max=b[1].float().abs().max().item()) for a,b in zip(values,reference)]
        print(json.dumps(dict(rank=comm.rank,fixed_l0=attempt,differences=differences)),flush=True)
net.linear=original_linear;net.lanes=original_lanes
values.clear();reference.clear();del fixed_x,x,res,y
caches.pool.release(0);caches.slots.give(slot)
