import json
reference=None
for attempt in range(6):
    caches.reset();slot=caches.slots.take(0);caches.pool.reserve(0,6912)
    local_ids=torch.tensor(ids[:6912].tolist(),dtype=torch.int64,device='cuda')
    step=Step.prefill(local_ids,0,0,slot);caches.prepare(step)
    embedded=net.embed(step.ids)
    x=embedded.chunk(4,0)[comm.rank]
    res=x[:,None,:].expand(x.shape[0],net.F.hc,net.F.hidden).contiguous()
    post,comb,x=net._hc_pre(0,res,'attn')
    gathered=net.prefill_transport.all_gather(x.contiguous())
    output=net._kda(0,gathered,step,caches,net.prefill_transport.reduce_scatter)
    actual=[local_ids,embedded,post,comb,x,gathered,output]
    torch.cuda.synchronize()
    if reference is None:reference=[v.clone() for v in actual]
    else:
        print(json.dumps(dict(rank=comm.rank,input_repeat=attempt,differences=[(a.float()-b.float()).abs().max().item() for a,b in zip(actual,reference)])),flush=True)
    caches.pool.release(0);caches.slots.give(slot)
reference.clear();actual.clear();del local_ids,embedded,post,comb,x,res,gathered,output
