import json
orig_forward=engine._forward
retained=[]
def holding_forward(step):
    result=orig_forward(step)
    retained.append((step,result))
    return result
for mode in ('none','stream','none','hold','none','device','none','hold'):
    caches.reset()
    slot=caches.slots.take(0); caches.pool.reserve(0,len(ids)+6)
    engine.add(0,ids.tolist(),max_new=64); engine.open(0,slot)
    engine._forward=holding_forward if mode=='hold' else orig_forward
    for ctx in range(0,len(ids),6912):
        engine.prefill(0,ctx,min(6912,len(ids)-ctx),caches.pool.row(0),slot)
        if mode=='stream':torch.cuda.current_stream().synchronize()
        if mode=='device':torch.cuda.synchronize()
    picks=engine.generated(0)
    print(json.dumps(dict(rank=comm.rank,mode=mode,first=picks,text=tok.decode(picks))),flush=True)
    torch.cuda.synchronize();retained.clear()
    caches.pool.release(0);engine.close(0);caches.slots.give(slot);engine.forget(0)
engine._forward=orig_forward
