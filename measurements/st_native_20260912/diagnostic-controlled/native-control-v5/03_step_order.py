import json
for sync_observe,checkpoint in ((False,False),(True,False),(False,True),(True,True)):
    caches.reset()
    slot=caches.slots.take(0)
    caches.pool.reserve(0,len(ids)+6)
    engine.add(0,ids.tolist(),max_new=64)
    engine.open(0,slot)
    for ctx in range(0,len(ids),6912):
        length=min(6912,len(ids)-ctx)
        engine.prefill(0,ctx,length,caches.pool.row(0),slot)
        if sync_observe:torch.cuda.synchronize()
        if checkpoint and length==6912:engine.checkpoint(0,ctx+length,0)
    picks=engine.generated(0)
    print(json.dumps(dict(rank=comm.rank,sync_observe=sync_observe,checkpoint=checkpoint,first=picks,text=tok.decode(picks))),flush=True)
    caches.pool.release(0);engine.close(0);caches.slots.give(slot);engine.forget(0)
