import json
for seq,grow,checkpoint in ((3,False,False),(3,True,False),(3,True,True),(0,True,True)):
    caches.reset()
    slot=caches.slots.take(seq)
    if not grow:caches.pool.reserve(seq,len(ids)+6)
    engine.add(seq,ids.tolist(),max_new=64)
    engine.open(seq,slot)
    for ctx in range(0,len(ids),6912):
        length=min(6912,len(ids)-ctx)
        if grow:caches.pool.reserve_to([seq],[ctx+length])
        engine.prefill(seq,ctx,length,caches.pool.row(seq),slot)
        if checkpoint and length==6912:engine.checkpoint(seq,ctx+length,0)
    picks=engine.generated(seq)
    print(json.dumps(dict(rank=comm.rank,seq=seq,grow=grow,checkpoint=checkpoint,first=picks,text=tok.decode(picks))),flush=True)
    caches.pool.release(seq);engine.close(seq);caches.slots.give(slot);engine.forget(seq)
