import json
for clear_pages in (False,True,False,True):
    if clear_pages:caches.reset()
    slot=caches.slots.take(0)
    caches.reset_slot(slot)
    caches.pool.reserve(0,len(ids)+6)
    print(json.dumps(dict(rank=comm.rank,clear_pages=clear_pages,blocks=list(caches.pool.row(0))[:15])),flush=True)
    for ctx in range(0,len(ids),6912):
        step=Step.prefill(ids[ctx:ctx+6912],ctx,0,slot)
        caches.prepare(step)
        h,aux=net.forward(step,caches,aux_layers=engine.aux_layers)
        engine.drafter.observe(caches.draft_ring(slot),torch.arange(ctx,ctx+len(step.ids),device='cuda'),aux)
        before=int(net.head_tokens(h[-1:],engine.decodable).item())
        print(json.dumps(dict(rank=comm.rank,clear_pages=clear_pages,ctx=ctx,token=before,finite=bool(torch.isfinite(h).all()))),flush=True)
    caches.pool.release(0);caches.slots.give(slot)
