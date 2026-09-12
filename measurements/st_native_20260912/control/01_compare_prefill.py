import json
from pathlib import Path
# Compare the standalone successful prefill with the actual auxiliary-state
# and drafter-observation path, using the same loaded model and cache ownership.
runner.prefix.clear()
for seq in list(engine.tokens):
    engine.forget(seq)
prompt=Path('/repo/prompt32k.txt').read_text()
ids=torch.tensor(tok.encode(render([dict(role='user',content=prompt)],dict(thinking=True))).ids,device='cuda',dtype=torch.int64)
for aux_on,observe_on in ((False,False),(True,False),(True,True),(False,False)):
    caches.reset()
    slot=caches.slots.take(0)
    caches.pool.reserve(0,len(ids)+6)
    print(json.dumps(dict(rank=comm.rank,mode=[aux_on,observe_on],blocks=list(caches.pool.row(0))[:15])),flush=True)
    for ctx in range(0,len(ids),6912):
        step=Step.prefill(ids[ctx:ctx+6912],ctx,0,slot)
        caches.prepare(step)
        result=net.forward(step,caches,aux_layers=engine.aux_layers if aux_on else None)
        h,aux=result if aux_on else (result,None)
        saved=h[-1:].clone()
        logits=net.head(saved)
        before=int(logits.argmax(-1).item())
        packed=int(net.head_tokens(saved,engine.decodable).item())
        if observe_on:
            engine.drafter.observe(caches.draft_ring(slot),torch.arange(ctx,ctx+len(step.ids),device='cuda'),aux)
        after=int(net.head_tokens(h[-1:],engine.decodable).item())
        delta=(saved.float()-h[-1:].float()).abs().max().item()
        print(json.dumps(dict(rank=comm.rank,mode=[aux_on,observe_on],ctx=ctx,before=before,packed=packed,after=after,delta=delta)),flush=True)
    caches.pool.release(0)
    caches.slots.give(slot)
