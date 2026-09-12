import json
trace=[];where=[0]
orig_methods={name:getattr(net,name) for name in ('_kda','_dsa','_dense','_moe')}
for name,fn in orig_methods.items():
    def traced(layer,x,*args,_fn=fn,_name=name,**kwargs):
        y=_fn(layer,x,*args,**kwargs)
        trace.append((where[0],_name,layer,x[-1:].detach().clone(),y[-1:].detach().clone()))
        return y
    setattr(net,name,traced)
results=[]
for mode in ('none','stream','none'):
    caches.reset();slot=caches.slots.take(0);caches.pool.reserve(0,len(ids)+6)
    engine.add(0,ids.tolist(),max_new=64);engine.open(0,slot)
    trace=[]
    for ctx in range(0,len(ids),6912):
        where[0]=ctx
        engine.prefill(0,ctx,min(6912,len(ids)-ctx),caches.pool.row(0),slot)
        if mode=='stream':torch.cuda.current_stream().synchronize()
    picks=engine.generated(0)
    print(json.dumps(dict(rank=comm.rank,trace_mode=mode,first=picks,text=tok.decode(picks))),flush=True)
    results.append([(a,b,c,x.cpu(),y.cpu()) for a,b,c,x,y in trace])
    caches.pool.release(0);engine.close(0);caches.slots.give(slot);engine.forget(0)
for name,fn in orig_methods.items():setattr(net,name,fn)
for ri in (0,2):
    differences=[]
    for a,b in zip(results[ri],results[1]):
        dx=(a[3].float()-b[3].float()).abs().max().item();dy=(a[4].float()-b[4].float()).abs().max().item()
        if dx or dy:differences.append(dict(ctx=a[0],op=a[1],layer=a[2],input_max=dx,output_max=dy,reference_max=b[4].float().abs().max().item()))
    print(json.dumps(dict(rank=comm.rank,trace_run=ri,differences=differences[:12])),flush=True)
    torch.save(dict(actual=results[ri],reference=results[1]),f'/home/choiceoh/glm53-logs/trace-rank{comm.rank}-{ri}.pt')
trace.clear();results.clear()
