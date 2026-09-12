class FoundMoe(Exception):pass
saved_moe=net._moe
moe_input=None
def take_moe(L,x,*args,**kwargs):
 global moe_input
 if L==3:
  moe_input=x.clone();raise FoundMoe()
 return saved_moe(L,x,*args,**kwargs)
for key in list(saved_prefix.entries):saved_prefix._evict(key)
caches.reset();engine._forward=saved_forward;engine.drafter.observe=saved_observe;net._moe=take_moe
engine.add(0,ids,max_new=1);runner.submit(0,len(ids),ids=ids)
try:runner.step()
except FoundMoe:pass
finally:
 net._moe=saved_moe;runner.cancel(0);engine.forget(0)
torch.cuda.synchronize()
sel,weights=net.route(3,moe_input)
params=net.p;prefix='L3.moe.'
def run_moe(x,s,w):
 return net.lanes.moe(x,s,w,params[prefix+'w13'],params[prefix+'w13_sf'],params[prefix+'w2'],params[prefix+'w2_sf'],F.swiglu_limit)
outs=[]
for rep in range(8):
 out=run_moe(moe_input,sel,weights).clone();torch.cuda.synchronize()
 if rep==0:ref=out
 diff=(out.float()-ref.float())
 result=dict(rank=comm.rank,moe_repeat=rep,shape=list(out.shape),different=int((out!=ref).sum()),rel=float(diff.norm()/ref.float().norm()),max=float(diff.abs().max()),last_rel=float(diff[-1].norm()/ref[-1].float().norm()))
 print(json.dumps(result),flush=True);outs.append(result)
Path(f'/diag/moe-repeat-rank{comm.rank}.json').write_text(json.dumps(outs)+'\n')
# Save the actual failing geometry for a bounded independent single-layer regression.
torch.save(dict(x=moe_input,sel=sel,w=weights,first=ref,last=out),f'/diag/moe-L3-input-rank{comm.rank}.pt')
