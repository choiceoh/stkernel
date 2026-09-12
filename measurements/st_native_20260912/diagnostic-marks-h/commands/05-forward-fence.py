saved_forward=engine._forward
def fenced_forward(*args,**kwargs):
 out=saved_forward(*args,**kwargs);torch.cuda.synchronize();return out
engine._forward=fenced_forward
for key in list(saved_prefix.entries):saved_prefix._evict(key)
caches.reset();engine.drafter.observe=saved_observe;runner.prefix=saved_prefix
engine.add(0,ids,max_new=1);runner.submit(0,len(ids),ids=ids)
begin=time.monotonic()
while runner.step() is not None:pass
result=dict(rank=comm.rank,variant='runner-sync-after-forward',generated=engine.generated(0),text=tok.decode(engine.generated(0)),seconds=time.monotonic()-begin)
print(json.dumps(result,ensure_ascii=False),flush=True)
Path(f'/diag/result-runner-sync-after-forward-rank{comm.rank}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
engine.forget(0);engine._forward=saved_forward
