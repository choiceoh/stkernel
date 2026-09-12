snap_lo=caches.snapshot_store.data_ptr();snap_hi=snap_lo+caches.snapshot_store.numel()
def is_snapshot(ring):return snap_lo<=ring.data_ptr()<snap_hi
def live_only(ring,pos,aux):
 if not is_snapshot(ring):saved_observe(ring,pos,aux)
def synchronized(ring,pos,aux):
 torch.cuda.synchronize();saved_observe(ring,pos,aux);torch.cuda.synchronize()
def scratch_snapshot(ring,pos,aux):
 saved_observe(ring.clone() if is_snapshot(ring) else ring,pos,aux)
for label,observe in [('runner-live-observe',live_only),('runner-sync-observe',synchronized),('runner-scratch-observe',scratch_snapshot)]:
 for key in list(saved_prefix.entries):saved_prefix._evict(key)
 caches.reset();engine.drafter.observe=observe;runner.prefix=saved_prefix
 engine.add(0,ids,max_new=1);runner.submit(0,len(ids),ids=ids)
 begin=time.monotonic()
 while runner.step() is not None:pass
 result=dict(rank=comm.rank,variant=label,generated=engine.generated(0),text=tok.decode(engine.generated(0)),seconds=time.monotonic()-begin)
 print(json.dumps(result,ensure_ascii=False),flush=True)
 Path(f'/diag/result-{label}-rank{comm.rank}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 engine.forget(0)
engine.drafter.observe=saved_observe
