saved_observe=engine.drafter.observe
saved_prefix=runner.prefix
for label,no_observe,no_prefix in [('runner-no-observe',True,False),('runner-no-prefix',False,True)]:
 for h in list(saved_prefix.entries):saved_prefix._evict(h)
 caches.reset()
 engine.drafter.observe=(lambda *args:None) if no_observe else saved_observe
 runner.prefix=None if no_prefix else saved_prefix
 engine.add(0,ids,max_new=1)
 runner.submit(0,len(ids),ids=ids)
 begin=time.monotonic()
 while runner.step() is not None:pass
 result=dict(rank=comm.rank,variant=label,generated=engine.generated(0),text=tok.decode(engine.generated(0)),seconds=time.monotonic()-begin)
 print(json.dumps(result,ensure_ascii=False),flush=True)
 Path(f'/diag/result-{label}-rank{comm.rank}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 engine.forget(0)
engine.drafter.observe=saved_observe
runner.prefix=saved_prefix
