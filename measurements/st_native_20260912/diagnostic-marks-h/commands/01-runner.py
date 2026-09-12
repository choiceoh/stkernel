# Replay actual model/runner prefix marking and drafter observation, without capture.
for h in list(runner.prefix.entries): runner.prefix._evict(h)
caches.reset()
engine.add(0,ids,max_new=192)
runner.submit(0,len(ids),ids=ids)
begin=time.monotonic()
while runner.step() is not None:
 if engine.generated_count(0)==1:
  print(json.dumps(dict(rank=comm.rank,runner_first=engine.generated(0))),flush=True)
result=dict(rank=comm.rank,variant='runner-marked',generated=engine.generated(0),text=tok.decode(engine.generated(0)),seconds=time.monotonic()-begin)
print(json.dumps(result,ensure_ascii=False),flush=True)
Path(f'/diag/result-runner-marked-rank{comm.rank}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
engine.forget(0)
