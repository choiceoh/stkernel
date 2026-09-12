import json,time
prefill_original=engine.prefill
def completed_prefill(*args,**kwargs):
    result=prefill_original(*args,**kwargs)
    torch.cuda.current_stream().synchronize()
    return result
engine.prefill=completed_prefill
caches.reset();runner.prefix.clear()
for seq in list(engine.tokens):engine.forget(seq)
request=json.loads(open('/repo/diagnostic-requests.json').read())[-1]
for attempt in range(2):
    tokens=tok.encode(render([dict(role='user',content=request['content'])],dict(thinking=True))).ids
    engine.add(attempt,tokens,max_new=1200);runner.submit(attempt,len(tokens))
    started=time.monotonic()
    while runner.step() is not None:pass
    generated=engine.generated(attempt)
    print(json.dumps(dict(rank=comm.rank,sync_generation=attempt,elapsed=time.monotonic()-started,tokens=len(generated),text=tok.decode(generated)),ensure_ascii=False),flush=True)
    engine.forget(attempt)
engine.prefill=prefill_original
