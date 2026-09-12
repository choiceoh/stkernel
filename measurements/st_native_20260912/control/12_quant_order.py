import json,time,flashinfer
original_quant=flashinfer.nvfp4_quantize
def ordered_quant(*args,**kwargs):
    kwargs['enable_pdl']=False
    return original_quant(*args,**kwargs)
flashinfer.nvfp4_quantize=ordered_quant
caches.reset();runner.prefix.clear()
for seq in list(engine.tokens):engine.forget(seq)
request=json.loads(open('/repo/diagnostic-requests.json').read())[-1]
for attempt in range(3):
    tokens=tok.encode(render([dict(role='user',content=request['content'])],dict(thinking=True))).ids
    engine.add(attempt,tokens,max_new=1200);runner.submit(attempt,len(tokens))
    started=time.monotonic()
    while runner.step() is not None:pass
    generated=engine.generated(attempt)
    print(json.dumps(dict(rank=comm.rank,quant_order=attempt,elapsed=time.monotonic()-started,tokens=len(generated),text=tok.decode(generated)),ensure_ascii=False),flush=True)
    engine.forget(attempt)
flashinfer.nvfp4_quantize=original_quant
