"""Full 128K regression with the native kernels and unmodified runner."""
import json,time
from pathlib import Path
import torch
from engine.base.comm import Comm
from engine.base.instruments import Recorder
from engine.profiles.glm53.boot import build,tokenizer,chat_renderer
from engine.profiles.glm53.lanes import served
comm=Comm.init(timeout_s=300)
engine=None
try:
 comm.prepare_oneshot()
 F,net,caches,engine,runner=build(comm,None,served(moe_static='t,r,sf6,q0',consume_scales=True),
  '/home/choiceoh/models/st-glm53-9391-up-gate-full',12,4,True,Recorder('long-fp32'),execution='native',
  ckpt_meta='/repo/st-glm53-meta',drafter_dir='/home/choiceoh/models/GLM-5.3-Flash-DFlash2')
 tok=tokenizer('/repo/st-glm53-meta');render=chat_renderer('/repo/st-glm53-meta')
 prompt=Path('/diag/prompt128k.txt').read_text()
 ids=tok.encode(render([dict(role='user',content=prompt)],dict(thinking=True))).ids
 for rep,count in ((0,1200),(1,1)):
  for key in list(runner.prefix.entries):runner.prefix._evict(key)
  caches.reset();engine.add(0,ids,max_new=count);runner.submit(0,len(ids),ids=ids)
  begin=time.monotonic();print(json.dumps(dict(rank=comm.rank,repeat=rep,begin=True,tokens=len(ids))),flush=True)
  while runner.step() is not None:pass
  generated=engine.generated(0)
  result=dict(rank=comm.rank,repeat=rep,generated=generated,text=tok.decode(generated),seconds=time.monotonic()-begin)
  print(json.dumps(result,ensure_ascii=False),flush=True)
  Path(f'/diag/result-repeat{rep}-rank{comm.rank}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  engine.forget(0);comm.barrier()
finally:
 if engine is not None:engine.close_decode()
 comm.close()
