"""Bounded full-model native long-context bisection; never a serving fallback."""
import json, time
from pathlib import Path
import torch
from engine.base.comm import Comm
from engine.base.instruments import Recorder
from engine.profiles.glm53.boot import build, tokenizer, chat_renderer
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.net import Step

comm=Comm.init(timeout_s=300)
engine=None
try:
 comm.prepare_oneshot()
 F,net,caches,engine,runner=build(comm,None,served(moe_static='t,r,sf6,q0',consume_scales=True),
  '/home/choiceoh/models/st-glm53-9391-up-gate-full',12,4,True,Recorder('long-diag'),execution='native',
  ckpt_meta='/repo/st-glm53-meta',drafter_dir='/home/choiceoh/models/GLM-5.3-Flash-DFlash2')
 tok=tokenizer('/repo/st-glm53-meta');render=chat_renderer('/repo/st-glm53-meta')
 prompt=Path('/diag/prompt128k.txt').read_text()
 ids=tok.encode(render([dict(role='user',content=prompt)],dict(thinking=True))).ids
 sp=net.prefill_transport
 pairs={name:layer.nvfp4 for name,layer in net.dense.items() if name!='head'}
 variants=[('native-no-marks',True,True),('fp8-no-marks',True,False),('nvfp4-no-sp',False,True)]
 for label,use_sp,use_nv in variants:
  net.prefill_transport=sp if use_sp else None
  for name,pair in pairs.items():net.dense[name].nvfp4=pair if use_nv else None
  caches.reset(); slot=caches.slots.take(0);caches.pool.reserve(0,len(ids)+80)
  begin=time.monotonic();print(json.dumps(dict(rank=comm.rank,variant=label,tokens=len(ids),begin=True)),flush=True)
  try:
   for ctx in range(0,len(ids),6912):
    step=Step.prefill(torch.tensor(ids[ctx:ctx+6912],device='cuda',dtype=torch.int64),ctx,0,slot)
    caches.prepare(step);h=net.forward(step,caches)
    if (ctx//6912)%4==0:print(json.dumps(dict(rank=comm.rank,variant=label,ctx=ctx)),flush=True)
   selected=int(net.head_tokens(h[-1:],engine.decodable).item()); generated=[selected]
   first=selected
   for i in range(63):
    step=Step.prefill(torch.tensor([selected],device='cuda',dtype=torch.int64),len(ids)+i,0,slot)
    caches.prepare(step);h=net.forward(step,caches)
    selected=int(net.head_tokens(h[-1:],engine.decodable).item()); generated.append(selected)
    if selected in engine.eos:break
   result=dict(rank=comm.rank,variant=label,first=first,generated=generated,text=tok.decode(generated),seconds=time.monotonic()-begin)
   print(json.dumps(result,ensure_ascii=False),flush=True)
   Path(f'/diag/result-{label}-rank{comm.rank}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
  finally:
   caches.pool.release(0);caches.slots.give(slot)
 comm.barrier()
finally:
 if engine is not None:engine.close_decode()
 comm.close()
