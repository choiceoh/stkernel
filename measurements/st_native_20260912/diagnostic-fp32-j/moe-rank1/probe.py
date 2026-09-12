import json, time
from pathlib import Path
import torch
from engine.base.loader import RankLoader
from engine.profiles.glm53.lanes import served

torch.set_num_threads(4)
sample=torch.load('/sample/moe-L3-input-rank1.pt',map_location='cuda',weights_only=True)
loader=RankLoader('/home/choiceoh/models/st-glm53-9391-up-gate-full/rank1of4.safetensors')
keys=['L3.moe.'+name for name in ('w13','w13_sf','w2','w2_sf')]
p=loader.load(keys)
lanes=served(moe_static='t,r,sf6,q0',consume_scales=True)
args=[p[k] for k in keys]
lanes.moe_prepare(*args,8,10.)
x,sel,w=(sample[k] for k in ('x','sel','w'))
print(json.dumps(dict(event='loaded',shape=list(x.shape),free_gib=torch.cuda.mem_get_info()[0]/2**30)),flush=True)
results=[]
for rep in range(8):
 t=time.perf_counter()
 out=lanes.moe(x,sel,w,*args,10.).clone()
 torch.cuda.synchronize()
 if rep==0: ref=out
 diff=out.float()-ref.float()
 item=dict(repeat=rep,different=int((out!=ref).sum()),rel=float(diff.norm()/ref.float().norm()),max=float(diff.abs().max()),last_rel=float(diff[-1].norm()/ref[-1].float().norm()),seconds=time.perf_counter()-t)
 print(json.dumps(item),flush=True);results.append(item)
Path('/diag/repeat.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['different']==0 for r in results),results
