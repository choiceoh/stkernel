import json,sys
sys.path.insert(0,'probes')
from pathlib import Path
import torch
from engine_kda_state_perf import baseline_lane,load
from engine.kernels.kda import fused_recurrent_kda
from engine.profiles.glm53.lanes import served
old=load(Path('baseline/kda.py'),'legacy_driver')
old.fused_recurrent_gated_delta_rule_fwd_kernel=load(Path('baseline/fused_recurrent.py'),'legacy_kernel').fused_recurrent_gated_delta_rule_fwd_kernel
rows=[]
torch.manual_seed(93814)
for t in (1,6,12):
 for seed in (0,1,2):
  q,k,v,g=[torch.randn(1,t,16,128,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
  b=torch.randn(1,t,16,device='cuda',dtype=torch.bfloat16)
  a=torch.randn(16,device='cuda');bias=torch.randn(2048,device='cuda')
  initial=torch.randn(1,16,128,128,device='cuda')*.1
  kw=dict(initial_state=initial,inplace_final_state=False,sigmoid_beta=True,a_log=a,g_bias=bias,compute_gate=True,lower_bound=-5.)
  expected=old.fused_recurrent_kda(q,k,v,g,b,**kw)
  actual=fused_recurrent_kda(q,k,v,g,b,**kw)
  assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(actual,expected))
  rows.append({'tokens':t,'case':seed,'legacy_output_and_state_bits_exact':True})
from collections import Counter
ops={}
args=(q[:,:6].contiguous(),k[:,:6].contiguous(),v[:,:6].contiguous(),g[:,:6].contiguous(),b[:,:6].contiguous(),a,bias,initial,-5.)
for name,fn in [('old',baseline_lane(Path('baseline'))),('new',served(reference_for=('expert',)).kda_recurrent)]:
 fn(*args);torch.cuda.synchronize()
 torch.cuda.reset_peak_memory_stats();start=torch.cuda.memory_allocated()
 result=fn(*args);torch.cuda.synchronize()
 peak=torch.cuda.max_memory_allocated()-start
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  result=fn(*args)
 torch.cuda.synchronize()
 ops[name]={'peak_increment_bytes':peak,'aten_ops':{e.key:e.count for e in prof.key_averages() if e.key.startswith('aten::')},
 'cuda_events':dict(Counter(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))}
report={'legacy':rows,'six_token_seeded_dense_adapter':ops}
Path('legacy-profile.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
