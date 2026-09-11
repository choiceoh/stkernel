import torch,triton,statistics,json
from engine.profiles.glm53.lanes import served
from candidates import causal_conv1d_single,_single_conv
old=served(reference_for=('expert',)).conv_prefill
p=torch.cuda.get_device_properties(0)
torch.cuda.set_per_process_memory_fraction(1.5*2**30/p.total_memory)
torch.manual_seed(1258)
for t in (1,2,3,6,8,9,24,64,256,1024,4096,6912):
 x=torch.randn(t,6416,device='cuda',dtype=torch.bfloat16)[:,:6144]; w=torch.randn(6144,4,device='cuda');s=torch.randn(6144,3,device='cuda',dtype=torch.bfloat16)
 expected=old(x,w,s)
 rows=[]
 for bc,nw,bt in ((64,1,8),(128,1,8),(128,2,8),(128,4,8),(256,4,8),(256,8,8),(128,4,16),(128,4,4)):
  y=torch.empty_like(expected[0]);f=torch.empty_like(s)
  def run():
   return _single_conv[(triton.cdiv(6144,bc),triton.cdiv(t,bt))](x,w,s,y,f,t,6144,*x.stride(),*w.stride(),*s.stride(),4,True,bc,bt,num_warps=nw,num_stages=2)
  k=run();torch.cuda.synchronize()
  assert torch.equal(y.view(torch.int16),expected[0].view(torch.int16)),(t,bc,nw,bt,(y!=expected[0]).sum().item(),(y.float()-expected[0].float()).abs().max().item())
  assert torch.equal(f,expected[1]),t
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(4):run()
  samples=[]
  for _ in range(7):
   g.replay();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/4)
  g.reset();rows.append([bc,nw,bt,statistics.median(samples),k.n_regs,k.metadata.shared,k.n_spills])
 print(json.dumps({'tokens':t,'rows':rows}),flush=True)
