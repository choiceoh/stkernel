import sys,json,statistics
sys.path.insert(0,'/work')
import torch,triton
from engine.kernels.draft_attention import _attend,_combine
from engine.kernels.dense.cublaslt import _measure
report=[]
for n in (1,2,4):
 b,h,hk,d,cells=8,8,2,128,2048
 q=torch.randn(n,b,h,d,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(n,b,hk,d,device='cuda',dtype=torch.bfloat16)
 v=torch.randn_like(k)
 ring=torch.randn(n,1,2,cells,hk,d,device='cuda',dtype=torch.bfloat16)
 pos=torch.full((n,),4096,device='cuda',dtype=torch.int64)
 slots=torch.arange(n,device='cuda')
 parts=max(1,min(triton.cdiv(cells+b,32),48//(n*hk)))
 span=triton.cdiv(triton.cdiv(cells+b,32),parts)*32
 parts=triton.cdiv(cells+b,span)
 def run(bq):
  tiles=triton.cdiv(b*(h//hk),bq);held=n*hk*tiles*parts*bq
  acc=torch.empty(held,d,device='cuda');scale=torch.empty(2,held,device='cuda');out=torch.empty_like(q)
  _attend[(n,hk*tiles,parts)](q,k,v,ring,pos,slots,acc,scale[0],scale[1],ring.stride(0),0,v.stride(0),v.stride(1),1,1,b,h,hk,hk,d,cells,cells*hk*d,d**-.5,32,span,tiles,bq,cells,num_warps=4,enable_fp_fusion=False)
  _combine[(n,b,h)](acc,scale[0],scale[1],out,b,h,hk,d,parts,triton.next_power_of_2(parts),tiles,bq,num_warps=4,enable_fp_fusion=False)
  return out
 base=run(32);candidate=run(16)
 exact=torch.equal(base,candidate)
 err=(base.float()-candidate.float()).abs().max().item()
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  _measure(lambda:run(32),None);_measure(lambda:run(16),None)
  times=[[_measure(lambda bq=bq:run(bq),None,repeats=12) for bq in (32,16,16,32)] for _ in range(2)]
 row=dict(n=n,exact=exact,max_abs=err,baab_ms=times)
 report.append(row);print(json.dumps(row),flush=True)
open('/out/draft-attention-layout.json','w').write(json.dumps(report,indent=2)+'\n')
