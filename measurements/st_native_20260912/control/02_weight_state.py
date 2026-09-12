import hashlib,json
fresh=tok.encode(render([dict(role='user',content=prompt)],dict(thinking=True))).ids
print(json.dumps(dict(rank=comm.rank,ids_equal=torch.equal(ids,torch.tensor(fresh,device='cuda')),ids_tail=ids[-20:].tolist())),flush=True)
def digest(t):
    raw=t.detach().contiguous().view(torch.uint8).reshape(-1)
    sha=hashlib.sha256()
    for chunk in raw.split(16<<20):sha.update(chunk.cpu().numpy())
    return sha.hexdigest()
weights={}
for key,layer in net.dense.items():
    fp8=layer if key=='head' else layer.fp8
    weights[key+'/fp8']=fp8.weight[0]
    weights[key+'/fp8scale']=fp8.weight[1]
    if key!='head':
        for i,t in enumerate(layer.nvfp4[:3]):weights[key+'/nvfp4/'+str(i)]=t
before_hash={key:digest(t) for key,t in weights.items()}
print('WEIGHT_BASELINE_READY',comm.rank,flush=True)
# Observe actual saved auxiliary states again, then repeat a fresh prompt.
if aux is not None:
    engine.drafter.observe(caches.draft_ring(1),torch.arange(len(aux),device='cuda'),aux)
after_hash={key:digest(t) for key,t in weights.items()}
print(json.dumps(dict(rank=comm.rank,changed_weights=[k for k in before_hash if before_hash[k]!=after_hash[k]])),flush=True)
