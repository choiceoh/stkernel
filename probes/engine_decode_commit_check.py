import torch
from engine.kernels.common.decode_commit import advance
from engine.base.sampler import commit_batch

cases = 0
for k in (1, 3, 7):
 for sampled in (False, True):
  for kept in range(k+1):
   for room in range(k+2):
    for stop_at in (0, kept, k+1):
     drafts = torch.arange(10, 10+k).view(1,k)
     picks = torch.cat((drafts[:,:kept], torch.tensor([[50]]), torch.full((1,k-kept),60)),1)
     end = int(picks[0,stop_at]) if stop_at < k+1 else 99
     state = dict(drafts=drafts, ends=torch.tensor([[end]]), alive=torch.tensor([True]),
                  generated=torch.tensor([0]), limit=torch.tensor([room]), ctx=torch.tensor([17]),
                  anchor=torch.tensor([9]), real_slot=torch.tensor([1]), slot=torch.tensor([1]))
     accepted=torch.tensor([kept]) if sampled else None
     expected=commit_batch(picks,drafts,state['alive'],state['generated'],state['limit'],state['ends'],accepted)
     got=advance(picks,state,accepted)
     for a,b in zip(got[:4],expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
     assert state['ctx'].item()==17+got[0].item()
     cases+=1
print(f'PASS Triton CPU interpreter: {cases} greedy/sampled K1/K3/K7 clipping cases; counts, tokens, done, kept and context agree')
