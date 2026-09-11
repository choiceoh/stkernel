"""Real KDA/DSA/MoE graph replay against eager, including state and paged bytes.

Default: isolate rank 0 arithmetic on one GPU (no TP correctness claim).
--distributed: execute the same test on all four ranks with real NCCL.
"""
import torch
from engine.base.instruments import Recorder
from engine.profiles.glm53.boot import build
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.net import Step
from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs

class IsolatedRank:
    rank=0
    world_size=4
    def all_reduce(self,x): return x
    def all_reduce_max(self,x): return x
    def all_gather(self,x,dim=-1): return x

def main():
    import argparse
    from engine.base.comm import Comm
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks',required=True)
    ap.add_argument('--ckpt-meta',default='/home/choiceoh/models/glm53-redhat-nvfp4')
    ap.add_argument('--distributed',action='store_true')
    a=ap.parse_args()
    comm=Comm.init(world=4) if a.distributed else IsolatedRank()
    try:
        return check(comm,a)
    finally:
        if a.distributed: comm.close()


def check(comm,a):
    torch.manual_seed(13)
    F,net,caches,engine,runner=build(comm,[0,3],served(),a.ranks,.25,2,False,Recorder('graph'),
        ckpt_meta=a.ckpt_meta)
    print('weights loaded; rank',comm.rank,'distributed',a.distributed,flush=True)
    for t in [1,6]:
        g=Glm53DecodeGraphs(net,caches,2,t)
        print('captured',t,flush=True)
        slots=[caches.slots.take(i) for i in range(2)]
        for i in range(2): caches.pool.reserve(i,4352)
        for contexts in [(0,0),(1,3),(7,8),(15,257),(255,511),(3,7),(4090,4091),(4095,4096)]:
            caches.reset()
            ids=[]
            for i,ctx in enumerate(contexts):
                values=torch.randint(0,30000,(ctx+t,),device='cuda')
                ids.append(values[-t:])
                if ctx:
                    st=Step.prefill(values[:ctx],0,i,slots[i]); caches.prepare(st); net.forward(st,caches)
            # Reverse order to exercise physical slot/request remapping on replay.
            step=Step.decode([(ids[i],contexts[i],i,slots[i]) for i in [1,0]])
            state,paged=caches.state.clone(),caches.paged.clone()
            caches.prepare(step)
            eager=net.forward(step,caches)
            estate,epaged=caches.state.clone(),caches.paged.clone()
            caches.state.copy_(state); caches.paged.copy_(paged)
            replay,_,_=g.run(step)
            torch.cuda.synchronize()
            rel=((replay.float()-eager.float()).abs().max()/eager.float().abs().max()).item()
            same_state=torch.equal(estate,caches.state); same_paged=torch.equal(epaged,caches.paged)
            print('case',t,contexts,'relative',rel,'state_exact',same_state,'paged_exact',same_paged,flush=True)
            assert rel<.02 and same_state and same_paged
            if t==6 and contexts==(15,257):
                # The preceding replay wrote all six speculative positions.
                # Accept only the anchor and overwrite the five rejected rows;
                # do not clear those future writes before this comparison.
                retry=Step.decode([(torch.randint(0,30000,(t,),device='cuda'),
                                    contexts[i]+1,i,slots[i]) for i in [0,1]])
                state,paged=caches.state.clone(),caches.paged.clone()
                caches.prepare(retry); eager=net.forward(retry,caches)
                estate,epaged=caches.state.clone(),caches.paged.clone()
                caches.state.copy_(state); caches.paged.copy_(paged)
                replay,_,_=g.run(retry)
                torch.cuda.synchronize()
                rel=((replay.float()-eager.float()).abs().max()/eager.float().abs().max()).item()
                same_state=torch.equal(estate,caches.state); same_paged=torch.equal(epaged,caches.paged)
                print('rejected-future',contexts,'relative',rel,'state_exact',same_state,'paged_exact',same_paged,flush=True)
                assert rel<.02 and same_state and same_paged
        for i,slot in enumerate(slots): caches.slots.give(slot); caches.pool.release(i)
        g.graphs.close()
    print('PASS',flush=True)

if __name__ == '__main__':
    main()
