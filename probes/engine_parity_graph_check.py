"""Real TP4 target layers and full DFlash2: candidate preparation and replay.

This bounded component run can coexist with serving. It is a correctness
gate, not a serving throughput result. Full-model onepass remains required.
"""
import argparse
import json
import torch

from engine.base.comm import Comm
from engine.base.instruments import Recorder
from engine.profiles.glm53.boot import build
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.net import Step
from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks',required=True)
    ap.add_argument('--ckpt-meta',required=True)
    ap.add_argument('--drafter-dir',required=True)
    args=ap.parse_args()
    torch.cuda.set_device(0)
    comm=Comm.init(timeout_s=300)
    target_graph=None
    drafter=None
    try:
        comm.prepare_oneshot()
        rec=Recorder('parity-components')
        F,net,caches,engine,runner=build(comm,[0,3],served(moe_static='t,r,sf6,q0',consume_scales=True),args.ranks,.4,2,
            True,rec,ckpt_meta=args.ckpt_meta,drafter_dir=args.drafter_dir,execution='parity')
        print(rec.table(),flush=True)
        print(json.dumps(rec.root.counters),flush=True)
        drafter=engine.drafter
        target_graph=Glm53DecodeGraphs(net,caches,2,6)
        drafter.capture_decode(caches)
        slots=[caches.slots.take(i) for i in range(2)]
        for i in range(2):caches.pool.reserve(i,4352)
        torch.manual_seed(28)
        for context in (0,17,128,256,4096,4097):
            caches.reset()
            for i in range(2):
                if context:
                    step=Step.prefill(torch.randint(0,30000,(context,),device='cuda'),0,i,slots[i])
                    caches.prepare(step);net.forward(step,caches)
            step=Step.decode([(torch.randint(0,30000,(6,),device='cuda'),context,i,slots[i]) for i in (1,0)])
            state,paged=caches.state.clone(),caches.paged.clone()
            caches.prepare(step);expected=net.forward(step,caches)
            state_expected,paged_expected=caches.state.clone(),caches.paged.clone()
            caches.state.copy_(state);caches.paged.copy_(paged)
            actual,_,_=target_graph.run(step)
            torch.cuda.synchronize()
            relative=((actual.float()-expected.float()).norm()/expected.float().norm()).item()
            assert relative<.01,relative
            assert torch.equal(state_expected,caches.state),'target state differs'
            assert torch.equal(paged_expected,caches.paged),'target paged cache differs'
            print(json.dumps(dict(rank=comm.rank,target_context=context,relative=relative,passed=True)),flush=True)
        for context in (0,1,2047,2048,2057):
            ring=caches.draft_ring(slots[context%2])
            positions=torch.arange(max(0,context-drafter.F.window),context,device='cuda')
            ring.zero_()
            aux=torch.randn(len(positions),20480,device='cuda',dtype=torch.bfloat16)*.02
            drafter.observe(ring,positions,aux)
            anchor=torch.tensor([1234],device='cuda',dtype=torch.int64)
            expected=drafter.propose_tensor(anchor,context,ring).clone()
            actual=drafter.decode_graphs.propose(1234,context,ring).clone()
            assert torch.equal(expected,actual),(context,expected,actual)
            same=comm.all_gather(actual,dim=0).reshape(4,-1)
            assert torch.equal(same,same[0].expand_as(same)),same
            saved=ring.clone()
            for count in (1,3,6):
                positions=context+torch.arange(count,device='cuda')
                aux=torch.randn(count,20480,device='cuda',dtype=torch.bfloat16)*.02
                ring.copy_(saved);drafter.observe(ring,positions,aux);expected=ring.clone()
                ring.copy_(saved);drafter.observe_decode(ring,positions,aux)
                assert torch.equal(ring,expected),(context,count)
            print(json.dumps(dict(rank=comm.rank,drafter_context=context,passed=True)),flush=True)
        comm.barrier()
    finally:
        if target_graph is not None:target_graph.graphs.close()
        if drafter is not None and drafter.decode_graphs is not None:
            drafter.decode_graphs.proposals.close();drafter.decode_graphs.observations.close()
        comm.close()


if __name__=='__main__':main()
