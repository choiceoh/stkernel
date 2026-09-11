"""Exercise the production ST MLA driver/JIT, graph replay and separate streams."""
import json
import torch
from engine.kernels import mla


def main():
    torch.cuda.set_per_process_memory_fraction((512 * 2**20) / torch.cuda.get_device_properties(0).total_memory)
    mla.maybe_arm()
    print("armed", "cluster_max", mla._MLA_CLUSTER_MAX, flush=True)
    torch.manual_seed(615)
    cache = torch.randn(4096,512,device="cuda").to(torch.float8_e4m3fn)
    reports = []
    for T,W in ((6,2048),(24,2048),(32,1),(32,2176),(33,33),(36,2048),
                (40,2048),(42,2048),(48,2048),(54,2048),(60,2048),(63,2176),(64,2176),(65,33)):
        q=torch.randn(T,16,512,dtype=torch.bfloat16,device="cuda")*.3
        slots=torch.randint(len(cache),(T,W),dtype=torch.int32,device="cuda")
        lens=torch.full((T,),W,dtype=torch.int32,device="cuda")
        out=torch.empty_like(q); ref=torch.empty_like(q)
        splits=mla.mla_splits(T)
        expected_cluster=mla._mla_uses_cluster(T,W,splits)
        partial=torch.empty(max(1,T*splits)*16*512,dtype=torch.float32,device="cuda")
        ml=torch.empty(max(1,T*splits)*32,dtype=torch.float32,device="cuda")
        counter=torch.zeros(8,dtype=torch.int32,device="cuda")
        def baseline():
            mla._EXT.run_mla([x.data_ptr() for x in (q,cache,slots,lens,ref,partial,ml,counter)],
                            [.0625,.7],[T,W,splits])
        def candidate():
            return mla.mla_decode(q,cache.view(torch.uint8),slots,lens,.0625,.7,out=out)
        # A qualified cluster must work even when legacy scratch is absent.
        old_ws=mla._MLA_WS; old_barriers=mla._WS
        if expected_cluster: mla._MLA_WS=None; mla._WS=None
        before=torch.cuda.memory_allocated()
        candidate()
        if expected_cluster:
            assert mla._MLA_WS is None and mla._WS is None
            assert torch.cuda.memory_allocated()==before
            mla._MLA_WS=old_ws; mla._WS=old_barriers
        baseline(); torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16),ref.view(torch.int16)),(T,W,"eager")
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): candidate()
        for iteration in range(5):
            lens.copy_(torch.arange(T,device="cuda",dtype=torch.int32)*31%(W+1))
            if iteration==0: lens.zero_()
            if iteration==1: lens.fill_(W)
            if iteration==2:
                slots[:,::2]=slots[:,0:1].clone()
            if iteration==3:
                lens[0]=0
                slots[0].fill_(-1)
            if iteration==4:
                slots.copy_(torch.randint(len(cache),(T,W),dtype=torch.int32,device="cuda"))
            q.mul_(.9)
            baseline(); graph.replay(); torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16),ref.view(torch.int16)),(T,W,iteration)
        reports.append({"T":T,"W":W,"splits":splits,"cluster":expected_cluster,
                        "eager_exact":True,"graph_replays_exact":5})
        print(json.dumps(reports[-1]),flush=True)
    # Independent cluster calls must not share a barrier or partial workspace.
    T,W=48,512
    q=torch.randn(T,16,512,dtype=torch.bfloat16,device="cuda")*.3
    slots=torch.randint(len(cache),(T,W),dtype=torch.int32,device="cuda")
    lens=torch.full((T,),W,dtype=torch.int32,device="cuda")
    expect=mla.mla_decode(q,cache.view(torch.uint8),slots,lens,.0625,.7)
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    outputs=[torch.empty_like(q),torch.empty_like(q)]
    for stream in streams: stream.wait_stream(torch.cuda.current_stream())
    for _ in range(8):
        for stream,out in zip(streams,outputs):
            with torch.cuda.stream(stream):
                mla.mla_decode(q,cache.view(torch.uint8),slots,lens,.0625,.7,out=out)
    torch.cuda.synchronize()
    assert all(torch.equal(out.view(torch.int16),expect.view(torch.int16)) for out in outputs)
    print("PASS: production JIT/boot gate, 14 eager cases, 70 graph replays, two concurrent streams",flush=True)


if __name__ == "__main__": main()
