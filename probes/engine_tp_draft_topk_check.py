"""TP4 candidate-gather correctness and alternating legacy/candidate timings.

No model weights. Each rank needs the production RoCE environment and a private
MASTER_PORT. Tests original CUDA topk tie ordering, masks, replay and RNG state.
Timings cover selection plus communication, not head GEMM or full draft latency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.comm import Comm
from engine.modules.vocab import topk


def capture(call):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):call()
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    torch.cuda.synchronize()
    return g,out


def paired(graphs,comm):
    values=[[],[]]
    for round_id in range(5):
        for which in ((0,1) if round_id%2==0 else (1,0)):
            comm.barrier()
            for _ in range(5):graphs[which].replay()
            events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(30)]
            for a,b in events:
                a.record();graphs[which].replay();b.record()
            events[-1][1].synchronize()
            values[which].append([a.elapsed_time(b)*1000 for a,b in events])
    comm.barrier()
    return {name:dict(median_us=statistics.median(sum(rows,[])),samples_us=rows)
            for name,rows in zip(('legacy','candidate'),values)}


def equivalent(a,b):
    torch.testing.assert_close(a.values,b.values,rtol=0,atol=0,equal_nan=True)
    assert torch.equal(a.indices,b.indices),(a.indices,b.indices)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction((512<<20)/torch.cuda.mem_get_info()[1])
    comm=Comm.init(world=4,timeout_s=45)
    report=dict(scope=__doc__,rank=comm.rank,torch=torch.__version__,torch_git=torch.version.git_version,
                device=torch.cuda.get_device_name(),nccl=torch.cuda.nccl.version(),cases=[],
                env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')},
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                    'engine/modules/vocab.py','engine/base/comm.py','probes/engine_tp_draft_topk_check.py')})
    try:
        gen=torch.Generator(device='cuda').manual_seed(715+comm.rank)
        for rows in (1,5,20):
            local=torch.randn(rows,38720,device='cuda',dtype=torch.bfloat16,generator=gen)
            for decodable in (153880,38710,7):
                def legacy():
                    full=comm.all_gather(local,dim=-1).float()
                    full[:,decodable:]=float('-inf')
                    return full.topk(16,dim=-1)
                def candidate():return topk(local,comm,comm.rank*38720,16,decodable)
                old_graph,old_out=capture(legacy)
                new_graph,new_out=capture(candidate)
                for mode in ('random','ties','nonfinite'):
                    local.normal_(generator=gen)
                    if mode=='ties':
                        local.fill_(-1);local[:,::1000]=5
                    elif mode=='nonfinite':
                        local.fill_(float('-inf'))
                        if rows>1:
                            local[1].zero_();local[1,::2]=-0.
                            local[2,::1000]=float('nan');local[3,::1000]=float('inf')
                    rng=torch.cuda.get_rng_state().clone()
                    reference=legacy()
                    eager=candidate()
                    old_graph.replay();new_graph.replay();torch.cuda.synchronize()
                    for value in (eager,old_out,new_out):equivalent(reference,value)
                    assert torch.equal(rng,torch.cuda.get_rng_state())
                    item=dict(rows=rows,decodable=decodable,mode=mode,exact=True,
                              local_candidate_bytes=rows*16*8,gathered_candidate_bytes=rows*16*8*4,
                              local_legacy_bytes=rows*38720*2)
                    if mode=='random' and decodable==153880:
                        item['timing']=paired([old_graph,new_graph],comm)
                    report['cases'].append(item)
                    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
                    print(rows,decodable,mode,'exact',flush=True)
                old_graph.reset();new_graph.reset()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    finally:comm.close()


if __name__=='__main__':main()
