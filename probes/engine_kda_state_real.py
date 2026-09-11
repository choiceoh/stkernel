"""Actual L0 KDA weights, synthetic activations; isolate one TP4 rank's block."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch
from engine.base.arena import Arena
from engine.base.params import bind, total_bytes
from engine.profiles.glm53 import facts, lanes
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.decode_graphs import DeviceStep, GraphCaches
from engine.profiles.glm53.net import Glm53Net, Step
from engine.profiles.glm53.weights import rank_loader
from engine_kda_state_perf import baseline_lane
from engine_causal_conv_perf import baseline_conv
from engine_kda_strides_perf import baseline_strides


class IsolatedRank:
    rank, world_size = 0, 4
    def all_reduce(self, x): return x


def relative(x, y):
    return ((x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-6)).item()


def same_bits(x, y):
    return (x.shape == y.shape and x.dtype == y.dtype and
            torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    baseline = ap.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-dir", type=Path)
    baseline.add_argument("--baseline-net", type=Path, help="frozen net.py for comparing the complete KDA method")
    baseline.add_argument("--baseline-lanes", type=Path, help="frozen lanes.py for comparing the conv adapter")
    baseline.add_argument("--baseline-strides", type=Path, help="frozen canonical driver and recurrent kernel directory")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rank-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timing-replays", type=int, default=1, help="graph replays per timed sample")
    a = ap.parse_args()
    if a.timing_replays < 1: ap.error("--timing-replays must be positive")
    torch.cuda.set_per_process_memory_fraction((1024*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(93812)
    F = facts.load(a.checkpoint)
    current = lanes.served(reference_for=("expert",))
    methods = [Glm53Net._kda,Glm53Net._kda]
    if a.baseline_net:
        spec = importlib.util.spec_from_file_location("engine.profiles.glm53.baseline_net",a.baseline_net)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        methods[0] = module.Glm53Net._kda
        tables = [current,current]
    elif a.baseline_lanes:
        tables = [replace(current,conv_prefill=baseline_conv(a.baseline_lanes)),current]
    elif a.baseline_strides:
        tables = [replace(current,kda_recurrent=baseline_strides(a.baseline_strides)),current]
    else:
        tables = [replace(current,kda_recurrent=baseline_lane(a.baseline_dir)),current]
    net = Glm53Net(F,IsolatedRank(),current,layers=[0])
    specs = [s for s in net.specs() if s.name.startswith("L0.kda.")]
    loader = rank_loader(a.rank_file)
    arena = Arena(total_bytes(specs)+256*(len(specs)+10)+2*layout(F,[0]).nbytes(2,2))
    net.p = bind(specs,loader.load([s.name for s in specs],arena=arena,max_run=32<<20))
    caches = [Glm53Caches(arena,F,[0],2,2) for _ in tables]
    checked, measurements = [], []
    def compare(outputs, label):
        output_error = relative(outputs[1],outputs[0])
        state_error = relative(caches[1]._fields["rec",0],caches[0]._fields["rec",0])
        assert output_error < .008 and state_error < 2e-6,(label,output_error,state_error)
        assert torch.equal(caches[1]._fields["conv",0],caches[0]._fields["conv",0]),label
        exact = same_bits(outputs[1],outputs[0]) and torch.equal(caches[1].state,caches[0].state)
        if a.baseline_net or a.baseline_lanes or a.baseline_strides: assert exact,label
        checked.append({"case":label,"output_relative_max":output_error,"ring_relative_max":state_error,
                        "output_and_state_bits_exact":exact})
    for cache in caches: cache.reset()
    for name,ctx,t in (("prefill",0,64),("verify",64,6),("reject_four",66,6),("decode",72,1)):
        x = torch.randn(t,F.hidden,device="cuda",dtype=torch.bfloat16)*.1
        step = Step.prefill(torch.zeros(t,device="cuda",dtype=torch.int64),ctx,0,1)
        outputs=[]
        for table,cache,method in zip(tables,caches,methods):
            net.lanes=table
            outputs.append(method(net,0,x,step,cache))
        compare(outputs,name)
    for t in (1,6):
        x=torch.randn(t,F.hidden,device="cuda",dtype=torch.bfloat16)*.1
        ctx=torch.zeros(1,device="cuda",dtype=torch.int64)
        slot=torch.ones(1,device="cuda",dtype=torch.int64)
        seq=torch.zeros(1,device="cuda",dtype=torch.int64)
        step=DeviceStep(torch.zeros(t,device="cuda",dtype=torch.int64),ctx,t)
        graphs,outputs=[],[]
        for table,cache,method in zip(tables,caches,methods):
            view=GraphCaches(cache,seq,slot,4096)
            net.lanes=table
            method(net,0,x,step,view)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out=method(net,0,x,step,view)
            graphs.append(graph);outputs.append(out)
        for position,physical in ((0,2),(1,1),(7,2),(8,1),(4095,2),(4096,1)):
            x.normal_(std=.1)
            ctx.fill_(position);slot.fill_(physical)
            for field in caches[0]._fields.values(): field.normal_(std=.1)
            caches[1].state.copy_(caches[0].state)
            original=caches[1].state.clone()
            for graph in graphs: graph.replay()
            compare(outputs,f"graph_t{t}_ctx{position}_slot{physical}")
            # Match the complete canonical arena against eager after replay.
            expected_state=caches[1].state.clone();expected_out=outputs[1].clone()
            caches[1].state.copy_(original)
            net.lanes=current
            eager_step=Step.prefill(step.ids,position,0,physical)
            eager=net._kda(0,x,eager_step,caches[1])
            assert same_bits(eager,expected_out) and torch.equal(caches[1].state,expected_state), {
                "tokens":t,"context":position,"slot":physical,
                "output_relative":relative(eager,expected_out),
                "state_byte_differences":int((caches[1].state!=expected_state).sum())}
        samples=[[],[]]
        for iteration in range(11):
            for index in ((0,1) if iteration%2==0 else (1,0)):
                graphs[index].replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(a.timing_replays): graphs[index].replay()
                end.record();end.synchronize()
                samples[index].append(start.elapsed_time(end)*1000/a.timing_replays)
        measurements.append({"tokens":t,"context":4096,"samples_us":samples,
                             "median_us":[statistics.median(s) for s in samples]})
        for graph in graphs: graph.reset()
    weights={}
    with a.rank_file.open("rb") as stream:
        for s in specs:
            lo,hi=loader.header[s.name]["data_offsets"]
            stream.seek(loader.data_base+lo)
            weights[s.name] = {"sha256":hashlib.sha256(stream.read(hi-lo)).hexdigest(),"bytes":hi-lo}
    result={"passed":True,"rank":0,"real_weight_bytes":total_bytes(specs),"weights":weights,
        "baseline_net_sha256":hashlib.sha256(a.baseline_net.read_bytes()).hexdigest() if a.baseline_net else None,
        "baseline_lanes_sha256":hashlib.sha256(a.baseline_lanes.read_bytes()).hexdigest() if a.baseline_lanes else None,
        "baseline_strides_sha256":{p:hashlib.sha256((a.baseline_strides/p).read_bytes()).hexdigest() for p in ("kda.py","fused_recurrent.py")} if a.baseline_strides else None,
        "config_sha256":hashlib.sha256((a.checkpoint/"config.json").read_bytes()).hexdigest(),
        "timing_replays":a.timing_replays,
        "synthetic_activations":True,"collectives":False,"graph_eager_bytes_exact":True,
        "checks":checked,"measurements":measurements}
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
