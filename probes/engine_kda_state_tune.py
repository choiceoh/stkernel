"""Bounded GB10 sweep: direct canonical KDA state addressing versus transpose copies."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

import torch
from engine.kernels.kda.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("engine.kernels.kda.baseline_recurrent", args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    original = old.fused_recurrent_gated_delta_rule_fwd_kernel
    torch.cuda.set_per_process_memory_fraction((768 * 2**20) / torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(6128)
    reports = []
    for tokens in (1, 6, 12, 32):
        heads, dim = 16, 128
        q,k,v,g = [torch.randn(1,tokens,heads,dim,device="cuda",dtype=torch.bfloat16) for _ in range(4)]
        beta = torch.randn(1,tokens,heads,device="cuda",dtype=torch.bfloat16)
        alog = torch.randn(heads,device="cuda") * .2
        bias = torch.randn(heads*dim,device="cuda") * .1
        initial = torch.randn(1,heads,dim,dim,device="cuda") * .1
        functions, values, metadata, graphs = {}, {}, {}, {}
        configs = [("baseline",8,1), ("kv8w1",8,1), ("kv16w1",16,1), ("kv16w2",16,2),
                   ("kv32w1",32,1), ("kv32w2",32,2), ("kv32w4",32,4)]
        for name,bv,warps in configs:
            output = torch.empty_like(v)
            states = torch.empty(tokens,heads,dim,dim,device="cuda")
            def run(name=name,bv=bv,warps=warps,output=output,states=states):
                h0 = initial.transpose(-1,-2).contiguous() if name=="baseline" else initial
                kernel = original if name=="baseline" else candidate
                compiled = kernel[(1,dim//bv,heads)](
                    q=q,k=k,v=v,g=g,beta=beta,o=output,h0=h0,ht=states,
                    cu_seqlens=None,ssm_state_indices=None,num_accepted_tokens=None,
                    a_log=alog,g_bias=bias,scale=dim**-.5,N=1,T=tokens,B=1,H=heads,HV=heads,K=dim,V=dim,
                    BK=dim,BV=bv,stride_init_state_token=heads*dim*dim,
                    stride_final_state_token=heads*dim*dim,stride_indices_seq=1,stride_indices_tok=1,
                    INPLACE_FINAL_STATE=False,IS_BETA_HEADWISE=False,USE_QK_L2NORM_IN_KERNEL=True,
                    IS_KDA=True,SIGMOID_BETA=True,COMPUTE_GATE=True,SAFE_GATE=True,LOWER_BOUND=-5.,
                    num_warps=warps,num_stages=3,**({} if name=="baseline" else {"STATE_KV":True}))
                result = states.transpose(-1,-2).contiguous() if name=="baseline" else states
                return (output,result), compiled
            result,kernel=run()
            functions[name]=run
            values[name]=result
            metadata[name]={"registers":kernel.n_regs,"shared":kernel.metadata.shared,
                            "bv":bv,"warps":warps}
        torch.cuda.synchronize()
        comparisons={}
        for name, (output,states) in values.items():
            ref_o,ref_s=values["baseline"]
            o_err=((output.float()-ref_o.float()).abs().max()/ref_o.float().abs().max().clamp_min(1e-12)).item()
            s_err=((states-ref_s).abs().max()/ref_s.abs().max().clamp_min(1e-12)).item()
            comparisons[name]={"output_bits_exact":torch.equal(output.view(torch.int16),ref_o.view(torch.int16)),
                               "state_bits_exact":torch.equal(states.view(torch.int32),ref_s.view(torch.int32)),
                               "output_relative_max":o_err,"state_relative_max":s_err}
            assert o_err<.02 and s_err<1e-5,(tokens,name,o_err,s_err)
        for name,run in functions.items():
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8): result,_=run()
            graphs[name]=graph
        samples={name:[] for name in graphs}
        for tick in range(9):
            names=list(graphs)
            if tick%2: names.reverse()
            for name in names:
                graphs[name].replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/8)
        row={"tokens":tokens,"comparisons":comparisons,"metadata":metadata,"samples_us":samples,
             "median_us":{name:statistics.median(s) for name,s in samples.items()}}
        reports.append(row)
        args.output.write_text(json.dumps(reports,indent=2)+"\n")
        print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
    print("PASS",flush=True)


if __name__ == "__main__": main()
