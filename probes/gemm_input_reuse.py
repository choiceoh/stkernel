#!/usr/bin/env python3
"""Bounded same-build input-quantization reuse experiment; no serving changes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
FLAGS=['-O2','-gencode','arch=compute_121a,code=sm_121a',
       '-DMK_FP8_PACK2_DEF=1','-DMK_GEMM_TRANSPOSE_M8_DEF=1',
       '-DMK_GEMM_COMPACT_M8_DEF=1','-DMK_M8_FASTPATH_DEF=1']


def replace_once(source, old, new):
    assert source.count(old)==1, (old,source.count(old))
    return source.replace(old,new,1)


def render(original):
    source=replace_once(original,'struct MKGemm2Ctx {',
        (ROOT/'probes/gemm_input_reuse.cuh').read_text()+'\nstruct MKGemm2Ctx {')
    source=replace_once(source,'void mk_run_gemm(torch::Tensor x,',
                        'int g_probe_input_reuse = 0;\nvoid mk_run_gemm(torch::Tensor x,')
    source=replace_once(source,'  mk_launch_gemm2(c2, stream);\n}', '''
  if (g_probe_input_reuse && c2.m >= 1 && c2.m <= 8 && !bg && !c2.lr_r) {
    mk_launch(mk_probe_pack_input, c2.k / KSTEP, 0, stream, MKProbeInput{c2.x, c2.m, c2.k});
    c2.a_ready = 1;
  }
  mk_launch_gemm2(c2, stream);
}''')
    source=replace_once(source,'  m.def("run_gemm",', '''
  m.def("set_input_reuse", [](int mode) {
    TORCH_CHECK(mode == 0 || mode == 1); g_probe_input_reuse = mode;
  });
  m.def("copy_input_groups", [](torch::Tensor q, torch::Tensor scales) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kUInt8 && q.is_contiguous()
                && q.numel() == sizeof(g_mk2_aq));
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32
                && scales.is_contiguous() && scales.numel() * sizeof(float) == sizeof(g_mk2_axs));
    auto stream = c10::cuda::getCurrentCUDAStream();
    MK_CHECK_CUDA(cudaMemcpyFromSymbolAsync(q.data_ptr(), g_mk2_aq, sizeof(g_mk2_aq),
                                           0, cudaMemcpyDeviceToDevice, stream));
    MK_CHECK_CUDA(cudaMemcpyFromSymbolAsync(scales.data_ptr(), g_mk2_axs, sizeof(g_mk2_axs),
                                           0, cudaMemcpyDeviceToDevice, stream));
  });
  m.def("run_gemm",''')
    return source


def build(directory):
    from torch.utils.cpp_extension import load
    original=(ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.cu').read_text()
    source=render(original)
    sha=hashlib.sha256(source.encode()).hexdigest()
    directory=directory/sha[:16]
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'candidate.cu'; path.write_text(source)
    ext=load(name='input_reuse_'+sha[:16],sources=[str(path)],
             extra_cuda_cflags=FLAGS,build_directory=str(directory),verbose=False)
    return ext,{'base_sha256':hashlib.sha256(original.encode()).hexdigest(),
                'candidate_sha256':sha,'flags':FLAGS}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--build-dir',type=Path,default=Path('/build'))
    ap.add_argument('--out',type=Path,default=Path('/evidence/result.json'))
    ap.add_argument('--samples',type=int,default=24)
    args=ap.parse_args()
    assert args.samples>=12 and args.samples%2==0
    os.environ['MAX_JOBS']='1'
    os.environ['VLLM_GLM53_MK_PDL']='1'
    ext,identity=build(args.build_dir)
    assert hasattr(ext,'set_input_reuse') and hasattr(ext,'copy_input_groups')
    print(json.dumps(identity),flush=True)
    if args.compile_only:
        print('PASS: native compilation and probe bindings; no CUDA context',flush=True)
        return
    import torch
    spec=importlib.util.spec_from_file_location('input_reuse_driver',ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.py')
    mk=importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    mk._EXT=ext
    assert ext.probe_device()[:3]==[12,1,48]
    torch.backends.cuda.matmul.allow_tf32=False
    identity.update(device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                    samples=args.samples,serving_changed=False)
    result={'identity':identity,'gates':[],'timings':[],'verdict':'RUNNING'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.out.write_text(json.dumps(result,indent=2)+'\n')
    save()
    flush=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda')
    shapes=((6,4096,4096),(6,6144,4096),(6,6528,4096),(8,4096,4096),
            (6,1024,4096),(6,4096,512))
    for m,n,k in shapes:
        torch.manual_seed(731+n+k+m)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.3
        weights=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.05
        pack=mk.build_mk_weight_w4(weights)
        wref=mk.mk_w4_dequant(pack[0],pack[1],n,pack[2],pack[3] if len(pack)>3 else None).float()
        del weights
        aq=torch.empty((32,32,128),device='cuda',dtype=torch.uint8)
        asc=torch.empty((32,32),device='cuda',dtype=torch.float32)
        graphs={}; outputs={}
        for mode in (0,1):
            ext.set_input_reuse(mode)
            for _ in range(2): mk._gemm_call(x,pack,n)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): outputs[mode]=mk._gemm_call(x,pack,n)
            graphs[mode]=graph
        # Changed values replay the same graph and overwrite all consumed groups.
        for case in ('random','zero','tiny','wide','random2'):
            if case=='zero': x.zero_()
            else: x.normal_().mul_({'tiny':1e-20,'wide':1e3}.get(case,.3))
            for mode in (0,1): graphs[mode].replay()
            torch.cuda.synchronize()
            exact=torch.equal(outputs[0],outputs[1])
            assert exact, ('candidate changed output',m,n,k,case)
            assert torch.isfinite(outputs[1]).all(), ('nonfinite',m,n,k,case)
            ref=mk._mk_quant_x_ref(x) @ wref.T
            rel,over=mk._exact_gate(outputs[1],ref)
            assert rel<=1e-3 and over==0, ('oracle',m,n,k,case,rel,over)
            ext.copy_input_groups(aq,asc)
            groups=x.float().reshape(m,k//128,128)
            scales=(groups.abs().amax(-1)*(1/448.)).clamp_min(1e-30)
            expected=(groups*(1/scales)[...,None]).clamp(-448.,448.).to(torch.float8_e4m3fn).view(torch.uint8)
            got_q=aq[:k//128,:m].permute(1,0,2).contiguous()
            assert torch.equal(got_q,expected), ('FP8 bytes',m,n,k,case)
            assert torch.equal(asc[:m,:k//128],scales), ('scales',m,n,k,case)
            result['gates'].append({'shape':[m,n,k],'case':case,'bit_equal':exact,
                                    'quant_bytes_equal':True,'scales_equal':True,'relative':rel,'over_ulp':over})
            save()
        x.normal_().mul_(.3)
        for cache in ('cold','warm'):
            times={0:[],1:[]}; paired=[]
            for rep in range(args.samples):
                order=(0,1) if rep%2==0 else (1,0)
                pair={}
                for mode in order:
                    for _ in range(2): graphs[mode].replay()
                    if cache=='cold': flush.zero_()
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); graphs[mode].replay(); end.record(); end.synchronize()
                    us=start.elapsed_time(end)*1000
                    times[mode].append(us); pair[mode]=us
                paired.append(100*(pair[0]-pair[1])/pair[0])
            baseline,candidate=median(times[0]),median(times[1])
            row={'shape':[m,n,k],'cache':cache,'plan':ext.gemm2_plan(m,n,k),
                 'baseline_us':baseline,'candidate_us':candidate,
                 'latency_reduction_pct':100*(baseline-candidate)/baseline,
                 'wins':sum(p>0 for p in paired),'samples':args.samples,
                 'baseline_samples_us':times[0],'candidate_samples_us':times[1],
                 'paired_reduction_pct':paired}
            result['timings'].append(row);save()
            print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list) or k in ('shape','plan')}),flush=True)
        del graphs,outputs,x,pack,wref,aq,asc
    result['verdict']='PASS numerical gates; timing verdict requires all recorded shapes'
    save();print(result['verdict'],flush=True)


if __name__=='__main__': main()
