#!/usr/bin/env python3
"""Same-build probe: keep eight split-K partials within one output-tile CTA."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
from gemm_input_reuse import FLAGS, ROOT, replace_once


def render(source):
    kernel=(ROOT/'probes/gemm_input_cta.cuh').read_text()
    source=replace_once(source, '// ===========================================================================\n// MK_SEG_MHC',
                        kernel+'\n// ===========================================================================\n// MK_SEG_MHC')
    source=replace_once(source, 'bool g_attrs_set = false;', '''
constexpr int INPUT_CTA_SMEM2=MK_SMEM_ALIGN+2*W4_RAW_BYTES+8*6*16*sizeof(float);
constexpr int INPUT_CTA_SMEM3=MK_SMEM_ALIGN+3*W4_RAW_BYTES+8*6*16*sizeof(float);
int g_probe_input_cta=0;
bool g_attrs_set = false;''')
    source=replace_once(source,'  if (g_attrs_set) return;', '''  if (g_attrs_set) return;
  MK_CHECK_CUDA(cudaFuncSetAttribute(mk_gemm_input_cta_kernel<2>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, INPUT_CTA_SMEM2));
  MK_CHECK_CUDA(cudaFuncSetAttribute(mk_gemm_input_cta_kernel<3>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, INPUT_CTA_SMEM3));''')
    source=replace_once(source,
        '    mk_launch(mk_gemm_input_kernel, nblk * c2.ksr, GEMM_INPUT_SMEM, stream, c2);', '''
    if (g_probe_input_cta && c2.ksr==8) {
      if (g_probe_input_cta==1)
        mk_launch(mk_gemm_input_cta_kernel<2>, c2.n_orig/16, INPUT_CTA_SMEM2, stream, c2);
      else
        mk_launch(mk_gemm_input_cta_kernel<3>, c2.n_orig/16, INPUT_CTA_SMEM3, stream, c2);
    } else {
      mk_launch(mk_gemm_input_kernel, nblk * c2.ksr, GEMM_INPUT_SMEM, stream, c2);
    }''')
    source=replace_once(source, '  m.def("run_gemm",', '''
  m.def("set_input_cta", [](int mode) {
    TORCH_CHECK(mode>=0 && mode<=2); g_probe_input_cta=mode;
  });
  m.def("input_cta_info", []() {
    set_kernel_attrs();
    std::vector<int64_t> out;
    auto note=[&](auto kernel,int smem) {
      cudaFuncAttributes a{};int bps=0;
      MK_CHECK_CUDA(cudaFuncGetAttributes(&a,kernel));
      MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps,kernel,MK_THREADS,smem));
      out.insert(out.end(),{a.numRegs,(int64_t)a.localSizeBytes,bps,smem});
    };
    note(mk_gemm_input_kernel,GEMM_INPUT_SMEM);
    note(mk_gemm_input_cta_kernel<2>,INPUT_CTA_SMEM2);
    note(mk_gemm_input_cta_kernel<3>,INPUT_CTA_SMEM3);
    return out;
  });
  m.def("run_gemm",''')
    return source


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir',type=Path,default=Path('/build'))
    ap.add_argument('--out',type=Path,default=Path('/evidence/result.json'))
    ap.add_argument('--samples',type=int,default=32)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--production',action='store_true',help='compile the actual serving source and its three CTA modes')
    args=ap.parse_args()
    os.environ.update(MAX_JOBS='1',VLLM_GLM53_MK_PDL='1')
    import torch
    from torch.utils.cpp_extension import load
    original=(ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.cu').read_text()
    args.production = args.production or 'int g_input_cta_mode = -1;' in original
    source=original if args.production else render(original)
    modes=(0,1,2,3) if args.production else (0,1,2)
    sha=hashlib.sha256(source.encode()).hexdigest()
    build=args.build_dir/sha[:16];build.mkdir(parents=True,exist_ok=True)
    cu=build/'candidate.cu';cu.write_text(source)
    ext=load(name='input_cta_'+sha[:16],sources=[str(cu)],extra_cuda_cflags=FLAGS,
             build_directory=str(build),verbose=False)
    if args.compile_only:
        print('PASS compile',sha,flush=True);return
    assert ext.probe_device()[:3]==[12,1,48]
    spec=importlib.util.spec_from_file_location('input_cta_driver', ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.py')
    mk=importlib.util.module_from_spec(spec);spec.loader.exec_module(mk);mk._EXT=ext
    torch.backends.cuda.matmul.allow_tf32=False
    ext.set_gemm_input(1);ext.set_gemm2(0)
    result={'source_sha256':sha,'flags':FLAGS,'torch':torch.__version__,
            'device':torch.cuda.get_device_name(),'mode_names':(['default','generic_nb2','fixed_nb2_bps3','fixed_nb2_bps4'] if args.production else ['default','generic_nb2','generic_nb3']),
            'kernel_info_regs_local_bps_smem':ext.input_cta_info(),
            'gates':[],'timings':[],'status':'RUNNING'}
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    args.out.parent.mkdir(parents=True,exist_ok=True);save()
    print(json.dumps({k:v for k,v in result.items() if k not in ('gates','timings')}),flush=True)
    flush=torch.zeros(64*1024*1024,dtype=torch.uint8,device='cuda')
    retained=[]
    # Two real-shape layers, plus all runtime selector boundaries.
    shapes=((6,6416,4096,False),(6,6416,4096,False),(6,6416,4096,True),
            (1,6416,4096,False),(8,6416,4096,False),(6,6528,4096,False),
            (6,6144,4096,False),(6,4096,512,True))
    for index,(m,n,k,bg) in enumerate(shapes):
        torch.manual_seed(20260907+index)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.3
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.05
        pack=mk.build_mk_weight_w4(w);del w
        wr=mk.mk_w4_dequant(pack[0],pack[1],n,pack[2],pack[3] if len(pack)>3 else None).float()
        graphs={};outputs={}
        for mode in modes:
            ext.set_input_cta(mode)
            if args.production:
                plan=ext.gemm_input_cta_plan(m,n,k,bg,False)
                active=(m,n,k,bg)==(6,6416,4096,False)
                assert plan[0]==(mode if active else 0),(m,n,k,bg,mode,plan)
                assert ext.gemm_input_cta_plan(m,n,k,bg,True)[0]==0
            for _ in range(2):mk._gemm_call(x,pack,n,bg=bg)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs[mode]=mk._gemm_call(x,pack,n,bg=bg)
            graphs[mode]=graph
        for case in ('random','zero','tiny','wide','random2'):
            if case=='zero':x.zero_()
            else:x.normal_().mul_({'tiny':1e-20,'wide':1e3}.get(case,.3))
            ref=mk._mk_quant_x_ref(x)@wr.T
            for mode in modes:
                graphs[mode].replay();torch.cuda.synchronize()
                rel,over=mk._exact_gate(outputs[mode],ref)
                finite=bool(torch.isfinite(outputs[mode]).all())
                exact=bool(torch.equal(outputs[0],outputs[mode]))
                row={'shape':[m,n,k],'index':index,'bg':bg,'mode':mode,'case':case,
                     'relative':rel,'over_ulp':over,'finite':finite,'exact':exact}
                result['gates'].append(row);save()
                assert finite and exact and rel<=1e-3 and over==0,row
        if index<2:retained.append((x,pack,wr,graphs,outputs))
        if index==0 and not args.check_only:
            x.normal_().mul_(.3)
            for cache in ('warm','read_evicted'):
                times={mode:[] for mode in modes}
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();b.record();b.synchronize()
                for rep in range(args.samples):
                    order=modes if rep%2==0 else tuple(reversed(modes))
                    for mode in order:
                        for _ in range(16):graphs[mode].replay()
                        if cache=='read_evicted':flush.sum()
                        a.record();graphs[mode].replay();b.record();b.synchronize()
                        times[mode].append(a.elapsed_time(b)*1000)
                med={mode:median(values) for mode,values in times.items()}
                row={'cache':cache,'median_us':med,'raw_us':times,
                     'reduction_pct':{mode:100*(med[0]-med[mode])/med[0] for mode in modes[1:]}}
                result['timings'].append(row);save();print(json.dumps(row),flush=True)
    for rep in range(20):
        poison=torch.full((512*1024,),0xA5,dtype=torch.uint8,device='cuda');del poison
        for x,pack,wr,graphs,outputs in retained:
            x.normal_().mul_(.3)
            for mode in reversed(modes):graphs[mode].replay()
        for x,pack,wr,graphs,outputs in reversed(retained):
            ref=mk._mk_quant_x_ref(x)@wr.T
            for mode in modes[1:]:
                rel,over=mk._exact_gate(outputs[mode],ref)
                assert rel<=1e-3 and over==0 and torch.equal(outputs[0],outputs[mode]),(rep,mode,rel,over)
    for mode in modes[1:]:
        ext.set_input_cta(mode)
        assert mk._selftest_input_reuse()
    result.update(status='PASS',alternating_graph_replays=20*len(retained)*(len(modes)-1),boot_gate=True);save()
    print('PASS exact outputs, independent oracle, retained graphs and startup gate',flush=True)


if __name__=='__main__':main()
