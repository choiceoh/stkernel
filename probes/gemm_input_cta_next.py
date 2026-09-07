#!/usr/bin/env python3
"""Compare input prefetch and vector reduction against the enabled CTA=2 kernel."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
from gemm_input_reuse import FLAGS, ROOT, replace_once

MODES = ('default_cta2', 'early_input', 'vector_reduce', 'early_vector',
         'early_vector_nb3', 'early_vector_nb4')


def render(source):
    start=source.index('template <int MODE>\n__global__ void __launch_bounds__(MK_THREADS,MODE==2?4:3)')
    end=source.index('// ===========================================================================\n// MK_SEG_MHC',start)
    kernel=source[start:end]
    kernel=replace_once(kernel,'template <int MODE>', 'template <int NEXT>')
    kernel=replace_once(kernel,'MODE==2?4:3', 'NEXT==5?2:3')
    kernel=replace_once(kernel,'mk_gemm_input_cta_kernel', 'mk_gemm_input_next_kernel')
    kernel=replace_once(kernel,'  constexpr int NB=2;', '  constexpr int MODE=1;\n  constexpr int NB=NEXT==4?3:NEXT==5?4:2;')
    kernel=replace_once(kernel,'    uint32_t l0a[2],l1a[2],l0b[2],l1b[2];int slot[2];', '''
    uint2 early_x[4];
    float early_s0=0.f,early_s1=0.f;
    if constexpr (NEXT!=2) {
#pragma unroll
      for(int ks=0;ks<4;++ks) {
        early_x[ks]=make_uint2(0,0);
        if(g<6)early_x[ks]=*(const uint2*)(c.input_q+(size_t)kb*1024+ks*256+lane*8);
      }
      early_s0=2*q<6?c.input_s[kb*8+2*q]*c.wgs:0.f;
      early_s1=2*q+1<6?c.input_s[kb*8+2*q+1]*c.wgs:0.f;
    }
    uint32_t l0a[2],l1a[2],l0b[2],l1b[2];int slot[2];''')
    kernel=replace_once(kernel,'      if(g<m)x=*(const uint2*)(c.input_q+(size_t)kb*1024+ks*256+lane*8);', '''
      if constexpr (NEXT!=2) x=early_x[ks];
      else if(g<m)x=*(const uint2*)(c.input_q+(size_t)kb*1024+ks*256+lane*8);''')
    kernel=replace_once(kernel,'    const float s0=2*q<m?c.input_s[kb*8+2*q]*c.wgs:0.f;\n    const float s1=2*q+1<m?c.input_s[kb*8+2*q+1]*c.wgs:0.f;', '''
    const float s0=NEXT!=2?early_s0:(2*q<m?c.input_s[kb*8+2*q]*c.wgs:0.f);
    const float s1=NEXT!=2?early_s1:(2*q+1<m?c.input_s[kb*8+2*q+1]*c.wgs:0.f);''')
    kernel=replace_once(kernel,'if(offset+1<4) {mk_cp_wait_upto(0);__syncwarp();}',
                        'if(offset+1<4) {mk_cp_wait_upto(min(DIST-1,2-offset));__syncwarp();}')
    before='''  for(int t=threadIdx.x;t<m*16;t+=MK_THREADS) {
    const int row=t/16,col=nt*16+t%16;
    float value=0.f;
#pragma unroll
    for(int s=0;s<8;++s)value+=partial[s*m*16+t];
    value*=c.rgs?c.rgs[col]:1.f;
    c.out[(size_t)row*c.n_orig+col]=__float2bfloat16(value);
  }'''
    after='''  if constexpr (NEXT>=2) {
    if(threadIdx.x<24) {
      const int t=threadIdx.x*4,row=t/16,col=nt*16+t%16;
      float4 value=make_float4(0.f,0.f,0.f,0.f);
#pragma unroll
      for(int s=0;s<8;++s) {
        const float4 v=*(const float4*)(partial+s*96+t);
        value.x+=v.x;value.y+=v.y;value.z+=v.z;value.w+=v.w;
      }
      const float4 rg=c.rgs?*(const float4*)(c.rgs+col):make_float4(1.f,1.f,1.f,1.f);
      __nv_bfloat162* out=(__nv_bfloat162*)(c.out+(size_t)row*c.n_orig+col);
      out[0]=__floats2bfloat162_rn(value.x*rg.x,value.y*rg.y);
      out[1]=__floats2bfloat162_rn(value.z*rg.z,value.w*rg.w);
    }
  } else {
'''+before+'''
  }'''
    kernel=replace_once(kernel,before,after)
    source=source[:end]+kernel+source[end:]
    source=replace_once(source,'bool g_attrs_set = false;', '''
int g_input_next=0;
constexpr int INPUT_NEXT_SMEM2=MK_SMEM_ALIGN+2*W4_RAW_BYTES+8*6*16*sizeof(float);
constexpr int INPUT_NEXT_SMEM3=MK_SMEM_ALIGN+3*W4_RAW_BYTES+8*6*16*sizeof(float);
constexpr int INPUT_NEXT_SMEM4=MK_SMEM_ALIGN+4*W4_RAW_BYTES+8*6*16*sizeof(float);
bool g_attrs_set = false;''')
    attrs='\n'.join(f'  MK_CHECK_CUDA(cudaFuncSetAttribute(mk_gemm_input_next_kernel<{m}>,cudaFuncAttributeMaxDynamicSharedMemorySize,INPUT_NEXT_SMEM{3 if m==4 else 4 if m==5 else 2}));' for m in range(1,6))
    source=replace_once(source,'  if (g_attrs_set) return;', '  if (g_attrs_set) return;\n'+attrs)
    launches='\n'.join(f'      {"if" if m==1 else "else if"} (g_input_next=={m}) mk_launch(mk_gemm_input_next_kernel<{m}>,c2.n_orig/16,INPUT_NEXT_SMEM{3 if m==4 else 4 if m==5 else 2},stream,c2);' for m in range(1,6))
    source=replace_once(source,'    if (cta && c2.ksr==8) {', '''    if (g_input_next && c2.ksr==8) {
'''+launches+'''
    } else if (cta && c2.ksr==8) {''')
    info='\n'.join(f'    note(mk_gemm_input_next_kernel<{m}>,INPUT_NEXT_SMEM{3 if m==4 else 4 if m==5 else 2});' for m in range(1,6))
    source=replace_once(source,'  m.def("run_gemm",', '''
  m.def("set_input_next",[](int mode){TORCH_CHECK(mode>=0 && mode<=5);g_input_next=mode;});
  m.def("input_next_info",[](){
    set_kernel_attrs();std::vector<int64_t> out;
    auto note=[&](auto kernel,int smem){
      cudaFuncAttributes a{};int bps=0;
      MK_CHECK_CUDA(cudaFuncGetAttributes(&a,kernel));
      MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps,kernel,MK_THREADS,smem));
      out.insert(out.end(),{a.numRegs,(int64_t)a.localSizeBytes,bps,smem});
    };
    note(mk_gemm_input_cta_kernel<1>,INPUT_CTA_SMEM);
'''+info+'''
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
    args=ap.parse_args()
    os.environ.update(MAX_JOBS='1',VLLM_GLM53_MK_PDL='1',VLLM_GLM53_MK_INPUT_REUSE='1',VLLM_GLM53_MK_INPUT_CTA='2')
    import torch
    from torch.utils.cpp_extension import load
    original=(ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.cu').read_text()
    source=render(original)
    sha=hashlib.sha256(source.encode()).hexdigest()
    build=args.build_dir/sha[:16];build.mkdir(parents=True,exist_ok=True)
    cu=build/'candidate.cu';cu.write_text(source)
    ext=load(name='input_next_'+sha[:16],sources=[str(cu)],extra_cuda_cflags=FLAGS,
             build_directory=str(build),verbose=False)
    if args.compile_only:
        print('PASS compile',sha,flush=True);return
    assert ext.probe_device()[:3]==[12,1,48]
    spec=importlib.util.spec_from_file_location('input_next_driver', ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.py')
    mk=importlib.util.module_from_spec(spec);spec.loader.exec_module(mk);mk._EXT=ext
    torch.backends.cuda.matmul.allow_tf32=False
    ext.set_gemm_input(1);ext.set_input_cta(2);ext.set_gemm2(0)
    modes=tuple(range(len(MODES)))
    result={'source_sha256':sha,'baseline_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'flags':FLAGS,'torch':torch.__version__,'device':torch.cuda.get_device_name(),
            'mode_names':MODES,'kernel_info_regs_local_bps_smem':ext.input_next_info(),
            'gates':[],'timings':[],'status':'RUNNING'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    save();print(json.dumps({k:v for k,v in result.items() if k not in ('gates','timings')}),flush=True)
    flush=torch.zeros(64*1024*1024,dtype=torch.uint8,device='cuda')
    retained=[]
    shapes=((6,6416,4096,False),(6,6416,4096,False),(6,6416,4096,True),
            (1,6416,4096,False),(8,6416,4096,False),(6,6528,4096,False),
            (6,6144,4096,False),(6,4096,512,True))
    for index,(m,n,k,bg) in enumerate(shapes):
        torch.manual_seed(20260908+index)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.3
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.05
        pack=mk.build_mk_weight_w4(w);del w
        wr=mk.mk_w4_dequant(pack[0],pack[1],n,pack[2],pack[3] if len(pack)>3 else None).float()
        graphs={};outputs={}
        for mode in modes:
            ext.set_input_next(mode)
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
    for mode in modes:
        ext.set_input_next(mode)
        assert mk._selftest_input_reuse()
    result.update(status='PASS',alternating_graph_replays=20*len(retained)*(len(modes)-1),boot_gate=True);save()
    print('PASS exact outputs, independent oracle, retained graphs and startup gate',flush=True)


if __name__=='__main__':main()
