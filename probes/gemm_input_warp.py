#!/usr/bin/env python3
"""Probe a warp-independent consumer of prepacked FP8 decode inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median

from gemm_input_reuse import FLAGS, ROOT, render as render_reuse, replace_once


def render(original):
    source=render_reuse(original)
    source=replace_once(source, '__global__ void mk_probe_pack_input(MKProbeInput args)',
                        'template <bool WARP_LAYOUT>\n__global__ void mk_probe_pack_input(MKProbeInput args)')
    source=replace_once(source,
        '    *(uint32_t*)(g_mk2_aq + ((size_t)kb * 32 + row) * KSTEP + lane * 4) = packed;', """
    size_t offset;
    if constexpr (WARP_LAYOUT) {
      const int q = lane >> 3, word = lane & 7;
      const int ks = ((word >> 1) - q) & 3;
      offset = (size_t)kb * 32 * KSTEP + ks * 256 + (row * 4 + q) * 8 + (word & 1) * 4;
    } else {
      offset = ((size_t)kb * 32 + row) * KSTEP + lane * 4;
    }
    *(uint32_t*)(g_mk2_aq + offset) = packed;""")
    source=replace_once(source,
        '    mk_launch(mk_probe_pack_input, c2.k / KSTEP, 0, stream, MKProbeInput{c2.x, c2.m, c2.k});', """
    if (g_probe_input_reuse == 3)
      mk_launch(mk_probe_pack_input<true>, c2.k / KSTEP, 0, stream, MKProbeInput{c2.x, c2.m, c2.k});
    else
      mk_launch(mk_probe_pack_input<false>, c2.k / KSTEP, 0, stream, MKProbeInput{c2.x, c2.m, c2.k});""")
    start=original.index('template <int RQ, bool LR, bool COMPACT = false>')
    end=original.index('// ===========================================================================\n// MK_SEG_MHC',start)
    kernel=original[start:end]
    kernel=replace_once(kernel, '''template <int RQ, bool LR, bool COMPACT = false>
__global__ void __launch_bounds__(MK_THREADS, (MK_COMPACT_M8 && COMPACT) ? 3 : 2)
mk_gemm2_kernel(const MKGemm2Ctx c) {''', '''template <bool WARP_LAYOUT>
__global__ void __launch_bounds__(MK_THREADS, 3)
mk_probe_warp_kernel(const MKGemm2Ctx c) {
  constexpr int RQ = 1;
  constexpr bool LR = false, COMPACT = true;''')
    kernel=replace_once(kernel, 'uint8_t* sraw = (uint8_t*)(sxs + 2 * 32);',
                        'uint8_t* sraw = sb0;')
    kernel=replace_once(kernel, 'const int t = (int)threadIdx.x + u * MK_THREADS;',
                        'const int t = ((int)threadIdx.x >> 5) * 64 + (threadIdx.x & 31) + u * 32;')
    kernel=replace_once(kernel, '''    if (threadIdx.x < SMEM_W_ROWS * 8 / 16)
      mk_cp_async16(d + W4_RAW_NIB + threadIdx.x * 16,
                    ssrc + (size_t)threadIdx.x * 16);''', '''    const int st = ((int)threadIdx.x >> 5) * 8 + (threadIdx.x & 31);
    if ((threadIdx.x & 31) < 8)
      mk_cp_async16(d + W4_RAW_NIB + st * 16, ssrc + (size_t)st * 16);''')
    a=kernel.index('  // x -> registers')
    b=kernel.index('  const int lane = threadIdx.x & 31;',a)
    kernel=kernel[:a]+kernel[b:]
    kernel=replace_once(kernel, 'auto mma_fold = [&](int rbuf, int abuf) {',
                        'auto mma_fold = [&](int rbuf, int kb) {\n    constexpr int abuf = 0;')
    kernel=replace_once(kernel, '''          x0 = *(const uint32_t*)(sa + g * SMEM_A_PITCH + mk_swz(g, koff));
          x1 = *(const uint32_t*)(sa + g * SMEM_A_PITCH + mk_swz(g, koff + 4));''', '''          const size_t xoff = WARP_LAYOUT ? ks * 256 + lane * 8 : g * KSTEP + koff;
          const uint2 xv = *(const uint2*)(g_mk2_aq + (size_t)kb * 32 * KSTEP + xoff);
          x0 = xv.x; x1 = xv.y;''')
    kernel=replace_once(kernel, 'const float s0 = (2 * q < c.m) ? sxs[abuf * 32 + 2 * q] : 0.0f;',
                        'const float s0 = (2 * q < c.m) ? g_mk2_axs[(2*q)*KBLK_MAX+kb] * c.wgs : 0.0f;')
    kernel=replace_once(kernel, 'const float s1 = (2 * q + 1 < c.m) ? sxs[abuf * 32 + 2 * q + 1] : 0.0f;',
                        'const float s1 = (2 * q + 1 < c.m) ? g_mk2_axs[(2*q+1)*KBLK_MAX+kb] * c.wgs : 0.0f;')
    a=kernel.index('  // ---- prologue:')
    b=kernel.index('  // pair_act:',a)
    loop=kernel[a:b]
    for line in ('  load_x(kb0);\n','  quant_x(0);\n',
                 '    if (kb + 1 < kbn) load_x(kb + 1);\n',
                 '    quant_x((kb + 1 - kb0) & 1);\n'):
        loop=replace_once(loop,line,'')
    loop=replace_once(loop,'mma_fold(kb % NB, (kb - kb0) & 1);','mma_fold(kb % NB, kb);')
    assert loop.count('__syncthreads();')==2
    loop=loop.replace('__syncthreads();','__syncwarp();')
    kernel=kernel[:a]+loop+kernel[b:]
    source=replace_once(source,'bool g_attrs_set = false;',
        'constexpr int MK_PROBE_WARP_SMEM = MK_SMEM_ALIGN + W4_RAW_NBUF2 * W4_RAW_BYTES;\n'+kernel+'\nbool g_attrs_set = false;')
    source=replace_once(source,'  if (g_attrs_set) return;', '''  if (g_attrs_set) return;
  MK_CHECK_CUDA(cudaFuncSetAttribute(mk_probe_warp_kernel<false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, MK_PROBE_WARP_SMEM));
  MK_CHECK_CUDA(cudaFuncSetAttribute(mk_probe_warp_kernel<true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, MK_PROBE_WARP_SMEM));''')
    source=replace_once(source,'    TORCH_CHECK(mode == 0 || mode == 1); g_probe_input_reuse = mode;',
                        '    TORCH_CHECK(mode >= 0 && mode <= 3); g_probe_input_reuse = mode;')
    source=replace_once(source,'  mk_launch_gemm2(c2, stream);\n}', '''  if (g_probe_input_reuse >= 2 && c2.m >= 1 && c2.m <= 8 && !bg && !c2.lr_r) {
    if (g_probe_input_reuse == 3)
      mk_launch(mk_probe_warp_kernel<true>, (c2.n / SMEM_W_ROWS) * c2.ksr,
                MK_PROBE_WARP_SMEM, stream, c2);
    else
      mk_launch(mk_probe_warp_kernel<false>, (c2.n / SMEM_W_ROWS) * c2.ksr,
                MK_PROBE_WARP_SMEM, stream, c2);
  } else {
    mk_launch_gemm2(c2, stream);
  }
}''')
    source=replace_once(source,'  m.def("run_gemm",', '''  m.def("warp_info", []() {
    cudaFuncAttributes a{};
    MK_CHECK_CUDA(cudaFuncGetAttributes(&a, mk_probe_warp_kernel<false>));
    int bps=0;
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, mk_probe_warp_kernel<false>, MK_THREADS, MK_PROBE_WARP_SMEM));
    std::vector<int64_t> info{a.numRegs, (int64_t)a.localSizeBytes, bps, MK_PROBE_WARP_SMEM};
    MK_CHECK_CUDA(cudaFuncGetAttributes(&a, mk_probe_warp_kernel<true>));
    MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, mk_probe_warp_kernel<true>, MK_THREADS, MK_PROBE_WARP_SMEM));
    info.insert(info.end(), {a.numRegs, (int64_t)a.localSizeBytes, bps, MK_PROBE_WARP_SMEM});
    return info;
  });
  m.def("run_gemm",''')
    return source


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir',type=Path,default=Path('/build'))
    ap.add_argument('--out',type=Path,default=Path('/evidence/result.json'))
    ap.add_argument('--samples',type=int,default=24)
    ap.add_argument('--compile-only',action='store_true')
    args=ap.parse_args()
    os.environ.update(MAX_JOBS='1',VLLM_GLM53_MK_PDL='1')
    import torch
    from torch.utils.cpp_extension import load
    original=(ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.cu').read_text()
    source=render(original); sha=hashlib.sha256(source.encode()).hexdigest()
    build=args.build_dir/sha[:16];build.mkdir(parents=True,exist_ok=True)
    cu=build/'candidate.cu';cu.write_text(source)
    ext=load(name='input_warp_'+sha[:16],sources=[str(cu)],extra_cuda_cflags=FLAGS,
             build_directory=str(build),verbose=False)
    if args.compile_only:
        print('PASS native build',sha,flush=True);return
    spec=importlib.util.spec_from_file_location('input_warp_driver',ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.py')
    mk=importlib.util.module_from_spec(spec);spec.loader.exec_module(mk);mk._EXT=ext
    assert ext.probe_device()[:3]==[12,1,48]
    torch.backends.cuda.matmul.allow_tf32=False
    identity={'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
              'generated_sha256':sha,'flags':FLAGS,'device':torch.cuda.get_device_name(),
              'torch':torch.__version__,'cuda':torch.version.cuda,
              'warp_info_natural_then_coalesced_regs_local_bytes_bps_smem':ext.warp_info()}
    r={'identity':identity,'gates':[],'timings':[],'status':'RUNNING'}
    def save():args.out.write_text(json.dumps(r,indent=2)+'\n')
    args.out.parent.mkdir(parents=True,exist_ok=True);save();print(json.dumps(identity),flush=True)
    flush=torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda')
    shapes=((6,6528,4096),(6,6144,4096),(6,4096,4096),(6,1024,4096),(6,4096,512),(1,6528,4096),(8,6528,4096))
    for m,n,k in shapes:
        torch.manual_seed(1847+m+n+k)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.3
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.05
        pack=mk.build_mk_weight_w4(w);del w
        wr=mk.mk_w4_dequant(pack[0],pack[1],n,pack[2],pack[3] if len(pack)>3 else None).float()
        default=ext.gemm2_plan(m,n,k)[0]
        configs=[('base',0,default),('reuse',1,default)]
        splits=(1,2,3,4,6,8) if m==6 and k==4096 and n>=4096 else (default,)
        configs += [('warp-k'+str(s),2,s) for s in splits]
        configs += [('coalesced-k'+str(s),3,s) for s in splits]
        graphs={};outputs={}
        for name,mode,split in configs:
            ext.set_input_reuse(mode);ext.set_gemm2(split)
            for _ in range(2):mk._gemm_call(x,pack,n)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):outputs[name]=mk._gemm_call(x,pack,n)
            graphs[name]=graph
        # Repeat with changing values to detect partial/scratch replay hazards.
        for case in ('random','zero','tiny','wide','random2'):
            if case=='zero':x.zero_()
            else:x.normal_().mul_({'tiny':1e-20,'wide':1e3}.get(case,.3))
            ref=mk._mk_quant_x_ref(x)@wr.T
            for name,mode,split in configs:
                graphs[name].replay();torch.cuda.synchronize()
                y=outputs[name]; rel,over=mk._exact_gate(y,ref)
                finite=bool(torch.isfinite(y).all()); exact=bool(torch.equal(y,outputs['base']))
                gate={'shape':[m,n,k],'case':case,'config':name,'split':split,
                      'relative':rel,'over_ulp':over,'finite':finite,'bit_equal_base':exact}
                r['gates'].append(gate);save()
                assert finite and rel<=1e-3 and over==0,gate
                if split==default:assert exact,gate
        x.normal_().mul_(.3)
        for cache in ('cold','warm'):
            for name,mode,split in configs[1:]:
                times={'base':[],name:[]}
                for rep in range(args.samples):
                    order=('base',name) if rep%2==0 else (name,'base')
                    for arm in order:
                        for _ in range(2):graphs[arm].replay()
                        if cache=='cold':flush.zero_()
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();graphs[arm].replay();b.record();b.synchronize()
                        times[arm].append(a.elapsed_time(b)*1000)
                base,cand=median(times['base']),median(times[name])
                row={'shape':[m,n,k],'cache':cache,'config':name,'split':split,
                     'baseline_us':base,'candidate_us':cand,'reduction_pct':100*(base-cand)/base,
                     'wins':sum(b>a for b,a in zip(times['base'],times[name])),
                     'samples':args.samples,'raw_us':times}
                r['timings'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='raw_us'}),flush=True)
        ext.set_gemm2(0)
        del graphs,outputs,x,pack,wr
    r['status']='PASS';save();print('PASS warp consumer sweep',flush=True)


if __name__=='__main__':main()
