#!/usr/bin/env python3
"""Three-slice CTA probe for foreground M6/N4096|6144/K4096."""
from pathlib import Path
from gemm_input_reuse import replace_once

MODES = ('default_cta2', 'three_warps_nb2', 'six_warps_nb2',
         'three_warps_nb3', 'six_warps_nb3')


def render(source):
    start=source.index('template <int MODE>\n__global__ void __launch_bounds__(MK_THREADS,MODE==2?4:3)')
    end=source.index('// ===========================================================================\n// MK_SEG_MHC',start)
    kernel=source[start:end]
    kernel=replace_once(kernel,'template <int MODE>', 'template <int TILES, int NB>')
    kernel=replace_once(kernel,'__launch_bounds__(MK_THREADS,MODE==2?4:3)',
                        '__launch_bounds__(TILES*96,TILES==1?6:3)')
    kernel=replace_once(kernel,'mk_gemm_input_cta_kernel', 'mk_gemm_input_cta3_kernel')
    kernel=replace_once(kernel,'  constexpr int NB=2;\n  const int m=MODE?6:c.m;',
                        '  constexpr int MODE=0,m=6;\n  constexpr int RAW_NIB=TILES*48*64,RAW_BYTES=TILES*48*72;')
    kernel=kernel.replace('W4_RAW_BYTES','RAW_BYTES').replace('W4_RAW_NIB','RAW_NIB')
    kernel=replace_once(kernel,'  const int kblk=MODE?32:c.k/KSTEP, nt=blockIdx.x, slice=warp;\n  const int kb0=MODE?warp*4:kblk*slice/c.ksr;\n  const int kbn=MODE?kb0+4:kblk*(slice+1)/c.ksr;',
                        '  const int kblk=32,nt=blockIdx.x*TILES+warp/3,slice=warp%3;\n  const int kb0=32*slice/3,kbn=32*(slice+1)/3;')
    # The dynamic loop preserves the baseline's 10/11/11 K-group partitions.
    # Each warp stages its own weights; only the final reduction is CTA-wide.
    kernel=replace_once(kernel,'  for(int t=threadIdx.x;t<m*16;t+=MK_THREADS) {\n    const int row=t/16,col=nt*16+t%16;',
                        '  for(int t=threadIdx.x;t<TILES*96;t+=TILES*96) {\n    const int tile=t/96,local=t%96,row=local/16;\n    const int col=(blockIdx.x*TILES+tile)*16+local%16;')
    kernel=replace_once(kernel,'    for(int s=0;s<8;++s)value+=partial[s*m*16+t];',
                        '    for(int s=0;s<3;++s)value+=partial[(tile*3+s)*96+local];')
    source=source[:end]+kernel+source[end:]
    source=replace_once(source,'bool g_attrs_set = false;', '''
int g_input_cta3=0;
template<int TILES,int NB> constexpr int CTA3_SMEM=MK_SMEM_ALIGN+NB*TILES*48*72+TILES*3*96*sizeof(float);
bool g_attrs_set = false;''')
    pairs=((1,2),(2,2),(1,3),(2,3))
    attrs='\n'.join(f'  MK_CHECK_CUDA(cudaFuncSetAttribute(mk_gemm_input_cta3_kernel<{t},{nb}>,cudaFuncAttributeMaxDynamicSharedMemorySize,CTA3_SMEM<{t},{nb}>));' for t,nb in pairs)
    source=replace_once(source,'  if (g_attrs_set) return;', '  if (g_attrs_set) return;\n'+attrs)
    # Use the same PDL contract, but exactly three or six active warps.
    start=source.index('template <typename K, typename A>\nvoid mk_launch(')
    end=source.index('\nint g_input_cta_mode',start)
    launcher=source[start:end].replace('template <typename K, typename A>',
                                     'template <int THREADS, typename K, typename A>')
    launcher=replace_once(launcher,'void mk_launch(', 'void mk_launch_cta3(')
    launcher=replace_once(launcher,'dim3(MK_THREADS)', 'dim3(THREADS)')
    source=source[:end]+launcher+source[end:]
    source=replace_once(source,'  const bool input_reuse = mk_gemm_input_mode() &&\n      mk_input_shape(c2.m, c2.n_orig, c2.k, bg != 0, c2.lr_r != 0);', '''
  const bool cta3=g_input_cta3 && !bg && !c2.lr_r && c2.m==6 && c2.k==4096
      && (c2.n_orig==4096 || c2.n_orig==6144) && c2.ksr==3;
  const bool input_reuse = cta3 || (mk_gemm_input_mode() &&
      mk_input_shape(c2.m, c2.n_orig, c2.k, bg != 0, c2.lr_r != 0));''')
    launches='\n'.join(f'      {"if" if mode==1 else "else if"} (g_input_cta3=={mode}) mk_launch_cta3<{t*96}>(mk_gemm_input_cta3_kernel<{t},{nb}>,c2.n_orig/{16*t},CTA3_SMEM<{t},{nb}>,stream,c2);' for mode,(t,nb) in enumerate(pairs,1))
    source=replace_once(source,'    if (cta && c2.ksr==8) {',
                        '    if (cta3) {\n'+launches+'\n    } else if (cta && c2.ksr==8) {')
    info='\n'.join(f'    note(mk_gemm_input_cta3_kernel<{t},{nb}>,{t*96},CTA3_SMEM<{t},{nb}>);' for t,nb in pairs)
    source=replace_once(source,'  m.def("run_gemm",', '''
  m.def("set_input_next",[](int mode){TORCH_CHECK(mode>=0 && mode<=4);g_input_cta3=mode;});
  m.def("input_next_info",[](){
    set_kernel_attrs();std::vector<int64_t> out;
    auto note=[&](auto kernel,int threads,int smem){
      cudaFuncAttributes a{};int bps=0;
      MK_CHECK_CUDA(cudaFuncGetAttributes(&a,kernel));
      MK_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps,kernel,threads,smem));
      out.insert(out.end(),{a.numRegs,(int64_t)a.localSizeBytes,bps,smem});
    };
    note(mk_gemm_input_cta_kernel<1>,MK_THREADS,INPUT_CTA_SMEM);
'''+info+'''
    return out;
  });
  m.def("run_gemm",''')
    return source


if __name__=='__main__':
    import gemm_input_cta_next as harness
    harness.render=render
    harness.MODES=MODES
    harness.SHAPES=((6,4096,4096,False),(6,6144,4096,False),
                    (6,6416,4096,False),(6,4096,4096,True),
                    (1,4096,4096,False),(8,6144,4096,False),
                    (6,6528,4096,False),(6,4096,512,True))
    harness.TIMED_INDICES=(0,1,2)
    harness.main()
