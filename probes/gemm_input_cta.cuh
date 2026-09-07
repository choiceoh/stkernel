// Private probe: one CTA per 16 output columns, one warp per original K slice.
// W4 packs, FP8 preparation, per-slice arithmetic and reduction order are unchanged.
// Exact route only: M6/N6416/K4096 and split8. NB is the per-warp W staging depth.
template <int NB>
__global__ void __launch_bounds__(MK_THREADS,3)
mk_gemm_input_cta_kernel(const MKGemm2Ctx c) {
  asm volatile("griddepcontrol.launch_dependents;");
  extern __shared__ uint8_t smem[];
  uint8_t* sraw=smem;
  const uint32_t sm=(uint32_t)__cvta_generic_to_shared(sraw);
  sraw+=(MK_SMEM_ALIGN-(sm&(MK_SMEM_ALIGN-1)))&(MK_SMEM_ALIGN-1);
  float* partial=reinterpret_cast<float*>(sraw+NB*W4_RAW_BYTES);
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5, g=lane>>2, q=lane&3;
  const int kblk=c.k/KSTEP, nt=blockIdx.x, slice=warp;
  const int kb0=kblk*slice/c.ksr, kbn=kblk*(slice+1)/c.ksr;
  constexpr int DIST=NB-1;
  auto stage_raw=[&](int kb,int buf) {
    const uint8_t* w=c.wq4+((size_t)(nt/8)*kblk+kb)*8192+(nt%8)*1024;
    const uint8_t* s=(const uint8_t*)c.ws4+((size_t)(nt/8)*kblk+kb)*1024+(nt%8)*128;
    uint8_t* d=sraw+buf*W4_RAW_BYTES;
#pragma unroll
    for(int u=0;u<2;++u) {
      const int t=warp*64+lane+u*32,r=t>>2,ch=t&3;
      mk_cp_async16(d+r*W4_RAW_PITCH+((ch^((r>>1)&3))<<4),w+(size_t)(lane+u*32)*16);
    }
    if(lane<8) {
      const int st=warp*8+lane;
      mk_cp_async16(d+W4_RAW_NIB+st*16,s+(size_t)lane*16);
    }
    mk_cp_commit();
  };
  float acc[4]={};
  auto mma_fold=[&](int kb) {
    const uint8_t* raw=sraw+(kb%NB)*W4_RAW_BYTES;
    uint32_t l0a[2],l1a[2],l0b[2],l1b[2];int slot[2];
#pragma unroll
    for(int j=0;j<2;++j) {
      const int r=warp*16+j*8+g;
      const uint32_t ex=*(const uint16_t*)(raw+W4_RAW_NIB+r*8+2*q);
      const uint32_t ea=ex&255u,eb=ex>>8;
      const unsigned long long la=((ea&7u)-1u)<5u?MK_E2M1_LUT64_B:MK_E2M1_LUT64;
      const unsigned long long lb=((eb&7u)-1u)<5u?MK_E2M1_LUT64_B:MK_E2M1_LUT64;
      l0a[j]=__vadd4((uint32_t)la,ea*0x01010100u);
      l1a[j]=__vadd4((uint32_t)(la>>32),ea*0x01010101u);
      l0b[j]=__vadd4((uint32_t)lb,eb*0x01010100u);
      l1b[j]=__vadd4((uint32_t)(lb>>32),eb*0x01010101u);
      slot[j]=r*W4_RAW_PITCH+((q^((r>>1)&3))<<4);
    }
    float ka[4]={};
#pragma unroll
    for(int ks=0;ks<4;++ks) {
      const int wsel=(ks+q)&3;uint32_t wb[2][2];
#pragma unroll
      for(int j=0;j<2;++j) {
        const uint32_t w=*(const uint32_t*)(raw+slot[j]+4*wsel);
        const uint32_t l0=wsel<2?l0a[j]:l0b[j],l1=wsel<2?l1a[j]:l1b[j];
        wb[j][0]=__byte_perm(l0,l1,w&0x7777u)|__byte_perm(0x8000u,0u,(w>>3)&0x1111u);
        wb[j][1]=__byte_perm(l0,l1,(w>>16)&0x7777u)|__byte_perm(0x8000u,0u,(w>>19)&0x1111u);
      }
      uint2 x=make_uint2(0,0);
      if(g<c.m)x=*(const uint2*)(c.input_q+(size_t)kb*1024+ks*256+lane*8);
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(ka[0]),"+f"(ka[1]),"+f"(ka[2]),"+f"(ka[3])
          : "r"(wb[0][0]),"r"(wb[1][0]),"r"(wb[0][1]),"r"(wb[1][1]),"r"(x.x),"r"(x.y));
    }
    const float s0=2*q<c.m?c.input_s[kb*8+2*q]*c.wgs:0.f;
    const float s1=2*q+1<c.m?c.input_s[kb*8+2*q+1]*c.wgs:0.f;
    acc[0]+=ka[0]*s0;acc[1]+=ka[1]*s1;acc[2]+=ka[2]*s0;acc[3]+=ka[3]*s1;
  };
#pragma unroll
  for(int d=0;d<DIST;++d)if(kb0+d<kbn)stage_raw(kb0+d,(kb0+d)%NB);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  mk_cp_wait_upto(min(DIST-1,kbn-kb0-1));__syncwarp();
  for(int kb=kb0;;++kb) {
    if(kb+DIST<kbn)stage_raw(kb+DIST,(kb+DIST)%NB);
    mma_fold(kb);
    if(kb+1>=kbn)break;
    mk_cp_wait_upto(min(DIST-1,kbn-kb-2));__syncwarp();
  }
  // Each warp owns one of the original eight K slices. Publish all
  // 6x16 partials inside this CTA, then reduce in exactly the old slice order.
  // This removes device-wide partial traffic, arrival atomics and fences.
#pragma unroll
  for(int i=0;i<4;++i) {
    const int row=2*q+(i&1),col=g+(i>=2?8:0);
    if(row<c.m)partial[(warp*c.m+row)*16+col]=acc[i];
  }
  __syncthreads();
  for(int t=threadIdx.x;t<c.m*16;t+=MK_THREADS) {
    const int row=t/16,col=nt*16+t%16;
    float value=0.f;
#pragma unroll
    for(int s=0;s<8;++s)value+=partial[s*c.m*16+t];
    value*=c.rgs?c.rgs[col]:1.f;
    c.out[(size_t)row*c.n_orig+col]=__float2bfloat16(value);
  }
}
