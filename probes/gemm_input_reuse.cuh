// Probe only: produce exactly the existing GEMM's per-row FP8 groups once.
// The single-stream probe reuses SMLP2's scratch and a_ready consumer. A
// serving implementation would need an explicit lifetime/stream contract.
struct MKProbeInput { const __nv_bfloat16* x; int m, k; };
__global__ void mk_probe_pack_input(MKProbeInput args) {
  const __nv_bfloat16* x = args.x;
  const int m = args.m, k = args.k;
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int row = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kb = blockIdx.x;
  float v[4] = {0, 0, 0, 0};
  float mx = 0;
  if (row < m) {
    const uint2 raw = *(const uint2*)(x + (size_t)row * k + kb * KSTEP + lane * 4);
    const __nv_bfloat16* bf = (const __nv_bfloat16*)&raw;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = __bfloat162float(bf[j]);
      mx = fmaxf(mx, fabsf(v[j]));
    }
  }
  mx = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(mx)));
  if (row < m) {
    const float sc = mk_act_scale(mx), inv = mk_act_rcp(sc);
    const uint32_t packed = mk_f32x4_to_e4m3(v[0]*inv, v[1]*inv, v[2]*inv, v[3]*inv);
    *(uint32_t*)(g_mk2_aq + ((size_t)kb * 32 + row) * KSTEP + lane * 4) = packed;
    if (lane == 0) g_mk2_axs[row*KBLK_MAX + kb] = sc;
  }
}
