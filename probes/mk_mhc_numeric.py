#!/usr/bin/env python3
"""Run MK_SEG_MHC on the GPU at both hidden sizes and check what it wrote.

The compile probe proves the 5120 instantiation builds and costs no registers.
It cannot prove the kernel writes the right places, and writing the right
places is exactly what the V4.1 change put at risk: HIDDEN 4096 -> 5120 moves
NCHUNK 16 -> 20 and MHC_EPT 16 -> 20, and the MHC_MAX_TOK split moves the yp/rp
strides 32 -> 128. All four are INDEXING, not arithmetic, so the failure mode is
a silently wrong address rather than a wrong formula or a crash.

Rather than port the sinkhorn to the host and risk grading the kernel against a
buggy reference, this feeds inputs whose answer is analytic:

    x_in         = 1
    residual_in  = k + 1            (bf16-exact small integers)
    post_mix_in  = 1
    comb_mix_in  = identity
      =>  r[j] = 1 * 1 + sum_k cm[k][j] * res[k] = 1 + (j + 1) = j + 2
      =>  residual_out[t, j, h] == j + 2  for EVERY t, j and h

That last line is the test. It is T x HC x HID elements, every one of them
determined, so a chunk loop that misses part of the hidden dim, a stride that
assumes the wrong NCHUNK, or a token stride left at 32 all show up as a wrong
value at a specific (t, j, h) rather than as an average that drifts.

There is a second, cross-size invariant. rp[c, t] sums HCHUNK * sum_j r[j]^2 =
256 * 54, and the RMS the tail takes is

    sum_c rp[c, t] / (HC * HID)  =  NCHUNK * 13824 / (4 * HID)

which is 13.5 at HIDDEN 4096 (NCHUNK 16) and 13.5 at 5120 (NCHUNK 20) -- equal
only because NCHUNK tracked HID. A 5120 build that kept NCHUNK at 16 gives 10.8,
and `layer_input` moves with it.

    python3 probes/mk_mhc_numeric.py [--grid 16] [--tokens 1,7,24,32,33,128]

Result, srv4, 2026-09-10, grid 16, T in {1, 7, 24, 32, 33, 64, 127, 128} at both
HIDDEN and HIDDEN_V41: residual_out exact on every element, rms 13.5000 on every
run, layer_input uniform. T of 33 and above would have been rejected outright
before MHC_MAX_TOK was split from MAX_TOK.

The test has teeth, and that was checked rather than assumed. Mutating
`mk_mhc_p1_impl`'s `constexpr int NCHUNK = HID / HCHUNK` back to `HIDDEN /
HCHUNK` -- the exact regression the HID template exists to prevent -- leaves
4096 passing and fails 5120 on all three checks at once:

    residual_out MISMATCH (28672 of 143360, first t0 j0 h4096 = 0)
    rms 10.8000  <-- expected 13.5
    layer_input VARIES

28672 is (5120 - 4096) x HC x T, i.e. exactly the tail of the hidden dim left
unwritten, and 10.8 is 16 * 13824 / 20480. The first-mismatch coordinate names
the bug.

Safety on a serving node: the grid is a launch parameter here, defaulted small
and passed through `-D MK_MHC_GRID_DEF`, so this occupies a handful of blocks
for a few milliseconds rather than asking the device for full residency. The
kernel takes no dynamic shared memory and its spin waits carry a deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
ap.add_argument("--arch", default="sm_121a")
ap.add_argument("--grid", type=int, default=16)
ap.add_argument("--tokens", default="1,7,24,32,33,128")
ap.add_argument("--keep", action="store_true")
args = ap.parse_args()

SRC = (Path(__file__).resolve().parents[1]
       / "overlay/modules/glm53_megakernel/glm53_megakernel.cu")
text = SRC.read_text()


def section(start: str, end: str) -> str:
    assert text.count(start) == 1, (start, text.count(start))
    i = text.index(start)
    return text[i:text.index(end, i)]


def one(pattern: str) -> str:
    found = re.search(pattern, text, re.S)
    assert found, pattern
    return found.group()


unit = ("#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n"
        "#include <stdint.h>\n#include <math.h>\n#include <stdio.h>\n"
        "#include <vector>\n\n")
for pattern in (r"constexpr int MK_THREADS = [0-9]+;",
                r"constexpr int MK_WARPS = [^;]+;",
                r"constexpr int HC = [0-9]+;[^\n]*",
                r"constexpr int NOUT = [^;]+;[^\n]*",
                r"constexpr int MAX_TOK = [0-9]+;[^\n]*",
                r"#define MHC_MAX_TOK_DEF [0-9]+",
                r"constexpr int MHC_MAX_TOK = [^;]+;",
                r"constexpr int HCHUNK = [0-9]+;",
                r"constexpr int HIDDEN = [0-9]+;",
                r"constexpr int HIDDEN_V41 = [0-9]+;",
                r"constexpr int NCHUNK = [^;]+;[^\n]*"):
    unit += one(pattern) + "\n"
unit += one(r"__device__ __forceinline__ float mk_sigmoid\(float x\) \{[^}]*\}") + "\n"
unit += ("\n#define MK_MHC_PROBE(slot) do {} while (0)\n"
         "#define MK_MHC_TS(slot) do {} while (0)\n"
         "#define MK_SPIN_WAIT(cond, ns, site) while (cond) { __nanosleep(ns); }\n"
         "#define TORCH_CHECK(...) do {} while (0)\n\n")
unit += section("struct MKMhcArgs {", "}  // namespace")

unit += r"""
// --------------------------------------------------------------------------
// host harness
// --------------------------------------------------------------------------
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); \
  return 2; } } while (0)

template <int HID>
static int run_one(int T, int grid) {
  constexpr int NCH = HID / HCHUNK;
  MKMhcArgs a{};
  a.num_tokens = T;
  a.grid = grid;
  a.rms_eps = 1e-6f; a.pre_eps = 1e-6f; a.sinkhorn_eps = 1e-6f;
  a.post_mult = 1.0f; a.norm_eps = 1e-6f; a.sinkhorn_repeat = 20;

  auto dev = [](void** p, size_t n) { return cudaMalloc(p, n); };
  std::vector<__nv_bfloat16> h_x((size_t)T * HID), h_res((size_t)T * HC * HID),
      h_nw(HID);
  std::vector<float> h_pm((size_t)T * HC), h_cm((size_t)T * HC * HC),
      h_fn((size_t)NOUT * HC * HID), h_hs(3, 1.0f), h_hb(NOUT, 0.0f);
  for (auto& v : h_x) v = __float2bfloat16(1.0f);
  for (auto& v : h_nw) v = __float2bfloat16(1.0f);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < HC; ++k)
      for (int h = 0; h < HID; ++h)
        h_res[((size_t)t * HC + k) * HID + h] = __float2bfloat16((float)(k + 1));
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < HC; ++j) h_pm[(size_t)t * HC + j] = 1.0f;
    for (int k = 0; k < HC; ++k)
      for (int j = 0; j < HC; ++j)
        h_cm[((size_t)t * HC + k) * HC + j] = (k == j) ? 1.0f : 0.0f;
  }
  // fn = 0: the projection contributes nothing, so p2 runs on hc_base alone
  // and stays deterministic without a host port of the sinkhorn.

  void *d_x, *d_res, *d_pm, *d_cm, *d_fn, *d_hs, *d_hb, *d_nw;
  void *d_ro, *d_pmo, *d_cmo, *d_li, *d_yp, *d_rp, *d_sq, *d_rsq, *d_px, *d_ol;
  void *d_bar;
  CK(dev(&d_x, h_x.size() * 2));       CK(dev(&d_res, h_res.size() * 2));
  CK(dev(&d_pm, h_pm.size() * 4));     CK(dev(&d_cm, h_cm.size() * 4));
  CK(dev(&d_fn, h_fn.size() * 4));     CK(dev(&d_hs, 3 * 4));
  CK(dev(&d_hb, NOUT * 4));            CK(dev(&d_nw, HID * 2));
  CK(dev(&d_ro, (size_t)T * HC * HID * 2));
  CK(dev(&d_pmo, (size_t)T * HC * 4)); CK(dev(&d_cmo, (size_t)T * HC * HC * 4));
  CK(dev(&d_li, (size_t)T * HID * 2));
  CK(dev(&d_yp, (size_t)NCH * MHC_MAX_TOK * NOUT * 4));
  CK(dev(&d_rp, (size_t)NCH * MHC_MAX_TOK * 4));
  CK(dev(&d_sq, MHC_MAX_TOK * 4));     CK(dev(&d_rsq, MHC_MAX_TOK * 4));
  CK(dev(&d_px, MHC_MAX_TOK * HC * 4));
  CK(dev(&d_ol, (size_t)MHC_MAX_TOK * HID * 2));
  CK(dev(&d_bar, 8 * 8));
  CK(cudaMemset(d_yp, 0, (size_t)NCH * MHC_MAX_TOK * NOUT * 4));
  CK(cudaMemset(d_rp, 0, (size_t)NCH * MHC_MAX_TOK * 4));
  CK(cudaMemset(d_bar, 0, 8 * 8));
  CK(cudaMemset(d_fn, 0, h_fn.size() * 4));
  CK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_res, h_res.data(), h_res.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pm, h_pm.data(), h_pm.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cm, h_cm.data(), h_cm.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_hs, h_hs.data(), 3 * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_hb, h_hb.data(), NOUT * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_nw, h_nw.data(), HID * 2, cudaMemcpyHostToDevice));

  a.x_in = (const __nv_bfloat16*)d_x; a.residual_in = (const __nv_bfloat16*)d_res;
  a.post_mix_in = (const float*)d_pm; a.comb_mix_in = (const float*)d_cm;
  a.fn = (const float*)d_fn; a.hc_scale = (const float*)d_hs;
  a.hc_base = (const float*)d_hb; a.norm_weight = (const __nv_bfloat16*)d_nw;
  a.residual_out = (__nv_bfloat16*)d_ro; a.post_mix_out = (float*)d_pmo;
  a.comb_mix_out = (float*)d_cmo; a.layer_input = (__nv_bfloat16*)d_li;
  a.yp = (float*)d_yp; a.rp = (float*)d_rp; a.sq = (float*)d_sq;
  a.rsq = (float*)d_rsq; a.pmix = (float*)d_px;
  a.ol_stash = (__nv_bfloat16*)d_ol;
  a.barrier_ctr = (unsigned long long*)d_bar;

  mk_mhc_kernel<HID><<<grid, MK_THREADS>>>(a);
  CK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> o_ro((size_t)T * HC * HID), o_li((size_t)T * HID);
  std::vector<float> o_rp((size_t)NCH * MHC_MAX_TOK);
  CK(cudaMemcpy(o_ro.data(), d_ro, o_ro.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o_li.data(), d_li, o_li.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o_rp.data(), d_rp, o_rp.size() * 4, cudaMemcpyDeviceToHost));

  // 1. every residual_out element is j + 2, exactly
  long bad = 0; int bt = -1, bj = -1, bh = -1; float bv = 0;
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < HC; ++j)
      for (int h = 0; h < HID; ++h) {
        float got = __bfloat162float(o_ro[((size_t)t * HC + j) * HID + h]);
        if (got != (float)(j + 2)) {
          if (!bad) { bt = t; bj = j; bh = h; bv = got; }
          ++bad;
        }
      }
  // 2. the RMS the tail takes: sum_c rp[c,t] / (HC * HID), equal at both HIDs
  double rms = 0.0;
  for (int c = 0; c < NCH; ++c) rms += o_rp[(size_t)c * MHC_MAX_TOK + 0];
  rms /= (double)(HC * HID);
  // 3. layer_input is uniform over h (vals does not depend on h here)
  float li0 = __bfloat162float(o_li[0]);
  long li_bad = 0;
  for (int h = 0; h < HID; ++h)
    if (__bfloat162float(o_li[h]) != li0) ++li_bad;

  printf("  HID %4d  T %3d | residual_out %s", HID, T,
         bad ? "MISMATCH" : "exact  ");
  if (bad) printf(" (%ld of %ld, first t%d j%d h%d = %g)", bad,
                  (long)T * HC * HID, bt, bj, bh, bv);
  printf(" | rms %.4f%s | layer_input %s (%.6g)\n", rms,
         (fabs(rms - 13.5) < 1e-3) ? "" : "  <-- expected 13.5",
         li_bad ? "VARIES" : "uniform", li0);
  cudaFree(d_x); cudaFree(d_res); cudaFree(d_pm); cudaFree(d_cm);
  cudaFree(d_fn); cudaFree(d_hs); cudaFree(d_hb); cudaFree(d_nw);
  cudaFree(d_ro); cudaFree(d_pmo); cudaFree(d_cmo); cudaFree(d_li);
  cudaFree(d_yp); cudaFree(d_rp); cudaFree(d_sq); cudaFree(d_rsq);
  cudaFree(d_px); cudaFree(d_ol); cudaFree(d_bar);
  return (bad || li_bad || fabs(rms - 13.5) > 1e-3) ? 1 : 0;
}

int main(int argc, char** argv) {
  int grid = argc > 1 ? atoi(argv[1]) : 16;
  int fail = 0;
  for (int i = 2; i < argc; ++i) {
    int T = atoi(argv[i]);
    if (T > MHC_MAX_TOK) { printf("  (T %d > MHC_MAX_TOK %d, skipped)\n",
                                  T, MHC_MAX_TOK); continue; }
    fail |= run_one<HIDDEN>(T, grid);
    fail |= run_one<HIDDEN_V41>(T, grid);
  }
  printf(fail ? "\nNUMERIC FAIL\n" : "\nNUMERIC PASS (both hidden sizes)\n");
  return fail;
}
"""

print("source_sha256   ", hashlib.sha256(SRC.read_bytes()).hexdigest(), flush=True)
print("extracted_sha256", hashlib.sha256(unit.encode()).hexdigest(), flush=True)

with tempfile.TemporaryDirectory(prefix="mk-mhc-numeric-") as tmp:
    cu = Path(tmp) / "num.cu"
    cu.write_text(unit)
    if args.keep:
        Path("/tmp/mk-mhc-numeric.cu").write_text(unit)
    binary = cu.with_suffix(".bin")
    build = [args.nvcc, "-O2", f"-arch={args.arch}", "-std=c++17",
             f"-DMK_MHC_GRID_DEF={args.grid}", str(cu), "-o", str(binary)]
    done = subprocess.run(build, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr[-4000:])
        raise SystemExit(f"nvcc failed ({done.returncode})")
    run = subprocess.run([str(binary), str(args.grid)] + args.tokens.split(","),
                         capture_output=True, text=True, timeout=600)
    sys.stdout.write(run.stdout)
    sys.stderr.write(run.stderr)
    raise SystemExit(run.returncode)
