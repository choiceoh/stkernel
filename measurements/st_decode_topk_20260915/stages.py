"""Stage breakdown of st_dsa_select: where the time goes after the DRAM sweep."""
import os
import re
import sys

import torch
from torch.cuda import _compile_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import plan  # noqa: E402
from bench_graph import graph_time  # noqa: E402

REPO = r"C:\Users\user\Downloads\github\stkernel\.claude\worktrees\heuristic-pare-a5e502"
K, THREADS, SMEM_LIMIT = 512, 1024, 96 * 1024
BODY = open(os.path.join(REPO, "engine", "kernels", "decode_topk.cu"),
            encoding="utf-8").read().split("#ifndef ST_DECODE_TOPK_NVRTC\nvoid run(")[0]

# each cut leaves a use of the stage's result so nothing upstream is dead-code eliminated
CUTS = {
    "sweep": ("  __syncthreads();\n  if (tx < ST_RADIX) {\n    int s = 0;",
              "  __syncthreads();\n  if (tx == 0) out[(long long)row * ST_K] = hist_rep[0][0];\n"
              "  if (length) return;\n  if (tx < ST_RADIX) {\n    int s = 0;"),
    "pick": ("  const int coarse_pop = hist[coarse] - hist[coarse + 1];",
             "  const int coarse_pop = hist[coarse] - hist[coarse + 1];\n"
             "  if (tx == 0) out[(long long)row * ST_K] = coarse + coarse_pop;\n"
             "  if (length) return;"),
    "sift": ("#pragma unroll 8\n  for (int round = 0; round < 8; ++round) {\n"
             "    const int ring = round & 1;",
             "  if (tx == 0) out[(long long)row * ST_K] = counter + num_input[0];\n"
             "  if (length) return;\n#pragma unroll 8\n  for (int round = 0; round < 8; ++round) {\n"
             "    const int ring = round & 1;"),
}


def build(stage=None, defines=()):
    body = BODY
    src = "#define ST_DECODE_TOPK_NVRTC 1\n"
    for name, value in defines:
        body = re.sub(rf"^#define {name} .*$", "", body, flags=re.M)
        src += f"#define {name} {value}\n"
    if stage:
        old, new = CUTS[stage]
        assert old in body, stage
        body = body.replace(old, new, 1)
    reps = dict(defines).get("ST_REPLICAS", 4)
    static = (257 + reps * 256 + 512) * 4 + 256
    k = _compile_kernel(src + body, "st_dsa_select", compute_capability="120")
    k.set_shared_memory_config(SMEM_LIMIT - static)
    return k, static


def run(kern, lg, ke, out, bin_bytes, stash):
    kern((lg.shape[0], 1, 1), (THREADS, 1, 1),
         (lg, lg.stride(0), lg.shape[1], ke, out, stash, bin_bytes),
         shared_mem=bin_bytes + 4 * stash * 4)


def main():
    dev = "cuda"
    gen = torch.Generator(device=dev)
    shapes = [(16, 4096), (16, 8192), (16, 32768), (16, 50688)]
    data = {}
    for rows, n in shapes:
        gen.manual_seed(1)
        v = torch.randn(rows, n, device=dev, generator=gen)
        lg = torch.where(v > 0.3, v * 3.0, torch.zeros_like(v)).contiguous()
        data[(rows, n)] = (lg, torch.full((rows,), n, device=dev, dtype=torch.int32),
                           torch.empty(rows, K, device=dev, dtype=torch.int32))

    stages = ["sweep", "pick", "sift", None]
    built = [build(s) for s in stages]
    print("== cumulative us by stage ==")
    print(f"{'shape':>13} {'sweep':>7} {'pick':>7} {'sift':>7} {'refine+pub':>11}   "
          f"{'| sweep':>8} {'pick':>6} {'sift':>6} {'refine':>7}")
    for rows, n in shapes:
        lg, ke, out = data[(rows, n)]
        bb, st = plan(n, SMEM_LIMIT - built[0][1])
        t = [graph_time(lambda k=k: run(k, lg, ke, out, bb, st)) for k, _ in built]
        d = [t[0], t[1] - t[0], t[2] - t[1], t[3] - t[2]]
        print(f"{rows:>4}x{n:<8} {t[0]:7.1f} {t[1]:7.1f} {t[2]:7.1f} {t[3]:11.1f}   "
              f"| {d[0]:6.1f} {d[1]:6.1f} {d[2]:6.1f} {d[3]:7.1f}")

    print("\n== replicas (full kernel, us) ==")
    reps = (1, 2, 4, 8)
    kr = {r: build(None, (("ST_REPLICAS", r),)) for r in reps}
    print(f"{'shape':>13} " + " ".join(f"{'R=' + str(r):>7}" for r in reps))
    for rows, n in shapes:
        lg, ke, out = data[(rows, n)]
        row = []
        for r in reps:
            k, static = kr[r]
            bb, st = plan(n, SMEM_LIMIT - static)
            row.append(graph_time(lambda k=k, bb=bb, st=st: run(k, lg, ke, out, bb, st)))
        print(f"{rows:>4}x{n:<8} " + " ".join(f"{v:7.1f}" for v in row))


if __name__ == "__main__":
    main()
