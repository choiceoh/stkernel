"""Is the sift's cost the DRAM gather of the candidates, or the scan itself?

Arms, all on the same source:
  base     the shipped kernel
  nogather the stash seeds its histogram from the index instead of re-reading in[idx]
           (WRONG results, timing only: it removes exactly the gather)
Also prints the threshold bin's population, which is how many gathers there are.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import plan  # noqa: E402
from bench_graph import graph_time  # noqa: E402
from stages import build, run, SMEM_LIMIT, K  # noqa: E402
from torch.cuda import _compile_kernel  # noqa: E402

REPO = r"C:\Users\user\Downloads\github\stkernel\.claude\worktrees\heuristic-pare-a5e502"
BODY = open(os.path.join(REPO, "engine", "kernels", "decode_topk.cu"),
            encoding="utf-8").read().split("#ifndef ST_DECODE_TOPK_NVRTC\nvoid run(")[0]


def build_variant(sub):
    body = BODY
    for old, new in sub:
        assert old in body, old[:60]
        body = body.replace(old, new)
    k = _compile_kernel("#define ST_DECODE_TOPK_NVRTC 1\n" + body, "st_dsa_select",
                        compute_capability="120")
    k.set_shared_memory_config(SMEM_LIMIT - 8192)
    return k


GATHER = "st_hist_add(hist, (int)((st_ordered_key(in[_idx]) >> 24) & 0xffu));"
NOGATHER = "st_hist_add(hist, _idx & 0xff);"


def main():
    dev = "cuda"
    gen = torch.Generator(device=dev)
    base = build_variant([])
    nog = build_variant([(GATHER, NOGATHER)])
    pick, _ = build("pick")

    print(f"{'shape':>13} {'coarse_pop':>11} {'base us':>8} {'no-gather us':>13} {'gather us':>10}")
    for rows, n in [(16, 4096), (16, 8192), (16, 32768), (16, 50688), (8, 50688)]:
        gen.manual_seed(1)
        v = torch.randn(rows, n, device=dev, generator=gen)
        lg = torch.where(v > 0.3, v * 3.0, torch.zeros_like(v)).contiguous()
        ke = torch.full((rows,), n, device=dev, dtype=torch.int32)
        out = torch.empty(rows, K, device=dev, dtype=torch.int32)
        bb, st = plan(n, SMEM_LIMIT - 8192)
        run(pick, lg, ke, out, bb, st)
        torch.cuda.synchronize()
        pop = int(out[0, 0]) - int(out[0, 0]) % 1  # coarse + coarse_pop, coarse < 256
        pop = int(out[0, 0])
        tb = graph_time(lambda: run(base, lg, ke, out, bb, st))
        tn = graph_time(lambda: run(nog, lg, ke, out, bb, st))
        print(f"{rows:>4}x{n:<8} {pop:11d} {tb:8.1f} {tn:13.1f} {tb - tn:10.1f}")


if __name__ == "__main__":
    main()
