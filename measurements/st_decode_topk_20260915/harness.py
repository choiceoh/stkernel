"""Local NVRTC harness for engine/kernels/decode_topk.cu: set equality vs torch.topk.

No nvcc/MSVC on this box, so the kernel is compiled with the nvrtc torch ships.
sm_120 here vs sm_121a on GB10 -- same ISA family, different SM count and memory.
"""
import os
import sys

import torch
from torch.cuda import _compile_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import plan, block_threads  # noqa: E402

REPO = r"C:\Users\user\Downloads\github\stkernel\.claude\worktrees\heuristic-pare-a5e502"
K, THREADS = 512, 1024
SMEM_LIMIT = 96 * 1024
STATIC = 8192


def build():
    src = open(os.path.join(REPO, "engine", "kernels", "decode_topk.cu"), encoding="utf-8").read()
    src = "#define ST_DECODE_TOPK_NVRTC 1\n" + src.split("#ifndef ST_DECODE_TOPK_NVRTC\nvoid run(")[0]
    kern = _compile_kernel(src, "st_dsa_select", compute_capability="120")
    kern.set_shared_memory_config(SMEM_LIMIT - STATIC)
    return kern


def launch(kern, logits, ke, out, force_no_cache=False, threads=None):
    rows, n = logits.shape
    bin_bytes, stash = plan(1 << 30 if force_no_cache else n, SMEM_LIMIT - STATIC)
    kern((rows, 1, 1), (threads or block_threads(n), 1, 1),
         (logits, logits.stride(0), n, ke, out, stash, bin_bytes),
         shared_mem=bin_bytes + 4 * stash * 4)


def reference(logits, ke, k=K):
    """What the served path produces today: mask the horizon, torch.topk, as a SET per row."""
    rows, n = logits.shape
    cols = torch.arange(n, device=logits.device)
    masked = logits.masked_fill(cols[None, :] >= ke[:, None], float("-inf"))
    idx = torch.topk(masked, k, dim=-1, sorted=False).indices
    return [set(i for i in idx[r].tolist() if i < min(int(ke[r]), n)) for r in range(rows)]


def make(kind, rows, n, dev, gen):
    if kind == "random":
        return torch.randn(rows, n, device=dev, generator=gen)
    if kind == "relu":  # the indexer's own shape: sum_h w_h relu(q.k), many exact zeros
        v = torch.randn(rows, n, device=dev, generator=gen)
        return torch.where(v > 0.3, v * 3.0, torch.zeros_like(v))
    if kind == "plateau":  # a forced tie straddling the k-th boundary
        v = torch.randn(rows, n, device=dev, generator=gen)
        for r in range(rows):
            hi = v[r].topk(max(1, K - 200)).values[-1]
            v[r, torch.randperm(n, device=dev, generator=gen)[:400]] = hi
        return v
    if kind == "allzero":  # pathological: nearly the whole row is one exact value
        v = torch.zeros(rows, n, device=dev)
        v[:, ::64] = torch.rand(rows, (n + 63) // 64, device=dev, generator=gen)
        return v
    if kind == "onevalue":  # every visible element identical: ties resolve purely by index
        return torch.full((rows, n), 1.25, device=dev)
    if kind == "negative":  # mostly-negative gates
        return -torch.randn(rows, n, device=dev, generator=gen).abs()
    if kind == "huge":  # past fp16 range, where the coarse bin saturates
        return torch.randn(rows, n, device=dev, generator=gen) * 1e30
    if kind == "tiny":  # subnormal in fp16, where the coarse bin underflows
        return torch.randn(rows, n, device=dev, generator=gen) * 1e-8
    raise ValueError(kind)


def main():
    dev = "cuda"
    gen = torch.Generator(device=dev)
    kern = build()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} SMs={props.multi_processor_count} "
          f"smem/block optin={getattr(props, 'shared_memory_per_block_optin', '?')}")
    print("\n== set equality vs masked torch.topk ==")
    bad = cases = 0
    for kind in ("random", "relu", "plateau", "allzero", "onevalue", "negative", "huge", "tiny"):
        for rows in (1, 8, 16, 20):
            for n in (600, 1024, 4096, 8192, 32768, 50688):
                gen.manual_seed(abs(hash((kind, rows, n))) & 0xFFFF)
                lg = make(kind, rows, n, dev, gen).contiguous()
                for horizon in ("full", "short", "mixed"):
                    if horizon == "full":
                        ke = torch.full((rows,), n, device=dev, dtype=torch.int32)
                    elif horizon == "short":
                        ke = torch.full((rows,), min(n, 300), device=dev, dtype=torch.int32)
                    else:
                        ke = torch.randint(0, n + 1, (rows,), device=dev,
                                           generator=gen).to(torch.int32)
                    want = reference(lg, ke)
                    for no_cache, width in ((False, None), (True, None), (False, 128), (False, 1024)):
                        out = torch.full((rows, K), -7, device=dev, dtype=torch.int32)
                        launch(kern, lg, ke, out, force_no_cache=no_cache, threads=width)
                        torch.cuda.synchronize()
                        for r in range(rows):
                            cases += 1
                            length = min(int(ke[r]), n)
                            exp = want[r] if length > K else set(range(length))
                            have = set(i for i in out[r].tolist() if 0 <= i < length)
                            if have != exp:
                                bad += 1
                                if bad <= 6:
                                    print(f"  MISMATCH {kind} rows={rows} n={n} {horizon} "
                                          f"cache={not no_cache} T={width} row={r} len={length} "
                                          f"got={len(have)} want={len(exp)} "
                                          f"missing={sorted(exp - have)[:4]} "
                                          f"extra={sorted(have - exp)[:4]}")
    print(f"  mismatched rows: {bad} of {cases}")


if __name__ == "__main__":
    main()
