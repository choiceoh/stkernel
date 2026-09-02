#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MK_SEG_MLA vs FlashInfer's SM90 wrapper: numerics, GPU time, host plan cost.

Run in a fresh container with the repo mounted at /repo and the fleet down
(the cache alone is ~150 MB and the comparison needs an idle GPU):

    docker run --rm --gpus all --entrypoint python3 \
      --mount type=bind,src=$REPO,dst=/repo,readonly glm53:v13-b12x \
      /repo/probes/mk_mla_bench.py

VLLM_GLM53_MK_MLA_PROBE=1 streams only, =2 adds the dot: the roofline that
showed the score phase's cross-lane reduction is the wall."""
import importlib.util, os, sys, time
os.environ.setdefault("VLLM_GLM53_MEGAKERNEL", "1"); os.environ.setdefault("VLLM_GLM53_MK_MLA", "1")
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import torch
spec = importlib.util.spec_from_file_location(
    "mk", "/repo/overlay/modules/glm53_megakernel/glm53_megakernel.py")
mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk); mk._build()
print("selftest:", "ARM" if mk._selftest_mla() else "DISARM", flush=True)

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm90 as fi
dev = torch.device("cuda")
D, H, W = mk.MLA_D, mk.MLA_H, 2048
NUM_SLOTS = 300000          # ~150 MB of latent, well past L2
torch.manual_seed(0)
cache8 = (torch.randn(NUM_SLOTS, D, device=dev) * 0.4).to(torch.float8_e4m3fn)
sm = D ** -0.5

def graph_us(fn, n=20, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        with torch.cuda.graph(g, stream=st):
            for _ in range(n): fn()
    torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (reps * n) * 1e3

state = None
for T in (8, 16, 32):
    q = (torch.randn(T, H, D, device=dev) * 0.3).to(torch.bfloat16)
    slots = torch.randint(0, NUM_SLOTS, (T, W), dtype=torch.int32, device=dev)
    lens = torch.full((T,), W, dtype=torch.int32, device=dev)
    ours = lambda: mk.mla_decode(q, cache8.view(torch.uint8), slots, lens, sm, 1.0)
    t_ours = graph_us(ours)
    if state is None:
        state = fi._SM90State(device=dev, num_heads=H, kv_dtype=torch.float8_e4m3fn,
                              max_tokens=32, topk_width=W, kv_lora_rank=D,
                              qk_rope_head_dim=0, sm_scale=sm)
    state.kv_indices[: T * W].copy_(slots.reshape(-1))
    lens_cpu = lens.cpu()
    state.plan(T, lens_cpu)
    flat = cache8.view(-1, 1, D)
    ckv = flat[..., :D]; kpe = flat[..., D:]
    qpe = torch.empty(T, H, 0, dtype=torch.bfloat16, device=dev)
    run = lambda: state.wrapper.run(q, qpe, ckv, kpe, ckv_scale=1.0, kpe_scale=1.0)
    t_fi = graph_us(run)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30): state.plan(T, lens_cpu)
    torch.cuda.synchronize()
    t_plan = (time.perf_counter() - t0) / 30 * 1e6
    o1 = mk.mla_decode(q, cache8.view(torch.uint8), slots, lens, sm, 1.0)
    o2 = run(); torch.cuda.synchronize()
    rel = mk._rel_err(o1.float(), o2.float())
    bw = T * W * D / (t_ours * 1e-6) / 1e9
    print(f"T={T:2d}: MK {t_ours:7.1f} us ({bw:5.0f} GB/s) | FI run {t_fi:7.1f} us + host plan {t_plan:7.1f} us"
          f" | 층당 절감 {(t_fi + t_plan) - t_ours:7.1f} us | rel(MK vs FI) {rel:.2e}", flush=True)
