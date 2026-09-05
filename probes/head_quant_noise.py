#!/usr/bin/env python3
"""Quantization noise of the vocab head, on the real checkpoint, CPU only.

The served head is fp8 (deepgemm per-128x128-block e4m3, ue8m0 scale); the
megakernel's head lane (EXP-22) serves a W4 pack (e2m1 x per-16-group e4m3
scale, the packer's 2-octave x 8-mantissa search). Both are rebuilt here in
numpy on a strided sample of the rank-0 shard (16 consecutive 128-row blocks,
so the fp8 block scales are the real ones) and compared in weight space
(relative rms error, per row) and, for an isotropic activation, as logit
noise over logit spread. No GPU, no torch, ~5 s on one core: run it niced.

  nice -n 19 taskset -c 19 python3 probes/head_quant_noise.py [MODEL_DIR]

09-06 (glm53-redhat-nvfp4, ledger 30차 §13): fp8 2.66 pct, W4 8.37 pct,
ratio 3.15 -- the W4 head triples the logit noise the fp8 head already adds.
"""
import json, os, struct, sys, time
import numpy as np
mp = sys.argv[1] if len(sys.argv) > 1 else "/home/choiceoh/models/glm53-redhat-nvfp4"
idx = json.load(open(os.path.join(mp, "model.safetensors.index.json")))
shard = idx["weight_map"]["lm_head.weight"]
path = os.path.join(mp, shard)
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
meta = hdr["lm_head.weight"]
print("lm_head:", meta["dtype"], meta["shape"], "shard", shard)
assert meta["dtype"] == "BF16"
V, K = meta["shape"]
off0, off1 = meta["data_offsets"]
mm = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + off0, shape=(V, K))
# rank-0 shard = rows [0, V/4); 16 consecutive 128-row blocks at strided starts
Vr = V // 4
starts = [int(j * (Vr // 128) / 16) * 128 for j in range(16)]
rows = np.concatenate([np.arange(s, s + 128) for s in starts])
w16 = np.asarray(mm[rows])                      # [2048, K] uint16
w = (w16.astype(np.uint32) << 16).view(np.float32)   # bf16 -> f32
R = w.shape[0]
print(f"sample rows {R} (16 blocks of 128 from the rank-0 shard of {Vr}), K={K}, |w| rms {np.sqrt((w**2).mean()):.4e}, amax {np.abs(w).max():.3e}")

def e4m3(x):
    """round-to-nearest-even onto the e4m3fn grid (|x| <= 448 assumed)."""
    a = np.abs(x).astype(np.float32)
    e = np.floor(np.log2(np.maximum(a, 1e-30)))
    e = np.maximum(e, -6.0)                     # subnormal step 2^-9 = 2^(-6-3)
    step = np.exp2(e - 3.0)
    q = np.rint(a / step) * step                # rint = ties to even
    q = np.minimum(q, 448.0)
    return np.sign(x) * q

# --- fp8 head (deepgemm per-block 128x128, ue8m0 scale = 2^ceil(log2(amax/448)))
t = time.time()
wb = w.reshape(16, 128, K // 128, 128)          # [block, 128 rows, kblock, 128]
amax = np.abs(wb).max(axis=(1, 3), keepdims=True)
sc8 = np.exp2(np.ceil(np.log2(np.maximum(amax, 1e-30) / 448.0)))
w8 = e4m3(wb / sc8) * sc8
d8 = (w8 - wb).reshape(R, K)
print(f"fp8 W8 block-quant done {time.time()-t:.1f}s")

# --- W4 pack (e2m1 x e4m3 scale per 16-group, the packer's 2-octave x 8-mantissa search)
t = time.time()
grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
mids = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
g = w.reshape(R, K // 16, 16)
ga = np.abs(g); sign = np.sign(g)
amax4 = ga.max(-1, keepdims=True)
e0 = np.ceil(np.log2(np.maximum(amax4, 1e-30) / 6.0))
best_err = None; best_deq = None
for j in (0.0, 1.0):
    e = e0 - j
    for kk in range(8):
        sc = (1.0 + kk / 8.0) * np.exp2(e)
        code = np.searchsorted(mids, ga / sc)    # nearest grid index
        deq = e4m3(grid[code] * sc) * sign
        err = ((deq - g) ** 2).sum(-1, keepdims=True)
        if best_err is None:
            best_err, best_deq = err, deq
        else:
            take = err < best_err
            best_err = np.where(take, err, best_err)
            best_deq = np.where(take, deq, best_deq)
d4 = (best_deq - g).reshape(R, K)
print(f"W4 group-quant done {time.time()-t:.1f}s")

def stats(d, name):
    rr = np.sqrt((d ** 2).mean(1)) / np.sqrt((w ** 2).mean(1))   # per-row rel rms
    tot = np.sqrt((d ** 2).sum()) / np.sqrt((w ** 2).sum())
    # logit-noise / logit-spread for isotropic x: ||dW_v|| / rms_v ||W_v||
    print(f"{name}: rel rms error overall {tot:.4f}; per-row median {np.median(rr):.4f}, p90 {np.percentile(rr, 90):.4f}, max {rr.max():.4f}")
    return tot
r8 = stats(d8, "fp8 (served head)")
r4 = stats(d4, "W4 pack (RTN, e2m1 x e4m3 scale)")
print(f"noise ratio W4 / fp8 = {r4 / r8:.2f}")
# The synthetic-x argmax flip rate is NOT reported: a Gaussian x has no real
# top-1 margins (every logit is near-equal), so its flips only restate the
# noise ratio above. The real flip / acceptance cost needs real hidden
# states: the draft-head bracket (acceptance-normalised step/s) and, for the
# target head, a greedy-divergence count against the fp8 boot.
