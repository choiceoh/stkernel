"""What does nvfp4 cost on the DENSE projections, against the fp8 we serve now?

nvfp4 runs them 2.3x faster (236 vs 104 TFLOP/s) and halves the pack. The
checkpoint's own recipe put fp4 on the EXPERTS only and left these alone, so
the question is whether that was necessary or just conservative. Measure on
the real weights, both operands quantized, against the bf16 reference."""
import glob, json, statistics as st, struct, sys, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from flashinfer import nvfp4_quantize
dev = "cuda"; torch.manual_seed(0)
F8, E4M3_MAX = torch.float8_e4m3fn, 448.0

def load(limit=6):
    want = ("self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj", "fused_qkv_a_proj")
    out = []
    for f in sorted(glob.glob("/models/glm/*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n)); base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__" or not k.endswith(".weight") or v.get("dtype") != "BF16":
                continue
            if not any(w in k for w in want): continue
            sh = v["shape"]
            if len(sh) != 2 or min(sh) < 512 or sh[0]*sh[1] > 4e7: continue
            s, e = v["data_offsets"]
            with open(f, "rb") as fh:
                fh.seek(base+s); raw = fh.read(e-s)
            out.append((k.split("model.language_model.")[-1],
                        torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*sh).to(dev)))
            if len(out) >= limit: return out
    return out

def qd_block(t, br, bc):
    n, k = t.shape
    br = n if br < 0 else br; bc = k if bc < 0 else bc
    pn, pk = -(-n//br)*br, -(-k//bc)*bc
    p = torch.zeros(pn, pk, device=t.device, dtype=torch.float32); p[:n,:k] = t.float()
    v = p.view(pn//br, br, pk//bc, bc)
    s = v.abs().amax(dim=(1,3), keepdim=True).clamp_min(1e-6)/E4M3_MAX
    return ((v/s).to(F8).float()*s).view(pn,pk)[:n,:k]

E2M1 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.], device=dev)
def qd_nvfp4(t, bs=16):
    """nvfp4 as the checkpoint uses it: per-16 block e4m3 scale + a global
    fp32 scale, values on the 8-point e2m1 grid."""
    n, k = t.shape
    x = t.float().view(n, k//bs, bs)
    gs = (x.abs().amax()/(6.0*E4M3_MAX)).clamp_min(1e-12)
    bsc = (x.abs().amax(dim=2, keepdim=True)/6.0/gs).to(F8).float()*gs
    bsc = bsc.clamp_min(1e-12)
    q = x/bsc
    idx = torch.bucketize(q.abs().contiguous(), (E2M1[1:]+E2M1[:-1])/2)
    return (torch.sign(q)*E2M1[idx]*bsc).view(n, k)

def acts(m, k, outliers):
    x = torch.randn(m, k, device=dev, dtype=torch.float32)
    if outliers:
        ch = torch.randperm(k, device=dev)[:max(1, k//256)]
        x[:, ch] *= 40.0
    return x

print(f"{'tensor':38s} {'outl':>5s} {'fp8 blk128':>11s} {'nvfp4':>9s} {'ratio':>6s}")
agg = {}
for name, w in load():
    for outl in (False, True):
        x = acts(1024, w.shape[1], outl)
        ref = x @ w.float().T; rn = ref.norm().item()
        e8 = ((qd_block(x,1,128) @ qd_block(w,128,128).T) - ref).norm().item()/rn
        e4 = ((qd_nvfp4(x) @ qd_nvfp4(w).T) - ref).norm().item()/rn
        agg.setdefault(outl, []).append((e8, e4))
        print(f"{name[:38]:38s} {str(outl):>5s} {e8:11.2e} {e4:9.2e} {e4/e8:5.1f}x")
for outl in (False, True):
    a=[x for x,_ in agg[outl]]; b=[x for _,x in agg[outl]]
    print(f"MEAN outliers={outl!s:5s} fp8 {st.mean(a):.2e}  nvfp4 {st.mean(b):.2e}"
          f"  -> nvfp4 is {st.mean(b)/st.mean(a):.1f}x the error")
