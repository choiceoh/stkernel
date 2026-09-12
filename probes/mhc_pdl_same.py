"""PDL must not change a single number. Each flag state runs in its own process (kernel cache) and writes
summaries; a third pass compares."""
import pathlib, shutil, sys
SRC, DST = pathlib.Path("/w/engine"), pathlib.Path("/tmp/e/engine")
if not DST.exists():
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
sys.path.insert(0, "/tmp/e")
K = DST / "kernels" / "mhc" / "tilelang_kernels.py"
import torch

HIDDEN, HC = 4096, 4
enable = sys.argv[1] == "on"
t = K.read_text()
K.write_text(t.replace("ENABLE_PDL = False", f"ENABLE_PDL = {enable}").replace("ENABLE_PDL = True", f"ENABLE_PDL = {enable}"))
from engine.kernels.mhc import mhc_post_tilelang as post, mhc_pre_tilelang as pre
from engine.kernels.mhc import tilelang_kernels as tk
assert tk.ENABLE_PDL is enable

def sig(x):
    f = x.float().flatten()
    w = torch.arange(f.numel(), device=f.device, dtype=torch.float32) % 9973.0
    return [float(f.sum()), float(f.abs().sum()), float(f.min()), float(f.max()), float((f * w).sum())]

dev = torch.device("cuda")
out = {}
for n in (65, 512, 2304, 6912):
    g = torch.Generator(device=dev).manual_seed(11)
    res = torch.randn(n, HC, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(HC * (2 + HC), HC * HIDDEN, generator=g, device=dev, dtype=torch.float32).contiguous()
    scale = torch.ones(3, device=dev, dtype=torch.float32)
    base = torch.zeros(HC * (2 + HC), device=dev, dtype=torch.float32)
    norm = torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
    p, c, x = pre(res, fn, scale, base, 1e-5, 1e-5, 1e-5, 1.0, 0, 1, norm, 1e-5)
    r = post(x, res, p, c)
    out[n] = {k: sig(v) for k, v in (("post_mix", p), ("comb", c), ("x", x), ("res", r))}
    del res, fn, p, c, x, r
    torch.cuda.empty_cache()
torch.save(out, f"/s_out/lane_{sys.argv[1]}.pt")
print(f"wrote {sys.argv[1]}")
