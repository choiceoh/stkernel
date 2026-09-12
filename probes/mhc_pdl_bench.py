"""Does the prefill lane's PDL overlap pay? The half probes/mhc_pdl_cost.py could not answer.

The cost side is empty (same registers, spills, smem, ld/st either way -- ledger 45차 §64). What is left is
whether letting each mHC kernel start on the SMs its predecessor frees is worth anything at prefill shapes. That
needs a device. It does NOT need the fleet: this times the lane's own entry points on one GPU, beside whatever
else is running, with well under a GiB.

    docker run --rm --gpus all -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/mhc_pdl_bench.py
"""
import pathlib
import shutil
import statistics
import sys

SRC, DST = pathlib.Path("/w/engine"), pathlib.Path("/tmp/e/engine")
if not DST.exists():
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
sys.path.insert(0, "/tmp/e")
K = DST / "kernels" / "mhc" / "tilelang_kernels.py"

import torch

HIDDEN, HC = 4096, 4
TOKENS = [512, 2304, 6912]            # the prefill chunk and two steps below it (facts.CHUNK_ALIGN, its 3x law)
WARMUP, ITERS = 20, 200


def load(enable):
    t = K.read_text()
    K.write_text(t.replace("ENABLE_PDL = False", f"ENABLE_PDL = {enable}").replace("ENABLE_PDL = True", f"ENABLE_PDL = {enable}"))
    for n in [n for n in sys.modules if n.startswith("engine.")]:
        del sys.modules[n]
    from engine.kernels.mhc import tilelang_kernels as tk
    assert tk.ENABLE_PDL is enable
    from engine.kernels.mhc import mhc_post_tilelang, mhc_pre_tilelang
    return mhc_pre_tilelang, mhc_post_tilelang


def inputs(n, dev):
    g = torch.Generator(device=dev).manual_seed(7)
    res = torch.randn(n, HC, HIDDEN, generator=g, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(HC * (2 + HC), HC * HIDDEN, generator=g, device=dev, dtype=torch.float32).contiguous()
    scale = torch.ones(3, device=dev, dtype=torch.float32)
    base = torch.zeros(HC * (2 + HC), device=dev, dtype=torch.float32)
    norm = torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
    return res, fn, scale, base, norm


def time_lane(pre, post, n, dev):
    res, fn, scale, base, norm = inputs(n, dev)
    def once():
        p, c, x = pre(res, fn, scale, base, 1e-5, 1e-5, 1e-5, 1.0, 0, 1, norm, 1e-5)
        return post(x, res, p, c)
    for _ in range(WARMUP):
        once()
    torch.cuda.synchronize()
    out = []
    for _ in range(ITERS):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); once(); b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000.0)      # us
    return statistics.median(out), min(out)


def main():
    """One flag state per PROCESS. Flipping it inside one process lets tilelang's kernel cache hand the second
    state the first state's compiled kernel, which would make the whole comparison a measurement of nothing."""
    enable = sys.argv[1] == "on"
    dev = torch.device("cuda")
    pre, post = load(enable)
    print(f"# ENABLE_PDL={enable} on {torch.cuda.get_device_name(0)}, free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")
    for n in TOKENS:
        median, fastest = time_lane(pre, post, n, dev)
        print(f"RESULT {enable} {n} {median:.2f} {fastest:.2f}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
