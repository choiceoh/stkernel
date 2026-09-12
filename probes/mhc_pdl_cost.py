"""What ENABLE_PDL costs, in registers and instructions, without a GPU.

Turning it on drops `__restrict__` from every kernel's pointer parameters (probes/mhc_pdl_lowering.py). That is a
no-alias guarantee the compiler loses, and whether it matters is not an argument -- it is ptxas output. This
generates the CUDA both ways, compiles each for sm_121a, and reports registers, spills, shared memory and SASS
instruction count. The GAIN (launch overlap) still needs a device; this is the other half.

    docker run --rm -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/mhc_pdl_cost.py

Answered 2026-09-12 (ledger 45차 §64): nothing. All six kernels that lower come out with identical register
counts, zero spills either way, identical shared memory, and the same number of ld.global and st.global. The
no-alias guarantee was not buying anything here -- tilelang's generated code already addresses through explicit
TMA descriptors and per-thread indices the compiler can disambiguate without being told.

Caveat worth keeping: registers, spills and smem come from ptxas, so allocation really is identical; the load and
store counts are PTX, so a scheduling difference below ptxas would not show here. With equal registers and no
spills that is a narrow channel, but it is not nothing, and the image has no nvdisasm to close it.
"""
import inspect
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

SRC, DST = pathlib.Path("/w/engine"), pathlib.Path("/tmp/e/engine")
if not DST.exists():
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
sys.path.insert(0, "/tmp/e")
K = DST / "kernels" / "mhc" / "tilelang_kernels.py"
import tilelang

ROOT_TL = pathlib.Path(tilelang.__file__).parent
TEMPLATES = ROOT_TL / "src"
CUTLASS = ROOT_TL / "3rdparty" / "cutlass" / "include"   # tl_templates/cuda/common.h includes cute/
TARGET = {"kind": "cuda", "arch": "sm_121a"}
GLM = {"hidden": 4096, "hidden_size": 4096, "hc": 4, "hc_mult": 4, "n_out": 24,
       "rms_eps": 1e-5, "hc_eps": 1e-5, "hc_pre_eps": 1e-5, "hc_sinkhorn_eps": 1e-5,
       "hc_post_mult_value": 1.0, "sinkhorn_repeat": 0, "norm_eps": 1e-5,
       "post_mult": 1.0, "sinkhorn": 0}
REG = re.compile(r"Used (\d+) registers")
SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
SMEM = re.compile(r"(\d+) bytes smem")


def load(enable):
    t = K.read_text()
    K.write_text(t.replace("ENABLE_PDL = False", f"ENABLE_PDL = {enable}").replace("ENABLE_PDL = True", f"ENABLE_PDL = {enable}"))
    for n in [n for n in sys.modules if "tilelang_kernels" in n or n.startswith("engine.")]:
        del sys.modules[n]
    from engine.kernels.mhc import tilelang_kernels as tk
    assert tk.ENABLE_PDL is enable
    return tk


def measure(source: str):
    """(registers, spill stores, spill loads, smem, SASS instructions) from ptxas, or an error string."""
    with tempfile.TemporaryDirectory() as tmp:
        cu = pathlib.Path(tmp) / "k.cu"
        cu.write_text(source)
        cubin = pathlib.Path(tmp) / "k.cubin"
        p = subprocess.run(["nvcc", "-arch=sm_121a", "-cubin", "-o", str(cubin), str(cu),
                            f"-I{TEMPLATES}", f"-I{CUTLASS}", "-std=c++17", "-DENABLE_BF16",
                            "--expt-relaxed-constexpr", "-Xptxas", "-v", "-O3"],
                           capture_output=True, text=True)
        if p.returncode:
            return " || ".join(p.stderr.strip().splitlines()[:3])[:200]
        log = p.stderr
        reg = int(REG.search(log).group(1)) if REG.search(log) else -1
        st, ld = (int(a) for a in SPILL.search(log).groups()) if SPILL.search(log) else (0, 0)
        smem = int(SMEM.search(log).group(1)) if SMEM.search(log) else 0
        # PTX, not SASS: the image has cuobjdump but no nvdisasm. Global loads and stores are what a lost
        # no-alias guarantee shows up as -- the compiler stops keeping a value in a register across a store
        # it can no longer prove is to a different buffer -- so they are the number to watch.
        ptx = pathlib.Path(tmp) / "k.ptx"
        subprocess.run(["nvcc", "-arch=sm_121a", "-ptx", "-o", str(ptx), str(cu), f"-I{TEMPLATES}",
                        f"-I{CUTLASS}", "-std=c++17", "-DENABLE_BF16", "--expt-relaxed-constexpr", "-O3"],
                       capture_output=True, text=True)
        text = ptx.read_text() if ptx.exists() else ""
        loads = sum(1 for line in text.splitlines() if "ld.global" in line)
        stores = sum(1 for line in text.splitlines() if "st.global" in line)
        return reg, st, ld, smem, (loads, stores)


def main():
    names = [n for n in dir(load(False)) if n.endswith("_tilelang")]
    got = {}
    for enable in (False, True):
        tk = load(enable)
        for name in names:
            fn = getattr(tk, name)
            if not hasattr(fn, "get_tir"):
                continue
            params = inspect.signature(fn.func if hasattr(fn, "func") else fn).parameters
            kw = {k: GLM[k] for k, v in params.items() if k in GLM and v.default is inspect.Parameter.empty}
            try:
                source = tilelang.compile(fn.get_tir(**kw), target=TARGET).get_kernel_source()
            except Exception:
                continue
            got.setdefault(name, {})[enable] = measure(source)

    print(f"  {'kernel':<44} {'registers':>16} {'spill':>12} {'smem':>14} {'ld.global':>14} {'st.global':>14}")
    for name, pair in sorted(got.items()):
        if False not in pair or True not in pair:
            continue
        off, on = pair[False], pair[True]
        if isinstance(off, str) or isinstance(on, str):
            print(f"  {name:<44} nvcc: {off if isinstance(off, str) else on}")
            continue
        def cell(a, b):
            return f"{a} -> {b}" + ("" if a == b else f" ({b - a:+d})")
        print(f"  {name:<44} {cell(off[0], on[0]):>16} {cell(off[1] + off[2], on[1] + on[2]):>12} "
              f"{cell(off[3], on[3]):>14} {cell(off[4][0], on[4][0]):>14} {cell(off[4][1], on[4][1]):>14}")


if __name__ == "__main__":
    main()
