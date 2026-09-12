"""Does the mHC tilelang lane lower for SM121 with PDL on? That was the recorded reason it is off.

`kernels/mhc/tilelang_kernels.ENABLE_PDL` has sat False with the note "SM12x lowering is unvalidated".
Validating it needs the compiler, not the GPU -- and production GLM-5.3 is this engine, so a window is downtime.
This lowers every PDL-bearing kernel for sm_121a with the flag both ways and reports what came out.

    docker run --rm -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/mhc_pdl_lowering.py

Answered 2026-09-12 (ledger 45차 §63): six of the eight lower, and with the flag on all six emit BOTH
cudaGridDependencySynchronize() and cudaTriggerProgrammaticLaunchCompletion(). What the flag costs, in all six,
is `__restrict__` on the kernel pointer parameters.

The other two (`mhc_pre_big_fuse_with_norm_tilelang` and its broadcast sibling) fail register allocation here --
IDENTICALLY with the flag off, so not PDL's doing -- and that failure says nothing about the kernels: the served
lane calls the first of them at every layer of every step with exactly the arguments used here, and production
is serving. Without a device, `determine_target` cannot be asked, and a hand-written target dict is evidently
not the same target. Treat those two rows as "this harness could not ask", not as a result.
"""
import inspect, pathlib, shutil, sys, traceback

SRC, DST = pathlib.Path("/w/engine"), pathlib.Path("/tmp/e/engine")
if not DST.exists():
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
sys.path.insert(0, "/tmp/e")
K = DST / "kernels" / "mhc" / "tilelang_kernels.py"
import tilelang

# GLM-5.3's own geometry for the REQUIRED parameters only. Anything with a default is left at it: overriding
# n_splits to 1 made two kernels fail register allocation and read as "PDL does not lower", which was this
# harness's doing and not PDL's -- they fail the same way with the flag off.
GLM = {"hidden": 4096, "hidden_size": 4096, "hc": 4, "hc_mult": 4, "n_out": 24,
       "rms_eps": 1e-5, "hc_eps": 1e-5, "hc_pre_eps": 1e-5, "hc_sinkhorn_eps": 1e-5,
       "hc_post_mult_value": 1.0, "sinkhorn_repeat": 0, "norm_eps": 1e-5,
       "post_mult": 1.0, "sinkhorn": 0}
TARGET = {"kind": "cuda", "arch": "sm_121a"}
WANT = ("cudaGridDependencySynchronize", "cudaTriggerProgrammaticLaunchCompletion")


def load(enable):
    text = K.read_text()
    K.write_text(text.replace("ENABLE_PDL = False", f"ENABLE_PDL = {enable}")
                     .replace("ENABLE_PDL = True", f"ENABLE_PDL = {enable}"))
    for name in [n for n in sys.modules if "tilelang_kernels" in n or n.startswith("engine.")]:
        del sys.modules[name]
    from engine.kernels.mhc import tilelang_kernels as tk
    assert tk.ENABLE_PDL is enable
    return tk


names = [n for n in dir(load(False)) if n.endswith("_tilelang")]
print(f"{len(names)} tilelang entry points\n")
for enable in (False, True):
    tk = load(enable)
    print(f"-- ENABLE_PDL = {enable}")
    for name in names:
        fn = getattr(tk, name)
        if not hasattr(fn, "get_tir"):
            continue
        try:
            params = inspect.signature(fn.func if hasattr(fn, "func") else fn).parameters
        except Exception:
            params = {}
        kw = {k: GLM[k] for k, v in params.items()
              if k in GLM and v.default is inspect.Parameter.empty}      # required only; defaults are the tuning
        try:
            tir = fn.get_tir(**kw)
            src = tilelang.compile(tir, target=TARGET).get_kernel_source()
        except Exception as exc:
            print(f"  {name:<46} SKIP  {type(exc).__name__}: {str(exc).splitlines()[-1][:70]}")
            continue
        hit = [w for w in WANT if w in src]
        flag = "OK  " if len(hit) == 2 else ("PARTIAL" if hit else "NO PDL")
        print(f"  {name:<46} {flag}  restrict={'yes' if '__restrict__' in src else 'NO '}  {len(src)}c")
