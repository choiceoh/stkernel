#!/usr/bin/env python3
"""Compile the verbatim MK_SEG_MHC device code at both hidden sizes; no GPU.

GLM-5.3-Flash and DeepSeek-V4-Flash are hidden 4096; DeepSeek-V4.1-Flash is
5120. The segment was written against `constexpr int HIDDEN = 4096` and is now
parameterized, so there are two questions this answers without a device:

  0. does the segment still compile once MHC_MAX_TOK is its own constant --
     it appears in eleven pointer strides and one device counter, so a
     mismatch between the two halves is a silent wrong-address bug rather
     than a build error;
  1. does HID = 5120 compile at all -- 5120 divides HCHUNK (NCHUNK 16 -> 20)
     and MK_THREADS (MHC_EPT 16 -> 20), so it should, but "should" is not a
     receipt; and
  2. what does it cost. MhcTailRegs holds res[HC][MHC_EPT] plus nw[MHC_EPT]
     and the compute pass adds vals[MHC_EPT], so a quarter more per-thread
     state rides on that constant. `-Xptxas=-v` prints the register count and
     any spill, and a spill in a persistent grid is not a slowdown -- it is a
     residency change, and this kernel's grid barrier deadlocks if residency
     drops below the cached grid.

Like probes/mk_mla_prefill32_compile.py this checks ptxas resource allocation
only: not the PyTorch extension ABI, not numerics, not the GPU.

    python3 probes/mk_mhc_hidden_compile.py
"""
import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
ap.add_argument("--arch", default="sm_121a")
ap.add_argument("--keep", action="store_true")
args = ap.parse_args()

source = (Path(__file__).resolve().parents[1]
          / "overlay/modules/glm53_megakernel/glm53_megakernel.cu")
text = source.read_text()


def section(start: str, end: str) -> str:
    # The start must be unique -- that is what makes the extraction verbatim
    # and unambiguous. The end only has to be the first one after it.
    assert text.count(start) == 1, (start, text.count(start))
    i = text.index(start)
    return text[i:text.index(end, i)]


def one(pattern: str) -> str:
    found = re.search(pattern, text, re.S)
    assert found, pattern
    return found.group()


# The segment's own constants, then the two macros its bodies expand, then the
# segment verbatim from its banner to the one after it.
code = ("#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n"
        "#include <stdint.h>\n#include <math.h>\n\n")
for pattern in (r"constexpr int MK_THREADS = [0-9]+;",
                r"constexpr int MK_WARPS = [^;]+;",
                r"constexpr int HC = [0-9]+;[^\n]*",
                r"constexpr int NOUT = [^;]+;[^\n]*",
                r"constexpr int MAX_TOK = [0-9]+;[^\n]*",
                r"#define MHC_MAX_TOK_DEF [0-9]+",
                r"constexpr int MHC_MAX_TOK = [^;]+;",
                r"constexpr int HCHUNK = [0-9]+;",
                r"constexpr int HIDDEN = [0-9]+;",
                r"constexpr int HIDDEN_V41 = [0-9]+;",
                r"constexpr int NCHUNK = [^;]+;[^\n]*"):
    code += one(pattern) + "\n"

# Device helpers the segment calls but does not define.
code += one(r"__device__ __forceinline__ float mk_sigmoid\(float x\) \{[^}]*\}") + "\n"

# The probe macros compile to nothing in this unit: their storage lives in the
# extension and none of it changes with HID.
code += ("\n#define MK_MHC_PROBE(slot) do {} while (0)\n"
         "#define MK_MHC_TS(slot) do {} while (0)\n"
         "#define MK_SPIN_WAIT(cond, ns, site) while (cond) { __nanosleep(ns); }\n"
         "#define TORCH_CHECK(...) do {} while (0)\n\n")

code += section("struct MKMhcArgs {", "}  // namespace")

# Instantiate explicitly: these are templates with a default, so nothing in
# this unit would emit code for them otherwise.
code += """
template __global__ void mk_mhc_kernel<HIDDEN>(const MKMhcArgs);
template __global__ void mk_mhc_bf16_kernel<HIDDEN>(const MKMhcArgs);
template __global__ void mk_mhc_ar_kernel<true, HIDDEN>(const MKMhcArgs);
template __global__ void mk_mhc_ar_kernel<false, HIDDEN>(const MKMhcArgs);
template __global__ void mk_mhc_kernel<HIDDEN_V41>(const MKMhcArgs);
template __global__ void mk_mhc_bf16_kernel<HIDDEN_V41>(const MKMhcArgs);
template __global__ void mk_mhc_ar_kernel<true, HIDDEN_V41>(const MKMhcArgs);
template __global__ void mk_mhc_ar_kernel<false, HIDDEN_V41>(const MKMhcArgs);
"""

print("source_sha256   ", hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
print("extracted_sha256", hashlib.sha256(code.encode()).hexdigest(), flush=True)

with tempfile.TemporaryDirectory(prefix="mk-mhc-hidden-") as tmp:
    unit = Path(tmp) / "mhc.cu"
    unit.write_text(code)
    command = [args.nvcc, "-O2", f"-arch={args.arch}", "--cubin", "-Xptxas=-v",
               "-Xptxas=--warn-on-spills", "-std=c++17",
               str(unit), "-o", str(unit.with_suffix(".cubin"))]
    done = subprocess.run(command, capture_output=True, text=True)
    if args.keep:
        Path("/tmp/mk-mhc-hidden.cu").write_text(code)
        print("kept /tmp/mk-mhc-hidden.cu")
    if done.returncode != 0:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        raise SystemExit(f"nvcc failed ({done.returncode})")

    # ptxas reports per function; pair each with the hidden size it came from
    # so the 16 -> 20 cost is readable rather than inferred.
    report = done.stderr
    print()
    fn = None
    for line in report.splitlines():
        name = re.search(r"Compiling entry function '([^']+)'", line)
        if name:
            mangled = name.group(1)
            hid = "5120" if "Li5120E" in mangled else "4096"
            kind = ("ar_bf16" if "ar_kernel" in mangled and "Lb1E" in mangled else
                    "ar" if "ar_kernel" in mangled else
                    "bf16" if "bf16_kernel" in mangled else "plain")
            fn = f"hidden {hid}  {kind:8s}"
            continue
        used = re.search(r"Used (\d+) registers", line)
        if used and fn:
            spill = re.search(r"(\d+) bytes spill stores", line)
            spilled = int(spill.group(1)) if spill else 0
            flag = "  <-- SPILL" if spilled else ""
            print(f"  {fn}  {used.group(1):>3} regs  "
                  f"spill {spilled:>4} B{flag}")
            fn = None
    if "spill" in report and re.search(r"[1-9]\d* bytes spill stores", report):
        print("\nSPILLS PRESENT -- a persistent grid's residency is cached from "
              "occupancy, so this changes the grid, not just the speed.")
    print("\nDEVICE COMPILE PASS (no GPU validation)")
