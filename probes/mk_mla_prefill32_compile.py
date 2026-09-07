#!/usr/bin/env python3
"""Compile the verbatim prefill32 device code with host nvcc; no GPU/runtime.

This checks ptxas resource allocation, not the PyTorch extension ABI or GPU
correctness. The numerical probe separately compiles the complete extension.
"""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
args = ap.parse_args()
source = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_megakernel/glm53_megakernel.cu"
text = source.read_text()
def section(start, end):
    assert text.count(start) == 1 and text.count(end) == 1, (start, end)
    return text[text.index(start):text.index(end)]
code = "#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <stdint.h>\n#include <math.h>\n"
code += re.search(r"constexpr int MK_THREADS = [0-9]+;", text).group() + "\n"
code += section("__device__ __forceinline__ void mk_cp_async16", "// wait_group takes an immediate")
code += section("constexpr int MLA_D =", "__global__ __launch_bounds__(MK_THREADS) void mk_mla_kernel")
code += section("constexpr int MLA_PREFILL_TILE =", "void mk_run_mla_prefill32")
print("source_sha256", hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
print("extracted_sha256", hashlib.sha256(code.encode()).hexdigest(), flush=True)
with tempfile.TemporaryDirectory(prefix="mla-prefill32-compile-") as tmp:
    unit = Path(tmp) / "prefill.cu"
    unit.write_text(code)
    command = [args.nvcc, "-O2", "-arch=sm_121a", "--cubin", "-Xptxas=-v",
               "-Xptxas=--warn-on-spills", str(unit), "-o", str(unit.with_suffix(".cubin"))]
    subprocess.run(command, check=True)
print("DEVICE COMPILE PASS (no GPU validation)", flush=True)
