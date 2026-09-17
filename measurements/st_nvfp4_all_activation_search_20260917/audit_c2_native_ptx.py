"""CPU-only native audit in the pinned serving image; does not initialize CUDA.

Set PYTHONPATH to the source snapshot and CUTE_DSL_ARCH=sm_121a. The final
JSON line records source identities and native instructions at the generic
global-store -> resident-grid-barrier -> TMA-load boundary, for both recipes
and both decode widths. This is an ordering audit, not a GPU race reproducer.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

import torch
import cutlass.cute as cute


def main():
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        root = Path(md.__file__).resolve().parents[3]
        files = ["moe_dispatch.py", "moe_static_kernel_v4.py", "moe_static_kernel_v5.py",
                 "moe_static_common.py", "fp4_scale_search.py"]
        sources = {f"engine/kernels/b12x/{name}": hashlib.sha256(
            (root / "engine/kernels/b12x" / name).read_bytes()).hexdigest() for name in files}
        original_compile = cute.compile

        def keep(*args, **kwargs):
            kwargs["options"] = kwargs.get("options", "") + " --keep-ptx --keep-cubin"
            return original_compile(*args, **kwargs)

        cells = []
        with patch.object(md, "get_num_sm", return_value=48), \
                patch.object(md, "get_max_active_clusters", return_value=48), \
                patch.object(md, "build_and_load_cute_dsl_kernel",
                             side_effect=lambda module, name, build, **kw: build()), \
                patch.object(cute, "compile", side_effect=keep):
            for m in map(int, os.environ.get("FC_AUDIT_ROWS", "8,16").split(",")):
                for arm in os.environ.get("FC_AUDIT_ARMS", "ss1,as1").split(","):
                    cfg = md._static_v2_decode_config(
                        md._parse_glm53_static_v2("t,r,sf6,batch," + arm), m)
                    cfg = md._static_v2_input_reuse_config(cfg, 288, 288, m, 4096, 512, 8, m*8)
                    fn, _ = md._get_static_kernel_v2(
                        288, 288, m, 4096, 512, 8, m*8, config=cfg, mac_override=48,
                        w13_chunk=256, activation="swigluoai_uninterleave",
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                    ptx = fn.__ptx__
                    assert isinstance(ptx, str) and ".entry" in ptx
                    lines = ptx.splitlines()
                    instructions = [(i+1, line.strip()) for i, line in enumerate(lines)
                                    if any(word in line for word in
                                           ("fence.", "membar.", "st.global",
                                            "cp.async.bulk.tensor", "ld.global.acquire",
                                            "st.global.release", "atom.global.add"))]
                    fences = [item for item in instructions if "fence." in item[1] or "membar." in item[1]]
                    binary = fn.__cubin__
                    if not isinstance(binary, bytes):
                        from cutlass.base_dsl.jit_executor import get_escaped_cubin_bytes
                        payloads = re.findall(r'llvm\.mlir\.global[^\n]*@\w+_binary\("([^"\n]*)"\)',
                                              str(fn.ir_module))
                        assert len(payloads) == 1
                        binary = get_escaped_cubin_bytes(payloads[0].encode())
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "kernel.cubin"
                        path.write_bytes(binary)
                        resources = subprocess.run(["cuobjdump", "-res-usage", str(path)],
                                                   capture_output=True, text=True, check=True).stdout
                        sass = subprocess.run(["cuobjdump", "-sass", str(path)],
                                              capture_output=True, text=True, check=True).stdout
                    cells.append(dict(m=m, arm=arm, input_reuse=cfg["input_reuse"],
                                      activation_radius=cfg["activation_scale_search"],
                                      fc2_radius=cfg["activation_scale_search"] or cfg["fc2_scale_search"],
                                      ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                                      cubin_sha256=hashlib.sha256(binary).hexdigest(),
                                      resources=resources,
                                      sass_fences=[line.strip() for line in sass.splitlines()
                                                   if re.search(r"MEMBAR|FENCE|ERRBAR|CCTL", line)],
                                      ptx_version=[line for line in lines if line.startswith((".version", ".target"))],
                                      fences=fences, memory_instructions=instructions))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(event="native_ptx_audit", gpu_used=False, torch=torch.__version__,
                          cuda=torch.version.cuda, sources=sources, cells=cells)), flush=True)


if __name__ == "__main__":
    main()
