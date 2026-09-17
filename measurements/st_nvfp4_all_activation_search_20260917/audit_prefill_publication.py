"""Compile the served short/long prefill paths without initializing CUDA.

Run in the pinned serving image with CUTE_DSL_ARCH=sm_121a and PYTHONPATH
pointing at the source snapshot. This inspects emitted instructions; it does
not launch a kernel or establish whether a permitted stale read occurred.
"""
import hashlib
import json
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
        root = Path(md.__file__).resolve().parent
        names = ["moe_dispatch.py", "_moe_dynamic/gated.py",
                 "moe_dynamic_gated_sf6.py", "moe_dynamic_gated_sf6_q0.py",
                 "moe_dynamic_gated_sf6_words.py", "moe_dynamic_gated_sf6_q0_words.py",
                 "moe_dynamic_gated_sf6_prefill.py", "fp4_scale_search.py"]
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in names}
        md.configure_static_v2("t,r,sf6,batch,as1")
        md.configure_tp_sf6_q0(True)
        original_compile = cute.compile
        launches = []

        def keep(launch, *args, **kwargs):
            owner = launch._kernel
            launches.append(dict(kernel_class=type(owner).__name__,
                                 activation_radius=getattr(owner, "activation_scale_search", None)))
            kwargs["options"] = kwargs.get("options", "") + " --keep-ptx --keep-cubin"
            return original_compile(launch, *args, **kwargs)

        cells = []
        with patch.object(md, "get_num_sm", return_value=48), \
                patch.object(md, "get_max_active_clusters", return_value=48), \
                patch.object(md, "build_and_load_cute_dsl_kernel",
                             side_effect=lambda module, name, build, **kw: build()), \
                patch.object(cute, "compile", side_effect=keep):
            for m in (2304, 32256):
                print(json.dumps(dict(event="compile_start", m=m)), flush=True)
                physical_tiles, _, _ = md._dynamic_task_geometry(
                    288, 512, m * 8, tile_m=128, tile_n=128)
                max_rows = physical_tiles * 128
                fn, mac = md._get_dynamic_kernel(
                    288, m, 4096, 512, 8, max_rows, tile_m=128,
                    tiled=True, reform_sf_pack=True, w13_chunk=256,
                    activation="swigluoai_uninterleave", swiglu_alpha=1.,
                    swiglu_beta=0., swiglu_limit=10.)
                ptx = fn.__ptx__
                assert isinstance(ptx, str) and ".entry" in ptx
                expected = ("MoEGatedDynamicKernelSF6Q0Words" if m <= 8192
                            else "MoEGatedDynamicKernelSF6Prefill")
                assert launches[-1] == dict(kernel_class=expected, activation_radius=1)
                lines = ptx.splitlines()
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
                selected = [(i + 1, line.strip()) for i, line in enumerate(lines)
                            if re.search(r"fence\.|membar\.|ld\.global\.acquire|"
                                         r"st\.global\.release|atom\.global\.add|"
                                         r"cp\.async\.bulk\.tensor", line)]
                cells.append(dict(m=m, mac=mac, max_rows=max_rows, **launches[-1],
                                  ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                                  cubin_sha256=hashlib.sha256(binary).hexdigest(),
                                  ptx_version=[s for s in lines if s.startswith((".version", ".target"))],
                                  generic_u64_stores=sum("st.global.u64" in s for s in lines),
                                  generic_byte_stores=sum(bool(re.search(r"st(?:\.global)?\.[usb]8", s))
                                                          for s in lines),
                                  global_proxy_fences=sum("fence.proxy.async.global" in s for s in lines),
                                  native_global_proxy_fences=sum("FENCE.VIEW.ASYNC.G" in s for s in sass.splitlines()),
                                  instructions=selected, resources=resources))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(event="prefill_publication_audit", gpu_used=False,
                          torch=torch.__version__, cuda=torch.version.cuda,
                          sources=sources, cells=cells)), flush=True)


if __name__ == "__main__":
    main()
