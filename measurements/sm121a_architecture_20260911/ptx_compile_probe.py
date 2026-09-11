"""Compile-only ISA probes; deliberately do not launch these synthetic kernels.

Negative architecture checks use a known supported target as a syntax control.
PTX inputs use dummy addresses and are not executable correctness tests.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile


REGS = """
.shared .align 128 .b8 scratch[1024];
.reg .b64 p;
.reg .b32 a<4>, b<2>, s, q;
.reg .b16 h;
.reg .b8 packed;
.reg .f32 c<4>, d<4>;
ld.param.u64 p, [output];
mov.b32 a0, 0; mov.b32 a1, 0; mov.b32 a2, 0; mov.b32 a3, 0;
mov.b32 b0, 0; mov.b32 b1, 0; mov.u32 s, scratch; add.u32 q, s, 512;
mov.b16 h, 0;
cvt.u8.u32 packed, 0;
mov.f32 c0, 0f00000000; mov.f32 c1, 0f00000000;
mov.f32 c2, 0f00000000; mov.f32 c3, 0f00000000;
"""
OPERANDS = "{d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}"
STORE_FLOAT = "st.global.f32 [p], d0;"


def mma(instruction, suffix=""):
    return instruction + " " + OPERANDS + suffix + ";\n" + STORE_FLOAT


PROBES = {
    "mma_bf16": (mma("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"), None),
    "mma_tf32": (mma("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"), None),
    "mma_fp8": (mma("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"), None),
    "mma_fp6": (mma("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32"), "sm_120a"),
    "mma_nvfp4": (mma("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3", ", s, {0,0}, q, {0,0}"), "sm_120a"),
    "mma_mxfp4": (mma("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0", ", s, {0,0}, q, {0,0}"), "sm_120a"),
    "cp_async": ("cp.async.ca.shared.global [s], [p], 16; cp.async.commit_group; cp.async.wait_group 0;", None),
    "tma_load_2d": ("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [s], [p, {0,0}], [q];", "sm_90a"),
    "tma_store_2d": ("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [p, {0,0}], [s]; cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;", "sm_90a"),
    "mbarrier": ("mbarrier.init.shared.b64 [s], 1; mbarrier.arrive.expect_tx.shared.b64 p, [s], 16;", "sm_90a"),
    "async_proxy_fence": ("fence.proxy.async.shared::cta;", "sm_90a"),
    "ldmatrix_b16": ("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {a0}, [s]; st.global.b32 [p], a0;", None),
    "ldmatrix_b8": ("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {a0,a1}, [s]; st.global.b32 [p], a0;", "sm_120a"),
    "stmatrix_b8": ("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [s], {a0};", "sm_120a"),
    "cvt_fp4_pack": ("cvt.rn.satfinite.e2m1x2.f32 packed, c0, c1; st.global.b8 [p], packed;", "sm_120a"),
    "cvt_fp4_unpack": ("cvt.rn.f16x2.e2m1x2 a0, packed; st.global.b32 [p], a0;", "sm_120a"),
    "pdl": ("griddepcontrol.launch_dependents; griddepcontrol.wait;", "sm_90a"),
    "cluster_barrier": ("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "sm_90a"),
    "cluster_shared_load": ("mapa.shared::cluster.u32 q, s, 1; ld.shared::cluster.b32 a0, [q]; st.global.b32 [p], a0;", "sm_90a"),
    "cluster_launch_control": ("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [s], [q];", "sm_100a"),
    "atomic_bf16x2": ("atom.global.add.noftz.bf16x2 a0, [p], b0; st.global.b32 [p], a0;", "sm_90a"),
    "warp_reduce": ("redux.sync.max.s32 a0, b0, 0xffffffff; st.global.b32 [p], a0;", None),
    "wgmma": ("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {d0,d1,d2,d3}, p, p, 0, 1, 1, 0, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0; " + STORE_FLOAT, "sm_90a"),
    "tcgen05_tmem": ("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [s], 32;", "sm_100a"),
    "cvt_stochastic_fp4": ("cvt.rs.satfinite.e2m1x4.f32 h, {c0,c1,c2,c3}, s; st.global.b16 [p], h;", "sm_100a"),
}

results = []
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    for name, (body, control) in PROBES.items():
        targets = ["sm_121a"]
        if control:
            targets.append(control)
        if name == "mma_nvfp4":
            targets.extend(["sm_121", "sm_120f", "sm_121f"])
        for target in targets:
            path = root / (name + "-" + target + ".ptx")
            path.write_text(
                f".version 9.0\n.target {target}\n.address_size 64\n"
                ".visible .entry probe(.param .u64 output) {\n" + REGS + body + "\nret;\n}\n"
            )
            binary = path.with_suffix(".cubin")
            proc = subprocess.run(
                ["ptxas", "-arch=" + target, str(path), "-o", str(binary)],
                text=True, capture_output=True, timeout=10,
            )
            sass = ""
            disassembly_status = "not assembled"
            if proc.returncode == 0 and shutil.which("nvdisasm"):
                disasm = subprocess.run(["cuobjdump", "--dump-sass", str(binary)],
                                        text=True, capture_output=True, timeout=10)
                if disasm.returncode:
                    raise RuntimeError(disasm.stderr)
                sass = disasm.stdout
                disassembly_status = "available"
            elif proc.returncode == 0:
                disassembly_status = "nvdisasm absent from this image"
            results.append({"probe": name, "target": target,
                            "returncode": proc.returncode,
                            "assembled": proc.returncode == 0,
                            "diagnostics": proc.stderr.replace(str(root), "<tmp>"),
                            "disassembly_status": disassembly_status, "sass": sass})

version = subprocess.run(["ptxas", "--version"], capture_output=True, text=True, check=True)
print(json.dumps({"ptxas_version": version.stdout, "method": "compile only; optional disassembly; no kernel launches",
                  "results": results}, indent=2))
