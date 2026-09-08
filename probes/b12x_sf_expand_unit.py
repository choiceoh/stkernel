#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the packed-scale expansion (cell q, 39차 §4c), on the GPU.

The MoE kernel's numerics gate says the expansion is wrong, while the host
packing round-trips on the device and the same DMA with unpacked stages (cell
q0) passes. This runs ONLY the expansion -- 128 threads, one 4096 B smem
stage, no pipeline, no MMA -- so a failure here is the expansion itself and a
pass sends the search back into the kernel's staging.

    bash probes/run_mk_probe.sh probes/b12x_sf_expand_unit.py
"""
import os
import sys

sys.path.insert(0, os.environ.get("MK_PKG_PATH", "/usr/local/lib/python3.12/dist-packages"))

import torch  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.pipeline as pipeline  # noqa: E402
from cutlass.cutlass_dsl import Int32  # noqa: E402
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_sf_pack import (  # noqa: E402
    SF_PACK_BLOCK,
    SF_STAGE_BYTES,
    pack_sf_inline,
    unpack_sf_inline,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_common import (  # noqa: E402
    _ld_shared_i32_volatile,
    _st_shared_i32,
    shared_ptr_to_u32,
)

_PLANE_A = SF_PACK_BLOCK // 2
_BASE_OFF = SF_PACK_BLOCK * 6 // 8


def expand(bar, stage_addr, tidx):
    """Byte for byte the kernel's _sf_expand_stage."""
    if True:
        a = []
        for w in range(4):
            a.append(_ld_shared_i32_volatile(stage_addr + Int32(16) * tidx + Int32(4 * w)))
        b = []
        for w in range(2):
            b.append(_ld_shared_i32_volatile(
                stage_addr + Int32(_PLANE_A) + Int32(8) * tidx + Int32(4 * w)))
        base = _ld_shared_i32_volatile(stage_addr + Int32(_BASE_OFF)) & Int32(0xFF)
        bar.arrive_and_wait()
        for j in range(8):
            word = Int32(0)
            for m in range(4):
                i = 4 * j + m
                nib = (a[i >> 3] >> Int32(8 * ((i >> 1) & 3) + 4 * (i & 1))) & Int32(0xF)
                hi = (b[i >> 4] >> Int32(8 * ((i >> 2) & 3) + 2 * (i & 3))) & Int32(0x3)
                val = (base + nib + (hi << Int32(4))) & Int32(0xFF)
                word = word | (val << Int32(8 * m))
            _st_shared_i32(stage_addr + Int32(32) * tidx + Int32(4 * j), word)


@cute.jit
def run(packed: cute.Tensor, out: cute.Tensor, stream):
    kernel(packed, out).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)


@cute.kernel
def kernel(packed: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()

    @cute.struct
    class Storage:
        buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, SF_PACK_BLOCK], 1024]

    storage = smem.allocate(Storage)
    base_addr = shared_ptr_to_u32(storage.buf.data_ptr())
    # stage the packed bytes: 3088 B over 128 threads, 4 B each, plain copies
    for w in range(SF_STAGE_BYTES // 4 // 128 + 1):
        idx = Int32(w) * Int32(128) + Int32(tidx)
        if idx * Int32(4) < Int32(SF_STAGE_BYTES):
            byte = idx * Int32(4)
            word = (packed[byte].to(Int32)
                    | (packed[byte + Int32(1)].to(Int32) << Int32(8))
                    | (packed[byte + Int32(2)].to(Int32) << Int32(16))
                    | (packed[byte + Int32(3)].to(Int32) << Int32(24)))
            _st_shared_i32(base_addr + byte, word)
    cute.arch.sync_threads()
    expand(pipeline.NamedBarrier(barrier_id=3, num_threads=128),
           base_addr, Int32(tidx))
    cute.arch.sync_threads()
    for j in range(SF_PACK_BLOCK // 4 // 128):
        byte = (Int32(j) * Int32(128) + Int32(tidx)) * Int32(4)
        word = _ld_shared_i32_volatile(base_addr + byte)
        for m in range(4):
            out[byte + Int32(m)] = ((word >> Int32(8 * m)) & Int32(0xFF)).to(cutlass.Uint8)


def main() -> int:
    torch.cuda.init()
    dev = "cuda"
    g = torch.Generator().manual_seed(23)
    bad = 0
    for base, span in ((0x5c, 45), (0x6e, 21), (0x40, 64), (0x00, 1)):
        sf = (base + torch.randint(0, span, (SF_PACK_BLOCK,), generator=g,
                                   dtype=torch.int16)).to(torch.uint8).to(dev)
        staged = pack_sf_inline(sf, SF_PACK_BLOCK).reshape(-1)
        assert torch.equal(unpack_sf_inline(staged.reshape(1, -1), SF_PACK_BLOCK), sf)
        out = torch.zeros(SF_PACK_BLOCK, dtype=torch.uint8, device=dev)
        compiled = cute.compile(
            run,
            cute.runtime.from_dlpack(staged),
            cute.runtime.from_dlpack(out),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        )
        compiled(cute.runtime.from_dlpack(staged), cute.runtime.from_dlpack(out))
        torch.cuda.synchronize()
        ok = torch.equal(out, sf)
        n_bad = int((out != sf).sum())
        print(f"base {base:#04x} span {span:2d}: {'EXACT' if ok else f'{n_bad} of 4096 bytes wrong'}")
        if not ok:
            bad += 1
            d = (out != sf).nonzero().flatten()[:8].tolist()
            print("   first wrong bytes:", [(int(i), int(out[i]), int(sf[i])) for i in d])
    print("VERDICT:", "PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
