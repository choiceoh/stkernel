"""Compile or check exact C1 activation stores and graph reuse; never reserves GPUs.

--cpu uses the real shared layout and store helper without creating a CUDA
context. --gpu requires an existing fleet GPU hold and checks bytes/canaries,
including changed payloads and partial rows. It does not measure throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu', action='store_true')
    mode.add_argument('--gpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cpu and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU compile requires CUDA_VISIBLE_DEVICES=')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cuda.bindings.driver as cuda
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x.moe_static_kernel_v4 import MoEStaticKernelV4
        from flashinfer.cute_dsl.fp4_common import shared_ptr_to_u32

    base, extent, generations = 1024, 3072, 4

    def compile_helper(packed_store):
        owner = MoEStaticKernelV4(16, 4, decode_reform=True, reform_sf_pack=True,
                                  packed_activation_store=packed_store)
        owner.a_dtype = cutlass.Float4E2M1FN
        owner.a_layout = utils.LayoutEnum.ROW_MAJOR

        @cute.kernel
        def write(src: cute.Tensor, rows: cute.Tensor, dst: cute.Tensor, layout: cute.ComposedLayout):
            tid, _, _ = cute.arch.thread_idx()
            smem = utils.SmemAllocator()
            data = smem.allocate(extent, byte_alignment=1024)
            raw = cute.make_tensor(cute.recast_ptr(data, dtype=cutlass.Uint8),
                                   cute.make_layout((extent,)))
            addr = shared_ptr_to_u32(data)
            i = cutlass.Int32(tid)
            while i < extent:
                raw[i] = cutlass.Uint8(0xA5)
                i += 128
            cute.arch.sync_threads()
            for generation in cutlass.range_constexpr(generations):
                row = cutlass.Int32(tid) // 8
                block = cutlass.Int32(tid) % 8
                if row < rows[generation]:
                    owner._store_packed_activation(addr+base, layout, row, block*8,
                                                   src[generation, row, block])
                cute.arch.sync_threads()
                i = cutlass.Int32(tid)
                while i < extent:
                    dst[generation, i] = raw[i]
                    i += 128
                cute.arch.sync_threads()

        @cute.jit
        def entry(src: cute.Tensor, rows: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
            layout = owner._make_a_smem_layout(16, 128, 1)
            write(src, rows, dst, layout).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

        src = cute.runtime.make_fake_compact_tensor(cutlass.Uint64, (generations, 16, 8),
            stride_order=(2, 1, 0), assumed_align=16)
        rows = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (generations,), assumed_align=16)
        dst = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (generations, extent),
            stride_order=(1, 0), assumed_align=16)
        return cute.compile(entry, src, rows, dst,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options='--opt-level 2 --enable-tvm-ffi')

    kernels = {flag: compile_helper(flag) for flag in (False, True)}
    records = []
    if args.gpu:
        source = torch.empty((generations, 16, 8), dtype=torch.uint64, device='cuda')
        rows = torch.empty((generations,), dtype=torch.int32, device='cuda')
        dest = torch.empty((generations, extent), dtype=torch.uint8, device='cuda')
        source.zero_()
        rows.zero_()
        for packed_store, kernel in kernels.items():
            kernel(source, rows, dest)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(source, rows, dest)
            rng = random.Random(915)
            for replay in range(64):
                payload = [[[rng.getrandbits(64) for b in range(8)] for r in range(16)]
                           for g in range(generations)]
                counts = [(replay*generations+g) % 17 for g in range(generations)]
                source.copy_(torch.tensor(payload, dtype=torch.uint64))
                rows.copy_(torch.tensor(counts, dtype=torch.int32))
                graph.replay()
                observed = dest.cpu().tolist()
                expected = bytearray([0xA5])*extent
                for g, valid in enumerate(counts):
                    for r in range(valid):
                        for b in range(8):
                            offset = base+r*64+((b*8)^(((r >> 1)&3)<<4))
                            expected[offset:offset+8] = payload[g][r][b].to_bytes(8, 'little')
                    if bytes(observed[g]) != expected:
                        raise AssertionError(('activation bytes/canaries', packed_store, replay, g))
            records.append(dict(packed_activation_store=packed_store, graph_replays=64, exact=True))
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU compile initialized CUDA')
    report = dict(status='PASS', mode='gpu' if args.gpu else 'cpu',
        compiled_helpers=len(kernels), checks=records,
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'engine/kernels/b12x/moe_static_kernel_v4.py', 'engine/kernels/b12x/moe_static_common.py')},
        scope='helper compile' if args.cpu else 'exact helper bytes, canaries and graph replay')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
