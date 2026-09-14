"""Exact engine SF6 byte/canary and repeated-graph gate; never reserves a GPU.

--cpu compiles the same helper without initializing CUDA. --gpu requires an
existing fleet GPU hold. Full real-weight MoE and consumer performance remain
separate gates; this probe has no throughput verdict.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
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
    import cuda.bindings.driver as cuda
    # Import-time admission only; no CUDA runtime call in CPU mode.
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x.moe_static_kernel_v4 import MoEStaticKernelV4
        from flashinfer.cute_dsl.fp4_common import shared_ptr_to_u32

    block, packed, slots = 2048, 1552, 2
    # Match the real Storage header's alignment and FC1 packed-ring offsets.
    source_base, dest_base, extent = 240, 4096, 4096+slots*block+16

    def compile_helper(separate, word_expand):
        owner = MoEStaticKernelV4(16, 4, decode_reform=True, reform_sf_pack=True,
                                  sf6_separate=separate, sf6_word_expand=word_expand)
        @cute.kernel
        def expand(src: cute.Tensor, dst: cute.Tensor):
            tid, _, _ = cute.arch.thread_idx()
            smem = cutlass.utils.SmemAllocator()
            data = smem.allocate(extent, byte_alignment=16)
            raw = cute.make_tensor(cute.recast_ptr(data, dtype=cutlass.Uint8),
                                   cute.make_layout((extent,)))
            addr = shared_ptr_to_u32(data)
            i = cutlass.Int32(tid)
            while i < extent:
                raw[i] = cutlass.Uint8(0xA5)
                i += 128
            cute.arch.sync_threads()
            for generation in cutlass.range_constexpr(4):
                slot = generation % slots
                input_offset = source_base+slot*packed if separate else dest_base+slot*block
                i = cutlass.Int32(tid)
                while i < packed:
                    raw[input_offset+i] = src[generation, i]
                    i += 128
                cute.arch.sync_threads()
                if cutlass.const_expr(separate):
                    owner._sf_expand_stage(addr+dest_base+slot*block, cutlass.Int32(tid), block,
                                           packed_addr=addr+input_offset)
                else:
                    owner._sf_expand_stage(addr+dest_base+slot*block, cutlass.Int32(tid), block)
                i = cutlass.Int32(tid)
                while i < extent:
                    dst[generation, i] = raw[i]
                    i += 128
                cute.arch.sync_threads()

        @cute.jit
        def entry(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
            expand(src, dst).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

        src = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (4, packed),
            stride_order=(1, 0), assumed_align=16)
        dst = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (4, extent),
            stride_order=(1, 0), assumed_align=16)
        return cute.compile(entry, src, dst,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options='--opt-level 2 --enable-tvm-ffi')

    cases = ((False, False), (True, False), (True, True))
    kernels = {case: compile_helper(*case) for case in cases}
    records = []
    if args.gpu:
        # Independent CPU encoding includes every base/code, modulo byte addition.
        from tests.test_engine_moe_sf6_staging import packed_codes
        source = torch.empty((4, packed), dtype=torch.uint8, device='cuda')
        dest = torch.empty((4, extent), dtype=torch.uint8, device='cuda')
        source.zero_()
        for (separate, word_expand), kernel in kernels.items():
            kernel(source, dest)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(source, dest)
            for replay in range(64):
                blocks = [packed_codes((replay*4+g) % 256, block, replay+g) for g in range(4)]
                source.copy_(torch.tensor([list(pair[0]) for pair in blocks], dtype=torch.uint8))
                graph.replay()
                observed = dest.cpu().tolist()
                expected = bytearray([0xA5])*extent
                for generation, (packed_bytes, raw_bytes) in enumerate(blocks):
                    slot = generation % slots
                    input_offset = source_base+slot*packed if separate else dest_base+slot*block
                    expected[input_offset:input_offset+packed] = packed_bytes
                    output_offset = dest_base+slot*block
                    expected[output_offset:output_offset+block] = raw_bytes
                    if bytes(observed[generation]) != expected:
                        raise AssertionError(('SF6 bytes/canaries', separate, word_expand, replay, generation))
            records.append(dict(separate=separate, word_expand=word_expand, graph_replays=64, exact=True))
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU compile initialized CUDA')
    report = dict(status='PASS', mode='gpu' if args.gpu else 'cpu',
        compiled_helpers=len(kernels), checks=records,
        source_sha256=hashlib.sha256((root/'engine/kernels/b12x/moe_static_kernel_v4.py').read_bytes()).hexdigest(),
        scope='helper compile' if args.cpu else 'exact helper bytes, canaries and graph replay')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
