"""Compile or compare direct SF6 registers against the actual dense copy layout.

CPU mode never initializes CUDA. GPU mode is a prepared byte/graph gate;
this command does not reserve a GPU or measure serving performance.
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
        raise RuntimeError('CPU mode requires CUDA_VISIBLE_DEVICES=')
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
    from tests.test_engine_moe_sf6_staging import packed_codes
    from tests.test_engine_moe_register_scales import load_words

    packed, raw_size, slots, generations = 1552, 2048, 2, 4
    source_base, dest_base, extent = 240, 4096, 8208
    mappings = {}

    def compile_helper(kind, direct):
        owner = MoEStaticKernelV4(16, 4, decode_reform=True, reform_sf_pack=True)
        owner.a_dtype = owner.b_dtype = cutlass.Float4E2M1FN
        owner.sf_dtype = cutlass.Float8E4M3FN
        owner.a_layout = owner.b_layout = owner.c_layout = utils.LayoutEnum.ROW_MAJOR
        blocks, words = (4, 4) if kind == 'fc1' else (2, 8)

        @cute.kernel
        def check(src: cute.Tensor, out: cute.Tensor, canary: cute.Tensor,
                  layout: cute.Layout, mma: cute.TiledMma, tile_shape: cutlass.Constexpr):
            tid, _, _ = cute.arch.thread_idx()
            data = utils.SmemAllocator().allocate(extent, byte_alignment=16)
            memory = cute.make_tensor(cute.recast_ptr(data, dtype=cutlass.Uint8), cute.make_layout((extent,)))
            address = shared_ptr_to_u32(data)
            scales = cute.make_tensor(cute.recast_ptr(data+dest_base, dtype=owner.sf_dtype), layout)
            tile = cute.local_tile(scales, cute.slice_(tile_shape, (0, None, None)), (0, 0, None))
            fragment = owner._partition_fragment_SFB(tile[None, None, 0], mma.get_slice(tid), tid)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), owner.sf_dtype)
            copy = cute.make_tiled_copy(atom, owner._get_layoutSFB_TV(mma),
                (cute.size(mma.permutation_mnk[1]), cute.size(mma.permutation_mnk[2])))
            thread = copy.get_slice(tid)
            source = thread.partition_S(tile)
            dest = cute.filter_zeros(thread.retile(fragment))
            i = cutlass.Int32(tid)
            while i < extent:
                memory[i] = cutlass.Uint8(0xA5)
                i += 128
            cute.arch.sync_threads()
            for generation in cutlass.range_constexpr(generations):
                slot = generation % slots
                input_offset = source_base+slot*packed
                i = cutlass.Int32(tid)
                while i < packed:
                    memory[input_offset+i] = src[generation, i]
                    i += 128
                cute.arch.sync_threads()
                if cutlass.const_expr(not direct):
                    owner._sf_expand_stage(address+dest_base+slot*raw_size, cutlass.Int32(tid), raw_size,
                                           packed_addr=address+input_offset)
                for kb in cutlass.range_constexpr(blocks):
                    if cutlass.const_expr(direct):
                        owner._sf6_load_fragment(dest[None, None, kb], address+input_offset, tid, kind, kb)
                    else:
                        original = cute.filter_zeros(source[None, None, None, slot])
                        cute.copy(copy, original[None, None, kb], dest[None, None, kb])
                    result = cute.recast_tensor(dest[None, None, kb], cutlass.Uint32)
                    for word in cutlass.range_constexpr(words):
                        out[generation, tid, kb, word] = result[word]
                cute.arch.sync_threads()
                i = cutlass.Int32(tid)
                while i < extent:
                    canary[generation, i] = memory[i]
                    i += 128
                cute.arch.sync_threads()

        @cute.jit
        def entry(src: cute.Tensor, out: cute.Tensor, canary: cute.Tensor, stream: cuda.CUstream):
            owner._setup_attributes(4096)
            mappings[kind] = owner.sf6_register_offsets[kind]
            if cutlass.const_expr(kind == 'fc1'):
                layout, mma, tile = owner.sfb1_smem_layout_staged, owner.tiled_mma1, owner.fc1_tile_shape_mnk
            else:
                layout, mma, tile = owner.sfb2_smem_layout_staged, owner.tiled_mma, owner.tile_shape_mnk
            check(src, out, canary, layout, mma, tile).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

        source = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (generations, packed),
            stride_order=(1, 0), assumed_align=16)
        output = cute.runtime.make_fake_compact_tensor(cutlass.Uint32, (generations, 128, blocks, words),
            stride_order=(3, 2, 1, 0), assumed_align=16)
        canary = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (generations, extent),
            stride_order=(1, 0), assumed_align=16)
        return cute.compile(entry, source, output, canary,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options='--opt-level 2 --enable-tvm-ffi')

    kernels = {(kind, direct): compile_helper(kind, direct)
               for kind in ('fc1', 'fc2') for direct in (False, True)}
    # Execute the exact integer/shuffle helper at every compiled copy offset.
    # A CPU quad is translated to each recorded lane's physical row here.
    cpu_words = 0
    for kind, lanes in mappings.items():
        for base in (0, 64, 127, 128, 192, 255):
            encoded, raw = packed_codes(base, raw_size, base*17)
            memory = bytearray([0xA5])*4096
            memory[256:256+packed] = encoded
            for tid, lane in enumerate(lanes):
                for offsets in lane:
                    actual, _ = load_words(memory, 256, offsets, tid//4*4)
                    expected = [int.from_bytes(raw[i:i+4], 'little') for i in offsets]
                    if actual != expected:
                        raise AssertionError((kind, base, offsets))
                    cpu_words += len(offsets)
    records = []
    if args.gpu:
        source = torch.zeros((generations, packed), dtype=torch.uint8, device='cuda')
        for (kind, direct), kernel in kernels.items():
            blocks, words = (4, 4) if kind == 'fc1' else (2, 8)
            output = torch.empty((generations, 128, blocks, words), dtype=torch.uint32, device='cuda')
            canary = torch.empty((generations, extent), dtype=torch.uint8, device='cuda')
            kernel(source, output, canary)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(source, output, canary)
            for replay in range(64):
                payloads = [packed_codes(replay*4+g, raw_size, replay+g) for g in range(generations)]
                source.copy_(torch.tensor([list(p) for p, _ in payloads], dtype=torch.uint8))
                graph.replay()
                observed, guards = output.cpu().tolist(), canary.cpu().tolist()
                memory = bytearray([0xA5])*extent
                for g, (encoded, raw) in enumerate(payloads):
                    slot = g % slots
                    memory[source_base+slot*packed:source_base+(slot+1)*packed] = encoded
                    if not direct:
                        memory[dest_base+slot*raw_size:dest_base+(slot+1)*raw_size] = raw
                    if bytes(guards[g]) != memory:
                        raise AssertionError(('shared canaries', kind, direct, replay, g))
                    expected = [[[int.from_bytes(raw[i:i+4], 'little') for i in offsets]
                                 for offsets in lane] for lane in mappings[kind]]
                    if observed[g] != expected:
                        raise AssertionError(('operand bytes', kind, direct, replay, g))
            graph.reset()
            records.append(dict(kind=kind, direct=direct, graph_replays=64, exact=True))
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU mode initialized CUDA')
    report = dict(status='PASS', gpu_used=args.gpu, compiled_helpers=len(kernels),
        cpu_operand_words=cpu_words, gpu_checks=records,
        source_sha256=hashlib.sha256((root/'engine/kernels/b12x/moe_static_kernel_v4.py').read_bytes()).hexdigest(),
        scope='native helper compile and CPU operand oracle' if args.cpu else 'exact GPU operand bytes/canaries/replay')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
