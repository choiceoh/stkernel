"""No-GPU full dense extension build and DSA/head-gate SM121 PTXAS compile."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--build-dir', type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('compile needs CUDA_VISIBLE_DEVICES= and NVIDIA_VISIBLE_DEVICES=void')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from torch.utils.cpp_extension import load
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.mla.decode_inputs import _latent_norm_write
    from engine.kernels.indexer_gate import _gate_partials
    from engine.kernels.decode_projection import _indexer_boundary
    from engine.kernels.mla.prefill_absorb import _absorb
    from engine.kernels.indexer import _pool_window, _update_pool_cache, _pool_slots
    from engine.kernels.kpool import _kpool_softmax_rotate_write_cache_kernel
    from engine.kernels.common.native_cache import prepare_sources
    python_source = root / 'engine/kernels/dense/__init__.py'
    body = next(n for n in ast.parse(python_source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'extension')
    flags = next(ast.literal_eval(n.value) for n in body.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'flags' for t in n.targets))
    key, directory, sources = prepare_sources(args.build_dir, [root / 'engine/kernels/dense/kernels.cu'],
                                              (flags, torch.__version__, torch.version.cuda))
    load(name='st_dense_' + key, sources=list(sources), extra_cuda_cflags=flags,
         build_directory=str(directory), verbose=True)
    records = []
    for xs, ls, ts0, ts1, offset in ((2048, 512, 176, 1, 0), (2056, 520, 352, 2, 7680)):
        src = ASTSource(fn=_latent_norm_write, signature=dict(X='*bf16', W='*bf16', LATENT='*fp8e4nv',
                        TABLE='*i32', CTX='*i64', EPS='fp32'),
                        constexprs=dict(XS=xs, LS=ls, TS0=ts0, TS1=ts1, BLOCK=768,
                                        BLOCK_STRIDE=8448, OFFSET=offset, TOKENS=8, D=512))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
        records.append(dict(input_stride=xs, latent_stride=ls, block=768, block_stride=8448,
                            layer_offset=offset, shared_bytes=kernel.metadata.shared,
                            status='PASS'))
    head_records = []
    for stride in (4096, 4104):
        src = ASTSource(fn=_gate_partials, signature=dict(X='*bf16', W='*fp32', P='*fp32'), constexprs=dict(XS=stride))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                               options=dict(num_warps=4, enable_fp_fusion=False))
        head_records.append(dict(kernel='gate_partials', input_stride=stride, shared_bytes=kernel.metadata.shared,
                                 status='PASS'))
    for splits in (1, 16):
        src = ASTSource(fn=_indexer_boundary,
                        signature=dict(Q='*bf16', K='*bf16', W='*fp32', NW='*fp32', NB='*fp32',
                                       Q8='*fp8e4nv', KO='*bf16', WE='*fp32'),
                        constexprs=dict(NH=32, KS=256, SCALE=128 ** -.5 * 32 ** -.5, HEAD_SPLITS=splits))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                               options=dict(num_warps=1, enable_fp_fusion=False))
        head_records.append(dict(kernel='indexer_boundary', head_splits=splits, shared_bytes=kernel.metadata.shared,
                                 status='PASS'))
    reader_records = []
    for mapped, score_dtype, cache_write in ((False, 'bf16', False), (False, 'fp32', False),
                                             (False, 'fp32', True), (True, 'bf16', False)):
        signature = dict(buf_fp8_ptr='*fp8e4nv', buf_fp32_ptr='*fp32', slot_k_ptr='*bf16',
                         slot_score_ptr='*' + score_dtype, ape_ptr='*fp32',
                         loc_ptr='*i64' if cache_write else '*fp8e4nv',
                         write_mask_ptr='*i1' if cache_write else '*fp8e4nv',
                         compressed_k_ptr='*fp8e4nv', compressed_scale_ptr='*fp32',
                         **{k: 'i32' for k in ('slot_k_stride_0', 'slot_k_stride_1', 'slot_score_stride_0',
                                              'slot_score_stride_1', 'ape_stride_0')})
        constants = dict(PAGE_SIZE=64 if cache_write else 1, BUF_NUMEL_PER_PAGE=64*132 if cache_write else 1,
                         POOL_SIZE=4, HEAD_DIM=128, S_OFFSET_NBYTES_IN_PAGE=64*128 if cache_write else 0,
                         ROUND_SCALE=True, HAS_WRITE_MASK=cache_write, RETURN_COMPRESSED=True, WRITE_CACHE=cache_write,
                         BLOCK_D=128, slot_k_stride_2=1, slot_score_stride_2=1, ape_stride_1=1,
                         WARP_LOCAL_ROTATION=not cache_write, MAPPED_INPUT=mapped,
                         MAX_POOLS=2 if mapped else 1, TOKENS=8 if mapped else 0, TAIL_WIDTH=10 if mapped else 0,
                         TAIL_STRIDE_0=3264 if mapped else 0, TAIL_STRIDE_1=272 if mapped else 0,
                         TAIL_STRIDE_2=136 if mapped else 0)
        if mapped:
            signature.update(tail_ptr='*bf16', physical_slots_ptr='*i64', contexts_ptr='*i64')
        else:
            constants.update(tail_ptr=None, physical_slots_ptr=None, contexts_ptr=None)
        kernel = triton.compile(ASTSource(fn=_kpool_softmax_rotate_write_cache_kernel,
                                         signature=signature, constexprs=constants),
                                target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4 if cache_write else 1))
        reader_records.append(dict(mapped=mapped, score_dtype=score_dtype, cache_write=cache_write,
                                   shared_bytes=kernel.metadata.shared, status='PASS'))
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    absorb_records = []
    for bm in (16, 32):
        for transpose in (False, True):
            inner, outer = (512, 256) if transpose else (256, 512)
            src = ASTSource(fn=_absorb, signature=dict(X='*bf16', W='*bf16', Y='*bf16', ROWS='i32'),
                            constexprs=dict(HEADS=16, INPUT=inner, OUTPUT=outer, WH=512*512, WR=512,
                                            TRANSPOSE=transpose, BM=bm, BN=64, BK=64))
            kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                                   options=dict(num_warps=4, num_stages=2))
            absorb_records.append(dict(tile_m=bm, transpose=transpose, shared_bytes=kernel.metadata.shared,
                                       status='PASS'))
    pool_records = []
    for mapped in (False, True):
        src = ASTSource(fn=_pool_window,
                        signature=dict(TAILS='*bf16', K='*bf16', GATE='*bf16', CTX='*i64', OUT_K='*bf16', OUT_G='*bf16',
                                       SLOTS='*i64', **{k: 'i32' for k in
                                       ('tail_s0', 'tail_s1', 'tail_s2', 'k_s0', 'k_s1', 'gate_s0', 'gate_s1')}),
                        constexprs=dict(T=8, KP=4, W=10, NPOS=8, D=128, MAPPED=mapped))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
        pool_records.append(dict(kernel='pool_window', mapped=mapped, shared_bytes=kernel.metadata.shared, status='PASS'))
    src = ASTSource(fn=_update_pool_cache,
                    signature=dict(PK='*u8', PS='*fp32', KEYS='*u8', SCALES='*fp32', TAIL='*bf16', SLOTS='*i64',
                                   CTX='*i64', TABLE='*i32', K='*bf16', GATE='*bf16', **{k: 'i32' for k in
                                   ('table_s0', 'table_s1', 'key_s0', 'scale_s0', 'tail_s0', 'tail_s1', 'tail_s2',
                                    'k_s0', 'k_s1', 'gate_s0', 'gate_s1')}),
                    constexprs=dict(PER=192, STRIDE=2112, OFFSET=1920, CAP=32768, KP=4, T=8, MAXP=2, W=10, D=128))
    kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
    pool_records.append(dict(kernel='update_pool_cache', shared_bytes=kernel.metadata.shared, status='PASS'))
    for dtype in ('i32', 'i64'):
        src = ASTSource(fn=_pool_slots,
                        signature=dict(ids='*' + dtype, lengths='*i32', table='*i32', out='*i32', counts='*i32',
                                       **{k: 'i32' for k in ('id_s0', 'id_s1', 'len_s0', 'table_s0', 'out_s0',
                                                            'out_s1', 'count_s0', 'block_stride', 'layer_offset', 'table_s1')}),
                        constexprs=dict(groups=512, block_size=768, POOL=4, MAPPED=True, BLOCK=512, TOKENS=8,
                                        paths=None, context=0, PATH_WIDTH=0))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
        pool_records.append(dict(kernel='pool_slots', input_dtype=dtype, shared_bytes=kernel.metadata.shared, status='PASS'))
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/query_pair.py', 'engine/kernels/mla/decode_inputs.py',
             'engine/kernels/indexer_gate.py', 'engine/kernels/decode_projection.py',
             'engine/kernels/mla/decode_absorb.py', 'engine/kernels/mla/prefill_absorb.py',
             'engine/kernels/indexer.py', 'engine/kernels/kpool.py')
    result = dict(status='PASS', gpu_used=False, torch=torch.__version__, triton=triton.__version__,
                  cuda=torch.version.cuda, flags=flags, native_cache_key=key, latent_kernels=records, head_gate_kernels=head_records,
                  decode_absorb_kernels=absorb_records,
                  pool_cache_kernels=pool_records,
                  pool_reader_kernels=reader_records,
                  source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
                  scope='full native build and PTXAS only; GPU bytes/replay/timing pending')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
