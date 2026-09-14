"""Compile C1/C2 SF6 tiles and operand controls in an existing CPU-only ST image."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import time
from unittest.mock import patch


def instruction_opcodes(sass):
    # CUDA 13.2 spells the packed-byte opcode VIADD.U8x4 (lowercase x).
    return re.findall(r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?'
                      r'([A-Z][A-Za-z0-9_.]*)(?:\s|;)', sass, re.M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sass', action='store_true', help='also disassemble and count native instructions')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--activation-store', action='store_true', help='compare packed activation stores')
    mode.add_argument('--fc2-words', action='store_true', help='compare in-place FC2 word restoration')
    mode.add_argument('--fc1-reuse', action='store_true', help='compare gate/up A/SFA register reuse')
    mode.add_argument('--compact-staging', action='store_true', help='compare compact FC1 inputs and disjoint FC2 scales')
    mode.add_argument('--register-scales', action='store_true', help='compare direct MMA scale registers')
    mode.add_argument('--batch-reform', action='store_true', help='compare the C2 M16 tile against the served M32 tile')
    mode.add_argument('--sync-cleanup', action='store_true', help='compare batched pipeline initialization and C1 publication')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    if args.sass and shutil.which('nvdisasm') is None:
        raise RuntimeError('native instruction checks require nvdisasm on PATH')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    os.environ['CUTE_DSL_KEEP'] = 'ptx,cubin'
    os.environ['CUTE_DSL_DUMP_DIR'] = str(args.output.parent / 'cute')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    records, selected = [], {}
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        from engine.kernels.b12x.moe_static_kernel_v4 import MoEStaticKernelV4
        import cutlass.cute as cute
        setup = MoEStaticKernelV4._setup_attributes
        validate_fragment = MoEStaticKernelV4._validate_fc1_reuse_fragment

        def checked_fragment(owner, fragment, name):
            counts = validate_fragment(owner, fragment, name)
            selected.setdefault('retained_fragment_elements', {})[name] = counts
            return counts

        def checked_setup(owner, hidden_size):
            setup(owner, hidden_size)
            a_bytes = sum(cute.size_in_bytes(dtype, cute.slice_(layout, (None, None, 0)))
                          for dtype, layout in ((owner.a_dtype, owner.a1_smem_layout_staged),
                                                (owner.sf_dtype, owner.sfa1_smem_layout_staged)))
            b_bytes = cute.size_in_bytes(owner.b_dtype,
                                        cute.slice_(owner.b1_smem_layout_staged, (None, None, 0)))
            weight_bytes = b_bytes + 1552*owner.sf1_packed_blocks
            selected.update(smem_bytes=owner.smem_bytes,
                            tile_m=owner.tile_m,
                            direct_scatter=owner.direct_scatter,
                            scatter_reuse=owner.scatter_reuse,
                            fc2_prefetch=owner.fc2_prefetch,
                            fc2_stages=owner.fc2_stages,
                            scatter_row_pairs=getattr(owner, 'scatter_row_pairs', None),
                            scatter_pair_rows=getattr(owner, 'scatter_pair_rows', None),
                            smem_capacity=owner.smem_capacity,
                            sf6_registers=owner.sf6_registers,
                            sf6_register_offsets=getattr(owner, "sf6_register_offsets", None),
                            separate=owner.sf6_separate, word_expand=owner.sf6_word_expand,
                            fc2_word_expand=owner.sf6_fc2_word_expand,
                            fc1_reuse_a=owner.fc1_reuse_a,
                            compact_staging=owner.compact_staging,
                            sync_cleanup=owner.sync_cleanup, a_barrier_count=owner.a_barrier_count,
                            fc1_input_stages=owner.fc1_input_stages,
                            fc1_input_bytes=sum(cute.size_in_bytes(dtype, layout) for dtype, layout in
                                ((owner.a_dtype, owner.a1_smem_layout_staged),
                                 (owner.sf_dtype, owner.sfa1_smem_layout_staged))),
                            fc2_packed_bytes=owner.sf2_packed_bytes,
                            fc1_a_sfa_bytes=a_bytes,
                            fc1_gate_bytes=weight_bytes+a_bytes,
                            fc1_up_bytes=weight_bytes+(0 if owner.fc1_reuse_a else a_bytes),
                            packed_activation_store=owner.packed_activation_store)

        def builder(module, name, build, **kwargs):
            start = time.monotonic()
            kernel = build()  # Actual CuTe lowering, PTXAS and TVM-FFI; no CUDA context.
            record = dict(selected, kernel=name, status='PASS', seconds=time.monotonic()-start)
            cubin = kernel.__cubin__
            if not isinstance(cubin, bytes):
                # The CUDA-dialect pipeline lowers its fatbin into an LLVM
                # byte string, while __cubin__ returns an unused dump path.
                # Decode that retained binary without loading a CUDA library.
                from cutlass.base_dsl.jit_executor import get_escaped_cubin_bytes
                payloads = re.findall(r'llvm\.mlir\.global[^\n]*@\w+_binary\("([^"\n]*)"\)',
                                      str(kernel.ir_module))
                if len(payloads) != 1:
                    raise RuntimeError(f'expected one retained native binary, got {len(payloads)}')
                cubin = get_escaped_cubin_bytes(payloads[0].encode())
            artifact = args.output.parent / (name + '.fatbin')
            artifact.write_bytes(cubin)
            resources = subprocess.run(['cuobjdump', '--dump-resource-usage', str(artifact)],
                                       check=True, capture_output=True, text=True).stdout
            record.update(native_binary_sha256=hashlib.sha256(cubin).hexdigest(), resources=resources)
            if args.sass:
                sass = subprocess.run(['cuobjdump', '--dump-sass', str(artifact)],
                                      check=True, capture_output=True, text=True).stdout
                instructions = instruction_opcodes(sass)
                if not instructions:
                    raise RuntimeError('no native instructions were captured')
                artifact.with_suffix('.sass').write_text(sass)
                record.update(sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),
                    static_instructions=len(instructions), opcode_counts=dict(sorted(Counter(instructions).items())))
            records.append(record)
            print(json.dumps(record), flush=True)
            return kernel

        # Each comparison changes only its named axis; in particular the
        # earlier FC1-reuse probe must not silently include compact staging.
        defaults = dict(sf6_separate=True, sf6_word_expand=True,
                        packed_activation_store=True, sf6_fc2_word_expand=True,
                        fc1_reuse_a=True, compact_staging=False, sf6_registers=False, sync_cleanup=False)
        if args.compact_staging or args.register_scales or args.sync_cleanup:
            defaults['compact_staging'] = True
        if args.register_scales or args.sync_cleanup:
            # The served C1 handle carries both; the cleanup compiles over registers.
            defaults['sf6_registers'] = True
        if args.sync_cleanup:
            defaults['sync_cleanup'] = True
        cases = [(rows, {}) for rows in (1, 7, 8)]
        if args.register_scales:
            cases += [(8, dict(sf6_registers=False)), (8, dict(stamps=True))]
        elif args.sync_cleanup:
            cases += [(8, dict(sync_cleanup=False)), (8, dict(sf6_registers=False)),
                      (8, dict(compact_staging=False)), (8, dict(fc1=1)), (8, dict(stamps=True))]
        elif args.compact_staging:
            cases += [(8, dict(compact_staging=False)), (8, dict(fc1_reuse_a=False))]
        elif args.fc1_reuse:
            cases += [(8, dict(fc1_reuse_a=False))]
        elif args.fc2_words:
            cases += [(8, dict(sf6_fc2_word_expand=False)), (8, dict(sf6_separate=False))]
        elif args.activation_store:
            cases += [(8, dict(packed_activation_store=False))]
        else:
            cases += [(8, dict(sf6_word_expand=False)),
                      (8, dict(sf6_separate=False, sf6_word_expand=False))]
        cases += [(rows, {}) for rows in (16, 32)]
        if args.batch_reform:
            # Isolate C2 tile/operand reuse from direct register scatter.
            # C1 retains its existing compiled handle.
            defaults = {}
            cases = [(8, {}), (16, {}),
                     (16, dict(batch_reform=True, c2_direct_scatter=False)),
                     (16, dict(batch_reform=True, c2_direct_scatter=True, c2_scatter_reuse=False)),
                     (16, dict(batch_reform=True, c2_direct_scatter=True, c2_scatter_reuse=True, c2_fc2_prefetch=False)),
                     (16, dict(batch_reform=True, c2_direct_scatter=True, c2_scatter_reuse=True, c2_fc2_prefetch=True))]
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder), \
                patch.object(MoEStaticKernelV4, '_validate_fc1_reuse_fragment', checked_fragment), \
                patch.object(MoEStaticKernelV4, '_setup_attributes', checked_setup):
            for rows, overrides in cases:
                selected.clear()
                requested = dict(defaults, **overrides)
                selected.update(rows=rows, requested=requested)
                config = {**md._parse_glm53_static_v2('t,r,sf6'), **requested}
                try:
                    config = md._static_v2_decode_config(config, rows)
                    md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, rows*8,
                        config=config, mac_override=48, activation='swigluoai_uninterleave',
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                except Exception as exc:
                    record = dict(selected, status='FAIL', error=repr(exc))
                    records.append(record)
                    print(json.dumps(record), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    passed = len(records) == len(cases) and all(r['status'] == 'PASS' for r in records)
    report = dict(status='PASS' if passed else 'FAIL', gpu_used=False, kernels=records,
                  nvdisasm_version=(subprocess.check_output(['nvdisasm', '--version'], text=True).strip()
                                    if args.sass else None),
                  scope='native compile and layout checks; GPU numerics/replay/timing pending',
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ('engine/kernels/b12x/moe_dispatch.py',
                                   'engine/kernels/b12x/moe_w4a16_fp4_helpers.py',
                                   'engine/kernels/b12x/moe_static_kernel_v4.py',
                                   'engine/kernels/b12x/moe_static_common.py',
                                   'engine/kernels/b12x/moe_static_kernel_v5.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise RuntimeError('one or more native SF6 handles failed')


if __name__ == '__main__':
    main()
