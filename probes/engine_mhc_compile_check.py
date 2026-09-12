"""Compile the actual ST mHC post body for SM121 without using a GPU.

The isolated module keeps the kernel and its decorator verbatim, omitting
unrelated FlashInfer/Triton imports. Meta tensors supply only shapes and dtypes.
This verifies lowering and device compilation, not CUDA execution or timing.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('engine/kernels/mhc/tilelang_kernels.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvcc', default='nvcc')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'mhc_post_tilelang')
    body = '\n'.join(source.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno])
    prefix = '''import math
import tilelang
import tilelang.language as T
ENABLE_PDL = False
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}
'''
    isolated = args.output / 'post_module.py'
    isolated.write_text(prefix + body + '\n')
    spec = importlib.util.spec_from_file_location('st_mhc_post_compile', isolated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import torch
    import tilelang
    tensors = [torch.empty(shape, dtype=dtype, device='meta') for shape,dtype in (
        ((6,4,4),torch.float32), ((6,4,4096),torch.bfloat16),
        ((6,4),torch.float32), ((6,4096),torch.bfloat16),
        ((6,4,4096),torch.bfloat16))]
    kernel = module.mhc_post_tilelang
    tir = kernel.get_tir(*tensors, 4, 4096)
    (args.output / 'input.tir').write_text(tir.script())
    target = tilelang.tvm.target.Target({'kind':'cuda','arch':'sm_121a'})
    with target, tilelang.tvm.transform.PassContext(opt_level=3, config=kernel.pass_configs):
        lowered = tilelang.lower(tir, target=target, enable_device_compile=False)
    cu = args.output / 'post.cu'
    cu.write_text(lowered.kernel_source.rstrip() + '\n')
    (args.output / 'host.tir').write_text(lowered.host_mod.script())
    (args.output / 'device.tir').write_text(lowered.device_mod.script())
    package = Path(tilelang.__file__).resolve().parent
    command = [args.nvcc, '-O3', '-std=c++17', '-arch=sm_121a', '--cubin',
               '--ptxas-options=-v', '-I'+str(package/'src'), '-I'+str(package/'3rdparty/cutlass/include'),
               str(cu), '-o', str(args.output/'post.cubin')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (args.output / 'nvcc.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    report = {'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'generated_cuda_sha256':hashlib.sha256(cu.read_bytes()).hexdigest(),
              'tilelang':tilelang.__version__, 'torch':torch.__version__,
              'target':'sm_121a', 'gpu_used':False, 'device_compile_pass':True}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
