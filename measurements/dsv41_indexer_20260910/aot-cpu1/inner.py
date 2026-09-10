
import hashlib, json, os, pathlib, sys
root, out = pathlib.Path('/repo'), pathlib.Path('/evidence')
assert os.environ['NVIDIA_VISIBLE_DEVICES'] == 'void'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
devices = [str(p) for pattern in ('nvidia*', 'dri/*', 'kfd')
           for p in pathlib.Path('/dev').glob(pattern)]
assert not devices, devices
import torch, triton
assert not torch.cuda.is_initialized()
sys.path.insert(0, str(root / 'overlay/modules/dsv41_model'))
from dsv41_indexer_triton import offline_compile
compiled = offline_compile(out / 'compiled')
assert compiled, 'no compiled specializations'
assert not torch.cuda.is_initialized()
assert not any(k == 'vllm' or k.startswith('vllm.') for k in sys.modules)
report = dict(schema=1, passed=True, compiled=compiled,
              torch_version=torch.__version__, triton_version=triton.__version__,
              torch_cuda_version=torch.version.cuda, devices=devices,
              cuda_initialized=False, gpu_numerics=False,
              gpu_performance=False, model_equivalence=False)
(out / 'aot.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
