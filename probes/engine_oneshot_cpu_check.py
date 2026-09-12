"""Build the actual Torch/verbs extension without opening a GPU or RDMA device."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root/'engine/kernels/oneshot'
    import torch
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location('st_oneshot_cpu_check', directory/'__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extension = module.build()
    assert hasattr(extension, 'oneshot_max_int64')
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', evidence='full Torch extension compile/load only', gpu_used=False,
                  torch=torch.__version__, cuda=torch.version.cuda, max_elements=module.MAX_ELEMENTS,
                  extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(directory.iterdir()) if p.is_file()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
