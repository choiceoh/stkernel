"""Run on stdin in the built image, without a repository mount or PYTHONPATH override."""
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pkgutil
import sys

assert importlib.util.find_spec("vllm") is None


class ForbidVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise AssertionError(fullname)
        return None


sys.meta_path.insert(0, ForbidVllm())
import torch
import deep_gemm
import engine.kernels
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.weights import WEIGHT_LAYOUT

names = [m.name for m in pkgutil.walk_packages(engine.kernels.__path__, "engine.kernels.")]
for name in names:
    importlib.import_module(name)
table = served()
root = Path(engine.kernels.__file__).resolve().parents[1]
assert root == Path("/opt/st/engine"), root
expected = json.loads((root / "runtime/dependencies.json").read_text())
packages = ("torch", "triton", "flashinfer-python", "tilelang", "nvidia-cutlass-dsl")
versions = {name: importlib.metadata.version(name) for name in packages}
for name, version in versions.items():
    assert version == expected[name], (name, version, expected[name])
assert torch.version.cuda == expected["cuda"]
assert callable(deep_gemm.fp8_fp4_mqa_logits)
assert callable(deep_gemm.tf32_hc_prenorm_gemm)
origin = json.loads(Path(deep_gemm.__file__).with_name("ST_SOURCE.json").read_text())
assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
device = torch.cuda.get_device_properties(0)
print(json.dumps({
    "passed": True,
    "engine_root": str(root),
    "kernel_modules": len(names),
    "served_table": table.name,
    "vllm_installed": False,
    "vllm_loaded": False,
    "weight_layout": WEIGHT_LAYOUT,
    "python": sys.version,
    "cuda": torch.version.cuda,
    "packages": versions,
    "deep_gemm_path": deep_gemm.__file__,
    "deep_gemm_source_files": len(origin["files"]),
    "gpu": device.name,
    "capability": [device.major, device.minor],
    "sms": device.multi_processor_count,
}, indent=2))
