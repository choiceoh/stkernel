"""Build-time extraction of the pinned image's standalone DeepGEMM library.

Run before removing vLLM from the seed image. This copies the extension and
the entire include tree together; runtime code only imports ``deep_gemm``.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sysconfig


def main():
    packages = Path(sysconfig.get_path("purelib"))
    source = packages / "vllm/third_party/deep_gemm"
    target = packages / "deep_gemm"
    if target.exists():
        raise FileExistsError(f"refusing to replace an existing DeepGEMM: {target}")
    if not (source / "include").is_dir() or not list(source.glob("_C*.so")):
        raise RuntimeError("seed image is missing DeepGEMM extension or JIT headers")
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    for name, expected in manifest.items():
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"DeepGEMM extraction changed {name}")
    provenance = {"seed_vllm": importlib.metadata.version("vllm"), "files": manifest}
    (target / "ST_SOURCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"extracted DeepGEMM: {len(manifest)} files, extension and JIT headers verified")


if __name__ == "__main__":
    main()
