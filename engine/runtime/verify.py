"""Fail closed on runtime ABI drift and report the deployed source identity."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(gpu=False):
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "runtime/dependencies.json").read_text())
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "triton", "flashinfer-python", "tilelang", "nvidia-cutlass-dsl")}
    for name, version in versions.items():
        if version != expected[name]:
            raise RuntimeError(f"ST ABI mismatch: {name}={version}, expected {expected[name]}")
    if not platform.python_version().startswith(expected["python"] + "."):
        raise RuntimeError("ST Python ABI differs from the pinned runtime")
    if importlib.util.find_spec("vllm") is not None:
        raise RuntimeError("standalone ST runtime must not contain vLLM")
    import torch
    import deep_gemm
    if torch.version.cuda != expected["cuda"]:
        raise RuntimeError(f"ST CUDA ABI mismatch: {torch.version.cuda}")
    library = Path(deep_gemm.__file__).resolve().parent
    provenance = json.loads((library / "ST_SOURCE.json").read_text())
    for name, digest in provenance["files"].items():
        if sha256(library / name) != digest:
            raise RuntimeError(f"DeepGEMM extension/header provenance changed: {name}")
    files = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    source = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    report = dict(passed=True, python=platform.python_version(), cuda=torch.version.cuda,
                  packages=versions, seed_image=expected["seed_image_id"],
                  engine_source_sha256=source, source_files=files,
                  deep_gemm_files=len(provenance["files"]), vllm_present=False)
    if gpu:
        from engine.profiles.glm53.facts import check_box
        report["device"] = check_box()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.gpu), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
