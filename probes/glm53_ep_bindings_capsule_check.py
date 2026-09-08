#!/usr/bin/env python3
"""No-device metadata and import check for the pinned CUDA bindings capsule.

Run only through run_glm53_ep_bindings_capsule_cpu.py. The check imports the
bindings but calls no CUDA API and never imports Torch. PASS is not compilation,
GPU compatibility, sanitizer, kernel correctness, or performance evidence.
"""
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import time

from glm53_ep_bindings_capsule import PINNED_WHEELS, check_dependencies, stage_capsule, validate_capsule


IMAGE = "sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211"
SOURCE_PATHS = (
    "probes/glm53_ep_bindings_capsule.py",
    "probes/glm53_ep_bindings_capsule_check.py",
    "probes/run_glm53_ep_bindings_capsule_cpu.py",
)
SCOPE = "No-device metadata and import identity only; no compilation or GPU compatibility proof"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    return {name: file_hash(Path(root) / name) for name in SOURCE_PATHS}


def device_nodes(root=Path("/dev")):
    return sorted({str(path) for pattern in ("nvidia*", "dri/*", "kfd", "dxg")
                   for path in root.glob(pattern)})


def assert_no_devices():
    devices = device_nodes()
    if devices:
        raise RuntimeError("CPU capsule container exposes accelerator devices: " + repr(devices))
    return devices


def accelerator_modules():
    return sorted(name for name in sys.modules
                  if name in ("torch", "cuda") or name.startswith(("torch.", "cuda.")))


def runtime_identity():
    result = dict(python=sys.version, implementation=sys.implementation.name,
                  machine=platform.machine(), platform=sys.platform)
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or platform.machine() != "aarch64" or sys.platform != "linux"):
        raise RuntimeError("capsule requires the pinned CPython 3.12 Linux aarch64 image")
    if not sys.dont_write_bytecode:
        raise RuntimeError("capsule imports require Python -B; no pycache writes allowed")
    return result


def snapshot_distributions():
    """Read every installed METADATA before altering the Python search path."""
    from packaging.utils import canonicalize_name
    records = []
    for distribution in metadata.distributions():
        base = Path(distribution._path)
        # Keep system/legacy egg metadata in the complete inventory too.
        path = next((path for path in (base / "METADATA", base / "PKG-INFO", base)
                     if path.is_file()), None)
        if path is None:
            raise RuntimeError("installed distribution metadata file is missing")
        raw = path.read_bytes()
        records.append(dict(name=canonicalize_name(distribution.metadata["Name"]),
                            version=distribution.version, requires_dist=distribution.requires or [],
                            metadata_path=str(path.resolve()), metadata_sha256=hashlib.sha256(raw).hexdigest(),
                            metadata_text=raw.decode("utf-8")))
    return sorted(records, key=lambda record: (record["name"], record["metadata_path"]))


def base_pathfinder_identity(records):
    selected = [record for record in records if record["name"] == "cuda-pathfinder"]
    if len(selected) != 1 or selected[0]["version"] != "1.7.0":
        raise RuntimeError("base cuda-pathfinder must be exactly 1.7.0")
    distribution = metadata.distribution("cuda-pathfinder")
    path = Path(distribution.locate_file("cuda/pathfinder/__init__.py")).resolve()
    return dict(version=distribution.version, path=str(path), sha256=file_hash(path),
                metadata_path=selected[0]["metadata_path"], metadata_sha256=selected[0]["metadata_sha256"])


def module_identity(module, root, manifest):
    """Require the imported module to be an exact file from the staged wheels."""
    path = Path(module.__file__).resolve()
    try:
        relative = path.relative_to(Path(root).resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("binding import escaped the capsule: " + module.__name__) from exc
    expected = manifest["files"].get(relative)
    digest = file_hash(path)
    if expected is None or expected != dict(sha256=digest, size=path.stat().st_size):
        raise RuntimeError("imported binding file differs from the pinned manifest")
    return dict(module=module.__name__, path=str(path), relative_path=relative,
                sha256=digest, size=path.stat().st_size)


def import_capsule(root, manifest, pathfinder):
    if accelerator_modules():
        raise RuntimeError("accelerator modules were imported before selecting the capsule")
    if not sys.dont_write_bytecode:
        raise RuntimeError("capsule imports require Python -B")
    sys.path.insert(0, str(Path(root).resolve()))
    importlib.invalidate_caches()
    selected = {}
    for record in manifest["distributions"]:
        distribution = metadata.distribution(record["name"])
        path = Path(distribution._path) / "METADATA"
        if (distribution.version != "13.0.3"
                or path.resolve() != (Path(root) / record["metadata_path"]).resolve()
                or file_hash(path) != record["metadata_sha256"]):
            raise RuntimeError("selected distribution metadata did not resolve to the capsule")
        selected[record["name"]] = dict(version=distribution.version,
            metadata_path=str(path.resolve()), metadata_sha256=file_hash(path))
    modules = [importlib.import_module(name) for name in (
        "cuda.bindings", "cuda.bindings.driver", "cuda.bindings._bindings.cydriver")]
    if modules[0].__version__ != "13.0.3":
        raise RuntimeError("imported CUDA bindings version is not 13.0.3")
    identities = [module_identity(module, root, manifest) for module in modules]
    finder = importlib.import_module("cuda.pathfinder")
    distribution = metadata.distribution("cuda-pathfinder")
    finder_path = Path(finder.__file__).resolve()
    metadata_path = (Path(distribution._path) / "METADATA").resolve()
    current = dict(version=distribution.version, path=str(finder_path), sha256=file_hash(finder_path),
                   metadata_path=str(metadata_path), metadata_sha256=file_hash(metadata_path))
    if current != pathfinder or finder_path.is_relative_to(Path(root).resolve()):
        raise RuntimeError("cuda-pathfinder did not remain the original base 1.7.0 distribution")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("capsule import unexpectedly imported Torch")
    return dict(distributions=selected, modules=identities, base_pathfinder=current,
                imported_accelerator_modules=accelerator_modules())


def run_check(wheels, output):
    root = Path(__file__).resolve().parents[1]
    wheels, output = Path(wheels), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "result.json"
    if receipt_path.exists() or (output / "base-distributions.json").exists():
        raise RuntimeError("refusing to reuse a capsule check receipt")
    receipt = dict(schema=1, verdict="FAIL", scope=SCOPE, image=IMAGE,
                   image_binding="fixed immutable wrapper Docker argv; no engine query in this process",
                   started=time.time(), sources=source_hashes(root),
                   probe_cuda_api_calls=False, cuda_context_queried=False, torch_imported=False)
    try:
        receipt["exposed_device_nodes"] = assert_no_devices()
        receipt["runtime"] = runtime_identity()
        if accelerator_modules():
            raise RuntimeError("accelerator modules were already loaded before the check")
        from packaging.markers import default_environment
        base = snapshot_distributions()
        metadata_path = output / "base-distributions.json"
        metadata_path.write_text(json.dumps(base, indent=2) + "\n")
        receipt["base_distributions"] = dict(path=str(metadata_path), count=len(base), sha256=file_hash(metadata_path))
        pathfinder = base_pathfinder_identity(base)
        staged = stage_capsule([wheels / PINNED_WHEELS[name]["filename"] for name in sorted(PINNED_WHEELS)],
                               output / "capsule")
        receipt["capsule_manifest_sha256"] = staged["manifest_sha256"]
        manifest = validate_capsule(output / "capsule", staged["manifest_sha256"])
        receipt["marker_environment"] = default_environment()
        report = check_dependencies(base, manifest["distributions"], marker_environment=receipt["marker_environment"])
        receipt["dependencies"] = report
        if not report["compatible"]:
            raise RuntimeError("capsule introduces or retains a selected-package dependency conflict")
        receipt["imports"] = import_capsule(output / "capsule", manifest, pathfinder)
        # Catch import-time file changes, including accidental __pycache__ files.
        validate_capsule(output / "capsule", staged["manifest_sha256"])
        receipt["exposed_device_nodes"] = assert_no_devices()
        if receipt["sources"] != source_hashes(root):
            raise RuntimeError("probe source changed during the check")
        receipt["verdict"] = "PASS"
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt["ended"] = time.time()
        receipt["torch_imported"] = any(name == "torch" or name.startswith("torch.") for name in sys.modules)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("CAPSULE CPU IMPORT PASS; no compilation or GPU compatibility proof", flush=True)
    return receipt


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("the fixed capsule check accepts no command-line overrides")
    run_check(Path("/wheels"), Path("/evidence"))
