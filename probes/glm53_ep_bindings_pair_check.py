#!/usr/bin/env python3
"""Same minimal binding observation with pre-API package identity checks.

Both arms retain V5's order: fresh Torch context, binding imports, first count
and version queries. Identity checks occur after import and before either
binding API. This does not execute MoE, CuTe or a performance measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

if __package__:
    from .glm53_ep_bindings_capsule import validate_capsule
else:
    from glm53_ep_bindings_capsule import validate_capsule

SITE = Path('/usr/local/lib/python3.12/dist-packages')
MODULE_FILES = {
    'cuda.bindings.driver': 'cuda/bindings/driver.cpython-312-aarch64-linux-gnu.so',
    'cuda.bindings._bindings.cydriver': 'cuda/bindings/_bindings/cydriver.cpython-312-aarch64-linux-gnu.so',
}
# Installed pinned-image binaries; exact docker-cp/RECORD receipts are archived
# separately. These are not the hashes of a newly downloaded 13.3.1 wheel.
BASELINE_FILES = {
    MODULE_FILES['cuda.bindings.driver']: '23596c158b5ef437517605405aa25978d7a442fa1c222d59a392300bc39a0119',
    MODULE_FILES['cuda.bindings._bindings.cydriver']: '6ae437b694ad31accb9cd8745fa5dd9e52e11af045e0da683966be7b3341e67e',
    'cuda_bindings-13.3.1.dist-info/METADATA': '0bfcbe77dade61b1cba9f201c0465f2353f5b50adf4e4f554bca35a35033166f',
    'cuda_python-13.3.1.dist-info/METADATA': '8bea0c1333aaef1205d3cdea2690d935b4127448c1ed78e55c179fe1658856fd',
}
VERSIONS = {'baseline': '13.3.1', 'candidate': '13.0.3'}


def verify_import_identity(arm, capsule_root, manifest_sha256):
    """Bind imported binaries and selected metadata before calling any APIs."""
    manifest = validate_capsule(capsule_root, manifest_sha256)
    version = VERSIONS[arm]
    root = SITE if arm == 'baseline' else Path(capsule_root).resolve(strict=True)
    files = BASELINE_FILES if arm == 'baseline' else {
        name: row['sha256'] for name, row in manifest['files'].items()}
    identity = dict(version=version, root=str(root), modules={}, distributions={})
    for name, relative in MODULE_FILES.items():
        module = importlib.import_module(name)
        actual = Path(module.__file__)
        expected = root/relative
        if actual != expected or actual.resolve(strict=True) != expected:
            raise RuntimeError('binding import origin mismatch: '+name)
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if digest != files[relative]:
            raise RuntimeError('binding import hash mismatch: '+name)
        identity['modules'][name] = dict(path=str(actual), sha256=digest)
    for name in ('cuda-bindings', 'cuda-python'):
        distribution = importlib.metadata.distribution(name)
        relative = name.replace('-', '_')+'-'+version+'.dist-info/METADATA'
        actual = Path(distribution.locate_file(relative))
        expected = root/relative
        if (distribution.version != version or actual != expected
                or actual.resolve(strict=True) != expected):
            raise RuntimeError('binding distribution origin/version mismatch: '+name)
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        selected_metadata = distribution.read_text('METADATA')
        if (digest != files[relative] or selected_metadata is None
                or hashlib.sha256(selected_metadata.encode()).hexdigest() != digest):
            raise RuntimeError('binding distribution metadata mismatch: '+name)
        identity['distributions'][name] = dict(version=version, path=str(actual), sha256=digest)
    return identity


def observe(result, arm, capsule_root, manifest_sha256):
    import torch
    result['cuda_initialized_before'] = torch.cuda.is_initialized()
    if result['cuda_initialized_before']:
        raise RuntimeError('fresh process with no existing CUDA context required')
    result['phase'] = 'torch-context'
    print('EP_BINDING_CONTEXT_BEGIN', flush=True)
    keepalive = torch.empty((1,), device='cuda')
    torch.cuda.synchronize()
    print('EP_BINDING_CONTEXT_READY', flush=True)
    result['cuda_initialized_after'] = torch.cuda.is_initialized()
    result['phase'] = 'binding-identity'
    result['binding_identity'] = verify_import_identity(arm, capsule_root, manifest_sha256)
    print('EP_BINDING_IDENTITY_VERIFIED', flush=True)
    result['phase'] = 'binding-device-count'
    print('EP_BINDING_DEVICE_COUNT_BEGIN', flush=True)
    from cuda.bindings import driver
    count_code, count = driver.cuDeviceGetCount()
    print('EP_BINDING_DEVICE_COUNT_END', flush=True)
    version_code, version = driver.cuDriverGetVersion()
    result.update(cuda_bindings=importlib.metadata.version('cuda-bindings'),
                  count_result=[int(count_code), count], version_result=[int(version_code), version],
                  tensor_numel=keepalive.numel())
    if (int(count_code) or int(version_code) or type(count) is not int or count != 1
            or type(version) is not int or version != 13000
            or result['cuda_bindings'] != VERSIONS[arm]):
        raise RuntimeError('unexpected binding count/version observation')
    result.update(verdict='OBSERVED', phase='complete')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=tuple(VERSIONS), required=True)
    ap.add_argument('--capsule-root', type=Path, required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = dict(verdict='RUNNING', phase='prepare', arm=args.arm,
                  performance_acceptance=False, full_gpu_acceptance=False,
                  capsule_manifest_sha256=args.manifest_sha256,
                  scope='Torch context then binding identity/count/version only; no MoE or CuTe')
    try:
        observe(result, args.arm, args.capsule_root, args.manifest_sha256)
    except BaseException as exc:
        result.update(verdict='FAIL', error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
