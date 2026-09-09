"""Bind full MoE compiler/GPU processes to the proven 13.0.3 capsule.

No CUDA APIs, subprocesses or Torch imports. Host validation checks immutable
receipt contents; runtime verification also inspects actual imported files.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import sys

if __package__:
    from .glm53_ep_bindings_capsule import validate_capsule
    from .glm53_ep_bindings_pair_check import verify_import_identity
else:
    from glm53_ep_bindings_capsule import validate_capsule
    from glm53_ep_bindings_pair_check import verify_import_identity

CAPSULE_MOUNT = '/opt/glm53-bindings-capsule'
CAPSULE_SHA256 = 'b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab'
SITE = '/usr/local/lib/python3.12/dist-packages'
PATHFINDER_FILE = SITE+'/cuda/pathfinder/__init__.py'
PATHFINDER_METADATA = SITE+'/cuda_pathfinder-1.7.0.dist-info/METADATA'


def expected_runtime_receipt():
    """Fresh exact value pinned by CPU2 imports and the GPU v6 candidate."""
    modules = {
        'cuda.bindings.driver': ('cuda/bindings/driver.cpython-312-aarch64-linux-gnu.so',
                                'dd32428c39930c2ceffe855ab06ac645e158eba1af9b6f665fa676fa0c71e89e'),
        'cuda.bindings._bindings.cydriver': ('cuda/bindings/_bindings/cydriver.cpython-312-aarch64-linux-gnu.so',
                                           '28a0267e633d526f47acf4920f7b7942e696d5eb6879d22e4f969ba55a6d292d'),
    }
    distributions = {
        'cuda-bindings': ('cuda_bindings-13.0.3.dist-info/METADATA',
                          '5659a955aa1bc509a7c06b939502271fe49acaf4f79adb58c0f73b3f0791a235'),
        'cuda-python': ('cuda_python-13.0.3.dist-info/METADATA',
                        'c4b239363756205466e7d427234a9b17df68034823a0e5758794ca6a3e6fea99'),
    }
    return dict(schema=1, capsule_manifest_sha256=CAPSULE_SHA256,
                binding_identity=dict(version='13.0.3', root=CAPSULE_MOUNT,
                    modules={name: dict(path=CAPSULE_MOUNT+'/'+path, sha256=digest)
                             for name, (path, digest) in modules.items()},
                    distributions={name: dict(version='13.0.3', path=CAPSULE_MOUNT+'/'+path, sha256=digest)
                                   for name, (path, digest) in distributions.items()}),
                pathfinder=dict(version='1.7.0', path=PATHFINDER_FILE,
                    sha256='4ea26c3f6f3bb3c7d2dc725beabfd9cb375e11142ff9a477290eb9b226fd9543',
                    metadata_path=PATHFINDER_METADATA,
                    metadata_sha256='b2d9b84b42ce29eff98b781efdccfd9cb75862d61cb6cab57fc37c6f52cc2548'))


def validate_runtime_receipt(receipt):
    if type(receipt) is not dict or receipt != expected_runtime_receipt():
        raise ValueError('missing or mismatched pinned capsule runtime identity')
    return receipt


def validate_capsule_input(path, manifest_sha256):
    if manifest_sha256 != CAPSULE_SHA256:
        raise ValueError('full MoE requires the proven capsule manifest SHA256')
    path = Path(path)
    if any(char in str(path) for char in (',', '\n', '\r', '\0')):
        raise ValueError('capsule mount path contains a separator or control character')
    resolved = path.resolve(strict=True)
    if any(char in str(resolved) for char in (',', '\n', '\r', '\0')):
        raise ValueError('resolved capsule path contains a mount separator')
    validate_capsule(resolved, manifest_sha256)
    return resolved


def docker_capsule_args(path, manifest_sha256):
    resolved = validate_capsule_input(path, manifest_sha256)
    return ['--mount', f'type=bind,source={resolved},target={CAPSULE_MOUNT},readonly',
            '-e', 'PYTHONPATH='+CAPSULE_MOUNT, '-e', 'PYTHONNOUSERSITE=1',
            '-e', 'PYTHONDONTWRITEBYTECODE=1']


def pathfinder_identity():
    module = importlib.import_module('cuda.pathfinder')
    actual = Path(module.__file__)
    distribution = metadata.distribution('cuda-pathfinder')
    meta = Path(distribution.locate_file('cuda_pathfinder-1.7.0.dist-info/METADATA'))
    if (str(actual) != PATHFINDER_FILE or actual.resolve(strict=True) != actual
            or distribution.version != '1.7.0' or str(meta) != PATHFINDER_METADATA
            or meta.resolve(strict=True) != meta):
        raise RuntimeError('base pathfinder import or metadata origin/version changed')
    text = distribution.read_text('METADATA')
    digest = hashlib.sha256(meta.read_bytes()).hexdigest()
    if text is None or hashlib.sha256(text.encode()).hexdigest() != digest:
        raise RuntimeError('selected pathfinder metadata differs from its file')
    return dict(version=distribution.version, path=str(actual),
                sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
                metadata_path=str(meta), metadata_sha256=digest)


def verify_runtime(path, manifest_sha256):
    resolved = validate_capsule_input(path, manifest_sha256)
    if str(resolved) != CAPSULE_MOUNT:
        raise RuntimeError('full MoE runtime requires the fixed read-only capsule mount')
    if (not sys.dont_write_bytecode or os.environ.get('PYTHONDONTWRITEBYTECODE') != '1'
            or os.environ.get('PYTHONNOUSERSITE') != '1'
            or os.environ.get('PYTHONPATH') != CAPSULE_MOUNT):
        raise RuntimeError('full MoE capsule requires -B and exact isolated Python environment')
    observed = dict(schema=1, capsule_manifest_sha256=manifest_sha256,
                    binding_identity=verify_import_identity('candidate', resolved, manifest_sha256),
                    pathfinder=pathfinder_identity())
    return validate_runtime_receipt(observed)
