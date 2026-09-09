"""Pinned, isolated wheel files for one CUDA-binding diagnostic experiment.

No download, package installation, CUDA import, or process launch occurs here.
The capsule root is a site directory containing only the complete two pinned
wheels. A caller supplies its manifest SHA from an earlier trusted CPU stage;
the manifest must never authorize itself. Runtime/image proof remains separate.
"""
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


MANIFEST_NAME = "capsule-manifest.json"
PINNED_WHEELS = {
    "cuda-bindings": dict(name="cuda-bindings", version="13.0.3",
        filename="cuda_bindings-13.0.3-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl",
        url="https://files.pythonhosted.org/packages/61/3c/c33fd3aa5fcc89aa1c135e477a0561f29142ab5fe028ca425fc87f7f0a74/cuda_bindings-13.0.3-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl",
        sha256="b899e5a513c11eaa18648f9bf5265d8de2a93f76ef66a6bfca0a2887303965cd",
        metadata_sha256="5659a955aa1bc509a7c06b939502271fe49acaf4f79adb58c0f73b3f0791a235"),
    "cuda-python": dict(name="cuda-python", version="13.0.3",
        filename="cuda_python-13.0.3-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/31/5f/beaa12a11b051027eec0b041df01c6690db4f02e3b2e8fadd5a0eeb4df52/cuda_python-13.0.3-py3-none-any.whl",
        sha256="914cd7e2dd075bd06a2d5121c1d9ccdd3d0c94b03ea5a44dbd98d24d8ed93bab",
        metadata_sha256="c4b239363756205466e7d427234a9b17df68034823a0e5758794ca6a3e6fea99"),
}
_DIST_INFO = {"cuda-bindings": "cuda_bindings-13.0.3.dist-info", "cuda-python": "cuda_python-13.0.3.dist-info"}
_SHA = re.compile("[a-f0-9]{64}")
_MAX_EXPANDED_BYTES = 256 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def _path(value, *, directory=False):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("invalid wheel path")
    value = value[:-1] if directory and value.endswith("/") else value
    if value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("unsafe wheel path")
    return value


def _allowed(path, package, *, directory=False):
    roots = [_DIST_INFO[package]] + (["cuda/bindings"] if package == "cuda-bindings" else [])
    return any(path.startswith(root + "/") or directory and (path == root or root.startswith(path + "/")) for root in roots)


def _metadata(files, package):
    prefix = _DIST_INFO[package]
    path = prefix + "/METADATA"
    raw = files[path]
    pin = PINNED_WHEELS[package]
    if _sha(raw) != pin["metadata_sha256"]:
        raise ValueError("metadata hash mismatch: " + package)
    parsed = BytesParser().parsebytes(raw)
    if parsed.get_all("Name") != [package] or parsed.get_all("Version") != [pin["version"]]:
        raise ValueError("metadata distribution identity mismatch")
    licenses = parsed.get_all("License-File", [])
    if not licenses:
        raise ValueError("wheel license metadata missing")
    for license_name in licenses:
        relative = _path(license_name)
        if not any(prefix + middle + relative in files for middle in ("/", "/licenses/")):
            raise ValueError("complete wheel license file missing")
    return dict(name=package, version=pin["version"], requires_dist=parsed.get_all("Requires-Dist", []),
                metadata_path=path, metadata_sha256=_sha(raw))


def _record(files, package):
    record = _DIST_INFO[package] + "/RECORD"
    required = {_DIST_INFO[package] + "/" + name for name in ("METADATA", "WHEEL", "RECORD")}
    if package == "cuda-bindings":
        required |= {"cuda/bindings/__init__.py", "cuda/bindings/driver.cpython-312-aarch64-linux-gnu.so",
                     "cuda/bindings/_bindings/cydriver.cpython-312-aarch64-linux-gnu.so"}
    if not required <= files.keys():
        raise ValueError("required complete wheel files missing")
    seen = set()
    for row in csv.reader(io.StringIO(files[record].decode("utf-8"))):
        if len(row) != 3:
            raise ValueError("invalid wheel RECORD row")
        name, digest, size = row
        _path(name)
        if name in seen or name not in files:
            raise ValueError("duplicate or unknown wheel RECORD file")
        seen.add(name)
        if name == record:
            if digest or size:
                raise ValueError("wheel RECORD must leave its own digest empty")
        else:
            expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).decode().rstrip("=")
            if digest != expected or size != str(len(files[name])):
                raise ValueError("wheel RECORD content mismatch")
    if seen != files.keys():
        raise ValueError("wheel files missing from RECORD")


def _wheel(path, package):
    path = Path(path)
    pin = PINNED_WHEELS[package]
    if path.name != pin["filename"] or path.is_symlink() or not path.is_file():
        raise ValueError("exact pinned regular wheel file required")
    raw = path.read_bytes()
    if _sha(raw) != pin["sha256"]:
        raise ValueError("official wheel hash mismatch: " + package)
    files, seen, expanded = {}, set(), 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            directory = info.is_dir()
            relative = _path(info.filename, directory=directory)
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind not in (0, stat.S_IFDIR if directory else stat.S_IFREG):
                raise ValueError("non-regular wheel entry")
            if relative in seen or not _allowed(relative, package, directory=directory):
                raise ValueError("duplicate or disallowed wheel path")
            seen.add(relative)
            if info.flag_bits & 1:
                raise ValueError("encrypted wheel member")
            expanded += info.file_size
            if expanded > _MAX_EXPANDED_BYTES:
                raise ValueError("wheel expanded size exceeds diagnostic limit")
            if not directory:
                files[relative] = archive.read(info)
    _record(files, package)
    return files, _metadata(files, package)


def stage_capsule(wheel_paths, destination):
    """Validate both pinned wheel byte streams, then write a new site directory.

    wheel_paths is an iterable containing exactly the two downloaded files.
    A failed/partial destination is never reused or accepted without validation.
    """
    paths = [Path(path) for path in wheel_paths]
    expected = {pin["filename"]: name for name, pin in PINNED_WHEELS.items()}
    if len(paths) != 2 or {path.name for path in paths} != set(expected):
        raise ValueError("exactly both pinned wheels required")
    destination = Path(destination)
    if os.path.lexists(destination):
        raise ValueError("capsule destination already exists")
    files, distributions = {}, []
    for path in sorted(paths, key=lambda path: path.name):
        selected, metadata = _wheel(path, expected[path.name])
        if files.keys() & selected.keys():
            raise ValueError("wheel file collision")
        files.update(selected)
        distributions.append(metadata)
    directories = sorted({str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent) != "."})
    manifest = dict(schema=1, diagnostic_only=True, wheels=[PINNED_WHEELS[name] for name in sorted(PINNED_WHEELS)],
                    distributions=distributions, directories=directories,
                    files={name: dict(sha256=_sha(raw), size=len(raw)) for name, raw in sorted(files.items())})
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for directory in directories:
        (destination / directory).mkdir(exist_ok=True)
    for name, raw in files.items():
        with (destination / name).open("xb") as output:
            output.write(raw)
    (destination / MANIFEST_NAME).write_bytes(encoded)
    digest = _sha(encoded)
    validate_capsule(destination, digest)
    return dict(manifest_sha256=digest, manifest=manifest)


def validate_capsule(root, expected_manifest_sha256):
    """Verify an externally pinned manifest and every site entry, without imports."""
    root = Path(root)
    if not isinstance(expected_manifest_sha256, str) or _SHA.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("external manifest SHA256 required")
    manifest_path = root / MANIFEST_NAME
    if (root.is_symlink() or not root.is_dir() or not manifest_path.exists()
            or not stat.S_ISREG(manifest_path.lstat().st_mode)):
        raise ValueError("capsule must be a regular directory with a regular manifest")
    encoded = manifest_path.read_bytes()
    if _sha(encoded) != expected_manifest_sha256:
        raise ValueError("external capsule manifest hash mismatch")
    manifest = _json(encoded)
    if (set(manifest) != {"schema", "diagnostic_only", "wheels", "distributions", "directories", "files"}
            or manifest["schema"] != 1 or manifest["diagnostic_only"] is not True
            or manifest["wheels"] != [PINNED_WHEELS[name] for name in sorted(PINNED_WHEELS)]):
        raise ValueError("invalid pinned capsule manifest")
    files, directories = {}, set()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("capsule contains a non-regular entry")
            if stat.S_ISDIR(mode):
                directories.add(relative)
            elif relative != MANIFEST_NAME:
                files[relative] = path.read_bytes()
    if directories != set(manifest["directories"]) or files.keys() != manifest["files"].keys():
        raise ValueError("extra or missing capsule entries")
    for name, raw in files.items():
        _path(name)
        if manifest["files"][name] != dict(sha256=_sha(raw), size=len(raw)):
            raise ValueError("capsule file changed: " + name)
    distributions = []
    for package in sorted(PINNED_WHEELS):
        selected = {name: raw for name, raw in files.items() if _allowed(name, package)}
        _record(selected, package)
        distributions.append(_metadata(selected, package))
    if len(files) != sum(sum(_allowed(name, package) for name in files) for package in PINNED_WHEELS):
        raise ValueError("capsule package roots are not allowlisted")
    if manifest["distributions"] != distributions:
        raise ValueError("capsule distribution metadata changed")
    return manifest


def check_dependencies(base_distributions, selected_metadata, *, marker_environment):
    """Compare supplied distribution metadata before/after the exact two overrides.

    No resolver installs anything. Requirements are evaluated with extra=''.
    A complete supplied inventory is needed for a complete report. Existing
    unrelated conflicts remain visible; only new or selected-package conflicts
    fail the capsule compatibility check. This is metadata, not import proof.
    """
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    required_environment = set(default_environment())
    if (not isinstance(marker_environment, dict) or not required_environment <= marker_environment.keys()
            or any(not isinstance(marker_environment[key], str) for key in required_environment)
            or marker_environment.get("extra", "") != ""):
        raise ValueError("explicit complete target marker environment with empty extras required")
    environment = {**marker_environment, "extra": ""}

    def inventory(records):
        result = {}
        for record in records:
            name = canonicalize_name(record["name"])
            Version(record["version"])
            if name in result or not isinstance(record["requires_dist"], list):
                raise ValueError("duplicate or invalid supplied distribution")
            result[name] = record
        return result
    base, selected = inventory(base_distributions), inventory(selected_metadata)
    if set(selected) != set(PINNED_WHEELS) or not set(selected) <= base.keys():
        raise ValueError("exactly both selected and original CUDA distributions required")
    if any(selected[name]["version"] != PINNED_WHEELS[name]["version"] for name in selected):
        raise ValueError("selected package version differs from capsule")

    def conflicts(distributions):
        issues, checked = [], []
        for owner, record in sorted(distributions.items()):
            for text in record["requires_dist"]:
                requirement = Requirement(text)
                if requirement.marker and not requirement.marker.evaluate(environment):
                    continue
                dependency = canonicalize_name(requirement.name)
                target = distributions.get(dependency)
                reason = None
                if requirement.extras and dependency in selected:
                    reason = "extras on selected packages are not admitted"
                elif requirement.url:
                    reason = "direct URL identity requires separate evidence"
                elif target is None:
                    reason = "dependency absent from supplied inventory"
                elif not requirement.specifier.contains(target["version"], prereleases=True):
                    reason = "installed version does not satisfy requirement"
                item = dict(owner=owner, requirement=text, dependency=dependency,
                            installed=None if target is None else target["version"])
                checked.append(item)
                if reason:
                    issues.append(dict(item, reason=reason))
        return issues, checked
    baseline, _ = conflicts(base)
    candidate, checked = conflicts({**base, **selected})
    baseline_keys = {(item["owner"], item["requirement"], item["reason"]) for item in baseline}
    introduced = [item for item in candidate if (item["owner"], item["requirement"], item["reason"]) not in baseline_keys]
    selected_conflicts = [item for item in candidate if item["owner"] in selected or item["dependency"] in selected]
    return dict(schema=1, scope="supplied distribution metadata; no import/runtime proof", extra="",
                compatible=not introduced and not selected_conflicts, baseline_conflicts=baseline,
                candidate_conflicts=candidate, introduced_conflicts=introduced,
                selected_package_conflicts=selected_conflicts, checked_requirements=checked)
