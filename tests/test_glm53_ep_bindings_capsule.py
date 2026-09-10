"""CPU fixtures: no downloaded wheel execution, CUDA imports or installation."""
import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import glm53_ep_bindings_capsule as m


def wheel(directory, package, *, extra=None, missing=None, bad_record=False, bad_name=False, symlink=None, duplicate=None):
    """Small signed-by-test-fixture wheel; production pins are never changed."""
    prefix = m._DIST_INFO[package]
    requires = (["cuda-pathfinder~=1.1"] if package == "cuda-bindings" else
                ["cuda-bindings~=13.0.3", "cuda-pathfinder~=1.1", 'cuda-bindings[all]~=13.0.3; extra == "all"'])
    metadata = ("Metadata-Version: 2.4\nName: " + ("wrong-package" if bad_name else package) +
                "\nVersion: 13.0.3\nLicense-File: LICENSE\n" +
                "".join("Requires-Dist: " + value + "\n" for value in requires) + "\nFixture only.\n").encode()
    files = {prefix + "/METADATA": metadata, prefix + "/WHEEL": b"Wheel-Version: 1.0\n",
             prefix + "/licenses/LICENSE": b"Complete fixture license\n"}
    if package == "cuda-bindings":
        files.update({"cuda/bindings/__init__.py": b"raise RuntimeError('must never import fixture')\n",
                      "cuda/bindings/driver.cpython-312-aarch64-linux-gnu.so": b"driver-fixture-not-executable",
                      "cuda/bindings/_bindings/cydriver.cpython-312-aarch64-linux-gnu.so": b"cydriver-fixture-not-executable"})
    files.update(extra or {})
    if missing:
        del files[missing]
    record_path = prefix + "/RECORD"
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for path, raw in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        writer.writerow([path, "sha256=" + ("bad" if bad_record else digest), str(len(raw))])
    writer.writerow([record_path, "", ""])
    files[record_path] = buf.getvalue().encode()
    path = directory / m.PINNED_WHEELS[package]["filename"]
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = ((stat.S_IFLNK if name == symlink else stat.S_IFREG) | 0o644) << 16
            archive.writestr(info, raw)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(duplicate, files[duplicate])
    pin = dict(m.PINNED_WHEELS[package], sha256=m._sha(path.read_bytes()), metadata_sha256=m._sha(metadata))
    return path, pin


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def stage(self, **binding_options):
        paths, pins = [], {}
        for package in m.PINNED_WHEELS:
            path, pin = wheel(self.root, package, **(binding_options if package == "cuda-bindings" else {}))
            paths.append(path)
            pins[package] = pin
        with patch.object(m, "PINNED_WHEELS", pins):
            receipt = m.stage_capsule(paths, self.root / "site")
        return receipt, pins

    def validate(self, receipt, pins):
        with patch.object(m, "PINNED_WHEELS", pins):
            return m.validate_capsule(self.root / "site", receipt["manifest_sha256"])

    def test_complete_payload_and_licenses_validate_without_importing_cuda(self):
        imported = {name for name in sys.modules if name == "torch" or name.startswith("cuda")}
        receipt, pins = self.stage()
        manifest = self.validate(receipt, pins)
        self.assertEqual(manifest, receipt["manifest"])
        self.assertEqual({record["name"]: record["version"] for record in manifest["distributions"]},
                         {"cuda-bindings": "13.0.3", "cuda-python": "13.0.3"})
        self.assertTrue(all(m._DIST_INFO[package] + "/licenses/LICENSE" in manifest["files"] for package in pins))
        self.assertFalse((self.root / "site/cuda/__init__.py").exists())
        self.assertEqual(imported, {name for name in sys.modules if name == "torch" or name.startswith("cuda")})

    def test_manifest_requires_an_external_digest_and_rejects_rehashed_mutation(self):
        receipt, pins = self.stage()
        manifest_path = self.root / "site" / m.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        name = "cuda/bindings/__init__.py"
        changed = b"modified code"
        (self.root / "site" / name).write_bytes(changed)
        manifest["files"][name] = dict(sha256=m._sha(changed), size=len(changed))
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "external capsule"):
            self.validate(receipt, pins)
        with patch.object(m, "PINNED_WHEELS", pins), self.assertRaisesRegex(ValueError, "RECORD"):
            m.validate_capsule(self.root / "site", m._sha(manifest_path.read_bytes()))
        with self.assertRaisesRegex(ValueError, "external manifest"):
            m.validate_capsule(self.root / "site", None)

    def test_extra_missing_mutated_files_and_empty_directories_are_rejected(self):
        for change in ("extra", "missing", "mutated", "directory"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                previous = self.root
                self.root = Path(directory)
                receipt, pins = self.stage()
                target = self.root / "site/cuda/bindings/__init__.py"
                if change == "extra":
                    (target.parent / "injected.py").write_text("injected")
                elif change == "missing":
                    target.unlink()
                elif change == "mutated":
                    target.write_text("mutated")
                else:
                    (target.parent / "empty").mkdir()
                with self.assertRaises(ValueError):
                    self.validate(receipt, pins)
                self.root = previous

    def test_symlinked_payload_directory_and_manifest_are_rejected(self):
        for change in ("file", "directory", "manifest"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                previous = self.root
                self.root = Path(directory)
                receipt, pins = self.stage()
                if change == "directory":
                    target = self.root / "site/cuda/bindings/_bindings"
                else:
                    target = self.root / "site" / (m.MANIFEST_NAME if change == "manifest" else "cuda/bindings/__init__.py")
                outside = self.root / "outside"
                target.rename(outside)
                target.symlink_to(outside, target_is_directory=change == "directory")
                with self.assertRaises(ValueError):
                    self.validate(receipt, pins)
                self.root = previous

    def test_archive_traversal_extra_package_and_symlink_never_create_site(self):
        for name in ("../escape", "/absolute", "cuda/bindings/../escape", "cuda\\bindings\\bad", "cuda/__init__.py", "cuda/pathfinder/__init__.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.stage(extra={name: b"unsafe"})
            self.assertFalse((self.root / "site").exists())
        with self.assertRaisesRegex(ValueError, "non-regular"):
            self.stage(symlink="cuda/bindings/__init__.py")
        self.assertFalse((self.root / "site").exists())

    def test_complete_record_metadata_license_and_unique_members_are_required(self):
        cases = (dict(bad_record=True), dict(bad_name=True),
                 dict(missing="cuda_bindings-13.0.3.dist-info/licenses/LICENSE"),
                 dict(missing="cuda/bindings/_bindings/cydriver.cpython-312-aarch64-linux-gnu.so"),
                 dict(duplicate="cuda/bindings/__init__.py"))
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.stage(**options)
            self.assertFalse((self.root / "site").exists())

    def test_unpinned_wheel_bytes_one_package_and_existing_destination_are_refused(self):
        paths = [wheel(self.root, package)[0] for package in m.PINNED_WHEELS]
        with self.assertRaisesRegex(ValueError, "official wheel"):
            m.stage_capsule(paths, self.root / "site")
        with self.assertRaisesRegex(ValueError, "both pinned"):
            m.stage_capsule(paths[:1], self.root / "site")
        self.stage()
        with self.assertRaisesRegex(ValueError, "already exists"):
            m.stage_capsule(paths, self.root / "site")


@unittest.skipUnless(importlib.util.find_spec("packaging"), "dependency metadata checks require packaging; staging does not")
class DependencyTests(unittest.TestCase):
    def setUp(self):
        from packaging.markers import default_environment
        self.environment = default_environment()
        self.environment.update(python_version="3.12", python_full_version="3.12.3", sys_platform="linux",
                                os_name="posix", platform_system="Linux", platform_machine="aarch64")
        self.base = [
            dict(name="cuda-bindings", version="13.3.1", requires_dist=["cuda-pathfinder>=1.4.2"]),
            dict(name="cuda-python", version="13.3.1", requires_dist=["cuda-bindings~=13.3.1", "cuda-core~=1.0.0"]),
            dict(name="cuda-pathfinder", version="1.7.0", requires_dist=[]),
            dict(name="cuda-core", version="1.0.1", requires_dist=["cuda-bindings>=12.0"]),
            dict(name="torch", version="2.13.0+cu130", requires_dist=['cuda-bindings>=13.0.3,<14; sys_platform == "linux"']),
            dict(name="nvidia-cutlass-dsl", version="4.6.2", requires_dist=["cuda-python>=12.8"]),
            dict(name="flashinfer-python", version="0.6.18", requires_dist=["cuda-python>=12.0", "nvidia-cutlass-dsl==4.7.0"]),
        ]
        self.selected = [
            dict(name="cuda-bindings", version="13.0.3", requires_dist=["cuda-pathfinder~=1.1", 'cuda-toolkit==13.*; extra == "all"']),
            dict(name="cuda-python", version="13.0.3", requires_dist=["cuda-bindings~=13.0.3", "cuda-pathfinder~=1.1",
                                                                     'cuda-bindings[all]~=13.0.3; extra == "all"']),
        ]

    def check(self):
        return m.check_dependencies(self.base, self.selected, marker_environment=self.environment)

    def test_matching_pair_satisfies_reverse_constraints_and_retains_unrelated_baseline_conflict(self):
        result = self.check()
        self.assertTrue(result["compatible"])
        self.assertEqual(result["introduced_conflicts"], [])
        self.assertEqual(result["selected_package_conflicts"], [])
        self.assertEqual(len(result["candidate_conflicts"]), 1)
        self.assertEqual(result["candidate_conflicts"][0]["dependency"], "nvidia-cutlass-dsl")
        self.assertFalse(any(row["dependency"] == "cuda-toolkit" for row in result["checked_requirements"]))

    def test_binding_only_override_is_not_accepted_as_the_matched_pair(self):
        with self.assertRaisesRegex(ValueError, "both selected"):
            m.check_dependencies(self.base, self.selected[:1], marker_environment=self.environment)
        self.selected[1]["version"] = "13.3.1"
        with self.assertRaisesRegex(ValueError, "selected package version"):
            self.check()

    def test_reverse_dependency_pathfinder_and_missing_target_fail(self):
        for change in ("reverse", "pathfinder", "missing"):
            old = copy.deepcopy(self.base)
            if change == "reverse":
                self.base[-1]["requires_dist"].append("cuda-bindings>=13.2")
            elif change == "pathfinder":
                self.base[2]["version"] = "2.0"
            else:
                self.base = [row for row in self.base if row["name"] != "cuda-pathfinder"]
            with self.subTest(change=change):
                result = self.check()
                self.assertFalse(result["compatible"])
                self.assertTrue(result["selected_package_conflicts"])
            self.base = old

    def test_target_markers_are_explicit_and_active_extras_cannot_hide_constraints(self):
        with self.assertRaisesRegex(ValueError, "target marker environment"):
            m.check_dependencies(self.base, self.selected, marker_environment={"sys_platform": "linux"})
        with self.assertRaisesRegex(ValueError, "empty extras"):
            m.check_dependencies(self.base, self.selected, marker_environment={**self.environment, "extra": "all"})
        self.base[-1]["requires_dist"].append('cuda-bindings>=99; sys_platform == "win32"')
        self.assertTrue(self.check()["compatible"])
        self.base[-1]["requires_dist"].append("cuda-bindings[all]>=13")
        self.assertFalse(self.check()["compatible"])

    def test_duplicate_distribution_and_direct_url_identity_fail_closed(self):
        self.base.append(dict(name="CUDA_Bindings", version="13.3.1", requires_dist=[]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check()
        self.base.pop()
        self.base[-1]["requires_dist"].append("cuda-bindings @ https://example.invalid/unverified.whl")
        self.assertFalse(self.check()["compatible"])


if __name__ == "__main__":
    unittest.main()
