"""Mock-only contracts: no Docker, CUDA imports, wheel staging, or GPU calls."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import glm53_ep_bindings_capsule_check as check
import run_glm53_ep_bindings_capsule_cpu as runner


def installed_distribution(root, directory, name, version, *, legacy=False):
    path = Path(root) / directory
    path.mkdir()
    (path / ("PKG-INFO" if legacy else "METADATA")).write_text(
        "Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version
        + "\nRequires-Dist: other>=1\n\nFull metadata body for " + directory + "\n")
    return check.metadata.PathDistribution(path)


class WrapperTests(unittest.TestCase):
    def test_fixed_no_device_command_and_distinct_mounts(self):
        args = runner.command(Path("/source"), Path("/wheels"), Path("/output"), "owned")
        for value in ("--runtime=runc", "--network=none", "--pull=never", "--memory=4g",
                      "--memory-swap=4g", "--cpus=2", "--pids-limit=128", "NVIDIA_VISIBLE_DEVICES=void",
                      "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "-B", check.IMAGE):
            self.assertIn(value, args)
        self.assertEqual(args[-3:], [check.IMAGE, "-B", "/repo/probes/glm53_ep_bindings_capsule_check.py"])
        mounts = [args[index + 1] for index, value in enumerate(args) if value == "--mount"]
        self.assertEqual(mounts, ["type=bind,source=/source,target=/repo,readonly",
                                  "type=bind,source=/wheels,target=/wheels,readonly",
                                  "type=bind,source=/output,target=/evidence"])
        self.assertFalse(any(arg.startswith(("--gpus", "--device", "--privileged")) for arg in args))
        for output in ("/source/new", "/wheels/new", "/", "/output,readonly"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                runner.command(Path("/source"), Path("/wheels"), Path(output), "owned")

    def test_unchanged_twelve_gib_guard_precedes_all_processes_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "capture"
            with patch.object(runner, "host_available_kib", return_value=12 * 1024 * 1024 - 1), \
                    patch.object(runner.subprocess, "run") as launch:
                with self.assertRaisesRegex(RuntimeError, "12 GiB"):
                    runner.run(Path(folder) / "absent-wheels", output)
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_no_arbitrary_image_or_command_cli_is_accepted(self):
        for extra in (("--image", "sha256:" + "b" * 64), ("--", "bash")):
            with patch.object(sys, "argv", ["check", "--wheels", "/wheels", "--output", "/output", *extra]), \
                    patch.object(runner, "run") as launch, redirect_stdout(io.StringIO()), \
                    patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main()
            launch.assert_not_called()

    def test_outer_failure_and_timeout_never_create_success_receipt(self):
        for outcome in (subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0),
                        subprocess.TimeoutExpired("docker run", 180)):
            with self.subTest(outcome=type(outcome).__name__), tempfile.TemporaryDirectory() as folder:
                wheels, output = Path(folder) / "wheels", Path(folder) / "capture"
                wheels.mkdir()
                (wheels / "fixture.whl").write_bytes(b"fixture")
                pins = {"fixture": dict(filename="fixture.whl", sha256=hashlib.sha256(b"fixture").hexdigest())}
                side_effect = ([outcome, subprocess.CompletedProcess([], 0)]
                               if isinstance(outcome, Exception) else [outcome])
                with patch.object(runner, "host_available_kib", return_value=12 * 1024 * 1024), \
                        patch.object(runner, "PINNED_WHEELS", pins), \
                        patch.object(runner.subprocess, "run", side_effect=side_effect) as launch:
                    with self.assertRaises((RuntimeError, FileNotFoundError, subprocess.TimeoutExpired)):
                        runner.run(wheels, output)
                receipt = json.loads((output / "receipt.json").read_text())
                self.assertEqual(receipt["verdict"], "FAIL")
                self.assertNotIn("capsule_manifest_sha256", receipt)
                argv = launch.call_args_list[0].args[0]
                if isinstance(outcome, subprocess.TimeoutExpired):
                    self.assertEqual(launch.call_args_list[1].args[0],
                                     ["docker", "rm", "-f", argv[argv.index("--name") + 1]])
                else:
                    self.assertEqual(launch.call_count, 1)


class InnerTests(unittest.TestCase):
    def test_device_guard_and_preimport_checks_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            dev = Path(folder)
            (dev / "nvidiactl").touch()
            (dev / "dri").mkdir()
            (dev / "dri/renderD128").touch()
            self.assertEqual(check.device_nodes(dev), sorted([str(dev / "nvidiactl"), str(dev / "dri/renderD128")]))
        with patch.object(check, "device_nodes", return_value=["/dev/nvidia0"]):
            with self.assertRaisesRegex(RuntimeError, "exposes"):
                check.assert_no_devices()
        with patch.object(check, "accelerator_modules", return_value=["cuda.bindings"]), \
                patch.object(check.importlib, "import_module") as imports:
            with self.assertRaisesRegex(RuntimeError, "before selecting"):
                check.import_capsule(Path("/capsule"), {}, {})
        imports.assert_not_called()
        with patch.object(check, "accelerator_modules", return_value=[]), \
                patch.object(sys, "dont_write_bytecode", False), \
                patch.object(check.importlib, "import_module") as imports:
            with self.assertRaisesRegex(RuntimeError, "Python -B"):
                check.import_capsule(Path("/capsule"), {}, {})
        imports.assert_not_called()

    def test_every_installed_metadata_record_is_kept_including_legacy_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            distributions = []
            for name, info_name, file_name in (("Cuda-Bindings", "bindings.dist-info", "METADATA"),
                                                ("unrelated", "other.egg-info", "PKG-INFO")):
                info = Path(folder) / info_name
                info.mkdir()
                raw = "Name: " + name + "\nVersion: 1.0\nRequires-Dist: other>=1\n\nEntire body retained.\n"
                (info / file_name).write_text(raw)
                distributions.append(SimpleNamespace(_path=info, metadata={"Name": name},
                                                       version="1.0", requires=["other>=1"]))
            with patch.object(check.metadata, "distributions", return_value=distributions):
                records = check.snapshot_distributions()
            self.assertEqual([record["name"] for record in records], ["cuda-bindings", "unrelated"])
            for record in records:
                self.assertIn("Entire body retained.", record["metadata_text"])
                self.assertEqual(record["metadata_sha256"], check.file_hash(record["metadata_path"]))
                self.assertEqual(record["requires_dist"], ["other>=1"])

    def test_runtime_resolution_preserves_all_shadowed_metadata_without_version_or_order_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            cryptography = [installed_distribution(folder, directory, "cryptography", version, legacy=legacy)
                            for directory, version, legacy in (
                                ("a-crypto.dist-info", "50.0.0", False),
                                ("b-crypto.egg-info", "41.0.7", True),
                                ("c-crypto.dist-info", "41.0.7", False))]
            six = installed_distribution(folder, "six.dist-info", "six", "1.17.0")
            distributions = [*cryptography, six]
            with patch.object(check.metadata, "distributions", return_value=distributions):
                records = check.snapshot_distributions()
            before = copy.deepcopy(records)
            selected = {"cryptography": cryptography[1], "six": six}
            with patch.object(check.metadata, "distribution", side_effect=selected.__getitem__) as lookup:
                result = check.resolve_distributions(records)
            self.assertEqual([call.args[0] for call in lookup.call_args_list], ["cryptography", "six"])
            self.assertEqual((result["raw_count"], result["effective_count"], result["shadowed_count"]), (4, 2, 2))
            self.assertEqual(result["effective_distributions"], [records[1], records[3]])
            self.assertEqual(result["shadowed_distributions"], [records[0], records[2]])
            self.assertEqual(result["selections"][0]["snapshot_index"], 1)
            self.assertEqual(result["selections"][0]["shadowed_snapshot_indices"], [0, 2])
            self.assertEqual(records, before)
            self.assertEqual(result["search_path"], sys.path)
            # Reordering the raw inventory must not change runtime selection.
            with patch.object(check.metadata, "distribution", side_effect=selected.__getitem__):
                reordered = check.resolve_distributions(list(reversed(records)))
            self.assertEqual(reordered["effective_distributions"], result["effective_distributions"])
            for record in result["shadowed_distributions"]:
                self.assertIn("Full metadata body", record["metadata_text"])
                self.assertEqual(record["metadata_sha256"], check.file_hash(record["metadata_path"]))

    def test_runtime_selection_requires_exact_saved_path_version_hash_and_dependency_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            selected = installed_distribution(folder, "selected.dist-info", "PyJWT", "2.13.0")
            record = check.distribution_record(selected)
            for field, value in (("metadata_path", str(Path(folder) / "unknown/METADATA")),
                                 ("version", "2.7.0"), ("metadata_sha256", "0" * 64),
                                 ("requires_dist", []), ("metadata_text", "different body")):
                changed = copy.deepcopy(record)
                changed[field] = value
                with self.subTest(field=field), patch.object(check.metadata, "distribution", return_value=selected):
                    with self.assertRaisesRegex(RuntimeError, "snapshot"):
                        check.resolve_distributions([changed])
            # A second saved record at the same resolved path is ambiguous,
            # even when every byte and version is identical.
            with patch.object(check.metadata, "distribution", return_value=selected):
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    check.resolve_distributions([record, copy.deepcopy(record)])
            # Runtime bytes changing after the snapshot cannot be adopted.
            Path(record["metadata_path"]).write_text(record["metadata_text"] + "changed after snapshot\n")
            with patch.object(check.metadata, "distribution", return_value=selected):
                with self.assertRaisesRegex(RuntimeError, "changed from its snapshot"):
                    check.resolve_distributions([record])

    def test_unknown_missing_or_wrong_name_runtime_lookup_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            original = installed_distribution(folder, "known.dist-info", "six", "1.17.0")
            unknown = installed_distribution(folder, "unknown.dist-info", "six", "1.17.0")
            wrong_name = installed_distribution(folder, "wrong-name.dist-info", "other", "1.17.0")
            records = [check.distribution_record(original)]
            for resolved in (unknown, wrong_name):
                with patch.object(check.metadata, "distribution", return_value=resolved):
                    with self.assertRaisesRegex(RuntimeError, "exactly one"):
                        check.resolve_distributions(records)
            with patch.object(check.metadata, "distribution", side_effect=check.metadata.PackageNotFoundError("six")):
                with self.assertRaisesRegex(RuntimeError, "lookup is missing"):
                    check.resolve_distributions(records)

    def test_actual_module_files_must_be_inside_capsule_and_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "capsule"
            root.mkdir()
            path = root / "driver.so"
            path.write_bytes(b"pinned import bytes")
            module = SimpleNamespace(__name__="cuda.bindings.driver", __file__=str(path))
            manifest = {"files": {"driver.so": dict(sha256=check.file_hash(path), size=path.stat().st_size)}}
            self.assertEqual(check.module_identity(module, root, manifest)["relative_path"], "driver.so")
            path.write_bytes(b"changed import bytes")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                check.module_identity(module, root, manifest)
            outside = Path(folder) / "base-driver.so"
            outside.write_bytes(b"pinned import bytes")
            module.__file__ = str(outside)
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                check.module_identity(module, root, manifest)

    def test_base_pathfinder_version_cannot_be_replaced(self):
        with patch.object(check.metadata, "distribution") as lookup:
            for records in ([], [{"name": "cuda-pathfinder", "version": "1.8.0"}]):
                with self.assertRaisesRegex(RuntimeError, "exactly 1.7.0"):
                    check.base_pathfinder_identity(records)
        lookup.assert_not_called()

    def test_selected_import_metadata_version_and_base_pathfinder_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "capsule"
            root.mkdir()
            manifest = dict(distributions=[], files={})
            distributions, modules = {}, {}
            for name in ("cuda-bindings", "cuda-python"):
                info = root / (name + ".dist-info")
                info.mkdir()
                (info / "METADATA").write_text("pinned " + name)
                distributions[name] = SimpleNamespace(_path=info, version="13.0.3")
                manifest["distributions"].append(dict(name=name, version="13.0.3",
                    metadata_path=str((info / "METADATA").relative_to(root)),
                    metadata_sha256=check.file_hash(info / "METADATA")))
            for name in ("cuda.bindings", "cuda.bindings.driver", "cuda.bindings._bindings.cydriver"):
                path = root / (name + ".so")
                path.write_text("pinned " + name)
                modules[name] = SimpleNamespace(__name__=name, __file__=str(path), __version__="13.0.3")
                manifest["files"][path.name] = dict(sha256=check.file_hash(path), size=path.stat().st_size)
            base = Path(folder) / "base"
            base.mkdir()
            (base / "METADATA").write_text("base pathfinder")
            (base / "__init__.py").write_text("base pathfinder import")
            distributions["cuda-pathfinder"] = SimpleNamespace(_path=base, version="1.7.0")
            modules["cuda.pathfinder"] = SimpleNamespace(__file__=str(base / "__init__.py"))
            finder = dict(version="1.7.0", path=str((base / "__init__.py").resolve()),
                sha256=check.file_hash(base / "__init__.py"), metadata_path=str((base / "METADATA").resolve()),
                metadata_sha256=check.file_hash(base / "METADATA"))
            for change in (None, "selected_metadata", "bindings_version", "pathfinder_version"):
                original_path = list(sys.path)
                try:
                    if change == "selected_metadata":
                        distributions["cuda-bindings"].version = "13.3.1"
                    elif change == "bindings_version":
                        modules["cuda.bindings"].__version__ = "13.3.1"
                    elif change == "pathfinder_version":
                        distributions["cuda-pathfinder"].version = "1.8.0"
                    with patch.object(check, "accelerator_modules", return_value=[]), \
                            patch.object(sys, "dont_write_bytecode", True), \
                            patch.object(check.metadata, "distribution", side_effect=distributions.__getitem__), \
                            patch.object(check.importlib, "import_module", side_effect=modules.__getitem__):
                        if change:
                            with self.subTest(change=change), self.assertRaises(RuntimeError):
                                check.import_capsule(root, manifest, finder)
                        else:
                            result = check.import_capsule(root, manifest, finder)
                            self.assertEqual(result["base_pathfinder"], finder)
                            self.assertEqual([item["module"] for item in result["modules"]],
                                ["cuda.bindings", "cuda.bindings.driver", "cuda.bindings._bindings.cydriver"])
                finally:
                    sys.path[:] = original_path
                    distributions["cuda-bindings"].version = "13.0.3"
                    modules["cuda.bindings"].__version__ = "13.0.3"
                    distributions["cuda-pathfinder"].version = "1.7.0"

    def test_full_metadata_and_dependency_decision_precede_import_and_postvalidate(self):
        for compatible in (False, True):
            with self.subTest(compatible=compatible), tempfile.TemporaryDirectory() as folder:
                output, events = Path(folder), []
                base = [dict(name="cuda-bindings", version="13.3.1", requires_dist=[]),
                        dict(name="cuda-bindings", version="13.0.1", requires_dist=[])]
                effective = base[:1]
                manifest = dict(distributions=[dict(name="cuda-bindings", version="13.0.3", requires_dist=[])])
                report = dict(compatible=compatible, baseline_conflicts=[{"owner": "unrelated"}])

                def resolve(actual_base):
                    self.assertEqual(json.loads((output / "base-distributions.json").read_text()), base)
                    self.assertEqual(actual_base, base)
                    events.append("resolve")
                    return dict(raw_count=2, effective_count=1, shadowed_count=1,
                                effective_distributions=effective, shadowed_distributions=base[1:])

                def stage(wheels, destination):
                    self.assertTrue((output / "base-distributions.json").is_file())
                    self.assertTrue((output / "distribution-resolution.json").is_file())
                    self.assertEqual(len(wheels), 2)
                    self.assertEqual(destination, output / "capsule")
                    events.append("stage")
                    return dict(manifest_sha256="a" * 64, manifest=manifest)

                def dependencies(actual_base, selected, *, marker_environment):
                    self.assertEqual(actual_base, effective)
                    self.assertEqual(selected, manifest["distributions"])
                    self.assertIn("python_version", marker_environment)
                    events.append("dependencies")
                    return report

                def validate(*args):
                    events.append("validate")
                    return manifest

                def imports(*args):
                    self.assertEqual(events[-1], "dependencies")
                    events.append("imports")
                    return {"modules": ["mock import identity"]}

                with patch.object(check, "source_hashes", return_value={"check": "pinned"}), \
                        patch.object(check, "assert_no_devices", return_value=[]), \
                        patch.object(check, "runtime_identity", return_value={}), \
                        patch.object(check, "accelerator_modules", return_value=[]), \
                        patch.object(check, "snapshot_distributions", return_value=base), \
                        patch.object(check, "resolve_distributions", side_effect=resolve), \
                        patch.object(check, "base_pathfinder_identity", return_value={"version": "1.7.0"}), \
                        patch.object(check, "stage_capsule", side_effect=stage), \
                        patch.object(check, "validate_capsule", side_effect=validate), \
                        patch.object(check, "check_dependencies", side_effect=dependencies), \
                        patch.object(check, "import_capsule", side_effect=imports) as import_check, \
                        redirect_stdout(io.StringIO()):
                    if compatible:
                        check.run_check(Path("/wheels"), output)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "dependency conflict"):
                            check.run_check(Path("/wheels"), output)
                        import_check.assert_not_called()
                receipt = json.loads((output / "result.json").read_text())
                self.assertEqual(receipt["dependencies"], report)
                self.assertEqual(receipt["base_distributions"]["count"], 2)
                self.assertEqual(receipt["distribution_resolution"]["effective_count"], 1)
                self.assertEqual(receipt["distribution_resolution"]["shadowed_count"], 1)
                self.assertEqual(receipt["verdict"], "PASS" if compatible else "FAIL")
                self.assertEqual(events, ["resolve", "stage", "validate", "dependencies"] +
                                 (["imports", "validate"] if compatible else []))


if __name__ == "__main__":
    unittest.main()
