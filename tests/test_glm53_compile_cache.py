"""Compile cache lifecycle, invalidation safety, and unchanged fleet provenance."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "launchers/lib/glm53-compile-cache.py"
spec = importlib.util.spec_from_file_location("compile_cache", HELPER)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
IMAGE = "sha256:" + "a" * 64


class CompileCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.manifest = self.root / "manifest.tsv"
        self.manifest.write_text("# source_commit=first\na.py\t/vllm/a.py\tabsent\n")
        self.source = self.root / "a.py"
        self.source.write_text("value = 1\n")
        self.artifact = self.cache / "vllm/torch_compile_cache/graph.bin"
        self.docker = patch.object(cc.subprocess, "run", side_effect=self.clear_cache).start()
        self.addCleanup(patch.stopall)

    def clear_cache(self, argv, check):
        self.assertTrue(check)
        self.assertEqual(argv, ["docker", "run", "--rm", "-v", f"{self.cache}:/cache",
                               "--entrypoint", "rm", IMAGE, "-rf", "/cache/vllm/torch_compile_cache"])
        shutil.rmtree(self.artifact.parent, ignore_errors=True)

    def prepare(self):
        return cc.prepare(self.cache, self.manifest, IMAGE)

    def seed(self):
        self.artifact.parent.mkdir(parents=True, exist_ok=True)
        self.artifact.write_bytes(b"compiled artifact")

    def test_metadata_only_deploy_reuses_artifact_and_updates_fleet_sha(self):
        self.assertEqual(self.prepare()["action"], "invalidate")
        self.seed()
        stat = self.artifact.stat()
        original = cc.fingerprints(self.manifest, IMAGE)
        self.manifest.write_text(self.manifest.read_text().replace("first", "second"))
        updated = cc.fingerprints(self.manifest, IMAGE)
        self.assertNotEqual(original[0], updated[0])
        self.assertEqual(original[1], updated[1])
        self.assertEqual(self.prepare()["action"], "reuse")
        self.assertEqual(self.artifact.read_bytes(), b"compiled artifact")
        self.assertEqual(self.artifact.stat().st_mtime_ns, stat.st_mtime_ns)
        self.assertEqual(self.artifact.stat().st_ino, stat.st_ino)
        self.assertEqual((self.cache / ".overlay-sha").read_text(), updated[0])
        self.assertEqual(self.prepare()["action"], "reuse")
        self.assertEqual(self.docker.call_count, 1)

    def test_same_length_runtime_edit_still_invalidates_with_unchanged_manifest(self):
        self.prepare()
        self.seed()
        previous = self.manifest.read_bytes()
        self.source.write_text("value = 2\n")
        self.assertEqual(self.prepare()["action"], "invalidate")
        self.assertFalse(self.artifact.exists())
        self.assertEqual(self.manifest.read_bytes(), previous)

    def test_bindings_base_contract_and_image_identity_invalidate(self):
        original = cc.fingerprints(self.manifest, IMAGE)[1]
        for row in ("a.py\t/vllm/b.py\tabsent\n", "a.py\t/vllm/a.py\t" + "f" * 64 + "\n"):
            self.manifest.write_text(row)
            self.assertNotEqual(cc.fingerprints(self.manifest, IMAGE)[1], original)
        self.manifest.write_text("a.py\t/vllm/a.py\tabsent\n")
        self.assertNotEqual(cc.fingerprints(self.manifest, "sha256:" + "b" * 64)[1], original)

    def test_order_and_comments_do_not_invalidate(self):
        (self.root / "b.py").write_text("value = 3\n")
        rows = ["a.py\t/vllm/a.py\tabsent\n", "b.py\t/vllm/b.py\tabsent\n"]
        self.manifest.write_text("".join(rows))
        original = cc.fingerprints(self.manifest, IMAGE)[1]
        self.manifest.write_text("# deployment\n" + "".join(reversed(rows)))
        self.assertEqual(cc.fingerprints(self.manifest, IMAGE)[1], original)

    def test_old_launcher_or_missing_corrupt_receipt_requires_invalidation(self):
        for replacement in (None, "garbage", "[]", '{"version": 2}'):
            self.prepare()
            receipt = self.cache / ".compile-overlay.json"
            if replacement is None:
                receipt.unlink()
            else:
                receipt.write_text(replacement)
            self.assertEqual(self.prepare()["action"], "invalidate")
        self.seed()
        (self.cache / ".overlay-sha").write_text("old-launcher-deployed-different-code")
        self.assertEqual(self.prepare()["action"], "invalidate")
        self.assertFalse(self.artifact.exists())

    def test_failed_deletion_does_not_publish_reuse_receipt(self):
        self.prepare()
        original = (self.cache / ".compile-overlay.json").read_bytes()
        self.source.write_text("changed\n")
        self.docker.side_effect = subprocess.CalledProcessError(1, "docker")
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()
        self.assertEqual((self.cache / ".compile-overlay.json").read_bytes(), original)
        self.assertEqual(cc.decision(self.cache, self.manifest, IMAGE)["action"], "invalidate")

    def test_interrupted_stamp_update_is_conservative(self):
        self.prepare()
        self.manifest.write_text(self.manifest.read_text().replace("first", "second"))
        write = cc.atomic_write
        def interrupt(path, text):
            if path.name == ".overlay-sha":
                raise OSError("interrupted")
            return write(path, text)
        with patch.object(cc, "atomic_write", side_effect=interrupt), self.assertRaises(OSError):
            self.prepare()
        self.assertEqual(cc.decision(self.cache, self.manifest, IMAGE)["action"], "invalidate")

    def test_missing_source_or_invalid_manifest_cannot_reuse(self):
        self.prepare()
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        for text in ("# empty\n", "a.py\t/vllm/a.py\n", "../a.py\t/vllm/a.py\tabsent\n"):
            self.manifest.write_text(text)
            with self.assertRaises(ValueError):
                self.prepare()

    def test_inspection_does_not_touch_cache_or_invoke_docker(self):
        self.seed()
        result = cc.decision(self.cache, self.manifest, IMAGE)
        self.assertEqual(result["action"], "invalidate")
        self.docker.assert_not_called()
        self.assertEqual(list(self.cache.glob(".*")), [])
        self.assertTrue(self.artifact.exists())


if __name__ == "__main__":
    unittest.main()
