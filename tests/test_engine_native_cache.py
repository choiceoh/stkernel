"""Native inputs must survive checkout churn and invalidate real build changes."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from engine.kernels.common.native_cache import prepare_sources

ROOT = Path(__file__).resolve().parents[1]


class NativeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "checkout one"
        self.checkout.mkdir()
        self.source = self.checkout / "kernel.cu"
        self.header = self.checkout / "constants.h"
        self.source.write_text('#include "constants.h"\nint answer() { return VALUE; }\n')
        self.header.write_text('#define VALUE 41\n')
        self.identity = (["-O2", "sm_121a"], ["-libverbs"], "torch-1", "cuda-13")

    def prepare(self, sources=None, identity=None):
        return prepare_sources(self.root / "cache", sources or (self.source, self.header),
                               self.identity if identity is None else identity)

    def test_same_content_keeps_source_paths_inode_and_mtime_across_checkouts(self):
        first = self.prepare()
        original = [(Path(p).stat().st_ino, Path(p).stat().st_mtime_ns) for p in first[2]]
        later = self.root / "another checkout"
        shutil.copytree(self.checkout, later)
        for p in (self.source, self.header, later / self.source.name, later / self.header.name):
            os.utime(p, ns=(2_000_000_000_000_000_000,) * 2)
        self.assertEqual(self.prepare(), first)
        self.assertEqual(self.prepare([later / p.name for p in (self.source, self.header)]), first)
        self.assertEqual([(Path(p).stat().st_ino, Path(p).stat().st_mtime_ns) for p in first[2]], original)

    def test_source_header_flags_and_runtime_changes_make_separate_builds(self):
        original = self.prepare()
        old_bytes = [Path(p).read_bytes() for p in original[2]]
        self.header.write_text('#define VALUE 42\n')
        header_change = self.prepare()
        self.assertNotEqual(original[0], header_change[0])
        self.source.write_text(self.source.read_text() + '// new source bytes\n')
        source_change = self.prepare()
        self.assertNotEqual(source_change[0], header_change[0])
        changes = ((["-O3", "sm_121a"], self.identity[1], *self.identity[2:]),
                   (self.identity[0], ["-lother"], *self.identity[2:]),
                   (*self.identity[:2], "torch-2", "cuda-13"),
                   (*self.identity[:3], "cuda-14"))
        keys = [source_change[0]] + [self.prepare(identity=v)[0] for v in changes]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual([Path(p).read_bytes() for p in original[2]], old_bytes)

    def test_source_key_and_staged_bytes_use_the_same_snapshot(self):
        original = Path.read_bytes
        reads = []
        def read(path):
            data = original(path)
            if path == self.source:
                reads.append(path)
                path.write_text("changed after the read\n")
            return data
        with patch.object(Path, "read_bytes", read):
            key, _, staged = self.prepare()
        self.assertEqual(len(reads), 1)
        self.assertIn(b"answer()", Path(staged[0]).read_bytes())
        self.assertNotEqual(self.prepare()[0], key)

    def test_corrupted_staged_input_is_repaired_without_removing_build_outputs(self):
        key, directory, staged = self.prepare()
        artifact = directory / "already-built.so"
        artifact.write_bytes(b"ninja owns this output")
        Path(staged[1]).write_bytes(b"incomplete")
        self.assertEqual(self.prepare()[0], key)
        self.assertEqual(Path(staged[1]).read_bytes(), self.header.read_bytes())
        self.assertEqual(artifact.read_bytes(), b"ninja owns this output")

    def test_duplicate_basenames_and_empty_source_lists_are_refused(self):
        other = self.root / self.source.name
        other.write_text("a different translation unit\n")
        with self.assertRaises(ValueError):
            self.prepare([self.source, other])
        with self.assertRaises(ValueError):
            prepare_sources(self.root / "cache", [], self.identity)

    def test_concurrent_processes_preserve_one_complete_source_snapshot(self):
        code = '''import json,sys
from pathlib import Path
from engine.kernels.common.native_cache import prepare_sources
key,directory,sources = prepare_sources(sys.argv[1],sys.argv[2:],("same-runtime",))
print(json.dumps([key,[(Path(p).read_text(),Path(p).stat().st_ino,Path(p).stat().st_mtime_ns) for p in sources]]))
'''
        command = [sys.executable, "-c", code, str(self.root / "parallel"), str(self.source), str(self.header)]
        processes = [subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True) for _ in range(4)]
        try:
            rows = []
            for process in processes:
                out, err = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, err)
                rows.append(json.loads(out))
            self.assertTrue(all(row == rows[0] for row in rows))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_runtime_defaults_and_launcher_use_the_same_persistent_roots(self):
        dockerfile = (ROOT / "engine/runtime/Dockerfile").read_text()
        launcher = (ROOT / "launchers/start-st-glm53.sh").read_text()
        for assignment in ("ST_DENSE_BUILD_ROOT=/cache/st-dense", "ST_ONESHOT_BUILD_ROOT=/cache/st-oneshot",
                           "ST_MLA_BUILD_ROOT=/cache/mla"):
            self.assertIn(assignment, dockerfile)
            self.assertIn(assignment, launcher)

    def test_native_builders_pass_staged_sources_and_retain_link_flags(self):
        # Execute each real build function with a loader that records its inputs;
        # no Torch import or CUDA context is required to check the build boundary.
        from types import SimpleNamespace
        for relative, function in (("dense/__init__.py", "extension"), ("mla/__init__.py", "_build"),
                                   ("oneshot/__init__.py", "build")):
            with self.subTest(builder=relative):
                path = ROOT / "engine/kernels" / relative
                tree = ast.parse(path.read_text())
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
                node.decorator_list = []
                calls = []
                ext = SimpleNamespace(probe_device=lambda: (12, 1, 48))
                def load(**kwargs):
                    calls.append(kwargs)
                    return ext
                fake_torch = SimpleNamespace(__version__="torch-test", version=SimpleNamespace(cuda="cuda-test"))
                namespace = dict(__file__=str(path), Path=Path, os=os, torch=fake_torch,
                                 _EXT=None, MAX_ELEMENTS=64*4096)
                env_name = "ST_" + relative.split('/')[0].upper() + "_BUILD_ROOT"
                with patch.dict(sys.modules, {"torch": fake_torch,
                                              "torch.utils.cpp_extension": SimpleNamespace(load=load)}), \
                        patch.dict(os.environ, {env_name: str(self.root / relative.split('/')[0])}):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                    self.assertIs(namespace[function](), ext)
                call = calls[0]
                for source in call['sources']:
                    self.assertEqual(Path(source).parent, Path(call['build_directory']) / 'src')
                    self.assertEqual(Path(source).read_bytes(), path.with_name(Path(source).name).read_bytes())
                if function == 'build':
                    self.assertEqual(call['extra_ldflags'], ['-libverbs'])
                    self.assertTrue((Path(call['sources'][0]).parent / 'dsv4_oneshot_transport.h').is_file())


if __name__ == "__main__":
    unittest.main()
