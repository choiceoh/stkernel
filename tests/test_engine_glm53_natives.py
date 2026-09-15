"""profiles/glm53/natives: every native extension is built on every rank before the ranks' first device collective."""
import ast
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "engine/kernels"


def native_modules():
    """Dotted names of the kernel modules that JIT-build a torch extension."""
    out = set()
    for path in KERNELS.rglob("*.py"):
        if "cpp_extension import load" in path.read_text():
            relative = path.relative_to(ROOT).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            out.add(".".join(parts))
    return out


class NativeListTests(unittest.TestCase):
    def test_every_native_extension_under_kernels_is_built_before_the_first_collective(self):
        from engine.profiles.glm53 import natives
        listed = {module: entry for _, module, entry in natives.MODULES + (natives.ONESHOT,)}
        self.assertEqual(set(listed), native_modules(),
                         "a native built at its first use makes the ranks wait for its compile inside a collective")
        for module, entry in listed.items():
            path = ROOT / (module.replace(".", "/") + ".py")
            if not path.is_file():
                path = ROOT / module.replace(".", "/") / "__init__.py"
            tree = ast.parse(path.read_text())
            self.assertIn(entry, {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}, module)

    def test_the_dense_build_touches_no_device_and_its_lane_probes_after_it(self):
        tree = ast.parse((KERNELS / "dense/__init__.py").read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        build = ast.unparse(functions["build"])
        self.assertNotIn("probe_device", build)
        self.assertNotIn("kernel_shape", build)
        extension = ast.unparse(functions["extension"])
        self.assertIn("build()", extension)
        self.assertIn("probe_device", extension)


class NativeBuildsTests(unittest.TestCase):
    def test_builds_run_at_once_and_report_each_in_order(self):
        from engine.profiles.glm53.natives import NativeBuilds
        threads = set()

        def slow(value):
            def run():
                threads.add(threading.get_ident())
                time.sleep(0.4)
                return value
            return run
        start = time.perf_counter()
        builds = NativeBuilds([(f"n{i}", slow(i)) for i in range(4)])
        seconds = builds.wait()
        wall = time.perf_counter() - start
        self.assertEqual(list(seconds), ["n0", "n1", "n2", "n3"])
        self.assertTrue(all(0.35 <= s < 1.0 for s in seconds.values()), seconds)
        self.assertLess(wall, 1.2, "four 0.4 s builds must overlap, not queue")
        self.assertEqual(len(threads), 4)
        self.assertNotIn(threading.get_ident(), threads)

    def test_a_failed_build_is_named_after_every_build_has_finished(self):
        from engine.profiles.glm53.natives import NativeBuilds
        finished = []

        def fine():
            time.sleep(0.2)
            finished.append("fine")

        def broken():
            raise OSError("ninja: build stopped: subcommand failed")
        with self.assertRaisesRegex(RuntimeError, "broken: OSError: ninja: build stopped"):
            NativeBuilds([("fine", fine), ("broken", broken)]).wait()
        self.assertEqual(finished, ["fine"])

    def test_modules_are_imported_on_the_callers_thread_and_one_shot_gets_the_served_build(self):
        import importlib
        from engine.profiles.glm53 import natives
        importers, calls = set(), []
        real = importlib.import_module
        fake = type("Module", (), {})()
        fake.build = fake._build = lambda: "module"
        fake_oneshot = type("Module", (), {})()
        fake_oneshot.build = lambda rails, *, inline_flags: calls.append((rails, inline_flags)) or "oneshot"

        def record(name, *args):
            importers.add(threading.get_ident())
            if name == "engine.kernels.oneshot":
                return fake_oneshot
            if name.startswith("engine.kernels.") and name not in ("engine.kernels.common.native_cache",
                                                                  "engine.kernels.native_root"):
                return fake
            return real(name, *args)
        with patch.object(natives.importlib, "import_module", side_effect=record):
            builds = natives.builds(2, True)
        self.assertEqual(importers, {threading.get_ident()})
        self.assertEqual([name for name, _ in builds],
                         ["dense", "mla", "prefill-topk", "decode-topk", "mapped-staging", "bounded-graph",
                          "decode-queue", "one-shot"])
        self.assertEqual(natives.NativeBuilds(builds).wait().keys(), dict(builds).keys())
        self.assertEqual(calls, [(2, True)])

    def test_the_line_names_every_build(self):
        from engine.profiles.glm53.natives import line
        self.assertEqual(line(3, {"dense": 0.21, "mla": 61.04}, 61.3),
                         "  rank3: native builds in 61.3 s (dense 0.2s, mla 61.0s)")


if __name__ == "__main__":
    unittest.main()
