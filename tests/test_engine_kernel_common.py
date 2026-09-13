"""engine/kernels/common: the model-free kernels, and the boundary that keeps them model-free.

On 2026-09-13 the kernels whose arguments are the shape -- the sampler, block verification, vocabulary
candidates, decode commit, norm+RoPE, SwiGLU and the native build cache -- moved into one package, and
engine/base/lanes binds them as the default lanes a profile inherits. What keeps that honest is checked
here without a GPU: a common kernel imports nothing model-bound, nothing imports the old paths, and the
GLM profile takes its common kernels through the base table rather than wiring them itself.
"""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "engine/kernels/common"
MOVED = ("sampler", "block_verify", "vocab_candidates", "decode_commit", "norm_rope", "swiglu", "native_cache")


def imported(path: Path) -> "list[str]":
    """Every module a file imports, spelled as absolute dotted names where the import is absolute, and as
    '.name' for relative ones. `from a.b import c` yields both 'a.b' and 'a.b.c' (c may be a module)."""
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.append("." * node.level + (node.module or ""))
            else:
                names.append(node.module or "")
                names += [f"{node.module}.{alias.name}" for alias in node.names]
    return names


class CommonKernelBoundaryTests(unittest.TestCase):
    def test_the_model_free_kernels_live_in_common_only(self):
        for name in MOVED:
            with self.subTest(kernel=name):
                self.assertTrue((COMMON / f"{name}.py").is_file())
                self.assertFalse((ROOT / "engine/kernels" / f"{name}.py").exists())

    def test_a_common_kernel_imports_nothing_model_bound(self):
        """No profile, no base (so no kernel shape), no other kernel package: the arguments are the shape."""
        for path in sorted(COMMON.glob("*.py")):
            for name in imported(path):
                with self.subTest(file=path.name, imports=name):
                    if name.startswith("."):
                        self.assertFalse(name.startswith(".."), "a relative import leaves the package")
                    elif name.split(".")[0] == "engine":
                        self.assertTrue(name.startswith("engine.kernels.common"), name)
            text = path.read_text()
            for marker in ("os.environ", "getenv("):              # the package reads no environment (D11)
                with self.subTest(file=path.name, marker=marker):
                    self.assertNotIn(marker, text)

    def test_nothing_reaches_a_common_kernel_by_its_old_path(self):
        old = {f"engine.kernels.{name}" for name in MOVED}
        for top in ("engine", "tests", "probes", "bench", "launchers"):
            for path in sorted((ROOT / top).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                names = imported(path)
                in_kernels = path.parent == ROOT / "engine/kernels"
                for name in names:
                    stale = name in old or any(name.startswith(o + ".") for o in old) or (
                        in_kernels and name.lstrip(".") in MOVED and name.startswith("."))
                    with self.subTest(file=str(path.relative_to(ROOT)), imports=name):
                        self.assertFalse(stale)
                text = path.read_text()
                for name in MOVED:
                    with self.subTest(file=str(path.relative_to(ROOT)), path_string=name):
                        self.assertNotIn("engine/kernels/" + name + ".py", text)

    def test_the_glm_profile_inherits_its_common_kernels_from_the_base_table(self):
        for rel in ("engine/profiles/glm53/drafter.py", "engine/profiles/glm53/pipeline.py", "engine/profiles/glm53/lanes.py"):
            names = imported(ROOT / rel)
            with self.subTest(file=rel):
                self.assertFalse([n for n in names if n.startswith("engine.kernels.common")], names)
                self.assertIn("engine.base.lanes", names)

    def test_the_base_table_binds_the_common_kernels(self):
        names = set(imported(ROOT / "engine/base/lanes.py"))
        for expected in ("engine.kernels.common.decode_commit.advance", "engine.kernels.common.norm_rope.norm",
                         "engine.kernels.common.norm_rope.add_norm", "engine.kernels.common.norm_rope.norm_rope",
                         "engine.kernels.common.norm_rope.warm", "engine.kernels.common.swiglu.swiglu"):
            with self.subTest(binds=expected):
                self.assertIn(expected, names)
        self.assertFalse([n for n in names if n.startswith("engine.") and not n.startswith("engine.kernels.common")],
                         "the base table binds kernels, nothing model-bound")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("triton") is not None,
                         "binding the kernels imports Triton")
    def test_the_table_is_the_kernels_and_is_built_once(self):
        from engine.base import lanes
        from engine.kernels.common import decode_commit, norm_rope, swiglu
        table = lanes.served()
        self.assertIs(lanes.served(), table)
        self.assertEqual((table.rmsnorm, table.add_rmsnorm, table.rmsnorm_rope, table.rope_table, table.swiglu, table.commit),
                         (norm_rope.norm, norm_rope.add_norm, norm_rope.norm_rope, norm_rope.warm, swiglu.swiglu,
                          decode_commit.advance))
        from engine.profiles.glm53 import drafter
        self.assertEqual((drafter.norm, drafter.norm_rope, drafter.add_norm, drafter.swiglu, drafter.warm_rotary),
                         (norm_rope.norm, norm_rope.norm_rope, norm_rope.add_norm, swiglu.swiglu, norm_rope.warm))


if __name__ == "__main__":
    unittest.main()
