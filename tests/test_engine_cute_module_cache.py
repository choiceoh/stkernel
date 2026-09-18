"""b12x CuTe-DSL kernel families keep separate on-disk modules: building one never wipes another's artifacts.

flashinfer invalidates a whole module directory when a kernel with other key files is built into it. While the
dynamic prefill kernels (static key files plus variant files) shared the static kernels' module, every boot erased
and recompiled both families (2026-09-15, "Invalidating stale CuTe-DSL module" three times in one boot). The same
wipe ran between trees: production's release and a Qwen3.8 window's tree built into one `st_b12x_moe` on the same
/cache and took turns recompiling (2026-09-18), so the module is named by its files' contents too.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "cutlass", "flashinfer")),
                     "requires the ST image (torch, CuTe DSL, flashinfer)")
class CuteModuleCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
        import torch
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)):
            from engine.kernels.b12x import moe_dispatch
        cls.md = moe_dispatch

    def keys(self, *names):
        here = os.path.dirname(self.md.__file__)
        return self.md._kernel_source_files() + tuple(os.path.join(here, name) for name in names)

    def test_each_key_set_names_its_own_module(self):
        md = self.md
        static = md._kernel_source_files()
        packets = self.keys("moe_dynamic_prefill_packets.py", "moe_w4a16_fp4_helpers.py")
        batch8 = self.keys("moe_prefill_q0_batch8.py", "_prefill_q0_batch8.py")
        self.assertTrue(md._cute_dsl_module(static).startswith(md._CUTE_DSL_MODULE + "_"))
        self.assertEqual(len({md._cute_dsl_module(k) for k in (static, packets, batch8)}), 3)
        self.assertEqual(md._cute_dsl_module(packets), md._cute_dsl_module(tuple(list(packets))))
        # the same files in another order hash differently in flashinfer's key, so they get another module too
        self.assertNotEqual(md._cute_dsl_module(packets), md._cute_dsl_module(packets[:-2] + packets[:-3:-1]))

    def test_building_one_family_keeps_the_other_familys_artifact(self):
        from flashinfer.jit import cute_dsl_core
        md = self.md

        class Compiled:
            def export_to_c(self, path, function_name):
                Path(path).write_bytes(b"object")

        static = md._kernel_source_files()
        dynamic = self.keys("moe_dynamic_prefill_packets.py", "moe_w4a16_fp4_helpers.py")

        def build(module, name, keys):
            cute_dsl_core.build_and_load_cute_dsl_kernel(module, name, Compiled, extra_key_files=keys)
            return cute_dsl_core.JitSpecCuteDsl(module, name, Compiled, cute_dsl_core._hash_source_files(tuple(keys)))

        with tempfile.TemporaryDirectory() as root, \
                patch.object(cute_dsl_core.jit_env, "FLASHINFER_JIT_DIR", Path(root)):
            first = build(md._cute_dsl_module(static), "static_m1", static)
            build(md._cute_dsl_module(dynamic), "dynamic_e288", dynamic)
            self.assertTrue(first.is_compiled, "building the dynamic family wiped the static artifact")
            # control: one shared module, as before -- the second family's build erases the first
            shared = build(md._CUTE_DSL_MODULE + "_shared", "static_m1", static)
            build(md._CUTE_DSL_MODULE + "_shared", "dynamic_e288", dynamic)
            self.assertFalse(shared.is_compiled)

    def test_two_trees_on_one_cache_keep_their_own_artifacts(self):
        """Production's release and a window's tree: the same key file names, other contents, one /cache."""
        from flashinfer.jit import cute_dsl_core
        md = self.md

        class Compiled:
            def export_to_c(self, path, function_name):
                Path(path).write_bytes(b"object")

        with tempfile.TemporaryDirectory() as root, \
                patch.object(cute_dsl_core.jit_env, "FLASHINFER_JIT_DIR", Path(root) / "jit"):
            trees = []
            for tree, body in (("release", "# production\n"), ("window", "# qwen38 window\n")):
                keys = []
                for name in ("moe_dispatch.py", "moe_static_kernel.py"):
                    path = Path(root) / "repo" / name          # one path, as every tree is mounted at /repo
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(body + name)
                    keys.append(str(path))
                module = md._cute_dsl_module(tuple(keys))
                cute_dsl_core.build_and_load_cute_dsl_kernel(module, "static_m1", Compiled, extra_key_files=keys)
                trees.append((module, cute_dsl_core._hash_source_files(tuple(keys))))
                md._MODULE_NAMES.clear()                        # the next tree is another process
            (release, release_sha), (window, _) = trees
            self.assertNotEqual(release, window)
            spec = cute_dsl_core.JitSpecCuteDsl(release, "static_m1", Compiled, release_sha)
            self.assertTrue(spec.is_compiled, "the window's build wiped the release's artifact")

    def test_an_unreadable_key_file_keeps_a_name_and_is_not_remembered(self):
        md = self.md
        missing = ("/nonexistent/moe_dispatch.py",)
        name = md._cute_dsl_module(missing)
        self.assertTrue(name.startswith(md._CUTE_DSL_MODULE + "_"))
        self.assertNotIn(missing, md._MODULE_NAMES)


if __name__ == "__main__":
    unittest.main()
