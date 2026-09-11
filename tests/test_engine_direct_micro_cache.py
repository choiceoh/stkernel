"""CPU checks for the direct micro MoE kernel's move onto flashinfer's on-disk
CuTe-DSL cache (2026-09-12): the TVM-FFI launch form, and the block-dim verdict
that travels beside the exported object as a sidecar."""
import contextlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:                       # CuTe DSL is an image dependency
        raise unittest.SkipTest(f"{name} unavailable here: {exc}")


class _FakeCompiled:
    """Stands in for a cute.compile result: exportable, callable, recognisable."""
    def __init__(self, tag):
        self.tag = tag

    def export_to_c(self, path, function_name):
        Path(path).write_bytes(f"{self.tag}:{function_name}".encode())


class _FakeKernel:
    launch_block_dim = 512

    def __cache_key__(self):
        return ("fake", 1, 2.0, None)


@contextlib.contextmanager
def _disk_cache(md, tmp, *, accepts, builds):
    """flashinfer's cache rooted at `tmp`, compile and probe replaced by fakes."""
    from flashinfer.jit import cute_dsl_core, env as jit_env

    def fake_compile(kernel, *, topk_ids_dtype, tvm_ffi=False):
        builds.append(tvm_ffi)
        return _FakeCompiled(f"build{len(builds)}")

    with mock.patch.dict(os.environ, {"FLASHINFER_CUTE_DSL_DISABLE_CACHE": "0"}), \
         mock.patch.object(jit_env, "FLASHINFER_JIT_DIR", Path(tmp)), \
         mock.patch.object(cute_dsl_core, "_get_compile_arch", lambda: "sm121a"), \
         mock.patch.object(cute_dsl_core.JitSpecCuteDsl, "_load_from_disk", lambda self: "LOADED"), \
         mock.patch.object(md, "compile_direct_micro_kernel", fake_compile), \
         mock.patch.object(md, "compiled_direct_micro_accepts_block_dim", lambda compiled, block_dim: accepts):
        yield


class DirectMicroDiskCacheTests(unittest.TestCase):
    def test_miss_builds_and_persists_the_verdict_then_hits_reload_it(self):
        md = _module("engine.kernels.b12x.moe_dispatch")
        kernel = _FakeKernel()
        key = ("direct_micro", kernel.__cache_key__(), torch.int32)
        builds = []
        with tempfile.TemporaryDirectory() as tmp, _disk_cache(md, tmp, accepts=True, builds=builds):
            compiled, ok = md._build_direct_micro_on_disk(kernel, "direct_micro_test", key, torch.int32)
            self.assertIsInstance(compiled, _FakeCompiled)              # a miss compiles in-process, TVM-FFI form ...
            self.assertEqual((ok, builds), (True, [True]))
            module_dir = next(Path(tmp).glob("st_b12x_direct_micro_*_cute_dsl"))
            obj = next(module_dir.glob("direct_micro_test_*.o"))
            sidecar = obj.with_name(obj.name[:-2] + ".blockdim.json")
            self.assertEqual(json.loads(sidecar.read_text()), {"block_dim": 512, "accepts": True})

            compiled, ok = md._build_direct_micro_on_disk(kernel, "direct_micro_test", key, torch.int32)
            self.assertEqual((compiled, ok, len(builds)), ("LOADED", True, 1))   # ... a hit reloads the .o and the verdict

            sidecar.unlink()                                                # an object without its verdict is rebuilt
            compiled, ok = md._build_direct_micro_on_disk(kernel, "direct_micro_test", key, torch.int32)
            self.assertIsInstance(compiled, _FakeCompiled)
            self.assertTrue(sidecar.exists())
            self.assertEqual(len(builds), 2)

            sidecar.write_text(json.dumps({"block_dim": 256, "accepts": True}))   # a stale verdict is rebuilt, never guessed
            compiled, ok = md._build_direct_micro_on_disk(kernel, "direct_micro_test", key, torch.int32)
            self.assertIsInstance(compiled, _FakeCompiled)
            self.assertEqual((json.loads(sidecar.read_text())["block_dim"], len(builds)), (512, 3))

    def test_a_refused_block_dim_is_persisted_and_reloaded_as_refused(self):
        md = _module("engine.kernels.b12x.moe_dispatch")
        kernel = _FakeKernel()
        key = ("direct_micro", kernel.__cache_key__(), torch.int64)
        builds = []
        with tempfile.TemporaryDirectory() as tmp, _disk_cache(md, tmp, accepts=False, builds=builds):
            self.assertFalse(md._build_direct_micro_on_disk(kernel, "p", key, torch.int64)[1])
            self.assertEqual(md._build_direct_micro_on_disk(kernel, "p", key, torch.int64), ("LOADED", False))
            self.assertEqual(len(builds), 1)

    def test_disabled_cache_compiles_in_process(self):
        md = _module("engine.kernels.b12x.moe_dispatch")
        kernel = _FakeKernel()
        with mock.patch.dict(os.environ, {"FLASHINFER_CUTE_DSL_DISABLE_CACHE": "1"}), \
             mock.patch.object(md, "compile_direct_micro_kernel", lambda k, *, topk_ids_dtype, tvm_ffi=False: _FakeCompiled("x")), \
             mock.patch.object(md, "compiled_direct_micro_accepts_block_dim", lambda compiled, block_dim: False):
            compiled, ok = md._build_direct_micro_on_disk(kernel, "p", ("direct_micro", kernel.__cache_key__(), torch.int32), torch.int32)
            self.assertIsInstance(compiled, _FakeCompiled)
            self.assertFalse(ok)


class DirectMicroLaunchFormTests(unittest.TestCase):
    def test_tvm_ffi_launch_passes_addresses_tensors_ints_and_no_stream(self):
        dm = _module("engine.kernels.b12x.moe_direct_micro_kernel")
        seen = []
        t = {name: torch.zeros(4, dtype=torch.uint8) for name in
             ("x", "w1", "w1s", "w1a", "a1", "a2", "inter", "w2", "w2s", "w2a", "ids", "tw", "out")}
        bc, be = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
        dm.MoEDirectMicroKernel.launch(
            lambda *args: seen.append(args),
            x=t["x"], w1_fp4=t["w1"], w1_blockscale=t["w1s"], w1_alphas=t["w1a"], a1_gscale=t["a1"], a2_gscale=t["a2"],
            inter_fp32=t["inter"], w2_fp4=t["w2"], w2_blockscale=t["w2s"], w2_alphas=t["w2a"], topk_ids=t["ids"],
            topk_weights=t["tw"], out=t["out"], barrier_count=bc, barrier_epoch=be, m=6, grid_x=48, tvm_ffi=True)
        args = seen[0]
        self.assertEqual(len(args), 17)                                   # 13 addresses + 2 tensors + 2 ints, no stream
        self.assertEqual(list(args[:13]), [v.data_ptr() for v in t.values()])
        self.assertTrue(args[13] is bc and args[14] is be)
        self.assertEqual(args[15:], (6, 48))

    def test_compile_signature_carries_the_tvm_ffi_form(self):
        import inspect
        dm = _module("engine.kernels.b12x.moe_direct_micro_kernel")
        self.assertIn("tvm_ffi", inspect.signature(dm.compile_direct_micro_kernel).parameters)
        src = (ROOT / "engine/kernels/b12x/moe_direct_micro_kernel.py").read_text()
        self.assertIn("make_fake_stream(use_tvm_ffi_env_stream=True) if tvm_ffi", src)
        self.assertIn('"--enable-tvm-ffi" not in (options or "")', src)
        dispatch = (ROOT / "engine/kernels/b12x/moe_dispatch.py").read_text()
        self.assertIn("tvm_ffi=True,\n        )\n        return scatter_output", dispatch)


if __name__ == "__main__":
    unittest.main()
