"""Exact W4 cache transport: CPU round trips; GPU checks use the fleet probe."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "CPU torch required")
class PackIOTests(unittest.TestCase):
    def setUp(self):
        self.common = load_file("pack_common", ROOT / "overlay/modules/glm53_model/glm53_startup_cache.py")
        self.mk = load_file("pack_mk", ROOT / "overlay/modules/glm53_megakernel/glm53_megakernel.py")
        modules = patch.dict(sys.modules, {"vllm.model_executor.layers.glm53_startup_cache": self.common})
        modules.start()
        self.addCleanup(modules.stop)
        env = patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_FAST_IO": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_md5_matches_legacy_for_views_dtypes_and_chunk_boundaries(self):
        with patch.object(self.common, "TRANSFER_BYTES", 1024):
            for dtype in (torch.bfloat16, torch.float32, torch.uint8):
                base = torch.arange(7000).to(dtype).reshape(70, 100)
                for weight in (base, base[1:43], base[:, ::2], base.T):
                    expected = hashlib.md5(weight.contiguous().view(torch.uint8).numpy()).hexdigest()
                    self.assertEqual(self.mk._weight_md5(weight), expected)

    def test_disabled_path_never_allocates_staging(self):
        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_FAST_IO": "0"}), \
                patch.object(self.mk, "_pack_staging", side_effect=AssertionError("staging used")):
            self.mk._weight_md5(torch.ones(4, 4))
            self.mk._pack_tensor_to_device(torch.ones(4, 4), "cpu", False)

    def test_restore_dense_offset_and_strided_views_and_optional_fields(self):
        with patch.object(self.mk, "_pack_staging", side_effect=AssertionError("CPU staging used")):
            for dtype in (torch.uint8, torch.int8, torch.float32, torch.bfloat16):
                base = torch.arange(7000).to(dtype).reshape(70, 100)
                for value in (base, base[1:43], base[:, ::2], base.T):
                    actual = self.mk._pack_tensor_to_device(value, "cpu", True)
                    expected = value.to("cpu")
                    self.assertIs(actual, value)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertTrue(torch.equal(actual, expected))
            self.assertIsNone(self.mk._pack_tensor_to_device(None, "cpu", True))

    def test_legacy_zip_and_mmap_blobs_restore_exact_same_pack(self):
        weight = torch.arange(129 * 256).reshape(129, 256).to(torch.bfloat16)
        blob = {"version": self.mk.MK_PACK_VERSION,
                "wq4": torch.randint(0, 256, (2, 2, 128, 64), dtype=torch.uint8),
                "ws4": torch.randint(-64, 64, (2, 2, 128, 8), dtype=torch.int8),
                "wgs": 0.125, "rgs": torch.arange(256).float(),
                "lr_a": torch.randn(256, 8).bfloat16(),
                "lr_b": torch.randn(8, 256).bfloat16()}
        with tempfile.TemporaryDirectory() as root:
            for zip_format in (False, True):
                path = str(Path(root) / "pack.pt")
                torch.save(blob, path, _use_new_zipfile_serialization=zip_format)
                packs = []
                for fast in ("0", "1"):
                    with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_FAST_IO": fast}), \
                            patch.object(self.mk, "_calib_hessian_for", return_value=None), \
                            patch.object(self.mk, "_pack_cache_path", return_value=path), \
                            patch.object(self.mk, "_w4_row_shift", side_effect=AssertionError("rebuilt")):
                        packs.append(self.mk.build_mk_weight_w4(weight))
                for left, right in zip(*packs):
                    if isinstance(left, torch.Tensor):
                        self.assertTrue(torch.equal(left, right))
                        self.assertEqual(left.stride(), right.stride())
                    else:
                        self.assertEqual(left, right)
        self.assertEqual(self.mk._PACK_IO_STATS["fast_hits"], 2)
        self.assertEqual(self.mk._PACK_IO_STATS["legacy_hits"], 2)

    def test_unrelated_load_error_is_not_retried_as_a_legacy_file(self):
        with patch.object(torch, "load", side_effect=RuntimeError("damaged storage")) as loader:
            with self.assertRaisesRegex(RuntimeError, "damaged"):
                self.mk._load_pack_blob("bad.pt", True)
            self.assertEqual(loader.call_count, 1)

    def test_transfer_switch_keeps_artifact_identity_but_numerics_rekeys(self):
        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_FAST_IO": "0"}):
            baseline = self.common.environment_identity()
        self.assertEqual(self.common.environment_identity(), baseline)
        with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE_PREFETCH": "1"}):
            self.assertEqual(self.common.environment_identity(), baseline)
        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_ROWSHIFT": "different"}):
            self.assertNotEqual(self.common.environment_identity(), baseline)


if __name__ == "__main__":
    unittest.main()
