"""CPU regressions for MK pack-cache hashing and lazy calibration loading."""
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


SOURCE = (Path(__file__).resolve().parents[1]
          / "overlay/modules/glm53_megakernel/glm53_megakernel.py")


@unittest.skipIf(torch is None, "CPU torch is required")
class CalibrationLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {
            "VLLM_GLM53_MK_CALIB_DIR": str(root / "calib"),
            "VLLM_GLM53_MK_PACK_CACHE": str(root / "packs"),
            "VLLM_GLM53_MK_PACK_GPTQ": "1",
            "VLLM_GLM53_MK_PACK_ROWSHIFT": "1",
            "VLLM_GLM53_MK_PACK_LORC": "0",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        spec = importlib.util.spec_from_file_location("mk_cache_test", SOURCE)
        self.mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mk)
        self.mk.logger = logging.getLogger("mk_cache_test")
        self.mk.logger.addHandler(logging.NullHandler())
        self.mk.logger.propagate = False
        self.mk._mk_rank = lambda: 0
        self.name = "Glm5Next/model.layers.0.q_proj"
        self.path = root / "calib" / "rank0" / (self.name + ".pt")
        self.path.parent.mkdir(parents=True)
        self.k = 128
        self.hessian = torch.diag(torch.arange(1, self.k + 1).float())
        self.real_load = torch.load

    def save_hessian(self, *, legacy=False):
        torch.save({"H": self.hessian, "ntok": 37}, self.path,
                   _use_new_zipfile_serialization=not legacy)

    @staticmethod
    def historical_weight_md5(weight):
        value = weight.detach().contiguous()
        raw = value.view(torch.uint8) if value.dtype != torch.uint8 else value
        return hashlib.md5(raw.cpu().numpy().tobytes()).hexdigest()

    def test_weight_hash_preserves_every_byte_across_dtypes_and_layouts(self):
        dtypes = (torch.bfloat16, torch.float16, torch.float32, torch.float64,
                  torch.uint8, torch.int8, torch.int16, torch.int32,
                  torch.int64, torch.bool, torch.complex64, torch.complex128,
                  torch.float8_e4m3fn, torch.float8_e5m2)
        for dtype in dtypes:
            width = torch.empty((), dtype=dtype).element_size()
            # Arbitrary raw bit patterns include NaNs, signed zeros and
            # unnormalised bool bytes; hashing must not reinterpret values.
            raw = torch.arange(128 * width).to(torch.uint8)
            raw[:2 * width] = 0
            raw[2 * width - 1] = 128
            raw[-width:] = 255
            value = raw.view(dtype).reshape(8, 16)
            layouts = {
                "contiguous": value,
                "transpose": value.T,
                "row_stride": value[::2],
                "column_stride": value[:, ::2],
                "offset": value[1:7],
                "broadcast": value[:1].expand(8, -1),
                "empty": value[:0],
            }
            for layout, tensor in layouts.items():
                with self.subTest(dtype=dtype, layout=layout):
                    self.assertEqual(self.mk._weight_md5(tensor),
                                     self.historical_weight_md5(tensor))

    def test_weight_hash_does_not_create_a_python_bytes_copy(self):
        import numpy as np

        class NoBytesCopy(np.ndarray):
            def tobytes(self, *args, **kwargs):
                raise AssertionError("weight hash copied the entire host buffer")

        weight = torch.arange(512).reshape(4, 128).to(torch.bfloat16)
        expected = self.historical_weight_md5(weight)
        real_numpy = torch.Tensor.numpy

        def numpy_view(tensor, *args, **kwargs):
            return real_numpy(tensor, *args, **kwargs).view(NoBytesCopy)

        with patch.object(torch.Tensor, "numpy", numpy_view):
            self.assertEqual(self.mk._weight_md5(weight), expected)

    def test_weight_hash_keeps_historical_cache_paths_and_detects_mutation(self):
        weight = torch.arange(512).reshape(4, 128).to(torch.bfloat16)
        root = Path(os.environ["VLLM_GLM53_MK_PACK_CACHE"]) / "rank0"
        historical_hash = self.historical_weight_md5(weight)
        for per_row in (False, True):
            for gptq in (False, True):
                for rank in (0, 8):
                    expected = root / (
                        f"{historical_hash}-4x128-bfloat16-v{self.mk.MK_PACK_VERSION}-"
                        f"{'row' if per_row else 'ten'}-{'gptq' if gptq else 'rtn'}-lr{rank}.pt")
                    self.assertEqual(Path(self.mk._pack_cache_path(weight, per_row, gptq, rank)),
                                     expected)
        before = self.mk._pack_cache_path(weight, True, True, 0)
        weight[0, 0] = -1
        self.assertNotEqual(self.mk._pack_cache_path(weight, True, True, 0), before)

    def test_zip_hessian_maps_storage_and_preserves_values(self):
        self.save_hessian()
        with patch.object(torch, "load", wraps=self.real_load) as load:
            hessian, ntok = self.mk._calib_hessian_for(self.name, self.k)
        self.assertTrue(torch.equal(hessian, self.hessian))
        self.assertEqual(ntok, 37)
        self.assertEqual(load.call_count, 1)
        self.assertTrue(load.call_args.kwargs["mmap"])

    def test_legacy_hessian_falls_back_and_preserves_values(self):
        self.save_hessian(legacy=True)
        with patch.object(torch, "load", wraps=self.real_load) as load:
            hessian, ntok = self.mk._calib_hessian_for(self.name, self.k)
        self.assertTrue(torch.equal(hessian, self.hessian))
        self.assertEqual(ntok, 37)
        self.assertEqual(load.call_count, 2)
        self.assertTrue(load.call_args_list[0].kwargs["mmap"])
        self.assertNotIn("mmap", load.call_args_list[1].kwargs)

    def test_old_torch_without_mmap_retains_ordinary_load(self):
        self.save_hessian()

        def old_load(*args, **kwargs):
            if "mmap" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'mmap'")
            return self.real_load(*args, **kwargs)

        with patch.object(torch, "load", side_effect=old_load) as load:
            hessian, ntok = self.mk._calib_hessian_for(self.name, self.k)
        self.assertTrue(torch.equal(hessian, self.hessian))
        self.assertEqual(ntok, 37)
        self.assertEqual(load.call_count, 2)

    def test_non_mmap_load_error_keeps_rtn_fallback(self):
        self.save_hessian()
        with patch.object(torch, "load", side_effect=RuntimeError("corrupt archive")) as load:
            self.assertIsNone(self.mk._calib_hessian_for(self.name, self.k))
        self.assertEqual(load.call_count, 1)

    def test_shape_mismatch_missing_file_and_disabled_gptq_keep_rtn(self):
        self.save_hessian()
        self.assertIsNone(self.mk._calib_hessian_for(self.name, self.k * 2))
        self.assertIsNone(self.mk._calib_hessian_for(self.name + ".missing", self.k))
        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_GPTQ": "0"}), \
                patch.object(torch, "load", side_effect=AssertionError("disabled GPTQ loaded")):
            self.assertIsNone(self.mk._calib_hessian_for(self.name, self.k))

    def test_warm_pack_hit_does_not_read_hessian_storage(self):
        self.save_hessian()
        weight = torch.ones(3, self.k, dtype=torch.bfloat16)
        cache = Path(self.mk._pack_cache_path(weight, True, True, 0))
        cache.parent.mkdir(parents=True)
        blob = {
            "version": self.mk.MK_PACK_VERSION,
            "wq4": torch.full((1, 1, 128, 64), 3, dtype=torch.uint8),
            "ws4": torch.full((1, 1, 128, 8), 5, dtype=torch.int8),
            "wgs": 1.0, "rgs": torch.ones(128), "lr_a": None, "lr_b": None,
        }
        torch.save(blob, cache)
        real_reader = torch.serialization._open_zipfile_reader
        opened = []

        class MetadataOnlyReader:
            """Reject torch's eager tensor-storage read, allow mmap metadata."""
            def __init__(self, reader):
                self.reader = reader

            def __getattr__(self, name):
                return getattr(self.reader, name)

            def get_storage_from_record(self, *args, **kwargs):
                raise AssertionError("pack hit materialized the Hessian storage")

        class GuardedOpen:
            def __init__(self, opener):
                self.opener = opener

            def __enter__(self):
                return MetadataOnlyReader(self.opener.__enter__())

            def __exit__(self, *args):
                return self.opener.__exit__(*args)

        def reader(path):
            opener = real_reader(path)
            if Path(getattr(path, "name", path)) == self.path:
                opened.append(path)
                return GuardedOpen(opener)
            return opener

        with patch.object(torch.serialization, "_open_zipfile_reader", side_effect=reader), \
                patch.object(self.mk, "_w4_row_shift", side_effect=AssertionError("cache miss")):
            pack = self.mk.build_mk_weight_w4(weight, self.name)
        self.assertEqual(len(opened), 1)
        self.assertEqual(self.mk._PACK_STATS["cached"], 1)
        self.assertEqual(self.mk._PACK_STATS["gptq"], 0)
        for actual, expected in zip(pack, [blob[key] for key in
                ("wq4", "ws4", "wgs", "rgs", "lr_a", "lr_b")]):
            if isinstance(expected, torch.Tensor):
                self.assertTrue(torch.equal(actual, expected))
            else:
                self.assertEqual(actual, expected)

    def test_gptq_cache_miss_consumes_exact_hessian_and_keeps_pack_bytes(self):
        self.save_hessian()
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn(3, self.k, generator=generator).to(torch.bfloat16)
        gptq = self.mk._w4_gptq_codes
        seen = []

        def capture(*args, **kwargs):
            self.assertTrue(torch.equal(args[3], self.hessian))
            seen.append(True)
            return gptq(*args, **kwargs)

        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_CACHE": "off"}), \
                patch.object(self.mk, "_w4_gptq_codes", side_effect=capture):
            mapped = self.mk.build_mk_weight_w4(weight, self.name)
            # The prior eager-load path provides the same GPTQ input and pack.
            eager = self.real_load(self.path, map_location="cpu")
            self.mk._CALIB_OVERRIDE = (eager["H"], eager["ntok"])
            ordinary = self.mk.build_mk_weight_w4(weight, self.name)
        self.assertEqual(len(seen), 2)
        self.assertEqual(self.mk._PACK_STATS["gptq"], 2)
        self.assertEqual(self.mk._PACK_STATS["gptq_failed"], 0)
        for actual, expected in zip(mapped, ordinary):
            if isinstance(expected, torch.Tensor):
                self.assertTrue(torch.equal(actual, expected))
            else:
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
