"""SHA256 W4 keys preserve source/recipe identity and reuse historical packs."""
import errno
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_glm53_pack_io as pack_io
from test_glm53_pack_io import ROOT, load_file, torch


@unittest.skipIf(torch is None, "CPU torch required")
class PackKeyTests(unittest.TestCase):
    setUp = pack_io.PackIOTests.setUp

    def test_exact_hashes_and_pack_bytes(self):
        proof = load_file("pack_key_proof", ROOT / "probes/glm53_pack_key_check.py")
        self.assertTrue(proof.check(self.mk, self.common, "cpu")["ok"])

    def test_identity_disabled_cache_and_control_namespace(self):
        weight = torch.arange(512).reshape(4, 128).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_CACHE": root,
                                       "VLLM_GLM53_MK_PACK_SHA256": "1"}), \
                patch.object(self.mk, "_mk_rank", return_value=3):
            key = self.mk._pack_cache_path(weight, True, False, 0)
            expected = hashlib.sha256(weight.view(torch.uint8).numpy()).hexdigest()
            self.assertEqual(Path(key).name, f"sha256-{expected}-4x128-bfloat16-v{self.mk.MK_PACK_VERSION}-row-rtn-lr0.pt")
            self.assertEqual(Path(key).parent, Path(root) / "rank3")
            variants = [self.mk._pack_cache_path(w, r, g, l) for w, r, g, l in (
                (weight.reshape(8, 64), True, False, 0), (weight.view(torch.float16), True, False, 0),
                (weight, False, False, 0), (weight, True, True, 0), (weight, True, False, 8))]
            weight[0, 0] += 1
            variants.append(self.mk._pack_cache_path(weight, True, False, 0))
            self.assertNotIn(key, variants)
            self.assertEqual(len(set(variants)), len(variants))
            with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "0"}):
                legacy = self.mk._pack_cache_path(weight, True, False, 0)
                self.assertEqual(Path(legacy).name.split('-')[0], self.mk._weight_md5(weight))
            for disabled in ("", "0", "off"):
                with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_CACHE": disabled}), \
                        patch.object(self.mk, "_weight_digest", side_effect=AssertionError("disabled hash")):
                    self.assertIsNone(self.mk._pack_cache_path(weight, True, False, 0))

    def test_alias_failure_and_concurrent_publication(self):
        weight = torch.ones(4, 128)
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_CACHE": root}), \
                patch.object(self.mk, "_mk_rank", return_value=0):
            with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "0"}):
                legacy = Path(self.mk._pack_cache_path(weight, True, False, 0))
                legacy.parent.mkdir()
                legacy.write_bytes(b"original")
            with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "1"}):
                for error in (OSError(errno.EROFS, "read-only"), OSError(errno.EXDEV, "cross device")):
                    with patch.object(os, "link", side_effect=error):
                        self.assertEqual(self.mk._pack_cache_path(weight, True, False, 0), str(legacy))
                def concurrent(source, destination):
                    Path(destination).write_bytes(b"concurrent winner")
                    raise FileExistsError(destination)
                with patch.object(os, "link", side_effect=concurrent):
                    winner = Path(self.mk._pack_cache_path(weight, True, False, 0))
                self.assertEqual(winner.read_bytes(), b"concurrent winner")
                self.assertEqual(legacy.read_bytes(), b"original")
                self.assertEqual(self.mk._PACK_IO_STATS["alias_errors"], 2)

    def test_cold_sha_pack_publishes_and_warm_restore_is_exact(self):
        weight = torch.arange(129 * 128).reshape(129, 128).to(torch.bfloat16) / 1000
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_CACHE": root,
                                       "VLLM_GLM53_MK_PACK_SHA256": "1",
                                       "VLLM_GLM53_MK_PACK_LORC": "0"}), \
                patch.object(self.mk, "_mk_rank", return_value=0), \
                patch.object(self.mk, "_calib_hessian_for", return_value=None):
            cold = self.mk.build_mk_weight_w4(weight)
            files = list(Path(root).glob("rank0/*.pt"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.startswith("sha256-"))
            with patch.object(self.mk, "_weight_md5", side_effect=AssertionError("warm MD5")), \
                    patch.object(self.mk, "_w4_row_shift", side_effect=AssertionError("rebuilt")):
                warm = self.mk.build_mk_weight_w4(weight)
            for left, right in zip(cold, warm):
                if isinstance(left, torch.Tensor):
                    self.assertTrue(torch.equal(left, right))
                else:
                    self.assertEqual(left, right)

    def test_key_switch_does_not_rekey_rank_or_fp8_artifacts(self):
        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "0"}):
            baseline = self.common.environment_identity()
        with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "1"}):
            self.assertEqual(self.common.environment_identity(), baseline)


if __name__ == "__main__":
    unittest.main()
