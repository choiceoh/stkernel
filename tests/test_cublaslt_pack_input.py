"""A comparison must preserve the named pack's bytes and reject wrong shapes."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class PackedInputTests(unittest.TestCase):
    def test_preserves_quantized_bytes_without_cuda_or_requantization(self):
        import torch
        from probes.engine_cublaslt_check import packed_weight
        q = torch.arange(128 * 256, dtype=torch.int32).remainder(100).to(torch.uint8)
        q = q.view(torch.float8_e4m3fn).view(128, 256)
        scale = torch.tensor([[.5, 2.]], dtype=torch.float32)
        identity = dict(kind='fp8', name='test/head', shape=(128, 256))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pack.pt'
            torch.save(dict(identity=identity, q=q, scale=scale), path)
            (got_q, got_scale), receipt = packed_weight(path, 128, 256)
            self.assertEqual(got_q.device.type, 'cpu')
            self.assertTrue(torch.equal(q.view(torch.uint8), got_q.view(torch.uint8)))
            self.assertTrue(torch.equal(scale, got_scale))
            expected = hashlib.sha256(q.view(torch.uint8).numpy().tobytes()
                                      + scale.view(torch.uint8).numpy().tobytes()).hexdigest()
            self.assertEqual(receipt['sha256'], expected)
            self.assertEqual(receipt['identity'], identity)
            with self.assertRaisesRegex(ValueError, 'padded weight shape'):
                packed_weight(path, 256, 256)

    def test_rejects_wrong_format_and_scale_geometry(self):
        import torch
        from probes.engine_cublaslt_check import packed_weight
        q = torch.zeros(128, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        good = dict(identity=dict(kind='fp8'), q=q, scale=torch.ones(1, 2))
        for changed in (dict(identity=dict(kind='w4')), dict(q=q.float()),
                        dict(scale=torch.ones(2, 1)), dict(scale=torch.ones(1, 2).double())):
            with self.subTest(changed=list(changed)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'pack.pt'
                torch.save(dict(good, **changed), path)
                with self.assertRaisesRegex(ValueError, 'padded weight shape'):
                    packed_weight(path, 128, 256)
