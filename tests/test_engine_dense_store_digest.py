"""kernels/dense/store: a weight's lanes hash its bytes and its calibration once, and find the packs filed before."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from engine.kernels.dense.store import PackStore


def filed_keys(store, weight, name, smooth):
    """The cache keys the store filed a weight's W4 and FP8 packs under before its lanes shared digests (main
    bb72154d): each lane copied and hashed the weight, and loaded, smoothed and hashed its calibration, itself."""
    n, k = weight.shape
    hessian = store._hessian(name, k, smooth)
    raw = weight.detach().contiguous().view(torch.uint8).cpu().numpy()
    weight_sha = hashlib.sha256(raw).hexdigest()
    calibration = hashlib.sha256(hessian.contiguous().numpy()).hexdigest() if hessian is not None else 'rtn'
    w4 = dict(version=2, weight=weight_sha, shape=(n, k), name=name, calibration=calibration,
              per_row=not name.startswith('DFlash2Qwen3ForCausalLM/'), algorithm=store.algorithm,
              smooth=store._smooth_sha(smooth), **store._tuning_identity(name))
    fp8 = dict(version=2, weight=weight_sha, shape=(n, k), name=name, kind='fp8', calibration=calibration,
               algorithm=store.algorithm, smooth=store._smooth_sha(smooth), **store._tuning_identity(name))
    key = lambda identity: hashlib.sha256(repr(identity).encode()).hexdigest()
    return (key(w4), w4), (key(fp8), fp8)


class DigestTests(unittest.TestCase):
    N, K = 128, 256
    NAME = 'Glm5NextForCausalLM/model.layers.0.self_attn.fused_qkv_a_proj'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        g = torch.Generator().manual_seed(11)
        self.weight = (torch.randn(self.N, self.K, generator=g) * 0.02).bfloat16()
        self.smooth = torch.exp2(torch.round(torch.log2(torch.rand(self.K, generator=g) + 0.5)))
        x = torch.randn(1024, self.K, generator=g)
        self.store = PackStore(self.root, 0)
        path = self.store.calibration_path(self.NAME)
        path.parent.mkdir(parents=True)
        torch.save(dict(H=x.T @ x, ntok=1024, name=self.NAME, amax=x.abs().amax(0)), path)

    def tearDown(self):
        self.store.release_pages()
        self.tmp.cleanup()

    def file_packs(self):
        """Pack blobs of the served layouts, filed under the keys the store used before this change."""
        (w4_key, w4), (fp8_key, fp8) = filed_keys(PackStore(self.root, 0), self.weight, self.NAME, self.smooth)
        packs = self.root / 'st-dense-packs'
        packs.mkdir(parents=True, exist_ok=True)
        padded = (self.N + 127) // 128 * 128
        torch.save(dict(identity=w4, data=torch.full((padded // 128, self.K // 128, 128, 64), 7, dtype=torch.uint8),
                        scale=torch.ones(padded // 128, self.K // 128, 128, 8, dtype=torch.int8),
                        rowscale=torch.ones(padded)), packs / f'{w4_key}.pt')
        torch.save(dict(identity=fp8, q=torch.zeros(padded, self.K).to(torch.float8_e4m3fn),
                        scale=torch.full((padded // 128, self.K // 128), 0.5)), packs / f'{fp8_key}.pt')

    def test_both_lanes_find_the_filed_packs_and_read_the_calibration_once(self):
        self.file_packs()
        read = mock.patch.object(PackStore, '_hessian', autospec=True, side_effect=PackStore._hessian)
        with read as hessian:
            digest = self.store.weight_digest(self.weight)
            pack = self.store.pack(self.weight, self.NAME, smooth=self.smooth, digest=digest)
            q, scale = self.store.pack_fp8(self.weight, self.NAME, smooth=self.smooth, digest=digest)
        self.assertEqual(hessian.call_count, 1, 'one load, check, smoothing and hash for both lanes')
        self.assertEqual((self.store.stats['cache'], self.store.stats['fp8_cache']), (1, 1), 'the filed packs, not new ones')
        self.assertEqual(self.store.stats['calibration_digest_reused'], 1)
        self.assertNotIn('built', self.store.stats)
        self.assertNotIn('fp8_built', self.store.stats)
        self.assertTrue(pack.calibrated)
        self.assertTrue(bool((pack.data == 7).all()))
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertTrue(bool((scale == 0.5).all()))
        self.assertEqual(len(list((self.root / 'st-dense-packs').glob('*.pt'))), 2, 'nothing filed beside them')

    def test_lanes_without_a_shared_digest_still_find_them(self):
        self.file_packs()
        store = PackStore(self.root, 0)
        read = mock.patch.object(PackStore, '_hessian', autospec=True, side_effect=PackStore._hessian)
        with read as hessian:
            store.pack(self.weight, self.NAME, smooth=self.smooth)
            store.pack_fp8(self.weight, self.NAME, smooth=self.smooth)
        self.assertEqual(hessian.call_count, 2)
        self.assertEqual((store.stats['cache'], store.stats['fp8_cache']), (1, 1))
        self.assertNotIn('calibration_digest_reused', store.stats)
        store.release_pages()

    def test_a_digest_answers_only_for_the_bytes_it_hashed(self):
        w = torch.randn(4, 16).bfloat16()
        expected = hashlib.sha256(w.view(torch.uint8).numpy()).hexdigest()
        digest = self.store.weight_digest(w)
        w.add_(1)                                                          # the hash is of a copy taken before this write
        self.assertFalse(digest.covers(w), 'a write since is other bytes')
        with self.assertRaisesRegex(ValueError, 'different bytes'):
            digest.of(w)
        self.assertEqual(digest._future.result(), expected)
        v = torch.randn(4, 16).bfloat16()
        digest = self.store.weight_digest(v)
        self.assertTrue(digest.covers(v[:, 0:4096].contiguous()), 'a whole-width tile is the same bytes')
        self.assertFalse(digest.covers(v[:, 0:8].contiguous()))
        self.assertFalse(digest.covers(v.clone()))
        self.assertEqual(digest.of(v), hashlib.sha256(v.view(torch.uint8).numpy()).hexdigest())

    def test_a_rewritten_calibration_is_read_again_and_files_a_new_pack(self):
        from engine.kernels.dense import W4Pack
        padded = (self.N + 127) // 128 * 128

        def packed(w, **kw):
            return W4Pack(torch.zeros(padded // 128, self.K // 128, 128, 64, dtype=torch.uint8),
                          torch.zeros(padded // 128, self.K // 128, 128, 8, dtype=torch.int8),
                          torch.ones(padded), self.N, self.K, kw.get('hessian') is not None)
        with mock.patch('engine.kernels.dense.pack_w4', side_effect=packed) as build, \
                mock.patch.object(PackStore, '_factor', return_value=None):
            digest = self.store.weight_digest(self.weight)
            self.store.pack(self.weight, self.NAME, smooth=self.smooth, digest=digest)
            path = self.store.calibration_path(self.NAME)
            torch.save(dict(H=torch.eye(self.K), ntok=64, name=self.NAME, amax=torch.ones(self.K), note='rewritten'), path)
            self.store.pack(self.weight, self.NAME, smooth=self.smooth, digest=digest)
            self.store.pack(self.weight, self.NAME, smooth=self.smooth)
        self.assertEqual(build.call_count, 2, 'the rewritten blob is another identity; its pack is then found')
        self.assertEqual(len(list((self.root / 'st-dense-packs').glob('*.pt'))), 2)

    def test_a_build_refuses_a_calibration_that_changed_under_it(self):
        real = PackStore._hessian
        calls = []

        def drifting(store, name, k, smooth=None):
            calls.append(name)
            h = real(store, name, k, smooth)
            return h if len(calls) == 1 else h * 2
        with mock.patch.object(PackStore, '_hessian', autospec=True, side_effect=drifting):
            with self.assertRaisesRegex(ValueError, 'changed while packing'):
                self.store.pack_fp8(self.weight, self.NAME, smooth=self.smooth)

    def test_release_returns_the_worker_and_a_later_digest_starts_another(self):
        self.store.weight_digest(self.weight)
        self.store.release_pages()
        self.assertIsNone(self.store._hasher)
        digest = self.store.weight_digest(self.weight)
        self.assertEqual(digest.of(self.weight), hashlib.sha256(self.weight.view(torch.uint8).numpy()).hexdigest())


if __name__ == '__main__':
    unittest.main()
