import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from probes.qwen38_gptq_subset import subset, file_sha
from probes.qwen38_gptq_offline import check_fit, manifest_files


class OfflineBoundaryTests(unittest.TestCase):
    def test_small_or_partial_calibration_cannot_enter_the_330k_job(self):
        rows = [dict(name=str(n), ntok=330234) for n in range(193)]
        report = dict(records=rows, sites=193, statistics_valid=True)
        check_fit(report)
        rows[-1]['ntok'] = 131184
        with self.assertRaisesRegex(ValueError, '330000'):
            check_fit(report)
        rows[-1]['ntok'] = 330234
        rows[-1]['name'] = rows[0]['name']
        with self.assertRaises(ValueError):
            check_fit(report)

    def test_dense_subset_preserves_bytes_and_excludes_experts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = dict(dense=dict(dtype='BF16', shape=[2, 2], data_offsets=[4, 12]),
                          expert=dict(dtype='BF16', shape=[1, 2], data_offsets=[0, 4]))
            raw = json.dumps(header).encode()
            source = root / 'source.safetensors'
            source.write_bytes(struct.pack('<Q', len(raw)) + raw + b'abcd12345678')
            output = root / 'subset.safetensors'
            result = subset(source, output, ['dense'])
            data = output.read_bytes()
            size = struct.unpack('<Q', data[:8])[0]
            self.assertEqual(set(json.loads(data[8:8+size])), {'dense'})
            self.assertEqual(data[8+size:], b'12345678')
            self.assertEqual(result['tensors'][0]['sha256'], hashlib.sha256(b'12345678').hexdigest())
            self.assertEqual(result['sha256'], file_sha(output))
            if importlib.util.find_spec('safetensors'):
                from safetensors import safe_open
                with safe_open(str(output), framework='pt') as tensors:
                    self.assertEqual(tuple(tensors.get_tensor('dense').shape), (2, 2))
            with self.assertRaises(FileExistsError):
                subset(source, output, ['dense'])
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                subset(source, root / 'truncated.safetensors', ['dense'])

    def test_changed_pack_or_false_serving_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = []
            for n in range(385):
                path = root / f'{n}.pt'
                path.write_bytes(b'pack')
                files.append(dict(filename=path.name, bytes=4, sha256=file_sha(path)))
            manifest = dict(stage='offline_packed_and_scored', serving_gptq_verified=False, packs=files)
            manifest_files(manifest, root)
            manifest['serving_gptq_verified'] = True
            with self.assertRaisesRegex(ValueError, 'serving claim'):
                manifest_files(manifest, root)
            manifest['serving_gptq_verified'] = False
            (root / '384.pt').write_bytes(b'fake')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                manifest_files(manifest, root)


@unittest.skipUnless(os.environ.get('ST_QWEN_5050_SMOKE') == '1', 'explicit synthetic 5050 smoke only')
class CudaPackSmoke(unittest.TestCase):
    def test_synthetic_pack_reload_and_heldout_energy(self):
        import torch
        from engine.kernels.dense.store import PackStore
        from engine.kernels.dense.packing import mk_w4_dequant
        from probes.qwen38_gptq_offline import verify_lane
        from probes.qwen38_gptq_score import output_error
        verify_lane()
        torch.set_num_threads(2)
        torch.manual_seed(919)
        torch.backends.cuda.matmul.allow_tf32 = False
        weight = torch.randn(128, 256, device='cuda').bfloat16()
        x = torch.randn(512, 256, device='cuda')
        held = torch.randn(71, 256, device='cuda', dtype=torch.float64)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'mkcalib/rank0/site.pt'
            path.parent.mkdir(parents=True)
            torch.save(dict(H=(x.T @ x).cpu(), ntok=512, weights_id='synthetic-smoke',
                            name='site', amax=x.abs().amax(0).cpu()), path)
            store = PackStore(root, 0, 'synthetic-smoke', require_identity=True)
            first = store.pack(weight, 'site')
            fp8 = store.pack_fp8(weight, 'site')
            second = PackStore(root, 0, 'synthetic-smoke', require_identity=True)
            reloaded = second.pack(weight, 'site')
            f_reloaded = second.pack_fp8(weight, 'site')
            self.assertTrue(first.calibrated and reloaded.calibrated)
            self.assertTrue(torch.equal(first.data, reloaded.data))
            self.assertTrue(torch.equal(fp8[0].view(torch.uint8), f_reloaded[0].view(torch.uint8)))
            self.assertEqual(second.stats['built'] + second.stats['fp8_built'], 0)
            q = mk_w4_dequant(first.data, first.scale, 128, rgs=first.rowscale)
            measured = output_error(weight, q, held.T @ held)['relative_rmse']
            explicit = float(((held @ (weight.double() - q.double()).T).square().sum()
                              / (held @ weight.double().T).square().sum()).sqrt())
            self.assertAlmostEqual(measured, explicit, places=10)
            store.release_pages()
            second.release_pages()


if __name__ == '__main__':
    unittest.main()
