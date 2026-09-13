"""CPU numerical and artifact contracts for the two optional draft changes."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from bench.draft_fc_bias import collect_fc_pairs, fit_fc_bias, fit_fc_bias_ranks
from engine.modules.draft_projection import project
from engine.profiles.glm53.draft_fc_bias import prepare_bias, reader_identity
from engine.profiles.glm53.draft_tuning import DraftTuning


def fixture(rank=0, world=1):
    source = torch.eye(2, dtype=torch.bfloat16)
    class Reader:
        weight = (source.clone(), torch.ones(1, 1))
        observer = None
        def __call__(self, x):
            return (x.float() + torch.tensor([1., -2.])).bfloat16()
    layer = SimpleNamespace(rows=2, cols=2, decode_precision='fp8', fp8=Reader(), decode_fp8=None)
    comm = SimpleNamespace(rank=rank, world_size=world, wait_prepared=lambda label: None,
                           gather_objects=lambda error: [error] * world)
    d = SimpleNamespace(F=SimpleNamespace(hidden=2, rms_eps=1e-6), target=SimpleNamespace(comm=comm),
        dense={'fc.weight': layer}, p={'fc.weight': source, 'hidden_norm.weight': torch.tensor([-1., 2.]).bfloat16()},
        tuning=DraftTuning())
    batches = [dict(aux=torch.tensor([[2., 3.], [float('nan'), 1.]]).bfloat16(),
                    keep=torch.tensor([True, False]), ids=[part, 'rejected'], split=part)
               for part in ('train', 'validation')]
    return d, batches


class ProjectionTests(unittest.TestCase):
    def test_output_rounding_can_hide_a_real_score_difference(self):
        x = torch.tensor([[1., 1 / 256]], dtype=torch.bfloat16)
        w = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.bfloat16)
        base, kept = project(x, w), project(x, w, fp32=True)
        self.assertEqual(base.tolist(), [[1., 1.]])
        self.assertEqual(kept.tolist(), [[1., 1.00390625]])
        self.assertEqual((base.argmax().item(), kept.argmax().item()), (0, 1))
        torch.testing.assert_close(kept.double(), x.double() @ w.double().T, rtol=0, atol=0)

    def test_empty_strided_and_invalid_inputs(self):
        x = torch.ones(7, 8, dtype=torch.bfloat16)[:, ::2]
        w = torch.ones(3, 4, dtype=torch.bfloat16)
        self.assertEqual(project(x, w, fp32=True).tolist(), [[4.] * 3] * 7)
        self.assertEqual(project(x[:0], w, fp32=True).shape, (0, 3))
        for a, b in ((x.float(), w), (x, w.float()), (x, w[:, :2]), (x[0], w)):
            with self.assertRaises(ValueError):
                project(a, b, fp32=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph numerics; no GPU is used in CPU CI')
    def test_fp32_projection_graph_replay_preserves_changed_close_scores(self):
        x = torch.zeros(7, 4096, dtype=torch.bfloat16, device='cuda')
        w = torch.zeros(256, 4096, dtype=torch.bfloat16, device='cuda')
        x[:, 0], x[:, 1], w[:, 0], w[1, 1] = 1., 1 / 256, 1., 1.
        for _ in range(3):
            project(x, w, fp32=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = project(x, w, fp32=True)
        try:
            for delta in (1 / 256, -1 / 256):
                x[:, 1] = delta
                graph.replay()
                expected = x.cpu().float() @ w.cpu().float().T
                torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        finally:
            graph.reset()


class BiasFitTests(unittest.TestCase):
    def test_cpu_cli_merges_rank_bundles_and_records_input_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / f'rank{rank}.pt' for rank in range(2)]
            for rank, path in enumerate(paths):
                torch.save(collect_fc_pairs(*fixture(rank, 2)), path)
            output = Path(root) / 'profile.json'
            result = subprocess.run([sys.executable, 'bench/draft_tune.py', 'fc-bias', str(paths[0]),
                '--peer-bundle', str(paths[1]), '--out', str(output)], cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text())
            self.assertEqual(set(DraftTuning.from_dict(value).fc_bias), {'0', '1'})
            self.assertEqual(len(value['evidence']['input_sha256']), 64)
            self.assertEqual(len(value['evidence']['peer_sha256']), 1)

    def test_collect_masks_ghost_rows_and_fits_the_correction_sign(self):
        d, batches = fixture()
        bundle = collect_fc_pairs(d, batches, max_rows=2)
        self.assertEqual(bundle['train']['ids'], ['train'])
        self.assertEqual(bundle['train']['actual'].tolist(), [[3., 1.]])
        result = fit_fc_bias_ranks([bundle])
        self.assertTrue(result['evidence']['selected'])
        self.assertEqual(result['fc_bias']['0']['values'], (-1., 2.))
        report = result['evidence']['ranks']['0']
        self.assertEqual(report['candidate'], dict(fc_error=0., norm_error=0.))
        self.assertFalse(report['live_acceptance'])
        with self.assertRaisesRegex(ValueError, 'budget'):
            collect_fc_pairs(d, batches, max_rows=1)

    def test_validation_can_veto_but_cannot_fit_the_vector(self):
        d, batches = fixture()
        bundle = collect_fc_pairs(d, batches)
        bundle['validation']['actual'] = torch.tensor([[1., 5.]]).bfloat16()
        result = fit_fc_bias(bundle)
        self.assertEqual(result['fc_bias'], {})
        self.assertFalse(result['evidence']['selected'])
        bundle['validation']['ids'] = ['train']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            fit_fc_bias(bundle)

    def test_normalized_error_can_veto_improved_fc_error(self):
        d, batches = fixture()
        bundle = collect_fc_pairs(d, batches)
        # Baseline is a scalar multiple of the reference and has exactly the
        # right direction. This mean bias improves FC MSE but hurts RMSNorm.
        bundle['train']['reference'] = torch.tensor([[2., 2.]]).bfloat16()
        bundle['train']['actual'] = torch.tensor([[3., 2.]]).bfloat16()
        bundle['validation']['reference'] = torch.tensor([[2., 2.]]).bfloat16()
        bundle['validation']['actual'] = torch.tensor([[4., 4.]]).bfloat16()
        result = fit_fc_bias(bundle)
        self.assertLess(result['evidence']['candidate']['fc_error'], result['evidence']['baseline']['fc_error'])
        self.assertGreater(result['evidence']['candidate']['norm_error'], result['evidence']['baseline']['norm_error'])
        self.assertEqual(result['fc_bias'], {})

    def test_fit_requires_finite_paired_rows_and_complete_distinct_ranks(self):
        a = collect_fc_pairs(*fixture(0, 2))
        b = collect_fc_pairs(*fixture(1, 2))
        result = fit_fc_bias_ranks([b, a])
        self.assertEqual(set(result['fc_bias']), {'0', '1'})
        for bundles in ([a], [a, a]):
            with self.assertRaisesRegex(ValueError, 'every TP rank'):
                fit_fc_bias_ranks(bundles)
        bad = copy.deepcopy(b)
        bad['train']['ids'], bad['validation']['ids'] = ['validation'], ['train']
        with self.assertRaisesRegex(ValueError, 'across TP'):
            fit_fc_bias_ranks([a, bad])
        bad = copy.deepcopy(a)
        bad['train']['actual'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite paired'):
            fit_fc_bias(bad)


class BindingTests(unittest.TestCase):
    def test_profile_rejects_malformed_corrections(self):
        for bias in ([], {'0': {}}, {'01': {}}, {'0': dict(reader_sha256='x', values=[0.])},
                     {'0': dict(reader_sha256='a' * 64, values=[float('inf')])},
                     {'0': dict(reader_sha256='a' * 64, values=[True])}):
            with self.assertRaises(ValueError):
                DraftTuning.from_dict(dict(version=1, fc_bias=bias))
        with self.assertRaisesRegex(ValueError, 'must be bool'):
            DraftTuning.from_dict(dict(version=1, selector_projection_fp32=1))

    def test_default_does_no_preparation_and_identity_pins_each_reader_input(self):
        d, _ = fixture()
        with patch('engine.profiles.glm53.draft_fc_bias.reader_identity', side_effect=AssertionError('unused')):
            self.assertIsNone(prepare_bias(d))
        def identity():
            return reader_identity(d.dense['fc.weight'], d.p['fc.weight'], d.p['hidden_norm.weight'], d.F.rms_eps)
        first = identity()
        for tensor in (d.p['fc.weight'], d.p['hidden_norm.weight'], *d.dense['fc.weight'].fp8.weight):
            saved = tensor.clone()
            tensor.reshape(-1)[0] += 1
            self.assertNotEqual(identity(), first)
            tensor.copy_(saved)
        d.F.rms_eps *= 2
        self.assertNotEqual(identity(), first)

    def test_binding_uses_own_rank_and_peer_errors_are_reported(self):
        d, batches = fixture(1, 2)
        bundle = collect_fc_pairs(d, batches)
        entry = fit_fc_bias(bundle)['fc_bias']['1']
        d.tuning = replace(d.tuning, fc_bias={'0': dict(entry, reader_sha256='0' * 64), '1': entry})
        self.assertEqual(prepare_bias(d).tolist(), [-1., 2.])
        d.target.comm.gather_objects = lambda error: ['peer reader mismatch', error]
        with self.assertRaisesRegex(ValueError, 'rank 0: peer reader mismatch'):
            prepare_bias(d)
        d.target.comm.gather_objects = lambda error: [error, error]
        d.p['fc.weight'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            prepare_bias(d)

    def test_w4_bias_is_refused_even_if_the_vector_is_well_formed(self):
        d, batches = fixture()
        d.tuning = DraftTuning.from_dict(fit_fc_bias(collect_fc_pairs(d, batches)))
        d.dense['fc.weight'].decode_precision = 'w4'
        with self.assertRaisesRegex(ValueError, 'fixed FP8'):
            prepare_bias(d)


if __name__ == '__main__':
    unittest.main()
