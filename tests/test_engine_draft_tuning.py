"""CPU contracts for the optional low-cost controls and their held-out fitting."""
import copy
import json
import math
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from bench.draft_tune import fit_packing, fit_selector, tag_selector
from engine.modules.draft_boundary import EMPTY, WIDTH, for_request, mask_ends, sampled, tensor
from engine.profiles.glm53.draft_diagnostics import DraftDiagnostics
from engine.profiles.glm53.draft_tuning import DraftTuning, load_agreed


class ProfileTests(unittest.TestCase):
    def test_baseline_and_explicit_names(self):
        self.assertEqual(DraftTuning().alphas(6), (1.,) * 6)
        for tuning in (DraftTuning(), DraftTuning.from_dict(dict(version=1))):
            self.assertTrue(tuning.selector_projection_fp32)
            self.assertTrue(tuning.fc_bias_auto)
            self.assertEqual(tuning.fc_bias, {})
        comm = SimpleNamespace(wait_prepared=lambda stage: None, gather_objects=lambda x: [x])
        self.assertIsInstance(load_agreed('', None, None, comm), DraftTuning)
        profile = DraftTuning.from_dict(dict(version=1, selector_alpha=[.75],
            smoothing_alpha={'layers.0.input_layernorm.weight': .25},
            gptq_damping={'layers.0.self_attn.qkv': .02}, request_boundaries=True))
        profile.validate(SimpleNamespace(k=6, layers=1), ['layers.0.self_attn.qkv'])
        self.assertEqual(profile.alphas(6), (.75,) * 6)
        with self.assertRaisesRegex(ValueError, 'names'):
            profile.validate(SimpleNamespace(k=6, layers=0), [])

    def test_malformed_profiles_cannot_silently_change_defaults(self):
        for values in ({'version': True}, {'selector_alpha': [float('nan')]},
                       {'selector_alpha': [-.1]}, {'selector_alpha': [True]},
                       {'gptq_damping': {'fc.weight': 0}}, {'smoothing_alpha': {'x': 1.1}},
                       {'trace_every': True}, {'request_boundaries': 1}, {'unknown': 1},
                       {'fc_bias_auto': 1}, {'selector_projection_fp32': 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DraftTuning.from_dict(dict({'version': 1}, **values))
        with self.assertRaisesRegex(ValueError, 'exactly 6'):
            DraftTuning.from_dict(dict(version=1, selector_alpha=[1., .5])).alphas(6)

    def test_profile_identity_and_errors_are_agreed_before_packing(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'profile.json'
            path.write_text(json.dumps(dict(version=1, selector_alpha=[.5])))
            calls = []
            comm = SimpleNamespace(wait_prepared=calls.append, gather_objects=lambda x: [x, x])
            facts = SimpleNamespace(k=3, layers=1)
            self.assertEqual(load_agreed(path, facts, [], comm).alphas(3), (.5,) * 3)
            self.assertEqual(calls, ['draft-tuning'])
            comm.gather_objects = lambda x: [x, dict(digest='other', error=None)]
            with self.assertRaisesRegex(ValueError, 'different profile digests'):
                load_agreed(path, facts, [], comm)
            with self.assertRaisesRegex(ValueError, 'different profile digests'):
                load_agreed('', facts, [], comm)
            comm.gather_objects = lambda x: [x, dict(digest=None, error='bad peer JSON')]
            with self.assertRaisesRegex(ValueError, 'rank 1: bad peer'):
                load_agreed(path, facts, [], comm)
            path.write_text('broken')
            comm.gather_objects = lambda x: [x]
            with self.assertRaisesRegex(ValueError, 'rank 0: JSONDecodeError'):
                load_agreed(path, facts, [], comm)


class SelectorFitTests(unittest.TestCase):
    def test_fitted_alpha_preserves_projection_precision_and_refuses_mixed_records(self):
        self.assertFalse(fit_selector(self.rows())['selector_projection_fp32'],
                         'legacy BF16 traces must not be reinterpreted using new serving defaults')
        records = [dict(row, selector_projection_fp32=True) for row in self.rows()]
        self.assertTrue(fit_selector(records)['selector_projection_fp32'])
        records[0]['selector_projection_fp32'] = False
        with self.assertRaisesRegex(ValueError, 'mix projection precision'):
            fit_selector(records)

    def rows(self):
        return [dict(kind='draft_selector', rank=0, sample_group=split, split=split, draft_width=3,
                     target=[10, 10], candidates=[[10, 11], [10, 11]],
                     unary=[[2., 0.], [2., 0.]], edge=[[0., 3.], [0., 3.]])
                for split in ('train', 'validation')]

    def test_position_fit_requires_a_held_out_gain_and_keeps_unobserved_positions(self):
        rows = self.rows()
        rows[1]['target'][1] = 11  # training's proposed change regresses this held-out position
        got = fit_selector(rows)
        self.assertEqual(got['selector_alpha'], [.5, 1., 1.])
        self.assertFalse(got['evidence']['live_acceptance'])
        self.assertEqual(got['evidence']['positions'][2]['validation_rows'], 0)
        rows[1]['target'][0] = 99  # both miss coverage: no pretend improvement on a tie
        self.assertEqual(fit_selector(rows)['selector_alpha'], [1., 1., 1.])

    def test_groups_cannot_leak_and_nonroot_or_policy_rows_do_not_fit(self):
        rows = self.rows()
        rows += [dict(rows[0], rank=1, unary=[[float('nan')]])]
        rows += [dict(rows[0], policy_modified=True, split='invalid')]
        self.assertEqual(fit_selector(rows)['selector_alpha'], [.5, .5, 1.])
        rows[1]['sample_group'] = 'train'
        with self.assertRaisesRegex(ValueError, 'both train and validation'):
            fit_selector(rows)
        with self.assertRaisesRegex(ValueError, 'independent'):
            fit_selector(self.rows()[:1])

    def test_recording_manifest_tags_request_families_without_editing_raw_trace(self):
        rows = [dict(row, request_token='recording', seq=i) for i, row in enumerate(self.rows())]
        manifest = {f'recording:{i}': dict(sample_group=f'prompt-{i}', split=row['split'])
                    for i, row in enumerate(rows)}
        tagged = tag_selector(rows, manifest)
        self.assertEqual(tagged[0]['sample_group'], 'prompt-0')
        self.assertEqual(rows[0]['sample_group'], 'train')
        self.assertEqual(fit_selector(tagged)['selector_alpha'], [.5, .5, 1.])
        with self.assertRaisesRegex(ValueError, 'missing request-family'):
            tag_selector(rows, {})


class BoundaryTests(unittest.TestCase):
    def request(self):
        return SimpleNamespace(drafter=SimpleNamespace(k=3, request_boundaries=True), options={1: {}},
            matchers={}, _generated_count=lambda seq: 4, min_new={1: 6}, ends={1: [7, 7]}, eos={7},
            thinking={1: True}, tokens={1: [3, 4, 5, 6, 6]}, prompt_len={1: 1}, decodable=8,
            F=SimpleNamespace(vocab=8))

    def test_packet_matches_minimum_budget_and_committed_thinking_state(self):
        e = self.request()
        e.options[1] = dict(reasoning_budget=5, reasoning_end=7)
        packet = for_request(e, 1)
        self.assertEqual(packet[:4], (2, 1, 7, 7))
        self.assertEqual(packet.count(7), 2, 'one force id and one unique end id')
        self.assertEqual(tensor(packet, 'cpu').shape, (WIDTH,))
        e.tokens[1].append(7)
        self.assertEqual(for_request(e, 1)[2], -1)
        e.options[1]['reasoning_end'] = 20
        self.assertEqual(for_request(e, 1)[2], -1, 'never force an out-of-vocab codebook read')
        e.options[1]['grammar'] = {'type': 'json_object'}
        self.assertIsNone(for_request(e, 1))
        e.options[1].clear()
        e.ends[1] = range(33)
        self.assertIsNone(for_request(e, 1))
        e.drafter.request_boundaries = False
        self.assertIsNone(for_request(e, 1))

    def test_eos_is_removed_before_topk_on_each_shard_with_replacement(self):
        packet = tensor((1, 0, -1, 1, 5) + (-1,) * 30, 'cpu')
        for start in (0, 4):
            logits = torch.tensor([[1., 100., 3., 2.], [1., 100., 3., 2.]])
            mask_ends(logits, start, packet)
            self.assertEqual(logits.topk(2, -1).indices.tolist(), [[2, 3], [1, 2]])
        with self.assertRaisesRegex(ValueError, 'geometry'):
            tensor([0], 'cpu')

    def walk(self, packet, first=(1, 2)):
        cand = torch.tensor([first, (1, 2), (1, 2)])
        unary = torch.tensor([[0., 20.], [0., 0.], [0., 0.]])
        pred = torch.tensor([[0.], [0.], [-20.], [0.], [20.], [0.], [0.], [0.]])
        succ = torch.tensor([[0.], [1.], [-1.], [0.], [0.], [0.], [0.], [0.]])
        return sampled(unary, cand, torch.tensor([0]), torch.ones(3, 1), pred, succ,
                       (1.,) * 3, 1., torch.tensor([.5, .5, .5]), tensor(packet, 'cpu'))

    def test_sampled_forcing_preserves_unique_support_and_uses_actual_predecessor(self):
        packet = (0, 0, 4) + (-1,) * 32
        for first in ((1, 2), (1, 4)):
            tokens, support, probs = self.walk(packet, first)
            self.assertEqual(tokens[:2].tolist(), [4, 1])
            self.assertEqual(probs[0].sum().item(), 1.)
            self.assertEqual(probs[0][support[0] == 4].item(), 1.)
            for ids, p, token in zip(support, probs, tokens):
                self.assertEqual(len(ids.unique()), len(ids))
                self.assertGreater(p[ids == token].item(), 0)
                self.assertAlmostEqual(p.sum().item(), 1., places=6)
        tokens, _, _ = self.walk((2, 0, 4, 4) + (-1,) * 31)
        self.assertEqual(tokens.tolist(), [2, 2, 4], 'minimum-token exclusion precedes reasoning forcing')
        tokens, _, _ = self.walk((0, 0, 999) + (-1,) * 32)
        self.assertEqual(tokens[0].item(), 2)

    def test_spontaneous_end_cancels_future_force(self):
        tokens, _, _ = self.walk((0, 1, 4) + (-1,) * 32, first=(1, 4))
        self.assertEqual(tokens[:2].tolist(), [4, 1])


class TraceTests(unittest.TestCase):
    def test_root_trace_records_only_observable_unconstrained_labels_and_samples_steps(self):
        d = DraftDiagnostics(torch.zeros(3, 2), 3, 2)
        d.enable_selector_trace(2, 'profile', rank=0)
        d.support[2] = torch.tensor([[10, 11], [12, 13], [14, 15]])
        d.selector_trace[0][2] = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        d.selector_trace[1][2].fill_(.5)
        records = []
        d.sink = lambda **row: records.append(row)
        for context in (100, 102, 104):
            d.note_sync(1, context, 2, 1, [10, 13, 99], 50, set())
        trace = [r for r in records if r['kind'] == 'draft_selector']
        self.assertEqual([r['context'] for r in trace], [100, 104])
        self.assertEqual(trace[0]['target'], [10, 13])
        self.assertEqual(trace[0]['unary'], [[1., 2.], [3., 4.]])
        d.note_sync(1, 200, 2, 0, [10], 50, set(), policy_modified=True)
        d.note_sync(1, 200, 2, 0, [10], 50, set(), trace_eligible=False)
        d.trace_rank = 1
        d.note_sync(1, 200, 2, 0, [10], 50, set())
        self.assertEqual(len([r for r in records if r['kind'] == 'draft_selector']), 2)
        d.trace_rank = 0
        for seq in range(2, 20):
            d.note_sync(seq, 200, 2, 0, [10], 50, set())
        self.assertEqual(len(d.trace_steps), 1, 'retired sequence ids do not accumulate')
        self.assertEqual(len([r for r in records if r['kind'] == 'draft_selector']), 20)
        d.close()
        self.assertIsNone(d.selector_trace)

    def test_debug_selector_features_append_hidden_projection_anchor_and_bonus(self):
        # debug (never merge): the feature file holds positions [0, count) as raw bf16 hidden then raw fp32 projection
        import tempfile
        d = DraftDiagnostics(torch.zeros(3, 2), 3, 2)
        d.enable_selector_trace(1, 'profile', rank=0, features=(5, 4))
        d.support[2] = torch.tensor([[10, 11], [12, 13], [14, 15]])
        d.selector_trace[0][2].fill_(1.)
        d.selector_trace[1][2].fill_(.5)
        hidden = torch.arange(15, dtype=torch.float32).view(3, 5).bfloat16()
        projection = torch.arange(12, dtype=torch.float32).view(3, 4) / 7
        d.selector_features[0][2].copy_(hidden)
        d.selector_features[1][2].copy_(projection)
        d.selector_features[2][2] = 777
        records = []
        d.sink = lambda **row: records.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            d.feature_dir = tmp
            d.note_sync(1, 100, 2, 1, [10, 13, 99], 50, set())          # accepted 1: labels 2 positions, no bonus
            d.note_sync(1, 102, 2, 3, [10, 12, 14, 42], 50, set())      # all 3 accepted: 3 positions, bonus 42
            rows = [r for r in records if r['kind'] == 'draft_selector']
            with open(f'{tmp}/selector-features.bin', 'rb') as f:
                blob = f.read()
        self.assertEqual([r['anchor'] for r in rows], [777, 777])
        self.assertEqual([r['bonus'] for r in rows], [None, 42])
        self.assertEqual([r['feature_offset'] for r in rows], [0, 2 * (5 * 2 + 4 * 4)])
        second = blob[rows[1]['feature_offset']: rows[1]['feature_offset'] + rows[1]['feature_bytes']]
        h = torch.frombuffer(bytearray(second[:3 * 5 * 2]), dtype=torch.bfloat16).view(3, 5)
        p = torch.frombuffer(bytearray(second[3 * 5 * 2:]), dtype=torch.float32).view(3, 4)
        self.assertTrue(torch.equal(h, hidden))
        self.assertTrue(torch.equal(p, projection))
        d.close()
        self.assertIsNone(d.selector_features)


class PackingFitTests(unittest.TestCase):
    def bundle(self):
        g = torch.Generator().manual_seed(35)
        w = torch.randn(8, 128, generator=g).bfloat16()
        return dict(train_ids=['calibration'], selection_ids=['selection'] * 2,
            validation_ids=['validation'] * 2, H=torch.eye(128), amax=torch.ones(128) * 3,
            weight_peaks=w.float().abs().amax(0), norm_weight=torch.ones(128).bfloat16(),
            selection=torch.randn(2, 128, generator=g).bfloat16(),
            validation=torch.randn(2, 128, generator=g).bfloat16(),
            weights={'layers.0.self_attn.qkv': w}, norm_key='layers.0.input_layernorm.weight')

    def test_actual_cpu_pack_fit_is_read_only_and_profiles_preserve_format(self):
        bundle = self.bundle()
        original = copy.deepcopy(bundle)
        got = fit_packing(bundle, alphas=(.25, .5), dampings=(.01,))
        DraftTuning.from_dict(got).validate(SimpleNamespace(k=6, layers=1), bundle['weights'])
        self.assertFalse(got['evidence']['live_acceptance'])
        self.assertEqual(len(got['evidence']['candidates']), 2)
        self.assertTrue(math.isfinite(got['evidence']['baseline']['audit_error']))
        self.assertLessEqual(got['evidence']['selected']['audit_error'], got['evidence']['baseline']['audit_error'])
        for key in ('norm_weight', 'H', 'amax', 'selection', 'validation'):
            self.assertTrue(torch.equal(bundle[key], original[key]))
        self.assertTrue(torch.equal(bundle['weights']['layers.0.self_attn.qkv'], original['weights']['layers.0.self_attn.qkv']))

    def test_no_fitting_on_overlapping_requests_or_wrong_reader_statistics(self):
        for problem in ('leak', 'missing_ids', 'peaks', 'reader'):
            bundle = self.bundle()
            if problem == 'leak':
                bundle['selection_ids'][0] = 'calibration'
            elif problem == 'missing_ids':
                bundle['validation_ids'].pop()
            elif problem == 'peaks':
                bundle['weight_peaks'].zero_()
            else:
                bundle['norm_key'] = 'layers.0.post_attention_layernorm.weight'
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                fit_packing(bundle)

    def test_damping_changes_factor_and_pack_identity_only_for_the_tuned_reader(self):
        from engine.kernels.dense.store import PackStore
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 0)
            h = torch.tensor([[2., .5], [.5, 1.]])
            first = store._factor('draft', h, 'none', 'cpu')
            self.assertIs(store._factor('draft', h, 'none', 'cpu'), first)
            self.assertEqual(store._tuning_identity('draft'), {})
            store.gptq_damping['draft'] = .02
            second = store._factor('draft', h, 'none', 'cpu')
            self.assertFalse(torch.equal(first[1], second[1]))
            self.assertEqual(store._tuning_identity('draft'), {'gptq_damping': .02})
            self.assertEqual(store._tuning_identity('target'), {})
            self.assertEqual(store.stats['factor_built'], 2)
            self.assertEqual(store.stats['factor_reused'], 1)

    def test_tuned_pack_cannot_reuse_default_cache_or_unattested_legacy_bytes(self):
        from engine.kernels.dense import W4Pack
        from engine.kernels.dense.store import PackStore
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 0)
            name = 'DFlash2Qwen3ForCausalLM/model.layers.0.self_attn.qkv_proj'
            weight, h = torch.ones(8, 128).bfloat16(), torch.eye(128)
            path = store.calibration_path(name)
            path.parent.mkdir(parents=True)
            torch.save(dict(H=h, ntok=128, name=name), path)
            def packed(w, **kw):
                return W4Pack(torch.zeros(1, 1, 128, 64, dtype=torch.uint8),
                    torch.zeros(1, 1, 128, 8, dtype=torch.int8), torch.ones(128), 8, 128, True)
            with patch('engine.kernels.dense.pack_w4', side_effect=packed) as build:
                store.pack(weight, name)
                digest = hashlib.sha256(weight.view(torch.uint8).numpy()).hexdigest()
                legacy = Path(root) / 'mkpacks/rank0' / f'sha256-{digest}-8x128-bfloat16-v4-ten-gptq-lr0.pt'
                legacy.parent.mkdir(parents=True)
                legacy.write_bytes(b'not a trusted tuned pack')
                store.gptq_damping[name] = .02
                store.pack(weight, name)
                store.pack(weight, name)
                store.gptq_damping.clear()
                store.pack(weight, name)
                self.assertEqual(build.call_count, 2)
            blobs = [torch.load(p, weights_only=True)['identity'] for p in (Path(root) / 'st-dense-packs').glob('*.pt')]
            self.assertEqual(len(blobs), 2)
            self.assertEqual(sorted(b.get('gptq_damping', .01) for b in blobs), [.01, .02])
            self.assertNotIn(legacy, store.read_files)


if __name__ == '__main__':
    unittest.main()
