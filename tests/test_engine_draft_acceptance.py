"""Independent FC precision/calibration arms and real first-rejection causes."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from engine.kernels.dense import DenseLinear
from engine.kernels.dense.calibration import Calibration, ROWS_FLOOR
from engine.kernels.dense.store import PackStore
from engine.profiles.glm53.draft_policy import (DraftPolicy, SERVING_POLICY, decode_name,
                                             require_decode_calibration, resolve_calibration)
from engine.profiles.glm53.draft_diagnostics import DraftDiagnostics, classify


class DraftAcceptanceTests(unittest.TestCase):
    def test_serving_boot_collects_missing_statistics_then_consumes_on_the_next_boot(self):
        from engine.base.comm import Comm
        with tempfile.TemporaryDirectory() as root:
            store, name, cols = PackStore(root, 0), 'draft/model.fc', 16
            first = resolve_calibration(SERVING_POLICY, store, name, cols, Comm())
            self.assertEqual(first, DraftPolicy('fp8', 'collect', True))
            self.assertFalse(first.separate_decode_fp8)
            with self.assertRaisesRegex(ValueError, 'missing'):
                resolve_calibration(DraftPolicy('fp8', 'decode', True), store, name, cols, Comm())
            key = decode_name(name)
            path = store.calibration_path(key)
            path.parent.mkdir(parents=True)
            torch.save(dict(name=key, input_scope='committed_decode_v1', ntok=ROWS_FLOOR,
                            H=torch.eye(cols), amax=torch.ones(cols)), path)
            original = path.read_bytes()
            second = resolve_calibration(SERVING_POLICY, store, name, cols, Comm())
            self.assertEqual(second, DraftPolicy('fp8', 'decode', True))
            self.assertTrue(second.separate_decode_fp8)
            self.assertEqual(path.read_bytes(), original)

    def test_one_missing_rank_keeps_every_rank_on_collection_without_overwriting_ready_files(self):
        with tempfile.TemporaryDirectory() as root:
            store, name, cols = PackStore(root, 0), 'draft/model.fc', 16
            key = decode_name(name)
            path = store.calibration_path(key)
            path.parent.mkdir(parents=True)
            torch.save(dict(name=key, input_scope='committed_decode_v1', ntok=ROWS_FLOOR,
                            H=torch.eye(cols), amax=torch.ones(cols)), path)
            original, calls = path.read_bytes(), []
            def gather(report):
                self.assertEqual(calls, ['draft-calibration'])
                self.assertEqual(report, dict(ready=True, error=None))
                return [report, dict(ready=False, error=None)]
            comm = SimpleNamespace(wait_prepared=calls.append, gather_objects=gather)
            got = resolve_calibration(SERVING_POLICY, store, name, cols, comm)
            self.assertEqual(got.fc_calibration, 'collect')
            self.assertEqual(store.missing_calibration(key, cols), [])
            self.assertEqual(path.read_bytes(), original)
            calls.clear()
            with self.assertRaisesRegex(ValueError, 'completed statistics on every rank'):
                resolve_calibration(DraftPolicy('fp8', 'decode', True), store, name, cols, comm)

    def test_corrupt_local_or_peer_statistics_fail_all_ranks_instead_of_downgrading_to_rtn(self):
        from engine.base.comm import Comm
        with tempfile.TemporaryDirectory() as root:
            store, name, cols = PackStore(root, 0), 'draft/model.fc', 16
            key = decode_name(name)
            path = store.calibration_path(key)
            path.parent.mkdir(parents=True)
            path.write_bytes(b'not a torch checkpoint')
            with self.assertRaisesRegex(ValueError, 'TP decode FC calibration failed: rank 0'):
                resolve_calibration(SERVING_POLICY, store, name, cols, Comm())
            path.unlink()
            calls = []
            comm = SimpleNamespace(wait_prepared=calls.append,
                gather_objects=lambda report: [report, dict(ready=False, error='bad scope')])
            with self.assertRaisesRegex(ValueError, 'rank 1: bad scope'):
                resolve_calibration(SERVING_POLICY, store, name, cols, comm)
            self.assertEqual(calls, ['draft-calibration'])
            # Fixed baseline/collector arms do not introduce a preparation vote.
            for mode in ('shared', 'collect'):
                policy = DraftPolicy(fc_calibration=mode)
                self.assertIs(resolve_calibration(policy, None, name, cols, None), policy)

    def test_arms_are_independent_and_combined_mode_is_explicit(self):
        self.assertFalse(DraftPolicy().active)
        self.assertTrue(DraftPolicy(fc_precision='fp8').active)
        self.assertTrue(DraftPolicy(fc_calibration='collect').active)
        self.assertTrue(DraftPolicy(diagnostics=True).active)
        self.assertTrue(DraftPolicy('fp8', 'decode', True).separate_decode_fp8)
        self.assertFalse(DraftPolicy('fp8', 'collect', True).separate_decode_fp8)
        for options in ({'fc_precision': 'bf16'}, {'fc_calibration': 'oops'}, {'diagnostics': 1}):
            with self.assertRaises(ValueError):
                DraftPolicy(**options)

    def test_fc_fp8_reuses_its_pack_and_observes_exactly_once(self):
        layer = DenseLinear.__new__(DenseLinear)
        layer.cols, layer.rows, layer.executed = 128, 8, 0
        layer.workspace, layer.packs = None, (object(),)
        observed, fp8_calls, w4_calls = [], [], []
        layer.observer = lambda x, mask: observed.append((x.clone(), mask))
        layer.fp8 = lambda x: fp8_calls.append(len(x)) or x[:, :8] + 2
        x = torch.ones(7, 128, dtype=torch.bfloat16)
        mask = torch.arange(7) < 3
        def w4(x, pack, workspace, *, bound_input=False):
            self.assertFalse(bound_input)  # target-model fastpaths do not opt the drafter in
            w4_calls.append(len(x))
            return x[:, :8] + 1
        with patch('engine.kernels.dense.w4_gemm', side_effect=w4):
            layer.decode_precision = 'w4'
            self.assertTrue(torch.equal(layer(x, mask), x[:, :8] + 1))
            layer.decode_precision = 'fp8'
            self.assertTrue(torch.equal(layer(x, mask), x[:, :8] + 2))
            layer(x, mask, observe=False)
            layer.decode_precision = 'w4'
            layer(torch.ones(64, 128, dtype=torch.bfloat16))
        self.assertEqual(w4_calls, [7])
        self.assertEqual(fp8_calls, [7, 7, 64])
        self.assertEqual(len(observed), 3)
        self.assertIs(observed[1][1], mask)
        self.assertEqual(layer.executed, 3)

    def test_decode_collection_ignores_prefill_capture_and_rejected_rows_and_preserves_shared_blob(self):
        torch.manual_seed(81)
        cols, name = 16, 'draft/model.fc'
        key = decode_name(name)
        layer = SimpleNamespace(input_dtype=torch.bfloat16, observer=None)
        c = Calibration('cpu', budget_bytes=1 << 20)
        c.attach(key, layer, PackStore.tiles(key, cols), True, decode_only=True)
        x = torch.randn(7, cols).bfloat16()
        mask = torch.arange(7) < 3
        layer.observer(x, mask)  # capture/preparation, disarmed
        c.arm()
        layer.observer(torch.randn(2048, cols).bfloat16(), None)
        layer.observer(x, None)  # even a 7-token prompt is not decode
        layer.observer(torch.randn(64, cols).bfloat16(), torch.ones(64, dtype=torch.bool))
        self.assertEqual(c.progress(), 0)
        layer.observer(x, mask)
        self.assertEqual(c.progress(), 3)
        torch.testing.assert_close(c.H[key], x[:3].float().T @ x[:3].float())
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 0)
            shared = store.calibration_path(name)
            shared.parent.mkdir(parents=True)
            torch.save({'H': torch.eye(cols), 'amax': torch.ones(cols), 'ntok': ROWS_FLOOR, 'name': name}, shared)
            original = shared.read_bytes()
            with self.assertRaisesRegex(ValueError, 'missing'):
                require_decode_calibration(store, name, cols)
            c.rows[key].fill_(ROWS_FLOOR)
            saved, = c.save(root, 0)
            self.assertNotEqual(saved, shared)
            self.assertEqual(shared.read_bytes(), original)
            self.assertEqual(require_decode_calibration(store, name, cols), key)
            blob = torch.load(saved, weights_only=True)
            self.assertEqual(blob['input_scope'], 'committed_decode_v1')
            blob.pop('input_scope')
            torch.save(blob, saved)
            with self.assertRaisesRegex(ValueError, 'scope'):
                require_decode_calibration(store, name, cols)

    def test_only_w4_pack_uses_decode_calibration_fp8_keeps_shared_identity(self):
        weight = SimpleNamespace(ndim=2, is_cuda=True, dtype=torch.bfloat16, shape=(128, 8192))
        w4_names, fp8_names = [], []
        store = SimpleNamespace(calibrated=lambda name: True,
            pack_wide=lambda w, name, **kw: w4_names.append(name) or [SimpleNamespace(calibrated=True)],
            pack_fp8=lambda w, name, **kw: fp8_names.append(name) or ('q', 'scale'))
        with patch('engine.kernels.dense.extension'), patch('engine.kernels.dense._fold', side_effect=lambda x: x), \
             patch('engine.kernels.dense.FP8Linear'):
            DenseLinear(weight, store=store, name='shared', decode_name='decoded')
        self.assertEqual(w4_names, ['decoded'])
        self.assertEqual(fp8_names, ['shared'])

    def test_combined_mode_calibrates_executed_fp8_and_preserves_shared_packs(self):
        weight = SimpleNamespace(ndim=2, is_cuda=True, dtype=torch.bfloat16, shape=(128, 8192))
        w4_names, fp8_names = [], []
        store = SimpleNamespace(calibrated=lambda name: True,
            pack_wide=lambda w, name, **kw: w4_names.append(name) or [SimpleNamespace(calibrated=True)],
            pack_fp8=lambda w, name, **kw: fp8_names.append(name) or ('q', 'scale'))
        with patch('engine.kernels.dense.extension'), patch('engine.kernels.dense._fold', side_effect=lambda x: x), \
             patch('engine.kernels.dense.FP8Linear'):
            layer = DenseLinear(weight, store=store, name='shared', decode_name='decoded', decode_precision='fp8')
        self.assertEqual(w4_names, [])
        self.assertEqual(layer.packs, ())
        self.assertFalse(layer.calibrated, 'an absent W4 pack must not count as GPTQ coverage')
        with self.assertRaisesRegex(ValueError, 'prepared W4'):
            layer.isolate_workspace()
        self.assertEqual(fp8_names, ['shared', 'decoded'])
        calls = []
        layer.fp8 = lambda x: calls.append('prefill') or x[:, :128] + 1
        layer.decode_fp8 = lambda x: calls.append('decode') or x[:, :128] + 2
        observed = []
        layer.observer = lambda x, mask: observed.append(mask)
        x = torch.ones(7, 8192, dtype=torch.bfloat16)
        # Identical row counts: a short prompt must keep shared calibration.
        self.assertTrue(torch.equal(layer(x), x[:, :128] + 1))
        self.assertTrue(torch.equal(layer(x, decode=True), x[:, :128] + 2))
        layer(x, decode=True, observe=False)
        layer(torch.ones(56, 8192, dtype=torch.bfloat16), decode=True)
        self.assertEqual(calls, ['prefill', 'decode', 'decode', 'decode'])
        self.assertEqual(len(observed), 3)
        store.pack_fp8 = lambda w, name, **kw: None
        with patch('engine.kernels.dense.extension'), patch('engine.kernels.dense._fold', side_effect=lambda x: x), \
             patch('engine.kernels.dense.FP8Linear'), self.assertRaisesRegex(ValueError, 'completed decode'):
            DenseLinear(weight, store=store, name='shared', decode_name='decoded', decode_precision='fp8')

    def test_decode_calibration_refuses_corruption_before_large_allocation(self):
        cols, name = 384, 'draft/model.fc'
        key = decode_name(name)
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 0)
            path = store.calibration_path(key)
            path.parent.mkdir(parents=True)
            def valid():
                return dict(name=key, ntok=ROWS_FLOOR, input_scope='committed_decode_v1',
                            H=torch.eye(cols), amax=torch.ones(cols))
            torch.save(valid(), path)
            real_isfinite, sizes = torch.isfinite, []
            def finite(x):
                sizes.append(x.numel())
                return real_isfinite(x)
            with patch.object(torch, 'isfinite', side_effect=finite):
                self.assertEqual(require_decode_calibration(store, name, cols), key)
            self.assertLessEqual(max(sizes), cols * 128)
            self.assertIn(path, store.read_files)
            for problem in ('foreign', 'few_rows', 'wrong_peaks', 'nan_off_diagonal',
                            'infinite_peak', 'negative_diagonal', 'empty', 'narrow_hessian', 'malformed'):
                with self.subTest(problem=problem):
                    blob = valid()
                    if problem == 'foreign':
                        blob['name'] = 'another/model.fc'
                    elif problem == 'few_rows':
                        blob['ntok'] = ROWS_FLOOR - 1
                    elif problem == 'wrong_peaks':
                        blob['amax'] = torch.ones(1)
                    elif problem == 'nan_off_diagonal':
                        blob['H'][257, 3] = float('nan')
                    elif problem == 'infinite_peak':
                        blob['amax'][0] = float('inf')
                    elif problem == 'negative_diagonal':
                        blob['H'][0, 0] = -1
                    elif problem == 'empty':
                        blob['H'].zero_()
                        blob['amax'].zero_()
                    elif problem == 'narrow_hessian':
                        blob['H'] = blob['H'].bfloat16()
                    else:
                        blob = []
                    torch.save(blob, path)
                    with self.assertRaisesRegex(ValueError, 'decode FC calibration'):
                        require_decode_calibration(store, name, cols)

    def test_committed_decode_mask_cannot_be_a_weight_or_broadcast_across_rows(self):
        cols, name = 16, 'draft/model.fc'
        layer = SimpleNamespace(input_dtype=torch.bfloat16, observer=None)
        c = Calibration('cpu', budget_bytes=1 << 20)
        c.attach(name, layer, PackStore.tiles(name, cols), True, decode_only=True)
        c.arm()
        x = torch.ones(7, cols, dtype=torch.bfloat16)
        for mask in (torch.ones(7), torch.ones(1, dtype=torch.bool), torch.ones(7, 1, dtype=torch.bool)):
            with self.assertRaisesRegex(ValueError, 'committed-row mask'):
                layer.observer(x, mask)
        self.assertEqual(c.progress(), 0)

    def test_first_rejection_covers_selection_support_eos_limits_and_inactive_rows(self):
        drafts = torch.tensor([[1, 2, 3]] * 7)
        picks = torch.tensor([[1, 9, 3, 4], [8, 2, 3, 4], [1, 2, 3, 4],
                              [1, 9, 3, 4], [1, 9, 3, 4], [8, 2, 3, 4], [8, 2, 3, 4]])
        support = torch.tensor([[[1, 8], [2, 5], [3, 6]]] * 7)
        alive = torch.tensor([True] * 6 + [False])
        remaining = torch.tensor([100, 100, 100, 1, 100, 100, 100])
        ends = torch.tensor([[-1], [-1], [-1], [-1], [1], [-1], [-1]])
        temps = torch.tensor([0., 0., 0., 0., 0., 1., 0.])
        actual = classify(picks, drafts, support, alive, remaining, ends, temps)
        self.assertEqual(actual.tolist(), [[1, 1], [0, 2], [3, 0], [1, 3], [1, 3], [0, -1], [0, -1]])
        # Attribution reads the proposals; it must not change generation inputs.
        self.assertEqual(drafts.tolist(), [[1, 2, 3]] * 7)

    def test_slot_owned_candidates_and_host_recording_survive_reuse(self):
        field = torch.zeros(5, 1, 2, 8, 1, 4, dtype=torch.bfloat16)
        d = DraftDiagnostics(field, 3, 2)
        self.assertEqual(d.slot(field[3]).tolist(), [3])
        with self.assertRaises(ValueError):
            d.slot(torch.zeros_like(field[0]))
        d.support[3] = torch.tensor([[1, 8], [2, 5], [3, 6]])
        records = []
        d.sink = lambda **row: records.append(row)
        d.note_sync(12, 2048, 3, 0, [8], 100, set())
        d.note_sync(13, 3000, 3, 1, [1, 9], 100, set())
        d.note_sync(14, 4000, 3, 1, [1, 9], 1, set())
        d.note_sync(15, 5000, 3, 1, [1, 9], 100, set(), policy_modified=True)
        self.assertEqual([x['reason'] for x in records],
                         ['selector_miss', 'candidate_miss', 'output_boundary', 'policy_modified'])
        self.assertEqual([x['seq'] for x in records], [12, 13, 14, 15])
        self.assertEqual(sum(x['count'] for x in d.snapshot()), 4)

    def test_async_and_bounded_serving_retire_diagnostics_once_without_changing_tokens(self):
        from tests.test_engine_burst_decode import CpuBurst, CpuQueue, engine
        from engine.profiles.glm53.pipeline import AsyncDecode
        for pipeline in ('async', 'burst', 'queue'):
            with self.subTest(pipeline=pipeline):
                def run(enabled):
                    e = engine(4)
                    records = []
                    if enabled:
                        e.draft_diagnostics = DraftDiagnostics(torch.zeros(5, 1), 1, 4)
                        e.draft_diagnostics.sink = lambda **row: records.append(row)
                        e.draft_diagnostics.support[:] = torch.tensor([7, 8, 9, 10])
                    p = AsyncDecode(e) if pipeline == 'async' else CpuBurst(e, 4)
                    if pipeline == 'queue':
                        p.queue = CpuQueue()
                    for seqs in ([1, 2, 3, 4], [2, 4], [1, 2, 4]):
                        pending = p.launch(seqs, seqs)
                        pending.resolve()
                    return e, records
                baseline, _ = run(False)
                candidate, records = run(True)
                self.assertEqual(candidate.tokens, baseline.tokens)
                self.assertEqual(candidate.ctx, baseline.ctx)
                self.assertEqual(candidate.accepted_total, baseline.accepted_total)
                self.assertEqual(sum(x['count'] for x in candidate.draft_diagnostics.snapshot()), len(records))
                self.assertEqual(len(records), candidate.drafted_total)
                self.assertEqual({r['reason'] for r in records}, {'all_accepted'})

    def test_diagnostic_release_drops_external_device_storage_and_sink(self):
        import weakref
        from engine.profiles.glm53.adapter import Glm53Engine
        d = DraftDiagnostics(torch.zeros(5, 1), 3, 2)
        held = weakref.ref(d.support)
        d.sink = lambda **row: None
        e = SimpleNamespace(tokens={}, slot={}, close_decode=lambda: None,
                            draft_diagnostics=d, drafter=SimpleNamespace(diagnostics=d),
                            vision=None, caches=SimpleNamespace())
        Glm53Engine.release(e)
        self.assertIsNone(held())
        self.assertIsNone(d.field)
        self.assertIsNone(d.sink)
        self.assertIsNone(e.draft_diagnostics)
        self.assertIsNone(e.drafter.diagnostics)


if __name__ == '__main__':
    unittest.main()
