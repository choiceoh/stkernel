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
from engine.profiles.glm53.draft_policy import DraftPolicy, decode_name, require_decode_calibration
from engine.profiles.glm53.draft_diagnostics import DraftDiagnostics, classify


class DraftAcceptanceTests(unittest.TestCase):
    def test_arms_are_independent_and_invalid_or_ineffective_choices_fail(self):
        self.assertFalse(DraftPolicy().active)
        self.assertTrue(DraftPolicy(fc_precision='fp8').active)
        self.assertTrue(DraftPolicy(fc_calibration='collect').active)
        self.assertTrue(DraftPolicy(diagnostics=True).active)
        for options in ({'fc_precision': 'bf16'}, {'fc_calibration': 'oops'}, {'diagnostics': 1},
                        {'fc_precision': 'fp8', 'fc_calibration': 'decode'}):
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
        def w4(x, pack, workspace):
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
