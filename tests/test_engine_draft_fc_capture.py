"""The draft FC pair collector: families stay whole, the budget holds, and it never breaks a step."""
import types
import tempfile
import unittest

import torch

from engine.profiles.glm53.draft_fc_capture import DraftFcCapture, _family_split, attach, retain_source

COLS = 64


def fake_graphs():
    """What a captured boot exposes: the pipeline calls these, never the drafter's own method."""
    graphs = types.SimpleNamespace(replayed=[])
    graphs.observe_rows = lambda slots, positions, aux, valid: graphs.replayed.append('rows')
    graphs.observe_prepared_rows = (
        lambda slots, positions, context, valid, aux: graphs.replayed.append('prepared'))
    return graphs


def fake_drafter(*, source=True, observer=None, rank=0, graphs=None):
    layer = types.SimpleNamespace(cols=COLS, observer=observer)
    drafter = types.SimpleNamespace(
        decode_graphs=graphs,
        dense={'fc.weight': layer},
        p={'fc.weight': torch.zeros(4, COLS, dtype=torch.bfloat16) if source else None},
        target=types.SimpleNamespace(comm=types.SimpleNamespace(rank=rank)),
        diagnostics=types.SimpleNamespace(note_sync=lambda *a, **k: 'inner'),
        tuning=types.SimpleNamespace(digest='d'),
    )
    drafter.observe_rows = lambda field, slots, positions, aux, valid: 'inner'
    return drafter


def rows(cap, n, t, valid, slots=None):
    slots = torch.arange(n) if slots is None else slots
    positions = torch.arange(n * t).reshape(n, t)
    aux = torch.randn(n * t, COLS, dtype=torch.bfloat16)
    cap.drafter.observe_rows(None, slots, positions, aux, torch.tensor(valid))


class FamilySplitTests(unittest.TestCase):
    def test_a_family_lands_in_one_split_and_the_salt_moves_it(self):
        names = [f'seq-{i}' for i in range(400)]
        first = {n: _family_split(n, .25, 'a') for n in names}
        self.assertEqual(first, {n: _family_split(n, .25, 'a') for n in names})   # deterministic
        share = sum(v == 'validation' for v in first.values()) / len(names)
        self.assertTrue(.18 < share < .33, share)
        moved = sum(first[n] != _family_split(n, .25, 'b') for n in names)
        self.assertTrue(moved, 'a different salt must be able to move families')

    def test_the_share_is_honoured_at_the_ends(self):
        names = [f'seq-{i}' for i in range(400)]
        low = sum(_family_split(n, .05, 's') == 'validation' for n in names) / len(names)
        high = sum(_family_split(n, .80, 's') == 'validation' for n in names) / len(names)
        self.assertTrue(low < .12 and high > .70, (low, high))


class GraphSeamTests(unittest.TestCase):
    """The first armed production boot collected 0 rows: it had wrapped the one seam a captured boot never calls."""

    def _rows(self, cap, call):
        positions = torch.arange(16).reshape(2, 8)
        aux = torch.randn(16, COLS, dtype=torch.bfloat16)
        call(torch.tensor([0, 1]), positions, aux, torch.tensor([8, 8]))

    def test_the_replayed_seam_is_what_records(self):
        graphs = fake_graphs()
        cap = DraftFcCapture(fake_drafter(graphs=graphs), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        self.assertIn('decode_graphs.observe_rows', cap.seams)
        self.assertIn('decode_graphs.observe_prepared_rows', cap.seams)
        self._rows(cap, cap.drafter.decode_graphs.observe_rows)
        self.assertEqual(sum(cap.kept.values()), 16)
        self.assertEqual(graphs.replayed, ['rows'], 'the replay still happens, unchanged')

    def test_the_early_observe_seam_records_too(self):
        graphs = fake_graphs()
        cap = DraftFcCapture(fake_drafter(graphs=graphs), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        context = torch.zeros(2, 8, 1)
        cap.drafter.decode_graphs.observe_prepared_rows(
            torch.tensor([0, 1]), torch.arange(16).reshape(2, 8), context, torch.tensor([8, 8]),
            torch.randn(16, COLS, dtype=torch.bfloat16))
        self.assertEqual(sum(cap.kept.values()), 16)
        self.assertEqual(graphs.replayed, ['prepared'])

    def test_a_graphless_boot_still_has_the_drafter_seam(self):
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        self.assertEqual(cap.seams, ['drafter.observe_rows'])

    def test_a_door_that_never_fills_both_splits_stops_syncing(self):
        """Recording costs a device sync a step; without a budget a one-family door would pay it forever."""
        graphs = fake_graphs()
        cap = DraftFcCapture(fake_drafter(graphs=graphs), '/tmp/unused', rows=8, salt='s')
        cap.attach()
        cap.kept['train'] = 8                                  # budget met on one side, the other never fed
        for _ in range(cap.calls_budget + 50):
            self._rows(cap, cap.drafter.decode_graphs.observe_rows)
        self.assertTrue(cap.full())
        self.assertEqual(cap.calls, cap.calls_budget)
        self.assertEqual(len(graphs.replayed), cap.calls_budget + 50, 'every step still replayed')


class CaptureTests(unittest.TestCase):
    def test_compaction_can_retire_the_device_source_without_breaking_collection(self):
        from bench.draft_fc_bias import collect_fc_pairs
        from tests.test_engine_draft_precision import fixture
        drafter, batches = fixture()
        expected = collect_fc_pairs(drafter, batches)
        retain_source(drafter)
        source = drafter.p.pop('fc.weight')
        source.zero_()  # compaction can reuse the original storage
        self.assertEqual(drafter.fc_capture_source.device.type, 'cpu')
        drafter.observe_rows = lambda *args: None
        with tempfile.TemporaryDirectory() as root:
            cap = attach(types.SimpleNamespace(drafter=drafter), root, rows=2)
            cap.batches, cap.kept = batches, dict(train=1, validation=1)
            result = torch.load(cap.close()['path'], weights_only=True)
        self.assertNotIn('fc.weight', drafter.p)
        self.assertEqual(result['reader_sha256'], expected['reader_sha256'])
        for split in ('train', 'validation'):
            for field in ('actual', 'reference'):
                torch.testing.assert_close(result[split][field], expected[split][field], rtol=0, atol=0)

    def test_it_wraps_without_replacing_and_keeps_families_whole(self):
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        self.assertEqual(cap.drafter.diagnostics.note_sync(7, 0, 2, 1, [], 1, ()), 'inner')
        self.assertEqual(cap.slot_family[2], 'seq-7')            # the map came from note_sync
        rows(cap, 4, 8, [8, 8, 8, 8])
        self.assertTrue(cap.batches)
        for batch in cap.batches:
            self.assertEqual(len(set(batch['ids'])), 1, 'a batch carries one family')
            self.assertEqual(batch['aux'].shape, (8, COLS))
            self.assertEqual(batch['keep'].shape, (8,))
        seen = {}
        for batch in cap.batches:                                 # and one family never splits
            seen.setdefault(batch['ids'][0], set()).add(batch['split'])
        self.assertTrue(all(len(v) == 1 for v in seen.values()), seen)

    def test_only_committed_rows_count_and_the_budget_is_never_exceeded(self):
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=6, salt='s')
        cap.attach()
        rows(cap, 2, 8, [3, 0])                                   # slot 0 commits 3, slot 1 commits none
        self.assertEqual(sum(cap.kept.values()), 3)
        self.assertEqual(len(cap.batches), 1, 'a slot with no committed row contributes nothing')
        self.assertTrue(bool(cap.batches[0]['keep'][:3].all()) and not bool(cap.batches[0]['keep'][3:].any()))
        for _ in range(20):
            rows(cap, 4, 8, [8, 8, 8, 8])
        self.assertTrue(max(cap.kept.values()) <= 6, cap.kept)

    def test_a_failure_stops_the_capture_and_not_the_step(self):
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        positions = torch.arange(8).reshape(1, 8)
        bad = torch.randn(8, COLS, dtype=torch.float32)           # wrong dtype: the guard must fire
        self.assertEqual(cap.drafter.observe_rows(None, torch.tensor([0]), positions, bad,
                                                  torch.tensor([8])), 'inner')
        self.assertIsNotNone(cap.stopped)
        self.assertEqual(cap.batches, [])
        self.assertIn('error', cap.close())

    def test_rows_are_held_on_the_host(self):
        """437 MiB of aux on the device would cost a fleet window; on the host it rides a serving boot."""
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        rows(cap, 2, 8, [8, 8])
        self.assertTrue(cap.batches)
        for batch in cap.batches:
            self.assertEqual(batch['aux'].device.type, 'cpu')
            self.assertEqual(batch['keep'].device.type, 'cpu')

    def test_arming_refuses_what_the_collector_cannot_read(self):
        with self.assertRaises(ValueError):
            attach(types.SimpleNamespace(drafter=None), '/tmp/unused')
        with self.assertRaises(ValueError):                        # the BF16 source was consumed
            attach(types.SimpleNamespace(drafter=fake_drafter(source=False)), '/tmp/unused')
        with self.assertRaises(ValueError):                        # a calibrating reader
            attach(types.SimpleNamespace(drafter=fake_drafter(observer=lambda *a: None)), '/tmp/unused')
        self.assertIsInstance(attach(types.SimpleNamespace(drafter=fake_drafter()), '/tmp/unused'),
                              DraftFcCapture)

    def test_attaching_twice_is_refused(self):
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=8)
        cap.attach()
        with self.assertRaises(ValueError):
            cap.attach()

    def test_a_bundle_needs_both_splits(self):
        cap = DraftFcCapture(fake_drafter(), '/tmp/unused', rows=64, salt='s')
        cap.attach()
        report = cap.close()                                       # nothing recorded at all
        self.assertIn('error', report)
        self.assertEqual(report['rank'], 0)


class ProfileDefaultTests(unittest.TestCase):
    def test_main_ships_the_collector_switched_off(self):
        """A collecting boot serves normally but writes 160 MiB of pairs; main must not do that."""
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1]
                  / 'engine/profiles/glm53/boot.py').read_text()
        self.assertIn('DRAFT_FC_CAPTURE = False', source)
        self.assertIn('DRAFT_FC_CAPTURE_ROWS = ', source)


if __name__ == '__main__':
    unittest.main()
