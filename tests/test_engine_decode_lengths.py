"""Length reuse belongs to one gathered batch, not to a warmup or prior replay."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

from engine.modules.sparse_indexer import row_lengths
from engine.profiles.glm53.decode_graphs import DeviceStep, GraphCaches


def caches(n=4):
    ids = torch.arange(n)
    real = NS(F=NS(kpool=4), layout=None, block_table=torch.zeros(8, 176, dtype=torch.int32))
    return GraphCaches(real, ids, ids+1, 131072)


class LengthLifetimeTests(unittest.TestCase):
    def test_shared_within_a_forward_and_recomputed_after_context_change(self):
        for n in (1, 2, 3, 4):
            cache, make = caches(n), Mock(wraps=row_lengths)
            contexts = torch.zeros(n, dtype=torch.int64)
            for phase, ctx in enumerate((0, 31997, 131061, 31990, 3)):
                contexts.copy_(torch.arange(n)+ctx)
                cache.gather()
                first = cache.row_lengths(contexts, 8, 4, make)
                for _ in range(10):
                    self.assertIs(cache.row_lengths(contexts, 8, 4, make), first)
                expected = (contexts[:, None]+torch.arange(8)+1).int().flatten()
                self.assertTrue(torch.equal(first[0], expected))
                self.assertTrue(torch.equal(first[1], expected//4))
                self.assertEqual(make.call_count, phase+1)

    def test_changed_batch_identity_width_pool_or_lane_requires_a_new_gather(self):
        cache, contexts, make = caches(1), torch.zeros(1, dtype=torch.int64), Mock(wraps=row_lengths)
        with self.assertRaisesRegex(RuntimeError, 'gathered graph caches'):
            cache.row_lengths(contexts, 8, 4, make)
        cache.gather()
        cache.row_lengths(contexts, 8, 4, make)
        for args in ((contexts.clone(), 8, 4, make), (contexts, 1, 4, make),
                     (contexts, 8, 8, make), (contexts, 8, 4, row_lengths)):
            with self.assertRaisesRegex(ValueError, 'one gathered batch'):
                cache.row_lengths(*args)
        self.assertEqual(make.call_count, 1)

    def test_split_groups_keep_independent_contexts_and_metadata(self):
        parent, make = caches(), Mock(wraps=row_lengths)
        contexts = torch.tensor([0, 31997, 131064, 32252])
        parent.gather()
        whole = parent.row_lengths(contexts, 8, 4, make)
        for a, b in ((0, 2), (2, 4)):
            child, ctx = parent.subset(a, b), contexts[a:b]
            pair = child.row_lengths(ctx, 8, 4, make)
            for _ in range(10):
                self.assertIs(child.row_lengths(ctx, 8, 4, make), pair)
            for got, expected in zip(pair, whole):
                self.assertTrue(torch.equal(got, expected[a*8:b*8]))
                self.assertNotEqual(got.data_ptr(), expected.data_ptr())
        self.assertEqual(make.call_count, 3)

    def test_actual_capture_forward_clears_metadata_on_success_and_failure(self):
        path = Path(__file__).resolve().parents[1]/'engine/profiles/glm53/decode_graphs.py'
        node = copy.deepcopy(next(n for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == 'forward'))
        cache, make = caches(1), Mock(wraps=row_lengths)
        step = DeviceStep(torch.zeros(8, dtype=torch.int64), torch.zeros(1, dtype=torch.int64), 8)
        expected, failing = [], [False]
        def model_forward(step, scratch, **kwargs):
            for _ in range(11):
                pair = scratch.row_lengths(step.contexts, 8, 4, make)
                expected.append(pair[0].clone())
                if failing[0]:
                    raise RuntimeError('injected model failure')
            return torch.zeros(8, 4)
        logits = torch.empty(8, 4)
        scope = dict(torch=torch, tokens=8,
                     net=NS(forward=model_forward, head_local=lambda h, *, out: out.copy_(h)),
                     self=NS(observe_stream=None, execution_plan=NS(direct_mhc=False), streams=None,
                             aux_layers=(), head_outputs={(1, 8): logits}))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        for phase, ctx in enumerate((31997, 131064, 31990)):
            step.contexts.fill_(ctx)
            scope['forward']((step, None, None, cache, logits))
            self.assertEqual(make.call_count, phase+1)
            self.assertIsNone(cache._decode_lengths)
            self.assertFalse(hasattr(cache, 'block_table'))
            self.assertTrue(torch.equal(expected[-1], torch.arange(8).int()+ctx+1))
        failing[0] = True
        with self.assertRaisesRegex(RuntimeError, 'injected model failure'):
            scope['forward']((step, None, None, cache, logits))
        self.assertIsNone(cache._decode_lengths)
        self.assertFalse(hasattr(cache, 'block_table'))


@unittest.skipUnless(torch.cuda.is_available(), 'requires the existing reserved GPU')
class CaptureTests(unittest.TestCase):
    def test_replays_recompute_lengths_for_new_contexts_and_rollback(self):
        from probes.engine_decode_lengths import check
        check(lambda *a, **kw: None)


if __name__ == '__main__':
    unittest.main()
