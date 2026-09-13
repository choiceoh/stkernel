"""The real warmup row lifecycle must satisfy the served sampling-key contract."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import torch

from engine.base import draws
from engine.profiles.glm53.adapter import Glm53Engine


class WarmupDrawTests(unittest.TestCase):
    def engine(self):
        caches = MagicMock()
        caches.device = torch.device('cpu')
        caches.pool.rows_in_use = 0
        caches.pool.num_blocks = 128
        caches.pool.max_seqs = 4
        caches.pool.row.side_effect = lambda row: row
        caches.slots.owner = [-1] * 6
        caches.slots.take.side_effect = lambda row: row + 1
        engine = Glm53Engine(None, caches, NS(spec_k=6, block=1, vocab=32), seed=7)
        engine.decode_graphs = object()
        engine.async_ready = lambda rows: False
        engine.admissions = 10
        return engine

    def test_all_warmup_widths_get_keys_and_preserve_real_admission_sequence(self):
        engine = self.engine()
        observed = []

        def decode(rows, physical, slots):
            values = engine._pick_uniforms([NS(seq=r, length=2) for r in rows])
            expected = []
            for row in rows:
                expected += draws.uniforms(draws.row_key(7, engine.nonces[row], 0), draws.PICK, 2)
            self.assertEqual(values, expected)
            observed.extend(engine.nonces[row] for row in rows)

        engine.decode = decode
        with patch('torch.cuda.synchronize'):
            paid = engine.warmup_shapes(lengths=(), widths=(1, 4))
        self.assertEqual(set(paid), {'decode/1', 'decode/4'})
        self.assertEqual(observed, [11, 12, 13, 14, 15])
        self.assertEqual(engine.nonces, {})
        self.assertEqual(engine.tokens, {})
        self.assertEqual(engine.admissions, 10)
        engine._admitted(3)
        self.assertEqual(engine.nonces[3], 11)

    def test_failed_warmup_also_restores_nonce_sequence_and_rows(self):
        engine = self.engine()

        def fail(rows, physical, slots):
            engine._pick_uniforms([NS(seq=r, length=1) for r in rows])
            raise RuntimeError('injected decode failure')

        engine.decode = fail
        with patch('torch.cuda.synchronize'), self.assertRaisesRegex(RuntimeError, 'injected decode failure'):
            engine.warmup_shapes(lengths=(), widths=(4,))
        self.assertEqual(engine.nonces, {})
        self.assertEqual(engine.tokens, {})
        self.assertEqual(engine.admissions, 10)


if __name__ == '__main__':
    unittest.main()
