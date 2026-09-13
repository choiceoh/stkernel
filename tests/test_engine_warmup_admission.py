"""Boot's synthetic rows exercise the serving draw keys without consuming user admissions."""
import importlib.util
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires PyTorch')
class WarmupDrawTests(unittest.TestCase):
    def engine(self, fail=False):
        from engine.base import draws
        from engine.base.kv import BlockPool, SlotPool
        from engine.profiles.glm53.adapter import Glm53Engine
        caches = NS(device='cpu', pool=BlockPool(64, 16, 4, 16), slots=SlotPool(5), reset=Mock())
        e = Glm53Engine(None, caches, NS(spec_k=6, block=16, vocab=128),
                       drafter=NS(k=6, aux_layers=[]), seed=913)
        e.admissions = 17
        e.decode_graphs = object()
        e.pipeline = NS(depth=2, pending=[], invalidate=Mock())
        seen = []

        def decode(rows, blocks, slots):
            # These are the actual host sampler inputs and the device-chain draw keys.
            host = e._pick_uniforms([NS(seq=r, length=1) for r in rows])
            expected = [draws.uniforms(draws.row_key(e.seed, e.nonces[r], 0), draws.PICK, 1)[0]
                        for r in rows]
            self.assertEqual(host, expected)
            self.assertEqual(len(set(e.nonces[r] for r in rows)), len(rows))
            self.assertEqual(len(blocks), len(slots))
            seen.append((tuple(rows), tuple(e.nonces[r] for r in rows)))
            if fail:
                raise RuntimeError('injected decode failure')

        def decode_async(rows, blocks, slots):
            decode(rows, blocks, slots)
            return NS(resolve=Mock())

        e.decode, e.decode_async = decode, decode_async
        e.async_ready = lambda rows: True
        return e, seen

    def empty(self, e):
        self.assertEqual(e.admissions, 17)
        for values in (e.tokens, e.prompt_len, e.nonces, e.slot, e.ctx, e.options, e.limits):
            self.assertEqual(values, {})
        self.assertEqual(e.caches.pool.rows_in_use, 0)
        self.assertEqual(e.caches.pool.available, 64)
        self.assertEqual(e.caches.slots.available, 4)

    def test_all_widths_and_async_steps_have_keys_and_leave_user_sequence_unchanged(self):
        e, seen = self.engine()
        with patch('torch.cuda.synchronize'):
            paid = e.warmup_shapes(lengths=())
            first = list(seen)
            self.empty(e)
            seen.clear()
            e.warmup_shapes(lengths=())
        self.assertEqual(set(paid), {f'decode{kind}/{w}' for w in range(1, 5) for kind in ('', '-async')})
        self.assertEqual(seen, first, 'repeating synthetic warmup must not advance request admissions')
        self.empty(e)
        e.add(0, [1, 2], max_new=8)
        self.assertEqual(e.nonces[0], 18, 'the first real request keeps its pre-warmup admission')

    def test_failed_decode_releases_rows_and_restores_admission_count(self):
        e, _ = self.engine(fail=True)
        with patch('torch.cuda.synchronize'), self.assertRaisesRegex(RuntimeError, 'injected decode failure'):
            e.warmup_shapes(lengths=(), widths=(4,))
        self.empty(e)


if __name__ == '__main__':
    unittest.main()
