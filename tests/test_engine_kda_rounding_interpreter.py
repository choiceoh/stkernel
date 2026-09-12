"""Execute the actual Triton rounding/writer on CPU, not just a mirror.

TRITON_INTERPRET=1 python -m unittest tests.test_engine_kda_rounding_interpreter
This proves scalar semantics and logical addressing, not CUDA graph or speed.
"""
import os
import unittest
import warnings

import torch


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'separate Triton interpreter process')
class InterpreterRoundingTests(unittest.TestCase):
    def test_actual_store_matches_independent_cpu_reference(self):
        from engine.kernels.kda.rounding import _copy
        from engine.modules.kda_storage import store_state, rounding_seed
        # Includes signed normals, half-subnormal thresholds, exponent edges,
        # and tiny underflow. Shapes exercise padding and source strides.
        values = [1 + 2**-12, -1 - 2**-12, 2**-25, -2**-25, 2**-14 + 2**-26,
                  2 - 2**-12, 0., -0., 65504., 2**-60, 2**-30, -2**-30]
        src = torch.tensor(values).repeat(2, 33, 1).transpose(1, 2)
        dst, expected = (torch.empty(src.shape, dtype=torch.float16) for _ in range(2))
        seed = rounding_seed(44, 3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # NumPy uint overflow is Philox arithmetic
            for position in (0, 32768, 2**32 + 32768):
                store_state(expected, src, position, seed)
                for block in (128, 256):
                    _copy[((src.numel()+block-1)//block,)](src, dst, position, *src.shape, *src.stride(), seed, block)
                    self.assertTrue(torch.equal(dst.view(torch.int16), expected.view(torch.int16)), (position, block))

    def test_actual_graph_writer_uses_absolute_position_and_preserves_other_slots(self):
        from engine.kernels.state import write_ring
        from engine.modules.kda_storage import store_state, rounding_seed
        src = torch.full((7, 2, 17, 33), 1 + 2**-12)
        seed = rounding_seed(2, 3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for slot, position in ((1, 32768), (3, 32768), (2, 65536)):
                actual = torch.full((4, 7, 2, 17, 33), -8., dtype=torch.float16)
                expected = actual.clone()
                for i in range(7):
                    store_state(expected[slot, (position+i)%7], src[i], position+i, seed)
                write_ring(src, actual, torch.tensor([slot]), torch.tensor(position), round_seed=seed)
                self.assertTrue(torch.equal(actual, expected))

    def test_actual_repeated_rounding_does_not_lose_small_updates(self):
        from tests.test_engine_kda_rounding import _walk
        n, steps = 1024, 2048
        out = torch.empty(n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _walk[(1,)](out, steps, n, False, n)
            self.assertTrue(torch.equal(out, torch.ones_like(out)))
            _walk[(1,)](out, steps, n, True, n)
        self.assertLess(abs(float(out.double().mean()) - (1 + steps*2**-12)), .005)


if __name__ == '__main__':
    unittest.main()
