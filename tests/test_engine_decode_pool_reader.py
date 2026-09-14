"""Direct pooling reads preserve the established kernel's pre-store values."""
import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


def inputs(n):
    tails = torch.randn(9, 12, 2, 136, dtype=torch.bfloat16)[:, :10, :, :128]
    source = torch.randn(n, 8, 264, dtype=torch.bfloat16)
    ape = torch.randn(8, 256)[::2, ::2]
    contexts = torch.full((n,), 31997, dtype=torch.int64)
    slots = torch.tensor([8, 2, 5, 0])[:n].clone()
    return tails, source[:, :, :128], source[:, :, 128:256], ape, contexts, slots


@unittest.skipUnless(importlib.util.find_spec('triton'), 'requires Triton')
class ContractTests(unittest.TestCase):
    def test_reference_bisection_also_replaces_the_captured_compressor(self):
        from dataclasses import replace
        from engine.profiles.glm53.lanes import reference, _apply_reference_lanes
        ref = reference()
        native = replace(ref, name='native', kpool_compress=object(),
                         decode_rows=replace(ref.decode_rows, compress=object()))
        untouched = _apply_reference_lanes(native, ref, ())
        self.assertIs(untouched, native)
        result = _apply_reference_lanes(native, ref, ('kpool_compress',))
        self.assertIs(result.kpool_compress, ref.kpool_compress)
        self.assertIs(result.decode_rows.compress, ref.decode_rows.compress)
        self.assertIs(result.decode_rows.update, native.decode_rows.update)
        with self.assertRaisesRegex(ValueError, 'names no lane'):
            _apply_reference_lanes(native, ref, ('not_a_lane',))

    def test_bound_wrapper_launches_once_and_preserves_source_views(self):
        from engine.kernels.kpool import compress_decode_pools
        for n in (1, 2, 3, 4):
            args = inputs(n)
            launch = MagicMock()
            with patch('engine.kernels.kpool._kpool_softmax_rotate_write_cache_kernel', launch):
                out, scale = compress_decode_pools(*args)
            self.assertEqual((out.shape, scale.shape), ((n * 2, 128), (n * 2, 1)))
            self.assertEqual((out.dtype, scale.dtype), (torch.float8_e4m3fn, torch.float32))
            launch.__getitem__.assert_called_once_with((n * 2,))
            called = launch.__getitem__.return_value.call_args
            self.assertIs(called.args[2], args[1])
            self.assertIs(called.args[3], args[2])
            self.assertIs(called.kwargs['tail_ptr'], args[0])
            self.assertIs(called.kwargs['physical_slots_ptr'], args[5])
            self.assertEqual(called.kwargs['num_warps'], 1)
            self.assertTrue(called.kwargs['MAPPED_INPUT'])

    def test_invalid_layout_is_rejected_before_launch(self):
        from engine.kernels.kpool import compress_decode_pools
        valid = inputs(2)
        for index, bad in ((0, valid[0][:, :9]), (1, valid[1][:, :7]), (2, valid[2].float()),
                           (3, valid[3].bfloat16()), (4, valid[4].int()), (5, valid[5][:1])):
            args = list(valid)
            args[index] = bad
            launch = MagicMock()
            with patch('engine.kernels.kpool._kpool_softmax_rotate_write_cache_kernel', launch):
                with self.assertRaisesRegex(ValueError, 'bound K=7'):
                    compress_decode_pools(*args)
            launch.__getitem__.assert_not_called()


@unittest.skipUnless(INTERPRET and importlib.util.find_spec('triton'), 'requires the no-GPU Triton interpreter')
class InterpreterTests(unittest.TestCase):
    def test_pre_fp8_values_and_scales_equal_materialized_windows(self):
        from engine.kernels.kpool import _kpool_softmax_rotate_write_cache_kernel as kernel
        from engine.modules.sparse_indexer import pool_window
        torch.manual_seed(923)
        for n in (1, 2, 3, 4):
            tail, k, gate, ape, contexts, slots = inputs(n)
            for phase, (ctx, magnitude) in enumerate(((0, 0.), (31997, 1e-6), (31998, 1.), (131059, 100.),
                                                     (131064, 10.), (31990, .1))):
                contexts.copy_(torch.tensor([max(0, ctx - i) for i in range(n)]))
                slots.copy_((torch.tensor([8, 2, 5, 0])[:n] + phase) % 9)
                tail.normal_().mul_(magnitude)
                k.normal_().mul_(magnitude)
                gate.normal_(std=10.)
                ape.normal_()
                window = pool_window(tail.index_select(0, slots), k, gate, contexts, 4, 2)
                outputs = []
                for mapped in (False, True):
                    x, score = (k, gate) if mapped else window
                    # FP32 diagnostic destinations compare the actual kernel's
                    # values BEFORE its FP8 store. This is not GPU FP8-cast proof.
                    out, scale = torch.full((n * 2, 128), float('nan')), torch.full((n * 2, 1), float('nan'))
                    kernel[(n * 2,)](
                        out, scale, x, score, ape, out, out, out, scale,
                        x.stride(0), x.stride(1), score.stride(0), score.stride(1), ape.stride(0),
                        PAGE_SIZE=1, BUF_NUMEL_PER_PAGE=1, POOL_SIZE=4, HEAD_DIM=128,
                        S_OFFSET_NBYTES_IN_PAGE=0, ROUND_SCALE=True, HAS_WRITE_MASK=False,
                        RETURN_COMPRESSED=True, WRITE_CACHE=False, BLOCK_D=128,
                        WARP_LOCAL_ROTATION=True, ape_stride_1=ape.stride(1),
                        MAPPED_INPUT=mapped, tail_ptr=tail, physical_slots_ptr=slots, contexts_ptr=contexts,
                        MAX_POOLS=2, TOKENS=8, TAIL_WIDTH=10,
                        TAIL_STRIDE_0=tail.stride(0), TAIL_STRIDE_1=tail.stride(1), TAIL_STRIDE_2=tail.stride(2),
                        num_warps=1)
                    outputs.append((out, scale))
                for got, want in zip(outputs[1], outputs[0]):
                    self.assertTrue(got.isfinite().all())
                    self.assertTrue(torch.equal(got, want), (n, ctx, magnitude))


if __name__ == '__main__':
    unittest.main()
