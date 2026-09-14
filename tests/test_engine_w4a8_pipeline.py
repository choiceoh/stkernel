"""Packed pair addresses, shared activation groups and workspace ownership."""
import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from engine.modules.w4a8_dataflow import W4A8PipelinePlan

TRITON = importlib.util.find_spec('triton') is not None
INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'

if TRITON:
    import triton
    import triton.language as tl
    from engine.kernels.w4a8_pipeline import _weight_tile

    @triton.jit
    def read_tiles(W, S, OUT, K: tl.constexpr):
        tile, block = tl.program_id(0), tl.program_id(1)
        weight = _weight_tile(W, S, tile, block, K)
        r, c = tile*128+tl.arange(0, 128), block*128+tl.arange(0, 128)
        tl.store(OUT+r[None, :]*K+c[:, None], weight.to(tl.uint8, bitcast=True))


@unittest.skipUnless(TRITON, 'requires Triton')
class WorkspaceTests(unittest.TestCase):
    @staticmethod
    def workspace(plans):
        from engine.kernels.w4a8_pipeline import Workspace
        empty = torch.empty
        with patch('torch.cuda.get_device_capability', return_value=(12, 1)), \
             patch('torch.cuda.is_current_stream_capturing', return_value=False), \
             patch('torch.empty', side_effect=lambda *a, **kw: empty(*a, **{**kw, 'device': 'cpu'})):
            return Workspace(plans, 'cuda:0')

    def test_shared_input_does_not_alias_live_activation_output_or_other_scales(self):
        plans = [W4A8PipelinePlan(m, h, i) for m, h, i in
                 ((1, 128, 128), (17, 384, 256), (16, 4096, 3072), (32, 8192, 8192))]
        # Only allocation/device admission are mocked. Inspect the real byte
        # slices: an output may be reused as next input, so XQ cannot alias it.
        workspace = self.workspace(plans)
        self.assertEqual(workspace.storage.numel(), max(p.scratch_bytes for p in plans))
        for p in plans:
            views = workspace.views[p.rows, p.hidden, p.intermediate]
            occupied = torch.zeros(p.scratch_bytes, dtype=torch.bool)
            for v in views:
                start = v.data_ptr()-workspace.storage.data_ptr()
                size = v.numel()*v.element_size()
                self.assertEqual(start % 16, 0)
                self.assertFalse(occupied[start:start+size].any())
                occupied[start:start+size] = True
            self.assertEqual(int(occupied.sum()), sum(v.numel()*v.element_size() for v in views))
            self.assertLess(p.scratch_bytes-int(occupied.sum()), 4*16)
            self.assertEqual(tuple(v.dtype for v in views),
                             (torch.uint8, torch.float32, torch.bfloat16, torch.uint8, torch.float32))

    def test_output_feedback_is_safe_but_quantizer_overlap_is_rejected_before_launch(self):
        from engine.kernels.w4a8_pipeline import execute
        from tests.test_engine_w4a8_dataflow import packed_weights
        plan = W4A8PipelinePlan(1, 128, 128)
        workspace = self.workspace((plan, W4A8PipelinePlan(32, 8192, 8192)))
        views = workspace.views[1, 128, 128]
        weights = packed_weights(128, 128)
        launches = [MagicMock() for _ in range(3)]
        with patch('engine.kernels.w4a8_pipeline._input', launches[0]), \
             patch('engine.kernels.w4a8_pipeline._gate_up', launches[1]), \
             patch('engine.kernels.w4a8_pipeline._down', launches[2]):
            self.assertIs(execute(plan, views[2], weights, 10., workspace), views[2])
            for launch in launches[1:]:
                self.assertEqual(launch.__getitem__.return_value.call_args.kwargs['arch'], 'sm120')
            for launch in launches:
                launch.__getitem__.assert_called_once()
                launch.reset_mock()
            for start, _ in plan.buffer_spans[3:]:
                x = workspace.storage[start:start+256].view(torch.bfloat16).view(1, 128)
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    execute(plan, x, weights, 10., workspace)
            for launch in launches:
                launch.__getitem__.assert_not_called()


@unittest.skipUnless(TRITON and INTERPRET, 'explicit no-GPU Triton interpreter check')
class InterpreterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_every_packed_byte_and_signed_scale_matches_the_deployed_byte_tables(self):
        # Cover all 256 packed bytes x all 256 signed scale bytes, including
        # both signs of zero, in multiple nonzero N/K tile positions.
        k, n = 256, 512
        data = torch.arange(256, dtype=torch.int32).repeat(256).to(torch.uint8).reshape(4, 2, 128, 64)
        scales = torch.arange(-128, 128, dtype=torch.int32).repeat_interleave(256).reshape_as(data)[..., ::8].to(torch.int8)
        actual = torch.empty(n, k, dtype=torch.uint8)
        read_tiles[(4, 2)](data, scales.contiguous(), actual, k)
        expected = torch.empty_like(actual)
        for tile in range(4):
            for block in range(2):
                for row in range(128):
                    for byte in range(64):
                        packed, d = int(data[tile, block, row, byte]), int(scales[tile, block, row, byte//8])
                        lut = (0, 48, 56, 61, 64, 69, 72, 77) if 1 <= (d & 7) <= 5 else (0, 48, 56, 60, 64, 68, 72, 76)
                        for half in range(2):
                            code = (packed >> (4*half)) & 15
                            mag = code & 7
                            value = (0 if mag == 0 else lut[mag]+d) | ((code & 8) << 4)
                            expected[tile*128+row, block*128+byte*2+half] = value & 255
        self.assertTrue(torch.equal(actual, expected))

    def test_shared_quantizer_group_layout_tail_masks_and_repeated_inputs(self):
        from engine.kernels.w4a8_pipeline import _input
        from engine.kernels.dense.packing import _mk_quant_x_ref
        for m, h in ((1, 128), (3, 384), (8, 4096), (17, 256), (32, 8192)):
            for repeat in range(2):
                groups = m*(h//128)
                # Representable FP8 values with a power-of-two scale avoid the
                # interpreter's known halfway-rounding bug. Scaling/addresses
                # execute the actual kernel; native RTNE is a separate GPU gate.
                values = torch.arange(128).remainder(17).sub(8).float().repeat(groups, 1)
                values[:, -1] = 448
                powers = torch.exp2(torch.arange(groups).remainder(11).float()-5+repeat)
                x = (values*powers[:, None]).reshape(m, h).bfloat16()
                x.view(-1, 128)[::7] = 0
                q = torch.full((m*h+128,), 93, dtype=torch.uint8)
                s = torch.full((groups+4,), -17., dtype=torch.float32)
                _input[(triton.cdiv(groups, 4),)](x, q, s, m, h)
                decoded = (q[:m*h].view(torch.float8_e4m3fn).float().view(groups, 128)*s[:groups, None]).view(m, h)
                self.assertTrue(torch.equal(decoded, _mk_quant_x_ref(x)))
                self.assertTrue((q[m*h:] == 93).all())
                self.assertTrue((s[groups:] == -17).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
