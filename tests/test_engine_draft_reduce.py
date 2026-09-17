"""Exactly one reduction per drafter boundary, with immediate packet consumption."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.profiles.glm53.drafter import Drafter
from engine.kernels.oneshot import RankPackets


def fixture(enabled=True):
    from engine.profiles.glm53.drafter import _bind_common_lanes
    _bind_common_lanes()
    d = Drafter.__new__(Drafter)
    d.F = NS(layers=5, conv_taps=2, conv_group=256, rms_eps=1e-6)
    d.k, d.max_block_rows = 7, 32
    d.fast_attention, d.reduce_packets = True, enabled
    d.reduce_packets_executed = set()
    d.decode_cell_rows = (8,)
    d.reduce_packet_rows = (8,)
    p = {'norm.weight': torch.ones(4096, dtype=torch.bfloat16)}
    for L in range(5):
        for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            p[f'layers.{L}.{name}'] = p['norm.weight']
        for name in ('attention_conv.base_kernel', 'mlp_conv.base_kernel'):
            p[f'layers.{L}.{name}'] = torch.ones(2, 2, 4096, dtype=torch.bfloat16)
    d.p = p
    owner = NS(pending=None, packet_failed=False)
    def exchange(x):
        if owner.pending is not None or owner.packet_failed:
            raise RuntimeError('previous exchange was not consumed')
        owner.pending = RankPackets(owner, x, torch.arange(4, dtype=torch.int64))
        return owner.pending
    transport = NS(exchange=Mock(side_effect=exchange))
    d.target = NS(comm=NS(transport=transport, _settled=lambda x: x,
                          all_reduce=Mock(side_effect=lambda x: x)),
                  embed=lambda ids: torch.ones(len(ids), 4096, dtype=torch.bfloat16))
    def linear(x, name, *unused):
        width = 64 if name.endswith('kernel_projection.weight') else 8192 if name.endswith('gate_up') else 4096
        return torch.ones(len(x), width, dtype=torch.bfloat16)
    d.linear = linear
    d._attn_rows = Mock(side_effect=lambda L, x, *args, **kw: x)
    d._conv_rows = Mock(side_effect=lambda x, *args: x)
    d._post_conv_norm = Mock(side_effect=lambda x, delta, base, res, weight, block: (res, x))
    d._finish_head_input = Mock(side_effect=lambda res, x, *args: x)
    return d, owner


def run_block(d):
    ids = torch.zeros(8, dtype=torch.int64)
    return d.block_rows(ids, ids, ids[:1], ids[:1], None, 1, 8, head_input=True)


class DraftReduceTests(unittest.TestCase):
    def test_all_ten_boundaries_consume_before_next_exchange(self):
        d, owner = fixture()
        with patch('torch.cuda.current_stream', return_value=NS(cuda_stream=13)), \
             patch('engine.kernels.draft_reduce.packet_tap_add_norm', side_effect=lambda x, pk, de, ba, r, *a: (r, x)) as norm, \
             patch('engine.kernels.draft_reduce.packet_tap_mix', side_effect=lambda x, *a: x) as mix:
            run_block(d)
        self.assertEqual(norm.call_count, 9)
        self.assertEqual(mix.call_count, 1)
        self.assertEqual(d.target.comm.transport.exchange.call_count, 10)
        d.target.comm.all_reduce.assert_not_called()
        self.assertIsNone(owner.pending)
        self.assertFalse(owner.packet_failed)
        self.assertEqual(d.reduce_packets_executed, {(s, L, 8) for s in ('attn', 'mlp') for L in range(5)})
        self.assertTrue(all(c.kwargs['reduce'] is False for c in d._attn_rows.call_args_list))
        self.assertTrue(d._finish_head_input.call_args.args[-1])

    def test_ordinary_transport_reduces_once_not_twice(self):
        d, _ = fixture(False)
        run_block(d)
        self.assertEqual(d.target.comm.all_reduce.call_count, 10)
        d.target.comm.transport.exchange.assert_not_called()
        self.assertEqual(d._post_conv_norm.call_count, 9)

    def test_failed_consumer_poison_is_not_retried_as_all_reduce(self):
        d, owner = fixture()
        with patch('torch.cuda.current_stream', return_value=NS(cuda_stream=13)), \
             patch('engine.kernels.draft_reduce.packet_tap_add_norm', side_effect=ValueError('bad consumer')):
            with self.assertRaisesRegex(ValueError, 'bad consumer'):
                run_block(d)
        self.assertTrue(owner.packet_failed)
        self.assertIsNone(owner.pending)
        d.target.comm.all_reduce.assert_not_called()
        self.assertFalse(d.reduce_packets_executed)

    def test_boot_requires_every_bound_layer_and_side(self):
        from engine.profiles.glm53.boot import drafter_reduce_report
        d, _ = fixture()
        with self.assertRaisesRegex(RuntimeError, 'not executed'):
            drafter_reduce_report(d)
        d.reduce_packets_executed = {(s, L, 8) for s in ('attn', 'mlp') for L in range(5)}
        self.assertEqual(len(drafter_reduce_report(d)['executed']), 10)
        d.reduce_packet_rows = (8, 32)
        with self.assertRaisesRegex(RuntimeError, 'not executed'):
            drafter_reduce_report(d)
        d.reduce_packets_executed.update((s, L, 32) for s in ('attn', 'mlp') for L in range(5))
        self.assertEqual(drafter_reduce_report(d)['rows'], [8, 32])
        d.reduce_packets = False
        self.assertEqual(drafter_reduce_report(d), {})

    def test_packet_rows_do_not_depend_on_dense_precision(self):
        d, _ = fixture()
        d.dense = {}
        self.assertEqual(d.bind_decode_cells((8, 16, 24, 32, 40)), [])
        self.assertEqual(d.decode_cell_rows, ())
        self.assertEqual(d.reduce_packet_rows, (8, 16, 24, 32))

    def test_other_block_lengths_keep_one_ordinary_reduction(self):
        d, _ = fixture()
        x = torch.zeros(6, 4096, dtype=torch.bfloat16)
        self.assertIs(d._reduce_post_conv(x, None, None, 6, ('mlp', 4)), x)
        d.target.comm.all_reduce.assert_called_once_with(x)
        d.target.comm.transport.exchange.assert_not_called()

    def test_replicated_reference_does_not_reduce(self):
        d, _ = fixture()
        d.fast_attention = False
        x = torch.zeros(8, 4096, dtype=torch.bfloat16)
        self.assertIs(d._reduce_post_conv(x, None, None, 8, ('mlp', 4)), x)
        d.target.comm.all_reduce.assert_not_called()
        d.target.comm.transport.exchange.assert_not_called()

    def test_raw_cpu_descriptors_are_refused_before_dereferencing(self):
        from engine.kernels.draft_reduce import packet_tap_mix
        x = torch.zeros(8, 4096, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, 'TP4 CUDA'):
            packet_tap_mix(x, torch.zeros(4, dtype=torch.int64), x, x, 256, 8)


if __name__ == '__main__':
    unittest.main()
