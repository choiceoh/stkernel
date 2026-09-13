"""Native coverage and finite-output gates must fail before admitting requests."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import torch

from engine.base.comm import Comm
from engine.profiles.glm53.adapter import Glm53Engine
from engine.profiles.glm53.boot import native_execution_report


class NativeQualificationTests(unittest.TestCase):
    def test_missing_native_implementation_cannot_pass_coverage(self):
        net = NS(layers=[0, 1], dense={'a': NS(executed=3), 'head': NS(executed=True)},
                 mhc=NS(executed={'a', 'b', 'c'}),
                 shared_mlp={1: NS(executed=True)},
                 shared_overlap=NS(executed=True),
                 _router_weights={1: None}, _router_tensorcore={1},
                 prefill_transport=NS(executed={'fp8_all_gather', 'fp8_reduce_scatter'}, project_tiles=False))
        drafter = NS(dense={'fc.weight': NS(executed=3), 'q': NS(executed=1)})
        self.assertEqual(native_execution_report(net, drafter)['target_fp8'], 1)
        required = {'fp8_all_gather', 'fp8_reduce_scatter'}
        for extra in (set(), {'fp8_tiled_projection'}, {'fp8_tiled_projection', 'fp8_packet_projection'}):
            net.prefill_transport.executed = required | extra
            with self.subTest(extra=extra):
                self.assertEqual(set(native_execution_report(net, drafter)['prefill_collectives']), required | extra)
        for missing in required:
            # Two markers can still be incomplete; additional optimized lanes
            # must never stand in for the missing gather or reduce-scatter.
            net.prefill_transport.executed = (required - {missing}) | {'fp8_tiled_projection'}
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, 'proof is incomplete'):
                native_execution_report(net, drafter)
        net.prefill_transport.project_tiles = True
        for missing in ('fp8_tiled_projection', 'fp8_packet_projection'):
            net.prefill_transport.executed = required | ({'fp8_tiled_projection', 'fp8_packet_projection'} - {missing})
            with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, 'proof is incomplete'):
                native_execution_report(net, drafter)
        net.prefill_transport.executed = required | {'fp8_tiled_projection', 'fp8_packet_projection'}
        native_execution_report(net, drafter)
        for obj, field, value in ((net.dense['a'], 'executed', 1),      # the decode lane alone: no prefill row ran
                                   (net.dense['a'], 'executed', 2),      # the prefill lane alone: no decode row ran
                                   (drafter.dense['q'], 'executed', 0),
                                   (net.mhc, 'executed', set()),
                                   (net.shared_mlp[1], 'executed', False),
                                   (net.shared_overlap, 'executed', False),
                                   (net, '_router_tensorcore', set())):
            before = getattr(obj, field)
            setattr(obj, field, value)
            with self.assertRaisesRegex(RuntimeError, 'proof is incomplete'):
                native_execution_report(net, drafter)
            setattr(obj, field, before)

    def test_nonfinite_prefill_releases_slots_and_refuses_readiness(self):
        for failure in ('hidden', 'aux', 'head'):
            with self.subTest(failure=failure):
                caches = MagicMock()
                caches.device, caches.snapshots = 'cpu', 2
                caches.pool.rows_in_use = 0
                caches.pool.num_blocks = 8
                caches.slots.owner = [-1, -1]
                caches.slots.take.return_value = 1
                h = torch.ones(1, 8)
                aux = torch.ones(4, 16)
                logits = torch.ones(1, 32)
                {'hidden': h, 'aux': aux, 'head': logits}[failure].flatten()[0] = float('nan')
                engine = NS(caches=caches, F=NS(block=1), prefill_chunk=4, max_context=8,
                            memory=MagicMock(), drafter=MagicMock(),
                            net=NS(comm=Comm(), head=lambda x: logits),
                            _prefill_forward=lambda step: (h, aux))
                with self.assertRaisesRegex(FloatingPointError, 'non-finite model output'):
                    Glm53Engine._warmup_prefill_memory(engine)
                caches.pool.release.assert_called_once_with(0)
                caches.slots.give.assert_called_once_with(1)
                caches.reset.assert_called_once()
                engine.drafter.observe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
