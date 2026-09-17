"""Safety contracts for the temporary same-boot scale controls."""
from types import SimpleNamespace
import unittest

import torch

from engine.profiles.glm53.incident_redhat_scales import bind_layer, select


class RedHatScaleControlTests(unittest.TestCase):
    def test_prepare_preserves_shared_weights_and_survives_consumed_scale_storage(self):
        first, second = torch.zeros(2, 4, 4, dtype=torch.uint8), torch.zeros(2, 4, 2, dtype=torch.uint8)
        seen = []

        def prepare(w13, sf13, w2, sf2, topk, limit, *, scales):
            self.assertIs(w13, first)
            self.assertIs(w2, second)
            self.assertEqual(sf13.numel(), 16)
            self.assertEqual(sf2.numel(), 8)
            seen.append((sf13.data_ptr(), sf2.data_ptr(), scales))
            # Model the serving prepare call retiring the raw scale owner.
            sf13.untyped_storage().resize_(0)
            sf2.untyped_storage().resize_(0)
            return object()

        net = SimpleNamespace(rank=0, F=SimpleNamespace(swiglu_limit=10., topk_experts=2),
                              p={'L3.moe.w13': first, 'L3.moe.w2': second},
                              lanes=SimpleNamespace(moe_prepare=prepare, moe=lambda *a, **k: None,
                                                    moe_packets=None, moe_packets_supported=None))
        values = dict(w13_sf=torch.ones(2, 8), w2_sf=torch.ones(2, 4),
                      w13_alpha=torch.tensor([.25, .5]), w2_alpha=torch.tensor([.5, .25]),
                      a13_scale=torch.tensor([.125, .25]), a2_scale=torch.tensor([.25, .125]))
        bind_layer(net, 3, values)
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0][:2], seen[1][:2])
        variants = net._incident_redhat_layers[3]
        self.assertTrue(torch.equal(variants['weight']['scales'].input13, torch.ones(2)))
        self.assertTrue(torch.equal(variants['calibrated']['scales'].alpha13, torch.tensor([.03125, .125])))
        self.assertTrue(torch.equal(variants['calibrated']['scales'].alpha2, torch.tensor([.125, .03125])))

    def test_selection_restores_all_owners_and_audit_after_failed_forward(self):
        attrs = ('_experts', '_expert_views', '_quant_scales', '_packet_experts', '_packet_capabilities')
        fields = ('expert', 'views', 'scales', 'packet', 'packet_supported')
        original = {attr: {3: object()} for attr in attrs}
        candidate = {field: object() for field in fields}
        net = SimpleNamespace(**original, layers=[0, 3], F=SimpleNamespace(is_moe=lambda layer: layer == 3),
                              incident_audit_root='audit', _incident_redhat_layers={3: {'calibrated': candidate}})
        with self.assertRaisesRegex(RuntimeError, 'failed forward'):
            with select(net, 'calibrated'):
                self.assertIsNone(net.incident_audit_root)
                for attr, field in zip(attrs, fields):
                    self.assertIs(getattr(net, attr)[3], candidate[field])
                raise RuntimeError('failed forward')
        for attr in attrs:
            self.assertIs(getattr(net, attr), original[attr])
        self.assertEqual(net.incident_audit_root, 'audit')


if __name__ == '__main__':
    unittest.main()
