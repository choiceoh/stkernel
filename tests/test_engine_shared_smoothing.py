"""A token sent by one TP rank must retain every receiving rank's linear map."""
import unittest

import torch

from engine.base.comm import LocalTP
from engine.profiles.glm53 import lanes
from engine.profiles.glm53.net import Glm53Net, rmsnorm
from tests.test_engine_glm53 import tiny_facts


class SharedInputSmoothingTests(unittest.TestCase):
    def run_case(self, calibrated):
        facts = tiny_facts()

        def worker(comm):
            rank, hidden = comm.rank, facts.hidden
            norm = torch.ones(hidden, dtype=torch.bfloat16)
            weight = (torch.eye(hidden)[:4] * (2 ** rank)).bfloat16()
            x = ((torch.arange(hidden) % 5 + rank + 1).view(1, -1)).bfloat16()
            amax = torch.full((hidden,), float(2 ** (3 * rank))) if rank in calibrated else None

            def make():
                net = Glm53Net(facts, comm, lanes.reference(), layers=[0])
                net.p = {'L0.in_norm': norm.clone(), 'L0.kda.in_proj': weight.clone()}
                return net

            logical = comm.all_gather(rmsnorm(x, norm, facts.rms_eps), dim=0)
            expected = torch.nn.functional.linear(logical, weight)
            old = make()
            old_packs = old.smooth_inputs(lambda _: amax)
            old_weight = old_packs.get('L0.kda.in_proj', (weight, None))[0]
            old_input = comm.all_gather(rmsnorm(x, old.p['L0.in_norm'], facts.rms_eps), dim=0)
            broken = torch.nn.functional.linear(old_input, old_weight)

            net = make()
            packs = net.smooth_inputs(lambda _: amax, shared_comm=comm)
            packed_weight = packs.get('L0.kda.in_proj', (weight, None))[0]
            received = comm.all_gather(rmsnorm(x, net.p['L0.in_norm'], facts.rms_eps), dim=0)
            actual = torch.nn.functional.linear(received, packed_weight)
            return dict(norm=net.p['L0.in_norm'], before=expected, after=actual,
                        old_equal=torch.equal(broken, expected), folded=bool(packs))

        rows = LocalTP(4, timeout_s=10).run(worker)
        for row in rows:
            torch.testing.assert_close(row['after'], row['before'], rtol=0, atol=0)
            torch.testing.assert_close(row['norm'], rows[0]['norm'], rtol=0, atol=0)
            self.assertEqual(row['folded'], bool(calibrated))
        return rows

    def test_cross_rank_inputs_preserve_each_receivers_projection(self):
        rows = self.run_case({0, 1, 2, 3})
        self.assertTrue(all(not row['old_equal'] for row in rows),
                        'independent rank factors must reproduce the broken all-gather path')

    def test_one_rank_calibration_supplies_the_common_domain(self):
        self.run_case({0})

    def test_no_calibration_leaves_every_rank_unfolded(self):
        self.run_case(set())


if __name__ == '__main__':
    unittest.main()
