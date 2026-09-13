"""Burst exit conditions and TP4 agreement without a CUDA context."""
from types import SimpleNamespace as NS
import unittest
import torch

from engine.base.comm import LocalTP
from engine.profiles.glm53.bounded_loop import stop_at_boundary, agree_stop


class BoundedLoopTests(unittest.TestCase):
    def controls(self, rows):
        return dict(before=torch.full((rows,), 100, dtype=torch.int64),
                    current=torch.full((rows,), 107, dtype=torch.int64),
                    alive=torch.ones(rows, dtype=torch.bool),
                    reserved_end=torch.full((rows,), 128, dtype=torch.int64),
                    bucket_end=256, interrupted=torch.zeros(1, dtype=torch.int64),
                    step_tokens=7, block=64)

    def test_any_row_exit_prevents_the_next_iteration(self):
        for rows in (1, 4):
            with self.subTest(rows=rows):
                self.assertEqual(stop_at_boundary(**self.controls(rows)).tolist(), [0])
                cases = (
                    ("eos_or_limit", lambda c: c["alive"].__setitem__(-1, False)),
                    ("prefix_crossed", lambda c: (c["before"].__setitem__(-1, 125), c["current"].__setitem__(-1, 129))),
                    ("reservation", lambda c: c["reserved_end"].__setitem__(-1, 113)),
                    ("bucket", lambda c: c.update(bucket_end=113)),
                    ("interruption", lambda c: c["interrupted"].fill_(1)),
                )
                for reason, change in cases:
                    with self.subTest(reason=reason):
                        controls = self.controls(rows)
                        change(controls)
                        self.assertEqual(stop_at_boundary(**controls).tolist(), [1])

    def test_exact_reservation_and_bucket_fit_are_allowed(self):
        controls = self.controls(4)
        controls["reserved_end"].fill_(114)
        controls["bucket_end"] = 114
        self.assertEqual(stop_at_boundary(**controls).tolist(), [0])

    def test_rank_local_stop_is_shared_before_any_rank_branches(self):
        def rank(comm):
            comm.transport = NS(eligible_max=lambda vote: True)  # CPU transport oracle only
            outputs = []
            for stopping_rank in (-1, 0, 1, 2, 3):
                vote = torch.tensor([int(comm.rank == stopping_rank)], dtype=torch.int64)
                outputs.append(agree_stop(comm, vote).item())
            return outputs
        self.assertEqual(LocalTP(4, timeout_s=10).run(rank), [[0, 1, 1, 1, 1]] * 4)

    def test_rejects_incompatible_control_geometry(self):
        mutations = (
            lambda c: c.update(step_tokens=0), lambda c: c.update(block=7),
            lambda c: c.update(bucket_end=True), lambda c: c.update(before=torch.zeros(5, dtype=torch.int64)),
            lambda c: c.update(current=torch.zeros(4, dtype=torch.int32)),
            lambda c: c.update(alive=torch.ones(4, dtype=torch.int64)),
            lambda c: c.update(interrupted=torch.zeros(2, dtype=torch.int64)),
            lambda c: c.update(reserved_end=torch.zeros(3, dtype=torch.int64)),
        )
        for mutate in mutations:
            controls = self.controls(4)
            mutate(controls)
            with self.assertRaises(ValueError):
                stop_at_boundary(**controls)
        for comm in (NS(world_size=1, transport=None), NS(world_size=4, transport=None),
                     NS(world_size=4, transport=NS(eligible_max=lambda vote: False))):
            with self.assertRaises(ValueError):
                agree_stop(comm, torch.zeros(1, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
