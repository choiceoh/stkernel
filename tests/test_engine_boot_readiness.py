"""Slow rank preparation must not consume the serving collective deadline."""
import importlib.util
import json
import multiprocessing
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

from engine.base.comm import Comm, LocalTP


def _worker(rank, directory, mode):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group('gloo', init_method=(root / 'store').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    # Match the separate, short-deadline serving control group. CPU-only: no NCCL/GPU.
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=1))
    preparation = dist.new_group(backend='gloo', timeout=timedelta(seconds=6))
    comm = Comm(2, rank, dist.group.WORLD, control, preparation=preparation)
    result = {}
    try:
        if mode == 'slow':
            # Test both a slow follower and a slow coordinator. Each gap is longer
            # than the group's ordinary timeout; the next real collective must work.
            for late in (1, 0):
                if rank == late:
                    time.sleep(2)
                comm.wait_prepared(f'loaded-{late}', timeout_s=6, final=late == 0)
                value = torch.tensor([rank + 1])
                comm.all_reduce(value)
                assert value.item() == 3
                assert comm.gather_objects(rank) == [0, 1]
            assert comm.preparation is None
            result['passed'] = True
        elif mode == 'mismatch':
            comm.wait_prepared(f'phase-{rank}', timeout_s=6)
            raise AssertionError('different preparation phases were accepted')
        elif mode == 'missing':
            if rank == 1:
                time.sleep(2)
            else:
                start = time.monotonic()
                try:
                    comm.wait_prepared('weights-loaded', timeout_s=.3)
                    raise AssertionError('missing rank was accepted')
                finally:
                    result['elapsed_s'] = time.monotonic() - start
        elif mode == 'dead':
            if rank == 1:
                dist.destroy_process_group()
                return
            comm.wait_prepared('weights-loaded', timeout_s=3)
            raise AssertionError('dead rank was accepted')
    except RuntimeError as exc:
        result['error'] = str(exc)
    finally:
        (root / f'{rank}.json').write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


class PreparationTests(unittest.TestCase):
    def test_world_one_needs_no_distributed_runtime(self):
        Comm().wait_prepared('loaded')
        for timeout in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                Comm().wait_prepared('loaded', timeout_s=timeout)
        with self.assertRaises(ValueError):
            Comm().wait_prepared('')
        with self.assertRaisesRegex(RuntimeError, 'boot Gloo group'):
            Comm(2).wait_prepared('loaded')

    def test_local_tp_phase_agreement_and_failure(self):
        tp = LocalTP(2, timeout_s=1)
        self.assertEqual(tp.run(lambda comm: comm.wait_prepared('loaded')), [None, None])
        with self.assertRaises(RuntimeError) as caught:
            tp.run(lambda comm: comm.wait_prepared(str(comm.rank)))
        self.assertIn('different preparation phases', str(caught.exception.__cause__))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch with Gloo')
class DistributedPreparationTests(unittest.TestCase):
    def run_case(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            ctx = multiprocessing.get_context('spawn')
            workers = [ctx.Process(target=_worker, args=(rank, directory, mode)) for rank in range(2)]
            try:
                for process in workers:
                    process.start()
                deadline = time.monotonic() + 45
                for process in workers:
                    process.join(max(0, deadline - time.monotonic()))
                    self.assertEqual(process.exitcode, 0, 'rank crashed or exceeded test deadline')
                return [json.loads((Path(directory) / f'{rank}.json').read_text()) for rank in range(2)]
            finally:
                for process in workers:
                    if process.is_alive():
                        process.kill()
                    if process.pid is not None:
                        process.join(5)

    def test_slow_preparation_outlives_serving_timeout_on_either_rank(self):
        self.assertEqual(self.run_case('slow'), [{'passed': True}] * 2)

    def test_different_phases_fail_on_every_rank(self):
        for result in self.run_case('mismatch'):
            self.assertIn('different preparation phases', result.get('error', ''))

    def test_missing_peer_fails_at_the_preparation_deadline(self):
        result = self.run_case('missing')[0]
        self.assertIn('weights-loaded failed on rank 0', result.get('error', ''))
        self.assertIn('1', result['error'])
        self.assertLess(result['elapsed_s'], 1.5)

    def test_dead_peer_fails(self):
        result = self.run_case('dead')[0]
        self.assertIn('weights-loaded failed on rank 0', result.get('error', ''))


if __name__ == '__main__':
    unittest.main()
