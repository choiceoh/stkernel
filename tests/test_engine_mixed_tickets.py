"""Ticket ownership and real four-process host agreement, without GPU work."""
from datetime import timedelta
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref

from engine.base.comm import Comm
from engine.modules.mixed_experts import ExpertInvocation, plan_experts
from engine.modules.mixed_completion import plan_cold
from engine.modules.mixed_tickets import MixedLayerScheduler, signature


class Fence:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


class Owner:
    """Only the GPU component is simulated; plans and scheduler are real."""
    def __init__(self, generation=1, quota=2, rank=0):
        self.plan = plan_experts([list(range(8))], [list(range(8))]*17,
            identity=ExpertInvocation(3, 1, generation, 1), hot_route_quota=0)
        self.cold = plan_cold(self.plan, task_quota=quota)
        self.state, self.next_window = 'new', 0
        self.stream = 'cpu-device-fixture'
        self.fence, self.calls, self.rank = Fence(), [], rank
        self.bad = self.fail_advance = self.fail_fence = False

    def validate(self, identity):
        if self.bad or identity != self.plan.identity:
            raise RuntimeError('source changed')

    def begin(self, identity):
        import torch
        self.state, self.next_window = 'decode', 0
        self.calls.append('begin')
        return torch.tensor([self.rank+1.])

    def advance(self, identity):
        if self.fail_advance:
            raise RuntimeError('injected cold dispatch failure')
        self.next_window += 1
        self.calls.append('advance')
        self.state = 'routed' if self.next_window == len(self.cold.windows) else 'cold'
        return self.state == 'routed'

    def finish(self, identity):
        import torch
        self.state = 'complete'
        self.calls.append('finish')
        return torch.tensor([10.*(self.rank+1)])

    def reader_fence(self):
        if self.fail_fence:
            raise RuntimeError('injected event recording failure')
        return self.fence


def complete(scheduler, key):
    scheduler.begin(key)
    while not scheduler.advance(key):
        pass
    scheduler.finish(key)


class TicketTests(unittest.TestCase):
    def test_quota_is_agreed_even_when_this_histogram_needs_only_one_window(self):
        a, b = Owner(quota=48), Owner(quota=96)
        self.assertEqual(a.cold.windows, b.cold.windows)
        self.assertNotEqual(signature(a), signature(b))

    def test_decode_first_and_bounded_cold_dispatch(self):
        s, owner = MixedLayerScheduler(3, Comm()), Owner()
        key = s.admit(owner, request='request', slot=0)
        with self.assertRaises(RuntimeError):
            s.result(key, prefill=True)
        s.begin(key)
        self.assertEqual(s.result(key, prefill=False)[0].item(), 1.)
        with self.assertRaises(RuntimeError):
            s.finish(key)
        for window in range(len(owner.cold.windows)):
            self.assertEqual(s.advance(key), window+1 == len(owner.cold.windows))
            self.assertEqual(owner.next_window, window+1)
        s.finish(key)
        self.assertEqual(s.result(key, prefill=True)[0].item(), 10.)
        self.assertEqual(owner.calls, ['begin']+['advance']*4+['finish'])
        with self.assertRaisesRegex(RuntimeError, 'consumer'):
            s.release(key)
        s.release(key, consumer_fence=Fence())
        self.assertTrue(s.reap(key))

    def test_cancel_keeps_owners_until_both_fences_and_refuses_stale_handles(self):
        s, owner, consumer = MixedLayerScheduler(3, Comm()), Owner(), Fence(False)
        reader, ref = owner.fence, weakref.ref(owner)
        reader.ready = False
        old = s.admit(owner, request='old', slot=0)
        s.begin(old); s.result(old, prefill=False)
        s.advance(old)
        owner.bad = True  # cancellation must not validate changed sources
        s.cancel(old, consumer_fence=consumer)
        del owner; gc.collect()
        self.assertIsNotNone(ref())
        self.assertFalse(s.reap(old))
        reader.ready = True
        self.assertFalse(s.reap(old))
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            s.admit(Owner(2), request='new', slot=0)
        consumer.ready = True
        self.assertTrue(s.reap(old)); gc.collect()
        self.assertIsNone(ref())
        with self.assertRaisesRegex(RuntimeError, 'generation'):
            s.admit(Owner(), request='new', slot=0)
        new = s.admit(Owner(2), request='new', slot=0)
        self.assertNotEqual(new.serial, old.serial)
        for method in (s.begin, s.advance, s.finish, s.cancel, s.reap):
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                method(old)
        s.cancel(new); self.assertTrue(s.reap(new))

    def test_complete_cancel_and_owner_alias_cannot_overwrite_borrowed_output(self):
        s, owner = MixedLayerScheduler(3, Comm()), Owner()
        first = s.admit(owner, request='a', slot=0)
        with self.assertRaisesRegex(RuntimeError, 'leased'):
            s.admit(owner, request='a', slot=1)
        complete(s, first)
        s.result(first, prefill=True)
        with self.assertRaisesRegex(RuntimeError, 'leased'):
            s.admit(owner, request='a', slot=1)
        s.cancel(first, consumer_fence=Fence())
        self.assertTrue(s.reap(first))
        second = s.admit(owner, request='a', slot=0)
        complete(s, second); s.release(second); self.assertTrue(s.reap(second))

    def test_source_change_out_of_band_dispatch_and_failed_publication_are_drainable(self):
        for failure in ('source', 'dispatch', 'fence'):
            with self.subTest(failure=failure):
                s, owner = MixedLayerScheduler(3, Comm()), Owner()
                key = s.admit(owner, request='a', slot=0)
                if failure == 'source':
                    owner.bad = True
                elif failure == 'dispatch':
                    owner.begin(owner.plan.identity)
                else:
                    owner.fail_fence = True
                with self.assertRaises(RuntimeError):
                    s.begin(key)
                owner.fail_fence = False
                s.cancel(key); self.assertTrue(s.reap(key))

    def test_failed_preparation_and_bad_bounds_never_occupy_a_slot(self):
        s = MixedLayerScheduler(3, Comm())
        for owner, request, slot, error in ((None, 'a', 0, 'local preparation failed'),
                (Owner(), '', 0, None), (Owner(), 'a', True, None), (Owner(), 'a', 4, None)):
            with self.assertRaises(RuntimeError):
                s.admit(owner, request=request, slot=slot, preparation_error=error)
            self.assertEqual(s._entries, {})
        for capacity in (0, 5, True):
            with self.assertRaises(ValueError):
                MixedLayerScheduler(3, Comm(), capacity=capacity)

    def test_tickets_cannot_race_the_native_shared_scratch_on_two_streams(self):
        s = MixedLayerScheduler(3, Comm())
        first = s.admit(Owner(), request='a', slot=0)
        other = Owner(); other.stream = 'other-stream'
        with self.assertRaisesRegex(RuntimeError, 'one eager'):
            s.admit(other, request='b', slot=1)
        s.cancel(first); self.assertTrue(s.reap(first))


def gloo_worker(rank, rendezvous):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=4,
        init_method='file://'+rendezvous, timeout=timedelta(seconds=25))
    try:
        comm = Comm(4, rank, dist.group.WORLD, dist.group.WORLD)
        def refused(fn):
            try:
                fn()
            except RuntimeError:
                return
            raise AssertionError('a rank proceeded after another rank refused work')
        for failure in ('prepare', 'descriptor', 'cold_descriptor', 'order', 'cold', 'publication',
                        'pending_packets', 'shape', 'packed_agreement', 'none'):
            s, owner = MixedLayerScheduler(3, comm), Owner(rank=rank)
            if failure in ('cold_descriptor', 'packed_agreement'):
                # Real ranks may build equivalent descriptors from different
                # host representations. Agree actual bytes, never object type.
                from dataclasses import replace
                import numpy as np
                from engine.modules.mixed_experts import plan_experts_packed
                if rank % 2:
                    owner.plan = plan_experts_packed(np.asarray(owner.plan.decode, dtype=np.int32),
                        np.asarray(owner.plan.prefill, dtype=np.int32), identity=owner.plan.identity,
                        hot_route_quota=owner.plan.quota)
                    owner.cold = plan_cold(owner.plan, task_quota=owner.cold.task_quota)
                if failure == 'cold_descriptor' and rank == 2:
                    sources = list(owner.cold.sources)
                    e, dest, row, slot = sources[0]
                    sources[0] = (e, dest+1, row, slot)
                    owner.cold = replace(owner.cold, sources=tuple(sources))
                if failure == 'cold_descriptor':
                    refused(lambda: s.admit(owner, request='a', slot=0))
                    assert not s._entries
                    continue
            if failure == 'none':
                # The fixed process-group sum never asks the optional native
                # transport for tensor-dependent eligibility.
                comm.transport = SimpleNamespace(pending=None, packet_failed=False)
            if failure == 'prepare':
                refused(lambda: s.admit(None if rank == 2 else owner, request='a', slot=0,
                    preparation_error='rank 2 could not prepare' if rank == 2 else None))
                assert not s._entries
                continue
            if failure == 'descriptor':
                owner = Owner(rank=rank, quota=1 if rank == 2 else 2)
                refused(lambda: s.admit(owner, request='a', slot=0))
                assert not s._entries
                continue
            key = s.admit(owner, request='a', slot=0)
            if failure == 'order':
                refused(lambda: s.begin(key) if rank != 2 else s.cancel(key))
            elif failure in ('pending_packets', 'shape'):
                if failure == 'pending_packets' and rank == 2:
                    comm.transport = SimpleNamespace(pending=object(), packet_failed=False)
                if failure == 'shape' and rank == 2:
                    original = owner.begin
                    owner.begin = lambda identity: original(identity).repeat(2)
                refused(lambda: s.begin(key))
                comm.transport = None
            elif failure == 'publication':
                owner.fail_fence = rank == 2
                refused(lambda: s.begin(key))
                owner.fail_fence = False
            else:
                s.begin(key)
                assert s.result(key, prefill=False)[0].item() == 10.
                if failure == 'cold':
                    owner.fail_advance = rank == 2
                    refused(lambda: s.advance(key))
                    # Different local cursors must not block cancellation.
                    assert owner.next_window == (0 if rank == 2 else 1)
                else:
                    while not s.advance(key):
                        pass
                    s.finish(key)
                    assert s.result(key, prefill=True)[0].item() == 100.
            owner.fence.ready = rank != 3
            consumer = Fence(rank != 1)
            s.cancel(key, consumer_fence=consumer)
            assert not s.reap(key)
            owner.fence.ready = True
            assert not s.reap(key)
            consumer.ready = True
            assert s.reap(key)
            assert not s._entries
    finally:
        dist.destroy_process_group()


class FourProcessAgreementTests(unittest.TestCase):
    def test_tp4_gloo_errors_order_reductions_and_slowest_rank_retirement(self):
        import torch.distributed as dist
        import torch.multiprocessing as mp
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest('Gloo is unavailable')
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(gloo_worker, args=(str(Path(temp)/'rendezvous'),), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
