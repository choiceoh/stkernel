"""Packed planning must preserve every scalar route and task descriptor."""
from dataclasses import fields, replace
from types import SimpleNamespace
import unittest

import numpy as np

from engine.modules.mixed_experts import ExpertInvocation, plan_experts, plan_experts_packed
from engine.modules.mixed_completion import plan_cold
from engine.modules.mixed_tickets import signature
from engine.modules.route_table import RouteTable

IDENTITY = ExpertInvocation(3, 1, 2, 3)


class PackedPlanTests(unittest.TestCase):
    def pair(self, decode, prefill, hot=128, cold=48):
        a = plan_experts(decode.tolist(), prefill.tolist(), identity=IDENTITY, hot_route_quota=hot)
        b = plan_experts_packed(decode, prefill, identity=IDENTITY, hot_route_quota=hot)
        ca, cb = plan_cold(a, task_quota=cold), plan_cold(b, task_quota=cold)
        for old, new in ((a, b), (ca, cb)):
            for field in fields(old):
                x, y = getattr(old, field.name), getattr(new, field.name)
                if isinstance(y, RouteTable):
                    np.testing.assert_array_equal(np.asarray(x, dtype=np.int32).reshape(-1, y.columns), y.array())
                else:
                    self.assertEqual(x, y, field.name)
        self.assertEqual(a.work(), b.work())
        self.assertEqual(signature(SimpleNamespace(plan=a, cold=ca)), signature(SimpleNamespace(plan=b, cold=cb)))
        return b, cb

    def test_all_widths_tails_quotas_and_zero_cold_match_the_scalar_reference(self):
        for d in (1, 7, 8, 9, 16, 24, 32):
            decode = np.tile(np.arange(8, dtype=np.int32), (d, 1))
            for p in (1, 14, 127, 128, 129, 140, 143, 256):
                for quota in (0, 25, 128):
                    with self.subTest(d=d, p=p, quota=quota):
                        self.pair(decode, np.tile(decode[0], (p, 1)), quota, 1 if p % 2 else 128)

    def test_long_shuffled_routes_retain_every_original_row_slot_and_padded_task(self):
        rng = np.random.default_rng(89520)
        for d, p in ((8, 9240), (32, 9240), (8, 32768), (32, 32768)):
            routes = np.argsort(rng.random((p, 288)), axis=1)[:, :8].astype(np.int32)
            for quota in (0, 128):
                with self.subTest(d=d, p=p, quota=quota):
                    self.pair(routes[:d], routes, quota)

    def test_host_storage_and_upload_are_owned_and_cannot_be_made_writable(self):
        import torch
        routes = np.tile(np.arange(8, dtype=np.int32), (140, 1))
        plan, cold = self.pair(routes[:8], routes)
        digest = signature(SimpleNamespace(plan=plan, cold=cold))
        routes[:] = 100
        for table in (plan.prefill, plan.cold_routes, cold.sources):
            with self.assertRaises(ValueError):
                table.array().setflags(write=True)
            with self.assertRaises(ValueError):
                table.array()[0, 0] = 99
            copied = torch.tensor(table.array(), dtype=torch.int32)
            copied.fill_(99)
            self.assertFalse(bool((table.array() == 99).all()))
        self.assertEqual(signature(SimpleNamespace(plan=plan, cold=cold)), digest)
        with self.assertRaises(AttributeError):
            plan.prefill.data = b''

    def test_noncontiguous_and_big_endian_inputs_have_the_same_descriptor(self):
        rows = np.tile(np.arange(16, dtype=np.int32), (140, 1))[:, ::2]
        plan, cold = self.pair(rows[:8], rows)
        other, rest = self.pair(rows[:8].astype('>i4'), rows.astype('>i4'))
        self.assertEqual(signature(SimpleNamespace(plan=plan, cold=cold)), signature(SimpleNamespace(plan=other, cold=rest)))

    def test_same_histogram_new_routes_and_each_cold_descriptor_field_change_digest(self):
        rows = (np.arange(140 * 8, dtype=np.int32).reshape(-1, 8) % 288)
        plan, cold = self.pair(rows[:8], rows)
        owner = SimpleNamespace(plan=plan, cold=cold)
        before = signature(owner)
        changed = plan_experts_packed(rows[:8], np.roll(rows, 1, axis=0), identity=IDENTITY)
        self.assertEqual(changed.prefill_counts, plan.prefill_counts)
        self.assertNotEqual(before, signature(SimpleNamespace(plan=changed, cold=plan_cold(changed))))
        for field in fields(cold):
            value = getattr(cold, field.name)
            if isinstance(value, RouteTable):
                array = value.array().copy(); array[0, 1] += 1
                value = RouteTable.pack(array)
            elif field.name == 'windows':
                value = ((value[0][0], value[0][1] - 1),) + value[1:]
            elif isinstance(value, tuple):
                value = (value[0] + 1,) + value[1:]
            else:
                value += 1
            with self.subTest(field=field.name):
                self.assertNotEqual(before, signature(SimpleNamespace(plan=plan, cold=replace(cold, **{field.name: value}))))
        for field in ('layer', 'epoch', 'slot_generation', 'source_generation'):
            identity = replace(IDENTITY, **{field: getattr(IDENTITY, field) + 1})
            self.assertNotEqual(before, signature(SimpleNamespace(plan=replace(plan, identity=identity), cold=cold)))

    def test_invalid_arrays_cannot_truncate_overflow_or_alias_routes(self):
        valid = np.arange(8, dtype=np.int32).reshape(1, 8)
        bad = (valid.tolist(), valid.astype(bool), valid.astype(np.int64), valid.astype(np.float32),
            np.empty((0, 8), np.int32), np.tile(valid, (33, 1)), valid[:, :7], valid.ravel(),
            np.zeros((1, 8), np.int32), valid - 1, valid + 288)
        for value in bad:
            with self.subTest(value=str(value)[:100]), self.assertRaises(ValueError):
                plan_experts_packed(value, valid, identity=IDENTITY)
        for quota in (-1, 129, True, 1.5):
            with self.assertRaises(ValueError):
                plan_experts_packed(valid, valid, identity=IDENTITY, hot_route_quota=quota)
        with self.assertRaises(ValueError):
            plan_experts_packed(valid, valid, identity=None)
        for data, width in ((bytearray(32), 8), (bytes(31), 8), (b'', 0), (b'', True)):
            with self.assertRaises(ValueError):
                RouteTable(data, width)


if __name__ == '__main__':
    unittest.main()
