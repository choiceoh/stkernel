"""Copy removal preserves BF16 payloads, output ownership and arena page maps."""
from types import SimpleNamespace as NS
import unittest

import torch

from engine.profiles.glm53.decode_graphs import GraphCaches
from engine.profiles.glm53.lanes import _mla_output


class CopyContracts(unittest.TestCase):
    def test_single_result_preserves_every_bf16_bit_pattern(self):
        result = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(8, 16, 512)
        expected = torch.cat([result], dim=1)
        got = _mla_output([result])
        self.assertIs(got, result)
        self.assertTrue(torch.equal(got.view(torch.int16), expected.view(torch.int16)))

    def test_explicit_destination_and_multiple_head_groups_keep_their_ownership(self):
        parts = [torch.full((8, 16, 32), i, dtype=torch.bfloat16) for i in range(4)]
        for count in (1, 4):
            owner = torch.full((10, 16*count, 32), -7, dtype=torch.bfloat16)
            destination = owner[1:9]
            got = _mla_output(parts[:count], destination)
            self.assertIs(got, destination)
            for i in range(count):
                self.assertTrue(torch.equal(got[:, i*16:(i+1)*16], parts[i]))
            self.assertTrue(bool((owner[0] == -7).all() & (owner[-1] == -7).all()))
        got = _mla_output(parts)
        self.assertEqual(got.shape, (8, 64, 32))
        self.assertTrue(all(got.data_ptr() != p.data_ptr() for p in parts))

    def test_gather_clamps_only_its_new_copy_after_row_changes(self):
        parent = torch.arange(6*24, dtype=torch.int32).reshape(6, 24) % 17 - 1
        table = parent[:, ::2]
        ids = torch.tensor([5, 2, 0, 3])
        caches = GraphCaches(NS(F=NS(kpool=4), layout=None, block_table=table), ids, ids, 4096)
        for phase in range(4):
            table[phase].fill_(-1)
            before = table.clone()
            ids.copy_((torch.arange(4)+phase).flip(0) % 6)
            caches.gather()
            self.assertTrue(torch.equal(caches.block_table, before.index_select(0, ids).clamp_min(0)))
            self.assertNotEqual(caches.block_table.data_ptr(), table.data_ptr())
            self.assertTrue(torch.equal(table, before))
            caches.block_table.fill_(73)
            self.assertTrue(torch.equal(table, before))


@unittest.skipUnless(torch.cuda.is_available(), 'requires the existing reserved GPU')
class CaptureContracts(unittest.TestCase):
    def test_changed_replays_preserve_payload_and_page_maps(self):
        from probes.engine_decode_no_copy import check
        check(lambda *a, **kw: None)


if __name__ == '__main__':
    unittest.main()
