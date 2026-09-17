"""The next arm must leave usable, prefix-aligned sampling evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from engine.profiles.glm53.incident_logits import capture


class IncidentLogitsTests(unittest.TestCase):
    def test_full_generation_records_restore_modes_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = [10, 11] + [12] * 7
            raw = torch.tensor([[1., 2., 3.]], dtype=torch.bfloat16)
            block = raw.float()
            dists = block.softmax(-1)
            uniforms = [torch.tensor([.25])]
            saved = [value.clone() for value in (raw, block, dists, uniforms[0])]
            adapter = SimpleNamespace(net=SimpleNamespace(rank=0, incident_audit_root=Path(directory)/'audit'),
                                      incident_modes={0: 20}, tokens={0: prefix}, prompt_len={0: 2},
                                      nonces={0: 1}, seeds={0: 7},
                                      _generated_count=lambda seq: 7, _row_key=lambda seq: 123)
            for admission, mode in enumerate((20, 21, 22, 23, 24), 1):
                adapter.nonces[0] = admission
                adapter.incident_modes[0] = mode
                path = capture(adapter, [(0, raw, [], None)], block, dists, [1.], [-1], [1.],
                               uniforms, [2], [(0, [1], None)])
                record = torch.load(path, weights_only=True)
                self.assertEqual(path.name, f'admit{admission}-gen7.pt')
                self.assertEqual(record['mode'], mode)
                self.assertEqual(record['picks'], [2])
                self.assertEqual(record['committed'], [1])
                self.assertEqual(record['prefix_ids_sha256'],
                                 hashlib.sha256(json.dumps(prefix, separators=(',', ':')).encode()).hexdigest())
                self.assertEqual(record['prompt_len'], 2)
                self.assertEqual(record['seed'], 7)
                for key, value in zip(('raw', 'processed', 'probabilities', 'uniforms'), saved):
                    self.assertTrue(torch.equal(record[key], value))
                with self.assertRaises(FileExistsError):
                    capture(adapter, [(0, raw, [], None)], block, dists, [1.], [-1], [1.],
                            uniforms, [2], [(0, [1], None)])
            for value, before in zip((raw, block, dists, uniforms[0]), saved):
                self.assertTrue(torch.equal(value, before))
            adapter.net.rank = 1
            self.assertIsNone(capture(adapter, [(0, raw, [], None)], block, dists, [], [], [], [], [], []))


if __name__ == '__main__':
    unittest.main()
