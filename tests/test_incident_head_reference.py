"""CPU gates for checkpoint provenance and the eager vocabulary-head control."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

from safetensors.torch import save_file
import torch

from engine.profiles.glm53.incident_head_reference import checkpoint_weight, project


class HeadReferenceTests(unittest.TestCase):
    def fixture(self, root, weight=None):
        weight = (torch.tensor([[1., 2.], [-3., 4.], [5., -6.]], dtype=torch.bfloat16)
                  if weight is None else weight)
        path = str(Path(root) / 'rank.safetensors')
        save_file({'head': weight}, path)
        native = torch.tensor([[7., 8., 9.]], dtype=torch.bfloat16)
        net = NS(incident_rank_file=path, rank=0, vp=3, F=NS(hidden=2),
                 incident_audit_root=Path(root) / 'incident-prefill-audit',
                 head_local=lambda hidden: native)
        return net, native, weight

    def test_same_hidden_pair_preserves_native_and_uses_actual_checkpoint(self):
        with TemporaryDirectory() as root:
            net, native, weight = self.fixture(root)
            hidden = torch.tensor([[2., -1.]], dtype=torch.bfloat16)
            for mode, admission in ((20, 1), (28, 2)):
                actual = project(net, hidden, mode=mode, admission=admission,
                                 generation=0, prefix_sha256='same-prefix')
                expected = native if mode == 20 else hidden @ weight.T
                self.assertTrue(torch.equal(actual, expected))
                record = torch.load(Path(root) / 'incident-head' / f'admit{admission}-gen0-rank0.pt',
                                    weights_only=True)
                self.assertTrue(torch.equal(record['hidden'], hidden))
                self.assertTrue(torch.equal(record['native'], native))
                self.assertTrue(torch.equal(record['bf16'], hidden @ weight.T))
                self.assertEqual(record['prefix_sha256'], 'same-prefix')
            self.assertIs(checkpoint_weight(net, hidden.device), net._incident_head_weight)
            self.assertIs(net.head_local(hidden), native)

    def test_wrong_dtype_or_shard_shape_is_rejected(self):
        with TemporaryDirectory() as root:
            for weight in (torch.zeros(3, 2), torch.zeros(4, 2, dtype=torch.bfloat16)):
                net, _, _ = self.fixture(root, weight)
                with self.assertRaisesRegex(ValueError, 'original BF16 vocabulary shard'):
                    checkpoint_weight(net, torch.device('cpu'))

    def test_control_is_inactive_for_other_modes_and_refuses_batched_rows(self):
        with TemporaryDirectory() as root:
            net, native, _ = self.fixture(root)
            args = dict(admission=1, generation=0, prefix_sha256='p')
            self.assertIs(project(net, torch.zeros(2, 2), mode=0, **args), native)
            self.assertFalse(hasattr(net, '_incident_head_weight'))
            with self.assertRaisesRegex(ValueError, 'one eager BF16 hidden row'):
                project(net, torch.zeros(2, 2, dtype=torch.bfloat16), mode=28, **args)


if __name__ == '__main__':
    unittest.main()
