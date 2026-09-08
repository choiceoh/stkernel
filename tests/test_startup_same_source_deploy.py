"""Publication replay must never introduce different code or an unadmitted SHA."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bench'))
from startup_same_source_deploy import require_identical


class SameSourceTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'manifest.tsv': 'admitted-revision-hash', 'kernel.cu': 'kernel-hash'}
        self.nodes = {f'srv{n}': {'files': {k: {'sha256': v} for k, v in self.expected.items()}}
                      for n in range(1, 5)}

    def test_all_nodes_with_exact_source_are_accepted(self):
        require_identical(self.expected, self.nodes)

    def test_one_node_with_different_code_or_revision_is_rejected(self):
        for node in self.nodes:
            for name in self.expected:
                with self.subTest(node=node, file=name), self.assertRaises(ValueError):
                    snapshots = copy.deepcopy(self.nodes)
                    snapshots[node]['files'][name]['sha256'] = 'different'
                    require_identical(self.expected, snapshots)

    def test_missing_or_extra_deployed_source_is_rejected(self):
        for extra in (False, True):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                snapshots = copy.deepcopy(self.nodes)
                if extra:
                    snapshots['srv3']['files']['new.cu'] = {'sha256': 'new'}
                else:
                    del snapshots['srv3']['files']['kernel.cu']
                require_identical(self.expected, snapshots)

    def test_partial_cluster_is_rejected(self):
        del self.nodes['srv4']
        with self.assertRaises(ValueError):
            require_identical(self.expected, self.nodes)


if __name__ == '__main__':
    unittest.main()
