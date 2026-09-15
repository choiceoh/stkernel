"""Reject unsupported per-layer architecture instead of silently running legacy geometry."""
import copy
import unittest

from engine.profiles.glm53.facts import architecture
from tests.test_engine_kernel_shape import GLM53_TEXT_CONFIG


def modern_config():
    t = copy.deepcopy(GLM53_TEXT_CONFIG)
    full = set(t['linear_attn_config']['full_attn_layers'])
    t.update(attention_bias=False,
             layer_types=['deepseek_sparse_attention' if i in full else 'linear_attention' for i in range(45)],
             mlp_layer_types=['dense'] * 3 + ['sparse'] * 42, indexer_types=['full'] * 45)
    return t


class ArchitectureTests(unittest.TestCase):
    def test_legacy_modern_and_full_attention_alias_describe_the_same_network(self):
        expected = architecture(GLM53_TEXT_CONFIG)
        t = modern_config()
        self.assertEqual(architecture({'text_config': t}), expected)
        t['layer_types'] = ['full_attention' if k == 'deepseek_sparse_attention' else k for k in t['layer_types']]
        self.assertEqual(architecture(t), expected)

    def test_unsupported_bias_and_conflicting_layer_descriptions_are_rejected(self):
        for key, index, value in (('attention_bias', None, True), ('layer_types', 0, 'deepseek_sparse_attention'),
                                  ('mlp_layer_types', 0, 'sparse'), ('indexer_types', 7, 'shared')):
            t = modern_config()
            if index is None:
                t[key] = value
            else:
                t[key][index] = value
            with self.subTest(key=key), self.assertRaisesRegex(AssertionError, key):
                architecture(t)

    def test_incomplete_and_unknown_layer_lists_are_rejected(self):
        for key in ('layer_types', 'mlp_layer_types', 'indexer_types'):
            for mutation in ('short', 'unknown'):
                t = modern_config()
                if mutation == 'short':
                    t[key].pop()
                else:
                    t[key][0] = 'unknown'
                with self.subTest(key=key, mutation=mutation), self.assertRaisesRegex(AssertionError, key):
                    architecture(t)

    def test_implicit_shared_indexer_schedules_are_rejected(self):
        for fields in ({'index_topk_freq': 2}, {'index_topk_pattern': 'F' * 7 + 'S' + 'F' * 37}):
            t = copy.deepcopy(GLM53_TEXT_CONFIG)
            t.update(fields)
            with self.assertRaisesRegex(AssertionError, 'indexer_types'):
                architecture(t)
        t = modern_config()
        t['index_topk_freq'] = 2  # an explicit list takes precedence, as in the model config
        self.assertEqual(architecture(t), architecture(GLM53_TEXT_CONFIG))


if __name__ == '__main__':
    unittest.main()
