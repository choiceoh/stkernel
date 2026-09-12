"""Full-source accounting must fail on omissions outside the explicit MTP policy."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from engine.profiles.glm53.modelopt_coverage import source_coverage


class ModelOptCoverageTests(unittest.TestCase):
    def fixture(self, directory, extra=None):
        root = Path(directory)
        keys = {'model.language_model.layers.0.weight': 4,
                'model.visual.weight': 6,
                'model.language_model.layers.1.weight': 8}
        keys.update(extra or {})
        header, offset = {}, 0
        for key, size in keys.items():
            header[key] = dict(dtype='U8', shape=[size], data_offsets=[offset, offset + size])
            offset += size
        raw = json.dumps(header).encode()
        (root / 'weights.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(offset))
        (root / 'config.json').write_text(json.dumps(dict(text_config=dict(num_hidden_layers=1, num_nextn_predict_layers=1))))
        (root / 'model.safetensors.index.json').write_text(json.dumps(dict(
            metadata=dict(total_size=offset), weight_map={k:'weights.safetensors' for k in keys})))
        return root, {'model.language_model.layers.0.weight'}, {'model.visual.weight'}

    def test_mtp_is_explicitly_excluded_without_copying_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root, text, vision = self.fixture(temp)
            result = source_coverage(root, text, vision)
            self.assertEqual(result['source_payload_bytes'], 18)
            self.assertEqual(result['text']['payload_bytes'], 4)
            self.assertEqual(result['vision']['payload_bytes'], 6)
            self.assertEqual(result['intentionally_excluded_mtp']['payload_bytes'], 8)
            self.assertFalse(result['intentionally_excluded_mtp']['written_to_preshards'])
            self.assertEqual(result['unaccounted_tensors'], 0)

    def test_omitted_real_layer_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root, text, vision = self.fixture(temp)
            with self.assertRaisesRegex(ValueError, 'unaccounted'):
                source_coverage(root, set(), vision)

    def test_unknown_auxiliary_tensor_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root, text, vision = self.fixture(temp, {'model.other.weight': 4})
            with self.assertRaisesRegex(ValueError, 'unaccounted'):
                source_coverage(root, text, vision)

    def test_missing_index_tensor_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root, text, vision = self.fixture(temp)
            p = root / 'model.safetensors.index.json'
            index = json.loads(p.read_text()); index['weight_map']['absent'] = 'weights.safetensors'
            p.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'absent header'):
                source_coverage(root, text | {'absent'}, vision)

    def test_truncated_payload_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root, text, vision = self.fixture(temp)
            p = root / 'weights.safetensors'; p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'outside shard'):
                source_coverage(root, text, vision)


if __name__ == '__main__':
    unittest.main()
