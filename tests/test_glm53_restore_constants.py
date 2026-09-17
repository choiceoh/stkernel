import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.restore_glm53_constants import copy_with_constants, read_header
from engine.profiles.glm53.weights import restored_constants_id


class RestoreConstantsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source, self.dest = (Path(self.temp.name) / name for name in ('old.safetensors', 'new.safetensors'))
        self.payload = b'expert-data!' + struct.pack('<ff', 1., 2.) + b'other-weights'
        self.header = {'__metadata__': {'weight_layout': 'original'},
                       'L3.moe.w13': {'dtype': 'U8', 'shape': [12], 'data_offsets': [0, 12]},
                       'L3.moe.bias': {'dtype': 'F32', 'shape': [2], 'data_offsets': [12, 20]},
                       'tail': {'dtype': 'U8', 'shape': [13], 'data_offsets': [20, 33]}}
        raw = json.dumps(self.header).encode()
        self.original = struct.pack('<Q', len(raw)) + raw + self.payload
        self.source.write_bytes(self.original)
        self.patch = {'L3.moe.bias': struct.pack('<ff', 1.001, 2.002)}

    def test_only_selected_payload_changes_and_source_is_unchanged(self):
        result = copy_with_constants(self.source, self.dest, self.patch, {'fp32_constants_sha256': 'receipt'})
        with self.dest.open('rb') as stream:
            header, base = read_header(stream)
            payload = stream.read()
        self.assertEqual(base % 8, 0)
        self.assertEqual(payload, self.payload[:12] + self.patch['L3.moe.bias'] + self.payload[20:])
        self.assertEqual(result['payload_sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(header['L3.moe.w13'], self.header['L3.moe.w13'])
        self.assertEqual(header['__metadata__']['weight_layout'], 'original')
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_existing_destination_is_never_overwritten(self):
        self.dest.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            copy_with_constants(self.source, self.dest, self.patch, {})
        self.assertEqual(self.dest.read_bytes(), b'existing')

    def test_expert_replacement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not a GLM control'):
            copy_with_constants(self.source, self.dest, {'L3.moe.w13': b'changed-data'}, {})
        self.assertFalse(self.dest.exists())

    def test_wrong_constant_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            copy_with_constants(self.source, self.dest, {'L3.moe.bias': b'bad'}, {})
        self.assertFalse(self.dest.exists())

    def test_restored_values_change_state_cache_identity(self):
        self.assertIsNone(restored_constants_id({}))
        first = {'fp32_constants_sha256': 'a' * 64, 'fp32_constants_kinds': 'router'}
        second = dict(first, fp32_constants_sha256='b' * 64)
        self.assertNotEqual(restored_constants_id(first), restored_constants_id(second))
        self.assertNotEqual(restored_constants_id(first), restored_constants_id(dict(first, fp32_constants_kinds='all')))

    def test_incomplete_identity_is_rejected(self):
        for metadata in ({'fp32_constants_sha256': 'a' * 64},
                         {'fp32_constants_kinds': 'router'},
                         {'fp32_constants_sha256': 'bad', 'fp32_constants_kinds': 'router'}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                restored_constants_id(metadata)
