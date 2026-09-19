import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from probes.qwen38_gptq_expand import guard_fragments, original_windows, overlaps, training_sources


class ExpansionTests(unittest.TestCase):
    def test_only_original_training_sessions_are_read(self):
        name = "client:main.jsonl"
        other = "client:main:00000000-0000-0000-0000-000000000001.jsonl"
        rows = [dict(id="a", split="train"), dict(id="b", split="test")]
        provenance = dict(rows=[dict(id="a", session=name, snapshot_bytes=20, source_sha256="a"),
                               dict(id="b", session=other, snapshot_bytes=30, source_sha256="b")])
        self.assertEqual(training_sources(rows, provenance), {name: (20, "a")})
        provenance['rows'][1]['session'] = name
        with self.assertRaisesRegex(ValueError, 'spans splits'):
            training_sources(rows, provenance)

    def test_original_prefix_is_frozen_and_credentials_and_future_records_are_excluded(self):
        messages = [dict(role="user", content="Original question " * 10 + "password=abcdefgh123456"),
                    dict(role="assistant", content=[dict(type="thinking", text="excluded"),
                                                   dict(type="text", text="The previous answer")]),
                    dict(role="user", content="A distinct next question " * 10)]
        raw = ''.join(json.dumps(m) + '\n' for m in messages).encode()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'client:main.jsonl'
            path.write_bytes(raw + b'{"role":"user","content":"future"}\n')
            windows = list(original_windows(path, len(raw), hashlib.sha256(raw).hexdigest()))
            self.assertEqual([i for i, _ in windows], [0, 2])
            text = json.dumps(windows)
            self.assertNotIn('abcdefgh123456', text)
            self.assertNotIn('excluded', text)
            self.assertNotIn('future', text)
            self.assertIn('[REDACTED]', text)
            path.write_bytes(b'x' + raw[1:])
            with self.assertRaisesRegex(ValueError, 'prefix changed'):
                list(original_windows(path, len(raw), hashlib.sha256(raw).hexdigest()))

    def test_substantial_copies_of_evaluation_text_are_rejected(self):
        text = ''.join(chr(0xAC00 + i) for i in range(256))
        rows = [dict(messages=[dict(role='user', content=text)])]
        guard = guard_fragments(rows)
        self.assertTrue(overlaps([dict(role='user', content='A different wrapper: ' + text)], guard))
        self.assertTrue(overlaps([dict(role='user', content=' '.join(text))], guard))
        self.assertFalse(overlaps([dict(role='user', content='unrelated input ' * 30)], guard))


if __name__ == '__main__':
    unittest.main()
