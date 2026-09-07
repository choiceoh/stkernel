"""The diagnostic must preserve the standard request, text and error behavior."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
from input_reuse_channels import recorder
spec = importlib.util.spec_from_file_location('channels_test_onepass', ROOT/'bench/onepass.py')
onepass = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onepass)


class Response:
    def __init__(self, deltas):
        self.frames = [json.dumps({'choices': [{'delta': d}]}).encode() for d in deltas]
        self.frames.append(json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}],
                                      'usage': {'prompt_tokens': 20, 'completion_tokens': 2048}}).encode())
        self.closed = False
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def __iter__(self): return iter([b'data: '+s+b'\n' for s in self.frames]+[b'data: [DONE]\n'])


class RecordingTests(unittest.TestCase):
    def run_case(self, deltas):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'channels.jsonl'
            response = Response(deltas)
            timing = {'rep': 2}
            with patch.object(urllib.request, 'urlopen', return_value=response) as opening:
                result = recorder(onepass.ask_stream, output)(
                    'http://example.invalid/v1/chat/completions', 'fixture', '질문', 2048,
                    timing, min_tokens=2048, seed=9)
                self.assertIs(urllib.request.urlopen, opening)
                body = json.loads(opening.call_args.args[0].data)
                self.assertEqual((body['seed'], body['min_tokens'], body['max_tokens']), (9,2048,2048))
            self.assertTrue(response.closed)
            row = json.loads(output.read_text())
            self.assertEqual(row['timing'], timing)
            self.assertEqual(row['finish_reason'], 'length')
            return result, row

    def test_separates_reasoning_without_changing_the_standard_combined_text(self):
        result, row = self.run_case([{'reasoning_content': 'Halvorsen博士'}, {'content': '할보르센 박사'}])
        self.assertEqual(result[0], 'Halvorsen博士할보르센 박사')
        self.assertEqual(row['channels']['reasoning_content'], 'Halvorsen博士')
        self.assertEqual(row['channels']['content'], '할보르센 박사')

    def test_keeps_both_channels_when_one_delta_contains_both(self):
        result, row = self.run_case([{'content': '답', 'reasoning': '추론'}])
        self.assertEqual(result[0], '답')
        self.assertEqual(row['channels']['reasoning'], '추론')

    def test_network_failure_propagates_and_restores_the_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'channels.jsonl'
            with patch.object(urllib.request, 'urlopen', side_effect=OSError('offline')) as opening:
                with self.assertRaisesRegex(OSError, 'offline'):
                    recorder(onepass.ask_stream, output)('http://example.invalid', 'fixture', '질문', 20)
                self.assertIs(urllib.request.urlopen, opening)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
