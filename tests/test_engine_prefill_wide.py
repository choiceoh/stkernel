"""Exercise actual native MLA dispatch without allocating CUDA tensors."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class Fallback(Exception):
    pass


class Tensor:
    def __init__(self, shape, dtype, contiguous=True, size=1):
        self.shape, self.dtype, self.contiguous, self.size = shape, dtype, contiguous, size

    def is_contiguous(self):
        return self.contiguous

    def element_size(self):
        return self.size


class WidePrefillTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'engine/kernels/mla/__init__.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mla_decode')
        self.torch = types.SimpleNamespace(bfloat16='bf16', int32='i32',
                    cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False))
        def fallback(*a):
            raise Fallback()
        self.ns = dict(MLA_H=16, MLA_D=512, ENABLE_MLA_PREFILL32=True,
                       _mla_prefill32=lambda *a: 'prefill', mla_splits=fallback,
                       logger=types.SimpleNamespace(warning=lambda *a: None))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), self.ns)

    def route(self, tokens, width=2176, dtype='bf16', cache_size=1):
        with patch.dict(sys.modules, torch=self.torch):
            try:
                return self.ns['mla_decode'](Tensor((tokens, 16, 512), dtype),
                    Tensor((134400, 512), 'u8', size=cache_size),
                    Tensor((tokens, width), 'i32'), Tensor((tokens,), 'i32'), .04, 1.)
            except Fallback:
                return 'fallback'

    def test_large_grid_uses_prefill_kernel(self):
        for tokens in (4096, 16128, 16384, 16385, 29952, 32256, 32768):
            for width in (1, 33, 2048, 2176):
                self.assertEqual(self.route(tokens, width), 'prefill')

    def test_decode_capture_and_invalid_storage_keep_existing_route(self):
        for tokens in (1, 7, 28, 64, 4095, 32769):
            self.assertEqual(self.route(tokens), 'fallback')
        self.assertEqual(self.route(32256, 2177), 'fallback')
        self.assertEqual(self.route(32256, dtype='fp16'), 'fallback')
        self.assertEqual(self.route(32256, cache_size=2), 'fallback')
        self.torch.cuda.is_current_stream_capturing = lambda: True
        self.assertEqual(self.route(32256), 'fallback')


if __name__ == '__main__':
    unittest.main()
