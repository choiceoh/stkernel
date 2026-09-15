"""Native router admission and FP32 output ownership, without GPU imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'engine/kernels/prefill_router.py'


class PrefillRouterTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        class Tensor:
            ndim,shape,dtype,is_cuda,device = 2,(32256,4096),'bf16',True,'cuda'
            contiguous = True
            def is_contiguous(self): return self.contiguous
        self.Tensor = Tensor
        self.x,self.w = Tensor(),Tensor()
        self.w.shape, self.w.dtype = (288,4096), 'fp32'
        self.capturing = False
        def capture():
            self.calls.append('capture')
            return self.capturing
        def project(x, w):
            self.calls.append(('project', x, w))
            return 'fp32-output'
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'router_logits')
        ns = dict(torch=SimpleNamespace(bfloat16='bf16',float32='fp32',
                                       cuda=SimpleNamespace(is_current_stream_capturing=capture)),
                  project=project)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
        self.fn = ns['router_logits']

    def test_long_prefill_uses_the_shared_fp32_operator(self):
        self.assertEqual(self.fn(self.x,self.w),'fp32-output')
        self.assertEqual(self.calls, ['capture', ('project', self.x, self.w)])

    def test_short_decode_unsupported_and_capture_do_not_launch(self):
        for rows in (1,6,24,64,289,4095,8192,32769):
            self.x.shape=(rows,4096)
            self.assertIsNone(self.fn(self.x,self.w))
            self.assertEqual(self.calls,[])
        self.x.shape=(32256,4096)
        for target,key,value in ((self.x,'dtype','fp32'),(self.w,'dtype','bf16'),
                                 (self.x,'is_cuda',False),(self.w,'device','cuda:1'),
                                 (self.w,'shape',(288,2048)),(self.x,'contiguous',False)):
            old=getattr(target,key); setattr(target,key,value)
            self.assertIsNone(self.fn(self.x,self.w))
            self.assertEqual(self.calls,[])
            setattr(target,key,old)
        self.capturing=True
        self.assertIsNone(self.fn(self.x,self.w))
        self.assertEqual(self.calls,['capture'])


if __name__ == '__main__':
    unittest.main()
