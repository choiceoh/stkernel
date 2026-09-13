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
        self.w.shape = (288,4096)
        self.capturing = False
        def capture():
            self.calls.append('capture')
            return self.capturing
        def empty(shape,**kwargs):
            self.calls.append(('allocate',shape,kwargs))
            return 'fp32-output'
        calls = self.calls
        class Kernel:
            def __getitem__(self,grid):
                return lambda *args,**kwargs: calls.append(('launch',grid,args,kwargs))
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'router_logits')
        ns = dict(torch=SimpleNamespace(bfloat16='bf16',float32='fp32',empty=empty,
                                       cuda=SimpleNamespace(is_current_stream_capturing=capture)),
                  triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),_router_gemm=Kernel())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),ns)
        self.fn = ns['router_logits']

    def test_long_prefill_uses_original_operands_and_separate_fp32_output(self):
        self.assertEqual(self.fn(self.x,self.w),'fp32-output')
        self.assertEqual(self.calls[1],('allocate',(32256,288),dict(device='cuda',dtype='fp32')))
        launch = self.calls[-1]
        self.assertEqual(launch[1],(2520,))
        self.assertEqual(launch[2],(self.x,self.w,'fp32-output',32256))
        self.assertIs(launch[3]['enable_fp_fusion'],False)

    def test_short_decode_unsupported_and_capture_do_not_launch(self):
        for rows in (1,6,24,64,289,4095,8192,32769):
            self.x.shape=(rows,4096)
            self.assertIsNone(self.fn(self.x,self.w))
            self.assertEqual(self.calls,[])
        self.x.shape=(32256,4096)
        for target,key,value in ((self.x,'dtype','fp32'),(self.w,'dtype','fp32'),
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
