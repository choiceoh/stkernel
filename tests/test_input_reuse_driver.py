"""Behavior of serving receipts and failure restoration without a GPU."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.py'


def load_function(name,namespace):
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace[name]


class ReuseDriver(unittest.TestCase):
    def setUp(self):
        self.calls=[];self.logs=[]
        self.ext=types.SimpleNamespace(gemm_input_plan=lambda *a:self.calls.append(a) or [1,8,3,33792])
        self.ns={'_EXT':self.ext,'_ARMED':{'gemm':True},'_INPUT_CAPTURED':set(),
                 'logger':types.SimpleNamespace(warning=lambda *a:self.logs.append(a))}
        self.note=load_function('_note_input_capture',self.ns)

    def test_ineligible_calls_do_not_plan_or_emit(self):
        for dims in ((6,0,1,False,False),(8,6528,4096,False,False),
                     (6,6528,4096,True,False),(6,6528,4096,False,True),
                     (6,6144,4096,False,False)):
            self.note(*dims)
        self.assertEqual(self.calls,[]);self.assertEqual(self.logs,[])

    def test_startup_gate_does_not_emit(self):
        self.ns['_ARMED']['gemm']=False
        self.note(6,6528,4096,False,False)
        self.assertEqual(self.calls,[])

    def test_only_a_real_capture_emits_once(self):
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
        with patch.dict(sys.modules,torch=torch):self.note(6,6528,4096,False,False)
        self.assertEqual(self.logs,[])
        torch.cuda.is_current_stream_capturing=lambda:True
        with patch.dict(sys.modules,torch=torch):
            self.note(6,6528,4096,False,False)
            self.note(6,6528,4096,False,False)
        self.assertEqual(len(self.logs),1)
        self.assertEqual(len(self.calls),2)

    def test_gate_exception_restores_arm_and_mode(self):
        class Injected(Exception):pass
        restored=[]
        self.ext.gemm_input_mode=lambda:1
        self.ext.set_gemm_input=lambda mode:restored.append(mode)
        def fail(*args,**kwargs):
            self.assertFalse(self.ns['_ARMED']['gemm'])
            raise Injected()
        torch=types.SimpleNamespace(randn=fail,bfloat16=object(),
            random=types.SimpleNamespace(fork_rng=lambda **kw:nullcontext()),
            cuda=types.SimpleNamespace(current_device=lambda:0))
        gate=load_function('_selftest_input_reuse',self.ns)
        with patch.dict(sys.modules,torch=torch):
            with self.assertRaises(Injected):gate()
        self.assertTrue(self.ns['_ARMED']['gemm'])
        self.assertEqual(restored,[1])


if __name__=='__main__':unittest.main()
