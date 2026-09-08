"""Independent CTA fallback and real-capture receipts without CUDA."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'overlay/modules/glm53_megakernel/glm53_megakernel.py'


def load(name, namespace):
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace[name]


class CtaDriver(unittest.TestCase):
    def gate_case(self, failure=None, cta=3):
        state={'cta':cta,'reuse':1};calls=[]
        ext=types.SimpleNamespace(
            gemm_input_cta_mode=lambda:state['cta'],
            set_input_cta=lambda mode:state.update(cta=mode),
            gemm_input_mode=lambda:state['reuse'],
            set_gemm_input=lambda mode:state.update(reuse=mode))
        ns={'_EXT':ext,'_ARMED':{'gemm':False},'logger':types.SimpleNamespace(warning=lambda *a:None),
            '_selftest_gemm':object(),'_selftest_input_reuse':object()}
        def gate(name, callback):
            calls.append((name,state.copy()))
            return name!=failure
        load('_arm_gemm',ns)(gate)
        return state,ns['_ARMED'],calls

    def test_cta_failure_preserves_enabled_default_input_reuse(self):
        state,armed,calls=self.gate_case('input_cta')
        self.assertEqual(state,{'cta':0,'reuse':1})
        self.assertTrue(armed['gemm'])
        self.assertEqual([s['cta'] for _,s in calls],[0,0,3])

    def test_default_failure_does_not_get_overridden_by_cta(self):
        state,armed,calls=self.gate_case('input_reuse')
        self.assertEqual(state['reuse'],0)
        self.assertTrue(armed['gemm'])
        self.assertEqual([n for n,_ in calls],['gemm','input_reuse'])
        _,armed,calls=self.gate_case('gemm')
        self.assertFalse(armed['gemm'])
        self.assertEqual([n for n,_ in calls],['gemm'])

    def test_success_preserves_requested_mode(self):
        state,armed,calls=self.gate_case()
        self.assertEqual(state,{'cta':3,'reuse':1})
        self.assertTrue(armed['gemm'])
        self.assertEqual([n for n,_ in calls],['gemm','input_reuse','input_cta'])

    def test_three_slice_failure_preserves_validated_cta2(self):
        state,armed,calls=self.gate_case('input_cta3',cta=4)
        self.assertEqual(state,{'cta':2,'reuse':1})
        self.assertTrue(armed['gemm'])
        self.assertEqual([s['cta'] for _,s in calls],[0,0,2,4])

    def test_three_slice_never_overrides_failed_existing_cta(self):
        state,armed,calls=self.gate_case('input_cta',cta=4)
        self.assertEqual(state,{'cta':0,'reuse':1})
        self.assertNotIn('input_cta3',[n for n,_ in calls])

    def test_three_slice_success_arms_requested_mode(self):
        state,armed,calls=self.gate_case(cta=4)
        self.assertEqual(state,{'cta':4,'reuse':1})
        self.assertEqual([n for n,_ in calls],['gemm','input_reuse','input_cta','input_cta3'])

    def test_receipt_requires_real_eligible_capture_and_enabled_plan(self):
        captured=set();logs=[];calls=[];mode=[2]
        def plan(*args):
            calls.append(args)
            return [mode[0],8,3,22528]
        ns={'_EXT':types.SimpleNamespace(gemm_input_cta_plan=plan),
            '_ARMED':{'gemm':True},'_INPUT_CTA_CAPTURED':captured,
            'logger':types.SimpleNamespace(warning=lambda *args:logs.append(args))}
        note=load('_note_input_cta_capture',ns)
        for dims in ((6,6416,4096,True,False),(6,6416,4096,False,True),
                     (8,6416,4096,False,False),(6,6528,4096,False,False)):
            note(*dims)
        self.assertEqual(calls,[])
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
        dims=(6,6416,4096,False,False)
        with patch.dict(sys.modules,torch=torch):
            note(*dims)
            torch.cuda.is_current_stream_capturing=lambda:True
            ns['_ARMED']['gemm']=False
            note(*dims)
            ns['_ARMED']['gemm']=True
            mode[0]=0;note(*dims)
            self.assertEqual(logs,[])
            mode[0]=2;note(*dims);note(*dims)
        self.assertEqual(len(logs),1)
        self.assertEqual(logs[0][1:],(6,6416,4096,2,8,3,22528))


if __name__=='__main__':unittest.main()
