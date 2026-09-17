"""Head producer dispatch, observation, compact row and MX input contracts."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch


class ProducerContracts(unittest.TestCase):
    def test_prepared_head_preserves_observer_logical_width_and_out(self):
        from engine.kernels.dense import FP8Linear
        from engine.profiles.glm53.net import Glm53Net
        h=torch.ones(7,128,dtype=torch.bfloat16)
        q=torch.zeros(7,128,dtype=torch.float8_e4m3fn)
        s=torch.zeros(512,dtype=torch.uint8)
        out=torch.empty(7,128,dtype=torch.bfloat16)
        observed=[];calls=[]
        def project(qq,ss,*,out):
            calls.append((qq,ss));return out.fill_(3)
        head=FP8Linear.__new__(FP8Linear)
        head.cublas=NS(project_mx=project);head.rows=127;head.cols=128
        head.observer=lambda x,rows:observed.append((x,rows));head.executed=False
        net=Glm53Net.__new__(Glm53Net);net.dense={'head':head}
        result=net.head_local(h,producer_pack=(q,s),out=out)
        self.assertEqual(result.shape,(7,127));self.assertEqual(result.data_ptr(),out.data_ptr())
        self.assertIs(observed[0][0],h);self.assertIsNone(observed[0][1])
        self.assertIs(calls[0][0],q);self.assertTrue(head.executed)
        with self.assertRaisesRegex(ValueError,'hidden rows'):
            head.project_mx(h[:1],q,s)
        head.cublas=None
        with self.assertRaisesRegex(RuntimeError,'prepared'):
            head.project_mx(h,q,s)

    def test_draft_finish_uses_compact_producer_only_for_requested_head_input(self):
        from engine.profiles.glm53.drafter import Drafter, _bind_common_lanes
        _bind_common_lanes()
        d=Drafter.__new__(Drafter);d.F=NS(rms_eps=1e-6)
        d.p={'norm.weight':torch.ones(128,dtype=torch.bfloat16)}
        d.target=NS(dense={})
        a=torch.ones(8,128,dtype=torch.bfloat16);b=a.clone()
        self.assertFalse(d._packed_head())
        self.assertEqual(d._finish_head_input(a,b,8,False).shape,a.shape)
        d.target.dense={'head':NS(cublas=object())};self.assertTrue(d._packed_head())
        with patch('engine.kernels.dense.cublaslt_producer.add_norm_head',return_value=('hidden','pack')) as producer:
            self.assertEqual(d._finish_head_input(a,b,8,True),('hidden','pack'))
            producer.assert_called_once_with(a,b,d.p['norm.weight'],1e-6,8)

    def test_default_proof_requires_the_drafter_head_producer_to_execute(self):
        from engine.profiles.glm53.cublas import execution_report
        reader=NS(split_decode=False,executed={'direct'},report=lambda: {})
        net=NS(cublas_readers={'head':NS(cublas=reader)},cublas_head_producer_required=True)
        with self.assertRaisesRegex(RuntimeError,'producer_mx'):
            execution_report(net)
        reader.executed.add('producer_mx')
        self.assertEqual(execution_report(net),{'head':{}})

    def test_native_mx_rejects_invalid_scale_abi_before_dispatch(self):
        from engine.kernels.dense.cublaslt_serving import Reader
        reader=Reader.__new__(Reader);reader.k=128
        q=torch.zeros(2,128,dtype=torch.float8_e4m3fn)
        reader.weight=(q,None)
        for s in (torch.zeros(2,1),torch.zeros(4,dtype=torch.uint8),torch.zeros(512,dtype=torch.int32)):
            with self.assertRaisesRegex(ValueError,'native MX'):
                reader.project_mx(q,s)


if __name__=='__main__':unittest.main()
