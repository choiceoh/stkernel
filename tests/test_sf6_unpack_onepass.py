"""Frozen SF6 unpack experiment inputs and parent compile-cache contract."""
import ast
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bench'))
from fleet_prepare import command_environment
from fleet_onepass import validate

KNOB = 'VLLM_GLM53_SF6_UNPACK_U8X4'


class UnpackOnepass(unittest.TestCase):
    def test_exact_two_arm_canonical_workload_and_private_endpoint(self):
        argv = json.loads((ROOT/'probes/sf6_unpack_onepass_v1.json').read_text())
        command, env = command_environment(argv, {})
        self.assertEqual(command, ['bash', 'bench/chain.sh',
            'sf6-unpack-0909v1A='+KNOB+'=1', 'sf6-unpack-0909v1B='])
        validate(argv, ROOT, ROOT)
        expected = {KNOB:'0', 'VLLM_GLM53_B12X_STATIC_V2':'t,r,sf6',
            'GLM53_API_HOST':'127.0.0.1','GLM53_API_PORT':'18000',
            'HEAD':'127.0.0.1','HEAD_URL':'http://127.0.0.1:18000',
            'VLLM_GLM53_AR_COMPACT_CTA':'0','VLLM_GLM53_AR_PROXY_INLINE':'0',
            'VLLM_GLM53_AR_CONSUMER_PDL':'1','VLLM_GLM53_MK_PDL':'1',
            'SPEC_K':'5','KV_TOKENS':'1100000','KV_HYBRID_BLOCKS':'187',
            'ONEPASS_FIXED_DECODE_TOKENS':'2048','ONEPASS_FIXED_DECODE_REPS':'3',
            'ONEPASS_REQUIRE_EXCLUSIVE':'1','ONEPASS_COMBINE_MIN_CTX':'32000',
            'PREFILL_WARMUP':'0','SKIP_BOOT':'0','QUALITY_CTX':'2000,32000,128000'}
        for key, value in expected.items():
            self.assertEqual(env[key],value,key)
        workload=json.loads(env['FLEET_WORKLOAD'])
        self.assertEqual(workload,dict(ctx=[2000,32000,128000],seed=7,max_tokens=400,
            combine_min_ctx=32000,fixed_decode_tokens=2048,fixed_decode_reps=3,require_exclusive=True))
        self.assertIn('\n'+KNOB+'=0\n',(ROOT/'profiles/glm53.env').read_text())

    def test_torch_compile_factor_distinguishes_both_modes(self):
        source=ROOT/'overlay/modules/glm53_model/glm53_fp8_dense.py'
        tree=ast.parse(source.read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.Expr)
            and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
            and n.value.func.id=='_register_compile_factor' and n.value.args
            and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value==KNOB]
        self.assertEqual(len(nodes),1)
        registered={}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),
             dict(os=os,_register_compile_factor=lambda name,getter:registered.update({name:getter})))
        getter=registered[KNOB]
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(getter(),'1')
            os.environ[KNOB]='0'
            self.assertEqual(getter(),'0')
            os.environ[KNOB]='1'
            self.assertEqual(getter(),'1')


if __name__=='__main__':
    unittest.main()
