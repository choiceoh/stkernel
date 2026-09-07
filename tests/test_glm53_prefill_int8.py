"""Default-off/mode isolation and the independent INT8 byte recipe."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'probes'))
import glm53_prefill_int8_check as m


class Int8Tests(unittest.TestCase):
    def test_invalid_or_incompatible_settings_fail_before_collectives(self):
        source=ROOT/'overlay/modules/glm53_runtime/glm53_prefill_collectives.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_parse_rs_int8')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        parse=ns['_parse_rs_int8']
        for enabled in (False,True):
            for fp8 in (False,True):
                self.assertFalse(parse('0',enabled=enabled,fp8_v3=fp8))
                if enabled and fp8:self.assertTrue(parse('1',enabled=enabled,fp8_v3=fp8))
                else:
                    with self.assertRaises(ValueError):parse('1',enabled=enabled,fp8_v3=fp8)
        for value in ('', '2', 'true', '-1'):
            with self.assertRaises(ValueError):parse(value,enabled=True,fp8_v3=True)
        self.assertIn('VLLM_GLM53_PREFILL_SP_RS_INT8=0',(ROOT/'profiles/glm53.env').read_text())

    def test_int8_dispatch_is_explicit_and_returns_the_given_output(self):
        source=ROOT/'overlay/modules/glm53_runtime/glm53_prefill_collectives.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_reduce_scatter_v3')
        output=object();call=Mock(return_value=output)
        ns=dict(_RS_INT8=True,_reduce_scatter_int8=call)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        tensor=object();self.assertIs(ns['_reduce_scatter_v3'](tensor,output,8192),output)
        call.assert_called_once_with(tensor,output,8192)

    def test_int8_proof_cannot_be_satisfied_by_generic_packed_or_arming_lines(self):
        sys.path.insert(0,str(ROOT/'bench'));import proof
        knob='VLLM_GLM53_PREFILL_SP_RS_INT8';table=proof.markers(str(ROOT/'bench/proof-markers.tsv'))
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'log'
            log.write_text('[prefill-sp] packed reduce-scatter engaged (one values+scales all-to-all)\n')
            self.assertFalse(proof.check([knob],str(log),table)['proof'][knob])
            log.write_text(table[knob][0]+'\n')
            self.assertTrue(proof.check([knob],str(log),table)['proof'][knob])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'run with actual torch in pinned CPU image')
    def test_reference_preserves_signed_ties_padding_and_packet_alignment(self):
        import torch
        value=torch.zeros((129,4096),dtype=torch.bfloat16)
        value[0,:8]=torch.tensor([-127.,-126.5,-1.5,-.5,.5,1.5,126.5,127.])
        value[0,2048:2052]=torch.tensor([-128.,-3.,3.,128.])
        packet,decoded,g=m.reference_packet(torch,value)
        self.assertEqual(g['padded_rows'],132)
        self.assertEqual(decoded[0,:8].tolist(),[-127.,-126.,-2.,0.,0.,2.,126.,127.])
        self.assertEqual(decoded[0,2048:2052].tolist(),[-128.,-4.,4.,128.])
        self.assertTrue(torch.equal(decoded[129:],torch.zeros_like(decoded[129:])))
        self.assertEqual(g['payload_bytes']%128,0)
        self.assertEqual(int(packet[0]),129)  # two's-complement -127
        self.assertEqual(int(packet[7]),127)
        for destination in range(4):
            tail=packet.reshape(4,-1)[destination,g['local_n']+4*(g['local_n']//2048):]
            self.assertTrue(torch.equal(tail,torch.zeros_like(tail)))
        self.assertTrue(torch.isfinite(decoded).all())


if __name__=='__main__':unittest.main()
