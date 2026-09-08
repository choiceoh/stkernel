import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tools.trace_prefill_attribution import analyze, category, occupancy
from probes.glm53_prefill_attribution import first_piece


class PrefillAttributionTests(unittest.TestCase):
    def test_overlap_is_not_double_counted(self):
        result = occupancy([
            dict(ts=0, dur=10, category='communication'),
            dict(ts=5, dur=10, category='dense'),
            dict(ts=7, dur=2, category='dense'),
        ], 0, 20)
        self.assertEqual(result['busy_ms'], .015)
        self.assertEqual(result['idle_ms'], .005)
        self.assertEqual(result['communication_compute_overlap_ms'], .005)
        self.assertEqual(result['categories']['dense']['occupied_ms'], .010)
        self.assertEqual(result['categories']['dense']['exclusive_ms'], .005)

    def test_prefill_ranges_exclude_decode_and_deduplicate_stream_annotations(self):
        trace = {'traceEvents': [
            dict(cat='cpu_op', name='glm53_fp8_dense::gemm_nvfp4', args={'External id':7}),
            dict(cat='gpu_user_annotation', name='execute_context_1(32)_generation_0(0)',
                 ts=0, dur=10, tid=1, args={'External id':1}),
            dict(cat='gpu_user_annotation', name='execute_context_1(32)_generation_0(0)',
                 ts=1, dur=8, tid=2, args={'External id':1}),
            dict(cat='gpu_user_annotation', name='execute_context_0(0)_generation_1(6)',
                 ts=20, dur=10, tid=1, args={'External id':2}),
            dict(cat='kernel', ph='X', name='_partials', ts=1, dur=3, tid=1,
                 args={'External id':7}),
            dict(cat='kernel', ph='X', name='decode_kernel', ts=21, dur=3, tid=1, args={}),
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.gz'
            with gzip.open(path,'wt') as f:
                json.dump(trace,f)
            result=analyze(path)
        self.assertEqual(result['prefill_tokens'],32)
        self.assertEqual(result['prefill_chunks'],1)
        self.assertEqual(result['selected_events'],1)
        self.assertEqual(result['excluded_events'],1)
        self.assertEqual(result['kernels'][0]['category'],'dense_gemm_and_quant')

    def test_refuses_unlabeled_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.gz'
            with gzip.open(path,'wt') as f:
                json.dump({'traceEvents':[]},f)
            with self.assertRaisesRegex(ValueError,'no explicit pure-prefill'):
                analyze(path)

    def test_codec_and_prenorm_are_not_generic_gemm(self):
        self.assertEqual(category('_pack_rs_payload'),'transport_codec')
        self.assertEqual(category('_unpack_sum_payload'),'transport_codec')
        self.assertEqual(category('void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl'),'mhc')

    def test_narrow_target_excludes_bf16_chunks(self):
        ev=[]
        for ext,rows,start in [(1,4096,0),(2,2048,20)]:
            ev.append(dict(cat='gpu_user_annotation',
                           name=f'execute_context_1({rows})_generation_0(0)',
                           ts=start,dur=10,tid=1,args={'External id':ext}))
            ev.append(dict(cat='kernel',ph='X',name='mhc_post_tilelang_kernel',
                           ts=start+1,dur=3,tid=1,args={}))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.gz'
            with gzip.open(path,'wt') as f:
                json.dump({'traceEvents':ev},f)
            result=analyze(path)
        self.assertEqual(result['rs_unpack_and_post_fp8_chunks']['occupied_ms'],.003)
        self.assertEqual(result['rs_unpack_and_post_fp8_chunks']['pct_span'],10)

    def test_capture_trigger_waits_for_actual_token(self):
        self.assertFalse(first_piece(b'data: {"choices":[{"delta":{"role":"assistant"}}]}'))
        self.assertFalse(first_piece(b'data: [DONE]'))
        for field in ['content','reasoning_content','reasoning']:
            self.assertTrue(first_piece(('data: '+json.dumps({'choices':[{'delta':{field:'text'}}]})).encode()))


if __name__ == '__main__':
    unittest.main()
