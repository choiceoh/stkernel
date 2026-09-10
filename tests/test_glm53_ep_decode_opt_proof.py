"""Decode optimization needs its kernel and fresh verified clone elimination."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
sys.path.insert(0, str(ROOT / 'tests'))
import glm53_launch_metadata as launch
import glm53_prep_proof as prep
import proof
from test_glm53_ep_tiled_proof import receipt, log
from test_onepass_speculation_proof import BOOT, command

KNOB = 'VLLM_GLM53_EP_DECODE_OPT'
TAG = 'glm53_ep_static_sf6_fc1_register_v2'


def optimized_receipt():
    record = receipt()
    for case in record['cases']:
        rows = case['rows']
        if rows > 32:
            continue
        key = ('glm53_ep_static_tiled_fp32_v1', rows, 256, 48, 'torch.int32', False, True,
            (16,128,256) if rows <= 8 else (32,64,512),
            (16,256,128) if rows <= 8 else (32,128,128),
            'nvfp4', 'sf6_v1', 'swigluoai_uninterleave', 1., 0., 10.,
            'bf16_scatter' if rows <= 8 else 'fp32_scatter')
        if rows <= 8:
            key += ('glm53_ep_static_sf6_a_ring_v1', 'glm53_ep_static_sf6_word_unpack_v1',
                    'glm53_ep_static_bf16_scatter_v1')
        key += ('glm53_ep_static_fused_route_v1', 288, 'torch.int32', 0)
        if rows <= 8:
            key += (TAG,)
        case['cache_evidence'] = dict(keys=[repr(key)], decode_opt=rows <= 8)
    return record


class DecodeOptimizationProofTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'head.log'

    def context(self):
        self.path.write_text('identified boot\n')
        actual = dict(launch.launch_speculation(command(5)), boot_id=BOOT,
            image='sha256:'+'b'*64, environment_spec_k='5', preparation_mode='1',
            preparation_kernel='cuda', shadow_every='1', selfcheck_every='64', ep_decode_opt='1')
        return dict(expected_mode='1', boot_id=BOOT, launch_before=actual,
            launch_after=copy.deepcopy(actual), exclusive=True, log_prefix=prep.log_prefix(self.path))

    def optimized_checkpoint(self):
        return dict(version=1, num_reqs=1, q=6, live_snapshot_clones_elided=True,
            first_plan_check_passed=True, fused_steps=64, live_diff_checks=2)

    def append(self, checkpoint=None, *, native=True):
        with self.path.open('a') as stream:
            if native:
                stream.write(log(optimized_receipt())+'\n')
            stream.write('[prep-fused] plan built: mode=on kernel=cuda groups=3 q=6 '
                         'shadow_every=1 selfcheck_every=64\n')
            stream.write('[prep-fused] on: fused_steps=64 stock_steps=8 checks ok=2 drift=0 '
                         'first_plan_check=False\n')
            stream.write('[prep-decode-opt] USED '+json.dumps(
                self.optimized_checkpoint() if checkpoint is None else checkpoint)+'\n')

    def test_native_requires_every_expected_shape_and_exact_opt_cache_namespace(self):
        record = optimized_receipt()
        self.assertTrue(proof._startup_proof(KNOB, log(record)))
        for mutate in (
            lambda r:r['cases'][0]['cache_evidence'].update(decode_opt=False),
            lambda r:r['cases'][1]['cache_evidence'].update(decode_opt=True),
            lambda r:r['cases'][0]['cache_evidence'].update(keys=['()']),
            lambda r:r['cases'][0]['cache_evidence'].update(keys=["('"+TAG+"',)"]),
            lambda r:r['cases'][0].update(rows=True),
            lambda r:r['cases'][0].pop('cache_evidence'),
        ):
            changed=copy.deepcopy(record); mutate(changed)
            self.assertFalse(proof._startup_proof(KNOB, log(changed)))
        self.assertFalse(proof._startup_proof(KNOB, log()))

    def test_full_candidate_needs_both_kernel_and_workload_preparation(self):
        context=self.context(); self.append()
        result=proof.check([KNOB],str(self.path),preparation=context)
        self.assertEqual(result['proof_ok'],'1/1')
        self.assertEqual(result['decode_optimization']['decode_opt_checkpoint_count'],1)
        self.assertEqual(result['decode_optimization']['decode_opt_last_checkpoint']['live_diff_checks'],2)
        context=self.context(); self.append(native=False)
        self.assertEqual(proof.check([KNOB],str(self.path),preparation=context)['proof_ok'],'0/1')
        context=self.context(); self.path.write_text(log(optimized_receipt()))
        self.assertEqual(proof.check([KNOB],str(self.path),preparation=context)['proof_ok'],'0/1')

    def test_old_or_armed_only_reuse_cannot_supply_fresh_workload_proof(self):
        context=self.context(); self.append()
        context['log_prefix']=prep.log_prefix(self.path)
        with self.path.open('a') as stream:
            stream.write('[prep-fused] on: fused_steps=128 stock_steps=8 checks ok=3 drift=0\n')
        self.assertEqual(prep.evidence(context,self.path,decode_opt=True)['verdict'],'REJECTED')
        context=self.context(); self.append()
        for state in ('launch_before','launch_after'):
            context[state]['ep_decode_opt']='0'
        self.assertEqual(prep.evidence(context,self.path,decode_opt=True)['verdict'],'REJECTED')

    def test_unverified_false_or_malformed_reuse_counters_fail_closed(self):
        for field,value in (('first_plan_check_passed',False), ('live_snapshot_clones_elided',1),
                            ('fused_steps',65), ('live_diff_checks',0),
                            ('live_diff_checks',3), ('q',4), ('version',True)):
            context=self.context(); checkpoint=self.optimized_checkpoint(); checkpoint[field]=value
            self.append(checkpoint)
            self.assertEqual(prep.evidence(context,self.path,decode_opt=True)['verdict'],'REJECTED')
        context=self.context(); self.append()
        with self.path.open('a') as stream:
            stream.write('[prep-decode-opt] USED {"version":1,"version":1}\n')
        self.assertEqual(prep.evidence(context,self.path,decode_opt=True)['verdict'],'REJECTED')


if __name__ == '__main__':
    unittest.main()
