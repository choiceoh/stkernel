import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
D=Path(__file__).parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
base=load('existing_proof_contracts',ROOT/'tests/test_glm53_ep_tiled_proof.py')
proof=load('private_hybrid_proof',ROOT/'bench/proof.py')
base.proof=proof
KNOB='VLLM_GLM53_EP_HYBRID_TP2'
TAG='glm53_ep2tp2_tiled_e144_i1024_v1'
DUAL_KNOB='VLLM_GLM53_EP_HYBRID_Q0_DUAL_WARP'
DUAL_TAG='glm53_ep2tp2_q0_dual_warp_v1'

def receipt(*, q0_dual_warp=False):
    record=base.receipt()
    record.update(schema=2,geometry=dict(E=144,K=4096,I=1024,top8=8),
        loader_identity=['glm53_ep2_tp2_loader_v1',288,144,4096,2048,1024,
                         0,0,0,0,144,0,1024])
    record['rank_geometry']=dict(physical_rank=0,world_size=4,ep_size=2,ep_rank=0,
        tp_size=2,tp_rank=0,expert_start=0,expert_stop=144,
        intermediate_start=0,intermediate_stop=1024,attention_shared_tp_size=4,
        terminal_output_sum_size=4,collective_execution_verified=False)
    for key in ('packed_before','packed_after'):
        for name,blocks in (('fc1',256),('fc2',128)):
            record[key]['planes'][name]['shape']=[144,blocks,1552]
    for case in record['cases']:
        m=case['rows']
        if m<=32:
            low=m<=8
            key=('glm53_ep_static_tiled_fp32_v1',m,256,48,'torch.int32',False,True,
                 (16,128,256) if low else (32,64,512),
                 (16,256,128) if low else (32,128,128),
                 'nvfp4','sf6_v1','swigluoai_uninterleave',1.,0.,10.,
                 'bf16_scatter' if low else 'fp32_scatter')
            if low:key+=('glm53_ep_static_sf6_a_ring_v1','glm53_ep_static_sf6_word_unpack_v1',
                         'glm53_ep_static_bf16_scatter_v1')
            key+=('glm53_ep_static_fused_route_v1',288,'torch.int32',0,TAG,144,1024)
        else:
            key=('dynamic','fp4','nvfp4',144,4096,1024,8,48,(128,128),'torch.int32',False,
                 True,'swigluoai_uninterleave',1.,0.,10.,False,True,
                 'glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1',TAG)
            if q0_dual_warp:key+=(DUAL_TAG,)
        text=repr(key)
        if m>32:text=text.replace("'torch.int32'",'torch.int32')
        case['cache_evidence']=dict(keys=[text],decode_opt=False)
    if q0_dual_warp:record['q0_dual_warp']=True
    return record

def log(record=None):
    return '\n'.join(('[ep-hybrid-selftest] PASS '+json.dumps(record or receipt()),
        '[ep-hybrid] LAUNCHED decode E144/H4096/I1024/top8 T=6',
        '[ep-hybrid] LAUNCHED prefill E144/H4096/I1024/top8 T=8192',
        base.log().splitlines()[-1]))

def dual_log(record=None, *, rows=8192):
    return log(receipt(q0_dual_warp=True) if record is None else record)+'\n'+(
        '[ep-hybrid-q0-dual-warp] LAUNCHED prefill E144/H4096/I1024/top8 T='+str(rows))

class HybridProofTests(unittest.TestCase):
    def test_exact_hybrid_executes_both_proofs(self):
        for k in (KNOB,'VLLM_GLM53_EP_TILED'):
            self.assertTrue(proof._startup_proof(k,log()))
        self.assertFalse(proof._startup_proof('VLLM_GLM53_EP_DECODE_OPT',log()))
        self.assertFalse(proof._startup_proof(KNOB,base.log()))

    def test_every_rank_geometry_and_bad_identity_rejected(self):
        for p in range(4):
            ep,tp=divmod(p,2);record=receipt()
            record['loader_identity'][6:]=[p,ep,tp,ep*144,(ep+1)*144,tp*1024,(tp+1)*1024]
            record['rank_geometry'].update(physical_rank=p,ep_rank=ep,tp_rank=tp,
                expert_start=ep*144,expert_stop=(ep+1)*144,
                intermediate_start=tp*1024,intermediate_stop=(tp+1)*1024)
            self.assertTrue(proof._startup_proof(KNOB,log(record)))
            for index in range(13):
                broken=copy.deepcopy(record)
                v=broken['loader_identity'][index]
                broken['loader_identity'][index]=False if type(v)is int else v+'-wrong'
                self.assertFalse(proof._startup_proof(KNOB,log(broken)))

    def test_wrong_geometry_packed_owner_row_and_cache_never_pass(self):
        mutations=(lambda r:r.update(schema=True),lambda r:r.update(geometry=dict(E=72,K=4096,I=2048,top8=8)),
            lambda r:r.pop('loader_identity'),lambda r:r['packed_before']['planes']['fc1'].update(shape=[72,512,1552]),
            lambda r:r['cases'][0].update(rows=7),lambda r:r['cases'][0]['candidate'][0].update(bad_rows=1),
            lambda r:r['cases'][0]['cache_evidence'].update(decode_opt=True),
            lambda r:r['cases'][0]['cache_evidence'].update(keys=["('retired',)"]),
            lambda r:r['cases'][4]['cache_evidence'].update(keys=["torch.execute('unsafe')"]),
            lambda r:r['cases'][4]['cache_evidence'].update(keys=[]),
            lambda r:r['rank_geometry'].update(collective_execution_verified=True),
            lambda r:r['rank_geometry'].update(ep_rank=False))
        for mutate in mutations:
            record=receipt();mutate(record)
            self.assertFalse(proof._startup_proof(KNOB,log(record)))
        for c in range(12):
            record=receipt()
            record['cases'][c]['cache_evidence']['keys'][0]=record['cases'][c]['cache_evidence']['keys'][0].replace(TAG,'wrong')
            self.assertFalse(proof._startup_proof(KNOB,log(record)))

    def test_failure_mixed_boot_partial_execution_or_duplicate_cannot_pass(self):
        text=log()
        for wrong in (text+'\n[ep-hybrid-selftest] FAIL {}',text+'\n'+base.log(),
                      '\n'.join(text.splitlines()[:-1]),text.replace('T=6','T=33'),
                      text+'\n'+text.splitlines()[0],text.replace('"schema": 2','"schema": 2,"schema": 2')):
            self.assertFalse(proof._startup_proof(KNOB,wrong))

    def test_dual_warp_actual_prefill_preserves_native_and_base_proofs(self):
        for rows in (33,2128,8192,16384):
            text=dual_log(rows=rows)
            for knob in (KNOB,'VLLM_GLM53_EP_TILED',DUAL_KNOB):
                self.assertIs(proof._startup_proof(knob,text),True)
        # Absence remains the original artifact; an explicit false is equivalent.
        for record in (receipt(),dict(receipt(),q0_dual_warp=False)):
            self.assertIs(proof._startup_proof(KNOB,log(record)),True)
            self.assertIs(proof._startup_proof(DUAL_KNOB,dual_log(record)),False)
        original=receipt();selected=receipt(q0_dual_warp=True)
        self.assertEqual([c for c in original['cases'] if c['rows']<=32],
                         [c for c in selected['cases'] if c['rows']<=32])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'actual.log';path.write_text(dual_log())
            result=proof.check([KNOB,'VLLM_GLM53_EP_TILED',DUAL_KNOB],str(path),table={})
            self.assertEqual(result['proof_ok'],'3/3')
            self.assertIs(result['proof'][DUAL_KNOB],True)

    def test_dual_warp_selection_and_every_dynamic_key_must_agree(self):
        for value in (None,0,1,'true',[],{}):
            record=receipt(q0_dual_warp=True);record['q0_dual_warp']=value
            self.assertIs(proof._startup_proof(KNOB,dual_log(record)),False)
        for value in (False,'missing'):
            record=receipt(q0_dual_warp=True)
            if value=='missing':record.pop('q0_dual_warp')
            else:record['q0_dual_warp']=value
            self.assertIs(proof._startup_proof(KNOB,dual_log(record)),False)
        for idx,case in enumerate(receipt()['cases']):
            if case['rows']<=32:continue
            for text in (case['cache_evidence']['keys'][0],
                         receipt(q0_dual_warp=True)['cases'][idx]['cache_evidence']['keys'][0].replace(DUAL_TAG,'retired')):
                record=receipt(q0_dual_warp=True)
                record['cases'][idx]['cache_evidence']['keys']=[text]
                self.assertIs(proof._startup_proof(DUAL_KNOB,dual_log(record)),False)
        # A correctly suffixed key cannot launder an additional stale artifact.
        record=receipt(q0_dual_warp=True)
        record['cases'][4]['cache_evidence']['keys']+=receipt()['cases'][4]['cache_evidence']['keys']
        self.assertIs(proof._startup_proof(DUAL_KNOB,dual_log(record)),False)

    def test_dual_warp_armed_canary_wrong_or_failed_serving_never_proves(self):
        candidate=log(receipt(q0_dual_warp=True))
        for text in (candidate,candidate+'\n'+DUAL_KNOB+'=1',
                     candidate+'\n[ep-hybrid-q0-dual-warp] ARMED',
                     dual_log().replace('LAUNCHED prefill E144','LAUNCHED decode E144'),
                     dual_log().replace('I1024/top8 T=8192','I2048/top8 T=8192')):
            self.assertIs(proof._startup_proof(DUAL_KNOB,text),False)
        for rows in (0,32,16385,'6.0','8192 trailing','-1',''):
            self.assertIs(proof._startup_proof(DUAL_KNOB,dual_log(rows=rows)),False)
        for failure in ('[ep-hybrid-q0-dual-warp] FAIL {}','[ep-hybrid-selftest] FAIL {}'):
            for text in (failure+'\n'+dual_log(),dual_log()+'\n'+failure):
                self.assertIs(proof._startup_proof(DUAL_KNOB,text),False)

    def test_dual_warp_marker_never_replaces_full_numerics_and_owner_proof(self):
        mutations=(lambda r:r['cases'][4]['candidate'][0].update(bad_rows=1),
                   lambda r:r['cases'][4].update(graph_replay=False),
                   lambda r:r['packed_after']['planes']['fc1'].update(sha256='b'*64),
                   lambda r:r['rank_geometry'].update(tp_size=4),
                   lambda r:r.update(cleanup_error='failed'))
        for mutate in mutations:
            record=receipt(q0_dual_warp=True);mutate(record)
            self.assertIs(proof._startup_proof(DUAL_KNOB,dual_log(record)),False)
        text=dual_log()
        for partial in ('\n'.join(line for line in text.splitlines() if 'LAUNCHED decode' not in line),
                        '\n'.join(line for line in text.splitlines() if 'FINALIZED' not in line),
                        text+'\n'+text.splitlines()[0],text+'\n'+log()):
            self.assertIs(proof._startup_proof(DUAL_KNOB,partial),False)

if __name__=='__main__':
    unittest.main()
