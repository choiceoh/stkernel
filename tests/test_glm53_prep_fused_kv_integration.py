"""Pinned source integration and actual host control flow; no serving imports/GPU."""
import ast
import copy
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'overlay/modules/glm53_runtime/glm53_prep_fused.py'
WORKER = ROOT / 'overlay/modules/glm53_runtime/kv_zero_worker_utils.py'
FIXTURE = ROOT / 'tests/fixtures/glm53_prep_fused_runtime'
REL = 'v1/worker/utils.py'
BASE = '3dcd6ad34ee1d1db2875f7f7dd51d90ee0e64041ab282180687770a38b26acb1'
MOUNTED = 'fd27b906f3363202a83ff79b451c69af305d806a60cda6905dc2ecdfa49bb49c'
RUNNER = 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415'


def fixture(name, expected):
    stored = (FIXTURE / name).read_bytes()
    metadata = json.loads((FIXTURE / 'identity.json').read_text())['files'][name]
    raw = gzip.decompress(stored)
    if (hashlib.sha256(stored).hexdigest() != metadata['stored_sha256']
            or hashlib.sha256(raw).hexdigest() != expected
            or metadata['original_sha256'] != expected or len(raw) != metadata['original_bytes']):
        raise AssertionError('source fixture identity changed')
    return raw


def load_defs(names, **extra):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names
             or isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', None) in names]
    ns = dict(__name__=__name__, os=os, hashlib=hashlib, logger=Mock(),
              dataclass=dataclass, field=field, **extra)
    module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], *nodes], type_ignores=[])
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns


def methods(raw, cls):
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {n.name:n for n in node.body if isinstance(n, ast.FunctionDef)}


class PrepFusedKvIntegrationTests(unittest.TestCase):
    def test_installed_source_pin_and_original_image_contract_are_distinct(self):
        pins = load_defs({'PREIMAGES'})['PREIMAGES']
        self.assertEqual(hashlib.sha256(WORKER.read_bytes()).hexdigest(), MOUNTED)
        self.assertEqual(pins[REL], MOUNTED)
        self.assertNotEqual(pins[REL], BASE)
        self.assertEqual(pins['v1/worker/gpu/model_runner.py'], RUNNER)
        rows = [line.split('\t') for line in (WORKER.parent/'manifest.tsv').read_text().splitlines()
                if line and not line.startswith('#')]
        self.assertEqual([r for r in rows if r[0] == WORKER.name], [[WORKER.name, 'vllm/'+REL, BASE]])
        fixture('model_runner.py.gz', RUNNER)
        fixture('worker_utils.image.py.gz', BASE)
        receipt = json.loads((FIXTURE/'identity.json').read_text())['runner_read']
        self.assertTrue(receipt['identity_equal'])
        self.assertEqual(receipt['sha256'], RUNNER)
        self.assertEqual(receipt['image'], 'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211')

    def test_reviewed_delta_is_confined_to_zero_kernel_and_its_owner(self):
        before = ast.parse(fixture('worker_utils.image.py.gz', BASE))
        after = ast.parse(WORKER.read_bytes())
        self.assertEqual(len(before.body), len(after.body))
        changed = []
        for a,b in zip(before.body, after.body):
            if ast.dump(a) == ast.dump(b):
                continue
            self.assertEqual(getattr(a,'name',None), getattr(b,'name',None))
            if isinstance(a, ast.ClassDef):
                self.assertEqual(a.name, 'KVBlockZeroer')
                self.assertEqual(len(a.body), len(b.body))
                for x,y in zip(a.body,b.body):
                    if ast.dump(x) != ast.dump(y):
                        self.assertEqual(x.name,y.name)
                        changed.append(a.name+'.'+x.name)
            else:
                changed.append(a.name)
        self.assertEqual(changed, ['_zero_kv_blocks_kernel','KVBlockZeroer.__init__','KVBlockZeroer.zero_block_ids'])

    def _installer(self, root):
        ns = load_defs({'check_preimages','install_glm53_prep_fused'}, PREIMAGES={REL:MOUNTED},
                       _INSTALLED=False, prep_fused_mode=lambda:'on', _ORIG={})
        untouched = object()
        runner = type('Runner', (), dict.fromkeys(('prepare_inputs','prepare_attn','capture_model',
                      'post_kv_cache_wake_up','execute_model','update_requests'), untouched))
        state = type('State', (), {'prepare_attn':untouched})
        for name in ('_patched_prepare_inputs','_patched_prepare_attn','_patched_capture_model',
                     '_patched_post_kv_cache_wake_up','_patched_ms_prepare_attn'):
            ns[name] = object()
        ns['_memo_slot_mappings_by_layer'] = lambda:object()
        vllm = ModuleType('vllm');vllm.__file__ = str(root/'__init__.py')
        mr = ModuleType('vllm.v1.worker.gpu.model_runner')
        mr.GPUModelRunner = runner;mr.build_slot_mappings_by_layer = untouched
        mh = ModuleType('vllm.v1.worker.gpu.model_states.mamba_hybrid');mh.MambaHybridModelState = state
        modules = {'vllm':vllm,mr.__name__:mr,mh.__name__:mh}
        return ns,runner,state,untouched,modules

    def test_drift_missing_and_original_image_bytes_disarm_before_any_patch(self):
        variants = (None, WORKER.read_bytes()[:-1]+b' ', fixture('worker_utils.image.py.gz', BASE))
        for data in variants:
            with self.subTest(data_sha=None if data is None else hashlib.sha256(data).hexdigest()), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);target=root/REL;target.parent.mkdir(parents=True)
                if data is not None:target.write_bytes(data)
                ns,runner,state,original,modules = self._installer(root)
                with patch.dict(sys.modules,modules):
                    self.assertFalse(ns['install_glm53_prep_fused']())
                self.assertIs(runner.prepare_inputs,original)
                self.assertIs(runner.prepare_attn,original)
                self.assertIs(state.prepare_attn,original)
                self.assertFalse(ns['_INSTALLED']);self.assertEqual(ns['_ORIG'],{})
                self.assertIn('DISARM',str(ns['logger'].warning.call_args))

    def test_exact_installed_source_arms_without_replacing_zeroing_or_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/REL;target.parent.mkdir(parents=True);target.write_bytes(WORKER.read_bytes())
            ns,runner,state,original,modules = self._installer(root)
            with patch.dict(sys.modules,modules):
                self.assertTrue(ns['install_glm53_prep_fused']())
                self.assertTrue(ns['install_glm53_prep_fused']())
            self.assertIs(runner.prepare_inputs,ns['_patched_prepare_inputs'])
            self.assertIs(state.prepare_attn,ns['_patched_ms_prepare_attn'])
            self.assertIs(runner.update_requests,original)
            self.assertIs(runner.execute_model,original)
            self.assertEqual(set(ns['_ORIG']), {'prepare_inputs','prepare_attn','capture_model',
                             'post_kv_cache_wake_up','ms_prepare_attn','build_slot_mappings_by_layer'})

    def test_actual_runner_zeroes_before_copy_publish_and_fused_preparation(self):
        tree = methods(fixture('model_runner.py.gz', RUNNER), 'GPUModelRunner')
        update = copy.deepcopy(tree['update_requests'])
        execute = copy.deepcopy(tree['execute_model']);execute.decorator_list=[]
        # Execute the authentic dispatch prefix, stopping after input/attention
        # preparation so no model forward is simulated or imported.
        for index,node in enumerate(execute.body):
            if isinstance(node,ast.If) and any(isinstance(n,ast.Call) and
                    getattr(n.func,'attr',None)=='prepare_inputs' for n in ast.walk(node)):
                end=next(i for i,n in enumerate(node.body) if any(isinstance(c,ast.Call) and
                         getattr(c.func,'attr',None)=='prepare_attn' for c in ast.walk(n)))
                node.body=node.body[:end+1];node.orelse=[];execute.body=execute.body[:index+1];break
        else:self.fail('authentic runner preparation boundary missing')
        events=[]
        ns=dict(np=SimpleNamespace(minimum=lambda *a,**k:None),
                copy_kv_cache_blocks_inplace=lambda *a:events.append('copy'),
                dispatch_cg_and_sync_dp=lambda *a,**k:(SimpleNamespace(num_tokens=6),None))
        exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],update,execute],type_ignores=[]),'pinned-runner-prefix','exec'),ns)
        runner=SimpleNamespace(req_states=SimpleNamespace(num_computed_tokens_np=[],prefill_len=SimpleNamespace(np=[]),num_computed_prefill_tokens=[]),
            block_tables=SimpleNamespace(apply_staged_writes=lambda:events.append('publish')),
            kv_block_zeroer=SimpleNamespace(zero_block_ids=lambda ids:events.append(('zero',tuple(ids)))),
            kv_caches=[],kv_cache_config=SimpleNamespace(num_blocks=8),lora_config=None,is_encoder_decoder=False,
            cudagraph_manager=None,dp_size=1,dp_rank=0,
            gather_batch_req_state=lambda *a:(SimpleNamespace(num_tokens=6),6),
            prepare_inputs=lambda *a:(events.append('fused_inputs') or object()),
            prepare_attn=lambda *a:(events.append('fused_attn') or (None,None)))
        for name in ('update_pp_decode_requests','finish_requests','free_states','add_requests'):
            setattr(runner,name,lambda *a:None)
        runner.update_requests=lambda out:ns['update_requests'](runner,out)
        for ids in ([0,7],[]):
            events.clear()
            output=SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=[],num_computed_tokens=[],new_block_ids=[]),
                new_block_ids_to_zero=ids,kv_cache_block_copies=[('cow',)],total_num_scheduled_tokens=6,num_scheduled_tokens={'r':6})
            ns['execute_model'](runner,output)
            expected=([('zero',(0,7))] if ids else [])+['copy','publish','fused_inputs','fused_attn']
            self.assertEqual(events,expected)

    def _runtime(self, mode):
        ns=load_defs({'_State','_ensure_plan','_patched_prepare_inputs',
                      '_patched_capture_model','_patched_post_kv_cache_wake_up'})
        st=ns['_State'](mode=mode,shadow_every=1,selfcheck_every=64)
        st.plan=object();st.metadata_cache['shape']='old';fused=object();stock=object()
        ns['_state_of']=lambda owner:st
        ns['_eligible']=lambda owner,*args:ns['_ensure_plan'](owner,st)
        ns['_fused_prepare_inputs']=Mock(return_value=fused)
        ns['_verify']=Mock(return_value=(stock,['input_ids']))
        ns['_ORIG']={'prepare_inputs':Mock(return_value=stock)}
        ns['build_plan']=Mock(side_effect=AssertionError('must stay disarmed'))
        return ns,st,fused,stock

    def test_numerical_drift_stays_disarmed_and_uses_stock_next_step(self):
        ns,st,fused,stock=self._runtime('on')
        call=ns['_patched_prepare_inputs']
        self.assertIs(call(object(),None,None,None),stock)
        self.assertEqual(st.checks_drift,1);self.assertTrue(st.plan_failed)
        self.assertFalse(st.plan_verified)
        self.assertIsNone(st.plan);self.assertEqual(st.metadata_cache,{})
        self.assertIs(call(object(),None,None,None),stock)
        self.assertEqual(ns['_fused_prepare_inputs'].call_count,1)
        ns['build_plan'].assert_not_called()
        ns['_ORIG']['prepare_inputs'].assert_called_once()

    def test_shadow_success_uses_fused_batch_but_drift_returns_stock_and_counts(self):
        ns,st,fused,stock=self._runtime('shadow')
        ns['_verify'].return_value=(stock,[])
        self.assertIs(ns['_patched_prepare_inputs'](object(),None,None,None),fused)
        self.assertEqual(st.checks_ok,1);self.assertEqual(st.steps_fused,1)
        ns['_verify'].return_value=(stock,['slot_mapping'])
        self.assertIs(ns['_patched_prepare_inputs'](object(),None,None,None),stock)
        self.assertEqual(st.checks_drift,1)
        self.assertIn('DRIFT',str(ns['logger'].warning.call_args))

    def test_on_verifies_first_use_even_with_periodic_disabled_then_keeps_cadence(self):
        for cadence in (64,0):
            with self.subTest(cadence=cadence):
                ns,st,fused,stock=self._runtime('on');st.selfcheck_every=cadence
                ns['_verify'].return_value=(stock,[])
                call=ns['_patched_prepare_inputs']
                self.assertIs(call(object(),None,None,None),fused)
                ns['_verify'].assert_called_once();self.assertTrue(st.plan_verified)
                first_log=ns['logger'].warning.call_args.args
                self.assertIn('first_plan_check=%s',first_log[0]);self.assertIs(first_log[-1],True)
                for _ in range(62):self.assertIs(call(object(),None,None,None),fused)
                ns['_verify'].assert_called_once()
                self.assertIs(call(object(),None,None,None),fused)
                self.assertEqual(ns['_verify'].call_count,2 if cadence else 1)
                self.assertEqual(st.steps_fused,64)

    def test_capture_and_kv_wake_require_first_comparison_of_the_rebuilt_plan(self):
        ns,st,fused,stock=self._runtime('on');ns['_verify'].return_value=(stock,[])
        owner=SimpleNamespace(_glm53_prep=st,model_state=SimpleNamespace())
        def plan(_):
            return SimpleNamespace(warmup=Mock(),G=7,gdn_groups=[],attn_g=0,factor=4,ratio=4,sbs=64,q=6)
        ns['build_plan']=Mock(side_effect=plan)
        ns['_ORIG']['capture_model']=Mock(return_value='capture')
        ns['_ORIG']['post_kv_cache_wake_up']=Mock(return_value='wake')
        call=ns['_patched_prepare_inputs']
        call(owner,None,None,None)
        for count,reset in enumerate(('_patched_capture_model','_patched_post_kv_cache_wake_up',
                                      '_patched_capture_model'),2):
            st.metadata_cache['old']='view'
            ns[reset](owner)
            self.assertFalse(st.plan_verified);self.assertEqual(st.metadata_cache,{})
            self.assertIs(call(owner,None,None,None),fused)
            self.assertTrue(st.plan_verified);self.assertEqual(ns['_verify'].call_count,count)
            self.assertIs(ns['logger'].warning.call_args.args[-1],True)
        self.assertEqual(ns['build_plan'].call_count,3)

    def test_drift_remains_disarmed_after_capture_and_kv_cache_wake(self):
        ns,st,fused,stock=self._runtime('on')
        owner=SimpleNamespace(_glm53_prep=st,model_state=SimpleNamespace())
        ns['_ORIG']['capture_model']=Mock(return_value='capture')
        ns['_ORIG']['post_kv_cache_wake_up']=Mock(return_value='wake')
        call=ns['_patched_prepare_inputs']
        self.assertIs(call(owner,None,None,None),stock)
        self.assertTrue(st.plan_failed)
        for reset in ('_patched_capture_model','_patched_post_kv_cache_wake_up'):
            ns[reset](owner)
            self.assertTrue(st.plan_failed);self.assertFalse(st.plan_verified)
            self.assertIsNone(st.plan)
            self.assertIs(call(owner,None,None,None),stock)
        ns['build_plan'].assert_not_called()
        self.assertEqual(ns['_verify'].call_count,1)
        self.assertEqual(ns['_fused_prepare_inputs'].call_count,1)
        self.assertEqual(ns['_ORIG']['prepare_inputs'].call_count,2)


if __name__ == '__main__':
    unittest.main()
