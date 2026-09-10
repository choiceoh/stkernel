"""Execute the C1/K5 host optimization with CPU storage/launch spies only."""
from dataclasses import fields
import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_glm53_prep_fused_kv_integration import load_defs


class Array:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, Array):
            return Array(self.values[i] for i in index.values)
        return self.values[index]

    def __mul__(self, value):
        return Array(x * value for x in self.values)

    def copy(self):
        return Array(self.values)


class Tensor:
    views = 0
    clones = 0
    pointers = 0

    def __init__(self, values, *, storage=None, start=0, length=None):
        self.storage = list(values) if storage is None else storage
        self.start = start
        self.shape = (len(self.storage) if length is None else length,)
        self.strides = (1,)
        self.dtype, self.device = 'int32', 'cuda:0'

    def data_ptr(self):
        Tensor.pointers += 1
        return id(self.storage) + self.start * 4

    def stride(self):
        return self.strides

    def values(self):
        return self.storage[self.start:self.start + self.shape[0]]

    def __getitem__(self, item):
        start, stop, step = item.indices(self.shape[0])
        assert step == 1
        Tensor.views += 1
        return Tensor([], storage=self.storage, start=self.start + start, length=stop-start)

    def clone(self):
        Tensor.clones += 1
        return Tensor(self.values())

    def __ne__(self, other):
        count = sum(a != b for a, b in zip(self.values(), other.values()))
        return NS(sum=lambda:NS(item=lambda:count))


def namespace():
    numpy = NS(int32='int32', arange=lambda n, **kw:Array(range(n)),
               zeros=lambda n, **kw:Array([0] * n),
               full=lambda n, value, **kw:Array([value] * n))
    numpy.add = lambda a, b, out:out.values.__setitem__(slice(None),
        [x+y for x, y in zip(a.values, b.values)])
    return load_defs({'PrepPlan', '_State', '_fused_prepare_inputs', '_report_decode_opt',
                      '_patched_prepare_inputs', '_patched_capture_model',
                      '_patched_post_kv_cache_wake_up'}, json=json, time=time, np=numpy,
                     torch=NS(equal=lambda a,b:a.shape == b.shape and a.values() == b.values(),
                              from_numpy=lambda a:Tensor(a.values)))


def runtime(ns, *, opt=True, q=6, mode='on'):
    plan = object.__new__(ns['PrepPlan'])
    plan.q, plan.decode_opt = q, opt
    plan.decode_live_diff_checks = 0
    plan.owned = {name:Tensor(range(96)) for name in (
        'expanded_idx', 'expanded_pos', 'logits_arange', 'cu_num_logits')}
    names = ('input_ids', 'positions', 'query_start_loc', 'seq_lens', 'is_padding',
             'req_id_buf', 'exp_bt', 'dec_seq_lens', 'dec_lens', 'per_req_dec_lens',
             'idx_bt', 'comp_slot', 'sched_buf')
    for name in names:
        setattr(plan, name, Tensor(range(96)))
    plan.bt = NS(slot_mappings=Tensor(range(96)),
                 input_block_tables=[Tensor(range(96)), Tensor(range(96))])
    plan.gdn_groups = [1]
    for name in ('gdn_state', 'gdn_mask', 'gdn_tok', 'gdn_qsl', 'gdn_nacc'):
        setattr(plan, name, [Tensor(range(96))])
    runner = NS(input_buffers=NS(**{n:getattr(plan,n) for n in (
        'input_ids', 'positions', 'query_start_loc', 'seq_lens', 'is_padding')}),
        req_states=NS(num_computed_tokens_np=Array([100, 200])))
    plan.launch = Mock(side_effect=lambda idx, n:Tensor(idx.values))
    st = ns['_State'](mode=mode, shadow_every=1, selfcheck_every=64, plan=plan)
    runner._glm53_prep = st
    return runner, st, plan


def request(index=0):
    return NS(req_ids=['r'+str(index)], idx_mapping_np=Array([index]),
              num_scheduled_tokens=Array([6]), prefill_len_np=Array([3]),
              num_computed_prefill_tokens_np=Array([3]), is_prefilling_np=Array([False]))


class PrepDecodeOptTests(unittest.TestCase):
    def test_exact_opt_flags_default_off(self):
        ns = namespace()
        factory = next(f for f in fields(ns['PrepPlan']) if f.name == 'decode_opt').default_factory
        for opt, ep, expected in (('0','1',False), ('1','0',False), ('true','1',False),
                                  ('1','1',True), ('1','true',False)):
            with patch.dict(os.environ, {'VLLM_GLM53_EP_DECODE_OPT':opt,
                                         'VLLM_GLM53_EP_TILED':ep}):
                self.assertIs(factory(), expected)

    def test_steady_input_preparation_keeps_fresh_views_without_metadata_guards(self):
        for opt in (False, True):
            ns = namespace(); runner, st, plan = runtime(ns, opt=opt)
            module = ModuleType('vllm.v1.worker.gpu.input_batch'); module.InputBatch = NS
            Tensor.views = Tensor.pointers = 0
            with patch.dict(sys.modules, {module.__name__:module}):
                first = ns['_fused_prepare_inputs'](runner, st,
                    NS(has_structured_output_requests=False), request(0), None)
                second = ns['_fused_prepare_inputs'](runner, st,
                    NS(has_structured_output_requests=True), request(1), None)
            self.assertIsNot(first, second)
            self.assertIsNot(first.input_ids, second.input_ids)
            self.assertEqual(Tensor.views, 18)
            self.assertEqual(Tensor.pointers, 0)
            self.assertEqual(first.idx_mapping.values(), [0])
            self.assertEqual(second.idx_mapping.values(), [1])
            self.assertEqual(first.seq_lens_cpu_upper_bound.values(), [106])
            self.assertEqual(second.seq_lens_cpu_upper_bound.values(), [206])
            self.assertEqual(first.num_computed_tokens_np.values, [100])
            self.assertEqual(second.num_computed_tokens_np.values, [200])
            for name in ('query_start_loc_np', 'cu_num_logits_np', 'num_draft_tokens_per_req'):
                self.assertIsNot(getattr(first, name), getattr(second, name))
            self.assertIsNot(first.query_start_loc_np, first.cu_num_logits_np)
            self.assertFalse(first.has_structured_output_reqs)
            self.assertTrue(second.has_structured_output_reqs)
            self.assertEqual(plan.launch.call_count, 2)
            self.assertEqual(plan.decode_live_diff_checks, 0)
            self.assertEqual(first.query_start_loc.shape, (2,))
            self.assertEqual(first.seq_lens.shape, (1,))
            self.assertEqual(first.input_ids.shape, (6,))

    def test_live_diff_preserves_all_checks_without_second_clone(self):
        for dirty in (False, True):
            observed = []
            for opt in (False, True):
                ns = namespace(); runner, st, plan = runtime(ns, opt=opt)
                Tensor.clones = 0
                snap = plan.snapshot(1)
                count = len(snap)
                self.assertEqual(Tensor.clones, count)
                if dirty:
                    plan.input_ids.storage[0] = -99
                    plan.bt.input_block_tables[1].storage[0] = -88
                    plan.gdn_nacc[0].storage[0] = -77
                bad = plan.diff(snap, 1, direct_views=True)
                observed.append(bad)
                self.assertEqual(Tensor.clones, count if opt else 2*count)
                self.assertEqual(plan.decode_live_diff_checks, int(opt and not dirty))
                # The original fused snapshot stays immutable after stock writes.
                self.assertEqual(snap['input_ids'].values()[0], 0)
            self.assertEqual(observed[0], observed[1])
            self.assertEqual(observed[1], ['input_ids(1)', 'bt1(1)', 'gdn0_nacc(1)'] if dirty else [])

    def test_live_diff_shape_change_is_still_reported(self):
        ns = namespace(); runner, st, plan = runtime(ns)
        snap = plan.snapshot(1)
        plan.comp_slot.shape = (95,)
        self.assertEqual(plan.diff(snap, 1, direct_views=True), ['comp_slot(-1)'])
        self.assertEqual(plan.decode_live_diff_checks, 0)

    def test_other_shapes_and_shadow_keep_two_snapshots(self):
        for q, n, direct in ((4,1,True), (6,2,True), (6,1,False)):
            ns = namespace(); runner, st, plan = runtime(ns, q=q)
            Tensor.clones = 0
            snap = plan.snapshot(n); plan.diff(snap,n,direct_views=direct)
            self.assertEqual(Tensor.clones, 2*len(snap))
            self.assertEqual(plan.decode_live_diff_checks, 0)

    def test_success_markers_follow_existing_first_and_64_step_checks(self):
        ns = namespace(); runner, st, plan = runtime(ns)
        ns['_state_of'] = lambda _:st; ns['_eligible'] = lambda *args:True
        ns['_ORIG'] = {'prepare_inputs':Mock()}
        batch = NS(num_reqs=1)
        ns['_fused_prepare_inputs'] = Mock(return_value=batch)
        def verify(*args):
            snap = plan.snapshot(1)
            return batch, plan.diff(snap, 1, direct_views=True)
        ns['_verify'] = Mock(side_effect=verify)
        for _ in range(128):ns['_patched_prepare_inputs'](runner, None, None, None)
        self.assertEqual(ns['_verify'].call_count, 3)
        messages = [json.loads(call.args[1]) for call in ns['logger'].warning.call_args_list
                    if call.args[0] == '[prep-decode-opt] USED %s']
        self.assertEqual([m['fused_steps'] for m in messages], [1,64,128])
        self.assertEqual([m['live_diff_checks'] for m in messages], [1,2,3])
        self.assertTrue(all(m['version']==1 and m['num_reqs']==1 and m['q']==6
            and m['live_snapshot_clones_elided'] and m['first_plan_check_passed'] for m in messages))
        calls = ns['logger'].warning.call_args_list
        for i, call in enumerate(calls):
            if call.args[0] == '[prep-decode-opt] USED %s':
                self.assertTrue(calls[i+1].args[0].startswith('[prep-fused] %s:'))
                self.assertEqual(calls[i+1].args[2], json.loads(call.args[1])['fused_steps'])
        ns['_verify'].side_effect = None
        ns['_verify'].return_value = (batch, ['input_ids'])
        for _ in range(64):ns['_patched_prepare_inputs'](runner, None, None, None)
        self.assertIsNone(st.plan); self.assertTrue(st.plan_failed)
        self.assertEqual(sum(c.args[0] == '[prep-decode-opt] USED %s'
                             for c in ns['logger'].warning.call_args_list), 3)

    def test_no_used_marker_without_selected_successful_comparison(self):
        for reason in ('unverified', 'no_diff', 'flag_off', 'shadow', 'q4', 'n2'):
            ns = namespace(); runner, st, plan = runtime(ns)
            st.plan_verified = True; st.steps_fused = 64
            plan.decode_live_diff_checks = 2
            batch = NS(num_reqs=1)
            if reason == 'unverified':st.plan_verified = False
            elif reason == 'no_diff':plan.decode_live_diff_checks = 0
            elif reason == 'flag_off':plan.decode_opt = False
            elif reason == 'shadow':st.mode = 'shadow'
            elif reason == 'q4':plan.q = 4
            elif reason == 'n2':batch.num_reqs = 2
            ns['_report_decode_opt'](st, batch)
            ns['logger'].warning.assert_not_called()

    def test_capture_wake_and_disarm_require_new_plan_comparison(self):
        for reset in ('capture', 'wake', 'disarm'):
            ns = namespace(); runner, st, old = runtime(ns)
            old.decode_live_diff_checks = 99
            st.plan_verified = True
            ns['_state_of'] = lambda _:st
            ns['_ensure_plan'] = Mock(return_value=False)
            ns['_ORIG'] = {name:Mock() for name in ('capture_model','post_kv_cache_wake_up')}
            if reset == 'disarm':st.disarm('test')
            else:ns['_patched_capture_model' if reset == 'capture' else
                    '_patched_post_kv_cache_wake_up'](runner)
            self.assertIsNone(st.plan); self.assertFalse(st.plan_verified)
            runner2, st2, new = runtime(ns)
            self.assertEqual(new.decode_live_diff_checks, 0)
            self.assertFalse(st2.plan_verified)
            ns['logger'].warning.reset_mock()
            ns['_report_decode_opt'](st2, NS(num_reqs=1))
            ns['logger'].warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
