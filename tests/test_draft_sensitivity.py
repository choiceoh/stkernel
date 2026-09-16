"""CPU contracts for real-state capture and causal proposal/cost attribution."""
import json
from pathlib import Path
import tempfile
import itertools
from types import SimpleNamespace
import unittest

import torch

from engine.profiles.glm53.draft_replay import Capture, agreed, cpu, dense_state, save_prepared
from probes.draft_sensitivity import (active_weight_bytes, case_files, prefix_bounds,
    cost_screen, load_state, replace_reader, restore_dense, source_weight, summarize)


class Scores(unittest.TestCase):
    def case(self, old, target):
        return dict(case_id='case-00000',temperature=0,k=len(old),baseline_drafts=old,
                    target=target,label_source='committed_greedy_continuation')

    def test_late_matches_do_not_count_after_first_rejection(self):
        c = self.case([9,2,3],[1,2,3])
        r = summarize([c],[[9,2,3]],[[1,9,3]])
        self.assertEqual(r['baseline_prefix_bounds'],[0,0])
        self.assertEqual(r['candidate_prefix_bounds'],[1,1])
        self.assertEqual(r['prefix_survival_candidate'],[[1,1],[0,0],[0,0]])

    def test_censored_labels_are_bounds_not_fake_rejections_or_baseline_labels(self):
        self.assertEqual(prefix_bounds([1,2,3],[1]),(1,3))
        self.assertEqual(prefix_bounds([2,2,3],[1]),(0,0))
        c = self.case([9,2,3],[1])
        r = summarize([c],[[9,2,3]],[[1,2,3]])
        self.assertEqual(r['prefix_gain_bounds'],[1,3])
        self.assertEqual(r['labels_complete'],0)

    def test_cost_is_native_latency_and_prefix_gain_not_weight_error(self):
        c = self.case([1,9,9],[1,2,3])
        r = summarize([c],[[1,9,9]],[[1,2,9]],
                      [dict(baseline_us=100,candidate_us=140)])
        self.assertEqual(r['extra_proposal_us_per_extra_token'],40)
        self.assertEqual(r['max_extra_step_fraction_bounds'],[.5,.5])
        self.assertFalse(r['live_toks_measured'])
        with self.assertRaisesRegex(ValueError,'timing'):
            summarize([c],[[1,9,9]],[[1,2,9]],[dict(baseline_us=0,candidate_us=1)])

    def test_rejects_baseline_drift_and_sampled_or_off_prefix_labels(self):
        c = self.case([1,2,3],[1,2,3])
        with self.assertRaisesRegex(ValueError,'baseline replay drift'):
            summarize([c],[[1,2,4]],[[1,2,3]])
        for changed in (dict(temperature=1),dict(label_source='target_picks_after_rejection')):
            with self.assertRaisesRegex(ValueError,'greedy'):
                summarize([dict(c,**changed)],[[1,2,3]],[[1,2,3]])

    def test_cost_screen_does_not_prefer_expensive_low_impact_reader(self):
        def row(name,gain,time,memory):
            return dict(reader=name,prefix_gain_bounds=[gain,gain],proposal_delta_us=time,
                        active_weight_byte_delta_per_rank=[memory]*4)
        r=cost_screen([row('large-error',.1,20,100),row('sensitive',.3,5,50),row('cheap',.2,2,20)])
        self.assertEqual(r['dominated_by']['large-error'],['sensitive','cheap'])
        self.assertEqual(r['non_dominated'],['sensitive','cheap'])
        self.assertIsNone(r['deployment_winner'])


class Recorder(unittest.TestCase):
    def make(self, directory, *, max_cases=1):
        e = SimpleNamespace(tokens={1:[7]}, limits={1:(100,0.)}, min_new={}, ends={}, eos={99})
        e._generated_count = lambda seq: len(e.tokens[seq])-1
        e._rich = lambda seq: False
        e._reasoning_boundary = lambda seq: False
        e.async_ready = lambda seqs: True
        e.drafter = SimpleNamespace(k=3,F=SimpleNamespace(mask_id=0),
            target=SimpleNamespace(embed=lambda ids: ids[:,None].repeat(1,4).bfloat16(),
                                   comm=SimpleNamespace(gather_objects=lambda x:[x])),
            propose=lambda *a,**kw:[4,5,6])
        chunks = itertools.cycle(([9],[10,11],[12]))
        def decode(seqs, blocks, slots):
            for seq in seqs:
                e.drafter.propose(e.tokens[seq][-1],len(e.tokens[seq]),torch.zeros(1,2,4,1,4))
                e.tokens[seq].extend(next(chunks))
            return [False]*len(seqs)
        e.decode, e.close = decode, lambda seq: None
        rec = Capture(e,directory,'state-hash',max_cases=max_cases,every=1).install()
        return e,rec

    def test_future_committed_tokens_supply_labels_and_completed_capture_restores_async(self):
        with tempfile.TemporaryDirectory() as root:
            e,r = self.make(root)
            self.assertFalse(e.async_ready([1]))
            e.decode([1],None,[0])
            self.assertFalse((Path(root)/'case-00000.json').exists())
            e.decode([1],None,[0])
            labels=json.loads((Path(root)/'case-00000.json').read_text())
            self.assertEqual(labels['target'],[9,10,11])
            self.assertTrue(labels['complete'])
            self.assertTrue(e.async_ready([1]))
            case=torch.load(Path(root)/'case-00000.pt',weights_only=True)
            self.assertEqual(case['baseline_drafts'],[4,5,6])
            self.assertEqual(case['ids'].tolist(),[7,0,0,0])
            self.assertEqual(r.captured,1)

    def test_close_preserves_censored_case_and_missing_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            e,r=self.make(root)
            e.decode([1],None,[0])
            with self.assertRaisesRegex(ValueError,'unfinished'):
                case_files(Path(root),{'state_sha256':'state-hash'})
            e.close(1)
            cases=case_files(Path(root),{'state_sha256':'state-hash'})
            self.assertEqual(cases[0]['target'],[9])
            self.assertFalse(cases[0]['complete'])
            with self.assertRaisesRegex(ValueError,'different capture'):
                case_files(Path(root),{'state_sha256':'another-hash'})
            with (Path(root)/'case-00000.pt').open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                case_files(Path(root),{'state_sha256':'state-hash'})

    def test_sampled_and_modified_requests_do_not_enter_greedy_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            e,r=self.make(root)
            e.limits[1]=(100,1.)
            e.decode([1],None,[0])
            self.assertEqual(r.captured,0)
            e.limits[1]=(100,0.)
            e._rich=lambda seq: True
            e.decode([1],None,[0])
            self.assertEqual(r.captured,0)

    def test_cpu_snapshot_does_not_serialize_a_shared_arena(self):
        arena=torch.zeros(10000)
        view=arena[12:28].view(4,4)
        owned=cpu(view)
        self.assertEqual(owned.untyped_storage().nbytes(),64)
        view.fill_(1)
        self.assertEqual(owned.sum(),0)

    def test_c1_capture_does_not_include_multi_request_batches(self):
        with tempfile.TemporaryDirectory() as root:
            e,r=self.make(root)
            e.tokens[2]=[8]
            e.limits[2]=(100,0.)
            e.decode([1,2],None,[0,1])
            self.assertEqual(r.captured,0)

    def test_empty_directory_does_not_fabricate_cases(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'no real replay cases'):
                case_files(Path(root),{'state_sha256':'x'})

    def test_one_request_cannot_consume_the_whole_capture_budget(self):
        with tempfile.TemporaryDirectory() as root:
            e,r=self.make(root,max_cases=4)
            for _ in range(3): e.decode([1],None,[0])
            self.assertEqual(r.captured,2)
            e.tokens[2]=[8]; e.limits[2]=(100,0.)
            e.decode([2],None,[1])
            self.assertEqual(r.captured,3)


class Readers(unittest.TestCase):
    def test_small_drafter_recomputes_blocks_head_and_selector_after_one_reader_change(self):
        from engine.base.comm import Comm
        from engine.profiles.glm53.drafter import Drafter, DrafterFacts, specs
        f=DrafterFacts(layers=2,hidden=16,heads=2,kv_heads=1,head_dim=4,inter=16,
            rms_eps=1e-5,rope_theta=10000.,window=8,block=4,mask_id=20,
            conv_taps=2,conv_group=4,sel_rank=4,sel_top_k=3,target_layers=(1,),k=3)
        g=torch.Generator().manual_seed(71)
        rand=lambda *shape:(torch.randn(*shape,generator=g)*.3).bfloat16()
        table,head=rand(21,16),rand(21,16)
        target=SimpleNamespace(comm=Comm(),rank=0,vp=21,embed=lambda ids:table[ids],
                               head_local=lambda h:torch.nn.functional.linear(h,head))
        d=Drafter(f,target,21)
        d.p={s.name:torch.ones(s.shape,dtype=s.dtype) if s.name.endswith('norm.weight') else rand(*s.shape) for s in specs(f)}
        ring=rand(2,2,8,1,4); before=ring.clone(); anchor=torch.tensor([7])
        name='layers.0.mlp.down_proj.weight'; w=d.p[name]
        d.dense[name]=lambda x:torch.nn.functional.linear(x,w)
        baseline=d.propose_tensor(anchor,9,ring).tolist()
        changed=[]
        for scale in (0,-32,32):
            with replace_reader(d,name,lambda x,s=scale:torch.nn.functional.linear(x,w*s)):
                changed.append(d.propose_tensor(anchor,9,ring).tolist())
        self.assertTrue(any(x!=baseline for x in changed))
        self.assertEqual(d.propose_tensor(anchor,9,ring).tolist(),baseline)
        self.assertTrue(torch.equal(ring,before))

    def test_full_prepared_snapshot_loads_without_boot_or_gpu(self):
        from engine.base.comm import Comm
        from engine.profiles.glm53.drafter import Drafter, DrafterFacts, specs
        from engine.kernels.dense import W4Pack
        facts=DrafterFacts(layers=1,hidden=128,heads=2,kv_heads=1,head_dim=64,inter=128,
            rms_eps=1e-5,rope_theta=10000.,window=8,block=4,mask_id=2,
            conv_taps=2,conv_group=16,sel_rank=4,sel_top_k=2,target_layers=(1,),k=3)
        head=SimpleNamespace(rows=128,cols=128,weight=(torch.zeros(128,128).to(torch.float8_e4m3fn),torch.ones(1,1)))
        target=SimpleNamespace(comm=Comm(),rank=0,vp=128,dense={'head':head})
        d=Drafter(facts,target,128)
        d.p={s.name:torch.zeros(s.shape,dtype=s.dtype) for s in specs(facts)}
        d.dense={'layers.0.mlp.down_proj.weight':SimpleNamespace(rows=128,cols=128,
            decode_precision='w4',decode_input_rows=(8,),smooth=None,fp8=None,decode_fp8=None,
            packs=(W4Pack(torch.zeros(1,1,128,64,dtype=torch.uint8),torch.zeros(1,1,128,8,dtype=torch.int8),
                          torch.ones(128),128,128,True),))}
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); checkpoint=root/'weights'; checkpoint.write_bytes(b'CPU fixture')
            identity=save_prepared(d,root,checkpoint)
            restored,state,manifest=load_state(root,Comm(),'cpu')
            self.assertEqual(identity,manifest['state_sha256'])
            self.assertEqual(restored.F,facts)
            self.assertEqual(set(restored.dense),set(d.dense))
            self.assertNotIn('fc.weight',restored.p)
            self.assertTrue(torch.equal(restored.p['norm.weight'],d.p['norm.weight']))
            with self.assertRaisesRegex(ValueError,'already contains'):
                save_prepared(d,root,checkpoint)
            with self.assertRaisesRegex(ValueError,'rank/world'):
                load_state(root,SimpleNamespace(rank=1,world_size=2),'cpu')

    def test_tp_qkv_shards_each_projection_before_concatenating_and_applies_smoothing(self):
        weights={f'layers.0.self_attn.{s}_proj.weight':torch.arange(n*4).view(n,4).bfloat16()+offset
                 for s,n,offset in [('q',8,0),('k',4,100),('v',4,200)]}
        model=SimpleNamespace(get_tensor=weights.__getitem__)
        smooth=torch.tensor([1.,2.,4.,8.])
        got=source_weight(model,'layers.0.self_attn.qkv',1,2,smooth)
        expected=torch.cat([(weights[f'layers.0.self_attn.{s}_proj.weight'].float()*smooth).bfloat16().chunk(2)[1]
                            for s in ('q','k','v')])
        self.assertTrue(torch.equal(got,expected))
        with self.assertRaisesRegex(ValueError,'fixed-ring'):
            source_weight(model,'fc.weight',0,2,None)

    def test_prepared_dense_roundtrip_preserves_bytes_and_dispatch(self):
        from engine.kernels.dense import W4Pack
        original=SimpleNamespace(rows=128,cols=128,decode_precision='w4',decode_input_rows=(8,),
            smooth=torch.ones(128),fp8=None,decode_fp8=None,
            packs=(W4Pack(torch.zeros(1,1,128,64,dtype=torch.uint8),
                          torch.zeros(1,1,128,8,dtype=torch.int8),torch.ones(128),128,128,True),))
        state=dense_state(original)
        restored=restore_dense(state,'cpu')
        self.assertEqual(restored.decode_input_rows,(8,))
        self.assertIsNone(restored.observer)
        self.assertEqual(active_weight_bytes(state),8192+1024+512)
        for a,b in zip((original.packs[0].data,original.packs[0].scale,original.packs[0].rowscale),
                       (restored.packs[0].data,restored.packs[0].scale,restored.packs[0].rowscale)):
            self.assertTrue(torch.equal(a,b))

    def test_reader_replacement_recomputes_downstream_and_restores_on_error(self):
        counts=[]
        d=SimpleNamespace(dense={'first':lambda x:x,'second':lambda x:x.square()})
        def proposal():
            h=d.dense['first'](torch.tensor([-2.,1.]))
            h=d.dense['second'](h)
            counts.append(h.tolist())
            return h.argmax().item()
        self.assertEqual(proposal(),0)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with replace_reader(d,'first',lambda x:torch.tensor([0.,3.])):
                self.assertEqual(proposal(),1)
                raise RuntimeError('stop')
        self.assertEqual(proposal(),0)
        self.assertEqual(counts,[[4.,1.],[0.,9.],[4.,1.]])

    def test_local_file_failure_is_reported_to_every_rank(self):
        from engine.base.comm import LocalTP
        def rank(comm):
            def load():
                if comm.rank==1: raise FileNotFoundError('missing rank state')
                return 3
            try: agreed(comm,load)
            except ValueError as exc: return str(exc)
            self.fail('peer failure was ignored')
        errors=LocalTP(2,timeout_s=5).run(rank)
        self.assertEqual(errors[0],errors[1])
        self.assertIn('missing rank state',errors[0])


if __name__=='__main__':
    unittest.main()
