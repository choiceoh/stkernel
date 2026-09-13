"""Encoding, scale direction and fixed dense routing for NVIDIA ST serving."""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from engine.base.preshard import RankWriter
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53 import lanes
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.profiles.glm53.net import Glm53Net
from engine.profiles.glm53.weights import MODELOPT_BF16_DENSE_LAYOUT, MODELOPT_WEIGHT_LAYOUT, WEIGHT_LAYOUT, rank_loader


def small_facts():
    return Facts(hidden=128, layers=4, kinds=('kda','kda','kda','dsa'), dense=(0,1,2),
        vocab=256, rms_eps=1e-5, kda_heads=4, kda_dim=32, conv=4, lower_bound=-5.,
        heads=4, qk_nope=128, v_dim=128, q_lora=128, kv_lora=128,
        idx_heads=4, idx_dim=128, topk=8, kpool=4, max_position=4096,
        experts=2, topk_experts=2, moe_inter=256, dense_inter=512,
        routed_scale=1., swiglu_limit=10., hc=4, hc_eps=1e-6, sinkhorn=20,
        post_mult=2., weight_layout=MODELOPT_WEIGHT_LAYOUT)


def model(layer, table=None):
    comm = SimpleNamespace(world_size=4, rank=0, all_reduce=Mock(side_effect=lambda x:x))
    net = Glm53Net(small_facts(), comm, table or lanes.reference(), layers=[layer])
    views = {}
    for spec in net.specs():
        fill = 0x12 if spec.dtype == torch.uint8 else 1.
        views[spec.name] = torch.full(spec.shape, fill, dtype=spec.dtype)
        if spec.name.endswith(('_alpha', '_scale')) and spec.dtype == torch.float32:
            views[spec.name].fill_(.125 if spec.name.endswith('_alpha') else .25)
    return net, views


class ModelOptServingTests(unittest.TestCase):
    def test_dense_w4a16_guard_rows_are_safe_and_configurable(self):
        self.assertEqual(lanes.dense_w4a16_guard_rows("4096"), 4096)
        self.assertEqual(lanes.dense_w4a16_guard_rows("0"), 0)
        with patch.dict(os.environ, {"STK_GLM53_DENSE_W4A16_GUARD_ROWS": "8192"}):
            self.assertEqual(lanes.dense_w4a16_guard_rows(), 8192)
        for raw in ("-1", "nope"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                lanes.dense_w4a16_guard_rows(raw)

    def test_replay_judge_distinguishes_one_bf16_step_from_scatter_drift(self):
        from probes.engine_modelopt_check import bf16_ulps, repeat_stable
        x=torch.tensor([-1.,-.5,-0.,0.,.5,1.],dtype=torch.bfloat16)
        next_x=torch.nextafter(x,torch.full_like(x,float('inf')))
        self.assertEqual(bf16_ulps(x,next_x),1)
        self.assertTrue(repeat_stable(.001269,1))
        self.assertFalse(repeat_stable(.0125,2))
        self.assertEqual(bf16_ulps(torch.tensor([-0.],dtype=torch.bfloat16),
                                   torch.tensor([0.],dtype=torch.bfloat16)),0)

    def scales(self):
        values = [torch.tensor(v) for v in ([.125,.5], [.25,.125], [.5,.25], [.125,.5])]
        return ModelOptScales.bind(*values, experts=2, device=torch.device('cpu'))

    def test_both_gemms_restore_activation_scale_and_keep_quantizer_divisors(self):
        s = self.scales()
        self.assertTrue(torch.equal(s.alpha13, torch.tensor([.03125,.0625])))
        self.assertTrue(torch.equal(s.input13, torch.tensor([.25,.125])))
        self.assertTrue(torch.equal(s.alpha2, torch.tensor([.0625,.125])))
        self.assertTrue(torch.equal(s.input2, torch.tensor([.125,.5])))
        # Reconstructed x/a times an unscaled weight needs a*w exactly once.
        x, w = torch.tensor([3.,-2.]), torch.tensor([2.,4.])
        for a, alpha, wg in ((s.input13,s.alpha13,s.weight13),
                             (s.input2,s.alpha2,s.weight2)):
            self.assertTrue(torch.equal((x/a)*w*alpha, x*w*wg))

    def test_bad_scales_fail_at_bind_before_capture(self):
        for bad in (0., -1., float('nan'), float('inf'), 1e-45):
            values=[torch.ones(2) for _ in range(4)]; values[1][0]=bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ModelOptScales.bind(*values, experts=2, device=torch.device('cpu'))
        with self.assertRaises(ValueError):
            ModelOptScales.bind(*(torch.ones(2,dtype=torch.bfloat16) for _ in range(4)),
                               experts=2,device=torch.device('cpu'))

    def test_dense_uses_one_fixed_expert_and_keeps_packed_weight_views(self):
        calls=[]
        table=replace(lanes.reference(), moe=lambda x,ids,w,**kw:(calls.append((ids,w,kw)) or x),
                      moe_prepare=Mock())
        net, views=model(0,table); net.bind(views)
        self.assertNotIn('L0.mlp.gate_up',net.p)
        self.assertIs(net.p['L0.mlp.w13'],views['L0.mlp.w13'])
        scale=net._quant_scales[0]
        pointers=[v.data_ptr() for v in (scale.alpha13,scale.input13,scale.alpha2,scale.input2)]
        for rows in (1,6,129):
            x=torch.zeros(rows,128,dtype=torch.bfloat16)
            self.assertIs(net._dense(0,x),x)
            ids,w,kw=calls[-1]
            self.assertEqual(ids.shape,(rows,1));self.assertEqual(ids.dtype,torch.int32)
            self.assertEqual(torch.count_nonzero(ids),0);self.assertTrue(torch.equal(w,torch.ones_like(w)))
            self.assertIs(kw['scales'],scale)
        self.assertEqual(pointers,[v.data_ptr() for v in (scale.alpha13,scale.input13,scale.alpha2,scale.input2)])
        self.assertEqual(net.comm.all_reduce.call_count,3)
        self.assertEqual(table.moe_prepare.call_args.args[-2],1)
        # Native sequence-parallel prefill supplies its own reduction.
        reduce=Mock(side_effect=lambda x:x)
        net._dense(0,torch.zeros(4,128,dtype=torch.bfloat16),reduce=reduce)
        reduce.assert_called_once()
        self.assertEqual(net.comm.all_reduce.call_count,3)

    def test_moe_passes_each_experts_scales_and_keeps_shared_expert(self):
        calls=[]
        table=replace(lanes.reference(),moe=lambda x,ids,w,**kw:(calls.append(kw) or torch.zeros_like(x)))
        net,views=model(3,table); net.bind(views)
        net.route=lambda L,x:(torch.tensor([[0,1]],dtype=torch.int32),torch.tensor([[.4,.6]]))
        x=torch.full((1,128),.01,dtype=torch.bfloat16)
        output=net._moe(3,x)
        self.assertIs(calls[0]['scales'],net._quant_scales[3])
        self.assertGreater(float(output.abs().sum()),0.)  # shared BF16 expert is still added
        net.comm.all_reduce.assert_called_once()

    def test_redhat_contract_keeps_dense_bf16_and_scale_free_lane_call(self):
        F=replace(small_facts(),weight_layout=WEIGHT_LAYOUT)
        comm=SimpleNamespace(world_size=4,rank=0,all_reduce=lambda x:x)
        net=Glm53Net(F,comm,lanes.reference(),layers=[0,3])
        names={s.name for s in net.specs()}
        self.assertIn('L0.mlp.gate_up',names);self.assertNotIn('L0.mlp.w13',names)
        self.assertNotIn('L3.moe.a13_scale',names)

    def test_rank_requires_all_separate_scales_and_matching_encoding(self):
        net,views=model(0)
        contracts=[s for s in net.specs() if s.name.startswith('L0.mlp.')]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'rank.safetensors'
            def write(specs):
                writer=RankWriter(p,specs,{'weight_layout':MODELOPT_WEIGHT_LAYOUT,'rank':0,'world':4})
                for spec in specs:writer.put(spec.name,views[spec.name])
                writer.close()
            write(contracts)
            rank_loader(p,expected_layout=MODELOPT_WEIGHT_LAYOUT)
            with self.assertRaisesRegex(ValueError,'layout mismatch'):rank_loader(p,expected_layout=WEIGHT_LAYOUT)
            write([s for s in contracts if not s.name.endswith('a2_scale')])
            with self.assertRaisesRegex(ValueError,'ModelOpt scale'):rank_loader(p)


class ModelOptBf16DenseTests(unittest.TestCase):
    """ModelOpt routed experts with BF16 dense MLPs: the config excludes the dense MLPs from NVFP4, the dense layers
    take the packed BF16 path Red Hat's ranks use, the MoE layers keep ModelOpt's separate scales."""

    def config(self, patterns):
        return {'quantization_config': {'quant_method': 'modelopt', 'exclude_modules': patterns}}

    def test_the_config_selects_it_all_or_nothing(self):
        from engine.profiles.glm53.modelopt_weights import dense_excluded
        F = small_facts()
        prefix = 'model.language_model.layers.'
        self.assertFalse(dense_excluded(self.config(['lm_head', prefix + '3.mlp.gate']), F))
        self.assertTrue(dense_excluded(self.config([prefix + f'{layer}.mlp*' for layer in (0, 1, 2)]), F))
        self.assertTrue(dense_excluded(self.config([prefix + f'{layer}.mlp.{p}_proj' for layer in (0, 1, 2)
                                                    for p in ('gate', 'up', 'down')]), F))
        with self.assertRaisesRegex(ValueError, 'all NVFP4 or all excluded'):
            dense_excluded(self.config([prefix + '0.mlp*']), F)

    def test_dense_layers_are_bf16_and_moe_layers_keep_modelopt_scales(self):
        F = replace(small_facts(), weight_layout=MODELOPT_BF16_DENSE_LAYOUT)
        comm = SimpleNamespace(world_size=4, rank=0, all_reduce=lambda x: x)
        net = Glm53Net(F, comm, lanes.reference(), layers=[0, 3])
        self.assertTrue(net.modelopt)
        self.assertFalse(net.dense_nvfp4)
        self.assertEqual(net._dense.__func__, Glm53Net._dense)
        by_name = {s.name: s for s in net.specs()}
        self.assertEqual(by_name['L0.mlp.gate_up'].dtype, torch.bfloat16)
        self.assertNotIn('L0.mlp.w13', by_name)
        self.assertNotIn('L0.mlp.a13_scale', by_name)
        for suffix in ('w13', 'w13_sf', 'w13_alpha', 'a13_scale', 'w2', 'w2_sf', 'w2_alpha', 'a2_scale'):
            self.assertIn('L3.moe.' + suffix, by_name)
        self.assertIn('L0.mlp.gate_up', Glm53Net.dense_weight_names(by_name))

    def test_bind_prepares_experts_for_moe_layers_only(self):
        calls = []
        table = replace(lanes.reference(), moe=lambda x, ids, w, **kw: (calls.append(kw) or torch.zeros_like(x)),
                        moe_prepare=Mock())
        F = replace(small_facts(), weight_layout=MODELOPT_BF16_DENSE_LAYOUT)
        comm = SimpleNamespace(world_size=4, rank=0, all_reduce=Mock(side_effect=lambda x: x))
        net = Glm53Net(F, comm, table, layers=[0, 3])
        views = {}
        for spec in net.specs():
            views[spec.name] = torch.full(spec.shape, 0x12 if spec.dtype == torch.uint8 else 1., dtype=spec.dtype)
            if spec.name.endswith(('_alpha', '_scale')) and spec.dtype == torch.float32:
                views[spec.name].fill_(.125 if spec.name.endswith('_alpha') else 1.)
        net.bind(views)
        self.assertEqual(sorted(net._experts), [3])
        self.assertEqual(sorted(net._quant_scales), [3])
        self.assertEqual(table.moe_prepare.call_count, 1)

    def test_rank_marker_is_checked_and_moe_scales_are_still_required(self):
        F = replace(small_facts(), weight_layout=MODELOPT_BF16_DENSE_LAYOUT)
        comm = SimpleNamespace(world_size=4, rank=0, all_reduce=lambda x: x)
        net = Glm53Net(F, comm, lanes.reference(), layers=[0, 3])
        contracts = [s for s in net.specs() if s.name.startswith(('L0.mlp.', 'L3.moe.'))]
        views = {s.name: torch.ones(s.shape, dtype=s.dtype) for s in contracts}
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'rank.safetensors'

            def write(specs):
                writer = RankWriter(p, specs, {'weight_layout': MODELOPT_BF16_DENSE_LAYOUT, 'rank': 0, 'world': 4})
                for spec in specs:
                    writer.put(spec.name, views[spec.name])
                writer.close()
            write(contracts)
            rank_loader(p, expected_layout=MODELOPT_BF16_DENSE_LAYOUT)
            with self.assertRaisesRegex(ValueError, 'layout mismatch'):
                rank_loader(p, expected_layout=MODELOPT_WEIGHT_LAYOUT)
            write([s for s in contracts if not s.name.endswith('L3.moe.a2_scale')])
            with self.assertRaisesRegex(ValueError, 'ModelOpt scale'):
                rank_loader(p)


if __name__=='__main__':unittest.main()
