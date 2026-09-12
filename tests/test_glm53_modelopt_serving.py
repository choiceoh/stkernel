"""Encoding, scale direction and fixed dense routing for NVIDIA ST serving."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import torch

from engine.base.preshard import RankWriter
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53 import lanes
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.profiles.glm53.net import Glm53Net
from engine.profiles.glm53.weights import MODELOPT_WEIGHT_LAYOUT, WEIGHT_LAYOUT, rank_loader


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
    def scales(self):
        values = [torch.tensor(v) for v in ([.125,.5], [.25,.125], [.5,.25], [.125,.5])]
        return ModelOptScales.bind(*values, experts=2, device=torch.device('cpu'))

    def test_both_gemms_restore_activation_scale_and_invert_quantizer_scale(self):
        s = self.scales()
        self.assertTrue(torch.equal(s.alpha13, torch.tensor([.03125,.0625])))
        self.assertTrue(torch.equal(s.quant13, torch.tensor([4.,8.])))
        self.assertTrue(torch.equal(s.alpha2, torch.tensor([.0625,.125])))
        self.assertTrue(torch.equal(s.quant2, torch.tensor([8.,2.])))
        # Reconstructed x/a times an unscaled weight needs a*w exactly once.
        x, w = torch.tensor([3.,-2.]), torch.tensor([2.,4.])
        for a, alpha, q, wg in ((s.input13,s.alpha13,s.quant13,s.weight13),
                                (s.input2,s.alpha2,s.quant2,s.weight2)):
            self.assertTrue(torch.equal((x*q)*w*alpha, x*w*wg))
            self.assertTrue(torch.equal(a*q,torch.ones(2)))

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
        pointers=[v.data_ptr() for v in (scale.alpha13,scale.quant13,scale.alpha2,scale.quant2)]
        for rows in (1,6,129):
            x=torch.zeros(rows,128,dtype=torch.bfloat16)
            self.assertIs(net._dense(0,x),x)
            ids,w,kw=calls[-1]
            self.assertEqual(ids.shape,(rows,1));self.assertEqual(ids.dtype,torch.int32)
            self.assertEqual(torch.count_nonzero(ids),0);self.assertTrue(torch.equal(w,torch.ones_like(w)))
            self.assertIs(kw['scales'],scale)
        self.assertEqual(pointers,[v.data_ptr() for v in (scale.alpha13,scale.quant13,scale.alpha2,scale.quant2)])
        self.assertEqual(net.comm.all_reduce.call_count,3)
        self.assertEqual(table.moe_prepare.call_args.args[-2],1)

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


if __name__=='__main__':unittest.main()
