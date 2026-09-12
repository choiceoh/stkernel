"""Independent placement and scale-convention checks for the offline importer."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from engine.modules.nvfp4_sf import unswizzle_sf
from engine.base.preshard import RankWriter
from engine.base.loader import RankLoader
from engine.profiles.glm53.modelopt_weights import quant_specs,WEIGHT_LAYOUT
from engine.profiles.glm53.weights import rank_loader


class ModelOptPreshardTests(unittest.TestCase):
    def fixture(self,layer=3):
        F=SimpleNamespace(hidden=128,experts=2,moe_inter_local=64,dense_inter_local=128,is_moe=lambda l:l>=3)
        dense=layer<3;inter=512 if dense else 256
        names=[f'model.language_model.layers.{layer}.mlp.'] if dense else [f'model.language_model.layers.{layer}.mlp.experts.{e}.' for e in range(2)]
        source={}
        for e,p in enumerate(names):
            for j,g in enumerate(('up','gate','down')):
                n,k=(128,inter) if g=='down' else (inter,128)
                source[p+g+'_proj.weight']=(torch.arange(n*k//2).reshape(n,k//2)%251+j).to(torch.uint8)
                source[p+g+'_proj.weight_scale']=((torch.arange(n*k//16).reshape(n,k//16)%7+1)/16).to(torch.float8_e4m3fn)
                source[p+g+'_proj.weight_scale_2']=torch.tensor((e+1)*.0001*(2 if g=='down' else 1))
                source[p+g+'_proj.input_scale']=torch.tensor(.003 if g=='down' else .002)
        return F,names,source

    def check_split(self,layer):
        F,names,source=self.fixture(layer);contracts=quant_specs(F,layer)
        for rank in range(4):
            got={s.name.rsplit('.',1)[1]:s.build(source,rank,4) for s in contracts}
            inter=F.moe_inter_local if layer>=3 else F.dense_inter_local
            for e,p in enumerate(names):
                expected=torch.cat([source[p+g+'_proj.weight'][rank*inter:(rank+1)*inter] for g in ('up','gate')])
                self.assertTrue(torch.equal(got['w13'][e],expected))
                expected_sf=torch.cat([source[p+g+'_proj.weight_scale'][rank*inter:(rank+1)*inter] for g in ('up','gate')]).view(torch.uint8)
                self.assertTrue(torch.equal(unswizzle_sf(got['w13_sf'][e].view(torch.uint8),2*inter,8),expected_sf))
                self.assertTrue(torch.equal(got['w2'][e],source[p+'down_proj.weight'][:,rank*inter//2:(rank+1)*inter//2]))
                self.assertTrue(torch.equal(unswizzle_sf(got['w2_sf'][e].view(torch.uint8),128,inter//16),source[p+'down_proj.weight_scale'][:,rank*inter//16:(rank+1)*inter//16].view(torch.uint8)))
                self.assertEqual(got['w13_alpha'][e].item(),source[p+'up_proj.weight_scale_2'].item())
                self.assertEqual(got['w2_alpha'][e].item(),source[p+'down_proj.weight_scale_2'].item())
                self.assertLess(got['w13_alpha'][e].item(),.001)
                self.assertEqual(got['a13_scale'][e].item(),source[p+'up_proj.input_scale'].item())

    def test_expert_tp_splits_preserve_packed_weights_raw_scales_and_multipliers(self):self.check_split(3)
    def test_dense_mlp_remains_packed_nvfp4(self):self.check_split(0)

    def test_incompatible_fused_gate_up_scales_are_rejected(self):
        F,names,source=self.fixture()
        source[names[0]+'gate_proj.weight_scale_2']=torch.tensor(.1)
        contract=next(s for s in quant_specs(F,3) if s.name.endswith('w13_alpha'))
        with self.assertRaisesRegex(ValueError,'gate/up'):contract.build(source,0,4)

    def test_modelopt_rank_roundtrip_and_explicit_layout_acceptance(self):
        F,_,source=self.fixture();contracts=quant_specs(F,3)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rank0of4.safetensors'
            writer=RankWriter(path,contracts,{'weight_layout':WEIGHT_LAYOUT,'rank':0,'world':4})
            for spec in contracts:writer.put(spec.name,spec.build(source,0,4))
            writer.close();loaded=RankLoader(path).load([s.name for s in contracts],device='cpu')
            for spec in contracts:
                self.assertTrue(torch.equal(loaded[spec.name].view(torch.uint8),spec.build(source,0,4).view(torch.uint8)))
            self.assertEqual(rank_loader(path, expected_layout=WEIGHT_LAYOUT).metadata['weight_layout'], WEIGHT_LAYOUT)
            with self.assertRaisesRegex(ValueError,'layout mismatch'):
                rank_loader(path, expected_layout='st-glm53-b12x-up-gate-v1')


if __name__=='__main__':unittest.main()
