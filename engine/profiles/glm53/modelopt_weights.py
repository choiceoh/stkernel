"""Lossless offline TP=4 layout for NVIDIA's GLM ModelOpt NVFP4 checkpoint.

Packed bytes and E4M3 block scales are split/reordered without arithmetic.
FP32 weight_scale_2 multipliers and input_scale values remain separate. The
first three dense MLPs remain NVFP4. This encoding is intentionally distinct
from the live folded-scale layout; it requires an explicit serving adapter.
"""
from pathlib import Path
import json

import torch

from engine.base.params import Spec
from engine.modules.nvfp4_sf import swizzle_sf
from engine.profiles.glm53 import facts, specs


WEIGHT_LAYOUT = 'st-glm53-modelopt-up-gate-v1'


def load_facts(path):
    config = json.loads((Path(path)/'config.json').read_text())
    q = config['quantization_config']
    if q.get('quant_method') != 'modelopt' or q.get('quant_algo') != 'NVFP4':
        raise ValueError('requires the NVIDIA ModelOpt NVFP4 checkpoint')
    groups = q['config_groups']
    if set(groups) != {'group_0'} or groups['group_0']['targets'] != ['Linear']:
        raise ValueError('unexpected ModelOpt quantization groups')
    for side in ('weights', 'input_activations'):
        scheme = groups['group_0'][side]
        if scheme['num_bits'] != 4 or scheme['type'] != 'float' or scheme['group_size'] != 16:
            raise ValueError('requires NVFP4 group 16 for weights and activations')
    return facts.architecture(config)


def quant_specs(F, layer):
    dense = not F.is_moe(layer)
    prefix = f'{specs.CK}layers.{layer}.mlp.'
    names = [prefix] if dense else [prefix+f'experts.{e}.' for e in range(F.experts)]
    out_prefix = f'L{layer}.' + ('mlp.' if dense else 'moe.')
    intermediate = F.dense_inter_local if dense else F.moe_inter_local
    experts, hidden = len(names), F.hidden
    suffixes = ('weight','weight_scale','weight_scale_2','input_scale')
    first_keys = tuple(p+projection+'_proj.'+suffix for p in names
                       for projection in ('up','gate') for suffix in suffixes)
    second_keys = tuple(p+'down_proj.'+suffix for p in names for suffix in suffixes)

    def packed_first(source, rank, world):
        return torch.stack([torch.cat([specs._split(source[p+g+'_proj.weight'],0,rank,world)
                                      for g in ('up','gate')]) for p in names]).contiguous()

    def packed_second(source, rank, world):
        return torch.stack([specs._split(source[p+'down_proj.weight'],1,rank,world) for p in names]).contiguous()

    def scale_first(source, rank, world):
        return torch.stack([swizzle_sf(torch.cat([specs._split(source[p+g+'_proj.weight_scale'],0,rank,world)
                           for g in ('up','gate')]).view(torch.uint8)).view(torch.float8_e4m3fn) for p in names]).contiguous()

    def scale_second(source, rank, world):
        return torch.stack([swizzle_sf(specs._split(source[p+'down_proj.weight_scale'],1,rank,world).contiguous()
                           .view(torch.uint8)).view(torch.float8_e4m3fn) for p in names]).contiguous()

    def scalar_first(suffix):
        def build(source, rank, world):
            values=[]
            for p in names:
                up,gate=(source[p+g+'_proj.'+suffix].float().reshape(()) for g in ('up','gate'))
                if not torch.equal(up,gate):
                    raise ValueError(f'{p}: gate/up {suffix} must match for fused FC1')
                values.append(up)
            result=torch.stack(values)
            if not torch.isfinite(result).all() or not (result>0).all():
                raise ValueError('global scales must be positive finite multipliers')
            return result
        return build

    def scalar_second(suffix):
        def build(source, rank, world):
            result=torch.stack([source[p+'down_proj.'+suffix].float().reshape(()) for p in names])
            if not torch.isfinite(result).all() or not (result>0).all():
                raise ValueError('global scales must be positive finite multipliers')
            return result
        return build

    return [
        Spec(out_prefix+'w13',(experts,2*intermediate,hidden//2),torch.uint8,first_keys,packed_first),
        Spec(out_prefix+'w13_sf',(experts,2*intermediate*(hidden//16)),torch.float8_e4m3fn,first_keys,scale_first),
        Spec(out_prefix+'w13_alpha',(experts,),torch.float32,first_keys,scalar_first('weight_scale_2')),
        Spec(out_prefix+'a13_scale',(experts,),torch.float32,first_keys,scalar_first('input_scale')),
        Spec(out_prefix+'w2',(experts,hidden,intermediate//2),torch.uint8,second_keys,packed_second),
        Spec(out_prefix+'w2_sf',(experts,hidden*(intermediate//16)),torch.float8_e4m3fn,second_keys,scale_second),
        Spec(out_prefix+'w2_alpha',(experts,),torch.float32,second_keys,scalar_second('weight_scale_2')),
        Spec(out_prefix+'a2_scale',(experts,),torch.float32,second_keys,scalar_second('input_scale')),
    ]


def layer_specs(F, layer):
    ordinary = specs.layer_specs(F,layer)
    replaced = (f'L{layer}.moe.w13',f'L{layer}.moe.w13_sf',f'L{layer}.moe.w2',f'L{layer}.moe.w2_sf') if F.is_moe(layer) else (f'L{layer}.mlp.gate_up',f'L{layer}.mlp.down')
    return [s for s in ordinary if s.name not in replaced]+quant_specs(F,layer)


def groups(F,layers):
    top=specs.top_specs(F)
    yield 'top',top
    for layer in layers:
        yield f'layer {layer}',layer_specs(F,layer)
