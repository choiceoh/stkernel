"""CPU execution of actual fused address/arithmetic bodies; no GPU proof."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


class Pointer:
    def __init__(self, data, offsets=0):
        self.data, self.offsets = data.reshape(-1), offsets

    def __add__(self, offsets):
        return Pointer(self.data, self.offsets + offsets)


def kernel():
    source = ROOT / 'engine/kernels/prefill_mhc.py'
    node = copy.deepcopy(next(n for n in ast.parse(source.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == '_post_prenorm'))
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    program = [0, 0]

    def load(pointer, mask, other=0):
        offsets, mask = torch.broadcast_tensors(pointer.offsets, mask)
        selected = offsets[mask]
        assert bool(((selected >= 0) & (selected < pointer.data.numel())).all())
        output = torch.full(offsets.shape, other, dtype=pointer.data.dtype)
        output[mask] = pointer.data[selected]
        return output

    def store(pointer, value, mask):
        offsets, mask = torch.broadcast_tensors(pointer.offsets, mask)
        assert bool(((offsets[mask] >= 0) & (offsets[mask] < pointer.data.numel())).all())
        pointer.data[offsets[mask]] = value.expand(offsets.shape)[mask]

    tl = NS(program_id=lambda d: program[d], arange=torch.arange, static_range=range,
            cdiv=lambda a,b: (a+b-1)//b, zeros=torch.zeros, load=load, store=store,
            float32=torch.float32, bfloat16=torch.bfloat16, int64=torch.int64,
            where=torch.where, sum=lambda x,axis: x.sum(dim=axis),
            fma=lambda a,b,c: a*b+c, dot=lambda a,b,c: a.float()@b.float()+c)
    scope = dict(tl=tl)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope[node.name], program


class MhcCpuTests(unittest.TestCase):
    def test_real_fused_body_preserves_rounded_residual_and_packed_weight_coordinates(self):
        run, program = kernel()
        generator = torch.Generator().manual_seed(93)
        for m, h, splits in ((65, 256, 1), (97, 512, 3)):
            rand = lambda *shape: torch.randn(*shape, generator=generator)
            residual, x = rand(m,4,h).bfloat16(), rand(m,h).bfloat16()
            post, comb, fn = rand(m,4), rand(m,4,4), rand(24,4*h).bfloat16()
            packed = fn.reshape(24,4,h).transpose(1,2).contiguous()
            out = torch.full_like(residual, float('nan'))
            mul, sqr = torch.full((splits,m,24), float('nan')), torch.full((splits,m), float('nan'))
            for split in range(splits):
                program[1] = split
                for tile in range((m+31)//32):
                    program[0] = tile
                    run(*(Pointer(v) for v in (comb,residual,post,x,packed,out,mul,sqr)),m,h,splits,32,128,32)
            reference = []
            for channel in range(4):
                r = post[:,channel,None]*x.float()
                for source in range(4):
                    r = comb[:,source,channel,None]*residual[:,source].float()+r
                reference.append(r.bfloat16())
            expected = torch.stack(reference, dim=1)
            self.assertTrue(torch.equal(out, expected))
            flat = expected.float().reshape(m,-1)
            torch.testing.assert_close(mul.sum(0), flat@fn.float().T, atol=.002, rtol=2e-5)
            torch.testing.assert_close(sqr.sum(0), flat.square().sum(1), atol=.03, rtol=2e-6)

    def test_prefill_uses_existing_lossless_pack_and_decode_entry_stays_separate(self):
        tree = ast.parse((ROOT/'engine/kernels/dense/mhc.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHC')
        node = copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prefill'))
        # The invalid cases return before importing a CUDA implementation.
        scope = {}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<real-MHC-prefill>','exec'),scope)
        for count, packed in ((7,object()), (64,object()), (32769,object()), (668,None)):
            result = scope['prefill'](NS(weights={'f':(None,packed)}),'f',NS(shape=(count,4096)),
                                      *([None]*6), 1e-6,1e-6,2.,20)
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
