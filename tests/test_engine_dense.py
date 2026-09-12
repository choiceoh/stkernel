"""Quantized dense execution against its arithmetic twin and the BF16 source."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT, "requires CUDA; " + REASON)
class DenseTests(unittest.TestCase):
    def test_retired_arena_storage_keeps_pack_bytes_and_graph_addresses(self):
        from engine.kernels.dense import DenseLinear
        torch.manual_seed(63)
        backing=torch.full((1024*512*2+512,),83,device='cuda',dtype=torch.uint8)
        source=backing[256:-256].view(torch.bfloat16).view(1024,512)
        source.normal_()
        layer=DenseLinear(source)
        inputs=[torch.randn(m,512,device='cuda',dtype=torch.bfloat16) for m in (6,33,1024)]
        expected=[layer(x).clone() for x in inputs]
        layer.consume_weight(source)
        for x,ref in zip(inputs,expected):
            torch.testing.assert_close(layer(x),ref,rtol=0,atol=0)
        self.assertTrue((backing[:256]==83).all() and (backing[-256:]==83).all())
        for tensor in (layer.packs[0].data,layer.packs[0].scale,layer.packs[0].rowscale,*layer.fp8.weight):
            self.assertGreaterEqual(tensor.data_ptr(),source.data_ptr())
            self.assertLess(tensor.data_ptr()+tensor.numel()*tensor.element_size(),source.data_ptr()+source.numel()*2+1)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=layer(inputs[0])
        try:
            for _ in range(3):
                inputs[0].normal_();ref=layer(inputs[0]);graph.replay()
                torch.testing.assert_close(out,ref,rtol=0,atol=0)
        finally:graph.reset()

    def test_gptq_calibration_and_cache_identity(self):
        import tempfile
        from pathlib import Path
        from engine.kernels.dense import pack_w4
        from engine.kernels.dense.store import PackStore
        from engine.kernels.dense.packing import mk_w4_dequant
        torch.manual_seed(993)
        w=(torch.randn(128,128,device='cuda')*.02).bfloat16()
        x=torch.randn(1024,128,device='cuda')
        x[:,1::2] += x[:,::2]*2
        h=x.T@x
        rtn=pack_w4(w)
        with tempfile.TemporaryDirectory() as directory:
            name='DFlash2Qwen3ForCausalLM/model.test'
            path=Path(directory)/'mkcalib/rank0'/(name+'.pt')
            path.parent.mkdir(parents=True)
            torch.save(dict(H=h.cpu(),ntok=1024,name=name),path)
            # Old GPTQ filenames did not bind H and could contain an RTN
            # fallback. An otherwise valid legacy pack must not be trusted.
            import hashlib
            digest=hashlib.sha256(w.view(torch.uint8).cpu().numpy()).hexdigest()
            legacy=Path(directory)/'mkpacks/rank0'/f'sha256-{digest}-128x128-bfloat16-v4-ten-gptq-lr0.pt'
            legacy.parent.mkdir(parents=True)
            torch.save(dict(version=4,shape=(128,128),name=name,wq4=rtn.data.cpu(),
                            ws4=rtn.scale.cpu(),wgs=1.,rgs=None,lr_a=None,lr_b=None),legacy)
            store=PackStore(directory,0)
            packed=store.pack(w,name)
            self.assertEqual(store.stats['legacy_unverified_skipped'],1)
            self.assertEqual(store.stats['built'],1)
            cached=store.pack(w,name)
            self.assertTrue(torch.equal(packed.data,cached.data))
            self.assertEqual(store.stats['cache'],1)
            store.release_pages()
            self.assertTrue(path.is_file())
            self.assertEqual(len(list((Path(directory)/'st-dense-packs').glob('*.pt'))),1)
            def error(p):
                dq=mk_w4_dequant(p.data,p.scale,128,1.,p.rowscale)
                return (x@(w.float()-dq).T).norm()
            self.assertLess(error(packed).item(),error(rtn).item())
            # Changing calibration must produce a new cache, even with the
            # same layer name, shape and weight bytes.
            torch.save(dict(H=torch.eye(128),ntok=128,name=name),path)
            store.pack(w,name)
            self.assertEqual(len(list((Path(directory)/'st-dense-packs').glob('*.pt'))),2)
            torch.save(dict(H=torch.eye(64),ntok=128,name=name),path)
            with self.assertRaisesRegex(ValueError,'incompatible calibration'):
                store.pack(w,name)
            torch.save(dict(H=torch.full((128,128),float('nan')),ntok=128,name=name),path)
            with self.assertRaisesRegex(ValueError,'incompatible calibration'):
                store.pack(w,name)

    def test_decode_prefill_and_replayed_changed_inputs(self):
        from engine.kernels.dense import DenseLinear
        from engine.kernels.dense.packing import mk_w4_dequant, _mk_quant_x_ref
        torch.manual_seed(772)
        for n,k in ((1024,4096),(6416,4096),(4096,2048),(4096,512),(512,8192)):
            weight=(torch.randn(n,k,device="cuda")*.02).bfloat16()
            layer=DenseLinear(weight)
            for m in (1,6,24,33,256,1023,1024):
                x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
                out=layer(x)
                ref=torch.nn.functional.linear(x,weight)
                relative=(out.float()-ref.float()).norm()/ref.float().norm()
                self.assertLess(relative.item(),.05 if 32<m<1024 else .16,(n,k,m,relative.item()))
                if m<=32:
                    parts=[]
                    for i,p in enumerate(layer.packs):
                        w=mk_w4_dequant(p.data,p.scale,n,1.,p.rowscale)
                        parts.append((_mk_quant_x_ref(x[:,i*4096:i*4096+p.cols])@w.T).bfloat16().float())
                    twin=sum(parts).bfloat16()
                    error=(out.float()-twin.float()).norm()/twin.float().norm()
                    self.assertLess(error.item(),.006,(n,k,m,error.item()))
            x=torch.randn(6,k,device="cuda",dtype=torch.bfloat16)
            layer(x)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=layer(x)
            try:
                for _ in range(3):
                    x.normal_();ref=layer(x);graph.replay()
                    self.assertTrue(torch.equal(out,ref))
            finally:graph.reset()


if __name__=="__main__":unittest.main()


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "the W4 lane is a device kernel")
class WidePackFoldTests(unittest.TestCase):
    """A weight wider than one GPTQ tile used to be several packs, summed after the fact (45차 §86).

    Every tile of a wide pack carries the same row shift, so the tiles are one quantisation cut into pieces and
    concatenating them along the k-tile axis is the pack the kernel wants. These pin that -- and that the fold
    does not happen where the shifts disagree, because there the sum is the answer.
    """
    def weight(self, rows, cols, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(rows, cols, device="cuda", generator=gen, dtype=torch.float32).bfloat16() / 64

    def test_a_wide_weight_becomes_one_pack(self):
        from engine.kernels.dense import DenseLinear, TILE
        layer = DenseLinear(self.weight(512, 4 * TILE), prefill=False, name="wide")
        self.assertEqual(len(layer.packs), 1)
        self.assertEqual(layer.packs[0].cols, 4 * TILE)

    def test_the_folded_product_is_the_sum_the_tiles_used_to_make(self):
        """Not bit for bit: the tiled call rounded every tile's output to bf16 before adding it, and the fold
        accumulates the k blocks in fp32 and rounds once. Both approximate the same weight, so that is what
        they are judged against -- and the fold is not the further of the two."""
        from engine.kernels.dense import DenseLinear, W4Pack, w4_gemm, TILE
        source = self.weight(512, 5 * TILE, seed=3)
        layer = DenseLinear(source, prefill=False, name="wide")
        whole = layer.packs[0]
        per = whole.data.shape[1] // 5
        tiles = [W4Pack(whole.data[:, i*per:(i+1)*per].contiguous(), whole.scale[:, i*per:(i+1)*per].contiguous(),
                        whole.rowscale, whole.rows, TILE, whole.calibrated) for i in range(5)]
        closer = further = 0
        for trial in range(4):
            gen = torch.Generator(device="cuda").manual_seed(100 + trial)
            x = torch.randn(6, 5 * TILE, device="cuda", generator=gen, dtype=torch.float32).bfloat16()
            truth = x.float() @ source.float().T
            summed = None
            for i, tile in enumerate(tiles):
                part = w4_gemm(x[:, i*TILE:(i+1)*TILE], tile).float()
                summed = part if summed is None else summed + part
            folded = layer(x).float()
            a = (summed.bfloat16().float() - truth).norm(dim=-1) / truth.norm(dim=-1)
            b = (folded - truth).norm(dim=-1) / truth.norm(dim=-1)
            self.assertLess((b - a).abs().max().item(), 1e-3)       # both live inside the 4-bit error
            closer += int((b < a).sum())
            further += int((b > a).sum())
        self.assertGreaterEqual(closer, further)

    def test_tiles_whose_shifts_disagree_are_left_alone(self):
        from engine.kernels.dense import W4Pack, _fold
        def tile(shift):
            return W4Pack(torch.zeros(4, 32, 128, 64, device="cuda", dtype=torch.uint8),
                          torch.zeros(4, 32, 128, 8, device="cuda", dtype=torch.int8),
                          torch.full((512,), shift, device="cuda"), 512, 4096, True)
        self.assertEqual(len(_fold([tile(1.0), tile(1.0)])), 1)
        self.assertEqual(len(_fold([tile(1.0), tile(0.5)])), 2)

    def test_a_weight_past_the_kernel_s_widest_k_is_not_folded(self):
        from engine.kernels.dense import W4Pack, _fold, KMAX, TILE
        def tile():
            return W4Pack(torch.zeros(1, TILE // 128, 128, 64, device="cuda", dtype=torch.uint8),
                          torch.zeros(1, TILE // 128, 128, 8, device="cuda", dtype=torch.int8),
                          torch.ones(128, device="cuda"), 128, TILE, True)
        self.assertEqual(len(_fold([tile() for _ in range(KMAX // TILE)])), 1)
        self.assertEqual(len(_fold([tile() for _ in range(KMAX // TILE + 1)])), KMAX // TILE + 1)
