"""CPU coverage for hardware-gated MLA dispatch and workspace ownership."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Tensor:
    is_cuda = True
    def __init__(self, shape, dtype, address, *, size=1, contiguous=True):
        self.shape, self.dtype, self.address = shape, dtype, address
        self.device, self.size, self.contiguous = "cuda:0", size, contiguous

    def data_ptr(self): return self.address
    def is_contiguous(self): return self.contiguous
    def element_size(self): return self.size


class MlaHardwareTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "engine/kernels/mla/__init__.py"
        spec = importlib.util.spec_from_file_location("mla_hardware_test", path)
        self.mla = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mla)
        self.mla.ENABLE_MLA_CLUSTER = True
        self.mla._MLA_CLUSTER_MAX = 8
        self.calls = []
        self.mla._EXT = SimpleNamespace(
            mla_grid=lambda: 96,
            mla_tree_cluster_max=lambda: 8,
            run_mla_cluster=lambda *args: self.calls.append(args),
        )

    def test_qualified_clusters_and_shape_limits(self):
        for t in (32, 36, 40, 42, 48, 54, 60, 63, 64):
            for w in (1, 33, 512, 2048, 2176):
                with self.subTest(t=t,w=w):
                    self.assertTrue(self.mla._mla_uses_cluster(t,w,self.mla.mla_splits(t)))
        for t,w,s in ((0,64,2),(12,2048,8),(16,2048,6),(24,2048,4),
                      (31,2048,3),(65,2048,2),(32,0,3),(32,2177,3),
                      (32,2048,1),(32,2048,4)):
            self.assertFalse(self.mla._mla_uses_cluster(t,w,s))

    def test_capability_and_disable_switch(self):
        self.mla._MLA_CLUSTER_MAX = 0
        self.assertFalse(self.mla._mla_uses_cluster(48,2048,2))
        self.mla._MLA_CLUSTER_MAX = 2
        self.assertTrue(self.mla._mla_uses_cluster(48,2048,2))
        self.assertFalse(self.mla._mla_uses_cluster(32,2048,3))
        self.mla.ENABLE_MLA_CLUSTER = False
        self.assertFalse(self.mla._mla_uses_cluster(48,2048,2))

    def test_cluster_reuses_output_without_allocating_global_scratch(self):
        q=Tensor((48,16,512),"bf16",16,size=2)
        cache=Tensor((4096,512),"u8",32)
        slots=Tensor((48,2048),"i32",48,size=4)
        lens=Tensor((48,),"i32",64,size=4)
        out=Tensor(q.shape,"bf16",80,size=2)
        torch=SimpleNamespace(int32="i32",bfloat16="bf16")
        def forbidden(*args): raise AssertionError("cluster allocated global scratch")
        self.mla._ensure_workspace=forbidden
        self.mla._mla_workspace=forbidden
        with patch.dict(sys.modules,torch=torch):
            self.assertIs(self.mla.mla_decode(q,cache,slots,lens,.0625,.7,out),out)
        self.assertEqual(self.calls,[([16,32,48,64,80],[.0625,.7],[48,2048,2])])

    def test_legacy_shapes_keep_workspace_path(self):
        class LegacySelected(Exception): pass
        def legacy(*args): raise LegacySelected()
        self.mla._ensure_workspace=legacy
        torch=SimpleNamespace(int32="i32",bfloat16="bf16")
        for t in (1,6,12,16,24,65):
            with self.subTest(t=t),patch.dict(sys.modules,torch=torch):
                with self.assertRaises(LegacySelected):
                    self.mla.mla_decode(Tensor((t,16,512),"bf16",16),
                        Tensor((4096,512),"u8",32),Tensor((t,2048),"i32",48),
                        Tensor((t,),"i32",64),.0625,.7)
        self.assertEqual(self.calls,[])

    def test_retained_queries_are_bound_to_ordinary_k7_cells(self):
        self.mla._EXT.run_mla = lambda *args: self.calls.append(args)
        self.mla._ensure_workspace = lambda device: {'barrier_mla': Tensor((8,), 'i32', 144)}
        self.mla._mla_workspace = lambda *args: {'part': Tensor((8,), 'f32', 160), 'pml': Tensor((8,), 'f32', 176)}
        torch = SimpleNamespace(int32='i32', bfloat16='bf16')
        for rows in (1, 8, 16, 24):
            for enabled in (False, True):
                self.mla.ENABLE_MLA_QREG = enabled
                q = Tensor((rows,16,512), 'bf16', 16)
                with patch.dict(sys.modules, torch=torch):
                    self.mla.mla_decode(q, Tensor((4096,512),'u8',32),
                        Tensor((rows,2048),'i32',48), Tensor((rows,),'i32',64),
                        .0625, 1., Tensor(q.shape,'bf16',80))
                self.assertEqual(self.calls[-1][2][-1], 2 if enabled and rows == 16 else 0)
        q = Tensor((16,16,512), 'bf16', 16)
        with patch.dict(sys.modules, torch=torch):
            self.mla.mla_decode(q, Tensor((4096,512),'u8',32),
                Tensor((16,33),'i32',48), Tensor((16,),'i32',64),
                .0625, 1., Tensor(q.shape,'bf16',80))
        self.assertEqual(self.calls[-1][2][-1], 0)

    def test_tree_banks_pass_direct_pointers_with_same_cluster_and_split_plan(self):
        self.mla._EXT.run_mla = lambda *args: self.calls.append(args)
        self.mla._ensure_workspace = lambda device: {'barrier_mla': Tensor((8,), 'i32', 144)}
        self.mla._mla_workspace = lambda *args: {'part': Tensor((8,), 'f32', 160), 'pml': Tensor((8,), 'f32', 176)}
        torch = SimpleNamespace(int32='i32', bfloat16='bf16', float8_e4m3fn='fp8',
                                cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        for rows in (8, 15, 32):
            q, cache = Tensor((rows,16,512),'bf16',16), Tensor((4096,512),'u8',32)
            slots, lens = Tensor((rows,2051),'i32',48), Tensor((rows,),'i32',64)
            out, private = Tensor(q.shape,'bf16',80), Tensor((rows,512),'fp8',96)
            with patch.dict(sys.modules, torch=torch):
                with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
                    if rows in (8, 32):
                        with self.assertRaisesRegex(RuntimeError, 'before graph capture'):
                            self.mla.mla_decode(q,cache,slots,lens,.0625,1.,out,branch=private)
                self.assertIs(self.mla.mla_decode(q,cache,slots,lens,.0625,1.,out,branch=private), out)
                ptrs, _, ints = self.calls[-1]
                self.assertEqual(ptrs, [16,32,48,64,80,96] if rows == 32 else [16,32,48,64,80,160,176,144,96])
                self.assertEqual(ints[:3], [rows,2051,self.mla.mla_splits(rows)])
                for bad in (Tensor((rows,512),'bf16',96), Tensor((rows+1,512),'fp8',96),
                            Tensor((rows,512),'fp8',96,contiguous=False)):
                    with self.assertRaises(ValueError):
                        self.mla.mla_decode(q,cache,slots,lens,.0625,1.,out,branch=bad)

    def test_decode_experiments_never_capture_tree_or_probe_dispatch(self):
        self.mla._EXT.run_mla = lambda *args: self.calls.append(args)
        self.mla._ensure_workspace = lambda device: {'barrier_mla': Tensor((8,), 'i32', 144)}
        self.mla._mla_workspace = lambda *args: {'part': Tensor((8,), 'f32', 160), 'pml': Tensor((8,), 'f32', 176)}
        torch = SimpleNamespace(int32='i32', bfloat16='bf16', float8_e4m3fn='fp8',
                                cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        for materialize in (False, True):
            self.mla.ENABLE_MLA_SYNC_CLEAN = True
            self.mla.ENABLE_MLA_BF16_TILE = materialize
            for rows in (1, 7, 8, 16, 24):
                q = Tensor((rows,16,512), 'bf16', 16)
                for mode in ('ordinary', 'tree', 'probe'):
                    kwargs = ({'branch': Tensor((rows,512), 'fp8', 96)} if mode == 'tree'
                              else {'probe': 1} if mode == 'probe' else {})
                    with patch.dict(sys.modules, torch=torch):
                        self.mla.mla_decode(q, Tensor((4096,512),'u8',32),
                            Tensor((rows,2048),'i32',48), Tensor((rows,),'i32',64),
                            .0625, 1., Tensor(q.shape,'bf16',80), **kwargs)
                    cell = self.calls[-1][2][-1]
                    expected = (8 if materialize else 4 if rows == 8 else 6) if rows in (8, 16) and mode == 'ordinary' else 0
                    self.assertEqual(cell, expected, (materialize, rows, mode))

    def test_tree_uses_own_capacity_before_selecting_cluster(self):
        self.mla._EXT.mla_tree_cluster_max = lambda: 2
        self.mla._EXT.run_mla = lambda *args: self.calls.append(args)
        self.mla._ensure_workspace = lambda device: {'barrier_mla': Tensor((8,), 'i32', 144)}
        self.mla._mla_workspace = lambda *args: {'part': Tensor((8,), 'f32', 160), 'pml': Tensor((8,), 'f32', 176)}
        torch = SimpleNamespace(int32='i32', bfloat16='bf16', float8_e4m3fn='fp8',
                                cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
        q, cache = Tensor((32,16,512),'bf16',16), Tensor((4096,512),'u8',32)
        slots, lens = Tensor((32,2051),'i32',48), Tensor((32,),'i32',64)
        out, private = Tensor(q.shape,'bf16',80), Tensor((32,512),'fp8',96)
        with patch.dict(sys.modules, torch=torch):
            for capturing in (False, True):
                with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=capturing):
                    self.mla.mla_decode(q,cache,slots,lens,.0625,1.,out,branch=private)
                    # Ordinary capacity=8, tree capacity=2: split=3 must use
                    # the global reduction. Never enter a rejected cluster launch.
                    self.assertEqual(self.calls[-1][0], [16,32,48,64,80,160,176,144,96])
                    self.assertEqual(self.calls[-1][2][:3], [32,2051,3])


if __name__ == "__main__": unittest.main()
