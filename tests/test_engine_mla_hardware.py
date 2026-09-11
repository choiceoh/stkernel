"""CPU coverage for hardware-gated MLA dispatch and workspace ownership."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Tensor:
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


if __name__ == "__main__": unittest.main()
