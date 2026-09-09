"""Actual-source host dispatch with a byte/view oracle, not device numerics.

The tensor model implements FP32->BF16 round-to-nearest-even explicitly. It
checks which bytes the real dispatch copies and owns, not Torch/CUDA lowering.
"""
from dataclasses import dataclass
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_glm53_ep_micro_scatter import load


@dataclass(frozen=True)
class Device:
    type: str = "cuda"
    index: int = 0


def bf16(bits):
    # Canonical quiet NaN; both old/new paths perform this conversion once.
    if bits & 0x7fffffff > 0x7f800000:
        return 0x7fc0
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


class Tensor:
    next_pointer = 0x100000

    def __init__(self, shape, dtype="bf16", device=Device(), *, events=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.events = events if events is not None else []
        self.contiguous = True
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 0x100000
        self.storage = [0] * self.numel()
        self.offset = 0
        self.owner = self
        self.reads = set()
        self.fail_copy = False

    def numel(self): return math.prod(self.shape)
    def size(self, dim): return self.shape[dim]
    def element_size(self): return {"bf16": 2, "f32": 4, "i32": 4}[self.dtype]
    def data_ptr(self): return self.pointer + self.offset * self.element_size()
    def is_contiguous(self): return self.contiguous

    def _view(self, shape, offset=0):
        view = object.__new__(Tensor)
        view.__dict__ = self.__dict__.copy()
        view.shape = tuple(shape)
        view.offset += offset
        return view

    def view(self, *shape):
        return self._view((self.numel(),) if shape == (-1,) else shape)

    def __getitem__(self, key):
        start, stop, step = key.indices(self.shape[0])
        assert step == 1
        stride = math.prod(self.shape[1:])
        return self._view((max(stop-start, 0), *self.shape[1:]), start * stride)

    def to(self, dtype):
        assert dtype == self.dtype, "fixtures use existing F32 metadata, no hidden cast"
        return self

    def record_stream(self, stream):
        self.events.append(("record", stream, self.data_ptr()))

    def copy_(self, source):
        if self.fail_copy:
            raise RuntimeError("copy failed")
        assert self.shape == source.shape
        self.events.append(("copy", self, source))
        values = source.storage[source.offset:source.offset + source.numel()]
        source.reads.update(range(source.offset, source.offset + source.numel()))
        if source.dtype == "f32" and self.dtype == "bf16":
            values = [bf16(value) for value in values]
        else:
            assert source.dtype == self.dtype
        self.storage[self.offset:self.offset+self.numel()] = values
        return self


class StaticWorkspace(SimpleNamespace):
    pass


class DynamicWorkspace(SimpleNamespace):
    pass


class Harness:
    def __init__(self):
        self.events = []
        self.stream = "default"
        self.failure = None
        self.kernel_args = []
        self.fixture = [0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
                        0x00000000, 0x80000000, 0x00008000, 0x00018000,
                        0x007fffff, 0x7f800000, 0xff800000, 0x7fc12345,
                        0xff812345, 0x3f7fffff, 0x3f810000, 0x00800000]
        def tensor(shape, dtype="bf16"):
            return Tensor(shape, dtype, events=self.events)
        self.tensor = tensor
        self.ws = StaticWorkspace(state_E=72,max_rows=64,device=Device(),dm_barrier_count=None)
        for name in ("compact_topk_ids", "packed_a_view", "packed_input_scale", "packed_a_flat",
                     "scale_flat", "barrier_count", "barrier_epoch", "row_counts",
                     "active_expert_count", "weight_expert_ids", "global_to_local_expert",
                     "token_map", "token_weights"):
            setattr(self.ws, name, tensor((72,), "i32"))
        self.ws.ep_micro_scatter_fp32 = tensor((8,4096), "f32")
        self.physical = tensor((8,4096))
        self.physical.storage[:] = [0x4321] * self.physical.numel()
        self.target = tensor((6,4096))
        self.target.storage[:] = [0x1234] * self.target.numel()
        self.weights = SimpleNamespace(tiled=False,w13_fp4=tensor((1,)),down_fp4=tensor((1,)),
                                       w1_alpha=tensor((72,),"f32"),w2_alpha=tensor((72,),"f32"))
        self.torch = SimpleNamespace(float32="f32",bfloat16="bf16",int32="i32",
            device=lambda value:Device(),cuda=SimpleNamespace(current_stream=lambda d:self.stream))
        self.ns = dict(__package__="ep_t6_test",torch=self.torch,
            _normalize_activation_precision=lambda x:x,_normalize_quant_mode=lambda *x:x[0],
            _normalize_source_format_for_quant_mode=lambda x,q:x,
            _activation_precision_from_quant_mode=lambda x:"fp4",
            _check_memref_limit=lambda *a:None,_expand_to_experts=lambda x,n:x,
            _FORCED_BACKEND=None,_GLM53_B12X_FORCE_BACKEND=None,
            _MICRO_SHARE_INPUT_ACROSS_EXPERTS=False,_MICRO_MAX_TOKENS=8,
            _DIRECT_MICRO_CUTOVER_PAIRS=0,_DIRECT_MICRO_MAX_N=1024,
            _MICRO_COMPACT_CUTOVER_PAIRS=40,_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK=40,
            _B12X_EP_ZERO_WEIGHT_MICRO=True,
            _b12x_ep_zero_weight_micro_expert_id=lambda **kw:72,
            get_num_sm=lambda _:48,get_max_active_clusters=lambda _:48,
            _STATIC_MAC_LADDER=(),_GLM53_B12X_STATIC_MAC_LADDER=None,
            _MICRO_MAC_LADDER=(),_GLM53_B12X_MICRO_MAC_LADDER=None,
            _lookup_mac_ladder=lambda *a:None,_scale_runtime_addresses=lambda *a,**kw:(1,2),
            _get_micro_kernel=lambda *a,**kw:(self.compiled,48),
            Sm120StaticMoEWorkspace=StaticWorkspace,Sm120DynamicMoEWorkspace=DynamicWorkspace,
            is_gated_activation=lambda x:True,_LEVEL_TILE_N=128,
            static_v2_weights_layout=lambda **kw:False,
            static_v2_weights_reform_sf_pack=lambda **kw:False,
            static_v2_weights_sf_pack=lambda **kw:False)
        for name in ("_ep_micro_scatter_fp32","_ep_micro_scatter_buffer",
                     "_validate_ep_micro_short_output","launch_sm120_static_moe","launch_sm120_moe"):
            load(name,self.ns)
        self.args = dict(workspace=self.ws,weights=self.weights,a=tensor((8,4096)),
            topk_ids=tensor((8,8),"i32"),topk_weights=tensor((8,8),"f32"),
            input_gs=tensor((72,),"f32"),down_input_scale=tensor((72,),"f32"),
            scatter_output=self.physical,num_experts=72,num_tokens=8,k=4096,n=2048,top_k=8,
            activation="swigluoai_uninterleave",swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,
            _ep_short_output=self.target)

    def compiled(self,*args):
        self.kernel_args.append(args)
        self.events.append(("launch",self.stream,args[21].data_ptr()))
        if self.failure:
            raise RuntimeError(self.failure)
        plane = args[21]
        plane.storage[:] = [self.fixture[i % len(self.fixture)] for i in range(plane.numel())]

    def run(self, *, unified=False):
        module = SimpleNamespace(compact_topk_ids=lambda *a:self.events.append(("compact",)))
        with patch.dict(sys.modules,{"ep_t6_test.triton_compact":module}):
            if not unified:
                return self.ns["launch_sm120_static_moe"](**self.args)
            args = {key:self.args[key] for key in ("a","topk_ids","topk_weights","top_k",
                "num_experts","scatter_output","activation","swiglu_alpha","swiglu_beta",
                "swiglu_limit","_ep_short_output") if key in self.args}
            raw_weight = self.tensor((1,))
            raw_weight.shape = (72,4096,2048)  # metadata only; no weight allocation
            args.update(w1_weight=raw_weight,w1_weight_sf=None,
                w1_alpha=self.weights.w1_alpha,w2_weight=None,w2_weight_sf=None,
                w2_alpha=self.weights.w2_alpha,num_local_experts=72,
                fc2_input_scale=self.args["down_input_scale"],_workspace=self.ws,
                _weight_views=self.weights,quant_mode="nvfp4")
            return self.ns["launch_sm120_moe"](**args)


class DirectOutputTests(unittest.TestCase):
    def test_exact_source_uses_same_m8_plane_and_one_six_row_cast_on_current_stream(self):
        h = Harness(); plane = h.ws.ep_micro_scatter_fp32
        for stream in ("default","side","capture-replay"):
            h.stream = stream; h.events.clear(); plane.reads.clear()
            self.assertIs(h.run(unified=True),h.target)
            self.assertIs(h.ws.ep_micro_scatter_fp32,plane)
            self.assertIs(h.kernel_args[-1][21],plane)
            self.assertEqual(h.kernel_args[-1][21].shape,(8,4096))
            self.assertEqual([event[0] for event in h.events],["record","compact","launch","copy"])
            self.assertEqual(h.events[0],("record",stream,plane.data_ptr()))
            self.assertEqual(h.events[2],("launch",stream,plane.data_ptr()))
            _,dst,src = h.events[-1]
            self.assertIs(dst,h.target); self.assertIs(src.owner,plane)
            self.assertEqual(src.shape,(6,4096)); self.assertEqual(src.data_ptr(),plane.data_ptr())
            self.assertEqual(plane.reads,set(range(6*4096)))
            self.assertEqual(h.physical.storage,[0x4321]*(8*4096))
            self.assertEqual(h.target.storage,[bf16(value) for value in plane.storage[:6*4096]])

    def test_rounding_bytes_match_previous_two_copies_including_ties_and_specials(self):
        h = Harness(); h.run()
        previous = h.tensor((8,4096)); previous.copy_(h.ws.ep_micro_scatter_fp32)
        caller = h.tensor((6,4096)); caller.copy_(previous[:6])
        self.assertEqual(h.target.storage,caller.storage)
        expected = [0x3f80,0x3f82,0xbf80,0xbf82,0,0x8000,0,2,
                    0x80,0x7f80,0xff80,0x7fc0,0x7fc0,0x3f80,0x3f81,0x80]
        self.assertEqual(h.target.storage[:len(expected)],expected)

    def test_omitted_override_preserves_full_output_identity_and_copy(self):
        h = Harness(); del h.args["_ep_short_output"]
        self.assertIs(h.run(unified=True),h.physical)
        self.assertEqual(h.events[-1][1:],(h.physical,h.ws.ep_micro_scatter_fp32))
        self.assertEqual(h.ws.ep_micro_scatter_fp32.reads,set(range(8*4096)))
        self.assertEqual(h.target.storage,[0x1234]*(6*4096))

    def test_kernel_or_final_copy_error_propagates_without_wrapper_fallback(self):
        for at in ("kernel","copy"):
            h = Harness()
            if at == "kernel": h.failure = "kernel failed"
            else: h.target.fail_copy = True
            with self.assertRaisesRegex(RuntimeError,at+" failed"): h.run()
            self.assertEqual(h.target.storage,[0x1234]*(6*4096))
            self.assertEqual(h.physical.storage,[0x4321]*(8*4096))
            self.assertFalse(any(event[0] == "copy" for event in h.events))

    def test_geometry_metadata_and_forced_modes_fail_before_any_device_work(self):
        changes = {"num_experts":288,"num_tokens":6,"k":4095,"n":2047,"top_k":1,
                   "activation":"silu","swiglu_alpha":1.1,"swiglu_beta":1.,
                   "swiglu_limit":9.,"quant_mode":"mxfp4"}
        for field,value in changes.items():
            h=Harness(); h.args[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):h.run()
            self.assertEqual(h.events,[])
        for name,value in (("_FORCED_BACKEND","micro"),("_FORCED_BACKEND","static"),
                           ("_B12X_EP_ZERO_WEIGHT_MICRO",False)):
            h=Harness();h.ns[name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):h.run()
            self.assertEqual(h.events,[])
        for obj,field,value in (("ws","state_E",288),("ws","max_rows",8),
                               ("weights","tiled",True),("target","shape",(8,4096)),
                               ("target","dtype","f32"),("target","contiguous",False),
                               ("target","device",Device("cpu"))):
            h=Harness();setattr(getattr(h,obj),field,value)
            with self.subTest(obj=obj,field=field),self.assertRaises(ValueError):h.run()
            self.assertEqual(h.events,[])

    def test_input_plane_shape_device_and_layout_mutations_refuse_without_replacement(self):
        for name in ("a","topk_ids","topk_weights","scatter_output","plane"):
            for field,value in (("shape",(7,7)),("device",Device(index=1)),("contiguous",False)):
                h=Harness();plane=h.ws.ep_micro_scatter_fp32
                target=plane if name == "plane" else h.args[name]
                setattr(target,field,value)
                with self.subTest(name=name,field=field),self.assertRaises(ValueError):h.run()
                self.assertEqual(h.events,[]);self.assertIs(h.ws.ep_micro_scatter_fp32,plane)
        for value in (None,Tensor((8,4096),"bf16")):
            h=Harness();h.ws.ep_micro_scatter_fp32=value
            with self.assertRaises(ValueError):h.run()
            self.assertEqual(h.events,[]);self.assertIs(h.ws.ep_micro_scatter_fp32,value)

    def test_all_alias_ranges_reject_but_adjacent_allocations_and_target_view_succeed(self):
        for name in ("a","topk_ids","topk_weights","scatter_output","plane"):
            for offset in (0,2,-2):
                h=Harness();source=h.ws.ep_micro_scatter_fp32 if name == "plane" else h.args[name]
                h.target.pointer=source.data_ptr()+offset
                with self.subTest(name=name,offset=offset),self.assertRaisesRegex(ValueError,"aliases"):
                    h.run()
                self.assertEqual(h.events,[])
        h=Harness();source=h.physical
        h.target.pointer=source.data_ptr()+source.numel()*source.element_size()
        self.assertIs(h.run(),h.target)
        h=Harness();whole=h.tensor((8,4096));whole.storage[:]=[0x4567]*whole.numel()
        h.target=whole[1:7];h.args["_ep_short_output"]=h.target
        self.assertIs(h.run(),h.target)
        self.assertEqual(whole.storage[:4096]+whole.storage[-4096:],[0x4567]*8192)

    def test_unified_override_refuses_missing_dynamic_workspace_and_non_nvfp4_source(self):
        h=Harness();launch=h.ns["launch_sm120_moe"]
        required=dict(a=None,topk_ids=None,topk_weights=None,w1_weight=None,w1_weight_sf=None,
            w1_alpha=None,w2_weight=None,w2_weight_sf=None,w2_alpha=None,num_experts=72,
            top_k=8,num_local_experts=72,scatter_output=None,_ep_short_output=h.target,
            quant_mode="nvfp4",source_format="modelopt",_workspace=h.ws)
        for change in ({"_workspace":None},{"_workspace":DynamicWorkspace()},
                       {"quant_mode":"w4a16"},{"quant_mode":"mxfp4"},
                       {"source_format":"other"}):
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,"explicit NVFP4"):
                launch(**(required|change))
        self.assertEqual(h.events,[])

    def test_lost_selected_plane_refuses_compiled_launch(self):
        h=Harness();wrong=h.tensor((8,4096),"f32")
        h.ns["_ep_micro_scatter_buffer"]=lambda *a:wrong
        with self.assertRaisesRegex(RuntimeError,"lost its FP32"):h.run()
        self.assertEqual(h.kernel_args,[])
        self.assertEqual(h.events,[("compact",)])
        self.assertEqual(h.target.storage,[0x1234]*(6*4096))


if __name__ == "__main__":
    unittest.main()
