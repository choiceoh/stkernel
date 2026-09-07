"""Dependency-free regression checks for eager TP4 collective contracts."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = (Path(__file__).resolve().parents[1]
          / "overlay/modules/glm53_runtime/glm53_prefill_collectives.py")


class Tensor:
    def __init__(self, rows=128, width=4096, dtype="bf16", device="cuda:0", contiguous=True):
        self.ndim = 2
        self.shape = (rows, width)
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith("cuda:")
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return self.shape[0] * self.shape[1]

    def __getitem__(self, index):
        return Tensor(rows=len(range(*index.indices(self.shape[0]))),
                      width=self.shape[1], dtype=self.dtype, device=self.device)


class ByteBuffer:
    def __init__(self, size, ptr=1024, device="cuda:0", dtype="u8"):
        self.shape, self.ptr = (size,), ptr
        self.dtype, self.device = dtype, device
        self.is_cuda = device.startswith("cuda:")

    def data_ptr(self):
        return self.ptr

    def is_contiguous(self):
        return True

    def __getitem__(self, index):
        return ByteBuffer(index.stop - index.start, self.ptr + index.start,
                          self.device, self.dtype)


class CollectiveContractTests(unittest.TestCase):
    def setUp(self):
        names = {"_PartialOutput", "_tp_comm", "partial_tp_output",
                 "maybe_partial_all_reduce", "_check", "_payload_bytes", "_use_fp8",
                 "prefill_all_gather", "prefill_reduce_scatter", "_exchange_packets",
                 "PackedPrefillRows", "is_packed_prefill", "prefill_mhc_post"}
        tree = ast.parse(SOURCE.read_text())
        body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
        self.ns = dict(contextmanager=contextmanager, ContextVar=ContextVar, dataclass=dataclass,
                       _PARTIAL=ContextVar("test_partial", default=None),
                       _ENABLED=True, _TP=4, _HIDDEN=4096, _BLOCK=2048,
                       _FP8=True, _FP8_V2=False, _FP8_V3=True, _FP8_V3_MIN_TOKENS=4096,
                       _FP8_AG_MIN_TOKENS=-1, _FP8_RS_MIN_TOKENS=-1,
                       _FUSE_MHC=False, _DIRECT_NCCL=False,
                       logger=SimpleNamespace(info_once=Mock()),
                       triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
                       torch=SimpleNamespace(bfloat16="bf16", uint8="u8", float32="f32",
                                             cuda=SimpleNamespace(
                           is_current_stream_capturing=lambda: False,
                           current_stream=lambda: "producer-stream")))
        exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), self.ns)
        # The device communicator intentionally has no .device. Only PyNCCL
        # offers the (out, input) API used by the candidate collectives.
        self.pynccl = SimpleNamespace(device="cuda:0", disabled=False, world_size=4)
        self.device_comm = SimpleNamespace(pynccl_comm=self.pynccl)
        self.group = SimpleNamespace(world_size=4, device_communicator=self.device_comm)
        distributed = ModuleType("vllm.distributed")
        distributed.get_tp_group = lambda: self.group
        modules = patch.dict(sys.modules, {"vllm": ModuleType("vllm"),
                                           "vllm.distributed": distributed})
        modules.start()
        self.addCleanup(modules.stop)

    def test_independent_gates_inherit_and_choose_different_transports(self):
        choose = self.ns["_use_fp8"]
        self.ns.update(_FP8_AG_MIN_TOKENS=6144, _FP8_RS_MIN_TOKENS=2048)
        self.assertFalse(choose(4096))
        self.assertTrue(choose(4096, reduce_scatter=True))
        self.assertTrue(choose(6144))
        self.assertFalse(choose(2047, reduce_scatter=True))
        self.ns.update(_FP8_V3_MIN_TOKENS=0, _FP8_AG_MIN_TOKENS=-1)
        self.assertTrue(choose(128))
        self.assertFalse(choose(128, reduce_scatter=True))

    def test_deferred_unpack_requires_explicit_mhc_consumer_and_eligible_v3(self):
        self.ns["torch"].empty = Mock(return_value=Tensor())
        dispatch = self.ns["_reduce_scatter_v3"] = Mock(return_value="packet")
        self.ns["_FUSE_MHC"] = True
        self.assertEqual(self.ns["prefill_reduce_scatter"](
            Tensor(rows=4097), defer_mhc_post=True), "packet")
        self.assertIsNone(dispatch.call_args.args[1])
        self.ns["torch"].empty.assert_not_called()
        self.ns["prefill_reduce_scatter"](Tensor(rows=4097))
        self.assertIsNotNone(dispatch.call_args.args[1])
        self.pynccl.reduce_scatter = Mock()
        self.ns["prefill_reduce_scatter"](Tensor(rows=2128), defer_mhc_post=True)
        self.pynccl.reduce_scatter.assert_called_once()

    def test_direct_exchange_groups_all_peers_on_producer_stream(self):
        events = []
        self.ns["_DIRECT_NCCL"] = True
        self.pynccl.group_start = lambda: events.append("start")
        self.pynccl.group_end = lambda: events.append("end")
        self.pynccl.send = lambda t, p, s: events.append(("send", t.ptr, t.shape, p, s))
        self.pynccl.recv = lambda t, p, s: events.append(("recv", t.ptr, t.shape, p, s))
        self.ns["_exchange_packets"](ByteBuffer(2048, 4096), ByteBuffer(2048), 512)
        expected = ["start"]
        for peer in range(4):
            expected += [("send", 1024 + 512 * peer, (512,), peer, "producer-stream"),
                         ("recv", 4096 + 512 * peer, (512,), peer, "producer-stream")]
        self.assertEqual(events, expected + ["end"])
        events.clear()
        self.pynccl.send = Mock(side_effect=RuntimeError("NCCL failure"))
        with self.assertRaisesRegex(RuntimeError, "NCCL failure"):
            self.ns["_exchange_packets"](ByteBuffer(2048, 4096), ByteBuffer(2048), 512)
        self.assertEqual(events, ["start", "end"])

    def test_direct_exchange_rejects_bad_buffers_before_opening_group(self):
        self.ns["_DIRECT_NCCL"] = True
        self.pynccl.group_start = Mock()
        for recv in (ByteBuffer(2048), ByteBuffer(1024, 4096),
                     ByteBuffer(2048, 4096, device="cpu"), ByteBuffer(2048, 4096, dtype="bf16")):
            with self.subTest(recv=vars(recv)), self.assertRaises(ValueError):
                self.ns["_exchange_packets"](recv, ByteBuffer(2048), 512)
        self.pynccl.group_start.assert_not_called()

    def test_packed_object_does_not_enter_model_custom_ops(self):
        path = SOURCE.parents[1] / "glm53_model/glm5next_model.py"
        tree = ast.parse(path.read_text())
        layer = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "Glm5NextDecoderLayer")
        methods = [n for n in layer.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("hc_post", "hc_fused_post_pre")]
        ns = dict(self.ns, _PREFILL_SP_ENABLED=True)
        ns["torch"].Tensor = Tensor
        mapped = object()
        ns["prefill_mhc_post"] = Mock(return_value=mapped)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), ns)
        fake = SimpleNamespace(hc_pre=Mock(return_value=("post", "comb", "input")),
                               mhc_post_op=Mock(return_value="plain"),
                               mhc_fused_post_pre_op=Mock(return_value="plain"),
                               rms_norm_eps=1e-6, hc_eps=1e-6, mhc_post_mult_value=2,
                               mhc_sinkhorn_iterations=20)
        packet = ns["PackedPrefillRows"](ByteBuffer(2048), 32, 512)
        for _ in range(2):  # auxiliary and next-layer consumers retain the packet
            self.assertIs(ns["hc_post"](fake, packet, "r", "p", "c"), mapped)
        result = ns["hc_fused_post_pre"](fake, packet, "r", "p", "c", "fn", "s", "b")
        self.assertEqual(result, (mapped, "post", "comb", "input"))
        fake.mhc_post_op.assert_not_called()
        fake.mhc_fused_post_pre_op.assert_not_called()
        self.assertEqual(ns["hc_post"](fake, Tensor(), "r", "p", "c"), "plain")

    def test_short_chunks_take_native_collectives_without_codec_allocations(self):
        allocations = []

        def empty(shape, *, device, dtype):
            allocations.append(shape)
            return Tensor(rows=shape[0], width=shape[1], dtype=dtype, device=device)

        self.ns["torch"].empty = empty
        self.pynccl.all_gather = Mock()
        self.pynccl.reduce_scatter = Mock()
        pad_calls = []
        self.ns["_copy_pad"] = { (8192,): lambda *a, **kw: pad_calls.append(kw) }
        # Both equal-shard 2K and the padding boundary must stay BF16. No
        # FP8 dtype/kernels exist in this fake: entering the codec fails.
        for rows in (128, 2128, 4095):
            with self.subTest(rows=rows):
                allocations.clear()
                local_rows = (rows + 3) // 4
                shard = Tensor(rows=local_rows)
                gathered = self.ns["prefill_all_gather"](shard, num_tokens=rows)
                reduced = self.ns["prefill_reduce_scatter"](Tensor(rows=rows))
                self.assertEqual(gathered.shape, (rows, 4096))
                self.assertEqual(reduced.shape, (local_rows, 4096))
                self.assertEqual(len(allocations), 3 if rows == 4095 else 2)
                self.assertIs(self.pynccl.all_gather.call_args.args[1], shard)
        self.assertEqual(self.pynccl.reduce_scatter.call_count, 3)
        self.assertEqual(len(pad_calls), 1)
        self.assertEqual(pad_calls[0]["N"], 4095 * 4096)

    def test_v3_large_chunks_and_zero_threshold_keep_packed_transport(self):
        self.ns["torch"].empty = lambda shape, **kw: Tensor(rows=shape[0], width=shape[1])
        packed = self.ns["_reduce_scatter_v3"] = Mock(return_value="packed")
        for rows in (4096, 4097, 6912):
            self.assertEqual(self.ns["prefill_reduce_scatter"](Tensor(rows=rows)), "packed")
            self.assertEqual(packed.call_args.args[2], ((rows + 3) // 4) * 4)
        self.ns["_FP8_V3_MIN_TOKENS"] = 0
        self.assertEqual(self.ns["prefill_reduce_scatter"](Tensor(rows=2128)), "packed")

    def test_gate_uses_global_real_rows_and_preserves_other_modes(self):
        choose = self.ns["_use_fp8"]
        for rows, expected in ((2128, False), (4093, False), (4095, False),
                               (4096, True), (4097, True), (6912, True)):
            self.assertIs(choose(rows), expected)
        self.ns["_FP8"] = False
        self.assertFalse(choose(128559))
        self.ns["_FP8"] = True
        self.ns["_FP8_V3"] = False
        self.assertTrue(choose(128))

    def test_buffer_collectives_use_pynccl(self):
        self.assertIs(self.ns["_check"](Tensor()), self.pynccl)
        self.assertEqual(self.ns["_tp_comm"](), (self.device_comm, self.pynccl))

    def test_packet_stride_aligns_every_peer_without_excess_padding(self):
        for rows in (32, 33, 34, 35, 128, 1728, 2047, 2048):
            with self.subTest(rows=rows):
                size = self.ns["_payload_bytes"](rows * 4096)
                self.assertGreaterEqual(size, rows * 4104)
                self.assertLess(size - rows * 4104, 128)
                for rank in range(4):
                    self.assertEqual((512 + rank * size) % 128, 0)

    def test_rejects_wrong_tensor_contract(self):
        for tensor in (Tensor(rows=31), Tensor(width=2048), Tensor(dtype="fp32"),
                       Tensor(device="cpu"), Tensor(device="cuda:1"), Tensor(contiguous=False)):
            with self.subTest(tensor=vars(tensor)), self.assertRaises(ValueError):
                self.ns["_check"](tensor)

    def test_scope_identity_is_device_communicator(self):
        tensor = Tensor()
        defer = self.ns["maybe_partial_all_reduce"]
        with self.ns["partial_tp_output"](num_tokens=128):
            self.assertIsNone(defer(self.pynccl, tensor))
            self.assertIsNone(defer(object(), tensor))
            self.assertIs(defer(self.device_comm, tensor), tensor)
        self.assertIsNone(self.ns["_PARTIAL"].get())
        self.assertIsNone(defer(self.device_comm, tensor))

    def test_empty_double_and_nested_scopes_reset(self):
        for case in ("empty", "double", "nested", "wrong_shape", "exception"):
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                with self.ns["partial_tp_output"](num_tokens=128):
                    if case == "double":
                        for _ in range(2):
                            self.ns["maybe_partial_all_reduce"](self.device_comm, Tensor())
                    elif case == "nested":
                        with self.ns["partial_tp_output"](num_tokens=128):
                            pass
                    elif case == "wrong_shape":
                        self.ns["maybe_partial_all_reduce"](self.device_comm, Tensor(rows=129))
                    elif case == "exception":
                        raise RuntimeError("attention failed")
            self.assertIsNone(self.ns["_PARTIAL"].get())

    def test_communicator_eligibility_is_checked_before_use(self):
        for case in ("disabled", "wrong_tp", "missing_device", "missing_pynccl", "graph", "disarmed"):
            with self.subTest(case=case):
                self.group.world_size = 3 if case == "wrong_tp" else 4
                self.group.device_communicator = None if case == "missing_device" else self.device_comm
                self.device_comm.pynccl_comm = None if case == "missing_pynccl" else self.pynccl
                self.pynccl.disabled = case == "disabled"
                self.ns["_ENABLED"] = case != "disarmed"
                self.ns["torch"].cuda.is_current_stream_capturing = lambda: case == "graph"
                with self.assertRaises(RuntimeError):
                    self.ns["_check"](Tensor())


if __name__ == "__main__":
    unittest.main()
