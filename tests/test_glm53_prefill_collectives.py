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


class CollectiveContractTests(unittest.TestCase):
    def setUp(self):
        names = {"_PartialOutput", "_tp_comm", "partial_tp_output",
                 "maybe_partial_all_reduce", "_check", "_payload_bytes", "_use_fp8",
                 "prefill_all_gather", "prefill_reduce_scatter"}
        tree = ast.parse(SOURCE.read_text())
        body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
        self.ns = dict(contextmanager=contextmanager, ContextVar=ContextVar, dataclass=dataclass,
                       _PARTIAL=ContextVar("test_partial", default=None),
                       _ENABLED=True, _TP=4, _HIDDEN=4096, _BLOCK=2048,
                       _FP8=True, _FP8_V2=False, _FP8_V3=True, _FP8_V3_MIN_TOKENS=4096,
                       logger=SimpleNamespace(info_once=Mock()),
                       triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
                       torch=SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
                           is_current_stream_capturing=lambda: False)))
        exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), self.ns)
        # The device communicator intentionally has no .device. Only PyNCCL
        # offers the (out, input) API used by the candidate collectives.
        self.pynccl = SimpleNamespace(device="cuda:0", disabled=False)
        self.device_comm = SimpleNamespace(pynccl_comm=self.pynccl)
        self.group = SimpleNamespace(world_size=4, device_communicator=self.device_comm)
        distributed = ModuleType("vllm.distributed")
        distributed.get_tp_group = lambda: self.group
        modules = patch.dict(sys.modules, {"vllm": ModuleType("vllm"),
                                           "vllm.distributed": distributed})
        modules.start()
        self.addCleanup(modules.stop)

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
