"""Dependency-free regression checks for eager TP4 collective contracts."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


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


class CollectiveContractTests(unittest.TestCase):
    def setUp(self):
        names = {"_PartialOutput", "_tp_comm", "partial_tp_output",
                 "maybe_partial_all_reduce", "_check", "_payload_bytes"}
        tree = ast.parse(SOURCE.read_text())
        body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
        self.ns = dict(contextmanager=contextmanager, ContextVar=ContextVar, dataclass=dataclass,
                       _PARTIAL=ContextVar("test_partial", default=None),
                       _ENABLED=True, _TP=4, _HIDDEN=4096, _BLOCK=2048,
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
