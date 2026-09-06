#!/usr/bin/env python3
"""Check the packed FP8 transport kernels without a model boot.

--compile-only uses an explicit SM121 target and needs no GPU. Otherwise four
rank packets are built on ONE GPU, exchanged locally, and compared bitwise
against v2 and an ordered FP32 reference. This checks packing, padding and
the dispatch's one-call contract; it does not validate real NCCL or serving
speed/quality. Use glm53_prefill_collectives_check.py for the real TP4 gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton


def load_helper(root):
    from vllm.distributed.device_communicators import glm53_prefill_collectives as helper

    source = Path(helper.__file__)
    expected = root / "overlay/modules/glm53_runtime/glm53_prefill_collectives.py"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != hashlib.sha256(expected.read_bytes()).hexdigest():
        raise AssertionError("imported prefill collective source differs from checkout")
    return helper, {"path": str(source), "sha256": sha}


def compile_kernels(helper):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from vllm.model_executor.layers import glm53_nvfp4_scale as scale

    compiled = []
    for function, signature, constants in (
        (helper._pack_rs_payload,
         {"X": "*bf16", "Packed": "*fp8e4nv", "Scales": "*fp32", "N": "i32",
          "LOCAL_N": "i32", "PAYLOAD_BYTES": "i32"}, {"BLOCK": 2048}),
        (helper._unpack_sum_payload,
         {"Packed": "*fp8e4nv", "Scales": "*fp32", "Out": "*bf16",
          "LOCAL_N": "i32", "PAYLOAD_BYTES": "i32"}, {"TP": 4, "BLOCK": 2048}),
        (scale._partials, {"X": "*bf16", "Partial": "*fp32", "N": "i32"},
         {"BLOCK": 8192}),
        (scale._finish,
         {"Partial": "*fp32", "WeightScale": "*fp32", "GlobalScale": "*fp32",
          "Alpha": "*fp32", "COUNT": "i32"}, {"BLOCK": 4096, "ALPHA_SCALE": 1.0}),
    ):
        kernel = triton.compile(ASTSource(function, signature, constexprs=constants),
                                target=GPUTarget("cuda", 121, 32),
                                options={"num_warps": 4, "enable_fp_fusion": function is not scale._finish})
        compiled.append({"kernel": function.__name__, "hash": kernel.hash,
                         "shared_bytes": kernel.metadata.shared})
    return compiled


def exact(actual, expected, label):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not torch.equal(actual.contiguous().view(torch.uint8),
                               expected.contiguous().view(torch.uint8))):
        raise AssertionError(label)


def run_case(helper, rows, case):
    from vllm import distributed

    local_rows = (rows + 3) // 4
    local_n = local_rows * 4096
    local_blocks = local_n // 2048
    blocks = local_blocks * 4
    n_padded = local_n * 4
    packet_bytes = helper._payload_bytes(local_n)
    values, old_data, old_scales, packets = [], [], [], []
    generator = torch.Generator(device="cuda").manual_seed(401)
    common = torch.randn((rows, 4096), device="cuda", generator=generator)
    for rank in range(4):
        x = torch.randn((rows, 4096), device="cuda", generator=generator)
        if case == "zeros":
            x.zero_()
        elif case == "cancellation":
            x = common * (1024.0, -1024.0, 0.125, -0.125)[rank]
            if rank == 3:
                x[:, ::257] += 0.0009765625
        elif case == "extreme_finite":
            x *= 2.0 ** -20
            x[:, 0] = (-1 if rank % 2 else 1) * 2.0 ** 60
            x[:, 2048] = 2.0 ** -60
        x = x.to(torch.bfloat16)
        before = x.clone()
        data = torch.empty(n_padded, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty(blocks, device="cuda", dtype=torch.float32)
        packet = torch.full((4 * packet_bytes,), 0xA5, device="cuda", dtype=torch.uint8)
        helper._pack_rs[(blocks,)](x, data, scales, N=x.numel(), OUT_N=n_padded, BLOCK=2048)
        helper._pack_rs_payload[(blocks,)](
            x, packet.view(torch.float8_e4m3fn), packet.view(torch.float32),
            N=x.numel(), LOCAL_N=local_n, PAYLOAD_BYTES=packet_bytes, BLOCK=2048,
        )
        # Verify every byte, including scales and all padded rows, on every
        # destination packet. Differing rank scales catch misplaced metadata.
        for dest in range(4):
            received = packet[dest * packet_bytes:(dest + 1) * packet_bytes]
            exact(received[:local_n], data[dest * local_n:(dest + 1) * local_n].view(torch.uint8),
                  f"{rows}/{case}: rank {rank}->{dest} values")
            scale_end = local_n + local_blocks * 4
            exact(received[local_n:scale_end], scales[dest * local_blocks:(dest + 1) * local_blocks].view(torch.uint8),
                  f"{rows}/{case}: rank {rank}->{dest} scales")
            exact(received[scale_end:], torch.zeros_like(received[scale_end:]),
                  f"{rows}/{case}: rank {rank}->{dest} alignment gap not initialized")
        exact(x, before, "pack changed input")
        values.append(x)
        old_data.append(data)
        old_scales.append(scales)
        packets.append(packet)

    for dest in range(4):
        recv = torch.cat([packet[dest * packet_bytes:(dest + 1) * packet_bytes]
                          for packet in packets])
        data = torch.cat([item[dest * local_n:(dest + 1) * local_n].view(torch.uint8)
                          for item in old_data]).view(torch.float8_e4m3fn)
        scales = torch.cat([item[dest * local_blocks:(dest + 1) * local_blocks]
                            for item in old_scales])
        out = torch.empty((local_rows, 4096), device="cuda", dtype=torch.bfloat16)
        old_out = torch.empty_like(out)
        helper._unpack_sum[(local_blocks,)](
            data, scales, old_out, LOCAL_N=local_n, NUM_BLOCKS=local_blocks, TP=4, BLOCK=2048,
        )
        helper._unpack_sum_payload[(local_blocks,)](
            recv.view(torch.float8_e4m3fn), recv.view(torch.float32), out,
            LOCAL_N=local_n, PAYLOAD_BYTES=packet_bytes, TP=4, BLOCK=2048,
        )
        reference = torch.zeros((local_blocks, 2048), device="cuda", dtype=torch.float32)
        for rank in range(4):
            reference.add_(data[rank * local_n:(rank + 1) * local_n].float().view(-1, 2048)
                           * scales[rank * local_blocks:(rank + 1) * local_blocks, None])
        exact(out, old_out, f"{rows}/{case}: v3 differs from v2 at rank {dest}")
        exact(out, reference.view(out.shape).to(torch.bfloat16), "ordered FP32 sum differs")
        if not torch.isfinite(out).all().item():
            raise AssertionError("nonfinite result")
        if dest == 3 and rows % 4:
            exact(out[-(4 * local_rows - rows):], torch.zeros_like(out[-(4 * local_rows - rows):]),
                  "nonzero padded output")

        sentinel_group = object()

        def exchange(actual_recv, actual_send, *, group):
            if group is not sentinel_group:
                raise AssertionError("wrong process group")
            exact(actual_send, packets[dest], "dispatch pack differs")
            actual_recv.copy_(recv)

        dispatched = torch.empty_like(out)
        with patch.object(distributed, "get_tp_group",
                          return_value=SimpleNamespace(device_group=sentinel_group)), \
                patch.object(torch.distributed, "all_to_all_single", side_effect=exchange) as call:
            result = helper._reduce_scatter_v3(values[dest], dispatched, 4 * local_rows)
        if result is not dispatched or call.call_count != 1:
            raise AssertionError("dispatch must use one all-to-all and return its output")
        exact(result, out, "full dispatch differs from isolated kernels")
    return {"rows": rows, "case": case, "packet_bytes": packet_bytes,
            "ranks_simulated": 4, "exact_v2_and_reference": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 129, 130, 131, 512, 8185, 8192])
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if any(rows < 128 for rows in args.rows):
        parser.error("rows must be >=128")
    helper, source = load_helper(args.source_root.resolve())
    print(json.dumps({"source": source, "triton": triton.__version__}), flush=True)
    if args.compile_only:
        print(json.dumps({"compiled": compile_kernels(helper), "gpu_execution": False}), flush=True)
        return
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("SM121 GPU required")
    for rows in args.rows:
        for case in ("zeros", "random", "cancellation", "extreme_finite"):
            print(json.dumps(run_case(helper, rows, case)), flush=True)
    print("PASS: packed v3 values, scales, padding, FP32 sum and one-call dispatch; NCCL simulated")


if __name__ == "__main__":
    main()
