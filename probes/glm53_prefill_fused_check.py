#!/usr/bin/env python3
"""Compare fused packet decode/MHC with the mounted production MHC kernels.

Single-GPU arithmetic, continuation, ownership and timing gate. Packet values
are built independently of the transport encoder. Real TP4 exchange is tested
by glm53_prefill_collectives_check.py; this is not a serving benchmark.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics

import torch

from glm53_prefill_transport_check import exact, load_helper


def inputs(helper, rows, case):
    gen = torch.Generator(device="cuda").manual_seed(931 + rows)
    n = rows * 4096
    stride = helper._payload_bytes(n)
    payload = torch.zeros(4 * stride, device="cuda", dtype=torch.uint8)
    common = torch.randn(n, device="cuda", generator=gen)
    for rank in range(4):
        q = torch.randn(n, device="cuda", generator=gen) * 32
        scales = torch.pow(2.0, torch.randint(-8, 4, (n // 2048,),
                                            device="cuda", generator=gen).float())
        if case == "zeros":
            q.zero_()
        elif case == "cancellation":
            q = common * (32, -32, 1, -1)[rank]
            scales.fill_(1 if rank < 2 else 2 ** -10)
            if rank == 3:
                q[::257] += 0.125
        elif case != "random":
            raise ValueError(case)
        packet = payload[rank * stride:(rank + 1) * stride]
        packet[:n].copy_(q.to(torch.float8_e4m3fn).view(torch.uint8))
        packet[n:n + (n // 2048) * 4].view(torch.float32).copy_(scales)
    residual = torch.randn((rows, 4, 4096), device="cuda", generator=gen).bfloat16()
    # Signed, asymmetric coefficients expose transpose and accumulation-order bugs.
    post = torch.randn((rows, 4, 1), device="cuda", generator=gen)
    comb = torch.randn((rows, 4, 4), device="cuda", generator=gen)
    return helper.PackedPrefillRows(payload, rows, stride), residual, post, comb


def unpack(helper, packet):
    x = torch.empty((packet.rows, 4096), device="cuda", dtype=torch.bfloat16)
    helper._unpack_sum_payload[(packet.rows * 2,)](
        packet.payload.view(torch.float8_e4m3fn), packet.payload.view(torch.float32), x,
        LOCAL_N=x.numel(), PAYLOAD_BYTES=packet.payload_bytes, TP=4, BLOCK=2048,
    )
    return x


def timing(old, fused, repeats):
    order = ("old", "fused", "fused", "old")
    lanes = {"old": old, "fused": fused}
    for name in order:
        lanes[name]()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rounds = []
    for _ in range(repeats):
        samples = []
        for name in order:
            start.record()
            result = lanes[name]()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
            del result
        rounds.append(samples)
    medians = {name: statistics.median(statistics.mean(r[j] for j, arm in enumerate(order)
                                                     if arm == name) for r in rounds)
               for name in lanes}
    return {"order": order, "rounds_us": rounds, "median_us": medians,
            "latency_reduction_pct": 100 * (1 - medians["fused"] / medians["old"])}


def check(helper, mhc, rows, case, repeats):
    packet, residual, post, comb = inputs(helper, rows, case)
    originals = [t.clone() for t in (packet.payload, residual, post, comb)]
    x = unpack(helper, packet)
    old = lambda: mhc.mhc_post_tilelang(unpack(helper, packet), residual, post, comb)
    fused = lambda: helper.prefill_mhc_post(packet, residual, post, comb)
    expected = mhc.mhc_post_tilelang(x, residual, post, comb)
    actual = fused()
    exact(actual, expected, f"{rows}/{case}: fused post differs from production MHC")
    exact(helper.prefill_mhc_post(packet, residual, post.squeeze(-1), comb), expected,
          "flat post coefficients differ")
    # The auxiliary-output path may consume the same packet a second time.
    again = fused()
    exact(again, expected, "second packet consumer differs")
    if again.data_ptr() == actual.data_ptr() or actual.data_ptr() == residual.data_ptr():
        raise AssertionError("output storage is shared across consumers")
    stream_outputs = []
    for stream in (torch.cuda.Stream(), torch.cuda.Stream()):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            stream_outputs.append((stream, fused()))
    for stream, output in stream_outputs:
        stream.synchronize()
        exact(output, expected, "nondefault-stream output differs")

    continuation = False
    if case == "random" and rows in (33, 532, 1728):
        gen = torch.Generator(device="cuda").manual_seed(471)
        fn = torch.randn((24, 16384), device="cuda", generator=gen) * 0.01
        scale = torch.tensor([0.05, 0.05, 0.05], device="cuda")
        base = torch.randn(24, device="cuda", generator=gen) * 0.01
        for norm in (None, torch.ones(4096, device="cuda", dtype=torch.bfloat16)):
            kwargs = dict(fn=fn, hc_scale=scale, hc_base=base, rms_eps=1e-6,
                          hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
                          hc_post_mult_value=2.0, sinkhorn_repeat=20,
                          norm_weight=norm, norm_eps=1e-6)
            reference = mhc.mhc_fused_post_pre_tilelang(x, residual, post, comb, **kwargs)
            mapped = fused()
            candidate = (mapped, *mhc.mhc_pre_tilelang(mapped, **kwargs))
            for i, (a, b) in enumerate(zip(candidate, reference, strict=True)):
                exact(a, b, f"{rows}: post/pre continuation output {i} differs")
        continuation = True
    for tensor, before in zip((packet.payload, residual, post, comb), originals, strict=True):
        exact(tensor, before, "input mutated")
    return {"rows_per_rank": rows, "case": case, "post_bit_exact": True,
            "continuation_checked": continuation, "stream_and_ownership_checked": True,
            "timing": timing(old, fused, repeats) if case == "random" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 33, 532, 1024, 1728, 2047, 2048])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows) < 32 or args.repeats < 1:
        parser.error("local rows >=32 and positive repeats required")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("SM121 GPU required")
    helper, source = load_helper(args.source_root)
    sources = {"helper": source}
    for name in ("tilelang", "tilelang_kernels"):
        module = importlib.import_module(f"vllm.model_executor.kernels.mhc.{name}")
        path = Path(module.__file__)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = args.source_root / f"overlay/modules/glm53_kernels/{name}.py"
        if sha != hashlib.sha256(expected.read_bytes()).hexdigest():
            raise AssertionError(f"mounted {name} source mismatch")
        sources[name] = {"path": str(path), "sha256": sha}
    mhc = importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")
    print(json.dumps({"sources": sources}), flush=True)
    for rows in args.rows:
        for case in ("zeros", "random", "cancellation"):
            print(json.dumps(check(helper, mhc, rows, case, args.repeats)), flush=True)
    print("PASS: bit-exact MHC post, continuation, repeated consumers and CUDA streams; no serving claim")


if __name__ == "__main__":
    main()
