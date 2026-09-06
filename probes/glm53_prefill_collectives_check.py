#!/usr/bin/env python3
"""Offline correctness checks for the actual eager TP4 prefill collectives.

Run BF16 and FP8 in separate processes because the helper latches its knobs.
The launcher must supply torchrun's WORLD_SIZE=4, RANK, LOCAL_RANK and rendezvous
environment. Four hosts with one GPU each use --nnodes=4 --nproc-per-node=1
and a distinct --node-rank on each host; this script does not launch peers.

Example after mounting the candidate overlay, on each of four hosts:
  torchrun --nnodes=4 --nproc-per-node=1 --node-rank=<0..3> \
    --master-addr=<rank-0-address> --master-port=<unused-port> \
    probes/glm53_prefill_collectives_check.py --transport bf16
Repeat with --transport fp8, fp8-v2 and fp8-v3. --source-root names the checkout whose overlay
must exactly match imported files on every rank. Optional --timing compares
isolated eager collectives in balanced order using the slowest rank per
sample. It does not measure model throughput or quality.
"""

import argparse
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
from unittest.mock import patch


SOURCES = {
    "vllm.distributed.device_communicators.glm53_prefill_collectives":
        "overlay/modules/glm53_runtime/glm53_prefill_collectives.py",
    "vllm.distributed.parallel_state":
        "overlay/modules/glm53_runtime/parallel_state.py",
    "vllm.distributed.device_communicators.cuda_communicator":
        "overlay/modules/glm53_runtime/cuda_communicator.py",
}


def _fingerprints(source_root):
    result = {}
    for name, relative in SOURCES.items():
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        expected = source_root / relative
        actual_sha = hashlib.sha256(actual.read_bytes()).hexdigest()
        expected_sha = hashlib.sha256(expected.read_bytes()).hexdigest()
        result[name] = {"path": str(actual), "sha256": actual_sha}
        if actual_sha != expected_sha:
            raise RuntimeError(f"imported source mismatch: {actual} != {expected}")
    probe = Path(__file__).resolve()
    probe_sha = hashlib.sha256(probe.read_bytes()).hexdigest()
    expected = source_root / "probes/glm53_prefill_collectives_check.py"
    if probe_sha != hashlib.sha256(expected.read_bytes()).hexdigest():
        raise RuntimeError("probe source differs from the checkout")
    result["probe"] = {"path": str(probe), "sha256": probe_sha}
    return result


def _require(condition, message, group):
    """Every rank reports the same validation failure before the next call."""
    reports = [None] * 4
    dist.all_gather_object(reports, None if bool(condition) else message,
                           group=group.cpu_group)
    failures = [f"rank {rank}: {item}" for rank, item in enumerate(reports) if item]
    if failures:
        raise AssertionError("; ".join(failures))


def _bits_equal(left, right):
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


def _allreduce(tensor, group, op=None):
    result = tensor.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM if op is None else op,
                    group=group.device_group)
    return result


def _input(rows, rank, case, device):
    generator = torch.Generator(device=device).manual_seed(1701 + rank)
    value = torch.randn((rows, 4096), generator=generator, device=device)
    if case == "zeros":
        value.zero_()
    elif case == "signed_headroom":
        # All four ranks add values on the exact +/-112 headroom boundary.
        value.fill_(112.0)
        value[:, 1::2] = -112.0
    elif case == "cancellation":
        # Large opposite terms and small opposite terms; relative-to-output
        # tolerances are inappropriate when the true sum is near zero.
        common = torch.Generator(device=device).manual_seed(1701)
        value = torch.randn((rows, 4096), generator=common, device=device)
        value *= (1024.0, -1024.0, 0.125, -0.125)[rank]
        if rank == 3:
            value[:, ::257] += 0.0009765625
    elif case == "extreme_finite":
        value.mul_(2.0 ** -20)
        value[:, 0] = (1 if rank % 2 == 0 else -1) * 2.0 ** 60
        value[:, 2048] = (1 if rank < 2 else -1) * 2.0 ** -60
    elif case != "random":
        raise ValueError(case)
    return value.to(torch.bfloat16).contiguous()


def _quant_reference(value, maxima, limit):
    blocks = value.float().view(-1, 2048)
    scales = torch.pow(2.0, torch.ceil(torch.log2(maxima.clamp_min(1.0e-30) / limit)))
    packed = (blocks / scales[:, None]).to(torch.float8_e4m3fn)
    decoded = (packed.float() * scales[:, None]).reshape(value.shape)
    return packed.reshape(value.shape), scales, decoded


def _scope_checks(helper, group, device, rows):
    from vllm.distributed import tensor_model_parallel_all_reduce

    x = _input(rows, group.rank_in_group, "random", device)
    before = x.clone()
    _require(group.use_custom_op_call, "CUDA GroupCoordinator custom op is disabled", group)
    with helper.partial_tp_output(num_tokens=rows):
        partial = tensor_model_parallel_all_reduce(x)
        scope = helper._PARTIAL.get()
        valid = (partial is x and partial.data_ptr() == x.data_ptr()
                 and scope.reductions == 1 and _bits_equal(x, before))
    _require(valid and helper._PARTIAL.get() is None,
             "scoped AR must bypass the custom op and return the original tensor once", group)

    def rejected(which):
        try:
            with helper.partial_tp_output(num_tokens=rows):
                if which == "empty":
                    pass
                elif which == "double":
                    group.all_reduce(x)
                    group.all_reduce(x)
                elif which == "shape":
                    group.all_reduce(x[:, :2048].contiguous())
                elif which == "stride":
                    group.all_reduce(torch.empty((rows, 8192), device=device,
                                                 dtype=torch.bfloat16)[:, ::2])
                elif which == "nested":
                    with helper.partial_tp_output(num_tokens=rows):
                        group.all_reduce(x)
        except RuntimeError:
            return helper._PARTIAL.get() is None
        return False

    for case in ("empty", "double", "shape", "stride", "nested"):
        _require(rejected(case), f"scope failed to reject/reset {case}", group)
    # Another communicator must not consume the TP scope's one reduction.
    with helper.partial_tp_output(num_tokens=rows):
        ignored = helper.maybe_partial_all_reduce(object(), x)
        partial = group.all_reduce(x)
    _require(ignored is None and partial is x, "communicator identity guard failed", group)

    native = _allreduce(x, group)
    outside = group.all_reduce(x)
    sum_abs = _allreduce(x.float().abs(), group)
    gamma = (3.0 / 256.0) / (1.0 - 3.0 / 256.0)
    error = (outside.float() - native.float()).abs()
    _require(outside.data_ptr() != x.data_ptr() and _bits_equal(x, before)
             and bool(torch.all(error <= 2 * gamma * sum_abs + 1.0e-30)),
             "outside-scope AR must produce a fresh sum without mutating input", group)


def _case(helper, group, rows, case, transport, device):
    rank = group.rank_in_group
    local_rows = (rows + 3) // 4
    padded_rows = local_rows * 4
    value = _input(rows, rank, case, device)
    before = value.clone()
    padded = torch.zeros((padded_rows, 4096), device=device, dtype=torch.bfloat16)
    padded[:rows].copy_(value)
    start, stop = rank * local_rows, (rank + 1) * local_rows
    shard = helper.prefill_shard(value)
    _require(tuple(shard.shape) == (local_rows, 4096) and shard.is_contiguous()
             and _bits_equal(shard, padded[start:stop])
             and _bits_equal(value, before),
             f"{rows}/{case}: trailing-zero padded shard mismatch", group)
    if stop <= rows:
        _require(shard.untyped_storage().data_ptr() == value.untyped_storage().data_ptr(),
                 f"{rows}/{case}: full shard should retain the contiguous view", group)
    else:
        _require(shard.untyped_storage().data_ptr() != value.untyped_storage().data_ptr(),
                 f"{rows}/{case}: padded shard must not overwrite source storage", group)
    shard_before = shard.clone()
    native_gather = torch.empty((padded_rows, 4096), device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(native_gather, shard, group=group.device_group)
    gathered = helper.prefill_all_gather(shard, num_tokens=rows)

    if transport == "bf16":
        gather_expected = native_gather
    else:
        local_max = shard.float().view(-1, 2048).abs().amax(dim=1)
        _, _, local_decoded = _quant_reference(shard, local_max, 448.0)
        gather_expected = torch.empty_like(native_gather)
        dist.all_gather_into_tensor(gather_expected, local_decoded.to(torch.bfloat16),
                                   group=group.device_group)
    _require(tuple(gathered.shape) == (rows, 4096) and gathered.is_contiguous()
             and _bits_equal(gathered, gather_expected[:rows]) and _bits_equal(shard, shard_before)
             and gathered.data_ptr() != shard.data_ptr(),
             f"{rows}/{case}: gather value/layout/input-preservation mismatch", group)
    if case == "zeros":
        untrimmed = helper.prefill_all_gather(shard)
        _require(tuple(untrimmed.shape) == (padded_rows, 4096)
                 and _bits_equal(untrimmed, gather_expected),
                 f"{rows}/{case}: optional untrimmed gather contract mismatch", group)

    native_sum = _allreduce(padded, group)
    sum_abs = _allreduce(padded.float().abs(), group)
    native_local = native_sum[start:stop]
    bf16_gamma = (3.0 / 256.0) / (1.0 - 3.0 / 256.0)
    quant_error = torch.zeros_like(sum_abs)
    reference = native_local.float()
    bound = 2 * bf16_gamma * sum_abs[start:stop] + 1.0e-30

    if transport == "fp8":
        maxima = padded.float().view(-1, 2048).abs().amax(dim=1)
        _require(torch.equal(helper._maxima(value, padded_numel=padded.numel()), maxima),
                 f"{rows}/{case}: block maxima mismatch", group)
        common_max = _allreduce(maxima, group, dist.ReduceOp.MAX)
        q_ref, scales_ref, decoded = _quant_reference(padded, common_max, 112.0)
        q, scales = helper._quantize(value, common_max, 112.0, num_rows=padded_rows)
        max_scales = _allreduce(scales, group, dist.ReduceOp.MAX)
        min_scales = _allreduce(scales, group, dist.ReduceOp.MIN)
        _require(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
                 and torch.equal(scales, scales_ref)
                 and torch.equal(max_scales, min_scales)
                 and bool(torch.all(q.float().abs() <= 112.0)),
                 f"{rows}/{case}: quantization/common scales/signed headroom mismatch", group)
        unpacked = torch.empty_like(padded)
        helper._decode[(scales.numel(),)](
            q, scales, unpacked, N=padded.numel(), BLOCK=2048,
        )
        _require(_bits_equal(unpacked, decoded.to(torch.bfloat16)),
                 f"{rows}/{case}: explicit decode mismatch", group)
        # A reference shared by all ranks, independent of native FP8 SUM's
        # reduction tree. gamma_3 bounds three rounded FP8 additions with
        # u=1/16. The extra BF16 term accounts for the final output cast.
        reference_full = _allreduce(decoded, group)
        encoded_abs_sum = _allreduce(decoded.abs(), group)
        quant_error = _allreduce((decoded - padded.float()).abs(), group)
        fp8_gamma = (3.0 / 16.0) / (1.0 - 3.0 / 16.0)
        bound = ((fp8_gamma + 1.0 / 255.0) * encoded_abs_sum[start:stop]
                 + 1.0e-30)
        reference = reference_full[start:stop]

    if transport in ("fp8-v2", "fp8-v3"):
        maxima = padded.float().view(-1, 2048).abs().amax(dim=1)
        _, _, decoded = _quant_reference(padded, maxima, 448.0)
        peers = [torch.empty_like(decoded) for _ in range(4)]
        dist.all_gather(peers, decoded, group=group.device_group)
        # Match the documented source-rank order, independently of the
        # transport's packet layout and of NCCL's native reduction tree.
        reference_full = torch.zeros_like(decoded)
        for peer in peers:
            reference_full.add_(peer)
        reference = reference_full[start:stop].to(torch.bfloat16).float()
        bound = torch.zeros_like(reference)

    reduced = helper.prefill_reduce_scatter(value)
    error = (reduced.float() - reference).abs()
    _require(tuple(reduced.shape) == (local_rows, 4096)
             and reduced.dtype == torch.bfloat16 and reduced.is_contiguous()
             and reduced.data_ptr() != value.data_ptr()
             and _bits_equal(value, before) and bool(torch.isfinite(reduced).all())
             and bool(torch.all(error <= bound)),
             f"{rows}/{case}: reduce-scatter/reference/bound/input mismatch", group)
    if case in ("zeros", "signed_headroom"):
        _require(_bits_equal(reduced, native_local),
                 f"{rows}/{case}: exact representable sum changed", group)
    native_error = (reduced.float() - native_local.float()).abs()
    if transport == "fp8":
        total_bound = (bound + quant_error[start:stop]
                       + bf16_gamma * sum_abs[start:stop])
        _require(bool(torch.all(native_error <= total_bound)),
                 f"{rows}/{case}: error exceeds quantization plus reduction bound", group)
    magnitude = sum_abs[start:stop].clamp_min(1.0e-30)
    return {
        "rows": rows, "case": case, "rank": rank,
        "max_abs_error_vs_native_bf16": native_error.max().item(),
        "max_error_over_sum_abs_inputs": (native_error / magnitude).max().item(),
        "max_abs_error_vs_transport_reference": error.max().item(),
    }


def _timing(helper, group, rows, device, repeats):
    """Probe-only mode overrides; every rank follows the same mirrored order."""
    value = _input(rows, group.rank_in_group, "random", device)
    shard = helper.prefill_shard(value)
    order = ("bf16", "fp8-v2", "fp8-v3", "fp8-v3", "fp8-v2", "bf16")

    def lane(mode):
        # The serving helper normally latches these on import. Isolate the
        # timing controls here; restore them before any other probe checks.
        return patch.multiple(helper, _FP8=mode != "bf16", _FP8_V2=mode == "fp8-v2",
                              _FP8_V3=mode == "fp8-v3", _FP8_V3_MIN_TOKENS=0)

    def invoke(operation):
        if operation == "all_gather":
            return helper.prefill_all_gather(shard, num_tokens=rows)
        return helper.prefill_reduce_scatter(value)

    result = []
    for operation in ("all_gather", "reduce_scatter"):
        for mode in order:
            with lane(mode):
                invoke(operation)
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        local_rounds = []
        for _ in range(repeats):
            samples = []
            for mode in order:
                with lane(mode):
                    dist.barrier(group=group.device_group)
                    start.record()
                    output = invoke(operation)
                    end.record()
                    end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
                del output
            local_rounds.append(samples)
        peers = [None] * 4
        dist.all_gather_object(peers, local_rounds, group=group.cpu_group)
        # A TP operation finishes only when its slowest rank does. Preserve
        # per-round pairs instead of comparing independent fastest samples.
        rounds = [[max(peer[r][j] for peer in peers) for j in range(len(order))]
                  for r in range(repeats)]
        per_arm = {mode: [statistics.mean([row[j] for j, name in enumerate(order) if name == mode])
                          for row in rounds] for mode in set(order)}
        result.append({"rows": rows, "operation": operation, "order": order,
                       "max_rank_rounds_us": rounds,
                       "median_us": {mode: statistics.median(samples) for mode, samples in per_arm.items()},
                       "v3_speedup_pct": {
                           baseline: statistics.median(100 * (a / b - 1)
                                                       for a, b in zip(per_arm[baseline], per_arm["fp8-v3"]))
                           for baseline in ("bf16", "fp8-v2")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("bf16", "fp8", "fp8-v2", "fp8-v3"), required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 129, 512, 8185, 8192])
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing", action="store_true",
                        help="balanced BF16/v2/v3 isolated collective timings after correctness")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fp8-min-tokens", type=int, default=0,
                        help="v3 dispatch threshold; default 0 measures the raw codec control")
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if world != 4:
        parser.error("real WORLD_SIZE=4 torchrun environment is required; no local emulation")
    if any(rows < 128 for rows in args.rows):
        parser.error("rows must be >=128")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.fp8_min_tokens < 0:
        parser.error("fp8 minimum tokens must be nonnegative")
    os.environ["VLLM_GLM53_PREFILL_SP"] = "1"
    os.environ["VLLM_GLM53_PREFILL_SP_FP8"] = {
        "bf16": "0", "fp8": "1", "fp8-v2": "2", "fp8-v3": "3",
    }[args.transport]
    os.environ["VLLM_GLM53_PREFILL_SP_FP8_MIN_TOKENS"] = str(args.fp8_min_tokens)
    # Keep the independent reference out of optional one-shot/custom AR arms.
    os.environ["VLLM_DSV4_ONESHOT_AR"] = "0"
    global torch, dist
    import torch
    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.device_communicators import glm53_prefill_collectives as helper

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (12, 1):
        raise RuntimeError("this pinned GLM candidate probe requires SM121")
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=4, pipeline_parallel_size=1,
        distributed_executor_backend="external_launcher",
    ))
    with set_current_vllm_config(config):
        ps.set_custom_all_reduce(False)
        try:
            ps.init_distributed_environment(
                world_size=4, rank=rank, local_rank=local_rank,
                distributed_init_method="env://", backend="nccl",
                timeout=timedelta(minutes=5),
            )
            ps.initialize_model_parallel(tensor_model_parallel_size=4,
                                         pipeline_model_parallel_size=1)
            group = get_tp_group()
            fingerprints, identity_error = {}, None
            try:
                fingerprints = _fingerprints(args.source_root.resolve())
            except Exception as exc:
                identity_error = str(exc)
            _require(identity_error is None, identity_error or "", group)
            peer_identity = [None] * 4
            dist.all_gather_object(peer_identity,
                                   ({name: item["sha256"] for name, item in fingerprints.items()},
                                    args.transport, args.rows, args.timing, args.repeats,
                                    args.fp8_min_tokens), group=group.cpu_group)
            _require(all(item == peer_identity[0] for item in peer_identity),
                     "ranks disagree on source/transport/row cases", group)
            comm = group.device_communicator
            _require(comm is not None and comm.pynccl_comm is not None
                     and not comm.pynccl_comm.disabled,
                     "active native PyNCCL communicator required", group)
            results = []
            for rows in args.rows:
                _scope_checks(helper, group, device, rows)
                reference_transport = ("bf16" if args.transport == "fp8-v3"
                                       and rows < args.fp8_min_tokens else args.transport)
                for case in ("zeros", "random", "cancellation", "extreme_finite", "signed_headroom"):
                    result = _case(helper, group, rows, case, reference_transport, device)
                    result["effective_transport"] = reference_transport
                    results.append(result)
            torch.cuda.synchronize(device)
            peer_results = [None] * 4
            dist.all_gather_object(peer_results, results, group=group.cpu_group)
            timings = []
            if args.timing:
                for rows in args.rows:
                    timings.extend(_timing(helper, group, rows, device, args.repeats))
            if rank == 0:
                report = {"status": "passed", "transport": args.transport,
                          "fp8_min_tokens": args.fp8_min_tokens,
                          "world_size": 4, "sources": fingerprints,
                          "checks": peer_results,
                          "timings": timings,
                          "scope": "eager collective numerics and optional isolated timing; no model quality/throughput claim"}
                rendered = json.dumps(report, indent=2)
                print(rendered, flush=True)
                if args.output:
                    args.output.write_text(rendered + "\n")
        finally:
            ps.destroy_model_parallel()
            ps.destroy_distributed_environment()


if __name__ == "__main__":
    main()
