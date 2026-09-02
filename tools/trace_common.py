# SPDX-License-Identifier: Apache-2.0
"""Shared loader / step cutter / classifier for torch-profiler decode traces.

Used by tools/trace_step_composition.py and tools/trace_step_timeline.py.
Steps are cut at the first prep kernel of each step; the anchor is looked up
in order because the mounted modules change the launch names:
  _gather_block_tables_kernel   stock V2 runner prep
  _glm53_prep_fused_kernel      glm53_prep_fused (EXP-7) replaces the stock chain
  _get_num_sampled_and_rejected census.py's sampler anchor (one per step)
Busy/idle use the interval UNION across streams, so concurrent kernels on the
shared-expert stream are not double-counted.
"""
from __future__ import annotations

import gzip
import json

STEP_ANCHORS = ("_gather_block_tables_kernel", "_glm53_prep_fused_kernel",
                "_get_num_sampled_and_rejected")


def load_kernel_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        tr = json.loads(fh.read())
    ev = [e for e in tr.get("traceEvents", []) if e.get("cat") == "kernel"
          and "ts" in e and "dur" in e and "name" in e]
    del tr
    ev.sort(key=lambda e: (e["ts"], e["dur"]))
    return ev


def cut_steps(ev: list[dict]) -> tuple[str, list[int]]:
    """(anchor used, indices of the first anchor kernel of every step)."""
    for anchor in STEP_ANCHORS:
        starts = [i for i, e in enumerate(ev) if e["name"].startswith(anchor)]
        if len(starts) >= 2:
            return anchor, starts
    raise SystemExit(f"no step anchor found among {STEP_ANCHORS}")


def union_busy_us(seg: list[dict]) -> float:
    """Length of the union of [ts, ts+dur) over all streams."""
    busy = 0.0
    cur_s = cur_e = None
    for e in sorted(seg, key=lambda e: e["ts"]):
        s, t = e["ts"], e["ts"] + e["dur"]
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, t
        elif t > cur_e:
            cur_e = t
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy


def stream_of(e: dict):
    return e.get("args", {}).get("stream")


def category(n: str) -> str:
    if n.startswith("k_oneshot"):
        return "AR k_oneshot"
    if "ncclDevKernel" in n:
        return "nccl"
    if "moecute" in n or "moe_static" in n:
        return "MoE expert kernel"
    if "deep_gemm::sm120_fp8_fp4_gemm" in n:
        return "deep_gemm fp8/fp4 GEMM"
    if "deep_gemm" in n:
        return "deep_gemm other"
    if ("cutlass_80_wmma" in n or "splitKreduce" in n or "gemmSN" in n
            or n.startswith("_gate_splitk")):
        return "bf16/cublas GEMM + head gate"
    if "fused_recurrent" in n or "conv1d" in n or "layer_norm_gated" in n:
        return "KDA"
    if "BatchMLAPaged" in n or "concat_and_cache_mla" in n:
        return "MLA decode"
    if ("mqa_logits" in n or "topKPerRow" in n or "expand_pools" in n or "kpool" in n.lower()):
        return "indexer"
    if "mhc" in n:
        return "MHC"
    if ("per_token_group_quant" in n or "act_and_mul" in n or "single_group_topk" in n
            or "_deneb_gate" in n):
        return "MoE glue"
    if ("flashinfer::sampling" in n or "_selector_walk" in n or "_cache_draft" in n
            or "kernel_mha" in n):
        return "drafter/sampler"
    if n.startswith("triton_poi") or "at::native" in n or "elementwise" in n:
        return "elementwise glue"
    return "other"


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[i]
