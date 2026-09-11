"""Qwen3.8-Flash-Next's budget lines (profile). Constants live here, not in base."""
from __future__ import annotations

from engine.base.budget import (GIB, MEASURED, LEDGER, DECLARED, ESTIMATED, READ,
                                Line, Budget, probe_box, host_box)
from engine.profiles.qwen38.plan import resident_gib

# Measured on qwen38 itself, 2026-09-11, HF Qwen4ExpTextDecoderLayer built on cuda
# with uninitialised weights and run at 1K/2K/4K tokens (shapes, not values):
#   GDN layer 0:  0.289 / 0.286 / 0.284 GiB per 1K -- linear.
#   QSA layer 3:  0.290 / 0.526 / 0.999 GiB per 1K -- QUADRATIC, because HF eager
#     materialises T x T index scores before top-k. That is the oracle's cost,
#     not a kernel's: vLLM's Triton qsa_pre_indexer selects in-kernel. Until the
#     engine's QSA module exists and is measured, the line takes the GDN number
#     and this comment is the debt.
ACTIVATION_GIB_PER_1K = 0.286
RESIDUAL_BYTES_PER_TOKEN = 4 * 2560 * 2          # hc_count x hidden, bf16

# Not measured on qwen38. GLM's load-model minus weights (8.77 GiB, with
# expandable_segments on) stays as an UPPER BOUND; qwen38 has no pack step.
CONSTRUCTION_UPPER_GIB = 8.77
RUNTIME_FLOOR_GIB = 5.54                         # GLM 40th boot table; re-measure
OS_RESERVE_MULTIPLE = 2.0


def budget(chunk: int = 4096, box_gib: "float | None" = None, tenants_gib: float = 0.0) -> Budget:
    host_total, _ = host_box()
    if box_gib is None:
        box_gib, _free = probe_box()
    floor = host_total * 0.05
    weights = resident_gib()
    lines = []
    if tenants_gib:
        lines.append(Line("other tenants on this box", tenants_gib, MEASURED,
                          "mem_get_info total-free before we allocate"))
    lines += [
        Line("reserve for the OS", floor * OS_RESERVE_MULTIPLE, DECLARED,
             f"{OS_RESERVE_MULTIPLE:g}x earlyoom's 5% floor ({floor:.2f} GiB)"),
        Line("runtime floor (CUDA ctx + NCCL)", RUNTIME_FLOOR_GIB, LEDGER,
             "GLM 40th boot table -- re-measure on qwen38"),
        Line("weights (this rank, TEP=4)", weights, READ,
             "plan.py: census x pinned placement rules; PLE 11.92 of it is the D1 line"),
        Line("allocator slack", weights * 0.001, MEASURED, "expandable_segments:True -> 0.1%"),
        Line("module construction (cuBLAS, init)", CONSTRUCTION_UPPER_GIB, ESTIMATED,
             "UPPER BOUND from GLM; qwen38 has no pack step"),
        Line(f"activation @ chunk {chunk:,}",
             chunk / 1024 * ACTIVATION_GIB_PER_1K + chunk * RESIDUAL_BYTES_PER_TOKEN / GIB, MEASURED,
             "GDN layer 0.286 GiB/1K linear (HF layer, shapes only); QSA kernel-bounded pending"),
    ]
    return Budget(box_gib, lines, label=f"Qwen3.8-Flash-Next, one rank of TEP=4, chunk {chunk:,}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--exclusive", action="store_true")
    a = ap.parse_args()
    total, free = probe_box()
    b = budget(a.chunk, total, 0.0 if a.exclusive else total - free)
    print(b.table()); print(); print("  " + b.verdict())
    if b.kv_gib > 0:
        from engine.profiles.qwen38.plan import max_context, text_config
        cfg = text_config()
        print(f"\n  what {b.kv_gib:.2f} GiB of KV buys (ceiling {cfg['max_position_embeddings']:,}):")
        for c in (1, 8, 32, 128):
            print(f"    concurrency {c:>3}: {max_context(b.kv_gib, c, cfg):>9,} tok")
