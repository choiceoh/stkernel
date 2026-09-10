"""What loading DSv4.1 weights costs ABOVE the weights themselves.

`engine/budget.py` carries one estimated line it cannot justify: load scratch,
taken as GLM-5.3's ratio (load-model 59.17 GiB against 50.4 GiB of weights =
17.4%). That number came from a different model, a different loader, and a
different quantization path. This measures it on dsv41's real rank file.

GB10 makes one variant suspicious a priori: CPU and CUDA tensors live in the
SAME physical pool, so `safe_open(device="cpu")` followed by `.to("cuda")`
holds both copies at once. On a discrete GPU that is a host page and a device
page; here it is 2x the box.

One point per process, because a caching allocator that has already grown does
not report the peak a fresh boot would see. Fit peak = fixed + slope * bytes
across sizes, the same shape as the 40th campaign's step-cost fit.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

GIB = 1 << 30


def header(path: Path) -> dict:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    raw.pop("__metadata__", None)
    return raw


def keys_for_layers(head: dict, layers: "list[int]") -> "list[str]":
    want = {f"layers.{i}." for i in layers}
    return sorted(k for k in head if any(k.startswith(w) for w in want))


def bytes_of(head: dict, keys: "list[str]") -> int:
    return sum(head[k]["data_offsets"][1] - head[k]["data_offsets"][0] for k in keys)


def run(path: Path, layers: "list[int]", mode: str) -> dict:
    import torch
    from safetensors import safe_open

    head = header(path)
    keys = keys_for_layers(head, layers)
    if not keys:
        raise SystemExit(f"no tensors for layers {layers}")
    want = bytes_of(head, keys)

    torch.cuda.init()
    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    held = {}
    if mode == "cuda":
        with safe_open(str(path), framework="pt", device="cuda") as f:
            for k in keys:
                held[k] = f.get_tensor(k)
    elif mode == "cpu":
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in keys:
                held[k] = f.get_tensor(k).to("cuda", non_blocking=False)
    elif mode == "cpu-stream":
        # release each staging tensor before taking the next -- the difference
        # between this and `cpu` is exactly what holding the whole slice costs.
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in keys:
                staging = f.get_tensor(k)
                held[k] = staging.to("cuda", non_blocking=False)
                del staging
    else:
        raise SystemExit(f"unknown mode {mode}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    free_after, _ = torch.cuda.mem_get_info()
    peak_alloc = torch.cuda.max_memory_allocated()
    resident_alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    return {
        "mode": mode,
        "layers": layers,
        "tensors": len(keys),
        "want_gib": want / GIB,
        "box_delta_gib": (free_before - free_after) / GIB,
        "torch_peak_gib": peak_alloc / GIB,
        "torch_resident_gib": resident_alloc / GIB,
        "torch_reserved_gib": reserved / GIB,
        "scratch_gib": (free_before - free_after) / GIB - want / GIB,
        "seconds": elapsed,
        "gib_per_s": (want / GIB) / elapsed if elapsed else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank-file",
                        default="/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4/rank0of4.safetensors")
    parser.add_argument("--layers", default="0",
                        help="comma list or a-b range of layer indices")
    parser.add_argument("--mode", default="cuda", choices=("cuda", "cpu", "cpu-stream"))
    parser.add_argument("--list", action="store_true", help="print layer sizes and exit")
    args = parser.parse_args()

    path = Path(args.rank_file)
    if args.list:
        head = header(path)
        sizes = {}
        for k, v in head.items():
            m = re.match(r"layers\.(\d+)\.", k)
            if m:
                i = int(m.group(1))
                sizes[i] = sizes.get(i, 0) + v["data_offsets"][1] - v["data_offsets"][0]
        for i in sorted(sizes):
            print(f"  layer {i:>3}  {sizes[i] / GIB:8.3f} GiB")
        print(f"  {'total':>9}  {sum(sizes.values()) / GIB:8.3f} GiB over {len(sizes)} layers")
        return 0

    if "-" in args.layers:
        lo, hi = args.layers.split("-")
        layers = list(range(int(lo), int(hi) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",") if x != ""]
    print(json.dumps(run(path, layers, args.mode)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
