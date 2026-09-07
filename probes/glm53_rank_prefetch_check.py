#!/usr/bin/env python3
"""Exact rank-cache GPU restoration on another stream; run under fleet holder."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import torch
from vllm.model_executor.layers import glm53_rank_cache as rank
from vllm.model_executor.layers.glm53_startup_cache import HostStaging, digest_json


def main():
    global rank
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--module", type=Path, help="private candidate module; serving imports stay unchanged")
    parser.add_argument("--full-cache", type=Path, help="read a real payload into a bounded reusable GPU destination")
    args = parser.parse_args()
    if args.module:
        spec = importlib.util.spec_from_file_location("rank_prefetch_candidate", args.module)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        rank = candidate
    torch.set_num_threads(1)
    chunk_bytes = rank.CHUNK_BYTES
    size = 8 * chunk_bytes + 17
    block = bytearray(os.urandom(chunk_bytes))
    owner = torch.empty(size + 31, dtype=torch.uint8, device="cuda")
    target = owner[31:]
    bf16 = torch.empty(32, 64, dtype=torch.bfloat16, device="cuda")
    state = {"weight": target, "alias": target, "bf16": bf16}
    stream = torch.cuda.Stream()
    report = {"checks": 0, "bytes": size + bf16.numel() * 2, "samples": [],
              "module_sha256": hashlib.sha256(Path(rank.__file__).read_bytes()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="rank-prefetch-gpu-") as temp:
        root = Path(temp)
        chunks = []
        with (root / "weights.bin").open("wb") as out:
            for start in range(0, size, chunk_bytes):
                # Distinct chunks catch lookahead/result association errors,
                # rather than accepting a shifted copy of repeated data.
                block[:8] = start.to_bytes(8, "little")
                data = memoryview(block)[:min(chunk_bytes, size - start)]
                chunks.append({"name": "weight", "start": start, "offset": out.tell(),
                               "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                out.write(data)
                data.release()
            extra = torch.arange(2048).reshape(32, 64).bfloat16().view(torch.uint8).numpy().tobytes()
            chunks.append({"name": "bf16", "start": 0, "offset": out.tell(),
                           "size": len(extra), "sha256": hashlib.sha256(extra).hexdigest()})
            out.write(extra)
            out.flush()
            os.fsync(out.fileno())
            rank._drop_file_pages(out.fileno(), 0, out.tell())
        del block, extra
        manifest = {"size": report["bytes"], "chunks": chunks, "loaded": ["weight", "bf16"]}
        checker = HostStaging()
        for prefetch in (0, 1, 1, 0):
            os.environ["VLLM_GLM53_RANK_CACHE_PREFETCH"] = str(prefetch)
            with torch.cuda.stream(stream):
                target.fill_(165)
                bf16.fill_(-7)
            stream.synchronize()
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                loaded = rank._restore(root, manifest, state)
            elapsed = time.perf_counter() - started
            assert loaded == {"weight", "bf16"}
            assert state["alias"].data_ptr() == target.data_ptr() == owner.data_ptr() + 31
            with torch.cuda.stream(stream):
                for chunk in chunks:
                    raw = state[chunk["name"]].reshape(-1).view(torch.uint8)
                    selected = raw[chunk["start"]:chunk["start"] + chunk["size"]]
                    assert checker.digest(selected) == chunk["sha256"]
                    report["checks"] += 1
            report["samples"].append({"prefetch": prefetch, "restore_s": elapsed})
        # A corrupt second chunk must never reach its target or the fallback.
        with (root / "weights.bin").open("r+b") as out:
            out.seek(chunk_bytes + 3)
            value = out.read(1)
            out.seek(chunk_bytes + 3)
            out.write(bytes([value[0] ^ 1]))
        for prefetch in (0, 1):
            os.environ["VLLM_GLM53_RANK_CACHE_PREFETCH"] = str(prefetch)
            with torch.cuda.stream(stream):
                target.fill_(165)
                try:
                    rank._restore(root, manifest, state)
                except RuntimeError as exc:
                    assert "checksum mismatch" in str(exc), exc
                else:
                    raise AssertionError("corrupt chunk was accepted")
                assert torch.all(target[chunk_bytes:2 * chunk_bytes] == 165).item()
            report["checks"] += 1
        stream.synchronize()
    report["median_restore_s"] = {str(p): statistics.median(
        row["restore_s"] for row in report["samples"] if row["prefetch"] == p) for p in (0, 1)}
    if args.full_cache:
        path = max(args.full_cache.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime_ns)
        envelope = json.loads(path.read_text())
        original = envelope["manifest"]
        assert digest_json(original) == envelope["sha256"]
        # Transport-only probe: reuse one 64 MiB destination, retaining each
        # real chunk's file offset/size/hash. No full model allocation or edits
        # to the existing cache. Full boot/alias correctness is a separate gate.
        manifest = {"size": original["size"], "loaded": ["weight"],
                    "chunks": [dict(c, name="weight", start=0) for c in original["chunks"]]}
        full_target = torch.empty(max(c["size"] for c in manifest["chunks"]), dtype=torch.uint8, device="cuda")
        samples = []
        for prefetch in (0, 1, 1, 0):
            os.environ["VLLM_GLM53_RANK_CACHE_PREFETCH"] = str(prefetch)
            with (path.parent / "weights.bin").open("rb") as source:
                rank._drop_file_pages(source.fileno(), 0, manifest["size"])
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                rank._restore(path.parent, manifest, {"weight": full_target})
            elapsed = time.perf_counter() - started
            last = manifest["chunks"][-1]
            with torch.cuda.stream(stream):
                assert checker.digest(full_target[:last["size"]]) == last["sha256"]
            report["checks"] += 1
            samples.append({"prefetch": prefetch, "restore_s": elapsed})
        report["full_payload"] = {"cache_manifest": str(path), "bytes": manifest["size"],
                                  "chunks": len(manifest["chunks"]), "samples": samples,
                                  "median_restore_s": {str(p): statistics.median(
                                      r["restore_s"] for r in samples if r["prefetch"] == p) for p in (0, 1)}}
    report["ok"] = True
    encoded = json.dumps(report, sort_keys=True)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
