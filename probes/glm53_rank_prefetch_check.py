#!/usr/bin/env python3
"""Exact rank-cache GPU restoration on another stream; run under fleet holder."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import torch
from vllm.model_executor.layers import glm53_rank_cache as rank
from vllm.model_executor.layers.glm53_startup_cache import HostStaging


def main():
    torch.set_num_threads(1)
    chunk_bytes = rank.CHUNK_BYTES
    size = 8 * chunk_bytes + 17
    block = bytearray(os.urandom(chunk_bytes))
    owner = torch.empty(size + 31, dtype=torch.uint8, device="cuda")
    target = owner[31:]
    bf16 = torch.empty(32, 64, dtype=torch.bfloat16, device="cuda")
    state = {"weight": target, "alias": target, "bf16": bf16}
    stream = torch.cuda.Stream()
    report = {"checks": 0, "bytes": size + bf16.numel() * 2, "samples": []}
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
    report["ok"] = True
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
