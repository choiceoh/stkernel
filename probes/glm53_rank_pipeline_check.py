#!/usr/bin/env python3
"""Exact-byte real-CUDA gate for pre-finalization rank-cache transport only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "overlay/modules/glm53_model"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULES / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    assert torch.cuda.is_available(), "real CUDA required"
    common = module("rank_probe_common", "glm53_startup_cache.py")
    with patch.dict(sys.modules, {"vllm.model_executor.layers.glm53_startup_cache": common}):
        rank = module("rank_probe", "glm53_rank_cache.py")
    chunk = rank.CHUNK_BYTES
    # Several full 64 MiB chunks, dtype boundaries, shared aliases, and tails.
    a = torch.arange(3 * chunk + 37, dtype=torch.uint8)
    b = torch.arange(chunk // 2 + 19, dtype=torch.int32).to(torch.bfloat16)
    expected = {"weight": a[11:-7], "alias": a[11:-7], "bf16": b,
                "scalar": torch.tensor(3.25), "empty": torch.empty(0)}
    report = dict(cuda=str(torch.version.cuda), torch=torch.__version__,
                  device=torch.cuda.get_device_name(), source_sha256={
                      p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (MODULES/"glm53_rank_cache.py", MODULES/"glm53_startup_cache.py")}, runs=[])
    with tempfile.TemporaryDirectory(prefix="rank-pipeline-gpu-") as temporary:
        artifact = Path(temporary)/"artifact"
        rank._write(artifact, {"probe": 1}, expected, set(expected))
        manifest = json.loads((artifact/"manifest.json").read_text())["manifest"]
        assert manifest["aliases"] == {"alias": "weight"}
        report["bytes"] = manifest["size"]
        report["chunks"] = len(manifest["chunks"])
        assert report["chunks"] >= 7
        stream = torch.cuda.Stream()
        raw_copy = torch.Tensor.copy_
        def delayed_copy(dst, src, *args, **kwargs):
            if dst.is_cuda and not src.is_cuda and src.is_pinned():
                torch.cuda._sleep(150_000_000)  # defer DMA to stress buffer reuse
            return raw_copy(dst, src, *args, **kwargs)
        for policy in ("0", "1", "0", "1"):
            with torch.cuda.stream(stream):
                backing = torch.full_like(a, 255, device="cuda")
                state = {"weight": backing[11:-7], "alias": backing[11:-7],
                         "bf16": torch.full_like(b, -99, device="cuda"),
                         "scalar": torch.full((), -99., device="cuda"),
                         "empty": torch.empty(0, device="cuda")}
                rank._read_manifest(artifact, {"probe": 1}, state)
                stream.synchronize()
                started = time.perf_counter()
                with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE_PIPELINE": policy}), \
                     patch.object(torch.Tensor, "copy_", delayed_copy):
                    loaded = rank._restore(artifact, manifest, state)
                elapsed = time.perf_counter()-started
                # _restore must already have drained every copy before hooks run.
                assert stream.query(), "copy still in flight when restore returned"
                assert loaded == set(expected)
                assert state["weight"].data_ptr() == state["alias"].data_ptr()
                for name, value in expected.items():
                    got = state[name].cpu()
                    assert torch.equal(got.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8)), name
                report["runs"].append(dict(policy=int(policy), exact=True, drained=True, seconds=elapsed))
                del state, backing
        # A later corrupted chunk is fatal; no source fallback or success receipt.
        bad = manifest["chunks"][2]
        with (artifact/"weights.bin").open("r+b") as f:
            f.seek(bad["offset"]); first=f.read(1); f.seek(bad["offset"]); f.write(bytes([first[0]^1]))
        with torch.cuda.stream(stream):
            backing = torch.full_like(a, 255, device="cuda")
            state = {"weight": backing[11:-7], "alias": backing[11:-7],
                     "bf16": torch.full_like(b, -99, device="cuda"),
                     "scalar": torch.full((), -99., device="cuda"), "empty": torch.empty(0, device="cuda")}
            try:
                with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE_PIPELINE": "1"}), \
                     patch.object(torch.Tensor, "copy_", delayed_copy):
                    rank._restore(artifact, manifest, state)
            except RuntimeError as exc:
                assert "checksum mismatch" in str(exc)
            else:
                raise AssertionError("corrupt artifact accepted")
            assert stream.query(), "failure left DMA in flight"
            assert torch.all(state["weight"][bad["start"]:bad["start"]+bad["size"]] == 255)
            report["corrupt_chunk_rejected"] = True
            report["failure_drained"] = True
    report["ok"] = True
    Path(os.environ.get("RANK_PIPELINE_GPU_REPORT", "/evidence/gpu-exact.json")).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
