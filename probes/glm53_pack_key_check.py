#!/usr/bin/env python3
"""Exact hash and legacy-pack alias proof; GPU execution requires fleet ownership.

Only writes a temporary directory. Production cache files remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch


def check(mk, common, device):
    torch.set_num_threads(1)
    checks = 0
    with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_FAST_IO": "1"}):
        # Bit patterns include NaNs, signed zeros, offset/strided views and tails.
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.uint8):
            width = torch.empty((), dtype=dtype).element_size()
            cpu = torch.arange(7000 * width).to(torch.uint8).view(dtype).reshape(70, 100)
            value = cpu.to(device)
            for left, right in ((cpu, value), (cpu[1:43], value[1:43]),
                                (cpu[:, ::2], value[:, ::2]), (cpu.T, value.T)):
                expected = hashlib.sha256(left.contiguous().view(torch.uint8).numpy()).hexdigest()
                with patch.object(common, "TRANSFER_BYTES", 1024):
                    mk._PACK_STAGING = None
                    assert mk._weight_digest(right, "sha256") == expected
                checks += 1
        mk._PACK_STAGING = None
        if torch.device(device).type == "cuda":
            # Real 64 MiB boundary and a producer on a non-default CUDA stream.
            stream = torch.cuda.Stream()
            cpu = torch.arange(common.TRANSFER_BYTES + 513).to(torch.uint8)
            expected = hashlib.sha256(cpu.numpy()).hexdigest()
            with torch.cuda.stream(stream):
                value = cpu.to(device)
                assert mk._weight_digest(value, "sha256") == expected
                checks += 1
                mk._PACK_STAGING.disabled = True
                assert mk._weight_digest(value, "sha256") == expected
                checks += 1
            del cpu, value
            mk._PACK_STAGING = None

        weight = torch.arange(129 * 256).reshape(129, 256).to(dtype=torch.bfloat16, device=device)
        blob = {"version": mk.MK_PACK_VERSION,
                "wq4": torch.arange(2 * 2 * 128 * 64).to(torch.uint8).reshape(2, 2, 128, 64),
                "ws4": torch.arange(2 * 2 * 128 * 8).to(torch.int8).reshape(2, 2, 128, 8),
                "wgs": 0.125, "rgs": torch.arange(256).float(),
                "lr_a": torch.arange(256 * 8).reshape(256, 8).bfloat16(),
                "lr_b": torch.arange(8 * 256).reshape(8, 256).bfloat16()}
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_CACHE": root,
                                       "VLLM_GLM53_MK_PACK_ROWSHIFT": "1",
                                       "VLLM_GLM53_MK_PACK_LORC": "0"}), \
                patch.object(mk, "_mk_rank", return_value=0), \
                patch.object(mk, "_calib_hessian_for", return_value=None), \
                patch.object(mk, "_w4_row_shift", side_effect=AssertionError("pack rebuilt")):
            for zip_format in (False, True):
                with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "0"}):
                    legacy = Path(mk._pack_cache_path(weight, True, False, 0))
                    legacy.parent.mkdir(exist_ok=True)
                    torch.save(blob, legacy, _use_new_zipfile_serialization=zip_format)
                    baseline = mk.build_mk_weight_w4(weight)
                before = hashlib.sha256(legacy.read_bytes()).hexdigest()
                with patch.dict(os.environ, {"VLLM_GLM53_MK_PACK_SHA256": "1"}):
                    alias = Path(mk._pack_cache_path(weight, True, False, 0))
                    assert alias.samefile(legacy)
                    assert hashlib.sha256(alias.read_bytes()).hexdigest() == before
                    checks += 2
                    with patch.object(mk, "_weight_md5", side_effect=AssertionError("warm MD5")):
                        candidate = mk.build_mk_weight_w4(weight)
                    for left, right in zip(baseline, candidate):
                        if isinstance(left, torch.Tensor):
                            assert left.dtype == right.dtype and left.stride() == right.stride()
                            assert torch.equal(left, right)
                        else:
                            assert left == right
                        checks += 1
                    alias.unlink()
                assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
                checks += 1
    return {"ok": True, "device": str(device), "checks": checks,
            "source_sha256": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (mk, common)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from vllm.model_executor.layers import glm53_megakernel as mk
    from vllm.model_executor.layers import glm53_startup_cache as common
    report = check(mk, common, "cuda")
    Path(args.out).write_text(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
