# SPDX-License-Identifier: Apache-2.0
"""Optional, exact FP8 startup artifacts. No CUDA work or vLLM import at import time."""
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import torch

logger = logging.getLogger(__name__)
FORMAT_VERSION = 1
_FILE_DIGESTS = {}
TRANSFER_BYTES = 64 * 1024 * 1024


class HostStaging:
    """One reusable pinned buffer; copies finish before reuse or CPU access.

    The first fleet trial spent 376 s copying a duplicated rank state through
    freshly allocated pageable .cpu() tensors. Bound and reuse the transfer
    allocation. Synchronize the current stream in BOTH directions: CPU hashing
    and overwriting pinned H2D inputs must never race CUDA's copy engine.
    """
    def __init__(self):
        self.buffer = None
        self.disabled = False

    def _get(self):
        if self.buffer is None and not self.disabled:
            try:
                self.buffer = torch.empty(TRANSFER_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True)
            except RuntimeError as exc:
                self.disabled = True
                logger.warning("[startup-cache] pinned staging unavailable; using synchronous copies: %r", exc)
        return self.buffer

    def chunks_to_cpu(self, raw, chunk_bytes=TRANSFER_BYTES):
        buffer = self._get() if raw.device.type == "cuda" else None
        step = min(chunk_bytes, TRANSFER_BYTES)
        for start in range(0, raw.numel(), step):
            chunk = raw[start:start + step]
            if buffer is None:
                yield start, chunk.cpu()
            else:
                cpu = buffer[:chunk.numel()]
                cpu.copy_(chunk, non_blocking=True)
                torch.cuda.current_stream(raw.device).synchronize()
                yield start, cpu

    def to_cpu(self, raw):
        if raw.device.type != "cuda":
            return raw.cpu()
        cpu = torch.empty(raw.numel(), dtype=torch.uint8, device="cpu")
        for start, chunk in self.chunks_to_cpu(raw):
            cpu[start:start + chunk.numel()].copy_(chunk)
        return cpu

    def digest(self, raw):
        digest = hashlib.sha256()
        for _, chunk in self.chunks_to_cpu(raw):
            digest.update(memoryview(chunk.numpy()))
        return digest.hexdigest()

    def copy_from_cpu(self, target, raw):
        buffer = self._get() if target.device.type == "cuda" else None
        if buffer is None:
            target.copy_(raw)
            return
        for start in range(0, raw.numel(), TRANSFER_BYTES):
            chunk = raw[start:start + TRANSFER_BYTES]
            cpu = buffer[:chunk.numel()]
            cpu.copy_(chunk)
            target[start:start + chunk.numel()].copy_(cpu, non_blocking=True)
            torch.cuda.current_stream(target.device).synchronize()


def cache_directory(value):
    # Profile loading preserves nonempty caller overrides. "0" therefore
    # provides an explicit rollback even when the profile names a cache path.
    return "" if (value or "").strip().lower() in ("", "0", "off", "false", "no") else value


def file_digest(path):
    """Bound the host allocation even for a large runtime shared library."""
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if key not in _FILE_DIGESTS:
        with path.open("rb") as source:
            _FILE_DIGESTS[key] = hashlib.file_digest(source, "sha256").hexdigest()
    return _FILE_DIGESTS[key]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tensor_digest(raw):
    """Hash contiguous CPU bytes without a second tensor-sized Python copy."""
    return hashlib.sha256(raw.numpy()).hexdigest()


def environment_identity():
    # Layout/JIT flags can change the result without changing package versions.
    # Store only a digest, never raw environment values in cache metadata.
    values = {k: v for k, v in os.environ.items()
              if k.startswith(("VLLM_", "DG_", "DEEP_GEMM_", "FLASHINFER_",
                               "CUTE_DSL_", "TORCH_", "CUDA_", "NVIDIA_TF32_"))
              and k not in ("VLLM_GLM53_RANK_CACHE", "VLLM_GLM53_FP8_CACHE")
              and not any(word in k for word in ("API_KEY", "SECRET", "PASSWORD", "ACCESS_TOKEN"))}
    return digest_json(values)


def _artifact_digest(data):
    return digest_json({"key": data["key"], **{
        name: {k: v for k, v in data[name].items() if k != "raw"}
        for name in ("q", "ws")}})


def runtime_identity():
    """Include imported vLLM/DeepGEMM implementations, not just release labels.

    These packages are locally overlaid. A version string alone cannot identify
    their loading/quantization semantics. Different import sets cause a safe miss.
    """
    sources = {}
    for name, module in sorted(sys.modules.copy().items()):
        if not name.startswith(("vllm.", "deep_gemm")):
            continue
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix in (".py", ".so"):
            sources[name] = file_digest(path)
    sources["startup_cache"] = file_digest(__file__)
    versions = {}
    for package in ("vllm", "deep-gemm", "flashinfer-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    capability = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    return {"format": FORMAT_VERSION, "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "capability": capability,
            "environment": environment_identity(), "versions": versions, "sources": sources}


def tensor_spec(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype)}


def _storage_record(tensor, staging):
    # DeepGEMM scales can have padded strides and a storage offset. Preserve
    # the ENTIRE backing allocation, including padding used by vector loads.
    raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))
    raw = staging.to_cpu(raw)
    return {**tensor_spec(tensor), "offset": tensor.storage_offset(),
            "raw": raw, "sha256": tensor_digest(raw)}


def _restore_storage(record, device, staging):
    raw = record["raw"]
    if not isinstance(raw, torch.Tensor) or raw.device.type != "cpu" or raw.dtype != torch.uint8:
        raise ValueError("invalid cached storage")
    if raw.ndim != 1 or not raw.is_contiguous() or tensor_digest(raw) != record["sha256"]:
        raise ValueError("cached storage checksum mismatch")
    dtype = getattr(torch, record["dtype"].removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("invalid cached dtype")
    shape, stride, offset = record["shape"], record["stride"], record["offset"]
    if len(shape) != 2 or len(stride) != 2 or not isinstance(offset, int) or offset < 0:
        raise ValueError("invalid cached layout")
    if (any(type(x) is not int or x <= 0 for x in shape)
            or any(type(x) is not int or x < 0 for x in stride)):
        raise ValueError("invalid cached layout")
    span = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
    if span * itemsize > raw.numel() or raw.numel() % itemsize:
        raise ValueError("cached layout exceeds its storage")
    storage = torch.empty(raw.numel(), dtype=torch.uint8, device=device)
    staging.copy_from_cpu(storage, raw)
    return torch.empty(0, dtype=dtype, device=device).set_(
        storage.untyped_storage(), offset, shape, stride)


class Fp8Cache:
    """Per-fold cache/timing state; an empty directory disables disk access."""
    def __init__(self, directory=None, identity=None):
        self.directory = cache_directory(directory if directory is not None else os.environ.get("VLLM_GLM53_FP8_CACHE", ""))
        self.identity = identity
        self.last_path = None
        self.hits = self.misses = self.errors = 0
        self.times = dict(key=0.0, read=0.0, quantize=0.0, write=0.0)
        self.staging = HostStaging()

    def quantize(self, weight, quantizer):
        self.last_path = None
        path = key = None
        if self.directory:
            start = time.perf_counter()
            try:
                if self.identity is None:
                    # Load the implementation before fingerprinting even on a hit.
                    import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
                    import vllm.utils.deep_gemm  # noqa: F401
                    self.identity = digest_json(runtime_identity())
                raw = weight.detach().contiguous().view(torch.uint8).reshape(-1)
                key = {"source": self.staging.digest(raw), "weight": tensor_spec(weight),
                       "runtime": self.identity, "recipe": "e4m3-ue8m0-block128-padded-v1"}
                del raw
                path = Path(self.directory) / (digest_json(key) + ".pt")
            except Exception as exc:
                self.errors += 1
                logger.warning("[fp8-cache] key unavailable; quantizing: %r", exc)
            finally:
                self.times["key"] += time.perf_counter() - start
        if path is not None:
            start = time.perf_counter()
            try:
                data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if data["key"] != key or _artifact_digest(data) != data["sha256"]:
                    raise ValueError("cached identity mismatch")
                q = _restore_storage(data["q"], weight.device, self.staging)
                ws = _restore_storage(data["ws"], weight.device, self.staging)
                rows, cols = weight.shape
                if q.dtype != torch.float8_e4m3fn or tuple(q.shape) != (
                        (rows + 127) // 128 * 128, (cols + 127) // 128 * 128):
                    raise ValueError("invalid FP8 shape/dtype")
                if ws.dtype not in (torch.float32, torch.int32):
                    raise ValueError("invalid DeepGEMM scale dtype")
                self.hits += 1
                self.last_path = path
                return q, ws, rows, cols
            except FileNotFoundError:
                pass
            except Exception as exc:
                # Do not hold a rejected GPU copy while rebuilding its replacement.
                data = q = ws = None
                self.errors += 1
                logger.warning("[fp8-cache] rejecting %s; rebuilding: %r", path.name, exc)
            finally:
                self.times["read"] += time.perf_counter() - start
        self.misses += 1
        start = time.perf_counter()
        result = quantizer(weight)
        self.times["quantize"] += time.perf_counter() - start
        if path is not None:
            start = time.perf_counter()
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                q, ws, _, _ = result
                data = {"key": key, "q": _storage_record(q, self.staging), "ws": _storage_record(ws, self.staging)}
                data["sha256"] = _artifact_digest(data)
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fp8-", delete=False) as out:
                    temporary = Path(out.name)
                    torch.save(data, out)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, path)
                self.last_path = path
            except Exception as exc:
                self.errors += 1
                logger.warning("[fp8-cache] write skipped: %r", exc)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("[fp8-cache] temporary cleanup failed: %s", temporary)
                self.times["write"] += time.perf_counter() - start
        return result

    def reject_last(self):
        """The existing direct-kernel source check remains authoritative."""
        if self.last_path is not None:
            try:
                self.last_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("[fp8-cache] could not remove rejected artifact %s", self.last_path)

    def report(self, label):
        logger.warning("[fp8-cache] %s enabled=%s hit=%d miss=%d errors=%d host-seconds=%s",
                       label, bool(self.directory), self.hits, self.misses, self.errors,
                       " ".join(f"{k}={v:.3f}" for k, v in self.times.items()))
