# SPDX-License-Identifier: Apache-2.0
"""Rank-local checkpoint cache at the GLM outer load_weights boundary.

Persist BEFORE GLM FP8/MK hooks and vLLM quant finalization. Hits restore only
raw parameters/buffers; the normal post-load hooks still run, in their original
order. This is not a dump of a serving model or vLLM's postprocessed sharded_state
format. One readiness vote keeps InstantTensor's distributed source iterator
on the same path on every rank. No process-wide loader patches are added.
"""
import hashlib
import json
import logging
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch

from vllm.model_executor.layers.glm53_startup_cache import (
    cache_directory, digest_json, environment_identity, runtime_identity, tensor_spec,
)

logger = logging.getLogger(__name__)
CHUNK_BYTES = 64 * 1024 * 1024
FORMAT_VERSION = 1


def _drop_file_pages(fd, offset, size, mapped=None):
    # A 50+ GiB rank artifact must not accumulate in host page cache on UMA.
    # Copies are synchronous; no consumer retains these mapped bytes here.
    start = offset // mmap.PAGESIZE * mmap.PAGESIZE
    length = offset + size - start
    if mapped is not None and hasattr(mapped, "madvise"):
        mapped.madvise(mmap.MADV_DONTNEED, start, length)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, start, length, os.POSIX_FADV_DONTNEED)


def checkpoint_identity(directory):
    """Local immutable-source identity without rereading the full checkpoint.

    A new inode, ctime, mtime, size, path, index or config causes a miss. ctime
    detects same-size edits even if mtime was restored. This is a local cache;
    identities deliberately do not transfer between hosts or source copies.
    """
    root = Path(directory).resolve(strict=True)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = sorted(p.name for p in root.glob("*.safetensors"))
    if not names:
        raise ValueError("rank cache requires a local safetensors checkpoint")
    files = {}
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe checkpoint index path")
        path = root / name
        st = path.stat()
        files[name] = [str(path.resolve()), st.st_dev, st.st_ino, st.st_size,
                       st.st_mtime_ns, st.st_ctime_ns]
    configs = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            configs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"root": str(root), "files": files, "configs": configs}


def _schema(state):
    result = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta" or not tensor.is_contiguous():
            raise ValueError(f"unsupported rank tensor {name}")
        result[name] = tensor_spec(tensor)
    return result


def _context(model):
    from vllm.distributed import get_tensor_model_parallel_rank

    config = model.vllm_config
    pc = config.parallel_config
    for name in ("pipeline_parallel_size", "data_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(pc, name, 1) != 1:
            raise ValueError("rank cache currently supports plain tensor parallelism only")
    if pc.enable_expert_parallel or getattr(pc, "enable_eplb", False) or config.lora_config is not None:
        raise ValueError("rank cache does not support EP/EPLB/LoRA")
    if config.load_config.load_format not in ("auto", "safetensors", "instanttensor"):
        raise ValueError("unsupported source loader for rank cache")
    if getattr(model, "secondary_weights", ()):
        raise ValueError("rank cache does not support secondary weight sources")
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if getattr(method, "uses_meta_device", False) or hasattr(method, "_q"):
            raise ValueError("rank cache requires a fresh model before quantization")
    mc = model.model_config
    source = getattr(mc, "model_weights", None) or mc.model
    # The pinned DefaultModelLoader reads mc.model, not model_weights. Do not
    # identify one source while its fallback actually reads another.
    if source != mc.model:
        raise ValueError("rank cache does not support model_weights overrides")
    identity = {"format": FORMAT_VERSION, "checkpoint": checkpoint_identity(source),
                "rank": get_tensor_model_parallel_rank(),
                "tp": pc.tensor_parallel_size, "class": type(model).__qualname__,
                "model_config": mc.hf_config.to_dict(), "dtype": str(mc.dtype),
                "quantization": mc.quantization, "revision": mc.revision,
                "env": environment_identity(), "runtime": runtime_identity()}
    identity["runtime"]["sources"]["rank_cache"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return source, identity


def _write(directory, identity, state, loaded):
    """Publish an immutable directory only after all bounded chunks are durable."""
    if directory.exists():
        return False
    schema = _schema(state)
    size = sum(t.numel() * t.element_size() for t in state.values())
    directory.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory.parent).free < size + 512 * 1024**2:
        raise OSError("insufficient disk space for rank checkpoint cache")
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".rank-"))
    try:
        chunks, offset = [], 0
        with (temporary / "weights.bin").open("wb") as out:
            for name, tensor in state.items():
                raw = tensor.detach().reshape(-1).view(torch.uint8)
                for start in range(0, raw.numel(), CHUNK_BYTES):
                    cpu = raw[start:start + CHUNK_BYTES].cpu()
                    buf = memoryview(cpu.numpy())
                    checksum = hashlib.sha256(buf).hexdigest()
                    count = out.write(buf)
                    if count != len(buf):
                        raise OSError("short rank-cache write")
                    chunks.append({"name": name, "start": start, "offset": offset,
                                   "size": count, "sha256": checksum})
                    # Drain dirty pages per bounded chunk before asking the
                    # kernel to evict them. This is a one-time cold-write cost.
                    out.flush()
                    getattr(os, "fdatasync", os.fsync)(out.fileno())
                    _drop_file_pages(out.fileno(), offset, count)
                    offset += count
                    del buf, cpu
            out.flush()
            os.fsync(out.fileno())
        manifest = {"identity": identity, "schema": schema, "chunks": chunks,
                    "size": offset, "loaded": sorted(loaded)}
        with (temporary / "manifest.json").open("w") as out:
            json.dump({"manifest": manifest, "sha256": digest_json(manifest)}, out)
            out.flush()
            os.fsync(out.fileno())
        # A concurrent successful writer wins. Never replace an existing
        # directory under a reader; corrupt artifacts require operator removal.
        if directory.exists():
            return False
        os.rename(temporary, directory)
        return True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_manifest(directory, identity, state):
    envelope = json.loads((directory / "manifest.json").read_text())
    manifest = envelope["manifest"]
    if digest_json(manifest) != envelope["sha256"] or manifest["identity"] != identity:
        raise ValueError("rank-cache manifest identity/checksum mismatch")
    if manifest["schema"] != _schema(state):
        raise ValueError("rank-cache schema mismatch")
    if not isinstance(manifest["loaded"], list) or any(not isinstance(k, str) for k in manifest["loaded"]):
        raise ValueError("invalid loaded-weight set")
    totals = {name: 0 for name in state}
    offset = 0
    for chunk in manifest["chunks"]:
        name, start, size = chunk["name"], chunk["start"], chunk["size"]
        if name not in state or type(start) is not int or type(size) is not int:
            raise ValueError("invalid rank-cache chunk")
        if (start != totals[name] or chunk["offset"] != offset or not 0 < size <= CHUNK_BYTES
                or size % state[name].element_size()):
            raise ValueError("invalid rank-cache chunk extent")
        totals[name] += size
        offset += size
    if any(totals[name] != tensor.numel() * tensor.element_size() for name, tensor in state.items()):
        raise ValueError("rank cache is incomplete")
    if offset != manifest["size"] or (directory / "weights.bin").stat().st_size != offset:
        raise ValueError("rank-cache file size mismatch")
    return manifest


def _restore(directory, manifest, state):
    """Checksum each mapped chunk before copying. A mid-restore error is fatal.

    Falling back after partial writes could leave mixed source generations or
    partially initialized parameters. The caller must not catch this failure.
    """
    if manifest["size"] == 0:
        return set(manifest["loaded"])
    with (directory / "weights.bin").open("rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
            with torch.no_grad():
                for chunk in manifest["chunks"]:
                    view = memoryview(mapped)[chunk["offset"]:chunk["offset"] + chunk["size"]]
                    raw = None
                    try:
                        if hashlib.sha256(view).hexdigest() != chunk["sha256"]:
                            raise RuntimeError("rank-cache payload checksum mismatch; remove cache and retry")
                        raw = torch.frombuffer(view, dtype=torch.uint8)
                        target = state[chunk["name"]].reshape(-1).view(torch.uint8)
                        target[chunk["start"]:chunk["start"] + chunk["size"]].copy_(raw)
                    finally:
                        del raw
                        view.release()
                    _drop_file_pages(source.fileno(), chunk["offset"], chunk["size"], mapped)
    return set(manifest["loaded"])


def _all_ranks_ready(ready):
    """InstantTensor uses the WORLD device group: never mix hits/source loads.

    Use that established group directly, avoiding both extra Gloo object
    exchanges and vLLM's custom all-reduce initialization on this control vote.
    Even a rank whose local identity/metadata check failed must participate.
    """
    from vllm.distributed import get_world_group
    import torch.distributed as dist

    world = get_world_group()
    if world.world_size == 1:
        return ready
    group = world.device_group
    device = (torch.device("cuda", torch.cuda.current_device())
              if dist.get_backend(group) == "nccl" else torch.device("cpu"))
    vote = torch.tensor(int(ready), dtype=torch.int32, device=device)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    return bool(vote.item())


def load_rank_cached(model, weights, load):
    """Use once, on the model that owns the complete checkpoint walk."""
    root = cache_directory(os.environ.get("VLLM_GLM53_RANK_CACHE", ""))
    if not root or getattr(model, "_rank_cache_attempted", False):
        return load(weights)
    model._rank_cache_attempted = True
    start = time.perf_counter()
    directory = None
    try:
        source, identity = _context(model)
        state = model.state_dict()
        initial_schema = _schema(state)
        directory = Path(root) / digest_json(identity)
    except Exception as exc:
        logger.warning("[rank-cache] unavailable; using source loader: %r", exc)
    manifest = None
    if directory is not None:
        try:
            if directory.is_dir():
                manifest = _read_manifest(directory, identity, state)
        except Exception as exc:
            logger.warning("[rank-cache] rejecting %s; using source loader: %r", directory, exc)
    if _all_ranks_ready(manifest is not None):
        # Deliberately outside the fallback catch: partial restoration is fatal.
        loaded = _restore(directory, manifest, state)
        if checkpoint_identity(source) != identity["checkpoint"]:
            raise RuntimeError("checkpoint changed during rank-cache restore")
        logger.warning("[rank-cache] hit rank=%d bytes=%d in %.3fs; post-load hooks follow",
                       identity["rank"], manifest["size"], time.perf_counter() - start)
        return loaded
    if manifest is not None:
        logger.warning("[rank-cache] another rank missed; all ranks use source loader")
    loaded = load(weights)
    if directory is None:
        return loaded
    try:
        state = model.state_dict()
        if _schema(state) != initial_schema or checkpoint_identity(source) != identity["checkpoint"]:
            raise ValueError("model schema or checkpoint changed while loading")
        if loaded is None:
            raise ValueError("source loader did not return a loaded-weight set")
        save_start = time.perf_counter()
        if _write(directory, identity, state, loaded):
            logger.warning("[rank-cache] saved rank=%d in %.3fs; post-load hooks follow",
                           identity["rank"], time.perf_counter() - save_start)
        else:
            logger.warning("[rank-cache] keeping existing artifact %s", directory)
    except Exception as exc:
        logger.warning("[rank-cache] write skipped; source weights remain active: %r", exc)
    return loaded
