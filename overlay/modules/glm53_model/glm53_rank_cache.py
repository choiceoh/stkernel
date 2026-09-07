# SPDX-License-Identifier: Apache-2.0
"""Rank-local checkpoint cache at the GLM outer load_weights boundary.

Persist BEFORE GLM FP8/MK hooks and vLLM quant finalization. Hits restore only
raw parameters/buffers; the normal post-load hooks still run, in their original
order. This is not a dump of a serving model or vLLM's postprocessed sharded_state
format. One readiness vote keeps InstantTensor's distributed source iterator
on the same path on every rank. No process-wide loader patches are added.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
    HostStaging, cache_directory, digest_json, environment_identity, runtime_identity, tensor_spec,
)

logger = logging.getLogger(__name__)
CHUNK_BYTES = 64 * 1024 * 1024
FORMAT_VERSION = 2


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


def _state_plan(state):
    """Write exact shared views once, preserving the registration alias graph.

    GLM registers the same decoder layers in both layers and _active_layers.
    state_dict includes both names, almost doubling a naive checkpoint dump.
    Disjoint views are not merged. Other overlaps are unsupported: independent
    restoration could otherwise corrupt a neighbor if its view offset changes.
    """
    unique, aliases, seen, extents = {}, {}, {}, {}
    for name, tensor in state.items():
        key = (str(tensor.device), tensor.untyped_storage().data_ptr(),
               tensor.storage_offset(), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
        if tensor.numel() and key in seen:
            aliases[name] = seen[key]
        else:
            if tensor.numel():
                start = tensor.storage_offset() * tensor.element_size()
                end = start + tensor.numel() * tensor.element_size()
                spans = extents.setdefault(key[:2], [])
                if any(start < hi and lo < end for lo, hi in spans):
                    raise ValueError(f"unsupported overlapping rank tensor {name}")
                spans.append((start, end))
            unique[name] = tensor
            seen[key] = name
    return unique, aliases


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
    # PretrainedConfig.to_dict() retains integer id2label keys (including
    # nested vision configs) and may contain tuples. JSON turns them into
    # strings/lists. Normalize BEFORE hashing and comparison, so a valid
    # persisted identity equals the next fresh model's identity.
    return source, json.loads(json.dumps(identity, allow_nan=False))


def _write(directory, identity, state, loaded):
    """Publish an immutable directory only after all bounded chunks are durable."""
    if directory.exists():
        return False
    schema = _schema(state)
    unique, aliases = _state_plan(state)
    size = sum(t.numel() * t.element_size() for t in unique.values())
    directory.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory.parent).free < size + 512 * 1024**2:
        raise OSError("insufficient disk space for rank checkpoint cache")
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".rank-"))
    try:
        chunks, offset = [], 0
        staging = HostStaging()
        with (temporary / "weights.bin").open("wb") as out:
            for name, tensor in unique.items():
                raw = tensor.detach().reshape(-1).view(torch.uint8)
                for start, cpu in staging.chunks_to_cpu(raw, CHUNK_BYTES):
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
        manifest = {"identity": identity, "schema": schema, "aliases": aliases, "chunks": chunks,
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
    unique, aliases = _state_plan(state)
    if manifest["aliases"] != aliases:
        raise ValueError("rank-cache registration alias mismatch")
    if not isinstance(manifest["loaded"], list) or any(not isinstance(k, str) for k in manifest["loaded"]):
        raise ValueError("invalid loaded-weight set")
    totals = {name: 0 for name in unique}
    offset = 0
    for chunk in manifest["chunks"]:
        name, start, size = chunk["name"], chunk["start"], chunk["size"]
        if name not in unique or type(start) is not int or type(size) is not int:
            raise ValueError("invalid rank-cache chunk")
        if (start != totals[name] or chunk["offset"] != offset or not 0 < size <= CHUNK_BYTES
                or size % state[name].element_size()):
            raise ValueError("invalid rank-cache chunk extent")
        totals[name] += size
        offset += size
    if any(totals[name] != tensor.numel() * tensor.element_size() for name, tensor in unique.items()):
        raise ValueError("rank cache is incomplete")
    if offset != manifest["size"] or (directory / "weights.bin").stat().st_size != offset:
        raise ValueError("rank-cache file size mismatch")
    return manifest


def _copy_streamed_chunk(target, raw):
    target.copy_(raw, non_blocking=target.device.type == "cuda" and raw.is_pinned())
    if target.device.type == "cuda":
        # The reader cannot overwrite this slot until DMA on the caller's
        # stream has finished, including when a non-default stream is active.
        torch.cuda.current_stream(target.device).synchronize()


def _init_reader_affinity():
    """Keep this short-lived reader on the allowed performance CPU tier."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return
    try:
        allowed = os.sched_getaffinity(0)
        capacities = {cpu: int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpu_capacity").read_text())
                      for cpu in allowed}
        if not capacities or min(capacities.values()) <= 0:
            return
        best = max(capacities.values())
        fast = {cpu for cpu, capacity in capacities.items() if capacity >= best * 0.9}
        if fast != allowed:
            # pid=0 changes only this calling worker thread. Never widen the
            # caller's affinity/cpuset, or change the model/serving thread.
            os.sched_setaffinity(0, fast)
            logger.warning("[rank-cache] reader affinity cpus=%s", ",".join(map(str, sorted(fast))))
    except (OSError, ValueError):
        # Uniform/unknown hardware and denied affinity retain normal scheduling.
        return


def _restore_streamed(directory, manifest, state):
    """One sequential reader fills/checks two bounded slots before GPU use."""
    started = time.perf_counter()
    pin = any(t.device.type == "cuda" for t in state.values())
    slots = []
    for _ in range(2):
        try:
            slots.append(torch.empty(CHUNK_BYTES, dtype=torch.uint8, device="cpu", pin_memory=pin))
        except RuntimeError:
            if not pin:
                raise
            logger.warning("[rank-cache] pinned read slot unavailable; using synchronous copies")
            slots.append(torch.empty(CHUNK_BYTES, dtype=torch.uint8, device="cpu"))
    read_time = checksum_time = wait_time = copy_time = discard_time = 0.0
    with (directory / "weights.bin").open("rb", buffering=0) as source:
        def read_chunk(chunk, slot):
            before = time.perf_counter()
            if source.tell() != chunk["offset"]:
                raise ValueError("non-sequential rank-cache chunk")
            with memoryview(slot[:chunk["size"]].numpy()) as view:
                done = 0
                while done < len(view):
                    count = source.readinto(view[done:])
                    if not count:
                        raise EOFError("short rank-cache read")
                    done += count
                read_s = time.perf_counter() - before
                before = time.perf_counter()
                value = hashlib.sha256(view).hexdigest()
                return value, read_s, time.perf_counter() - before

        # A single reader preserves sequential disk access. Its next slot can
        # overlap the current GPU copy, but a slot is only queued again after
        # the copy's stream synchronization. Join before closing the file even
        # on I/O, checksum, or copy failure; workers never invoke CUDA.
        with ThreadPoolExecutor(max_workers=1, initializer=_init_reader_affinity) as pool, torch.no_grad():
            remaining = iter(manifest["chunks"])
            pending = deque()
            for slot in slots:
                chunk = next(remaining, None)
                if chunk is not None:
                    pending.append((chunk, slot, pool.submit(read_chunk, chunk, slot)))
            while pending:
                chunk, slot, future = pending.popleft()
                before = time.perf_counter()
                value, read_s, checksum_s = future.result()
                wait_time += time.perf_counter() - before
                read_time += read_s
                checksum_time += checksum_s
                if value != chunk["sha256"]:
                    raise RuntimeError("rank-cache payload checksum mismatch; remove cache and retry")
                before = time.perf_counter()
                target = state[chunk["name"]].reshape(-1).view(torch.uint8)
                _copy_streamed_chunk(target[chunk["start"]:chunk["start"] + chunk["size"]], slot[:chunk["size"]])
                copy_time += time.perf_counter() - before
                before = time.perf_counter()
                _drop_file_pages(source.fileno(), chunk["offset"], chunk["size"])
                discard_time += time.perf_counter() - before
                following = next(remaining, None)
                if following is not None:
                    pending.append((following, slot, pool.submit(read_chunk, following, slot)))
    logger.warning("[rank-cache-io] prefetch=1 chunks=%d bytes=%d hash_work_s=%.3f hash_wait_s=%.3f copy_s=%.3f discard_s=%.3f total_s=%.3f read_s=%.3f checksum_s=%.3f strategy=readinto",
                   len(manifest["chunks"]), manifest["size"], read_time + checksum_time, wait_time,
                   copy_time, discard_time, time.perf_counter() - started, read_time, checksum_time)
    return set(manifest["loaded"])


def _restore(directory, manifest, state):
    """Checksum each mapped chunk before copying. A mid-restore error is fatal.

    Falling back after partial writes could leave mixed source generations or
    partially initialized parameters. The caller must not catch this failure.
    """
    if manifest["size"] == 0:
        return set(manifest["loaded"])
    if os.environ.get("VLLM_GLM53_RANK_CACHE_PREFETCH", "0") == "1":
        return _restore_streamed(directory, manifest, state)
    staging = HostStaging()
    started = time.perf_counter()
    hash_work = hash_wait = copy_time = discard_time = 0.0
    with (directory / "weights.bin").open("rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
            def checksum(chunk):
                before = time.perf_counter()
                with memoryview(mapped)[chunk["offset"]:chunk["offset"] + chunk["size"]] as view:
                    value = hashlib.sha256(view).hexdigest()
                return value, time.perf_counter() - before

            with torch.no_grad():
                for chunk in manifest["chunks"]:
                    before = time.perf_counter()
                    value, elapsed = checksum(chunk)
                    hash_wait += time.perf_counter() - before
                    hash_work += elapsed
                    if value != chunk["sha256"]:
                        raise RuntimeError("rank-cache payload checksum mismatch; remove cache and retry")
                    view = memoryview(mapped)[chunk["offset"]:chunk["offset"] + chunk["size"]]
                    raw = None
                    try:
                        before = time.perf_counter()
                        raw = torch.frombuffer(view, dtype=torch.uint8)
                        target = state[chunk["name"]].reshape(-1).view(torch.uint8)
                        staging.copy_from_cpu(target[chunk["start"]:chunk["start"] + chunk["size"]], raw)
                        copy_time += time.perf_counter() - before
                    finally:
                        del raw
                        view.release()
                    before = time.perf_counter()
                    _drop_file_pages(source.fileno(), chunk["offset"], chunk["size"], mapped)
                    discard_time += time.perf_counter() - before
    logger.warning("[rank-cache-io] prefetch=%d chunks=%d bytes=%d hash_work_s=%.3f hash_wait_s=%.3f copy_s=%.3f discard_s=%.3f total_s=%.3f",
                   0, len(manifest["chunks"]), manifest["size"], hash_work, hash_wait,
                   copy_time, discard_time, time.perf_counter() - started)
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


def _evict_stale(root, current, keep):
    """Keep the newest `keep` artifacts under `root` (the one just written
    counts), remove the rest and any abandoned `.rank-*` staging directory.

    Every boot on a new overlay/profile build writes a fresh 45 GB artifact per
    node and nothing removed the old ones (39차 §4m: five per node in one day,
    srv1 full twice). Errors are logged, never raised: the served weights are
    already active when this runs."""
    if keep <= 0:
        return
    try:
        entries = []
        for child in Path(root).iterdir():
            if child.name.startswith(".rank-"):
                try:
                    if time.time() - child.stat().st_mtime > 24 * 3600:
                        shutil.rmtree(child, ignore_errors=True)
                        logger.warning("[rank-cache] removed abandoned staging %s", child)
                except OSError:
                    pass
                continue
            if not child.is_dir() or child == current:
                continue
            if not (child / "manifest.json").exists() and not (child / "weights.bin").exists():
                continue
            entries.append((child.stat().st_mtime, child))
        entries.sort(reverse=True)
        for _, stale in entries[max(keep - 1, 0):]:
            try:
                size = sum(f.stat().st_size for f in stale.rglob("*") if f.is_file())
            except OSError:
                size = -1
            shutil.rmtree(stale, ignore_errors=True)
            logger.warning("[rank-cache] evicted stale artifact %s (%.1f GB; keep=%d)",
                           stale, size / 1e9, keep)
    except Exception as exc:  # noqa: BLE001 -- hygiene must never fail a boot
        logger.warning("[rank-cache] eviction skipped: %r", exc)


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
        initial_aliases = _state_plan(state)[1]
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
    save_ready = False
    try:
        if directory is None:
            raise ValueError("local rank cache context is unavailable")
        state = model.state_dict()
        if (_schema(state) != initial_schema or _state_plan(state)[1] != initial_aliases
                or checkpoint_identity(source) != identity["checkpoint"]):
            raise ValueError("model schema or checkpoint changed while loading")
        if loaded is None:
            raise ValueError("source loader did not return a loaded-weight set")
        unique, _ = _state_plan(state)
        size = sum(t.numel() * t.element_size() for t in unique.values())
        directory.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(directory.parent).free
        if not directory.exists() and free < size + 512 * 1024**2:
            raise OSError(f"insufficient disk space for rank checkpoint cache: need={size} free={free}")
        save_ready = True
    except Exception as exc:
        logger.warning("[rank-cache] save unavailable; source weights remain active: %r", exc)
    # An unusable rank must still vote, so peers do not waste minutes writing
    # artifacts that the all-rank hit gate can never use.
    if not _all_ranks_ready(save_ready):
        logger.warning("[rank-cache] save skipped on all ranks; a peer cannot publish")
        return loaded
    try:
        save_start = time.perf_counter()
        if _write(directory, identity, state, loaded):
            logger.warning("[rank-cache] saved rank=%d bytes=%d in %.3fs; post-load hooks follow",
                           identity["rank"], size, time.perf_counter() - save_start)
            _evict_stale(root, directory,
                         int(os.environ.get("VLLM_GLM53_RANK_CACHE_KEEP", "2") or 0))
        else:
            logger.warning("[rank-cache] keeping existing artifact %s", directory)
    except Exception as exc:
        logger.warning("[rank-cache] write skipped; source weights remain active: %r", exc)
    return loaded
