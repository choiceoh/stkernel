"""Qwen3.8-Flash-Next on the fleet (profile): one rank of TP=4, served through the door.

    RANK=r WORLD_SIZE=4 MASTER_ADDR=10.10.10.2 python3 -m engine.profiles.qwen38.fleet \\
        --ranks /home/choiceoh/models/st-qwen38-tep4 --ckpt-meta /home/choiceoh/models/st-qwen38-tep4 --port 8000

The phases are GLM-5.3's fleet boot's (engine/profiles/glm53/boot.py `fleet` and `build`), the pieces Qwen3.8 needs:

    box and shape     the GB10 asserted; the kernel shape bound from the record the preshard wrote (or derived from the
                      config) BEFORE anything reads it -- one-shot sizes its cell from `bound().comm`
    comm              NCCL + gloo control; the one-shot RDMA transport
    lanes             lanes.served(); lanes.qualify() holds the lanes that own arithmetic to their oracles (D3)
    admission         the arena's bytes declared (weights + caches + snapshots + workspace ceiling) and admitted on
                      every rank before any rank allocates (base/arena.prepare_allocation, base/runtime_memory)
    load              the rank file's views carved from the arena, bound; the PLE table opened beside the rank file
                      (ple-r{r}of4.weight on the SSD, ple_table.py -- not in the arena); the dense lanes packed (PackStore)
    engine            caches, the served composition behind base/composed.ComposedModel (adapter.py), the runner
    capture           the target's verify graphs and the MTP head's draft graphs, every row count and context bucket
                      (decode_graphs.py), before the door admits work; `--spec-k K` (K > 1) chains the head K-1
                      times inside the draft replay and widens the verify step to K+1 (the checkpoint's own is 1)
    serve             base/serve.Server on every rank (rank 0 answers HTTP; the others follow the control plane)

The boot's host work runs where it is already waiting, as GLM-5.3's does (base/background): the kernel packages import
under the rendezvous, and the door's host half -- the tokenizer, the chat template and what the door reads off it --
builds under the load and the packs and is joined before the capture, which is Python dispatch and needs the GIL. Every
rank writes its phase table (boot-rank{r}.json) and memory ledger (memory-rank{r}.json) under --dump-dir, and rank 0
prints the table: the first fleet boot's 107.4 s and 40.1 s had no rows, only container timestamps.

Not here yet: the asynchronous decode pipeline, the NVMe tier, self-calibration and the vision tower. Prefill runs
eagerly.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from functools import partial
from pathlib import Path

_IMPORTS_BEGAN = time.perf_counter()

import torch                                                    # noqa: E402

from engine.profiles.qwen38 import facts                        # noqa: E402

_IMPORT_SECONDS = time.perf_counter() - _IMPORTS_BEGAN

GIB = 1 << 30
KV_GIB = 16.0               # the vLLM stack's fixed KV (boot 8: KV_CACHE_MEMORY 16 GiB); Qwen3.8's KV is 15 KiB a token
MAX_SEQS = 4
TOKEN_BUDGET = 32768        # a prefill chunk of whole blocks after the draft reservation (GLM-5.3's)
MAX_WAIT_S = 0.0
WORKSPACE_GIB = 12.0        # everything outside the arena, base/runtime_memory's enforced ceiling (GLM-5.3's value)
OS_RESERVE_GIB = 12.0       # twice earlyoom's 6 GiB floor
SNAPSHOT_GIB = 2.0
MODEL_NAME = "qwen3.8-flash-next"
DUMP_DIR = "/home/choiceoh/glm53-logs/st-qwen38-dumps"   # the launcher mounts /home/choiceoh/glm53-logs on every node


def door_host_half(ckpt_meta, *, renderer: bool) -> dict:
    """What the door reads off the checkpoint, on the host: the tokenizer (every rank), and on the rank that renders,
    the chat template with the think block, tool call layout and effort rungs read off it. No CUDA and nothing the
    engine builds, so the boot runs it on a thread beside the load (base/background)."""
    from engine.base import tool_formats
    from engine.base.serve import effort_rungs_checked, reasoning_marks
    from engine.profiles.qwen38.boot import EFFORT_RUNGS, chat_renderer, generation_defaults, tokenizer
    tok = tokenizer(Path(ckpt_meta))
    chat = chat_renderer(Path(ckpt_meta)) if renderer else None
    end, tail = reasoning_marks(tok, chat) if chat is not None else (None, ())
    return {"tok": tok, "chat": chat, "end": end, "tail": tail,
            "tools": tool_formats.detect(chat) if chat is not None else None,
            "efforts": effort_rungs_checked(chat, EFFORT_RUNGS) if chat is not None else None,
            "generation": generation_defaults(Path(ckpt_meta))}


def timed_store(store) -> dict:
    """The pack store's entry points timed where they stand -> {entry: [calls, seconds]}, for `prepare dense`'s gauges.

    The row was one number, and the two things inside it that a boot pays every time are different levers: the weight
    hash (a device-to-host copy on the caller's thread and sha256 on the store's worker, for every dense weight) and
    the packs, read from /cache after the first boot. `pack` and `pack_fp8` include the hash they wait for, and a
    `weight_digest` a lane calls itself is counted under its own name as well."""
    spent = {}

    def timed(entry, call):
        def run(*args, **kwargs):
            began = time.perf_counter()
            try:
                return call(*args, **kwargs)
            finally:
                calls, seconds = spent.get(entry, (0, 0.0))
                spent[entry] = (calls + 1, seconds + time.perf_counter() - began)
        return run

    for entry in ("weight_digest", "pack", "pack_fp8"):
        setattr(store, entry, timed(entry, getattr(store, entry)))
    return spent


def rank_loader(path, *, expected_layout: str):
    """The rank file, refused unless its layout marker is this profile's (a file written for another layout loads and
    computes garbage)."""
    from engine.base.loader import RankLoader
    loader = RankLoader(path)
    marker = (loader.metadata or {}).get("weight_layout")
    if marker != expected_layout:
        raise ValueError(f"{path}: weight layout {marker!r}, this profile reads {expected_layout!r}; "
                         "regenerate rank files with engine/profiles/qwen38/preshard.py")
    return loader


def build(comm, lanes, ranks_dir, ckpt_meta, *, kv_gib: float, max_seqs: int, recorder, max_new: int,
          temperature: float, seed: int, drafter: bool, workspace_gib: float = WORKSPACE_GIB, hc_fp8: bool = False,
          spec_k: "int | None" = None, prelude=None, query_shards: bool = True, mtp_precision: str = "bf16",
          draft_index: "tuple[int, int] | None" = None, mtp_experts: str = "bf16", mtp_experts_dir: "str | None" = None,
          shared_overlap: "bool | str" = False, tap_rows: int = 0):
    """One rank's engine, admitted, loaded, packed and captured -> (F, net, caches, model, runner). `prelude` (a started
    base/background.Background) is joined in its own row before the capture: the capture is Python dispatch, and a host
    thread still running there would take the GIL from it."""
    from engine.base import scheduler as sched
    from engine.base.arena import Arena, host_reclaim, prepare_allocation
    from engine.base.params import total_bytes
    from engine.base.prefix import PrefixCache
    from engine.base.record import Ring
    from engine.base.runner import STEP_RECORD, Runner
    from engine.base.runtime_memory import RuntimeMemory, reclaim_preparation_pages
    from engine.kernels.dense.store import PackStore
    from engine.profiles.qwen38.adapter import build_model, capture, close
    from engine.profiles.qwen38.boot import eos_ids, generation_defaults
    from engine.profiles.qwen38.caches import Qwen38Caches, cache_capacity, layout, snapshot_layout
    from engine.profiles.qwen38.net import Qwen38Net

    F = facts.load(ckpt_meta)
    if spec_k is not None and spec_k != F.spec_k:
        if spec_k < 1:
            raise ValueError("--spec-k drafts at least one token a step (--no-drafter serves without the head)")
        # the head chains its draft: K picks a step from one MTP layer, the verify step K+1 wide; the rings the
        # caches derive from spec_k follow, the fixed ones are checked (caches.check_rings)
        F = dataclasses.replace(F, spec_k=spec_k)
    net = Qwen38Net(F, comm, lanes, mtp=drafter, hc_fp8=hc_fp8, query_shards=query_shards, mtp_precision=mtp_precision,
                    mtp_experts=mtp_experts, shared_overlap=shared_overlap)
    specs = net.specs()
    if tap_rows and drafter and comm.rank == 0:
        # the draft queries the head's argmax reads, and its picks, recorded inside the captured draft graphs
        from engine.kernels.common.row_tap import RowTap
        net.draft_tap = RowTap(tap_rows, F.hidden, "cuda")
    nb, snapshots = cache_capacity(F, net.layers, kv_gib, max_seqs, SNAPSHOT_GIB, mtp=drafter)
    if nb < 2:
        raise MemoryError(f"KV {kv_gib} GiB leaves {nb} blocks")
    cache_layout = layout(F, net.layers, mtp=drafter)
    snapshot_bytes = snapshot_layout(F, net.layers)[0]
    rank = rank_loader(Path(ranks_dir) / f"rank{comm.rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    store = PackStore("/cache", comm.rank)
    arena_bytes = (total_bytes(specs) + 256 * (len(specs) + 64) + cache_layout.nbytes(nb, max_seqs)
                   + snapshots * snapshot_bytes)
    files = sorted(Path(ranks_dir).glob("rank*of4.safetensors"))
    side = {s.name for s in net.side_specs()}
    if side:
        # the MTP head's experts from their side file (mtp_side.py: BF16 by default), loaded into the arena beside the
        # rank file's views
        from engine.profiles.qwen38 import mtp_side
        side_file = mtp_side.path(mtp_experts_dir or mtp_side.DIRS[net.mtp_experts], comm.rank, net.mtp_experts)
        if not side_file.is_file():
            raise FileNotFoundError(f"{side_file}: the MTP head's {net.mtp_experts} experts are served from side files "
                                    f"(python3 -m engine.profiles.qwen38.mtp_side --precision {net.mtp_experts} ...); "
                                    "--mtp-experts nvfp4 serves the rank file's")
        side_rank = rank_loader(side_file, expected_layout=mtp_side.LAYOUTS[net.mtp_experts])
        files.append(side_file)
    failure = memory = None
    try:
        report = prepare_allocation(arena_bytes, files, int((workspace_gib + OS_RESERVE_GIB) * GIB),
                                    lambda: torch.cuda.mem_get_info()[0], cache_roots=(Path(ranks_dir).parent,),
                                    host_reclaim=host_reclaim)
        memory = RuntimeMemory(arena_bytes, int(workspace_gib * GIB), int(OS_RESERVE_GIB * GIB), comm=comm,
                               host_budget_bytes=0,
                               reclaim=partial(reclaim_preparation_pages, cache_roots=(Path(ranks_dir).parent,),
                                               host_reclaim=host_reclaim))
        recorder.gauge("boot_immediately_free_GiB", round(report["immediately_free"] / GIB, 3))
    except (MemoryError, OSError, RuntimeError) as exc:
        failure = exc
    comm.wait_prepared("arena-admission")
    failed = comm.all_reduce(torch.tensor([int(failure is not None)], device="cuda"))
    if int(failed.item()):
        if memory is not None:
            memory.close()
        raise MemoryError(f"TP arena admission failed: {failure or 'a peer has insufficient free memory'}") from failure
    model = None
    try:
        with recorder.phase("arena"):
            arena = Arena(arena_bytes)
        with recorder.phase("load"):
            views = rank.load([s.name for s in specs if s.name not in side], arena=arena, recorder=recorder)
            if side:
                views.update(side_rank.load(sorted(side), arena=arena, recorder=recorder))
            net.bind(views)
        with recorder.phase("ple table"):
            # the PLE table is not in the rank file: the rank's rows come off its SSD file beside it (ple_table.py)
            from engine.profiles.qwen38.ple_table import PLETable
            net.attach_ple(PLETable.open(ranks_dir, comm.rank, F), max_rows=max_seqs * (F.spec_k + 1))
        with recorder.phase("prepare dense"):
            # the packs move into their BF16 sources' arena regions (all but the shared expert's padded down projection)
            spent = timed_store(store)
            net.prepare_dense(store, consume_weights=True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            store.release_pages()
            for name, (calls, seconds) in spent.items():
                recorder.gauge(f"{name}_calls", calls)
                recorder.gauge(f"{name}_s", round(seconds, 3))
            for name, count in sorted(store.stats.items()):
                recorder.gauge(f"packs_{name}", count)
        if side:
            with recorder.phase("mtp experts"):
                # D3: the side-file experts' kernel held to its torch form before a draft reads it
                from engine.kernels.moe_rows import qualify as qualify_rows
                recorder.gauge("mtp_experts_qualify",
                               max(qualify_rows(torch.device("cuda"), precision=net.mtp_experts).values()))
        if drafter and draft_index is not None:
            with recorder.phase("draft index"):
                # the drafter's argmax from an inverted-file index over the head's rows (dense/ivf_head)
                for name, value in net.prepare_draft_head(*draft_index).items():
                    recorder.gauge(f"draft_index_{name}", value)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        with recorder.phase("caches"):
            caches = Qwen38Caches(arena, F, net.layers, nb, max_seqs, snapshots, mtp=drafter)
        with recorder.phase("engine"):
            gen = generation_defaults(Path(ckpt_meta))
            model, _store = build_model(net, caches, F, eos_ids=eos_ids(Path(ckpt_meta), F.config), max_new=max_new,
                                        temperature=temperature, top_p=float(gen.get("top_p", 1.0)), seed=seed,
                                        drafter=drafter)
            k = model.k
            contract = sched.Contract(chunk_align=F.chunk_align, token_budget=TOKEN_BUDGET, draft_slots=k,
                                      max_wait_s=MAX_WAIT_S, max_running=max_seqs,
                                      decode_token_budget=F.chunk_align + k)
            model.memory, model.arena = memory, arena
        with recorder.phase("wait for weight preparation"):
            comm.wait_prepared("weights-loaded", final=True)
        if memory is not None:
            memory.checkpoint("loaded")
        with recorder.phase("runner"):
            prefix = PrefixCache(F.block, sched.chunk_for(contract.chunk_align, contract.token_budget, k), snapshots)
            runner = Runner(model, contract, caches.pool, caches.slots, Ring(4096, STEP_RECORD.size), recorder,
                            keep_idle=True, prefix=prefix)
        if prelude is not None:
            with recorder.phase("wait for the prelude"):
                prelude.take()
            recorder.gauge("prelude_s", round(prelude.seconds, 3))
        with recorder.phase("warm prefill"):
            # the largest chunk held to the memory ceiling, and every prefill kernel family compiled, before the door:
            # the first request used to pay both (warmup.py)
            from engine.profiles.qwen38.warmup import warmup
            paid = warmup(net, caches, memory=memory, chunk=sched.chunk_for(contract.chunk_align, contract.token_budget, k),
                          max_context=model.max_context, mtp=model.drafter is not None)
            if comm.rank == 0:
                print("  warm prefill: " + ", ".join(f"{name} {seconds}s" for name, seconds in paid.items()), flush=True)
        with recorder.phase("warm eager moe"):
            # the eager MoE's decode-sized launches at the one capacity they will keep: the 2026-09-19 K=3 window's
            # first requests compiled six of them mid-request (warmup.eager_moe)
            from engine.profiles.qwen38.warmup import eager_moe
            paid = eager_moe(net)
            if comm.rank == 0:
                print("  warm eager moe: " + ", ".join(f"{name} {seconds}s" for name, seconds in paid.items()), flush=True)
        with recorder.phase("capture decode"):
            capture(model, max_seqs, memory=memory)
        if memory is not None:
            memory.checkpoint("ready")
            memory.ready = True
        recorder.gauge("blocks", nb)
        recorder.gauge("arena_GiB", round(arena.used / GIB, 3))
        return F, net, caches, model, runner
    except BaseException as exc:
        if getattr(comm, "preparation", None) is not None:
            try:
                comm.wait_prepared(f"failed: rank {comm.rank}: {type(exc).__name__}: {str(exc)[:200]}", timeout_s=60.,
                                   final=True)
            except BaseException:                                      # noqa: BLE001 -- it raises by design
                pass
        if model is not None:
            # a failure after capture (the "ready" vote a failed peer casts) must not leave captured NCCL graphs alive
            # past the process group
            from engine.base.graphs import cleanup_after_error
            cleanup_after_error(exc, lambda: close(model), "close decode graphs after a failed build")
        if memory is not None:
            memory.close()
        raise


def drain_draft_tap(tap, directory, every_s: float = 30.0) -> None:
    """Rank 0's draft queries to `directory` as they come (the tap is drained on a stream of its own), one npz a drain:
    `rows` the BF16 queries as int16 bits, `ids` the picks. Runs until the process ends -- a stopped container runs no
    `finally`, so nothing waits for the end to write."""
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tap.drained = int(tap.count.to("cpu"))              # the boot's warmup and capture rows are not queries
    part = 0
    while True:
        time.sleep(every_s)
        rows, ids, count = tap.drain()
        if len(ids):
            np.savez(directory / f"draft-queries-{part:05d}.npz", rows=rows.view(torch.int16).numpy(),
                     ids=ids.numpy(), count=count)
            part += 1


def draft_index(text: "str | None") -> "tuple[int, int] | None":
    """`--draft-index CLUSTERS/PROBES` -> (clusters, probes), or None for the whole head."""
    if text is None:
        return None
    try:
        clusters, probes = (int(v) for v in text.split("/"))
    except ValueError:
        raise SystemExit(f"--draft-index {text!r}: CLUSTERS/PROBES, e.g. 1024/32") from None
    if not 1 <= probes <= clusters:
        raise SystemExit(f"--draft-index {text!r}: 1 <= PROBES <= CLUSTERS")
    return clusters, probes


def write_dumps(rec, memory, dump_dir, rank: int) -> None:
    """This rank's phase table and memory ledger under `dump_dir`. A dump that cannot be written is said and skipped:
    the boot serves either way."""
    try:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        rec.dump(Path(dump_dir) / f"boot-rank{rank}.json")
        if memory is not None:
            memory.write(Path(dump_dir) / f"memory-rank{rank}.json")
    except OSError as exc:
        print(f"  rank {rank}: no boot dump under {dump_dir}: {type(exc).__name__}: {exc}", flush=True)


def main(argv=None) -> int:
    opened = time.perf_counter()
    from engine.base import instruments, kernel_shape
    from engine.base.background import Background
    from engine.base.comm import Comm
    from engine.base.instruments import Recorder
    from engine.base.serve import Server
    from engine.profiles.qwen38 import lanes as lane_tables
    from engine.profiles.qwen38.boot import EFFORT_ALIASES

    ap = argparse.ArgumentParser(prog="python3 -m engine.profiles.qwen38.fleet", description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=str(facts.RANKS), help="rank{r}of4.safetensors and the kernel shape record")
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT), help="config.json, tokenizer.json, generation_config.json")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--kv-gib", type=float, default=KV_GIB)
    ap.add_argument("--max-seqs", type=int, default=MAX_SEQS)
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-drafter", action="store_true", help="serve without the MTP head")
    ap.add_argument("--hc-fp8", action="store_true",
                    help="the hyper-connection mixers on block-scaled FP8 (half the bytes a step reads from them; the mixer's numbers change, so a quality bracket judges it)")
    ap.add_argument("--mtp-precision", choices=("bf16", "fp8", "w4"), default="bf16",
                    help="the MTP head's dense projections: the checkpoint's BF16 (default), block-scaled FP8, or the "
                         "target layers' W4A8 at decode rows (before 2026-09-19); acceptance moves, output does not")
    ap.add_argument("--draft-index", default=None, metavar="CLUSTERS/PROBES",
                    help="the drafter's argmax from an inverted-file index over the head's rows (e.g. 1024/32): a few MB "
                         "a draft instead of the head's 159; unset, the whole head. Acceptance moves, output does not")
    ap.add_argument("--mtp-experts", choices=("bf16", "fp8", "nvfp4"), default="bf16",
                    help="the MTP head's routed experts: the checkpoint's original BF16 (default; the operator's rule of "
                         "2026-09-19) or the export's FP8 from side files (engine/profiles/qwen38/mtp_side.py), or the "
                         "rank file's NVFP4 re-encoding")
    ap.add_argument("--mtp-experts-dir", default=None,
                    help="the side files' directory (default /home/choiceoh/models/st-qwen38-mtp-<precision>)")
    ap.add_argument("--tap-draft-queries", type=int, default=0, metavar="ROWS",
                    help="rank 0 records the MTP head's draft queries and picks in a ring of ROWS inside the captured "
                         "graphs and writes them under --dump-dir/draft-queries every 30 s (the IVF head's real recall)")
    ap.add_argument("--no-oneshot", action="store_true",
                    help="every collective on NCCL: the one-shot RDMA transport is not bound (its hidden-2560 cell is unmeasured; "
                         "the first fleet boot, 2026-09-18, stalled in it at every sum)")
    ap.add_argument("--no-query-shards", action="store_true",
                    help="every rank scores every index query of a prefill step, as before carry Q11: the rollback of the "
                         "quarter-a-rank scoring, on by the operator's decision of 2026-09-18 with the fleet unmeasured")
    ap.add_argument("--shared-overlap", choices=("off", "one", "all"), default="off",
                    help="a captured step's shared expert on a second stream beside its routed experts (carry M5): 'one' for "
                         "steps of one request's rows, 'all' for every captured step; off until a GB10's step says it pays")
    ap.add_argument("--dump-dir", default=DUMP_DIR, help="where every rank writes boot-rank{r}.json and memory-rank{r}.json")
    ap.add_argument("--spec-k", type=int, default=None,
                    help="drafts a step from the MTP head (the checkpoint's 1): K > 1 chains the head K-1 times inside "
                         "the draft replay and the verify step is K+1 tokens wide")
    a = ap.parse_args(argv)

    started = time.perf_counter()
    print(f"  box: {facts.check_box()}", flush=True)       # CUDA is initialised here, on this thread, before any other
    boxed = time.perf_counter()
    shape, source = kernel_shape.bind_recorded(a.ranks, Path(a.ckpt_meta) / "config.json",
                                               lambda: facts.load(a.ckpt_meta).kernel_shape())
    print(f"  kernel shape ({source}): {shape.describe()}", flush=True)
    rec = Recorder("boot")
    # What sat before the recorder -- python, torch, this module's imports, the box check and the shape -- as its first
    # row (base/instruments.process_seconds), so the table's total is the boot's.
    front = instruments.process_seconds()
    if front is not None:
        rec.mark("front", front, import_s=round(_IMPORT_SECONDS + (started - opened), 3),
                 box_s=round(boxed - started, 3), shape_s=round(time.perf_counter() - boxed, 3))
    # The kernel packages import while the ranks meet: the first ranks up wait at the rendezvous for the last one.
    imports = Background(lane_tables.import_kernels, "kernel-imports").start()
    with rec.phase("comm"):
        comm = Comm.init()
    rec.root.name = f"rank{comm.rank}"
    model = None
    try:
        with rec.phase("prepare one-shot"):
            if not a.no_oneshot:
                comm.prepare_oneshot()
        print(f"  collectives: {'NCCL' if a.no_oneshot else 'one-shot RDMA (NCCL where ineligible)'}", flush=True)
        with rec.phase("lanes"):
            with rec.phase("wait for the imports"):
                imports.take()
            rec.gauge("kernel_imports_s", round(imports.seconds, 3))
            lanes = lane_tables.served()
            # every b12x kernel this boot builds or reads, for the next tree's prebuild (kernels/b12x_requests)
            import os
            from engine.kernels import b12x_requests
            b12x_requests.record_loaded("qwen38", b12x_requests.path_under(
                os.environ.get("FLASHINFER_WORKSPACE_BASE"), "qwen38"))
        F = facts.load(a.ckpt_meta)
        with rec.phase("qualify lanes"):
            qualified = lane_tables.qualify(torch.device("cuda"), F)
        print(f"  lanes qualified: {qualified}", flush=True)
        # The door's host half builds under the load and the packs; build() joins it before the capture.
        prelude = Background(partial(door_host_half, a.ckpt_meta, renderer=comm.rank == 0), "boot-prelude").start()
        F, net, caches, model, runner = build(comm, lanes, a.ranks, a.ckpt_meta, kv_gib=a.kv_gib, max_seqs=a.max_seqs,
                                              recorder=rec, max_new=a.max_new, temperature=a.temperature, seed=a.seed,
                                              drafter=not a.no_drafter, hc_fp8=a.hc_fp8, spec_k=a.spec_k, prelude=prelude,
                                              query_shards=not a.no_query_shards, mtp_precision=a.mtp_precision,
                                              shared_overlap={"off": False, "one": True, "all": "all"}[a.shared_overlap],
                                              draft_index=draft_index(a.draft_index), mtp_experts=a.mtp_experts,
                                              mtp_experts_dir=a.mtp_experts_dir,
                                              tap_rows=a.tap_draft_queries)
        if getattr(net, "draft_tap", None) is not None:
            import threading
            threading.Thread(target=drain_draft_tap, args=(net.draft_tap, Path(a.dump_dir) / "draft-queries"),
                             name="draft-tap", daemon=True).start()
        print(f"  drafter: {'MTP head, K=' + str(model.k) if model.drafter is not None else 'none'} "
              f"(verify step {model.k + 1} tokens a row)", flush=True)
        with rec.phase("door"):
            door = prelude.take()
            tok, chat, tools, efforts = door["tok"], door["chat"], door["tools"], door["efforts"]
            server = Server(model, runner, comm, port=a.port, tokenizer=tok, chat=chat, model_name=MODEL_NAME,
                            generation=door["generation"], reasoning_end=door["end"], reasoning_tail=door["tail"],
                            effort_rungs=efforts, reasoning_effort_aliases=EFFORT_ALIASES if efforts is not None else None,
                            tool_parser=tools.parse if tools else None, tool_stream=tools.partial if tools else None,
                            tool_grammar=tools.grammar if tools else None,
                            tool_call_start=tools.start_token(tok) if tools else None)
        write_dumps(rec, model.memory, a.dump_dir, comm.rank)
        if comm.rank == 0:
            print(rec.table(), flush=True)
        print(f"  rank {comm.rank}: ready in {time.perf_counter() - started:.1f} s, door on port {a.port}", flush=True)
        server.loop()
        return 0
    finally:
        if model is not None:
            from engine.profiles.qwen38.adapter import close
            close(model)                        # the graphs' NCCL references go before the process group
        comm.close()


if __name__ == "__main__":
    sys.exit(main())
