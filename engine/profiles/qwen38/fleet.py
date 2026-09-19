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
    tiers             the NVMe tiers (D16, base/tiered_kv.open_tiers): a finished turn parks its blocks and state slot
                      under its conversation, an evicted prefix boundary its blocks and snapshot -- in the directory
                      GLM-5.3's boot uses, under the same caps (the two never serve at once); `--tier-dir ''` is none
    grammar          structured output on every rank (base/grammar): the compiler built off the prelude, its mask
                      kernel proven on the device (`bind_grammars`) -- what response_format and the door's tool-call
                      grammar need, or every request with `tools` is refused
    capture           the target's verify graphs and the MTP head's draft graphs, every row count and context bucket
                      (decode_graphs.py), before the door admits work; `--spec-k K` (K > 1) chains the head K-1
                      times inside the draft replay and widens the verify step to K+1 (the checkpoint's own is 1)
    serve             base/serve.Server on every rank (rank 0 answers HTTP; the others follow the control plane)

Pictures (`--vision auto|on|off`, auto by default): the vision tower (vision.py) whole on every rank from
vision.safetensors next to the rank files (preshard.py --vision), in the arena; rank 0's door turns a picture into its
placeholder run and every rank encodes it at the prefill piece that reaches it. `auto` serves pictures when every rank has
the file (and refuses to boot when only some do), `on` requires it, `off` serves text only. The net carries each row's
mRoPE delta in its captured graphs only when it serves pictures.

The boot's host work runs where it is already waiting, as GLM-5.3's does (base/background): the kernel packages import
under the rendezvous, and the door's host half -- the tokenizer, the grammar compiler, the chat template and what the
door reads off it -- builds under the load and the packs and is joined before the capture, which is Python dispatch and
needs the GIL. Every rank writes its phase table (boot-rank{r}.json) and memory ledger (memory-rank{r}.json) under
--dump-dir, and rank 0 prints the table: the first fleet boot's 107.4 s and 40.1 s had no rows, only container
timestamps.

Not here yet: the asynchronous decode pipeline and video. Target GPTQ self-calibration uses the shared
collector, disarmed through warmup/capture; the next boot reads its stamped Hessians. Prefill runs
eagerly.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from functools import partial
from pathlib import Path

_IMPORTS_BEGAN = time.perf_counter()

import torch                                                    # noqa: E402

from engine.base.tiered_kv import PREFIX_TIER_GIB, TIER_GIB, TIER_ROOT, tier_line   # noqa: E402
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
# the MTP head's window (Windowed-MTP): its first group and its last 511 -- the 2,048 positions it attends at most,
# chosen by recency instead of scored. On by the operator's decision of 2026-09-19 ("전부 켜"), acceptance unmeasured;
# fleet --mtp-window off (the launcher's ST_MTP_WINDOW=off) serves the scored selection
MTP_WINDOW = (1, 511)
# the drafts a step verifies end before the first pick the head gives less than this (LibraSpec's rule; adapter.ServedMTP
# `threshold`), and steps of up to NARROW_ROWS rows replay a verify graph as wide as what was proposed -- on by the same
# decision, the value a guess until the draft ledger's curve says (the ledger is on too: DRAFT_LEDGER); `off` verifies
# every draft
DRAFT_THRESHOLD = 0.1
NARROW_ROWS = 2
DRAFT_CANDIDATES = 20       # the served default top_k (generation_config.json): a sampled draft's whole nucleus
# rank 0 records what the head observes, the fine-tuning data (MTPInputTap) -- on by the same decision, at most
# TAP_CAP_GIB under --dump-dir/mtp-inputs, counting what earlier boots left there
TAP_CAP_GIB = 64.0
MODEL_NAME = "qwen3.8-flash-next"
DUMP_DIR = "/home/choiceoh/glm53-logs/st-qwen38-dumps"   # the launcher mounts /home/choiceoh/glm53-logs on every node
# A finished turn shorter than this is released, not parked: parking writes the whole 109 MiB state slot a rank (K=3)
# whatever the turn's length, and prefilling 128 tokens again costs less than reading that back -- GLM-5.3's floor
# (profiles/glm53/boot.py PARK_MIN_TOKENS, set against health pings that pushed real conversations out of the tier)
PARK_MIN_TOKENS = 128


def door_host_half(ckpt_meta, *, renderer: bool) -> dict:
    """What the door reads off the checkpoint, on the host: the tokenizer and the grammar compiler over it (every rank),
    and on the rank that renders, the chat template with the think block, tool call layout and effort rungs read off
    it. No CUDA and nothing the engine builds, so the boot runs it on a thread beside the load (base/background).

    The compiler is every rank's, not the renderer's: each rank builds its own matcher for a grammar row and advances it
    with the same committed tokens (base/grammar), so a rank without one could not follow a request rank 0 admitted. It
    reads the door's tokenizer (base/grammar.tokenizer_info), as GLM-5.3's prelude does, rather than parsing
    tokenizer.json again through transformers: on this checkpoint the two give xgrammar the same inputs, in about 2 s
    against 6 s a rank (measurements/qwen38_fleet_grammars_20260919). It is built for the vocabulary and end tokens read
    off the checkpoint here, which `bind_grammars` holds to the engine's; the mask kernel is proven there, on the main
    thread."""
    from engine.base import grammar, tool_formats
    from engine.base.serve import effort_rungs_checked, reasoning_marks
    from engine.profiles.qwen38.boot import EFFORT_RUNGS, chat_renderer, eos_ids, generation_defaults, tokenizer
    tok = tokenizer(Path(ckpt_meta))
    F = facts.load(ckpt_meta)
    stops = eos_ids(Path(ckpt_meta), F.config)
    chat = chat_renderer(Path(ckpt_meta)) if renderer else None
    end, tail = reasoning_marks(tok, chat) if chat is not None else (None, ())
    return {"tok": tok, "chat": chat, "end": end, "tail": tail,
            "tools": tool_formats.detect(chat) if chat is not None else None,
            "efforts": effort_rungs_checked(chat, EFFORT_RUNGS) if chat is not None else None,
            "generation": generation_defaults(Path(ckpt_meta)),
            # the compiler alone, None where xgrammar is not installed (the door then refuses structured output, D3)
            "grammars": grammar.for_checkpoint(ckpt_meta, F.vocab, None, stops, tokenizer=tok),
            "grammar_vocab": F.vocab, "grammar_stops": stops}


def bind_grammars(model, door: dict, device) -> None:
    """The prelude's grammar compiler onto the served model, its mask kernel proven on `device` first (Grammars.qualify:
    the kernel and its Triton JIT here, not inside the first structured request -- GLM-5.3's `qualify grammar`).

    Unbound, the model refused every grammar, and the door arms one for every request that carries `tools` (the
    tool-call grammar, lazily at the call marker): Qwen3.8's fleet answered each of them 400, "no grammar compiler is
    bound" (2026-09-19, main a8e3c4de). The thread read the vocabulary and the end tokens off the checkpoint again, so
    a compiler built for any others than the engine's is refused here rather than masking against the wrong table.
    None (no xgrammar) binds nothing and the door refuses what needs one."""
    grammars = door.get("grammars")
    if grammars is None:
        return
    vocab, stops = door["grammar_vocab"], door["grammar_stops"]
    if vocab != model.vocab or set(stops) != set(model.eos):
        raise RuntimeError(f"the boot prelude built a grammar compiler for vocab {vocab} and ends {sorted(stops)}; "
                           f"this engine has {model.vocab} and {sorted(model.eos)}")
    grammars.qualify(device)
    model.grammars = grammars


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
          shared_overlap: "bool | str" = False, tap_rows: int = 0, draft_threshold: "float | None" = None,
          draft_ledger=None, narrow_rows: int = 0, mtp_window: "tuple[int, int] | None" = None,
          mtp_tuned_dir: "str | None" = None, draft_ahead: bool = False, draft_candidates: int = 0,
          vision: str = "auto", self_calibrate: bool = True, tier_dir: "str | None" = None,
          lease_owner: "str | None" = None, mapped_staging: bool = True):
    """One rank's engine, admitted, loaded, packed and captured -> (F, net, caches, model, runner). `prelude` (a started
    base/background.Background of `door_host_half`) is joined in its own row before the capture: the capture is Python
    dispatch, and a host thread still running there would take the GIL from it. Its grammar compiler is bound to the
    model in the row after (`bind_grammars`); without a prelude the model serves no grammar. `draft_ledger`: a factory
    of rank 0's ledger (DraftLedger); the other ranks record to nothing, their draft graphs the same as its.
    `tier_dir`: the NVMe tiers' root (module docstring), None for none; `lease_owner` claims this rank's directory
    under it (base/tenancy); `mapped_staging` stages through one GB10 host mapping instead of pinned + device copies."""
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
    from engine.profiles.qwen38 import calibration as calibrate

    F = facts.load(ckpt_meta)
    if spec_k is not None and spec_k != F.spec_k:
        if spec_k < 1:
            raise ValueError("--spec-k drafts at least one token a step (--no-drafter serves without the head)")
        # the head chains its draft: K picks a step from one MTP layer, the verify step K+1 wide; the rings the
        # caches derive from spec_k follow, the fixed ones are checked (caches.check_rings)
        F = dataclasses.replace(F, spec_k=spec_k)
    net = Qwen38Net(F, comm, lanes, mtp=drafter, hc_fp8=hc_fp8, query_shards=query_shards, mtp_precision=mtp_precision,
                    mtp_experts=mtp_experts, shared_overlap=shared_overlap)
    if mtp_window is not None:
        # the head attends a sink and a recent window of groups instead of scoring (Windowed-MTP; its index keys are
        # never written) -- acceptance moves, output does not
        if not (mtp_window[0] >= 0 and mtp_window[1] > 0 and sum(mtp_window) <= F.index_blocks):
            raise ValueError(f"--mtp-window {mtp_window}: SINK >= 0 and RECENT > 0 groups, {F.index_blocks} at most")
        net.mtp_window = tuple(mtp_window)
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
    # the vision tower (module docstring): this rank's view of whether it serves pictures; the ranks agree after admission
    from engine.profiles.qwen38 import vision as eyes
    vision_file = Path(ranks_dir) / eyes.FILE
    if vision not in ("auto", "on", "off"):
        raise ValueError(f"--vision {vision!r}: auto, on or off")
    if vision == "on" and not vision_file.is_file():
        raise FileNotFoundError(f"{vision_file}: --vision on serves pictures from it "
                                f"(python3 -m engine.profiles.qwen38.preshard --vision --out {ranks_dir})")
    VF = eyes.load(ckpt_meta) if vision != "off" and vision_file.is_file() else None
    vspecs = eyes.specs(VF) if VF is not None else []
    arena_bytes = (total_bytes(specs) + 256 * (len(specs) + 64) + cache_layout.nbytes(nb, max_seqs)
                   + snapshots * snapshot_bytes + net.router_nbytes())
    files = sorted(Path(ranks_dir).glob("rank*of4.safetensors"))
    if VF is not None:
        arena_bytes += total_bytes(vspecs) + 256 * (len(vspecs) + 64)
        files.append(vision_file)
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
    tuned = set()
    if mtp_tuned_dir is not None and drafter:
        # the head's dense weights fine-tuned on the target's own streams (mtp_tune.py export), in the rank file's place
        from engine.profiles.qwen38 import mtp_tune
        tuned_path = Path(mtp_tuned_dir) / mtp_tune.tuned_file(comm.rank)
        tuned_rank = rank_loader(tuned_path, expected_layout=mtp_tune.LAYOUT)
        tuned = set(mtp_tune.served_names(F)) & {s.name for s in specs}
        files.append(tuned_path)
    calib_files = [*files, Path(ranks_dir) / facts.ple_file(comm.rank, facts.TP)]
    weights_id = calibrate.identity(rank.metadata, calib_files, F.config, hc_fp8=hc_fp8)
    store = PackStore("/cache", comm.rank, weights_id=weights_id, require_identity=True)
    calib_plan, calib_bytes, deferred = calibrate.plan(net, specs, store) if self_calibrate else ([], 0, [])
    arena_bytes += calib_bytes
    store.release_pages()
    recorder.gauge("calibration_GiB", round(calib_bytes / GIB, 3))
    recorder.gauge("calibration_deferred", len(deferred))
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
    seeing = int(comm.all_reduce(torch.tensor([int(VF is not None)], device="cuda")).item())
    if 0 < seeing < facts.TP:
        if memory is not None:
            memory.close()
        raise RuntimeError(f"{seeing} of {facts.TP} ranks have {eyes.FILE}: pictures are served by every rank or none "
                           "(fan the file out, or boot with --vision off)")
    net.serves_pictures = VF is not None                 # before any graph is captured: the rows carry their mRoPE deltas
    model = None
    try:
        with recorder.phase("arena"):
            arena = Arena(arena_bytes)
        with recorder.phase("load"):
            views = rank.load([s.name for s in specs if s.name not in side and s.name not in tuned], arena=arena,
                              recorder=recorder)
            if side:
                views.update(side_rank.load(sorted(side), arena=arena, recorder=recorder))
            if tuned:
                views.update(tuned_rank.load(sorted(tuned), arena=arena, recorder=recorder))
            net.bind(views)
            vviews = None
            if VF is not None:
                vviews = rank_loader(vision_file, expected_layout=eyes.LAYOUT).load([s.name for s in vspecs], arena=arena,
                                                                                   recorder=recorder)
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
        with recorder.phase("prepare precision"):
            net.prepare_routers(arena)
            # GLM's IEEE GEMM extension must build outside CUDA graph capture.
            from engine.kernels.router_fp32 import build as build_router
            build_router()
            calibration = calibrate.attach(net, calib_plan, arena,
                                           max_decode_rows=max(32, max_seqs * (F.spec_k + 1)))
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
            # the ledger's rank writes it; the others hand theirs to a no-op, so every rank's draft graphs report the
            # picks' probabilities (the collective that sums them runs on all four or none)
            ledger = None if draft_ledger is None else (draft_ledger() if comm.rank == 0 else (lambda record: None))
            model, _store = build_model(net, caches, F, eos_ids=eos_ids(Path(ckpt_meta), F.config), max_new=max_new,
                                        temperature=temperature, top_p=float(gen.get("top_p", 1.0)), seed=seed,
                                        drafter=drafter, draft_threshold=draft_threshold, draft_ledger=ledger,
                                        draft_candidates=draft_candidates, draft_ahead=draft_ahead)
            k = model.k
            contract = sched.Contract(chunk_align=F.chunk_align, token_budget=TOKEN_BUDGET, draft_slots=k,
                                      max_wait_s=MAX_WAIT_S, max_running=max_seqs,
                                      decode_token_budget=F.chunk_align + k)
            model.memory, model.arena = memory, arena
            model.calibration, model.calibration_root = calibration, store.root
            model.calibration_weights_id = weights_id
            if VF is not None:
                model.composition.vision = eyes.Vision(VF, vviews, comm)
        with recorder.phase("wait for weight preparation"):
            comm.wait_prepared("weights-loaded", final=True)
        if memory is not None:
            memory.checkpoint("loaded")
        with recorder.phase("runner"):
            tiered = prefix_tier = None
            if tier_dir:
                # D16: a finished turn parks on NVMe and its row, blocks and slot go back; an evicted leaf boundary
                # parks its blocks and snapshot. The directory and the caps are GLM-5.3's (base/tiered_kv): a layout's
                # files are foreign to the other, counted against the cap and the first forgotten.
                from engine.base.tiered_kv import open_tiers
                from engine.profiles.qwen38.caches import state_format
                tiered, prefix_tier, left = open_tiers(
                    caches.pool, caches.layout.block_bytes, tier_dir, comm.rank,
                    state_format=state_format(F, caches.layout, caches.snapshot_bytes_n, mtp=drafter),
                    owner=lease_owner, mapped_staging=mapped_staging)
                if left:
                    print(f"  rank{comm.rank}: tenant state cleared -- the fleet changed hands from {left}", flush=True)
                recorder.gauge("nvme_mapped_staging", int(mapped_staging))
            prefix = PrefixCache(F.block, sched.chunk_for(contract.chunk_align, contract.token_budget, k), snapshots)
            runner = Runner(model, contract, caches.pool, caches.slots, Ring(4096, STEP_RECORD.size), recorder,
                            tiered=tiered, keep_idle=True, prefix=prefix)
            runner.prefix_tier = prefix_tier
        if prelude is not None:
            with recorder.phase("wait for the prelude"):
                door = prelude.take()
            recorder.gauge("prelude_s", round(prelude.seconds, 3))
            with recorder.phase("qualify grammar"):
                # response_format and the door's tool-call grammar, every rank: the compiler came off the prelude
                bind_grammars(model, door, caches.device)
        with recorder.phase("warm eager moe"):
            # the eager MoE's decode-sized launches at the one capacity they will keep: the 2026-09-19 K=3 window's
            # first requests compiled six of them mid-request (warmup.eager_moe). Before the prefill passes: their
            # decode-sized widths route a few pairs to a rank too, and a smaller first capacity would build its own
            from engine.profiles.qwen38.warmup import eager_moe
            paid = eager_moe(net)
            if comm.rank == 0:
                print("  warm eager moe: " + ", ".join(f"{name} {seconds}s" for name, seconds in paid.items()), flush=True)
        with recorder.phase("warm prefill"):
            # the largest chunk held to the memory ceiling, and every prefill kernel family compiled, before the door:
            # the first request used to pay both (warmup.py)
            from engine.profiles.qwen38.warmup import warmup
            paid = warmup(net, caches, memory=memory, chunk=sched.chunk_for(contract.chunk_align, contract.token_budget, k),
                          max_context=model.max_context, mtp=model.drafter is not None, head=k + 1)
            if comm.rank == 0:
                print("  warm prefill: " + ", ".join(f"{name} {seconds}s" for name, seconds in paid.items()), flush=True)
        if VF is not None:
            with recorder.phase("qualify vision"):
                # the largest picture the processor makes, under the memory ceiling, before the door opens (D3)
                for name, seconds in model.composition.vision.qualify().items():
                    recorder.gauge(name.replace("/", "_") + "_s", seconds)
        with recorder.phase("capture decode"):
            capture(model, max_seqs, memory=memory, narrow_rows=narrow_rows if draft_threshold else 0)
        if calibration is not None:
            calibration.arm()
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


class DraftQueries:
    """Rank 0's draft queries to `directory` as they come (the tap is drained on a stream of its own), one npz a drain:
    `rows` the BF16 queries as int16 bits, `ids` the picks -- every `every_s` seconds on a thread of its own, and once
    more at `close` (close_on_exit), the rows still behind the counter's slack included."""

    def __init__(self, tap, directory, every_s: float = 30.0):
        import threading
        self.tap, self.directory, self.every_s = tap, Path(directory), every_s
        self.directory.mkdir(parents=True, exist_ok=True)
        self.part = 0
        self._lock = threading.Lock()                   # a drain at a time: each moves the tap's `drained`
        self._closed = threading.Event()
        tap.drained = int(tap.count.to("cpu"))          # the boot's warmup and capture rows are not queries
        threading.Thread(target=self._run, name="draft-tap", daemon=True).start()

    def _run(self) -> None:
        while not self._closed.wait(self.every_s):
            self._drain(final=False)

    def _drain(self, *, final: bool) -> int:
        import numpy as np
        with self._lock:
            rows, ids, count = self.tap.drain(final=final)
            if len(ids):
                np.savez(self.directory / f"draft-queries-{self.part:05d}.npz", rows=rows.view(torch.int16).numpy(),
                         ids=ids.numpy(), count=count)
                self.part += 1
            return len(ids)

    def close(self, timeout_s: float = 20.0) -> str:
        """The last drain, waited for at most `timeout_s`: it reads the device, which a wedged step may never give back
        (so close_on_exit runs it after the host-side recorders, and an exit never hangs on it)."""
        import threading
        self._closed.set()
        drained = []
        last = threading.Thread(target=lambda: drained.append(self._drain(final=True)), name="draft-tap-last",
                                daemon=True)
        last.start()
        last.join(timeout_s)
        return (f"draft queries: {drained[0]} rows in the last drain" if drained
                else f"draft queries: the last drain did not return in {timeout_s:.0f} s")


class DraftLedger:
    """Rank 0's draft ledger (adapter.ServedMTP.record): one JSON line a verified row -- every pick the head made and
    its probability, how many were proposed, how many the target kept -- under `directory`, flushed every `every`
    records or second, whichever first. Both are judged when a record arrives, so a boot's last records wait in the
    buffer for the next one; `close` (close_on_exit) writes them at the process's end."""

    def __init__(self, directory, every: int = 64):
        import threading
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"draft-ledger-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.file = open(self.path, "a", buffering=1 << 16)
        self.every, self.count, self.flushed = every, 0, time.monotonic()
        self._lock = threading.Lock()                   # a text file is not safe across threads: `close` runs on another

    def __call__(self, record: dict) -> None:
        import json
        record["t"] = round(time.time(), 3)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            self.file.write(line)
            self.count += 1
            now = time.monotonic()
            if self.count % self.every == 0 or now - self.flushed > 1.0:
                self.file.flush()
                self.flushed = now

    def close(self, timeout_s: float = 0.0) -> str:
        """The buffer to the file. The file stays open: the step loop may still hand a record, which nothing waits for."""
        with self._lock:
            self.file.flush()
            return f"draft ledger: {self.count} records in {self.path.name}"


class MTPInputTap:
    """Rank 0's record of what the MTP head observes (adapter.ServedMTP.observe): at every kept position the target's
    streams before its closing mixer and the token after it -- the head's fine-tuning data (mtp_tune.py). Shards of up
    to `rows` rows under `directory`, handed to a thread of their own and written there: `streams` [R, hc*H] BF16 as
    int16 bits, `meta` [R, 4] int64 (sequence, position, next token, 1 where a verify step kept it). The copy to the
    host waits for the device -- a data window's cost, not a measured one's. What is held is written at least every
    `every_s` seconds, and at the process's end by `close` (close_on_exit: the interpreter's exit, or the SIGTERM the
    launcher's `stop` sends rank 0). A SIGKILL still loses what the timer had not written."""

    def __init__(self, directory, rows: int = 4096, every_s: float = 30.0, cap_bytes: "int | None" = None):
        """`cap_bytes`: the directory's shards stop growing past it, what earlier boots wrote there counted -- a
        default-on tap must not fill rank 0's disk."""
        import queue
        import threading
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows, self.every_s = rows, every_s
        self.cap_bytes = cap_bytes
        self.written = sum(f.stat().st_size for f in self.directory.glob("mtp-inputs-*.npz"))
        self.full = cap_bytes is not None and self.written >= cap_bytes
        self.closed = False
        self.failed = 0                                 # shards the writer could not write (said when it happened)
        self.prefix = f"mtp-inputs-{time.strftime('%Y%m%d-%H%M%S')}"
        self._held, self._count, self._part = [], 0, 0
        self._lock = threading.Lock()
        self._pending = 0                               # shards handed to the writer and not yet done with
        self._in_place = threading.Condition(self._lock)
        self._queue = queue.Queue()
        self._last = time.monotonic()
        threading.Thread(target=self._write, name="mtp-inputs", daemon=True).start()

    def __call__(self, seq: int, ctx: int, next_ids, hidden, decoded: bool) -> None:
        if self.full or self.closed:
            return
        rows = hidden.detach().to("cpu")
        n = rows.shape[0]
        meta = torch.tensor([[seq, ctx + j, int(next_ids[j]), int(decoded)] for j in range(n)], dtype=torch.int64)
        with self._lock:
            if self.closed:                             # closed while the copy ran: not recorded
                return
            self._held.append((rows, meta))
            self._count += n
            if self._count >= self.rows:
                self._flush()

    def _flush(self) -> None:
        if self._count:
            rows = torch.cat([r for r, _ in self._held])
            meta = torch.cat([m for _, m in self._held])
            self._queue.put((self._part, rows, meta))
            self._pending += 1
            self._held, self._count, self._part = [], 0, self._part + 1
        self._last = time.monotonic()

    def close(self, timeout_s: float = 20.0) -> str:
        """What is held handed to the writer, then every shard it was handed renamed into place, waiting at most
        `timeout_s`. Nothing observed after it is recorded. From any thread, and again: a second close waits the same."""
        with self._lock:
            self.closed = True
            held = self._count
            self._flush()
            self._in_place.wait_for(lambda: self._pending == 0, timeout_s)
            return (f"mtp inputs: {held} rows held at close; this boot {self._part - self._pending - self.failed} "
                    f"shards written, {self._pending} still waiting, {self.failed} failed")

    def _write(self) -> None:
        import numpy as np
        import os
        import queue
        while True:
            try:
                part, rows, meta = self._queue.get(timeout=self.every_s / 4)
            except queue.Empty:
                with self._lock:
                    if time.monotonic() - self._last >= self.every_s:
                        self._flush()
                continue
            # Written under a name `mtp_tune.shards` cannot glob, then renamed into place. The trainer reads this
            # directory while the fleet is still writing it, and `np.savez` straight to the final name means a
            # reader sooner or later loads a truncated zip -- EOFError, in the middle of a data window.
            final = self.directory / f"{self.prefix}-{part:05d}.npz"
            partial = self.directory / f".{final.name}.part"
            try:
                with open(partial, "wb") as handle:
                    np.savez(handle, streams=rows.view(torch.int16).numpy(), meta=meta.numpy())
                os.replace(partial, final)
                self.written += final.stat().st_size
            except OSError as exc:                      # a full disk loses this shard, not the writer: `close` still returns
                self.failed += 1
                print(f"  mtp inputs: {final.name} not written: {type(exc).__name__}: {exc}", flush=True)
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
            finally:
                with self._lock:
                    self._pending -= 1
                    self._in_place.notify_all()
            if self.cap_bytes is not None and self.written >= self.cap_bytes and not self.full:
                self.full = True
                print(f"  mtp inputs: {self.directory} holds {self.written / 2**30:.1f} GiB, the cap -- recording stops",
                      flush=True)


CLOSE_S = 20.0      # what the recorders get at the process's end: under the 30 s the launcher's `stop` gives rank 0


def close_on_exit(closers: list, timeout_s: float = CLOSE_S) -> None:
    """The recorders' `close` in `closers` (in order; the boot appends them as it makes them, the device's last) run
    when the process ends: at the interpreter's exit, and on SIGTERM -- after which the process exits 143.

    SIGTERM is `docker stop`'s, which the launcher's `stop` sends rank 0 before it removes the containers. The
    container's python is its PID 1 (`exec python3`, no --init), and a PID 1 without a handler never receives SIGTERM
    at all. A handler alone is not enough either: Python runs it on the main thread between bytecodes, never while
    that thread sits in a wedged CUDA or NCCL call (base/stall.py). So the handler here does nothing but make the
    signal arrive -- its C half writes the signal's number to the wakeup fd the moment it does, and a thread of its
    own reads it there, closes the recorders within one `timeout_s` between them, and ends the process with
    `os._exit`, whatever the main thread is doing. Main thread only; the process's one wakeup fd (nothing else in
    the fleet sets one)."""
    import atexit
    import os
    import signal
    import threading

    def close_all(why: str) -> None:
        deadline = time.monotonic() + timeout_s
        for close in list(closers):
            try:
                said = close(max(0.0, deadline - time.monotonic()))
            except Exception as exc:                   # noqa: BLE001 -- one recorder's failure leaves the rest to write
                said = f"{type(exc).__name__}: {exc}"
            print(f"  {why}: {said}", flush=True)

    def watch(woken: int) -> None:
        while byte := os.read(woken, 1):
            if byte[0] == signal.SIGTERM:
                try:
                    close_all("SIGTERM")
                finally:                                # a SIGTERM ends the process, whatever the closing did
                    os._exit(128 + signal.SIGTERM)

    atexit.register(close_all, "exit")
    woken, wake = os.pipe()
    os.set_blocking(wake, False)
    signal.set_wakeup_fd(wake, warn_on_full_buffer=False)
    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    threading.Thread(target=watch, args=(woken,), name="sigterm", daemon=True).start()


def draft_threshold(text: "str | None") -> "float | None":
    """`--draft-threshold P` -> P in [0, 1); `off` (or None) -> None, every draft verified."""
    if text is None or text == "off":
        return None
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"--draft-threshold {text!r}: a probability, e.g. 0.3") from None
    if not 0.0 <= value < 1.0:
        raise SystemExit(f"--draft-threshold {text!r}: 0 <= P < 1")
    return value


def mtp_window(text: "str | None") -> "tuple[int, int] | None":
    """`--mtp-window SINK,RECENT` -> (sink, recent) groups of idx_ratio positions; `off` (or None) -> None, the scored
    selection."""
    if text is None or text == "off":
        return None
    try:
        sink, recent = (int(v) for v in text.split(","))
    except ValueError:
        raise SystemExit(f"--mtp-window {text!r}: SINK,RECENT groups, e.g. 1,511") from None
    if sink < 0 or recent <= 0:
        raise SystemExit(f"--mtp-window {text!r}: SINK >= 0, RECENT > 0")
    return sink, recent


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
                    help="the hyper-connection mixers on block-scaled FP8 (half the bytes a step reads from them). Off: "
                         "at decode it is slower (C=1 +9.6%% a step on the fleet, 2026-09-19, K=1), and it changes the "
                         "mixer's NUMBERS with no quality bracket to judge them (D4) -- the one lever here that moves "
                         "the output")
    ap.add_argument("--mtp-precision", choices=("bf16", "fp8", "w4"), default="bf16",
                    help="the MTP head's dense projections: the checkpoint's BF16 (default), block-scaled FP8, or the "
                         "target layers' W4A8 at decode rows (before 2026-09-19); acceptance moves, output does not")
    ap.add_argument("--draft-index", default=None, metavar="CLUSTERS/PROBES",
                    help="the drafter's argmax from an inverted-file index over the head's rows (e.g. 1024/32): a few MB "
                         "a draft instead of the head's 159; unset, the whole head. Acceptance moves, output does not. "
                         "Off by default: its acceptance is unmeasured (PR #1226/#1232) and the drafter is where this "
                         "engine spends precision rather than bytes (the operator's rule, PR #1235)")
    ap.add_argument("--mtp-experts", choices=("bf16", "fp8", "nvfp4"), default="bf16",
                    help="the MTP head's routed experts: the checkpoint's original BF16 (default; the operator's rule of "
                         "2026-09-19) or the export's FP8 from side files (engine/profiles/qwen38/mtp_side.py), or the "
                         "rank file's NVFP4 re-encoding")
    ap.add_argument("--mtp-experts-dir", default=None,
                    help="the side files' directory (default /home/choiceoh/models/st-qwen38-mtp-<precision>)")
    ap.add_argument("--tap-draft-queries", type=int, default=0, metavar="ROWS",
                    help="rank 0 records the MTP head's draft queries and picks in a ring of ROWS inside the captured "
                         "graphs and writes them under --dump-dir/draft-queries every 30 s (the IVF head's real recall)")
    ap.add_argument("--draft-threshold", default=str(DRAFT_THRESHOLD), metavar="P|off",
                    help="a row's drafts end before the first pick the MTP head gives less than P (LibraSpec's rule): "
                         "the verify step is as wide as what is proposed -- steps of up to --narrow-rows rows replay "
                         f"narrower graphs, captured at boot; more rows are never cut. Default {DRAFT_THRESHOLD}, "
                         "`off` verifies every draft")
    ap.add_argument("--narrow-rows", type=int, default=NARROW_ROWS,
                    help="with --draft-threshold: the row counts whose narrower verify widths are captured (1..N)")
    ap.add_argument("--draft-candidates", type=int, default=DRAFT_CANDIDATES, metavar="C",
                    help="a sampled row (temperature > 0, no rich options) draws its drafts from the MTP head's "
                         "distribution over its C largest logits under the row's own sampler, and the verify step "
                         "keeps them by block verification (base/sampler.block_verify_batch) -- the target's own "
                         "distribution out, more drafts kept than the exact match of the head's argmax, which keeps a "
                         f"draft only where the target's draw lands on it. Default {DRAFT_CANDIDATES}; 0 keeps the "
                         "argmax and the exact match. A greedy row always does")
    ap.add_argument("--mtp-tuned", default=None, metavar="DIR",
                    help="the MTP head's dense weights from mtp_tune.py's export (mtp-tuned-r{r}of4.safetensors) instead "
                         "of the rank file's: the head fine-tuned on the target's own streams; acceptance moves, output "
                         "does not")
    ap.add_argument("--tap-mtp-inputs", action=argparse.BooleanOptionalAction, default=True,
                    help="rank 0 records what the MTP head observes -- the target's streams and the next token at every "
                         "kept position -- under --dump-dir/mtp-inputs (the head's fine-tuning data, mtp_tune.py); "
                         f"20 KB a position, the host copying each after its verify step's read, {TAP_CAP_GIB:.0f} GiB "
                         "at most (--tap-mtp-inputs-cap-gib). On by default")
    ap.add_argument("--tap-mtp-inputs-cap-gib", type=float, default=TAP_CAP_GIB,
                    help="the tap directory's cap, earlier boots' shards counted: a data window that prefills more than "
                         f"the default {TAP_CAP_GIB:.0f} GiB (every prefilled position is recorded, the prompt's too) "
                         "raises it for its own boots")
    ap.add_argument("--draft-ahead", action=argparse.BooleanOptionalAction, default=True,
                    help="behind a verify step whose rows are all greedy and plain, the next draft step runs on the device "
                         "before the host reads the picks (adapter.ServedModel._verify_ahead): the host's read, commit and "
                         "scheduling beside the draft replay instead of between the replays. The same tokens; each row "
                         "reserves 2K positions ahead instead of K+1. On by default (CHARTER D11), unmeasured on the "
                         "fleet; --no-draft-ahead is the rollback")
    ap.add_argument("--draft-ledger", action=argparse.BooleanOptionalAction, default=True,
                    help="rank 0 writes one JSON line a verified row under --dump-dir/draft-ledger: the head's picks, "
                         "their probabilities, how many were proposed and kept (the threshold's curve). On by default")
    ap.add_argument("--mtp-window", default=f"{MTP_WINDOW[0]},{MTP_WINDOW[1]}", metavar="SINK,RECENT|off",
                    help="the MTP head attends its first SINK and last RECENT groups (4 positions each) instead of its "
                         "scored selection -- Windowed-MTP: no index scoring in the draft; acceptance moves, output does "
                         f"not. SINK + RECENT <= 512; default {MTP_WINDOW[0]},{MTP_WINDOW[1]}, `off` the scored selection")
    ap.add_argument("--no-oneshot", action="store_true",
                    help="every collective on NCCL: the one-shot RDMA transport is not bound (its hidden-2560 cell is unmeasured; "
                         "the first fleet boot, 2026-09-18, stalled in it at every sum)")
    ap.add_argument("--no-query-shards", action="store_true",
                    help="every rank scores every index query of a prefill step, as before carry Q11: the rollback of the "
                         "quarter-a-rank scoring, on by the operator's decision of 2026-09-18 with the fleet unmeasured")
    ap.add_argument("--shared-overlap", choices=("off", "one", "all"), default="one",
                    help="a captured step's shared expert on a second stream beside its routed experts (carry M5): 'one' (the "
                         "default: steps of one request's rows, C=1 -5%% a step on the fleet, measurements/"
                         "qwen38_shared_overlap_20260919), 'all' (every captured step: C=4 +6%%), 'off' (the rollback)")
    ap.add_argument("--leave", choices=lane_tables.LEAVES, default=lane_tables.LEAVE,
                    help="how a leave meets the TP sum before it (carry H4): 'prefetch' (the default) launches it as the "
                         "sum's programmatic dependent and pulls the site's down projection into L2 while the sum waits "
                         "for the other ranks; 'pdl' the dependent alone; 'off' the ordinary launch after the sum (the "
                         "rollback). The same bytes every way")
    ap.add_argument("--no-self-calibrate", action="store_true",
                    help="skip collecting missing target GPTQ Hessians (up to 8 GiB); existing matching blobs still pack")
    ap.add_argument("--vision", choices=("auto", "on", "off"), default="auto",
                    help="pictures: auto serves them when every rank has vision.safetensors next to its rank file, on "
                         "requires it, off serves text only (module docstring)")
    ap.add_argument("--dump-dir", default=DUMP_DIR, help="where every rank writes boot-rank{r}.json and memory-rank{r}.json")
    ap.add_argument("--tier-dir", default=TIER_ROOT,
                    help="the NVMe tiers' root, one rank<N> directory a rank under it -- GLM-5.3's, shared; '' is none "
                         "(finished turns then stay in their rows until evicted)")
    ap.add_argument("--nvme-mapped-staging", action=argparse.BooleanOptionalAction, default=True,
                    help="stage tier transfers through one GB10 host mapping (GLM-5.3's default) instead of pinned "
                         "and device copies")
    ap.add_argument("--spec-k", type=int, default=None,
                    help=f"drafts a step from the MTP head (this profile serves {facts.SPEC_K}; the checkpoint has one "
                         "MTP layer and K > 1 chains it K-1 times inside the draft replay, the verify step K+1 tokens "
                         "wide). --spec-k 1 is the rollback")
    a = ap.parse_args(argv)
    if (draft_threshold(a.draft_threshold) is not None or a.draft_ledger or a.draft_candidates) \
            and a.draft_index is not None:
        raise SystemExit("--draft-threshold, --draft-ledger and --draft-candidates (all on by default) read the head's "
                         "whole row; --draft-index reads a few of its clusters: pass --draft-threshold off "
                         "--no-draft-ledger --draft-candidates 0 with it")
    if a.draft_candidates < 0:
        raise SystemExit(f"--draft-candidates {a.draft_candidates}: a candidate count, or 0 for the argmax drafts")

    started = time.perf_counter()
    print(f"  box: {facts.check_box()}", flush=True)       # CUDA is initialised here, on this thread, before any other
    boxed = time.perf_counter()
    # Rank 0's recorders, appended below as the boot makes them, write what they hold when the process ends -- at its
    # exit, and on the SIGTERM the launcher's `stop` sends (close_on_exit). Any rank exits 143 on a SIGTERM.
    closers = []
    close_on_exit(closers)
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
    model = runner = None
    try:
        with rec.phase("prepare one-shot"):
            if not a.no_oneshot:
                comm.prepare_oneshot()
        print(f"  collectives: {'NCCL' if a.no_oneshot else 'one-shot RDMA (NCCL where ineligible)'}", flush=True)
        with rec.phase("lanes"):
            with rec.phase("wait for the imports"):
                imports.take()
            rec.gauge("kernel_imports_s", round(imports.seconds, 3))
            lanes = lane_tables.served(leave=a.leave)
            # every b12x kernel this boot builds or reads, for the next tree's prebuild (kernels/b12x_requests)
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
                                              tap_rows=a.tap_draft_queries,
                                              draft_threshold=draft_threshold(a.draft_threshold),
                                              draft_ledger=partial(DraftLedger, Path(a.dump_dir) / "draft-ledger")
                                              if a.draft_ledger else None,
                                              narrow_rows=a.narrow_rows,
                                              mtp_window=mtp_window(a.mtp_window), mtp_tuned_dir=a.mtp_tuned,
                                              vision=a.vision, self_calibrate=not a.no_self_calibrate,
                                              draft_candidates=a.draft_candidates, tier_dir=a.tier_dir or None,
                                              lease_owner=os.environ.get("ST_LEASE_OWNER") or None,
                                              mapped_staging=a.nvme_mapped_staging, draft_ahead=a.draft_ahead)
        if a.tap_mtp_inputs and comm.rank == 0 and model.drafter is not None:
            model.drafter.inputs_tap = MTPInputTap(Path(a.dump_dir) / "mtp-inputs",
                                                   cap_bytes=int(a.tap_mtp_inputs_cap_gib * 2**30))
            closers.append(model.drafter.inputs_tap.close)
        if isinstance(getattr(model.drafter, "ledger", None), DraftLedger):
            closers.append(model.drafter.ledger.close)
        if getattr(net, "draft_tap", None) is not None:
            closers.append(DraftQueries(net.draft_tap, Path(a.dump_dir) / "draft-queries").close)   # the device's: last
        print("  shared expert: " + {False: "unforked", True: "forked at one request's rows", "all": "forked at every captured step"}
              [net.shared_overlap], flush=True)
        leave = {"off": "launched after its sum", "pdl": "its sum's programmatic dependent",
                 "prefetch": "its sum's programmatic dependent, the mixer's down projection prefetched"}[lanes.leave]
        if lanes.leave == "prefetch" and net.hc_fp8:
            # every mixer reads its FP8 lanes (net._mixer_weight): no BF16 projection to prefetch, the dependent only
            leave = "its sum's programmatic dependent, nothing prefetched (--hc-fp8: the mixers read FP8 weights)"
        print("  leave: " + leave, flush=True)
        print(f"  drafter: {'MTP head, K=' + str(model.k) if model.drafter is not None else 'none'} "
              f"(verify step {model.k + 1} tokens a row"
              + (f"; drafts cut below p={model.drafter.threshold}, narrow widths to {a.narrow_rows} rows"
                 if model.drafter is not None and model.drafter.threshold is not None else "")
              + ("; draft ledger" if a.draft_ledger else "")
              + ("; draft step ahead of the host's read" if model.draft_ahead else "")
              + ("; mtp inputs recorded" if a.tap_mtp_inputs else "") + ")", flush=True)
        vision = model.composition.vision
        print("  pictures: " + ("served, the tower on every rank (--vision " + a.vision + ")" if vision is not None
                                else "not served (--vision " + a.vision + ")"), flush=True)
        print(f"  structured output: {'on' if model.grammars is not None else 'off (no xgrammar)'}", flush=True)
        with rec.phase("door"):
            from engine.profiles.qwen38 import vision as eyes
            door = prelude.take()
            tok, chat, tools, efforts = door["tok"], door["chat"], door["tools"], door["efforts"]
            server = Server(model, runner, comm, port=a.port, tokenizer=tok, chat=chat, model_name=MODEL_NAME,
                            generation=door["generation"], reasoning_end=door["end"], reasoning_tail=door["tail"],
                            effort_rungs=efforts, reasoning_effort_aliases=EFFORT_ALIASES if efforts is not None else None,
                            tool_parser=tools.parse if tools else None, tool_stream=tools.partial if tools else None,
                            tool_grammar=tools.grammar if tools else None,
                            tool_call_start=tools.start_token(tok) if tools else None,
                            vision=eyes.Door(vision.V) if comm.rank == 0 and vision is not None else None,
                            park_min_tokens=PARK_MIN_TOKENS)
        write_dumps(rec, model.memory, a.dump_dir, comm.rank)
        if comm.rank == 0:
            print(rec.table(), flush=True)
            if runner.tiered is not None:     # after the door's reconciliation: what every rank holds alike
                print(tier_line(runner.tiered.tier, TIER_GIB, "conversations"), flush=True)
                print(tier_line(runner.prefix_tier.tier, PREFIX_TIER_GIB, "prefix boundaries"), flush=True)
            else:
                print("  NVMe tier: none (--tier-dir '') -- a finished turn stays in its row until evicted", flush=True)
        print(f"  rank {comm.rank}: ready in {time.perf_counter() - started:.1f} s, door on port {a.port}", flush=True)
        server.loop()
        return 0
    finally:
        try:
            if model is not None:
                from engine.profiles.qwen38.adapter import close
                try:
                    model.file_calibration()
                finally:
                    close(model)                # the graphs' NCCL references go before the process group
        finally:
            for tier in (getattr(runner, "tiered", None), getattr(runner, "prefix_tier", None)):
                if tier is None:
                    continue
                try:
                    tier.close()                # a park still writing lands before the staging goes
                except Exception as exc:        # noqa: BLE001 -- a shutdown does not fail a shutdown
                    print(f"  rank {comm.rank}: a tier did not close: {type(exc).__name__}: {exc}", flush=True)
            comm.close()


if __name__ == "__main__":
    sys.exit(main())
