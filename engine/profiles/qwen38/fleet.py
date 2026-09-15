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
    load              the rank file's views carved from the arena, bound; the dense lanes packed (PackStore)
    engine            caches, the served composition behind base/composed.ComposedModel (adapter.py), the runner
    capture           the target's verify graphs and the MTP head's draft graphs, every row count and context bucket
                      (decode_graphs.py), before the door admits work
    serve             base/serve.Server on every rank (rank 0 answers HTTP; the others follow the control plane)

Not here yet: the asynchronous decode pipeline, the NVMe tier, self-calibration and the vision tower. Prefill runs
eagerly.
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch

from engine.profiles.qwen38 import facts

GIB = 1 << 30
KV_GIB = 16.0               # the vLLM stack's fixed KV (boot 8: KV_CACHE_MEMORY 16 GiB); Qwen3.8's KV is 15 KiB a token
MAX_SEQS = 4
TOKEN_BUDGET = 32768        # a prefill chunk of whole blocks after the draft reservation (GLM-5.3's)
MAX_WAIT_S = 0.0
WORKSPACE_GIB = 12.0        # everything outside the arena, base/runtime_memory's enforced ceiling (GLM-5.3's value)
OS_RESERVE_GIB = 12.0       # twice earlyoom's 6 GiB floor
SNAPSHOT_GIB = 2.0
MODEL_NAME = "qwen3.8-flash-next"


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
          temperature: float, seed: int, drafter: bool, workspace_gib: float = WORKSPACE_GIB):
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
    net = Qwen38Net(F, comm, lanes, mtp=drafter)
    specs = net.specs()
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
            views = rank.load([s.name for s in specs], arena=arena, recorder=recorder)
            net.bind(views)
        with recorder.phase("prepare dense"):
            net.prepare_dense(store)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            store.release_pages()
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


def main(argv=None) -> int:
    from engine.base import kernel_shape
    from engine.base.comm import Comm
    from engine.base.instruments import Recorder
    from engine.base.serve import Server, effort_rungs_checked, reasoning_marks
    from engine.base import tool_formats
    from engine.profiles.qwen38 import lanes as lane_tables
    from engine.profiles.qwen38.boot import EFFORT_ALIASES, EFFORT_RUNGS, chat_renderer, generation_defaults, tokenizer

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
    a = ap.parse_args(argv)

    started = time.perf_counter()
    print(f"  box: {facts.check_box()}", flush=True)
    shape, source = kernel_shape.bind_recorded(a.ranks, Path(a.ckpt_meta) / "config.json",
                                               lambda: facts.load(a.ckpt_meta).kernel_shape())
    print(f"  kernel shape ({source}): {shape.describe()}", flush=True)
    rec = Recorder("boot")
    comm = Comm.init()
    model = None
    try:
        comm.prepare_oneshot()
        lanes = lane_tables.served()
        F = facts.load(a.ckpt_meta)
        print(f"  lanes qualified: {lane_tables.qualify(torch.device('cuda'), F)}", flush=True)
        F, net, caches, model, runner = build(comm, lanes, a.ranks, a.ckpt_meta, kv_gib=a.kv_gib, max_seqs=a.max_seqs,
                                              recorder=rec, max_new=a.max_new, temperature=a.temperature, seed=a.seed,
                                              drafter=not a.no_drafter)
        tok = tokenizer(Path(a.ckpt_meta))
        chat = chat_renderer(Path(a.ckpt_meta)) if comm.rank == 0 else None
        end, tail = reasoning_marks(tok, chat) if chat is not None else (None, ())
        tools = tool_formats.detect(chat) if chat is not None else None
        efforts = effort_rungs_checked(chat, EFFORT_RUNGS) if chat is not None else None
        server = Server(model, runner, comm, port=a.port, tokenizer=tok, chat=chat, model_name=MODEL_NAME,
                        generation=generation_defaults(Path(a.ckpt_meta)), reasoning_end=end, reasoning_tail=tail,
                        effort_rungs=efforts, reasoning_effort_aliases=EFFORT_ALIASES if efforts is not None else None,
                        tool_parser=tools.parse if tools else None, tool_stream=tools.partial if tools else None,
                        tool_grammar=tools.grammar if tools else None,
                        tool_call_start=tools.start_token(tok) if tools else None)
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
