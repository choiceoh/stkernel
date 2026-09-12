"""Boot GLM-5.3 on the ST engine (profile): facts -> comm -> arena -> loader
-> caches -> lanes -> runner. One function, two comms.

    python3 engine/profiles/glm53/boot.py --local --layers 0-4 --prompt 300 --max-new 8
        four ranks as threads on this box (LocalTP), reference lanes, a layer
        subset -- the whole runner path, no fleet, tokens are not language
    python3 engine/profiles/glm53/boot.py                  (on each of the four nodes, inside the glm53 image)
        the fleet: NCCL comm, served lanes, all 45 layers, then the serve loop

Nothing is discovered: the arena is weights + KV as facts.KV_GIB (the 40th
boot's measured KV), blocks = KV / block_bytes, slots = max_seqs + null.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# The arena is one virtual range backed by 20 MiB physical chunks (base/arena.py): the regime the
# box serves vLLM in. Set before torch reads it, so every segment of this process maps that way.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch                                                     # noqa: E402

from engine.base import scheduler as sched                       # noqa: E402
from engine.base.arena import Arena, prepare_allocation          # noqa: E402
from engine.base.runtime_memory import RuntimeMemory, reclaim_preparation_pages  # noqa: E402
from engine.base.comm import Comm, LocalTP                       # noqa: E402
from engine.base.config import Config, Fact, Knob                # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.base.record import DeathDump, Ring                   # noqa: E402
from engine.base.runner import STEP_RECORD, Runner               # noqa: E402
from engine.base.serve import Server                             # noqa: E402
from engine.base.kv_tier import NvmeTier                         # noqa: E402
from engine.base.prefix import PrefixCache                       # noqa: E402
from engine.base.shapes import chunk_for                         # noqa: E402
from engine.base.tiered_kv import TieredKV                       # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.caches import Glm53Caches, layout, snapshot_layout, stage_bytes   # noqa: E402
from engine.profiles.glm53 import drafter as drafter_mod           # noqa: E402
from engine.profiles.glm53.adapter import Glm53Engine, NullDrafter             # noqa: E402
from engine.profiles.glm53.net import Glm53Net                   # noqa: E402
from engine.profiles.glm53.weights import rank_loader            # noqa: E402
from engine.profiles.glm53 import vision as vision_mod           # noqa: E402

GIB = 1 << 30
KV_GIB = 24.0                       # production parity (vLLM's 24.02 GiB/rank, 28차 §8); the ST budget table leaves 41.6 GiB, 45차 §23
TOKEN_BUDGET = 8192                 # MAX_BATCHED: the 6,912 chunk law follows (shapes.py)
MAX_WAIT_S = 20.0                   # D10's one starvation valve
MAX_SEQS = 4                        # launcher MAX_SEQS
PREFIX_TIER_STAGE = 32 << 20        # the prefix tier's pinned staging + device scratch
PREFIX_SNAPSHOTS = 96               # block-boundary checkpoints: ~45 MiB/rank with the native two-head drafter KV shard.
                                    # The unit is the 768 block (nine per 6,912 chunk); boundaries a request adopted
                                    # outlive the ones nobody asked for (prefix._victim), so churn cannot flush them.


def tokenizer(ckpt=facts.CKPT):
    """The checkpoint's tokenizer, and nothing else it carries: this tokenizer.json ships a truncation rule
    (max_length 2048, direction Right) that `Tokenizer.from_file` honours and transformers' AutoTokenizer -- what
    vLLM tokenizes with -- ignores. Left in, the door silently cut every prompt to its first 2,048 tokens and
    answered about the head of a document whose question sat at the end (45차 §23: onepass 32K/128K 0/3)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(ckpt) / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return tok


def generation_defaults(ckpt=facts.CKPT) -> dict:
    """What a request may omit: the checkpoint's generation_config (vLLM applies it the same way -- temperature 1.0 here)."""
    import json
    g = json.loads((Path(ckpt) / "generation_config.json").read_text())
    return {k: g[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty") if k in g}


def grammars(ckpt, vocab: int, device=None, stop_token_ids=None):
    """base/grammar.Grammars over the checkpoint's tokenizer, on every rank (each row's matcher runs everywhere), or None
    where xgrammar is not installed -- then response_format is refused at the door (D3), never silently unenforced.

    `device`: prove the mask kernel here and pay its Triton JIT here (45차 §23 B2, the same rule as every other
    first-use cost -- and the same shape as `vision.qualify`: what cannot be served does not boot, D3)."""
    from engine.base import grammar
    if not grammar.available():
        return None
    from transformers import AutoTokenizer
    g = grammar.Grammars(AutoTokenizer.from_pretrained(str(ckpt)), vocab, stop_token_ids=stop_token_ids)
    if device is not None:
        g.qualify(device)
    return g


CHAT_TEMPLATE = "chat_template_mm_v2.jinja"     # what production serves with (launchers/lib/glm53-chat.sh); honours the `thinking` kwarg
REASONING_END = "</think>"                       # the model closes its reasoning with this token; the door splits content there
REQUEST_TIMEOUT_S = 3600.0                       # a request older than this is cancelled (the production probe's long-ingest bound x12)


def chat_renderer(ckpt=facts.CKPT):
    """messages -> prompt text through the checkpoint's chat template (the door's OpenAI chat endpoint). transformers'
    template engine renders it (the template needs its filters); chat_template_kwargs (`thinking`, ...) pass through."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(ckpt))
    t.chat_template = (Path(ckpt) / CHAT_TEMPLATE).read_text()

    def render(messages, kwargs, *, generation_prompt: bool = True, continue_final: bool = False):
        """`continue_final` resumes inside the last assistant turn instead of opening a new
        one, which is what a caller wants when it is handing back a partial answer to extend.
        It is passed only when asked for, so a template engine without it keeps working."""
        resume = {"continue_final_message": True} if continue_final else {}
        return t.apply_chat_template(messages, add_generation_prompt=generation_prompt,
                                     tokenize=False, **resume, **kwargs)
    return render


def eos_ids(ckpt=facts.CKPT) -> "list[int]":
    import json
    g = json.loads((Path(ckpt) / "generation_config.json").read_text())
    e = g.get("eos_token_id", [])
    return list(e) if isinstance(e, list) else [e]


def declared(a, comm_world: int) -> Config:
    """Native execution is fixed; only unqualified MLA and context experiments expire.

    Production declares no knobs and rejects every STK_* override. The
    retired execution, MoE, lane and eager-decode bisects cannot reappear
    through an old environment file.
    """
    import datetime as _dt
    facts_ = [
        Fact("model", str(a.ckpt_meta), "the checkpoint's config/tokenizer (facts.CKPT or a copy of those files)"),
        Fact("ranks", str(a.ranks), "preshard output"),
        Fact("world", comm_world, "facts.TP: four Sparks"),
        Fact("block", facts.BLOCK, "launcher --block-size"),
        Fact("spec_k", facts.SPEC_K, "launcher SPEC_K with DFlash2"),
        Fact("kv_gib", float(a.kv_gib), "40th boot's measured KV" if a.kv_gib == KV_GIB else "--kv-gib (local)"),
        Fact("port", int(a.port), "--port"),
        Fact("prefix_snapshots", PREFIX_SNAPSHOTS, "block-boundary checkpoints for prefix reuse (boot.PREFIX_SNAPSHOTS)"),
    ]
    fixed = dict(moe_static=lane_tables.MOE_STATIC_PRODUCTION,
                 lanes="served", decode_eager=0, execution="native")
    facts_ += [Fact(k, v, "native TP4 execution") for k, v in fixed.items()]
    if getattr(a, "production", False):
        defaults = dict(mla_prefill="stock", context_ceiling=0)
        return Config(facts_ + [Fact(k, v, "qualified production default") for k, v in defaults.items()], knobs=[])
    knobs = [
        Knob("mla_prefill", "stock", _dt.date(2026, 9, 30),
             "large-M MLA prefill candidates (39차: pair, pair4, tile32; production keeps them off pending numerics + a TTFT bracket)",
             "STK_mla_prefill=stock"),
        Knob("context_ceiling", 0, _dt.date(2026, 9, 30),
             "the served context ceiling: the door refuses a longer horizon and the decode ladder captures no bucket above it. "
             "0 = the checkpoint's trained positions (1,048,576), which is nine buckets and 36 target graphs; the boot's "
             "'target/<shape>/' memory rows carry each bucket's seconds, so a boot pair prices the cut before it is taken",
             "STK_context_ceiling=0", int),
    ]
    cfg = Config(facts_, knobs=knobs)
    return cfg


def decodable_vocab(tok) -> int:
    """Rows of the head the tokenizer has a token for: ids past this are masked (as served)."""
    return max(tok.get_vocab().values()) + 1


def build(comm, layers, lanes, ranks_dir, kv_gib: float, max_seqs: int, use_drafter: bool, recorder: Recorder,
          max_new: int = 256, temperature: float = 0.0, seed: int = 0, tier_dir: "str | None" = None,
          context_ceiling: "int | None" = None, execution: str = "stock",
          ckpt_meta: "str | Path" = facts.CKPT, drafter_dir: "str | Path" = drafter_mod.DRAFTER):
    """`ckpt_meta`: where config.json / tokenizer.json / generation_config.json are -- the HF checkpoint dir, or a
    copy of just those files: a node needs its rank file, the drafter and this, not the 185 GB checkpoint."""
    F = facts.load(ckpt_meta)
    if execution not in ("stock", "native"):
        raise ValueError("execution must be stock or native")
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    drafter_dir = Path(drafter_dir)
    D = drafter_mod.load(drafter_dir) if use_drafter else None
    dspecs = drafter_mod.specs(D) if D else []
    # Native DFlash stores only this rank's KV heads. The direct-ring lane
    # reads that shard; reserving the replicated ring would also waste three
    # quarters of every persistent prefix snapshot's drafter state.
    draft_heads = (D.kv_heads // comm.world_size if execution == "native" else D.kv_heads) if D else 0
    draft_cells = (D.window if execution == "native" else drafter_mod.ring_cells(D)) if D else 0
    draft_shape = (D.layers, draft_cells, draft_heads, D.head_dim) if D else None
    cache_layout = layout(F, net.layers, draft_shape)
    bb, sb = cache_layout.block_bytes, cache_layout.slot_bytes
    ns = max_seqs + 1
    # the persistent int32 block table is part of the same declared budget
    nb = int((kv_gib * GIB - ns * sb) // (bb + max_seqs * 4))
    if nb < 2:
        raise MemoryError(f"KV {kv_gib} GiB leaves {nb} blocks after {ns} slots of {sb / 2**20:.0f} MiB")
    rank = rank_loader(Path(ranks_dir) / f"rank{comm.rank}of{facts.TP}.safetensors")
    snapshot_bytes = snapshot_layout(F, net.layers, draft_shape)[0]
    # the vision tower (45차 §23 A7): whole on every rank, from vision.safetensors next to the rank files (preshard.py --vision);
    # absent, the door refuses pictures -- the fleet boot requires it (production serves images, PR #431)
    vision_file = Path(ranks_dir) / vision_mod.FILE
    VF = vision_mod.load(ckpt_meta) if vision_file.exists() else None
    vspecs = vision_mod.specs(VF) if VF else []
    arena_bytes = (total_bytes(specs) + total_bytes(dspecs) + total_bytes(vspecs) + 256 * (len(specs) + len(dspecs) + len(vspecs) + 64)
                   + cache_layout.nbytes(nb, max_seqs) + PREFIX_SNAPSHOTS * snapshot_bytes + stage_bytes(F, net.layers, max_seqs))
    memory = None
    if len(net.layers) == F.layers:
        # Fixed byte ceilings, not a measured workspace claim. Preparation
        # records peaks for the largest prefill and every declared graph.
        workspace_bytes, os_reserve_bytes = 12 * GIB, 4 * GIB
        files = sorted(Path(ranks_dir).glob("rank*of4.safetensors"))
        if D:
            files.append(drafter_dir / "model.safetensors")
        if VF:
            files.append(vision_file)
        failure = None
        try:
            report = prepare_allocation(arena_bytes, files, workspace_bytes + os_reserve_bytes,
                                        lambda: torch.cuda.mem_get_info()[0],
                                        cache_roots=(Path(ranks_dir).parent, drafter_dir.parent))
            memory = RuntimeMemory(arena_bytes, workspace_bytes, os_reserve_bytes, comm=comm,
                                   reclaim=partial(reclaim_preparation_pages,
                                                   cache_roots=(Path(ranks_dir).parent, drafter_dir.parent)))
        except (MemoryError, OSError, RuntimeError) as exc:
            failure = exc
        # A failed rank must prevent peers from starting their large CUDA
        # allocations; closing NCCL only after one rank fails is too late.
        failed = comm.all_reduce(torch.tensor([int(failure is not None)], device="cuda"))
        if int(failed.item()):
            if memory is not None:
                memory.close()
            raise MemoryError(f"TP arena admission failed: {failure or 'a peer has insufficient immediately free memory'}") from failure
        recorder.gauge("boot_immediately_free_GiB", round(report["immediately_free"] / GIB, 3))
        recorder.gauge("boot_reclaimed_GiB", round(report["reclaimed"] / GIB, 3))
        recorder.gauge("boot_model_cache_files_returned", report["cache_files"])
        # D1: the box declared, with every line's provenance, before the arena is allocated
        from engine.profiles.glm53 import budget as budget_mod
        b = budget_mod.budget(kv_gib, max_seqs, chunk=sched.chunk_for(F.chunk_align, TOKEN_BUDGET, D.k if D else 0), ckpt=ckpt_meta,
                              ranks_dir=ranks_dir, rank=comm.rank, drafter_dir=drafter_dir if D else None, snapshots=PREFIX_SNAPSHOTS,
                              draft_tp=comm.world_size if execution == "native" else 1,
                              draft_native=execution == "native")
        recorder.gauge("budget_unassigned_GiB", round(b.kv_gib - b.kv_declared_gib, 2))
        if comm.rank == 0:
            print(budget_mod.report(b))
    try:
        with recorder.phase("arena"):
            arena = Arena(arena_bytes)
        recorder.gauge("arena_expandable", int(arena.expandable))
        with recorder.phase("load"):
            views = rank.load(
                [s.name for s in specs], arena=arena, recorder=recorder)
            net.bind(views)
        tok = tokenizer(ckpt_meta)
        decodable = decodable_vocab(tok)
        drafter = NullDrafter()
        if D:
            with recorder.phase("load drafter"):
                dviews = RankLoader(drafter_dir / "model.safetensors").load([s.name for s in dspecs], arena=arena, recorder=recorder)
            drafter = drafter_mod.Drafter(D, net, decodable)
            drafter.bind(dviews)
        if execution == "native":
            from engine.kernels.dense.store import PackStore
            from engine.kernels.prefill_collectives import PrefillCollectives
            store = PackStore("/cache", comm.rank)
            with recorder.phase("prepare native execution"):
                net.prepare_dense(store, consume_weights=True)
                net.prefill_transport = PrefillCollectives(comm)
                if D:
                    drafter.prepare_fast(store, consume_weights=True)
            for name, count in store.stats.items():
                recorder.gauge("dense_pack_"+name, count)
            recorder.gauge("target_native_linears", len(net.dense)-1)
            recorder.gauge("drafter_native_linears", len(drafter.dense) if D else 0)
            # Scratch from one-time quantization must not consume the workspace
            # measured for prefill/capture. Live packs and graph owners remain.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            store.release_pages()
        vision = None
        if VF:
            with recorder.phase("load vision"):
                vviews = RankLoader(vision_file).load([s.name for s in vspecs], arena=arena, recorder=recorder)
                vision = vision_mod.Vision(VF, vviews, comm)
        # Everything from here to the first ledger row was 7.5 s of a measured boot with no name
        # (boot-time study 5-c): the caches and their zeroing, the engine, the tier's pinned staging,
        # the prefix snapshots and the runner.
        with recorder.phase("caches"):
            caches = Glm53Caches(arena, F, net.layers, nb, max_seqs, draft=draft_shape, snapshots=PREFIX_SNAPSHOTS, stage=True)
        with recorder.phase("engine"):
            # the aux layers must lie inside the chain: a layer subset (the local smoke) clips them to its last layer -- plumbing only
            aux = [min(L, net.layers[-1]) for L in drafter.aux_layers] if D else None
            engine = Glm53Engine(net, caches, F, drafter, max_new=max_new, eos_ids=eos_ids(ckpt_meta), temperature=temperature, seed=seed,
                                 decodable=decodable, aux_layers=aux, context_ceiling=context_ceiling)
            contract = sched.Contract(chunk_align=F.chunk_align, token_budget=TOKEN_BUDGET, draft_slots=drafter.k,
                                      max_wait_s=MAX_WAIT_S, max_running=max_seqs)
            engine.memory = memory
            engine.vision = vision
            engine.prefill_chunk = sched.chunk_for(contract.chunk_align, contract.token_budget, contract.draft_slots)
        if memory is not None:
            memory.checkpoint("loaded")
        with recorder.phase("runner"):
            tiered = prefix_tier = None
            if tier_dir:                                                                # D16: idle conversations park on NVMe, per rank
                tier = NvmeTier(Path(tier_dir) / f"rank{comm.rank}", block_bytes=cache_layout.block_bytes)   # a block is one NVMe unit (block-major)
                tiered = TieredKV(caches.pool, tier)
                # the prefix tier (45차 §23 A): evicted leaf boundaries -- their blocks and snapshot -- live on beside the parked
                # conversations, in their own directory and keyspace (a boundary's key is 56 bits of its hash)
                prefix_tier = TieredKV(caches.pool, NvmeTier(Path(tier_dir) / f"rank{comm.rank}" / "prefix",
                                                             block_bytes=cache_layout.block_bytes, stage_bytes=PREFIX_TIER_STAGE))
            prefix = PrefixCache(F.block, engine.prefill_chunk, PREFIX_SNAPSHOTS)      # boundaries = every 768 block (base/prefix.py)
            runner = Runner(engine, contract, caches.pool, caches.slots, Ring(4096, STEP_RECORD.size), recorder, tiered=tiered,
                            keep_idle=tiered is not None, prefix=prefix)                # with a tier, conversations live on and park
            runner.prefix_tier = prefix_tier
        recorder.gauge("blocks", nb); recorder.gauge("slots", ns); recorder.gauge("arena_GiB", round(arena.used / GIB, 3))
        recorder.gauge("prefix_snapshots", PREFIX_SNAPSHOTS); recorder.gauge("snapshot_MiB", round(snapshot_bytes / 2**20, 1))
        return F, net, caches, engine, runner
    except BaseException:
        if memory is not None:
            memory.close()
        raise


def run_prompts(engine: Glm53Engine, runner: Runner, prompts: "dict[int, list[int]]"):
    """Submit every prompt, step until idle. Returns {seq: generated ids}."""
    for seq, ids in prompts.items():
        engine.add(seq, ids)
        runner.submit(seq, len(ids))
    out = {seq: None for seq in prompts}
    while runner.step() is not None:
        for seq in list(out):
            if out[seq] is None and seq not in runner.state.running and seq not in runner.state.waiting:
                out[seq] = engine.generated(seq)
    return out


def native_execution_report(net, drafter):
    """Reject a prepared but unused lane before the full-model door opens."""
    target = [layer for name, layer in net.dense.items() if name != 'head']
    draft = list(drafter.dense.values())
    expected_mhc = 2*len(net.layers)-1  # first attn pre has no preceding post
    proof = dict(target_w4=sum(bool(p.executed & 1) for p in target),
                 target_nvfp4=sum(bool(p.executed & 4) for p in target),
                 target_fp8=sum(bool(p.executed & 2) for p in target),
                 head_fp8=net.dense['head'].executed,
                 drafter_w4=sum(bool(p.executed & 1) for p in draft),
                 drafter_context_fp8=bool(drafter.dense['fc.weight'].executed & 2),
                 mhc=len(net.mhc.executed),
                 prefill_collectives=sorted(net.prefill_transport.executed))
    if (proof['target_w4'] != len(target) or proof['target_nvfp4'] != len(target)
            or proof['target_fp8'] != len(target)
            or proof['drafter_w4'] != len(draft) or not proof['head_fp8']
            or not proof['drafter_context_fp8'] or proof['mhc'] != expected_mhc
            or len(proof['prefill_collectives']) != 2):
        raise RuntimeError(f'native execution proof is incomplete: {proof}')
    return proof


def local(a) -> int:
    print(f"  box: {facts.check_box()}")
    print(declared(a, facts.TP).table())
    layers = [int(x) for x in a.layers.split("-")]; layers = list(range(layers[0], layers[-1] + 1))
    torch.manual_seed(a.seed)
    prompts = {seq: torch.randint(0, 100_000, (a.prompt + 7 * seq,)).tolist() for seq in range(a.seqs)}
    tp = LocalTP(facts.TP)
    lanes = lane_tables.reference()
    if a.park:                                            # a run-private tier: parked ids from an earlier smoke must not collide
        import tempfile
        Path(a.tier_dir).mkdir(parents=True, exist_ok=True)
        a.tier_dir = tempfile.mkdtemp(prefix="local-", dir=a.tier_dir)

    def rank_main(comm):
        rec = Recorder(f"rank{comm.rank}")
        F, net, caches, engine, runner = build(comm, layers, lanes, a.ranks, a.kv_gib, MAX_SEQS, a.drafter, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed,
                                               tier_dir=a.tier_dir if a.park else None,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir)
        t0 = time.perf_counter()
        with rec.phase("generate"):
            out = run_prompts(engine, runner, prompts)
            torch.cuda.synchronize()
        parked = None
        if a.park:
            # D16 on the real caches: the finished conversation 0 still holds its blocks (keep_idle); park it, the arena
            # gets them back; resume into fresh blocks; wake and decode 4 more tokens -- they must equal a straight run's
            seq, straight = 0, 2
            with rec.phase("park"):
                free_before = caches.pool.available
                wrote = runner.park(seq)
                free_after = caches.pool.available
                got = runner.resume(seq)
                torch.cuda.synchronize()
            with rec.phase("continue"):
                runner.wake(seq)
                engine.limits[seq] = (engine.limits[seq][0] + 4, engine.limits[seq][1])
                while seq in runner.state.running and runner.step(now=0.0) is not None:
                    pass
                torch.cuda.synchronize()
            continued = engine.generated(seq)[a.max_new:]
            with rec.phase("straight"):                                          # the same prompt, max_new + 4 in one go
                engine.add(straight, prompts[seq], max_new=a.max_new + 4)
                runner.submit(straight, len(prompts[seq]), now=0.0)
                while straight not in runner.idle and runner.step(now=0.0) is not None:
                    pass
                torch.cuda.synchronize()
            parked = {"wrote": wrote, "got": got, "free_before": free_before, "free_after": free_after,
                      "continued": continued, "straight_tail": engine.generated(straight)[a.max_new:]}
        return {"rec": rec, "out": out, "steps": runner.steps, "ring": runner.ring.count, "secs": time.perf_counter() - t0,
                "kinds": [STEP_RECORD.unpack(r)[2] for r in runner.ring.ordered()], "blocks": caches.pool.available, "slots": caches.slots.available,
                "accepted": engine.accepted_total, "drafted": engine.drafted_total, "k": engine.drafter.k, "parked": parked}

    if a.serve:
        try:
            return local_serve(a, tp, lanes, layers, prompts)
        finally:
            if a.park:
                import shutil
                shutil.rmtree(a.tier_dir, ignore_errors=True)
    outs = tp.run(rank_main)
    r0 = outs[0]
    print(r0["rec"].table())
    same = all(o["out"] == r0["out"] for o in outs[1:])
    kinds = r0["kinds"]
    print(f"  layers {layers[0]}-{layers[-1]}, {a.seqs} prompts of ~{a.prompt} tokens, max_new {a.max_new}: {r0['steps']} steps "
          f"({kinds.count(1)} prefill, {kinds.count(2)} decode) in {r0['secs']:.1f} s; ring {r0['ring']} records; "
          f"blocks/slots returned: {r0['blocks']}/{r0['slots']}; drafter K={r0['k']}: {r0['accepted']}/{r0['drafted']} drafts accepted")
    for seq, ids in r0["out"].items():
        print(f"    seq {seq}: {len(ids)} tokens {ids[:12]}{'...' if len(ids) > 12 else ''}")
    print(f"  four ranks produced identical tokens: {same}")
    ok = same and all(len(ids) >= a.max_new for ids in r0["out"].values())
    if r0["parked"] is not None:
        pk = r0["parked"]
        same_p = all(o["parked"]["continued"] == pk["continued"] for o in outs[1:])
        print(f"  park/resume (D16 on the real caches): wrote {pk['wrote'] / 2**20:.1f} MiB (arena free {pk['free_before']} -> {pk['free_after']} blocks), "
              f"read back {pk['got'] / 2**20:.1f} MiB, continued {pk['continued']} == straight run's {pk['straight_tail']}: "
              f"{pk['continued'] == pk['straight_tail']}; ranks agree {same_p}")
        ok = ok and same_p and pk["wrote"] == pk["got"] and pk["free_after"] > pk["free_before"] and pk["continued"] == pk["straight_tail"]
    print("\n  " + ("PASS: the runner drove prefill and decode through the engine on four ranks" if ok else "FAIL"))
    if a.park:
        import shutil
        shutil.rmtree(a.tier_dir, ignore_errors=True)
    return 0 if ok else 1


def local_serve(a, tp, lanes, layers, prompts) -> int:
    """The serve loop itself, on four threads: rank 0 opens the door, a client
    thread posts the prompts over HTTP, the loop ends when they are answered."""
    import json
    import urllib.request
    port = a.port
    results = {}

    def rank_main(comm):
        rec = Recorder(f"rank{comm.rank}")
        F, net, caches, engine, runner = build(comm, layers, lanes, a.ranks, a.kv_gib, MAX_SEQS, a.drafter, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed,
                                               tier_dir=a.tier_dir if a.park else None,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir)
        tok = tokenizer(a.ckpt_meta)
        from engine.profiles.glm53.tools import parse_tool_calls, partial_tool_calls, tool_call_token, tool_grammar
        engine.grammars = grammars(a.ckpt_meta, F.vocab, caches.device, engine.eos)
        server = Server(engine, runner, comm, port=port, tokenizer=tok, chat=chat_renderer(a.ckpt_meta) if comm.rank == 0 else None,
                        model_name="glm-5.3-flash", reasoning_end=tok.token_to_id(REASONING_END), request_timeout_s=REQUEST_TIMEOUT_S,
                        tool_parser=parse_tool_calls, tool_stream=partial_tool_calls, tool_grammar=tool_grammar,
                        tool_call_start=tool_call_token(tok), generation=generation_defaults(a.ckpt_meta),
                        vision=vision_mod.Door(engine.vision.V, tok) if comm.rank == 0 and engine.vision is not None else None)
        httpd = None
        if comm.rank == 0:
            httpd = server._serve_http()                       # the door opens before the loop
            def post(body, path="/v1/completions"):
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=3600) as r:
                    return json.loads(r.read())

            def stream_chat(body):                            # the bench's dialect: SSE chunks, usage at the end
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                text, usage, finish = [], {}, None
                with urllib.request.urlopen(req, timeout=3600) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        obj = json.loads(line[5:])
                        usage = obj.get("usage") or usage
                        for ch in obj.get("choices") or []:
                            d = ch.get("delta") or {}
                            text.append(d.get("content") or d.get("reasoning_content") or "")
                            finish = ch.get("finish_reason") or finish
                return {"text": "".join(text), "usage": usage, "finish_reason": finish}

            def client():
                try:
                    for seq, ids in prompts.items():
                        results[seq] = post({"ids": ids, "max_tokens": a.max_new})
                    if a.park:                                # a second turn on conversation 0: parked on NVMe after its first, resumed here
                        results["turn2"] = post({"conversation": results[0]["conversation"], "ids": prompts[0][:5], "max_tokens": 4})
                    results["chat"] = stream_chat({"messages": [{"role": "user", "content": "안녕? 한 줄로 답해."}], "max_tokens": 6,
                                                   "stream": True, "stream_options": {"include_usage": True},
                                                   "chat_template_kwargs": {"thinking": True}})
                except Exception as e:                        # noqa: BLE001
                    results["error"] = repr(e)
                server.alive = False                          # rank 0 stops after the last answer ...
            threading.Thread(target=client, daemon=True).start()
        # ... and tells the others through the same broadcast the arrivals travel on
        while True:
            stop = comm.broadcast_object(not server.alive if comm.rank == 0 else None)
            if stop:
                break
            if not server.once():
                time.sleep(0.002)
        if httpd is not None:
            httpd.shutdown()
        if comm.rank == 0 and "error" in results:
            raise RuntimeError(f"client: {results['error']}")
        if comm.rank == 0 and server.pending:
            raise RuntimeError("rank 0 stopped with answers pending")
        return {"served": server.served, "steps": runner.steps}

    import threading
    t0 = time.perf_counter()
    outs = tp.run(rank_main)
    secs = time.perf_counter() - t0
    answers = {k: v for k, v in results.items() if k not in ("turn2", "chat")}
    ok = len(answers) == len(prompts) and all(len(r["ids"]) == a.max_new for r in answers.values())
    for seq, r in sorted(answers.items()):
        print(f"    seq {seq}: {r['completion_tokens']} tokens in {r['seconds']} s  {r['ids'][:8]}...")
    if a.park:
        t2 = results.get("turn2", {})
        print(f"    turn 2 on conversation {t2.get('conversation')}: {t2.get('completion_tokens')} tokens in {t2.get('seconds')} s (resumed from NVMe, 5 new prompt tokens)")
        ok = ok and t2.get("completion_tokens") == 4
    chat = results.get("chat", {})
    print(f"    chat (v2 template, streamed): {chat.get('usage', {}).get('completion_tokens')} tokens, finish {chat.get('finish_reason')}, text {chat.get('text', '')[:60]!r}")
    ok = ok and chat.get("usage", {}).get("completion_tokens") == 6 and chat.get("finish_reason") in ("stop", "length")
    print(f"  serve loop on four ranks: {outs[0]['steps']} steps, {outs[0]['served']} answered over HTTP in {secs:.1f} s")
    print("\n  " + ("PASS: requests in at rank 0, tokens out, every rank in lockstep" if ok else "FAIL"))
    return 0 if ok else 1


def fleet_lease_of() -> "dict | None":
    """The reservation the launcher took for this boot, if it took one.

    With it the engine publishes what it is doing and can be ASKED to hand the fleet
    over -- it finishes, parks its conversations where they survive (D16), and lets go.
    Without it the engine serves exactly as before; a lease is a reservation, not a
    dependency.
    """
    owner, path = os.environ.get("ST_LEASE_OWNER"), os.environ.get("ST_LEASE_PATH")
    return {"owner": owner, "path": path} if owner and path else None


def fleet(a) -> int:
    """One rank per node, inside the glm53 image: served lanes (D3: all or nothing), every layer, then serve."""
    print(f"  box: {facts.check_box()}")
    cfg = declared(a, facts.TP)
    # The rendezvous and the kernel imports are boot time too: 15.6 s of a measured 90.2 s boot sat
    # outside this table (boot-time study, 2026-09-11), so the recorder opens before them.
    rec = Recorder("boot")
    with rec.phase("comm"):
        comm = Comm.init()
    rec.root.name = f"rank{comm.rank}"
    engine = dump = None
    try:
        if comm.rank == 0:
            print(cfg.table())
        with rec.phase("prepare one-shot"):
            comm.prepare_oneshot()
        with rec.phase("lanes"):
            lanes = lane_tables.served(moe_static=cfg["moe_static"], mla_prefill=cfg["mla_prefill"],
                                       consume_scales=True)
        F, net, caches, engine, runner = build(comm, None, lanes, a.ranks, a.kv_gib, MAX_SEQS, True, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed, tier_dir=a.tier_dir,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir,
                                               context_ceiling=cfg["context_ceiling"] or None,
                                               execution=cfg["execution"])

        # "무장 != 서빙": which lanes and kernel cells this process actually bound, readable at
        # scrape time instead of inferred from a boot log nobody kept (45차 §17 lesson).
        engine.lane_info = {"lanes": lanes.name, "moe_static": cfg["moe_static"],
                            "mla_prefill": cfg["mla_prefill"], "spec_k": str(engine.drafter.k),
                            "context_ceiling": str(engine.max_context)}
        # a stale tier under one rank diverges the ranks (45th 21): find it in seconds, not after the capture
        Server._agree_on_parked(comm, sorted(runner.parked_keys()))
        with rec.phase("capture decode"):
            engine.capture_decode(MAX_SEQS)
        with rec.phase("warmup shapes"):
            paid = engine.warmup_shapes()                   # first-use JIT paid at boot, not on the first user (45차 §23 B2)
        if engine.vision is None:                           # production serves images and video (PR #431): so does this boot, or it does not boot
            raise RuntimeError(f"{vision_mod.FILE} is missing from {a.ranks}: write it once per node with "
                               f"`python3 engine/profiles/glm53/preshard.py --vision --out {a.ranks}` (45차 §23 A7)")
        with rec.phase("qualify vision"):
            paid.update(engine.vision.qualify())            # the largest image and video, before the door opens (D3)
        with rec.phase("qualify grammar"):
            engine.grammars = grammars(a.ckpt_meta, F.vocab, caches.device, engine.eos)   # response_format (json_object / json_schema), every rank
        if engine.memory is None or not engine.memory.ready:
            raise RuntimeError("full-model serving requires runtime memory qualification")
        engine.memory.checkpoint("production/ready")
        import json
        proof = native_execution_report(net, engine.drafter)
        print('ST_NATIVE_EXECUTION '+json.dumps(dict(rank=comm.rank, **proof)), flush=True)
        engine.memory.write(Path(a.dump_dir) / f"memory-rank{comm.rank}.json")
        dump = DeathDump(a.dump_dir, runner.ring, boot_id=f"glm53-r{comm.rank}-{int(time.time())}")
        if comm.rank == 0:
            print(rec.table())
            print(f"  ST engine: GLM-5.3, TP={facts.TP}, lanes={lanes.name}, KV {a.kv_gib} GiB, serving on :{a.port}")
            if runner.tiered is not None:
                t = runner.tiered.tier
                print(f"  NVMe tier: {sum(1 for k in t.index if t.has(int(k)))} conversations parked from before, {len(t.stale())} under another layout (kept, not resumable)")
        with rec.phase("door"):
            tok = tokenizer(a.ckpt_meta)
            renderer = chat_renderer(a.ckpt_meta) if comm.rank == 0 else None
        from engine.profiles.glm53.tools import parse_tool_calls, partial_tool_calls, tool_call_token, tool_grammar
        if comm.rank == 0:
            print("  warmup: " + ", ".join(f"{k} {v}s" for k, v in paid.items()) + (f"; structured output: {'on' if engine.grammars else 'off (no xgrammar)'}"))
        Server(engine, runner, comm, port=a.port, tokenizer=tok, chat=renderer,
               model_name="glm-5.3-flash", reasoning_end=tok.token_to_id(REASONING_END), request_timeout_s=REQUEST_TIMEOUT_S,
               tool_parser=parse_tool_calls, tool_stream=partial_tool_calls, tool_grammar=tool_grammar,
                        tool_call_start=tool_call_token(tok), generation=generation_defaults(a.ckpt_meta),
               vision=vision_mod.Door(engine.vision.V, tok) if comm.rank == 0 else None,
               lease=fleet_lease_of()).loop()
    finally:
        try:
            if dump is not None:
                dump.close()
            if engine is not None:
                try:
                    if engine.memory is not None:
                        engine.memory.write(Path(a.dump_dir) / f"memory-rank{comm.rank}.json")
                finally:
                    engine.close_decode()
        finally:
            comm.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="four ranks as threads on this box, reference lanes")
    ap.add_argument("--production", action="store_true", help="fixed serving defaults without expiring experiment knobs; rejects STK_* overrides")
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--kv-gib", type=float, default=KV_GIB)
    ap.add_argument("--prompt", type=int, default=300)
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serve", action="store_true", help="with --local: through the HTTP door and the lockstep loop")
    ap.add_argument("--drafter", action="store_true", help="with --local: DFlash2 drafts (aux layers clipped to the chain: plumbing, not quality)")
    ap.add_argument("--park", action="store_true", help="with --local: park a finished conversation on NVMe, resume, continue (D16)")
    ap.add_argument("--tier-dir", default="/home/choiceoh/glm53-logs/st-tier")
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT), help="dir with config.json, tokenizer.json, generation_config.json")
    ap.add_argument("--drafter-dir", default=str(drafter_mod.DRAFTER), help="DFlash2 config and model.safetensors directory")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dump-dir", default="/home/choiceoh/glm53-logs/st-dumps")
    a = ap.parse_args(argv)
    if a.local:
        if a.kv_gib == KV_GIB:
            a.kv_gib = 1.0                                      # a layer subset on one box
        return local(a)
    return fleet(a)


if __name__ == "__main__":
    raise SystemExit(main())
