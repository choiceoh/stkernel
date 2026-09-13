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
import pathlib
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
from engine.base import tenancy                                   # noqa: E402
from engine.base import kernel_shape                              # noqa: E402
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
from engine.profiles.glm53.caches import (Glm53Caches, layout, snapshot_layout, stage_bytes,
                                        cache_capacity, state_dtype)   # noqa: E402
from engine.profiles.glm53 import drafter as drafter_mod           # noqa: E402
from engine.profiles.glm53.adapter import Glm53Engine, NullDrafter             # noqa: E402
from engine.profiles.glm53.net import Glm53Net                   # noqa: E402
from engine.profiles.glm53.weights import rank_loader            # noqa: E402
from engine.profiles.glm53 import vision as vision_mod           # noqa: E402

GIB = 1 << 30
KV_GIB = 24.0                       # production parity (vLLM's 24.02 GiB/rank, 28차 §8); the ST budget table leaves 41.6 GiB, 45차 §23
TOKEN_BUDGET = 32768                # fourteen aligned blocks -> 32,256 prefill tokens after draft reservation
# The previous KV2 consumer measured <=4.23 GiB of prefill workspace.
# The unchanged 12 GiB ceiling must qualify this larger expert batch; that
# earlier measurement is not proof that the new allocation fits.
# Boot must qualify the new largest shape at both ends of the actual KV pool;
# throughput and answer quality still require the candidate consumer gate.
MAX_WAIT_S = 0.0                    # admit into a free decode row at the next chunk boundary
MAX_SEQS = 4
"""Resident decode rows: state slots, captured decode widths and the context ceiling follow.

8 was chosen for kernel coverage (48 target tokens at K=5, which mHC and one-shot reach) and
nothing else, and no release has ever served it -- every production release pins 4. Measured
side by side from two boot ledgers on the same commit (45차 §72, 2026-09-12):

    width          graph pool   captured ceiling   target graphs   boot     state slots
    1-4 (this)       0.60 GiB          1,035,264              36   161.9 s     1.21 GiB
    1-8              2.50 GiB            364,032              64   208.7 s     2.17 GiB

2.86 GiB of a box that reached 7.09 GiB free during capture, and 47 s of boot, to buy a
concurrency nothing serves. And the rows are not free of each other: the state slots come out
of the same `kv_gib`, so at 7.0 the pool is 1,314 blocks at four rows and 1,095 at eight
(budget.budget, same argument). Fewer blocks is a shorter longest sequence, which is why the
captured ladder tops out lower -- the width that was supposed to serve more requests serves
each of them less context.

Coverage still holds at 4: 24 target tokens is inside the same kernels. The repo default and
what production serves are now one number -- they disagreed, and that is exactly how two
onepass runs 27 minutes apart on one commit came out incomparable (45차 §72).
"""
PREFIX_TIER_STAGE = 32 << 20        # the prefix tier's pinned staging + device scratch
TIER_GIB = 64.0
"""What a rank's parked conversations may occupy on NVMe, and its evicted prefix boundaries below.

Declared, because "the filesystem decides" is not a decision (D1). Until 45차 §53 neither tier
had a capacity at all, so the only brake was `reserve_bytes` -- one gigabyte of free space --
on a root that also carries the checkpoints, the images and the logs. It had eaten 75 GiB of a
disk that was 99% full, and nothing in the engine had ever deleted a byte of it.

A parked conversation is ~260 MiB here (one block plus its 247 MiB state slot, the size 45차
§49 left open), so 64 GiB is about 250 of them and 16 GiB is about 30 prefix boundaries. The
prefix tier gets the smaller share on purpose: a boundary is a cache that recomputes, a
conversation is a turn the user may come back to (D16). Past the cap the LRU forgets, foreign
layouts first (`NvmeTier.oldest`).
"""
PREFIX_TIER_GIB = 16.0
TIER_RESERVE_GIB = 16.0             # free space a tier leaves on the filesystem whatever its own cap allows
PREFIX_SNAPSHOT_GIB = 2.125
PREFIX_UNTIERED_SNAPSHOT_GIB = 4.25
PREFIX_COMPRESSED_BYTES = 1 << 30
"""Tiered serving keeps 48 native raw snapshots instead of 96, plus at most
1 GiB of compressed cold snapshots. Original FP32/BF16 bits are preserved.
The existing prefix tier owns asynchronous spill/restore and a durable NVMe
copy. Its compressed cache only changes where a restore reads, never rank
ownership. Untiered local runs retain the original raw budget.

At least nine raw slots remain for a 6,912-token chunk. Compression ratios
are data dependent; the cap includes entries being built. The codec uses
bounded chunks through existing tier staging, plus its declared workspace.
"""


def prefix_host_bytes(tier_enabled: bool = True) -> int:
    from engine.base.compressed_snapshots import WORKSPACE_BYTES
    return PREFIX_COMPRESSED_BYTES + WORKSPACE_BYTES if tier_enabled else 0


def snapshot_count(snapshot_bytes: int, gib: float = PREFIX_SNAPSHOT_GIB) -> int:
    """How many boundaries fit the declared budget. At least a chunk's worth (nine blocks)."""
    return max(9, int(gib * GIB) // max(1, int(snapshot_bytes)))


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
REASONING_EFFORT_ALIASES = {"max": "high"}       # accept existing clients while capping this model at high
REASONING_END = "</think>"                       # the model closes its reasoning with this token; the door splits content there
REQUEST_TIMEOUT_S = 3600.0                       # a request older than this is cancelled (the production probe's long-ingest bound x12)


def chat_renderer(ckpt=facts.CKPT):
    """messages -> prompt text through the checkpoint's chat template (the door's OpenAI chat endpoint). transformers'
    template engine renders it (the template needs its filters); chat_template_kwargs (`thinking`, ...) pass through."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(ckpt))
    template = Path(ckpt) / CHAT_TEMPLATE
    if not template.exists():
        # NVIDIA ships only chat_template.jinja, which always opens <think>.
        # ST's bundled template preserves thinking=False and multimodal/tools
        # semantics for both checkpoint encodings, including direct boot.py.
        template = Path(__file__).resolve().parents[3] / 'launchers' / CHAT_TEMPLATE
    t.chat_template = template.read_text()

    def render(messages, kwargs, *, generation_prompt: bool = True, continue_final: bool = False):
        """`continue_final` resumes inside the last assistant turn instead of opening a new
        one, which is what a caller wants when it is handing back a partial answer to extend.
        It is passed only when asked for, so a template engine without it keeps working."""
        resume = {"continue_final_message": True} if continue_final else {}
        # Keep the profile default even if the checkpoint has an older template.
        if kwargs.get("reasoning_effort") in (None, "max"):
            kwargs = {**kwargs, "reasoning_effort": "high"}
        elif kwargs["reasoning_effort"] not in ("low", "high"):
            raise ValueError("GLM-5.3-Flash reasoning_effort must be low, high, or max")
        return t.apply_chat_template(messages, add_generation_prompt=generation_prompt,
                                     tokenize=False, **resume, **kwargs)
    return render


def eos_ids(ckpt=facts.CKPT) -> "list[int]":
    import json
    g = json.loads((Path(ckpt) / "generation_config.json").read_text())
    e = g.get("eos_token_id", [])
    return list(e) if isinstance(e, list) else [e]


def declared(a, comm_world: int) -> Config:
    """Native execution and the qualified large-prefill MLA lane are fixed.

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
        Fact("max_seqs", MAX_SEQS, "resident rows, state slots and captured decode widths"),
        Fact("prefix_snapshot_gib", PREFIX_SNAPSHOT_GIB, "tiered raw block-boundary checkpoints; count follows the shape"),
        Fact("prefix_untiered_snapshot_gib", PREFIX_UNTIERED_SNAPSHOT_GIB, "raw checkpoints when no prefix tier is configured"),
        Fact("prefix_compressed_bytes", PREFIX_COMPRESSED_BYTES, "prefix tier's bounded lossless RAM cache; codec workspace declared separately"),
    ]
    fixed = dict(moe_static=lane_tables.MOE_STATIC_PRODUCTION,
                 lanes="served", decode_eager=0, execution="native")
    facts_ += [Fact(k, v, "native TP4 execution") for k, v in fixed.items()]
    # One serving recipe in both modes. These four were enabled by operator
    # request; their component gates are not full-model performance proof.
    gb10_defaults = dict(direct_mhc=1, prefill_project_tiles=1,
                         nvme_mapped_staging=1, decode_iterations=4)
    if getattr(a, "production", False):
        # tile32 passed the full GPU numerical/graph and matched 2K/32K/128K
        # serving brackets. Keep it in the production contract so a stale
        # STK_* environment cannot silently restore the stock long-prefill
        # path.
        defaults = dict(mla_prefill="tile32", context_ceiling=0, kda_state_dtype=facts.KDA_STATE_DTYPE,
                        execution_overlap=0, early_observe=0, prefill_tiles=1, **gb10_defaults)
        return Config(facts_ + [Fact(k, v, "production default") for k, v in defaults.items()], knobs=[])
    knobs = [
        Knob("decode_iterations", gb10_defaults["decode_iterations"], _dt.date(2026, 9, 30),
             "Bounded greedy TP4 decode: reserve/read back 2 or 4 iterations with rank-agreed exits",
             "STK_decode_iterations=1", int),
        Knob("nvme_mapped_staging", gb10_defaults["nvme_mapped_staging"], _dt.date(2026, 9, 30),
             "One mapped GB10 staging allocation for NVMe host I/O and GPU gather/scatter",
             "STK_nvme_mapped_staging=0", int),
        Knob("prefill_project_tiles", gb10_defaults["prefill_project_tiles"], _dt.date(2026, 9, 30),
             "Overlap TP4 prefill tile arrival with independent KDA input projection",
             "STK_prefill_project_tiles=0", int),
        Knob("direct_mhc", gb10_defaults["direct_mhc"], _dt.date(2026, 9, 30),
             "TP4 rank packets consumed inside native MHC; exact rounding, C=1/C=4 latency and quality",
             "STK_direct_mhc=0", int),
        Knob("execution_overlap", 0, _dt.date(2026, 9, 30),
             "GB10 C=4 ordered TP/compute overlap: matched onepass C=1/C=4 latency and acceptance",
             "STK_execution_overlap=0", int),
        Knob("early_observe", 0, _dt.date(2026, 9, 30),
             "DFlash2 context preparation overlaps target tail using private W4 scratch; FP32 state unchanged",
             "STK_early_observe=0", int),
        Knob("prefill_tiles", 1, _dt.date(2026, 9, 30),
             "layer-major prefill windows of 1/2/4 native tiles, separate from decoding; 32K/128K TTFT and memory",
             "STK_prefill_tiles=1", int),
        Knob("kda_state_dtype", facts.KDA_STATE_DTYPE, _dt.date(2026, 9, 30),
             "FP16 recurrent storage with FP32 arithmetic: matched C=1/C=4 onepass quality, latency and memory",
             "STK_kda_state_dtype=fp32", state_dtype),
        Knob("mla_prefill", "tile32", _dt.date(2026, 9, 30),
             "large-M MLA prefill: tile32 is the qualified production default, stock the baseline "
             "(the pair/pair4 union candidates were measured and retired -- 45차 §23 조사 16차, PR #698)",
             "STK_mla_prefill=tile32"),
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
          ckpt_meta: "str | Path" = facts.CKPT, drafter_dir: "str | Path" = drafter_mod.DRAFTER,
          lease_owner: "str | None" = None, kda_state_dtype: "str | None" = None, execution_plan=None,
          nvme_mapped_staging=False):
    """`ckpt_meta`: where config.json / tokenizer.json / generation_config.json are -- the HF checkpoint dir, or a
    copy of just those files: a node needs its rank file, the drafter and this, not the 185 GB checkpoint."""
    F = facts.load(ckpt_meta)
    if kda_state_dtype is not None:
        from dataclasses import replace
        F = replace(F, kda_state_dtype=state_dtype(kda_state_dtype))
    if execution not in ("stock", "native"):
        raise ValueError("execution must be stock or native")
    if type(nvme_mapped_staging) is not bool or (nvme_mapped_staging and execution != "native"):
        raise ValueError("mapped NVMe staging requires the native GB10 profile")
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    drafter_dir = Path(drafter_dir)
    D = drafter_mod.load(drafter_dir) if use_drafter else None
    if D:
        # the draft kernels admit this head width (base/kernel_shape.drafter), bound once the drafter's facts are known
        kernel_shape.bind_drafter(kernel_shape.Drafter(head_dim=D.head_dim, kv_heads=D.kv_heads, layers=D.layers, window=D.window))
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
    # Keep FP32's capacity: narrower state must return bytes, not buy more KV.
    nb, snapshots = cache_capacity(F, net.layers, draft_shape, kv_gib, max_seqs,
                                   PREFIX_SNAPSHOT_GIB if tier_dir else PREFIX_UNTIERED_SNAPSHOT_GIB)
    if nb < 2:
        raise MemoryError(f"KV {kv_gib} GiB leaves {nb} blocks after {ns} slots of {sb / 2**20:.0f} MiB")
    rank = rank_loader(Path(ranks_dir) / f"rank{comm.rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    recorder.gauge('weight_layout', F.weight_layout)
    snapshot_bytes = snapshot_layout(F, net.layers, draft_shape)[0]
    reference_snapshot_bytes = snapshot_layout(F, net.layers, draft_shape, state_storage="fp32")[0]
    recorder.gauge("kda_state_dtype", F.kda_state_dtype)
    saved = ns * (layout(F, net.layers, draft_shape, state_storage="fp32").slot_bytes - sb)
    saved += snapshots * (reference_snapshot_bytes - snapshot_bytes)
    saved += ns * (snapshot_layout(F, net.layers, state_storage="fp32")[0]
                   - snapshot_layout(F, net.layers)[0])
    recorder.gauge("kda_state_storage_saved_bytes", saved)
    host_budget_bytes = prefix_host_bytes(bool(tier_dir))
    # the vision tower (45차 §23 A7): whole on every rank, from vision.safetensors next to the rank files (preshard.py --vision);
    # absent, the door refuses pictures -- the fleet boot requires it (production serves images, PR #431)
    vision_file = Path(ranks_dir) / vision_mod.FILE
    VF = vision_mod.load(ckpt_meta) if vision_file.exists() else None
    vspecs = vision_mod.specs(VF) if VF else []
    # Self-calibration (kernels/dense/calibration): the packs the store cannot build GPTQ, summed from this boot's
    # serving, drafter first, within a fixed budget of the arena; the next boot packs GPTQ from the blobs.
    store = calib_plan = None
    calib_bytes = 0
    if execution == "native":
        from engine.kernels.dense.calibration import BUDGET_BYTES, Calibration
        from engine.kernels.dense.store import PackStore
        store = PackStore("/cache", comm.rank)
        calib_plan = []                                                   # (module, weight key, store name, missing tiles, small rows)
        if D:
            for key, (_rows, cols) in drafter_mod.dense_shapes(D, comm.world_size).items():
                missing = store.missing_calibration(drafter_mod.store_name(key), cols)
                if missing:
                    calib_plan.append(("drafter", key, missing, True))
        shapes = {sp.name: sp.shape for sp in specs}
        for key, name in net.dense_weight_names(shapes).items():
            missing = store.missing_calibration(name, shapes[key][1])
            if missing:
                calib_plan.append(("target", key, missing, False))
        from engine.profiles.glm53.net import HEAD_NAME
        missing = store.missing_calibration(HEAD_NAME, shapes["head"][1])
        if missing:
            calib_plan.append(("target", "head", missing, True))      # its rows are the batch's own, prefill and decode alike
        budget = BUDGET_BYTES
        for _module, _key, missing, _small in calib_plan:
            from engine.kernels.dense import DenseLinear
            need = Calibration.nbytes(missing, max_decode_rows=max_seqs * (D.k + 1 if D else 1) if _small else 0,
                                      input_dtype=DenseLinear.input_dtype if _module == "drafter" else torch.float32)
            if need <= budget:
                budget -= need
                calib_bytes += need
    router_bytes = net.router_nbytes() if execution == "native" else 0
    projection_bytes = net.decode_projection_nbytes() if execution == "native" else 0
    draft_bytes = total_bytes(dspecs)
    if D and execution == "native":
        from engine.profiles.glm53.drafter_storage import nbytes as draft_resident_bytes
        draft_bytes = draft_resident_bytes(D, comm.world_size, max_seqs)
        recorder.gauge("drafter_source_bytes", total_bytes(dspecs))
        recorder.gauge("drafter_resident_bytes", draft_bytes)
        recorder.gauge("drafter_arena_saved_bytes", total_bytes(dspecs) - draft_bytes)
    arena_bytes = (total_bytes(specs) + draft_bytes + total_bytes(vspecs) + router_bytes + projection_bytes + 256 * (len(specs) + len(dspecs) + len(vspecs) + 64)
                   + cache_layout.nbytes(nb, max_seqs) + snapshots * snapshot_bytes + stage_bytes(F, net.layers, max_seqs) + calib_bytes)
    memory = None
    redeclare = None                    # the same table, re-runnable once a ledger exists (45차 §51)
    if len(net.layers) == F.layers:
        # Fixed byte ceilings, not a measured workspace claim. Preparation
        # records peaks for the largest prefill and every declared graph.
        # One source, not two: the same literals lived here and in budget.py, and the budget
        # table is what anyone reads to decide whether a boot fits. The reserve is the box's
        # own kill line plus a margin (budget.os_reserve_gib), not a number we picked.
        from engine.profiles.glm53 import budget as _budget_mod
        workspace_bytes = int(_budget_mod.WORKSPACE_GIB * GIB)
        os_reserve_bytes = int(_budget_mod.OS_RESERVE_GIB * GIB)
        files = sorted(Path(ranks_dir).glob("rank*of4.safetensors"))
        if D:
            files.append(drafter_dir / "model.safetensors")
        if VF:
            files.append(vision_file)
        failure = None
        try:
            report = prepare_allocation(arena_bytes, files, workspace_bytes + os_reserve_bytes + host_budget_bytes,
                                        lambda: torch.cuda.mem_get_info()[0],
                                        cache_roots=(Path(ranks_dir).parent, drafter_dir.parent))
            memory = RuntimeMemory(arena_bytes, workspace_bytes, os_reserve_bytes, comm=comm,
                                   host_budget_bytes=host_budget_bytes,
                                   reclaim=partial(reclaim_preparation_pages,
                                                   cache_roots=(Path(ranks_dir).parent, drafter_dir.parent)))
        except (MemoryError, OSError, RuntimeError) as exc:
            failure = exc
        # A failed rank must prevent peers from starting their large CUDA
        # allocations; closing NCCL only after one rank fails is too late.
        comm.wait_prepared("arena-admission")
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
        redeclare = partial(budget_mod.budget, kv_gib, max_seqs,
                            chunk=sched.chunk_for(F.chunk_align, TOKEN_BUDGET, D.k if D else 0), ckpt=ckpt_meta,
                            ranks_dir=ranks_dir, rank=comm.rank, drafter_dir=drafter_dir if D else None,
                            snapshots=snapshots, tier_enabled=bool(tier_dir), kda_state_dtype=F.kda_state_dtype,
                            draft_tp=comm.world_size if execution == "native" else 1,
                            draft_native=execution == "native", router_bytes=router_bytes, projection_bytes=projection_bytes)
        # With THIS boot's floor, not vLLM's 40th-boot constant. RuntimeMemory measured it
        # seconds ago in __init__, and this print is the moment anyone decides how much KV to
        # ask for: without it the first table said 42.77 GiB of KV remained on a box that had
        # 9.27, because the 33.50 GiB already on the node was nowhere in it (2026-09-12).
        b = redeclare(ledger=memory.report())
        recorder.gauge("budget_unassigned_GiB", round(b.kv_gib - b.kv_declared_gib, 2))
        recorder.gauge("budget_tenant_floor_GiB", round(memory.floor_bytes / GIB, 2))
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
        def load_drafter():
            with recorder.phase("load drafter"):
                # Native packs and surviving BF16 readers are copied into their
                # compact region below. The full checkpoint is temporary scratch.
                dviews = RankLoader(drafter_dir / "model.safetensors").load(
                    [s.name for s in dspecs], arena=None if execution == "native" else arena, recorder=recorder)
            result = drafter_mod.Drafter(D, net, decodable)
            result.bind(dviews)
            return result  # the loader's mapping must not outlive this call
        if D and execution != "native":
            drafter = load_drafter()
        calibration = None
        if execution == "native":
            from engine.kernels.prefill_collectives import PrefillCollectives
            with recorder.phase("prepare native execution"):
                net.prepare_routers(arena)
                recorder.gauge('router_resident_bytes', router_bytes)
                net.prepare_dense(store, consume_weights=True)
                net.prepare_decode_projections(arena)
                recorder.gauge('decode_projection_resident_bytes', projection_bytes)
                net.prefill_transport = PrefillCollectives(comm, project_tiles=bool(
                    execution_plan is not None and execution_plan.prefill_project_tiles))
                if D:
                    # Do not overlap the temporary checkpoint with target packing.
                    drafter = load_drafter()
                    drafter.prepare_fast(store, max_seqs=max_seqs, compact_into=arena)
                    recorder.gauge("drafter_block_fp8_packs", sum(
                        layer.fp8 is not None for name, layer in drafter.dense.items() if name != "fc.weight"))
            if calib_plan:                                            # this boot sums what the store lacked, within the budget
                calibration = Calibration(torch.device("cuda"), BUDGET_BYTES, arena=arena,
                                          max_decode_rows=max_seqs * (1 + drafter.k))
                for module, key, missing, small in calib_plan:
                    layer = (drafter if module == "drafter" else net).dense[key]
                    calibration.attach(layer.name, layer, missing, small, unsmooth=getattr(layer, "smooth", None))
                recorder.gauge("calibration_blobs", len(calibration.rows))
                recorder.gauge("calibration_hessians", len(calibration.H))
                recorder.gauge("calibration_deferred", len(calibration.deferred))
            for name, count in store.stats.items():
                recorder.gauge("dense_pack_"+name, count)
            layers = list(net.dense.values()) + (list(drafter.dense.values()) if D else [])
            recorder.gauge("dense_calibrated", sum(1 for layer in layers if getattr(layer, "calibrated", False)))
            recorder.gauge("dense_fp8_calibrated", sum(1 for layer in layers if getattr(getattr(layer, "fp8", layer), "calibrated", False)))
            recorder.gauge("dense_smoothed", sum(1 for layer in layers if getattr(layer, "smooth", None) is not None))
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
            caches = Glm53Caches(arena, F, net.layers, nb, max_seqs, draft=draft_shape, snapshots=snapshots, stage=True)
        with recorder.phase("engine"):
            # the aux layers must lie inside the chain: a layer subset (the local smoke) clips them to its last layer -- plumbing only
            aux = [min(L, net.layers[-1]) for L in drafter.aux_layers] if D else None
            engine = Glm53Engine(net, caches, F, drafter, max_new=max_new, eos_ids=eos_ids(ckpt_meta), temperature=temperature, seed=seed,
                                 decodable=decodable, aux_layers=aux, context_ceiling=context_ceiling,
                                 execution_plan=execution_plan)
            plan = engine.execution_plan
            token_budget = plan.tile_rows * plan.prefill_tiles + drafter.k if plan.prefill_tiles > 1 else TOKEN_BUDGET
            contract = sched.Contract(chunk_align=F.chunk_align, token_budget=token_budget, draft_slots=drafter.k,
                                      max_wait_s=MAX_WAIT_S, max_running=max_seqs,
                                      decode_token_budget=F.chunk_align + drafter.k,
                                      prefill_tail_multiple=facts.TP)
            engine.memory = memory
            engine.budget = redeclare           # printed once from guesses at boot, once from this boot's ledger
            engine.arena = arena                # every device tensor is a view of it: `release` needs the last reference
            engine.vision = vision
            engine.calibration, engine.calibration_root = calibration, (str(store.root) if store is not None else None)
            engine.pack_stats = dict(store.stats) if store is not None else {}
            if store is not None:
                dense_layers = list(net.dense.values()) + (list(drafter.dense.values()) if D else [])
                engine.pack_stats["fp8_gptq"] = sum(1 for layer in dense_layers if getattr(getattr(layer, "fp8", layer), "calibrated", False))
                engine.pack_stats["smoothed"] = sum(1 for layer in dense_layers if getattr(layer, "smooth", None) is not None)
            engine.prefill_chunk = sched.chunk_for(contract.chunk_align, contract.token_budget, contract.draft_slots)
        # Per-rank GPTQ/calibration caches can take very different times to
        # prepare. A fast rank used to enqueue the memory vote while a peer
        # still packed weights, exhausting NCCL's 120 s serving deadline.
        # Meet on the CPU control plane before any post-load device collective.
        with recorder.phase("wait for weight preparation"):
            comm.wait_prepared("weights-loaded", final=True)
        if memory is not None:
            memory.checkpoint("loaded")
        with recorder.phase("runner"):
            tiered = prefix_tier = None
            if tier_dir:                                                                # D16: idle conversations park on NVMe, per rank
                if lease_owner:
                    # A restart keeps its conversations (D16). A HANDOVER does not: the previous holder's
                    # clients are gone and its prefix tier is warm with boundaries this one never computed,
                    # which is poison for anything anybody measures next (base/tenancy).
                    left = tenancy.claim(Path(tier_dir) / f"rank{comm.rank}", lease_owner)
                    if left:
                        print(f"  rank{comm.rank}: tenant state cleared -- the fleet changed hands from {left}")
                # Missing format tags name historical FP32 bytes. FP16 cannot
                # discover or restore those conversations/prefix snapshots.
                state_format = "glm53-kda-fp16-v1" if F.kda_state_dtype == "fp16" else ""
                tier = NvmeTier(Path(tier_dir) / f"rank{comm.rank}", block_bytes=cache_layout.block_bytes,  # a block is one NVMe unit (block-major)
                                capacity_bytes=int(TIER_GIB * GIB), reserve_bytes=int(TIER_RESERVE_GIB * GIB),
                                state_format=state_format, mapped_staging=nvme_mapped_staging)
                tiered = TieredKV(caches.pool, tier)
                # the prefix tier (45차 §23 A): evicted leaf boundaries -- their blocks and snapshot -- live on beside the parked
                # conversations, in their own directory and keyspace (a boundary's key is 56 bits of its hash)
                prefix_tier = TieredKV(caches.pool, NvmeTier(Path(tier_dir) / f"rank{comm.rank}" / "prefix",
                                                             block_bytes=cache_layout.block_bytes, stage_bytes=PREFIX_TIER_STAGE,
                                                             capacity_bytes=int(PREFIX_TIER_GIB * GIB),
                                                             reserve_bytes=int(TIER_RESERVE_GIB * GIB),
                                                             snapshot_cache_bytes=PREFIX_COMPRESSED_BYTES,
                                                             state_format=state_format, mapped_staging=nvme_mapped_staging))
                recorder.gauge("nvme_mapped_staging", int(nvme_mapped_staging))
                recorder.gauge("nvme_staging_saved_bytes", (tier.stage_bytes + prefix_tier.tier.stage_bytes
                               - tier.staging_padding_bytes - prefix_tier.tier.staging_padding_bytes)
                               if nvme_mapped_staging else 0)
            prefix = PrefixCache(F.block, engine.prefill_chunk, snapshots)      # boundaries = every 768 block (base/prefix.py)
            # Reducing hot slots must not also halve the metadata budget for
            # cold boundaries whose KV blocks remain reusable.
            prefix.max_faded = 4 * snapshot_count(reference_snapshot_bytes, PREFIX_UNTIERED_SNAPSHOT_GIB)
            runner = Runner(engine, contract, caches.pool, caches.slots, Ring(4096, STEP_RECORD.size), recorder, tiered=tiered,
                            keep_idle=tiered is not None, prefix=prefix)                # with a tier, conversations live on and park
            runner.prefix_tier = prefix_tier
        recorder.gauge("blocks", nb); recorder.gauge("slots", ns); recorder.gauge("arena_GiB", round(arena.used / GIB, 3))
        recorder.gauge("prefix_compressed_budget_bytes", host_budget_bytes)
        recorder.gauge("prefix_snapshots", snapshots); recorder.gauge("snapshot_MiB", round(snapshot_bytes / 2**20, 1))
        return F, net, caches, engine, runner
    except BaseException as exc:
        # The peers wait at "weights-loaded" for up to 1800 s, and what ends their wait is the
        # deadline. A rank that failed on the way meets them there instead, with a phase that says
        # so: the rendezvous compares phases and every rank raises now, naming this one's error.
        if getattr(comm, "preparation", None) is not None:
            try:
                comm.wait_prepared(f"failed: rank {comm.rank}: {type(exc).__name__}: {str(exc)[:200]}", timeout_s=60., final=True)
            except BaseException:                     # noqa: BLE001 -- it raises by design; `exc` is the cause
                pass
        if memory is not None:
            memory.close()
        raise


CLEAN_RELEASE_BYTES = 64 << 20
"""What may still be ALLOCATED on the device after a release and still count as clean.

Not a tolerance for leaks -- a floor under the things a CUDA process keeps for as long as it
has a context: NCCL's own buffers, the runtime memory gate's status word, whatever a kernel
module built once at import. Anything above it is a holder `Glm53Engine.release` did not find,
and the block sizes printed beside it are the lead (45차 §52).
"""


def release_all(engine: Glm53Engine, runner: "Runner | None" = None) -> dict:
    """Hand the box back. The tiers go first: their staging is outside the arena, so nothing
    the engine does can reach it. Never raises -- a shutdown does not fail a shutdown."""
    staging = 0
    for tier in (getattr(runner, "tiered", None), getattr(runner, "prefix_tier", None)):
        if tier is None:
            continue
        try:
            staging += tier.close()
        except Exception as exc:                            # noqa: BLE001
            print(f"  released: a tier could not close: {type(exc).__name__}: {exc}", flush=True)
    report = engine.release()
    report["tier_staging_bytes"] = staging
    return report


def release_line(report: dict, rank: int = 0) -> str:
    """What came back, and -- when something did not -- the sizes of what stayed."""
    line = (f"  released: rank {rank} gave back {report['returned'] / GIB:.2f} GiB of "
            f"{report['arena_bytes'] / GIB:.2f} GiB arena plus {report.get('tier_staging_bytes', 0) / 2**20:.0f} MiB "
            f"of tier memory; {report['reserved_after'] / GIB:.2f} GiB reserved and "
            f"{report['allocated_after'] / 2**20:.0f} MiB allocated still")
    if report["allocated_after"] > CLEAN_RELEASE_BYTES:
        held = ", ".join(f"{n / 2**20:.0f} MiB" for n in report.get("still_held") or ())
        line += (f"\n  released: rank {rank} did NOT come back clean -- "
                 f"{report['allocated_after'] / 2**20:.0f} MiB is still held by live tensors"
                 + (f", largest blocks {held}" if held else ""))
    return line


def tier_line(tier, cap_gib: float, what: str) -> str:
    """What is on the disk, in bytes -- the boot used to print counts and leave the size a mystery."""
    live = sum(1 for k in tier.index if tier.has(int(k)))
    stale, stale_bytes = len(tier.stale()), tier.stale_bytes()
    line = (f"  NVMe tier: {live} {what} parked from before, {tier.used_bytes() / GIB:.1f} GiB of "
            f"{cap_gib:.0f} GiB")
    if stale:
        line += (f"; {stale} under another layout holding {stale_bytes / GIB:.1f} GiB -- not resumable, "
                 f"and the first thing forgotten when the cap bites")
    return line


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
                 target_fp8=sum(bool(p.executed & 2) for p in target),
                 head_fp8=net.dense['head'].executed,
                 drafter_w4=sum(bool(p.executed & 1) for p in draft),
                 drafter_context_fp8=bool(drafter.dense['fc.weight'].executed & 2),
                 mhc=len(net.mhc.executed),
                 shared_mlp=sum(p.executed for p in net.shared_mlp.values()),
                 shared_overlap=bool(net.shared_overlap and net.shared_overlap.executed),
                 router_tensorcore=len(net._router_tensorcore),
                 prefill_collectives=sorted(net.prefill_transport.executed))
    if (proof['target_w4'] != len(target) or proof['target_fp8'] != len(target)
            or proof['drafter_w4'] != len(draft) or not proof['head_fp8']
            or not proof['drafter_context_fp8'] or proof['mhc'] != expected_mhc
            or proof['shared_mlp'] != len(net.shared_mlp)
            or (net.shared_mlp and not proof['shared_overlap'])
            or net._router_tensorcore != set(net._router_weights)
            or len(proof['prefill_collectives']) != 2):
        raise RuntimeError(f'native execution proof is incomplete: {proof}')
    return proof


# A test boot runs the SAME path production runs -- the served lanes and captured decode graphs -- because a
# boot that decodes eagerly is a different engine and its numbers answer about nothing (45차 §94). What it
# drops is everything the path does not go through, and what it opens is the measurement.
#
#   same      served lanes; decode replays captured graphs, as production's does.
#   off       the qualification a fleet boot owes its door before opening it -- the vision tower, the
#             grammars, the parked-conversation tier, the calibration sums.
#   fast      one captured decode width instead of max_seqs of them. Capture is three quarters of a fleet
#             boot's 87 seconds ([[stkernel-st-boot-time]]), and it is paid per width.
#   open      every step timed instead of one in sixty-four, and /v1/engine/profile for the kernels.
#
# What it is NOT is a speed measurement. Four ranks are four threads on ONE GPU here, so the device does four
# ranks' arithmetic and a step takes what a step takes on this box, not on the fleet. D17 says a change that
# claims speed is not finished until the fleet has measured it, and this mode does not change that.
TEST_WIDTHS = 1            # captured decode widths: production's four cost four captures
TEST_CLOCK_EVERY = 1       # a test boot times every step; production samples one in sixty-four
TEST_FLOOR_GIB = 16.0      # what a --test boot must leave the box, over and above its own KV


def arm_test_measurement(engine, recorder):
    """Capture what production captures, then time every step of it.

    The decode graphs are the point: a boot that decodes eagerly runs different kernels in a different order
    and answers about nothing. One width is captured rather than `max_seqs` of them -- production serves one
    sequence today, and each width is its own capture.
    """
    from engine.base.stage_clock import StageClock
    with recorder.phase("capture decode"):
        engine.capture_decode(TEST_WIDTHS)
    pipeline = getattr(engine, "pipeline", None)
    if pipeline is None:                                   # no drafter: there is no async decode to time
        return None
    pipeline.clock = StageClock(every=TEST_CLOCK_EVERY, device=engine.caches.device)
    return pipeline.clock


def stage_table(clock, steps: int) -> str:
    """What a decode step is made of. These are the stages production exports, sampled at every step."""
    if clock is None or not clock.totals or not clock.samples:
        return "  decode: nothing timed (no drafter, or no decode step ran)"
    total = sum(clock.totals.values())
    lines = [f"  decode, by stage over {clock.samples} of {steps} steps ({total / clock.samples * 1e3:.1f} ms each;"
             f" this box's time, not the fleet's):"]
    for stage, seconds in sorted(clock.totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {stage:12s}{seconds / clock.samples * 1e3:9.2f} ms{seconds / total * 100:8.1f}%")
    return "\n".join(lines)


def memory_left(kv_gib: float) -> float:
    """MemAvailable less the KV this boot declares, in GiB. The weights and the arena come out of the rest."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20 - kv_gib
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def guard_test_memory(kv_gib: float, floor: float = TEST_FLOOR_GIB) -> None:
    """A test boot must not be the thing that takes production down.

    GB10 has one pool for host and device, earlyoom's floor is absolute, and the engine is a preferred kill
    target -- on 2026-09-11 a smoke test beside production killed the fleet's worker, not itself. So this
    refuses rather than guesses (D3), and prints the two numbers the caller needs to make it fit.
    """
    left = memory_left(kv_gib)
    if left < floor:
        raise SystemExit(
            f"  --test refuses: {left:.1f} GiB would be left after this boot's {kv_gib:.1f} GiB of KV and the "
            f"floor is {floor:.1f}. Narrow --layers, lower --kv-gib, or wait for the box. A test boot that "
            f"earlyooms production has not tested anything.")
    print(f"  memory: {left:.1f} GiB left after {kv_gib:.1f} GiB of KV, floor {floor:.1f} -- room for the weights")


def local(a) -> int:
    print(f"  box: {facts.check_box()}")
    print(declared(a, facts.TP).table())
    kernel_shape.bind_recorded(a.ranks, Path(a.ckpt_meta) / "config.json",   # before the lanes, as the fleet boot does
                               lambda: facts.load(a.ckpt_meta).kernel_shape())
    layers = [int(x) for x in a.layers.split("-")]; layers = list(range(layers[0], layers[-1] + 1))
    torch.manual_seed(a.seed)
    prompts = {seq: torch.randint(0, 100_000, (a.prompt + 7 * seq,)).tolist() for seq in range(a.seqs)}
    tp = LocalTP(facts.TP)
    if a.lanes == "served":
        guard_test_memory(a.kv_gib)
        print("  lanes: served, decode captured -- the path production runs, on this box alone. This is not "
              "the fleet and it holds no lease; four ranks are four threads on ONE GPU, so the PATH is "
              "production's and the TIMES are this box's. D17 still wants the fleet for a speed claim.")
    lanes = lane_tables.served() if a.lanes == "served" else lane_tables.reference()
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
        clock = arm_test_measurement(engine, rec) if a.lanes == "served" else None
        t0 = time.perf_counter()
        with rec.phase("generate"):
            out = run_prompts(engine, runner, prompts)
            torch.cuda.synchronize()
        if a.lanes == "served" and comm.rank == 0:
            print(stage_table(clock, runner.steps), flush=True)
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
        report = {"rec": rec, "out": out, "steps": runner.steps, "ring": runner.ring.count, "secs": time.perf_counter() - t0,
                  "kinds": [STEP_RECORD.unpack(r)[2] for r in runner.ring.ordered()], "blocks": caches.pool.available, "slots": caches.slots.available,
                  "accepted": engine.accepted_total, "drafted": engine.drafted_total, "k": engine.drafter.k, "parked": parked}
        report["release"] = release_all(engine, runner)   # last: it forgets every request this dict just read
        return report

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
    dirty = [(rank, o) for rank, o in enumerate(outs) if o["release"]["allocated_after"] > CLEAN_RELEASE_BYTES]
    print(f"  release: {sum(o['release']['arena_bytes'] for o in outs) / GIB:.2f} GiB of arena back over four ranks, "
          f"most still allocated after {max(o['release']['allocated_after'] for o in outs) / 2**20:.0f} MiB "
          f"({'clean' if not dirty else f'{len(dirty)} rank(s) NOT clean'})")
    for rank, o in dirty:
        print(release_line(o["release"], rank).split("\n")[-1])
    ok = ok and not dirty and all(o["release"]["arena_bytes"] > 0 for o in outs)
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
                        reasoning_effort_aliases=REASONING_EFFORT_ALIASES,
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
        # The smoke is the only door that runs a whole serve and then stops on purpose, so it is
        # where the release is gated: every rank hands the box back and says what stayed (45차 §52).
        back = release_all(engine, runner)
        print(release_line(back, comm.rank), flush=True)
        return {"served": server.served, "steps": runner.steps,
                "released": back["arena_bytes"], "allocated_after": back["allocated_after"]}

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
    dirty = [o for o in outs if o["allocated_after"] > CLEAN_RELEASE_BYTES]
    print(f"  release on four ranks: {sum(o['released'] for o in outs) / GIB:.2f} GiB of arena back, "
          f"most still allocated after {max(o['allocated_after'] for o in outs) / 2**20:.0f} MiB "
          f"({'clean' if not dirty else f'{len(dirty)} rank(s) NOT clean'})")
    ok = ok and not dirty and all(o["released"] > 0 for o in outs)
    print("\n  " + ("PASS: requests in at rank 0, tokens out, every rank in lockstep" if ok else "FAIL"))
    return 0 if ok else 1


def fleet_lease_of() -> dict:
    """The reservation this boot holds, or the boot does not happen.

    It used to say "a lease is a reservation, not a dependency" -- the engine served
    with or without one. That made the reservation advisory, and an advisory
    reservation is not one: on 2026-09-12 a yield was asked for and granted, the
    holder parked its conversations and let go, and the containers came straight back
    up on all four nodes because whatever started them never needed the lock. Nobody
    was at fault; there was nothing to be at fault against.

    So it is checked HERE, in the engine, and not only in the launcher. A launcher can
    be bypassed by one `docker run`; the engine is the thing that actually occupies the
    fleet, and the only place a rule about occupying it can be enforced (D3: refuse,
    do not serve anyway). `launchers/start-st-glm53.sh` acquires before it starts, so
    the ordinary path is unaffected -- this only stops the paths that never asked.
    """
    owner, path = os.environ.get("ST_LEASE_OWNER"), os.environ.get("ST_LEASE_PATH")
    if not owner or not path:
        raise RuntimeError(
            "this boot holds no fleet reservation: ST_LEASE_OWNER and ST_LEASE_PATH are unset. "
            "Start through launchers/start-st-glm53.sh, which acquires the lease first; "
            "`start-st-glm53.sh yield` asks a current holder to hand over.")
    from engine.base import fleet_lease
    record = fleet_lease.read(pathlib.Path(path))
    # The lock is ONE file and it lives on the head node -- homes are not shared between the
    # Sparks, which is the whole reason launchers/lib/fleet-lease.sh pipes the module there.
    # So ranks 1..3 read a path that is theirs and empty, and requiring a record of them
    # refused every boot of the fleet: rank 0 came up, the other three exited in under a
    # second (2026-09-12, first boot of this gate on real nodes). What those ranks CAN be held
    # to is the thing a bare `docker run` still would not have -- an owner and a path in the
    # environment, which only the launcher sets -- and agreement with any record they can read.
    rank = os.environ.get("RANK", "0")
    if not record:
        if rank == "0":
            raise RuntimeError(f"the fleet lock at {path} is empty: nothing reserved this boot")
        return {"owner": owner, "path": path}      # the head holds it; this node has no copy to hold
    held = record.get("owner")
    if held != owner:
        raise RuntimeError(
            f"the fleet is reserved by {held!r}, not by this boot ({owner!r}). "
            f"Ask for it: launchers/start-st-glm53.sh yield '<why>'")
    return {"owner": owner, "path": path}


def fleet(a) -> int:
    """One rank per node, inside the glm53 image: served lanes (D3: all or nothing), every layer, then serve."""
    # Before the 67 GiB, not after it: a boot with no reservation must cost nothing. `local` is exempt --
    # it does not take the fleet.
    lease = fleet_lease_of()
    print(f"  fleet reserved by {lease['owner']}")
    print(f"  box: {facts.check_box()}")
    cfg = declared(a, facts.TP)
    # The checkpoint's kernel shape, bound before any transport or lane reads it (base/kernel_shape):
    # the geometry every kernel is admitted for. The record the shape wizard wrote beside the rank
    # files when the model was taken in (preshard) is bound when present and still describes this
    # config.json; without one the shape is derived from the config as before. GLM's equals the
    # kernels' measured cell, so nothing served changes; a checkpoint that differs is refused by the
    # lanes that cannot serve it, by name (D3).
    shape, shape_source = kernel_shape.bind_recorded(a.ranks, Path(a.ckpt_meta) / "config.json",
                                                     lambda: facts.load(a.ckpt_meta).kernel_shape())
    # The rendezvous and the kernel imports are boot time too: 15.6 s of a measured 90.2 s boot sat
    # outside this table (boot-time study, 2026-09-11), so the recorder opens before them.
    rec = Recorder("boot")
    with rec.phase("comm"):
        comm = Comm.init()
    rec.root.name = f"rank{comm.rank}"
    engine = dump = runner = None
    serving = False                                     # the door is open: from here the loop keeps its own books
    try:
        if comm.rank == 0:
            print(cfg.table())
            print(f"  kernel shape ({shape_source}): {shape.describe()}")
        with rec.phase("prepare one-shot"):
            comm.prepare_oneshot()
        with rec.phase("lanes"):
            lanes = lane_tables.served(moe_static=cfg["moe_static"], mla_prefill=cfg["mla_prefill"],
                                       consume_scales=True)
        from engine.profiles.glm53.execution import ExecutionPlan
        if any(cfg[k] not in (0, 1) for k in ("execution_overlap", "early_observe", "direct_mhc", "prefill_project_tiles", "nvme_mapped_staging")):
            raise ValueError("execution switches must be 0 or 1")
        plan = ExecutionPlan(bool(cfg["execution_overlap"]), bool(cfg["early_observe"]), cfg["prefill_tiles"],
                             sched.chunk_for(facts.CHUNK_ALIGN, TOKEN_BUDGET, facts.SPEC_K),
                             direct_mhc=bool(cfg["direct_mhc"]), prefill_project_tiles=bool(cfg["prefill_project_tiles"]),
                             decode_iterations=cfg["decode_iterations"])
        F, net, caches, engine, runner = build(comm, None, lanes, a.ranks, a.kv_gib, MAX_SEQS, True, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed, tier_dir=a.tier_dir,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir,
                                               context_ceiling=cfg["context_ceiling"] or None,
                                               execution=cfg["execution"], lease_owner=lease["owner"],
                                               kda_state_dtype=cfg["kda_state_dtype"], execution_plan=plan,
                                               nvme_mapped_staging=bool(cfg["nvme_mapped_staging"]))

        # "무장 != 서빙": which lanes and kernel cells this process actually bound, readable at
        # scrape time instead of inferred from a boot log nobody kept (45차 §17 lesson).
        engine.lane_info = {"lanes": lanes.name, "moe_static": cfg["moe_static"],
                            "execution_plan": plan.label(),
                            "nvme_mapped_staging": str(cfg["nvme_mapped_staging"]),
                            "kda_state_dtype": F.kda_state_dtype,
                            "mla_prefill": cfg["mla_prefill"], "spec_k": str(engine.drafter.k),
                            "context_ceiling": str(engine.max_context),
                            "packs": f"gptq {engine.pack_stats.get('gptq', 0)} rtn {engine.pack_stats.get('rtn', 0)}",   # what the store built or read
                            "fp8_gptq": str(engine.pack_stats.get("fp8_gptq", 0)),                             # FP8 lane weights GPTQ'd on their grid
                            "smoothed": str(engine.pack_stats.get("smoothed", 0)),                             # inputs' channel smoothing folded into their norms
                            "calibration": engine.calibration.status() if engine.calibration is not None else "complete",
                            "dense_w4a16_guard_rows": str(lane_tables.dense_w4a16_guard_rows())}
        # a stale tier under one rank diverges the ranks (45th 21): find it in seconds, not after the capture
        Server._agree_on_parked(comm, sorted(runner.parked_keys()))
        # and the seed the drafts and samples are drawn from: nothing checked it, and a rank booted by
        # hand with another one would have sampled its own tokens for as long as the fleet stood
        from engine.base.tripwire import Tripwire
        Tripwire.of(comm).agree("boot:seed", [int(a.seed)])
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
        # Vision/grammar qualification can leave several GiB of inactive
        # allocator blocks behind (3.8 GiB on the decode22 boot). Return those
        # once before admission; live tensors and graph pools remain owned.
        with rec.phase("release warmup cache"):
            reserved = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            rec.gauge("production_warmup_cache_returned_bytes", reserved - torch.cuda.memory_reserved())
        engine.memory.checkpoint("production/ready")
        import json
        proof = native_execution_report(net, engine.drafter)
        print('ST_NATIVE_EXECUTION '+json.dumps(dict(rank=comm.rank, **proof)), flush=True)
        engine.memory.write(Path(a.dump_dir) / f"memory-rank{comm.rank}.json")
        if comm.rank == 0 and engine.budget is not None:
            # D1 a second time, with this boot's own numbers. The table above was printed
            # before a byte was allocated, so its two hardest lines were guesses -- a runtime
            # floor carried over from vLLM's 40th-boot table and a 12 GiB workspace CEILING
            # standing in for activations nobody here had measured. The ledger written a line
            # ago has both (45차 §51), and what it costs to say so is one more table.
            from engine.profiles.glm53 import budget as budget_mod
            print(budget_mod.report(engine.budget(ledger=engine.memory.report())))
        dump = DeathDump(a.dump_dir, runner.ring, boot_id=f"glm53-r{comm.rank}-{int(time.time())}")
        if comm.rank == 0:
            print(rec.table())
            print(f"  ST engine: GLM-5.3, TP={facts.TP}, lanes={lanes.name}, KV {a.kv_gib} GiB, serving on :{a.port}")
            if runner.tiered is not None:
                print(tier_line(runner.tiered.tier, TIER_GIB, "conversations"))
                if runner.prefix_tier is not None:
                    print(tier_line(runner.prefix_tier.tier, PREFIX_TIER_GIB, "prefix boundaries"))
        with rec.phase("door"):
            tok = tokenizer(a.ckpt_meta)
            renderer = chat_renderer(a.ckpt_meta) if comm.rank == 0 else None
        from engine.profiles.glm53.tools import parse_tool_calls, partial_tool_calls, tool_call_token, tool_grammar
        if comm.rank == 0:
            print("  warmup: " + ", ".join(f"{k} {v}s" for k, v in paid.items()) + (f"; structured output: {'on' if engine.grammars else 'off (no xgrammar)'}"))
        if engine.calibration is not None:                  # every warm-up and capture is behind us: from here the sums are the served traffic
            engine.calibration.arm()
            print(f"  calibration: rank {comm.rank} summing the inputs of {len(engine.calibration.rows)} uncalibrated pack tiles "
                  f"({len(engine.calibration.deferred)} deferred) -> {engine.calibration_root}/mkcalib/rank{comm.rank}/ "
                  "(filed on its own at 32K rows, at shutdown, or on POST /v1/engine/calibration; the next boot packs GPTQ from them)", flush=True)
        from engine.base.stall import StepWatch
        server = Server(engine, runner, comm, port=a.port, tokenizer=tok, chat=renderer,
               reasoning_effort_aliases=REASONING_EFFORT_ALIASES,
               step_watch=StepWatch(comm.rank, notes_dir=a.dump_dir, dump=dump.write_now),
               model_name="glm-5.3-flash", reasoning_end=tok.token_to_id(REASONING_END), request_timeout_s=REQUEST_TIMEOUT_S,
               tool_parser=parse_tool_calls, tool_stream=partial_tool_calls, tool_grammar=tool_grammar,
                        tool_call_start=tool_call_token(tok), generation=generation_defaults(a.ckpt_meta),
               vision=vision_mod.Door(engine.vision.V, tok) if comm.rank == 0 else None,
               latency_root=Path(a.dump_dir) / 'onepass-latency', lease=lease)
        serving = True
        server.loop()
    except BaseException as exc:
        # Before the door opens every peer is at a ledger vote (a capture's, a qualification's,
        # "production/ready"): a rank that dies here without voting is heard of at NCCL's deadline.
        # One failed vote pairs with their next row and stops every rank now, naming a peer.
        # After the door opens the loop keeps its own books (base/tripwire, base/stall).
        memory = getattr(engine, "memory", None)
        if not serving and memory is not None:
            try:
                memory.checkpoint("boot/failed", failed=f"rank {comm.rank}: {type(exc).__name__}: {str(exc)[:300]}")
            except BaseException:                     # noqa: BLE001 -- it raises by design; `exc` is the cause
                pass
        if not serving:
            from engine.base.tripwire import death_note
            death_note(getattr(a, "dump_dir", None), comm.rank, exc, phase="boot")
        raise
    finally:
        try:
            if dump is not None:
                dump.close()
            if engine is not None:
                try:
                    if engine.memory is not None:
                        engine.memory.write(Path(a.dump_dir) / f"memory-rank{comm.rank}.json")
                    if getattr(engine, "calibration", None) is not None:
                        written = engine.file_calibration()
                        if written:
                            print(f"  calibration: rank {comm.rank} filed {len(written)} blobs under {engine.calibration_root}/mkcalib/rank{comm.rank}/", flush=True)
                finally:
                    # The door is shut and the last blob is filed: hand the box back before NCCL
                    # teardown, not after the process happens to die (45차 §51). A handover has the
                    # next holder already asking for the same 55 GiB. The tiers go first because
                    # their staging lives outside the arena, so nothing else can reach it.
                    try:
                        print(release_line(release_all(engine, runner), comm.rank), flush=True)
                    except Exception as exc:                  # noqa: BLE001 -- a shutdown never fails a shutdown
                        print(f"  released: rank {comm.rank} could not: {type(exc).__name__}: {exc}", flush=True)
                        engine.close_decode()
        finally:
            comm.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="four ranks as threads on this box")
    ap.add_argument("--lanes", choices=("reference", "served"), default="reference",
                    help="with --local: `reference` proves the plumbing, `served` runs the kernels production "
                         "runs -- the only way to judge them without taking the fleet")
    ap.add_argument("--test", action="store_true",
                    help="--local --lanes served: a real engine on one box, no lease, refused if it would "
                         "leave the box under the memory floor")
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
    if a.test:
        a.local, a.lanes = True, "served"
    if a.local:
        if a.kv_gib == KV_GIB:
            a.kv_gib = 1.0                                      # a layer subset on one box
        return local(a)
    return fleet(a)


if __name__ == "__main__":
    raise SystemExit(main())
