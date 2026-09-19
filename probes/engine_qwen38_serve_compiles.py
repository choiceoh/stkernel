"""Qwen3.8 served on one GB10 the way the fleet boots it, then the window's requests: what compiles after the door (probe).

The 2026-09-19 serving window's K=3 boot answered its first requests with six decode steps over 2 s
(measurements/qwen38_serve_window_20260919). Six b12x micro kernels compiled mid-request (warmup.eager_moe now takes
those before the door), but the numbers do not close on them alone: the 3,223-token request and a 474-token one each
held a step over 2 s with no CuTe compile in the log. The boot's warm passes run prefill widths (64 and up) through the
target and the MTP head; nothing before the door runs an eager step of decode size -- the MTP head observing a parked
row's 1..K+1 positions -- and a Triton kernel is compiled per specialization of its arguments (a size of 1, a multiple of
16). So: what does a served request still compile?

This builds the served model for ONE rank of TP=4 from the rank file's own weights -- layers 1 (the PLE injection and a
GDN layer) and 7 (a QSA layer), the MTP head, K=3, four rows -- in the fleet boot's order (engine/profiles/qwen38/fleet
.build): the eager MoE warm pass, the prefill warm passes, the decode graphs captured; then drives base/runner.Runner
with the window's seven requests (random ids of the same prompt lengths, the same generated lengths, greedy, C=1, a
prefix cache), and around every runner step counts every Triton JIT function's compiled kernels and the b12x
dispatcher's kernel caches. A step that added a kernel is reported with the kernel's function and specialization.

    python3 probes/engine_kernel_check.py --lanes qwen38_serve_compiles --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-serve-compiles.json                                (the queue's single-GPU lane)

One rank, two layers: a 48-layer rank compiles the same kernels (every layer of a type takes the same shapes), not
more. OneRankComm: no collective. The PLE table is zeros. Step times are this small net's and are not a claim (D17);
what is recorded is which kernels appeared after the door and at which step.

"Appeared" is a kernel's first use in this process: Triton and the b12x getters put a kernel read back from their disk
caches into the same in-process caches as one they compiled. On the lane the disk caches (/cache) outlive a run, so a
kernel an earlier run built appears here as a read of milliseconds, where a fleet node without it compiles for seconds:
the list is what a tree's first boot would build after its door; the step's seconds say which kind this run saw.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAYERS = (1, 7)
SPEC_K = 3                     # the operator's K (#1182); `--lanes qwen38_serve_compiles:1` for the checkpoint's
MAX_SEQS = 4                   # the launcher's default rows
BLOCKS = 24                    # KV pages of 768 tokens: every request here fits with the draft chain's reach
REQUESTS = (("17x23", 28, 4), ("capital", 30, 21), ("sky", 27, 256), ("transformer", 25, 474),
            ("long-summary", 3223, 96), ("transformer-again", 25, 450), ("sky-again", 27, 256))
"""(name, prompt tokens, generated tokens): the window's K=3 requests (measurements/qwen38_serve_window_20260919)."""


class Census:
    """Every compiled kernel the process holds, counted cheaply per step and named when the count moves."""

    def __init__(self):
        from engine.kernels.b12x import moe_dispatch as md
        self.md = md
        self.caches = {name: getattr(md, name) for name in dir(md) if name.endswith("_KERNEL_CACHE")
                       and isinstance(getattr(md, name), dict)}
        self.jit = []
        self.sweep()
        self.seen = self.keys()

    def sweep(self) -> int:
        """Find every Triton JIT function alive now (a module imported mid-request brings new ones)."""
        from triton.runtime.jit import JITFunction
        known = {id(fn) for fn in self.jit}
        fresh = [o for o in gc.get_objects() if isinstance(o, JITFunction) and id(o) not in known]
        self.jit.extend(fresh)
        return len(fresh)

    def count(self) -> int:
        n = sum(len(cache) for cache in self.caches.values())
        for fn in self.jit:
            for entry in getattr(fn, "device_caches", {}).values():
                n += len(entry[0])
        return n

    def keys(self) -> set:
        out = {(f"b12x:{name}", repr(key)[:400]) for name, cache in self.caches.items() for key in cache}
        for fn in self.jit:
            for entry in getattr(fn, "device_caches", {}).values():
                out |= {(f"triton:{getattr(fn, '__name__', '?')}", repr(key)[:400]) for key in entry[0]}
        return out

    def new(self) -> list:
        now = self.keys()
        added, self.seen = sorted(now - self.seen), now
        return [{"kernel": name, "key": key} for name, key in added]


def build(ranks: Path, rank: int, layers=LAYERS, *, spec_k: int = SPEC_K):
    """fleet.build's order on one rank: net, weights, PLE (zeros), dense packs, caches, model, contract, runner -- then
    the eager MoE warm pass, the prefill (and head) warm passes and the capture -> (F, net, caches, model, runner, boot seconds by phase)."""
    import inspect
    import torch
    from engine.base import scheduler as sched
    from engine.base.arena import Arena
    from engine.base.params import total_bytes
    from engine.base.prefix import PrefixCache
    from engine.base.record import Ring
    from engine.base.runner import STEP_RECORD, Runner
    from engine.profiles.qwen38 import facts, lanes as lane_tables
    from engine.profiles.qwen38.adapter import build_model, capture
    from engine.profiles.qwen38.caches import Qwen38Caches, layout, snapshot_layout
    from engine.profiles.qwen38.fleet import MAX_WAIT_S, TOKEN_BUDGET, rank_loader
    from engine.profiles.qwen38.net import Qwen38Net
    from engine.profiles.qwen38 import warmup as warm
    from probes.engine_qwen38_step import OneRankComm, ZeroPLETable

    phases = {}
    began = time.perf_counter()
    F = facts.load(ranks)
    if F.spec_k != spec_k:
        F = dataclasses.replace(F, spec_k=spec_k)
    net = Qwen38Net(F, OneRankComm(rank), lane_tables.served(), layers=list(layers), mtp=True)
    specs = net.specs()
    snapshots = 9
    arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + layout(F, net.layers, mtp=True).nbytes(BLOCKS, MAX_SEQS)
                  + snapshots * snapshot_layout(F, net.layers)[0])
    loader = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    net.bind(loader.load([s.name for s in specs], arena=arena))
    if net._ple is not None:
        net.attach_ple(ZeroPLETable(F.ple_rows_per_rank, F.ple_head_dim, float(net._ple_scale)),
                       max_rows=MAX_SEQS * (F.spec_k + 1))
    net.prepare_dense(None)
    caches = Qwen38Caches(arena, F, net.layers, BLOCKS, MAX_SEQS, snapshots, mtp=True)
    model, _ = build_model(net, caches, F, eos_ids=[F.vocab + 7], max_new=2048, temperature=0.0, top_p=1.0, seed=0,
                           drafter=True)
    k = model.k
    contract = sched.Contract(chunk_align=F.chunk_align, token_budget=TOKEN_BUDGET, draft_slots=k, max_wait_s=MAX_WAIT_S,
                              max_running=MAX_SEQS, decode_token_budget=F.chunk_align + k)
    chunk = sched.chunk_for(contract.chunk_align, contract.token_budget, k)
    runner = Runner(model, contract, caches.pool, caches.slots, Ring(4096, STEP_RECORD.size), None, keep_idle=False,
                    prefix=PrefixCache(F.block, chunk, snapshots))
    torch.cuda.synchronize()
    phases["build"] = round(time.perf_counter() - began, 2)
    if hasattr(warm, "eager_moe"):
        began = time.perf_counter()
        phases["warm_eager_moe_passes"] = warm.eager_moe(net)
        phases["warm_eager_moe"] = round(time.perf_counter() - began, 2)
    began = time.perf_counter()
    head = {"head": k + 1} if "head" in inspect.signature(warm.warmup).parameters else {}   # a tree before the head passes
    phases["warm_prefill_passes"] = warm.warmup(net, caches, memory=None, chunk=chunk, max_context=model.max_context,
                                                mtp=True, **head)
    phases["warm_prefill"] = round(time.perf_counter() - began, 2)
    began = time.perf_counter()
    capture(model, MAX_SEQS)
    torch.cuda.synchronize()
    phases["capture"] = round(time.perf_counter() - began, 2)
    return F, net, caches, model, runner, phases


def serve(F, model, runner, census: Census, requests=REQUESTS, seed: int = 0) -> list:
    """The requests one after another on row 0, the door's way (engine.add, runner.submit, runner.step until the row
    leaves) -> a row a request: its steps' seconds and every kernel a step added."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for name, prompt, made in requests:
        ids = torch.randint(0, F.vocab, (prompt,), generator=gen).tolist()
        model.add(0, ids, max_new=made, temperature=0.0)
        runner.submit(0, len(ids), ids=ids)
        steps, compiled = [], []
        while True:
            before = census.count()
            began = time.perf_counter()
            step = runner.step()
            seconds = time.perf_counter() - began
            if step is None:
                if 0 not in runner.state.running and 0 not in runner.state.waiting:
                    break
                continue
            added = census.new() if census.count() != before else []
            steps.append({"kind": step.kind, "s": round(seconds, 4), "compiled": len(added)})
            for entry in added:
                compiled.append(dict(entry, step=len(steps) - 1, kind=step.kind, s=round(seconds, 3)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tokens = len(model.tokens.get(0, [])) - prompt
        model.forget(0)
        if census.sweep():                                  # functions a lazy import brought in during this request
            for entry in census.new():
                compiled.append(dict(entry, step=None, kind="after the request", s=None))
        decode = [s["s"] for s in steps if s["kind"] != "prefill"]
        rows.append({"request": name, "prompt": prompt, "generated": tokens, "steps": len(steps),
                     "decode_steps": len(decode), "decode_s": round(sum(decode), 3),
                     "slowest_decode_s": round(max(decode), 3) if decode else None,
                     "prefill_s": round(sum(s["s"] for s in steps if s["kind"] == "prefill"), 3),
                     "compiled": compiled, "step_s": [s["s"] for s in steps]})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "step_s"}), flush=True)
    return rows


def run(output=None, ranks=None, *, rank: "int | None" = None, spec_k: int = SPEC_K) -> dict:
    import torch
    from engine.base import kernel_shape
    from engine.profiles.qwen38 import facts
    from probes.engine_qwen38_prefill import ceiling_gib

    assert torch.cuda.get_device_capability() == (12, 1), "requires GB10"
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, ceiling_gib() * (1 << 30) / total))
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        rank = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))[-1]
    _, shape_source = kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    F, net, caches, model, runner, phases = build(ranks, rank, spec_k=spec_k)
    census = Census()
    at_door = census.count()
    rows = serve(F, model, runner, census)
    after = [c for r in rows for c in r["compiled"]]
    by_kernel = {}
    for c in after:
        by_kernel[c["kernel"]] = by_kernel.get(c["kernel"], 0) + 1
    record = {"lane": "qwen38_serve_compiles", "rank": rank, "layers": list(LAYERS), "spec_k": spec_k, "max_seqs": MAX_SEQS,
              "kernel_shape": shape_source, "boot": phases, "kernels_at_door": at_door,
              "compiled_after_door": len(after), "compiled_by_kernel": dict(sorted(by_kernel.items(), key=lambda kv: -kv[1])),
              "requests": rows, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2), "free_GiB_at_start": round(free / 2**30, 1)}
    print(json.dumps({k: v for k, v in record.items() if k != "requests"}), flush=True)
    if output:
        Path(output).write_text(json.dumps(record) + "\n")
    return record
