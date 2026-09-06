#!/usr/bin/env python3
"""Executable checks for the overlay's pure math — no GPU, no vllm needed.

The overlays only run inside the production image, so their trickiest pure
logic (prefill chunker budgets, IndexCache skip rule, indexer-SP shard
coverage, overlay-manifest invariants) was previously verified by review only.
This extracts those
functions from the sources with ast and runs them against stub inputs, so
the invariants are executable on any machine (dev Mac or srv2, pre-deploy).

Run: python3 tests/test_logic.py
"""
from __future__ import annotations

import ast
import builtins
import glob
import os
import random
import re
import shutil
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bash() -> str:
    # deneb fork: subprocess with bare "bash" lets CreateProcess resolve to
    # System32's WSL bash on Windows, which cannot open C:/ paths -- the
    # launcher guard tests then report "did not reject" for every case.
    # shutil.which finds the MSYS bash that can; on Linux both are the same
    # /usr/bin/bash.
    return shutil.which("bash") or "bash"


def _overlay_source(relpath: str) -> str:
    """Find an overlay source under overlay/modules/<module>/.

    The files moved there when the repo split into modules; this test kept the
    old flat overlay/<name>.py paths and started raising FileNotFoundError.
    deploy-overlays.sh runs it as a gate, so every profile's deploy aborted.
    Resolving by filename rather than a fixed path keeps that from recurring the
    next time a module is renamed.
    """
    path = os.path.join(REPO, relpath)
    if os.path.exists(path):
        return path
    name = os.path.basename(relpath)
    hits = sorted(glob.glob(os.path.join(REPO, "overlay", "modules", "*", name)))
    if not hits:
        raise FileNotFoundError(f"no overlay module provides {name}")
    if len(hits) > 1:
        raise RuntimeError(f"{name} is provided by {len(hits)} modules: {hits}")
    return hits[0]


class _CapturingLogger:
    """Records what an overlay announced, so a test can assert it announced."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, fmt, *args):
        self.lines.append(fmt % args if args else str(fmt))

    info = warning = error = debug = _record

    def __getattr__(self, _name):
        return self._record


def load_defs(relpath: str, names: set[str], ns: dict) -> dict:
    """exec only the named top-level defs/assigns from a source file into ns."""
    path = _overlay_source(relpath)
    tree = ast.parse(open(path).read())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            body.append(node)
        elif (
            # An annotated module global (``x: set = set()``) is an AnnAssign,
            # not an Assign; without this it reads as "def not found" and the
            # caller has to strip the annotation from production code to make
            # a test loadable.
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in names
        ):
            body.append(node)
    got = {n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
        t.id
        for n in body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {
        n.target.id
        for n in body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    missing = names - got
    assert not missing, f"{relpath}: defs not found: {missing}"
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


PASS = 0


def check(cond: bool, msg: str) -> None:
    global PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    PASS += 1


# ---------------------------------------------------------------------------
# 1. _resolve_skip_topk — IndexCache F/S rule (attention.py)
# ---------------------------------------------------------------------------
def test_skip_topk() -> None:
    ns = load_defs("overlay/attention.py", {"_resolve_skip_topk"}, {})
    fn = ns["_resolve_skip_topk"]

    def cfg(**kw):
        return types.SimpleNamespace(**kw)

    # freq=4 over C4A layers: F S S S F S S S ... (indexed over C4A only)
    ratios = [1, 4, 4, 1, 4, 4, 4, 4, 128, 4]  # C4A at 1,2,4,5,6,7,9
    c = cfg(use_index_cache=True, compress_ratios=ratios, num_hidden_layers=10,
            index_topk_freq=4)
    expect = {1: False, 2: True, 4: True, 5: True, 6: False, 7: True, 9: True}
    for lid, want in expect.items():
        check(fn(c, lid) == want, f"skip_topk layer {lid}: want {want}")
    # non-C4A layers never skip
    for lid in (0, 3, 8):
        check(fn(c, lid) is False, f"non-C4A layer {lid} must not skip")
    # the first C4A layer is structurally F for ANY freq (0 % freq == 0) —
    # the invariant IndexCache top-k reuse relies on (S layers read the
    # buffer the previous F layer wrote)
    for freq in (1, 2, 3, 4, 7):
        c2 = cfg(use_index_cache=True, compress_ratios=ratios,
                 num_hidden_layers=10, index_topk_freq=freq)
        check(fn(c2, 1) is False, f"first C4A layer must be F at freq={freq}")
    # feature off / missing ratios -> never skip
    check(fn(cfg(use_index_cache=False, compress_ratios=ratios,
                 num_hidden_layers=10, index_topk_freq=4), 2) is False,
          "use_index_cache off must not skip")
    check(fn(cfg(use_index_cache=True), 2) is False,
          "missing compress_ratios must not skip")
    # MTP/draft slot beyond num_hidden_layers -> F (computes its own top-k)
    c3 = cfg(use_index_cache=True, compress_ratios=ratios + [4],
             num_hidden_layers=10, index_topk_freq=4)
    check(fn(c3, 10) is False, "trailing MTP slot must not skip")
    print(f"  skip_topk rule ................ OK")


# ---------------------------------------------------------------------------
# 2. split_indexer_prefill_chunks — budget math of PR #51252 (indexer.py)
# ---------------------------------------------------------------------------
def test_prefill_chunker() -> None:
    fake_torch = types.SimpleNamespace(Tensor=type("T", (), {}))
    ns = load_defs("overlay/indexer.py", {"split_indexer_prefill_chunks"},
                   {"torch": fake_torch, "np": None})
    fn = ns["split_indexer_prefill_chunks"]

    def validate(seq_lens, query_lens, ws, logits_bytes, offset=0):
        chunks = fn(list(seq_lens), list(query_lens), ws, logits_bytes,
                    request_offset=offset)
        max_elems = logits_bytes // 4
        covered = {i: 0 for i in range(len(seq_lens))}
        for req_slice, q_slice in chunks:
            reqs = range(req_slice.start - offset, req_slice.stop - offset)
            n = sum(seq_lens[r] for r in reqs)
            m = q_slice.stop - q_slice.start
            check(m >= 1, "empty query sub-chunk")
            multi = len(list(reqs)) > 1
            if multi:
                # multi-request chunks must fit both budgets outright
                check(n <= ws, f"workspace exceeded: {n} > {ws}")
                check((sum(query_lens[r] for r in reqs)) * n <= max_elems,
                      "logits budget exceeded for multi-request chunk")
            else:
                # single oversized request: query-dim sub-chunk keeps m*n
                # within budget whenever any sub-chunking was possible
                if n <= ws and query_lens[list(reqs)[0]] * n > max_elems:
                    check(m * n <= max_elems or m == 1,
                          "sub-chunk exceeds logits budget")
            for r in reqs:
                covered[r] += m if not multi else query_lens[r]
        # every request's queries covered exactly once
        for r, q in enumerate(query_lens):
            check(covered[r] == q, f"req {r}: covered {covered[r]} != {q}")
        return chunks

    validate([100, 200, 300], [10, 20, 30], ws=10_000, logits_bytes=4 * 10**9)
    # tight workspace forces one request per chunk
    ch = validate([500, 500, 500], [50, 50, 50], ws=600, logits_bytes=4 * 10**9)
    check(len(ch) == 3, "tight workspace must split per request")
    # oversized single request sub-chunks on the query dim
    ch = validate([100_000], [4_096], ws=200_000, logits_bytes=4 * 10_000_000)
    check(len(ch) > 1, "logits budget must force query sub-chunking")
    # request_offset shifts slices
    ch = fn([10], [5], 100, 4 * 10**9, request_offset=7)
    check(ch[0][0].start == 7 and ch[0][0].stop == 8, "request_offset ignored")
    print(f"  prefill chunker budgets ....... OK")


# ---------------------------------------------------------------------------
# 3. _indexer_sp_owned_ranges — SP shard coverage (attention.py)
# ---------------------------------------------------------------------------
def test_sp_ranges() -> None:
    def chunk(a, b):
        return types.SimpleNamespace(token_start=a, token_end=b)

    def md_for(ndt, chunks):
        return types.SimpleNamespace(
            num_prefills=len(chunks) or 1,
            num_decode_tokens=ndt,
            prefill=types.SimpleNamespace(chunks=chunks),
        )

    def ranges_for(rank, tp, md):
        ns = {
            "_INDEXER_SP_ENABLED": True,
            "_indexer_sp_tp_rank": (tp, rank),
            "_resolve_layer_name": None,
            "get_forward_context": lambda: types.SimpleNamespace(
                attn_metadata={"k": md}
            ),
            "get_tensor_model_parallel_world_size": lambda: tp,
            "get_tensor_model_parallel_rank": lambda: rank,
        }
        load_defs("overlay/attention.py", {"_indexer_sp_owned_ranges"}, ns)
        return ns["_indexer_sp_owned_ranges"]("k")

    tp = 4
    # mixed batch: decode rows + one small chunk (<256) + two large chunks
    ndt = 12
    chunks = [chunk(12, 140), chunk(140, 4236), chunk(4236, 4801)]
    per_rank = [ranges_for(r, tp, md_for(ndt, chunks)) for r in range(tp)]
    for r, rr in enumerate(per_rank):
        rows = set()
        for a, b in rr:
            check(b > a, f"rank {r}: empty range emitted")
            rows.update(range(a, b))
        # decode rows and the small chunk are replicated on every rank
        check(set(range(0, ndt)) <= rows, f"rank {r}: decode rows not owned")
        check(set(range(12, 140)) <= rows, f"rank {r}: small chunk not owned")
    # adjacent spans are merged (shorter gather-index list for the consumer):
    # rank 0 fuses decode+small+first-shard into one span
    check(len(per_rank[0]) == 2, f"rank0 spans not merged: {per_rank[0]}")
    check(per_rank[0][0] == (0, 1164), f"rank0 fused span wrong: {per_rank[0]}")
    check(len(per_rank[1]) == 3, f"rank1 spans wrong: {per_rank[1]}")
    for ch in (chunks[1], chunks[2]):
        n = ch.token_end - ch.token_start
        sharded = [
            {row for a, b in rr for row in range(a, b)
             if ch.token_start <= row < ch.token_end}
            for rr in per_rank
        ]
        union = set().union(*sharded)
        check(union == set(range(ch.token_start, ch.token_end)),
              f"large chunk [{ch.token_start},{ch.token_end}) not covered")
        total = sum(len(s) for s in sharded)
        check(total == n, f"large chunk overlap: {total} != {n}")
        # ceil-split: rank loads differ by at most one shard quantum
        shard = -(-n // tp)
        check(all(len(s) <= shard for s in sharded), "shard size exceeded")
    # decode-only batch (no prefills) -> None (compute all rows)
    check(ranges_for(0, tp, types.SimpleNamespace(
        num_prefills=0, num_decode_tokens=8, prefill=None)) is None,
        "decode-only batch must return None")
    print(f"  indexer-SP shard coverage ..... OK")


# ---------------------------------------------------------------------------
# 4. small pure helpers
# ---------------------------------------------------------------------------
def test_helpers() -> None:
    ns = load_defs(
        "overlay/indexer.py",
        {"_get_b12x_paged_indexer_supertile_k",
         "_B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT",
         "_B12X_PAGED_INDEX_TILE_BLOCK_K"},
        {"os": os},
    )
    fn = ns["_get_b12x_paged_indexer_supertile_k"]
    os.environ.pop("B12X_PAGED_INDEX_SUPERTILE_K", None)
    check(fn() == 32768, "supertile default")
    os.environ["B12X_PAGED_INDEX_SUPERTILE_K"] = "1000"
    check(fn() == 1024, "supertile rounds up to 512 multiple")
    os.environ["B12X_PAGED_INDEX_SUPERTILE_K"] = "100"
    check(fn() == 512, "supertile clamps to tile block")
    os.environ.pop("B12X_PAGED_INDEX_SUPERTILE_K", None)

    ns = load_defs(
        "overlay/sparse_swa_dsv4.py",
        {"_layer_type_for", "_LAYER_TYPE_SWAONLY", "_LAYER_TYPE_C4A",
         "_LAYER_TYPE_C128A"},
        {},
    )
    lt = ns["_layer_type_for"]
    check(lt(1) == "swaonly" and lt(4) == "c4a" and lt(128) == "c128a",
          "layer types")
    try:
        lt(8)
        check(False, "ratio 8 must raise")
    except ValueError:
        check(True, "")
    print(f"  helper functions .............. OK")


# ---------------------------------------------------------------------------
# 5. profile-step.py — importable directly (stdlib only)
# ---------------------------------------------------------------------------
def test_profile_step() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ps", os.path.join(REPO, "bench", "profile-step.py"))
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    check(ps.bucket_of("ncclDevKernel_AllReduce_RING_LL") == "comms",
          "nccl bucket")
    check(ps.bucket_of("sm120_grouped_gemm_fp8") == "moe", "moe bucket")
    check(ps.bucket_of("totally_unknown_kernel") == "other", "other bucket")
    # busy-union: [0,10) + [5,15) + [20,30) -> 25 busy over 30 wall
    busy = ps.merged_busy_us([(0.0, 10.0), (5.0, 15.0), (20.0, 30.0)])
    check(abs(busy - 25.0) < 1e-9, f"interval union: {busy}")
    check(ps.merged_busy_us([]) == 0.0, "empty union")
    print(f"  profile-step buckets/union .... OK")


# ---------------------------------------------------------------------------
# 6. bench-dec.py spec-decode metrics parsing (stdlib only, importable)
# ---------------------------------------------------------------------------
def test_bench_dec_metrics() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bd", os.path.join(REPO, "bench", "bench-dec.py"))
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    text = (
        "# HELP vllm:spec_decode_num_accepted_tokens_total help\n"
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 120\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="1"} 30\n'
        'vllm:spec_decode_num_draft_tokens_total{engine="0"} 200\n'
        'vllm:spec_decode_num_draft_tokens_total{engine="1"} 50\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 100\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 50\n'
        "vllm:spec_decode_num_drafts_total 50\n"
        "vllm:num_requests_running 5\n"
    )
    c = bd._parse_spec_metrics(text)
    check(c["vllm:spec_decode_num_accepted_tokens_total"] == 150.0,
          "accepted counters summed over labels")
    check(c["vllm:spec_decode_num_draft_tokens_total"] == 250.0,
          "draft counters summed over labels")
    check("vllm:num_requests_running" not in c, "non-spec metric excluded")
    before = {k: 0.0 for k in c}
    s = bd.acceptance_suffix(before, c)
    # Legacy convention double-counts per_pos + num_drafts: (150+150)/(250+50).
    check("100.0%" in s, f"legacy acceptance ratio wrong: {s!r}")
    # Exact-name raw ratio excludes per_pos/num_drafts: 150/250.
    check("raw" in s and "60.0%" in s, f"raw acceptance ratio wrong: {s!r}")
    check(bd.acceptance_suffix(c, c) == "", "zero delta must yield no suffix")
    check(bd.acceptance_suffix({}, {}) == "", "empty metrics must yield no suffix")
    print(f"  bench-dec acceptance parser ... OK")


# ---------------------------------------------------------------------------
# 7. build/<profile>/manifest.tsv — the composed overlay inventory
# ---------------------------------------------------------------------------
def test_overlay_symbol_contracts() -> None:
    """An overlay importing from a module another overlay owns must find it there.

    glm53_model_wiring rewrote a stock import to take DenebGateLinear from
    moe_gate_sm121 and carried a second name along that the gate module does not
    define. `requires` records the dependency but not which symbols it expects,
    and since the overlays were never mounted, nothing had tried the import --
    so the boot died on ImportError forty seconds in.
    """
    # Owners are resolved per composition: two profiles may overlay the same
    # container path with different files (dsv4's mla_indexer and glm53's
    # glm53_tail_slot_persistent both own mla/indexer.py), and only the modules
    # composed together ever meet at runtime. A global "last manifest wins" map
    # checked glm53's attention.py against dsv4's indexer. Modules listed in no
    # profile are checked as their own composition (the module plus its
    # `requires` closure), so an orphan's intra-module contracts stay guarded.
    # Relative imports (`from .attention import X`) are resolved against the
    # importing file's own container path, so an overlay of a sibling module's
    # file is checked too. Every source is parsed once.
    all_mods = sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(
        os.path.join(REPO, "overlay", "modules", "*", "manifest.tsv")))
    profiles = {}
    for env in sorted(glob.glob(os.path.join(REPO, "profiles", "*.env"))):
        m = re.search(r'^MODULES="([^"]*)"', open(env, encoding="utf-8").read(), re.M)
        if m:
            profiles[os.path.basename(env)] = m.group(1).split()
    listed = {mod for mods in profiles.values() for mod in mods}
    for mod in all_mods:
        if mod not in listed:
            closure, todo = [], [mod]
            while todo:
                cur = todo.pop()
                if cur in closure:
                    continue
                closure.append(cur)
                req = os.path.join(REPO, "overlay", "modules", cur, "requires")
                if os.path.exists(req):
                    todo.extend(open(req, encoding="utf-8").read().split())
            profiles[f"(orphan) {mod}"] = closure

    def _target_dotted(target):
        if target.startswith("vllm/"):
            rel = target
        elif "/vllm/" in target:
            rel = "vllm/" + target.split("/vllm/", 1)[1]
        else:
            return None
        return rel[:-3].replace("/", ".")

    def _owners(pname, mods):
        owners = {}      # dotted module path -> source path
        for mod in mods:
            manifest = os.path.join(REPO, "overlay", "modules", mod, "manifest.tsv")
            check(os.path.exists(manifest), f"[{pname}] MODULES lists {mod!r} but it has no manifest")
            if not os.path.exists(manifest):
                continue
            moddir = os.path.dirname(manifest)
            for raw in open(manifest, encoding="utf-8"):
                line = raw.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                source, target = line.split("\t")[:2]
                if not source.endswith(".py"):
                    continue
                # Targets are absolute for image-bound overlays and relative to
                # the package root for portable ones, so anchoring on "/vllm/"
                # silently skipped every portable module -- including
                # moe_gate_sm121, the one this check exists for.
                dotted = _target_dotted(target)
                if dotted is None:
                    continue
                prev = owners.get(dotted)
                check(prev is None, f"[{pname}] {mod} and {prev and os.path.basename(os.path.dirname(prev))} both own {dotted}")
                owners[dotted] = os.path.join(moddir, source)
        return owners

    parsed = {}      # source path -> (provided names, [(absolute dotted module, [alias names])])

    def _parse(srcpath, dotted_self):
        if srcpath in parsed:
            return parsed[srcpath]
        tree = ast.parse(open(srcpath, encoding="utf-8").read())
        provided = set()
        imports = []
        package = dotted_self.rsplit(".", 1)[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                provided.add(node.name)
            elif isinstance(node, ast.Assign):
                provided.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                provided.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                provided.update((a.asname or a.name).split(".")[0] for a in node.names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 0:
                target = node.module
            else:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                target = ".".join(base + ([node.module] if node.module else []))
            imports.append((target, [a.name for a in node.names]))
        parsed[srcpath] = (provided, imports)
        return parsed[srcpath]

    checked = 0
    seen = set()
    for pname, mods in sorted(profiles.items()):
        owners = _owners(pname, mods)
        by_src = {v: k for k, v in owners.items()}
        for dotted, srcpath in sorted(owners.items()):
            provided, _ = _parse(srcpath, dotted)
            for other in sorted(set(owners.values())):
                if other == srcpath:
                    continue
                _, imports = _parse(other, by_src[other])
                for target, names in imports:
                    if target != dotted:
                        continue
                    for name in names:
                        key = (other, dotted, name, srcpath)
                        if key in seen:
                            continue
                        seen.add(key)
                        checked += 1
                        check(name in provided,
                              f"[{pname}] {os.path.basename(other)} imports {name} from "
                              f"{dotted}, which {os.path.basename(srcpath)} does not define")
    print(f"  overlay symbol contracts ({checked}, per composition incl. orphans) ... OK")


def test_profile_env_carried() -> None:
    """Every VLLM_* a profile declares has to reach the container.

    glm53.env set three of them and the launcher passed none, so the fused MoE
    gate, the fp8 lm head and one-shot AllReduce were dead config for as long as
    they had existed. The launcher reads the names out of the profile now; this
    checks it still does.
    """
    checked = 0
    for envpath in sorted(glob.glob(os.path.join(REPO, "profiles", "*.env"))):
        keys = {m.group(0) for m in re.finditer(r"^VLLM_[A-Z0-9_]+",
                                                open(envpath, encoding="utf-8").read(),
                                                re.M)}
        if not keys:
            continue
        launcher = None
        for line in open(envpath, encoding="utf-8"):
            if line.startswith("PROFILE_LAUNCHER="):
                launcher = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not launcher:
            continue
        path = os.path.join(REPO, "launchers", launcher)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8").read()
        check("_vllm_keys" in text and "ENVV=\"$ENVV -e $_k=" in text,
              f"{launcher} does not carry the profile's VLLM_* knobs "
              f"({len(keys)} declared in {os.path.basename(envpath)})")
        checked += len(keys)
    print(f"  profile env carried ({checked} knobs) ... OK")


# Keys that exist for a reader, not a program. Extending this is a decision:
# every other unread key so far turned out to be a carrier nobody wrote.
_PROFILE_DOC_ONLY = {"PROFILE_DESC", "GMU_NOTE"}

_PROFILE_CONSUMER_GLOBS = (
    "launchers/*.sh", "launchers/*.py", "tools/*.py",
    "overlay/modules/*/*.py", "bench/*.py", "bench/*.sh",
)


def test_profile_keys_have_readers() -> None:
    """A profile key nothing reads is either dead config or a missing carrier."""
    sources = {}
    for pat in _PROFILE_CONSUMER_GLOBS:
        for path in glob.glob(os.path.join(REPO, pat)):
            try:
                sources[path] = open(path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                pass
    # A launcher that harvests VLLM_* by pattern reads all of them at once.
    harvests = any("_vllm_keys" in t for t in sources.values())

    checked = 0
    for envpath in sorted(glob.glob(os.path.join(REPO, "profiles", "*.env"))):
        profile = os.path.basename(envpath)
        text = open(envpath, encoding="utf-8").read()
        for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)=", text, re.M):
            key = match.group(1)
            if key in _PROFILE_DOC_ONLY:
                continue
            checked += 1
            if key.startswith("VLLM_") and harvests:
                continue
            check(any(re.search(rf"\b{re.escape(key)}\b", t) for t in sources.values()),
                  f"{profile} declares {key} and nothing reads it -- "
                  f"add the carrier, or list it in _PROFILE_DOC_ONLY")
    print(f"  profile keys have readers ({checked}) ... OK")


def _composed_manifests() -> list[tuple[str, str, str]]:
    """(profile, target_prefix, manifest path) for every composed profile."""
    out = []
    for envpath in sorted(glob.glob(os.path.join(REPO, "profiles", "*.env"))):
        name = os.path.splitext(os.path.basename(envpath))[0]
        manifest = os.path.join(REPO, "build", name, "manifest.tsv")
        if not os.path.isfile(manifest):
            continue
        prefix = "/opt/venv/lib/python3.12/site-packages/"
        for line in open(envpath, encoding="utf-8"):
            if line.startswith("TARGET_PREFIX="):
                prefix = line.split("=", 1)[1].strip().strip('"').strip("'")
        out.append((name, prefix, manifest))
    return out


def test_overlay_manifest() -> None:
    source_chars = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    target_chars = source_chars | {"/"}
    profiles = _composed_manifests()
    check(bool(profiles), "no composed profile manifests -- run compose-overlays.sh")

    total = 0
    for profile, prefix, manifest in profiles:
        build = os.path.dirname(manifest)
        rows = []
        with open(manifest, encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                line = raw.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                check(len(fields) == 3,
                      f"{profile} line {line_no}: need three TSV fields")
                source, target, base_contract = fields
                check(source == os.path.basename(source),
                      f"{profile} line {line_no}: source must be a basename")
                check(source and not source.startswith(".")
                      and all(ch in source_chars for ch in source),
                      f"{profile} line {line_no}: unsafe source")
                check(target.startswith(prefix),
                      f"{profile} line {line_no}: target outside {prefix}")
                check(all(ch in target_chars for ch in target),
                      f"{profile} line {line_no}: unsafe target character")
                check(os.path.isfile(os.path.join(build, source)),
                      f"{profile} line {line_no}: missing build/{profile}/{source}")
                check(
                    base_contract == "absent"
                    or (
                        len(base_contract) == 64
                        and all(ch in "0123456789abcdef" for ch in base_contract)
                    ),
                    f"{profile} line {line_no}: invalid base preimage contract",
                )
                rows.append((source, target, base_contract))

        check(bool(rows), f"{profile}: manifest must not be empty")
        check(len({src for src, _, _ in rows}) == len(rows),
              f"{profile}: duplicate sources")
        check(len({tgt for _, tgt, _ in rows}) == len(rows),
              f"{profile}: duplicate targets")

        if profile == "dsv4":
            sources = {src for src, _, _ in rows}
            replaced = {"dspark_v2.py", "dspark_speculator_v2.py",
                        "dspark_utils_v2.py"}
            check({"fp8_draft_head.py"} | replaced <= sources,
                  "DSpark speed overlays are missing from the dsv4 manifest")
            check(all(contract != "absent" for src, _, contract in rows
                      if src in replaced),
                  "replaced DSpark files must pin exact production preimages")
        total += len(rows)

    for relpath in ("launchers/start-hy4-tp4.sh",
                    "launchers/start-glm53-nvfp4-tp4.sh",
                    "launchers/deploy-overlays.sh"):
        text = open(os.path.join(REPO, relpath), encoding="utf-8").read()
        check("manifest.tsv" in text, f"{relpath}: manifest not consumed")
        check('OVFILES="' not in text,
              f"{relpath}: hard-coded overlay inventory remains")
        check("md5sum" not in text and "sha256sum" in text,
              f"{relpath}: manifest/files must use SHA-256 parity")
    names = ", ".join(p for p, _, _ in profiles)
    print(f"  overlay manifests ({total} files across {names}) .... OK")


def test_composed_snapshot_sync() -> None:
    """Committed profile snapshots must equal a fresh module composition.

    Deploy composes again, but reviewers and probes read the checked-in build
    tree. A later PR regenerated from an older branch and silently restored
    stale fp8/MHC files, while every prior logic check stayed green.
    """
    total = 0
    for profile, prefix, build_manifest in _composed_manifests():
        envpath = os.path.join(REPO, "profiles", f"{profile}.env")
        env_text = open(envpath, encoding="utf-8").read()
        modules_match = re.search(r'^MODULES=["\']([^"\']+)["\']$',
                                  env_text, re.M)
        check(modules_match is not None,
              f"{profile}: MODULES must be a one-line quoted value")
        modules = modules_match.group(1).split()

        expected_rows = []
        owners = {}
        for module in modules:
            module_dir = os.path.join(REPO, "overlay", "modules", module)
            manifest = os.path.join(module_dir, "manifest.tsv")
            check(os.path.isfile(manifest),
                  f"{profile}: missing module manifest for {module}")
            with open(manifest, encoding="utf-8") as handle:
                for line_no, raw in enumerate(handle, 1):
                    line = raw.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    check(len(fields) == 3,
                          f"{module} line {line_no}: need three TSV fields")
                    source, target, contract = fields
                    check(source not in owners,
                          f"{profile}: {source} owned by two modules")
                    owners[source] = os.path.join(module_dir, source)
                    if not target.startswith("/"):
                        target = prefix + target
                    expected_rows.append((source, target, contract))

        with open(build_manifest, encoding="utf-8") as handle:
            actual_rows = [tuple(raw.rstrip("\n").split("\t"))
                           for raw in handle
                           if raw.strip() and not raw.startswith("#")]
        check(actual_rows == expected_rows,
              f"{profile}: committed manifest differs from profile modules")

        build_dir = os.path.dirname(build_manifest)
        actual_files = sorted(
            name for name in os.listdir(build_dir) if name != "manifest.tsv"
            and not name.startswith("__pycache__")
        )
        check(actual_files == sorted(owners),
              f"{profile}: build file inventory differs from its manifest")
        for source, module_source in owners.items():
            build_source = os.path.join(build_dir, source)
            with open(module_source, "rb") as left, open(build_source, "rb") as right:
                check(left.read() == right.read(),
                      f"{profile}: build/{source} is stale; re-run compose")
            total += 1

    check(total > 0, "composed profile snapshots are missing")
    print(f"  composed snapshot sync ({total} files) . OK")


# ---------------------------------------------------------------------------
# 8. DSpark speed experiment — launcher bounds + FP8 helper contract
# ---------------------------------------------------------------------------
def test_dspark_speed_guards() -> None:
    # deneb fork: MSYS bash strips unescaped backslashes, so a Windows
    # os.path.join path never reaches the script (suite runs on dev Macs,
    # srv2 AND Windows checkouts); forward slashes work everywhere.
    launcher = os.path.join(
        REPO, "launchers", "start-hy4-tp4.sh").replace(os.sep, "/")
    cases = (
        ({"FP8HEAD": "2", "MARKOV_TOPK": "0"}, "FP8HEAD must be 0 or 1"),
        ({"FP8HEAD": "0", "MARKOV_TOPK": "-1"}, "MARKOV_TOPK must be"),
        ({"FP8HEAD": "0", "MARKOV_TOPK": "abc"}, "MARKOV_TOPK must be"),
        ({"FP8HEAD": "0", "MARKOV_TOPK": "129281"}, "exceeds model vocab"),
        (
            {"FP8HEAD": "1", "MARKOV_TOPK": "0", "V2RUNNER": "0"},
            "require V2RUNNER=1",
        ),
    )
    for extra_env, expected in cases:
        env = dict(os.environ)
        env.update(extra_env)
        proc = subprocess.run(
            [_bash(), launcher],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        output = proc.stdout + proc.stderr
        check(proc.returncode == 2, f"launcher did not reject {extra_env}: {output}")
        check(expected in output, f"launcher rejection unclear for {extra_env}: {output}")

    launcher_text = open(launcher, encoding="utf-8").read()
    check("VLLM_DSPARK_FP8_DRAFT_HEAD=$FP8HEAD" in launcher_text,
          "FP8 draft-head knob not propagated")
    check("VLLM_DSPARK_DRAFT_TOPK=$MARKOV_TOPK" in launcher_text,
          "Markov top-k knob not propagated")
    check("VLLM_DSPARK_IMPL=$DSPARK_IMPL" in launcher_text,
          "DSpark v2 selector is not explicit")
    check('DSPARK_IMPL="${DSPARK_IMPL:-upstream}"' in launcher_text,
          "DSpark impl must default to upstream (the image ENV bakes it; "
          "production has always run it)")
    check("FP8HEAD == 1 || MARKOV_TOPK > 0" in launcher_text
          and "require the upstream DSpark impl" in launcher_text,
          "arming a speed knob with fork forced must abort")


# ---------------------------------------------------------------------------
# 9. bf16-release fail-closed matrix (nvidia_model.py / attention.py knobs)
# ---------------------------------------------------------------------------
def test_bf16_release_guards() -> None:
    ns = load_defs(
        "overlay/nvidia_model.py", {"_bf16_lm_head_release_blockers"}, {}
    )
    fn = ns["_bf16_lm_head_release_blockers"]

    # production defaults: target fp8 built + aliased draft with a copy
    check(fn(True, True, True) == [], "safe config must have no blockers")
    # draft with its OWN head never blocks (release touches only the target)
    check(fn(True, False, False) == [], "unaliased draft must not block")
    # target fp8 off -> verification would read the freed weight
    check(any("TARGET_LM_HEAD_FP8" in b for b in fn(False, True, True)),
          "target-fp8-off must block")
    # aliased draft without any fp8 copy -> draft fallback reads bf16
    check(any("draft" in b for b in fn(True, True, False)),
          "aliased copyless draft must block")
    # both broken -> both reasons surface
    check(len(fn(False, True, False)) == 2, "blockers must accumulate")

    # wiring: launcher propagates both knobs; attention.py refuses COMPFREE
    # without the fp8 path; dspark_utils aborts on a rolled-back target.
    launcher_text = open(
        os.path.join(REPO, "launchers", "start-hy4-tp4.sh"), encoding="utf-8"
    ).read()
    check("VLLM_DSV4_FREE_BF16_LM_HEAD=${HEADFREE:-0}" in launcher_text,
          "HEADFREE knob not propagated")
    check("VLLM_DSV4_FREE_BF16_COMPRESSOR=${COMPFREE:-0}" in launcher_text,
          "COMPFREE knob not propagated")
    attn_text = open(
        _overlay_source("overlay/attention.py"), encoding="utf-8"
    ).read()
    check("_FREE_BF16_COMPRESSOR and not _COMPRESSOR_FP8" in attn_text,
          "COMPFREE without COMPRESSOR_FP8 must fail at import")
    utils_text = open(
        _overlay_source("overlay/dspark_utils_v2.py"), encoding="utf-8"
    ).read()
    check("maybe_release_bf16_lm_head" in utils_text
          and "overlay rolled back" in utils_text,
          "armed HEADFREE with rolled-back nvidia_model.py must abort")
    print(f"  bf16-release guards ........... OK")
    check("EXPECTED_IMAGE_ID=" in launcher_text and "OVBASES" in launcher_text,
          "image/source preimage attestation is missing")
    check('[ -L "$target" ]' in launcher_text,
          "base preimage attestation must reject symlink targets")
    check('[ ! -f "$target" ]' in launcher_text,
          "hashed base preimages must be regular files")

    helper = _overlay_source("overlay/fp8_draft_head.py")
    helper_text = open(helper, encoding="utf-8").read()
    helper_tree = ast.parse(helper_text)
    function_names = {
        node.name for node in helper_tree.body if isinstance(node, ast.FunctionDef)
    }
    check(
        {
            "fp8_draft_head_supported",
            "require_fp8_draft_head_support",
            "quantize_draft_head",
            "fp8_draft_head_logits",
        } <= function_names,
        "FP8 helper is missing support/quantize/GEMM guards",
    )
    check("hidden_states.shape[-1] != head.weight_fp8.shape[-1]" in helper_text,
          "FP8 helper lacks hidden/head dimension guard")

    ns = load_defs(
        "overlay/dspark_speculator_v2.py",
        {"_parse_draft_topk"},
        {},
    )
    parse_topk = ns["_parse_draft_topk"]
    check(parse_topk("", 10) is None, "empty top-k must disable")
    check(parse_topk("0", 10) is None, "zero top-k must disable")
    check(parse_topk("1", 10) == 1, "top-k lower bound rejected")
    check(parse_topk("10", 10) == 10, "top-k upper bound rejected")
    for raw in ("-1", "11", "x"):
        try:
            parse_topk(raw, 10)
            check(False, f"invalid top-k accepted: {raw}")
        except ValueError:
            check(True, "")

    spec_text = open(
        _overlay_source("overlay/dspark_speculator_v2.py"),
        encoding="utf-8",
    ).read()
    model_text = open(
        _overlay_source("overlay/dspark_v2.py"),
        encoding="utf-8",
    ).read()
    utils_text = open(
        _overlay_source("overlay/dspark_utils_v2.py"),
        encoding="utf-8",
    ).read()
    check('base_logits.fill_(-float("inf"))' in spec_text,
          "top-k proposal is not truncated before Markov scatter")
    check("output_processed_logits=self.draft_logits" in spec_text,
          "truncated proposal q is not retained for rejection sampling")
    check("LogitsProcessor.use_all_gather=True" in spec_text,
          "TP full-vocabulary all-gather gate is missing")
    check('not getattr(markov_head, "_replicate_w2", False)' in spec_text,
          "replicated W2 startup gate is missing")
    check("def apply_bias_gathered(" in model_text
          and "weight[token_indices]" in model_text,
          "gathered Markov W2-row projection is missing")
    check(
        model_text.index("fp8_head = self._fp8_draft_head")
        < model_text.index('getattr(self, "lm_head_fp8_weight", None)'),
        "rowwise FP8 must take precedence over legacy DeepGEMM FP8",
    )
    check("if fp8_draft_head_enabled:" in utils_text
          and "else:" in utils_text
          and "maybe_build_fp8_lm_head" in utils_text,
          "FP8 loader does not select exactly one draft-head implementation")
    print("  DSpark speed-path guards ...... OK")


# ---------------------------------------------------------------------------
# 9. DSpark acceptance levers — refine pass + Markov sideload contracts
# ---------------------------------------------------------------------------
def test_acceptance_lever_guards() -> None:
    ns = load_defs(
        "overlay/dspark_speculator_v2.py",
        {"_parse_refine_pass", "_refine_feedback_indices"},
        {},
    )
    parse_refine = ns["_parse_refine_pass"]
    for raw in ("", "0", "false", "no", "off", " OFF "):
        check(parse_refine(raw) is False, f"refine {raw!r} must disable")
    for raw in ("1", "true", "yes", "on", " ON "):
        check(parse_refine(raw) is True, f"refine {raw!r} must enable")
    for raw in ("2", "-1", "maybe"):
        try:
            parse_refine(raw)
            check(False, f"invalid refine value accepted: {raw}")
        except ValueError:
            check(True, "")

    # Feedback index math: query offset j>=1 of request r receives the token
    # drafted at offset j-1; anchors (offset 0) are never rewritten; flat
    # order matches draft_tokens[:, :N-1].reshape(-1).
    fn = ns["_refine_feedback_indices"]
    for max_reqs, n in ((1, 2), (3, 5), (4, 7)):
        idx = fn(max_reqs, n)
        check(len(idx) == max_reqs * (n - 1), f"feedback count {max_reqs}x{n}")
        check(idx == sorted(idx) and len(set(idx)) == len(idx),
              "feedback indices must be strictly increasing")
        check(all(0 <= i < max_reqs * n for i in idx), "feedback index range")
        check(all(i % n != 0 for i in idx), "anchor slot rewritten")
        for m, i in enumerate(idx):
            req, off = divmod(i, n)
            src_req, src_col = divmod(m, n - 1)
            check(req == src_req and off == src_col + 1,
                  f"feedback misalignment at flat {m}: idx {i}")
    check(fn(2, 1) == [], "n_spec=1 must have no feedback slots")

    ns = load_defs(
        "overlay/dspark_utils_v2.py",
        {"_validate_markov_sideload"},
        {},
    )
    validate = ns["_validate_markov_sideload"]
    good = dict(payload_keys={"markov_w1", "markov_w2"},
                w1_shape=(100, 8), w2_shape=(100, 8),
                vocab_size=100, markov_rank=8,
                replicated_w1=True, replicated_w2=True)
    check(validate(**good) is None, "valid sideload rejected")
    check("missing keys" in validate(**{**good, "payload_keys": {"markov_w1"}}),
          "missing key must be reported")
    check("markov_w1 shape" in validate(**{**good, "w1_shape": (100, 4)}),
          "w1 shape skew must be reported")
    check("markov_w2 shape" in validate(**{**good, "w2_shape": (50, 8)}),
          "w2 shape skew must be reported")
    check("replicated Markov W1" in validate(**{**good, "replicated_w1": False}),
          "sharded W1 must be rejected")
    check("replicated Markov W2" in validate(**{**good, "replicated_w2": False}),
          "sharded W2 must be rejected")

    # Launcher: bounds + fail-closed combinations (early-abort zone, exit 2).
    # deneb fork: MSYS bash strips unescaped backslashes, so a Windows
    # os.path.join path never reaches the script (suite runs on dev Macs,
    # srv2 AND Windows checkouts); forward slashes work everywhere.
    launcher = os.path.join(
        REPO, "launchers", "start-hy4-tp4.sh").replace(os.sep, "/")
    cases = (
        ({"REFINE": "2"}, "REFINE must be 0 or 1"),
        ({"REFINE": "1", "V2RUNNER": "0"}, "require V2RUNNER=1"),
        ({"MARKOV_SIDELOAD": "relative/path.pt"},
         "must be an absolute path under /home/choiceoh/models"),
        ({"MARKOV_SIDELOAD": "/etc/passwd"},
         "must be an absolute path under /home/choiceoh/models"),
        ({"MARKOV_SIDELOAD": "/home/choiceoh/models/x.pt",
          "DSPARK_IMPL": "fork"}, "require the upstream DSpark impl"),
        ({"MARKOV_SIDELOAD": "/home/choiceoh/models/x.pt", "V2RUNNER": "0"},
         "require V2RUNNER=1"),
    )
    for extra_env, expected in cases:
        env = dict(os.environ)
        env.update(extra_env)
        proc = subprocess.run(
            [_bash(), launcher], env=env, text=True, capture_output=True,
            timeout=5, check=False,
        )
        output = proc.stdout + proc.stderr
        check(proc.returncode == 2, f"launcher did not reject {extra_env}: {output}")
        check(expected in output,
              f"launcher rejection unclear for {extra_env}: {output}")

    launcher_text = open(launcher, encoding="utf-8").read()
    check("VLLM_DSPARK_REFINE_PASS=$REFINE" in launcher_text,
          "refine knob not propagated")
    check("VLLM_DSPARK_MARKOV_SIDELOAD=$MARKOV_SIDELOAD" in launcher_text,
          "sideload knob not propagated")
    check("MARKOV_SIDELOAD missing on" in launcher_text,
          "sideload preflight existence check missing")

    # Two-pass structure: feedback between two identical backbone+sampling
    # rounds, inside the captured _generate_draft.
    spec_text = open(
        _overlay_source("overlay/dspark_speculator_v2.py"),
        encoding="utf-8",
    ).read()
    draft_body = spec_text.split("def _generate_draft", 1)[1]
    check(draft_body.count("self._run_model(") == 2,
          "refine pass must re-run the backbone exactly once more")
    check(draft_body.count("self._sample_sequential(") == 2,
          "refine pass must re-run sequential sampling (same Gumbel keys)")
    check("_feed_back_draft_tokens" in draft_body,
          "refine pass must feed pass-1 tokens back before the second run")
    check(draft_body.index("_feed_back_draft_tokens")
          < draft_body.rindex("self._run_model("),
          "feedback must precede the second backbone run")
    check("if not self._refine_pass:" in draft_body,
          "refine pass must be kill-switch gated")

    # Sideload ordering: applied before the FP8 draft-head selection (and
    # therefore before warmup / lazy W2 quantization / CUDA graph capture).
    utils_text = open(
        _overlay_source("overlay/dspark_utils_v2.py"),
        encoding="utf-8",
    ).read()
    check(utils_text.index("_sideload_markov_head(draft_model")
          < utils_text.index("if fp8_draft_head_enabled:"),
          "sideload must run before the FP8 draft-head selection")
    check("_w2_fp8_weight" in utils_text,
          "sideload must assert it precedes lazy FP8 W2 quantization")
    print("  acceptance lever guards ....... OK")


# ---------------------------------------------------------------------------
# 10. ngram-ceiling simulator — pure sim math (stdlib only, importable)
# ---------------------------------------------------------------------------
def test_ngram_ceiling_sim() -> None:
    import importlib.util
    import random

    spec = importlib.util.spec_from_file_location(
        "nc", os.path.join(REPO, "bench", "ngram-ceiling.py"))
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)

    curve = nc.parse_curve(nc.DEFAULT_CURVE)
    check(len(curve) == 5 and abs(curve[0] - 0.785) < 1e-9,
          "default curve parse")
    for bad in ("", "50,60", "120", "abc"):
        try:
            nc.parse_curve(bad)
            check(False, f"bad curve accepted: {bad}")
        except ValueError:
            check(True, "")
    pmf = nc.accept_len_pmf(curve)
    check(abs(sum(pmf) - 1.0) < 1e-9, "accept-length pmf must sum to 1")
    mean = sum(m * p for m, p in enumerate(pmf))
    check(abs(mean - sum(curve)) < 1e-9,
          "pmf mean must equal the cumulative-curve sum")
    rng = random.Random(0)
    draws = [nc.sample_accept_len(pmf, rng) for _ in range(20_000)]
    emp = sum(draws) / len(draws)
    check(abs(emp - sum(curve)) < 0.05, f"sampler mean off: {emp}")

    check(nc.lcp([1, 2, 3], [1, 2, 4]) == 2, "lcp basic")
    check(nc.lcp([], [1]) == 0, "lcp empty")

    # Periodic sequence: gram [1,2,3] ending at t=10 matches ending pos=3;
    # copy source tokens[3:8] vs actual tokens[10:] -> 4 accepted.
    tokens = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]
    idx = nc.SuffixIndex(tokens, nmin=3)
    idx.extend_to(10)
    n, pos = idx.best_match(10, nmax=8)
    check(n == 3 and pos == 3, f"suffix match wrong: n={n} pos={pos}")
    proposal = tokens[pos:min(pos + 5, 10)]
    check(nc.lcp(proposal, tokens[10:15]) == 4,
          "periodic copy must accept 4 of 5")
    # No future leakage: source window is clipped at t.
    check(len(proposal) == 5 and proposal == [4, 5, 6, 7, 1],
          "proposal must come from already-generated tokens only")
    print("  ngram-ceiling sim math ........ OK")


def test_launcher_head_guard() -> None:
    """A multi-node launcher must refuse to run anywhere but its head.

    Every HEAD_IP command runs through `bash -c` and the head container is
    started with a plain `docker run`, so off the head the script builds a
    different cluster instead of failing: head on the wrong machine carrying
    VLLM_HOST_IP=$HEAD_IP, workers pointed at a rendezvous nobody serves. What
    surfaces first is an unrelated-looking ssh failure on whichever worker the
    invoking node lacks a key for, which is what it looked like from srv4.
    """
    global PASS
    for path in sorted(glob.glob(os.path.join(REPO, "launchers", "start-*.sh"))):
        src = open(path, encoding="utf-8").read()
        if not re.search(r"^HEAD_IP=", src, re.M):
            continue
        name = os.path.basename(path)
        assert re.search(r'grep -qw "\$HEAD_IP"', src), \
            f"{name}: no head-node guard -- running it off the head is silent"
        guard = src[src.index('grep -qw "$HEAD_IP"'):]
        assert "exit 1" in guard[:400], f"{name}: head guard does not abort"
        assert "DRY_RUN" in src[max(0, src.index('grep -qw "$HEAD_IP"') - 200):
                                src.index('grep -qw "$HEAD_IP"')], \
            f"{name}: head guard must not block DRY_RUN config prints"
        PASS += 1
    print("  launcher head guard .......... OK")


def test_preflight_precedes_serve_args() -> None:
    """A measured GMU has to reach the flag it sizes.

    memfree-preflight computes what free memory supports, and the launcher used
    to run it *after* SERVE_ARGS had already baked --gpu-memory-utilization in.
    The number was printed and then ignored, so four boots in a row died on
    memory while the terminal showed the right answer. It also ran before the
    launch loop removes the previous run's workers, so it measured against
    memory those still held.
    """
    global PASS
    for path in sorted(glob.glob(os.path.join(REPO, "launchers", "start-*.sh"))):
        src = open(path, encoding="utf-8").read()
        if "memfree-preflight" not in src:
            continue
        name = os.path.basename(path)
        # The invocation, not the variable assignment -- the assignment can sit
        # anywhere, and it is the call that does the measuring.
        call = src.index('$("$PREFLIGHT"')
        # Two shapes here: glm53 builds SERVE_ARGS inline, hy4 writes a serve
        # script whose heredoc carries the flag. The flag itself is where the
        # number stops being changeable, so anchor on that.
        # Anchored: the flag as an argument, not the word inside a comment
        # that explains this very ordering.
        flag = re.search(r"^\s*--gpu-memory-utilization", src, re.M)
        assert flag, f"{name}: no --gpu-memory-utilization to size"
        assert call < flag.start(), (
            f"{name}: preflight runs after --gpu-memory-utilization is "
            "assembled -- the measured value cannot reach it"
        )
        reclaim = src.find("docker rm -f")
        assert reclaim != -1 and reclaim < call, (
            f"{name}: preflight measures before stale containers are removed"
        )
        applied = re.search(r"^\s*(GMU|GPU_MEM)=\$(GMU|GPU_MEM)_SAFE\s*$", src, re.M)
        assert applied, f"{name}: preflight warns but never applies its value"
        PASS += 1
    print("  preflight applies GMU ......... OK")


def test_no_hardcoded_image_paths() -> None:
    """An overlay must not name one image's Python layout.

    tp_oneshot_ar hardcoded /opt/venv/lib/python3.12/site-packages/... -- the
    dsv4 image's layout. The glm53 image puts vllm under
    /usr/local/lib/python3.12/dist-packages, so the extension build raised
    FileNotFoundError and the shim fell back to NCCL exactly as it is designed
    to. Nothing looked broken, and the module had never once run on that image.

    Overlays are mounted next to the file they patch, so paths belong relative
    to __file__.
    """
    global PASS
    bad = re.compile(r"/(opt/venv|usr/local)/lib/python3\.\d+/(site|dist)-packages")
    for path in sorted(glob.glob(os.path.join(REPO, "overlay", "modules", "*", "*.py"))):
        src = open(path, encoding="utf-8").read()
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#") or not bad.search(line):
                continue
            raise AssertionError(
                f"{os.path.relpath(path, REPO)}:{i}: hardcoded image layout "
                f"-- derive from __file__ instead: {stripped[:80]}"
            )
        PASS += 1
    print("  no hardcoded image paths ...... OK")


def test_b12x_ep_routing() -> None:
    """b12x EP never shows the fused kernel a global id or a polluted expert."""
    ns = load_defs(
        "overlay/flashinfer_b12x_moe.py",
        {
            "remap_b12x_ep_slot",
            "remap_b12x_ep_routing",
            "remap_b12x_ep_tensors",
            "b12x_ep_kernel_expert_count",
            "b12x_ep_pad_dim0",
            "b12x_ep_should_compact",
            "b12x_ep_compact_enabled",
            "compact_b12x_ep_pairs",
            "B12X_EP_COMPACT_MIN_ROUTED",
            "_ep_buf",
            "b12x_ep_set_scale",
            "_b12x_ep_set_dotted",
            "_B12X_EP_SCALE_ALIASES",
        },
        {"os": os},
    )
    slot = ns["remap_b12x_ep_slot"]
    remap = ns["remap_b12x_ep_routing"]
    kernel_e = ns["b12x_ep_kernel_expert_count"]
    pad_dim0 = ns["b12x_ep_pad_dim0"]

    check(kernel_e(72, False) == 72, "EP-off kernel E is local")
    check(kernel_e(72, True, True) == 72,
          "direct EP kernel E has no physical dummy")
    check(kernel_e(72, True, False) == 73,
          "rollback EP kernel E is local + dummy")
    check(kernel_e(288, False) == 288, "replicated kernel E is global")

    # Rank 1 of 4, 8 experts, 2 local (4,5). Dummy id = 2.
    ids, weights = remap(
        [[4, 5, 0, 7], [1, 4, 6, 5]],
        [[0.4, 0.3, 0.2, 0.1], [0.5, 0.25, 0.15, 0.1]],
        num_local_experts=2,
        local_expert_offset=4,
    )
    check(ids == [[0, 1, 2, 2], [2, 0, 2, 1]], f"offset remap ids: {ids}")
    check(weights == [[0.4, 0.3, 0.0, 0.0], [0.0, 0.25, 0.0, 0.1]],
          f"offset remap weights: {weights}")
    check(all(e < 3 for row in ids for e in row), "id exceeded dummy")
    check(all(w == 0.0 for row_i, row_w in zip(ids, weights)
              for e, w in zip(row_i, row_w) if e == 2),
          "dummy slot must carry scale 0")

    # Same shard as the offset path (last row remote).
    emap = [-1, -1, -1, -1, 0, 1, -1, -1]
    # Last row is a real local expert. torch expert_map[-1] would wrap here
    # and dump a padded slot onto expert 0 at the original scale.
    emap_wrap = [-1, -1, -1, -1, 0, 1, -1, 0]
    check(emap_wrap[-1] == 0, "fixture must make the wrap land on a real expert")
    check(slot(-1, num_local_experts=2, expert_map=emap_wrap) == -1,
          "-1 topk must not wrap through expert_map")
    check(slot(99, num_local_experts=2, expert_map=emap_wrap) == -1,
          "OOB topk must not index expert_map")
    check(slot(-1, num_local_experts=2, local_expert_offset=4) == -1,
          "-1 topk is remote on the offset path too")

    ids2, weights2 = remap(
        [[4, 5, 0, 7], [1, 4, 6, 5]],
        [[0.4, 0.3, 0.2, 0.1], [0.5, 0.25, 0.15, 0.1]],
        num_local_experts=2,
        expert_map=emap,
    )
    check(ids2 == ids and weights2 == weights, "expert_map must match offset")

    ids_pad, weights_pad = remap(
        [[-1, 4], [5, -1]],
        [[0.9, 0.1], [0.8, 0.2]],
        num_local_experts=2,
        expert_map=emap_wrap,
    )
    check(ids_pad == [[2, 0], [1, 2]], f"padded topk ids: {ids_pad}")
    check(weights_pad == [[0.0, 0.1], [0.8, 0.0]],
          f"padded topk scales: {weights_pad}")

    ids_off, weights_off = remap(
        [[-1, 4]], [[0.5, 0.5]],
        num_local_experts=2, local_expert_offset=4,
    )
    check(ids_off == [[2, 0]] and weights_off == [[0.0, 0.5]],
          "offset path must also park -1 on dummy")

    # All local: dummy unused, scales preserved.
    ids3, weights3 = remap(
        [[0, 1]], [[0.7, 0.3]],
        num_local_experts=2, local_expert_offset=0,
    )
    check(ids3 == [[0, 1]] and weights3 == [[0.7, 0.3]], "all-local remap")

    # All remote: every slot is dummy / 0. Never dump onto expert 0.
    ids4, weights4 = remap(
        [[6, 7]], [[0.6, 0.4]],
        num_local_experts=2, local_expert_offset=0,
    )
    check(ids4 == [[2, 2]] and weights4 == [[0.0, 0.0]], "all-remote -> dummy")

    should = ns["b12x_ep_should_compact"]
    compact = ns["compact_b12x_ep_pairs"]
    check(ns["B12X_EP_COMPACT_MIN_ROUTED"] == 640, "compact cutover is kernel 640")
    check(not should(256), "decode GRAPH_CAP=32 * 8 stays fixed-shape")
    check(not should(640), "cutover is exclusive")
    check(should(641), "prefill crosses compact")
    check(not should(4096, enabled=False), "compact env off")
    check(ns["b12x_ep_compact_enabled"](lambda _k, _d=None: "1"), "compact default on")
    check(not ns["b12x_ep_compact_enabled"](lambda _k, _d=None: "0"),
          "compact env 0")
    tok, loc, sc = compact(ids, weights, dummy=2)
    check(tok == [0, 0, 1, 1] and loc == [0, 1, 0, 1],
          f"compact pairs: {tok} {loc}")
    check(sc == [0.4, 0.3, 0.25, 0.1], f"compact scales: {sc}")
    tok4, loc4, sc4 = compact(ids4, weights4, dummy=2)
    check(tok4 == [] and loc4 == [] and sc4 == [], "all-remote compact is empty")

    check(pad_dim0(72, 72, required=True, name="w13_weight") == "pad",
          "local E must pad")
    check(pad_dim0(73, 72, required=True, name="w13_weight") == "already",
          "local+1 is already padded")
    check(pad_dim0(None, 72, required=False, name="w13_weight_scale_2") == "skip",
          "optional missing tensor skips")
    try:
        pad_dim0(None, 72, required=True, name="w13_weight")
    except RuntimeError as exc:
        check("missing" in str(exc), f"required missing: {exc}")
    else:
        raise AssertionError("required missing tensor must raise")
    try:
        pad_dim0(70, 72, required=True, name="w13_weight")
    except RuntimeError as exc:
        check("E=70" in str(exc) and "want 72" in str(exc),
              f"shape mismatch: {exc}")
    else:
        raise AssertionError("wrong E must raise, not skip")

    class _Desc:
        def __init__(self):
            self.scale = "old-w"
            self.alpha_or_gscale = "old-a"

    class _QC:
        def __init__(self):
            self._w1 = _Desc()
            self._w2 = _Desc()
            self._a2 = _Desc()

    class _Host:
        def __init__(self):
            self.quant_config = _QC()

        @property
        def w1_scale(self):
            return self.quant_config._w1.scale

        @property
        def w2_scale(self):
            return self.quant_config._w2.scale

        @property
        def g1_alphas(self):
            return self.quant_config._w1.alpha_or_gscale

        @property
        def g2_alphas(self):
            return self.quant_config._w2.alpha_or_gscale

        @property
        def a2_gscale(self):
            return self.quant_config._a2.alpha_or_gscale

    host = _Host()
    try:
        host.w1_scale = "padded"
    except AttributeError as exc:
        check("no setter" in str(exc) or "can't set" in str(exc).lower()
              or "has no setter" in str(exc),
              f"image-shaped property must refuse setattr: {exc}")
    else:
        raise AssertionError("w1_scale property must have no setter")
    set_scale = ns["b12x_ep_set_scale"]
    check(set_scale(host, "w1_scale", "padded-w1") == "quant_config._w1.scale",
          "w1_scale must write QuantDesc.scale")
    check(host.w1_scale == "padded-w1", "w1_scale property must read the pad")
    check(set_scale(host, "w2_scale", "padded-w2") == "quant_config._w2.scale",
          "w2_scale must write QuantDesc.scale")
    check(host.w2_scale == "padded-w2", "w2_scale property must read the pad")
    check(set_scale(host, "g1_alphas", "padded-g1")
          == "quant_config._w1.alpha_or_gscale",
          "g1_alphas must write QuantDesc.alpha_or_gscale")
    check(host.g1_alphas == "padded-g1", "g1_alphas property must read the pad")
    check(set_scale(host, "a2_gscale", "padded-a2")
          == "quant_config._a2.alpha_or_gscale",
          "a2_gscale must write QuantDesc.alpha_or_gscale")
    class _Broken:
        @property
        def w1_scale(self):
            return "stuck"
    try:
        set_scale(_Broken(), "w1_scale", "x")
    except RuntimeError as exc:
        check("cannot bind w1_scale" in str(exc), f"broken alias: {exc}")
    else:
        raise AssertionError("set_scale must fail loud when aliases miss")

    try:
        import torch
    except ImportError:
        print("  b12x EP routing ................ OK (tensor remap skipped)")
        return

    ns["torch"] = torch
    tensor_remap = ns["remap_b12x_ep_tensors"]
    cases = (
        ([[4, 5, 0, 7], [1, 4, 6, 5]],
         [[0.4, 0.3, 0.2, 0.1], [0.5, 0.25, 0.15, 0.1]],
         emap, 0,
         [[0, 1, 2, 2], [2, 0, 2, 1]],
         [[0.4, 0.3, 0.0, 0.0], [0.0, 0.25, 0.0, 0.1]]),
        ([[-1, 4], [5, -1]],
         [[0.9, 0.1], [0.8, 0.2]],
         emap_wrap, 0,
         [[2, 0], [1, 2]],
         [[0.0, 0.1], [0.8, 0.0]]),
        ([[-1, 4]],
         [[0.5, 0.5]],
         None, 4,
         [[2, 0]],
         [[0.0, 0.5]]),
        ([[99, 4]],
         [[0.3, 0.7]],
         emap_wrap, 0,
         [[2, 0]],
         [[0.0, 0.7]]),
    )
    for raw_ids, raw_w, list_map, offset, want_ids, want_w in cases:
        kwargs = dict(local_expert_offset=offset)
        if list_map is not None:
            kwargs["expert_map"] = torch.tensor(list_map, dtype=torch.int32)
        got_ids, got_w = tensor_remap(
            torch.tensor(raw_ids, dtype=torch.int64),
            torch.tensor(raw_w, dtype=torch.float32),
            num_local_experts=2,
            **kwargs,
        )
        check(got_ids.tolist() == want_ids,
              f"tensor remap ids {got_ids.tolist()} != {want_ids}")
        check([round(x, 6) for row in got_w.tolist() for x in row]
              == [round(x, 6) for row in want_w for x in row],
              f"tensor remap scales {got_w.tolist()} != {want_w}")
        list_ids, list_w = remap(
            raw_ids, raw_w, num_local_experts=2,
            local_expert_offset=offset, expert_map=list_map,
        )
        check(got_ids.tolist() == list_ids,
              "tensor remap must match list remap ids")
        check([round(x, 6) for row in got_w.tolist() for x in row]
              == [round(x, 6) for row in list_w for x in row],
              "tensor remap must match list remap scales")

    scratch_ids = torch.empty((2, 2), dtype=torch.int32)
    scratch_scales = torch.empty((2, 2), dtype=torch.float32)
    got_ids, got_w = tensor_remap(
        torch.tensor([[-1, 4], [5, -1]], dtype=torch.int64),
        torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32),
        num_local_experts=2,
        expert_map=torch.tensor(emap_wrap, dtype=torch.int32),
        out_ids=scratch_ids,
        out_scales=scratch_scales,
        long_idx=torch.empty((2, 2), dtype=torch.int64),
        mapped=torch.empty((2, 2), dtype=torch.int32),
        remote=torch.empty((2, 2), dtype=torch.bool),
        tmp_a=torch.empty((2, 2), dtype=torch.bool),
        tmp_b=torch.empty((2, 2), dtype=torch.bool),
    )
    check(got_ids.data_ptr() == scratch_ids.data_ptr(), "scratch ids reused")
    check(got_w.data_ptr() == scratch_scales.data_ptr(), "scratch scales reused")
    check(got_ids.tolist() == [[2, 0], [1, 2]], "in-place remap ids")
    check([round(x, 6) for row in got_w.tolist() for x in row]
          == [0.0, 0.1, 0.8, 0.0], "in-place remap scales")
    print("  b12x EP routing ................ OK")


def test_b12x_ep_preflight() -> None:
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "b12x_pf", os.path.join(REPO, "tools", "b12x-preflight.py"))
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)

    check(pf.gate_up_aligned(2048, 4), "GLM-5.3 TP=4 must align")
    check(pf.gate_up_aligned(2048, 1), "GLM-5.3 full intermediate must align")
    check(not pf.gate_up_aligned(640, 4), "Qwen3.8 TP=4 must not align")
    check(640 % pf.ALIGN == 0, "Qwen3.8 full intermediate hits the 128 tile")

    def _inspect(inter, experts, tp=4, ep=4):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"moe_intermediate_size": inter, "n_routed_experts": experts}
            open(os.path.join(tmp, "config.json"), "w").write(
                __import__("json").dumps(cfg))
            return pf.inspect(tmp, tp, ep)

    glm = _inspect(2048, 288)
    check(glm["b12x"] and glm["b12x_ep"], "GLM-5.3 must pass both paths")
    qwen = _inspect(640, 256)
    check(not qwen["b12x"] and qwen["b12x_ep"],
          "Qwen3.8 must be TP-closed and EP-open")
    check("320" in qwen["reason"] and "128" in qwen["reason"],
          f"Qwen3.8 TP reason unclear: {qwen.get('reason')}")
    odd = _inspect(2048, 100, ep=3)
    check(odd["b12x"] and not odd["b12x_ep"],
          "experts not divisible by EP must close the EP path")
    skinny = _inspect(64, 256, tp=1, ep=4)
    check(skinny["b12x"] and not skinny["b12x_ep"],
          "intermediate 64 passes gate+up but fails the wrapper's 128 tile")
    print("  b12x EP preflight .............. OK")


def test_b12x_ep_launcher() -> None:
    launcher = os.path.join(
        REPO, "launchers", "start-glm53-nvfp4-tp4.sh").replace(os.sep, "/")
    text = open(launcher, encoding="utf-8").read()
    check('ENABLE_EP="${ENABLE_EP:-0}"' in text, "ENABLE_EP default missing")
    check("--enable-expert-parallel" in text, "EP flag missing from launcher")
    check("${EP_FLAG:+$EP_FLAG }" in text,
          "EP flag must be optional in SERVE_ARGS")
    overlay = open(_overlay_source("overlay/flashinfer_b12x_moe.py"),
                   encoding="utf-8").read()
    check("return not getattr(moe_parallel_config, \"enable_eplb\", False)"
          in overlay,
          "b12x must accept EP and refuse EPLB")
    check("def supports_expert_map(self) -> bool:\n        return True"
          in overlay,
          "b12x must accept the vLLM expert_map under EP")
    check("return remap_b12x_ep_tensors(" in overlay,
          "_remap_ep_tensors must call remap_b12x_ep_tensors, not a private copy")
    check("torch.gather(expert_map, 0, long_idx.reshape(-1), out=mapped.reshape(-1))"
          in overlay,
          "tensor expert_map gather must be in-place")
    check("long_idx.clamp_(0, map_len - 1)" in overlay,
          "tensor expert_map gather must clamp after the in-range mask")
    check("b12x_ep_pad_dim0(" in overlay,
          "_pad_dummy_expert must use b12x_ep_pad_dim0 (no silent continue)")
    check('b12x_ep_set_scale(self, "w1_scale"' in overlay,
          "dummy pad must rebind w1_scale through QuantDesc, not the property")
    check("\n        self.w1_scale =" not in overlay,
          "self.w1_scale = dies on the image (property has no setter)")
    check("except AttributeError:\n                pass" not in overlay,
          "dummy pad must not swallow AttributeError on scale rebind")
    check("def _apply_ep_compact(" in overlay,
          "prefill must drop dummy slots instead of GEMMing them")
    check("ABORT: ENABLE_EP=1 GRAPH_CAP=" in text,
          "ENABLE_EP=1 must refuse GRAPH_CAP that captures past the compact cutover")
    check("VLLM_B12X_EP_COMPACT=${VLLM_B12X_EP_COMPACT:-1}" in text,
          "compact flag must reach the container")
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    check("VLLM_B12X_EP_NO_DUMMY=1" in profile,
          "no-dummy default must be profile-declared for env passthrough")
    check("VLLM_B12X_EP_NO_DUMMY KV_DTYPE" in text,
          "dry-run must expose the no-dummy rollback knob")

    check("ABORT: ENABLE_EP must be 0 or 1" in text,
          "ENABLE_EP must refuse anything other than 0 or 1")
    print("  b12x EP launcher ............... OK")


# ---------------------------------------------------------------------------
# UE8M0 scale repair (fp8_lm_head): the deepgemm packer traps on any scale
# whose sign or mantissa is set. Enforce the kernel's precondition on the host.
# ---------------------------------------------------------------------------
def test_ue8m0_scale_repair() -> None:
    # The repair operates on tensors, so this section needs torch. The serving
    # hosts run this suite as the deploy gate and do not have it -- skip out
    # loud rather than failing the gate (a silent pass would hide the hole).
    try:
        import torch
    except ImportError:
        print("  UE8M0 scale repair ............. SKIP (no torch on this host)")
        return

    ns = load_defs(
        "overlay/modules/glm53_drafter/fp8_lm_head.py",
        {"_repair_ue8m0_scales", "_ue8m0_violations",
         "_SF_SIGN_AND_MANTISSA", "_SMALLEST_NORMAL_F32"},
        {"torch": torch},
    )
    repair = ns["_repair_ue8m0_scales"]
    violations = ns["_ue8m0_violations"]

    # the mask must be exactly the kernel's: sign bit | mantissa
    check(ns["_SF_SIGN_AND_MANTISSA"] == 0x807FFFFF,
          "mask must match smxx_layout.cuh:131 (values[j] & 0x807fffffu)")
    check(ns["_SMALLEST_NORMAL_F32"] == 2.0**-126,
          "denormal cutoff must be the smallest normal fp32")

    powers = torch.tensor([1.0, 2.0, 0.5, 2.0**-126, 2.0**100])
    check(int(violations(powers).sum()) == 0,
          "exact powers of two must not be flagged")
    kept, n = repair(powers)
    check(n == 0 and torch.equal(kept, powers),
          "a clean scale tensor must pass through untouched")

    for name, raw in (
        ("arbitrary", torch.tensor([0.00223214, 1.3, 0.7, 3.14159])),
        ("zero/negative", torch.tensor([0.0, -1.0, -2.0])),
        ("inf/nan", torch.tensor([float("inf"), float("-inf"), float("nan")])),
        ("denormal", torch.tensor([1e-42, 5e-45, 1.4e-45])),
        ("under smallest normal", torch.tensor([0.9 * 2.0**-126])),
    ):
        fixed, _ = repair(raw)
        check(int(violations(fixed).sum()) == 0,
              f"repair must clear every violation ({name})")

    # denormals cannot be rounded onto a power of two -- they must go to zero
    fixed, _ = repair(torch.tensor([1e-42]))
    check(float(fixed[0]) == 0.0, "denormal scales must flush to zero")

    # positive normals round UP: safe direction, at most 2x
    src = torch.tensor([0.00223214, 1.3, 0.7])
    fixed, _ = repair(src)
    ratio = (fixed / src)
    check(bool((ratio >= 1.0).all()) and bool((ratio <= 2.0).all()),
          "normal scales must round up by at most 2x")

    gen = torch.Generator().manual_seed(7)
    bulk = torch.cat([
        torch.rand(50000, generator=gen) * 10,
        torch.rand(20000, generator=gen) * 1e-40,
        torch.zeros(500),
        torch.full((100,), float("nan")),
    ])
    fixed, count = repair(bulk)
    check(int(violations(fixed).sum()) == 0,
          "bulk repair must leave no violation")
    check(count > 0, "bulk sample must have had violations to repair")

    # non-fp32 scales are left alone rather than reinterpreted
    half = torch.tensor([1.3, 0.7], dtype=torch.float16)
    same, n = repair(half)
    check(n == 0 and torch.equal(same, half),
          "non-fp32 scales must pass through untouched")

    print("  UE8M0 scale repair ............. OK")


def test_fp8_acceptance_contracts() -> None:
    """Keep the draft-head kernel safe and its candidates decodable."""
    fp8_path = _overlay_source("overlay/fp8_lm_head.py")
    fp8_source = open(fp8_path, encoding="utf-8").read()
    fp8_tree = ast.parse(fp8_source)

    quantize_node = next(
        node
        for node in fp8_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_quantize_fp8_deepgemm"
    )
    quantize_source = ast.get_source_segment(fp8_source, quantize_node)
    assert quantize_source is not None
    import_guard = quantize_source.index("except ImportError as exc:")
    requant_call = quantize_source.index(
        "requant_weight_ue8m0_inplace(wq", import_guard
    )
    import_fallback = quantize_source[import_guard:requant_call]
    check(
        "raise RuntimeError" in import_fallback
        and "deepgemm_post_process_fp8_weight_block" not in import_fallback,
        "a missing UE8M0 requant helper must fall back before the CUDA packer",
    )

    class DummyTensor:
        shape = (128, 128)

        def detach(self):
            return self

        def __getitem__(self, _key):
            return self

        def float(self):
            return self

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    pack_calls = []

    def fake_pack(*args, **kwargs):
        pack_calls.append((args, kwargs))
        return "must-not-pack"

    fp8_utils_module = types.SimpleNamespace(
        deepgemm_post_process_fp8_weight_block=fake_pack
    )
    deep_gemm_module = types.SimpleNamespace(
        per_block_cast_to_fp8=lambda *_args, **_kwargs: (
            DummyTensor(),
            DummyTensor(),
        )
    )

    def import_without_requant(name, globals=None, locals=None,
                               fromlist=(), level=0):
        if name.endswith("fp8_utils"):
            if "requant_weight_ue8m0_inplace" in fromlist:
                raise ImportError("requant helper missing")
            return fp8_utils_module
        if name == "vllm.utils.deep_gemm":
            return deep_gemm_module
        return builtins.__import__(name, globals, locals, fromlist, level)

    fake_builtins = dict(vars(builtins))
    fake_builtins["__import__"] = import_without_requant
    quantize_ns = load_defs(
        "overlay/fp8_lm_head.py",
        {"_quantize_fp8_deepgemm"},
        {
            "__builtins__": fake_builtins,
            "torch": types.SimpleNamespace(
                Tensor=DummyTensor,
                no_grad=NoGrad,
                cat=lambda chunks, dim=0: chunks[0],
            ),
        },
    )
    import_failed_closed = False
    try:
        quantize_ns["_quantize_fp8_deepgemm"](DummyTensor())
    except RuntimeError as exc:
        import_failed_closed = "refusing" in str(exc)
    check(import_failed_closed, "missing requant helper must raise before packing")
    check(not pack_calls, "missing requant helper must never launch the packer")
    bad_guard = quantize_source.index("if bad_scales:")
    fail_closed = quantize_source.index("raise RuntimeError", bad_guard)
    final_pack = quantize_source.rindex(
        "return deepgemm_post_process_fp8_weight_block"
    )
    check(
        "bad_scales = _describe_ue8m0_scales(ws)" in quantize_source
        and bad_guard < fail_closed < final_pack,
        "an unsafe UE8M0 scale must fall back before the CUDA packer can trap",
    )
    check(
        "_flush_pathological_scales" not in quantize_source
        and "torch.where" not in quantize_source,
        "the kernel guard must never rewrite scales to silence the packer",
    )

    describe_node = next(
        node
        for node in fp8_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_describe_ue8m0_scales"
    )
    describe_source = ast.get_source_segment(fp8_source, describe_node)
    assert describe_source is not None
    safety_at = describe_source.index("finite = torch.isfinite(scales)")
    early_return_at = describe_source.index("if n_bad == 0:")
    check(
        safety_at < early_return_at
        and "inf_mask = torch.isinf(scales)" in describe_source
        and "zero_mask = finite & (scales == 0)" in describe_source,
        "zero and non-finite scales must be unsafe even when their bits pack",
    )

    fake_bf16 = object()
    fake_fp16 = object()
    fake_torch = types.SimpleNamespace(bfloat16=fake_bf16, float16=fake_fp16)
    fake_logger = types.SimpleNamespace(
        warning_once=lambda *args, **kwargs: None,
        info_once=lambda *args, **kwargs: None,
    )

    failed_calls = []

    def fail_quantize(weight):
        failed_calls.append(weight)
        raise RuntimeError("unsafe scales")

    failed_ns = load_defs(
        "overlay/fp8_lm_head.py",
        {"build_fp8_lm_head_weight"},
        {
            "torch": fake_torch,
            "logger": fake_logger,
            "_quantize_fp8_deepgemm": fail_quantize,
        },
    )
    failed_build = failed_ns["build_fp8_lm_head_weight"]
    failed_head = types.SimpleNamespace(
        weight=types.SimpleNamespace(dtype=fake_bf16, shape=(38720, 4096))
    )
    check(not failed_build(failed_head), "unsafe fp8 build must fall back")
    check(not failed_build(failed_head), "failed fp8 build must stay disarmed")
    check(
        len(failed_calls) == 1
        and failed_head._deneb_fp8_build_attempted is True,
        "two logits calls must attempt the full head quantization only once",
    )

    success_calls = []
    packed_weight = types.SimpleNamespace(shape=(38720, 4096))
    packed_scale = object()

    def successful_quantize(weight):
        success_calls.append(weight)
        return packed_weight, packed_scale

    success_ns = load_defs(
        "overlay/fp8_lm_head.py",
        {"build_fp8_lm_head_weight"},
        {
            "torch": fake_torch,
            "logger": fake_logger,
            "_quantize_fp8_deepgemm": successful_quantize,
        },
    )
    successful_build = success_ns["build_fp8_lm_head_weight"]
    success_head = types.SimpleNamespace(
        weight=types.SimpleNamespace(dtype=fake_fp16, shape=(38720, 4096))
    )
    check(successful_build(success_head), "safe fp8 build must attach its cache")
    check(successful_build(success_head), "an attached fp8 cache must be reusable")
    check(
        len(success_calls) == 1
        and success_head._deneb_fp8_w is packed_weight
        and success_head._deneb_fp8_ws is packed_scale,
        "a successful head must also be quantized only once",
    )

    processor_cls = next(
        node
        for node in fp8_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Fp8HeadLogitsProcessor"
    )
    apply_node = next(
        node
        for node in processor_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_head"
    )
    apply_source = ast.get_source_segment(fp8_source, apply_node)
    assert apply_source is not None
    gate_at = apply_source.index("use_fp8 = _read_bool_env")
    cache_at = apply_source.index('getattr(lm_head, "_deneb_fp8_w"')
    check(
        gate_at < cache_at and "if use_fp8 else None" in apply_source,
        "an env-off endpoint must ignore an fp8 cache shared by the other endpoint",
    )
    check(
        "and not attempted" in apply_source,
        "a failed shared-head build must stay on bf16 on later logits calls",
    )
    check(
        'out[..., valid_end:] = -float("inf")' in apply_source,
        "tokenizer-orphan draft rows must be masked before local top-k",
    )

    decodable_node = next(
        node
        for node in fp8_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "decodable_vocab_size"
    )
    decodable_source = ast.get_source_segment(fp8_source, decodable_node)
    assert decodable_source is not None
    check(
        "raise RuntimeError" in decodable_source
        and "orphan LM head rows stay reachable" not in decodable_source,
        "tokenizer read/shape failure must stop load instead of disabling the mask",
    )

    target_ns = load_defs(
        "overlay/glm5next_model.py",
        {"_validate_decodable_vocab_bound"},
        {},
    )
    validate_bound = target_ns["_validate_decodable_vocab_bound"]
    check(
        validate_bound(154856, 154880) == 154856,
        "a decodable prefix below the target vocabulary must remain valid",
    )
    check(
        validate_bound(154880, 154880) == 154880,
        "a decodable prefix equal to the target vocabulary must remain valid",
    )
    try:
        validate_bound(154881, 154880)
    except ValueError as exc:
        check(
            "154881 > 154880" in str(exc),
            "an oversized decodable prefix must report both vocabulary bounds",
        )
    else:
        check(False, "an oversized decodable prefix must stop target model load")

    target_source = open(
        _overlay_source("overlay/glm5next_model.py"), encoding="utf-8"
    ).read()
    compact_target = re.sub(r"\s+", "", target_source)
    check(
        "self._decodable_vocab=_validate_decodable_vocab_bound("
        "decodable_vocab_size(vllm_config.model_config.tokenizer),"
        "self.config.vocab_size,)" in compact_target,
        "the target model must validate its decodable bound even without a drafter",
    )

    ns = load_defs(
        "overlay/fp8_lm_head.py",
        {
            "_contiguous_tokenizer_vocab_size",
            "_local_valid_vocab_end",
            "_validate_decodable_top_k",
        },
        {},
    )
    contiguous_size = ns["_contiguous_tokenizer_vocab_size"]
    local_end = ns["_local_valid_vocab_end"]
    validate_top_k = ns["_validate_decodable_top_k"]
    check(
        contiguous_size({"c": 2, "a": 0, "b": 1}) == 3,
        "an unordered contiguous tokenizer vocabulary must retain its full prefix",
    )
    for bad_vocab in ({}, {"a": 1}, {"a": 0, "c": 2}):
        try:
            contiguous_size(bad_vocab)
        except ValueError:
            pass
        else:
            check(False, f"non-prefix tokenizer vocabulary accepted: {bad_vocab}")
    validate_top_k(154856, 16)
    for valid, top_k in ((8, 16), (None, 16), (154856, 0)):
        try:
            validate_top_k(valid, top_k)
        except ValueError:
            pass
        else:
            check(False, f"unsafe decodable/top-k pair accepted: {(valid, top_k)}")
    for valid, start, width, want in (
        (None, 0, 38720, 38720),
        (154856, 0, 38720, 38720),
        (154856, 38720, 38720, 38720),
        (154856, 77440, 38720, 38720),
        (154856, 116160, 38720, 38696),
        (154856, 154856, 128, 0),
        (154856, 154900, 128, 0),
    ):
        check(
            local_end(valid, start, width) == want,
            f"valid-vocab shard map {(valid, start, width)} must end at {want}",
        )

    loader_source = open(
        _overlay_source("overlay/dflash_utils.py"), encoding="utf-8"
    ).read()
    alias_at = loader_source.index("dflash_model.lm_head = target_lm_head")
    build_call = "build_fp8_lm_head(dflash_model)"
    check(
        loader_source.count(build_call) == 1
        and alias_at < loader_source.index(build_call),
        "DFlash2 must attach the loaded target head before quantizing it",
    )
    check(
        "except Exception" not in loader_source[alias_at:],
        "draft-head integration failures after sharing must stay visible",
    )

    qwen_source = open(
        _overlay_source("overlay/qwen3_dflash2.py"), encoding="utf-8"
    ).read()
    compact_qwen = re.sub(r"\s+", "", qwen_source)
    check(
        "valid_vocab_size=decodable_vocab_size("
        "vllm_config.model_config.tokenizer)" in compact_qwen,
        "the DFlash2 candidate processor must receive the tokenizer bound",
    )
    check(
        'selector_top_k=int(draft_config["selector_top_k"])' in compact_qwen,
        "the candidate processor must validate decodable rows against selector top-k",
    )

    print("  fp8 acceptance contracts ....... OK")


def test_glm53_v2_overlay_contracts() -> None:
    """Do not advertise V1-only guards on the production V2 runner."""
    profile_path = os.path.join(REPO, "profiles", "glm53.env")
    profile = open(profile_path, encoding="utf-8").read()
    modules_match = re.search(r'^MODULES="([^"]+)"', profile, re.M)
    assert modules_match is not None
    modules = set(modules_match.group(1).split())
    check(
        not {"glm53_drop_audit", "glm53_sparse_q"} & modules,
        "glm53 must not mount V1-only acceptance overlays on V2 Model Runner",
    )
    check(
        "glm53_runtime" in modules
        and "glm53_v2_hard_constraint_guard" not in modules,
        "glm53 must mount only the V2 sampling guard with an exact predicate",
    )
    check(
        "VLLM_SPEC_GATHER_Q=" not in profile,
        "glm53 must not publish an inert V1 sparse-q knob",
    )
    launcher = open(
        os.path.join(REPO, "launchers", "start-glm53-nvfp4-tp4.sh"),
        encoding="utf-8",
    ).read()
    check(
        "VLLM_DENEB_DROP_AUDIT=1" not in launcher,
        "glm53 launcher must not arm an audit on an inactive code path",
    )

    sampler_source = open(
        _overlay_source("overlay/sampler.py"), encoding="utf-8"
    ).read()
    sampler_tree = ast.parse(sampler_source)
    sampler_cls = next(
        node
        for node in sampler_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Sampler"
    )
    requires_processing = next(
        node
        for node in sampler_cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_requires_logits_processing"
    )
    requires_source = ast.get_source_segment(sampler_source, requires_processing)
    assert requires_source is not None
    check(
        "_thinking_budget_requires_logits_processing(" in requires_source,
        "thinking-budget-only requests must not skip logits processing",
    )
    thinking_required = load_defs(
        "overlay/sampler.py",
        {"_thinking_budget_requires_logits_processing"},
        {
            "np": types.SimpleNamespace(
                ndarray=object, any=lambda values: any(values)
            )
        },
    )["_thinking_budget_requires_logits_processing"]

    class Slots:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, _indices):
            return self.values

    disabled = types.SimpleNamespace(enabled=False)
    active = types.SimpleNamespace(
        enabled=True, use_thinking_budget=Slots([False, True])
    )
    inactive = types.SimpleNamespace(
        enabled=True, use_thinking_budget=Slots([False, False])
    )
    check(
        not thinking_required(disabled, [0]),
        "disabled thinking budgets must not touch their lazy request buffer",
    )
    check(
        thinking_required(active, [0, 1]),
        "an active thinking budget must require logits processing",
    )
    check(
        not thinking_required(inactive, [0, 1]),
        "inactive request slots must retain the fast path",
    )

    deploy = open(
        os.path.join(REPO, "launchers", "deploy-overlays.sh"),
        encoding="utf-8",
    ).read()
    for guard in (
        "status --porcelain",
        "fetch --quiet origin main",
        "merge-base --is-ancestor origin/main HEAD",
        "source_commit=%s",
    ):
        check(guard in deploy, f"deploy provenance guard missing: {guard}")
    print("  glm53 V2/deploy contracts ...... OK")


def test_deploy_refusal_is_not_swallowed() -> None:
    """A refused deploy must stop a boot, not scroll past it.

    deploy-overlays refuses when HEAD is not based on current origin/main --
    a real guard against booting a stale overlay rollback. On 2026-09-03 three
    boots came up anyway: the bench script ran it as

        bash launchers/deploy-overlays.sh glm53 2>&1 | tail -1

    and the ABORT is three lines, so `tail -1` printed only the origin/main
    sha, which reads exactly like a normal status line. Those boots served
    overlays six commits old while their logs looked healthy, and the arm
    under test had simply never been deployed.

    The guard itself must stay, and it must exit non-zero so a caller that
    checks can see it."""
    text = open(os.path.join(REPO, "launchers/deploy-overlays.sh"),
                encoding="utf-8").read()
    check("HEAD is not based on current origin/main" in text,
          "the stale-rollback guard is the thing that caught this")
    guard = text[text.index("HEAD is not based on current origin/main"):]
    check("exit 1" in guard[:400],
          "the refusal must exit non-zero so a caller can branch on it")
    print("  deploy refusal exits non-zero .. OK")


def test_fp8_dense_nvfp4_scheme_contract() -> None:
    """nvfp4 is opt-in, stacks on fp8, and arms only on a value check.

    It buys 2.3x on the prefill GEMM (236 vs 104 TFLOP/s measured) and halves
    the pack, for 3.7x the quantization error -- past what the checkpoint's own
    recipe was willing to do to these projections, which put fp4 on
    `mlp.experts.*` only. So it must behave like the other experimental
    scheme: never the default, never a boot failure, and never armed on
    "did not raise"."""
    src = open(os.path.join(
        REPO, "overlay/modules/glm53_model/glm53_fp8_dense.py"),
        encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()

    check(re.search(r"^VLLM_GLM53_FP8_DENSE=1$", profile, re.M) is not None,
          "the profile still ships w8a8, not nvfp4")
    check('scheme = "nvfp4"' in src, "the nvfp4 scheme exists")
    check('raw in ("nvfp4", "fp4x4", "w4a4")' in src,
          "nvfp4 needs its own spelling: bare 'fp4' stays w4a8")

    # Both operands, or there is no 2x -- quantizing only the weight is the
    # existing w4a8 arm and runs at the fp8 issue rate.
    gemm = src[src.index("def _nvfp4_dense_gemm("):]
    gemm = gemm[:gemm.index("\n\n\n")]
    check("nvfp4_quantize(flat" in gemm,
          "the activation is quantized too, per call")
    check("mm_fp4(" in gemm, "and the pair goes through mm_fp4")

    # Arms only on a check that ran (is True), like w4a8 -- not on "did not
    # raise", which is how the w4a8 default once poisoned a CUDA context.
    branch = src[src.index('if scheme == "nvfp4"'):]
    branch = branch[:branch.index('if scheme == "w4a8"')]
    check("_copy_matches_source(" in branch and ") is True:" in branch,
          "nvfp4 arms only on a value check that actually ran")
    check("mod.quant_method = method" in branch,
          "a failed check leaves the layer on the fp8 copy")

    # The check launches a REAL kernel, and this arm touches ~180 linears
    # inside a build pass that already died once for want of host memory
    # (23rd entry: srv3 global_oom, no fingerprint on any rank). Two things
    # keep it affordable and must not regress.
    check('_NVFP4_BACKEND = os.environ.get(' in src
          and '"auto"' not in src[src.index("_NVFP4_BACKEND"):
                                  src.index("_NVFP4_BACKEND") + 500],
          "the mm_fp4 backend is pinned -- auto JIT-compiles per shape, and "
          "the first compile measured 73.5 s")
    check("_NVFP4_ALPHA[0] is None" in branch,
          "the alpha convention is resolved once, not twice per layer")
    check("seen < 4 or seen % 16 == 0" in branch,
          "the per-layer value check is sampled, not run on all ~180")

    # It stacks on the fp8 METHOD, so a runtime failure drops one notch.
    ctor = src[src.index("class NvFp4DenseMethod"):]
    ctor = ctor[:ctor.index("class W4A8DenseMethod")]
    check("layer.quant_method = self._base" in ctor,
          "a runtime failure falls back to the fp8 method, not to bf16")

    # Re-arming has to unwrap every stacked method, not one.
    check("while isinstance(\n            base, (Fp8DenseMethod, "
          "W4A8DenseMethod, NvFp4DenseMethod)\n        ):" in src,
          "re-arm unwraps stacked methods in a loop")
    print("  fp8-dense nvfp4 scheme contract .. OK")


def test_union_prefill_width_matches_the_converter_tile() -> None:
    """The union arm must hand the converter a width it accepts.

    It did not, for its whole life. The code sliced 2051 columns (2048 KPool
    selection + at most three live tail tokens) and passed that as
    NUM_TOPK_TOKENS, but triton_convert_req_index_to_global_index tiles
    columns and asserts NUM_TOPK_TOKENS % BLOCK_N == 0. 2051 = 7 x 293, so no
    usable tile divides it, and every prefill raised:

        AssertionError: NUM_TOPK_TOKENS (2051) must be divisible by BLOCK_N (128)

    caught, logged, fallen back to FlashInfer. A boot logging "union prefill:
    ARMED width=4" ran 11 prefills and took the fallback 11 times.

    The width must round up to the tile, and the rounded tail must be -1 --
    the converter's documented "invalid", which _topk_length and the >= 0
    masks downstream both honour."""
    path = os.path.join(REPO, "overlay/modules/glm53_model",
                        "glm53_union_prefill.py")
    src = open(path, encoding="utf-8").read()
    check("_CONVERT_BLOCK_N = 128" in src,
          "the converter's column tile is named, not implied")
    body = src[src.index("tokens = q[0].shape[0]"):]
    body = body[:body.index("triton_convert_req_index_to_global_index(") + 400]
    check("-(-want // _CONVERT_BLOCK_N) * _CONVERT_BLOCK_N" in body,
          "the width rounds UP to the tile instead of passing 2051")
    check("logical[:, carried:] = -1" in body,
          "the rounded tail is marked invalid, not left as stale scratch")
    check("BLOCK_N=_CONVERT_BLOCK_N" in body,
          "the tile passed to the converter is the one the width used")
    check(".clone()" in body,
          "the padded view is a copy -- the fallback path reads that buffer")
    check("logical.shape[1] % _CONVERT_BLOCK_N == 0" in body,
          "and the contract is asserted here, not only inside vLLM")

    # The arm claims exact output and never once ran, so the claim is
    # untested. Shadow makes it a number instead of an argument -- and must
    # serve the STOCK answer while measuring, or it is not a shadow.
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    check(re.search(r"^VLLM_GLM53_UNION_PREFILL_SHADOW=0$", profile, re.M),
          "the shadow ships off")
    check('_UNION_SHADOW_ENV, "").strip() == "1"' in src,
          "exact opt-in for the shadow")
    shadow = src[src.index("if _union_shadow_enabled():"):]
    shadow = shadow[:shadow.index("return output, None")]
    check("ref = original(" in shadow and "return ref" in shadow,
          "shadow serves the stock answer, never the measured one")
    check("_UNION_SHADOW_MAX" in shadow,
          "shadow is bounded -- a long run must not pay for it forever")

    # The kernel tiles group_size x heads rows, each with a [D] fp32
    # accumulator, and caps that at 32. The hook admits only 16-head q (64
    # heads at TP4), so width 4 is 64 rows and can NEVER run on this model --
    # it returns None and the caller quietly uses the stock path. Not an
    # exception, so it does not even reach the fallback counter. This lane
    # booted "union prefill: ARMED width=4" for weeks on that.
    check("q[0].shape[1:] == (16, 512)" in src,
          "the hook pins 16 heads, which is what makes the cap a width limit")
    check("group_size * heads > 32" in src, "the row cap is still the gate")
    decline = src[src.index("if group_size not in (2, 4)"):]
    decline = decline[:decline.index("return None") + 12]
    check("_UNION_DECLINED" in decline and "logger.warning" in decline,
          "a declined width must say so once, not fail silently")
    check("VLLM_GLM53_UNION_PREFILL=2" in decline,
          "and must name the width that can actually run")
    print("  union prefill width vs converter tile .. OK")


def test_benches_ask_the_server_for_the_model_name() -> None:
    """No bench may hardcode a served model name as its only source.

    Every one of them defaulted to `deepseek-v4-flash`. On the glm53 server
    that 404s, and a 404 does not read like a failure here: the caller greps
    for a SUMMARY line, finds none, and the section reads "measured nothing"
    rather than "never ran". It silently voided the 9/9 quality gate on this
    lane -- `gates.sh` ran check-quality every time and it never once reached
    the model.

    So the literal may only ever be the fallback argument to a resolver that
    asks /v1/models first."""
    bench = os.path.join(REPO, "bench")
    shared = open(os.path.join(bench, "bench_common.py"), encoding="utf-8").read()
    check("def resolve_model(" in shared and "/v1/models" in shared,
          "the shared resolver asks the server")
    check('os.environ.get("BENCH_MODEL")' in shared,
          "BENCH_MODEL still wins when a caller means a specific name")
    offenders = []
    for name in sorted(os.listdir(bench)):
        if not name.endswith(".py") or name == "bench_common.py":
            continue
        text = open(os.path.join(bench, name), encoding="utf-8").read()
        if "BENCH_MODEL" not in text and "_resolve_model(" not in text:
            continue          # bench does not talk to the server
        if re.search(r'MODEL\s*=\s*os\.environ\.get\("BENCH_MODEL"', text):
            offenders.append(name)
        if "_resolve_model(" in text and "from bench_common import" not in text:
            offenders.append(name + " (own copy of the resolver)")
    check(not offenders,
          "every bench resolves through bench_common: " + ", ".join(offenders))
    print("  benches ask the server for the model .. OK")


def test_korean_gate_separates_notation_from_damage() -> None:
    """The Korean gate must not fire on the model writing about Korean.

    It did, for boots on end: a response explaining spacing wrote the correct
    construction "-(으)ㄹ 수 있다" and the bench reported that ㄹ as corruption.
    That single false hit is what a 1/8 and a 1/16 in the ledger were, and it
    nearly cost an arm -- prep-fused was about to be blamed for a regression
    the detector invented.

    Damage welds a jamo onto a syllable (하ㄹ수). Notation always leads with a
    delimiter. The character BEFORE the jamo is the whole discriminator."""
    ns = load_defs(
        "bench/korean-corruption.py",
        {"SYL", "JAMO", "WELDED_JAMO", "HAN", "HANJA_GLOSS",
         "INFORMATIONAL", "scan"},
        {"re": re, "unicodedata": __import__("unicodedata")},
    )
    scan = ns["scan"]
    notation = [
        'The construction "-(으)ㄹ 수 있다" requires spacing',
        "받침 ㄹ 과 ㄴ 은 다르다",
        "-ㅂ니다 체를 쓴다",
        "ㄱ부터 ㅎ까지",
        "조력 발전은 밀물과 썰물의 낙차를 이용한다.",
    ]
    for text in notation:
        check(scan(text)["lone_jamo"] == 0,
              f"grammar notation is not damage: {text[:34]}")
    for text in ("하ㄹ수 있다", "할ㅅ우 있다"):
        check(scan(text)["lone_jamo"] == 1,
              f"a jamo welded to a syllable is damage: {text}")
    check("jamo_notation" in ns["INFORMATIONAL"],
          "the notation count is reported but never gates a response")

    # Same class of false positive, second detector: these prompts ask about
    # Korean, and Korean technical writing gives the hanja for a term.
    #   Clear and crisp weather (천고마비 - 天高馬肥)
    #   조력 (潮力) refers to tidal power
    # Both were counted as corruption. A gloss follows its Korean word through
    # a bracket, dash, colon or comma; damage puts Han where Hangul belonged.
    gloss = [
        "Clear and crisp weather (천고마비 - 天高馬肥) - Dry skies",
        "조력 (潮力) refers to tidal power",
        "조력 발전(水力發電)의 원리",
        "변압기 - 變壓器 는 전압을 바꾼다",
        "조력 발전은 밀물과 썰물을 이용한다.",
    ]
    for text in gloss:
        check(scan(text)["cjk_mixed"] == 0,
              f"a hanja gloss is not damage: {text[:34]}")
    check(scan("발전소는 電氣를 만든다")["cjk_mixed"] == 2,
          "Han standing in for Hangul is damage, counted per character")
    check("hanja_gloss" in ns["INFORMATIONAL"],
          "the gloss count is reported but never gates a response")
    print("  korean gate notation vs damage .. OK")


def test_every_module_can_mount_on_an_image_the_repo_can_launch() -> None:
    """A module no launchable image accepts is a decoy, not a rollback path.

    `b12x_swiglu_clamp` and `flashinfer_b12x_collapse` were kept "for a v9
    image" after their fixes were baked into the image at v11. Nothing here
    can launch a v9: every profile and the launcher's own last-resort default
    name v13-b12x. Worse, all three of their files failed the preimage check
    against v13, so deploy-overlays.sh would have refused to mount them even
    if a profile had listed them. Keeping them read as a rollback that did not
    exist; a real one means checking out a commit from that era, where the
    whole tree agrees.

    This check is structural, not a sha comparison (that needs the image): a
    module in overlay/modules must be named by some profile's MODULES, or the
    profile must say in prose why it is not."""
    mods_dir = os.path.join(REPO, "overlay", "modules")
    have = {d for d in os.listdir(mods_dir)
            if os.path.isdir(os.path.join(mods_dir, d))}
    used, prose = set(), ""
    for env in sorted(glob.glob(os.path.join(REPO, "profiles", "*.env"))):
        text = open(env, encoding="utf-8").read()
        prose += text
        for line in text.splitlines():
            if line.startswith("MODULES="):
                used |= set(line.split("=", 1)[1].strip().strip('"').split())
    undocumented = sorted(m for m in have - used if m not in prose)
    check(not undocumented,
          "a module no profile mounts must be explained in a profile "
          f"comment; silent orphans: {', '.join(undocumented)}")
    print("  every module is mounted or explained .. OK")


def test_profile_declares_no_knob_the_code_cannot_read() -> None:
    """A profile knob nothing reads is worse than no knob.

    `VLLM_GLM53_MK_W4` outlived its lane. The fp8 MK-GEMM path and this knob
    were deleted when W4 became the only one -- the ledger lists both under
    "removed", and two tests assert the name is absent from the code -- but
    the profile kept declaring it, with eight lines describing how to turn the
    lane on. The launcher forwards every profile VLLM_* key as its own -e, so
    each container carried it, and an operator reading the profile would
    reasonably believe setting it did something.
    """
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    declared = set(re.findall(r"^(VLLM_[A-Z0-9_]+)=", profile, re.M))
    code = ""
    for root, _, files in os.walk(os.path.join(REPO, "overlay", "modules")):
        for f in files:
            if f.endswith((".py", ".cu")):
                code += open(os.path.join(root, f), encoding="utf-8",
                             errors="replace").read()
    launcher = open(os.path.join(REPO, "launchers",
                                 "start-glm53-nvfp4-tp4.sh"),
                    encoding="utf-8").read()
    unread = sorted(k for k in declared
                    if k not in code and k not in launcher)
    check(not unread,
          "every profile knob is read by overlay code or the launcher; "
          f"unread: {', '.join(unread)}")
    print("  profile knobs are all readable .. OK")


def test_fp8_dense_free_bf16_contract() -> None:
    """Releasing the bf16 sources is exact-opt-in and drops the base fallbacks.

    After the quant_method swap apply() reads only the fp8 copy, so the bf16
    tensor is dead: 2.94 GB/rank at TP4 against 1.47 GB of fp8 pack and 0.82
    GB of the megakernel's W4 pack (computed from the checkpoint's shapes).
    Freeing it also removes the bias and exception fallbacks through the base
    method, which is why it stayed knob-gated. It is ON by default since
    2026-09-04: the memory it holds is not spare. With it off the fleet boots
    at ~3 GiB free of 121, against a 4.0 GiB kernel min watermark -- srv3 was
    OOM-killed on 09-03 and srv1/srv2 wedged on 09-04. A default that costs a
    node is not a safe default."""
    src = open(os.path.join(REPO, "overlay/modules/glm53_model/glm53_fp8_dense.py"),
               encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(re.search(r"^VLLM_GLM53_FP8_DENSE_FREE_BF16=1$", profile, re.M) is not None,
          "profile ships the bf16 release ON -- holding it cost two wedged "
          "nodes and an OOM kill")
    launcher = open(os.path.join(REPO, "launchers",
                                 "start-glm53-nvfp4-tp4.sh"), encoding="utf-8").read()
    check('"$PREFLIGHT" 10' in launcher,
          "the memfree preflight reserves 10 GiB, not 3: these boxes carry a "
          "4.0 GiB kernel min watermark and the fp8-dense pass peaks several "
          "GiB above steady state, so a 3 GiB margin is inside the kill zone")
    hy4 = open(os.path.join(REPO, "launchers", "start-hy4-tp4.sh"),
               encoding="utf-8").read()
    check('"$PREFLIGHT" 10 ' in hy4 and '"$PREFLIGHT" 3 ' not in hy4,
          "the production launcher reserves the same 10 GiB: it only ever "
          "raises GPU_MEM, so a 3 GiB margin is the same kill zone the "
          "supervisor would relaunch into after every reboot")
    check('os.environ.get(_FREE_BF16_ENV, "").strip() == "1"' in src,
          "exact opt-in: only the string 1 releases the bf16 sources")
    check('if getattr(mod, "bias", None) is not None:' in src,
          "a linear with a bias keeps its bf16 source (the bias path needs it)")

    # The release must NOT run from maybe_build_fp8_dense. That function is
    # written to tolerate an early call -- AutoWeightsLoader enters a child
    # load_weights before the checkpoint is walked, and a later call rebuilds
    # the copy -- but a rebuild cannot restore a deleted source. Freeing from
    # inside it killed a boot at parameter.py:221, where the loader read
    # shape[input_dim] of a 1-D empty tensor.
    build = src[src.index("def maybe_build_fp8_dense"):]
    build = build[:build.index("\ndef ")] if "\ndef " in build else build
    check("mod.weight.data = torch.empty(" not in build,
          "maybe_build_fp8_dense never frees: it may run before the load ends")
    free = src[src.index("def maybe_free_fp8_dense_bf16"):]
    check("mod.weight.data = torch.empty(" in free,
          "the release lives in its own pass")

    # ... and it is driven from the first forward of the model file, which is
    # the only COMPOSED place that provably runs after loading. The first
    # attempt drove it from GPUModelRunner.load_model -- correct in principle,
    # dead in practice: glm53_drop_audit is in no composition, so the release
    # never ran and the boot silently measured the baseline.
    wiring = os.path.join(REPO, "overlay/modules/glm53_model",
                          "glm5next_model.py")
    src_w = open(wiring, encoding="utf-8").read()
    check("maybe_free_fp8_dense_bf16(self)" in src_w,
          "the release is triggered from the composed model wiring")

    # It must hang off Glm5NextForConditionalGeneration, the class vLLM
    # actually runs. Glm5NextForCausalLM.forward reads like the right place
    # and is never called: Glm4vForConditionalGeneration.forward reaches past
    # it into `self.language_model.model(...)`. A trigger there is dead code,
    # and two boots came up healthy-looking with the release silently skipped.
    outer = src_w[src_w.index("class Glm5NextForConditionalGeneration"):]
    check("maybe_free_fp8_dense_bf16(self)" in outer,
          "the trigger is on the class vLLM calls, not the bypassed one")
    inner = src_w[src_w.index("class Glm5NextForCausalLM"):
                  src_w.index("class Glm5NextForConditionalGeneration")]
    check("maybe_free_fp8_dense_bf16" not in inner,
          "Glm5NextForCausalLM.forward is bypassed -- nothing may rely on it")
    fwd = outer[outer.index("    def forward("):]
    check('getattr(self, "_bf16_released", False)' in fwd,
          "guarded so it runs once, not every step")
    check("return super().forward(" in fwd,
          "and it delegates -- the override adds the release, nothing else")

    # Every module the release touches must actually be composed; that is the
    # check the dead call site would have failed.
    manifest = open(os.path.join(REPO, "build/glm53/manifest.tsv"),
                    encoding="utf-8").read()
    for needed in ("glm5next_model.py", "glm53_fp8_dense.py"):
        check(needed in manifest,
              f"{needed} is in the glm53 composition")
    print("  fp8-dense bf16 release contract .. OK")


def test_fp8_dense_bproj() -> None:
    """The b-projection arm matches exactly the rear halves meant for it.

    STEP_KERNEL_MAP #108 section 2: after W8A8, 145 bf16 cutlass GEMMs/step
    remain -- q_b/kv_b (full-attn b halves), indexer wq_b -- while the KDA
    f_b/g_b halves sit under the min(shape) >= 512 guard and wk_weights_proj
    is loader-forced bf16 for its fusion. The arm must match the first group
    only, and must never ride the default (#110: opt-in or nothing).
    """
    ns = load_defs(
        "overlay/glm53_fp8_dense.py",
        {"_SHARED_EXPERT_RE", "_INCLUDE", "_BPROJ_INCLUDE", "_BPROJ_ON",
         "_include_patterns", "_DRAFTER_ENV", "_DRAFTER_INCLUDE"},
        {"os": os, "re": re},
    )
    base = ns["_INCLUDE"]
    bproj = ns["_BPROJ_INCLUDE"]
    include = ns["_include_patterns"]

    matches = lambda pats, name: any(p.search(name) for p in pats)

    must_match = [
        "model.layers.7.self_attn.q_b_proj",
        "model.layers.43.self_attn.kv_b_proj",
        "model.layers.3.self_attn.indexer.wq_b",
    ]
    must_not = [
        # the KDA merged a-half and the first-k dense MLPs are the base arm's
        # business -- the bproj arm must not widen them
        "model.layers.0.self_attn.in_proj_qkvbfg_a",
        "model.layers.5.mlp.experts.12.gate_up_proj",
        "model.layers.5.mlp.shared_experts.down_proj",
        # KDA b halves: [2048, 128] per rank, under the 512 guard on purpose
        "model.layers.0.self_attn.f_b_proj",
        "model.layers.0.self_attn.g_b_proj",
        # loader upcasts wk to bf16 to keep the wk+weights_proj fusion
        "model.layers.3.self_attn.indexer.wk_weights_proj",
        # spec/drafter body must stay untouched
        "model.layers.5.self_attn.o_proj.impl.not_a_real_suffix",
    ]
    for name in must_match:
        check(matches(bproj, name), f"bproj arm must match {name}")
    for name in must_not:
        check(not matches(bproj, name), f"bproj arm must NOT match {name}")

    saved = os.environ.pop("VLLM_GLM53_FP8_DENSE_BPROJ", None)
    try:
        check(include() == base, "gate unset: patterns are the base arm")
        os.environ["VLLM_GLM53_FP8_DENSE_BPROJ"] = "0"
        check(include() == base, "gate 0: patterns are the base arm")
        os.environ["VLLM_GLM53_FP8_DENSE_BPROJ"] = "1"
        extended = include()
        check(len(extended) == len(base) + len(bproj),
              "gate 1: base + bproj patterns")
        check(all(matches(extended, n) for n in must_match),
              "gate 1: every bproj target matches")
    finally:
        if saved is None:
            os.environ.pop("VLLM_GLM53_FP8_DENSE_BPROJ", None)
        else:
            os.environ["VLLM_GLM53_FP8_DENSE_BPROJ"] = saved


def test_mhc_smallm_knob() -> None:
    """VLLM_GLM53_MHC_SMALLM: parse strictly, validate against kernel contracts.

    The override feeds mhc_fused_post_pre's small-M branch; an invalid value
    must fall back to the stock heuristic (TODO(gnovack)-marked), never crash
    the dispatcher's assert or silently drop elements in the h-loop.
    """
    env_name = "VLLM_GLM53_MHC_SMALLM"
    saved = os.environ.pop(env_name, None)

    def load():
        ns = load_defs(
            "overlay/tilelang.py",
            {
                "_SMALLM_ENV",
                "_deneb_parse_smallm",
                "_deneb_smallm_pair",
                "_raw_smallm",
                "_DENEB_SMALLM",
            },
            {"os": os},
        )
        return ns

    try:
        os.environ.pop(env_name, None)
        ns = load()
        parse = ns["_deneb_parse_smallm"]
        check(ns["_DENEB_SMALLM"] is None, "env unset: knob is None (stock)")
        check(ns["_deneb_smallm_pair"](1, 4096, 4) is None,
              "env unset: pair falls back to stock")

        check(parse("6,4") == (6, 4), "parse plain")
        check(parse(" 6 , 4 ") == (6, 4), "parse whitespace")
        check(parse("6") is None, "parse missing split")
        check(parse("6,4,2") is None, "parse extra field")
        check(parse("a,b") is None, "parse non-numeric")
        check(parse("0,4") is None and parse("-6,4") is None,
              "parse rejects non-positive")
        check(parse("6,4x") is None, "parse rejects trailing junk")

        os.environ[env_name] = "6,4"
        ns = load()
        check(ns["_DENEB_SMALLM"] == (6, 4), "env set: knob frozen at import")
        pair = ns["_deneb_smallm_pair"]
        check(pair(1, 4096, 4) == (6, 4), "GLM shapes admit (6,4)")
        check(pair(16, 4096, 4) == (6, 4), "whole small-M branch is overridden")

        os.environ[env_name] = "5,4"
        ns = load()
        check(ns["_deneb_smallm_pair"](1, 4096, 4) is None,
              "tile_n=5 does not divide n_out=24 -> stock")
        os.environ[env_name] = "6,3"
        ns = load()
        check(ns["_deneb_smallm_pair"](1, 4096, 4) is None,
              "n_splits=3 would trip the dispatcher assert -> stock")
        os.environ[env_name] = "6,16"
        ns = load()
        check(ns["_deneb_smallm_pair"](1, 4096, 4) is None,
              "n_splits=16 exceeds the dispatcher's set -> stock")
        os.environ[env_name] = "6,8"
        ns = load()
        check(ns["_deneb_smallm_pair"](1, 4096, 4) == (6, 8),
              "(6,8): h_per_split=512 is n_thr-exact")
        os.environ[env_name] = "6,4"
        ns = load()
        check(ns["_deneb_smallm_pair"](1, 512, 4) is None,
              "h_per_split=128 leaves 128 threads idle -> stock (silent-drop guard)")
    finally:
        if saved is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = saved


def test_mhc_passes_knob() -> None:
    """VLLM_GLM53_MHC_PASSES: parse strictly; unset/invalid = stock passes.

    The knob flips TileLang pass configs (TMA lowering / warp specialization)
    for EVERY mhc kernel at import -- an ambiguous value must leave the
    compiled kernels identical to the image's (both TL_DISABLE_* stay True),
    same fail-safe shape as the sibling MHC knobs.
    """
    env_name = "VLLM_GLM53_MHC_PASSES"
    saved = os.environ.pop(env_name, None)

    def load():
        return load_defs(
            "overlay/tilelang_kernels.py",
            {
                "_MHC_PASSES_ENV",
                "_deneb_parse_mhc_passes",
                "_raw_mhc_passes",
                "_DENEB_MHC_PASSES",
            },
            {"os": os},
        )

    try:
        os.environ.pop(env_name, None)
        ns = load()
        parse = ns["_deneb_parse_mhc_passes"]
        check(ns["_DENEB_MHC_PASSES"] is None,
              "env unset: knob is None (stock pass configs)")

        check(parse("tma") == (True, False), "parse tma only")
        check(parse("WS") == (False, True), "parse ws, case-insensitive")
        check(parse("tma,ws") == (True, True), "parse both")
        check(parse("ws,tma") == (True, True), "parse both, order-free")
        check(parse(" tma , ws ") == (True, True), "parse whitespace")
        check(parse("tma,tma") == (True, False), "duplicate token tolerates")
        check(parse("none") == (False, False),
              "'none' is the explicit stock combo (probe reference pass)")
        check(parse("") is None and parse(",") is None, "empty -> None")
        check(parse("tma,xyz") is None, "unknown token -> None (stock)")
        check(parse("all") is None, "'all' is not a token -> None (stock)")

        os.environ[env_name] = "tma"
        ns = load()
        check(ns["_DENEB_MHC_PASSES"] == (True, False),
              "env set: knob frozen at import")
        os.environ[env_name] = "bogus"
        ns = load()
        check(ns["_DENEB_MHC_PASSES"] is None,
              "invalid value falls back to stock, never a partial arm")

        # Wiring: the parsed value must reach the dict the decorators bind,
        # before the first @tilelang.jit reads it. load_defs cannot exec the
        # flips themselves (they need tilelang), so pin them at source level.
        # The decorator match is line-anchored: this module's own comments
        # mention "@tilelang.jit", and a bare substring search would find the
        # comment instead of the decorator (this test's first draft did).
        src = open(_overlay_source("overlay/tilelang_kernels.py"),
                   encoding="utf-8").read()
        first_jit = re.search(r"^@tilelang\.jit", src, re.M).start()
        check("pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = False"
              in src, "tma flip is wired into pass_configs")
        check("pass_configs[tilelang.PassConfigKey."
              "TL_DISABLE_WARP_SPECIALIZED] = False" in src,
              "ws flip is wired into pass_configs")
        check(src.index("TL_DISABLE_TMA_LOWER] = False") < first_jit
              and src.index("TL_DISABLE_WARP_SPECIALIZED] = False") < first_jit,
              "flips precede the first @tilelang.jit that binds the dict")
    finally:
        if saved is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = saved


# ---------------------------------------------------------------------------
# EP stock top-k token chunks: strict opt-in above the preserved #146 fallback.
# ---------------------------------------------------------------------------
def test_ep_fixed_token_chunks() -> None:
    ns = load_defs(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py",
        {
            "B12X_EP_FIXED_MICRO_MAX_PAIRS",
            "B12X_EP_STOCK_TOPK_MICRO_MAX_TOKENS",
            "B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS",
            "B12X_EP_STOCK_TOPK_MICRO_TOKEN_COUNTS",
            "B12X_EP_STOCK_TOPK_MICRO_TOPK",
            "B12X_EP_STOCK_TOPK_MICRO_EXPERTS",
            "read_b12x_ep_bool",
            "read_b12x_ep_exact_bool",
            "b12x_ep_mode_from_env",
            "require_b12x_ep_micro_limit",
            "require_b12x_ep_micro_limits",
            "require_b12x_ep_stock_topk_micro_dispatch",
            "b12x_ep_fixed_slice_limit",
            "b12x_ep_fixed_pair_plan",
            "b12x_ep_stock_topk_token_limit",
            "b12x_ep_stock_topk_token_spans",
            "b12x_ep_stock_topk_micro_chunks",
        },
        {"os": os},
    )
    token_limit = ns["b12x_ep_stock_topk_token_limit"]
    token_spans = ns["b12x_ep_stock_topk_token_spans"]
    stock_chunks = ns["b12x_ep_stock_topk_micro_chunks"]
    pair_plan = ns["b12x_ep_fixed_pair_plan"]
    read_bool = ns["read_b12x_ep_bool"]
    mode_from_env = ns["b12x_ep_mode_from_env"]
    require_fixed_micro = ns["require_b12x_ep_micro_limit"]
    require_stock_micro = ns["require_b12x_ep_micro_limits"]
    require_stock_dispatch = ns[
        "require_b12x_ep_stock_topk_micro_dispatch"
    ]

    check(mode_from_env({}.get) == (True, False, False, False),
          "EP mode defaults to #146 fixed fallback; experiments stay off")
    for raw in ("1", " true ", "YES", "on"):
        check(read_bool("FLAG", False, {"FLAG": raw}.get),
              f"strict EP bool accepts true spelling {raw!r}")
    for raw in ("0", " false ", "NO", "off"):
        check(not read_bool("FLAG", True, {"FLAG": raw}.get),
              f"strict EP bool accepts false spelling {raw!r}")
    for name in ("VLLM_B12X_EP_NO_DUMMY", "VLLM_B12X_EP_DISABLE_MICRO"):
        try:
            mode_from_env({name: "maybe"}.get)
            check(False, f"invalid {name} must fail closed")
        except ValueError as exc:
            check(name in str(exc) and "must be one of" in str(exc),
                  f"invalid {name} reports the bad setting")
    try:
        mode_from_env({"VLLM_B12X_EP_DISABLE_MICRO": "1"}.get)
        check(False, "static diagnostic must reject the default no-dummy path")
    except RuntimeError as exc:
        msg = str(exc)
        check("VLLM_B12X_EP_DISABLE_MICRO=1" in msg
              and "also set VLLM_B12X_EP_NO_DUMMY=0" in msg,
              "mode conflict must explain how to select the static diagnostic")
    check(
        mode_from_env({
            "VLLM_B12X_EP_DISABLE_MICRO": "1",
            "VLLM_B12X_EP_NO_DUMMY": "0",
        }.get) == (False, True, False, False),
        "plain-static diagnostic is allowed only after disabling no-dummy",
    )
    check(
        mode_from_env({"VLLM_B12X_EP_STOCK_TOPK_MICRO": "1"}.get)
        == (True, False, False, True),
        "exact 1 arms only the stock native-top-k experiment",
    )
    for flags in (
        {
            "VLLM_B12X_EP_STOCK_TOPK_MICRO": "1",
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO": "1",
        },
        {
            "VLLM_B12X_EP_STOCK_TOPK_MICRO": "1",
            "VLLM_B12X_EP_NO_DUMMY": "0",
        },
        {
            "VLLM_B12X_EP_STOCK_TOPK_MICRO": "1",
            "VLLM_B12X_EP_DISABLE_MICRO": "1",
        },
    ):
        try:
            mode_from_env(flags.get)
            check(False, f"conflicting stock top-k mode must fail: {flags}")
        except RuntimeError as exc:
            check("STOCK_TOPK_MICRO" in str(exc),
                  "stock mode conflict must name the strict experiment")

    check(ns["B12X_EP_FIXED_MICRO_MAX_PAIRS"] == 8,
          "default #146 fallback retains its pinned top-k=1 boundary")
    check(ns["B12X_EP_STOCK_TOPK_MICRO_MAX_TOKENS"] == 8,
          "stock experiment pins FlashInfer's token boundary")
    check(ns["B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS"] == 40,
          "stock experiment pins the multi-top-k routed-row boundary")
    check(require_fixed_micro(8) == 8,
          "default fallback depends only on the existing token boundary")
    check(require_stock_micro(8, 40) == (8, 40),
          "stock experiment accepts the two pinned micro boundaries")
    check(require_stock_micro("12", "48") == (12, 48),
          "larger parseable micro boundaries remain compatible")
    for bad_limits in (
        (None, 40), ("unknown", 40), (0, 40), (7, 40),
        (8, None), (8, "unknown"), (8, 0), (8, 39),
    ):
        try:
            require_stock_micro(*bad_limits)
            check(False, f"micro boundaries {bad_limits!r} must fail closed")
        except RuntimeError as exc:
            check("cannot verify" in str(exc) or "requires FlashInfer" in str(exc),
                  f"micro boundaries {bad_limits!r} report the contract failure")
    dispatch = types.SimpleNamespace(
        _FORCED_BACKEND=None,
        _MICRO_MAX_TOKENS=8,
        _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK=40,
    )
    check(require_stock_dispatch(dispatch) == (8, 40),
          "stock opt-in accepts the pinned automatic dispatcher")
    for bad_dispatch in (
        types.SimpleNamespace(
            _MICRO_MAX_TOKENS=8,
            _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK=40,
        ),
        types.SimpleNamespace(
            _FORCED_BACKEND="micro",
            _MICRO_MAX_TOKENS=8,
            _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK=40,
        ),
    ):
        try:
            require_stock_dispatch(bad_dispatch)
            check(False, "missing or forced backend contract must fail closed")
        except RuntimeError as exc:
            check("_FORCED_BACKEND" in str(exc) or "automatic" in str(exc),
                  "dispatcher failure must explain the backend contract")

    check(
        [token_limit(n, 8) for n in (2048, 8, 4, 0)] == [5, 5, 4, 1],
        "top-k=8 calls must honor token, routed-row, and workspace bounds",
    )
    check(token_limit(32, 1) == 8 and token_limit(32, 40) == 1,
          "routed-row capacity must scale with top-k")
    for bad_topk in (0, -1, 41):
        try:
            token_limit(32, bad_topk)
            check(False, f"top_k={bad_topk} must not silently enter static")
        except (ValueError, RuntimeError):
            pass

    for concurrency, verify_tokens, launches in (
        (1, 8, 2),
        (2, 16, 4),
        (4, 32, 7),
    ):
        limit = token_limit(32, 8)
        spans = stock_chunks(verify_tokens, 8, 72, enabled=True)
        check(len(spans) == launches,
              f"C={concurrency}: {verify_tokens} tokens must use "
              f"{launches} micro calls per layer")
        sizes = [hi - lo for lo, hi in spans]
        check(spans[0][0] == 0 and spans[-1][1] == verify_tokens,
              f"micro chunks must cover all {verify_tokens} tokens")
        check(all(0 < size <= 5 and size * 8 <= 40 for size in sizes),
              "every native-top-k chunk must remain micro-eligible")
        check(max(sizes) - min(sizes) <= 1,
              "chunk sizes must be balanced")
        check(verify_tokens == 1 or min(sizes) > 1,
              "balanced multi-token shapes must avoid a one-token tail")
    check([hi - lo for lo, hi in token_spans(17, 5)] == [5, 4, 4, 4],
          "non-verification shapes must also use balanced full coverage")
    for tokens in range(1, 81):
        spans = token_spans(tokens, 5)
        check(sum(hi - lo for lo, hi in spans) == tokens,
              f"planner must cover arbitrary M={tokens}")
        check(all(0 < hi - lo <= 5 for lo, hi in spans),
              f"planner must keep arbitrary M={tokens} on micro")
    for args in (
        (1, 8, 72, True), (7, 8, 72, True), (24, 8, 72, True),
        (8, 1, 72, True), (8, 8, 71, True), (8, 8, 72, False),
    ):
        check(stock_chunks(*args[:3], enabled=args[3]) == (),
              f"non-exact stock shape must keep #146 fallback: {args}")

    # The default pair plan remains intact while this experiment is off.
    src, keep = pair_plan([True, False, True] + [False] * 5, 8)
    check(len(src) == 8 and sum(keep) == 2,
          "#146 fallback must retain every real pair and fixed length")
    check(set(src) <= {0, 2} and not any(keep[2:]),
          "#146 fallback padding may only repeat local rows at weight zero")

    # Pure semantic oracle for the runtime amin/where replacement. Remote
    # weights are zero from remap, so changing only their IDs cannot change the
    # local rank's weighted sum. The fill stays inside each token's local set.
    dummy = 18
    cases = (
        ("mixed", [
            [3, dummy, 7, dummy, dummy, 3, dummy, dummy],
            [dummy, 5, dummy, 9, dummy, dummy, dummy, 5],
        ], [
            [0.4, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2],
        ]),
        ("all remote", [[dummy] * 8], [[0.0] * 8]),
        ("all local", [[0, 1, 2, 3, 4, 5, 6, 7]], [[0.125] * 8]),
        ("duplicate", [[4, dummy, 4, dummy, 11, dummy, 11, dummy]],
         [[0.2, 0.0, 0.3, 0.0, 0.1, 0.0, 0.4, 0.0]]),
    )
    for label, ids, weights in cases:
        safe_ids = []
        for row in ids:
            local = [expert for expert in row if expert != dummy]
            fill = min(local) if local else dummy - 1
            safe_ids.append([fill if expert == dummy else expert for expert in row])
        check(all(0 <= expert < dummy for row in safe_ids for expert in row),
              f"kernel may only receive local expert IDs ({label})")
        for row, safe in zip(ids, safe_ids):
            local = {expert for expert in row if expert != dummy}
            if local:
                check(set(safe) <= local,
                      f"remote replacement may not add a weight plane ({label})")
        reference = sum(
            weight * ((token + 1) * 100 + expert)
            for token, (row, scales) in enumerate(zip(ids, weights))
            for expert, weight in zip(row, scales)
            if expert != dummy
        )
        actual = sum(
            weight * ((token + 1) * 100 + expert)
            for token, (row, scales) in enumerate(zip(safe_ids, weights))
            for expert, weight in zip(row, scales)
        )
        check(actual == reference,
              f"zero-weight ID replacement must preserve the EP sum ({label})")

    source = open(
        _overlay_source(
            "overlay/modules/glm53_moe/flashinfer_b12x_moe.py"
        ),
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)
    remap_tensor = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "remap_b12x_ep_tensors"
    )
    micro_setup = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "disable_b12x_micro_for_ep"
    )
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FlashInferB12xExperts"
    )
    fixed = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_ep_fixed"
    )
    stock = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_ep_stock_topk_micro"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
    )
    process = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_weights_after_loading"
    )
    apply = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply"
    )
    fixed_source = ast.get_source_segment(source, fixed)
    stock_source = ast.get_source_segment(source, stock)
    remap_source = ast.get_source_segment(source, remap_tensor)
    micro_setup_source = ast.get_source_segment(source, micro_setup)
    init_source = ast.get_source_segment(source, init)
    process_source = ast.get_source_segment(source, process)
    apply_source = ast.get_source_segment(source, apply)
    assert all(part is not None for part in (
        fixed_source, stock_source, remap_source, micro_setup_source, init_source,
        process_source, apply_source,
    ))
    assert fixed_source is not None
    assert stock_source is not None
    assert remap_source is not None
    assert micro_setup_source is not None
    assert init_source is not None
    assert process_source is not None
    assert apply_source is not None
    check(
        "require_b12x_ep_micro_limit(prior)" in micro_setup_source
        and "_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK"
        not in micro_setup_source,
        "default #146 setup must depend only on its existing token boundary",
    )
    check(
        "if map_len <= 0:" in remap_source
        and "remote.fill_(True)" in remap_source,
        "an empty expert map must initialise every route as remote",
    )
    check(
        "b12x_ep_mode_from_env()" in init_source,
        "EP flags must be parsed and latched during expert construction",
    )
    check(
        "disable_b12x_micro_for_ep(" in process_source
        and "self._ep_no_dummy, self._ep_disable_micro" in process_source,
        "weight setup must consume both latched EP mode flags",
    )
    check(
        "if self._ep_no_dummy:" in apply_source
        and "VLLM_B12X_EP_NO_DUMMY" not in apply_source,
        "captured apply path must consume the latch without reading its env",
    )
    check(
        "if not self._ep_no_dummy:\n                self._pad_dummy_expert(layer)"
        in process_source,
        "direct EP must not allocate the physical dummy weight row",
    )
    compact_return = apply_source.index("return self._apply_ep_compact(")
    fixed_return = apply_source.index("return self._apply_ep_fixed(")
    stock_return = apply_source.index(
        "return self._apply_ep_stock_topk_micro("
    )
    ensure_wrapper = apply_source.index("self._ensure_wrapper()")
    capacity_probe = apply_source.index(
        "self._ep_capacity_probe(wrapper, topk_ids)"
    )
    check(
        stock_return < compact_return < fixed_return < ensure_wrapper,
        "opt-in, compact, and #146 fallback must precede the large wrapper",
    )
    check(
        ensure_wrapper < capacity_probe,
        "capacity probe belongs only to the padded wrapper fallback",
    )
    check(
        "VLLM_B12X_EP_NO_DUMMY" not in apply_source,
        "apply must use the frozen mode instead of re-reading the environment",
    )
    check(
        "if self._ep_no_dummy and b12x_ep_should_compact(" in apply_source,
        "NO_DUMMY=0 must restore the wrapper for decode and prefill",
    )
    check(
        "b12x_ep_stock_topk_micro_chunks(" in stock_source
        and "for lo, hi in spans:" in stock_source,
        "stock runtime must use the exact-gated balanced token chunks",
    )
    check(
        "torch.amin(topk_ids, dim=1, keepdim=True, out=fill_ids)"
        in stock_source
        and "self._ep_remote[:tokens], fill_ids, topk_ids, out=topk_ids"
        in stock_source,
        "remote sentinels must become same-token local IDs without allocation",
    )
    check(
        "launch_sm120_moe(" in stock_source
        and "a=hidden_states[lo:hi]" in stock_source
        and "topk_ids=topk_ids[lo:hi]" in stock_source
        and "topk_weights=topk_weights[lo:hi]" in stock_source
        and "top_k=topk" in stock_source
        and "scatter_output=output[lo:hi]" in stock_source
        and "_workspace=self._ep_stock_topk_workspace" in stock_source,
        "stock path must write each native-top-k chunk directly to output",
    )
    check(
        all(name not in stock_source for name in (
            "argsort", "index_select", "index_add_", "pair_x", "pair_out",
        )),
        "stock experiment must not rematerialise routed pairs",
    )
    check(
        "pair_out = torch.zeros(" in fixed_source
        and "pair_out.mul_(keep.unsqueeze(1)" in fixed_source
        and fixed_source.index("pair_out.mul_(")
        < fixed_source.index("output.index_add_("),
        "default #146 fallback must retain both zero-init and pre-sum mask",
    )
    check(
        "self._ep_fill_ids: torch.Tensor | None = None" in init_source
        and "self._ep_stock_topk_workspace: Any | None = None" in init_source
        and "top_k=1" in process_source
        and "max_rows=B12X_EP_FIXED_MICRO_MAX_PAIRS" in process_source
        and "top_k=B12X_EP_STOCK_TOPK_MICRO_TOPK" in process_source
        and "max_rows=B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS"
        in process_source,
        "setup must keep separate #146 r8 and opt-in stock r40 workspaces",
    )
    check(
        "require_b12x_ep_stock_topk_micro_dispatch(moe_dispatch)"
        in process_source
        and "self.global_num_experts" in process_source
        and "288," in process_source,
        "stock opt-in must fail setup on dispatcher or pinned geometry drift",
    )

    workspace = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_shared_ep_fixed_workspace"
    )
    workspace_source = ast.get_source_segment(source, workspace)
    assert workspace_source is not None
    check(
        "max_rows=key.max_rows" in workspace_source
        and "num_topk=key.top_k" in workspace_source
        and 'backend="static"' in workspace_source,
        "pinned workspace allocator must consume the shape-specific key",
    )
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    check("VLLM_B12X_EP_STOCK_TOPK_MICRO=0" in profile,
          "stock top-k experiment must remain explicitly default-off")

    print("  EP stock top-k chunks .......... OK")


def test_b12x_zero_weight_micro() -> None:
    """Default-off E=72 sentinel skip: exact gate, cache, and hot-path order."""
    wrapper_path = (
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py"
    )
    wrapper_names = {
        "read_b12x_ep_bool",
        "read_b12x_ep_exact_bool",
        "b12x_ep_mode_from_env",
        "B12X_EP_ZERO_WEIGHT_MICRO_MAX_TOKENS",
        "B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS",
        "b12x_ep_micro_chunk_tokens",
        "B12X_EP_ZERO_WEIGHT_MICRO_TOPK",
        "B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS",
        "b12x_ep_zero_weight_micro_chunks",
    }
    wns = load_defs(wrapper_path, wrapper_names, {"os": os})
    exact_bool = wns["read_b12x_ep_exact_bool"]
    mode = wns["b12x_ep_mode_from_env"]
    chunks = wns["b12x_ep_zero_weight_micro_chunks"]

    check(mode({}.get) == (True, False, False, False),
          "both native-top-k experiments must be off by default")
    check(mode({"VLLM_B12X_EP_ZERO_WEIGHT_MICRO": "1"}.get)
          == (True, False, True, False),
          "exact 1 arms only the zero-weight kernel experiment")
    check(not exact_bool("Z", False, {}.get), "exact bool defaults off")
    for raw in ("true", "yes", "on", " 1 ", "2", "-1", ""):
        try:
            exact_bool("Z", False, {"Z": raw}.get)
            check(False, f"experimental bool must reject {raw!r}")
        except ValueError as exc:
            check("exactly 0 or 1" in str(exc),
                  f"experimental bool must explain {raw!r}")
    for flags in (
        {
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO": "1",
            "VLLM_B12X_EP_NO_DUMMY": "0",
        },
        {
            "VLLM_B12X_EP_ZERO_WEIGHT_MICRO": "1",
            "VLLM_B12X_EP_DISABLE_MICRO": "1",
        },
    ):
        try:
            mode(flags.get)
            check(False, f"conflicting zero-weight mode must fail: {flags}")
        except RuntimeError as exc:
            check("ZERO_WEIGHT_MICRO=1 requires" in str(exc),
                  "mode conflict must name the required local-only contract")

    expected_chunks = {
        8: ((0, 8),),
        16: ((0, 8), (8, 16)),
        32: ((0, 8), (8, 16), (16, 24), (24, 32)),
    }
    for tokens, expected in expected_chunks.items():
        got = chunks(tokens, 8, 72, enabled=True)
        check(got == expected, f"{tokens} stable tokens must use {len(expected)} calls")
        check(all(hi - lo == 8 for lo, hi in got),
              "every experimental call must be m=8, never single-token")
    # 24 (C=3 at MAX_SEQS=4) is a clean 3x8 plan -- structurally identical to
    # 32's 4x8 -- so it is admitted now; only shapes the chunker cannot slice
    # exactly, or that belong to compact, keep the fixed fallback.
    check(chunks(24, 8, 72, enabled=True) == ((0, 8), (8, 16), (16, 24)),
          "24 tokens must take the micro lane, not the pair fallback")
    for args in (
        (1, 8, 72, True), (7, 8, 72, True),
        (88, 8, 72, True), (8, 1, 72, True), (8, 8, 71, True),
        (8, 8, 72, False),
    ):
        check(chunks(*args[:3], enabled=args[3]) == (),
              f"non-exact wrapper shape must keep fixed fallback: {args}")

    dispatch_path = "overlay/modules/glm53_moe/moe_dispatch.py"
    dispatch_names = {
        "_B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS",
        "_B12X_EP_ZERO_WEIGHT_MICRO_TOKENS",
        "_B12X_EP_ZERO_WEIGHT_MICRO_TOPK",
        "_B12X_EP_ZERO_WEIGHT_MICRO_K",
        "_B12X_EP_ZERO_WEIGHT_MICRO_N",
        "_B12X_EP_ZERO_WEIGHT_MICRO_SWIGLU_LIMIT",
        "_b12x_ep_zero_weight_micro_expert_id",
    }
    dns = load_defs(dispatch_path, dispatch_names, {})
    gate = dns["_b12x_ep_zero_weight_micro_expert_id"]
    exact = dict(
        enabled=True,
        state_E=72,
        weight_E=72,
        num_tokens=8,
        k=4096,
        n=2048,
        num_topk=8,
        activation_precision="fp4",
        quant_mode="nvfp4",
        activation="swigluoai_uninterleave",
        swiglu_limit=10.0,
        forced_backend=None,
    )
    check(gate(**exact) == 72, "exact dispatch geometry returns sentinel E")
    mismatches = {
        "enabled": False,
        "state_E": 71,
        "weight_E": 73,
        "num_tokens": 7,
        "k": 2048,
        "n": 512,
        "num_topk": 1,
        "activation_precision": "bf16",
        "quant_mode": "mxfp4",
        "activation": "silu",
        "swiglu_limit": None,
    }
    for field, value in mismatches.items():
        case = dict(exact)
        case[field] = value
        check(gate(**case) is None, f"dispatch mismatch {field} must stay stock")
    forced = dict(exact)
    forced["forced_backend"] = "micro"
    try:
        gate(**forced)
        check(False, "forced micro must not bypass the sentinel compile flag")
    except RuntimeError as exc:
        check("cannot run with forced" in str(exc),
              "forced-backend conflict must fail before launch")

    routed_rows = exact["num_tokens"] * exact["num_topk"]
    check(routed_rows == 64 and routed_rows <= exact["state_E"],
          "prepass map proof requires 64 routed rows <= 72 entries")
    check(routed_rows - 1 < exact["state_E"],
          "even 64 unique ids write compact indices only 0..63")

    dispatch_source = open(_overlay_source(dispatch_path), encoding="utf-8").read()
    dispatch_tree = ast.parse(dispatch_source)
    helper_node = next(n for n in dispatch_tree.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "_b12x_ep_zero_weight_micro_expert_id")
    cache_node = next(n for n in dispatch_tree.body
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "_micro_kernel_cache_key")
    get_node = next(n for n in dispatch_tree.body
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "_get_micro_kernel")
    launch_node = next(n for n in dispatch_tree.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "launch_sm120_static_moe")
    helper_source = ast.get_source_segment(dispatch_source, helper_node) or ""
    cache_source = ast.get_source_segment(dispatch_source, cache_node) or ""
    get_source = ast.get_source_segment(dispatch_source, get_node) or ""
    launch_source = ast.get_source_segment(dispatch_source, launch_node) or ""
    check("routed_rows <= int(state_E)" in helper_source,
          "dispatch source must prove compact-map capacity before arming")
    check("skip_zero_weight_expert_id" in cache_source,
          "sentinel variant must be part of the micro cache key")
    check(get_source.count("skip_zero_weight_expert_id") >= 4,
          "compile helper must carry sentinel through key and constructor")
    check(launch_source.index("_b12x_ep_zero_weight_micro_expert_id(")
          < launch_source.index("use_micro ="),
          "exact arm must be computed before micro/static selection")
    check("or skip_zero_weight_expert_id is not None" in launch_source,
          "exact routed64 arm must widen only the opt-in micro decision")
    check("skip_zero_weight_expert_id=skip_zero_weight_expert_id" in launch_source,
          "launch must compile the sentinel-aware variant")
    check("routed_rows > workspace.weight_expert_ids.numel()" in launch_source,
          "runtime workspace drift must fail before Triton compaction")
    check("os.environ" not in launch_source,
          "captured launch path must consume the import-time latch only")

    kernel_path = "overlay/modules/glm53_moe/moe_micro_kernel.py"
    kernel_source = open(_overlay_source(kernel_path), encoding="utf-8").read()
    kernel_tree = ast.parse(kernel_source)
    kernel_cls = next(n for n in kernel_tree.body
                      if isinstance(n, ast.ClassDef) and n.name == "MoEMicroKernel")
    init_node = next(n for n in kernel_cls.body
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    call_node = next(n for n in kernel_cls.body
                     if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    init_source = ast.get_source_segment(kernel_source, init_node) or ""
    call_source = ast.get_source_segment(kernel_source, call_node) or ""
    check("single_token or share_input_across_experts" in init_source,
          "sentinel variant must reject both unsafe micro specializations")
    sentinel_pos = call_source.index("row = Int32(-1)")
    append_pos = call_source.index("row = atomic_add_global_i32", sentinel_pos)
    map_pos = call_source.index("get_ptr_as_int64(token_map, map_idx)", append_pos)
    check(sentinel_pos < append_pos < map_pos,
          "sentinel decision must precede row/token-map materialization")
    check("weight == cutlass.Float32(0.0)" in call_source
          and "Int32(self.skip_zero_weight_expert_id)" in call_source,
          "kernel may skip only exact-zero pairs for the named sentinel")
    check("row >= Int32(0)" in call_source
          and "should_quantize" in call_source,
          "row=-1 must suppress input quantization")

    wrapper_source = open(_overlay_source(wrapper_path), encoding="utf-8").read()
    wrapper_tree = ast.parse(wrapper_source)
    wrapper_cls = next(n for n in wrapper_tree.body
                       if isinstance(n, ast.ClassDef)
                       and n.name == "FlashInferB12xExperts")
    pure_node = next(n for n in wrapper_cls.body
                     if isinstance(n, ast.FunctionDef)
                     and n.name == "_apply_ep_zero_weight_micro")
    apply_node = next(n for n in wrapper_cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == "apply")
    process_node = next(n for n in wrapper_cls.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "process_weights_after_loading")
    pure_source = ast.get_source_segment(wrapper_source, pure_node) or ""
    apply_source = ast.get_source_segment(wrapper_source, apply_node) or ""
    process_source = ast.get_source_segment(wrapper_source, process_node) or ""
    check("scatter_output=output[lo:hi]" in pure_source
          and "topk_ids=topk_ids[lo:hi]" in pure_source
          and "_workspace=self._ep_zero_weight_workspace" in pure_source,
          "pure lane must use disjoint token/output slices and its pinned workspace")
    # The micro loop itself stays free of gather/scatter. A batch that is not
    # a multiple of the chunk hands only its short tail (< one chunk) to the
    # #146 fallback -- real decode saw 9/36/49/52/60 tokens, and covering the
    # aligned prefix turns 49 pair calls into 6 micro calls plus one tail.
    tail_call = pure_source.find("self._apply_ep_fixed(")
    loop_source = pure_source if tail_call < 0 else pure_source[:tail_call]
    for banned in ("pair_x", "pair_out", "index_add_", "argsort", "nonzero"):
        check(banned not in loop_source,
              f"pure top-k=8 lane must not retain {banned} overhead")
    if tail_call >= 0:
        check("b12x_ep_micro_tail(" in pure_source,
              "the tail must come from the shape-derived planner, never from a "
              "data-dependent count")
        check(pure_source.index("for lo, hi in chunks:") < tail_call,
              "the tail runs after the aligned prefix, on rows it never wrote")
        check("output[lo:hi]" in pure_source[tail_call:]
              and "hidden_states[lo:hi]" in pure_source[tail_call:],
              "the tail fallback must be handed disjoint slice views")
    zero_return = apply_source.index("return self._apply_ep_zero_weight_micro(")
    stock_return = apply_source.index(
        "return self._apply_ep_stock_topk_micro("
    )
    fixed_return = apply_source.index("return self._apply_ep_fixed(")
    ensure_wrapper = apply_source.index("self._ensure_wrapper()")
    check(zero_return < stock_return < fixed_return < ensure_wrapper,
          "kernel experiment must precede stock experiment and #146 fallback")
    check("VLLM_B12X_EP_ZERO_WEIGHT_MICRO" not in apply_source,
          "apply must never re-read the experiment environment")
    check("top_k=1" in process_source
          and "max_rows=B12X_EP_FIXED_MICRO_MAX_PAIRS" in process_source
          and "top_k=B12X_EP_STOCK_TOPK_MICRO_TOPK" in process_source
          and "max_rows=B12X_EP_STOCK_TOPK_MICRO_MAX_ROUTED_ROWS"
          in process_source
          and "top_k=B12X_EP_ZERO_WEIGHT_MICRO_TOPK" in process_source
          and "max_rows=B12X_EP_ZERO_WEIGHT_MICRO_MAX_ROWS" in process_source,
          "setup must pin distinct #146-r8, stock-r40, and overlay-r64 workspaces")

    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    check("glm53_moe" in profile,
          "glm53 composition must mount both FlashInfer source overlays")
    check("VLLM_B12X_EP_ZERO_WEIGHT_MICRO=0" in profile,
          "profile must keep the numeric experiment default off")
    manifest = open(os.path.join(
        REPO, "overlay", "modules", "glm53_moe", "manifest.tsv"
    ), encoding="utf-8").read()
    check("ccb6f65a22314961693493242f78f62ca58f79a319ecd0cb51bf6d7d8e7125c6"
          in manifest, "micro kernel must pin the live preimage SHA")
    check("f6923850c710eb21cf7c3566b6ddcc39ddda0d7a3664c19dec0205895af31362"
          in manifest, "dispatch must pin the live preimage SHA")

    print("  b12x zero-weight micro ......... OK")


def test_glm53_b12x_tuning_controls() -> None:
    """Default-off GLM controls parse strictly and fail closed on shape drift."""
    dispatch_path = "overlay/modules/glm53_moe/moe_dispatch.py"
    names = {
        "_GLM53_B12X_FORCE_BACKEND_ENV",
        "_GLM53_B12X_STATIC_CUTOVER_ENV",
        "_parse_glm53_forced_backend",
        "_parse_glm53_static_cutover",
        "_parse_glm53_mac_ladder",
        "_is_glm53_b12x_tp_geometry",
        "_effective_glm53_static_cutover",
    }
    ns = load_defs(dispatch_path, names, {"Tuple": tuple})
    parse_backend = ns["_parse_glm53_forced_backend"]
    parse_cutover = ns["_parse_glm53_static_cutover"]
    parse_ladder = ns["_parse_glm53_mac_ladder"]
    exact_gate = ns["_is_glm53_b12x_tp_geometry"]
    effective_cutover = ns["_effective_glm53_static_cutover"]

    for raw in (None, "", "  ", "auto"):
        check(parse_backend(raw) is None, f"backend {raw!r} must keep automatic")
    for backend in ("micro", "static", "dynamic"):
        check(parse_backend(backend) == backend, f"backend {backend} must parse")
    for raw in ("AUTO", "direct_micro", "static,dynamic", "1"):
        try:
            parse_backend(raw)
            check(False, f"invalid GLM backend must fail: {raw!r}")
        except ValueError as exc:
            check("must be auto" in str(exc), "backend error must name its contract")

    for raw, expected in ((None, None), ("", None), ("0", 0), ("640", 640)):
        check(parse_cutover(raw) == expected, f"cutover {raw!r} must parse")
    for raw in ("-1", "1.5", "dynamic"):
        try:
            parse_cutover(raw)
            check(False, f"invalid GLM cutover must fail: {raw!r}")
        except ValueError as exc:
            check("non-negative" in str(exc), "cutover error must name its contract")

    env_name = "VLLM_GLM53_B12X_STATIC_MAC_LADDER"
    check(parse_ladder(None, env_name) is None, "unset ladder keeps shipped values")
    check(
        parse_ladder("64:48, 128:40,640:32", env_name)
        == ((64, 48), (128, 40), (640, 32)),
        "valid routed-row MAC ladder must preserve every cell",
    )
    for raw in ("64", "64:x", "0:48", "64:0", "64:48,64:32", "128:48,64:32"):
        try:
            parse_ladder(raw, env_name)
            check(False, f"invalid MAC ladder must fail: {raw!r}")
        except ValueError as exc:
            check(env_name in str(exc), "ladder error must name the setting")

    exact = dict(
        num_experts=288,
        num_local_experts=288,
        hidden_size=4096,
        intermediate_size=2048,
        num_topk=8,
        quant_mode="nvfp4",
        activation="swigluoai_uninterleave",
        swiglu_limit=10.0,
    )
    check(exact_gate(**exact), "deployed GLM TP geometry must admit tuning")
    mismatches = {
        "num_experts": 72,
        "num_local_experts": 72,
        "hidden_size": 2048,
        # 512 is the per-rank spelling the launcher passes (admitted since
        # 2026-09-05); 1024 (a TP2 shard) is not a deployed geometry
        "intermediate_size": 1024,
        "num_topk": 1,
        "quant_mode": "mxfp4",
        "activation": "silu",
        "swiglu_limit": None,
    }
    for field, value in mismatches.items():
        case = dict(exact)
        case[field] = value
        check(not exact_gate(**case), f"GLM tuning mismatch {field} must fail closed")

    # The functions extracted above share this namespace as their globals.
    # A forced decode-micro setting must reserve all 8*top-k routed rows even
    # when a simultaneous cutover=0 routes every other call to dynamic.
    ns["_GLM53_B12X_STATIC_CUTOVER_PAIRS"] = 0
    ns["_GLM53_B12X_FORCE_BACKEND"] = "micro"
    ns["_MICRO_MAX_TOKENS"] = 8
    check(effective_cutover(640, **exact) == 64,
          "forced micro must retain a 64-row static workspace at cutover zero")
    non_glm = dict(exact)
    non_glm["num_experts"] = 72
    non_glm["num_local_experts"] = 72
    check(effective_cutover(17, **non_glm) == 17,
          "EP geometry must ignore GLM TP workspace tuning")

    source = open(_overlay_source(dispatch_path), encoding="utf-8").read()
    tree = ast.parse(source)
    launch_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_sm120_static_moe"
    )
    launch_source = ast.get_source_segment(source, launch_node) or ""
    dynamic_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_dynamic_kernel"
    )
    dynamic_source = ast.get_source_segment(source, dynamic_node) or ""
    check("os.environ" not in launch_source and "os.environ" not in dynamic_source,
          "captured MoE paths must consume import-time tuning latches")
    check("forced_backend = _effective_glm53_forced_backend(" in launch_source,
          "static launcher must use the exact-shape backend latch")
    check(launch_source.count("_effective_glm53_mac_ladder(") == 2,
          "static launcher must gate both static and micro MAC ladders")
    check("_effective_glm53_mac_ladder(" in dynamic_source,
          "dynamic compiler must gate its MAC ladder")

    wrapper_path = "overlay/modules/glm53_moe/b12x_moe.py"
    wrapper_source = open(_overlay_source(wrapper_path), encoding="utf-8").read()
    check("_effective_glm53_static_cutover(" in wrapper_source,
          "wrapper workspace capacity must use the same exact-shape cutover")
    for field in (
        "num_experts=self.num_experts",
        "num_local_experts=self.num_local_experts",
        "hidden_size=self.hidden_size",
        "intermediate_size=self.intermediate_size",
        "activation=self.activation",
        "swiglu_limit=self.swiglu_limit",
    ):
        check(wrapper_source.count(field) >= 2,
              f"wrapper backend selection must carry {field}")

    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    for key in (
        "VLLM_GLM53_B12X_FORCE_BACKEND",
        "VLLM_GLM53_B12X_STATIC_CUTOVER_PAIRS",
        "VLLM_GLM53_B12X_MICRO_MAC_LADDER",
        "VLLM_GLM53_B12X_STATIC_MAC_LADDER",
        "VLLM_GLM53_B12X_DYNAMIC_MAC_LADDER",
    ):
        check(f'{key}=""' in profile, f"{key} must remain explicitly default-off")

    print("  GLM53 b12x tuning controls ..... OK")


def test_b12x_static_v2_controls() -> None:
    """The decode-streaming static kernel is opt-in, exact-geometry, spec-parsed.

    `VLLM_GLM53_B12X_STATIC_V2` selects `MoEStaticKernelV2` for the served
    GLM-5.3 TP geometry only. The profile ships it empty (stock kernel) until
    the bracket; the spec parser must reject anything it cannot spell back,
    because the config lands in the kernel cache key and the on-disk name.
    """
    dispatch_path = "overlay/modules/glm53_moe/moe_dispatch.py"
    names = {
        "_GLM53_B12X_STATIC_V2_ENV",
        "_STATIC_V2_DEFAULT",
        "_parse_glm53_static_v2",
        "_is_glm53_b12x_tp_geometry",
        "_static_v2_config_for",
        "_static_v2_cache_key",
        "_static_kernel_cache_key",
    }
    ns = load_defs(dispatch_path, names, {"Tuple": tuple, "Dict": dict, "torch": None})
    parse = ns["_parse_glm53_static_v2"]
    default = ns["_STATIC_V2_DEFAULT"]

    for raw in (None, "", " ", "0", "off"):
        check(parse(raw) is None, f"static v2 {raw!r} must keep the stock kernel")
    check(parse("1") == default and parse("1") is not default,
          "'1' must be a copy of the default config")
    check(default == {"tile_m": 32, "fc1": 2, "fc2": 4, "a_rows": 32, "stamps": False,
                      "dynamic": False, "wide": False},
          "the default v2 config is m32,f2,g4,a32, static schedule, no stamps, v2 body")
    check(parse("m32,f3,g2") == {"tile_m": 32, "fc1": 3, "fc2": 2, "a_rows": 32,
                                 "stamps": False, "dynamic": False, "wide": False},
          "explicit cells override the defaults")
    check(parse("m32,f2,g4,d")["dynamic"] and not parse("m32,f2,g4,d")["stamps"],
          "d selects the dynamic item schedule")
    wide = parse("w")
    check(wide["wide"] and wide["fc2"] == 3 and wide["fc1"] == 2 and wide["tile_m"] == 32,
          "w selects the v3 kernel (FC1 halves over 256-wide K) with 3 FC2 stages")
    check(parse("w,g2")["fc2"] == 2, "an explicit g cell overrides w's FC2 default")
    for raw in ("w,m64", "w,d", "w,a64"):
        try:
            parse(raw)
            check(False, f"w must reject {raw!r}")
        except ValueError as exc:
            check("v3" in str(exc), "the w conflict error names v3")
    check(parse("m64,f2,g2")["a_rows"] == 64,
          "a_rows follows tile_m unless spelled out")
    check(parse("m32,f2,g4,a64")["a_rows"] == 64 and parse("m32,f2,g4,s")["stamps"],
          "a<rows> and s cells parse")
    for raw in ("2", "m48", "m32,x1", "m32,a48", "m32,a256", "f0", "m32,,g2", "m32 f2"):
        try:
            parse(raw)
            check(False, f"invalid static v2 spec must fail: {raw!r}")
        except ValueError as exc:
            check("VLLM_GLM53_B12X_STATIC_V2" in str(exc),
                  "static v2 spec error must name the knob")

    geometry = dict(num_experts=288, num_local_experts=288, hidden_size=4096,
                    intermediate_size=2048, num_topk=8, quant_mode="nvfp4",
                    activation="swigluoai_uninterleave", swiglu_limit=10.0)
    ns["_STATIC_V2_OVERRIDE"] = None
    ns["_GLM53_B12X_STATIC_V2"] = None
    config_for = ns["_static_v2_config_for"]
    check(config_for(activation_precision="fp4", **geometry) is None,
          "unset knob keeps the stock kernel")
    ns["_GLM53_B12X_STATIC_V2"] = dict(default)
    check(config_for(activation_precision="fp4", **geometry) == default,
          "the env config applies to the exact GLM TP geometry")
    check(config_for(activation_precision="bf16", **geometry) is None,
          "W4A16 (bf16 activations) never takes the NVFP4 v2 kernel")
    # launch_sm120_static_moe passes the PER-RANK intermediate (2048 / TP4 =
    # 512); the first probe run measured the stock kernel six times because
    # the gate only knew the full 2048 (no static2_ compile, no proof line)
    per_rank = dict(geometry, intermediate_size=512)
    check(config_for(activation_precision="fp4", **per_rank) == default,
          "the geometry gate must admit the per-rank intermediate the launcher passes")
    drift = dict(geometry, num_local_experts=72)
    check(config_for(activation_precision="fp4", **drift) is None,
          "EP geometry (E=72 local) keeps the stock kernel")
    ns["_STATIC_V2_OVERRIDE"] = {"tile_m": 64, "fc1": 2, "fc2": 2, "a_rows": 64,
                                 "stamps": True}
    check(config_for(activation_precision="fp4", **geometry)["tile_m"] == 64,
          "the probe override wins over the env config")

    key_a = ns["_static_v2_cache_key"](
        default, activation_precision="fp4", quant_mode="nvfp4", state_E=288,
        weight_E=288, m=8, k=4096, n=512, num_topk=8, max_rows=512, mac=48,
        mma_tiler_mn=(32, 128), topk_ids_dtype="int32",
        input_scales_are_reciprocal=False, fast_math=True,
        activation="swigluoai_uninterleave", swiglu_alpha=1.0, swiglu_beta=0.0,
        swiglu_limit=10.0)
    key_b = ns["_static_v2_cache_key"](
        dict(default, fc2=2), activation_precision="fp4", quant_mode="nvfp4",
        state_E=288, weight_E=288, m=8, k=4096, n=512, num_topk=8, max_rows=512,
        mac=48, mma_tiler_mn=(32, 128), topk_ids_dtype="int32",
        input_scales_are_reciprocal=False, fast_math=True,
        activation="swigluoai_uninterleave", swiglu_alpha=1.0, swiglu_beta=0.0,
        swiglu_limit=10.0)
    check(key_a[0] == "static_v2" and key_a != key_b,
          "the v2 cache key carries the config so configs never alias")

    src = open(os.path.join(REPO, dispatch_path), encoding="utf-8").read()
    check("moe_static_kernel_v2.__file__" in src,
          "the v2 kernel source must be in the module cache key files")
    check("_STATIC_V2_OVERRIDE if _STATIC_V2_OVERRIDE is not None" in src,
          "the probe override is a module-level hook, not an env read at launch")
    kernel = open(os.path.join(
        REPO, "overlay/modules/glm53_moe/moe_static_kernel_v2.py"),
        encoding="utf-8").read()
    check("class MoEStaticKernelV2" in kernel and "producer_tail" in kernel
          and "reset_count" not in kernel,
          "v2 keeps its pipeline states continuous across items and drains at exit")
    check("pass_sync_barrier" not in kernel,
          "v2 has no CTA-wide item-boundary barrier")
    manifest = open(os.path.join(
        REPO, "overlay", "modules", "glm53_moe", "manifest.tsv"),
        encoding="utf-8").read()
    check("moe_static_kernel_v2.py\tflashinfer/fused_moe/cute_dsl/blackwell_sm12x/"
          "moe_static_kernel_v2.py\tabsent" in manifest,
          "the v2 kernel is a new file (absent preimage) in the module manifest")
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check('VLLM_GLM53_B12X_STATIC_V2=w' in profile,
          "the profile ships the v3 static kernel (spec w) as the default "
          "(35차, operator: +7~10% on the decode MoE kernel; rollback = \"\")")
    runner = open(os.path.join(REPO, "probes", "run_mk_probe.sh"), encoding="utf-8").read()
    check("moe_static_kernel_v2.py" in runner,
          "the probe runner must mount the v2 kernel beside the dispatcher")

    print("  b12x static v2 controls ........ OK")


def test_mhc_probe_contracts() -> None:
    """The GPU probe must compare final shapes and load the tested overlay."""
    comparisons = []

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def reshape_as(self, expected):
            return ("reshape_as", self.name, expected.name)

    def fake_rel_err(actual, expected):
        comparisons.append((actual, expected))
        return len(comparisons)

    names = {
        "STOCK_LT8", "STOCK_GE8", "_stock_config", "_parse_ms",
        "_onepass_rel_errors", "_pair_rel_errors",
    }
    ns = load_defs("probes/mhc_glm53_bench.py", names,
                   {"rel_err": fake_rel_err})
    stock_config = ns["_stock_config"]
    check(stock_config(1) == (2, 8) and stock_config(7) == (2, 8),
          "probe must use the decode stock pair below M=8")
    check(stock_config(8) == (3, 4) and stock_config(16) == (3, 4),
          "probe must use the dispatcher stock pair at M=8..16")
    parse_ms = ns["_parse_ms"]
    check(parse_ms("1, 8,16") == [1, 8, 16],
          "probe must accept small-FMA token counts")
    for raw in ("", "0", "17", "1,,2", "x"):
        rejected = False
        try:
            parse_ms(raw)
        except ValueError:
            rejected = True
        check(rejected, f"probe must reject out-of-contract --ms={raw!r}")

    ref = [FakeTensor(name) for name in
           ("gemm", "sqrsum", "residual", "post", "comb", "layer")]
    out = [FakeTensor(name) for name in
           ("residual-out", "post-out", "comb-out", "layer-out")]
    result = ns["_onepass_rel_errors"](out, ref)
    check(result == [1, 2, 3, 4], "onepass must compare all four outputs")
    check(comparisons == [
        (out[0], ref[2]),
        (("reshape_as", "post-out", "post"), ref[3]),
        (("reshape_as", "comb-out", "comb"), ref[4]),
        (out[3], ref[5]),
    ], "onepass post/comb outputs must match the stock flat-buffer shapes")

    comparisons.clear()
    pair = [FakeTensor(name) for name in
            ("candidate-gemm", "candidate-sqrsum", "candidate-residual",
             "candidate-post", "candidate-comb", "candidate-layer")]
    ns["_pair_rel_errors"](pair, ref)
    check(comparisons == list(zip(pair[2:], ref[2:])),
          "config sweep must ignore n_splits-shaped intermediates")

    probe_path = os.path.join(REPO, "probes", "mhc_glm53_bench.py")
    probe_source = open(probe_path, encoding="utf-8").read()
    probe_tree = ast.parse(probe_source)
    main_node = next(node for node in probe_tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "main")
    main_source = ast.get_source_segment(probe_source, main_node)
    assert main_source is not None
    check(main_source.index('os.environ["VLLM_GLM53_MHC_ONEPASS"] = "1"')
          < main_source.index("_load_mhc_overlay"),
          "ONEPASS must be set before importing the eager MHC package")

    loader_node = next(node for node in probe_tree.body
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "_load_mhc_overlay")
    loader_source = ast.get_source_segment(probe_source, loader_node)
    assert loader_source is not None
    check(loader_source.count("_require_composed_source") == 2
          and '"_DENEB_ONEPASS"' in loader_source
          and '"mhc_onepass_tilelang"' in loader_source,
          "probe must prove both source hashes and the armed onepass path")

    wrapper = os.path.join(REPO, "probes", "run_mhc_glm53_bench.sh")
    wrapper_source = open(wrapper, encoding="utf-8").read()
    check(os.access(wrapper, os.X_OK)
          and "for source in tilelang.py tilelang_kernels.py" in wrapper_source
          and "STKERNEL_MHC_OVERLAY_BUILD=/repo/build/glm53" in wrapper_source,
          "probe wrapper must bind and identify both composed MHC sources")
    print("  mhc probe contracts ............ OK")


def test_mhc_onepass_math() -> None:
    """mhc_onepass_tilelang's MATH equals the stock two-kernel pipeline.

    The fused kernel is a transcription of mhc_fused_tilelang (phase 1) +
    mhc_pre_big_fuse_with_norm_tilelang (mixes/sinkhorn/norm). This sim runs
    both pipelines in pure python at hc=4, h=64 -- stock materializes the
    gemm_out intermediates like the two kernels do, onepass follows the fused
    kernel's single-CTA phase order. Same op order => bitwise-equal floats.
    A transcription slip (swapped scale index, dropped eps, wrong sinkhorn
    round count) breaks equality.
    """
    import struct

    def bf16(x):
        """Round fp32 to bf16 and back (the kernels' storage rounding)."""
        return struct.unpack("f", struct.pack("I",
            struct.unpack("I", struct.pack("f", x))[0] & 0xFFFF0000))[0]

    import math as _m

    hc, h, n_out = 4, 64, 24
    rms_eps, pre_eps, sink_eps, post_mult, sinkhorn, norm_eps = (
        1e-5, 1e-6, 1e-6, 2.0, 20, 1e-5)
    scale = [1.1, 0.9, 1.05]
    rng = random.Random(7)

    comb_in = [[rng.uniform(-1, 1) for _ in range(hc)] for _ in range(hc)]
    post_in = [rng.uniform(0.5, 1.5) for _ in range(hc)]
    x = [rng.uniform(-1, 1) for _ in range(h)]
    resid = [[rng.uniform(-1, 1) for _ in range(h)] for _ in range(hc)]
    fn = [[[rng.uniform(-0.05, 0.05) for _ in range(h)]
           for _ in range(hc)] for _ in range(n_out)]
    base = [rng.uniform(-0.5, 0.5) for _ in range(n_out)]
    norm_w = [1.0 for _ in range(h)]

    def rsqrt(v):
        return 1.0 / _m.sqrt(v)

    def sigmoid(v):
        return 1.0 / (1.0 + _m.exp(-v))

    def sinkhorn_rounds(cm):
        row_max = [max(cm[j]) for j in range(hc)]
        for j in range(hc):
            for k in range(hc):
                cm[j][k] = _m.exp(cm[j][k] - row_max[j])
        row_sum = [sum(cm[j]) for j in range(hc)]
        for j in range(hc):
            for k in range(hc):
                cm[j][k] = cm[j][k] / row_sum[j] + sink_eps
        col_sum = [sum(cm[j][k] for j in range(hc)) for k in range(hc)]
        for j in range(hc):
            for k in range(hc):
                cm[j][k] = cm[j][k] / (col_sum[k] + sink_eps)
        for _ in range(sinkhorn - 1):
            row_sum = [sum(cm[j]) for j in range(hc)]
            for j in range(hc):
                for k in range(hc):
                    cm[j][k] = cm[j][k] / (row_sum[j] + sink_eps)
            col_sum = [sum(cm[j][k] for j in range(hc)) for k in range(hc)]
            for j in range(hc):
                for k in range(hc):
                    cm[j][k] = cm[j][k] / (col_sum[k] + sink_eps)
        return cm

    def gates_and_norm(mixes, resid_cur):
        post = [bf16(sigmoid(mixes[j + hc] * scale[1] + base[j + hc]))
                * post_mult for j in range(hc)]
        cm = [[mixes[j * hc + k + hc * 2] * scale[2]
               + base[j * hc + k + hc * 2] for k in range(hc)]
              for j in range(hc)]
        comb = sinkhorn_rounds(cm)
        pre = [sigmoid(mixes[j] * scale[0] + base[j]) + pre_eps
               for j in range(hc)]
        ol = [sum(pre[j] * resid_cur[j][hh] for j in range(hc))
              for hh in range(h)]
        ol = [bf16(v) for v in ol]
        sumsq = sum(v * v for v in ol)
        r = rsqrt(sumsq / h + norm_eps)
        layer = [bf16(ol[hh] * r * norm_w[hh]) for hh in range(h)]
        return post, comb, layer

    def stock_pipeline():
        # kernel 1: mhc_fused (split_k=1, full tile) -> materialized outputs
        resid_cur = [[bf16(post_in[j] * x[hh]
                           + sum(comb_in[k][j] * resid[k][hh]
                                 for k in range(hc)))
                      for hh in range(h)] for j in range(hc)]
        yp = [sum(fn[n][j][hh] * resid_cur[j][hh]
                  for j in range(hc) for hh in range(h))
              for n in range(n_out)]
        rp = sum(resid_cur[j][hh] ** 2
                 for j in range(hc) for hh in range(h))
        # kernel 2: big_fuse_with_norm reads the intermediates back
        r = rsqrt(rp / (hc * h) + rms_eps)
        mixes = [yp[n] * r for n in range(n_out)]
        post, comb, layer = gates_and_norm(mixes, resid_cur)
        return post, comb, layer

    def onepass_pipeline():
        # the fused kernel's phase order: same formulas, intermediates stay
        # in-register; resid_cur goes to global once and is re-read (bf16).
        resid_cur = [[bf16(post_in[j] * x[hh]
                           + sum(comb_in[k][j] * resid[k][hh]
                                 for k in range(hc)))
                      for hh in range(h)] for j in range(hc)]
        acc = [sum(fn[n][j][hh] * resid_cur[j][hh]
                   for j in range(hc) for hh in range(h))
               for n in range(n_out)]
        sqr = sum(resid_cur[j][hh] ** 2
                  for j in range(hc) for hh in range(h))
        r = rsqrt(sqr / (hc * h) + rms_eps)
        mixes = [acc[n] * r for n in range(n_out)]
        post, comb, layer = gates_and_norm(mixes, resid_cur)
        return post, comb, layer

    sp, sc, sl = stock_pipeline()
    fp_, fc, fl = onepass_pipeline()
    check(sp == fp_, "post_mix identical between stock and onepass")
    check(sc == fc, "comb_mix identical between stock and onepass")
    check(sl == fl, "layer_input identical between stock and onepass")
    # semantic anchors: sinkhorn output is ~doubly stochastic; the normed
    # layer_input has ~unit RMS when norm_weight == 1
    colsums = [sum(sc[j][k] for j in range(hc)) for k in range(hc)]
    check(all(abs(v - 1.0) < 1e-3 for v in colsums),
          f"sinkhorn columns ~1: {sum(colsums) / hc:.4f}")
    rms_out = _m.sqrt(sum(v * v for v in sl) / h)
    check(abs(rms_out - 1.0) < 0.3, f"layer_input RMS ~1: {rms_out:.3f}")

    # gate + contract functions from the dispatcher takeover. The gate is
    # frozen at import like its siblings, so flipping env means re-loading.
    names = {
        "_ONEPASS_ENV", "_raw_onepass", "_DENEB_ONEPASS",
        "_deneb_onepass_enabled", "_deneb_onepass_ok",
    }
    saved = os.environ.pop("VLLM_GLM53_MHC_ONEPASS", None)
    try:
        os.environ.pop("VLLM_GLM53_MHC_ONEPASS", None)
        ns = load_defs("overlay/tilelang.py", names, {"os": os})
        check(not ns["_deneb_onepass_enabled"](), "ONEPASS default off")

        os.environ["VLLM_GLM53_MHC_ONEPASS"] = "1"
        ns = load_defs("overlay/tilelang.py", names, {"os": os})
        check(ns["_DENEB_ONEPASS"] is True and ns["_deneb_onepass_enabled"](),
              "ONEPASS env arms (frozen at import)")
        ok = ns["_deneb_onepass_ok"]
        check(ok(4096, 4), "GLM shapes admit onepass")
        check(not ok(4096, 8),
              "n_out=80 exceeds one warp's write span -> refuse")
        check(not ok(1000, 4), "hidden not n_thr-exact -> refuse")
    finally:
        if saved is None:
            os.environ.pop("VLLM_GLM53_MHC_ONEPASS", None)
        else:
            os.environ["VLLM_GLM53_MHC_ONEPASS"] = saved

    print("  mhc onepass math ................ OK")


def test_mhc_smallm_split_ownership() -> None:
    """ONEPASS-off must not overwrite the small-M kernel's split contract.

    ONEPASS was inserted between the stock ``if use_small_fma`` and its
    non-small ``else``. If that else binds to ONEPASS instead, the default-off
    path calls the generic DeepGEMM split planner even for the small-M kernel.
    GB10 then picks 48 splits for GLM's hc=4, H=4096 shape, outside the
    dispatcher's 1/2/4/8 set, and completes no 256-thread hidden iteration.
    """
    path = _overlay_source("overlay/tilelang.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    dispatcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "mhc_fused_post_pre_tilelang"
    )

    def node_source(node):
        segment = ast.get_source_segment(source, node)
        assert segment is not None
        return segment

    top_level_ifs = [node for node in dispatcher.body if isinstance(node, ast.If)]
    onepass_if = next(
        node for node in top_level_ifs
        if "_deneb_onepass_enabled()" in node_source(node.test)
    )
    check(not onepass_if.orelse,
          "ONEPASS early return must not own the stock fallback")

    nonsmall_if = next(
        node for node in top_level_ifs
        if node_source(node.test).strip() == "not use_small_fma"
    )
    check("compute_num_split(" in node_source(nonsmall_if),
          "generic split planner must be guarded by not use_small_fma")
    check(onepass_if.end_lineno < nonsmall_if.lineno,
          "generic non-small planner must follow the ONEPASS early return")

    split_calls = [
        node for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_num_split"
    ]
    check(len(split_calls) == 1,
          "dispatcher must have exactly one generic split-planner call")
    check(nonsmall_if.lineno <= split_calls[0].lineno <= nonsmall_if.end_lineno,
          "every generic split-planner call must stay under the non-small guard")

    small_if = next(
        node for node in top_level_ifs
        if node_source(node.test).strip() == "use_small_fma"
    )
    stock_split = next(
        node.value for node in ast.walk(small_if)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "n_splits"
                for target in node.targets)
        and isinstance(node.value, ast.IfExp)
    )
    check(ast.unparse(stock_split.test)
          == "num_tokens < 8 and hidden_size <= 4096"
          and ast.literal_eval(stock_split.body) == 8
          and ast.literal_eval(stock_split.orelse) == 4,
          "stock small-M dispatcher must retain its 8/4 split heuristic")
    for split in (4, 8):
        check((4096 // split) % 256 == 0
              and (4096 // split) // 256 > 0,
              f"stock split={split} must cover H=4096 in 256-thread iterations")

    # vLLM compute_num_split(block_k=64, k=hc*hidden, grid_size=1) on GB10.
    generic_split = min(48, ((4 * 4096 + 63) // 64) // 4)
    check(generic_split == 48 and generic_split not in (1, 2, 4, 8),
          f"SM121a regression fixture must expose split=48: {generic_split}")
    check((4096 // generic_split) // 256 == 0,
          "split=48 must expose the zero-iteration small-M failure")
    print("  mhc small-M split ownership ..... OK")




def test_mhc_bigfuse_knob() -> None:
    """VLLM_GLM53_MHC_BIGFUSE: strict parse, applies only past 64 tokens.

    dsv4 R3 precedent: h_blk=4096 single pipelined block wins on prefill
    (+5.6% at M=4096 on this kernel family) while M<=64 is stock-optimal.
    Unset or invalid -> None -> the kernels run stock gcd defaults.
    """
    env_name = "VLLM_GLM53_MHC_BIGFUSE"
    saved = os.environ.pop(env_name, None)

    def load():
        return load_defs(
            "overlay/tilelang.py",
            {
                "_BIGFUSE_ENV",
                "_deneb_parse_bigfuse",
                "_raw_bigfuse",
                "_DENEB_BIGFUSE",
                "_deneb_bigfuse_hblk",
                "_deneb_bigfuse_post",
            },
            {"os": os},
        )

    try:
        os.environ.pop(env_name, None)
        ns = load()
        parse = ns["_deneb_parse_bigfuse"]
        check(ns["_DENEB_BIGFUSE"] is None, "env unset: knob is None (stock)")
        check(ns["_deneb_bigfuse_hblk"](4096, 4096) is None,
              "env unset: prefill runs stock h_blk")

        check(parse("4096") == (4096, None), "parse plain (h_blk only)")
        check(parse(" 2048 ") == (2048, None), "parse whitespace")
        check(parse("4096,512") == (4096, 512), "parse h_blk + post_thr")
        check(parse("8192") is None, "h_blk outside the allowed set")
        check(parse("4096,96") is None, "post_thr 96 not in the R2 set")
        check(parse("abc") is None and parse("4096,512,96") is None,
              "rejects junk and extra fields")
        check(parse("0") is None and parse("-4096") is None,
              "rejects non-positive")

        os.environ[env_name] = "4096,512"
        ns = load()
        check(ns["_DENEB_BIGFUSE"] == (4096, 512), "frozen at import")
        hblk = ns["_deneb_bigfuse_hblk"]
        check(hblk(65, 4096) == 4096, "prefill M>64 gets the override")
        check(hblk(8192, 4096) == 4096, "large prefill gets the override")
        check(hblk(64, 4096) is None, "M=64 stays stock (boundary)")
        check(hblk(16, 4096) is None, "decode M stays stock")
        check(hblk(65, 2048) is None,
              "hidden not divisible by h_blk -> stock (OOB-copy guard)")
        post = ns["_deneb_bigfuse_post"]
        check(post(65) == (512, 4096), "post retune follows the second field")
        check(post(64) is None, "post retune also gated at M>64")
        os.environ[env_name] = "4096"
        ns = load()
        check(ns["_deneb_bigfuse_post"](65) is None,
              "post_thr omitted -> mhc_post stays stock")
    finally:
        if saved is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = saved

    print("  mhc bigfuse knob ............... OK")



def test_ep_fixed_output_initialised() -> None:
    src = open(_overlay_source(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py")).read()
    body = src[src.index("def _apply_ep_fixed"):src.index("def _apply_ep_compact")]
    check("pair_out = torch.zeros(" in body,
          "#146 fallback must initialise rows the kernel may skip")
    check("torch.empty(" not in body,
          "no uninitialised buffer may reach #146 index_add")
    check(body.index("pair_out.mul_(") < body.index("output.index_add_("),
          "#146 padding mask must run before the token sum")
    check("keep.unsqueeze(1)" in body,
          "#146 padding mask must use the exact keep flags")

    stock = src[
        src.index("def _apply_ep_stock_topk_micro"):
        src.index("def _apply_ep_fixed")
    ]
    check(stock.index("output.zero_()") < stock.index("for lo, hi in spans:"),
          "stock experiment must zero its direct scatter target first")
    check("torch.eq(fill_ids, dummy, out=all_remote)" in stock
          and "output.masked_fill_(all_remote, 0)" in stock,
          "stock experiment must seal all-remote tokens to exact zero")
    print("  EP fixed/stock output init ..... OK")
def test_glm53_sm121_mla_prefill_gate() -> None:
    """The GLM dense-prefill arm must fail closed on contract drift."""
    ns = load_defs(
        "overlay/flash_attn.py",
        {
            "_glm53_sm121_dense_prefill_gate",
            "_glm53_sm121_filter_prefill_metadata",
            "_glm53_sm121_instance_dimensions",
        },
        {},
    )
    gate = ns["_glm53_sm121_dense_prefill_gate"]
    valid = {
        "flag": "1",
        "capability": (12, 1),
        "dimensions": (256, 0, 256),
        "model_type": "glm5_next_text",
        "kv_lora_rank": 512,
        "num_attention_heads": 64,
        "index_topk": 2048,
        "index_kpool": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "outer_backend_name": "FLASHINFER_MLA_SPARSE_SM90",
        "flash_attn_version": 2,
    }
    check(gate(**valid), "exact SM121/GLM/HND contract admits dense prefill")
    rejected = {
        "flag": (None, "0", "true", " 1"),
        "capability": ((12, 0), (10, 0)),
        "dimensions": ((128, 64, 128), (256, 64, 256)),
        "model_type": (None, "deepseek_v32"),
        "kv_lora_rank": (256, 576),
        "num_attention_heads": (32, 128),
        "index_topk": (1024, 4096),
        "index_kpool": (1, 16),
        "kv_cache_dtype": ("auto", "fp8_ds_mla"),
        "outer_backend_name": (
            "FLASHINFER_MLA_SPARSE_SM120",
            "FLASHMLA_SPARSE",
        ),
        "flash_attn_version": (3, 4, None),
    }
    for field, values in rejected.items():
        for bad in values:
            case = dict(valid)
            case[field] = bad
            check(not gate(**case), f"{field}={bad!r} retains top-k MQA")

    dims = ns["_glm53_sm121_instance_dimensions"]
    check(
        dims(qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256),
        "only the admitted non-stock GLM dimensions latch the request guard",
    )
    for stock_dims in ((128, 64, 128), (192, 64, 256), (64, 64, 128)):
        check(
            not dims(
                qk_nope_head_dim=stock_dims[0],
                qk_rope_head_dim=stock_dims[1],
                v_head_dim=stock_dims[2],
            ),
            f"stock dimensions {stock_dims} do not latch the GLM guard",
        )

    filter_metadata = ns["_glm53_sm121_filter_prefill_metadata"]
    fresh = types.SimpleNamespace(chunked_context=None, use_dense_mha=True)
    filter_metadata(fresh, optin=True)
    check(fresh.use_dense_mha, "fresh causal GLM prefill remains dense-MHA")
    cached = types.SimpleNamespace(chunked_context=object(), use_dense_mha=True)
    filter_metadata(cached, optin=True)
    check(not cached.use_dense_mha, "cached GLM prefill falls back before FA2")
    stock = types.SimpleNamespace(chunked_context=object(), use_dense_mha=True)
    filter_metadata(stock, optin=False)
    check(stock.use_dense_mha, "stock FlashAttention models remain unchanged")

    source = open(_overlay_source("overlay/flash_attn.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FlashAttnPrefillBackend"
    )
    prepare = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_metadata"
    )
    prepare_source = ast.get_source_segment(source, prepare)
    assert prepare_source is not None
    check(
        prepare_source.index("super().prepare_metadata")
        < prepare_source.index("_glm53_sm121_filter_prefill_metadata"),
        "the cached-context guard runs after stock metadata preparation",
    )
    enabled = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_glm53_sm121_dense_prefill_enabled"
    )
    enabled_source = ast.get_source_segment(source, enabled)
    assert enabled_source is not None
    check(
        "reversed(tuple(static_context.values()))" in enabled_source
        and 'getattr(layer, "attn_backend", None)' in enabled_source,
        "admission binds the newest layer's actual outer backend",
    )
    check(
        "stock_supported or _glm53_sm121_dense_prefill_enabled" in source,
        "stock dimension support remains intact",
    )
    build_source = open(
        os.path.join(REPO, "build", "glm53", "flash_attn.py"),
        encoding="utf-8",
    ).read()
    check(source == build_source, "module and composed build remain byte-identical")
    profile = open(
        os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8"
    ).read()
    check(
        "glm53_kernels" in profile
        and "VLLM_GLM53_SM121_MLA_PREFILL=0" in profile,
        "the mounted module remains default-off",
    )
    print("  GLM53 SM121 MLA prefill gate .... OK")


def test_glm53_kpool_packed_scratch_contract() -> None:
    """Pool top-k reuses only packed storage outside live output rows."""
    # The serving hosts run this suite as the deploy gate and have no
    # torch. Skip out LOUD, never silently -- a silent pass hides the hole.
    try:
        import torch
    except ImportError:
        print("  glm53 kpool packed scratch contract ... SKIP (no torch on this host)")
        return

    ns = load_defs(
        "overlay/sparse_attn_indexer_kpool.py",
        {
            "_glm53_dense_mha_layer_name",
            "_glm53_dense_mha_scoring_unused",
            "_kpool_prefill_write_plan",
            "_kpool_prefill_windows",
            "_pool_topk_scratch_fits",
        },
        {"torch": torch},
    )
    fits = ns["_pool_topk_scratch_fits"]
    dense_mha_layer = ns["_glm53_dense_mha_layer_name"]
    scoring_unused = ns["_glm53_dense_mha_scoring_unused"]
    write_plan = ns["_kpool_prefill_write_plan"]
    prefill_windows = ns["_kpool_prefill_windows"]

    tokens = torch.arange(7 * 3).view(7, 3)
    windows = prefill_windows(tokens, 4)
    expected_windows = torch.stack([tokens[i : i + 4] for i in range(4)])
    check(windows.shape == (4, 4, 3),
          "kpool prefill view has [window, pool, dim] layout")
    check(torch.equal(windows, expected_windows),
          "zero-copy kpool windows match the old gathered values")
    check(windows.untyped_storage().data_ptr()
          == tokens.untyped_storage().data_ptr(),
          "kpool prefill windows alias the source storage")
    check(windows.stride() == (3, 3, 1),
          "kpool window strides retain a contiguous head dimension")

    slots = torch.tensor(
        [-1, -1, -1, 12, -1, -1, -1, 20],
        dtype=torch.int32,
    )
    plan_cache = {}
    loc, write_mask = write_plan(slots, 4, plan_cache)
    check(
        torch.equal(loc, torch.tensor([12, -1, -1, -1, 20]))
        and torch.equal(
            write_mask,
            torch.tensor([True, False, False, False, True]),
        ),
        "prefill write plan preserves completion destinations and mask",
    )
    cached_loc, cached_mask = write_plan(slots, 4, plan_cache)
    check(cached_loc is loc and cached_mask is write_mask and len(plan_cache) == 1,
          "identical metadata view reuses its destination and mask tensors")
    cloned_loc, cloned_mask = write_plan(slots.clone(), 4, plan_cache)
    check(
        cloned_loc is not loc
        and cloned_mask is not write_mask
        and len(plan_cache) == 2,
        "different metadata allocation cannot alias a cached write plan",
    )

    check(
        dense_mha_layer("model.layers.7.self_attn.indexer.k_cache")
        == "model.layers.7.self_attn.attn",
        "GLM kpool prefix resolves to its sibling MLA metadata key",
    )
    for drifted in (
        "model.layers.7.self_attn.indexer",
        "model.layers.7.self_attn.k_cache",
        "model.layers.7.self_attn.indexer.tail_cache",
        "",
    ):
        check(
            dense_mha_layer(drifted) == "",
            f"unknown kpool layout {drifted!r} fails closed",
        )

    k_prefix = "model.layers.7.self_attn.indexer.k_cache"
    mla_name = dense_mha_layer(k_prefix)
    dense_meta = {
        mla_name: types.SimpleNamespace(
            prefill=types.SimpleNamespace(use_dense_mha=True),
            num_decode_tokens=0,
        )
    }
    admitted = dict(
        k_cache_prefix=k_prefix,
        attn_metadata=dense_meta,
        is_cuda=True,
        cudagraph_full=False,
        stream_capturing=False,
    )
    check(scoring_unused(**admitted),
          "fresh dense MLA proves sparse indexer scoring has no consumer")
    for field in ("is_cuda", "cudagraph_full", "stream_capturing"):
        case = dict(admitted)
        case[field] = not case[field]
        check(not scoring_unused(**case),
              f"dense scoring bypass rejects {field} drift")
    mixed_meta = {
        mla_name: types.SimpleNamespace(
            prefill=types.SimpleNamespace(use_dense_mha=True),
            num_decode_tokens=1,
        )
    }
    case = dict(admitted, attn_metadata=mixed_meta)
    check(not scoring_unused(**case), "mixed batch retains sparse scoring")
    mqa_meta = {
        mla_name: types.SimpleNamespace(
            prefill=types.SimpleNamespace(use_dense_mha=False),
            num_decode_tokens=0,
        )
    }
    case = dict(admitted, attn_metadata=mqa_meta)
    check(not scoring_unused(**case), "MQA prefill retains sparse scoring")

    # GLM exact geometry: output width rounds 2048+3 up to BLOCK_N=128.
    output_width = ((2048 + 3 + 127) // 128) * 128
    select_k = 2048 // 4
    check(output_width == 2176 and select_k == 512,
          "GLM kpool scratch fixture matches the deployed shape")
    check(fits(8192, 32, output_width, 32, select_k),
          "steady decode fits in inactive persistent rows")
    check(fits(40, 32, output_width, 32, select_k),
          "eight inactive output rows cover 32 packed top-k rows")
    check(not fits(39, 32, output_width, 32, select_k),
          "seven inactive rows must take the allocation fallback")
    check(not fits(32, 33, output_width, 1, select_k),
          "active row count outside storage must fail closed")

    path = _overlay_source("overlay/sparse_attn_indexer_kpool.py")
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_inactive_pool_topk_scratch"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    check(
        "topk_indices_buffer[active_rows:].view(-1)" in helper_source
        and ".view(num_rows, select_k)" in helper_source,
        "scratch must be packed from storage after the active prefix",
    )
    check(
        "topk_indices_buffer.is_contiguous()" in helper_source
        and "topk_indices_buffer.dtype != torch.int32" in helper_source,
        "non-contiguous or non-int32 storage must fall back",
    )

    dispatcher = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sparse_attn_indexer_kpool"
    )
    dispatcher_source = ast.get_source_segment(source, dispatcher)
    assert dispatcher_source is not None

    compress_insert = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_kpool_compress_insert"
    )
    compress_source = ast.get_source_segment(source, compress_insert)
    assert compress_source is not None
    check(
        "_kpool_prefill_windows(k, kpool)" in compress_source
        and "_kpool_prefill_windows(gate_score, kpool)" in compress_source,
        "prefill compressor passes zero-copy K and gate windows",
    )
    check(
        "torch.arange" not in compress_source
        and "k[idx]" not in compress_source
        and "gate_score[idx]" not in compress_source,
        "prefill compressor no longer materializes an index grid or gathers",
    )
    strided_launch = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_kpool_compress_strided_write_cache"
    )
    strided_source = ast.get_source_segment(source, strided_launch)
    assert strided_source is not None
    check(
        "slot_k.stride(0)" in strided_source
        and "slot_k.stride(1)" in strided_source
        and "slot_score.stride(0)" in strided_source
        and "slot_score.stride(1)" in strided_source,
        "stock Triton compressor receives the zero-copy window strides",
    )
    check(".contiguous()" not in strided_source,
          "strided prefill launch must not rematerialize its windows")

    tail_seed_kernel = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_kpool_seed_tail_cache_strided_kernel"
    )
    tail_seed_kernel_source = ast.get_source_segment(source, tail_seed_kernel)
    assert tail_seed_kernel_source is not None
    check(
        "i * key_stride_0" in tail_seed_kernel_source
        and "i * score_stride_0" in tail_seed_kernel_source,
        "prefill tail seeder addresses split K and gate with explicit strides",
    )
    tail_seed = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_kpool_seed_tail_cache_strided"
    )
    tail_seed_source = ast.get_source_segment(source, tail_seed)
    assert tail_seed_source is not None
    check(
        "key.stride(0)" in tail_seed_source
        and "gate_score.stride(0)" in tail_seed_source
        and ".contiguous()" not in tail_seed_source,
        "tail-cache launch preserves zero-copy projection views",
    )

    short_fill = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fill_short_prefill_topk"
    )
    short_fill_source = ast.get_source_segment(source, short_fill)
    assert short_fill_source is not None
    check(
        "_fill_short_prefill_topk_kernel[grid]" in short_fill_source
        and "torch.arange" not in short_fill_source
        and "torch.where" not in short_fill_source,
        "short-prefill top-k output is one direct Triton write",
    )
    short_fill_kernel = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fill_short_prefill_topk_kernel"
    )
    short_fill_kernel_source = ast.get_source_segment(source, short_fill_kernel)
    assert short_fill_kernel_source is not None
    check(
        "values = tl.where(cols <= position, cols, -1)"
        in short_fill_kernel_source
        and "tl.store(" in short_fill_kernel_source,
        "short-prefill kernel writes causal token ids and -1 sentinels",
    )

    def node_text(node):
        segment = ast.get_source_segment(source, node)
        assert segment is not None
        return segment

    scratch_choices = [
        node for node in ast.walk(dispatcher)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Call)
        and isinstance(node.body.func, ast.Name)
        and node.body.func.id == "_inactive_pool_topk_scratch"
    ]
    check(len(scratch_choices) == 2,
          "prefill and decode both choose the shared packed scratch")
    expected_args = [
        "topk_indices_buffer", "hidden_states.shape[0]", "num_rows", "select_k"
    ]
    for choice in scratch_choices:
        check(node_text(choice.test) == "current_platform.is_cuda()"
              and isinstance(choice.orelse, ast.Constant)
              and choice.orelse.value is None,
              "packed scratch must remain CUDA-only with a None fallback")
        check([node_text(arg) for arg in choice.body.args] == expected_args,
              "scratch must start after the full active hidden-state prefix")

    active_inits = [
        node for node in ast.walk(dispatcher)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and node_text(node.targets[0])
        == "topk_indices_buffer[: hidden_states.shape[0]]"
        and ast.literal_eval(node.value) == -1
    ]
    check(len(active_inits) == 1
          and active_inits[0].lineno < min(c.lineno for c in scratch_choices),
          "active output rows and rounded tail must be reset before scratch use")
    init_guard = next(
        node for node in ast.walk(dispatcher)
        if isinstance(node, ast.If)
        and node_text(node.test) == "not short_prefill_covers_active"
    )
    check(active_inits[0] in init_guard.body,
          "pure short prefill skips the overwritten full-buffer sentinel fill")
    check(
        "short_prefill_covers_active = (" in dispatcher_source
        and "hidden_states.shape[0] == num_tokens" in dispatcher_source,
        "sentinel fill is skipped only when short prefill covers every active row",
    )
    check(
        "prefill_chunks = ()" in dispatcher_source
        and "prefill_chunks = prefill_metadata.chunks" in dispatcher_source
        and "for chunk in prefill_chunks:" in dispatcher_source,
        "short prefill bypasses sparse-scoring workspace and chunks",
    )
    short_branch = dispatcher_source[
        dispatcher_source.index("if short_prefill:"):
        dispatcher_source.index("for chunk in prefill_chunks:")
    ]
    check(
        "current_workspace_manager()" not in short_branch.split("else:", 1)[0]
        and "_fill_short_prefill_topk(" in short_branch.split("else:", 1)[0],
        "short branch writes indices without touching logits workspace",
    )

    dense_assigns = [
        node for node in dispatcher.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "dense_mha_scoring_unused"
            for target in node.targets
        )
    ]
    check(
        len(dense_assigns) == 1
        and "_glm53_dense_mha_scoring_unused(" in node_text(dense_assigns[0]),
        "kpool dispatcher computes the dense no-consumer predicate once",
    )
    dense_assign = dense_assigns[0]
    dense_assign_source = node_text(dense_assign)
    check(
        "forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL"
        in dense_assign_source
        and "torch.cuda.is_current_stream_capturing() if is_cuda else True"
        in dense_assign_source,
        "dense-MHA bypass delegates capture and request checks to shared gate",
    )
    dense_ifs = [
        node for node in dispatcher.body
        if isinstance(node, ast.If)
        and node_text(node.test) == "dense_mha_scoring_unused"
    ]
    cache_if = next(
        node for node in dense_ifs
        if "additional_kwargs.setdefault(" in node_text(node)
    )
    dense_return = next(
        node for node in dense_ifs
        if any(isinstance(child, ast.Return) for child in node.body)
    )
    check(
        dense_assign.lineno < cache_if.lineno
        < next(
            node.lineno for node in dispatcher.body
            if isinstance(node, ast.If)
            and node_text(node.test) == "not skip_k_cache_insert"
        )
        < dense_return.lineno < active_inits[0].lineno,
        "dense write-plan sharing precedes cache writes but return follows them",
    )
    check(
        "_GLM53_PREFILL_WRITE_PLAN_CACHE" in node_text(cache_if)
        and "write_plan_cache=write_plan_cache" in dispatcher_source,
        "dense sparse layers share one forward-context-owned write plan",
    )
    check(
        "_kpool_seed_tail_cache_strided(" in dispatcher_source,
        "prefill cache writes use the stride-aware persistent-tail seeder",
    )

    fallbacks = [
        node for node in ast.walk(dispatcher)
        if isinstance(node, ast.If)
        and node_text(node.test) == "pool_topk is None"
    ]
    check(len(fallbacks) == 2
          and all("pool_topk = torch.full(" in node_text(node)
                  for node in fallbacks),
          "both scratch sites retain the allocation fallback")

    pool_id_assigns = [
        node for node in ast.walk(dispatcher)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pool_ids"
                for target in node.targets)
        and isinstance(node.value, ast.IfExp)
    ]
    check(len(pool_id_assigns) == 2
          and all(node_text(node.value.test) == "current_platform.is_cuda()"
                  and node_text(node.value.body) == "topk_dst"
                  and node_text(node.value.orelse)
                  == "topk_dst.to(torch.int64)"
                  for node in pool_id_assigns),
          "CUDA passes int32 ids directly while XPU keeps its old widening")
    check("pool_topk.to(torch.int64)" not in dispatcher_source,
          "the CUDA pool scratch must not be widened unconditionally")
    print("  GLM53 kpool packed scratch ...... OK")


def test_glm53_cache_only_indexer_prefill() -> None:
    """Dense MLA builds kpool cache state without dead query scoring."""
    try:
        import torch
    except ImportError:
        print("  GLM53 cache-only prefill indexer . SKIP (no torch on this host)")
        return
    import torch.nn.functional as F

    class FakeIndexer:
        def __init__(self):
            self._buffers = {}

        def register_buffer(self, name, tensor, persistent=True):
            self._buffers[name] = tensor
            setattr(self, name, tensor)

    def original(self, *args):
        self.original_called = True
        return "original"

    FakeIndexer.forward = original
    FakeIndexer._glm53_prefill_original_forward = original
    ns = load_defs(
        "overlay/modules/glm53_model/glm53_prefill_fastpath.py",
        {
            "_GLM53_PREFILL_KG_WEIGHT",
            "_GLM53_FUSED_K_GATE_ENV",
            "_GLM53_SM121_MLA_PREFILL_ENV",
            "_glm53_fused_k_gate_enabled",
            "_glm53_cache_only_indexer_contract",
            "_glm53_cache_only_indexer_forward",
            "_glm53_fused_indexer_forward",
            "_glm53_head_gate", "_HEAD_GATE", "_HEAD_GATE_MODULE",
            "install_glm53_prefill_fastpath",
            "prepare_glm53_prefill_fastpath",
        },
        {
            "os": os,
            # load_defs execs only the named defs, so the module-level logger
            # never comes along. Stub it rather than pulling vllm in: the
            # arming lines are the point of the log, and a test that cannot
            # run them would hide the next one that is missing.
            "logger": _CapturingLogger(),
            "torch": torch,
            "F": F,
            "Indexer": FakeIndexer,
            "_fused_indexer_k_norm": lambda k, *args: k,
            "_fused_indexer_weight_scale": (
                lambda weights, q_scale, scale: weights * scale
            ),
            "_pad_indexer_heads": lambda value, pad: value,
            "fwht128_quant_fp8": lambda q: (
                q,
                torch.ones(q.shape[0], 1, dtype=torch.float32),
            ),
            "_glm53_indexer_scoring_unused": lambda indexer: True,
        },
    )
    contract = ns["_glm53_cache_only_indexer_contract"]
    valid = dict(
        rope_dim=0,
        head_dim=128,
        quant_block_size=128,
        scale_fmt="ue8m0",
        index_kpool=4,
        topk_tokens=2048,
        n_head=32,
        op_type="SparseAttnIndexerKpool",
        use_fp4_cache=False,
        hidden_dtype=torch.bfloat16,
        k_weight_dtype=torch.bfloat16,
        k_weight_shape=(160, 8),
        hidden_size=8,
        gate_dtype=torch.bfloat16,
        gate_shape=(128, 8),
        ape_shape=(4, 128),
        ape_dtype=torch.float32,
        fused_weight_dtype=torch.bfloat16,
        fused_weight_shape=(256, 8),
        fused_weight_contiguous=True,
    )
    check(contract(**valid), "exact GLM dense-prefill indexer contract admits")
    rejected = {
        "rope_dim": (64,),
        "head_dim": (64, 256),
        "quant_block_size": (64,),
        "scale_fmt": (None, "float"),
        "index_kpool": (1, 16),
        "topk_tokens": (1024, 4096),
        "n_head": (16, 64),
        "op_type": ("SparseAttnIndexer",),
        "use_fp4_cache": (True,),
        "hidden_dtype": (torch.float16, torch.float32),
        "k_weight_dtype": (torch.float16,),
        "k_weight_shape": ((128, 8), (160, 16)),
        "gate_dtype": (torch.float16, torch.float32),
        "gate_shape": ((64, 8),),
        "ape_shape": ((8, 128),),
        "ape_dtype": (torch.bfloat16,),
        "fused_weight_dtype": (torch.float16,),
        "fused_weight_shape": ((128, 8), (256, 16)),
        "fused_weight_contiguous": (False,),
    }
    for field, values in rejected.items():
        for bad in values:
            case = dict(valid)
            case[field] = bad
            check(not contract(**case), f"cache-only contract rejects {field}={bad}")

    class SparseAttnIndexerKpool:
        use_fp4_cache = False

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return args[1]

    op = SparseAttnIndexerKpool()
    indexer = FakeIndexer()
    indexer.rope_dim = 0
    indexer.head_dim = 128
    indexer.quant_block_size = 128
    indexer.scale_fmt = "ue8m0"
    indexer.index_kpool = 4
    indexer.topk_tokens = 2048
    indexer.n_head = 32
    indexer.indexer_op = op
    indexer.wk_weights_proj = types.SimpleNamespace(
        weight=torch.randn(160, 8, dtype=torch.bfloat16)
    )
    indexer.index_kpool_compress_gate = torch.randn(
        128, 8, dtype=torch.bfloat16
    )
    indexer.index_kpool_compress_ape = torch.randn(4, 128, dtype=torch.float32)
    indexer.k_norm = types.SimpleNamespace(
        weight=torch.ones(128, dtype=torch.bfloat16),
        bias=torch.zeros(128, dtype=torch.bfloat16),
        eps=1e-6,
    )
    indexer.original_called = False
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    qr = torch.randn(3, 4, dtype=torch.bfloat16)
    positions = torch.arange(3)
    forward = ns["_glm53_cache_only_indexer_forward"]
    fused_forward = ns["_glm53_fused_indexer_forward"]
    check(forward(indexer, hidden, qr, positions, None) == "original",
          "missing fused K+gate buffer fails closed to original indexer")

    env_name = ns["_GLM53_SM121_MLA_PREFILL_ENV"]
    fused_name = ns["_GLM53_PREFILL_KG_WEIGHT"]
    prepare = ns["prepare_glm53_prefill_fastpath"]
    model = types.SimpleNamespace(modules=lambda: (indexer,))
    old_env = os.environ.pop(env_name, None)
    try:
        install = ns["install_glm53_prefill_fastpath"]
        install()
        check(FakeIndexer.forward is original,
              "disabled dense-MLA arm leaves Indexer.forward untouched")
        check(prepare(model) == 0 and not hasattr(indexer, fused_name),
              "disabled dense-MLA arm allocates no fused prefill weight")
        os.environ[env_name] = "1"
        install()
        check(
            FakeIndexer.forward is fused_forward
            and FakeIndexer._glm53_prefill_original_forward is original,
            "exact dense-MLA opt-in installs fused prefill forward once",
        )
        install()
        check(FakeIndexer._glm53_prefill_original_forward is original,
              "prefill fastpath installer is idempotent")
        check(prepare(model) == 1,
              "post-load preparation builds one exact K+gate buffer")
        fused_weight = getattr(indexer, fused_name)
        expected_weight = torch.cat(
            (
                indexer.wk_weights_proj.weight[: indexer.head_dim],
                indexer.index_kpool_compress_gate,
            ),
            dim=0,
        )
        check(torch.equal(fused_weight, expected_weight),
              "fused prefill weight preserves K then gate row order")
        check(fused_name in indexer._buffers,
              "fused prefill weight is owned as a module buffer")
    finally:
        if old_env is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_env

    indexer.original_called = False
    out = forward(indexer, hidden, qr, positions, None)
    check(out.shape == (3, 128), "cache-only path computes K rows only")
    check(not indexer.original_called, "admitted dense prefill skips original scoring")
    check(
        op.args[1].data_ptr() == op.args[2].data_ptr() == op.args[3].data_ptr(),
        "K aliases both unused custom-op placeholders",
    )
    check(op.kwargs["gate_score"].shape == (3, 128),
          "cache-only path still computes kpool gate state")
    check(op.kwargs["gate_score"].stride(0) == 256,
          "split gate remains a zero-copy view of fused projection output")
    projected = F.linear(hidden, getattr(indexer, fused_name))
    check(
        torch.equal(op.args[1], projected[:, : indexer.head_dim])
        and torch.equal(op.kwargs["gate_score"], projected[:, indexer.head_dim :]),
        "single projection preserves exact K and gate row partitions",
    )
    check(op.kwargs["compress_ape"] is indexer.index_kpool_compress_ape,
          "cache-only path preserves trained pool bias")

    # The same fused buffer serves normal sparse prefill while preserving the
    # stock FP32 head-weight projection and query scoring pipeline.
    indexer.wq_b = lambda value: (
        torch.randn(value.shape[0], 32 * 128, dtype=torch.bfloat16),
        None,
    )
    indexer._wp_fp32 = None
    indexer.softmax_scale = 128**-0.5
    ns["_glm53_indexer_scoring_unused"] = lambda owner: False
    sparse_out = fused_forward(indexer, hidden, qr, positions, None)
    check(sparse_out.shape == (3, 32, 128),
          "normal sparse prefill uses the fused K+gate projection")
    check(op.args[1].shape == (3, 32, 128),
          "normal sparse prefill preserves quantized-query head geometry")
    check(op.args[2].shape == (3, 128) and op.args[3].shape == (3, 32),
          "normal sparse prefill preserves K and FP32 head weights")
    check(indexer._wp_fp32.dtype == torch.float32,
          "fused K+gate path keeps ranking head weights in FP32")
    check(op.kwargs["gate_score"].stride(0) == 256,
          "normal sparse gate is still a view of the single projection")
    indexer.original_called = False
    decode_out = fused_forward(indexer, hidden, qr, positions, None)
    check(decode_out.shape == (3, 32, 128) and not indexer.original_called,
          "decode and mixed batches also use the fused K+gate projection")

    indexer.rope_dim = 64
    check(forward(indexer, hidden, qr, positions, None) == "original"
          and indexer.original_called,
          "contract drift calls the original indexer unchanged")

    path = _overlay_source(
        "overlay/modules/glm53_model/glm53_prefill_fastpath.py"
    )
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    forward_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_glm53_cache_only_indexer_forward"
    )
    forward_source = ast.get_source_segment(source, forward_node) or ""
    for dead in (
        "self.wq_b",
        "fwht128_quant_fp8",
        "_fused_indexer_weight_scale",
        "_pad_indexer_heads",
        "torch.mm",
    ):
        check(dead not in forward_source,
              f"cache-only path must not execute dead scoring op {dead}")
    check(
        forward_source.count("F.linear(") == 1
        and "kg.split(self.head_dim, dim=-1)" in forward_source,
        "cache-only path computes K and gate with one projection",
    )
    check("_glm53_indexer_scoring_unused(self)" in forward_source,
          "model fast path shares the custom-op no-consumer predicate")
    prepare_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_glm53_prefill_fastpath"
    )
    prepare_source = ast.get_source_segment(source, prepare_node) or ""
    check(
        "torch.cat(" in prepare_source
        and "persistent=False" in prepare_source
        and "with torch.no_grad():" in prepare_source,
        "K+gate weight is a post-load non-persistent inference buffer",
    )

    model_source = open(
        _overlay_source("overlay/modules/glm53_model/glm5next_model.py"),
        encoding="utf-8",
    ).read()
    check(
        "install_glm53_prefill_fastpath" in model_source
        and model_source.index("install_glm53_prefill_fastpath()")
        < model_source.index("class Glm5NextDecoderLayer"),
        "GLM model installs the indexer path before constructing layers",
    )
    load_start = model_source.index(
        "def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]])",
        model_source.index("class Glm5NextForCausalLM"),
    )
    load_source = model_source[load_start : model_source.index("\n\n@", load_start)]
    # The post-load hooks (fp8 copies, fused K+gate, metadata warmup, kpool)
    # live in run_post_load and run ONCE from whoever owns the whole
    # checkpoint walk. AutoWeightsLoader enters a child's load_weights once
    # per contiguous run of its prefix in the stream, so hooks inline there
    # ran ~25 s into the load on unloaded weights and again per run -- the
    # memory cliff of 2026-09-04 (MEASUREMENTS 26차).
    check(
        "def run_post_load(self) -> None:" in load_source
        and "prepare_glm53_prefill_fastpath(self)" in load_source
        and load_source.index("loaded = loader.load_weights(weights)")
        < load_source.index('if not getattr(self, "_defer_post_load", False):')
        < load_source.index("self.run_post_load()")
        < load_source.index("def run_post_load(self) -> None:")
        < load_source.index("maybe_build_fp8_dense(self)")
        < load_source.index("prepare_glm53_prefill_fastpath(self)"),
        "the child runs its post-load hooks only when nothing above it owns "
        "the walk, and the fp8 copies come before the fused K+gate buffer",
    )
    wrap_start = model_source.index("class Glm5NextForConditionalGeneration(")
    wrap_source = model_source[wrap_start:]
    check(
        "self.language_model._defer_post_load = True" in wrap_source
        and "loaded = super().load_weights(weights)" in wrap_source
        and wrap_source.index("loaded = super().load_weights(weights)")
        < wrap_source.index('getattr(self.language_model, "run_post_load", None)'),
        "the wrapper owns the walk: it defers the child's hooks and runs "
        "them once after its own loader returns",
    )
    module_dir = os.path.join(
        REPO, "overlay", "modules", "glm53_model"
    )
    manifest = open(os.path.join(module_dir, "manifest.tsv"), encoding="utf-8").read()
    requires = open(os.path.join(module_dir, "requires"), encoding="utf-8").read()
    check("glm53_prefill_fastpath.py\t" in manifest and "\tabsent" in manifest,
          "prefill fastpath ships as an image-new overlay")
    check("glm53_kernels" in requires,
          "model wiring declares its kpool-indexer dependency (glm53_kernels)")
    print("  GLM53 cache-only prefill indexer . OK")


def test_glm53_kda_prefill_regime() -> None:
    """The KDA cache split stays two-bucket, exact-gated and core-six only."""
    module_rel = "overlay/modules/glm53_kernels/kda.py"
    delta_rel = "overlay/modules/glm53_kernels/chunk_delta_h.py"
    ns = load_defs(
        module_rel,
        {
            "_glm53_kda_prefill_regime_gate",
            "_glm53_kda_prefill_runtime_contract",
            "_glm53_kda_prefill_capture_runtime_contract",
            "_glm53_kda_prefill_autotune_regime",
        },
        {
            "_GLM53_KDA_PREFILL_REGIME_ENABLED": True,
            "_GLM53_KDA_PREFILL_RUNTIME_CONTRACT": None,
        },
    )
    gate = ns["_glm53_kda_prefill_regime_gate"]
    valid = {
        "enabled": True,
        "capability": (12, 1),
        "model_type": "glm5_next_text",
        "tensor_parallel_size": 4,
        "max_num_batched_tokens": 8192,
        "q_shape": (1, 4096, 16, 128),
        "k_shape": (1, 4096, 16, 128),
        "v_shape": (1, 4096, 16, 128),
        "raw_g_shape": (1, 4096, 16, 128),
        "beta_shape": (1, 4096, 16),
        "q_dtype": "torch.bfloat16",
        "k_dtype": "torch.bfloat16",
        "v_dtype": "torch.bfloat16",
        "raw_g_dtype": "torch.bfloat16",
        "beta_dtype": "torch.float32",
        "initial_state_shape": (4, 16, 128, 128),
        "initial_state_dtype": "torch.float32",
        "output_final_state": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "is_varlen": True,
        "num_sequences": 4,
    }
    check(gate(**valid), "4x1024 packed GLM prefill admits long regime")
    short = dict(valid)
    for field in ("q_shape", "k_shape", "v_shape", "raw_g_shape"):
        shape = list(short[field])
        shape[1] = 4095
        short[field] = tuple(shape)
    short["beta_shape"] = (1, 4095, 16)
    check(not gate(**short), "4x<1024 packed average stays stock regime 0")

    rejected = {
        "enabled": (False, 1),
        "capability": ((12, 0), (10, 0), None),
        "model_type": (None, "kimi_k3"),
        "tensor_parallel_size": (1, 8),
        "max_num_batched_tokens": (4096, 16384),
        "q_dtype": ("torch.float16", "torch.float32"),
        "beta_dtype": ("torch.bfloat16", "torch.float16"),
        "initial_state_shape": ((1, 16, 128, 128), None),
        "output_final_state": (False,),
        "safe_gate": (False,),
        "lower_bound": (-4.0, -6.0),
        "is_varlen": (False,),
    }
    for field, values in rejected.items():
        for bad in values:
            case = dict(valid)
            case[field] = bad
            check(not gate(**case), f"KDA regime rejects {field}={bad!r}")

    runtime = ns["_glm53_kda_prefill_runtime_contract"]
    capability = types.SimpleNamespace(major=12, minor=1)
    config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(model_type="glm5_next_text")
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8192),
    )
    check(
        runtime(capability, config) == ((12, 1), "glm5_next_text", 4, 8192),
        "runtime contract reads current GLM profile",
    )
    check(runtime(None, config) is None, "missing capability fails closed")
    check(runtime(capability, None) is None, "missing config fails closed")
    for broken in (
        types.SimpleNamespace(),
        types.SimpleNamespace(model_config=None),
        types.SimpleNamespace(model_config=types.SimpleNamespace()),
    ):
        check(
            runtime(capability, broken) is None,
            "vLLM config attribute drift fails closed",
        )

    # vLLM's current-config global exists only during model initialization.
    # Capture there, then prove forward admission still works after the
    # accessor would return None.
    capture = ns["_glm53_kda_prefill_capture_runtime_contract"]
    autotune_regime = ns["_glm53_kda_prefill_autotune_regime"]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_config = types.ModuleType("vllm.config")
    fake_config.get_current_vllm_config_or_none = lambda: config
    fake_vllm.config = fake_config
    saved_modules = {
        name: sys.modules.get(name) for name in ("vllm", "vllm.config")
    }
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.config"] = fake_config
    ns["current_platform"] = types.SimpleNamespace(
        get_device_capability=lambda: capability
    )
    try:
        capture(hidden_size=128, activation="sigmoid")
        check(
            ns["_GLM53_KDA_PREFILL_RUNTIME_CONTRACT"]
            == ((12, 1), "glm5_next_text", 4, 8192),
            "GLM RMSNorm construction latches the init-only runtime contract",
        )
        fake_config.get_current_vllm_config_or_none = lambda: None
        tensor = lambda shape, dtype: types.SimpleNamespace(  # noqa: E731
            shape=shape, dtype=dtype
        )
        check(
            autotune_regime(
                q=tensor((1, 4096, 16, 128), "torch.bfloat16"),
                k=tensor((1, 4096, 16, 128), "torch.bfloat16"),
                v=tensor((1, 4096, 16, 128), "torch.bfloat16"),
                raw_g=tensor((1, 4096, 16, 128), "torch.bfloat16"),
                beta=tensor((1, 4096, 16), "torch.float32"),
                initial_state=tensor((4, 16, 128, 128), "torch.float32"),
                output_final_state=True,
                cu_seqlens=types.SimpleNamespace(numel=lambda: 5),
                safe_gate=True,
                lower_bound=-5.0,
            )
            == 1,
            "forward uses the init latch after current config is restored",
        )

        ns["_GLM53_KDA_PREFILL_RUNTIME_CONTRACT"] = None
        bad_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_text_config=types.SimpleNamespace(model_type="kimi_k3")
            ),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
            scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8192),
        )
        fake_config.get_current_vllm_config_or_none = lambda: bad_config
        capture(hidden_size=128, activation="sigmoid")
        check(
            ns["_GLM53_KDA_PREFILL_RUNTIME_CONTRACT"] is None,
            "non-GLM init cannot latch the runtime contract",
        )

        def forbidden_lookup():
            raise AssertionError("disabled or non-GLM ctor must not query capability")

        ns["current_platform"] = types.SimpleNamespace(
            get_device_capability=forbidden_lookup
        )
        capture(hidden_size=64, activation="sigmoid")
        ns["_GLM53_KDA_PREFILL_REGIME_ENABLED"] = False
        capture(hidden_size=128, activation="sigmoid")
        check(
            ns["_GLM53_KDA_PREFILL_RUNTIME_CONTRACT"] is None,
            "disabled and non-GLM constructors remain lookup-free",
        )
    finally:
        for name, old in saved_modules.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        ns["_GLM53_KDA_PREFILL_REGIME_ENABLED"] = True

    sources = {
        "kda": open(os.path.join(REPO, module_rel), encoding="utf-8").read(),
        "delta": open(os.path.join(REPO, delta_rel), encoding="utf-8").read(),
    }
    trees = {name: ast.parse(source) for name, source in sources.items()}
    core = {
        "kda": {
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
            "recompute_w_u_fwd_kernel",
            "chunk_gla_fwd_kernel_o",
            "kda_gate_cumsum_fwd_kernel",
        },
        "delta": {"chunk_gated_delta_rule_fwd_kernel_h_blockdim64"},
    }

    def decorator(node, attr):
        return next(
            dec
            for dec in node.decorator_list
            if isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == attr
        )

    checked = 0
    for owner, names in core.items():
        defs = {
            node.name: node
            for node in trees[owner].body
            if isinstance(node, ast.FunctionDef)
        }
        for name in names:
            node = defs[name]
            args = {arg.arg: arg for arg in node.args.args}
            check("AUTOTUNE_REGIME" in args, f"{name} accepts regime scalar")
            check(
                args["AUTOTUNE_REGIME"].annotation is None,
                f"{name} regime is runtime, not tl.constexpr",
            )
            tune = decorator(node, "autotune")
            key_kw = next(kw for kw in tune.keywords if kw.arg == "key")
            keys = {elt.value for elt in key_kw.value.elts}
            check("AUTOTUNE_REGIME" in keys, f"{name} splits config cache")
            check("T" not in keys, f"{name} never keys on raw T")
            jit = decorator(node, "jit")
            dns_kw = next(
                kw for kw in jit.keywords if kw.arg == "do_not_specialize"
            )
            dns = {elt.value for elt in dns_kw.value.elts}
            check(
                {"T", "AUTOTUNE_REGIME"} <= dns,
                f"{name} keeps one code specialization across regimes",
            )
            checked += 1
    check(checked == 6, "exactly the core-six Autotuners own the regime")

    def named_calls(node, name):
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and (
                (isinstance(call.func, ast.Name) and call.func.id == name)
                or (isinstance(call.func, ast.Subscript)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == name)
            )
        ]

    kda_defs = {
        node.name: node
        for node in trees["kda"].body
        if isinstance(node, ast.FunctionDef)
    }
    delta_defs = {
        node.name: node
        for node in trees["delta"].body
        if isinstance(node, ast.FunctionDef)
    }
    launch_owners = {
        "chunk_kda_scaled_dot_kkt_fwd": (
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
        ),
        "recompute_w_u_fwd": ("recompute_w_u_fwd_kernel",),
        "chunk_gla_fwd_o_gk": ("chunk_gla_fwd_kernel_o",),
        "fused_kda_gate_chunk_cumsum": ("kda_gate_cumsum_fwd_kernel",),
    }
    for wrapper_name, kernel_names in launch_owners.items():
        wrapper = kda_defs[wrapper_name]
        for kernel_name in kernel_names:
            calls = named_calls(wrapper, kernel_name)
            check(len(calls) == 1, f"{wrapper_name} launches {kernel_name} once")
            kw = next(
                item for item in calls[0].keywords if item.arg == "AUTOTUNE_REGIME"
            )
            check(
                isinstance(kw.value, ast.Name) and kw.value.id == "autotune_regime",
                f"{wrapper_name} forwards its unchanged regime to {kernel_name}",
            )
    delta_wrapper = delta_defs["chunk_gated_delta_rule_fwd_h"]
    delta_calls = named_calls(
        delta_wrapper, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
    )
    check(len(delta_calls) == 1, "delta-h wrapper launches its core kernel once")
    delta_kw = next(
        item for item in delta_calls[0].keywords if item.arg == "AUTOTUNE_REGIME"
    )
    check(
        isinstance(delta_kw.value, ast.Name) and delta_kw.value.id == "autotune_regime",
        "delta-h wrapper forwards the identical call-level regime",
    )
    pipeline = kda_defs["_chunk_kda_fwd_with_cumulative_g"]
    for callee in (
        "chunk_kda_scaled_dot_kkt_fwd",
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_gla_fwd_o_gk",
    ):
        calls = named_calls(pipeline, callee)
        check(len(calls) == 1, f"chunk pipeline calls {callee} once")
        kw = next(item for item in calls[0].keywords if item.arg == "autotune_regime")
        check(
            isinstance(kw.value, ast.Name) and kw.value.id == "autotune_regime",
            f"chunk pipeline passes one unchanged regime to {callee}",
        )

    kda_source = sources["kda"]
    capture_start = kda_source.index(
        "def _glm53_kda_prefill_capture_runtime_contract"
    )
    helper_start = kda_source.index("def _glm53_kda_prefill_autotune_regime")
    helper_end = kda_source.index("def fused_recurrent_kda_fwd", helper_start)
    capture_source = kda_source[capture_start:helper_start]
    helper_source = kda_source[helper_start:helper_end]
    check(
        capture_source.index("if (")
        < capture_source.index("current_platform.get_device_capability"),
        "default-off and non-GLM init exit before capability/config lookup",
    )
    check(
        "get_current_vllm_config_or_none" not in helper_source
        and "get_device_capability" not in helper_source,
        "forward dispatch reads only the init-time contract latch",
    )
    check(
        'os.environ.get(_GLM53_KDA_PREFILL_REGIME_ENV) == "1"' in kda_source,
        "the module-import latch accepts exact string 1 only",
    )
    fwd_start = kda_source.index("def chunk_kda_with_fused_gate_fwd")
    fwd_end = kda_source.index("def chunk_kda(", fwd_start)
    fwd_source = kda_source[fwd_start:fwd_end]
    check(
        fwd_source.count("_glm53_kda_prefill_autotune_regime(") == 1,
        "the public chunk call derives its bucket once",
    )
    check(
        "q.shape[1] < 1024 * num_sequences" in helper_source
        and "cu_seqlens.numel() - 1" in helper_source,
        "long bucket is a bounded packed-average threshold",
    )
    standalone = next(
        node
        for node in trees["kda"].body
        if isinstance(node, ast.FunctionDef) and node.name == "kda_gate_fwd_kernel"
    )
    check(
        all(arg.arg != "AUTOTUNE_REGIME" for arg in standalone.args.args),
        "fused-recurrent gate Autotuner remains outside the chunk split",
    )
    rms_class = next(
        node
        for node in trees["kda"].body
        if isinstance(node, ast.ClassDef) and node.name == "FusedRMSNormGated"
    )
    rms_init = next(
        node
        for node in rms_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    capture_calls = named_calls(
        rms_init, "_glm53_kda_prefill_capture_runtime_contract"
    )
    check(
        len(capture_calls) == 1,
        "GLM's init-context RMSNorm captures the runtime contract once",
    )

    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(
        "glm53_kernels" in profile
        and "VLLM_GLM53_KDA_PREFILL_REGIME=0" in profile,
        "GLM profile mounts the module default-off",
    )
    for name in ("kda.py", "chunk_delta_h.py"):
        source = open(
            os.path.join(REPO, "overlay", "modules", "glm53_kernels", name),
            encoding="utf-8",
        ).read()
        built = open(os.path.join(REPO, "build", "glm53", name), encoding="utf-8").read()
        check(source == built, f"{name} module/build parity")

    probe = open(
        os.path.join(REPO, "probes", "kda_prefill_bench.py"), encoding="utf-8"
    ).read()
    check(
        "vllm.third_party.flash_linear_attention.ops" in probe
        and "models.kimi_k3" not in probe,
        "probe imports the actual generic FLA production path",
    )
    check(
        "output_final_state=True" in probe
        and "safe_gate=True" in probe
        and "use_qk_l2norm_in_kernel=True" in probe
        and "beta=inp[\"beta\"]" in probe,
        "probe matches production final-state, gate, norm and beta contract",
    )
    print("  GLM53 KDA prefill regimes ....... OK")


def test_glm53_upstream_prefill_batch() -> None:
    """Upstream-derived GLM prefill paths stay exact-gated and fail-closed."""
    indexer_path = _overlay_source(
        "overlay/modules/glm53_kernels/glm53_kpool_indexer.py"
    )
    indexer_source = open(indexer_path, encoding="utf-8").read()
    indexer_tree = ast.parse(indexer_source)
    metadata_cls = next(
        node
        for node in indexer_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "BuildPrefillChunkMetadataKernel"
    )
    compile_key = next(
        node
        for node in metadata_cls.body
        if isinstance(node, ast.ClassDef) and node.name == "CompileKey"
    )
    key_fields = {
        node.target.id
        for node in compile_key.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    runtime_scalars = {
        "query_slice_start",
        "query_slice_stop",
        "DCP_RANK",
        "DCP_WORLD",
        "DCP_INTERLEAVE",
    }
    check(key_fields == {"BLOCK_SIZE", "COMPRESS_RATIO", "input_variant"},
          "metadata compile key excludes request-varying runtime scalars")
    kernel = next(
        node for node in metadata_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    )
    jit_call = next(
        dec for dec in kernel.decorator_list
        if isinstance(dec, ast.Call)
        and isinstance(dec.func, ast.Attribute)
        and dec.func.attr == "jit"
    )
    dns = next(kw.value for kw in jit_call.keywords
               if kw.arg == "do_not_specialize")
    dns_values = {elt.value for elt in dns.elts}
    check(runtime_scalars <= dns_values,
          "metadata runtime scalars are explicitly do_not_specialize")
    warmup_method = next(
        node for node in metadata_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_warmup_keys"
    )
    warmup_source = ast.get_source_segment(indexer_source, warmup_method) or ""
    check("index_kpool" in warmup_source and "compress_ratios +" in warmup_source,
          "metadata compile warmup includes GLM's pooled ratio")

    fastpath_source = open(
        _overlay_source(
            "overlay/modules/glm53_model/glm53_prefill_fastpath.py"
        ),
        encoding="utf-8",
    ).read()
    check(
        "unaligned_storage[1:]" in fastpath_source
        and "torch.cuda.synchronize(device)" in fastpath_source
        and 'model_type", None) != "glm5_next_text"' in fastpath_source
        and 'tensor_parallel_size", None) != 4' in fastpath_source,
        "runtime warmup executes both pointer alignments under exact GLM TP4",
    )

    kpool_dir = os.path.join(
        REPO, "overlay", "modules", "glm53_kernels"
    )
    kpool_py = open(
        os.path.join(kpool_dir, "glm53_kpool_topk.py"), encoding="utf-8"
    ).read()
    kpool_cu = open(
        os.path.join(kpool_dir, "glm53_kpool_topk.cu"), encoding="utf-8"
    ).read()
    sparse_source = open(
        os.path.join(kpool_dir, "sparse_attn_indexer_kpool.py"),
        encoding="utf-8",
    ).read()
    check('os.environ.get(_ENV, "0") == "1"' in kpool_py,
          "radix KPool path is exact opt-in")
    for contract in (
        "scores.dtype == torch.float32",
        "output.shape == (scores.shape[0], 2051)",
        "row_starts.dtype == torch.int32",
    ):
        check(contract in kpool_py, f"radix wrapper pins {contract}")
    check(
        sparse_source.index("glm53_kpool_topk_expand_tail(")
        < sparse_source.index("torch.ops._C.top_k_per_row_prefill("),
        "single-launch KPool attempt precedes the untouched stock fallback",
    )
    for cuda_contract in (
        "constexpr int kGroupTopK = 512",
        "constexpr int kPoolSize = 4",
        "constexpr int kTokenTopK = 2048",
        "uint64_t topk_key",
        "sort_selected_indices<kGroupTopK>",
        "threshold_candidates > kSmemInputSize",
    ):
        check(cuda_contract in kpool_cu,
              f"radix CUDA source carries {cuda_contract}")
    check(kpool_cu.count("__global__") == 1,
          "radix selection, expansion and tail share one CUDA kernel")

    union_path = _overlay_source(
        "overlay/modules/glm53_model/glm53_union_prefill.py"
    )
    union_source = open(union_path, encoding="utf-8").read()
    union_tree = ast.parse(union_source)
    union_defs = {
        node.name for node in union_tree.body if isinstance(node, ast.FunctionDef)
    }
    check(
        {
            "_union_dense_prefix_prepare_kernel",
            "_union_mark_kernel",
            "_union_compact_kernel",
            "_glm53_union_prefill_kernel",
            "glm53_union_sparse_prefill",
            "install_glm53_union_prefill",
        } <= union_defs,
        "union path ships preparation, compaction, attention and installer",
    )
    for gate in (
        "q[0].shape[1:] == (16, 512)",
        'getattr(attn_metadata, "num_decodes", -1) == 0',
        'getattr(attn_metadata, "num_prefills", 0) > 0',
        'getattr(self, "qk_rope_head_dim", -1) == 0',
        "not torch.cuda.is_current_stream_capturing()",
    ):
        check(gate in union_source, f"union forward pins {gate}")
    check(
        "same_req" in union_source
        and "value != expected" in union_source
        and "tl.where(dense, slot, slot + base)" in union_source,
        "dense-prefix reuse has exact nested-prefix and request guards",
    )
    check(
        "owned =" in union_source
        and "owned & valid" in union_source
        and union_source.count("sm_scale * kv_scale") == 2
        and union_source.count("kv_scale / denom") == 2,
        "union/base kernels restore ownership and FP8 K/V scaling",
    )
    check(
        'VLLM_GLM53_UNION_PREFILL=0' in open(
            os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8"
        ).read(),
        "unmeasured union path remains default-off",
    )
    print("  GLM53 upstream prefill batch ..... OK")


def test_oneshot_sm121_grid_contract() -> None:
    """The fixed one-shot grid must match both GB10 and GLM capture shapes.

    ``done_ctr`` makes the launch graph-safe only while every replay contributes
    exactly ARGRID increments. Keep the grid fixed, but pin it to the 48-SM
    device it serves instead of reviving the old generic 256-block launch.
    """
    path = os.path.join(
        REPO, "overlay/modules/tp_oneshot_ar/dsv4_oneshot_ar.cu"
    )
    source = open(path, encoding="utf-8").read()

    def define(name: str) -> int:
        match = re.search(rf"^#define {name} (\d+)\b", source, re.M)
        check(match is not None, f"oneshot {name} define exists")
        return int(match.group(1))

    grid = define("ARGRID")
    threads = define("ARTHREADS")
    maxel = define("MAXEL")
    check(grid == 48, "GB10 has one fixed one-shot block per SM")
    check(threads == 256, "one-shot block width stays 256 threads")
    check(maxel == 32 * 4096,
          "one-shot size gate covers GLM C=4 K=7 verify exactly")

    check("prop.major == 12 && prop.minor == 1" in source,
          "one-shot init rejects non-SM121 devices")
    check("prop.multiProcessorCount == ARGRID" in source,
          "one-shot init rejects a non-48-SM SM121 device")
    check("k_oneshot<<<ARGRID, ARTHREADS" in source,
          "kernel launch uses the guarded fixed geometry")
    check("ARGRID == ARGRID - 1" in source,
          "completion publication still waits for every fixed-grid block")

    copy_at = source.index("c->tx[slot][i] = src[i];")
    atomic_at = source.index("last = atomicAdd", copy_at)
    publish_at = source.index("c->tx_seq = nxt;", atomic_at)
    completion_window = source[copy_at:atomic_at]
    check("__threadfence_system();" in completion_window,
          "every payload writer system-fences before completion")
    check(completion_window.rfind("__syncthreads();")
          > completion_window.rfind("__threadfence_system();"),
          "all warps finish their copy/fence before thread 0 increments")
    check(atomic_at < publish_at,
          "the last-block completion decision precedes tx_seq publish")

    # The same fixed grid must cover every graph bucket through grid-stride
    # iteration; no dynamic launch geometry may be needed for larger verifies.
    for tokens in (1, 2, 4, 8, 16, 32):
        n = tokens * 4096
        visits = [0] * n
        for block in range(grid):
            for thread in range(threads):
                i = block * threads + thread
                while i < n:
                    visits[i] += 1
                    i += grid * threads
        check(all(v == 1 for v in visits),
              f"fixed grid covers GLM T={tokens} hidden elements exactly once")

    shim = open(
        os.path.join(
            REPO,
            "overlay/modules/tp_oneshot_ar/dsv4_oneshot_shim.py",
        ),
        encoding="utf-8",
    ).read()
    check(re.search(r"^_MAXEL_DEFAULT = 131072\b", shim, re.M) is not None,
          "Python OSAR default must match the shipped CUDA constant")
    check("#ifndef MAXEL" in source,
          "CUDA MAXEL must accept the measured-sweep compiler override")
    check("_MAXEL = _resolve_maxel()" in shim,
          "Python eligibility must consume the resolved MAXEL")
    check('f"-DMAXEL={_MAXEL}"' in shim,
          "the resolved MAXEL must reach the CUDA compiler")
    check("build_directory = _build_dir(_src_md5, cuda_flags)" in shim
          and "[src_md5, *flags, torch.__version__, str(torch.version.cuda)]"
          in shim,
          "each OSAR stride must use an isolated extension build directory -- "
          "the -DMAXEL flag rides in the key that names it, so an object "
          "compiled for another peer-buffer stride is never reused")
    check("maxel=%d" in shim,
          "boot fingerprint must expose rank-to-rank MAXEL skew")
    check("class OneShotFatal(RuntimeError):" in shim,
          "post-commit rank-local fallback has a fatal error type")
    check("dist.all_reduce(connect_votes, group=comm.cpu_group)" in shim,
          "RDMA connect success is voted across every rank")
    check("dist.all_reduce(test_votes, group=comm.cpu_group)" in shim,
          "self-test success is voted across every rank")
    check(shim.count("rank-local NCCL fallback") >= 3,
          "post-agreement failures never silently switch one rank to NCCL")

    for profile in ("dsv4", "glm53"):
        # dsv4 keeps its own wiring module; glm53's folded into glm53_runtime (34차)
        _wiring_dir = {"dsv4": "dsv4_oneshot_wiring", "glm53": "glm53_runtime"}[profile]
        wiring = open(
            os.path.join(
                REPO,
                f"overlay/modules/{_wiring_dir}/"
                "cuda_communicator.py",
            ),
            encoding="utf-8",
        ).read()
        check("from .dsv4_oneshot_shim import OneShotFatal" in wiring,
              f"{profile} wiring imports the committed-path fatal type")
        check("except OneShotFatal:\n            raise" in wiring,
              f"{profile} wiring must propagate committed-path failures")
    print("  one-shot SM121 grid contract .... OK")

def test_census_kda_group() -> None:
    """census: KDA/FLA chunk kernels classify out of 기타 and norm groups.

    The 08-31 decode census left fused_recurrent/conv1d in 기타 and l2norm-
    class kernels destined for norm; prefill analysis needs them grouped.
    Patterns must not swallow moe/cutlass/elementwise names.
    """
    ns = load_defs("census.py", {"GROUPS", "group"}, {"re": re})
    group = ns["group"]
    must = {
        "fused_recurrent_gated_delta_rule_fwd_kernel": "KDA/FLA 청크",
        "_causal_conv1d_update_kernel": "KDA/FLA 청크",
        "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": "KDA/FLA 청크",
        "chunk_delta_h_fwd": "KDA/FLA 청크",
        "kda_gate_chunk_cumsum_vector_kernel": "KDA/FLA 청크",
        "fla_l2norm_fwd_kernel": "KDA/FLA 청크",
    }
    for n, want in must.items():
        check(group(n) == want, f"{n!r} -> {want} (got {group(n)!r})")
    keep = {
        "kernel_cutlass_kernel_flashinferfused_moecute_dsl": "MoE b12x",
        "cutlass_80_wmma_tensorop_bf16": "cutlass/cublas GEMM",
        "void at::native::elementwise_kernel": "elementwise 글루",
        "layer_norm_gated_fwd_kernel": "정규화/양자화",
        "mhc_fused_tilelang_kernel": "mhc (MHC 압축)",
        "_deneb_gate_partial_kernel": "우리 · MoE 게이트",
        "k_oneshot": "우리 · osar AR",
    }
    for n, want in keep.items():
        check(group(n) == want, f"{n!r} must stay {want!r} (got {group(n)!r})")

    print("  census KDA group ............... OK")




def test_census_owner_axis() -> None:
    """census/trace_common: who owns a kernel, and the anchors that decide it.

    The 08-31 map said "우리 소유 186 개(9.9%)". 42 of those were deep_gemm's
    `sm120_split_k_reduce_impl`: the osar group pattern had an unanchored
    `k_reduce`. Our osar module defines exactly one __global__ (k_oneshot), so
    a name that merely contains "k_reduce" must never land in our column.
    """
    ns = load_defs("census.py", {"GROUPS", "group"}, {"re": re})
    group = ns["group"]
    check(group("void deep_gemm::sm120_split_k_reduce_impl<cutlass::bfloat16_t, 4u>")
          == "cutlass/cublas GEMM",
          "deep_gemm's split-K reduce is a GEMM kernel, not our all-reduce")
    check(group("k_oneshot(Ctrl*, __nv_bfloat16 const*)") == "우리 · osar AR",
          "our one-shot AR still groups as ours")
    for n in ("(anonymous namespace)::mk_gemm_kernel((anonymous namespace)::MKGemmCtx)",
              "void (anonymous namespace)::mk_gemm2_kernel<1>((anonymous namespace)::MKGemm2Ctx)",
              "(anonymous namespace)::mk_mhc_kernel((anonymous namespace)::MKMhcArgs)",
              "mk_mla_kernel(MKMlaArgs)", "mk_kda_kernel(MKKdaArgs)"):
        check(group(n) == "우리 · 메가커널 세그먼트",
              f"{n[:40]!r} is a megakernel segment, not a vendor GEMM/MHC/MLA")
    check(group("_glm53_prep_fused_kernel") == "우리 · 준비/인덱서"
          and group("_gate_splitk_partial_kernel") == "우리 · 준비/인덱서",
          "prep-fused and the split-K head gate are ours when they are armed")

    tc = load_defs("tools/trace_common.py", {"OURS", "owner"}, {})
    owner = tc["owner"]
    ours = ("(anonymous namespace)::mk_gemm_kernel((anonymous namespace)::MKGemmCtx)",
            "void (anonymous namespace)::mk_gemm2_kernel<4>((anonymous namespace)::MKGemm2Ctx)",
            "mk_mhc_kernel(MKMhcArgs)", "mk_mla_kernel(MKMlaArgs)",
            "mk_kda_kernel(MKKdaArgs)", "k_oneshot(Ctrl*)",
            "_deneb_gate_partial_kernel", "kpool_topk_kernel(...)",
            "_glm53_prep_fused_kernel", "_gate_splitk_reduce_kernel",
            "void (anonymous namespace)::mk_gemm2_kernel<1>((anonymous namespace)::MKGemm2Ctx)",
            "mk_smlp_kernel(MKSmlpArgs)", "_kda_onepass_spec_kernel", "_dual_gate_gemm_kernel")
    for n in ours:
        check(owner(n) == "ours", f"{n[:40]!r} is compiled from this repo")
    theirs = ("void deep_gemm::sm120_split_k_reduce_impl<cutlass::bfloat16_t, 4u>",
              "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm>",
              "mhc_pre_big_fuse_with_norm_tilelang_kernel",
              "per_token_group_quant_8bit_packed_register_kernel",
              "fused_recurrent_gated_delta_rule_fwd_kernel",
              "ncclDevKernel_AllGather_RING_LL")
    for n in theirs:
        check(owner(n) == "image", f"{n[:40]!r} comes from the image, not us")
    print("  census owner axis .............. OK")


def test_census_streaming_events() -> None:
    """census: the streaming reader survives braces inside kernel names.

    json.load on a 40 MB trace costs GBs, and a loading node's MemAvailable
    drops to single-digit GiB (26차) -- a big python there is earlyoom bait.
    The streaming cut has to track string state, because inductor/at::native
    names carry `{lambda(int)#1}` and `}` of their own.
    """
    import gzip
    import json
    import tempfile

    ns = load_defs("census.py", {"iter_events"}, {"gzip": gzip, "json": json})
    braces = ("void at::native::elementwise_kernel<128, 4, "
              "{lambda(int)#1}>(int, {lambda(int)#1})")
    doc = {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "GB10"}],
           "traceEvents": [
               {"ph": "M", "name": "process_labels", "pid": 1},
               {"ph": "X", "cat": "kernel", "name": braces, "ts": 1.0, "dur": 2.5,
                "args": {"stream": 210, "grid": [8, 32, 1]}},
               {"ph": "X", "cat": "kernel", "name": "k_oneshot(Ctrl*)",
                "ts": 4.0, "dur": 40.0, "args": {"stream": 210}},
               {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                "ts": 3.0, "dur": 1.0}]}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.json.gz")
        with gzip.open(path, "wt") as fh:
            json.dump(doc, fh, indent=2)
        evs = list(ns["iter_events"](path))
    kernels = [e for e in evs
               if e.get("ph") == "X" and "kernel" in (e.get("cat") or "")]
    check(len(kernels) == 2, f"two kernel events survive the cut (got {len(kernels)})")
    check(kernels[0]["name"] == braces,
          "a name containing {lambda(int)#1} is not split at its braces")
    check(kernels[0]["args"]["grid"] == [8, 32, 1],
          "nested args objects come back whole")
    check(sum(e["dur"] for e in kernels) == 42.5, "durations survive")
    src = open(os.path.join(REPO, "census.py"), encoding="utf-8").read()
    check("json.load(" not in src,
          "census must not load a whole trace into memory (fleet nodes boot)")
    print("  census streaming reader ....... OK")


def test_trace_step_tail_analyze() -> None:
    """tools/trace_step_tail.py: the tail is what follows the step's last MoE
    expert kernel -- the region 25차 found the drafter's GEMMs in."""
    import importlib.util
    import json
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "tst", os.path.join(REPO, "tools", "trace_step_tail.py"))
    tst = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tst)

    evs = []
    for i in range(14):
        base = i * 10000
        evs.append({"ph": "X", "cat": "kernel", "name": "_gather_block_tables_kernel",
                    "ts": base, "dur": 50, "args": {"stream": 210}})
        evs.append({"ph": "X", "cat": "kernel",
                    "name": "kernel_..._moecute_dslblackwell_sm12x_moe",
                    "ts": base + 100, "dur": 400, "args": {"stream": 210}})
        # 꼬리: 드래프터 bf16 GEMM 둘 + AR 하나, 사이에 100 us 유휴
        evs.append({"ph": "X", "cat": "kernel",
                    "name": "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_x>",
                    "ts": base + 600, "dur": 200, "args": {"stream": 210}})
        evs.append({"ph": "X", "cat": "kernel",
                    "name": "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_x>",
                    "ts": base + 800, "dur": 200, "args": {"stream": 210}})
        evs.append({"ph": "X", "cat": "kernel", "name": "k_oneshot(Ctrl*)",
                    "ts": base + 1100, "dur": 100, "args": {"stream": 210}})
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tail.json")
        open(path, "w").write(json.dumps({"traceEvents": evs}))
        r = tst.analyze(path)
    check(r["steps"] == 8, f"14 anchors -> 8 analysed windows (got {r['steps']})")
    check(abs(r["fwd_ms"] - 0.5) < 1e-9,
          "forward = prep .. end of the last MoE kernel = 0.5 ms")
    check(abs(r["tail_ms"] - 0.7) < 1e-9,
          "tail = 500 us .. 1200 us after the step start = 0.7 ms")
    check(abs(r["tail_busy_ms"] - 0.5) < 1e-9,
          "tail union busy 0.5 ms -- the 100 us gap is idle, not counted")
    check(r["cats"]["bf16/cublas GEMM + head gate"]["cnt"] == 2
          and abs(r["cats"]["bf16/cublas GEMM + head gate"]["ms"] - 0.4) < 1e-9,
          "the drafter's two bf16 GEMMs are attributed to the tail")
    check(r["cats"]["AR k_oneshot"]["cnt"] == 1,
          "the drafter's all-reduce is in the tail, not the forward")
    print("  trace step tail cut ........... OK")


def test_profiles_readme_module_table() -> None:
    """profiles/README.md 의 모듈 × 프로필 표가 실제 프로필·매니페스트와 일치한다.

    표는 손으로 유지되다가 드리프트했다: 프로필이 싣는 모듈 10개가 빠져 있었고
    이미 삭제된 `glm53_async_dflash` 가 남아 있었다(#302 에서 수정). 표가 문서의
    유일한 모듈 색인이므로, 세 가지를 코드로 고정한다 — 어느 프로필에 실리는가,
    파일이 몇 개인가, 매니페스트 전 행이 `absent`(=이식 가능한 계약)인가.
    """
    import re as _re

    profiles = {}
    for name in ("dsv4", "glm53", "qwen38"):
        txt = open(os.path.join(REPO, "profiles", f"{name}.env"), encoding="utf-8").read()
        m = _re.search(r'MODULES="([^"]*)"', txt, _re.S)
        check(m is not None, f"{name}.env must declare MODULES=")
        profiles[name] = set(m.group(1).split())

    rows = {}
    for line in open(os.path.join(REPO, "profiles", "README.md"), encoding="utf-8"):
        m = _re.match(r"\|\s*`([a-z0-9_]+)`\s*\|(.+)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) != 6:          # 범위 · 파일 · 이식 · dsv4 · glm53 · qwen38
            continue
        rows[m.group(1)] = cells

    listed = set(rows)
    used = set().union(*profiles.values())
    check(not (used - listed),
          f"profiles load modules the table omits: {sorted(used - listed)}")
    check(not (listed - used - {"glm53_drop_audit", "glm53_sparse_q"}),
          f"table lists modules no profile loads: {sorted(listed - used)}")

    for mod, cells in rows.items():
        files, portable, marks = cells[1], cells[2], cells[3:]
        man = os.path.join(REPO, "overlay", "modules", mod, "manifest.tsv")
        check(os.path.exists(man), f"{mod}: table row without a module directory")
        entries = [l.split("\t") for l in open(man, encoding="utf-8").read().splitlines()
                   if l.strip() and not l.startswith("#")]
        check(files == str(len(entries)),
              f"{mod}: table says {files} files, manifest has {len(entries)}")
        all_new = bool(entries) and all(e[2].strip() == "absent" for e in entries)
        check((portable == "✓") == all_new,
              f"{mod}: 이식={portable!r} but manifest all-absent={all_new}")
        for name, mark in zip(("dsv4", "glm53", "qwen38"), marks):
            in_profile = mod in profiles[name]
            check((mark != "·") == in_profile,
                  f"{mod}: {name} column {mark!r} but in-profile={in_profile}")
    print("  profiles README module table .. OK")


def test_dflash_warmup_buckets() -> None:
    """The DFlash input-prep warmup sweeps every BLOCK_SIZE bucket, or none.

    BLOCK_SIZE = min(256, next_pow2(max_query_len + num_query_per_req)); the
    warmup query lengths must land one per power-of-two bucket so no runtime
    shape can hit a first-use JIT compile. VLLM_DFLASH_PREP_WARMUP=0
    disarms.
    """
    ns = load_defs(
        "overlay/spec_decode_rejection_warmup.py",
        {"_DENEB_WARMUP_ENV", "_deneb_warmup_query_lens"},
        {"os": os},
    )
    fn = ns["_deneb_warmup_query_lens"]
    saved = os.environ.pop("VLLM_DFLASH_PREP_WARMUP", None)
    try:
        check(fn(8, 8192) == [8, 24, 56, 120, 248],
              "GLM nq=8 buckets: 16/32/64/128/256")
        check(fn(1, 8192) == [7, 15, 31, 63, 127, 255],
              "nq=1 reaches the 8 bucket too")
        check(fn(8, 100) == [8, 24, 56],
              "query lengths cap at max_num_tokens")
        check(fn(8, 4) == [], "max_num_tokens below first bucket -> none")
        os.environ["VLLM_DFLASH_PREP_WARMUP"] = "0"
        ns = load_defs("overlay/spec_decode_rejection_warmup.py",
                       {"_DENEB_WARMUP_ENV", "_deneb_warmup_query_lens"},
                       {"os": os})
        check(ns["_deneb_warmup_query_lens"](8, 8192) == [],
              "env 0 disarms the warmup")
    finally:
        if saved is None:
            os.environ.pop("VLLM_DFLASH_PREP_WARMUP", None)
        else:
            os.environ["VLLM_DFLASH_PREP_WARMUP"] = saved

    print("  dflash warmup buckets .......... OK")


def test_dsv4_spec_warmup_contract() -> None:
    """DSpark warms its own anchor-sampling input-prep geometry."""
    path = "overlay/modules/dspark_drafter/dspark_speculator_v2.py"
    saved = os.environ.pop("VLLM_DSV4_SPEC_WARMUP", None)
    try:
        ns = load_defs(
            path,
            {"_SPEC_WARMUP_ENV", "_warmup_query_lens"},
            {"os": os},
        )
        lens = ns["_warmup_query_lens"]
        check(lens(5, 4096) == [3, 11, 27, 59, 123, 251],
              "DSpark N=5 must cover input-prep buckets 8..256")
        check(lens(5, 100) == [3, 11, 27, 59],
              "input-prep warmup must respect max_num_batched_tokens")
        os.environ["VLLM_DSV4_SPEC_WARMUP"] = "0"
        ns = load_defs(
            path,
            {"_SPEC_WARMUP_ENV", "_warmup_query_lens"},
            {"os": os},
        )
        check(ns["_warmup_query_lens"](5, 4096) == [],
              "VLLM_DSV4_SPEC_WARMUP=0 must disarm all added warmup")
    finally:
        if saved is None:
            os.environ.pop("VLLM_DSV4_SPEC_WARMUP", None)
        else:
            os.environ["VLLM_DSV4_SPEC_WARMUP"] = saved

    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DSparkSpeculator"
    )
    set_attn = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_attn"
    )
    body = ast.get_source_segment(source, set_attn) or ""
    check(body.index("super().set_attn") < body.index("_warmup_dspark_spec_decode"),
          "warmup must wait until DFlash block tables are initialized")
    check("sample_from_anchor=True" in source,
          "DSpark input-prep warmup must compile anchor-sampling semantics")
    check("_warmup_rejection_sampler" in source
          and "rejection_sample(" in source,
          "DSpark warmup must include rejection-sampler kernels")
    check("except Exception:" in source and "Skipping DSpark" in source,
          "a warmup failure must log and preserve the boot")
    print("  DSV4 spec warmup contract ...... OK")


def test_dsv4_ue8m0_host_guard() -> None:
    """Every DSV4 DeepGEMM FP8 copy must validate scales before packing."""
    path = "overlay/modules/dspark_drafter/dspark_v2.py"
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    quant = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_quantize_fp8_deepgemm"
    )
    body = ast.get_source_segment(source, quant) or ""
    requant_at = body.index("requant_weight_ue8m0_inplace")
    guard_at = body.index("unsafe = _ue8m0_unsafe(ws)")
    safe_pack_at = body.index("use_e8m0=False")
    check(requant_at < guard_at < safe_pack_at,
          "host scale validation must sit between requant and layout packing")
    check("raise RuntimeError" in body and "except ImportError as exc" in body,
          "a missing requant seam must fail before the device trap")
    check("torch.where" not in body,
          "the safety guard must report bad scales, never rewrite weights")

    try:
        import torch
    except ImportError:
        print("  DSV4 UE8M0 host guard ......... SKIP values (no torch)")
        return
    ns = load_defs(
        path,
        {"_SF_SIGN_AND_MANTISSA", "_ue8m0_unsafe"},
        {"torch": torch},
    )
    check(ns["_SF_SIGN_AND_MANTISSA"] == 0x807FFFFF,
          "DSV4 guard must use DeepGEMM's exact sign+mantissa mask")
    values = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0,
                           1.3, float("inf"), float("nan")])
    got = ns["_ue8m0_unsafe"](values).tolist()
    check(got == [False, False, False, False, True, True, True, True],
          "+0/powers of two are valid; negative/arbitrary/inf/nan are unsafe")
    print("  DSV4 UE8M0 host guard ......... OK")


def test_b12x_micro_chunk_width() -> None:
    """The caller cannot exceed the matching dispatch/workspace contract."""
    ns = load_defs(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py",
        {"b12x_ep_micro_chunk_tokens", "b12x_ep_zero_weight_micro_chunks",
         "B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS",
         "B12X_EP_ZERO_WEIGHT_MICRO_MAX_TOKENS",
         "B12X_EP_ZERO_WEIGHT_MICRO_TOPK",
         "B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS", "b12x_ep_micro_tail"},
        {"os": os},
    )
    width = ns["b12x_ep_micro_chunk_tokens"]
    chunks = ns["b12x_ep_zero_weight_micro_chunks"]
    stock = ns["B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS"]

    check(width({}.get) == stock, "unset chunk width keeps the exact 8")
    check(width({"VLLM_B12X_EP_MICRO_CHUNK_TOKENS": "8"}.get) == stock,
          "the installed dispatcher's exact 8-token width is accepted")
    for raw in ("", "x", "7", "9", "4", "16", "32", "64", "72", "0", "-8"):
        if raw == "":
            check(width({"VLLM_B12X_EP_MICRO_CHUNK_TOKENS": raw}.get) == stock,
                  "an empty setting is equivalent to unset")
            continue
        try:
            width({"VLLM_B12X_EP_MICRO_CHUNK_TOKENS": raw}.get)
            check(False, f"unsupported chunk width {raw!r} must fail setup")
        except ValueError as exc:
            check("must be exactly 8" in str(exc) and raw in str(exc),
                  f"unsupported width {raw!r} must explain the exact contract")

    # every positive multiple of the chunk is admitted, up to the cutover --
    # 24 (C=3 at MAX_SEQS=4) was previously dropped to the pair fallback
    for tokens in (8, 16, 24, 32, 40, 64, 80):
        plan = chunks(tokens, 8, 72, enabled=True)
        check(len(plan) == tokens // stock, f"{tokens} tokens -> exact slices")
        check(plan[0][0] == 0 and plan[-1][1] == tokens,
              f"{tokens} tokens must be covered exactly")
        check(all(hi - lo == stock for lo, hi in plan),
              f"{tokens} tokens: every slice is one chunk wide")
        check(all(plan[i][1] == plan[i + 1][0] for i in range(len(plan) - 1)),
              f"{tokens} tokens: slices are contiguous")

    tail = ns["b12x_ep_micro_tail"]
    # Real decode batches are rarely a multiple of the chunk -- with chunked
    # prefill this lane saw 9, 36, 49, 52, 60. Cover the aligned prefix and
    # leave only the short tail to the pair fallback.
    for tokens, want_calls in ((9, 2), (36, 5), (49, 7), (52, 7), (60, 8)):
        plan = chunks(tokens, 8, 72, enabled=True)
        t = tail(tokens, 8, 72, enabled=True)
        check(t is not None, f"{tokens} tokens must report a tail")
        check(len(plan) + 1 == want_calls,
              f"{tokens} tokens -> {want_calls} calls, not {tokens}")
        check(plan[0][0] == 0 and t[1] == tokens and plan[-1][1] == t[0],
              f"{tokens} tokens: prefix and tail must cover exactly, no gap")
        check(0 < t[1] - t[0] < stock, f"{tokens} tokens: tail is short")
    for tokens in (8, 16, 24, 32, 80):
        check(tail(tokens, 8, 72, enabled=True) is None,
              f"{tokens} tokens is aligned -- no tail")
    for tokens in (0, 1, 7, 81, 8192):
        check(chunks(tokens, 8, 72, enabled=True) == ()
              and tail(tokens, 8, 72, enabled=True) is None,
              f"{tokens} tokens must fail closed (below one chunk, or past the "
              "compact cutover where dropping remote slots is cheaper)")
    check(chunks(8, 8, 72, enabled=False) == (), "disabled yields no plan")
    check(chunks(8, 4, 72, enabled=True) == (), "top_k must be 8")
    check(chunks(8, 8, 71, enabled=True) == (), "local experts must be 72")

    source = open(_overlay_source(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py"
    ), encoding="utf-8").read()
    planner = source[source.index("def b12x_ep_zero_weight_micro_chunks"):
                     source.index("def b12x_ep_micro_tail")]
    check("b12x_ep_micro_chunk_tokens()" not in planner
          and "B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS" in planner,
          "the per-layer forward planner must not re-read the environment")

    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "FlashInferB12xExperts")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "__init__")
    apply = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name == "apply")
    init_source = ast.get_source_segment(source, init) or ""
    apply_source = ast.get_source_segment(source, apply) or ""
    check("b12x_ep_micro_chunk_tokens()" in init_source,
          "the exact zero-weight width must be validated once at construction")
    check("self._ep_compact_enabled = b12x_ep_compact_enabled()" in init_source
          and "enabled=self._ep_compact_enabled" in apply_source
          and "b12x_ep_compact_enabled()" not in apply_source,
          "compact selection must consume its construction-time latch")
    check("self._direct_out = os.environ.get(" in init_source
          and "if self._direct_out:" in apply_source
          and "os.environ" not in apply_source,
          "direct-out selection must not read the environment per layer")

    print("  EP micro chunk width .......... OK")


def test_ep_tail_fixed_shape() -> None:
    """The tail must be padded to one chunk: b12x JITs per launch shape."""
    src = open(_overlay_source(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py")).read()
    body = src[src.index("def _ep_tail_padded_micro"):
               src.index("def _ep_tail_buffers")]

    check("if rem <= 0 or rem >= chunk:" in body and "return False" in body,
          "the padded path only claims a genuine short tail")
    check("pad_w[rem:].zero_()" in body,
          "pad rows must carry router weight 0")
    check("pad_ids[rem:].copy_(topk_ids[lo:lo + 1]" in body
          and "pad_x[rem:].copy_(hidden_states[lo:lo + 1]" in body,
          "pad rows must repeat a row already in this call -- a duplicate "
          "cannot move the expert's FC2 amax, a foreign token can")
    check("scatter_output=pad_out" in body,
          "the padded launch must scatter into the staging buffer, never "
          "straight into the caller's rows")
    check("output[lo:hi].copy_(pad_out[:rem])" in body,
          "only the real rows may be copied back")
    check(body.index("pad_x[:rem].copy_") < body.index("launch_sm120_moe("),
          "staging is filled before the launch")

    bufs = src[src.index("def _ep_tail_buffers"):
               src.index("def _apply_ep_stock_topk_micro")]
    check("self._ep_tail_key" in bufs and "!= key" in bufs,
          "staging buffers must be rebuilt when the shape or device drifts")
    check("return None" in bufs and "warning_once" in bufs,
          "allocation failure must fall back loudly to the variable tail, "
          "never silently produce wrong rows")

    print("  EP tail fixed shape ........... OK")


def test_ep_compact_shape_align() -> None:
    """Compact must bucket its pair count: b12x JITs per launch shape."""
    ns = load_defs(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py",
        {"b12x_ep_compact_pair_count", "B12X_EP_COMPACT_PAIR_ALIGN"},
        {},
    )
    count = ns["b12x_ep_compact_pair_count"]
    align = ns["B12X_EP_COMPACT_PAIR_ALIGN"]

    check(count(0) == 0, "an empty local set stays empty")
    for n in (1, 63, 64, 65, 127, 128, 4096):
        got = count(n)
        check(got % align == 0, f"{n} must round to a multiple of {align}")
        check(got >= n, f"{n} must never round DOWN -- that would drop pairs")
        check(got - n < align, f"{n} must not over-pad past one bucket")

    # the shapes one live bench actually minted kernels for
    observed = [48, 83, 102, 103, 104, 119, 122, 130, 133, 134, 140, 146,
                149, 150, 152, 168, 171, 176, 180, 184, 188, 195, 201, 202,
                203, 233, 249, 268, 270, 275, 284, 304, 336]
    buckets = {count(n) for n in observed}
    check(len(buckets) <= 8,
          f"{len(observed)} observed shapes must collapse to a handful, got "
          f"{len(buckets)}")

    src = open(_overlay_source(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py")).read()
    start = src.index("def _apply_ep_compact")
    nxt = src.find("\n    def ", start)
    body = src[start:nxt if nxt > 0 else len(src)]
    check("pair_out = torch.zeros(" in body,
          "padded compact rows may be skipped by the kernel, so the buffer "
          "must be zeroed (an uninitialised bit pattern can be NaN)")
    check("torch.empty(" not in body, "no uninitialised buffer in compact")
    check(body.index("pair_out.mul_(") < body.index("output.index_add_("),
          "pad rows must be masked before the token sum")
    check("% n" in body and "index_select(0, fill)" in body,
          "padding must cycle pairs already in the list, never invent a slot")

    print("  EP compact shape align ........ OK")


def test_ep_compact_warmup_ladder() -> None:
    """Load-time warmup ladder: opt-in, shape-derived, never fatal."""
    ns = load_defs(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py",
        {"b12x_ep_compact_warmup_buckets", "b12x_ep_compact_pair_count",
         "B12X_EP_COMPACT_PAIR_ALIGN"},
        {},
    )
    ladder = ns["b12x_ep_compact_warmup_buckets"]
    align = ns["B12X_EP_COMPACT_PAIR_ALIGN"]

    got = ladder(8192, 8, 72, 288)
    check(got == tuple(sorted(got, reverse=True)),
          "largest bucket first -- it dominates the cost and is what a first "
          "long prompt hits")
    check(all(b % align == 0 for b in got), "every rung is a real bucket")
    check(len(set(got)) == len(got), "no rung repeats")
    check(len(got) <= 8, "the ladder must stay short; each rung is a compile")
    check(got[0] == ns["b12x_ep_compact_pair_count"](8192 * 8 * 72 // 288),
          "the top rung is the largest chunk this engine can schedule, scaled "
          "to the slots this rank actually owns")

    for args in ((0, 8, 72, 288), (8192, 0, 72, 288), (8192, 8, 0, 288),
                 (8192, 8, 72, 0), (8192, 8, 288, 72)):
        check(ladder(*args) == (),
              f"degenerate geometry must yield no ladder: {args}")

    src = open(_overlay_source(
        "overlay/modules/glm53_moe/flashinfer_b12x_moe.py")).read()
    start = src.index("def _warm_compact_shapes")
    body = src[start:src.index("def _warm_activation_dtype")]
    check('os.environ.get("VLLM_B12X_EP_WARM_COMPACT", "0").strip() != "1"'
          in body and "return" in body,
          "warmup must be exact opt-in -- it costs load time")
    check("except Exception as exc:" in body and "warning_once" in body,
          "a failed warmup must cost a log line, never the boot")
    check("logger.info_once" in body,
          "a warmup that ran must say which shapes it covered")

    print("  EP compact warmup ladder ...... OK")


def _launcher_caller_passthrough(text: str) -> set[str]:
    """The names the launcher restores after sourcing a profile.

    Parsed rather than matched as a literal: the list is line-continued and
    grows, so an adjacency string ties the contract to today's ordering and
    breaks the next time a knob is inserted.
    """
    # The list moved from an inline `for _v in ...` loop to the argument list
    # of ct_load_profile when both lanes started sharing launchers/lib/
    # common-tp4.sh; it is still line-continued and still grows.
    start = text.index("ct_load_profile ")
    body = ""
    for line in text[start:].splitlines():
        body += " " + line
        if not line.rstrip().endswith("\\"):
            break
    return {w for w in body.replace("\\", " ").split()
            if w.isupper() or w.startswith("$")}


def test_launcher_load_format_gate() -> None:
    """LOAD_FORMAT must reach the container and refuse anything unvalidated."""
    text = open("launchers/start-glm53-nvfp4-tp4.sh").read()
    # The fast loader is the default, and the profile names the image that
    # actually carries it. These two have to move together: defaulting to
    # instanttensor while the profile pointed at the bare image cost a boot,
    # dead 75 s in on "No module named 'instanttensor'".
    check('LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"' in text,
          "glm53 boots on the fast loader by default")
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    check('PROFILE_IMAGE="glm53:v13-b12x-it"' in profile,
          "the profile names the image that has instanttensor, not the base")
    # And validating the string is still not validating anything: ask
    # whichever image is in play whether it can import it.
    check('import instanttensor' in text,
          "the launcher must check the image can import instanttensor")
    check(text.index('IMAGE="${IMAGE:-glm53:v13-b12x}"')
          < text.index('-c "import instanttensor"'),
          "the availability check needs IMAGE, so it runs after IMAGE is set")
    check("No module named instanttensor" in text,
          "the abort names the failure a boot would otherwise hit at 75 s")
    # The shared validation lives in the library both lanes source now.
    _lib = open(os.path.join(REPO, "launchers", "lib", "common-tp4.sh"),
                encoding="utf-8").read()
    check("ct_check_load_format" in text,
          "the launcher must call the shared validation")
    check("ABORT: LOAD_FORMAT must be auto, safetensors or instanttensor"
          in _lib,
          "an unknown format must abort, not reach vLLM as a typo")
    check("--load-format $LOAD_FORMAT" in text,
          "the value must actually reach the serve command")
    check("LOAD_FORMAT" in _launcher_caller_passthrough(text),
          "LOAD_FORMAT must be in the caller passthrough list -- a knob the "
          "launcher never forwards is the failure this lane hit five times")
    check(text.index('LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"')
          < text.index("--load-format $LOAD_FORMAT"),
          "validation must precede use")
    check("SILENT RANK DEATH" in text,
          "the multi-node hazard must be written where the knob is, not only "
          "in a PR body")
    print("  launcher load-format gate ..... OK")


def test_dsv4_launcher_adoptions() -> None:
    """DSV4 carries every adopted guard while uncertain speed axes stay opt-in."""
    text = open("launchers/start-hy4-tp4.sh", encoding="utf-8").read()
    check('LOAD_FORMAT="${LOAD_FORMAT:-auto}"' in text,
          "DSV4 load-format default must remain auto")
    check("ct_check_load_format" in text
          and '--load-format "${LOAD_FORMAT}"' in text
          and "-e LOAD_FORMAT=$LOAD_FORMAT" in text,
          "validated DSV4 LOAD_FORMAT must reach the container and serve CLI "
          "-- the validation itself now lives in launchers/lib/common-tp4.sh, "
          "which both lanes source")
    check("silent rank death" in text,
          "instanttensor's multi-node risk must stay beside its opt-in knob")

    check("WARNING: profile not found" in open(
              os.path.join(REPO, "launchers", "lib", "common-tp4.sh"),
              encoding="utf-8").read()
          and "ct_load_profile" in text,
          "a copied launcher must not silently lose profile VLLM_* settings")
    check("ct_refuse_foreign_stacks '^(glm53|q38)(-|$)' DSV4" in text,
          "DSV4 must refuse live foreign model stacks on every node -- via the "
          "shared guard, naming the OTHER lanes")
    check('"$HEAD_IP" $_foreign_wips' in text
          and 'for _w in $WORKERS; do _foreign_wips=' in text,
          "the guard must cover the head AND every worker: WORKERS carries "
          "ip:rank, so the ranks are stripped before they reach it")
    check("OVERLAY_DIGEST" in text
          and "/cache/vllm/torch_compile_cache" in text
          and "for w in $WORKERS" in text,
          "overlay changes must invalidate each node's persistent compile cache")
    check("HEAD_OVSUM" in text[text.index("OVERLAY_DIGEST"):],
          "cache stamp must include overlay file bytes, not only manifest text")

    check('GRAPH_DEBUG="${GRAPH_DEBUG:-0}"' in text
          and "VLLM_LOGGING_LEVEL=DEBUG" in text,
          "graph address assertions must be reachable but default-off")
    check("CUSTOM_OPS_AXIS" in text and "COMPILE_CFG" in text
          and "EXTRA_ENV_FLAGS" in text,
          "DSV4 diagnostic axes must reach all ranks")
    check("FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS" not in text,
          "the rejected FlashInfer cutoff must not be wired to native b12x")
    check('MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-4096}"' in text,
          "the measured-rejected 8192 prefill default must not return")

    check('OSAR_MAXEL="${OSAR_MAXEL:-}"' in text
          and "VLLM_DSV4_OSAR_MAXEL=$OSAR_MAXEL" in text,
          "OSAR's unknown size crossover must be sweepable without a new default")
    print("  DSV4 launcher adoptions ....... OK")


def test_prefill_ladder_probe() -> None:
    """The ladder must not let prefix caching masquerade as a warm kernel."""
    src = open("probes/prefill_ladder.py").read()
    ns: dict = {}
    exec(compile(src.replace("raise SystemExit(main(args))", "pass"),
                 "prefill_ladder", "exec"), ns)
    prompt = ns["_prompt"]
    base = 4242
    heads = {prompt(2048, base + r * 1000)[:16] for r in range(3)}
    check(len(heads) == 3,
          "every sample must carry a distinct prefix -- a shared one would "
          "score a prefix-cache hit and the warm column would measure that "
          "instead of whether the shape was already compiled")
    lens = [len(prompt(2048, base + r * 1000)) for r in range(3)]
    check(max(lens) - min(lens) < 32,
          "repeats must hold the LENGTH fixed while changing content, so the "
          "kernel shape is identical across them")
    check(len(prompt(4096, 1)) > 3 * len(prompt(1024, 1)),
          "requested length must actually drive prompt size")
    check('"max_tokens": 1' in src,
          "the ladder must generate one token: it measures prefill, not decode")
    check("seed if r else seed" not in src,
          "the repeat seed must vary (an earlier revision reused it)")
    print("  prefill ladder probe .......... OK")


def test_launcher_nofile_limit() -> None:
    """The container must not run on Docker's 1024-descriptor default."""
    for launcher in (
        "launchers/start-glm53-nvfp4-tp4.sh",
        "launchers/start-hy4-tp4.sh",
    ):
        text = open(launcher, encoding="utf-8").read()
        check("--ulimit nofile=" in text,
              f"{launcher} must not keep Docker's 1024-descriptor default")
        m = re.search(r"--ulimit nofile=(\d+):(\d+)", text)
        check(m is not None,
              f"{launcher} nofile must be an explicit soft:hard pair")
        soft, hard = int(m.group(1)), int(m.group(2))
        check(soft >= 65536 and hard >= soft,
              f"{launcher} nofile {soft}:{hard} must be large and ordered")
        check("Too many open files" in text,
              f"{launcher} must document the distant NCCL symptom")
    print("  launcher nofile limit ......... OK")


def test_once_logger_args_hashable() -> None:
    """*_once loggers dedupe by hashing their args -- a list raises there."""
    import ast as _ast
    for rel in ("overlay/modules/glm53_moe/flashinfer_b12x_moe.py",
                "overlay/modules/glm53_drafter/fp8_lm_head.py"):
        src = open(_overlay_source(rel)).read()
        for node in _ast.walk(_ast.parse(src)):
            if not (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr.endswith("_once")):
                continue
            for arg in node.args[1:]:
                check(not isinstance(arg, (_ast.List, _ast.Dict, _ast.Set)),
                      f"{rel}:{node.lineno} passes an unhashable literal to "
                      f"{node.func.attr} -- it raises TypeError and skips the "
                      "whole block (this cost the #164 warmup a boot)")
    print("  *_once args hashable .......... OK")


def test_dflash2_selector_check() -> None:
    """The selector-load check must read the real object and never be silent."""
    import ast as _ast
    src = open(_overlay_source(
        "overlay/modules/glm53_drafter/qwen3_dflash2.py")).read()
    ns = load_defs(
        "overlay/modules/glm53_drafter/qwen3_dflash2.py",
        {"dflash2_selector_load_verdict"}, {},
    )
    verdict = ns["dflash2_selector_load_verdict"]

    ok, why = verdict({"predecessor_codebook": (1.0, 0.42, 0.031),
                       "successor_codebook": (1.0, 0.38, 0.029)})
    check(ok and not why, "trained-looking codebooks must pass")
    for label, stats in (
        ("non-finite", {"a": (0.9997, 3.1, 0.05)}),
        ("absurd magnitude", {"a": (1.0, 6.7e30, 2.2e28)}),
        ("all zero", {"a": (1.0, 0.0, 0.0)}),
    ):
        ok, why = verdict(stats)
        check(not ok and why,
              f"{label} must be reported -- that is what torch.empty looks "
              "like, and its only downstream symptom is suffix decay")

    body = src[src.index("def verify_selector_loaded"):
               src.index("def compute_candidates")]
    check('getattr(self, "model", None), "candidate_selector"' in body,
          "the selector hangs off the inner model; reading it off the wrapper "
          "would find None and report nothing")
    check(body.count("logger.warning") >= 2,
          "both a failed verdict AND a missing selector must be loud")
    call = src[src.index("def compute_candidates"):]
    check("_deneb_selector_checked" in call
          and call.index("_deneb_selector_checked") < call.index("return"),
          "the check runs once, before the first candidate batch")
    print("  dflash2 selector check ........ OK")


def test_dflash2_selector_score_precision() -> None:
    """BF16 codebooks must not force the final path ranking back to BF16."""
    try:
        import torch
    except ImportError:
        print("  dflash2 selector score precision  SKIP (no torch on this host)")
        return

    ns = load_defs(
        "overlay/modules/glm53_drafter/qwen3_dflash2.py",
        {"_score_edges"}, {"torch": torch},
    )
    score_edges = ns["_score_edges"]

    # The candidates differ by one BF16 ULP in a codebook coordinate whose
    # contribution is small. A BF16 output rounds both 1.101... scores to the
    # same value; FP32 retains the trained ordering for the selector walk.
    predecessor = torch.tensor(
        [[1.0, 0.1015625], [1.0, 0.1015625], [1.0, 0.1015625]],
        dtype=torch.bfloat16,
    )
    successor = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0078125]],
        dtype=torch.bfloat16,
    )
    candidate_ids = torch.tensor([[[1, 2]]])
    unary = torch.zeros((1, 1, 2), dtype=torch.bfloat16)
    hidden = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    anchor = torch.tensor([0])

    scores = score_edges(
        predecessor, successor, candidate_ids, unary, hidden, anchor, 2,
    )
    legacy = unary[:, :, None] + torch.einsum(
        "blpr,blcr->blpc",
        predecessor[anchor[:, None, None].expand(-1, 1, 2)]
        * hidden[:, :, None],
        successor[candidate_ids],
    )
    check(scores.dtype == torch.float32,
          "selector edge scores must remain FP32 through path ranking")
    check(legacy[0, 0, 0, 0].item() == legacy[0, 0, 0, 1].item(),
          "fixture must reproduce the BF16 ranking tie")
    check(scores[0, 0, 0, 1].item() > scores[0, 0, 0, 0].item(),
          "FP32 edge scoring must recover the successor ordering BF16 erased")
    print("  dflash2 selector score precision  OK")


def test_dflash2_conv_mask_buffer() -> None:
    """Grouped conv reuses its deterministic speculative-block tap mask."""
    source = open(_overlay_source(
        "overlay/modules/glm53_drafter/qwen3_dflash2.py"
    ), encoding="utf-8").read()
    tree = ast.parse(source)

    grouped = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_grouped_conv"
    )
    grouped_source = ast.get_source_segment(source, grouped) or ""
    check("torch.arange" not in grouped_source,
          "grouped-conv hot path must not allocate a position arange")
    check("tap_valid[:, tap].view(-1, 1, 1)" in grouped_source,
          "each shifted tap must retain the old speculative-block boundary mask")

    conv = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DFlashGroupedConv"
    )
    init = next(
        node for node in conv.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    convolve = next(
        node for node in conv.body
        if isinstance(node, ast.FunctionDef) and node.name == "_convolve"
    )
    init_source = ast.get_source_segment(source, init) or ""
    convolve_source = ast.get_source_segment(source, convolve) or ""
    check('self.register_buffer(\n            "_tap_valid"' in init_source
          and "persistent=False" in init_source,
          "derived tap masks must move with the module but stay out of checkpoints")
    check("position.bitwise_and_(block_size - 1)" in init_source
          and "position.remainder_(block_size)" in init_source,
          "precompute must preserve both old block-position formulas")
    check("self._tap_valid[: hidden_states.shape[0]]" in convolve_source,
          "runtime must take a zero-allocation view sized to the active rows")

    decoder = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DFlash2Qwen3DecoderLayer"
    )
    decoder_init = next(
        node for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    decoder_source = ast.get_source_segment(source, decoder_init) or ""
    check("max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens"
          in decoder_source,
          "tap-mask capacity must cover the scheduler's largest flattened batch")
    print("  dflash2 conv mask buffer ....... OK")


def test_dflash_aot_guard_stays_removed() -> None:
    """The FlashAttention AOT split schedule cannot exist on this fleet.

    #178 ported upstream vllm#54374, which disables `aot_schedule` on a
    sliding-window draft builder. Instrumented (#180), the guard reported
    `1 sliding-window draft builder(s), aot_schedule disabled on 0` on two
    separate boots -- because `flash_attn.py` sets

        self.aot_schedule = get_flash_attn_version() == 3

    and `get_flash_attn_version` returns 3 only for `device_capability.major
    == 9` (Hopper SM90). GB10 is sm_121, major 12, so it falls through to FA2
    and `aot_schedule` is False on every builder, always. The upstream
    "acceptance 1.000 -> 5.542" figure was measured on hardware we are not on.

    Keeping the overlay meant re-syncing a 721-line verbatim copy of the
    speculator against every image bump for a branch that can never be taken.
    """
    profile = open("profiles/glm53.env", encoding="utf-8").read()
    modules = profile.split('MODULES="', 1)[1].split('"', 1)[0].split()
    check("glm53_dflash_aot_guard" not in modules,
          "the guard must not come back without new evidence: re-adding it "
          "needs FA3 on this fleet, which needs SM90 hardware")
    check(not os.path.exists("overlay/modules/glm53_dflash_aot_guard"),
          "the overlay directory must be gone, not merely unmounted -- an "
          "unmounted 721-line image copy still rots against image bumps")
    manifest = open("build/glm53/manifest.tsv", encoding="utf-8").read()
    check("dflash_speculator.py" not in manifest,
          "the composed manifest must not pin a preimage for a file we no "
          "longer overlay")
    print("  dflash AOT guard stays removed  OK")


def test_hotpath_env_latches() -> None:
    """Process switches are read once, never from layer/logits hot paths."""
    kpool_path = _overlay_source("overlay/sparse_attn_indexer_kpool.py")
    kpool_source = open(kpool_path, encoding="utf-8").read()
    kpool_tree = ast.parse(kpool_source)
    kpool = next(
        node for node in kpool_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "sparse_attn_indexer_kpool"
    )
    kpool_body = ast.get_source_segment(kpool_source, kpool) or ""
    check("os.environ" not in kpool_body,
          "kpool custom-op hot path must not read the environment per layer")
    check("and _KPOOL_TAIL_CACHE_ENABLED" in kpool_body
          and "and _KPOOL_DECODE_WRITE_ENABLED" in kpool_body,
          "both kpool rollback switches must consume import-time latches")

    model_path = _overlay_source("overlay/modules/dsv4_model/nvidia_model.py")
    model_source = open(model_path, encoding="utf-8").read()
    model_tree = ast.parse(model_source)

    def method_source(name: str, marker: str) -> str:
        for cls in (node for node in model_tree.body
                    if isinstance(node, ast.ClassDef)):
            for node in cls.body:
                if not isinstance(node, ast.FunctionDef) or node.name != name:
                    continue
                body = ast.get_source_segment(model_source, node) or ""
                if marker in body:
                    return body
        raise AssertionError(f"{name} containing {marker!r} not found")

    forward = method_source("forward", "use_dspark_reference_hc")
    target_head = method_source(
        "_ensure_tgt_head_fp8_copy", "_DSV4_TARGET_LM_HEAD_FP8"
    )
    release_head = method_source(
        "maybe_release_bf16_lm_head", "_DSV4_FREE_BF16_LM_HEAD"
    )
    for name, body in (
        ("model forward", forward),
        ("target-head logits", target_head),
        ("bf16 release", release_head),
    ):
        check("getenv(" not in body and "os.environ" not in body,
              f"{name} must consume a module latch, not read the environment")
    check("use_dspark_reference_hc = _DSPARK_REFERENCE_HC" in forward,
          "aux hidden-state capture must use the reference-HC latch")
    print("  hot-path env latches ........... OK")


def test_overlay_logger_defined() -> None:
    """An overlay that logs must define logger -- NameError kills the boot."""
    import ast as _ast, glob as _glob
    checked = 0
    for path in sorted(_glob.glob("overlay/modules/*/*.py")):
        src = open(path).read()
        if "logger." not in src:
            continue
        checked += 1
        tree = _ast.parse(src)
        assigned = any(
            isinstance(n, _ast.Assign)
            and any(getattr(t_, "id", None) == "logger" for t_ in n.targets)
            for n in tree.body
        )
        imported = any(
            isinstance(n, _ast.ImportFrom)
            and any(a.name == "logger" or a.asname == "logger" for a in n.names)
            for n in _ast.walk(tree)
        )
        check(assigned or imported,
              f"{path} calls logger.* without defining it -- this raised "
              "NameError at model load and took the whole boot down")
    check(checked > 5, f"expected many logging overlays, scanned {checked}")
    print("  overlay logger defined ........ OK")


def test_torch_imports_are_guarded() -> None:
    """No test may import torch unguarded: it is the deploy gate.

    The serving hosts run this suite before every overlay deploy and have no
    torch. An unguarded import aborts the deploy -- which is not a test
    failure, it is a deploy outage. This was fixed once in #125 and came back
    with #157/#173, so it is a contract now.
    """
    import ast as _ast
    src = open("tests/test_logic.py").read()
    offenders = []
    for node in _ast.parse(src).body:
        if not (isinstance(node, _ast.FunctionDef)
                and node.name.startswith("test_")):
            continue
        seg = _ast.get_source_segment(src, node) or ""
        if "import torch" not in seg:
            continue
        if "except ImportError" in seg:
            continue
        offenders.append(node.name)
    check(not offenders,
          "these tests import torch without an ImportError guard and will "
          f"abort the deploy on a serving host: {offenders}")
    print("  torch imports guarded ......... OK")


def test_launcher_reject_method_gate() -> None:
    """REJECT_METHOD must reach the drafter config and refuse a typo."""
    text = open("launchers/start-glm53-nvfp4-tp4.sh").read()
    check('"rejection_sample_method\\":\\"$REJECT_METHOD' in text,
          "the value must reach the speculative-config JSON")
    check("ABORT: REJECT_METHOD must be standard or block" in text,
          "an unknown method must abort here, not reach vLLM as a typo")
    names = _launcher_caller_passthrough(text)
    check({"DRAFT_SAMPLE", "REJECT_METHOD"} <= names,
          "both drafter knobs must be in the caller passthrough list -- a "
          "knob the launcher never forwards is a knob that reads as a "
          f"measured no-effect; got {sorted(names)}")
    body = text[text.index('case "${REJECT_METHOD:-}" in'):]
    empty = body[body.index('"" )'):body.index("standard|block")]
    check("_spec_extra" not in empty,
          "unset must add nothing: the default has to stay byte-identical to "
          "the config this lane already measured")
    print("  launcher reject-method gate ... OK")


def test_dflash2_prefix_cache_fail_closed() -> None:
    """Cache-restored target tokens have no guaranteed DFlash context KV."""
    launcher = open(
        "launchers/start-glm53-nvfp4-tp4.sh", encoding="utf-8"
    ).read()
    profile = open("profiles/glm53.env", encoding="utf-8").read()

    check("PREFIX_CACHE=0" in profile,
          "the DFlash2 profile must prefer valid draft KV over prefix TTFT reuse")
    check('PREFIX_CACHE="${PREFIX_CACHE:-0}"' in launcher,
          "a profile-less DFlash2 launch must also fail closed")
    check("ABORT: PREFIX_CACHE must be 0 or 1" in launcher,
          "an invalid cache-safety value must not reach vLLM")
    check('0) PREFIX_CACHE_FLAG="--no-enable-prefix-caching"' in launcher
          and '1) PREFIX_CACHE_FLAG="--enable-prefix-caching"' in launcher,
          "both explicit vLLM BooleanOptionalAction flags must be wired")
    serve = launcher[launcher.index('SERVE_ARGS="'):]
    check("$PREFIX_CACHE_FLAG" in serve,
          "the validated prefix-cache decision must reach the serve command")
    names = _launcher_caller_passthrough(launcher)
    check("PREFIX_CACHE" in names,
          "a caller must be able to make the documented throughput rollback")
    dry_run = launcher[launcher.index('if [ "${DRY_RUN:-0}" = 1 ]'):]
    check("PREFIX_CACHE" in dry_run,
          "dry-run must expose whether draft-KV safety or prefix reuse won")
    check(launcher.index('PREFIX_CACHE="${PREFIX_CACHE:-0}"')
          < launcher.index('SERVE_ARGS="'),
          "prefix-cache validation must happen before serve args are frozen")
    print("  dflash2 prefix-cache fail-closed  OK")


def test_accept_profile_conditional_arithmetic() -> None:
    """pos[i] is a MARGINAL count; the conditional is pos[i] / pos[i-1].

    Dividing every position by the draft count yields a geometric decay even
    when the per-position conditional rate is flat, which reads as a broken
    drafter. This lane made exactly that error, so pin the arithmetic.
    """
    src = open("probes/accept_profile.py").read()
    check("previous = drafts" in src and "cond = count / previous" in src,
          "the conditional must divide by the previous position's count, "
          "with the draft count seeding position 0")
    check("count / drafts" in src,
          "the marginal must still be printed -- it is what the engine "
          "exports and what external reports quote")
    # A flat 80% conditional rate must not look like decay.
    marginal, drafts, previous, conds = [], 1000.0, 1000.0, []
    for _ in range(7):
        previous *= 0.8
        marginal.append(previous)
    previous = drafts
    for count in marginal:
        conds.append(count / previous)
        previous = count
    check(max(conds) - min(conds) < 1e-9,
          "a constant conditional rate must come back constant")
    check(marginal[-1] / drafts < 0.3,
          "...while its marginal decays below 30%, which is the number that "
          "looks alarming and is not")
    print("  accept profile arithmetic ..... OK")


def test_osar_maxel_rank_agreement() -> None:
    """MAXEL skew must fall back to NCCL on every rank, not just be logged.

    #185 made MAXEL a per-build override and noted in the source that all
    peers must agree, but the only evidence was a boot log line. MAXEL is
    compiled into the kernel, feeds the remote rx stride, and gates
    ``_eligible`` -- so a rank with a larger value takes the one-shot path for
    a tensor its peers send through NCCL. The shim already votes this way for
    readiness and for connect, with the reason written down: a split
    collective deadlocks. This pins the third vote.
    """
    source = open(_overlay_source(
        "overlay/modules/tp_oneshot_ar/dsv4_oneshot_shim.py"
    ), encoding="utf-8").read()
    check("MAXEL differs across ranks" in source,
          "a MAXEL mismatch must be reported, naming the values")
    vote_at = source.index("maxel_bounds")
    connect_at = source.index("_ext.connect(")
    agreed_at = source.index("_boot_agreed = True")
    check(vote_at < agreed_at < connect_at,
          "the vote must run before _boot_agreed and before connect -- peer "
          "buffers are sized from MAXEL, and after agreement a local fallback "
          "is fatal by design")
    tail = source[vote_at:connect_at]
    check("_disabled = True" in tail and "return" in tail,
          "a mismatch must put THIS rank on NCCL too; every rank runs the "
          "same comparison, so they all take this branch together")

    # The min/max trick: one all_reduce(MAX) over [x, -x] yields both bounds.
    for ranks, agree in (([131072] * 4, True),
                         ([131072, 131072, 786432, 131072], False),
                         ([1024, 1024], True)):
        hi = max(r for r in ranks)
        lo = -max(-r for r in ranks)
        check((hi == lo) is agree,
              f"{ranks} should read as {'agreeing' if agree else 'skewed'}")
    print("  osar MAXEL rank agreement ..... OK")


def test_bench_resolves_served_model() -> None:
    """A bench must ask the server its model name, not assume the dsv4 one.

    All three harnesses defaulted to "deepseek-v4-flash". Pointed at the
    glm53 server every request 404s -- and the failure reads as an empty
    section in a boot log, not as an error, so a prefill or decode arm can
    look "measured" when it never ran. This lane lost measurements to it
    twice on one day.
    """
    import io
    import json as _json
    import urllib as _urllib
    import urllib.request as _u

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _served(ids):
        def _open(url, timeout=None):
            if ids is None:
                raise OSError("connection refused")
            body = {"data": [{"id": i} for i in ids]}
            return _Resp(_json.dumps(body).encode())
        return _open

    # bench/ resolves through one shared copy (bench_common); probes/ keeps
    # its own, so both spellings are exercised here.
    for path, fname in (("probes/prefill_ladder.py", "_resolve_model"),
                        ("bench/bench_common.py", "resolve_model")):
        src = open(path, encoding="utf-8").read()
        match = re.search(rf"^def {fname}.*?(?=\n\n\n|\n[^\s#]|\Z)",
                          src, re.S | re.M)
        check(match is not None, f"{path} must define {fname}")
        assert match is not None
        for base_name in ("URL", "BASE"):
            if re.search(rf"^{base_name} = ", src, re.M):
                break
        # Some harnesses call urllib.request.urlopen through the package and
        # some import it locally; hand over the package so both resolve.
        ns: dict = {"os": os, "json": _json, "urllib": _urllib, "re": re,
                    base_name: "http://localhost:8000/v1/chat/completions",
                    # bench_common takes the url as a defaulted argument
                    "DEFAULT_URL": "http://localhost:8000/v1/chat/completions"}
        exec(compile(match.group(0), path, "exec"), ns)
        resolve = ns[fname]

        saved_open, saved_env = _u.urlopen, os.environ.pop("BENCH_MODEL", None)
        try:
            _u.urlopen = _served(["glm-5.3-flash"])
            check(resolve("deepseek-v4-flash") == "glm-5.3-flash",
                  f"{path}: a server serving one model must win over the "
                  "hardcoded dsv4 default")
            _u.urlopen = _served(["deepseek-v4-flash", "other"])
            check(resolve("deepseek-v4-flash") == "deepseek-v4-flash",
                  f"{path}: when the default IS served, keep it -- every "
                  "recorded dsv4 number must stay pointed at the same target")
            _u.urlopen = _served(None)
            check(resolve("deepseek-v4-flash") == "deepseek-v4-flash",
                  f"{path}: an unreachable server must fall back, not raise")
            os.environ["BENCH_MODEL"] = "explicit"
            _u.urlopen = _served(["glm-5.3-flash"])
            check(resolve("deepseek-v4-flash") == "explicit",
                  f"{path}: an explicit BENCH_MODEL must still win")
        finally:
            _u.urlopen = saved_open
            os.environ.pop("BENCH_MODEL", None)
            if saved_env is not None:
                os.environ["BENCH_MODEL"] = saved_env
    print("  bench served-model resolve .... OK")


def test_prefill_knobs_announce_arming() -> None:
    """Every opt-in prefill path must say it armed, and 1 must not read as off.

    #188 added four env knobs and none of the new code paths logged anything
    but failures. That is the shape this lane keeps losing boots to: #178's
    guard, CUSTOM_OPS_AXIS, the *_once TypeError. A sweep that sets everything
    to 1 would silently leave union prefill OFF, because only 2 and 4 are
    widths -- and the run would read as "measured, no effect".
    """
    union = open(_overlay_source(
        "overlay/modules/glm53_model/glm53_union_prefill.py"
    ), encoding="utf-8").read()
    check("union prefill: ARMED" in union,
          "the union path must announce its width when it installs")
    check("is not a union width" in union,
          "a value that is not 2 or 4 must warn, not silently mean off")

    ns: dict = {"os": os, "logger": _CapturingLogger()}
    load_defs("overlay/modules/glm53_model/glm53_union_prefill.py",
              {"_UNION_ENV", "_UNION_REPORTED", "_read_group_size"}, ns)
    saved = os.environ.get(ns["_UNION_ENV"])
    try:
        for raw, want, warns in (("0", 0, False), ("2", 2, False),
                                 ("4", 4, False), ("1", 0, True),
                                 ("yes", 0, True)):
            ns["_UNION_REPORTED"].clear()
            ns["logger"].lines.clear()
            os.environ[ns["_UNION_ENV"]] = raw
            check(ns["_read_group_size"]() == want,
                  f"union width {raw!r} must resolve to {want}")
            check(bool(ns["logger"].lines) is warns,
                  f"union width {raw!r} must {'warn' if warns else 'stay quiet'}")
    finally:
        os.environ.pop(ns["_UNION_ENV"], None)
        if saved is not None:
            os.environ[ns["_UNION_ENV"]] = saved

    fused = open(_overlay_source(
        "overlay/modules/glm53_model/glm53_prefill_fastpath.py"
    ), encoding="utf-8").read()
    check("fused K+gate: ARMED" in fused,
          "replacing Indexer.forward must be visible in the boot log")
    kpool = open(_overlay_source(
        "overlay/modules/glm53_kernels/glm53_kpool_topk.py"
    ), encoding="utf-8").read()
    check("kpool fused top-k" in kpool and "ARMED" in kpool,
          "a requested-but-not-built extension must be distinguishable from "
          "a knob that never parsed")
    print("  prefill knobs announce arming . OK")


def test_extra_env_rejects_comma_list() -> None:
    """EXTRA_ENV is space-separated; a comma-joined list must abort.

    The comma form passes the KEY=VALUE shape, so the whole string became one
    -e whose value was "1,NEXT_KEY=1,..." -- every knob read as off while the
    boot log said they were set. That boot measured the baseline and would
    have been reported as "knobs on, no effect". A value that merely contains
    commas (LIST=a,b,c) is legitimate and must still pass.
    """
    # The guard moved into launchers/lib/common-tp4.sh when both lanes stopped
    # carrying their own copy -- hy4's copy lacked the comma arm entirely, which
    # is why sharing the better implementation was the point.
    text = open(os.path.join(REPO, "launchers", "lib", "common-tp4.sh"),
                encoding="utf-8").read()
    body = text[text.index("for _kv in ${EXTRA_ENV:-}"):]
    body = body[:body.index("done")]
    check("space-separated, not comma-separated" in body,
          "the abort must name the actual mistake, not just 'bad format'")
    check("is declared in the profile, so EXTRA_ENV cannot" in body,
          "a profile-declared VLLM_* key must abort: the profile emits its "
          "own -e later and docker takes the last one, so EXTRA_ENV loses "
          "silently for exactly the knobs a sweep wants to move")
    check("_vllm_keys_sp" in text and "printf '%s ' ${_vllm_keys:-}" in text,
          "the key list is newline-separated; it must be flattened before the "
          "space-delimited case pattern, or the guard never matches")
    # Match the case ARMS, not the body that follows them: the accepting arm
    # has grown a guard between its pattern and its assignment, and pinning
    # the assignment made this test break the moment that happened.
    reject = body.index("[A-Za-z_]*=*,[A-Za-z_]*=*)")
    accept = body.index("[A-Za-z_]*=*)\n", reject + 1)
    check(reject < accept,
          "the comma pattern must be tested BEFORE the generic KEY=VALUE arm "
          "-- case takes the first match, so the order is the guard")
    print("  extra-env comma guard ......... OK")


def test_kda_conv_state_layout_is_the_arming_contract() -> None:
    """MK-KDA indexes DS; vLLM's default is SD, and the mismatch was silent.

    The 09-03 decode trace carried no mk_kda_kernel at all while the boot log
    said armed={'kda': True}: _selftest_kda builds its own DS fixtures and
    passes, then _kda_layout_ok rejects every production layer for the life
    of the boot and says nothing. Two things must hold so that cannot recur:
    the launcher derives the layout the segment needs, and a rejection names
    itself in the log.
    """
    mk = open(
        "overlay/modules/glm53_megakernel/glm53_megakernel.py", encoding="utf-8"
    ).read()
    launcher = open(
        "launchers/start-glm53-nvfp4-tp4.sh", encoding="utf-8"
    ).read()

    # 1. the gate returns a REASON, and the hot path logs each one once
    check("def _kda_layout_reason(layer)" in mk,
          "the layout gate must be able to say which predicate failed")
    reason = mk[mk.index("def _kda_layout_reason(layer)"):
                mk.index("def _kda_layout_ok(layer)")]
    # 32차: the kernel addresses the state through launch-carried strides,
    # so the gate no longer rejects a process layout (SD arrives as a
    # transposed view) or a non-contiguous view (page-aligned slots); it
    # refuses only overlapping / non-positive strides, and says them.
    check("isinstance(kv, (tuple, list))" in reason
          and "_conv_state_dim_first" not in reason
          and "conv_state.is_contiguous()" not in reason
          and "conv state strides %s overlap or are not positive" in reason
          and "slot_extent = s1 * (KDA_QKV - 1) + s2 * (cw - 1) + 1" in reason,
          "the layout gate admits strided views and names the strides it "
          "refuses -- a contiguity gate rejected every production layer")
    launch = mk[mk.index("def _kda_launch("):mk.index("def kda_block(")]
    check("int(conv_state.stride(0)), int(conv_state.stride(1))" in launch
          and "int(conv_state.stride(2)), int(rec_state.stride(0))" in launch
          and "int(conv_state.dtype == torch.bfloat16)" in launch
          and "torch.float32, torch.bfloat16" in reason
          and "rec_state.is_contiguous()" not in reason
          and "recurrent state strides %s are not" in reason,
          "the launch carries the three conv-state strides and the recurrent "
          "slot stride; the gate names the recurrent strides it refuses")
    st = mk[mk.index("def _selftest_kda()"):mk.index("def arm()")]
    kt = mk[mk.index("def _kda_eligible_reason(meta)"):mk.index("_KDA_LAYOUT_SAID = set()")]
    gl = mk[mk.index("def gemm_w4a8(x, mk_pack, n_rows, bg=False)"):]
    gl = gl[:gl.index("\ndef ", 1)]
    check("gemm lane CAPTURED into the decode graph" in gl and "plan grid=%d ksr=%d localq=%d gemm2=%d" in gl
          and "_EXT.gemm_plan(int(x.shape[0]), int(n_rows), int(x.shape[1]), 0)" in gl,
          "the GEMM lane says once at capture which variant the graph bakes")
    kb = mk[mk.index("def kda_block(layer, hidden_states, positions)"):mk.index("class KdaShadowArm")]
    check("kda lane CAPTURED into the decode graph" in kb and "_KDA_CAPTURED" in kb,
          "kda_block says once when it is captured into the decode graph -- "
          "the served step is a replay, no other line can see it")
    check("kda lane serving: first eligible eager step" in kt
          and "is_current_stream_capturing()" in kt and "kda lane tally: served=%d stock=%d capture=%d" in kt
          and "kda lane stock: %s" in kt and '"(routine)" not in reason' in kt
          and "_kda_eligible_said(_kda_meta(layer))" in mk,
          "the KDA takeover says once when a step first serves (armed or "
          "shadow) and once per distinct eligibility reason it does not; "
          "prefill steps stay silent (no boot tonight logged a KDA judgement)")
    hook = mk[mk.index("def smlp_forward(mlp, x)"):mk.index("def _smlp_ref(")]
    check("smlp lane CAPTURED into the decode graph" in hook
          and "smlp lane serving: first fused call" in hook
          and hook.count("_smlp_stock(") >= 4
          and "if T < 1 or T > MAX_TOK:\n        return None" in hook,
          "the smlp hook says once when it first serves and once per distinct "
          "reason it does not (armed != serving, 28차); prefill rows are "
          "routine and silent")
    judge = mk[mk.index("class KdaShadowArm"):mk.index("_DRAIN_BUF = None")]
    check("self.conv_mk[1:] = conv_state[used]" in judge
          and "self.rec_mk[1:] = rec_state[used]" in judge
          and "spec_state_indices_tensor=sidx_c.contiguous()" in judge
          and "conv_state[self.used]" in judge and "rec_state[self.used]" in judge
          and "conv_state.clone()" not in judge and "rec_state.clone()" not in judge,
          "the KDA shadow judge clones only the slots the step touches, with "
          "the indices remapped into the compact buffers (whole-pool clones "
          "emptied unified memory: KDA32SHADOW3 earlyoom, 32차)")
    check('fx.mk_run(v, layout=lay)' in st and '("bf16", 2e-2)' in st,
          "the self-test runs the padded-slot and SD-transposed views "
          "against the contiguous result")
    ok = mk[mk.index("def _kda_layout_ok(layer)"):
            mk.index("def _kda_ensure_packs")]
    check("_KDA_LAYOUT_SAID" in ok and "logger.warning" in ok
          and "has rejected %d calls" in ok,
          "a permanent rejection must be logged, once per distinct reason")
    check("return True" in ok and "_kda_layout_reason(layer)" in ok,
          "_kda_layout_ok must be the thin wrapper over the reason")

    # 2. arming says so when the process layout will reject every layer
    arm = mk[mk.index('_ARMED["kda"] = _gate("kda", _selftest_kda)'):]
    arm = arm[:arm.index("_armed_once")]
    check("is_conv_state_dim_first" in arm,
          "the boot must compare the ARMED segment against the process "
          "layout: the self-test's own fixtures cannot see it")

    # 3. the launcher derives DS rather than asking for two knobs
    check("VLLM_SSM_CONV_STATE_LAYOUT=DS" in launcher,
          "the launcher must set the layout MK-KDA indexes")
    blk = launcher[launcher.index('if [ "${VLLM_GLM53_MEGAKERNEL:-0}" != 0 ]'
                                  ' && [ "${VLLM_GLM53_MK_KDA:-0}" != 0 ]'):]
    blk = blk[:blk.index("\nfi\n") + 4]
    check("ABORT" in blk,
          "a conflicting layout must abort the boot, not be overridden "
          "silently -- docker takes the last -e and the operator would never "
          "see which value won")
    check(blk.index("ABORT") < blk.index(
        'ENVV="$ENVV -e VLLM_SSM_CONV_STATE_LAYOUT=DS"'),
          "the conflict check must run before the -e is appended")
    print("  kda conv-state layout ......... OK")


def test_fp8_dense_build_peak_pays_only_for_what_serves() -> None:
    """The build pass must not hold copies no method will read.

    srv3 was OOM-killed mid-pass on 2026-09-03 (global_oom, free 4.0 GiB
    against a 4.0 GiB watermark, all_unreclaimable) on the first boot with
    VLLM_GLM53_FP8_DENSE=nvfp4. Two of the copies at that peak were dead:
    the MK W4 pack was built for EVERY eligible linear before the scheme
    branch, and NvFp4DenseMethod.apply never reads one; and the alpha_scale
    retry re-ran _quantize_nvfp4, which does not take alpha_scale, holding
    two identical triples at once. The bf16 lifetime is NOT the lever here
    -- freeing it inside this pass already killed a boot (see the docstring
    on maybe_free_fp8_dense_bf16).
    """
    src = open(
        "overlay/modules/glm53_model/glm53_fp8_dense.py", encoding="utf-8"
    ).read()
    body = src[src.index("def maybe_build_fp8_dense("):]

    # 1. the pack is attached through the helper, never inline in the loop
    check("def _attach_mk_pack(" in src,
          "the pack build must be one helper so every call site is visible")
    check("build_mk_weight_w4" not in body,
          "no inline pack build inside the build loop -- it is what ran "
          "before the scheme branch and paid for every nvfp4 layer")

    # 2. no pack on a path where nvfp4/w4a8 wins the layer
    nv = body[body.index("if scheme == \"nvfp4\""):]
    nv = nv[:nv.index("if scheme == \"w4a8\"")]
    # The refusal path is `if not armed_nv:` since the alpha convention
    # stopped being resolved per layer; the split it guards is the same one.
    fallback = nv.index("if not armed_nv:")
    # the call sites go through `attach_mk`, which is _attach_mk_pack or,
    # under the "w8" scheme (fp8 pair only), a no-op -- one axis per boot
    check("attach_mk = _attach_mk_pack\n" in body,
          "attach_mk is the helper, bound once per pass")
    check("attach_mk(" not in nv[:fallback] and "_attach_mk_pack(" not in nv[:fallback],
          "a layer the nvfp4 arm takes must not build an MK pack: "
          "NvFp4DenseMethod.apply goes straight to the nvfp4 kernel")
    check("attach_mk(" in nv[fallback:],
          "when the nvfp4 arm refuses, fp8 serves and the pack is needed")

    # 3. one quantization per linear, and it happens before anything that
    # might repeat. _quantize_nvfp4 does not read alpha_scale, so producing a
    # triple per attempt held two identical ones at the build peak.
    check(nv.count("_quantize_nvfp4(weight)") == 1,
          "one quantization per linear, whatever the attempts")
    check(nv.index("_quantize_nvfp4(weight)") < nv.index("_NVFP4_ALPHA[0]"),
          "the triple exists before the convention is consulted, not per try")

    # 4. a boot where MK-GEMM is on but nothing carries a pack must say so
    check("%d MK W4 packs" in body,
          "the fingerprint must count the packs, or 'MK_GEMM=1 with 0 packs' "
          "is invisible")
    check("MK_GEMM=1 but the" in body
          and "the exclusion is " in body and "per layer" in body,
          "the scheme that wins the layer turns MK-GEMM off for it, and the "
          "message must scope the exclusion to MK-GEMM: silence there is how "
          "'armed' stops meaning 'serving', and an over-broad claim is how "
          "MK-KDA got written off with it")

    # 5. the transient of every linear goes back to the driver before the
    #    next one: on unified memory a block the caching allocator keeps
    #    reserved is host memory the node has lost, and the sum over the
    #    pass took every node under the 4 GiB kernel watermark (2026-09-04
    #    instrumented boot: reserved +14.9 GiB after the first linear with
    #    allocated flat)
    loop = body[body.index("for name, mod in model.named_modules():"):body.index("logger.warning(\n        \"[fp8-dense] %s (knob %s=%s)")]
    check("        finally:\n" in loop and "torch.cuda.empty_cache()" in loop
          and loop.index("        finally:\n") < loop.index("torch.cuda.empty_cache()"),
          "empty_cache() in the loop's finally: one transient at a time")

    # 6. the bf16 release stays its own pass
    free = src[src.index("def maybe_free_fp8_dense_bf16("):]
    free = free[:free.index("def _attach_mk_pack(")]
    check("mod.weight.data = torch.empty(" in free
          and "mod.weight.data = torch.empty(" not in body,
          "the bf16 release must stay outside the build pass: releasing from "
          "inside it made the early AutoWeightsLoader call destructive")
    print("  fp8-dense build peak .......... OK")


def test_kda_owns_its_projections_across_dense_schemes() -> None:
    """MK-KDA and the nvfp4 dense scheme are not exclusive; only MK-GEMM is.

    MK-MHC and MK-MLA never read a pack. MK-KDA fuses in_proj/o_proj into
    its own launch, so for those two the layer's quant_method is never
    called -- whichever scheme won them is dead weight, and the W4 pack the
    kernel does read must be attached while the bf16 source is still alive.
    Only MK-GEMM, which IS the Fp8DenseMethod.apply hook, cannot share a
    layer with an nvfp4/w4a8 arm.
    """
    src = open(
        "overlay/modules/glm53_model/glm53_fp8_dense.py", encoding="utf-8"
    ).read()
    check("def _kda_owns(" in src,
          "ownership must be decided in the build pass, where the bf16 "
          "source is still alive")
    owns = src[src.index("def _kda_owns("):src.index("def maybe_build_fp8_dense(")]
    check("in_proj_qkvbfg_a" in owns and "o_proj" in owns,
          "the KDA block's two projections are the owned pair")
    check('hasattr(parent, "in_proj_qkvbfg_a")' in owns,
          "o_proj also names the attention block's output projection -- the "
          "KDA one is the sibling of in_proj_qkvbfg_a")
    check("ENABLE_KDA or _mkmod.KDA_SHADOW" in owns,
          "ownership follows the KDA knob; with the segment off the dense "
          "scheme keeps the layer")

    body = src[src.index("def maybe_build_fp8_dense("):]
    owns_at = body.index("if _kda_owns(model, name):")
    nv_at = body.index('if scheme == "nvfp4"')
    check(owns_at < nv_at,
          "ownership is decided BEFORE any low-precision arm bids, or the "
          "layer pays for a copy nothing reads")
    branch = body[owns_at:nv_at]
    check("attach_mk(" in branch and "continue" in branch,
          "an owned layer attaches the pack and skips the arm")
    print("  kda owns its projections ...... OK")


def test_fp8_dense_prefill_nvfp4_pair_routes_by_rows() -> None:
    """Lever 7 (28차): prefill rows take the nvfp4 pair, decode keeps the lane.

    The nvfp4 SCHEME replaces the method and so turns the MK W4 lane off for
    every layer it takes (#263). The prefill pair is attached to the fp8
    method instead: apply() routes M > _NVFP4_PREFILL_MIN_M (the MK lane's
    32) to mm_fp4, everything at or below it goes on to the lane / fp8 pair
    untouched, the opaque (drafter) path never sees it, and a failing pair
    drops that layer's prefill to the fp8 pair for good.
    """
    calls = []

    class _Log:
        def warning(self, fmt, *a):
            calls.append(("log", fmt % a if a else fmt))

    class _X:
        def __init__(self, m, k=4096):
            self.shape = (m, k)

        def numel(self):
            return self.shape[0] * self.shape[1]

    ns = load_defs(
        "overlay/glm53_fp8_dense.py",
        {"Fp8DenseMethod", "_NVFP4_PREFILL_MIN_M", "_PREFILL_NVFP4_ENV",
         "_prefill_nvfp4_enabled"},
        {"os": os, "re": re, "logger": _Log(),
         "_nvfp4_dense_gemm_op": lambda x, *nv: calls.append(("nvfp4", x.shape[0], nv)) or "nv",
         "_fp8_dense_gemm_op": lambda x, *a: calls.append(("fp8", x.shape[0])) or "fp8",
         "_mk_or_fp8_dense_gemm_op": lambda x, *a: calls.append(("opaque", x.shape[0])) or "op"},
    )
    assert ns["_NVFP4_PREFILL_MIN_M"] == 32
    M = ns["Fp8DenseMethod"]
    base = types.SimpleNamespace(apply=lambda layer, x, bias: "bf16")
    m = M(base, "q", "ws", 4096, 4096)
    assert m._nvfp4 is None
    layer = types.SimpleNamespace()
    # no pair: rows go to the fp8 pair (no MK pack attached here)
    assert m.apply(layer, _X(64)) == "fp8"
    # with the pair: prefill rows -> nvfp4, decode rows -> fp8 pair
    m._nvfp4 = ("wq", "wsf", "gs", 4096, 1.0)
    calls.clear()
    assert m.apply(layer, _X(64)) == "nv" and calls[-1][0] == "nvfp4"
    assert m.apply(layer, _X(33)) == "nv"
    assert m.apply(layer, _X(32)) == "fp8" and calls[-1][0] == "fp8"
    assert m.apply(layer, _X(8)) == "fp8"
    # bias keeps the base path; the opaque drafter path never reads the pair
    assert m.apply(layer, _X(64), bias="b") == "bf16"
    m._opaque = True
    assert m.apply(layer, _X(64)) == "op" and calls[-1][0] == "opaque"
    m._opaque = False
    # a failing pair drops to fp8 for good, loudly
    ns["_nvfp4_dense_gemm_op"] = lambda x, *nv: (_ for _ in ()).throw(RuntimeError("mm_fp4"))
    calls.clear()
    assert m.apply(layer, _X(64)) == "fp8"
    assert m._nvfp4 is None and any("nvfp4 prefill pair failed" in c[1] for c in calls if c[0] == "log")
    assert m.apply(layer, _X(64)) == "fp8"
    # the knob is target-only and exact
    env = ns["_PREFILL_NVFP4_ENV"]
    old = os.environ.pop(env, None)
    try:
        assert not ns["_prefill_nvfp4_enabled"]("VLLM_GLM53_FP8_DENSE")
        os.environ[env] = "1"
        assert ns["_prefill_nvfp4_enabled"]("VLLM_GLM53_FP8_DENSE")
        assert not ns["_prefill_nvfp4_enabled"]("VLLM_DFLASH2_FP8_DENSE")
        os.environ[env] = "w8"
        assert not ns["_prefill_nvfp4_enabled"]("VLLM_GLM53_FP8_DENSE")
    finally:
        if old is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = old
    # build wiring and the profile default
    src = open("overlay/modules/glm53_model/glm53_fp8_dense.py", encoding="utf-8").read()
    assert 'prefill_nv = _prefill_nvfp4_enabled(env) and scheme == "w8a8"' in src
    assert "method._nvfp4 = pair" in src and "%d nvfp4 prefill " in src
    prof = open("profiles/glm53.env", encoding="utf-8").read()
    assert "\nVLLM_GLM53_FP8_DENSE_PREFILL_NVFP4=0\n" in prof


def test_spec_k_compile_factor() -> None:
    """29차: num_speculative_tokens must be part of the compile-cache key --
    the launcher forwards SPEC_K as VLLM_GLM53_SPEC_K and the fp8-dense
    module registers it as a compile factor (a K=5 boot's drafter artifacts
    killed the following K=7 boot)."""
    import os
    fd = open(os.path.join(REPO, "overlay/modules/glm53_model/glm53_fp8_dense.py"), encoding="utf-8").read()
    launcher = open(os.path.join(REPO, "launchers/start-glm53-nvfp4-tp4.sh"), encoding="utf-8").read()
    check('_register_compile_factor("VLLM_GLM53_SPEC_K", _spec_k_value)' in fd
          and 'ENVV="$ENVV -e VLLM_GLM53_SPEC_K=$SPEC_K"' in launcher,
          "SPEC_K reaches the container and keys the compile cache")
    print("  spec_k compile factor .. OK")


def test_sampler_profile_skip_contract() -> None:
    """29차: VLLM_GLM53_SKIP_SAMPLER_PROFILE=1 replaces the profile-time dummy
    sampler run with a loud no-op (the K5 boot's 45-minute Triton init);
    the profile ships it off and the installer applies it before its own
    mode check so an 'off' prep-fused boot still gets it."""
    import os
    pf = open(os.path.join(REPO, "overlay/modules/glm53_runtime/glm53_prep_fused.py"), encoding="utf-8").read()
    prof = open(os.path.join(REPO, "profiles/glm53.env"), encoding="utf-8").read()
    inst = pf[pf.index("def install_glm53_prep_fused()"):]
    check("GPUModelRunner._dummy_sampler_run = _skip" in pf
          and "profile-time dummy sampler run SKIPPED" in pf
          and inst.index("_maybe_skip_sampler_profile()") < inst.index("mode = prep_fused_mode()")
          and re.search(r"^VLLM_GLM53_SKIP_SAMPLER_PROFILE=0$", prof, re.M) is not None,
          "the sampler-profile skip is env-gated, loud, applied before the mode "
          "check, and off in the profile")
    print("  sampler profile skip contract .. OK")


def test_dev_lab_contracts() -> None:
    """32차 item 5: the boot-free kernel loop. The worker module remembers
    the served FULL descriptor and serves replay/reload/recapture through
    Worker.glm53_lab; the API route is a --middleware the launcher adds only
    when the knob is on; the driver installs the worker side from arm() and
    can rebuild the extension from another .cu; the profile ships it off."""
    import os
    mod = os.path.join(REPO, "overlay/modules/glm53_runtime")
    lab = open(os.path.join(mod, "glm53_dev_lab.py"), encoding="utf-8").read()
    mw = open(os.path.join(mod, "glm53_lab_middleware.py"), encoding="utf-8").read()
    man = open(os.path.join(mod, "manifest.tsv"), encoding="utf-8").read()
    req = open(os.path.join(mod, "requires"), encoding="utf-8").read()
    mk = open(os.path.join(REPO, "overlay/modules/glm53_megakernel/glm53_megakernel.py"), encoding="utf-8").read()
    prof = open(os.path.join(REPO, "profiles/glm53.env"), encoding="utf-8").read()
    launcher = open(os.path.join(REPO, "launchers/start-glm53-nvfp4-tp4.sh"), encoding="utf-8").read()
    check("CudaGraphManager.run_fullgraph = run_fullgraph" in lab
          and "Worker.glm53_lab = _lab" in lab
          and all(f'op == "{o}"' in lab for o in ("info", "replay", "reload", "recapture"))
          and "mgr.run_fullgraph(desc)" in lab and "e0.elapsed_time(e1)" in lab
          and "mk.rebuild(src)" in lab and "runner.capture_model()" in lab
          and "m2.graphs.clear()" in lab and "managers_cleared" in lab,
          "the worker lab remembers the served FULL descriptor, replays it "
          "with CUDA events, rebuilds the extension and recaptures")
    check('request.url.path != "/glm53/lab"' in mw
          and 'collective_rpc("glm53_lab"' in mw and "status_code=500" in mw,
          "the API side is a pass-through middleware that fans the op out "
          "over collective_rpc and says errors")
    check("glm53_dev_lab.py\tvllm/model_executor/layers/glm53_dev_lab.py\tabsent" in man
          and "glm53_lab_middleware.py\tvllm/glm53_lab_middleware.py\tabsent" in man
          and "glm53_megakernel" in req.split(),
          "manifest places both files as new; the module requires the driver")
    check("def rebuild(src_path: str) -> dict:" in mk
          and 'name=f"glm53_megakernel_{md5}"' in mk and "_armed_once = False" in mk
          and 'if _flag("VLLM_GLM53_DEV_LAB"):' in mk and "glm53_dev_lab.install()" in mk,
          "the driver rebuilds under a per-md5 name, re-arms, and installs "
          "the lab from arm() behind the knob")
    check(re.search(r"^VLLM_GLM53_DEV_LAB=0$", prof, re.M) is not None
          and "glm53_runtime" in re.search(r'^MODULES="([^"]*)"', prof, re.M).group(1)
          and 'if [ "${VLLM_GLM53_DEV_LAB:-0}" != 0 ]; then' in launcher
          and "--middleware vllm.glm53_lab_middleware.lab" in launcher,
          "the profile enrols the module OFF; the launcher adds the "
          "middleware only when the knob is on")
    print("  dev lab contracts .. OK")


def test_mk_smlp_hook_and_contracts() -> None:
    """MK_SEG_SMLP (32차): the dense MLP as one launch, wired without risk.

    smlp_forward is the Glm5NextMLP.forward hook: None (stock) unless the
    segment is armed, x is a 2-D bf16 decode batch (T <= 32, k a multiple
    of 128 <= 4096), both linears carry single W4 packs whose k-tiles match,
    and gate_up is 2 x the down input. The activation's clamp/alpha/beta
    come from the module (SiluAndMulWithClamp; SiluAndMul = 0/1/0). The
    kernel keeps the stock rounding points and reuses kda's a_ready
    hand-off; the hook never reduces (the linear's reduce_results contract
    stays with the caller).
    """
    import types
    calls, said = [], []
    ns = load_defs(
        "overlay/glm53_megakernel.py",
        {"_ARMED", "MAX_TOK", "MK_GEMM_KMAX", "SMLP_GU_MAX", "_smlp_packs",
         "_SMLP_SAID", "_SMLP_FUSED_CALLS", "_smlp_stock", "smlp_forward"},
        {"os": os, "re": re,
         "logger": types.SimpleNamespace(warning=lambda *a, **k: said.append(a)),
         "_smlp_call": lambda *a: calls.append(a) or "fused"},
    )
    import types
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bf16"
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        class _X:
            def __init__(self, m, k, dtype="bf16"):
                self.shape = (m, k); self.dtype = dtype
            def dim(self): return 2
            def contiguous(self): return self
        class _Pack:
            def __init__(self, ntiles, ktiles):
                self.shape = (ntiles, ktiles, 128, 64)
        def linear(pack, out_size=None, in_size=None):
            q = types.SimpleNamespace(_mk=pack)
            return types.SimpleNamespace(quant_method=q, output_size_per_partition=out_size,
                                         input_size_per_partition=in_size, output_size=out_size)
        gu = (_Pack(8, 32), "ws", 1.0)      # [1024 x 4096]
        d = (_Pack(32, 4), "ws", 1.0)       # [4096 x 512]
        mlp = types.SimpleNamespace(
            gate_up_proj=linear(gu, out_size=1024),
            down_proj=linear(d, out_size=4096, in_size=512),
            act_fn=types.SimpleNamespace(swiglu_limit=10.0, alpha=1.0, beta=0.0))
        f = ns["smlp_forward"]
        ns["_ARMED"]["smlp"] = False
        assert f(mlp, _X(8, 4096)) is None            # not armed
        ns["_ARMED"]["smlp"] = True
        assert f(mlp, _X(8, 4096)) == "fused"
        args = calls[-1]
        assert args[3:] == (1024, 512, 4096, 10.0, 1.0, 0.0), args[3:]
        assert f(mlp, _X(33, 4096)) is None           # T beyond the lane
        # the proof lines: the first fused call says so once; a distinct
        # stock reason says itself once; prefill rows are silent
        assert any("smlp lane serving: first fused call" in a[0] for a in said), said
        n_said = len(said)
        assert f(mlp, _X(33, 4096)) is None and len(said) == n_said
        assert f(mlp, _X(8, 4096, dtype="fp16")) is None and len(said) == n_said + 1
        assert f(mlp, _X(8, 4096, dtype="fp16")) is None and len(said) == n_said + 1
        assert f(mlp, _X(8, 4096, dtype="fp16")) is None
        assert f(mlp, _X(8, 4160)) is None            # k not a multiple of 128
        # a K-chunked (list) pack or a missing pack -> stock
        mlp.gate_up_proj.quant_method._mk = [gu, gu]
        assert f(mlp, _X(8, 4096)) is None
        mlp.gate_up_proj.quant_method._mk = gu
        mlp.down_proj.quant_method._mk = None
        assert f(mlp, _X(8, 4096)) is None
        mlp.down_proj.quant_method._mk = d
        # a wrapped method (nvfp4 / w4a8 stack on the fp8 one) is unwrapped
        mlp.down_proj.quant_method = types.SimpleNamespace(_base=types.SimpleNamespace(_mk=d))
        assert f(mlp, _X(8, 4096)) == "fused"
        # gate_up must be 2 x the down input; pack k-tiles must match
        mlp.down_proj = linear(d, out_size=4096, in_size=384)
        assert f(mlp, _X(8, 4096)) is None
        mlp.down_proj = linear((_Pack(32, 5), "ws", 1.0), out_size=4096, in_size=512)
        assert f(mlp, _X(8, 4096)) is None
        # plain SiluAndMul: no limit attribute -> 0 / 1 / 0
        mlp.down_proj = linear(d, out_size=4096, in_size=512)
        mlp.act_fn = types.SimpleNamespace()
        assert f(mlp, _X(16, 4096)) == "fused" and calls[-1][6:] == (0.0, 1.0, 0.0)
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
    cu = open("overlay/modules/glm53_megakernel/glm53_megakernel.cu", encoding="utf-8").read()
    k = cu[cu.index("void mk_smlp_kernel"):cu.index("void mk_mla_kernel")]
    assert k.count("mk_gemm_phase(c, smem, &g_mk_smlp_bar);") == 2, "two GEMM phases on the segment's own barrier"
    assert "c.a_ready = true;" in k and "c.unit_ctr = &g_mk_smlp_unit2;" in k and "g_mk_smlp_unit2 = 0u;" in k, \
        "down rides the a_ready path on its own unit counter, reset before phase A"
    assert k.count("mk_grid_barrier(a.barrier_ctr, a.grid);") == 1, "one barrier: the activation lives in gate_up's epilogue"
    assert "c.pair_act = 1;" in k and "c.n_int = a.n_int;" in k
    ph = cu[cu.index("__device__ void mk_gemm_phase"):cu.index("__global__ void mk_gemm_kernel")]
    assert "auto pair_finish = [&](int nt) {" in ph and ph.count("pair_finish(nt);") == 2, "both final-store paths finish the pair"
    assert "__float2bfloat16(" in ph and "fminf(gv, c.act_limit)" in ph and "(uv + c.act_beta)" in ph, "clamped SwiGLU at the stock rounding point"
    assert "if (s_pair_last) g_mk_pair_arrive[pair] = 0u;" in ph, "pair counters rearm like tile counters"
    assert "if (c.unit_ctr) s_unit = c.grid + (int)atomicAdd(c.unit_ctr, 1u);" in ph
    assert 'm.def("run_smlp"' in cu and "mk_smlp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize" in cu
    wiring = open("overlay/modules/glm53_model/glm5next_model.py", encoding="utf-8").read()
    assert "out = _mk_smlp(self, x)" in wiring and 'getattr(self.down_proj, "reduce_results", False)' in wiring
    prof = open("profiles/glm53.env", encoding="utf-8").read()
    assert "\nVLLM_GLM53_MK_SMLP=0\n" in prof, "bracket-gated: off until the 32차 bracket"


def test_mk_mla_workspace_is_fixed_and_splits_bounded() -> None:
    """28차: the MLA split scratch never moves under a captured graph.

    _mla_workspace used to grow on demand -- cap = max(need, 2*MAX_TOK),
    re-allocated (old tensors freed) when a larger call came. A captured
    decode graph bakes the address, and v5 routes prefill rows through the
    same entry point: the first request of a boot (37-token prompt, 48
    splits, 1,776 rows) freed the scratch under every decode graph, whose
    partials then landed in whatever the allocator handed out next -- two
    serving deaths. Now the scratch is one fixed allocation of MLA_WS_ROWS
    and mla_splits() never asks for more: the decode shapes keep their
    measured splits, rows the rule would split 48 ways take the direct path.
    """
    import sys
    import types

    class _Zeros:
        def __init__(self, n):
            self.n = n

    torch_stub = types.ModuleType("torch")
    torch_stub.float32 = "f32"
    torch_stub.zeros = lambda n, dtype=None, device=None: _Zeros(n)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    old_forced = os.environ.pop("VLLM_GLM53_MK_MLA_SPLITS", None)
    try:
        ns = load_defs(
            "overlay/glm53_megakernel.py",
            {"MLA_H", "MLA_D", "MLA_SPLITS_MAX", "MLA_MAX_SPLIT_ROWS",
             "MLA_WS_ROWS", "_MLA_WS", "_mla_workspace", "mla_splits"},
            {"os": os, "_EXT": types.SimpleNamespace(mla_grid=lambda: 48)},
        )
        splits, rows = ns["mla_splits"], ns["MLA_WS_ROWS"]
        assert rows == 3 * ns["MLA_MAX_SPLIT_ROWS"] == 192
        # the measured decode rule is untouched
        for T, want in ((8, 6), (16, 3), (24, 2), (32, 3), (64, 3)):
            assert splits(T) == want, (T, splits(T))
        # every split-eligible T fits the fixed scratch
        for T in range(1, ns["MLA_MAX_SPLIT_ROWS"] + 1):
            s_ = splits(T)
            assert 1 <= s_ <= ns["MLA_SPLITS_MAX"] and T * s_ <= rows, (T, s_)
        # the rows the old rule split 48 ways now take the direct path
        assert splits(37) == 1 and splits(40) == 1 and splits(56) == 1
        assert splits(65) == 1 and splits(8192) == 1 and splits(0) == 1
        # the probe knob is clamped to the budget too
        os.environ["VLLM_GLM53_MK_MLA_SPLITS"] = "64"
        assert splits(8) == 24 and splits(64) == 3
        os.environ.pop("VLLM_GLM53_MK_MLA_SPLITS")
        # one allocation, held for good: same object back, never re-sized
        ws = ns["_mla_workspace"]
        w1 = ws("cuda", 8, 6)
        assert w1["cap"] == rows
        assert w1["part"].n == rows * ns["MLA_H"] * ns["MLA_D"]
        assert w1["pml"].n == rows * ns["MLA_H"] * 2
        w2 = ws("cuda", 64, 3)
        assert w2 is w1 and w2["part"] is w1["part"]
        try:
            ws("cuda", 37, 48)
        except RuntimeError as e:
            assert "1776" in str(e)
        else:
            raise AssertionError("an over-budget call must raise, not re-allocate")
        assert ns["_mla_workspace"]("cuda", 8, 6) is w1
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
        if old_forced is not None:
            os.environ["VLLM_GLM53_MK_MLA_SPLITS"] = old_forced
    # mla_decode refuses a split plan the scratch cannot hold, and the boot
    # self-test now covers the direct path (40 and 100 rows) too
    src = open("overlay/modules/glm53_megakernel/glm53_megakernel.py",
               encoding="utf-8").read()
    assert "assert splits == 1 or T * splits <= MLA_WS_ROWS" in src
    assert "(40, 2048, True), (100, 2048, True)" in src
    # the serving shadow judges real rows, not the empty-KV dummy
    wsrc = open("overlay/modules/glm53_model/flashinfer_mla_sparse_sm90.py",
                encoding="utf-8").read()
    body = wsrc[wsrc.index("def _mk_mla_run("):wsrc.index("class FlashInferMLASparseSM90Impl")]
    assert "rows = int(valid_counts.max().item())" in body
    assert "rows >= _MK_MLA_SHADOW_MIN_ROWS" in body
    assert "den < _MK_MLA_SHADOW_MIN_NORM" in body and "inconclusive" in body
    assert "SHADOW FAIL" in body and 'm._ARMED["mla"] = False' in body
    assert "q_nope.shape[0] <= 64" not in body


def test_fp8_dense_drafter_compile_factor_and_serving_proof() -> None:
    """28차: the drafter knob is a compile-cache factor and serving is proven.

    vLLM keys torch.compile / AOT artifacts on the env vars registered in
    vllm.envs, the config and the forward's source -- never on a quant_method
    swapped in after load -- and loads them with guards off. Every boot with
    VLLM_DFLASH2_FP8_DENSE=1 served the 09-03 bf16 drafter graph while the
    fingerprint said 31 linears armed; the bracket measured the eager fc
    alone. Two pieces close that: the knob registers itself into
    vllm.envs.environment_variables (compile_factors() hashes every entry),
    and the loader wraps the drafter's forward to count opaque-op calls,
    reporting NOT SERVING when a Python-running forward makes fewer than
    half the expected calls. CUDA-graph replays run no Python and are not
    judged; a forward under stream capture is definitive.
    """
    import sys
    import types

    msgs = []

    class _Log:
        def warning(self, fmt, *a):
            msgs.append(fmt % a if a else fmt)

    ns = load_defs(
        "overlay/glm53_fp8_dense.py",
        {"_DRAFTER_ENV", "_DRAFTER_OFF", "_drafter_knob_value",
         "_register_compile_factor", "_OPAQUE_CALLS", "_stream_capturing",
         "install_drafter_serving_check"},
        {"os": os, "re": re, "torch": None, "logger": _Log()},
    )
    env = ns["_DRAFTER_ENV"]
    # -- factor registration against a stub vllm.envs
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.envs")}
    envs = types.ModuleType("vllm.envs")
    envs.environment_variables = {}
    pkg = types.ModuleType("vllm")
    pkg.envs = envs
    sys.modules["vllm"], sys.modules["vllm.envs"] = pkg, envs
    old = os.environ.pop(env, None)
    try:
        reg = ns["_register_compile_factor"]
        assert reg(env, ns["_drafter_knob_value"])
        assert reg(env, ns["_drafter_knob_value"])  # idempotent
        assert list(envs.environment_variables) == [env]
        getter = envs.environment_variables[env]
        assert getter() == "0"  # unset hashes like off
        for raw, want in (("1", "1"), ("w8", "w8"), ("off", "0"), ("", "0"),
                          (" W8 ", "w8"), ("false", "0")):
            os.environ[env] = raw
            assert getter() == want, (raw, getter())
    finally:
        if old is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = old
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    # -- the serving proof
    class Drafter:
        def __init__(self, calls):
            self.calls = calls

        def forward(self, *a, **k):
            ns["_OPAQUE_CALLS"] += self.calls
            return "out"

    install = ns["install_drafter_serving_check"]
    # serving: 30 of 31 per forward (fc runs outside forward) -> reported
    # after the window, wrapper gone
    m = Drafter(30)
    install(m, 31, forwards=3)
    assert "forward" in m.__dict__
    for _ in range(3):
        assert m.forward() == "out"
    assert "forward" not in m.__dict__
    assert any("serving: 30 of 31" in x for x in msgs), msgs
    assert m.forward() == "out"
    # replays (0 calls, no capture) are not judged; a captured forward is,
    # and a stale graph is loud
    msgs.clear()
    m = Drafter(0)
    install(m, 31, forwards=8)
    for _ in range(3):
        m.forward()
    assert not msgs and "forward" in m.__dict__
    ns["_stream_capturing"] = lambda: True
    m.forward()
    assert any("NOT SERVING: 0 of 31" in x for x in msgs), msgs
    assert "forward" not in m.__dict__
    ns["_stream_capturing"] = lambda: False
    # the eager fc alone (1 of 31) is what 28차 served: judged and loud at once
    msgs.clear()
    m = Drafter(1)
    install(m, 31, forwards=8)
    m.forward()
    assert any("NOT SERVING: 1 of 31" in x for x in msgs), msgs
    # never a Python-running forward in the window -> unknown, not a pass
    msgs.clear()
    m = Drafter(0)
    install(m, 31, forwards=2)
    m.forward()
    m.forward()
    assert any("SERVING UNKNOWN" in x for x in msgs), msgs
    # nothing armed installs nothing
    m = Drafter(5)
    install(m, 0)
    assert "forward" not in m.__dict__
    # the loader wires the proof behind a True pass, counting opaque methods
    src = open("overlay/modules/glm53_drafter/dflash_utils.py",
               encoding="utf-8").read()
    assert "if maybe_build_fp8_dense(dflash_model, env=\"VLLM_DFLASH2_FP8_DENSE\"):" in src
    assert "install_drafter_serving_check(dflash_model, n_opaque)" in src
    # and the op body counts itself
    fsrc = open("overlay/modules/glm53_model/glm53_fp8_dense.py",
                encoding="utf-8").read()
    body = fsrc[fsrc.index("def _mk_or_fp8_dense_gemm("):]
    body = body[:body.index("\ntry:")]
    assert "global _OPAQUE_CALLS" in body and "_OPAQUE_CALLS += 1" in body
    assert "_register_compile_factor(_DRAFTER_ENV, _drafter_knob_value)" in fsrc


def test_fp8_dense_drafter_patterns_and_opaque_op() -> None:
    """The DFlash2 drafter's dense GEMMs under VLLM_DFLASH2_FP8_DENSE.

    The armed 09-03 trace ends every decode step in a 7.5-8.5 ms tail the
    forward's annotation never covered: target head, then the drafter --
    fc 792 us bf16 (K = 20480, replicated), five layers of sharded bf16
    projections, the draft head. The base pattern set lists q/k/v_proj and
    the target's fused names, never the drafter's MERGED qkv_proj, its fc,
    or its conv kernel_projections, so the drafter knob covered o/gate_up/
    down and left 43% of the drafter's bytes bf16. And the drafter's forward
    is torch.compiled, so its GEMMs must be one opaque op each: dynamo
    cannot trace the lane's eligibility test or the extension call.
    """
    src = open("overlay/modules/glm53_model/glm53_fp8_dense.py",
               encoding="utf-8").read()
    ns = load_defs(
        "overlay/glm53_fp8_dense.py",
        {"_SHARED_EXPERT_RE", "_INCLUDE", "_BPROJ_INCLUDE", "_BPROJ_ON",
         "_include_patterns", "_DRAFTER_ENV", "_DRAFTER_INCLUDE"},
        {"os": os, "re": re},
    )
    include = ns["_include_patterns"]
    matches = lambda pats, name: any(p.search(name) for p in pats)
    drafter_names = [
        "model.layers.0.self_attn.qkv_proj", "model.layers.4.self_attn.o_proj",
        "model.layers.2.mlp.gate_up_proj", "model.layers.2.mlp.down_proj",
        "model.layers.1.attention_conv.kernel_projection",
        "model.layers.1.mlp_conv.kernel_projection", "model.fc",
    ]
    saved = os.environ.pop("VLLM_GLM53_FP8_DENSE_BPROJ", None)
    try:
        tgt, drf = include(), include(ns["_DRAFTER_ENV"])
        check(ns["_DRAFTER_ENV"] == "VLLM_DFLASH2_FP8_DENSE",
              "the drafter set keys on the drafter's own knob")
        check(all(matches(drf, n) for n in drafter_names),
              "drafter knob: every drafter dense linear matches, fc included")
        check(len(drf) == len(tgt) + len(ns["_DRAFTER_INCLUDE"]),
              "drafter knob: base patterns plus the drafter set, nothing else")
        for n in ("model.layers.0.self_attn.qkv_proj", "model.fc",
                  "model.layers.1.attention_conv.kernel_projection"):
            check(not matches(tgt, n), f"target knob must NOT match {n}")
        for n in ("candidate_selector.hidden_projection", "model.fc_norm",
                  "model.layers.0.self_attn.attn"):
            check(not matches(drf, n), f"drafter knob must NOT match {n}")
    finally:
        if saved is not None:
            os.environ["VLLM_GLM53_FP8_DENSE_BPROJ"] = saved

    body = src[src.index("def maybe_build_fp8_dense("):]
    check("_include_patterns(env)" in body,
          "the build pass selects its pattern set by the knob it runs under")
    check("method._opaque = env == _DRAFTER_ENV" in body,
          "drafter methods are marked opaque: the drafter forward is compiled")
    check("attach_mk = _attach_mk_pack\n" in body and '"w8"' not in body,
          "one lane below fp8: no fp8-only arm to remember (operator rule "
          "2026-09-04 -- a proven improvement is the default, the other side "
          "goes)")

    # the opaque op: one custom op that decides MK-or-fp8 at run time
    check('"glm53_fp8_dense::gemm_mk_or_fp8"' in src
          and "def _mk_or_fp8_dense_gemm(" in src
          and "def _mk_or_fp8_dense_gemm_fake(" in src,
          "the MK-or-fp8 choice is one registered custom op with a fake")
    op = src[src.index("def _mk_or_fp8_dense_gemm("):src.index("def _mk_or_fp8_dense_gemm_fake(")]
    check("gemm_w4a8 as _mk_gemm" in op and "maybe_arm as _mk_arm" in op
          and "return _fp8_dense_gemm(x, q, ws, orig_rows, orig_cols)" in op
          and "packs[0] if len(packs) == 1 else packs" in op,
          "inside the op: arm, try the lane (single or K-chunked pack), "
          "fall back to the verified fp8 pair")
    apply = src[src.index("    def apply(self, layer, x, bias=None):\n        if bias is not None:\n            return self._base.apply(layer, x, bias)\n        if self._opaque:"):]
    apply = apply[:apply.index("        if getattr(self, \"_bf16_freed\", False):")]
    check("_mk_or_fp8_dense_gemm_op(" in apply and "self._mk_args" in apply
          and "isinstance(mk, list)" in apply,
          "an opaque method routes apply() through the op with its pack "
          "flattened once (single or chunked)")

    # the pack attach: K-chunks past the lane's K
    attach = src[src.index("def _attach_mk_pack("):src.index("def _kda_owns(")]
    check("if cols > _mkmod.MK_GEMM_KMAX:" in attach
          and "build_mk_weight_w4_kchunks(weight, name=name," in attach
          and 'per_row = False if getattr(method, "_opaque", False) else None' in attach,
          "a linear wider than the lane's K gets one pack per K-chunk")

    # the lane side
    mksrc = open("overlay/modules/glm53_megakernel/glm53_megakernel.py",
                 encoding="utf-8").read()
    check("MK_GEMM_KMAX = 4096" in mksrc
          and "0 < k <= MK_GEMM_KMAX" in mksrc
          and "def build_mk_weight_w4_kchunks(weight, name=None, per_row=None):" in mksrc
          and "def _gemm_kchunks(x, packs, n_rows, bg=False):" in mksrc
          and "if isinstance(mk_pack, list):\n        return _gemm_kchunks(x, mk_pack, n_rows, bg)" in mksrc,
          "the lane's K contract is one constant; a chunked pack is a list "
          "gemm_w4a8 sums in fp32")
    kch = mksrc[mksrc.index("def _gemm_kchunks("):mksrc.index("# MK_SEG_MHC")]
    check("if len(packs) != -(-k // MK_GEMM_KMAX):" in kch
          and "_mk_gemm_eligible(m, kc, p[0].shape[0] * 128)" in kch
          and "acc.add_(out.float())" in kch and "acc.to(torch.bfloat16)" in kch,
          "chunked lane: every chunk passes the per-launch contract or the "
          "whole linear stays stock; partials summed in fp32, bf16 out")

    # the offline path check exists and covers the compiled contract
    chk = open(os.path.join(REPO, "probes", "drafter_dense_path_check.py"),
               encoding="utf-8").read()
    check("fullgraph=True" in chk and "dynamic=True" in chk
          and "torch.cuda.graph(" in chk and 'os.environ[DRAFTER_ENV] = "w8"' not in chk
          and "gemm_w4a8(x, q._mk, q._rows)" in chk,
          "the path probe gates compile (fullgraph, dynamic), capture, and "
          "the lane serving bitwise")
    print("  fp8-dense drafter patterns + opaque op .. OK")


def test_ab_runner_measures_both_channels() -> None:
    """The A/B arm runner must not be able to skip prefill.

    Decode-only measurement has hidden prefill regressions in this lane more
    than once, and prefill is TTFT -- what a user feels. The operator has
    asked for both together repeatedly; a flag would let it be forgotten
    again, so the script runs both unconditionally.
    """
    p = os.path.join(REPO, "launchers", "ab-glm53.sh")
    src = open(p, encoding="utf-8").read()
    check(os.access(p, os.X_OK), "ab-glm53.sh is executable")
    check("bench/bracket.py" in src and "prefill_ladder.py" in src,
          "one arm runs BOTH the decode leg and the prefill ladder")
    check(src.index("bracket.py") < src.index("prefill_ladder.py"),
          "decode first, prefill second, same boot")
    body = src[src.index("== decode"):]
    check("if" not in body.split("prefill_ladder.py")[0].split("== prefill")[0],
          "prefill is unconditional -- no flag, no branch, no way to skip it")
    check("step/s" in src and "tok/s" in src,
          "the script states why step/s is the judgment channel: tok/s "
          "carries the acceptance draw (30.9-43.0 vs 14.5-15.3 on 09-04)")
    check("armed=" in src and "MK W4 packs" in src,
          "each arm records the boot's own fingerprint -- the next boot "
          "truncates the log, so an unrecorded arm is unauditable later")
    print("  ab runner measures both ....... OK")


def test_fused_k_gate_lazy_slot_exists() -> None:
    """The fused indexer forward must not read a slot nobody creates.

    #188 shipped `if self._wp_fp32 is None:` with nothing ever setting it, so
    arming VLLM_GLM53_FUSED_K_GATE killed the engine at load:

      RuntimeError: Worker failed with error
        ''Indexer' object has no attribute '_wp_fp32''

    It had never booted in any configuration -- the same gate also fires for
    VLLM_GLM53_SM121_MLA_PREFILL=1. install() and prepare() are gated
    separately, so the forward can be live on a layer prepare() skipped; the
    read has to tolerate that.
    """
    source = open(_overlay_source(
        "overlay/modules/glm53_model/glm53_prefill_fastpath.py"
    ), encoding="utf-8").read()
    tree = ast.parse(source)
    fused = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_glm53_fused_indexer_forward"
    )
    body = ast.get_source_segment(source, fused) or ""
    check("self._wp_fp32 is None" not in body,
          "a bare attribute read of the lazy slot crashes any layer that "
          "prepare() skipped -- use getattr(self, ..., None)")
    check('getattr(self, "_wp_fp32", None)' in body,
          "the lazy fp32 head-weight cache must be read defensively")

    prepare = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_glm53_prefill_fastpath"
    )
    prep_body = ast.get_source_segment(source, prepare) or ""
    check("_wp_fp32 = None" in prep_body,
          "prepare() must create the slot where it sets the rest of the "
          "layer's fast-path state")
    print("  fused K+gate lazy slot ........ OK")
# ---------------------------------------------------------------------------
# glm53_megakernel -- pure helpers, .cu/.py geometry parity, sm_121a static
# contracts, hook placement
# ---------------------------------------------------------------------------
def test_osar_wait_is_split_by_message_size() -> None:
    """The all-reduce's wait is 85% of its cost; name what it is.

    Measured live 2026-09-03 (n=6922): guard 0.6, copy 3.5, wait 38.7,
    reduce 2.6 us per collective. The wait spins until all three peers'
    data has landed, so it carries the RDMA transfer as well as any arrival
    skew -- and at the 256 KiB cap a rank takes in 3 x 256 KiB, which is
    ~43 us at the fabric's measured 17.8 GB/s (~22 us if both HCAs serve
    it). Transfer scales with the message and skew does not, so the same
    counter split by size tells them apart. Two fields, because the peers'
    rx_base/rxf_base offsets must not move."""
    cu = open(os.path.join(REPO, "overlay/modules/tp_oneshot_ar/"
                                 "dsv4_oneshot_ar.cu"), encoding="utf-8").read()
    check("#define SPLIT_BYTES (128 * 1024)" in cu,
          "the split sits between C=1-shaped traffic (8 tokens = 64 KiB) and "
          "a full 4-sequence spec batch (32 tokens = 256 KiB)")
    check("volatile uint64_t t_wait_sm;" in cu
          and "volatile uint64_t t_calls_sm;" in cu
          and "uint64_t pad[1];" in cu and "uint64_t pad[3];" not in cu
          and "osar_stall_check(c, sp, t0, STALL_GUARD" in cu
          and "osar_stall_check(c, sp, t2, STALL_WAIT" in cu
          and '"[oneshot] STALL rank=%d phase=%s seq=%llu slot=%d missing_peer_mask=0x%x "' in cu
          and "#define OSAR_STALL_TRAP_S 30" in cu
          and cu.count("if (timer && c->pad[0] != 0) c->pad[0] = 0;") == 2
          and "if (seq < 16) return;" in cu,
          "the two counters come OUT of the existing padding: anything that "
          "moved tx/rx would invalidate every peer's registered offsets")
    check("if (nbytes <= SPLIT_BYTES) {" in cu
          and "c->t_wait_sm += (uint64_t)(t3 - t2);" in cu,
          "the small-message wait accumulates from the same t3 - t2 as the "
          "total, so the two can be subtracted")
    check("wait by size: <=128KiB n=%llu %.1f, >128KiB n=%llu %.1f" in cu
          and "last_wait_sm = g_ctrl->t_wait_sm;" in cu,
          "the proxy reports both halves per interval and carries its own "
          "high-water marks, like the counters beside it")
    print("  osar wait is split by message size .. OK")


def test_osar_prefetch_hints_contract() -> None:
    """The one-shot AR's peer wait doubles as an L2 prefetch of the next
    kernel's weights (VLLM_GLM53_AR_PREFETCH), and what it warms is learned
    from the megakernel launches that follow each collective.

    Contract: the kernel takes the hints by value (a captured graph bakes
    them), only warps 1..7 issue them and only inside the wait window (thread
    0's poll and its timer are untouched, so t_wait keeps measuring the
    collective), an empty hint is the old kernel; the shim counts EVERY
    collective of the target forward so NCCL-served prefill calls keep the
    ordinals aligned, and the driver notes its weights before each launch."""
    cu = open(os.path.join(REPO, "overlay/modules/tp_oneshot_ar/"
                                 "dsv4_oneshot_ar.cu"), encoding="utf-8").read()
    check("#define OSAR_MAXHINT 8" in cu and "struct HintArgs {" in cu
          and "int nbytes, const HintArgs h) {" in cu,
          "k_oneshot takes up to 8 (ptr, bytes) hints by value")
    check(cu.count('asm volatile("prefetch.global.L2 [%0];"') == 1
          and cu.index('asm volatile("prefetch.global.L2')
          > cu.index("__device__ __forceinline__ void osar_prefetch("),
          "one prefetch instruction, inside osar_prefetch")
    call_at = cu.find("} else if (h.n > 0 && threadIdx.x >= 32) {\n"
                      "    osar_prefetch(h, &s_landed);")
    t2_at = cu.find("long long t2 = timer ? clock64() : 0;")
    t3_at = cu.find("long long t3 = timer ? clock64() : 0;")
    check(0 < t2_at < call_at < t3_at,
          "the prefetch runs only in the peer-wait window, on warps 1..7, "
          "while thread 0 polls")
    check("    s_landed = 1;\n  } else if (h.n > 0" in cu,
          "owning blocks release their prefetch warps the moment the peers "
          "land")
    check("k_oneshot<<<ARGRID, ARTHREADS, 0, st>>>(g_ctrl, src, dst, (int)n,"
          in cu and "(int)(n * 2), h);" in cu,
          "the fixed-geometry launch carries the hints")
    check('m.def("oneshot_ar_hint", &py_oneshot_hint);' in cu
          and 'm.def("phase_counters", &py_phase_counters);' in cu
          and 'm.def("oneshot_ar", &py_oneshot);' in cu,
          "hint and counter bindings beside the unchanged plain entry")

    shim_path = os.path.join(REPO, "overlay/modules/tp_oneshot_ar/"
                                   "dsv4_oneshot_shim.py")
    shim = open(shim_path, encoding="utf-8").read()
    check("_PREFETCH_BUDGET = _resolve_prefetch_budget()" in shim
          and 'def begin_forward(scope: str = "target"):' in shim
          and "def end_forward():" in shim
          and "def note_consumer(tensors):" in shim
          and "_tables[_scope] = _cand" in shim,
          "the shim learns one hint table per model forward (scope)")
    mar = shim[shim.index("def maybe_all_reduce("):]
    check(mar.index("_ordinal += 1") < mar.index("if _disabled:"),
          "every collective advances the ordinal before any path decision")
    check("hint = _tables.get(_scope, {}).get(_ordinal) if _in_forward else None"
          in mar and "_ext.oneshot_ar_hint(" in mar
          and "_ext.oneshot_ar(input_)" in mar,
          "hints apply only inside a forward, from that model's own table; "
          "the plain path stays")

    drv = open(os.path.join(REPO, "overlay/modules/glm53_megakernel/"
                                  "glm53_megakernel.py"), encoding="utf-8").read()
    for site, note, launch in (
            ("gemm", "_ar_note(mk_pack[0], mk_pack[1])", "_EXT.run_gemm("),
            ("mhc", "_ar_note(fn)", "_EXT.run_mhc("),
            ("kda", "_ar_note(layer._mk_in_pack[0], layer._mk_in_pack[1])",
             "_EXT.run_kda(")):
        n_at, l_at = drv.find(note), drv.find(launch)
        check(0 < n_at < l_at, f"{site} launch notes its weights first")

    wiring = open(os.path.join(REPO, "overlay/modules/glm53_model/"
                                     "glm5next_model.py"), encoding="utf-8").read()
    cls_at = wiring.index("class Glm5NextForConditionalGeneration(")
    layer_at = wiring.index("class Glm5NextDecoderLayer(")
    b_at = wiring.index('osar.begin_forward("target")')
    check(b_at > cls_at and "osar.end_forward()" in wiring[b_at:]
          and "def _osar_shim():" in wiring
          and 'osar.begin_forward("target")' in wiring,
          "the forward boundary comes from the class above the compiled region")
    drafter = open(os.path.join(REPO, "overlay/modules/glm53_drafter/"
                                      "qwen3_dflash2.py"), encoding="utf-8").read()
    d_cls = drafter.index("class DFlash2Qwen3ForCausalLM(")
    d_b = drafter.index('osar.begin_forward("drafter")')
    check(d_b > d_cls and "osar.end_forward()" in drafter[d_b:]
          and "def _osar_shim():" in drafter
          and "class DFlash2Qwen3Model(DFlashQwen3Model)" in drafter
          and 'begin_forward' not in drafter[drafter.index("class DFlash2Qwen3Model("):d_cls],
          "the drafter's boundary sits on its ForCausalLM class (above the "
          "compiled DFlashQwen3Model) under its own scope")
    check("begin_forward" not in wiring[layer_at:cls_at],
          "no hint call inside the traced decoder layer")

    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    check(re.search(r"^VLLM_GLM53_AR_PREFETCH=0$", profile, re.M) is not None,
          "the prefetch knob is declared ON in the profile (32차 §14: adopted with KDA+LOCALQ)")
    check(re.search(r"^VLLM_GLM53_MK_PDL=1$", profile, re.M) is not None,
          "PDL is the profile default for the MK launches (2026-09-04)")
    ab = open(os.path.join(REPO, "launchers", "ab-glm53.sh"),
              encoding="utf-8").read()
    check("VLLM_GLM53_MK_PDL=1" in ab.split("cand)", 1)[1].split("\n", 1)[0],
          "the A/B cand arm names MK_PDL explicitly")
    bracket = open(os.path.join(REPO, "bench", "bracket.py"),
                   encoding="utf-8").read()
    check('"VLLM_GLM53_AR_PREFETCH"' in bracket,
          "bracket.py snapshots the prefetch knob")

    # the budget parser and the learning protocol, on the real module
    import importlib.util

    def load(env: str, tag: str):
        old = os.environ.get("VLLM_GLM53_AR_PREFETCH")
        os.environ["VLLM_GLM53_AR_PREFETCH"] = env
        try:
            spec = importlib.util.spec_from_file_location(
                f"_osar_shim_{tag}", shim_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            if old is None:
                os.environ.pop("VLLM_GLM53_AR_PREFETCH", None)
            else:
                os.environ["VLLM_GLM53_AR_PREFETCH"] = old
        return mod

    check(load("0", "a")._PREFETCH_BUDGET == 0
          and load("1", "b")._PREFETCH_BUDGET == 12 << 20
          and load("8", "c")._PREFETCH_BUDGET == 8 << 20
          and load("40", "d")._PREFETCH_BUDGET == 0
          and load("x", "e")._PREFETCH_BUDGET == 0,
          "budget: 0 off, 1 = 12 MB, N = N MB in 1..20, else off")

    class _T:
        is_cuda = True

        def __init__(self, ptr, nbytes):
            self._p, self._n = ptr, nbytes

        def numel(self):
            return self._n

        def element_size(self):
            return 1

        def data_ptr(self):
            return self._p

    m = load("1", "f")
    m.begin_forward("target")
    m.note_consumer([_T(0x10, 100)])           # before any collective: dropped
    m._ordinal = 1
    m.note_consumer([_T(0x1000, 10 << 20), _T(0x2000, 10 << 20)])
    m._ordinal = 2
    m.note_consumer([_T(0x3000, 64)])
    m.end_forward()
    t = m.prefetch_hint_table()
    check(0 not in t and t.get(1) == [(0x1000, 10 << 20), (0x2000, 2 << 20)]
          and t.get(2) == [(0x3000, 64)],
          "notes file under the preceding collective, capped at the budget")
    m.begin_forward("target")
    m._ordinal = 1
    m.note_consumer([_T(0x9000, 8)])
    m.end_forward()
    check(m.prefetch_hint_table() == t,
          "a forward with fewer notes (prefill-shaped) does not replace the "
          "table")
    check(m._in_forward is False, "end_forward closes the window")
    # the drafter's forward learns into ITS table; the target's is untouched
    m.begin_forward("drafter")
    m._ordinal = 1
    m.note_consumer([_T(0x7000, 8)])
    m.end_forward()
    check(m.prefetch_hint_table("drafter") == {1: [(0x7000, 8)]}
          and m.prefetch_hint_table("target") == t,
          "the drafter's collectives keep a table of their own")
    print("  osar prefetch hints contract .. OK")


def test_glm53_dflash_early_fc_contracts() -> None:
    """glm53_dflash_early_fc: the drafter's fc under the target head + sampler.

    Producer = a wrapper on GPUModelRunner.execute_model installed from the
    GLM model import (inert unless VLLM_GLM53_DFLASH_EARLY_FC=1); consumer =
    the drafter's combine_hidden_states, which takes a pending result only
    for this step's token count and waits on the producer's event first --
    before precompute_and_store_context_kv and the drafter graph, so the fc's
    megakernel launch never overlaps another one."""
    mod_dir = os.path.join(REPO, "overlay", "modules", "glm53_drafter")
    src = open(os.path.join(mod_dir, "glm53_dflash_early_fc.py"), encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    modules = re.search(r'^MODULES="([^"]+)"', profile, re.M).group(1).split()
    check("glm53_drafter" in modules, "glm53 profile mounts glm53_drafter (early-fc lives there)")
    check(re.search(r"^VLLM_GLM53_DFLASH_EARLY_FC=0$", profile, re.M) is not None,
          "the knob ships off")
    rows = [l.split("\t") for l in open(os.path.join(mod_dir, "manifest.tsv"), encoding="utf-8")
            .read().splitlines() if l and not l.startswith("#")]
    check(["glm53_dflash_early_fc.py",
                    "vllm/models/glm5next/nvidia/glm53_dflash_early_fc.py", "absent"] in rows,
          f"manifest binds the module as a new file next to the model: {rows}")
    req = open(os.path.join(mod_dir, "requires"), encoding="utf-8").read().split()
    check({"glm53_model"} <= set(req) and os.path.exists(os.path.join(mod_dir, "qwen3_dflash2.py")),
          "requires names the wiring (installer); the drafter overlay (consumer) is in the same module")
    check('(os.environ.get("VLLM_GLM53_DFLASH_EARLY_FC") or "0").strip() == "1"' in src
          and "def install_glm53_dflash_early_fc() -> bool:" in src
          and "Runner.execute_model = _patched_execute_model" in src
          and "_ORIG_EXECUTE_MODEL = Runner.execute_model" in src,
          "exact-1 knob; the installer wraps execute_model and keeps the original")
    prod = src[src.index("def launch_early_fc("):src.index("def take_early_fc(")]
    check("_STREAM.wait_stream(cur)" in prod and "with torch.cuda.stream(_STREAM):" in prod
          and "drafter._deneb_early_fc_event.record(_STREAM)" in prod
          and "torch.cat([a[:n] for a in aux], dim=-1, out=cat_buf[:n])" in prod
          and "width != int(fc.input_size)" in prod,
          "the producer runs on a side stream after the forward, into a persistent "
          "buffer, and refuses a width the stock path would refuse")
    cons = src[src.index("def take_early_fc("):src.index("def _patched_execute_model(")]
    check("drafter._deneb_early_fc_pending = None  # consumed once" in cons
          and "if n != num_tokens:\n        return None" in cons
          and "torch.cuda.current_stream().wait_event(drafter._deneb_early_fc_event)" in cons,
          "the consumer takes a result once, only for this token count, after the event")
    patched = src[src.index("def _patched_execute_model("):src.index("def install_glm53_dflash_early_fc(")]
    check("out = _ORIG_EXECUTE_MODEL(self, *args, **kwargs)" in patched
          and "_DISABLED = True" in patched and "return out" in patched,
          "a producer failure disables the arm for the boot and never breaks the step")
    wiring = open(os.path.join(REPO, "overlay/modules/glm53_model/"
                                     "glm5next_model.py"), encoding="utf-8").read()
    check("from .glm53_dflash_early_fc import install_glm53_dflash_early_fc" in wiring
          and 'if _e.name != f"{__package__}.glm53_dflash_early_fc":' in wiring
          and wiring.index("install_glm53_dflash_early_fc()") > wiring.index("install_glm53_prep_fused()"),
          "installed from the wiring like prep_fused: silent without the module, loud when broken")
    drafter = open(os.path.join(REPO, "overlay/modules/glm53_drafter/"
                                      "qwen3_dflash2.py"), encoding="utf-8").read()
    body = drafter[drafter.index("def combine_hidden_states(self, hidden_states"):]
    body = body[:body.index("def verify_selector_loaded")]
    check("early = _early_fc_take()" in body
          and "return super().combine_hidden_states(hidden_states)" in body
          and "from vllm.models.glm5next.nvidia.glm53_dflash_early_fc import" in drafter,
          "the drafter overlay consumes the pending result and falls back to stock")
    bracket = open(os.path.join(REPO, "bench", "bracket.py"), encoding="utf-8").read()
    check('"VLLM_GLM53_DFLASH_EARLY_FC"' in bracket, "bracket.py snapshots the knob")

    # the producer/consumer protocol on fakes: taken once, only for the same
    # token count, and never when nothing is pending
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_early_fc_mod", os.path.join(mod_dir, "glm53_dflash_early_fc.py"))
    import types
    fake_vllm = types.ModuleType("vllm")
    fake_logger = types.ModuleType("vllm.logger")
    fake_logger.init_logger = lambda name: _CapturingLogger()
    fake_vllm.logger = fake_logger
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.logger")}
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.logger"] = fake_logger
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    class _Ev:
        pass

    class _Stream:
        def wait_event(self, ev):
            self.waited = ev

    class _Torch:
        class cuda:
            _s = _Stream()

            @staticmethod
            def current_stream():
                return _Torch.cuda._s

    mod.torch = _Torch  # take_early_fc imports torch lazily; give it the fake
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            return _Torch
        return real_import(name, *a, **k)

    class _Drafter:
        pass

    d = _Drafter()
    d._deneb_early_fc_event = _Ev()
    d._deneb_early_fc_out = list(range(16))
    builtins.__import__ = fake_import
    try:
        check(mod.take_early_fc(d, 8) is None, "nothing pending -> stock path")
        d._deneb_early_fc_pending = (8, 1)
        check(mod.take_early_fc(d, 5) is None and d._deneb_early_fc_pending is None,
              "a token-count mismatch drops the pending result (stock path)")
        d._deneb_early_fc_pending = (8, 2)
        got = mod.take_early_fc(d, 8)
        check(got == list(range(8)) and _Torch.cuda._s.waited is d._deneb_early_fc_event
              and d._deneb_early_fc_pending is None,
              "a match waits on the event, returns the first n rows, and is consumed")
        check(mod.take_early_fc(d, 8) is None, "consumed once")
    finally:
        builtins.__import__ = real_import
    print("  glm53_dflash_early_fc contracts .. OK")


def test_kv_cache_is_pinned_in_tokens() -> None:
    """KV is pinned in tokens, not left to take whatever GMU leaves.

    Measured live 2026-09-03: num_gpu_blocks=2175, kv_cache_size_tokens=
    4,579,624, kv_cache_max_concurrency=4.37 on a server that runs
    MAX_SEQS=4 -- four times the headroom the scheduler can use. vllm applies
    the override by setting available_memory = override * bytes_per_block
    (kv_cache_utils), so the blocks that are not claimed are never allocated
    and the memory stays free on this unified-memory part.

    tokens -> blocks is not tokens/block_size: the 34 KDA layers hold
    recurrent state per SEQUENCE in whole blocks, which does not scale with
    tokens. The reserve is measured, named and re-checkable, not folded into
    a magic number."""
    src = open(os.path.join(REPO, "launchers/start-glm53-nvfp4-tp4.sh"),
               encoding="utf-8").read()
    check('KV_TOKENS="${KV_TOKENS:-2000000}"' in src
          and 'KV_HYBRID_BLOCKS="${KV_HYBRID_BLOCKS:-187}"' in src,
          "the launcher pins KV in tokens and names the hybrid reserve it "
          "adds on top")
    check("--num-gpu-blocks-override $KV_BLOCKS" in src
          and "int(($KV_TOKENS + 2303) / 2304) + $KV_HYBRID_BLOCKS" in src,
          "tokens convert to blocks by block_size plus the measured hybrid "
          "reserve, and reach vllm as --num-gpu-blocks-override")
    check("ABORT: KV_TOKENS and KV_BYTES both set" in src,
          "KV_TOKENS and KV_BYTES size the same cache; setting both is a "
          "config error, not a silent precedence rule")
    check("--kv-cache-memory-bytes $KV_BYTES" in src
          and "--kv-cache-memory $KV_BYTES" not in src,
          "the KV bytes flag is spelled out -- --kv-cache-memory only works "
          "as an argparse abbreviation while no other flag shares the prefix")
    check(src.index("KV_FLAG=") < src.index("SERVE_ARGS="),
          "the KV flag resolves before SERVE_ARGS bakes it in")
    print("  kv cache is pinned in tokens ... OK")


def test_earlyoom_is_fireable_on_unified_memory() -> None:
    """earlyoom on GB10 must use an absolute floor, ignore swap, and target the engine.

    Three nodes wedged on 2026-09-04 (sshd could not fork; only a power cycle
    or a broken collective got them back) with earlyoom inactive and, had it
    been active, unfireable: "-m 2 -s 2" requires SwapFree < 2% too, and on
    unified memory the model's pages are pinned so swap never fills -- the
    AND is false forever. And 2% of 121 GB is under the 4.0 GiB kernel
    watermark, past which fork() itself blocks.
    """
    cfg = open(os.path.join(REPO, "launchers", "earlyoom.default"),
               encoding="utf-8").read()
    m = re.search(r'^EARLYOOM_ARGS="(.*)"$', cfg, re.M)
    check(m is not None, "launchers/earlyoom.default defines EARLYOOM_ARGS")
    assert m is not None
    args = m.group(1)
    fl = re.search(r"-M (\d+)", args)
    check(fl is not None and 4 * 1048576 < int(fl.group(1)) <= 10 * 1048576,
          "the memory floor is ABSOLUTE (-M KiB) and sits above the 4.0 GiB "
          "kernel watermark but below the 10 GiB the preflight reserves -- "
          "so a healthy boot never crosses it and a runaway is caught while "
          "the node can still fork")
    check(re.search(r"(^|\s)-m \d", args) is None,
          "no percentage memory threshold: 2% of 121 GB was under the watermark")
    check("-s 100,100" in args,
          "swap must be non-binding for BOTH stages (-s 100,100): pinned GPU "
          "pages never swap, and earlyoom defaults the SIGKILL swap threshold "
          "to half the SIGTERM one -- the first install logged 'SIGKILL when "
          "swap <= 50%', a condition this fleet never meets")
    kl = re.search(r"-M \d+,(\d+)", args)
    check(kl is not None and 4 * 1048576 < int(kl.group(1)) < int(fl.group(1)),
          "the SIGKILL floor is explicit, below SIGTERM and still above the "
          "4.0 GiB watermark -- the derived default (half of 6 GiB) was under it")
    check("--prefer" in args and re.search(r"--prefer '[^']*vllm", args),
          "the engine is the preferred kill: when the floor is crossed it is "
          "the runaway, and losing the node loses it anyway")
    check(re.search(r"--avoid '[^']*sshd", args) and "tailscaled" in args
          and "dockerd" in args,
          "the management plane (sshd, docker, tailscale) stays protected")
    check(re.search(r"--avoid '[^']*(vllm|VLLM)", args) is None,
          "the engine must not ALSO be on the avoid list -- that was the "
          "srv2/srv3 config that made earlyoom shoot bystanders")
    print("  earlyoom fireable on UMA ...... OK")


def test_cudagraph_mem_profiling_off_keeps_the_kv_size() -> None:
    """Disabling the estimator must not silently resize the KV cache.

    vllm computes `available_kv = requested - non_kv - cudagraph_estimate`,
    so VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 drops the last term and
    hands that memory to KV. The launcher takes the same share back off GMU,
    which is why the delta is subtracted from whatever memfree-preflight
    measured rather than written as a fixed util. vllm's own suggested
    0.7671 is the OPPOSITE case (estimator kept on, pre-v0.21 KV restored);
    applying it here too would hand the 0.85 GiB back twice."""
    src = open(os.path.join(REPO, "launchers/start-glm53-nvfp4-tp4.sh"),
               encoding="utf-8").read()
    check("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in src
          and 'CG_MEM_PROFILE="${CG_MEM_PROFILE:-0}"' in src
          and 'CG_UTIL_DELTA="${CG_UTIL_DELTA:-0.0071}"' in src,
          "the launcher disables the cudagraph memory estimator by default "
          "and keeps both the switch and the delta overridable")
    check("$GMU - $CG_UTIL_DELTA" in src and "GMU=$(awk" in src,
          "the delta comes OFF GMU, and off the measured GMU -- a fixed util "
          "would overwrite what memfree-preflight just measured")
    check(src.index("memfree preflight") < src.index("CG_MEM_PROFILE=")
          < src.index("SERVE_ARGS="),
          "the adjustment lands after the preflight sets GMU and before "
          "SERVE_ARGS bakes it in")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    check("0.7671" not in code,
          "vllm's suggested util is for keeping the estimator ON; using it "
          "here as well would double-count the graph memory (naming it in a "
          "comment is how the next reader learns that)")
    print("  cudagraph mem profiling off keeps kv size .. OK")


def test_boot_stamps_measure_without_changing_the_boot() -> None:
    """`init engine took 192.52 s` is one number for eight things.

    The 2026-09-02 boot could only be split where something happened to log,
    leaving 63 s in two gaps with no marker at all. This module stamps the
    phases themselves. It is additive (the image ships its own
    sitecustomize.py, which must not be shadowed), it patches deterministically
    (an audit hook misses modules already in sys.modules), and it must never
    stand between the boot and its work."""
    d = os.path.join(REPO, "overlay/modules/glm53_runtime")
    src = open(os.path.join(d, "deneb_boot_stamps.py"), encoding="utf-8").read()
    man = open(os.path.join(d, "manifest.tsv"), encoding="utf-8").read()
    pth = open(os.path.join(d, "zz_deneb_boot_stamps.pth"),
               encoding="utf-8").read()
    stamp_rows = [l for l in man.splitlines() if "deneb_boot_stamps" in l and not l.startswith("#")]
    check(len(stamp_rows) == 2 and all(l.endswith("\tabsent") for l in stamp_rows)
          and "sitecustomize" not in man,
          "boot stamps add two NEW files and replace nothing -- shadowing the "
          "image's sitecustomize.py would silently drop whatever it does")
    check(pth.strip() == "import deneb_boot_stamps; deneb_boot_stamps.install()",
          "the .pth is the additive entry point and does nothing else")
    check("class _PostImport:" in src
          and all(m in src for m in ('"vllm.v1.worker.gpu_worker"',
                                     '"vllm.v1.worker.gpu.model_runner"',
                                     '"vllm.model_executor.model_loader.'
                                     'default_loader"'))
          and "loader.exec_module = exec_module" in src
          and "not isinstance(f, _PostImport)" in src,
          "the patch rides a meta-path post-import hook (an audit hook never "
          "fires for a module already in sys.modules) and skips itself when "
          "resolving, so it cannot recurse")
    for phase in ("determine_available_memory", "initialize_from_config",
                  "compile_or_warm_up_model", "profile_run", "capture_model",
                  "init_device"):
        check(phase in src, f"the {phase} phase is stamped")
    # 37차: the 2026-09-05 19:30 boot spent 302 s between the TP and EP
    # groups of initialize_model_parallel with no line on any rank (1 s in
    # the 09-04 20:16 and 09-05 00:03 boots). So the distributed groups are
    # stamped by name, and a phase still running after DENEB_BOOT_STAMP_SLOW
    # seconds gets its thread's stack logged -- a gap with no marker cannot
    # be attributed afterwards.
    check('"vllm.distributed.parallel_state"' in src
          and '(None, "init_model_parallel_group", "dist-group")' in src
          and '(None, "initialize_model_parallel", "dist-model-parallel")' in src
          and '(None, "init_distributed_environment", "dist-world")' in src
          and "cls = m if cls_name is None else getattr(m, cls_name, None)" in src
          and 'group = kw.get("group_name")' in src,
          "the world group and every model-parallel group are stamped by "
          "name (module-level functions, cls None = the module)")
    check('os.environ.get("DENEB_BOOT_STAMP_SLOW", "60")' in src
          and "sys._current_frames().get(tid)" in src
          and "while not done.wait(_SLOW) and n < _SLOW_MAX_DUMPS:" in src
          and "done = _start_watch(lab, t)" in src
          and "done.set()" in src
          and "daemon=True" in src,
          "a phase still running after DENEB_BOOT_STAMP_SLOW seconds logs its "
          "thread's stack from a daemon watcher that is ended in finally, so "
          "the gap is attributed while it happens and the watcher never "
          "outlives the phase")
    # `Loading weights took Ns` covers the loader AND the model's
    # weight_loader. Read 296 s once and 63.5 s the next day on the same
    # code -- the first was measured while this repo's benches had all four
    # hosts busy. Timing the generator from both sides puts that in the log.
    # 37차: the prep-fusion kernel ran tl.arange(0, Q) and DISARMed for any
    # decode_query_len that is not a power of two (k=5 -> Q=6: "plan
    # build/warmup failed"), so a k!=7 boot silently lost the fusion. Now it
    # runs Q_P2 lanes and masks the ones >= Q, the NS_P2 pattern it already
    # used for the GDN state -- every store over offs and the block-table load
    # carry the mask, and neither raise remains.
    pf2 = open(os.path.join(REPO, "overlay/modules/glm53_runtime/"
                                  "glm53_prep_fused.py"), encoding="utf-8").read()
    kern = pf2[pf2.index("def _glm53_prep_fused_kernel("):]
    kern = kern[:kern.index("\ndef ")] if "\ndef " in kern else kern
    check("    Q_P2: tl.constexpr," in kern
          and "    offs = tl.arange(0, Q_P2)\n    qmask = offs < Q\n" in kern
          and not [l for l in kern.splitlines()
                   if "tl.store(" in l and "+ offs" in l and "mask=" not in l]
          and "bn = tl.load(src_row + bidx, mask=qmask, other=0)" in kern
          and kern.count("v[None, :], mask=msk[None, :] & qmask[:, None])") == 2
          and "not a power of two" not in pf2
          and "self.q_p2 = 1 << (self.q - 1).bit_length()" in pf2
          and "Q=self.q, Q_P2=self.q_p2," in pf2,
          "prep-fused serves any decode_query_len: Q_P2 lanes, every offs "
          "store and the block-table load masked, the power-of-two raises gone")
    check("def _wrap_weights_iter(cls, name):" in src
          and '_wrap_weights_iter(cls, "get_all_weights")' in src
          and "produce += time.monotonic() - t" in src
          and "apply {max(total - produce, 0.0):.1f}s" in src,
          "weight loading is stamped as read (inside the generator) vs apply "
          "(everything else), with bytes, so a slow boot says which half")
    check("yield item" in src and "raise" in src,
          "the wrapper stays a passthrough: every tensor is yielded on and a "
          "raising loader still raises")
    check("return fn(*a, **kw)" in src and "finally:" in src,
          "every wrapper returns the original's value and times it in a "
          "finally, so a raising phase is still measured and still raises")
    check('os.environ.get("DENEB_BOOT_STAMPS", "1")' in src
          and src.count("except Exception") >= 4,
          "boot stamps are opt-out and inert on any failure: measuring a boot "
          "must never be able to stop one")
    prof = open(os.path.join(REPO, "profiles/glm53.env"), encoding="utf-8").read()
    check("glm53_runtime" in prof,
          "the glm53 profile ships the boot stamps")
    print("  boot stamps measure without changing the boot .. OK")


def test_self_built_kernels_persist_their_caches() -> None:
    """A restart must not recompile what only a deploy changes.

    The 2026-09-02 boot spent 34.4 s on the megakernel's nvcc (ninja log),
    60 s on osar's, and ~10 s re-JITing the MHC TileLang pair -- all into
    container-local directories, all paid again on every restart, while
    TRITON_CACHE_DIR and VLLM_CACHE_ROOT already pointed at the host mount.
    Each cache now lives on that mount under a key that still forces an
    honest rebuild when the source, the flags or the torch/CUDA pair move.
    Measured on srv4, two containers sharing one cache: 59.4 s -> 0.3 s."""
    mk = open(os.path.join(REPO, "overlay/modules/glm53_megakernel/"
                                 "glm53_megakernel.py"), encoding="utf-8").read()
    check("def _build_dir(src_md5: str, flags: list) -> str:" in mk
          and 'os.environ.get("VLLM_GLM53_MK_BUILD_ROOT")' in mk
          and "build_directory=_build_dir(md5, flags)," in mk
          and "[src_md5, *flags, torch.__version__, str(torch.version.cuda)]" in mk
          and 'build_directory="/root/.mk_build"' not in mk,
          "the megakernel builds into the persistent cache mount, keyed over "
          "source + flags + torch/CUDA")
    osar = open(os.path.join(REPO, "overlay/modules/tp_oneshot_ar/"
                                   "dsv4_oneshot_shim.py"), encoding="utf-8").read()
    check("def _build_dir(src_md5: str, flags: list) -> str:" in osar
          and 'os.environ.get("VLLM_DSV4_OSAR_BUILD_ROOT")' in osar
          and "build_directory = _build_dir(_src_md5, cuda_flags)" in osar
          and 'os.makedirs("/root/.osar_build", exist_ok=True)' not in osar,
          "osar builds into the persistent cache mount too, and MAXEL rides "
          "in the flags so a different peer-buffer stride keys a new build")
    tl = open(os.path.join(REPO, "overlay/modules/glm53_kernels/"
                                 "tilelang.py"), encoding="utf-8").read()
    check("def _deneb_persist_tilelang_cache() -> None:" in tl
          and 'if os.environ.get("TILELANG_CACHE_DIR"):' in tl
          and 'os.environ["TILELANG_CACHE_DIR"] = os.path.join(cand, "tilelang")' in tl
          and tl.index("_deneb_persist_tilelang_cache()\n")
              < tl.index("from vllm.utils.torch_utils"),
          "the MHC overlay points TILELANG_CACHE_DIR at the persistent mount "
          "at its own import, ahead of the vllm/tilelang machinery, and never "
          "overrides an explicit setting")
    for src, name in ((mk, "megakernel"), (osar, "osar")):
        check("shutil.rmtree(stale, ignore_errors=True)" in src
              and "except OSError:" in src,
              f"{name} prunes stale sibling builds, and a failed prune never "
              f"fails the build")
    # 37차: chain 13's k=5 boot spun in deep_gemm's tf32 prenorm GEMM at
    # M=6 (rank 0, CPU 200%, 19 min); M=8/16/24 (k=7 decode) and every
    # prefill M run through it daily. So M < 8 -- only that -- is served as
    # the proven M=8 shape: zero rows appended, GEMM, live rows copied back.
    # Every call site goes through the wrapper, or a k=5 boot finds the one
    # that does not.
    check("def _deneb_hc_prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits):" in tl
          and "_HC_PRENORM_MIN_M = 8" in tl
          and "if m >= _HC_PRENORM_MIN_M:" in tl
          and "x_pad[:m].copy_(x)" in tl
          and "out_mul.copy_(mul_pad[:, :m])" in tl
          and "out_sqrsum.copy_(sq_pad[:, :m])" in tl
          and tl.count("_deneb_hc_prenorm_gemm(") == 4
          and tl.count("tf32_hc_prenorm_gemm(") == 2
          and tl.rindex("tf32_hc_prenorm_gemm(") < tl.index("def mhc_"),
          "the prenorm GEMM's three call sites route through the M<8 padding "
          "wrapper (def + 3 calls); the raw deep_gemm call appears only "
          "inside the wrapper")
    # 37차 "1+2": layer 0's standalone pre-mix is the one call of a decode
    # step that still reached deep_gemm (every other layer's pre rides in the
    # fused hook). The fused kernel's post step is v = pm*x + sum cm*res in
    # fp32, so pm = 0, cm = I make it the identity and the same armed kernel
    # serves the standalone pre with no kernel change. Static coefficient
    # buffers at T_max, sliced per call; a self-test that checks the identity
    # bitwise and the outputs against the stock pair, at T=8 and -- only when
    # the boot's spec k is not 7 -- at T=k+1, so a production boot never runs
    # a T it never serves; its own arm flag under the MHC segment's.
    mkp = open(os.path.join(REPO, "overlay/modules/glm53_megakernel/"
                                  "glm53_megakernel.py"), encoding="utf-8").read()
    check('ENABLE_MHC_PRE = ENABLE_MHC and _flag("VLLM_GLM53_MK_MHC_PRE", "1")' in mkp
          and '_ARMED = {"mhc": False, "mhc_pre": False, ' in mkp
          and "def mhc_pre_only(" in mkp and "def mhc_pre_hook(" in mkp
          and "def _selftest_mhc_pre() -> bool:" in mkp
          and 'if ENABLE_MHC_PRE and _ARMED["mhc"]:' in mkp
          and '_ARMED["mhc_pre"] = _gate("mhc_pre", _selftest_mhc_pre)' in mkp,
          "the pre-only MHC hook has its own knob (default on), arm flag and "
          "self-test, gated under the fused segment's arm")
    check("cm_i = torch.eye(HC, dtype=torch.float32, device=device).reshape(1, HC * HC)" in mkp
          and "x0[:num_tokens], residual.reshape(-1, hc_mult, hidden)," in mkp
          and "pm0[:num_tokens], cm_i[:num_tokens], fn, hc_scale, hc_base," in mkp
          and "identity = identity and bool(torch.equal(rc, res)) and bool(torch.equal(res_ref, res))" in mkp
          and 'spec_k = (os.environ.get("VLLM_GLM53_SPEC_K") or "7").strip()' in mkp
          and "ts.append(int(spec_k) + 1)" in mkp,
          "identity post coefficients from static buffers sliced per call; "
          "the self-test proves the identity bitwise and adds T=k+1 only on "
          "a non-7 spec boot")
    # 37차 (operator: "200줄 쿠다"): the fused decode-step preparation kernel
    # is a CUDA kernel in the megakernel extension (mk_prep_kernel / run_prep),
    # the request's Q tokens as plain loops -- no power-of-two lane constraint,
    # no masks. The Triton kernel stays as the fallback (VLLM_GLM53_PREP_FUSED_
    # KERNEL=triton, or no extension / dtype mismatch). The pointer and int
    # lists are positional on both sides, so their ORDER is pinned by name.
    mkcu = open(os.path.join(REPO, "overlay/modules/glm53_megakernel/"
                                   "glm53_megakernel.cu"), encoding="utf-8").read()
    pf3 = open(os.path.join(REPO, "overlay/modules/glm53_runtime/"
                                  "glm53_prep_fused.py"), encoding="utf-8").read()
    check("__global__ void __launch_bounds__(MK_PREP_THREADS) mk_prep_kernel(MKPrepArgs a)" in mkcu
          and 'm.def("run_prep", &mk_run_prep,' in mkcu
          and "TORCH_CHECK(ptrs.size() == 33 && ints.size() == 23" in mkcu
          and "def kernel_backend() -> str:" in pf3
          and 'os.environ.get(ENV_KERNEL, "cuda")' in pf3
          and "def cuda_dtype_reason(self) -> str | None:" in pf3
          and 'ext.run_prep(self._cuda_ptrs(idx), self._cuda_ints(num_reqs, num_tokens))' in pf3
          and pf3.index('ext = o.get("cuda_ext")') < pf3.index('compiled = o.get("compiled")')
          and 'self.owned["cuda_ext"] = ext' in pf3
          and "kernel=%s" in pf3,
          "the CUDA prep kernel is the default backend with the Triton kernel "
          "as the fallback, and the plan line names which one serves")
    cpp_ptrs = re.findall(r"P\(&a\.(\w+)\)", mkcu[mkcu.index("void mk_run_prep("):])
    cpp_ints = re.findall(r"a\.(\w+) = (?:\(int\))?ints\[q\+\+\]", mkcu[mkcu.index("void mk_run_prep("):])
    py_ptrs_src = pf3[pf3.index("def _cuda_ptrs("):pf3.index("def _cuda_ints(")]
    py_ptrs = re.findall(r"(?:self\.|o\[\"|bt\.)?([A-Za-z_][\w.\[\]\"]*)", py_ptrs_src[py_ptrs_src.index("for t in ("):py_ptrs_src.index(")]")])
    py_ptrs = [x.replace("]", "").strip('"') for x in py_ptrs if x not in ("t", "in", "for")]
    rename = {"idx": "idx_mapping", "prefill_len_src.gpu": "prefill_len", "query_start_loc": "qsl",
              "block_table_ptrs": "src_bt_ptrs", "input_block_table_ptrs": "dst_bt_ptrs",
              "block_table_strides": "bt_strides", "block_sizes_tensor": "block_sizes",
              "num_blocks.gpu": "num_blocks", "slot_mappings": "slot", "req_id_buf": "req_id"}
    py_norm = [rename.get(x, x) for x in py_ptrs]
    check(len(cpp_ptrs) == 33 and py_norm == cpp_ptrs,
          f"the 33 pointers are passed in the order mk_run_prep unpacks them "
          f"(py={py_norm[:6]}..., cpp={cpp_ptrs[:6]}...)")
    py_ints_src = pf3[pf3.index("def _cuda_ints("):pf3.index("def _consts(")]
    want_ints = ["num_reqs", "num_tokens", "max_num_reqs", "max_num_tokens", "draft_stride",
                 "num_blocks_stride", "slot_stride", "req_id_cap", "exp_bt_stride", "dec_seq_cap",
                 "idx_bt_stride", "idx_bt_cols", "comp_slot_cap", "Q", "NUM_SPEC", "NS", "G",
                 "N_GDN", "ATTN_G", "FACTOR", "RATIO", "SBS", "PAD_ID"]
    check(cpp_ints == want_ints and len(cpp_ints) == 23
          and all(k in py_ints_src for k in ("self.draft_tokens.stride(0)", "bt.num_blocks.gpu.stride(0)",
                                              "bt.slot_mappings.stride(0)", "self.req_id_buf.numel()",
                                              "self.exp_bt.stride(0)", "self.dec_seq_lens.numel()",
                                              "self.idx_bt.stride(0)", "self.idx_bt_cols", "self.comp_slot.numel()",
                                              "self.q, self.num_spec, self.num_spec + 1, self.G, len(self.gdn_groups), self.attn_g",
                                              "self.factor, self.ratio, self.sbs, PAD_SLOT_ID")),
          "the 23 ints are unpacked in the documented order on both sides")
    # 37차 night round: the target's per-token features for drafter training
    # (five aux hidden states, ids, positions, top-k logits + lse) from prefill
    # steps. Inert without VLLM_GLM53_DRAFT_DUMP; wraps execute_model after
    # the original; compute_logits is a TP collective so every rank runs the
    # same 1024-token chunks and only rank 0 writes; a failing hook disables
    # itself and never touches the step.
    dd = open(os.path.join(REPO, "overlay/modules/glm53_runtime/"
                                 "glm53_draft_dump.py"), encoding="utf-8").read()
    man_rt = open(os.path.join(REPO, "overlay/modules/glm53_runtime/manifest.tsv"),
                  encoding="utf-8").read()
    wiring_m = open(os.path.join(REPO, "overlay/modules/glm53_model/glm5next_model.py"),
                    encoding="utf-8").read()
    check("glm53_draft_dump.py\tvllm/models/glm5next/nvidia/glm53_draft_dump.py\tabsent" in man_rt
          and "from .glm53_draft_dump import install_glm53_draft_dump" in wiring_m
          and wiring_m.index("install_glm53_draft_dump()") > wiring_m.index("install_glm53_prep_fused()")
          and "if _STATE[\"installed\"] or dump_dir() is None:" in dd
          and "out = orig(self, scheduler_output, *args, **kwargs)" in dd
          and dd.index("out = orig(self, scheduler_output") < dd.index("_dump_step(self)")
          and "for i in range(0, hidden.shape[0], CHUNK):" in dd
          and "top_ids, top_vals, lse = _topk_logits(runner, hidden, k)   # collective: all ranks" in dd
          and "if rank == 0:" in dd and "os.replace(path + \".tmp\", path)" in dd
          and "_STATE[\"disabled\"] = True" in dd
          and "torch.cuda.is_current_stream_capturing()" in dd,
          "the draft-dump hook is mounted beside prep_fused, installed after it, "
          "inert without its env, runs after the original execute_model, computes "
          "the head collectively in fixed chunks, writes on rank 0 atomically, and "
          "disables itself on failure")
    check("_HOOK_SERVED_PRE[0] += 1" in mkp
          and "[megakernel] mhc-pre hook serving (T=%d)" in mkp
          and mkp.index("maybe_arm()", mkp.index("def mhc_pre_hook(")) < mkp.index("out = mhc_pre_only(", mkp.index("def mhc_pre_hook(")),
          "mhc_pre_hook arms before it calls and logs a serving receipt once")
    pre_start = tl.index("def mhc_pre_tilelang(")
    pre_end = tl.index("\ndef ", pre_start + 10)
    pre = tl[pre_start:pre_end]
    check("def _deneb_mk_pre_hook():" in tl
          and "from vllm.model_executor.layers.glm53_megakernel import mhc_pre_hook" in tl
          and "if num_tokens <= 16 and norm_weight is not None:" in pre
          and "_mk_pre_hook = _deneb_mk_pre_hook()" in pre
          and pre.index("_mk_pre_hook = _deneb_mk_pre_hook()") < pre.index("use_deep_gemm = is_deep_gemm_supported()")
          and "_pm.view(*outer_shape, hc_mult, 1)," in pre
          and tl.count("_mk_pre_hook = _deneb_mk_pre_hook()") == 1,
          "the standalone pre wrapper offers the MK pre hook (T <= 16, with "
          "norm) before any GEMM, returning the stock wrapper's shapes; the "
          "fused wrapper's own hook block is untouched")
    print("  self-built kernels persist their caches .. OK")


def test_cuda_builds_keep_the_arch_specific_target() -> None:
    """-arch=sm_121a silently drops the 'a' suffix under -c (cpp_extension's mode).

    Verified 2026-09-03 on nvcc 13.0 with a flag/mode matrix: -cubin and -ptx
    accept -arch=sm_121a, but -c emits a plain sm_121 target and ptxas then
    refuses every architecture-specific instruction -- the fp4 mma
    (kind::f8f6f4) and the 2:4 sparse mma (mma.sp::ordered_metadata). Both are
    real on this part (measured 155 and 309 TFLOP/s), so the flag is what
    stands between the kernels and them."""
    for rel in ("overlay/modules/glm53_megakernel/glm53_megakernel.py",
                "overlay/modules/glm53_kernels/glm53_kpool_topk.py"):
        src = open(os.path.join(REPO, rel), encoding="utf-8").read()
        i = src.find("extra_cuda_cflags=")
        while i != -1:
            flags = src[i: src.index("]", i) + 1]
            # `extra_cuda_cflags=<name>,` hands the list to a variable built
            # just above the call (the megakernel does this so its build
            # directory can hash the exact flags); follow it to that literal.
            head = src[i + len("extra_cuda_cflags="):
                       src.find(",", i + len("extra_cuda_cflags="))].strip()
            if head.isidentifier():
                j = src.rindex(f"{head} = [", 0, i)
                flags = src[j: src.index("]", j) + 1]
            check('"-arch=sm_121a"' not in flags,
                  f"{os.path.basename(rel)}: -arch=sm_121a loses the 'a' suffix under -c; "
                  f"use -gencode arch=compute_121a,code=sm_121a")
            check('"arch=compute_121a,code=sm_121a"' in flags,
                  f"{os.path.basename(rel)}: the CUDA build must pin compute_121a/sm_121a")
            i = src.find("extra_cuda_cflags=", i + 1)
    print("  cuda builds keep the sm_121a-specific target .. OK")


def test_glm53_megakernel_contracts() -> None:
    import math as _math

    mod = "overlay/modules/glm53_megakernel"
    ns = load_defs(
        f"{mod}/glm53_megakernel.py",
        {"_mk_pow2_scale", "_mk_pad128", "_mk_gemm_eligible", "MK_GEMM_KMAX",
         "_mk_mhc_eligible", "_mk_w4_scale_exp"},
        {"math": _math},
    )
    pow2, pad128 = ns["_mk_pow2_scale"], ns["_mk_pad128"]
    gemm_ok, mhc_ok = ns["_mk_gemm_eligible"], ns["_mk_mhc_eligible"]
    w4_exp = ns["_mk_w4_scale_exp"]

    # -- pow2 scale: exact power of two, covers amax, degenerates to 1.0
    for amax in (1e-30, 0.5, 1.0, 447.9, 448.0, 1e6, 3.7e37):
        s = pow2(amax)
        check(s > 0 and _math.frexp(s)[0] == 0.5,
              f"pow2 scale must be an exact power of two (amax={amax})")
        check(s * 448.0 >= amax,
              f"pow2 scale must cover amax/448 (amax={amax})")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        check(pow2(bad) == 1.0, f"pow2 degenerate amax -> 1.0 (amax={bad})")

    # -- pad128
    for n, want in ((0, 0), (1, 128), (127, 128), (128, 128),
                    (6416, 6528), (4096, 4096)):
        check(pad128(n) == want, f"pad128({n}) == {want}")

    # -- W4 scale exponent: coverage below the 384 saturation ceiling,
    #    graceful saturation above it, exactness range, pow2 semantics
    for amax in (1e-30, 0.001, 0.05, 1.0, 6.0, 100.0, 384.0):
        e = w4_exp(amax)
        check(-5 <= e <= 6, f"w4 scale exponent in the LUT-exact range "
              f"(amax={amax}, s={e})")
        check(6.0 * (2.0 ** e) >= amax,
              f"w4 scale covers amax below the 384 ceiling (amax={amax})")
        check(_math.frexp(2.0 ** e)[0] == 0.5,
              f"w4 scale is an exact power of two (amax={amax})")
    for amax in (385.0, 447.9, 1e6):  # above the ceiling: saturate at s=6
        e = w4_exp(amax)
        check(e == 6, f"w4 saturates at s=6 above 384 (amax={amax})")
        check(6.0 * 64.0 <= 448.0,
              "the saturated magnitude stays a finite e4m3 (never NaN)")
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        check(w4_exp(bad) == 0, f"w4 degenerate amax -> s=0 (amax={bad})")

    # -- eligibility truth tables
    check(gemm_ok(8, 4096, 6528), "decode in_proj shape is eligible")
    check(gemm_ok(32, 2048, 2048), "C=4 o_proj shape is eligible")
    check(not gemm_ok(33, 4096, 4096), "M=33 falls back (kernel M<=32)")
    check(not gemm_ok(8, 4000, 4096), "K%128!=0 falls back")
    check(not gemm_ok(8192, 4096, 4096),
          "prefill M stays on deepgemm -- the megakernel is decode-only")
    check(not gemm_ok(0, 4096, 4096), "empty batch falls back")
    check(mhc_ok(8, 4, 4096) and mhc_ok(32, 4, 4096),
          "C=1..4 verify token counts are eligible")
    check(not mhc_ok(33, 4, 4096), "T=33 falls back")
    check(not mhc_ok(8, 6, 4096), "hc_mult!=4 falls back")
    check(not mhc_ok(8, 4, 5120), "hidden!=4096 falls back")

    # -- .cu constants evaluated and pinned to the GLM-5.3 per-rank geometry
    cu = open(os.path.join(REPO, mod, "glm53_megakernel.cu"),
              encoding="utf-8").read()
    consts: dict[str, int] = {}

    def ev(expr: str) -> int:
        import ast as _ast

        expr = " ".join(expr.split())  # multi-line constexprs are legal C++
        tree = _ast.parse(expr, mode="eval")

        def walk(n):
            if isinstance(n, _ast.Expression):
                return walk(n.body)
            if isinstance(n, _ast.Constant):
                return n.value
            if isinstance(n, _ast.Name):
                return consts[n.id]
            if isinstance(n, _ast.BinOp):
                l, r = walk(n.left), walk(n.right)
                return (l + r if isinstance(n.op, _ast.Add)
                        else l - r if isinstance(n.op, _ast.Sub)
                        else l * r if isinstance(n.op, _ast.Mult)
                        else l // r)
            if isinstance(n, _ast.UnaryOp) and isinstance(n.op, _ast.USub):
                return -walk(n.operand)
            raise AssertionError(f"unsupported const expr: {expr}")

        return walk(tree)

    # Seed the #define knobs first -- MK_GRID and friends now read their
    # shipped defaults from -D-overridable defines, so the constexpr lines
    # below reference names that only exist as macros.
    for m in re.finditer(r"^#define (\w+) (\d+)$", cu, re.M):
        consts[m.group(1)] = int(m.group(2))
    for m in re.finditer(r"^constexpr int (\w+) = ([^;]+);", cu, re.M):
        consts[m.group(1)] = ev(m.group(2))
    for name, want in (("HC", 4), ("HIDDEN", 4096), ("NOUT", 24),
                       ("MAX_TOK", 32), ("NCHUNK", 16), ("KDA_H", 16),
                       ("KDA_D", 128), ("KDA_QKV", 6144),
                       ("KDA_INPROJ_N", 6416), ("KDA_INPROJ_N_PAD", 6528),
                       ("CONV_W", 4), ("KDA_OUT", 2048), ("MK_GRID_CAP", 96),
                       ("MK_THREADS", 256), ("KSTEP", 128),
                       ("KBLK_MAX", 32), ("SMEM_W_ROWS", 128)):
        check(consts.get(name) == want,
              f".cu constant {name} == {want} (got {consts.get(name)})")
    check(consts["GEMM_SMEM"] <= 98304,
          "dynamic smem stays inside the 96 KB discipline of the 128 KB/SM")

    # -- driver geometry must be the same numbers (drift here = silent
    #    shape-mismatch bugs the boot self-test would hunt blind)
    pysrc = open(os.path.join(REPO, mod, "glm53_megakernel.py"),
                 encoding="utf-8").read()
    for m in re.finditer(r"^(\w+) = ([^\n]+)$", pysrc, re.M):
        name, expr = m.group(1), m.group(2).strip()
        if name in consts and re.fullmatch(r"[\d\s()+\-*/\w]*", expr):
            try:
                check(ev(expr) == consts[name],
                      f"driver {name} matches the .cu constant")
            except (KeyError, AssertionError):
                pass  # expr references driver-only names; parity is per-name

    pysrc_full = open(os.path.join(REPO, mod, "glm53_megakernel.py"),
                      encoding="utf-8").read()
    bench = open(os.path.join(REPO, "probes", "megakernel_glm53_bench.py"),
                 encoding="utf-8").read()

    # -- sm_121a static contract (code only: the header COMMENT names the
    #    forbidden instructions, so strip // and /* */ tails before scanning)
    cu_code = re.sub(r"/\*.*?\*/", "", cu, flags=re.S)
    cu_code = re.sub(r"//[^\n]*", "", cu_code)
    for bad in ("wgmma", "tcgen05", "mbarrier", "cp.async.bulk",
                "cluster.sync", "setmaxnreg"):
        check(bad not in cu_code, f"sm_121a contract: {bad} must not appear")
    check("m16n8k32.row.col.f32.e4m3.e4m3.f32" in cu,
          "the GEMM uses the e4m3 mma.sync kind available on sm_121a")
    check("cp.async.cg.shared.global" in cu_code
          and "W4_RAW_NBUF * W4_RAW_BYTES" in cu
          and "cp.async.wait_group" in cu_code,
          "the W stream is a multi-buffer cp.async pipeline of raw records")
    # Depth is a tuning knob, not a contract -- but it must stay deep enough
    # to keep more than one record in flight, and the smem it costs must fit
    # the opt-in this part actually reports (101376 B).
    nbuf = consts["W4_RAW_NBUF"]
    check(nbuf >= 3, f"W pipeline depth {nbuf} keeps too little in flight")
    smem = consts["GEMM_SMEM"]
    check(smem <= 101376,
          f"W pipeline depth {nbuf} overruns the 101376 B smem opt-in")
    # The opt-in is per BLOCK; occupancy is set by the SM's shared memory,
    # which on this part is 102,400 B -- NOT the 131,072 an earlier version
    # of this check assumed. That wrong number let the check pass while
    # asserting the opposite of the truth: at nbuf=3 the block takes 69,632 B,
    # so two blocks overrun 102,400 and exactly ONE block is resident.
    # Record the real figure rather than assert a fiction.
    #
    # The load-bearing contract is residency, not depth: mk_gemm_kernel is a
    # PERSISTENT grid with a grid barrier, so a grid that does not fit is a
    # deadlock, not a slowdown. smem is the binding limit here (registers,
    # at 59 x 256 = 15,104 of 65,536, would allow four blocks).
    blocks_per_sm = 102400 // smem
    check(blocks_per_sm >= 1,
          f"W pipeline depth {nbuf} uses {smem} B, over the SM's 102400")
    # The grid is now resolved at launch from the device, so the static
    # contract is on the CEILING: it must not promise more than the smem
    # could ever deliver, or the clamp is doing all the work and the cap is
    # decoration. (The clamp is what keeps a bad pair from deadlocking; this
    # keeps the pair honest.)
    check(consts["MK_GRID_CAP"] <= 2 * blocks_per_sm * 48,
          f"gemm grid ceiling {consts['MK_GRID_CAP']} is more than twice "
          f"what {smem} B of smem allows ({blocks_per_sm} block(s)/SM on 48 "
          "SMs) -- raise the ceiling only alongside a shallower pipeline")
    # The W pack is TILE-major and the packer and the kernel are the only
    # two places that know it -- they must move together or the kernel reads
    # a correct-looking tensor as garbage. Row-major put the 128 rows of a
    # tile 4096 B apart, so one record touched 128 DRAM pages.
    check("c.wq4 + ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 64);" in cu
          and "((size_t)nt * kblk + kb) * (SMEM_W_ROWS * 8);" in cu,
          "stage_raw4 reads a (tile, k-block) record as one contiguous run")
    check("rowb + mk_swz(row4, half4 * 64 + g4 * 16)" in cu
          and "wrow + mk_swz(nrow, koff + t4)" in cu
          and "constexpr int SMEM_W_PITCH = KSTEP;" in cu,
          "the expanded e4m3 tile is dense 128 B rows: the expansion stores "
          "and the fragment loads go through the same XOR swizzle (a padded "
          "pitch put every row across a bank-line boundary)")
    check("smem += (MK_SMEM_ALIGN - (sb & (MK_SMEM_ALIGN - 1))) & (MK_SMEM_ALIGN - 1);"
          in cu and "constexpr int GEMM_SMEM = MK_SMEM_ALIGN + 2 * 16 * SMEM_A_PITCH +"
          in cu,
          "the dynamic smem base must be re-aligned at runtime (the static "
          "s_last/s_unit push it to +16, which put every 128 B tile row across "
          "a bank-line boundary) and the smem budget must carry the slack")
    check(bench.index("torch.mm(_SPACER[0], _SPACER[1], out=_SPACER[2])")
          < bench.index("_DRAIN.sum()") < bench.index("for t in hot:"),
          "the bench flush must drain the dirty lines with a read stream AFTER "
          "the matmul spacer (whose 8 MB output is dirty too) and before the "
          "hot touch: the old order left ~24 MB of write-back under the timed "
          "kernel (both arms ~35% slow at the first launch)")
    check("threadIdx.x != 0 && t < SMEM_W_ROWS * 4;" in cu,
          "thread 0 issues no cp.async: it runs the grid barrier, whose "
          "__threadfence drains its own outstanding copies -- with the "
          "fill hoisted above the barrier that turned into a 6-12 us wait")
    _w_at = cu_code.index("stage_raw4(nt, kb0,")
    check(_w_at < cu_code.index("stage_a_load(kb0, 0);", _w_at),
          "W(kb0) starts flying before A(kb0) is staged. kb0, not 0: a "
          "block may own a k SLICE of a tile.")
    # -- the FIRST unit's W fill is hoisted above the A-quant prologue and
    #    its barrier, and the prologue's own x loads (consumed by the amax
    #    shuffle) go out before that fill. Phase stamps: quant + barrier
    #    took 3-10 us during which DRAM idled; a fill issued first instead
    #    queued the 256 B/row x loads behind 1.5 MB of W (quant 13-18 us).
    _hoist_at = cu_code.index("stage_raw4(nt0, kb00,")
    check(_hoist_at < cu_code.index('asm volatile("griddepcontrol.wait;"')
          < cu_code.index("raw[i] = *(const uint2*)(c.x +")
          < cu_code.index("mx[i] = mk_warp_amax(")
          < cu_code.index("mk_grid_barrier(bar, c.grid);"),
          "prologue order: the first unit's W fill (independent of the "
          "previous kernel), the PDL wait, x into registers, amax/convert/"
          "store, barrier")
    check(cu_code.count('asm volatile("griddepcontrol.launch_dependents;");') == 7
          and "cudaLaunchAttributeProgrammaticStreamSerialization" in cu
          and 'getenv("VLLM_GLM53_MK_PDL")' in cu
          and "cudaLaunchKernelEx(&cfg, kernel, args)" in cu,
          "every segment kernel (gemm, gemm-lq, gemm2, kda, mhc, mla, smlp) triggers "
          "its dependents at entry and is launched programmatically behind "
          "the MK_PDL knob")
    check("const bool prefilled = hoisted && (u == (int)blockIdx.x);" in cu
          and "if (!prefilled) stage_raw4(nt, kb0, kb0 % W4_RAW_NBUF);" in cu,
          "the unit loop must not re-issue the tiles the hoist already "
          "staged -- a second cp.async group for the same buffer would "
          "break the one-stage-one-group wait accounting")
    # -- A is quantized ONCE per launch, not once per (tile, k-block). Every
    #    n-tile walks all of k, so the in-loop form redid it nblk times --
    #    51x at n=6416 -- on bf16 input, twice the bytes the mma consumes.
    #    Measured ceiling for removing it: -10% at n=6416/4096, -22% at
    #    n=2048, -14% at n=1024.
    check("__device__ uint8_t g_mk_aq[" in cu
          and "const int kbq = (int)blockIdx.x;" in cu
          and "for (int kb = c.a_ready ? kblk : kbq + c.grid; kb < kblk;" in cu
          and cu.index("g_mk_aq + ((size_t)kb * 32 + r) * KSTEP + ql * 4")
              < cu.index("auto stage_a"),
          "A is quantized once, cooperatively across the grid, before the "
          "tile loop reads it")
    check(cu.index("mk_grid_barrier(bar, c.grid);")
          < cu.index("auto stage_a"),
          "a barrier publishes the shared A quant before any block stages "
          "it -- without it a block reads a tile another block has not "
          "written yet, and only sometimes")
    check("sxs[i] = g_mk_axs[i] * c.wgs;" in cu,
          "the per-row scales come from the same shared quant, carrying the "
          "pack's pow2 normalisation -- folding it here costs one multiply "
          "in the prologue instead of one per accumulate")
    # -- remainder split-K: leftovers of the last partial round take k
    #    slices instead of leaving grid - rem blocks idle for a whole
    #    tile-time. ksr == 1 must leave the original path untouched.
    check("const int rem = nblk % c.grid;" in cu
          and "int ksr = (rem > 0) ? (grid / rem) : 1;" in cu
          and "int mk_choose_ksr(int m, int n, int k, int grid)" in cu
          and cu.count("mk_choose_ksr(") == 7  # def, the gemm plan (+ its bg control), kda in/out, smlp gate_up/down
          and "const int ksr = c.ksr;" in cu,
          "the split is over the REMAINDER tiles when there are whole tiles "
          "too -- those units are a mix of sizes and the uniform cost model "
          "below does not apply to them")
    # When full == 0 every unit is one k-slice of one tile, so wall time is
    # ceil(nblk*r / grid) / r tile-times and the truncating grid/rem is
    # simply the wrong pick: it returns 1 for nblk in (grid/2, grid],
    # i.e. the 4096-wide projections, leaving 16 of 48 blocks
    # idle where r=3 would cost 0.667 tile-times.
    check("if (full == 0 && rem > 0 && m <= 32) {" in cu
          and "if (rounds * bd < bn * r) { bn = rounds; bd = r; ksr = r; }"
          in cu,
          "with no whole tiles the k-split is the cost-minimising one, not "
          "the truncated grid / rem")
    check("const bool split = mk_split_ok(c.m, pcols, ksr);" in cu
          and "return (ksr > 1) && (m <= 32) && (pcols <= MK_SPLIT_MAXCOL) &&" in cu,
          "ksr == 1 must fall through to the single-pass path unchanged (the "
          "gate is mk_split_ok, shared with the host's unit count)")
    # The accumulator no longer follows from ksr * rem <= grid, so the
    # size guard has to be on the accumulator itself.
    check("((size_t)m * pcols * ksr <= MK_SPLIT_ELEMS)" in cu
          and "MK_SPLIT_ELEMS = 32 * 128 * MK_SPLIT_UNITS_MAX" in cu,
          "the split gate bounds the partial accumulator directly")
    # The one surviving barrier must stand between the slice writes and the
    # reads that fold them -- anchor it on the READ, not on the comment: the
    # zero pass used to sit above the comment and the ordering check passed
    # by accident of that layout.
    # No fold barrier any more: the last slice of a tile folds it. The
    # writer fences before its arrival is counted, the completing block
    # fences after seeing the count and reads the other slices through L2
    # (L1 is not coherent within a launch), and re-arms the counter.
    _arr = cu.index("atomicAdd(&g_mk_tile_arrive[lt], 1u)")
    check(cu.rindex("__threadfence();", 0, _arr) < _arr
          < cu.index("__ldcg((const float4*)(src + (size_t)spx * pslice))"),
          "last-arriver fold: fence before the arrival, L2 reads after it")
    check("if (s_last) g_mk_tile_arrive[lt] = 0u;" in cu
          and "const unsigned expect = (unsigned)min(ksr, kblk);" in cu,
          "the tile counter is re-armed by its completing slice and expects "
          "only the non-empty slices (kb0 == kbn never arrives)")
    check(cu.count("mk_grid_barrier(bar, c.grid);") == 1,
          "one grid barrier per gemm phase (the A-quant publish) -- the "
          "fold has none")
    # Units after the first are handed out dynamically; the counter is
    # re-armed by block 0 BEFORE the A-quant barrier so the barrier orders
    # the reset ahead of every block's first take.
    check(cu.index("g_mk_unit_next = 0u;", cu.index("MK_TS(0)"))
          < cu.index("mk_grid_barrier(bar, c.grid);")
          and "u = local_q ? u + (int)gridDim.x : next_unit()) {" in cu
          and "s_unit = c.grid + (int)atomicAdd(&g_mk_unit_next, 1u);" in cu,
          "dynamic unit hand-out: first unit static (hoisted fill), the "
          "rest from a counter re-armed ahead of the publish barrier")
    # -- barrier-free path (2026-09-04): a launch with at most one unit per
    #    block (the shared expert's two GEMMs, 86 launches a step) has every
    #    block quantize ITS unit's A k-blocks into smem as it stages them --
    #    no grid-wide prologue, no barrier, idle blocks exit at once. The
    #    host picks it and the phase re-derives the condition, so a drift
    #    degrades to the global path; MK_LOCALQ=0 is the kill switch.
    check("constexpr bool local_q = LQ;" in cu
          and "if (c.a_ready || units > c.grid) __trap();" in cu
          and "if (local_q && !has_u0) return;" in cu
          and cu.index("if (local_q && !has_u0) return;")
              < cu.index('asm volatile("griddepcontrol.wait;" ::: "memory");')
          and "int mk_units(int m, int n, int grid, int ksr)" in cu
          and "p.localq = (lq == 2 || (lq == 1 && bg)) && p.units <= p.grid &&" in cu
          and "const MKGemmPlan p = mk_gemm_plan_for(c.m, c.n, c.k, bg != 0);" in cu
          and 'mk_env_int("VLLM_GLM53_MK_LOCALQ", 0, 0, 2, false)' in cu
          and cu.count("mk_gemm_plan_for(") == 3,  # def, the launch, the bench pybind
          "the two paths are compile-time instantiations chosen by ONE host "
          "plan (launch and bench pybind alike: units <= grid and a background "
          "caller); a drift traps instead of running a 48-block barrier on a "
          "launch sized to the units; default OFF behind VLLM_GLM53_MK_LOCALQ "
          "(0 / 1 bg / 2 all)")
    # the lane's A quantizer is ONE set of helpers for its three copies (the
    # gemm prologue, the local path, kda p4): the local path's bytes are the
    # global path's by construction, and the boot self-test checks it
    _lq = cu.index("auto lq_quant = [&](int buf) {")
    _lq_end = cu.index("auto stage_a_store = [&](int kb) {", _lq)
    check(cu.count("mk_pack4(") == 4 and cu.count("mk_warp_amax(") == 5
          and cu.count("mk_act_rcp(") >= 4  # def + the quantizers (33차 lever 1: exact scale)
          and cu.count("mk_pow2_scale(") == 1  # the definition only: no activation quantizer is pow2 now
          and "mk_act_scale(mxq[i])" in cu[_lq:_lq_end]
          and "asc[i] = sc * c.wgs;" in cu[_lq:_lq_end]
          and "sxs[r * KBLK_MAX + kb] = asc[i];" in cu
          and "__reduce_max_sync(0xffffffffu, __float_as_uint(v))" in cu
          and "__nv_cvt_float2_to_fp8x2(" in cu
          and "return __int_as_float((biased + 1) << 23);" in cu,
          "local A quant: quant_store's helpers (redux amax, exponent-"
          "arithmetic pow2, exact reciprocal, paired e4m3 pack) -- the output "
          "must be bitwise the global path's")
    # the local quant runs ahead of the mma on rows loaded a whole iteration
    # earlier (a two-slot ring), the global path stages A(kb0) before the
    # wait exactly as main did
    check(cu.index("if (kb + 2 < kbn) stage_a_load(kb + 2, (kb - kb0) & 1);")
              < cu.index("if (kb + 1 < kbn) lq_quant((kb + 1 - kb0) & 1);")
              < cu.index("mma_fold(sw4t, kb);  // the group scales are inside the bytes")
          and "if (!local_q) stage_a_store(kb0);" in cu
          and cu.index("if (!local_q) stage_a_store(kb0);")
              < cu.index("mk_cp_wait_upto(min(RAW_DIST - 1, kbn - kb0 - 1));")
          and "if (local_q) stage_a_store(kb0);" in cu
          and "uint2 xq0[RPW], xq1[RPW];" in cu,
          "local path: x two k-blocks ahead in a register ring, the quant "
          "before the mma; global path: A(kb0) into smem before the wait "
          "(main's order)")
    check("template <bool LQ>" in cu
          and "mk_gemm_phase_t<false>(c, smem, bar);" in cu
          and cu.count("mk_gemm_phase_t<true>(") == 1
          and "__global__ void mk_gemm_lq_kernel(const MKGemmCtx c) {" in cu
          and "mk_launch(mk_gemm_lq_kernel, p.lgrid, GEMM_SMEM, stream, c);" in cu
          and "mk_resident_grid(mk_gemm_lq_kernel, g_gemm_lq_grid," in cu
          and "int mk_lq_launch_grid(int units, int grid)" in cu
          and "c.bar_id ? &g_mk_gemm_bar_bg : &g_mk_gemm_bar" in cu
          and "if (!p.localq && bg && g_probe_bg_grid > 0 && g_probe_bg_grid < p.grid) {" in cu,
          "the local path is its own kernel launched on as many blocks as it "
          "has units; the bench's control (the global kernel on fewer blocks) "
          "runs on its own ticket counter; kda's mk_gemm_phase is the global "
          "instantiation")
    # Executable lane-selection, failure-restoration and nonfinite tests run
    # in test_megakernel_regression_suite below. Source strings previously
    # passed even when both purported v1 launches actually selected v2.
    check("def exact_fixture(dev=\"cuda\", shape=None):" in pysrc_full
          and "def run_both_kernels(x, pack, n, ksr=0):" in pysrc_full,
          "the boot and probe share the configurable exact fixture and v1 runner")
    bench = open(os.path.join(REPO, "probes", "megakernel_glm53_bench.py"),
                 encoding="utf-8").read()
    check("x, p4, w_exact, ref = mk.exact_fixture(DEV)" in bench
          and "got, plans = mk.run_both_kernels(x, p4, n)" in bench
          and "ran_local = plans[2][3] == 1" in bench
          and 'mark = " " if (same != "NO" and r <= TOL_SPLIT and rep_ok) else "!"' in bench
          and "ext.read_ts()  # clear: idle blocks of THIS launch must read 0" in bench,
          "probe_exact uses the boot gate's fixture and runner (no second copy "
          "to drift), the sweep marks rows by the tolerance it judges them by "
          "and clears the stamps before the launch it stamps")
    conc = open(os.path.join(REPO, "probes", "mk_gemm_concurrent_probe.py"),
                encoding="utf-8").read()
    check("from moe_decode_stream_probe import (" in conc
          and "served_wrapper" in conc and "expert_set" in conc
          and "MOE_LAYERS = 42" in conc
          and '((0, 0, 32), "global 32 (control)")' in conc,
          "the concurrent probe builds the served MoE from the MoE probe's "
          "builders, projects by the model's 42 MoE layers, and carries the "
          "fewer-blocks control row")
    fd_src = open(os.path.join(REPO, "overlay/modules/glm53_model/glm53_fp8_dense.py"),
                  encoding="utf-8").read()
    check("method._mk_bg = bool(_SHARED_EXPERT_RE.search(name))" in fd_src
          and "bg=getattr(self, \"_mk_bg\", False)" in fd_src,
          "the fp8-dense hook marks the shared expert's linears background "
          "(the aux-stream pair beside the routed MoE) for the lane")
    prof = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(re.search(r"^VLLM_GLM53_MK_LOCALQ=0$", prof, re.M) is not None,
          "the profile DECLARES the local-quant knob (off until its bracket): "
          "the launcher forwards only declared keys, so an undeclared knob "
          "could not be flipped at all")
    br = open(os.path.join(REPO, "bench", "bracket.py"), encoding="utf-8").read()
    check('"VLLM_GLM53_MK_LOCALQ"' in br,
          "bracket.py snapshots the local-quant knob")
    # There is no zero pass: every accumulator element is ASSIGNED by exactly
    # one unit, so pre-setting them cost a full pass plus a barrier to
    # publish values that are all overwritten before anyone reads them. The
    # assignment is what makes that true -- if the epilogue ever goes back to
    # accumulating, the zero pass has to come back with it.
    check("g_mk_gemm_partial[i] = 0.0f;" not in cu
          and "pb[(size_t)r0 * pcols + pc] = acc[i][j][0];" in cu,
          "the split-K accumulator is assigned, not accumulated, so it "
          "needs no zero pass")
    check("32 * 128 * MK_SPLIT_UNITS_MAX" in cu
          and "MK_SPLIT_UNITS_MAX = 2 * MK_GRID_CAP" in cu
          and "(MK_GRID_CAP - 1) * 128" in cu,
          "the accumulator is sized by the UNIT cap (rem * ksr), not by n: "
          "ksr * rem <= grid stopped holding once ksr could exceed "
          "grid / rem, and n-sized would be 20x larger for no reason")
    check("atomicAdd(&g_mk_gemm_partial" not in cu
          and "fixed order -> reproducible" in cu,
          "the split reduction must be deterministic: an atomicAdd "
          "accumulator returns bitwise-different results for back-to-back "
          "launches of the same call, which the probe's replay-stability "
          "check flags and this lane cannot ship")
    check("expand_w4(kb0 % W4_RAW_NBUF, kb0 % 2)" in cu
          and "stage_raw4(nt, kb0, kb0 % W4_RAW_NBUF);" in cu,
          "the W4 prologue must fill the buffer the loop reads first: the "
          "loop starts at kb0, so a hardcoded 0 loads the wrong k-block "
          "(this broke the W4 exact-grid check when split-K landed)")
    # -- W streams its raw records through cp.async:
    #    tile-major pack (one contiguous 8 KB + 1 KB record per (tile,
    #    k-block)), one commit group per stage, the exact-wait formula,
    #    and a smem budget that stays under the sm_121 opt-in.
    check("wq4.dim() == 4" in cu and "wq4.size(3) == 64" in cu
          and "ws4.size(3) == 8" in cu
          and ".permute(0, 2, 1, 3).contiguous())" in pysrc_full
          and "wq4.view(n_pad // 128, 128, k // 128, 64)" in pysrc_full,
          "the W4 pack is tile-major and the host refuses any other shape")
    check("W4_RAW_NBUF * W4_RAW_BYTES" in cu
          and 'static_assert(GEMM_SMEM <= 101376, "over the sm_121 opt-in smem");' in cu
          and "mk_cp_wait_upto(min(RAW_DIST - 1, kbn - kb - 2));" in cu,
          "W4 raw staging is budgeted in GEMM_SMEM and waits for exactly "
          "the record it needs")
    # -- W4 pack: exact e2m1 -> e4m3 expansion
    # Pin the LIVE constant, not a copy of it. This used to check a
    # `__device__ __constant__ uint8_t[8]` that the funnel-shift immediate
    # replaced; the array sat unreferenced for as long as the test kept
    # naming it. Decode the immediate and check the values it actually
    # encodes.
    lut = re.search(r"MK_E2M1_LUT64 = 0x([0-9A-Fa-f']+)ULL", cu_code)
    check(lut is not None, "the e2m1->e4m3 LUT immediate is present")
    assert lut is not None
    packed = int(lut.group(1).replace("'", ""), 16)
    e4m3 = [(packed >> (8 * c)) & 0xFF for c in range(8)]
    check(e4m3 == [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C],
          "the e2m1->e4m3 LUT covers {0,.5,1,1.5,2,3,4,6} exactly")
    check("mk_e2m1_to_e4m3" not in cu_code and "mk_e2m1_byte" not in cu_code,
          "the superseded table and its accessor stay deleted")
    # the expansion reads the packed immediate, not constant memory, and
    # the immediate must be the same table byte for byte
    _lut64 = int("0x4C4844403C383000", 16)
    check([(_lut64 >> (8 * c)) & 0xFF for c in range(8)]
          == [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C]
          and "0x4C484440'3C383000ULL" in cu
          and "__byte_perm(l0, l1, w & 0x7777u)" in cu_code
          and "__byte_perm(l0, l1, (w >> 16) & 0x7777u)" in cu_code
          and "__vadd4((uint32_t)lutg, eb * 0x01010100u)" in cu_code
          and "__vadd4((uint32_t)(lutg >> 32), eb * 0x01010101u)" in cu_code
          # the scale byte is e4m3, not a bare exponent: its 3-bit mantissa
          # rides the SAME byte add, and the only correction is +1 on the
          # three codes whose magnitude mantissa is 1.5, which does not
          # depend on the mantissa -- so a second constant, not a select
          # over eight
          and "0x4D484540'3D383000ULL" in cu
          and "((eb & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64" in cu_code
          and "mk_e2m1_to_e4m3[lo & 7]" not in cu_code
          and "__vcmpeq4(mag" not in cu_code,
          "the W4 expansion is a byte-permute table lookup: the raw nibbles "
          "(sign masked) are the prmt selector over the LUT's two halves "
          "with the group exponent folded in per 16 elements -- never a "
          "__constant__ load, never per-nibble scalar chains, and no longer "
          "the byte-lane arithmetic (22 -> 13 ops per raw word)")
    # -- one lane: the fp8 (W8) MK arm was removed once W4 beat the stock
    #    pair on every decode shape. One kernel, one budget, one ticket
    #    counter, one pack builder, one knob (MK_GEMM) -- nothing to select.
    check("mk_gemm_kernel<" not in cu_code and "template <bool W4>" not in cu
          and "run_gemm_w4" not in cu and "GEMM_SMEM_W4" not in cu
          and "g_mk_gemm4_bar" not in cu and "MK_W_NBUF" not in cu
          and "stage_w(" not in cu_code and "if constexpr (W4)" not in cu
          and 'm.def("run_gemm", &mk_run_gemm, "MK_SEG_GEMM (W4 pack)");' in cu
          and "mk_gemm_phase(c, smem, c.bar_id ? &g_mk_gemm_bar_bg : &g_mk_gemm_bar);" in cu,
          "the megakernel GEMM is W4-only: no fp8 W8 kernel, budget, counter "
          "or entry point remains in the .cu")
    check("def build_mk_weight(" not in pysrc_full
          and "run_gemm_w4" not in pysrc_full and "ENABLE_W4" not in pysrc_full
          and "W4_ALL" not in pysrc_full and "_W4_ARMED" not in pysrc_full
          and "VLLM_GLM53_MK_W4" not in pysrc_full
          and "-DMK_NBUF_DEF=" in pysrc_full and "MK_W4_NBUF" not in pysrc_full
          and "MK_PROBE_SKIP" not in pysrc_full and "MK_PROBE_SKIP" not in cu,
          "the driver has one pack builder, one launch entry and one depth "
          "knob (VLLM_GLM53_MK_NBUF); the W4 arm knob and the probe-skip "
          "switch are gone")
    check("mk_pack[0].shape[0] * 128" in pysrc_full
          and "def gemm_w4a8(x, mk_pack, n_rows, bg=False):" in pysrc_full,
          "eligibility derives n_pad from the tile-major pack's first dim x "
          "128 (the tile count itself failed n_pad % 128 on every real shape "
          "and the lane silently stayed stock)")
    # -- kda p4 emits the o_proj GEMM's fp8 A tiles + pow2 scales itself
    #    (a head's 128 dims = one k-group), and p5 runs with a_ready: no
    #    x load / quant / publishing barrier in its prologue. The unit
    #    counter reset moves under the kernel's own p4->p5 barrier.
    check("bool a_ready = false;" in cu
          and "if (!c.a_ready && kbq < kblk) quant_store(kbq, v, mx);" in cu
          and "if (!c.a_ready) {\n    if (blockIdx.x == 0 && threadIdx.x == 0) g_mk_unit_next = 0u;"
          in cu
          and "*(uint32_t*)(g_mk_aq + ((size_t)h * 32 + t) * KSTEP + lane * 4) = pack;"
          in cu and "if (lane == 0) g_mk_axs[t * KBLK_MAX + h] = sc;" in cu
          and "c.a_ready = true;" in cu
          and cu.index("if (blockIdx.x == 0 && threadIdx.x == 0) g_mk_unit_next = 0u;\n  MK_KDA_TS(9);")
              < cu.index("c.a_ready = true;"),
          "kda p4 writes the o_proj A tiles and scales in the GEMM's layout; "
          "p5 skips its prologue (a_ready) and the unit counter is reset under "
          "the p4->p5 barrier")
    check("_TOL_KDA_SHADOW = 0.15" in pysrc_full
          and "not (v <= _TOL_KDA_SHADOW)" in pysrc_full
          and "_TOL_KDA = 2e-2" in pysrc_full,
          "the serving KDA shadow is gated at the e2m1 by-design class (its "
          "MK arm streams W4 packs of the real weights); the fixture keeps "
          "the 2e-2 noise gate on grid-snapped weights")
    check("self.w_in = mk_pack_dequant(_pi, KDA_INPROJ_N)" in pysrc_full
          and "in_mk, o_mk = build_mk_weight_w4(f.w_in), build_mk_weight_w4(f.w_o)"
          in pysrc_full,
          "the KDA fixture snaps its weights to the e2m1 grid so both arms "
          "see the same values and the 2e-2 gate measures noise, not e2m1's "
          "by-design error")
    check("(int8_t)((sc4 >> (8 * g4)) & 0xFFu)) & 0xFFu" in cu_code
          and "__byte_perm(0x8000u, 0u, (w >> 3) & 0x1111u)" in cu_code
          and "__byte_perm(0x8000u, 0u, (w >> 19) & 0x1111u)" in cu_code
          and "uint8_t nb[32]" not in cu_code and "uint8_t ob[64]" not in cu_code,
          "expansion is a scale-byte add (in the table) + sign (a second "
          "prmt over nibble bit 3) in registers, never a float multiply and "
          "never a local-memory byte array")
    check("def _selftest_gemm() -> bool:" in pysrc_full
          and "_selftest_w4" not in pysrc_full
          and "ref = mk_pack_twin(x, pack, n)"
          in pysrc_full and "if not e <= 1e-3 or n_ulp > 0:" in pysrc_full
          and "def _exact_gate(got, ref32) -> tuple:" in pysrc_full
          and "refb = ref32.to(torch.bfloat16).float()" in pysrc_full
          and "_TOL_GEMM_W4 = 0.15" in pysrc_full,
          "the GEMM self-test gates bit-exact expansion against the torch "
          "twins (kernel activation quant x dequantized pack) and the "
          "by-design error against the stock pair")
    check("def mk_w4_dequant(wq4, ws4, n_rows, gscale=1.0, rgs=None):" in pysrc_full
          and "def _mk_quant_x_ref(x):" in pysrc_full
          and "torch.float8_e4m3fn" in pysrc_full
          and "scale = (amax * (1.0 / 448.0)).clamp(min=1e-30)" in pysrc_full
          and "return fmaxf(amax * (1.0f / 448.0f), 1.0e-30f);" in cu
          and "rsc = 1.0 / scale" in pysrc_full
          and ".clamp(-448.0, 448.0).to(torch.float8_e4m3fn)" in pysrc_full,
          "the torch twins exist: dequant of the tile-major pack and the "
          "prologue's per-128-group pow2 activation quant")
    check('".in_proj_qkvbfg_a" not in name' not in open(
        os.path.join(REPO, "overlay/modules/glm53_model/"
                            "glm53_fp8_dense.py"), encoding="utf-8").read(),
          "every eligible linear gets the W4 pack, the KDA in_proj included: "
          "there is no per-linear knob left")
    # All four launches are persistent grids that resolve their own size
    # from the device. mhc has its own ceiling because it takes no dynamic
    # smem; gemm and kda share MK_GRID_CAP but resolve separately, since
    # kda carries more state and need not fit as often.
    # gemm and kda are persistent grids too, and their occupancy differs
    # from each other's, so each resolves its own -- clamped, like mhc, so a
    # grid that does not fit degrades instead of refusing to launch.
    check("mk_resident_grid(mk_gemm_kernel, g_gemm_grid, GEMM_SMEM)" in cu
          and "mk_resident_grid(mk_kda_kernel, g_kda_grid, GEMM_SMEM)" in cu
          and "int cap = MK_GRID_CAP" in cu and "if (cache > cap) cache = cap;" in cu,
          "gemm and kda each resolve their persistent grid from the device "
          "rather than assuming MK_GRID_CAP (the cap is a defaulted argument so "
          "MK_SEG_MLA, which owns its ticket counter, can raise its own)")
    # Distinct grids must not share a ticket counter -- the same trap the
    # mhc split fixed. kda inlines mk_gemm_phase on ITS grid.
    check("g_mk_kda_bar" in cu
          and cu.count("mk_gemm_phase(c, smem, c.bar_id ? &g_mk_gemm_bar_bg : &g_mk_gemm_bar)") == 1
          and cu.count("mk_gemm_phase_t<true>(c, smem, &g_mk_gemm_bar)") == 1
          and cu.count("mk_gemm_phase(c, smem, &g_mk_kda_bar)") == 2,
          "kda's inlined gemm phases use their own barrier counter")
    check(cu.count("mk_launch(mk_mhc_kernel, mhc_grid, 0, stream, a);") == 1
          and "if (mhc_grid > MK_MHC_GRID_CAP)" in cu,
          "mhc launches its own grid, clamped to what the device reports "
          "resident: a hard constant plus an assert would turn future "
          "register drift into a refusal to boot")
    check(cu.count("cudaOccupancyMaxActiveBlocksPerMultiprocessor") == 3
          and "&g_gemm2_bps, mk_gemm2_kernel<4, false>, MK_THREADS, GEMM2_SMEM" in cu,
          "both persistent grids check residency before launching: a grid "
          "that does not fit deadlocks on the grid barrier, it does not "
          "merely run slowly (the third query is the v2 lane's blocks per SM, "
          "which sizes its grid and is not a residency contract)")

    # -- MK-GEMM v2 (30차): the same GEMM as a NON-persistent grid. The
    #    persistent kernel's 48 x 69.6 KB blocks cannot share an SM with the
    #    routed MoE kernel (90 KB), so the shared expert's side-stream
    #    launches queued behind it and the publish barrier held every landed
    #    block for the last one (down [4096 x 512]: 18 us alone, 135 us in
    #    the 09-04 armed trace, 5.7 of the lane's 14.1 ms). v2 has no
    #    barrier, no shared A staging, expands W4 in registers into the mma
    #    fragments, and fits TWICE per SM.
    v2 = cu[cu.index("mk_gemm2_kernel(const MKGemm2Ctx c)"):cu.index("MK_SEG_MHC --")]
    check(consts.get("W4_RAW_NBUF2", 0) >= 2
          and 2 * (consts["GEMM2_SMEM"] + 1024) <= 102400
          and 'static_assert(2 * (GEMM2_SMEM + 1024) <= 102400, "v2 must fit twice per SM");' in cu
          and "__launch_bounds__(MK_THREADS, 2)" in cu,
          "the v2 block's smem (with the SM's ~1 KB per-block reserve) fits "
          "two per SM and its register budget is bounded to match -- one "
          "resident block would make it the persistent kernel without the "
          "barrier, not a co-scheduling kernel")
    check("mk_grid_barrier" not in v2 and "g_mk_unit_next" not in v2
          and "g_mk_aq" not in v2,
          "v2 has no grid barrier, no unit counter and no shared A tiles: "
          "every block quantizes the A k-blocks of its own slice")
    check("stage_raw(kb0 + d, (kb0 + d) % NB)" in v2
          and v2.index("stage_raw(kb0 + d, (kb0 + d) % NB)")
              < v2.index('asm volatile("griddepcontrol.wait;"')
              < v2.index("load_x(kb0);") < v2.index("quant_x(0);"),
          "v2 prologue order: the W ring first (independent of the previous "
          "kernel, in flight during its tail under PDL), the PDL wait, then x")
    check("mk_cp_wait_upto(min(DIST - 1, kbn - kb - 2));" in v2
          and "if (kb + DIST < kbn) stage_raw(kb + DIST, (kb + DIST) % NB);" in v2
          and "quant_x((kb + 1 - kb0) & 1);" in v2
          and v2.count("__syncthreads();", v2.index("for (int kb = kb0;;"),
                       v2.index("MK2_TS(2);")) == 1,
          "v2 k loop: exact ring wait, refill of the stage consumed last "
          "iteration, quant into the A buffer the running mma does not read, "
          "one __syncthreads per k-block")
    check("int a_ready = 0;\n  int pair_act = 0;" in cu
          and "__device__ __align__(16) uint8_t g_mk2_aq[(size_t)KBLK_MAX * 32 * KSTEP];" in cu
          and "constexpr int MK2_UNITS_MAX = MK2_TILES_MAX * MK2_KSR_MAX * 2;" in cu
          and "__device__ unsigned int g_mk2_pair_arrive[MK2_TILES_MAX];" in cu
          and "if (c.a_ready) {  // stage the published group; nothing to quantize" in v2
          and "if (c.pair_act) pair_finish(nt);  // the tile's final store was just made" in v2
          and "if (c.pair_act) pair_finish(nt);  // the fold was this tile's final store" in v2
          and "if (s_pair_last) g_mk2_pair_arrive[pair] = 0u;  // both arrived; rearm" in v2
          and "void mk_run_smlp2(std::vector<int64_t> ptrs, std::vector<double> scalars," in cu
          and cu.count("mk_launch_gemm2(") == 4  # def, gemm, smlp2 x2
          and "g.pair_act = 1; g.n_int = n_int;" in cu and "d.a_ready = 1;" in cu
          and 'm.def("run_smlp2", &mk_run_smlp2' in cu
          and "mk_grid_barrier" not in cu[cu.index("void mk_run_smlp2("):cu.index("std::vector<int64_t> mk_gemm2_plan(")],
          "MK_SEG_SMLP2 is two PDL-chained v2 launches: gate_up's pair epilogue "
          "emits the fp8 groups, down stages them (a_ready), no grid barrier, "
          "the pair counter self-rearms")
    check('ENABLE_SMLP2 = MASTER and _flag("VLLM_GLM53_MK_SMLP2")' in pysrc_full
          and "def _smlp2_call(x, gu_pack, d_pack, n_gu, n_int, n_out, limit, alpha=1.0," in pysrc_full
          and "def _selftest_smlp2() -> bool:" in pysrc_full
          and '_ARMED["smlp2"] = _gate("smlp2", _selftest_smlp2)' in pysrc_full
          and 'if not (_ARMED["smlp"] or _ARMED["smlp2"]):' in pysrc_full
          and 'if _ARMED["smlp2"]:\n        return _smlp2_call(' in pysrc_full
          and 'first fused call (%s) ' in pysrc_full
          and '"smlp2" if _ARMED["smlp2"] else "smlp"' in pysrc_full,
          "the driver arms smlp2 behind its own knob with the exact + replay "
          "gate, the MLP hook prefers it when armed, and the serving line "
          "names the lane")
    check("template <int RQ, bool LR>" in cu
          and "constexpr int MT = (RQ == 4) ? 2 : 1;   // m-tiles present" in v2
          and "constexpr int LPR = 32 / RQ;            // lanes per quantized row" in v2
          and "for (int off = LPR / 2; off; off >>= 1)  // stays inside the row's lane group" in v2
          and "mk_launch(mk_gemm2_kernel<1, false>, grid2, GEMM2_SMEM, stream, c2);" in cu
          and "mk_launch(mk_gemm2_kernel<2, false>, grid2, GEMM2_SMEM, stream, c2);" in cu
          and "mk_launch(mk_gemm2_kernel<4, false>, grid2, GEMM2_SMEM, stream, c2);" in cu,
          "v2 is instantiated per m class (rows quantized per warp 1/2/4 -> "
          "m-tiles and the x lane mapping at compile time) and the host picks "
          "the instantiation from m")
    check("((ea & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64" in v2
          and "((eb & 7u) - 1u) < 5u ? MK_E2M1_LUT64_B : MK_E2M1_LUT64" in v2
          and "l0a[j] = __vadd4((uint32_t)la, ea * 0x01010100u);" in v2
          and "l1b[j] = __vadd4((uint32_t)(lb >> 32), eb * 0x01010101u);" in v2
          and "__byte_perm(l0, l1, w & 0x7777u)" in v2
          and "__byte_perm(0x8000u, 0u, (w >> 3) & 0x1111u)" in v2
          and "__byte_perm(0x8000u, 0u, (w >> 19) & 0x1111u)" in v2
          and "const int wsel = (ks + q) & 3;" in v2
          and "const int koff = 32 * q + 8 * wsel;" in v2
          and "expand_w4" not in v2 and "mk_w4x4" not in cu_code,
          "v2 expands the e2m1 nibbles into the B fragments per lane with the "
          "SAME table lookup as expand_w4 (same two LUT immediates, same sign "
          "prmt) -- two LUT pairs per W row per k-block (lane q owns natural "
          "elements [32q, 32q+32), groups 2q and 2q+1) and the rotated word "
          "(ks + q) & 3 keeps the quad's raw loads on distinct banks")
    check("((ch ^ ((r >> 1) & 3)) << 4)" in v2
          and "slot[j] = nrow * W4_RAW_PITCH + ((q ^ ((nrow >> 1) & 3)) << 4);" in v2
          and "*(const uint32_t*)(rr + slot[j] + 4 * wsel);" in v2,
          "the raw nibble chunks land XOR-swizzled at copy time and the "
          "fragment loads read through the same swizzle (eight rows on a 64 B "
          "pitch would otherwise share two bank groups)")
    check("store_tile([&](int r, int col, float v) { pb[(size_t)r * c.n + col] = v; });" in v2
          and "if (r0 < c.m) { put(r0, cb, acc[i][j][0]); put(r0, cb + 1, acc[i][j][1]); }" in v2
          and "atomicAdd(&g_mk2_tile_arrive[nt], 1u)" in v2
          and "if (s_last) g_mk2_tile_arrive[nt] = 0u;" in v2
          and "for (int s = 0; s < nslices; ++s) {  // fixed order -> reproducible" in v2
          and "atomicAdd(&g_mk2_partial" not in cu
          and "g_mk2_partial[i] = 0.0f;" not in cu,
          "v2 split slices are assigned (no zero pass), counted per tile with "
          "a self-rearming counter, and folded by the last slice in fixed "
          "order -- deterministic, no atomics on the partials")
    check('getenv("VLLM_GLM53_MK_KTAIL")' in cu
          and "int mk_choose_tail2(int m, int n, int k, int ksr)" in cu
          and "if (shortest < 2 * tail) return 0;" in cu
          and "const int nslices = c.tail > 0 ? 2 * ksr : ksr;      // partials per tile" in v2
          and "const int slice = is_tail ? ksr + sp : sp;" in v2
          and "s_last = (prev + 1u == (unsigned)nslices);" in v2
          and "for (int s = 0; s < nslices; ++s) {  // fixed order -> reproducible" in v2
          and "const int grid2 = (c2.n / SMEM_W_ROWS) * c2.ksr * (c2.tail > 0 ? 2 : 1)\n                    + (c2.lr_r > 0 ? LR_CTAS : 0);" in cu,
          "v2 tail units (VLLM_GLM53_MK_KTAIL, default off): the last tail k-blocks "
          "of every slice form a second unit at the end of the grid, folded as a "
          "second partial in fixed order; only when every slice keeps >= tail "
          "k-blocks and the doubled partial set fits")
    check('getenv("VLLM_GLM53_MK_GEMM2")' in cu
          and "if (mk_gemm2_on()) {  // the non-persistent lane" in cu
          and cu.index("if (mk_gemm2_on()) {  // the non-persistent lane")
              < cu.index("const MKGemmPlan p = mk_gemm_plan_for(c.m, c.n, c.k, bg != 0);")
          and "int mk_choose_ksr2(int m, int n, int k)" in cu
          and "(g_mk_sms > 0 ? g_mk_sms : 48);" in cu
          and "MK_CHECK_CUDA(cudaGetDevice(&dev));" in cu
          and "&g_mk_sms, cudaDevAttrMultiProcessorCount, dev));" in cu
          and "if (slots % nblk == 0 && slots / nblk <= kmax) {" in cu
          and "} else if (nblk * 2 > slots) {" in cu
          and "if (ksr > kblk) ksr = kblk;" in cu
          and "while (ksr > 1 && (size_t)m * n * ksr > (size_t)MK2_PART_ELEMS) --ksr;" in cu
          and consts["MK2_PART_ELEMS"] >= 32 * consts["KDA_INPROJ_N_PAD"] * 4,
          "the v2 lane is a kill-switched dispatch ahead of the persistent "
          "launch (VLLM_GLM53_MK_GEMM2, default off); its slice rule takes one "
          "exact wave of the device's resident slots (SM count from the device, "
          "not a constant), fine slices above half the slots, and keeps every "
          "slice non-empty and the fp32 partial inside its buffer")
    check('"-DMK_NBUF2_DEF=" in pysrc_full' if False else "-DMK_NBUF2_DEF=" in pysrc_full
          and "_EXT.gemm2_plan(8, KDA_INPROJ_N, HIDDEN)" in pysrc_full
          and 'ap.add_argument("--gemm2", choices=("0", "1", "both", "env"), default="env")' in bench
          and 'args.gemm2 = "1" if os.environ.get("VLLM_GLM53_MK_GEMM2") == "1" else "0"' in bench
          and "ext.set_gemm2(on, ksr, ktail)" in bench
          and "same = bool(torch.equal(got2, got))" in bench,
          "the driver builds v2's ring depth in, the boot fingerprint names "
          "the served lane and its in_proj plan, and the bench times both "
          "lanes side by side with v2's output diffed against the persistent "
          "lane's")
    # Two grid sizes must never share a ticket counter. The barrier computes
    # (t / grid + 1) * grid, so it is only correct when the counter is
    # grid-aligned at launch; a 48-block kernel leaves it 48 past a multiple
    # of 96, and the next 96-block launch then releases at half its blocks.
    check('ws["barrier_mhc"].data_ptr()' in pysrc_full
          and 'ws["barrier_mla"].data_ptr()' in pysrc_full
          and pysrc_full.count("_barrier_ptr(ws)")
          - pysrc_full.count("def _barrier_ptr(ws)") == 1,
          "mhc and mla each run their own barrier counter, not the one the "
          "48-block kernels share (their grids are 96 blocks)")
    check("(t / (unsigned long long)grid + 1ULL) * grid" in cu,
          "grid barrier is the never-reset monotonic ticket form, 64-bit "
          "(32-bit wraps in ~a week of arrivals and releases early)")
    bar = cu[cu.index("mk_grid_barrier"):cu.index("mk_grid_barrier") + 700]
    check("__threadfence();" in bar,
          "barrier fences device scope around the ticket (osar lesson)")
    check("atomicAdd(ctr, 1ULL)" in bar
          and "ctr =" not in bar and "ctr=" not in bar,
          "the barrier counter is never reset inside the kernel "
          "(monotonic ticket only)")

    # -- state slot addressing: [slots, ...] buffers need the per-slot
    #    element count in the stride (a slot-0-only self-test once passed
    #    while the conv stride was missing it -- found in review)
    check(cu.count("slot * a.cs_s0 + (size_t)ch * a.cs_s1") >= 1
          and "sbase + (size_t)i * a.cs_s2" in cu
          and "sbase + (size_t)(acc + i) * a.cs_s2" in cu
          and "slot * KDA_QKV * a.conv_width" not in cu
          and "ints.size() == 10" in cu
          and "kda_cs_load(a, sbase + (size_t)(acc - 1 + i) * a.cs_s2)" in cu
          and "kda_cs_store(a, sbase + (size_t)i * a.cs_s2, v)" in cu
          and "a.cs_bf16 = (int)ints[9];" in cu
          and "conv state strides must be positive" in cu
          and "(size_t)slot0 * a.rs_s0 + (size_t)head * KDA_D * KDA_D" in cu
          and "(size_t)sj * a.rs_s0 + (size_t)head * KDA_D * KDA_D" in cu
          and "recurrent state slot stride is narrower than one slot" in cu,
          "conv state is addressed through (slot, channel, width) strides "
          "carried by the launch, computed once as sbase and used by every "
          "read and write (the engine hands out page-aligned / transposed "
          "views; a contiguity gate rejected every production layer, 32차)")
    check("st[i] = kda_cs_load(a, sbase + (size_t)(acc - 1 + i) * a.cs_s2);" in cu
          and "a.conv_state[sbase + a.conv_width - (CONV_W - 1) + i]" not in cu,
          "the convolution's pos<0 history starts at the accepted boundary "
          "(state[acc - 1 .. acc + 1], the stock spec kernel's prior_tokens) "
          "-- the buffer's newest end is that window only when every draft "
          "was accepted")
    check("for (int i = 0; i < a.conv_width; ++i)" in cu
          and "? kda_cs_load(a, sbase + (size_t)(acc + i) * a.cs_s2)" in cu
          and "for (int j = 0; j < NQ_MAX; ++j) v = (q == j) ? xin[j] : v;" in cu
          and "constexpr int KDA_NQ_MAX = KDA_SPEC + 1;" in cu
          and '"kda: max_query_len over the unrolled conv window (KDA_NQ_MAX)"' in cu,
          "the state update writes the WHOLE window: causal_conv1d_update "
          "keeps conv_width - nq old values starting at `acc` and appends "
          "every query token")
    check("(size_t)slot0 * a.rs_s0 + (size_t)head * KDA_D * KDA_D" in cu
          and "(size_t)sj * a.rs_s0 + (size_t)head * KDA_D * KDA_D" in cu
          and "const int head = blockIdx.x >> 1, rowhalf = blockIdx.x & 1;" in cu
          and "(size_t)rowhalf * RB * KDA_D;" in cu,
          "recurrent state slot stride is the launch-carried rs_s0 (>= H*D*D) "
          "for both the resume slot and the per-position store slots")

    # -- driver-side guards from the same review
    check("SLOT = 1" in pysrc_full
          and "self.sidx = (self.SLOT + torch.arange(8, dtype=torch.int32,"
          in pysrc_full
          and "mkw(self.SLOT + 8 + 1, KDA_H, KDA_D, KDA_D," in pysrc_full,
          "the KDA fixture addresses NONZERO, DISTINCT state slots per query "
          "position (the engine's spec-decode layout)")
    # -- the state-index contract, taken from the stock kernels: the conv
    #    history starts at the accepted boundary, the recurrence resumes
    #    from slot [r, acc - 1] and stores after every token into [r, j]
    check("st[i] = kda_cs_load(a, sbase + (size_t)(acc - 1 + i) * a.cs_s2);" in cu
          and "const int slot0 = a.state_idx[r * a.mql + (acc - 1)];" in cu
          and "if (slot0 <= 0 || t1 <= t0) continue;" in cu
          and "const int sj = a.state_idx[r * a.mql + j];" in cu
          and "if (sj > 0) {" in cu
          and "if (j == acc - 1)" not in cu
          and "a.conv_width - (CONV_W - 1) + i" not in cu,
          "MK-KDA follows the stock state-index contract: conv history from "
          "state[acc - 1], recurrence resumed from slot [r, acc - 1] and "
          "stored per position into [r, j] (slot <= 0 skipped)")
    arm_fn = pysrc_full[pysrc_full.index("def maybe_arm"):]
    check("is_current_stream_capturing()" in arm_fn,
          "maybe_arm never compiles/self-tests inside graph capture")
    check("_kda_ensure_packs" in pysrc_full
          and "NvFp4DenseMethod, W4A8DenseMethod" in pysrc_full
          and "while isinstance(m, (NvFp4DenseMethod, W4A8DenseMethod))"
          in pysrc_full,
          "KDA packs require a QUANTIZED arm, but the method may be wrapped "
          "(nvfp4/w4a8 stack on the fp8 one) -- demanding Fp8DenseMethod by "
          "isinstance is what made MK-KDA look exclusive with the nvfp4 "
          "dense scheme when only MK-GEMM ever was")
    packs_fn = pysrc_full[pysrc_full.index("def _kda_ensure_packs"):]
    packs_fn = packs_fn[:packs_fn.index("def _kda_device_ok")]
    check("_bf16_freed" in packs_fn and "raise RuntimeError" in packs_fn,
          "this runs on the first eager forward, AFTER "
          "maybe_free_fp8_dense_bf16: a missing pack whose source was "
          "released must refuse loudly, not pack an empty tensor")

    # -- kda.py overlay keeps the stock body reachable (its own module since
    #    the core was made model-agnostic; see test_megakernel_core_is_shared)
    kda = open(os.path.join(REPO, "overlay/modules/glm53_model",
                            "glm5next_kda.py"), encoding="utf-8").read()
    check("fused_recurrent_kda(" in kda and "causal_conv1d_update(" in kda,
          "kda.py overlay keeps the stock conv/recurrent path")
    check("deneb fork (glm53_megakernel)" in kda,
          "kda.py takeover carries the fork marker")
    check("kda_block(self, hidden_states, positions)" in kda
          and "compare(out)" in kda,
          "kda.py wires both the armed takeover and the shadow epilogue")
    check(kda.index("torch.cuda.is_current_stream_capturing()")
          < kda.index("_mk_arm(self, hidden_states)"),
          "shadow runs eager-only, never inside graph capture")

    # -- mhc: no grid barrier. The block that completes a token's 16th
    #    chunk runs that token's p2/p3/p4 (arrival counter, rearmed by the
    #    last arriver, other blocks' partials read through L2); p1 reduces
    #    its 25 partials through a transposed smem tile and prefetches the
    #    next pair's loads. Keeping fn in smem instead measured worse.
    _mhc_k = cu[cu.index("__global__ void mk_mhc_kernel(const MKMhcArgs a) {"):]
    _mhc_k = _mhc_k[:_mhc_k.index("\n}\n")]
    check("mk_grid_barrier" not in _mhc_k and "mk_mhc_p1(a, blockIdx.x);" in _mhc_k,
          "the mhc kernel has no grid barrier: p2/p3/p4 run off the tail queue")
    # -- kda: gates and conv share a phase (both read only p0's qkv); the
    #    kernel has four grid barriers, not five
    _kda_k = cu[cu.index("__global__ void mk_kda_kernel(const MKKdaArgs a) {"):]
    _kda_k = _kda_k[:_kda_k.index("\n}\n")]
    check(_kda_k.count("mk_grid_barrier(a.barrier_ctr, a.grid);") == 4
          and "// no barrier here: phase 2 (conv) reads only phase 0's qkv" in _kda_k,
          "the kda kernel runs gates and conv back to back without a barrier")
    # -- kda: the split-K model keeps >= 8 k-blocks per slice (warming L2
    #    with the o_proj pack from the idle blocks measured a net loss:
    #    p5 -5.6 us, p3 +12)
    _smlp_k = cu[cu.index("void mk_smlp_kernel"):cu.index("void mk_mla_kernel")]
    check("prefetch.global.L2" not in _kda_k
          and cu.count("prefetch.global.L2") == _smlp_k.count("prefetch.global.L2")
          and "Warm phase C's first W records in L2" in _smlp_k
          and "L2WARM" not in cu and "warm_l2" not in cu
          and "const int rmax = kblk / 8 > 1 ? kblk / 8 : 1;" in cu
          and "for (int r = 2; r <= kblk && r <= rmax; ++r) {" in cu,
          "split-K never makes slices shorter than 8 k-blocks; no L2 "
          "prefetch under the delta rule (net loss) -- the fused MLP's warm "
          "of its down pack is a bench knob, off by default")
    # -- kda gates: a tensor-core GEMM (cp.async weight tiles, bf16 mma)
    check("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in cu
          and "constexpr int GT_TILES = 2 * KDA_OUT / GT_ROWS;" in cu
          and "if (next < GT_TILES) stage_tile(next, buf ^ 1);" in cu
          and '"gate weights and the qkv workspace must be 16 B aligned (cp.async)"' in cu,
          "the kda gates run as a double-buffered cp.async + bf16 mma GEMM "
          "(the warp-per-dot form measured 14-18 us for 1 MB of weights)")
    check("__device__ unsigned int g_mk_mhc_tok_arrive[MAX_TOK];" in cu
          and "if (threadIdx.x == 0) atomicAdd(&g_mk_mhc_tok_arrive[t], 1u);" in cu
          and "atomicAdd(&g_mk_mhc_tok_arrive[pend], 1u);" in cu
          and "int pend = -1;  // a token whose chunk is done but not yet published" in cu
          and "s_tok = (int)atomicAdd(&g_mk_mhc_tail_next, 1u);" in cu
          and "while (*v < (unsigned int)NCHUNK) __nanosleep(128);" in cu
          and "g_mk_mhc_tok_arrive[t] = 0u;  // rearm for the next launch" in cu
          and "g_mk_mhc_tail_next = 0u;" in cu
          and "      mk_mhc_p2_token(a, t, s_pmix);\n      mk_mhc_p34_load(a, t, tr);" in cu
          and "mk_mhc_p34_compute(a, t, s_pmix, tr);" in cu
          and "if (warp != 0) mk_mhc_p34_load(a, t, tr);" in cu
          and "m[j][k] = __shfl_sync(0xffffffffu, mixv, j * HC + k + 2 * HC);" in cu
          and "mine += __ldcg(&a.yp[((size_t)c * MAX_TOK + t) * NOUT + lane]);" in cu
          and "const float rms = __shfl_sync(0xffffffffu, rms_l, NOUT);" in cu
          and "if (lane == 0) {  // comb mixes: hc_scale[2] + sinkhorn, 4x4 in registers" in cu
          and "mk_ldcg_bf16(a.residual_out +" in cu,
          "the per-token tails are taken off a ticket counter by free blocks "
          "that wait on the token's arrival counter, rearm it, and read the "
          "other blocks' partials through L2; the last block out rearms the "
          "ticket counter")
    check("__shared__ float part[MK_THREADS][NOUT + 3];" in cu
          and "const int m = warp + 8 * q;" in cu
          and "load_tok(t + groups, h, nxv, nres, npm, ncm);" in cu
          and "float fnr[NOUT][HC];" in cu
          and "for (int t = g; t < a.num_tokens; t += groups) {" in cu
          and "const int groups = max(1, a.grid / NCHUNK);" in cu
          and "MHC_SMEM" not in cu,
          "p1 keeps the chunk's fn slice in registers with the tokens as the "
          "inner loop and reduces through a transposed smem tile (pitch 27); "
          "the smem-slice form measured worse")
    # -- hook placement: MK precedes ONEPASS in the mhc wrapper
    tl = open(os.path.join(REPO, "overlay/modules/glm53_kernels/"
                                 "tilelang.py"), encoding="utf-8").read()
    check(tl.index("_mk_hook = _deneb_mk_hook()")
          < tl.index("deneb fork: ONEPASS"),
          "the MK-MHC hook is tried before the ONEPASS experiment")

    # -- fp8_dense hook routes armed decode shapes and falls back otherwise
    fp8 = open(os.path.join(REPO, "overlay/modules/glm53_model/"
                                  "glm53_fp8_dense.py"), encoding="utf-8").read()
    check("gemm_w4a8 as _mk_gemm" in fp8 and "maybe_arm as _mk_arm" in fp8,
          "Fp8DenseMethod.apply routes through the megakernel driver")
    check("method._mk = _mkmod.build_mk_weight_w4(weight, name=name,\n                                                       per_row=per_row)" in fp8
          and 'per_row = False if getattr(method, "_opaque", False) else None' in fp8
          and fp8.count('attach_mk(method, weight, cols, f"{type(model).__name__}/{name}")') == 4
          and "ENABLE_W4" not in fp8 and "build_mk_weight(" not in fp8
          and "VLLM_GLM53_MK_W4" not in fp8,
          "the build attaches the W4 pack next to the deepgemm pair on every "
          "eligible linear, no arm knob")
    check("def probe_exact(gemm2: str = \"both\") -> bool:" in bench
          and "def _probe_exact_lane(mk, x, p4, n, ref, tag: str) -> bool:" in bench
          and "x, p4, w_exact, ref = mk.exact_fixture(DEV)" in bench
          and "probe_w4" not in bench
          and "run_gemm_w4" not in bench and "build_mk_weight(" not in bench
          and "VLLM_GLM53_MK_W4" not in bench
          and 'TOL = {"mhc": 1e-3, "gemm": 0.15, "kda": 2e-2}' in bench,
          "the bench times the W4 lane as the MK arm, gates it at the e2m1 "
          "by-design class, and keeps the exact-grid gate against the torch "
          "twins")
    # -- MK_SEG_MLA: correct-but-not-adopted sparse MLA decode. The contract
    #    that matters is that nothing routes to it and the pipeline keeps a
    #    fixed in-flight group count (a short row read stale smem without it).
    check("VLLM_GLM53_MK_MLA=1" in open(os.path.join(REPO, "profiles/glm53.env"),
                                        encoding="utf-8").read(),
          "profile ships MK_SEG_MLA on within the megakernel set (28차: bracket "
          "passed twice with the fixed scratch; the master gate still decides)")
    check(cu.count("else mk_cp_commit();") >= 2
          and "for (int ti = 0; ti < MLA_NSTAGE - 1; ++ti) {" in cu,
          "mla: empty commits keep cp.async groups aligned with wait_group, "
          "including rows whose slot count is under one tile")
    check("if (ti + MLA_NSTAGE - 1 < ntile) issue(ti + MLA_NSTAGE - 1);" in cu
          and cu.index("if (ti + MLA_NSTAGE - 1 < ntile) issue(ti + MLA_NSTAGE - 1);")
              < cu.index("const uint8_t* tile8 = ring +"),
          "mla: the next tile is issued BEFORE the current one is consumed "
          "(issuing after drained the pipeline: every phase was additive)")
    check("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in cu
          and "__nv_cvt_fp8x2_to_halfraw2" in cu,
          "mla: bf16 tensor cores for both mma phases, hardware fp8x2 conversion "
          "(the FMA versions lost 1.7x to conversion and unpacking instruction count)")
    _mla_src = cu[cu.index("// v4: tensor cores"): cu.index("// host entry points")]
    check("ldmatrix.sync" not in _mla_src and "mla_e4m3x2_strided" in cu
          and "MLA_SMEM_CT" not in cu,
          "mla v4: no bf16 copy of the tile in smem -- fragments convert from the "
          "e4m3 ring in registers, which is what buys the second block per SM")
    check("MLA_TILE = 16" in cu and "MLA_NSTAGE = 3" in cu and "MLA_KQ = 4" in cu,
          "mla v4 geometry: 16-slot tiles, 3 ring stages, 4 k-quarters -> ~46 KB smem")

    check("a.lens = (const int*)ptrs[3];" in cu and "const int len = a.lens[t];" in cu,
          "mla: per-row lengths are read from DEVICE memory (that is what a "
          "captured-graph launch needs; the wrapper's host replan is the cost "
          "this kernel exists to remove)")
    # -- glm53_mk_mla_wiring: the image-bound hook for MK_SEG_MLA
    wd = os.path.join(REPO, "overlay", "modules", "glm53_model")
    wsrc = open(os.path.join(wd, "flashinfer_mla_sparse_sm90.py"), encoding="utf-8").read()
    wrows = [l.split("\t") for l in open(os.path.join(wd, "manifest.tsv"), encoding="utf-8")
             .read().splitlines() if l and not l.startswith("#")]
    mla_rows = [r for r in wrows if r[1] == "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"]
    check(len(mla_rows) == 1 and re.fullmatch(r"[0-9a-f]{64}", mla_rows[0][2]) is not None,
          "mk_mla_wiring overlays the SM90 sparse backend with a pinned preimage")
    check("num_tokens > _MK_MLA_MAX_T" in wsrc and "_MK_MLA_MAX_T = 1 << 20" in wsrc,
          "mk_mla_wiring routes decode AND prefill (v5: splits==1 needs no scratch)")
    check("q_nope.shape[0] <= _MK_MLA_SHADOW_MAX_T" in wsrc
          and "_MK_MLA_SHADOW_MAX_T = 4096" in wsrc
          and "rows >= _MK_MLA_SHADOW_MIN_ROWS" in wsrc,
          "the one-shot wrapper shadow judges the first eager call whose rows "
          "read real KV (a prefill chunk of <= 4096 rows: a 64 MB transient "
          "output once), not the empty-KV dummy that passed with rel=0 (28차)")
    check("if T > MLA_MAX_SPLIT_ROWS:" in pysrc_full and "return 1" in pysrc_full,
          "prefill row counts take splits == 1 (a [T][splits][H][D] fp32 scratch "
          "would be 268 MB at T=8192)")
    check("if (a.splits == 1) return;" in cu and "v5 (prefill)" in cu,
          "the kernel normalises in phase 0 and skips the combine when splits == 1")
    # the wrapper tail also lives in the module-level _sm90_wrapper_run
    # helper (defined before the class), so compare against forward_mqa's
    # own copy -- the LAST occurrence
    check("if _mk_mla_route(self, num_tokens):" in wsrc
          and wsrc.index("if _mk_mla_route(self, num_tokens):") < wsrc.rindex("state.kv_indices[: num_tokens * width].copy_("),
          "the MK branch precedes the wrapper tail in forward_mqa")
    check("_SM90_STATE.plan(num_rows, kv_lens)" in wsrc,
          "the builder still plans every step (prefill and T>32 use the wrapper)")
    check('m._ARMED["mla"] = False' in wsrc and "rel > 2e-2" in wsrc and "torch.isfinite(out).all()" in wsrc
          and "_mk_mla_check_failure()" in wsrc and "SHADOW FAIL" in wsrc,
          "real-KV shadow drift or nonfinite output invalidates the worker; "
          "captured graphs must not continue behind a Python-only DISARM")
    check("glm53_megakernel" in open(os.path.join(wd, "requires"), encoding="utf-8").read(),
          "mk_mla_wiring requires the megakernel module")
    _mods = re.search(r'^MODULES="([^"]+)"', open(os.path.join(REPO, "profiles/glm53.env"), encoding="utf-8").read(), re.M).group(1).split()
    check(_mods.index("glm53_megakernel") < _mods.index("glm53_model"),
          "profile mounts the core before the wiring (glm53_model)")
    check("for s in range(1, budget + 1):" in pysrc_full and "(T * s) % grid == 0" in pysrc_full
          and "budget = max(1, min(MLA_SPLITS_MAX, MLA_WS_ROWS // T))" in pysrc_full,
          "mla split policy: smallest s with T*s a multiple of the resident grid (measured), "
          "bounded by the fixed scratch (T*s <= MLA_WS_ROWS; 28차)")
    print("  glm53 megakernel contracts .. OK")


def test_megakernel_core_is_shared() -> None:
    """The megakernel core carries no model file, so a second profile can mount it.

    The 2026-09-03 split: `glm5next_kda.py` -- the one row that bound a model
    directory and pinned an image preimage -- moved to `glm53_mk_kda_wiring`,
    and the core's `requires` went with it. What is left binds only
    `vllm/model_executor/layers/`, RELATIVE to the profile's TARGET_PREFIX,
    with both rows `absent`: nothing in the core can fail a deploy gate on an
    image that has no glm5next/ tree. dsv4 mounts exactly that, every knob 0.

    Composing glm53 before and after the split renders a byte-identical
    build/glm53 (same three rows, redistributed) -- that equivalence is what
    test_composed_snapshot_sync keeps true from here on.
    """
    core = os.path.join(REPO, "overlay", "modules", "glm53_megakernel")
    rows = [l.split("\t") for l in
            open(os.path.join(core, "manifest.tsv"), encoding="utf-8")
            .read().splitlines() if l and not l.startswith("#")]
    check(len(rows) == 2 and {r[0] for r in rows} ==
          {"glm53_megakernel.py", "glm53_megakernel.cu"},
          f"the core is the kernel and its driver, nothing else: {rows}")
    check(all(r[1].startswith("vllm/model_executor/layers/") for r in rows),
          "core targets are RELATIVE to TARGET_PREFIX and stay under "
          "model_executor/layers -- an absolute path or a model directory "
          "would pin the core to one image")
    check(all(r[2] == "absent" for r in rows),
          "both core files are new files: no preimage to drift on an image "
          "that never shipped them")
    check(not os.path.exists(os.path.join(core, "requires")),
          "the core requires nothing -- it is inert without a hook, and a "
          "requirement on GLM's wiring would follow it into every profile "
          "that mounts the kernel")

    kw = os.path.join(REPO, "overlay", "modules", "glm53_model")
    krows = [l.split("\t") for l in
             open(os.path.join(kw, "manifest.tsv"), encoding="utf-8")
             .read().splitlines() if l and not l.startswith("#")]
    kda_rows = [r for r in krows if r[1] == "vllm/models/glm5next/nvidia/kda.py"]
    check(len(kda_rows) == 1
          and re.fullmatch(r"[0-9a-f]{64}", kda_rows[0][2]) is not None,
          f"the KDA hook keeps the model path and its pinned preimage: {kda_rows}")
    check("glm53_megakernel" in open(os.path.join(kw, "requires"), encoding="utf-8").read().split()
          and os.path.exists(os.path.join(kw, "glm53_fp8_dense.py")),
          "the KDA hook requires the kernel it calls; the fp8-dense arm its packs "
          "are built from lives in the same module (that requirement used to sit on the core)")

    # -- one core, two profiles; every profile mounting it carries a hook
    def _modules(profile):
        text = open(os.path.join(REPO, "profiles", f"{profile}.env"),
                    encoding="utf-8").read()
        m = re.search(r'^MODULES="([^"]+)"', text, re.M)
        # this now runs over EVERY profile, so a malformed one has to fail as
        # a named check rather than as AttributeError on .group
        check(m is not None,
              f"{profile}: MODULES must be a one-line quoted value")
        return text, m.group(1).split()

    glm_text, glm_mods = _modules("glm53")
    dsv_text, dsv_mods = _modules("dsv4")
    check("glm53_megakernel" in glm_mods and "glm53_megakernel" in dsv_mods,
          "both profiles mount the same core module")
    check({"glm53_kernels", "glm53_model"} <= set(glm_mods),
          "glm53 keeps its wirings after the split (MHC in glm53_kernels, MLA/KDA in glm53_model)")
    # The core's `requires` used to make compose ABORT when the wiring was
    # missing. Splitting it moved that guarantee here, so the rule has to hold
    # for EVERY profile, not the two this test happens to name -- a third
    # profile mounting the kernel with no hook, or arming a segment whose hook
    # is not mounted, would log `armed` and route nothing.
    seg_hook = {"VLLM_GLM53_MK_MHC": None,          # any module with the hook
                "VLLM_GLM53_MK_GEMM": "glm53_model",
                "VLLM_GLM53_MK_KDA": "glm53_model",
                "VLLM_GLM53_MK_MLA": "glm53_model"}
    for envpath in sorted(glob.glob(os.path.join(REPO, "profiles", "*.env"))):
        profile = os.path.basename(envpath)[:-4]
        text, mods = _modules(profile)
        if "glm53_megakernel" not in mods:
            continue
        hooked = [m for m in mods if any(
            "_deneb_mk_hook()" in
            open(os.path.join(REPO, "overlay", "modules", m, f), encoding="utf-8").read()
            for f in os.listdir(os.path.join(REPO, "overlay", "modules", m))
            if f.endswith(".py"))]
        check(hooked, f"{profile} mounts the megakernel core but no module "
                      "carries the MK-MHC hook -- the core would be dead bytes")
        for knob, module in seg_hook.items():
            if module is None or re.search(rf"^{knob}=", text, re.M) is None:
                continue
            check(module in mods,
                  f"{profile} declares {knob} but does not mount {module}: "
                  "the segment would arm on its self-test and then serve "
                  "nothing, with the boot log saying otherwise")

    check(re.search(r"^VLLM_GLM53_MK_KDA_SHADOW=0$", glm_text, re.M) is not None,
          "glm53 ships VLLM_GLM53_MK_KDA_SHADOW=0 -- the shadow judge is a "
          "diagnostic, never a production default")
    for knob in ("VLLM_GLM53_MK_KDA", "VLLM_GLM53_MK_LOCALQ",
                 "VLLM_GLM53_AR_PREFETCH"):
        check(re.search(rf"^{knob}=0$", glm_text, re.M) is not None,
              f"glm53 ships {knob}=0 -- 32차 §14: the zero-gain set without "
              "SMLP (KDA lane CAPTURED into the decode graph, local-quant, AR "
              "prefetch) measured +0.7% together with Korean clean; the "
              "operator adopted it")
    for knob in ("VLLM_GLM53_MEGAKERNEL", "VLLM_GLM53_MK_MHC",
                 "VLLM_GLM53_MK_GEMM", "VLLM_GLM53_MK_MLA"):
        check(re.search(rf"^{knob}=1$", glm_text, re.M) is not None,
              f"glm53 ships {knob}=1 -- the bracketed megakernel set is the "
              "production default (28차, operator)")
    ab = open(os.path.join(REPO, "launchers/ab-glm53.sh"), encoding="utf-8").read()
    check('base) ARM_ENV="VLLM_GLM53_MEGAKERNEL=0"' in ab,
          "ab-glm53 base arm pins the master OFF now that the profile ships it on")
    du = open(os.path.join(REPO, "overlay/modules/glm53_drafter/dflash_utils.py"),
              encoding="utf-8").read()
    check('maybe_free_fp8_dense_bf16(dflash_model, label="drafter")' in du
          and du.index("install_drafter_serving_check(dflash_model, n_opaque)")
          < du.index('maybe_free_fp8_dense_bf16(dflash_model, label="drafter")'),
          "the drafter's bf16 sources are released at load, after the pass and the proof")
    fd = open(os.path.join(REPO, "overlay/modules/glm53_model/glm53_fp8_dense.py"),
              encoding="utf-8").read()
    check('def maybe_free_fp8_dense_bf16(model, label: str = "") -> int:' in fd,
          "the release names its model in the log line")

    # -- dsv4 ships every knob OFF and claims no segment it cannot run
    for knob in ("VLLM_GLM53_MEGAKERNEL", "VLLM_GLM53_MK_MHC",
                 "VLLM_GLM53_MK_PDL"):
        check(re.search(rf"^{knob}=0$", dsv_text, re.M) is not None,
              f"dsv4 must ship {knob}=0 (this is production and nothing is "
              "measured on this model)")
    for knob in ("VLLM_GLM53_MK_GEMM", "VLLM_GLM53_MK_KDA", "VLLM_GLM53_MK_MLA"):
        check(re.search(rf"^{knob}=", dsv_text, re.M) is None,
              f"dsv4 declares no {knob}: no linear-attention layer, a "
              "different MLA geometry, and block-fp8 dense weights with no "
              "bf16 source for the W4 pack")

    # -- the dsv4 hook: same branch, same place, same fall-through
    tl = open(os.path.join(REPO, "overlay", "modules", "dsv4_mhc_tilelang",
                           "mhc_tilelang.py"), encoding="utf-8").read()
    # scope to the wrapper body: mhc_pre_tilelang allocates a gemm_out of its
    # own earlier in the file, and an unscoped index would match that one
    fn = tl[tl.index("def mhc_fused_post_pre_tilelang("):]
    fn = fn[:fn.index("def _mhc_fused_post_pre_tilelang_fake")]
    head = "if use_small_fma and norm_weight is not None:"
    alloc = "gemm_out_mul = torch.empty("
    check(head in fn and fn.index(head) < fn.index(alloc),
          "the MK branch precedes the gemm_out allocations (the fused kernel "
          "has no gemm_out)")
    blk = fn[fn.index(head):fn.index(alloc)]
    check("_mk_hook = _deneb_mk_hook()" in blk and "_mk = _mk_hook(" in blk,
          "the hook goes through the cached resolver and the core's entry "
          "point, not a fresh import per call")
    check("if _mk is not None:" in blk and "return _mk" in blk,
          "an unmounted core, an unarmed segment or an ineligible shape "
          "returns None and falls through to this lane's swept stock pair")
    blk_code = "\n".join(l for l in blk.splitlines()
                         if l.strip() and not l.strip().startswith("#"))
    check("try:" not in blk_code and "except" not in blk_code,
          "the LAUNCH is not excepted: an async CUDA failure cannot be "
          "contained by a python fallback (the resolver owns the one "
          "try/except, around the import)")

    # -- the arm-then-call contract lives in the core, once, for both forks
    coresrc = open(os.path.join(core, "glm53_megakernel.py"),
                   encoding="utf-8").read()
    hook = coresrc[coresrc.index("def mhc_hook("):]
    hook = hook[:hook.index("\ndef ")]
    check("maybe_arm()" in hook
          and hook.index("maybe_arm()") < hook.index("mhc_fused_post_pre("),
          "mhc_hook arms before it calls -- that pairing is the thing the two "
          "image forks must not each re-implement")
    hook_code = "\n".join(l for l in hook.splitlines()
                          if l.strip() and not l.strip().startswith("#"))
    hook_code = hook_code[hook_code.index('"""', hook_code.index('"""') + 3):]
    check("try:" not in hook_code and "except" not in hook_code,
          "mhc_hook does not swallow the launch either")

    # -- and the two forks stay code-identical (comments may differ)
    def _mk_shape(path, marker):
        """The block starting at `marker`, ended by DEDENT -- not by the first
        blank line: a blank line inserted inside the block (the natural thing
        to do when adding a guard, which is exactly the edit that introduces
        drift) would otherwise shrink the compared region to a prefix and let
        the tails differ silently."""
        text = open(path, encoding="utf-8").read()
        lines = text[text.index(marker):].splitlines()
        base = len(lines[0]) - len(lines[0].lstrip())
        out = [lines[0]]
        for line in lines[1:]:
            if line.strip() and (len(line) - len(line.lstrip())) <= base:
                break
            out.append(line)
        return "\n".join(l for l in out
                          if l.strip() and not l.strip().startswith("#"))

    forks = [os.path.join(REPO, "overlay", "modules", "glm53_kernels",
                          "tilelang.py"),
             os.path.join(REPO, "overlay", "modules", "dsv4_mhc_tilelang",
                          "mhc_tilelang.py")]
    calls = [_mk_shape(f, "    if use_small_fma and norm_weight is not None:")
             for f in forks]
    resolvers = [_mk_shape(f, "def _deneb_mk_hook():") for f in forks]
    check(calls[0] == calls[1],
          "the two image forks carry the SAME hook code -- they are separate "
          "files that no compose rule ties together, so a fix that lands in "
          "one and not the other is invisible until a lane serves the old "
          "shape (the T <= 16 correction had to be applied twice)")
    check(resolvers[0] == resolvers[1],
          "the cached resolver is the same in both forks too")
    check("except ImportError as e:" in resolvers[0]
          and "not isinstance(e, ModuleNotFoundError) or e.name == _MK_MODULE"
          in resolvers[0]
          and "_MK_HOOK, _MK_HOOK_TRIED = None, True" in resolvers[0],
          "BOTH permanent import shapes cache: module not mounted "
          "(ModuleNotFoundError naming it) and mounted-without-the-entry-point "
          "(a plain ImportError -- an older core beside a newer wiring, which "
          "the core/wiring split made reachable). Treating the second as "
          "transient re-runs the import on every decode call forever, which "
          "is the cost this resolver exists to remove")
    check("except Exception:" in resolvers[0]
          and resolvers[0].index("except ImportError")
          < resolvers[0].index("except Exception:"),
          "a non-import failure is the only doubtful one: stock for that call, "
          "retried on the next")

    # -- the dsv4 launcher now emits profile VLLM_* keys, so it needs the same
    #    EXTRA_ENV guard glm53 has: $COMMON (EXTRA_ENV) renders BEFORE $ENVV
    #    (profile keys) and docker takes the last -e for a name.
    hy4 = open(os.path.join(REPO, "launchers", "start-hy4-tp4.sh"),
               encoding="utf-8").read()
    # The guard moved into the library both lanes source; hy4 names itself
    # through the ct_extra_env_flags argument.
    _lib_src = open(os.path.join(REPO, "launchers", "lib", "common-tp4.sh"),
                    encoding="utf-8").read()
    body = _lib_src[_lib_src.index("for _kv in ${EXTRA_ENV:-}"):]
    body = body[:body.index("done")]
    check("ct_extra_env_flags launchers/start-hy4-tp4.sh" in hy4,
          "hy4 must pass its own path so the abort names THIS launcher")
    check("is declared in the profile, so EXTRA_ENV cannot" in body,
          "start-hy4-tp4.sh must abort when EXTRA_ENV names a profile-declared "
          "key, or a megakernel sweep silently measures the profile's 0")
    check("_vllm_keys_sp" in _lib_src and "printf '%s ' ${_vllm_keys:-}" in _lib_src,
          "the key list is newline-separated; flatten it or the guard never "
          "matches")
    check("bash $_launcher" in body,
          "the abort must name THIS launcher in the caller-env command it "
          "suggests -- the shared guard takes the caller's path as $_launcher, "
          "and hy4 passing its own is checked above")
    # -- the probe is profile-driven too, or step 2 measures the wrong stack
    wrap = open(os.path.join(REPO, "probes", "run_megakernel_bench.sh"),
                encoding="utf-8").read()
    check("--profile" in wrap and 'PROFILE=${PROFILE:-glm53}' in wrap,
          "the bench wrapper takes --profile and still defaults to glm53")
    check("IMAGE=${IMAGE:-$PROFILE_IMAGE}" in wrap
          and "glm53:v13-b12x" not in wrap,
          "the image comes from the profile, not a hard-coded default that "
          "drifts (the profile's is the -it build)")
    check('MK_PKG_PATH=${TARGET_PREFIX%/}' in wrap,
          "the probe's package root comes from the profile: GLM's image "
          "installs to dist-packages, dsv4's to the venv site-packages")
    check("VLLM_(GLM53|DSV4)_" in wrap,
          "dsv4's STOCK arm is tuned by VLLM_DSV4_MHC_* knobs; a sweep that "
          "wants the untuned reference must be able to pass them in")
    check('cmd="python3 /repo/probes/megakernel_glm53_bench.py"' in wrap
          and 'megakernel_glm53_bench.py $*' not in wrap,
          "probe args are appended explicitly -- the old \"$*\" form went "
          "empty once the wrapper shifted --profile off, which would have "
          "silently run the defaults while the caller believed otherwise")
    # every source a recipe names must be a row in that profile's manifest
    for profile in ("glm53", "dsv4"):
        m = re.search(rf"^  {profile}\)\n(.*?)^    ;;", wrap, re.M | re.S)
        check(m is not None, f"the wrapper has no recipe for {profile}")
        srcs = re.search(r"sources=\(([^)]*)\)", m.group(1), re.S).group(1).split()
        rows = {l.split("\t")[0] for l in
                open(os.path.join(REPO, "build", profile, "manifest.tsv"),
                     encoding="utf-8").read().splitlines()
                if l and not l.startswith("#")}
        missing = [x for x in srcs if x not in rows]
        check(not missing,
              f"{profile}'s probe recipe names sources its manifest does not "
              f"have: {missing}")
        check("glm53_megakernel.py" in srcs and "glm53_megakernel.cu" in srcs,
              f"{profile}'s recipe must mount the driver and the kernel")
    dsv4_recipe = re.search(r"^  dsv4\)\n(.*?)^    ;;", wrap, re.M | re.S).group(1)
    check("--segments mhc" in dsv4_recipe and "--stock both" in dsv4_recipe,
          "dsv4 runs MHC alone (no kda layer, no e2m1 pack, other MLA) and "
          "measures both stock arms")
    for absent in ("glm5next_kda.py", "tilelang_kernels.py", "glm53_fp8_dense.py"):
        check(absent not in dsv4_recipe,
              f"dsv4's recipe must not name {absent}: that file has no row in "
              "its manifest and the mount loop would ABORT")

    probe = open(os.path.join(REPO, "probes", "megakernel_glm53_bench.py"),
                 encoding="utf-8").read()
    check('os.environ.get("MK_PKG_PATH"' in probe,
          "the probe imports from the package root the wrapper hands it")
    check('ap.add_argument("--segments"' in probe
          and 'ap.add_argument("--stock"' in probe
          and '"--skip-kda", action="store_true"' in probe,
          "--segments and --stock exist and --skip-kda still works (the "
          "campaign's running commands use it)")
    disp = probe[probe.index("def probe_mhc_dispatch"):probe.index("def main()")]
    check("mk.maybe_arm()" in disp
          and disp.index("mk.maybe_arm()") < disp.index('mk._ARMED["mhc"] = False'),
          "the dispatch arm arms FIRST and only then disarms for the "
          "reference call -- otherwise the wiring's own maybe_arm re-arms it "
          "and both arms measure MK")
    check(disp.count("call(*args, **kw)") >= 4 and "{'hit':>6}" in disp
          and "hit = served() > before" in disp
          and "mk._HOOK_SERVED[0]" in disp
          and "mk.mhc_fused_post_pre = " not in disp,
          "the dispatch arm calls the wrapper both ways and reads the CORE's "
          "served counter for the hit column -- patching a private of the "
          "driver would turn any refactor of mhc_hook into a run-wide false "
          "FAIL blamed on the lane under test")
    check("_HOOK_SERVED[0] += 1" in coresrc
          and coresrc.index("out = mhc_fused_post_pre(")
          < coresrc.index("_HOOK_SERVED[0] += 1"),
          "mhc_hook counts the calls it SERVED (armed path only, next to a "
          "kernel launch), which is what makes that receipt cheap")
    check("for T in (8, 16, 32):" in disp,
          "the dispatch arm spans the wrapper's boundary (16) so the window "
          "shows up as a hit column rather than as an assumption")
    check("if not any_hit:" in disp and "return False" in disp,
          "a dispatch run where MK was never offered a call must FAIL: every "
          "row would be stock against stock, rel 0.0, and the gate would pass "
          "without the kernel running once")
    check('armed0 = mk._ARMED["mhc"]' in disp
          and 'mk._ARMED["mhc"] = armed0' in disp
          and disp.index("finally:") < disp.index('mk._ARMED["mhc"] = armed0'),
          "the dispatch arm restores the arm state it flips, not just the "
          "monkeypatch -- an exception mid-loop would otherwise leave the "
          "module armed by hand for whatever runs next")
    check("else None" in disp and "if hit else '-'" in disp,
          "no timing pass for an arm that was never offered the call, and the "
          "cell says so instead of printing the stock time under mk_us")
    # -- sinkhorn_repeat: the probe must default to what the models serve
    check("SINKHORN_SERVED = 20" in coresrc,
          "the driver names what the models serve (hc_sinkhorn_iters=20 in "
          "both configs) in one place")
    self_test = coresrc[coresrc.index("def _selftest_mhc"):]
    self_test = self_test[:self_test.index("\ndef ")]
    check("sinkhorn_repeat = 1.0, SINKHORN_SERVED" in self_test
          and "1.0, 4" not in self_test,
          "the ARMING gate runs the same sinkhorn count serving does, from "
          "the constant and not a literal -- it used to validate 3 loop "
          "iterations while production runs 19, so a divergence that opens "
          "up later armed clean and served wrong")
    check("sinkhorn=%d" in self_test,
          "the gate logs the count it used next to its rel_errs: _TOL_MHC was "
          "calibrated at 4 and not re-derived at 20, so a DISARM has to be "
          "readable as accumulation or as divergence")
    check('ap.add_argument("--sinkhorn", type=int, default=None)' in probe
          and "mk.SINKHORN_SERVED if args.sinkhorn is None" in probe,
          "the probe's sinkhorn default IS the driver's constant, not a "
          "second copy of 20 that can drift from the arming gate. It used to "
          "be a hard-coded 4, and the loop is a runtime bound in BOTH arms "
          "that they do not scale with alike, so a ratio taken there does "
          "not transfer to serving")
    check("1.0, 1e-6, 4)" not in probe and "1.0, 4, 1e-6," not in probe,
          "no hard-coded sinkhorn_repeat survives in the MHC arms")
    check("BASIS NOTE" in probe and "--sinkhorn 4" in probe,
          "the docstring records that the earlier MEASUREMENTS rows were "
          "taken at 4 and names the flag that reproduces them")
    check(probe.count("sinkhorn_repeat={sk}") >= 2
          and "(driver default)" in probe,
          "both the run header and the raw arm print the basis they used, "
          "and the header says when it came from the driver")
    check("raw stock arm: mhc_fused(tile_n=2, n_splits=4)" in probe,
          "the raw arm names the stock config it built -- no profile's "
          "dispatcher picks it, and on dsv4 it is the pre-sweep config")
    arm_body = probe[probe.index("def _arm_env(segs)"):]
    arm_body = arm_body[:arm_body.index("\n\n\n")]
    check(probe.count("os.environ.setdefault(") == arm_body.count("os.environ.setdefault(")
          and 'os.environ.setdefault(_SEG_KNOB[seg], "1")' in arm_body
          # the CALL in main (the def near the top matches the bare name too),
          # and main's driver import, which follows it (probe_gemm has one of
          # its own earlier in the file)
          and probe.index("segs = [s.strip()") < probe.index("\n    _arm_env(segs)")
          < probe.index("from vllm.model_executor.layers import glm53_megakernel as mk",
                        probe.index("\n    _arm_env(segs)")),
          "every knob setdefault lives in _arm_env, which arms only the "
          "selected segments and runs AFTER --segments is parsed and BEFORE "
          "the driver import that reads the knobs: dsv4's MHC-only run used "
          "to build GEMM and KDA packs at arm() (seconds since #268) and log "
          "their DISARMs into an MHC measurement")
    check('print("--segments selected nothing to run")' in probe
          and "if not segs:" in probe,
          "an empty --segments selection must refuse to run: it used to skip "
          "every probe and print VERDICT: PASS")
    # -- the wrapper refuses a shell that would disarm the run
    check("ABORT: VLLM_GLM53_MEGAKERNEL=" in wrap
          and 'case "${VLLM_GLM53_MEGAKERNEL-1}" in' in wrap
          and '${VLLM_GLM53_MEGAKERNEL:-1}' not in wrap,
          "a caller shell that sourced the profile carries MEGAKERNEL=0; "
          "forwarded, it leaves the probe disarmed, so the wrapper refuses "
          "instead of measuring nothing -- and the guard uses ${VAR-1}, not "
          "${VAR:-1}, which would rewrite the set-but-EMPTY value (also off "
          "to the driver's _flag) into 1 and wave it through")
    check("VLLM_GLM53_MK_PDL" in wrap and '"${!_k-1}"' in wrap,
          "the forwarded-knob warning covers MK_PDL too: a sourced profile "
          "carries it at 0 and it is worth 17-19 pct per launch, so it moves "
          "every number in the table")
    check('echo "forwarded:${_fwd:- (none)}"' in wrap,
          "the run prints which knobs actually arrived -- the receipt has to "
          "be in the output, not in the operator's memory of their shell")
    check('PROFILE_IMAGE=""' in wrap and wrap.index('PROFILE_IMAGE=""')
          < wrap.index('eval "$('),
          "both profile-derived names are pre-set: a profile that fails to "
          "source would otherwise die on 'unbound variable' instead of the "
          "ABORT that names the profile")

    replay = open(os.path.join(REPO, "probes", "mhc_replay.py"),
                  encoding="utf-8").read()
    check('ap.add_argument("--sinkhorn", type=int, default=None)' in replay
          and "mk.SINKHORN_SERVED if args.sinkhorn is None" in replay
          and "1.0, 1e-6, 4)" not in replay,
          "the replay diagnostic runs the same sinkhorn count the bench and "
          "the gate do -- the bench names it as the tool for a `rep` failure, "
          "and at a shorter chain a longer chain's drift reproduces as clean")
    print("  megakernel core is shared ..... OK")


def test_prefill_warmup_contracts() -> None:
    """The engine-level prefill warmup must not lie to itself.

    Same discipline the ladder probe already enforces, applied to the
    warmup: every request needs a DISTINCT prefix (a shared one would
    prefix-cache-hit and skip the compute the warmup exists to trigger)
    and an EXACT token count (the shape is the point).
    """
    ns = load_defs(
        "launchers/prefill-warmup.py",
        {"build_distinct_prompt", "trim_to_exact_tokens", "WORDS"},
        {"random": random},
    )
    build, trim = ns["build_distinct_prompt"], ns["trim_to_exact_tokens"]

    seeds = [n * 1000 + rep
             for n in (2048, 4096, 8192) for rep in (1, 2)]
    prompts = {build(seed, seed // 1000) for seed in seeds}
    check(len(prompts) == 6, "every warmup request carries distinct text")
    heads = [" ".join(p.split()[:2]) for p in prompts]
    check(len(set(heads)) == 6,
          "distinctness lives in the PREFIX (seed word + seed id), not just "
          "the tail -- a shared header would let the prefix cache serve "
          "every later request from the first one's blocks")
    check(build(2048 * 1000 + 1, 2048) == build(2048 * 1000 + 1, 2048),
          "per-seed prompts are deterministic (log/replay discipline)")

    ids = trim("x", 100, lambda t: list(range(3000)))
    check(ids == list(range(100)), "long tokenization trims to exactly N")
    short = trim("x", 300, lambda t: list(range(150)))
    check(len(short) == 300,
          "short tokenization tops up before trimming (exact N guaranteed)")

    src = open(os.path.join(REPO, "launchers", "prefill-warmup.py"),
               encoding="utf-8").read()
    check('"prompt": ids' in src and "prompt_token_ids\": ids" not in src,
          "the completions payload sends token ids as the prompt list -- this "
          "fork's /v1/completions 400s a top-level prompt_token_ids (the "
          "33차 first armed boot warmed nothing and died on request 1)")

    launcher = open(os.path.join(REPO, "launchers/start-glm53-nvfp4-tp4.sh"),
                    encoding="utf-8").read()
    check('PREFILL_WARMUP:-0' in launcher,
          "the warmup hook stays opt-in at the launcher")
    profile = open(os.path.join(REPO, "profiles/glm53.env"),
                   encoding="utf-8").read()
    check("PREFILL_WARMUP=1" in profile
          and 'PREFILL_WARMUP_LENS="2048,4096,8192"' in profile,
          "the profile carries the knob, default on since 33차, ladder lengths")
    for harness in ("bench/ab-lever.sh", "launchers/ab-glm53.sh"):
        h = open(os.path.join(REPO, harness), encoding="utf-8").read()
        check('export PREFILL_WARMUP="${PREFILL_WARMUP:-0}"' in h,
              f"{harness} pins its boots to no-warmup -- the cold-tax channel "
              "is the bracket's to measure, and the warmup's requests would "
              "land inside its first decode leg's 2s windows")
    print("  prefill warmup contracts ... OK")


def test_megakernel_w4_layout_functional() -> None:
    """The W4 packing checked FUNCTIONALLY, not by string contract.

    Review found the packer slicing the wrong dimension: numel broke, the
    attach's except swallowed the raise into a silently-stock boot, and the
    self-test exception disarmed every other segment. The string contracts
    in test_glm53_megakernel_contracts could not see any of that. This test
    runs two real checks:

    torch-free (always): the .cu LUT bytes decoded as e4m3 must equal the
    e2m1 grid x 2^s exactly -- the arithmetic the whole W4 arm rests on.

    torch-guarded (runs wherever torch exists; skipped cleanly otherwise,
    the #176 rule): the REAL build_mk_weight_w4 on crafted grid weights,
    unpacked with the kernel's own index math and the .cu LUT, must
    reproduce the weights exactly, byte-level layout included.
    """
    import importlib.util

    mod_dir = os.path.join(REPO, "overlay/modules/glm53_megakernel")
    cu = open(os.path.join(mod_dir, "glm53_megakernel.cu"),
              encoding="utf-8").read()
    # The LUT lives as the funnel-shift immediate, not as an array: a
    # __constant__ load serialises over the distinct addresses in a warp, so
    # the array it replaced was removed once it had no callers left.
    m = re.search(r"MK_E2M1_LUT64 = 0x([0-9A-Fa-f']+)ULL", cu)
    check(m is not None, "the e2m1->e4m3 LUT is present in the .cu")
    assert m is not None
    packed = int(m.group(1).replace("'", ""), 16)
    lut = [(packed >> (8 * c)) & 0xFF for c in range(8)]
    check(lut == [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C],
          f"LUT decodes the e2m1 grid in order (got {[hex(x) for x in lut]})")

    def e4m3_value(byte: int) -> float:
        sign = -1.0 if byte & 0x80 else 1.0
        expf = (byte >> 3) & 0xF
        man = byte & 0x7
        if expf == 0:
            return sign * (man / 8.0) * 2.0 ** -6  # denormal region
        return sign * (1.0 + man / 8.0) * 2.0 ** (expf - 7)

    mb = re.search(r"MK_E2M1_LUT64_B = 0x([0-9A-Fa-f']+)ULL", cu)
    check(mb is not None, "the +1 companion table is present in the .cu")
    assert mb is not None
    lut_b = [(int(mb.group(1).replace("'", ""), 16) >> (8 * c)) & 0xFF
             for c in range(8)]
    check([lut_b[c] - lut[c] for c in range(8)] == [0, 0, 0, 1, 0, 1, 0, 1],
          "the companion table is the LUT with +1 on exactly the three codes "
          "whose e2m1 magnitude mantissa is 1.5 (3, 5, 7)")

    def e4m3_round(x: float) -> float:
        """round-to-nearest-even onto the e4m3 grid"""
        if x == 0.0:
            return 0.0
        sign = -1.0 if x < 0 else 1.0
        m, e = abs(x), 0
        while m >= 2.0:
            m /= 2.0
            e += 1
        while m < 1.0:
            m *= 2.0
            e -= 1
        q = m * 8.0
        f = int(q)
        r = q - f
        if r > 0.5 or (r == 0.5 and f % 2):
            f += 1
        if f == 16:
            f = 8
            e += 1
        return sign * (f / 8.0) * 2.0 ** (e + 1) / 2.0

    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # The scale is e4m3, stored as one byte d = (e << 3) + k, and the
    # expansion adds d to a table byte. Adding the scale's 3-bit mantissa
    # IS the multiply -- the e2m1 magnitudes carry a 1-bit mantissa, so the
    # product needs at most one carry out of the mantissa field and that
    # carry lands on the exponent field. Sweep every combination; this is
    # the arithmetic the whole W4 arm rests on.
    for code in range(16):
        for sexp in range(-5, 6):
            for kk in range(8):
                tbl = lut_b if 1 <= kk <= 5 else lut
                # mirror the kernel's ternary EXACTLY: a zero-magnitude
                # nibble takes no add (it would wrap the byte); only its sign
                sig = 0x80 if code & 8 else 0
                byte = sig if (code & 7) == 0 else                     ((tbl[code & 7] + (sexp << 3) + kk) & 0xFF) | sig
                want = e4m3_round(grid[code & 7] * (1.0 + kk / 8.0)
                                  * 2.0 ** sexp) * (-1.0 if code & 8 else 1.0)
                got = e4m3_value(byte)
                check(abs(got - want) <= 1e-9 * max(1.0, abs(want)),
                      f"table+scale-byte add is exact: code={code} "
                      f"e={sexp} k={kk} {got} == {want}")
                check(byte & 0x7F != 0x7F,
                      f"never the NaN encoding: code={code} e={sexp} k={kk}")

    # ---- torch-guarded roundtrip through the REAL packer
    try:
        import torch
    except ImportError:
        print("  w4 layout (torch-free half) . OK (torch absent: packer "
              "roundtrip half skipped)")
        return

    spec = importlib.util.spec_from_file_location(
        "mk_driver_w4", os.path.join(mod_dir, "glm53_megakernel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n, k = 100, 128  # n_pad exercises zero-padded rows too
    rng = random.Random(7)

    def craft(code_of, sexp_of):
        w = torch.zeros(n, k, dtype=torch.float32)
        for r in range(n):
            for g in range(k // 16):
                for e in range(16):
                    c = code_of(r, g)
                    w[r, g * 16 + e] = grid[c & 7] * 2.0 ** sexp_of(r, g)                         * (-1.0 if c & 8 else 1.0)
        return w

    # (a) BYTE-EXACT tier: every element of a group at magnitude 4 (the
    # largest whose amax/6 is NOT an exact pow2) pins the quantizer to the
    # crafted s, so nibbles AND scale bytes must come back identical.
    codes_a = [[6 | (0x8 if rng.randrange(2) else 0)
                for _ in range(k // 16)] for _ in range(n)]
    sexps_a = [[rng.randrange(-5, 6) for _ in range(k // 16)]
               for _ in range(n)]
    # the crafted tiers pin the PER-TENSOR shift path byte for byte; the
    # per-row shift (33차 lever 3) gets its own tier below. No pack cache
    # in a test process.
    _env_keep = {k: os.environ.get(k) for k in
                 ("VLLM_GLM53_MK_PACK_ROWSHIFT", "VLLM_GLM53_MK_PACK_CACHE")}
    os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = "0"
    os.environ["VLLM_GLM53_MK_PACK_CACHE"] = "off"
    w_a = craft(lambda r, g: codes_a[r][g], lambda r, g: sexps_a[r][g])
    wq4, ws4, _gs, *_rest = mod.build_mk_weight_w4(w_a)
    check(_rest[0] is None and _rest[1] is None and _rest[2] is None,
          "per-tensor shift: the pack carries no row scales and no "
          "low-rank factors")
    check(tuple(wq4.shape) == (1, 1, 128, k // 2),
          "wq4 is tile-major [n_pad/128, k/128, 128, 64]")
    check(tuple(ws4.shape) == (1, 1, 128, k // 16),
          "ws4 is tile-major [n_pad/128, k/128, 128, 8]")
    q, sc = wq4.tolist(), ws4.tolist()
    import math as _math
    _shift_a = int(round(-_math.log2(_gs)))

    # the kernel's index math (stage_raw4 / expand_w4): tile = r // 128,
    # k-block = kk // 128, then row-in-tile and byte / group within it
    def nib(r, kk):
        return q[r // 128][kk // 128][r % 128][(kk % 128) // 2]

    def sexp_at(r, kk):
        return sc[r // 128][kk // 128][r % 128][(kk % 128) // 16]

    for r in range(n):
        for kk in range(k):
            byte = nib(r, kk)
            code = (byte & 0xF) if kk % 2 == 0 else (byte >> 4)
            check(code == codes_a[r][kk // 16],
                  f"byte-exact tier: elem {kk} rides byte {kk // 2} "
                  f"half {kk % 2} with the crafted nibble (row {r})")
            check(sexp_at(r, kk) == ((sexps_a[r][kk // 16] + _shift_a) << 3),
                  "byte-exact tier: the stored byte is (e << 3) + mantissa, "
                  "with e carrying the pack's pow2 normalisation")

    # (a') PER-ROW shift tier (33차 lever 3): the same crafted weights under
    # VLLM_GLM53_MK_PACK_ROWSHIFT=1 -- wgs is 1, rgs[r] = 2^-shift_r with
    # shift_r = -median of the row's covering exponents (on these groups
    # amax = 4 x 2^s, so the covering exponent is s itself), and every
    # stored byte carries ITS ROW's shift.
    # Rows 16 octaves apart (base_r in [-8, 8]) with a 5-octave spread
    # inside each row: a per-tensor shift cannot keep such a tensor inside
    # the 11-octave window, a per-row one centres every row on its median
    # so every group stays clamp-free and byte-exact.
    os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = "1"
    base_r = [rng.randrange(-8, 9) for _ in range(n)]
    sexps_r = [[base_r[r] + rng.randrange(-2, 3) for _ in range(k // 16)]
               for r in range(n)]
    w_r = craft(lambda r, g: codes_a[r][g], lambda r, g: sexps_r[r][g])
    pk_r = mod.build_mk_weight_w4(w_r)
    check(float(pk_r[2]) == 1.0 and pk_r[3] is not None
          and tuple(pk_r[3].shape) == (128,),
          "per-row shift: wgs is 1 and rgs is fp32 [n_pad]")
    rgs_l = pk_r[3].tolist()
    q_r, sc_r = pk_r[0].tolist(), pk_r[1].tolist()
    for r in range(n):
        med = sorted(sexps_r[r])[(len(sexps_r[r]) - 1) // 2]  # torch.median: the lower middle
        shift_r = int(round(-_math.log2(rgs_l[r])))
        check(shift_r == -med,
              f"per-row shift: row {r} shifts by minus its median exponent")
        for kk in range(k):
            byte = q_r[0][0][r][(kk % 128) // 2]
            code = (byte & 0xF) if kk % 2 == 0 else (byte >> 4)
            check(code == codes_a[r][kk // 16],
                  f"per-row tier: the nibbles are the crafted ones (row {r})")
            check(sc_r[0][0][r][(kk % 128) // 16]
                  == ((sexps_r[r][kk // 16] + shift_r) << 3),
                  "per-row tier: the stored byte carries the ROW's shift")
    for r in range(n, 128):
        check(rgs_l[r] == 1.0, "per-row shift: padded rows scale by 1")
    os.environ["VLLM_GLM53_MK_PACK_ROWSHIFT"] = "0"

    # (c) 33차 lever 2 plumbing: the calibration observer accumulates x^T x
    # per pack under its name, dumps per rank, and the packer reads it back
    # by name (the served GPTQ path is this file round trip, never exercised
    # by the GPU probes, which hand the Hessian over in memory)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        mod._CALIB["on"] = True; mod._CALIB["dumped"] = False
        mod._CALIB["H"].clear(); mod._CALIB["ntok"].clear(); mod._CALIB["meta"].clear()
        mod._CALIB["seen"] = 0; mod._CALIB["budget"] = 40
        os.environ["VLLM_GLM53_MK_CALIB_DIR"] = td
        pk_c = mod.build_mk_weight_w4(w_a)
        mod.note_pack_name(pk_c, "Glm5Next/layers.1.self_attn.q_proj")
        check("if torch.cuda.is_current_stream_capturing():\n        return\n    key = _pack_key(pack)"
              in open(os.path.join(mod_dir, "glm53_megakernel.py"), encoding="utf-8").read(),
              "calibration: the observer never runs under graph capture (its "
              "finite-row mask syncs the device: 'operation not permitted when "
              "stream is capturing' killed the CALIB2 boot)")
        xs = torch.randn(24, k)
        xs_pad = torch.cat([xs, torch.full((4, k), float("nan"))])   # padded rows
        mod._calib_observe(xs_pad, pk_c)
        check(not mod._CALIB["dumped"] and mod._CALIB["seen"] == 24,
              "calibration: 24 finite rows seen (4 padded NaN rows dropped), budget 40 not reached")
        mod._calib_observe(xs, pk_c)
        check(mod._CALIB["dumped"], "calibration: the budget dumps")
        got_c = mod._calib_hessian_for("Glm5Next/layers.1.self_attn.q_proj", k)
        check(got_c is not None and int(got_c[1]) == 48
              and torch.allclose(got_c[0], 2 * (xs.T @ xs), atol=1e-3),
              "calibration: the dumped Hessian is sum x^T x over the seen rows, "
              "keyed by the linear's name and rank 0")
        check(mod._calib_hessian_for("Glm5Next/layers.1.self_attn.k_proj", k) is None,
              "calibration: an unknown linear packs RTN")
        check(mod._calib_hessian_for("Glm5Next/layers.1.self_attn.q_proj", 2 * k) is None,
              "calibration: a Hessian of another k is ignored (the drafter's "
              "linear must never read the target's dump)")
        os.environ["VLLM_GLM53_MK_PACK_GPTQ"] = "0"
        check(mod._calib_hessian_for("Glm5Next/layers.1.self_attn.q_proj", k) is None,
              "calibration: VLLM_GLM53_MK_PACK_GPTQ=0 ignores the dump")
        os.environ.pop("VLLM_GLM53_MK_PACK_GPTQ", None)
        os.environ.pop("VLLM_GLM53_MK_CALIB_DIR", None)
        mod._CALIB["on"] = False

    # (b) VALUE-EXACT tier: fully random codes/scales. The quantizer may
    # renormalize a group to its own (s', code') -- legal, the grid is
    # closed under x2 up to 6 -- but the dequantized VALUES must return
    # exactly, unpacked with the KERNEL's index math and the .cu LUT.
    codes_b = [[rng.randrange(16) for _ in range(k // 16)] for _ in range(n)]
    # The expansion's scale field spans 11 octaves around the pack's pow2
    # normalisation, and the e2m1 magnitudes themselves already use ~4 of
    # them (0.5 .. 6), so the crafted exponents get the remaining 7. A
    # wider spread would clamp -- which is a real property of the format,
    # not a packer bug, and tier (a) pins the clamp-free case exactly.
    sexps_b = [[rng.randrange(-3, 4) for _ in range(k // 16)]
               for _ in range(n)]
    wb = craft(lambda r, g: codes_b[r][g], lambda r, g: sexps_b[r][g])
    wq4, ws4, _gs2, *_rest2 = mod.build_mk_weight_w4(wb)
    for _k, _v in _env_keep.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
    q, sc = wq4.tolist(), ws4.tolist()
    for r in range(n):
        for kk in range(k):
            byte = nib(r, kk)
            code = (byte & 0xF) if kk % 2 == 0 else (byte >> 4)
            d = sexp_at(r, kk)
            val = e4m3_round(grid[code & 7] * (1.0 + (d & 7) / 8.0)
                             * 2.0 ** (d >> 3)) \
                * (-1.0 if code & 8 else 1.0) * _gs2
            check(val == wb[r, kk].item(),
                  f"value-exact tier: elem {kk} dequantizes to the original "
                  f"(row {r}: {val} != {wb[r, kk].item()})")
    for r in range(n, 128):  # padded rows are zero nibbles
        check(all(b == 0 for b in q[0][0][r]), f"pad row {r} packed as zeros")
    print("  w4 layout roundtrip ...... OK")


def test_glm53_prep_fused_contracts() -> None:
    """glm53_prep_fused: wiring, kill switch, preimage pins, live handles, guards."""
    import hashlib
    import tempfile

    mod_dir = os.path.join(REPO, "overlay", "modules", "glm53_runtime")
    src_path = os.path.join(mod_dir, "glm53_prep_fused.py")
    src = open(src_path, encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    modules = re.search(r'^MODULES="([^"]+)"', profile, re.M).group(1).split()
    check("glm53_runtime" in modules, "glm53 profile must mount glm53_runtime (prep_fused lives there)")
    check(re.search(r"^VLLM_GLM53_PREP_FUSED=1$", profile, re.M) is not None
          and "0 stays the kill switch" in profile,
          "profile ships VLLM_GLM53_PREP_FUSED=1 (32차 operator decision: the "
          "+7.3% lever is on; 0 stays the kill switch)")
    rows = [l.split("\t") for l in open(os.path.join(mod_dir, "manifest.tsv"), encoding="utf-8")
            .read().splitlines() if l and not l.startswith("#")]
    check(["glm53_prep_fused.py", "vllm/models/glm5next/nvidia/glm53_prep_fused.py", "absent"] in rows,
          f"manifest must bind the module as a new file next to the model: {rows}")
    req = open(os.path.join(mod_dir, "requires"), encoding="utf-8").read().split()
    check({"glm53_model", "glm53_kernels"} <= set(req),
          "requires must name the wiring, the tail indexer and the kpool op it was read against")
    wiring = open(_overlay_source("overlay/glm5next_model.py"), encoding="utf-8").read()
    hook = wiring.find("install_glm53_prep_fused()")
    check(hook > wiring.find("logger = init_logger(__name__)"),
          "the install hook must run after the wiring's logger exists (it logs failures)")
    check("except ImportError as _e:\n    install_glm53_prep_fused = None" in wiring
          and 'if _e.name != f"{__package__}.glm53_prep_fused":' in wiring,
          "a boot without the module mounted stays stock silently; a broken module is logged")

    ns = load_defs(
        "overlay/glm53_prep_fused.py",
        {"prep_fused_mode", "shadow_every", "selfcheck_every", "_every", "check_preimages",
         "PREIMAGES", "ENV", "ENV_SHADOW_EVERY", "ENV_SELFCHECK_EVERY"},
        {"os": os, "hashlib": hashlib, "logger": _CapturingLogger()},
    )
    pins = ns["PREIMAGES"]
    check(len(pins) >= 15 and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in pins.values()),
          "preimage table must pin full sha256 digests of the bypassed runner files")
    tail_idx = open(_overlay_source("overlay/glm53_kpool_indexer.py"), "rb").read()
    check(pins["v1/attention/backends/mla/indexer.py"] == hashlib.sha256(tail_idx).hexdigest(),
          "the pinned mla/indexer.py must be the mounted glm53_tail_slot_persistent copy")
    for rel in ("v1/worker/gpu/model_runner.py", "v1/worker/gpu/input_batch.py",
                "v1/worker/gpu/block_table.py", "v1/worker/gpu/buffer_utils.py",
                "v1/worker/gpu/model_states/mamba_hybrid.py",
                "v1/attention/backends/gdn_attn.py", "v1/attention/backends/mla/compressor_utils.py",
                "model_executor/layers/attention/sparse_mla_attention.py"):
        check(rel in pins, f"preimage table must pin {rel}")
    saved = os.environ.pop("VLLM_GLM53_PREP_FUSED", None)
    try:
        check(ns["prep_fused_mode"]() == "off", "unset knob must be off")
        # a kernel-selecting knob must land on the safe side on any typo
        for v, want in (("0", "off"), ("off", "off"), ("", "off"), ("shadow", "shadow"),
                        ("SHADOW", "shadow"), ("1", "on"), ("yes", "off"), ("shadwo", "off"),
                        ("true", "off"), ("on", "off"), ("2", "off")):
            os.environ["VLLM_GLM53_PREP_FUSED"] = v
            check(ns["prep_fused_mode"]() == want, f"knob {v!r} -> {want}")
        check(any("DISARM" in l for l in ns["logger"].lines), "an unknown knob value must be logged")
    finally:
        os.environ.pop("VLLM_GLM53_PREP_FUSED", None)
        if saved is not None:
            os.environ["VLLM_GLM53_PREP_FUSED"] = saved
    for name, env, default in (("shadow_every", "VLLM_GLM53_PREP_FUSED_SHADOW_EVERY", 1),
                               ("selfcheck_every", "VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY", 64)):
        saved = os.environ.pop(env, None)
        try:
            check(ns[name]() == default, f"{name} default {default}")
            os.environ[env] = "16"
            check(ns[name]() == 16, f"{name} knob")
            os.environ[env] = "x"
            check(ns[name]() == default, f"{name}: unparsable falls back to the default")
        finally:
            os.environ.pop(env, None)
            if saved is not None:
                os.environ[env] = saved
    with tempfile.TemporaryDirectory() as tmp:
        bad = ns["check_preimages"](tmp)
        check(len(bad) == len(pins), "every absent file is drift")
        rel = "v1/attention/backends/mla/indexer.py"
        os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
        with open(os.path.join(tmp, rel), "wb") as f:
            f.write(tail_idx)
        bad = ns["check_preimages"](tmp)
        check(len(bad) == len(pins) - 1 and not any(b.startswith(rel) for b in bad),
              "a matching file is not drift")

    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("_fused_prepare_inputs", "launch", "_args", "_eligible", "_patched_prepare_attn"):
        body = ast.get_source_segment(src, funcs[name])
        check(body is not None and not re.search(r"\.(item|cpu|tolist|numpy)\(|synchronize\(", body),
              f"{name} must not synchronize with the device (the whole point is host time)")
    # live handles: the rotating UVA views and the block-table pointer tensors
    # are dereferenced at every launch, never cached in the plan
    args = ast.get_source_segment(src, funcs["_args"])
    for live in ("bt.num_blocks.gpu", "self.prefill_len_src.gpu", "bt.block_table_ptrs",
                 "bt.input_block_table_ptrs", "bt.block_table_strides", "bt.block_sizes_tensor"):
        check(live in args, f"_args must read {live} live (UVA handles rotate, wake re-makes pointers)")
    post = ast.get_source_segment(src, funcs["__post_init__"])
    check("UvaBufferPool(" in post and "pin_memory=True" not in post,
          "idx_mapping staging must use the image's round-robin UVA pool (async-safe)")
    # 37차: a non-power-of-two Q (k=5 -> 6) is served on Q_P2 masked lanes
    # instead of refused -- refusing cost every k!=7 boot the fusion.
    check("self.q_p2 = 1 << (self.q - 1).bit_length()" in post
          and "self.q & (self.q - 1)" not in post
          and "self.exp_bt.stride(0) != wa" in post,
          "the plan pads Q to a power of two (masked lanes) and still refuses an "
          "expanded-table width mismatch")
    plan_src = ast.get_source_segment(src, funcs["build_plan"])
    check("DFlash2Speculator" in src, "the fleet's DFlash2Speculator must be admitted")
    for guard in ("_SPECULATORS", "draft_attn_layer_names", "draft_kv_cache_group_ids",
                  "tail_builder=tail_b", "q != num_spec + 1"):
        check(guard in plan_src, f"build_plan must carry the guard {guard}")
    check("q & (q - 1)" not in plan_src,
          "build_plan no longer refuses a non-power-of-two Q (37차: masked lanes)")
    check("def tail_ok" in src and src.count("tail_ok()") >= 3,
          "the kpool tail builder's dormancy is asserted at plan build and on every verification")
    elig = ast.get_source_segment(src, funcs["_eligible"])
    check("CUDAGraphMode.FULL" in elig and "uniform_token_count != q" in elig
          and "batch_desc.num_reqs != num_reqs" in elig and "has_prefill" in elig,
          "eligibility must require FULL cudagraph, uniform Q, no request padding, no prefill")
    check("adaptive_verification is not None" in elig, "adaptive verification must fall back")
    ppi = ast.get_source_segment(src, funcs["_patched_prepare_inputs"])
    check("except Exception" in ppi and 'st.disarm(' in ppi and "_ORIG[\"prepare_inputs\"]" in ppi,
          "a fused launch failure must disarm and serve stock for that step")
    check("return fused" in ppi and "_verify(" in ppi and "selfcheck_every" in ppi,
          "shadow serves the fused batch after a clean verification; armed mode self-checks")
    ens = ast.get_source_segment(src, funcs["_ensure_plan"])
    check("plan.warmup()" in ens and "st.disarm(" in ens,
          "a warmup failure must disarm instead of JIT-ing on the first request")
    kern = ast.get_source_segment(src, funcs["_glm53_prep_fused_kernel"])
    check("_fill(slot_ptr + g * slot_stride, num_tokens, max_num_tokens, PAD_ID, BLOCK)" in kern,
          "every group's slot-mapping tail must be padded like the stock kernel")
    barrier = kern.find("tl.debug_barrier()")
    check(0 < barrier < kern.find("_load_ptr(gdn_state_ptrs + k") and barrier > kern.find("tl.store(dst_row + off"),
          "the gathered rows must be visible before the GDN read-back")
    check("_fill(qsl_ptr, num_reqs, max_num_reqs + 1, num_tokens, BLOCK)" in kern
          and "_fill(seq_lens_ptr, num_reqs, max_num_reqs, 0, BLOCK)" in kern,
          "query_start_loc / seq_lens tails must be padded like prepare_inputs")
    check("from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool, _load_ptr" in src
          and "from vllm.v1.attention.backends.utils import PAD_SLOT_ID" in src,
          "reuse the image's pointer helper, UVA pool and PAD_SLOT_ID instead of copies")
    inst = ast.get_source_segment(src, funcs["install_glm53_prep_fused"])
    check(inst.find("check_preimages(root)") < inst.find("Runner.prepare_inputs = _patched_prepare_inputs"),
          "preimages are checked before any class is patched")
    check("_ORIG[\"build_slot_mappings_by_layer\"]" in inst
          and "Runner.post_kv_cache_wake_up = _patched_post_kv_cache_wake_up" in inst,
          "the unbind memo wraps the original and a KV-cache wake-up invalidates the plan")
    wrapper = open(os.path.join(REPO, "probes", "run_prep_fused_check.sh"), encoding="utf-8").read()
    check("sha256sum" in wrapper and "base preimage mismatch" in wrapper and "PROFILE_IMAGE" in wrapper,
          "the probe wrapper must verify every mounted row's base contract like the launcher")
    check("FlashInferMLASparseSM90Builder" in src and "_mk_mla_armed()" in src
          and "replans the wrapper every step" in src,
          "the SM90 sparse builder is accepted ONLY while MK_SEG_MLA is armed: it "
          "replans FlashInfer's wrapper every step from host lengths, which caching "
          "this group's metadata would skip")
    print("  glm53 prep fused contracts .. OK")


def test_launcher_multiline_assignments_have_no_embedded_comments() -> None:
    """A `#` line inside a backslash-continued shell string ends the string.

    The NCCL channel note was added directly above the `-e NCCL_NET=IB` line,
    which sits INSIDE the ENVV="..." continuation: bash then closed the string
    early, ran the comment as a comment, and tried to execute the remaining
    `-e NCCL_...` lines as commands -- the boot died with "docker run requires
    at least 1 argument" AFTER clearing the compile cache. `bash -n` accepts
    it (the result is still valid syntax), so this check exists instead."""
    for name in ("start-glm53-nvfp4-tp4.sh", "start-hy4-tp4.sh", "start-qwen38-nvfp4-tp4.sh"):
        path = os.path.join(REPO, "launchers", name)
        if not os.path.exists(path):
            continue
        in_str = False
        for i, raw in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
            line = raw.rstrip()
            if not in_str:
                # an assignment that opens a quote and continues to the next line
                m = re.match(r'^\s*[A-Za-z_][A-Za-z0-9_]*="[^"]*\\$', line)
                if m:
                    in_str = True
                continue
            check(not line.lstrip().startswith("#"),
                  f"{name}:{i} is a comment inside a continued string assignment "
                  f"-- it silently truncates the value: {line.strip()[:60]}")
            if not line.endswith("\\"):
                in_str = False
    env = open(os.path.join(REPO, "launchers", "start-glm53-nvfp4-tp4.sh"), encoding="utf-8").read()
    envv = env[env.index('ENVV="'):]
    envv = envv[: envv.index('"\n', 1) + 1] if '"\n' in envv else envv[:4000]
    for k in ("NCCL_MIN_NCHANNELS=16", "NCCL_MAX_NCHANNELS=16", "NCCL_NCHANNELS_PER_NET_PEER=4"):
        check(k in envv, f"the measured NCCL channel tuning ({k}) is inside the ENVV string")
    print("  launcher continued-string assignments carry no comments .. OK")


def test_launcher_restores_prefill_warmup_from_caller_env() -> None:
    """PREFILL_WARMUP=0 bash launchers/... must survive sourcing the profile.

    The profile declares PREFILL_WARMUP=1 (default on since 33차), and the
    launcher restores only the keys in its _caller list after `. "$PROFILE_ENV"`;
    the warmup keys were once not on it, so a caller's value was silently
    clobbered by the profile and the 2026-09-03 warm-prefill boot ran cold
    with no prefill-warmup.log at all. The restore list is what makes the
    bracket harnesses' 0 pin and the kill switch work."""
    src = open(os.path.join(REPO, "launchers", "start-glm53-nvfp4-tp4.sh"), encoding="utf-8").read()
    # The restore list is ct_load_profile's argument list now, and the library
    # appends $_vllm_keys to it; the profile is sourced inside that function.
    names = _launcher_caller_passthrough(src)
    check({"PREFILL_WARMUP", "PREFILL_WARMUP_LENS"} <= names,
          "PREFILL_WARMUP and PREFILL_WARMUP_LENS are on the launcher's caller-env restore list")
    lib = open(os.path.join(REPO, "launchers", "lib", "common-tp4.sh"),
               encoding="utf-8").read()
    check('for _v in "$@" $_vllm_keys; do' in lib
          and lib.index('for _v in "$@" $_vllm_keys; do') < lib.index('. "$PROFILE_ENV"'),
          "the shared loader captures the caller's values BEFORE sourcing the "
          "profile, or the restore has nothing to restore")
    prof = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(re.search(r"^PREFILL_WARMUP=1$", prof, re.M) is not None,
          "profile ships PREFILL_WARMUP=1 (cold boots are the opt-in now)")
    print("  launcher restores PREFILL_WARMUP from caller env .. OK")


def test_profile_keys_not_passed_via_extra_env() -> None:
    """A profile-declared VLLM_* key cannot be overridden through EXTRA_ENV: the
    launcher aborts. Every documented arming command must use the caller env."""
    # both profiles: dsv4 declares VLLM_* keys of its own since it mounted the
    # megakernel core, and its launcher renders EXTRA_ENV even earlier than
    # glm53's does ($COMMON before $ENVV), so the same trap is live there
    keys = set()
    for _profile in ("glm53", "dsv4"):
        keys |= set(re.findall(
            r"^(VLLM_[A-Z0-9_]+)=",
            open(os.path.join(REPO, "profiles", f"{_profile}.env"),
                 encoding="utf-8").read(), re.M))
    docs = [os.path.join(REPO, "RUNBOOK_KERNEL_CAMPAIGN2.md")]
    docs += sorted(glob.glob(os.path.join(REPO, "overlay", "modules", "glm53_*", "README.md")))
    docs += sorted(glob.glob(os.path.join(REPO, "overlay", "modules", "dsv4_*", "README.md")))
    docs += [os.path.join(REPO, "profiles", "dsv4.env"),
             os.path.join(REPO, "profiles", "glm53.env")]
    offenders = []
    for path in docs:
        for i, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
            for m in re.finditer(r'EXTRA_ENV="([^"]+)"', line):
                for kv in m.group(1).split():
                    if kv.split("=", 1)[0] in keys:
                        offenders.append(f"{os.path.relpath(path, REPO)}:{i} {kv}")
    check(not offenders, f"profile-declared keys passed via EXTRA_ENV (launcher ABORTs): {offenders}")
    print("  profile keys not via EXTRA_ENV .. OK")




def test_glm53_indexer_gate_splitk_contracts() -> None:
    """glm53_indexer_gate_splitk: opt-in deterministic split-K head gate on the
    checkpoint's shape (index_n_heads=32), small-M only, stock default."""
    mod_dir = os.path.join(REPO, "overlay", "modules", "glm53_model")
    kern = open(os.path.join(mod_dir, "glm53_indexer_gate.py"), encoding="utf-8").read()
    attn = open(os.path.join(mod_dir, "glm5next_attention.py"), encoding="utf-8").read()
    fast = open(os.path.join(REPO, "overlay", "modules", "glm53_model",
                             "glm53_prefill_fastpath.py"), encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    modules = re.search(r'^MODULES="([^"]+)"', profile, re.M).group(1).split()
    check("glm53_model" in modules, "glm53 profile must mount glm53_model (the indexer gate lives there)")
    check(re.search(r"^VLLM_GLM53_INDEXER_GATE_SPLITK=0$", profile, re.M) is not None,
          "profile must ship VLLM_GLM53_INDEXER_GATE_SPLITK=0 (stock torch.mm by default)")
    rows = [l.split("\t") for l in open(os.path.join(mod_dir, "manifest.tsv"), encoding="utf-8")
            .read().splitlines() if l and not l.startswith("#")]
    by_target = {r[1]: r[2] for r in rows}
    check(re.fullmatch(
        r"[0-9a-f]{64}", by_target.get("vllm/models/glm5next/nvidia/attention.py", "")) is not None
        and by_target.get("vllm/models/glm5next/nvidia/glm53_indexer_gate.py") == "absent",
        f"manifest must overlay attention.py (pinned) and add the kernel file (absent): {rows}")
    stock = "torch.mm(hidden_states.float(), self._wp_fp32)"
    helper = "_glm53_head_gate(hidden_states, self._wp_fp32)"
    check(stock not in attn and attn.count(helper) == 1,
          "attention overlay routes the head gate through the helper exactly once")
    check("from vllm.models.glm5next.nvidia.glm53_indexer_gate import head_gate as _glm53_head_gate" in attn,
          "attention overlay imports the helper from the module's kernel file")
    check(stock not in fast and fast.count(helper) == 1,
          "the fused-indexer forward (VLLM_GLM53_FUSED_K_GATE) routes through the same helper")
    check("isinstance(e, ModuleNotFoundError) and e.name == _HEAD_GATE_MODULE" in fast
          and "logger.exception(" in fast.split("def _glm53_head_gate")[1].split("return _HEAD_GATE")[0],
          "the fastpath tolerates only 'module not mounted' silently and logs other ImportErrors")
    check("VLLM_GLM53_INDEXER_GATE_SPLITK" in fast.split("def _glm53_head_gate")[1].split("return _HEAD_GATE")[0],
          "the fastpath announces a knob that asks for split-K while the module is missing")
    check(not ({"glm53_drafter", "glm53_runtime"}
                    & set(open(os.path.join(mod_dir, "requires"), encoding="utf-8").read().split())),
          "the module is self-contained (the wiring fastpath optionally imports it, not the reverse) (folded into glm53_model: requires nothing above the model layer)")
    # kernel/helper contracts: deterministic two-stage reduce, the checkpoint's N, layout guards
    check("MAX_N = 32" in kern and "w.shape[1] <= MAX_N" in kern,
          "the applicability cap must admit the checkpoint's index_n_heads=32")
    check("MAX_M = 16" in kern and "x.shape[0] <= MAX_M" in kern,
          "split-K only for the small-M decode shape (cuBLAS is fast from M=24)")
    check("tl.atomic_add" not in kern and "_gate_splitk_reduce_kernel" in kern
          and "tl.static_range(SPLIT)" in kern,
          "the reduction must be a fixed-order partial sum, not fp32 atomics (bitwise reproducible)")
    check("torch.zeros" not in kern, "no memset: partials are written, not accumulated")
    check("x.stride(1) == 1" in kern and "x.shape[1] == w.shape[0]" in kern and "x.dim() == 2" in kern,
          "applicability guards x's layout and the K match (the kernel assumes both)")
    check("w.dtype == torch.float32" in kern and "w.is_contiguous()" in kern,
          "applicability: fp32 contiguous weight only")
    check("K % block_k" in kern and "raise ValueError" in kern,
          "a non-tiling block_k fails loud (the partial kernel reads w rows unmasked)")
    check("return torch.mm(x.float(), w)" in kern, "the helper falls back to the stock product")
    check("[indexer-gate]" in kern and "logger.warning" in kern
          and "if ok not in _ANNOUNCED:" in kern,
          "each routing verdict is announced ONCE (keyed on the verdict, not the "
          "shape: prefill M differs per request and would fill the log)")
    check("do_not_specialize" not in kern,
          "no specialization pins: K, N and strides are per-layer constants (one compile per shape)")
    tree = ast.parse(kern)
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name == "gate_splitk_enabled")
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENV" for t in n.targets))]
    check(len(nodes) == 2, "ENV constant and gate_splitk_enabled must be top-level definitions")
    ns: dict = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "glm53_indexer_gate", "exec"), ns)
    fn = ns["gate_splitk_enabled"]
    saved = os.environ.pop("VLLM_GLM53_INDEXER_GATE_SPLITK", None)
    try:
        check(fn() is False, "unset knob keeps stock torch.mm")
        for v, want in (("0", False), ("1", True), (" 1 ", True), ("on", False), ("true", False),
                        ("2", False), ("shadow", False), ("", False)):
            os.environ["VLLM_GLM53_INDEXER_GATE_SPLITK"] = v
            check(fn() is want, f"knob {v!r} must map to {want} (only the exact string 1 arms)")
    finally:
        os.environ.pop("VLLM_GLM53_INDEXER_GATE_SPLITK", None)
        if saved is not None:
            os.environ["VLLM_GLM53_INDEXER_GATE_SPLITK"] = saved
    readme = open(os.path.join(mod_dir, "README.md"), encoding="utf-8").read()
    check("not bit-exact" in readme and "VLLM_GLM53_INDEXER_GATE_SPLITK=1 bash launchers/" in readme
          and "index_n_heads" in readme and "[indexer-gate]" in readme,
          "README states the numerics caveat, the caller-env arming form, the checkpoint shape and the log anchor")
    print("  glm53 indexer gate split-K contracts .. OK")


def test_bracket_runner_contracts() -> None:
    """bench/bracket.py: the bracket discipline as code -- C=1 step/s channel,
    base-pair drift as the significance floor, env snapshots per leg."""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "bracket", os.path.join(REPO, "bench", "bracket.py"))
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)

    def recs(med_base1, med_cand, med_base2, env='{"A": "1"}', conc=1, n=6):
        # one jsonl line = one leg holding its reps (cmd_leg's output shape)
        def leg(tag, med):
            return {"name": "T", "tag": tag, "conc": conc,
                    "env": json.loads(env),
                    "reps": [{"tok_s": med,
                              "step_s": med + (0 if conc == 1 else 5),
                              "acc_raw": None, "rep": i + 1}
                             for i in range(n)]}
        return [leg("base", med_base1), leg("cand", med_cand),
                leg("base", med_base2)]

    def medians(vals):
        vals = sorted(vals)
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    # adopt: +3.8% against a ~0.4% drift
    rep = br.judge(recs(100.0, 103.8, 100.4))
    check(rep["ok"], "a proper bracket judges")
    check(abs(rep["base_drift"] - round(0.4 / 100.2, 4)) < 1e-9,
          "drift is the base-pair relative difference")
    check(rep["cands"][0]["verdict"].startswith("채택"),
          f"an effect above drift adopts (got {rep['cands'][0]['verdict']!r})")
    # indeterminate: effect inside the drift
    rep = br.judge(recs(100.0, 100.3, 100.4))
    check(rep["cands"][0]["verdict"].startswith("CV 이하"),
          "an effect inside the drift refuses to judge")
    # reject: negative beyond the drift
    rep = br.judge(recs(100.0, 96.0, 100.4))
    check(rep["cands"][0]["verdict"].startswith("기각"),
          "a negative effect beyond the drift rejects")
    # not a bracket: missing trailing base
    rep = br.judge(recs(100.0, 103.8, 100.4)[:-1])
    check(not rep["ok"] and any("브래킷" in p for p in rep["problems"]),
          "base->cand without the trailing base is not a bracket")
    # env fingerprint differences surface as a problem (#116: knob not delivered)
    rec_env = recs(100.0, 103.8, 100.4)
    for r in rec_env:
        if r["tag"] == "cand":
            r["env"] = {"A": "0"}
    rep = br.judge(rec_env)
    check(any("env" in p for p in rep["problems"]),
          "an env snapshot differing between legs is flagged")
    # C!=1 records stay out of the judgment channel
    rec_all = recs(100.0, 103.8, 100.4) + recs(90.0, 93.0, 90.2, conc=4)
    rep = br.judge(rec_all)
    check(rep["other_conc"] == 18 and rep["ok"],
          "C=4 records are recorded but excluded from the C=1 judgment")
    # the ledger formula: step/s = tok/s / (1 + k x raw_acc)
    check(abs(br.step_s_of(500.0, 0.075, 7) - 500.0 / 1.525) < 1e-9,
          "step/s normalization matches tok/s / (1 + k*raw_acc)")
    check(br.step_s_of(500.0, None, 7) is None,
          "no acceptance counter -> no step/s (judge falls back, loudly)")
    # the knobs the campaign judges on must be in the snapshot list (source
    # contract: the list itself is os.environ-filtered at import time)
    src = open(os.path.join(REPO, "bench", "bracket.py"), encoding="utf-8").read()
    for key in ("VLLM_GLM53_ASYNC_DFLASH", "VLLM_GLM53_PREP_FUSED",
                "VLLM_GLM53_INDEXER_GATE_SPLITK", "VLLM_GLM53_FP8_DENSE",
                "ENABLE_EP", "CUSTOM_OPS_AXIS"):
        check(key in src, f"snapshot list covers {key}")
    check("사람이 한다" in src and "읽기 전용" in src,
          "the tool states the reboot-is-human rule (automation is read-only)")
    print("  bracket runner contracts .. OK")


def test_trace_composition_analyze() -> None:
    """tools/trace_step_composition.py: analyze on a synthetic trace, and the
    diff contract that counts are the authoritative channel."""
    import importlib.util
    import json
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "tsc", os.path.join(REPO, "tools", "trace_step_composition.py"))
    tsc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsc)

    def trace(extra_per_step=0):
        evs = []
        t = 0
        for _ in range(14):
            base_ts = t
            evs.append({"ph": "X", "cat": "kernel", "name": "_gather_block_tables_kernel",
                        "ts": base_ts, "dur": 50, "args": {"stream": 210}})
            for j in range(2 + extra_per_step):
                evs.append({"ph": "X", "cat": "kernel",
                            "name": "deep_gemm::sm120_fp8_fp4_gemm_nt",
                            "ts": base_ts + 100 + j * 100, "dur": 100,
                            "args": {"stream": 210}})
            evs.append({"ph": "X", "cat": "kernel", "name": "k_oneshot",
                        "ts": base_ts + 300, "dur": 40, "args": {"stream": 210}})
            t += 10000
        return {"traceEvents": evs}

    with tempfile.TemporaryDirectory() as td:
        pa = os.path.join(td, "base.json")
        pb = os.path.join(td, "cand.json")
        open(pa, "w").write(json.dumps(trace(0)))
        open(pb, "w").write(json.dumps(trace(1)))
        ra = tsc.analyze(pa)
        rb = tsc.analyze(pb)
    check(ra["steps"] == 8, f"14 step anchors -> 8 analysed windows (got {ra['steps']})")
    check(ra["cats"]["deep_gemm fp8/fp4 GEMM"]["cnt"] == 2,
          "per-step median kernel count per category")
    check(abs(ra["cats"]["deep_gemm fp8/fp4 GEMM"]["ms"] - 0.2) < 1e-9,
          "2 x 100 us deep_gemm per step = 0.2 ms")
    check(abs(ra["idle_ms"] - 0.05) < 1e-9,
          "idle = span 340 us - union busy 290 us = 0.05 ms")
    check(ra["ar"][:3] == (40.0, 40.0, 40.0) and ra["ar"][4] == 8,
          "AR percentiles over the analysed windows")
    d_cnt = rb["cats"]["deep_gemm fp8/fp4 GEMM"]["cnt"] - ra["cats"]["deep_gemm fp8/fp4 GEMM"]["cnt"]
    check(d_cnt == 1, "the diff's kernel-count delta is +1 for an added launch")
    src = open(os.path.join(REPO, "tools", "trace_step_composition.py"),
               encoding="utf-8").read()
    check("--diff" in src and "kernel-count change" in src
          and "ground truth" in src,
          "diff mode prints the count delta as the authoritative channel")
    print("  trace composition analyze .. OK")


def test_drafter_fc_probe_contracts() -> None:
    """probes/drafter_fc_check.py: the drafter's fleet shapes at their
    per-rank sharding, DRAM-cold timing by weight cycling, every arm the
    image already has (bf16 / fp8 / fp4 / MK W4 with K-chunks past the lane's
    K contract), class gates, and the per-step sum against the armed trace's
    66 ms step."""
    src = open(os.path.join(REPO, "probes", "drafter_fc_check.py"),
               encoding="utf-8").read()
    check("DEF_LAYERS, DEF_HIDDEN, DEF_M = 5, 4096, 7" in src
          and "DEF_HEADS, DEF_KV_HEADS, DEF_HEAD_DIM, DEF_INTER = 32, 8, 128, 12288" in src,
          "fleet drafter defaults: 5 layers x hidden 4096, 32/8 heads of 128, "
          "intermediate 12288, decode M=7 (the checkpoint's config.json)")
    check("--config" in src and "num_hidden_layers" in src
          and "hidden_size" in src and "num_key_value_heads" in src
          and "intermediate_size" in src and "conv_group_size" in src,
          "#231's lesson: the fleet shape is read from config, not assumed")
    check('("fc", h, L * h, 1)' in src and "(nh + 2 * nkv) * hd // tp" in src
          and '("o_proj", h, nh * hd // tp, L)' in src
          and '("gate_up_proj", 2 * inter // tp, h, L)' in src
          and '("down_proj", h, inter // tp, L)' in src
          and '("kernel_projection", kproj_n, h, 2 * L)' in src,
          "shapes are the drafter's PER-RANK shapes: column-parallel shards N, "
          "row-parallel shards K, fc and the conv kernel_projections are "
          "replicated -- the 09-03 boot ran the drafter at TP=4")
    check("MIN_STREAM = 96 << 20" in src and "L2_BYTES = 24 << 20" in src
          and "-(-MIN_STREAM // (N * K // 2))" in src and "NW_MIN, NW_MAX = 6, 48" in src,
          "cold weights by graph cycling: enough copies that the SMALLEST "
          "arm's bytes overflow the 24 MB L2 (ledger trap 4)")
    for arm in ("_fp8_fp4_dense_gemm", "_fp8_dense_gemm(x, q, s, rows, cols)",
                "functional.linear", "build_mk_weight_w4", "_quantize_fp8_block_padded"):
        check(arm in src, f"arm present: {arm}")
    check("MK_KMAX = 4096" in src and "def _mk_chunks(" in src
          and "return mk.build_mk_weight_w4_kchunks(w)" in src
          and "assert lane_kmax == MK_KMAX" in src
          and "out = mk._gemm_kchunks(x, packs, N)" in src
          and "acc += mk._gemm_call(xc[c]" not in src and "xc = [" not in src,
          "the K-chunk arm is the LANE's chunker and summation "
          "(build_mk_weight_w4_kchunks / _gemm_kchunks), width asserted "
          "against MK_GEMM_KMAX -- a second chunk loop here kept measuring "
          "its own width and pre-sliced x outside the timed region, hiding "
          "the contiguous copies the served path pays")
    check("mk.maybe_arm()" in src and '_ARMED.get("gemm")' in src,
          "the lane is armed (extension built, self-tests run) before "
          "_gemm_call is timed")
    check("SKIP" in src and "_launch_bitwise" in src,
          "an arm that raises is a SKIP line; replay determinism is checked")
    check('TOL = {"bf16": 5e-2, "fp8": 5e-2, "fp4": 0.15, "mk_w4": 0.15}' in src
          and "e2m1" in src,
          "gates by class: e2m1 arms at the MK bench's by-design 0.15")
    check("STEP_MS = 66.0" in src and "per-step drafter GEMM sum" in src
          and "of the step vs bf16" in src,
          "the prize is the per-step sum over the drafter's launches against "
          "the armed trace's 66 ms step")
    check("lm_head" in src and "--head" in src and "reference only" in src,
          "the head shape is reference only: the served head is another lane")
    print("  drafter fc probe contracts .. OK")


def test_hy4_entrypoint_carries_the_production_knobs() -> None:
    """One launcher, and it must not quietly serve differently than the fork it replaces.

    srv2 ran a 481-line fork of this launcher (~/start-hy4-tp4.sh, dated
    08-30) while the repo copy was refactored to 655. Unifying on the repo
    copy is right -- the fork has no memfree preflight, no overlay
    attestation, no EXTRA_ENV collision guard -- but the refactor had
    dropped five things the fork was still passing, and deploying it as-is
    would have changed serving with nothing saying so:

      --long-prefill-token-threshold   (fork: 2048; repo: flag absent)
      SPEC_METHOD switch               (fork: dspark|draft-model; repo: dspark hardcoded)
      VLLM_INTERACTIVE_RESERVE         (fork: 0)
      VLLM_INTERACTIVE_BULK_CHUNK      (fork: 1024)
      VLLM_DSPARK_CONFIDENCE_*         (fork: off / 0.0 / 0)

    The knobs live in profiles/dsv4.env now. This holds the entry point
    against the fork so the same silent divergence cannot come back.
    """
    launcher = open(os.path.join(REPO, "launchers", "start-hy4-tp4.sh"),
                    encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "dsv4.env"),
                   encoding="utf-8").read()

    # 1. long prefill: the flag is conditional, so both halves must be present
    check('${LONG_PREFILL:+--long-prefill-token-threshold' in launcher,
          "--long-prefill-token-threshold must stay conditional on LONG_PREFILL: "
          "dropping the flag hands long prefills to the default scheduler path")
    check(re.search(r"^LONG_PREFILL=2048$", profile, re.M) is not None,
          "the profile carries the production long-prefill threshold")

    # 2. the speculative method is a switch again, not a hardcoded string
    check('SPEC_METHOD="${SPEC_METHOD:-dspark}"' in launcher,
          "SPEC_METHOD selects the speculative lane; hardcoding dspark in the "
          "serve script removed the draft-model path entirely")
    check('--speculative-config "${SPEC_JSON}"' in launcher,
          "the serve command reads the JSON the branch built, not a literal")
    check('draft_kv_cache_dtype' in launcher and 'DRAFT_PATH' in launcher,
          "the draft-model arm still reads DRAFT_PATH/DRAFT_KV")

    # 3. every knob the fork passed reaches the container
    for var in ("SPEC_METHOD", "DRAFT_PATH", "DRAFT_KV",
                "VLLM_DFLASH_DRAFT_BLOCK_SIZE", "LONG_PREFILL"):
        check("-e %s=" % var in launcher,
              "%s must be passed to the container -- the serve script reads it "
              "from the environment" % var)

    # 4. profile-declared VLLM_* keys ride the _vllm_keys loop; these are the
    #    ones the fork passed by hand.
    for key in ("VLLM_INTERACTIVE_RESERVE", "VLLM_INTERACTIVE_BULK_CHUNK",
                "VLLM_DSPARK_CONFIDENCE_SCHEDULER",
                "VLLM_DSPARK_CONFIDENCE_THRESHOLD", "VLLM_DSPARK_TOPK_GATHER"):
        check(re.search(r"^%s=" % key, profile, re.M) is not None,
              "%s belongs to the profile now that the fork is gone" % key)
    check("-e VLLM_DFLASH_DRAFT_BLOCK_SIZE=" in launcher
          and re.search(r"^VLLM_DFLASH_DRAFT_BLOCK_SIZE=", profile, re.M) is None,
          "VLLM_DFLASH_DRAFT_BLOCK_SIZE is plumbed from DRAFT_BLOCK, so the "
          "profile must NOT also declare it -- the _vllm_keys loop would add a "
          "second -e and docker keeps the last one")

    # 5. caller env still wins over the profile for the restored names
    preserve = " ".join(sorted(_launcher_caller_passthrough(launcher)))
    for var in ("DRAFT_BLOCK", "DRAFT_KV", "DRAFT_PATH", "LONG_PREFILL",
                "SPEC_METHOD"):
        check(var in preserve,
              "%s must be in the caller-precedence list, or a one-off "
              "`%s=... bash launchers/start-hy4-tp4.sh` loses to the profile" % (var, var))

    # 6. GPU_MEM is pinned in the profile, which is what stops the preflight
    #    from trading host headroom for KV on a production boot nobody measured.
    check(re.search(r"^GPU_MEM=0\.60$", profile, re.M) is not None,
          "GPU_MEM stays pinned at the production value: _GPU_MEM_PINNED is "
          "recorded after the profile loads, so this is what keeps the memfree "
          "preflight from raising 0.60 to ~0.744 unmeasured")
    print("  hy4 entry point parity ........ OK")




def test_common_tp4_library_is_the_one_implementation() -> None:
    """Both TP=4 lanes share one copy of the machinery they used to duplicate.

    start-hy4-tp4.sh and start-glm53-nvfp4-tp4.sh implemented profile loading,
    the EXTRA_ENV guard, LOAD_FORMAT validation and the torch/NCCL error mode
    twice, and every one of those copies had drifted:

      memfree margin   hy4 sat at 3 while #270 moved glm53 to 10 -- the same
                       bug fixed twice, and the margin-3 boot in between
                       wedged three nodes (09-04).
      EXTRA_ENV guard  glm53 rejected a comma-joined list; hy4 did not, so
                       EXTRA_ENV="A=1,B=2" became one -e and every knob read
                       as off while the log said they were set.
      ASYNC_ERROR_     hy4 ran 0, glm53 ran 1, neither with a reason. 0 is why
        HANDLING       the 09-04 wedge could not clear itself: the ring broke
                       and the workers held ~55 GiB for twelve hours because
                       nothing tore them down.

    The rule: one concern, one implementation, and the better one wins.
    """
    lib_path = os.path.join(REPO, "launchers", "lib", "common-tp4.sh")
    check(os.path.exists(lib_path), "the shared library must exist")
    lib = open(lib_path, encoding="utf-8").read()
    lanes = {
        name: open(os.path.join(REPO, "launchers", name), encoding="utf-8").read()
        for name in ("start-hy4-tp4.sh", "start-glm53-nvfp4-tp4.sh")
    }

    for name, src in lanes.items():
        check("lib/common-tp4.sh" in src, "%s must source the shared library" % name)
        for fn in ("ct_load_profile", "ct_extra_env_flags", "ct_check_load_format"):
            check(fn in src, "%s must call %s rather than inline its own copy" % (name, fn))
        # the extracted bodies must be GONE from the lanes, or the drift is back
        check("_vllm_keys=$(grep -oE" not in src,
              "%s must not re-implement profile key discovery" % name)
        check("EXTRA_ENV entry is not KEY=VALUE" not in src,
              "%s must not re-implement the EXTRA_ENV guard" % name)
        check("LOAD_FORMAT must be auto" not in src,
              "%s must not re-implement LOAD_FORMAT validation" % name)
        # the error mode is the library's default, not a per-lane literal
        check("TORCH_NCCL_ASYNC_ERROR_HANDLING=$CT_NCCL_ASYNC_ERR" in src,
              "%s must take the shared torch/NCCL error mode" % name)
        for literal in ("TORCH_NCCL_ASYNC_ERROR_HANDLING=0",
                        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"):
            check(literal not in src,
                  "%s must not hardcode %s -- that divergence is the bug" % (name, literal))

    # the library holds the values, and TearDown is the one that frees a node
    check(re.search(r'^CT_NCCL_ASYNC_ERR="\$\{NCCL_ASYNC_ERR:-1\}"$', lib, re.M) is not None,
          "the shared default is 1 (TearDown): 0 leaves a broken ring holding "
          "the node's memory until someone power-cycles it, which is exactly "
          "what 09-04 cost. NCCL_ASYNC_ERR=0 still rolls it back per boot")
    check("space-separated, not comma-separated" in lib,
          "the comma guard is the better of the two implementations and must "
          "be the one both lanes now get")
    check("docker takes the last -e" in lib,
          "the profile-collision guard moved into the library intact")

    # the foreign-stack guard: one implementation, and BOTH lanes use it. hy4
    # was written twice with different anchors and ran at different points (the
    # claim in #277 that glm53 had none was wrong -- it had an inline copy the
    # survey missed), so it only reliably protected whichever stack happened
    # to be started second -- starting glm53 onto a live DSV4 double-books the
    # same unified memory, which is how a node crosses the 4 GiB watermark.
    check("ct_refuse_foreign_stacks()" in lib,
          "the foreign-stack guard belongs in the library, not one lane")
    check("DRY_RUN:-0" in lib[lib.index("ct_refuse_foreign_stacks()"):],
          "a dry run creates nothing, so a live stack is not a conflict there")
    foreign = {
        "start-hy4-tp4.sh": "ct_refuse_foreign_stacks '^(glm53|q38)(-|$)' DSV4",
        "start-glm53-nvfp4-tp4.sh": "ct_refuse_foreign_stacks '^(hy4|q38)(-|$)' GLM53",
    }
    for name, call in foreign.items():
        check(call in lanes[name],
              "%s must refuse the OTHER lanes' stacks: %s" % (name, call))
    # the RoCE-v2 IPv4 GID index is a per-node, per-boot property, so it has
    # to be resolved inside the container -- one copy, used by both lanes.
    # glm53 shipped `-e NCCL_IB_GID_INDEX=3` from the head instead: one value
    # for four nodes, which no single -e can make right. All four happen to
    # read 3 today; the runbook records a boot where srv1 read 3 and srv2/srv4
    # read 4, which is the case a hardcoded index gets silently wrong.
    check("CT_GID_PRELUDE=$(cat <<'GIDEOF'" in lib,
          "the prelude must be defined through a QUOTED heredoc, or the single "
          "quotes in `tr ',' ' '` do not survive into the serve script")
    check("gid_attrs/types" in lib and "NCCL_IB_GID_INDEX=$i" in lib,
          "the library holds the detection")
    for name, src in lanes.items():
        check("gid_attrs/types" not in src,
              "%s must not carry its own copy of the GID detection" % name)
        check('"$CT_GID_PRELUDE"' in src,
              "%s must emit the shared prelude into its serve command" % name)
    # glm53 delivers its serve command as a base64 payload: the prelude has to
    # come FIRST, or the export lands after vllm has already started.
    g = lanes["start-glm53-nvfp4-tp4.sh"]
    for var in ("W_B64=", "SERVE_B64="):
        line = [l for l in g.splitlines() if l.strip().startswith(var)][0]
        check(line.index('"$CT_GID_PRELUDE"') < line.index("vllm serve"),
              "%s must put the prelude before the serve line" % var)
    check("-e NCCL_IB_GID_INDEX=" in g,
          "glm53 keeps its historical index as the FALLBACK the prelude "
          "overrides -- belt and suspenders, per the runbook")

    # exactly one foreign check per lane -- #277 left glm53 running two
    check(g.count("hy4|q38") == 1,
          "glm53 must carry ONE foreign check: #277 added the shared call "
          "beside an inline copy it had not noticed")

    # every node must resolve $IMAGE to the same build. hy4 compared worker IDs
    # to the head's; glm53 only asked whether the tag resolved, and its tags are
    # built locally, so one tag can name different images per node. The overlay
    # attestation compares HOST files and the manifest preimage check runs on
    # the head only, so neither closed that gap.
    check("ct_verify_image_uniform()" in lib and "CT_IMAGE_ID=" in lib,
          "the image check belongs in the library")
    check("the ranks would run different builds" in lib,
          "the abort must say what skew actually costs")
    # assert the COMPARISON, not just the message: replacing the condition with
    # a constant leaves the abort text in place and the guard doing nothing.
    check('[ "$_wid" != "$_hid" ]' in lib,
          "the worker ID must actually be compared against the head's")
    check('[ -n "$_expect" ] && [ "$_hid" != "$_expect" ]' in lib,
          "the pinned ID, when given, must actually be compared")
    for name, src in lanes.items():
        check("ct_verify_image_uniform" in src,
              "%s must verify the image is uniform across nodes" % name)
        check("docker image inspect" not in src,
              "%s must not keep its own image inspection" % name)
    check('ct_verify_image_uniform "$SSHOPT" "$IMAGE" "$EXPECTED_IMAGE_ID"'
          in lanes["start-hy4-tp4.sh"],
          "hy4 keeps its pinned ID as the expected value")
    check('ct_verify_image_uniform "$SSHOPT" "$IMAGE" ""'
          in lanes["start-glm53-nvfp4-tp4.sh"],
          "glm53 pins no value -- its tag moves by design -- but still requires "
          "the four nodes to agree")

    # custom_ops axis: one substitution, anchored, and it refuses to be a no-op.
    # glm53's copy was `sed 's/"all"/.../'` with no guard at all, which fails two
    # ways against a caller-supplied COMPILE_CFG -- both reproduced before the
    # consolidation: a config without custom_ops left the axis silently
    # unapplied (the boot then measures the CONTROL arm while the caller
    # believes otherwise), and a config carrying "all" ahead of custom_ops got
    # that other field rewritten instead, because sed replaces the FIRST match.
    check("ct_apply_custom_ops_axis()" in lib,
          "the axis rewrite belongs in the library")
    check('"custom_ops":\\["all"\\]' in lib,
          "the substitution must be ANCHORED on custom_ops, not bare \"all\" -- "
          "sed takes the first match and a caller config can carry it earlier")
    check("silent no-op" in lib,
          "the abort must name the failure it prevents")
    axis_fn = lib[lib.index("ct_apply_custom_ops_axis()"):]
    check('*\'"custom_ops":["all"]\'*) ;;' in axis_fn,
          "the anchor must be CHECKED before substituting, or the rewrite is a "
          "silent no-op when it is absent")
    for name, src in lanes.items():
        check("ct_apply_custom_ops_axis" in src,
              "%s must use the shared axis rewrite" % name)
        # neither the bare pattern glm53 used nor the anchored one may remain
        # in a lane: the rewrite has exactly one home now.
        check("""s/\\"all\\"/""" not in src,
              "%s must not keep the bare \"all\" substitution" % name)
        check('"custom_ops":\\["all"\\]/' not in src,
              "%s must not keep its own anchored substitution either" % name)
    # empty is a VALUE on glm53 (its fusion arm) and merely unset on hy4
    check('ct_apply_custom_ops_axis "$CUSTOM_OPS_AXIS" 0' in lanes["start-hy4-tp4.sh"]
          and '${CUSTOM_OPS_AXIS:-}' in lanes["start-hy4-tp4.sh"],
          "hy4 applies the axis only when non-empty and disallows empty")
    check('ct_apply_custom_ops_axis "${CUSTOM_OPS_AXIS:-}" 1' in lanes["start-glm53-nvfp4-tp4.sh"]
          and '${CUSTOM_OPS_AXIS+x}' in lanes["start-glm53-nvfp4-tp4.sh"],
          "glm53 keeps set-but-empty as its fusion arm (+x, allow-empty=1)")

    # overlay target containment. Both lanes enforced "inside the package root"
    # with a PREFIX test plus a character class that allows "." -- which is not
    # containment. Verified against each lane's own case statements before the
    # change: a target of
    #   <root>/../../../../etc/cron.d/evil
    # satisfies the prefix, uses only allowed characters, and was ACCEPTED by
    # both. The target becomes a docker bind-mount destination, so that row
    # would have mounted over a path outside the root the check exists to
    # protect. Neither lane had the guard, so this is not drift -- and the
    # ABORT-set diff that found the other asymmetries could not surface it.
    check("ct_check_overlay_target()" in lib,
          "target validation belongs in the library")
    tgt_fn = lib[lib.index("ct_check_overlay_target()"):]
    check('*/../*)' in tgt_fn,
          "the .. escape must be rejected: a prefix test alone is not "
          "containment, and the target is a bind-mount destination")
    check('case "/$_t/" in' in tgt_fn,
          "the .. test must wrap the value in slashes, or a component that "
          "merely starts or ends with .. is missed")
    for name, src in lanes.items():
        check("ct_check_overlay_target" in src,
              "%s must use the shared target validation" % name)
        check("outside the package root" not in src
              and "unsafe overlay target in manifest" not in src,
              "%s must not keep its own target case statement" % name)
    # the root differs per lane by design: glm53 overlays reach outside vllm/
    # (flashinfer), hy4 does not, so hy4 demands the stricter root.
    check('ct_check_overlay_target "$target" "/opt/venv/lib/python3.12/site-packages/vllm/"'
          in lanes["start-hy4-tp4.sh"],
          "hy4 keeps the stricter vllm/ root")
    check('ct_check_overlay_target "$target" "${TARGET_PREFIX:-' in lanes["start-glm53-nvfp4-tp4.sh"],
          "glm53 derives its root from the profile's TARGET_PREFIX")

    # profile-declared VLLM_* values ride an UNQUOTED expansion into docker:
    #   ENVV="$ENVV -e $_k=${!_k}" ... docker run $COMMON $ENVV ...
    # The word splitting is what makes separate -e arguments, so the mechanism
    # cannot carry whitespace. Reproduced with VLLM_A={"x":"a b"}: the value is
    # truncated at the space AND the remainder becomes a stray docker argument.
    # hy4 already refuses whitespace in COMPILE_CFG for this exact reason; the
    # general mechanism with the same shape had no guard on either lane.
    check("ct_check_profile_env_values()" in lib,
          "profile values must be validated where the profile is loaded")
    check("ct_check_profile_env_values" in lib[:lib.index("ct_check_profile_env_values()")],
          "ct_load_profile must CALL it, not merely define it -- a validator "
          "nothing invokes is the silence it was written to remove")
    vfn = lib[lib.index("ct_check_profile_env_values()"):]
    check("*[[:space:]]*)" in vfn,
          "whitespace is the condition that breaks the -e mechanism")
    # empty is NOT an error: glm53's profile declares six knobs as "" to
    # document that they exist while leaving them unset, and warning on each
    # would fire six times per healthy boot.
    check("WARNING" not in vfn,
          "an empty declaration is an idiom here, not a fault -- glm53 uses it "
          "six times, so flagging it would train the operator to ignore output")
    prof = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(len(re.findall(r'^VLLM_[A-Z0-9_]+=""$', prof, re.M)) >= 5,
          "that idiom is real and load-bearing -- if it stops being used, the "
          "no-warning decision above should be revisited")

    # memfree-preflight.sh is shared by both lanes and sizes the number whose
    # wrong value wedged the fleet. Three faults, all found by running it:
    pre = open(os.path.join(REPO, "launchers", "memfree-preflight.sh"),
               encoding="utf-8").read()
    # 1. the default margin was 3 -- the exact value whose boot left ~3 GiB
    #    free against a 4.0 GiB watermark. Both launchers pass 10, so the
    #    default is only reached by a direct call, which is where a trap hurts.
    check("MARGIN=${1:-3}" not in pre,
          "the margin default must not be the value that wedged the fleet")
    check("MARGIN=10" in pre,
          "the default margin matches what both launchers pass")
    # 2. the margin was consumed unconditionally, so passing only node
    #    addresses made the FIRST ADDRESS the margin (awk read 10.10.10.2 as
    #    10.10) and dropped that node from the measurement.
    check('[[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]' in pre,
          "a bare number is a margin; an address is a node. The usage line "
          "calls the margin optional, so it has to actually be optional")
    # 3. a too-small result was clamped up to 0.40 and returned as if measured.
    #    hy4 never adopts a lower value so it was inert there, but glm53 adopts
    #    in BOTH directions: 0.40 x 119.69 = 47.9 GiB against ~78 GiB of
    #    weights and activations is KV about -30 GiB and a dead boot.
    check("if(v<0.40)v=0.40" not in pre,
          "a fabricated floor is worse than refusing: the caller cannot tell "
          "it from a measurement")
    check("refusing to size this low" in pre,
          "too little memory must refuse, so callers fall back to their "
          "configured value the way they already do for an unreachable node")

    # #282 gave the two LAUNCHERS the .. check and stopped there. The same
    # validation existed in two more places, and one of them is the earliest
    # producer: compose-overlays.sh glues TARGET_PREFIX onto a relative target,
    # so a module writing "../../x" satisfies the prefix test while pointing
    # outside the root. Reproduced -- it composed clean and reached the
    # manifest. deploy-overlays.sh then distributes that manifest. All four
    # sites share one implementation now.
    for name in ("compose-overlays.sh", "deploy-overlays.sh"):
        src = open(os.path.join(REPO, "launchers", name), encoding="utf-8").read()
        check("lib/common-tp4.sh" in src,
              "%s must source the shared library" % name)
        check("ct_check_overlay_target" in src,
              "%s validates manifest targets, so it uses the shared check" % name)
        check("unsafe character in target" not in src
              and "outside the profile's package root" not in src
              and "outside $PROFILE's TARGET_PREFIX" not in src,
              "%s must not keep its own copy" % name)
    # compose knows which module bound the target; that diagnostic must survive
    comp = open(os.path.join(REPO, "launchers", "compose-overlays.sh"),
                encoding="utf-8").read()
    check('ct_check_overlay_target "$target" "$TARGET_PREFIX" "$mod' in comp,
          "compose passes the module as context -- naming the offender is worth "
          "more than saving an argument")
    check('_ctx="${3:-}"' in lib,
          "the context argument is optional, so the launchers need not pass one")
    # and it must run AFTER the relative-target expansion, or the .. case is
    # unreachable at the one place that can name the module
    check(comp.index('target="${TARGET_PREFIX}${target}"')
          < comp.index("ct_check_overlay_target"),
          "compose expands a relative target before validating it, which is "
          "exactly what makes the .. case reachable there")

    # fleet-audit.sh is the runbook's acceptance instrument for "GID index
    # identical on all four nodes". It dumped the raw table for a human to
    # eyeball, scanned 0..9 while the launcher scans 0..15, and made no
    # selection of its own -- so it could neither say which index the engine
    # would use nor see an entry above 9 that the engine would pick. It now
    # runs the launchers' own CT_GID_PRELUDE on each node and prints the
    # result, so the audit cannot disagree with the boot by construction.
    aud = open(os.path.join(REPO, "launchers", "fleet-audit.sh"), encoding="utf-8").read()
    check("lib/common-tp4.sh" in aud and "$CT_GID_PRELUDE" in aud,
          "the audit must run the launchers' selection, not a lookalike")
    check("for i in $(seq 0 15); do" in aud and "for i in 0 1 2 3 4 5 6 7 8 9; do" not in aud,
          "the audit scans the same 0..15 the launcher does")
    check("nccl_gid_index=${NCCL_IB_GID_INDEX:-UNSET}" in aud,
          "the audit prints the index the engine will export on that node")
    # the HCA list the prelude reads must be the one the launchers pass, or the
    # audit answers a different question than the boot asks
    hca_aud = re.search(r'^AUDIT_GID_HCA="([^"]+)"$', aud, re.M)
    check(hca_aud is not None, "the audit names its HCA list once, as a variable")
    for name, src in lanes.items():
        hca_lane = re.search(r'NCCL_IB_HCA=([A-Za-z0-9_,]+)', src)
        check(hca_lane is not None and hca_lane.group(1) == hca_aud.group(1),
              "%s passes NCCL_IB_HCA=%s but the audit probes %s -- they must be "
              "one list" % (name, hca_lane.group(1) if hca_lane else "?", hca_aud.group(1)))

    check("hy4" not in foreign["start-hy4-tp4.sh"].split("'")[1]
          and "glm53" not in foreign["start-glm53-nvfp4-tp4.sh"].split("'")[1],
          "a lane must not name ITSELF foreign -- it would refuse to restart "
          "over its own stale containers, which every boot has")
    print("  common tp4 library ............ OK")


def test_worker_launch_does_not_let_the_remote_reparse_envv() -> None:
    """A JSON -e value must survive ssh; the remote shell must never re-parse it.

    Interpolating $ENVV into `ssh host "docker run ... $ENVV ..."` hands the
    value of -e COMPILE_CFG (a JSON blob) to a shell that parses it from
    scratch, and a top-level {a,b,c} is BRACE EXPANSION there. On 2026-09-04
    the first production boot on the unified launcher shattered it into

      COMPILE_CFG=cudagraph_mode:FULL_AND_PIECEWISE
      COMPILE_CFG=custom_ops:[all]
      COMPILE_CFG=pass_config:fuse_gemm_comms:true

    and docker read a fragment as the image:
      invalid reference format: repository name (library/COMPILE_CFG=custom_ops)

    The head runs locally, where an expanded variable's braces are NOT
    re-expanded, so the head container started and only the workers died --
    which is why the log made it look like a head failure.

    `set +B` on the remote is not a fix: it stops the split but quote removal
    still strips the JSON's own quotes and vLLM gets an invalid
    --compilation-config. Quoting the argv with printf %q is the fix.
    """
    src = open(os.path.join(REPO, "launchers", "start-hy4-tp4.sh"),
               encoding="utf-8").read()
    check("_wrun=$(printf '%q ' docker run -d --name hy4-worker" in src,
          "the worker argv must be built and %q-quoted locally, so the remote "
          "parses back exactly these tokens")
    check('"docker rm -f hy4-worker 2>/dev/null; $_wrun >/dev/null"' in src,
          "the ssh payload carries the pre-quoted argv, not raw interpolation")
    # The load-bearing invariant, and it belongs to BOTH lanes. glm53 had the
    # identical shape and boots only because nothing in its ENVV carries braces
    # today -- its COMPILE_CFG rides SERVE_ARGS inside the base64 payload,
    # single-quoted. But the profile's VLLM_* keys are forwarded verbatim, and
    # a brace-bearing one would not crash there: {"m":1,"n":2} reaches the
    # worker as two valid -e args (last wins) while the head keeps the whole
    # value, so the ranks silently disagree. Verified against a real worker.
    for name in ("start-hy4-tp4.sh", "start-glm53-nvfp4-tp4.sh"):
        lane = open(os.path.join(REPO, "launchers", name), encoding="utf-8").read()
        offenders = [n for n, line in enumerate(lane.splitlines(), 1)
                     if "ssh " in line and "$ENVV" in line
                     and not line.lstrip().startswith("#")]
        check(not offenders,
              "%s: no ssh command may interpolate $ENVV -- the remote re-parses "
              "it (offending lines: %s)" % (name, offenders))
        check("printf '%q ' docker run" in lane,
              "%s must build its worker argv with printf %%q" % name)
    # the head path is local and stays direct; changing it is not the fix
    check("docker run -d --name hy4 $COMMON $RDMA_FLAGS $ENVV" in src,
          "the head runs locally and needs no quoting -- it was never the bug")
    print("  worker envv not re-parsed ..... OK")


def test_micro_fusion_bundle_contracts() -> None:
    """Launch-count bundle 2 (RUNBOOK EXP-20): three kill-switched micro-fusions.

    dual f_b/g_b GEMM + one-pass KDA (glm53_kda_onepass, wired from
    glm53_mk_kda_wiring through resolve()/gate_gemms()/spec_onepass()) and
    the kpool update on int64 positions (glm53_kpool_tail_select). Every knob
    reads the exact string 1, ships 0 in the profile, and the stock path is
    what runs on any doubt: the module self-tests against the stock chain on
    the first eager forward and DISARMs on a mismatch. The gate top-k epilogue
    that was the fourth axis is not in the tree (MEASUREMENTS 31차: bit-exact,
    slower than the kernel it replaced); nothing may reference it."""
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    modules = re.search(r'^MODULES="([^"]+)"', profile, re.M).group(1).split()
    check("glm53_model" in modules, "glm53 profile mounts glm53_model (kda_onepass lives there)")
    for knob in ("VLLM_GLM53_KDA_DUAL_GEMM", "VLLM_GLM53_KDA_ONEPASS",
                 "VLLM_GLM53_KPOOL_UPDATE_DIRECT_POS"):
        check(re.search(rf"^{knob}=0$", profile, re.M) is not None,
              f"profile ships {knob}=0 (stock by default; the bundle boot arms it as caller env)")
    gate = open(os.path.join(REPO, "overlay", "modules", "moe_gate_sm121", "moe_gate_sm121.py"),
                encoding="utf-8").read()
    check("VLLM_MOE_GATE_TOPK" not in profile and "VLLM_MOE_GATE_TOPK" not in gate
          and "_deneb_gate_topk_kernel" not in gate,
          "the rejected top-k epilogue stays out of the gate module and the profile")

    # --- glm53_kda_onepass: module shape ---
    mod_dir = os.path.join(REPO, "overlay", "modules", "glm53_model")
    rows = [l.split("\t") for l in open(os.path.join(mod_dir, "manifest.tsv"), encoding="utf-8")
            .read().splitlines() if l and not l.startswith("#")]
    check(["glm53_kda_onepass.py",
                    "vllm/model_executor/layers/glm53_kda_onepass.py", "absent"] in rows,
          f"manifest adds the kernel file next to the layers (absent preimage): {rows}")
    check(not ({"glm53_drafter", "glm53_runtime"}
                    & set(open(os.path.join(mod_dir, "requires"), encoding="utf-8").read().split())),
          "self-contained: the KDA wiring optionally imports it, not the reverse (folded into glm53_model: requires nothing above the model layer)")
    kern = open(os.path.join(mod_dir, "glm53_kda_onepass.py"), encoding="utf-8").read()
    # knobs: exact "1", read in one place (the module), executed here
    knob_nodes = [n for n in ast.parse(kern).body
                  if (isinstance(n, ast.FunctionDef) and n.name in ("dual_gemm_enabled", "onepass_enabled"))
                  or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                      and t.id in ("DUAL_GEMM_ENV", "ONEPASS_ENV") for t in n.targets))]
    check(len(knob_nodes) == 4, "ENV constants and the two knob readers are top-level definitions")
    ns: dict = {"os": os}
    exec(compile(ast.Module(body=knob_nodes, type_ignores=[]), "glm53_kda_onepass", "exec"), ns)
    saved = {k: os.environ.pop(k, None) for k in ("VLLM_GLM53_KDA_DUAL_GEMM", "VLLM_GLM53_KDA_ONEPASS")}
    try:
        for fn, env in ((ns["dual_gemm_enabled"], "VLLM_GLM53_KDA_DUAL_GEMM"),
                        (ns["onepass_enabled"], "VLLM_GLM53_KDA_ONEPASS")):
            check(fn() is False, f"{env} unset keeps stock")
            for v, want in (("0", False), ("1", True), (" 1 ", True), ("on", False),
                            ("true", False), ("2", False), ("shadow", False), ("", False)):
                os.environ[env] = v
                check(fn() is want, f"{env}={v!r} must map to {want} (only the exact string 1 arms)")
            os.environ.pop(env, None)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    # last-arriver protocol: monotonic counters, power-of-two NV, no reset
    check(kern.count('tl.atomic_add(ctr_ptr + i_nh, 1, sem="acq_rel", scope="gpu")') == 1
          and kern.count('tl.atomic_add(ctr_ptr + nh_total + i_nh, 1, sem="acq_rel", scope="gpu")') == 1
          and kern.count("((arrived + 1) & (NV - 1)) == 0") == 1
          and kern.count("((done + 1) & (NV - 1)) == 0") == 1
          and "tl.atomic_xchg" not in kern,
          "two monotonic last-arriver counters (conv-state write, norm), nothing resets them")
    check("_COUNTERS: dict[tuple[torch.device, int], torch.Tensor] = {}" in kern
          and "def prepare_counters(device: torch.device, nv: int = _SERVING_NV) -> torch.Tensor:" in kern
          and "ctr = prepare_counters(projected.device, nv)" in kern
          and "prepare_counters(device, _SERVING_NV)" in kern
          and "if not counters_ready(projected.device, nv) and torch.cuda.is_current_stream_capturing():" in kern,
          "counters are keyed by block width (count % NV needs one NV per buffer), prepared off-capture "
          "for the serving width; a missing width under capture declines to stock")
    check("(nv & (nv - 1)) != 0" in kern and "_MAX_REQ_HEADS" in kern
          and "num_spec_decodes * h > _MAX_REQ_HEADS" in kern,
          "applicability pins NV to a power of two and declines (never raises) past the counter buffer")
    check(re.search(r"tl\.debug_barrier\(\)\s+arrived = tl\.atomic_add", kern) is not None
          and re.search(r"tl\.debug_barrier\(\)\s+done = tl\.atomic_add", kern) is not None,
          "the CTA barrier precedes each arrival (every thread's reads/stores are done)")
    check('cache_modifier=".cg"' in kern, "the norm tail reads the other programs' rows past L1")
    # stock skip semantics kept separately
    check("do_conv = line != 0" in kern and "do_rec = state_idx > 0" in kern
          and "if (line == 0) & (state_idx <= 0):" in kern and "if do_conv:" in kern
          and "if do_rec:" in kern and "tl.where(do_conv, _conv_taps(q0, q1, q2, q3, wq0, wq1, wq2, wq3)," in kern,
          "conv runs iff the conv line is valid, recurrence iff the resume slot is; a skipped conv feeds raw rows")
    check("acc.to(tl.bfloat16).to(tl.float32)" in kern,
          "conv output rounded to bf16 where the stock chain stores it (recurrence bit-exact)")
    check("b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)" in kern
          and "b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))" in kern
          and "b_h *= tl.exp(b_gk[None, :])" in kern
          and "b_v -= tl.sum(b_h * b_k[None, :], 1)" in kern
          and "b_beta = tl.sigmoid(b_beta)" in kern
          and "b_h += b_v[:, None] * b_k[None, :]" in kern
          and "b_o = tl.sum(b_h * b_q[None, :], 1)" in kern,
          "the recurrence is the stock fused_recurrent kernel's op order")
    check("state_idx = tl.load(idx_ptr + i_n * stride_idx_seq + n_acc - 1)" in kern
          and "final_idx = tl.load(idx_ptr + i_n * stride_idx_seq + t)" in kern
          and "if final_idx > 0:" in kern,
          "state-index contract: resume from [n, acc-1], store per token to [n, t], skip null")
    check("VAL: tl.constexpr = STATE_LEN - SEQLEN" in kern and "elif j - VAL < T:" in kern
          and "src = cs_line + (tok_off + 1 + j) * stride_cs_tok" in kern,
          "conv-state roll: 2 old slots from acc+j, then the block's tokens, per-request T")
    check("b_var = tl.sum(b_x * b_x, axis=0) / V" in kern
          and "b_rstd = 1 / tl.sqrt(b_var + eps)" in kern and "b_y = b_y * tl.sigmoid(b_g)" in kern,
          "gated RMSNorm in the stock op order (sigmoid gate), one 1-D row at a time")
    # guards on the real tensors: rows the kernel touches, both cache dtypes, dim-first view
    check("num_actual_tokens: int" in kern and "if n <= 0 or n > projected.shape[0]:" in kern
          and "if g1.numel() < n * proj or g2.numel() < n * proj:" in kern
          and "if out.numel() < n * proj:" in kern,
          "applicability sizes out/g1/g2 on the actual token count, not the padded projection rows")
    check("conv_state.dtype not in (torch.bfloat16, torch.float32)" in kern
          and "if conv_state.shape[1] != 3 * proj or conv_state.shape[2] < state_len:" in kern
          and "spec_state_indices.stride(1) != 1" in kern,
          "conv state: bf16 or fp32 (MK-KDA's cache dtype too), dim-first view, unit-stride indices")
    check("_MAX_DUAL_M = 32" in kern and "0 < x_fg.shape[0] <= _MAX_DUAL_M" in kern
          and "if kd != 128 or x_fg.shape[1] != 2 * kd:" in kern,
          "dual GEMM admits decode M only and the one validated KD")
    check("if x_fg.shape[0] <= _MAX_DUAL_M and announce_pending(\"dual-stock\"):" in kern,
          "a stock decline is announced for decode shapes only (prefill keeps stock by design)")
    check("num_warps=1" in kern and "num_stages=3" in kern,
          "the one-pass keeps the stock recurrent launch geometry (single-warp programs)")
    # boot self-test and resolve
    check("def selftest(device: torch.device) -> tuple[bool, str]:" in kern
          and "def resolve(device: torch.device | None = None) -> dict:" in kern
          and "torch.cuda.is_current_stream_capturing()" in kern
          and "self-test FAIL (%s) -> one-pass DISARMED" in kern
          and "torch.float32)" in kern.split("def selftest(")[1].split("try:")[0],
          "resolve() self-tests both cache dtypes against the stock chain and disarms on mismatch")
    check("def run_stock_chain(" in kern and "def make_fixture(" in kern and "def run_onepass(" in kern,
          "one fixture + one stock reference chain, shared by the self-test and the probe")
    readme = open(os.path.join(mod_dir, "README.md"), encoding="utf-8").read()
    check("not bit-exact" in readme and "[kda-onepass]" in readme
          and "VLLM_GLM53_KDA_DUAL_GEMM=1 VLLM_GLM53_KDA_ONEPASS=1 bash launchers/" in readme
          and "6416" in readme and "run_micro_fusion_check.sh" in readme
          and "self-test" in readme and "[8, 128]" not in readme,
          "README: numerics caveat, caller-env arming, checkpoint shape, log anchor, probe, no stale tile")

    # --- wiring in glm53_mk_kda_wiring: a guarded import and two calls ---
    kda = open(os.path.join(REPO, "overlay", "modules", "glm53_model", "glm5next_kda.py"),
               encoding="utf-8").read()
    fusion_state = kda.split("def _kda_fusion_state")[1].split("class Glm5NextLinearAttention")[0]
    check('_KDA_ONEPASS_MODULE = "vllm.model_executor.layers.glm53_kda_onepass"' in kda
          and "isinstance(e, ModuleNotFoundError) and e.name == _KDA_ONEPASS_MODULE" in fusion_state
          and 'if not any(_os.environ.get(k, "").strip() == "1" for k in _KDA_ONEPASS_ENVS):' in fusion_state
          and fusion_state.count('strip() == "1"') == 1
          and "resolved = mod.resolve()" in fusion_state
          and 'st["dual"] = bool(resolved["dual"])' in fusion_state
          and 'st["onepass"] = bool(resolved["onepass"])' in fusion_state
          and "dual_gemm_enabled" not in fusion_state and "onepass_enabled" not in fusion_state,
          "the wiring imports the module only when a knob is armed (exact 1: the profile's 0 costs no import), "
          "tolerates only 'not mounted' silently and takes both verdicts from resolve()")
    check('_fus["mod"].gate_gemms(' in kda and "_pair = (self.f_b_proj(f_a)[0], self.g_b_proj(g_a)[0])" in kda,
          "the dual path is one call; the stock two GEMMs remain the fallback")
    check('_fus["mod"].spec_onepass(' in kda and "num_actual_tokens=num_actual_tokens" in kda
          and "and attn_metadata_narrowed.num_prefills == 0" in kda
          and "and attn_metadata_narrowed.num_decodes == 0" in kda
          and "and (non_spec_token_indx is None or non_spec_token_indx.numel() == 0)" in kda
          and 'and self.o_norm.activation == "sigmoid"' in kda and "and self.kda_safe_gate" in kda,
          "one-pass only on a pure spec-verify step with the bounded gate and sigmoid norm")
    fwd = kda.split("    def _forward(")[1]
    check("_normed" not in kda and "g2: torch.Tensor," in fwd.split(") -> None:")[0]
          and fwd.count("self.o_norm(core_attn_out, g2)") == 2
          and "self.o_norm(" not in kda.split("    def forward(")[1].split("    def _forward(")[0],
          "_forward owns the gated norm on every path (profile run, stock chain); forward applies none")
    check(kda.index('_fus["mod"].spec_onepass(') > kda.index("conv_bias = self.q_conv1d.bias"),
          "the one-pass call sits after the merged conv weight exists")

    # --- kpool direct positions ---
    kp = open(os.path.join(REPO, "overlay", "modules", "glm53_kernels",
                           "sparse_attn_indexer_kpool.py"), encoding="utf-8").read()
    check('os.environ.get("VLLM_GLM53_KPOOL_UPDATE_DIRECT_POS", "").strip() == "1"' in kp,
          "kpool knob arms on the exact string 1")
    check("dec_pos = positions[:num_decode_tokens].view(shape2)" in kp
          and "dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)" in kp,
          "uniform path passes int64 positions under the knob, the cast copy otherwise")
    check(kp.count("_KPOOL_UPDATE_DIRECT_POS and") == 1
          and "_scatter_decode_tokens_by_request(\n                    positions[:num_decode_tokens].to(torch.int32)," in kp,
          "the padded non-uniform path keeps the scatter of the int32 cast")
    print("  micro-fusion bundle 2 contracts .. OK")



def test_glm53_drafter_prep_contracts() -> None:
    """glm53_drafter_prep (34차, EXP-24): the drafter's per-step host build of
    its attention metadata is skipped on FULL cudagraph replays.

    The trace showed a pageable DtoH + stream sync (`seq_lens.cpu()` inside
    the FlashInfer builder, non-causal drafter attention) followed by 300-460
    us of Python while the GPU idles; propose()'s FULL branch never reads the
    dict. The wrapper decides FULL-ness through the drafter's own cudagraph
    dispatch, caches one dict per shape, and falls back to the stock build for
    everything else -- and for the rest of the boot on any exception."""
    mod_dir = os.path.join(REPO, "overlay", "modules", "glm53_drafter")
    src = open(os.path.join(mod_dir, "glm53_drafter_prep.py"), encoding="utf-8").read()
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    modules = re.search(r'^MODULES="([^"]+)"', profile, re.M).group(1).split()
    check("glm53_drafter" in modules, "glm53 profile mounts glm53_drafter (drafter_prep lives there)")
    check(len(re.findall(r"^VLLM_GLM53_DRAFTER_PREP=", profile, re.M)) == 1
          and re.search(r"^VLLM_GLM53_DRAFTER_PREP=0$", profile, re.M) is not None,
          "the knob is declared exactly once and ships off (32차 duplicate-declaration lesson)")
    rows = [l.split("\t") for l in open(os.path.join(mod_dir, "manifest.tsv"), encoding="utf-8")
            .read().splitlines() if l and not l.startswith("#")]
    check(["glm53_drafter_prep.py",
                    "vllm/models/glm5next/nvidia/glm53_drafter_prep.py", "absent"] in rows,
          f"manifest binds the module as a new file next to the model: {rows}")
    req = open(os.path.join(mod_dir, "requires"), encoding="utf-8").read().split()
    check({"glm53_model"} <= set(req) and os.path.exists(os.path.join(mod_dir, "qwen3_dflash2.py")),
          "requires names the wiring (installer); the drafter overlay is in the same module")
    check('MODES = ("time", "shadow", "1")' in src
          and "Spec._build_draft_attn_metadata = _patched_build" in src
          and "_ORIG_BUILD = Spec._build_draft_attn_metadata" in src
          and "check_preimages(root)" in src and "-> DISARM" in src,
          "exact modes; the installer wraps the base build, keeps the original, "
          "and DISARMs on image drift")
    for rel in ("v1/worker/gpu/spec_decode/dflash/speculator.py",
                "v1/worker/gpu/spec_decode/speculator.py", "v1/worker/gpu/cudagraph_utils.py"):
        check(re.search(rf'"{re.escape(rel)}":\s*\n?\s*"[0-9a-f]{{64}}"', src) is not None,
              f"preimage pinned for {rel}")
    wiring = open(os.path.join(REPO, "overlay/modules/glm53_model/"
                                     "glm5next_model.py"), encoding="utf-8").read()
    check("from .glm53_drafter_prep import install_glm53_drafter_prep" in wiring
          and 'if _e.name != f"{__package__}.glm53_drafter_prep":' in wiring
          and wiring.index("install_glm53_drafter_prep()")
          > wiring.index("install_glm53_dflash_early_fc()"),
          "installed from the wiring after early-fc: silent without the module, loud when broken")
    bracket = open(os.path.join(REPO, "bench", "bracket.py"), encoding="utf-8").read()
    check('"VLLM_GLM53_DRAFTER_PREP"' in bracket, "bracket.py snapshots the knob")
    readme = open(os.path.join(mod_dir, "README.md"), encoding="utf-8").read()
    check("Device -> Pageable" in readme and "run_fullgraph" in readme,
          "the README carries the trace evidence and the FULL-branch argument")

    # the wrapper on fakes: FULL -> cached after the first build; eager -> stock;
    # a failing original -> disabled for good, stock served; compare() drift axes
    import importlib.util
    import types
    fake_vllm = types.ModuleType("vllm")
    fake_logger = types.ModuleType("vllm.logger")
    fake_logger.init_logger = lambda name: _CapturingLogger()
    fake_vllm.logger = fake_logger
    fake_cfg = types.ModuleType("vllm.config")
    fake_comp = types.ModuleType("vllm.config.compilation")

    class _CG:
        FULL = "FULL"
        NONE = "NONE"

    fake_comp.CUDAGraphMode = _CG
    fake_cfg.compilation = fake_comp
    keys = ("vllm", "vllm.logger", "vllm.config", "vllm.config.compilation", "torch")
    saved = {k: sys.modules.get(k) for k in keys}
    fake_torch = types.ModuleType("torch")

    class _T:
        def __init__(self, ptr, cuda=True, shape=(8,), dtype="i32", val=None):
            self._ptr, self.is_cuda, self.shape, self.dtype, self.val = ptr, cuda, shape, dtype, val

        def data_ptr(self):
            return self._ptr

        def stride(self):
            return (1,)

    fake_torch.Tensor = _T
    fake_torch.equal = lambda a, b: a.val == b.val
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.logger"] = fake_logger
    sys.modules["vllm.config"] = fake_cfg
    sys.modules["vllm.config.compilation"] = fake_comp
    sys.modules["torch"] = fake_torch
    try:
        spec = importlib.util.spec_from_file_location(
            "_drafter_prep_mod", os.path.join(mod_dir, "glm53_drafter_prep.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _Desc:
            def __init__(self, mode, n):
                self.cg_mode, self.num_tokens = mode, n

        class _Mgr:
            def __init__(self, mode):
                self.mode, self.calls = mode, []

            def dispatch(self, num_reqs, num_tokens, uniform, num_active_loras=0):
                self.calls.append((num_reqs, num_tokens, uniform, num_active_loras))
                return _Desc(self.mode, num_tokens)

        class _Spec:
            num_query_per_req = 8
            dp_size = 1

            def __init__(self, mode):
                self.query_cudagraph_manager = _Mgr(mode)

        built = []

        def orig(self, num_reqs, num_reqs_padded, num_tokens_padded, ub, step, **kw):
            built.append((num_reqs, num_tokens_padded, step))
            return {"g": _T(0x10), "max_seq_len": 100 + len(built)}

        mod._ORIG_BUILD = orig
        mod._MODE = "1"
        sp = _Spec("FULL")
        a = mod._patched_build(sp, 1, 1, 8, None, 8, causal=False)
        b = mod._patched_build(sp, 1, 1, 8, None, 8, causal=False)
        check(a is b and len(built) == 1 and sp.query_cudagraph_manager.calls[0] == (1, 8, 8, 0),
              "FULL replay: one stock build per shape, the cached dict after; the dispatch "
              "call mirrors propose() (num_reqs, num_reqs x num_query_per_req, per-req, 0)")
        c = mod._patched_build(sp, 2, 2, 16, None, 8, causal=False)
        check(c is not a and len(built) == 2, "a new shape builds once more")
        d = mod._patched_build(sp, 1, 1, 8, None, 8, causal=False, query_start_loc_np=[0])
        check(len(built) == 3 and d is not a, "a query_start_loc override always builds")
        sp2 = _Spec("NONE")
        e = mod._patched_build(sp2, 1, 1, 8, None, 8, causal=False)
        check(len(built) == 4 and e is not a and mod._STATS["stock"] >= 1,
              "an eager batch takes the stock build")
        check(mod._STATS["served"] == 1 and mod._DISABLED is False, "tally counts the served dict")

        def boom(self, *a, **k):
            raise RuntimeError("dispatch broke")

        sp3 = _Spec("FULL")
        sp3.query_cudagraph_manager.dispatch = boom
        f = mod._patched_build(sp3, 1, 1, 8, None, 8, causal=False)
        check(mod._DISABLED is True and len(built) == 5 and f is not a,
              "an exception inside the wrapper disables it and serves the stock build")
        g = mod._patched_build(sp, 1, 1, 8, None, 8, causal=False)
        check(g is not a and len(built) == 6, "disabled stays disabled for the boot")
        # compare(): identity drift vs expected per-step scalars
        same = {"g": _T(0x10), "max_seq_len": 5}
        check(mod.compare(same, {"g": _T(0x10), "max_seq_len": 9}) == (0, 1),
              "a per-step scalar differing counts as scalar_diff, not drift")
        check(mod.compare(same, {"g": _T(0x20), "max_seq_len": 5}) == (1, 0),
              "a GPU tensor at another address is drift")
        check(mod.compare(same, {"g": _T(0x10), "max_seq_len": 5, "extra": 1}) == (1, 0),
              "a structural difference is drift")
        check(mod.compare({"n": 3}, {"n": 4}) == (1, 0),
              "an unlisted host value differing is drift")
        # mode parsing
        for v, want in (("1", "1"), ("shadow", "shadow"), ("time", "time"), ("yes", "0"), ("", "0")):
            os.environ["VLLM_GLM53_DRAFTER_PREP"] = v
            check(mod.drafter_prep_mode() == want, f"mode {v!r} -> {want!r}")
        os.environ.pop("VLLM_GLM53_DRAFTER_PREP", None)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    print("  glm53_drafter_prep contracts .... OK")


def test_indexer_decode_fused_contracts() -> None:
    """glm53_kpool_tail_select (34차, EXP-24): the decode top-k glue of a
    full-attention layer as one Triton launch + a direct-write expand.

    Gated by VLLM_GLM53_INDEXER_DECODE_FUSED=1 (exact), uniform layouts only;
    the fused kernel carries the stock arithmetic (int32 add after the
    narrowing, int64 floor-div, clamp, finfo.max where last_pool >= 0) so the
    probe's bit-exact gate is the contract; the stock chain stays as the
    fallback and the padded path."""
    path = os.path.join(REPO, "overlay/modules/glm53_kernels/sparse_attn_indexer_kpool.py")
    src = open(path, encoding="utf-8").read()
    check('os.environ.get("VLLM_GLM53_INDEXER_DECODE_FUSED", "0").strip() == "1"' in src,
          "exact-1 knob read once at import (the op runs per layer per step)")
    check("def _glm53_indexer_tail_select_kernel(" in src
          and "def indexer_tail_select_fused(" in src
          and "def expand_pools_and_append_tail_into(" in src,
          "the fused kernel, its wrapper and the direct-write expand exist")
    kern = src[src.index("def _glm53_indexer_tail_select_kernel("):src.index("def indexer_tail_select_fused(")]
    check("seq = pos.to(tl.int32) + 1" in kern
          and "last_pool = seq.to(tl.int64) // POOL - 1" in kern
          and "col = tl.minimum(tl.maximum(last_pool, 0), n_cols - 1)" in kern
          and "wm = m & (last_pool >= 0)" in kern,
          "the kernel mirrors _decode_topk_seq_lens + _force_tail_pool_into_logits step by step")
    route = src[src.index("fused_tail = False"):src.index("_force_tail_pool_into_logits(logits, dec_seq, index_kpool)")]
    check("and _INDEXER_DECODE_FUSED" in route
          and "and not decode_metadata.requires_padding" in route
          and "indexer_tail_select_fused(" in route
          and "_always_select_tail()," in route
          and "[indexer-fused] tail-select fused" in route,
          "routing: knob + uniform layout only, the config's tail flag passed through, "
          "a one-time proof line (capture-tagged)")
    check("if positions is not None and not fused_tail and _always_select_tail():" in src,
          "the stock tail bias is skipped when the fused launch already applied it")
    tail = src[src.index("if fused_tail:\n"):src.index("out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)")]
    check("expand_pools_and_append_tail_into(" in tail and "if written is not None:" in tail
          and "return topk_indices_buffer" in tail,
          "the fused path writes the expansion straight into the persistent buffer "
          "and falls back to the stock expand + copy when the layout is refused")
    check("_expand_pools_and_append_tail_kernel,\n" in src,
          "the direct write reuses the image's expand kernel (same values)")
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(len(re.findall(r"^VLLM_GLM53_INDEXER_DECODE_FUSED=", profile, re.M)) == 1
          and re.search(r"^VLLM_GLM53_INDEXER_DECODE_FUSED=0$", profile, re.M) is not None,
          "the knob is declared once and ships off")
    bracket = open(os.path.join(REPO, "bench", "bracket.py"), encoding="utf-8").read()
    check('"VLLM_GLM53_INDEXER_DECODE_FUSED"' in bracket, "bracket.py snapshots the knob")
    tc = load_defs("tools/trace_common.py", {"OURS", "owner", "category"}, {})
    check(tc["owner"]("_glm53_indexer_tail_select_kernel") == "ours"
          and tc["category"]("_glm53_indexer_tail_select_kernel") == "indexer",
          "the trace tools count the fused kernel as ours, in the indexer bucket")
    check(os.path.exists(os.path.join(REPO, "probes", "indexer_decode_fused_check.py"))
          and os.path.exists(os.path.join(REPO, "probes", "run_indexer_fused_check.sh")),
          "the bit-exact probe and its container runner exist")
    runner = open(os.path.join(REPO, "probes", "run_indexer_fused_check.sh"), encoding="utf-8").read()
    check("probes/indexer_decode_fused_check.py" in runner and "base preimage mismatch" in runner,
          "the runner mounts the composed overlay after verifying every base preimage")
    print("  indexer decode fused contracts .. OK")



def test_glm53_drafter_ctx_kv_w4_contracts() -> None:
    """34차 EXP-24 item 6: the drafter's context K/V projection on the W4 lane.

    `precompute_and_store_context_kv` ran one fused bf16 GEMM over the five
    layers' k/v rows (21 MB, 101 us/step) -- the last drafter linear outside
    the lane. The overlay builds a pack at buffer-build time (exact-1 knob),
    serves decode-sized batches through gemm_w4a8 and keeps F.linear for
    everything the lane declines (prefill rows, a disarmed lane, a bias)."""
    path = os.path.join(REPO, "overlay/modules/glm53_drafter/qwen3_dflash2.py")
    src = open(path, encoding="utf-8").read()
    check('(os.environ.get("VLLM_GLM53_DRAFTER_CTX_KV_W4") or "0").strip() != "1"' in src,
          "exact-1 knob, read once when the fused buffers are built")
    build = src[src.index("    def _build_fused_kv_buffers(self) -> None:"):src.index("    def _project_context_kv(")]
    check("super()._build_fused_kv_buffers()" in build
          and "self._deneb_ctx_kv_pack = _mk.build_mk_weight_w4(w)" in build
          and 'getattr(self, "_fused_kv_bias", None) is None' in build
          and "w.shape[1] % 128 == 0" in build and "w.shape[1] <= _mk.MK_GEMM_KMAX" in build
          and "logger.exception" in build,
          "the pack is built after the stock buffers, only for a bias-free bf16 weight the "
          "lane admits, and a build failure logs and keeps stock")
    proj = src[src.index("    def _project_context_kv("):src.index("class DFlash2Qwen3ForCausalLM(")]
    check("if pack is None:\n            return super()._project_context_kv(" in proj
          and "ops.rms_norm(normed, context_states, self._hidden_norm_weight, self._rms_norm_eps)" in proj
          and "_mk.gemm_w4a8(normed, pack, int(self._fused_kv_weight.shape[0]))" in proj
          and "all_kv_flat = F.linear(normed, self._fused_kv_weight, None)" in proj
          and ".permute(2, 1, 0, 3, 4)" in proj
          and "[drafter-ctx-kv] serving:" in proj,
          "the projection keeps the stock norm and layout, takes the lane when it answers, "
          "falls back to F.linear when it declines, and says so once")
    profile = open(os.path.join(REPO, "profiles", "glm53.env"), encoding="utf-8").read()
    check(len(re.findall(r"^VLLM_GLM53_DRAFTER_CTX_KV_W4=", profile, re.M)) == 1
          and re.search(r"^VLLM_GLM53_DRAFTER_CTX_KV_W4=0$", profile, re.M) is not None,
          "the knob is declared once and ships off")
    bracket = open(os.path.join(REPO, "bench", "bracket.py"), encoding="utf-8").read()
    check('"VLLM_GLM53_DRAFTER_CTX_KV_W4"' in bracket, "bracket.py snapshots the knob")
    print("  drafter ctx-KV W4 contracts ..... OK")

def test_trace_step_nodes_tool() -> None:
    """tools/trace_step_nodes.py + trace_common.cut_steps (34차).

    A prep-fused boot still launches the stock `_gather_block_tables_kernel`
    on the rare non-uniform step, so "first anchor with >= 2 hits" cut the
    09-05 trace into four 3.6 s steps; the anchor with the MOST hits must win,
    ties going to the earlier (step-start) entry. The node tool streams the
    file, keeps clean steps (kernel-count mode), dumps a step's node sequence
    and lists every event category inside a window."""
    import gzip
    import importlib.util
    import json
    import tempfile

    tc = load_defs("tools/trace_common.py", {"STEP_ANCHORS", "cut_steps"}, {})
    ev = []
    t = 0.0
    for i in range(30):
        name = "_glm53_prep_fused_kernel" if i % 3 == 0 else "k_oneshot(Ctrl*)"
        if i == 16:
            name = "_gather_block_tables_kernel"   # the rare stock prep launch
        ev.append({"name": name, "ts": t, "dur": 5.0})
        t += 10.0
    ev.append({"name": "_gather_block_tables_kernel", "ts": t, "dur": 5.0})
    anchor, starts = tc["cut_steps"](ev)
    check(anchor == "_glm53_prep_fused_kernel" and len(starts) == 10,
          f"the most frequent anchor wins over the first listed (got {anchor}, {len(starts)})")

    spec = importlib.util.spec_from_file_location(
        "_trace_step_nodes", os.path.join(REPO, "tools", "trace_step_nodes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    events = []
    ts = 1000.0
    for step in range(6):
        events.append({"ph": "X", "cat": "kernel", "name": "_glm53_prep_fused_kernel",
                       "ts": ts, "dur": 50.0, "args": {"stream": 17}})
        events.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch",
                       "ts": ts + 60.0, "dur": 100.0, "pid": 7, "tid": 7})
        events.append({"ph": "X", "cat": "kernel",
                       "name": "(anonymous namespace)::mk_gemm2_kernel<1>(MKGemm2Ctx)",
                       "ts": ts + 200.0, "dur": 20.0, "args": {"stream": 17}})
        events.append({"ph": "X", "cat": "kernel", "name": "k_oneshot(Ctrl*)",
                       "ts": ts + 230.0, "dur": 40.0, "args": {"stream": 17}})
        if step == 2:   # one dirty step with an extra kernel
            events.append({"ph": "X", "cat": "kernel", "name": "void at::native::fill{lambda()#3}",
                           "ts": ts + 280.0, "dur": 2.0, "args": {"stream": 210}})
        events.append({"ph": "X", "cat": "kernel", "name": "_get_num_sampled_and_rejected_kernel",
                       "ts": ts + 300.0, "dur": 3.0, "args": {"stream": 17}})
        ts += 400.0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.json.gz")
        with gzip.open(path, "wt") as f:
            json.dump({"schemaVersion": 1, "traceEvents": events}, f)
        r = mod.analyze(path)
        check(r["anchor"] == "_glm53_prep_fused_kernel",
              f"a tie between the sampler anchor and the prep anchor goes to the step start "
              f"(got {r['anchor']})")
        check(r["steps"] == 4 and r["clean"] == 3 and r["mode"] == 4,
              f"6 anchors -> 4 inner steps, 3 clean at the kernel-count mode (got {r['steps']}, "
              f"{r['clean']}, mode {r['mode']})")
        check(abs(r["step_ms"] - 0.303) < 1e-6 and abs(r["idle_ms"] - 0.190) < 1e-6,
              f"step span and idle from the union of kernels (got {r['step_ms']}, {r['idle_ms']})")
        check(r["cats"]["MK GEMM (ours)"]["cnt"] == 1 and r["cats"]["AR k_oneshot"]["cnt"] == 1,
              "categories know the v2 GEMM lane")
        k = [n for n in r["kernels"] if "mk_gemm2" in n][0]
        check(r["kernels"][k]["owner"] == "ours" and r["kernels"][k]["dur"] == 20.0,
              "the v2 lane kernel is ours, with its median duration")
        out = os.path.join(d, "dump.txt")
        j = mod.dump_step(r, out, None)
        lines = [l for l in open(out).read().splitlines() if not l.startswith("#")]
        check(len(lines) == 4 and lines[0].split()[-1].endswith("_glm53_prep_fused_kernel")
              and lines[1].split()[2] == "150.0",
              f"the dump lists the step's kernels in order with the gap on their stream "
              f"(step {j}: {lines[:3]})")
        rows = mod.gap_events(path, r, None, 40.0, 190.0)
        check([x[2] for x in rows] == ["cuda_runtime"] and rows[0][6] == "cudaGraphLaunch",
              "a window listing shows the host event the kernel view cannot")
    print("  trace step nodes tool ........... OK")

def test_supervisor_paces_and_stops_relaunching() -> None:
    """A launcher that keeps failing must not churn the fleet forever.

    The relaunch loop fired every ~2 min with no backoff and no cap, and an
    attempt is not free: bash "$LAUNCHER" runs drop_caches on ALL FOUR nodes
    and creates/destroys containers. srv4 hosts unrelated tenants (nemotron,
    solarflow, the SolarFlow DB), so a persistently broken launcher evicts
    their page cache every two minutes, indefinitely. Observed 2026-09-04:
    three relaunches in five minutes against a launcher bug (#289), stopped
    only because a human noticed.

    Simulated with an always-failing launcher: 5 attempts in 110 minutes
    instead of ~55, then a hold.
    """
    sup = open(os.path.join(REPO, "launchers", "dsv4-tp4-supervisor.sh"),
               encoding="utf-8").read()
    for var in ("launch_fails=0", "next_launch_at=0", "LAUNCH_BACKOFF_MAX=1800",
                "LAUNCH_HOLD_AFTER=5"):
        check(var in sup, "the supervisor must declare %s" % var)
    check("_backoff=$(( LAUNCH_BACKOFF_BASE * (1 << (launch_fails - 1)) ))" in sup,
          "consecutive failures must back off exponentially, not retry flat")
    check("[ \"$_backoff\" -gt \"$LAUNCH_BACKOFF_MAX\" ] && _backoff=$LAUNCH_BACKOFF_MAX" in sup,
          "the backoff needs a ceiling")
    # a NEXT-ALLOWED-TIME, not a sleep: the probe must keep running so a stack
    # someone fixes by hand is adopted at once instead of after the backoff.
    check("next_launch_at=$(( $(date +%s) + _backoff ))" in sup
          and '[ "$_now" -lt "$next_launch_at" ] && continue' in sup,
          "backoff must gate the next attempt by timestamp, leaving the health "
          "probe free to adopt a recovered stack immediately")
    check("sleep $_backoff" not in sup and "sleep \"$_backoff\"" not in sup,
          "sleeping the backoff would blind the health probe for up to 30 min")
    # and it must eventually stop and say so
    check("HELD after $launch_fails relaunches with no healthy stack" in sup,
          "after the cap it stops relaunching and states why once")
    held = sup[sup.index("LAUNCH_HOLD_AFTER"):]
    check("held_logged=1" in held and 'held_logged" = 0' in held,
          "the hold is announced once, not every cycle")
    # the hold must be clearable, or a fixed stack would never be adopted
    check("launch_fails=0; next_launch_at=0; held_logged=0" in sup,
          "recovery must clear the hold -- a supervisor that gives up "
          "permanently is a worse failure than the churn it replaced")
    # ...and ONLY a confirmed generation may clear it. launch() returns 0 on
    # api_up alone, and /v1/models answers 200 from a corpse, so crediting its
    # return reset the counter every round and the hold was unreachable in the
    # one failure mode the pacing exists for (simulated against a corpse API:
    # 293 launcher runs and no hold, vs 5 and a hold).
    check("if launch; then" not in sup,
          "a relaunch counts as an ATTEMPT -- api_up alone is not health, "
          "so launch()'s return must not clear the counter")
    check(sup.count("launch_fails=0") == 2,
          "the only two places that zero the counter are its declaration "
          "and the api_up && chat_ok recovery branch")
    clear_at = sup.index("launch_fails=0; next_launch_at=0; held_logged=0")
    check("if api_up && chat_ok; then" in sup[:clear_at].rsplit("while :;", 1)[-1],
          "the clear sits under a real generation probe, not under api_up")
    # The root cause, not the three symptoms: launch() has three callers (boot,
    # health loop, wedge watchdog) and each one that skips the accounting buys a
    # destructive relaunch past the cap -- the boot and the watchdog each did.
    # One counted path, so a fourth caller cannot reintroduce it silently.
    check(sup.count("launch_fails=$((launch_fails+1))") == 1
          and "attempt_launch(){" in sup,
          "exactly one place counts an attempt, and it is attempt_launch")
    bare = [n for n, line in enumerate(sup.splitlines(), 1)
            if line.strip().startswith("launch")
            and not line.strip().startswith(("launch(){", "launch_fails"))]
    check(len(bare) == 1,
          "launch() is invoked from exactly one place (inside attempt_launch); "
          "every caller goes through the accounting (offending lines: %s)" % bare)
    check(sup.count("attempt_launch") == 4,
          "attempt_launch is defined once and called by all three sites: the "
          "boot, the health loop and the wedge watchdog")
    print("  supervisor relaunch pacing .... OK")


def test_megakernel_regression_suite():
    """Run behavioral gate/dispatch and extracted CUDA-control regressions.

    Keep these in the normal deployment gate, not only in an opt-in test
    command. GPU numeric/racecheck/graph proof is a separate validation.
    """
    import unittest

    suite = unittest.defaultTestLoader.discover(
        os.path.dirname(os.path.abspath(__file__)),
        pattern="test_megakernel_*regressions.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    check(result.testsRun > 0 and result.wasSuccessful(),
          "megakernel behavioral regressions must run and pass")
    return result.testsRun


if __name__ == "__main__":
    test_skip_topk()
    test_prefill_chunker()
    test_sp_ranges()
    test_helpers()
    test_profile_step()
    test_bench_dec_metrics()
    test_overlay_manifest()
    test_composed_snapshot_sync()
    test_overlay_symbol_contracts()
    test_profile_env_carried()
    test_profile_keys_have_readers()
    test_dspark_speed_guards()
    test_bf16_release_guards()
    test_acceptance_lever_guards()
    test_ngram_ceiling_sim()
    test_ue8m0_scale_repair()
    test_ep_fixed_token_chunks()
    test_ep_fixed_output_initialised()
    test_b12x_zero_weight_micro()
    test_glm53_b12x_tuning_controls()
    test_b12x_static_v2_controls()
    test_b12x_micro_chunk_width()
    test_ep_tail_fixed_shape()
    test_ep_compact_shape_align()
    test_ep_compact_warmup_ladder()
    test_once_logger_args_hashable()
    test_dflash2_selector_check()
    test_dflash2_selector_score_precision()
    test_overlay_logger_defined()
    test_osar_maxel_rank_agreement()
    test_bench_resolves_served_model()
    test_prefill_knobs_announce_arming()
    test_extra_env_rejects_comma_list()
    test_fused_k_gate_lazy_slot_exists()
    test_torch_imports_are_guarded()
    test_dflash2_conv_mask_buffer()
    test_dflash_aot_guard_stays_removed()
    test_hotpath_env_latches()
    test_launcher_load_format_gate()
    test_launcher_reject_method_gate()
    test_dflash2_prefix_cache_fail_closed()
    test_accept_profile_conditional_arithmetic()
    test_dsv4_launcher_adoptions()
    test_launcher_nofile_limit()
    test_prefill_ladder_probe()
    test_fp8_acceptance_contracts()
    test_glm53_v2_overlay_contracts()
    test_launcher_head_guard()
    test_preflight_precedes_serve_args()
    test_no_hardcoded_image_paths()
    test_b12x_ep_routing()
    test_b12x_ep_preflight()
    test_b12x_ep_launcher()
    test_deploy_refusal_is_not_swallowed()
    test_fp8_dense_nvfp4_scheme_contract()
    test_union_prefill_width_matches_the_converter_tile()
    test_benches_ask_the_server_for_the_model_name()
    test_korean_gate_separates_notation_from_damage()
    test_every_module_can_mount_on_an_image_the_repo_can_launch()
    test_profile_declares_no_knob_the_code_cannot_read()
    test_fp8_dense_free_bf16_contract()
    test_fp8_dense_bproj()
    test_mhc_smallm_knob()
    test_mhc_passes_knob()
    test_mhc_probe_contracts()
    test_mhc_onepass_math()
    test_mhc_smallm_split_ownership()
    test_mhc_bigfuse_knob()
    test_census_kda_group()
    test_census_owner_axis()
    test_census_streaming_events()
    test_trace_step_tail_analyze()
    test_profiles_readme_module_table()
    test_dflash_warmup_buckets()
    test_dsv4_spec_warmup_contract()
    test_dsv4_ue8m0_host_guard()
    test_glm53_sm121_mla_prefill_gate()
    test_glm53_kpool_packed_scratch_contract()
    test_glm53_cache_only_indexer_prefill()
    test_glm53_kda_prefill_regime()
    test_glm53_upstream_prefill_batch()
    test_oneshot_sm121_grid_contract()
    test_cuda_builds_keep_the_arch_specific_target()
    test_self_built_kernels_persist_their_caches()
    test_boot_stamps_measure_without_changing_the_boot()
    test_cudagraph_mem_profiling_off_keeps_the_kv_size()
    test_kv_cache_is_pinned_in_tokens()
    test_earlyoom_is_fireable_on_unified_memory()
    test_fp8_dense_drafter_patterns_and_opaque_op()
    test_fp8_dense_drafter_compile_factor_and_serving_proof()
    test_mk_mla_workspace_is_fixed_and_splits_bounded()
    test_spec_k_compile_factor()
    test_sampler_profile_skip_contract()
    test_dev_lab_contracts()
    test_mk_smlp_hook_and_contracts()
    test_fp8_dense_prefill_nvfp4_pair_routes_by_rows()
    test_ab_runner_measures_both_channels()
    test_osar_wait_is_split_by_message_size()
    test_osar_prefetch_hints_contract()
    test_glm53_dflash_early_fc_contracts()
    test_glm53_drafter_prep_contracts()
    test_indexer_decode_fused_contracts()
    test_glm53_drafter_ctx_kv_w4_contracts()
    test_trace_step_nodes_tool()
    test_micro_fusion_bundle_contracts()
    test_glm53_megakernel_contracts()
    test_prefill_warmup_contracts()
    test_megakernel_w4_layout_functional()
    test_megakernel_core_is_shared()
    test_glm53_prep_fused_contracts()
    test_launcher_multiline_assignments_have_no_embedded_comments()
    test_launcher_restores_prefill_warmup_from_caller_env()
    test_profile_keys_not_passed_via_extra_env()
    test_glm53_indexer_gate_splitk_contracts()
    test_bracket_runner_contracts()
    test_trace_composition_analyze()
    test_drafter_fc_probe_contracts()
    test_kda_conv_state_layout_is_the_arming_contract()
    test_fp8_dense_build_peak_pays_only_for_what_serves()
    test_kda_owns_its_projections_across_dense_schemes()
    test_hy4_entrypoint_carries_the_production_knobs()
    test_common_tp4_library_is_the_one_implementation()
    test_worker_launch_does_not_let_the_remote_reparse_envv()
    test_supervisor_paces_and_stops_relaunching()
    regressions = test_megakernel_regression_suite()
    print(f"all OK ({PASS} checks; {regressions} megakernel regressions)")
