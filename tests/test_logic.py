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
        if isinstance(node, ast.FunctionDef) and node.name in names:
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
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)} | {
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
    owners = {}          # dotted module path -> (module dir, source path)
    for manifest in sorted(glob.glob(
            os.path.join(REPO, "overlay", "modules", "*", "manifest.tsv"))):
        moddir = os.path.dirname(manifest)
        for raw in open(manifest, encoding="utf-8"):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            source, target = line.split("\t")[:2]
            if not source.endswith(".py"):
                continue
            # Targets are absolute for image-bound overlays and relative to the
            # package root for portable ones, so anchoring on "/vllm/" silently
            # skipped every portable module -- including moe_gate_sm121, the one
            # this check exists for.
            if target.startswith("vllm/"):
                rel = target
            elif "/vllm/" in target:
                rel = "vllm/" + target.split("/vllm/", 1)[1]
            else:
                continue
            dotted = rel[:-3].replace("/", ".")
            owners[dotted] = os.path.join(moddir, source)

    checked = 0
    for dotted, srcpath in sorted(owners.items()):
        provided = set()
        tree = ast.parse(open(srcpath, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                provided.add(node.name)
            elif isinstance(node, ast.Assign):
                provided.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                provided.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                provided.update((a.asname or a.name).split(".")[0] for a in node.names)

        for other in sorted(set(owners.values())):
            if other == srcpath:
                continue
            for node in ast.walk(ast.parse(open(other, encoding="utf-8").read())):
                if not isinstance(node, ast.ImportFrom) or node.module != dotted:
                    continue
                for alias in node.names:
                    checked += 1
                    check(alias.name in provided,
                          f"{os.path.basename(other)} imports {alias.name} from "
                          f"{dotted}, which {os.path.basename(srcpath)} does not define")
    print(f"  overlay symbol contracts ({checked}) ... OK")


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
        "overlay/modules/fp8_lm_head/fp8_lm_head.py",
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
        "glm53_v2_sampler_guards" in modules
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
        {"_INCLUDE", "_BPROJ_INCLUDE", "_BPROJ_ON", "_include_patterns"},
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


# ---------------------------------------------------------------------------
# EP stock top-k token chunks: strict opt-in above the preserved #146 fallback.
# ---------------------------------------------------------------------------
def test_ep_fixed_token_chunks() -> None:
    ns = load_defs(
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py",
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
            "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py"
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py"
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

    dispatch_path = "overlay/modules/b12x_zero_weight_micro/moe_dispatch.py"
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

    kernel_path = "overlay/modules/b12x_zero_weight_micro/moe_micro_kernel.py"
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
    check("b12x_zero_weight_micro" in profile,
          "glm53 composition must mount both FlashInfer source overlays")
    check("VLLM_B12X_EP_ZERO_WEIGHT_MICRO=0" in profile,
          "profile must keep the numeric experiment default off")
    manifest = open(os.path.join(
        REPO, "overlay", "modules", "b12x_zero_weight_micro", "manifest.tsv"
    ), encoding="utf-8").read()
    check("ccb6f65a22314961693493242f78f62ca58f79a319ecd0cb51bf6d7d8e7125c6"
          in manifest, "micro kernel must pin the live preimage SHA")
    check("f6923850c710eb21cf7c3566b6ddcc39ddda0d7a3664c19dec0205895af31362"
          in manifest, "dispatch must pin the live preimage SHA")

    print("  b12x zero-weight micro ......... OK")


def test_glm53_b12x_tuning_controls() -> None:
    """Default-off GLM controls parse strictly and fail closed on shape drift."""
    dispatch_path = "overlay/modules/b12x_zero_weight_micro/moe_dispatch.py"
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
        "intermediate_size": 512,
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

    wrapper_path = "overlay/modules/glm53_b12x_out/b12x_moe.py"
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py")).read()
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
        "glm53_sm121_mla_prefill" in profile
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
        "overlay/modules/glm53_model_wiring/glm53_prefill_fastpath.py",
        {
            "_GLM53_PREFILL_KG_WEIGHT",
            "_GLM53_FUSED_K_GATE_ENV",
            "_GLM53_SM121_MLA_PREFILL_ENV",
            "_glm53_fused_k_gate_enabled",
            "_glm53_cache_only_indexer_contract",
            "_glm53_cache_only_indexer_forward",
            "_glm53_fused_indexer_forward",
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
        "overlay/modules/glm53_model_wiring/glm53_prefill_fastpath.py"
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
        _overlay_source("overlay/modules/glm53_model_wiring/glm5next_model.py"),
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
    check(
        "prepare_glm53_prefill_fastpath(self)" in load_source
        and load_source.index("loaded = loader.load_weights(weights)")
        < load_source.index("prepare_glm53_prefill_fastpath(self)"),
        "fused K+gate buffer is built only after outer checkpoint loading",
    )
    module_dir = os.path.join(
        REPO, "overlay", "modules", "glm53_model_wiring"
    )
    manifest = open(os.path.join(module_dir, "manifest.tsv"), encoding="utf-8").read()
    requires = open(os.path.join(module_dir, "requires"), encoding="utf-8").read()
    check("glm53_prefill_fastpath.py\t" in manifest and "\tabsent" in manifest,
          "prefill fastpath ships as an image-new overlay")
    check("glm53_kpool_tail_select" in requires,
          "model wiring declares its shared-gate dependency")
    print("  GLM53 cache-only prefill indexer . OK")


def test_glm53_kda_prefill_regime() -> None:
    """The KDA cache split stays two-bucket, exact-gated and core-six only."""
    module_rel = "overlay/modules/glm53_kda_prefill_regime/kda.py"
    delta_rel = "overlay/modules/glm53_kda_prefill_regime/chunk_delta_h.py"
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
        "glm53_kda_prefill_regime" in profile
        and "VLLM_GLM53_KDA_PREFILL_REGIME=0" in profile,
        "GLM profile mounts the module default-off",
    )
    for name in ("kda.py", "chunk_delta_h.py"):
        source = open(
            os.path.join(REPO, "overlay", "modules", "glm53_kda_prefill_regime", name),
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
        "overlay/modules/glm53_tail_slot_persistent/glm53_kpool_indexer.py"
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
            "overlay/modules/glm53_model_wiring/glm53_prefill_fastpath.py"
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
        REPO, "overlay", "modules", "glm53_kpool_tail_select"
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
        "overlay/modules/glm53_model_wiring/glm53_union_prefill.py"
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
    check('f"/root/.osar_build_maxel{_MAXEL}"' in shim,
          "each OSAR stride must use an isolated extension build directory")
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
        wiring = open(
            os.path.join(
                REPO,
                f"overlay/modules/{profile}_oneshot_wiring/"
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py",
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py"
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py")).read()
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py",
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py")).read()
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py",
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
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py")).read()
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
    start = text.index("for _v in ")
    body = text[start:text.index("; do", start)]
    return {w for w in body.replace("\\\n", " ").split()
            if w.isupper() or w.startswith("$")}


def test_launcher_load_format_gate() -> None:
    """LOAD_FORMAT must reach the container and refuse anything unvalidated."""
    text = open("launchers/start-glm53-nvfp4-tp4.sh").read()
    check('LOAD_FORMAT="${LOAD_FORMAT:-auto}"' in text,
          "the default must stay auto -- instanttensor is opt-in")
    check("ABORT: LOAD_FORMAT must be auto, safetensors or instanttensor"
          in text,
          "an unknown format must abort, not reach vLLM as a typo")
    check("--load-format $LOAD_FORMAT" in text,
          "the value must actually reach the serve command")
    check("LOAD_FORMAT" in _launcher_caller_passthrough(text),
          "LOAD_FORMAT must be in the caller passthrough list -- a knob the "
          "launcher never forwards is the failure this lane hit five times")
    check(text.index('LOAD_FORMAT="${LOAD_FORMAT:-auto}"')
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
    check("ABORT: LOAD_FORMAT must be auto, safetensors or instanttensor" in text
          and '--load-format "${LOAD_FORMAT}"' in text
          and "-e LOAD_FORMAT=$LOAD_FORMAT" in text,
          "validated DSV4 LOAD_FORMAT must reach the container and serve CLI")
    check("silent rank death" in text,
          "instanttensor's multi-node risk must stay beside its opt-in knob")

    check("WARNING: profile not found" in text,
          "a copied launcher must not silently lose profile VLLM_* settings")
    check("_foreign_stack" in text
          and "'^(glm53|q38)(-|$)'" in text,
          "DSV4 must refuse live foreign model stacks on every node")
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
    for rel in ("overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py",
                "overlay/modules/fp8_lm_head/fp8_lm_head.py"):
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
        "overlay/modules/glm53_dflash2_fp8_head/qwen3_dflash2.py")).read()
    ns = load_defs(
        "overlay/modules/glm53_dflash2_fp8_head/qwen3_dflash2.py",
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
        "overlay/modules/glm53_dflash2_fp8_head/qwen3_dflash2.py",
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
        "overlay/modules/glm53_dflash2_fp8_head/qwen3_dflash2.py"
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

    for path in ("probes/prefill_ladder.py", "bench/bench-dec.py",
                 "bench/bench-tp4.py"):
        src = open(path, encoding="utf-8").read()
        match = re.search(r"^def _resolve_model.*?(?=\n\n\n|\n[^\s#])",
                          src, re.S | re.M)
        check(match is not None, f"{path} must define _resolve_model")
        assert match is not None
        for base_name in ("URL", "BASE"):
            if re.search(rf"^{base_name} = ", src, re.M):
                break
        # Some harnesses call urllib.request.urlopen through the package and
        # some import it locally; hand over the package so both resolve.
        ns: dict = {"os": os, "json": _json, "urllib": _urllib, "re": re,
                    base_name: "http://localhost:8000/v1/chat/completions"}
        exec(compile(match.group(0), path, "exec"), ns)
        resolve = ns["_resolve_model"]

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
        "overlay/modules/glm53_model_wiring/glm53_union_prefill.py"
    ), encoding="utf-8").read()
    check("union prefill: ARMED" in union,
          "the union path must announce its width when it installs")
    check("is not a union width" in union,
          "a value that is not 2 or 4 must warn, not silently mean off")

    ns: dict = {"os": os, "logger": _CapturingLogger()}
    load_defs("overlay/modules/glm53_model_wiring/glm53_union_prefill.py",
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
        "overlay/modules/glm53_model_wiring/glm53_prefill_fastpath.py"
    ), encoding="utf-8").read()
    check("fused K+gate: ARMED" in fused,
          "replacing Indexer.forward must be visible in the boot log")
    kpool = open(_overlay_source(
        "overlay/modules/glm53_kpool_tail_select/glm53_kpool_topk.py"
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
    text = open("launchers/start-glm53-nvfp4-tp4.sh").read()
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
        "overlay/modules/glm53_model_wiring/glm53_prefill_fastpath.py"
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
def test_glm53_megakernel_contracts() -> None:
    import math as _math

    mod = "overlay/modules/glm53_megakernel"
    ns = load_defs(
        f"{mod}/glm53_megakernel.py",
        {"_mk_pow2_scale", "_mk_pad128", "_mk_gemm_eligible",
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
          and "MK_W_NBUF * SMEM_W_ROWS * SMEM_W_PITCH" in cu
          and "cp.async.wait_group" in cu_code,
          "the W stream is a multi-buffer cp.async pipeline")
    # Depth is a tuning knob, not a contract -- but it must stay deep enough
    # to keep more than one tile in flight, and the smem it costs must fit
    # the opt-in this part actually reports (101376 B).
    nbuf = consts["MK_W_NBUF"]
    check(nbuf >= 3, f"W pipeline depth {nbuf} keeps too little in flight")
    smem = 2 * 16 * 132 + nbuf * 128 * 144 + 32 * 32 * 4
    check(smem <= 101376,
          f"W pipeline depth {nbuf} overruns the 101376 B smem opt-in")
    # The opt-in is per BLOCK; occupancy is set by the SM's shared memory,
    # which on this part is 102,400 B -- NOT the 131,072 an earlier version
    # of this check assumed. That wrong number let the check pass while
    # asserting the opposite of the truth: at nbuf=3 the block takes 63,616 B,
    # so 2 x 63,616 = 127,232 overruns 102,400 and exactly ONE block is
    # resident. (It also explains why deepening 3 -> 4 measured zero: both
    # depths were already at one block/SM, so there was no occupancy left
    # to lose.) Record the real figure rather than assert a fiction.
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
    # The fp8 W pack is TILE-major and the packer and the kernel are the only
    # two places that know it -- they must move together or the kernel reads
    # a correct-looking tensor as garbage. Row-major put the 128 rows of a
    # tile 4096 B apart, so one 16 KB tile touched 128 DRAM pages.
    check("c.wq + ((size_t)nt * kblk + kb) * (SMEM_W_ROWS * KSTEP)" in cu
          and "wsrc + (size_t)t * 16" in cu,
          "stage_w reads a tile as one contiguous run")
    check("wq.dim() == 4" in cu and "wq.size(3) == KSTEP" in cu
          and "wq.is_contiguous()" in cu,
          "run_gemm rejects a wq that is not the tile-major pack -- the "
          "shape is the only thing standing between a stale row-major "
          "tensor and silently wrong output")
    check(".permute(0, 2, 1, 3).contiguous())" in pysrc_full
          and "q.view(n_pad // 128, 128, k // 128, 128)" in pysrc_full,
          "build_mk_weight emits [n/128, k/128, 128, 128]")
    check("mk_cp_async16(d0 + (size_t)t * 16, wsrc + (size_t)t * 16);" in cu
          and "src = (torch.arange(8, device=q.device)[None, :] ^ (rows[:, None] & 7))"
          in pysrc_full and "wrow + mk_swz(nrow, koff + t4)" in cu,
          "the W8 tile copy is a straight memcpy of a PRE-SWIZZLED pack into "
          "dense 128 B rows (every thread issues copies), and the fragment "
          "loads read through the same XOR -- the padded pitch cost the copy "
          "16% of its bandwidth")
    check("smem += (MK_SMEM_ALIGN - (sb & (MK_SMEM_ALIGN - 1))) & (MK_SMEM_ALIGN - 1);"
          in cu and "constexpr int GEMM_SMEM = MK_SMEM_ALIGN + 2 * 16 * SMEM_A_PITCH +"
          in cu and "constexpr int GEMM_SMEM_W4 = MK_SMEM_ALIGN + 2 * 16 * SMEM_A_PITCH +"
          in cu,
          "the dynamic smem base must be re-aligned at runtime (the static "
          "s_last/s_unit push it to +16, which put every 128 B tile row across "
          "a bank-line boundary: dense rows measured no faster until this) and "
          "both smem budgets must carry the alignment slack")
    check("stage_w(nt0, kb00, kb00 % MK_W_NBUF, false);" in cu
          and "stage_w(nt, kb + DIST, (kb + DIST) % MK_W_NBUF, true);" in cu
          and "if (!prefilled) stage_w(nt, kb0, kb0 % MK_W_NBUF, true);" in cu,
          "only the fill hoisted above the grid barrier leaves thread 0 out of "
          "the copies (the barrier's fence would drain them); every later "
          "stage_w uses all 256 threads, 4 chunks each")
    check(bench.index("torch.mm(_SPACER[0], _SPACER[1], out=_SPACER[2])")
          < bench.index("_DRAIN.sum()") < bench.index("for t in hot:"),
          "the bench flush must drain the dirty lines with a read stream AFTER "
          "the matmul spacer (whose 8 MB output is dirty too) and before the "
          "hot touch: the old order left ~24 MB of write-back under the timed "
          "kernel (both arms ~35% slow at the first launch)")
    check("threadIdx.x != 0\n           && t < SMEM_W_ROWS * MK_W_CHUNKS; t += MK_THREADS - 1)" in cu
          and "threadIdx.x != 0 && t < SMEM_W_ROWS * 4;" in cu,
          "thread 0 issues no cp.async in the hoisted fill: it runs the grid barrier, whose "
          "__threadfence drains its own outstanding copies -- with the "
          "fill hoisted above the barrier that turned into a 6-12 us wait")
    _w8_at = cu_code.index("stage_w(nt, kb0,")
    check(_w8_at < cu_code.index("stage_a(kb0);", _w8_at),
          "W(kb0) starts flying before A(kb0) is staged (the W8 pipeline "
          "fill; the W4 branch above fills its own buffers first by "
          "construction). kb0, not 0: a block may own a k SLICE of a tile.")
    # -- the FIRST unit's W fill is hoisted above the A-quant prologue and
    #    its barrier, and the prologue's own x loads (consumed by the amax
    #    shuffle) go out before that fill. Phase stamps: quant + barrier
    #    took 3-10 us during which DRAM idled; a fill issued first instead
    #    queued the 256 B/row x loads behind 1.5 MB of W (quant 13-18 us).
    _hoist_at = cu_code.index("stage_w(nt0, kb00,")
    check(_hoist_at < cu_code.index('asm volatile("griddepcontrol.wait;"')
          < cu_code.index("raw[i] = *(const uint2*)(c.x +")
          < cu_code.index("__shfl_xor_sync(0xffffffffu, mx[i], off)")
          < cu_code.index("mk_grid_barrier(bar, c.grid);"),
          "prologue order: the first unit's W fill (independent of the "
          "previous kernel), the PDL wait, x into registers, amax/convert/"
          "store, barrier")
    check(cu_code.count('asm volatile("griddepcontrol.launch_dependents;");') == 3
          and "cudaLaunchAttributeProgrammaticStreamSerialization" in cu
          and 'getenv("VLLM_GLM53_MK_PDL")' in cu
          and "cudaLaunchKernelEx(&cfg, kernel, args)" in cu,
          "gemm, kda and mhc kernels trigger their dependents at entry and "
          "are launched programmatically behind the MK_PDL knob (default off)")
    check("const bool prefilled = hoisted && (u == (int)blockIdx.x);" in cu
          and "if (!prefilled) stage_w(nt, kb0, kb0 % MK_W_NBUF, true);" in cu,
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
          and "for (int kb = kbq + c.grid; kb < kblk; kb += c.grid)" in cu
          and cu.index("g_mk_aq + ((size_t)kb * 32 + r) * KSTEP + ql * 4")
              < cu.index("auto stage_a"),
          "A is quantized once, cooperatively across the grid, before the "
          "tile loop reads it")
    check(cu.index("mk_grid_barrier(bar, c.grid);")
          < cu.index("auto stage_a"),
          "a barrier publishes the shared A quant before any block stages "
          "it -- without it a block reads a tile another block has not "
          "written yet, and only sometimes")
    check("sxs[i] = g_mk_axs[i];" in cu,
          "the per-row scales come from the same shared quant")
    # -- remainder split-K: leftovers of the last partial round take k
    #    slices instead of leaving grid - rem blocks idle for a whole
    #    tile-time. ksr == 1 must leave the original path untouched.
    check("const int rem = nblk % c.grid;" in cu
          and "int ksr = (rem > 0) ? (grid / rem) : 1;" in cu
          and "int mk_choose_ksr(int m, int n, int k, int grid)" in cu
          and cu.count("mk_choose_ksr(") == 5
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
    check("const bool split = (ksr > 1)" in cu,
          "ksr == 1 must fall through to the single-pass path unchanged")
    # The accumulator no longer follows from ksr * rem <= grid, so the
    # size guard has to be on the accumulator itself.
    check("(size_t)c.m * pcols * ksr <= MK_SPLIT_ELEMS" in cu
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
          and "for (int u = blockIdx.x; u < units; u = next_unit())" in cu
          and "s_unit = c.grid + (int)atomicAdd(&g_mk_unit_next, 1u);" in cu,
          "dynamic unit hand-out: first unit static (hoisted fill), the "
          "rest from a counter re-armed ahead of the publish barrier")
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
    # -- W4 streams its raw records through cp.async like the W8 tiles:
    #    tile-major pack (one contiguous 8 KB + 1 KB record per (tile,
    #    k-block)), one commit group per stage, the exact-wait formula,
    #    and a smem budget that stays under the sm_121 opt-in.
    check("wq4.dim() == 4" in cu and "wq4.size(3) == 64" in cu
          and "ws4.size(3) == 8" in cu
          and ".permute(0, 2, 1, 3).contiguous())" in pysrc_full
          and "wq4.view(n_pad // 128, 128, k // 128, 64)" in pysrc_full,
          "the W4 pack is tile-major and the host refuses any other shape")
    check("W4_RAW_NBUF * W4_RAW_BYTES" in cu
          and "static_assert(GEMM_SMEM <= 101376 && GEMM_SMEM_W4 <= 101376" in cu
          and "mk_cp_wait_upto(min(RAW_DIST - 1, kbn - kb - 2));" in cu,
          "W4 raw staging is budgeted in GEMM_SMEM and waits for exactly "
          "the record it needs")
    # -- W4 pack: exact e2m1 -> e4m3 expansion
    check("mk_e2m1_to_e4m3[8]" in cu_code
          and "0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C" in cu,
          "the e2m1->e4m3 LUT covers {0,.5,1,1.5,2,3,4,6} exactly")
    # the expansion reads the packed immediate, not constant memory, and
    # the immediate must be the same table byte for byte
    _lut64 = int("0x4C4844403C383000", 16)
    check([(_lut64 >> (8 * c)) & 0xFF for c in range(8)]
          == [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C]
          and "0x4C484440'3C383000ULL" in cu
          and "__vcmpeq4(mag, 0x01010101u)" in cu_code
          and "__byte_perm(out2[0], out2[1], 0x5140)" in cu_code
          and "mk_e2m1_to_e4m3[lo & 7]" not in cu_code,
          "the W4 expansion is byte-lane SIMD on the LUT's closed form -- "
          "never a __constant__ load (it serialises over the warp's "
          "distinct codes) and never per-nibble scalar chains")
    check("mk_gemm_kernel<true>" in cu_code and "run_gemm_w4" in cu
          and "GEMM_SMEM_W4" in cu and "g_mk_gemm4_bar" in cu,
          "the W4 path is its own kernel instantiation with its own smem "
          "budget and ticket counter; the W8 kernel keeps the 63,616 B "
          "budget it had before the W4 pipeline")
    check("(int8_t)((sc4 >> (8 * g4)) & 0xFFu) << 3)" in cu_code
          and "(code & 0x08080808u) << 4" in cu_code
          and "uint8_t nb[32]" not in cu_code and "uint8_t ob[64]" not in cu_code,
          "expansion is exponent-field add + sign in registers, never a "
          "float multiply and never a local-memory byte array")
    check(pysrc_full.count("_selftest_w4") >= 1
          and "1e-5" in pysrc_full and "0.15" in pysrc_full,
          "the W4 self-test gates bit-exact expansion and by-design error")
    check('".in_proj_qkvbfg_a" not in name' in open(
        os.path.join(REPO, "overlay/modules/glm53_fp8_dense/"
                            "glm53_fp8_dense.py"), encoding="utf-8").read(),
          "the W4 attach skips the KDA in_proj unless the knob is 'all'")
    # All four launches are persistent grids that resolve their own size
    # from the device. mhc has its own ceiling because it takes no dynamic
    # smem; gemm and kda share MK_GRID_CAP but resolve separately, since
    # kda carries more state and need not fit as often.
    # gemm and kda are persistent grids too, and their occupancy differs
    # from each other's, so each resolves its own -- clamped, like mhc, so a
    # grid that does not fit degrades instead of refusing to launch.
    check("mk_resident_grid(mk_gemm_kernel<false>, g_gemm_grid, GEMM_SMEM)" in cu
          and "mk_resident_grid(mk_gemm_kernel<true>, g_gemm4_grid, GEMM_SMEM_W4)"
          in cu
          and "mk_resident_grid(mk_kda_kernel, g_kda_grid, GEMM_SMEM)" in cu
          and "if (cache > MK_GRID_CAP) cache = MK_GRID_CAP;" in cu,
          "gemm, gemm_w4 and kda each resolve their persistent grid from "
          "the device rather than assuming MK_GRID_CAP")
    # Distinct grids must not share a ticket counter -- the same trap the
    # mhc split fixed. kda inlines mk_gemm_phase on ITS grid.
    check("g_mk_kda_bar" in cu
          and cu.count("mk_gemm_phase<W4>(c, smem, W4 ? &g_mk_gemm4_bar : &g_mk_gemm_bar)") == 1
          and cu.count("mk_gemm_phase<false>(c, smem, &g_mk_kda_bar)") == 2,
          "kda's inlined gemm phases use their own barrier counter")
    check(cu.count("mk_launch(mk_mhc_kernel, mhc_grid, 0, stream, a);") == 1
          and "if (mhc_grid > MK_MHC_GRID_CAP)" in cu,
          "mhc launches its own grid, clamped to what the device reports "
          "resident: a hard constant plus an assert would turn future "
          "register drift into a refusal to boot")
    check(cu.count("cudaOccupancyMaxActiveBlocksPerMultiprocessor") == 2,
          "both persistent grids check residency before launching: a grid "
          "that does not fit deadlocks on the grid barrier, it does not "
          "merely run slowly")
    # Two grid sizes must never share a ticket counter. The barrier computes
    # (t / grid + 1) * grid, so it is only correct when the counter is
    # grid-aligned at launch; a 48-block kernel leaves it 48 past a multiple
    # of 96, and the next 96-block launch then releases at half its blocks.
    check('ws["barrier_mhc"].data_ptr()' in pysrc_full
          and pysrc_full.count("_barrier_ptr(ws)")
          - pysrc_full.count("def _barrier_ptr(ws)") == 1,
          "mhc runs its own barrier counter, not the one the 48-block "
          "kernels share")
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
    check(cu.count("slot * KDA_QKV * a.conv_width") >= 1
          and "sbase + i" in cu and "sbase + acc + i" in cu,
          "conv state slot stride is KDA_QKV*conv_width, computed once as "
          "sbase and used by every read and write "
          "(spec allocates a wider window; runtime width is the stride)")
    check("a.conv_width - (CONV_W - 1) + i" in cu,
          "the convolution's pos<0 history is the NEWEST end of the state "
          "buffer -- reading the front took the oldest entries once the "
          "width grew past CONV_W - 1")
    check("for (int i = 0; i < a.conv_width; ++i)" in cu
          and "(i < keep) ? kept[i] : hist(i - keep)" in cu,
          "the state update writes the WHOLE window: causal_conv1d_update "
          "keeps conv_width - nq old values starting at `acc` and appends "
          "every query token")
    check("slot * KDA_H + head) * KDA_D * KDA_D" in cu,
          "recurrent state slot stride is H*D*D")

    # -- driver-side guards from the same review
    check("SLOT = 1" in pysrc_full
          and "torch.full((1, 8), self.SLOT" in pysrc_full,
          "the KDA fixture addresses a NONZERO state slot")
    arm_fn = pysrc_full[pysrc_full.index("def maybe_arm"):]
    check("is_current_stream_capturing()" in arm_fn,
          "maybe_arm never compiles/self-tests inside graph capture")
    check("_kda_ensure_packs" in pysrc_full
          and "isinstance(in_m, Fp8DenseMethod)" in pysrc_full,
          "KDA packs build themselves and only against a W8A8 stock arm")

    # -- kda.py overlay keeps the stock body reachable
    kda = open(os.path.join(REPO, mod, "glm5next_kda.py"),
               encoding="utf-8").read()
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

    # -- hook placement: MK precedes ONEPASS in the mhc wrapper
    tl = open(os.path.join(REPO, "overlay/modules/glm53_mhc_tilelang/"
                                 "tilelang.py"), encoding="utf-8").read()
    check(tl.index("mhc_fused_post_pre as _mk_mhc")
          < tl.index("deneb fork: ONEPASS"),
          "the MK-MHC hook is tried before the ONEPASS experiment")

    # -- fp8_dense hook routes armed decode shapes and falls back otherwise
    fp8 = open(os.path.join(REPO, "overlay/modules/glm53_fp8_dense/"
                                  "glm53_fp8_dense.py"), encoding="utf-8").read()
    check("gemm_w8a8 as _mk_gemm" in fp8 and "maybe_arm as _mk_arm" in fp8,
          "Fp8DenseMethod.apply routes through the megakernel driver")
    check("build_mk_weight" in fp8 and "build_mk_weight_w4" in fp8
          and "ENABLE_W4" in fp8,
          "the build attaches the MK pack (fp8 or W4) next to the deepgemm "
          "pair")
    print("  glm53 megakernel contracts .. OK")


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

    launcher = open(os.path.join(REPO, "launchers/start-glm53-nvfp4-tp4.sh"),
                    encoding="utf-8").read()
    check('PREFILL_WARMUP:-0' in launcher,
          "the warmup hook stays opt-in at the launcher")
    profile = open(os.path.join(REPO, "profiles/glm53.env"),
                   encoding="utf-8").read()
    check("PREFILL_WARMUP=0" in profile
          and 'PREFILL_WARMUP_LENS="2048,4096,8192"' in profile,
          "the profile carries the knob, default off, ladder lengths")
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
    m = re.search(r"mk_e2m1_to_e4m3\[8\] = \{([^}]*)\};", cu, re.S)
    check(m is not None, "the e2m1->e4m3 LUT is present in the .cu")
    lut = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]{2}", m.group(1))]
    check(lut == [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C],
          f"LUT decodes the e2m1 grid in order (got {[hex(x) for x in lut]})")

    def e4m3_value(byte: int) -> float:
        sign = -1.0 if byte & 0x80 else 1.0
        expf = (byte >> 3) & 0xF
        man = byte & 0x7
        if expf == 0:
            return sign * (man / 8.0) * 2.0 ** -6  # denormal region
        return sign * (1.0 + man / 8.0) * 2.0 ** (expf - 7)

    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for code in range(16):
        for sexp in (-5, -2, 0, 3, 6):
            # mirror the kernel's ternary EXACTLY: a zero-magnitude nibble
            # takes no exponent add (it would wrap the byte); only its sign
            sig = 0x80 if code & 8 else 0
            byte = sig if (code & 7) == 0 else                 (lut[code & 7] + (sexp << 3)) | sig
            want = grid[code & 7] * 2.0 ** sexp * (-1.0 if code & 8 else 1.0)
            got = e4m3_value(byte)
            check(abs(got - want) <= 1e-9 * max(1.0, abs(want)),
                  f"LUT+exp-add is exact: code={code} s={sexp} "
                  f"{got} == {want}")

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
    sexps_a = [[rng.randrange(-5, 7) for _ in range(k // 16)]
               for _ in range(n)]
    wq4, ws4 = mod.build_mk_weight_w4(craft(lambda r, g: codes_a[r][g],
                                            lambda r, g: sexps_a[r][g]))
    check(tuple(wq4.shape) == (1, 1, 128, k // 2),
          "wq4 is tile-major [n_pad/128, k/128, 128, 64]")
    check(tuple(ws4.shape) == (1, 1, 128, k // 16),
          "ws4 is tile-major [n_pad/128, k/128, 128, 8]")
    q, sc = wq4.tolist(), ws4.tolist()

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
            check(sexp_at(r, kk) == sexps_a[r][kk // 16],
                  "byte-exact tier: scale index follows the 16-group")

    # (b) VALUE-EXACT tier: fully random codes/scales. The quantizer may
    # renormalize a group to its own (s', code') -- legal, the grid is
    # closed under x2 up to 6 -- but the dequantized VALUES must return
    # exactly, unpacked with the KERNEL's index math and the .cu LUT.
    codes_b = [[rng.randrange(16) for _ in range(k // 16)] for _ in range(n)]
    sexps_b = [[rng.randrange(-5, 7) for _ in range(k // 16)]
               for _ in range(n)]
    wb = craft(lambda r, g: codes_b[r][g], lambda r, g: sexps_b[r][g])
    wq4, ws4 = mod.build_mk_weight_w4(wb)
    q, sc = wq4.tolist(), ws4.tolist()
    for r in range(n):
        for kk in range(k):
            byte = nib(r, kk)
            code = (byte & 0xF) if kk % 2 == 0 else (byte >> 4)
            val = grid[code & 7] * 2.0 ** sexp_at(r, kk)                 * (-1.0 if code & 8 else 1.0)
            check(val == wb[r, kk].item(),
                  f"value-exact tier: elem {kk} dequantizes to the original "
                  f"(row {r}: {val} != {wb[r, kk].item()})")
    for r in range(n, 128):  # padded rows are zero nibbles
        check(all(b == 0 for b in q[0][0][r]), f"pad row {r} packed as zeros")
    print("  w4 layout roundtrip ...... OK")


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
    test_fp8_dense_bproj()
    test_mhc_smallm_knob()
    test_mhc_probe_contracts()
    test_mhc_onepass_math()
    test_mhc_smallm_split_ownership()
    test_mhc_bigfuse_knob()
    test_census_kda_group()
    test_dflash_warmup_buckets()
    test_dsv4_spec_warmup_contract()
    test_dsv4_ue8m0_host_guard()
    test_glm53_sm121_mla_prefill_gate()
    test_glm53_kpool_packed_scratch_contract()
    test_glm53_cache_only_indexer_prefill()
    test_glm53_kda_prefill_regime()
    test_glm53_upstream_prefill_batch()
    test_oneshot_sm121_grid_contract()
    test_glm53_megakernel_contracts()
    test_prefill_warmup_contracts()
    test_megakernel_w4_layout_functional()
    print(f"all OK ({PASS} checks)")
