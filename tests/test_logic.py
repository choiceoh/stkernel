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
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)} | {
        t.id
        for n in body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
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
    launcher = os.path.join(REPO, "launchers", "start-hy4-tp4.sh")
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
            ["bash", launcher],
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
    launcher = os.path.join(REPO, "launchers", "start-hy4-tp4.sh")
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
            ["bash", launcher], env=env, text=True, capture_output=True,
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
    launcher = os.path.join(REPO, "launchers", "start-glm53-nvfp4-tp4.sh")
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
# EP fixed pair plan: the decode path that drops the dummy expert entirely.
# ---------------------------------------------------------------------------
def test_ep_fixed_pair_plan() -> None:
    ns = load_defs(
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py",
        {
            "B12X_EP_FIXED_MICRO_MAX_PAIRS",
            "read_b12x_ep_bool",
            "read_b12x_ep_exact_bool",
            "b12x_ep_mode_from_env",
            "require_b12x_ep_micro_limit",
            "b12x_ep_fixed_slice_limit",
            "b12x_ep_fixed_pair_plan",
        },
        {"os": os},
    )
    plan = ns["b12x_ep_fixed_pair_plan"]
    slice_limit = ns["b12x_ep_fixed_slice_limit"]
    read_bool = ns["read_b12x_ep_bool"]
    mode_from_env = ns["b12x_ep_mode_from_env"]
    require_micro = ns["require_b12x_ep_micro_limit"]

    check(mode_from_env({}.get) == (True, False, False),
          "EP mode defaults to fixed no-dummy with micro enabled")
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
        }.get) == (False, True, False),
        "plain-static diagnostic is allowed only after disabling no-dummy",
    )

    check(ns["B12X_EP_FIXED_MICRO_MAX_PAIRS"] == 8,
          "fixed EP boundary must match FlashInfer _MICRO_MAX_TOKENS")
    check(require_micro(8) == 8 and require_micro("12") == 12,
          "fixed EP setup accepts a verified micro boundary of at least eight")
    for bad_limit in (None, "unknown", 0, 7):
        try:
            require_micro(bad_limit)
            check(False, f"micro boundary {bad_limit!r} must fail closed")
        except RuntimeError as exc:
            check("cannot verify" in str(exc) or "requires FlashInfer" in str(exc),
                  f"micro boundary {bad_limit!r} reports the contract failure")
    check(
        [slice_limit(n) for n in (2048, 8, 4, 0)] == [8, 8, 4, 1],
          "fixed EP calls must honor both micro and workspace bounds")
    for concurrency, verify_tokens, launches in (
        (1, 8, 8),
        (2, 16, 16),
        (4, 32, 32),
    ):
        pairs = verify_tokens * 8
        limit = slice_limit(32)
        spans = [
            (lo, min(lo + limit, pairs))
            for lo in range(0, pairs, limit)
        ]
        check(len(spans) == launches,
              f"C={concurrency}: {verify_tokens}x8={pairs} pairs must use "
              f"{launches} micro calls per layer")
        check(spans[0][0] == 0 and spans[-1][1] == pairs,
              f"micro slices must cover all {pairs} pairs")
        check(all(0 < hi - lo <= 8 for lo, hi in spans),
              f"no {pairs}-pair slice may enter static")
    odd_spans = [
        (lo, min(lo + slice_limit(32), 17))
        for lo in range(0, 17, slice_limit(32))
    ]
    check(len(odd_spans) == 3 and odd_spans[-1] == (16, 17),
          "non-verification shapes must still be fully micro-sliced")

    for label, flags in (
        ("typical", [True] * 64 + [False] * 192),
        ("local last", [False] * 192 + [True] * 64),
        ("all local", [True] * 256),
        ("one local", [True] + [False] * 255),
        ("interleaved", [i % 4 == 0 for i in range(256)]),
        ("mixed final slice", [True] * 10 + [False] * 22),
    ):
        src, keep = plan(flags, len(flags))
        local = [i for i, f in enumerate(flags) if f]
        n = len(local)
        check(len(src) == len(flags) and len(keep) == len(flags),
              f"plan must keep the input length ({label})")
        check(sum(keep) == n, f"exactly the local pairs are kept ({label})")
        check(src[:n] == local,
              f"every real local pair survives, in order ({label})")
        check(all(not k for k in keep[n:]),
              f"padding carries weight 0 ({label})")
        check(all(i in local for i in src),
              f"padding may only repeat pairs already present ({label})")
        if len(flags) > n > 1:
            check(len(set(src[n:])) > 1,
                  f"padding spreads instead of piling on one pair ({label})")
        for lo in range(0, len(flags), 8):
            hi = min(lo + 8, len(flags))
            real_src = {src[i] for i in range(lo, hi) if keep[i]}
            padding_src = {src[i] for i in range(lo, hi) if not keep[i]}
            if real_src:
                check(padding_src <= real_src,
                      f"mixed micro slice may only repeat its own rows ({label})")

    # no local pairs: shape held, nothing kept, no index out of range
    src, keep = plan([False] * 8, 8)
    check(len(src) == 8 and not any(keep) and set(src) == {0},
          "an empty local set must still yield a valid fixed-length plan")

    # the plan never names an expert the caller did not already route to --
    # that is what removes the dummy expert's 12 MiB/layer weight read
    flags = [i % 3 == 0 for i in range(60)]
    src, _ = plan(flags, 60)
    check(set(src) <= {i for i, f in enumerate(flags) if f},
          "plan must never introduce a slot outside the local set")

    source = open(
        _overlay_source(
            "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py"
        ),
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)
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
    micro_setup_source = ast.get_source_segment(source, micro_setup)
    init_source = ast.get_source_segment(source, init)
    process_source = ast.get_source_segment(source, process)
    apply_source = ast.get_source_segment(source, apply)
    assert all(part is not None for part in (
        fixed_source, micro_setup_source, init_source, process_source, apply_source
    ))
    assert fixed_source is not None
    assert micro_setup_source is not None
    assert init_source is not None
    assert process_source is not None
    assert apply_source is not None
    check(
        "require_b12x_ep_micro_limit(prior)" in micro_setup_source,
        "no-dummy setup must verify the live FlashInfer micro boundary",
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
    ensure_wrapper = apply_source.index("self._ensure_wrapper()")
    capacity_probe = apply_source.index(
        "self._ep_capacity_probe(wrapper, topk_ids)"
    )
    check(
        compact_return < ensure_wrapper and fixed_return < ensure_wrapper,
        "both direct EP paths must return before the large wrapper is built",
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
        "limit = b12x_ep_fixed_slice_limit(self.max_num_tokens)"
        in fixed_source
        and "for lo in range(0, pairs, limit):" in fixed_source,
        "runtime fixed path must use the tested micro slice limit",
    )
    check(
        "local_in_slice" in fixed_source
        and "padding_src = torch.where" in fixed_source,
        "runtime padding must be planned inside each micro slice",
    )
    check(
        "launch_sm120_moe(" in fixed_source
        and "top_k=1" in fixed_source
        and "scatter_output=pair_out[lo:hi]" in fixed_source
        and "_workspace=self._ep_fixed_workspace" in fixed_source,
        "fixed path must pin each top_k=1 slice to its graph workspace",
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

    print("  EP fixed pair plan ............. OK")


def test_b12x_zero_weight_micro() -> None:
    """Default-off E=72 sentinel skip: exact gate, cache, and hot-path order."""
    wrapper_path = (
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py"
    )
    wrapper_names = {
        "read_b12x_ep_bool",
        "read_b12x_ep_exact_bool",
        "b12x_ep_mode_from_env",
        "B12X_EP_ZERO_WEIGHT_MICRO_TOKEN_COUNTS",
        "B12X_EP_ZERO_WEIGHT_MICRO_CHUNK_TOKENS",
        "B12X_EP_ZERO_WEIGHT_MICRO_TOPK",
        "B12X_EP_ZERO_WEIGHT_MICRO_EXPERTS",
        "b12x_ep_zero_weight_micro_chunks",
    }
    wns = load_defs(wrapper_path, wrapper_names, {"os": os})
    exact_bool = wns["read_b12x_ep_exact_bool"]
    mode = wns["b12x_ep_mode_from_env"]
    chunks = wns["b12x_ep_zero_weight_micro_chunks"]

    check(mode({}.get) == (True, False, False),
          "zero-weight micro must be off by default")
    check(mode({"VLLM_B12X_EP_ZERO_WEIGHT_MICRO": "1"}.get)
          == (True, False, True), "exact 1 arms the local-only experiment")
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
    for args in (
        (1, 8, 72, True), (7, 8, 72, True), (9, 8, 72, True),
        (24, 8, 72, True), (8, 1, 72, True), (8, 8, 71, True),
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
    for banned in ("pair_x", "pair_out", "index_add_", "argsort", "nonzero"):
        check(banned not in pure_source,
              f"pure top-k=8 lane must not retain {banned} overhead")
    zero_return = apply_source.index("return self._apply_ep_zero_weight_micro(")
    fixed_return = apply_source.index("return self._apply_ep_fixed(")
    ensure_wrapper = apply_source.index("self._ensure_wrapper()")
    check(zero_return < fixed_return < ensure_wrapper,
          "exact lane must return before fixed fallback and large wrapper")
    check("VLLM_B12X_EP_ZERO_WEIGHT_MICRO" not in apply_source,
          "apply must never re-read the experiment environment")
    check("top_k=1" in process_source
          and "max_rows=B12X_EP_FIXED_MICRO_MAX_PAIRS" in process_source
          and "top_k=B12X_EP_ZERO_WEIGHT_MICRO_TOPK" in process_source
          and "max_rows=B12X_EP_ZERO_WEIGHT_MICRO_MAX_ROWS" in process_source,
          "setup must pin distinct top-k=1 and top-k=8 workspaces")

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



def test_ep_fixed_pair_out_initialised() -> None:
    src = open(_overlay_source(
        "overlay/modules/b12x_shared_workspace/flashinfer_b12x_moe.py")).read()
    body = src[src.index("def _apply_ep_fixed"):src.index("def _apply_ep_compact")]
    check("pair_out = torch.zeros(" in body,
          "fixed decode must allocate pair_out zeroed -- three quarters of its "
          "rows carry weight 0, the kernel may skip them, and an uninitialised "
          "bit pattern can be NaN")
    check("torch.empty(" not in body,
          "no uninitialised buffer may reach index_add_ in the fixed path")
    check(body.index("pair_out.mul_(") < body.index("output.index_add_("),
          "padding rows must be masked BEFORE the sum, in case the kernel "
          "wrote them without applying token_final_scales")
    check("keep.unsqueeze(1)" in body, "the mask must be the keep flags")
    print("  EP fixed pair_out init ......... OK")
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
    # The Python size gate and the CUDA #define have to stay in lockstep: a
    # tensor the shim admits but the kernel refuses is a TORCH_CHECK abort, and
    # a MAXEL the peers disagree on puts every rx write at the wrong stride.
    # The constant is overridable now (VLLM_DSV4_OSAR_MAXEL) because the
    # built-in 131072 = 32 tokens at hidden 4096 is GLM's captured verify, not
    # this stack's 192 -- so both sides must read the SAME override, and the
    # default must remain the exact constant that has always shipped.
    cu = open(
        os.path.join(
            REPO, "overlay/modules/tp_oneshot_ar/dsv4_oneshot_ar.cu"),
        encoding="utf-8",
    ).read()
    check(re.search(r"^_MAXEL_DEFAULT = 131072\b", shim, re.M) is not None,
          "the Python default drifted from the shipped CUDA constant")
    check(re.search(r"^#define MAXEL 131072$", cu, re.M) is not None,
          "the CUDA default drifted from the Python one")
    check("#ifndef MAXEL" in cu,
          "the CUDA constant is not overridable, so -DMAXEL would be a "
          "redefinition rather than the override the shim thinks it passes")
    check('_MAXEL = _resolve_maxel()' in shim,
          "the Python gate no longer follows the resolved override")
    check('f"-DMAXEL={_MAXEL}"' in shim,
          "the resolved MAXEL never reaches the compiler, so Python would "
          "admit tensors the kernel TORCH_CHECKs on")
    check('build_dir = f"/root/.osar_build_maxel{_MAXEL}"' in shim,
          "an overridden build shares the default build directory -- a stale "
          "object at the old MAXEL would put peer writes at the wrong stride")
    check('"[osar] source md5=%s kernels=%d maxel=%d (%s)"' in shim,
          "MAXEL is not in the boot fingerprint, so a rank-to-rank split is "
          "unreadable from the logs")
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


def _launcher_src(name: str) -> str:
    return open(os.path.join(REPO, "launchers", name), encoding="utf-8").read()


def test_custom_ops_axis_contract() -> None:
    """The fusion arm has to be spellable, and the empty string is not it.

    vLLM validates every custom_ops entry against {all, none, +op, -op}
    (config/compilation.py). CUSTOM_OPS_AXIS="" -- the value both the GLM
    profile and its launcher documented as "the experiment" -- therefore dies
    in config validation: ValueError on the GLM image, IndexError (op[0] on an
    empty string) on the older copy the DSV4 image ships. Neither reaches the
    compiler, so the fusion arm never ran, and the boot failure is
    indistinguishable from the profile's documented "barrier necessary"
    verdict -- a false negative that would have closed the axis with evidence
    pointing the wrong way.
    """
    for name in ("start-glm53-nvfp4-tp4.sh", "start-hy4-tp4.sh"):
        src = _launcher_src(name)
        check("CUSTOM_OPS_AXIS" in src, f"{name}: no custom_ops axis at all")
        # every substitution must use the checked value, never the :- default
        # that turns "unset" into an empty entry.
        check("${CUSTOM_OPS_AXIS:-}" not in src,
              f"{name}: an unvalidated ${{CUSTOM_OPS_AXIS:-}} can emit an "
              'empty custom_ops entry, which vLLM rejects')
        check("all|none|[+-]?*)" in src,
              f"{name}: CUSTOM_OPS_AXIS is not validated against the "
              "{all,none,+op,-op} vocabulary vLLM enforces")
        guard = src[src.index("all|none|[+-]?*)"):]
        check("exit 2" in guard[:400],
              f"{name}: the axis validator does not abort on a bad value")
    profile = open(os.path.join(REPO, "profiles", "glm53.env"),
                   encoding="utf-8").read()
    axis_doc = profile[profile.index("custom_ops fusion-barrier"):]
    axis_doc = axis_doc[:axis_doc.index("CUSTOM_OPS_AXIS=")]
    check('"none"' in axis_doc,
          "glm53.env documents the fusion arm but not as none")
    check('#   "" =' not in axis_doc,
          'glm53.env still offers "" as the fusion arm')
    print("  custom_ops axis contract ....... OK")


def test_dsv4_launcher_axes() -> None:
    """The GLM lane's launcher axes, ported to DSV4 without moving the default.

    Three knobs the GLM launcher grew and this one never did: an overridable
    compilation config (#88/#112), the diagnostic env passthrough (#118), and
    a documented chunked-prefill budget (#117). All three must be inert when
    unset -- this is the production launcher, and the boot it produces has to
    stay byte-identical until a bracket says otherwise.
    """
    src = _launcher_src("start-hy4-tp4.sh")

    # 1. the default compilation config still says exactly what the serve
    # script carried inline before it became overridable. This literal is the
    # production compile config; moving it is a bracket, not an edit, so the
    # test pins the string rather than deriving it from git history (which
    # stops being a baseline the moment this lands).
    inline = (
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],'
        '"pass_config":{"fuse_gemm_comms":true,"fuse_allreduce_rms":true,'
        '"fuse_rope_kvcache_cat_mla":true,"fuse_attn_quant":true}}'
    )
    now = re.search(r"^\[ -n \"\$\{COMPILE_CFG:-\}\" \] \|\| COMPILE_CFG='(.*)'$",
                    src, re.M)
    check(now is not None, "COMPILE_CFG has no single-quoted default")
    check(now.group(1) == inline,
          "COMPILE_CFG default drifted from the production compile config:"
          f"\n  want {inline}\n  got  {now.group(1)}")
    # ${COMPILE_CFG:-{...}} is the shape that cannot work: the JSON's own
    # brace closes the expansion and the leftover }} lands on the caller's
    # value. The GLM lane lost this knob to it once (#88).
    code = "\n".join(line for line in src.splitlines()
                      if not line.lstrip().startswith("#"))
    check("${COMPILE_CFG:-{" not in code,
          "COMPILE_CFG default uses brace expansion the JSON closes early")

    # 2. it has to reach the container and be read there, not re-inlined.
    check("-e COMPILE_CFG=$COMPILE_CFG" in src,
          "COMPILE_CFG never reaches the container")
    serve = src[src.index("cat > /tmp/serve-hy4.sh"):src.index("SERVEEOF\n", src.index("cat > /tmp/serve-hy4.sh") + 40)]
    check('--compilation-config "${COMPILE_CFG}"' in serve,
          "the serve script does not read the carried COMPILE_CFG")
    check('--compilation-config "{' not in serve,
          "the serve script still hardcodes a compilation config")
    # it rides an unquoted $ENVV, so a value with a space would split into
    # two docker arguments and silently truncate the config.
    check("must not contain whitespace" in src,
          "COMPILE_CFG is not checked for the whitespace that would split it")

    # 3. EXTRA_ENV: KEY=VALUE only, and it has to reach BOTH containers.
    check("EXTRA_ENV_FLAGS" in src, "no EXTRA_ENV passthrough")
    extra = src[src.index("EXTRA_ENV_FLAGS=\"\""):]
    check("exit 2" in extra[:600],
          "EXTRA_ENV accepts entries that are not KEY=VALUE")
    check('COMMON="$COMMON $EXTRA_ENV_FLAGS"' in src,
          "EXTRA_ENV never reaches the containers")
    head = src.index("docker run -d --name hy4 $COMMON")
    worker = src.index("docker run -d --name hy4-worker $COMMON")
    check(head > src.index('COMMON="$COMMON $EXTRA_ENV_FLAGS"')
          and worker > src.index('COMMON="$COMMON $EXTRA_ENV_FLAGS"'),
          "EXTRA_ENV is appended after the containers are started")

    # 4. the profile source must not eat a caller's value for any of them.
    preserve = src[src.index("for _v in IMAGE"):]
    preserve = preserve[:preserve.index("do")]
    for key in ("COMPILE_CFG", "CUSTOM_OPS_AXIS", "EXTRA_ENV",
                "MAX_NUM_BATCHED"):
        check(key in preserve,
              f"{key} is not preserved across the profile source, so a "
              "caller's A/B value would be silently replaced")

    # 5. the prefill budget: 8192 adopted 2026-09-01 without a dsv4 bracket,
    # so the grounds have to travel with it and the gap has to stay named.
    check(re.search(r'^MAX_NUM_BATCHED="\$\{MAX_NUM_BATCHED:-8192\}"$', src, re.M),
          "MAX_NUM_BATCHED default moved again -- adopting or reverting this "
          "is a ledger entry, not a silent edit")
    budget = src[:src.index('MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8192}"')][-2400:]
    for token in ("#117", "rows-per-EXPERT", "NOT ESTABLISHED",
                  "GPU KV cache size"):
        check(token in budget,
              f"the chunked-prefill rationale lost {token!r} -- this value was "
              "never measured on dsv4, and the comment is the only place that "
              "says so, along with what it costs and how to revert")
    print("  dsv4 launcher axes ............. OK")



def test_dsv4_ue8m0_host_guard() -> None:
    """The DSV4 fork of the fp8 quant block must catch what the kernel traps on.

    deepgemm packs only an fp32 scale's exponent and device-asserts that sign
    and mantissa are clear (smxx_layout.cuh:131). The assert is `asm("trap;")`,
    which destroys the CUDA context, so the failure surfaces seconds later at
    an unrelated site -- the GLM lane lost two boots to that signature before
    reading the assert. DSV4 reaches the same one-call path from four sites
    that are all armed by default (markov W2, draft lm_head, TARGET lm_head,
    attention compressor), so the condition is evaluated on the host first.

    The contract is trap-faithful, deliberately NOT the GLM module's stricter
    value test: everything the kernel traps on must raise, +inf must raise
    (it packs clean and then produces garbage, which is worse than a trap),
    and +0.0 must NOT -- an all-zero 128x128 block has no other scale, the
    kernel consumes it correctly, and rejecting it would abort a healthy boot.
    """
    try:
        import struct

        import torch
    except ImportError:
        print("  dsv4 UE8M0 host guard ......... SKIP (no torch)")
        return

    ns = load_defs(
        "overlay/modules/dspark_drafter/dspark_v2.py",
        {"_ue8m0_unsafe", "_SF_SIGN_AND_MANTISSA"},
        {"torch": torch},
    )
    unsafe = ns["_ue8m0_unsafe"]
    check(ns["_SF_SIGN_AND_MANTISSA"] == 0x807FFFFF,
          "the host mask is not the kernel's mask (smxx_layout.cuh:131)")

    def kernel_traps(value: float) -> bool:
        return (struct.unpack("<I", struct.pack("<f", value))[0] & 0x807FFFFF) != 0

    cases = {
        "+0.0": 0.0, "-0.0": -0.0, "1.0": 1.0, "2.0": 2.0, "0.5": 0.5,
        "min-normal": 2.0 ** -126, "denormal": 5e-40, "-1.0": -1.0,
        "1.5": 1.5, "3.0": 3.0, "+inf": float("inf"), "-inf": float("-inf"),
        "nan": float("nan"), "2^100": 2.0 ** 100,
    }
    flags = unsafe(torch.tensor(list(cases.values()),
                                dtype=torch.float32)).tolist()
    for (name, value), flagged in zip(cases.items(), flags):
        if kernel_traps(value):
            check(flagged,
                  f"{name} trips the deepgemm assert but the host guard "
                  "lets it through -- that is a destroyed CUDA context")
        elif value in (float("inf"), float("-inf")):
            check(flagged, f"{name} packs clean and computes garbage; "
                           "it has to be refused, not accepted")
        else:
            check(not flagged,
                  f"{name} is a scale the kernel accepts; refusing it turns "
                  "a healthy boot into an abort")
    check(not flags[0],
          "+0.0 must stay accepted -- zero-filled padding blocks produce it "
          "and 0x00000000 passes the kernel's mask")

    # The guard must be a superset of the assert on arbitrary bit patterns,
    # not just on the hand-picked ones above.
    gen = torch.Generator().manual_seed(0)
    scales = torch.randn(50000, generator=gen) * torch.exp2(
        torch.randint(-140, 40, (50000,), generator=gen).float())
    trapped = (scales.view(torch.int32) & 0x807FFFFF) != 0
    check(bool((trapped <= unsafe(scales)).all()),
          "the host guard is not a superset of the device assert")

    # And the seam has to exist at all: a single use_e8m0=True call puts the
    # trap between the requant and the layout transform, where nothing can
    # look at it.
    src = open(_overlay_source("overlay/modules/dspark_drafter/dspark_v2.py"),
               encoding="utf-8").read()
    quant = src[src.index("def _quantize_fp8_deepgemm"):]
    quant = quant[:quant.index("\ndef ", 1)]
    check("requant_weight_ue8m0_inplace" in quant,
          "the requant is not run separately, so there is no seam to inspect")
    check("use_e8m0=False" in quant,
          "the layout transform still asks for the fused requant")
    check("use_e8m0=True" in quant,
          "the ImportError fallback to the original single call is gone -- a "
          "renamed helper would take the boot with it")
    print("  dsv4 UE8M0 host guard ......... OK")



def test_launcher_parity_guards() -> None:
    """Guards the GLM launcher grew that the production one never got.

    The two launchers drove the same fleet and diverged: DSV4's guard set is
    the stronger one overall (preimage skew, image-ID drift, unmounted-source
    attestation), which is exactly why the three holes below went unnoticed --
    a count of ABORTs says DSV4 is ahead.

    1. A missing profile has to be announced. In production this launcher runs
       from a COPY outside the checkout, where the relative profile path does
       not resolve; the profile is then skipped in silence and every VLLM_*
       it declares stops reaching the container.
    2. It must refuse to start over a live bring-up stack. The GLM launcher
       already refuses to start over hy4; nothing enforced the reverse, and
       this is the one an operator starts reflexively.
    3. The cudagraph replay address assert must be reachable. vLLM has the
       check and gates it behind VLLM_LOGGING_LEVEL=DEBUG, so by default a
       graph reading a stale address is silent.
    """
    src = _launcher_src("start-hy4-tp4.sh")

    # 1 -- the warning, and it must come from a real absence test
    check("profile not found at $PROFILE_ENV" in src,
          "a missing profile is still silent; the built-in defaults take over "
          "and the profile's VLLM_* knobs never reach the container")
    check('if [ ! -f "$PROFILE_ENV" ]; then' in src,
          "the profile warning is not guarded by an absence test")
    check(src.index('if [ ! -f "$PROFILE_ENV" ]') < src.index('if [ -f "$PROFILE_ENV" ]'),
          "the warning must come before the profile is sourced")

    # 2 -- foreign-stack refusal on the head AND every worker
    check("_foreign_stack" in src, "no refusal to start over a live glm53/q38")
    check(src.count("runs glm53/q38") == 2,
          "the foreign-stack refusal must cover the head and the workers "
          f"(found {src.count('runs glm53/q38')} of 2)")
    check("^(glm53|q38)" in src,
          "the foreign-stack pattern must be anchored, or 'my-glm53-notes' "
          "would match a container name")
    # anchor on the step itself, not on prose that names it
    retire = src.index('echo "=== [1/5] retire old containers')
    for site in [m.start() for m in re.finditer(r"runs glm53/q38", src)]:
        check(site < retire,
              "the refusal must run BEFORE containers are retired -- after it, "
              "the foreign stack is still resident while KV is sized")
    check(src.count('"${DRY_RUN:-0}" != 1') >= 2,
          "the foreign-stack checks must be skipped under DRY_RUN like every "
          "other check that asks a machine about itself")

    # 3b -- the b12x MoE cutover has to be reachable and integer-only
    check('MOE_CUTOVER="${MOE_CUTOVER:-}"' in src, "no MOE_CUTOVER knob")
    check("-e FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=$MOE_CUTOVER" in src,
          "MOE_CUTOVER never reaches the container")
    moe = src[src.index('MOE_CUTOVER="${MOE_CUTOVER:-}"'):]
    check("exit 2" in moe[:500], "MOE_CUTOVER accepts a non-integer")
    check('if [ -n "$MOE_CUTOVER" ]; then\n  ENVV="$ENVV -e '
          'FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=$MOE_CUTOVER"' in src,
          "the cutover must be exported only when set -- an unconditional "
          "-e ...= passes an EMPTY value, and int('') raises inside "
          "flashinfer's _get_static_compact_cutover_pairs")
    # the arithmetic that makes this an axis at all has to stay in the file:
    # top_k=6 * 6 tokens per seq puts the static->dynamic switch at C~17.8,
    # inside the sweep that adopted MAX_NUM_SEQS=32.
    rationale = src[src.index("b12x MoE backend cutover"):
                    src.index('MOE_CUTOVER="${MOE_CUTOVER:-}"')]
    for token in ("top_k=6", "C~17.8", "864 rows", "1152 rows"):
        check(token in rationale,
              f"the cutover rationale lost {token!r} -- without the routed-row "
              "arithmetic nobody can see that the static->dynamic switch lands "
              "inside the sweep that adopted MAX_NUM_SEQS=32")

    # 3c -- MoE activation precision: the stack runs W4A4, and the knob that
    # says otherwise must be opt-in and 0/1 only.
    check('MOE_W4A16="${MOE_W4A16:-0}"' in src, "no MOE_W4A16 knob")
    check('if [ "$MOE_W4A16" = 1 ]; then\n  ENVV="$ENVV -e '
          'FLASHINFER_B12X_FORCE_MOE_W4A16=1"' in src,
          "MOE_W4A16 must arm flashinfer's own override, and only when 1 -- "
          "exporting =0 would still be an override, not the image default")
    w4a16 = src[src.index('MOE_W4A16="${MOE_W4A16:-0}"'):]
    check("exit 2" in w4a16[:300], "MOE_W4A16 accepts values other than 0/1")
    # the evidence has to travel with the knob: this contradicts a label the
    # ledger has carried since 08-09, so the next reader needs the derivation.
    prec = src[src.index("# MoE activation precision."):
               src.index('MOE_W4A16="${MOE_W4A16:-0}"')]
    for token in ("W4A4", "neither activation_precision", 'default applies'):
        check(token in prec,
              f"the W4A4 rationale lost {token!r} -- without it this reads as "
              "a preference rather than what the image actually resolves")
    check("MICRO" in src and "C=1" in src,
          "the C=1 micro-kernel boundary is not recorded where the cutover "
          "arithmetic lives")

    # 3 -- GRAPH_DEBUG reaches the container, and only as 0/1
    check('GRAPH_DEBUG="${GRAPH_DEBUG:-0}"' in src, "no GRAPH_DEBUG knob")
    check('if [ "$GRAPH_DEBUG" = 1 ]; then ENVV="$ENVV -e VLLM_LOGGING_LEVEL=DEBUG"; fi'
          in src,
          "GRAPH_DEBUG does not arm vLLM's replay address assert")
    graph = src[src.index('GRAPH_DEBUG="${GRAPH_DEBUG:-0}"'):]
    check("exit 2" in graph[:400], "GRAPH_DEBUG accepts values other than 0/1")

    # and the GLM launcher must keep refusing the reverse, or the pair is
    # only half a rule.
    glm = _launcher_src("start-glm53-nvfp4-tp4.sh")
    check("runs hy4/q38" in glm,
          "the GLM launcher stopped refusing to start over production")
    print("  launcher parity guards ......... OK")


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
    test_ep_fixed_pair_plan()
    test_ep_fixed_pair_out_initialised()
    test_b12x_zero_weight_micro()
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
    test_glm53_sm121_mla_prefill_gate()
    test_oneshot_sm121_grid_contract()
    test_custom_ops_axis_contract()
    test_dsv4_launcher_axes()
    test_dsv4_ue8m0_host_guard()
    test_launcher_parity_guards()
    print(f"all OK ({PASS} checks)")
