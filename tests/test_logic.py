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
import glob
import os
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
        args = re.search(r"^SERVE_ARGS=", src, re.M)
        assert args, f"{name}: no SERVE_ARGS to size"
        assert call < args.start(), (
            f"{name}: preflight runs after SERVE_ARGS -- the measured GMU "
            "cannot reach --gpu-memory-utilization"
        )
        reclaim = src.find("docker rm -f")
        assert reclaim != -1 and reclaim < call, (
            f"{name}: preflight measures before stale containers are removed"
        )
        assert "GMU=$GMU_SAFE" in src, f"{name}: preflight warns but never applies"
        PASS += 1
    print("  preflight applies GMU ......... OK")


if __name__ == "__main__":
    test_skip_topk()
    test_prefill_chunker()
    test_sp_ranges()
    test_helpers()
    test_profile_step()
    test_bench_dec_metrics()
    test_overlay_manifest()
    test_overlay_symbol_contracts()
    test_profile_env_carried()
    test_profile_keys_have_readers()
    test_dspark_speed_guards()
    test_bf16_release_guards()
    test_acceptance_lever_guards()
    test_ngram_ceiling_sim()
    test_launcher_head_guard()
    test_preflight_precedes_serve_args()
    print(f"all OK ({PASS} checks)")
