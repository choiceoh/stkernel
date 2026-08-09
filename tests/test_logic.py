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
import os
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_defs(relpath: str, names: set[str], ns: dict) -> dict:
    """exec only the named top-level defs/assigns from a source file into ns."""
    path = os.path.join(REPO, relpath)
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
    # the invariant the globalize-reuse cache and spec_enabled rely on
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
    # adjacent spans are merged so the consumer can take the zero-copy
    # single-span path: rank 0 fuses decode+small+first-shard into one span
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
    check("60.0%" in s, f"acceptance ratio wrong: {s!r}")
    check(bd.acceptance_suffix(c, c) == "", "zero delta must yield no suffix")
    check(bd.acceptance_suffix({}, {}) == "", "empty metrics must yield no suffix")
    print(f"  bench-dec acceptance parser ... OK")


# ---------------------------------------------------------------------------
# 7. overlay/manifest.tsv — the only overlay inventory
# ---------------------------------------------------------------------------
def test_overlay_manifest() -> None:
    manifest = os.path.join(REPO, "overlay", "manifest.tsv")
    source_chars = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    target_chars = source_chars | {"/"}
    rows = []
    with open(manifest, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            check(len(fields) == 3, f"manifest line {line_no}: need three TSV fields")
            source, target, base_contract = fields
            check(source == os.path.basename(source),
                  f"manifest line {line_no}: source must be a basename")
            check(source and not source.startswith(".")
                  and all(ch in source_chars for ch in source),
                  f"manifest line {line_no}: unsafe source")
            check(target.startswith(
                "/opt/venv/lib/python3.12/site-packages/vllm/"),
                f"manifest line {line_no}: target outside vllm package")
            check(all(ch in target_chars for ch in target),
                  f"manifest line {line_no}: unsafe target character")
            check(os.path.isfile(os.path.join(REPO, "overlay", source)),
                  f"manifest line {line_no}: missing overlay/{source}")
            check(
                base_contract == "absent"
                or (
                    len(base_contract) == 64
                    and all(ch in "0123456789abcdef" for ch in base_contract)
                ),
                f"manifest line {line_no}: invalid base preimage contract",
            )
            rows.append((source, target, base_contract))

    check(bool(rows), "overlay manifest must not be empty")
    check(len({source for source, _, _ in rows}) == len(rows),
          "overlay manifest has duplicate sources")
    check(len({target for _, target, _ in rows}) == len(rows),
          "overlay manifest has duplicate targets")
    manifest_sources = {source for source, _, _ in rows}
    check(
        {
            "fp8_draft_head.py",
            "dspark_v2.py",
            "dspark_speculator_v2.py",
            "dspark_utils_v2.py",
        } <= manifest_sources,
        "DSpark speed overlays are missing from the canonical manifest",
    )
    check(
        all(
            contract != "absent"
            for source, _, contract in rows
            if source in {
                "dspark_v2.py",
                "dspark_speculator_v2.py",
                "dspark_utils_v2.py",
            }
        ),
        "replaced DSpark files must pin exact production preimages",
    )

    for relpath in ("launchers/start-hy4-tp4.sh",
                    "launchers/deploy-overlays.sh"):
        text = open(os.path.join(REPO, relpath), encoding="utf-8").read()
        check("manifest.tsv" in text, f"{relpath}: manifest not consumed")
        check('OVFILES="' not in text,
              f"{relpath}: hard-coded overlay inventory remains")
        check("md5sum" not in text and "sha256sum" in text,
              f"{relpath}: manifest/files must use SHA-256 parity")
    print(f"  overlay manifest ({len(rows)} files) .... OK")


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
    check('DSPARK_IMPL="${DSPARK_IMPL:-fork}"' in launcher_text,
          "DSpark impl must default to the fork path when knobs are off")
    check("DSPARK_IMPL=upstream" in launcher_text
          and "FP8HEAD == 1 || MARKOV_TOPK > 0" in launcher_text,
          "arming a speed knob must force the upstream impl")
    check("EXPECTED_IMAGE_ID=" in launcher_text and "OVBASES" in launcher_text,
          "image/source preimage attestation is missing")
    check('[ -L "$target" ]' in launcher_text,
          "base preimage attestation must reject symlink targets")
    check('[ ! -f "$target" ]' in launcher_text,
          "hashed base preimages must be regular files")

    helper = os.path.join(REPO, "overlay", "fp8_draft_head.py")
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
        os.path.join(REPO, "overlay", "dspark_speculator_v2.py"),
        encoding="utf-8",
    ).read()
    model_text = open(
        os.path.join(REPO, "overlay", "dspark_v2.py"),
        encoding="utf-8",
    ).read()
    utils_text = open(
        os.path.join(REPO, "overlay", "dspark_utils_v2.py"),
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


if __name__ == "__main__":
    test_skip_topk()
    test_prefill_chunker()
    test_sp_ranges()
    test_helpers()
    test_profile_step()
    test_bench_dec_metrics()
    test_overlay_manifest()
    test_dspark_speed_guards()
    print(f"all OK ({PASS} checks)")
