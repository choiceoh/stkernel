"""Shares of the ST engine/kernel optimizations by model dependence, from classification.jsonl (one row per item of
collect.py; the classifier's instructions are rubric.md).

    python3 measurements/st_model_dependence_20260917/summarize.py            # -> summary.txt
    python3 measurements/st_model_dependence_20260917/summarize.py --table    # -> table.md (the per-PR list)

Scopes: U model-free, S an op every model has but tuned/compiled at GLM-5.3's cell, F one architecture family,
G GLM-5.3 only, O another model only. "Live" drops the items marked rejected.

The carry-over to Qwen3.8 and DeepSeek-V4.1 is not a judgment made here: each F/S item's feature maps to the lanes it
optimizes, and engine/kernels/cells.admission() says for that model's shape (the fixtures of
tests/test_engine_kernel_shape.py) whether the lane is admitted (as-is), unmeasured on its own kernel (re-measure),
served by the same compiled kernel through an adapter (glue), or refused / served by another kernel / absent (no).
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOPES = "USFGO"

# PRs whose point is low-bit accuracy (GPTQ, smoothing) more than speed or memory: the rubric counted them (a GPTQ pack
# enables a low-bit lane); the sensitivity row drops them.
ACCURACY_ONLY = {650, 659, 661, 669}
# U items whose benefit lands only in one family's kernels (a b12x compile cache, constants built at DSA sites, prefix
# snapshots carrying KDA state): the sensitivity row moves them to F.
BORDERLINE_U = {981, 983, 580, 540, 590, 549, 553}

FEATURE_LANES = {
    "moe": ("moe",), "boot_load_compile": ("moe",),       # the F boot items are NVFP4 expert preshard / b12x compile caches
    "kda_linear": ("kda_recurrent", "kda_ring", "kda_chunk"), "indexer": ("indexer",), "mla_dsa": ("mla",),
    "mhc": ("mhc_decode", "mhc_prefill"), "drafter": ("draft",), "dense_gemm": ("dense",),
    "comm": ("oneshot", "prefill_collectives"),
}
CARRY = ("as-is", "re-measure", "glue", "no")


def load():
    return [json.loads(l) for l in (HERE / "classification.jsonl").read_text().splitlines() if l.strip()]


def pct(n, d):
    return f"{n} ({100 * n / d:.1f}%)" if d else "0"


def shares(rows, label):
    c = collections.Counter(r["scope"] for r in rows)
    print(f"  {label:<44} n={len(rows):3d}  " + "  ".join(f"{s} {pct(c.get(s, 0), len(rows))}" for s in SCOPES))


def lane_carry(verdicts: dict, lane: str) -> str:
    v = verdicts.get(lane)
    if v is None:
        return "no"                                        # the model has no such layer
    if v.status == "admitted":
        return "as-is"
    tier = v.serve.tier if v.serve else None
    if tier == "glue":
        return "glue"
    if v.status == "unmeasured" and tier == "specialized":
        return "re-measure"
    return "no"                                            # refused: another kernel, a generic one, or nothing fast


def carry(r, verdicts, home=False):
    if r["scope"] == "U" or (home and r["scope"] in "SG"):
        return "as-is"
    if r["scope"] == "G":
        return "no"
    lanes = FEATURE_LANES.get(r["feature"])
    if not lanes:
        return "re-measure" if r["scope"] == "S" else "no"   # a generic knob (chunk size, width) chosen at GLM's cell
    return max((lane_carry(verdicts, lane) for lane in lanes), key=CARRY.index)


def admissions():
    """GLM-5.3 is kernel_shape.MEASURED (its derived shape plus the DFlash2 drafter boot binds) -- the sanity row every
    lane admits; the other two are the test fixtures' shapes read off the fleet's config.json files."""
    sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
    import test_engine_kernel_shape as fixtures
    from engine.base.kernel_shape import MEASURED
    from engine.kernels import cells
    out = {}
    for model, shape in (("glm53", lambda: MEASURED), ("qwen38", fixtures.qwen_shape), ("dsv41", fixtures.dsv41_shape)):
        verdicts = cells.admission(shape())
        out[model] = ({v.lane: v for v in verdicts}, cells.counts(verdicts))
    return out


def model_free_files(r):
    args = (["diff", "--name-only", f"{r['hash']}^1", r["hash"]] if r["kind"] == "merge"
            else ["show", "--name-only", "--format=", r["hash"]])
    names = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.split()
    return [n for n in names if n.startswith("engine/") and n.endswith((".py", ".cu", ".cpp", ".h"))]


MODEL_FREE = ("engine/base/", "engine/kernels/common/", "engine/kernels/bounded_graph/", "engine/kernels/decode_queue/",
              "engine/kernels/mapped_staging/", "engine/kernels/native_root.py", "engine/kernels/token_embedding.py",
              "engine/runtime/")

LOC_BUCKETS = (
    ("model-free (common, bounded_graph, decode_queue, mapped_staging, native_root, token_embedding)",
     ("common", "bounded_graph", "decode_queue", "mapped_staging", "native_root.py", "token_embedding.py")),
    ("dense GEMM (dense, w4a8_pipeline, decode_inputs, decode_projection, tile_dataflow)",
     ("dense", "w4a8_pipeline.py", "decode_inputs.py", "decode_projection.py", "tile_dataflow.py")),
    ("comm (oneshot, prefill_collectives)", ("oneshot", "prefill_collectives")),
    ("MoE (b12x, router_fp32, glm_pointwise, prefill_router, moe_output)",
     ("b12x", "router_fp32.py", "router_fp32.cpp", "glm_pointwise.py", "prefill_router.py", "moe_output.py")),
    ("MLA/DSA + indexer (mla, kpool, indexer*, decode/prefill_topk, deep_gemm)",
     ("mla", "kpool.py", "indexer.py", "indexer_gate.py", "decode_topk.py", "decode_topk.cu", "prefill_topk.py",
      "prefill_topk.cu", "deep_gemm.py")),
    ("KDA (kda, causal_conv*, state, linear_decay)",
     ("kda", "causal_conv.py", "causal_conv_ring.py", "causal_conv_single.py", "state.py", "linear_decay.py")),
    ("mHC (mhc, prefill_mhc, mhc_contract)", ("mhc", "prefill_mhc.py", "mhc_contract.py")),
    ("DFlash drafter (draft_*)", ("draft_attention.py", "draft_select.py", "draft_conv.py", "draft_observe.py",
                                  "draft_boundary.py")),
    ("Qwen3.8 only (qsa, gdn, gated_residual)", ("qsa.py", "gdn.py", "gated_residual.py")),
    ("frame (cells, arch, __init__)", ("cells.py", "arch.py", "__init__.py")),
)


def kernel_loc():
    base = ROOT / "engine" / "kernels"
    per = collections.Counter()
    for p in base.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".cu", ".cpp", ".h", ".c"):
            per[p.relative_to(base).parts[0]] += sum(1 for _ in p.open(errors="ignore"))
    total = sum(per.values())
    seen = set()
    print(f"\n== engine/kernels lines by bucket (total {total}; b12x includes vendored CuTe-DSL sources)")
    for label, names in LOC_BUCKETS:
        n = sum(per.get(x, 0) for x in names)
        seen.update(names)
        print(f"  {pct(n, total):>16}  {label}")
    rest = {k: v for k, v in per.items() if k not in seen}
    if rest:
        print(f"  unbucketed: {rest}")


def summary(rows):
    opt = [r for r in rows if r["opt"] == "yes"]
    live = [r for r in opt if r["status"] != "rejected"]
    kinds = collections.Counter(r["kind"] for r in rows)
    print(f"items: {len(rows)} first-parent main commits touching engine/ ({kinds['squash']} squash PRs, {kinds['merge']} "
          f"merged PRs, {kinds['direct']} direct), {rows[-1]['date']} .. {rows[0]['date']}, newest {rows[0]['hash']}")
    print(f"optimizations: {len(opt)}; status " + ", ".join(f"{k} {v}" for k, v in
                                                          collections.Counter(r['status'] for r in opt).most_common()))

    print("\n== scope shares")
    shares(opt, "all merged optimizations")
    shares(live, "live (rejected dropped)")
    shares([r for r in opt if r["status"] == "default"], "default-on only")
    no_acc = [r for r in live if r["pr"] not in ACCURACY_ONLY]
    shares(no_acc, "live, accuracy-only GPTQ dropped")
    flipped = [dict(r, scope="F") if r["pr"] in BORDERLINE_U and r["scope"] == "U" else r for r in live]
    shares(flipped, "live, borderline U moved to F")
    shares([r for r in flipped if r["pr"] not in ACCURACY_ONLY], "live, both")

    print("\n== live: feature x scope")
    fx = collections.Counter((r["feature"], r["scope"]) for r in live)
    for f in sorted({f for f, _ in fx}, key=lambda f: (-sum(v for (g, _), v in fx.items() if g == f), f)):
        print(f"  {f:<18} " + "  ".join(f"{s} {fx.get((f, s), 0):3d}" for s in SCOPES))

    print("\n== live: hardware-bound (GB10/SM121/TP4 RoCE) and evidence, within each scope")
    for s in SCOPES:
        sub = [r for r in live if r["scope"] == s]
        if sub:
            hw = sum(bool(r["hw"]) for r in sub)
            ev = collections.Counter(r["measured"] for r in sub)
            print(f"  {s}: n={len(sub):3d}  hw-bound {pct(hw, len(sub))}  evidence " +
                  ", ".join(f"{k} {v}" for k, v in ev.most_common()))
    ev = collections.Counter(r["measured"] for r in live)
    print(f"  all: evidence " + ", ".join(f"{k} {pct(v, len(live))}" for k, v in ev.most_common()))

    print("\n== carry-over: which live GLM-era optimizations serve another model's shape (O items excluded)")
    adm = admissions()
    base = [r for r in live if r["scope"] != "O"]
    for model, (verdicts, counts) in adm.items():
        lanes = ", ".join(f"{v.lane} {v.status}" + (f"/{v.serve.tier}" if v.serve else "") for v in verdicts.values())
        print(f"  {model} lanes: {counts}\n    {lanes}")
    for model, (verdicts, _) in adm.items():
        home = model == "glm53"
        c = collections.Counter(carry(r, verdicts, home) for r in base)
        cum, parts = 0, []
        for k in CARRY[:-1]:
            cum += c[k]
            parts.append(f"{k} {c[k]} (cum {100 * cum / len(base):.1f}%)")
        print(f"  {model}: n={len(base)}  " + "  ".join(parts) + f"  no {pct(c['no'], len(base))}")
        by = collections.Counter((r["feature"] if r["scope"] in "SF" else r["scope"], carry(r, verdicts, home))
                                 for r in base if carry(r, verdicts, home) != "as-is")
        print("    not as-is: " + ", ".join(f"{f}->{k} {v}" for (f, k), v in sorted(by.items())))

    print("\n== where the live U items landed (engine code files each changed)")
    where = collections.Counter()
    glm_only = []
    for r in (r for r in live if r["scope"] == "U"):
        files = model_free_files(r)
        free = any(f.startswith(MODEL_FREE) for f in files)
        glm = any(f.startswith("engine/profiles/glm53/") for f in files)
        k = ("model-free layers only" if free and not glm else "model-free layers + GLM profile" if free
             else "GLM profile, no model-free layer" if glm else "other kernels only")
        where[k] += 1
        if k == "GLM profile, no model-free layer":
            glm_only.append(f"#{r['pr']}")
    for k, v in where.most_common():
        print(f"  {k}: {v}")
    print(f"  GLM profile only: {' '.join(glm_only)}")

    kernel_loc()


def table(rows):
    opt = [r for r in rows if r["opt"] == "yes"]          # rejected ones included: the status column says so
    names = {"U": "U — 모델 무관", "S": "S — 일반 연산, GLM 셀에서 튜닝", "F": "F — 아키텍처 계열 전용",
             "G": "G — GLM-5.3 전용", "O": "O — 다른 모델 전용"}
    out = ["# PR별 분류 — ST 최적화의 모델 의존도 (2026-09-17)", "",
           "`summarize.py --table` 이 `classification.jsonl` 에서 만든다. 손으로 고치지 않는다.",
           "상태: default 기본 경로 · optin 기본 꺼짐/실험 · rejected 기각·되돌림. hw: GB10/SM121/TP4 전용 여부. "
           "근거: e2e 종단 실측 · component 커널/프로브·발사 수 · none 수치 없음.", ""]
    for s in SCOPES:
        sub = sorted((r for r in opt if r["scope"] == s), key=lambda r: (r["feature"], -(r["pr"] or 0)))
        if not sub:
            continue
        out += [f"## {names[s]} ({len(sub)})", "", "| PR | 요지 | 상태 | 특징 | hw | 근거 | 이유 |", "|---|---|---|---|---|---|---|"]
        for r in sub:
            pr = f"#{r['pr']}" if r["pr"] else f"`{r['hash']}`"
            mixed = f" (+{','.join(r['mixed'])})" if r["mixed"] else ""
            cell = lambda t: str(t).replace("|", "\\|")
            out.append(f"| {pr} | {cell(r['gist'])} | {r['status']} | {r['feature']}{mixed} | "
                       f"{'yes' if r['hw'] else 'no'} | {r['measured']} | {cell(r['why'])} |")
        out.append("")
    (HERE / "table.md").write_text("\n".join(out))
    print(f"wrote {HERE / 'table.md'} ({len(opt)} optimizations)")


if __name__ == "__main__":
    rows = load()
    table(rows) if "--table" in sys.argv else summary(rows)
