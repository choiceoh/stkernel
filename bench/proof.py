#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Armed is not serving: prove the arm's lanes from the head log.

For every knob the serving container carries at a non-default value (or every
knob you name), look for its serving marker (bench/proof-markers.tsv) in the
head log with a FIXED-STRING search, and say PASS / ---- / no-marker. The
result rides in the onepass record (`proof`, `proof_ok`) so a measurement can
never again be filed without knowing whether its lanes ran.

    python3 bench/proof.py                       # knobs from the container, log = head log
    python3 bench/proof.py --log boot-FUS7.log   # a preserved per-arm log
    python3 bench/proof.py --knobs VLLM_GLM53_KDA_ONEPASS,VLLM_GLM53_MK_SMLP2
    python3 bench/proof.py --json                # machine form

The head log is /glmlogs/glm53.log inside the head container -- the server's
stdout, rewritten by every boot (bench/ab-lever.sh keeps a per-arm copy as
boot-<NAME>.log). Serving markers appear only AFTER traffic, so check after
the leg, not after health.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKERS = os.path.join(HERE, "proof-markers.tsv")
HEAD_LOG = os.environ.get("MK_HEAD_LOG", "/home/choiceoh/glm53-logs/glm53.log")


def _json_marker_receipts(log: str, prefix: str) -> list[dict] | None:
    """Decode complete literal JSON records; duplicates/nonfinite values fail."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")

    def finite_float(value):
        number = float(value)
        if not -float("inf") < number < float("inf"):
            raise ValueError("nonfinite JSON number")
        return number

    records = []
    for line in log.splitlines():
        if prefix not in line:
            continue
        try:
            record = json.loads(line.split(prefix, 1)[1].strip(),
                                object_pairs_hook=pairs, parse_constant=invalid_constant,
                                parse_float=finite_float)
            if not isinstance(record, dict):
                return None
        except (ValueError, TypeError):
            return None
        records.append(record)
    return records


def _ep_tiled_case_proof(case):
    if (not isinstance(case, dict) or case.get("verdict") != "PASS"
            or case.get("phase") != "complete"
            or any(key in case for key in ("error", "cleanup_error", "diagnostic_error", "first_failure_rows"))):
        return False
    graph = case.get("case") in {"mixed6", "balanced12", "concentrated24", "zeros32", "remote33", "balanced8192",
                                 "mixed4", "balanced8", "concentrated16"}
    if case.get("graph_replay") is not graph:
        return False
    labels = ("C1-eager", "C2-graph-current" if graph else "C2-eager",
              "C3-graph-side" if graph else "C3-side")
    phases = [phase + "-" + label for phase in ("initial", "changed") for label in labels]
    items = case.get("candidate")
    if not isinstance(items, list) or len(items) != 6:
        return False
    metrics = ("max_row_relative_l2", "max_row_relative_abs",
               "stock_max_row_relative_l2", "stock_max_row_relative_abs")
    return all(isinstance(item, dict) and item.get("phase") == phase
               and type(item.get("bad_rows")) is int and item["bad_rows"] == 0
               and all(type(item.get(key)) in (int, float) and item[key] >= 0 for key in metrics)
               and not any(key in item for key in ("error", "cleanup_error", "diagnostic_error"))
               for item, phase in zip(items, phases))


def _ep_tiled_sf6_proof(record):
    before, after = record.get("packed_before"), record.get("packed_after")
    if (record.get("actual_packed_owner") is not True or not isinstance(before, dict)
            or before != after or before.get("raw_sources_retained") is not True
            or before.get("raw_release_acceptance") is not False):
        return False
    planes = before.get("planes")
    if not isinstance(planes, dict) or set(planes) != {"fc1", "fc2"}:
        return False
    for name, blocks in (("fc1", 512), ("fc2", 256)):
        plane = planes[name]
        if (not isinstance(plane, dict) or plane.get("shape") != [72, blocks, 1552]
                or plane.get("dtype") != "torch.uint8"
                or type(plane.get("data_ptr")) is not int or plane["data_ptr"] <= 0
                or re.fullmatch(r"[0-9a-f]{64}", str(plane.get("sha256"))) is None):
            return False
    return planes["fc1"]["data_ptr"] != planes["fc2"]["data_ptr"]


def _ep_tiled_sf6_release_proof(log):
    records = _json_marker_receipts(log, "[ep-tiled-sf6] FINALIZED ")
    if not records:
        return False
    for record in records:
        layers = record.get("layers")
        if (record.get("format") != "sf6_v1" or record.get("packed_only") is not True
                or type(layers) is not int or layers <= 0
                or any(key.endswith("error") for key in record)):
            return False
        expected = dict(raw_bytes_released=layers*72*768*2048,
                        packed_bytes=layers*72*768*1552,
                        storage_bytes_saved=layers*72*768*(2048-1552))
        if any(type(record.get(key)) is not int or record[key] != value
               for key, value in expected.items()):
            return False
    return True


def _startup_proof(knob: str, log: str) -> bool | None:
    """Composite execution evidence; armed or partial progress is insufficient."""
    if knob == "VLLM_GLM53_EP_DECODE_OPT":
        if _startup_proof("VLLM_GLM53_EP_TILED", log) is not True:
            return False
        records = _json_marker_receipts(log, "[ep-tiled-selftest] PASS ")
        try:
            for record in records:
                native = {}
                for case in record["cases"]:
                    rows = case["rows"]
                    if type(rows) is not int:
                        return False
                    if rows > 32:
                        continue
                    if rows not in (4, 6, 8, 12, 16, 24, 32) or rows in native:
                        return False
                    native[rows] = True
                    evidence = case["cache_evidence"]
                    if evidence.get("decode_opt") is not (rows <= 8):
                        return False
                    expected = ("glm53_ep_static_tiled_fp32_v1", rows, 256, 48,
                        "torch.int32", False, True,
                        (16, 128, 256) if rows <= 8 else (32, 64, 512),
                        (16, 256, 128) if rows <= 8 else (32, 128, 128),
                        "nvfp4", "sf6_v1", "swigluoai_uninterleave", 1., 0., 10.,
                        "bf16_scatter" if rows <= 8 else "fp32_scatter")
                    if rows <= 8:
                        expected += ("glm53_ep_static_sf6_a_ring_v1",
                            "glm53_ep_static_sf6_word_unpack_v1",
                            "glm53_ep_static_bf16_scatter_v1")
                    expected += ("glm53_ep_static_fused_route_v1", 288, "torch.int32", 0)
                    if rows <= 8:
                        expected += ("glm53_ep_static_sf6_q1_register_max_v5",)
                    keys = evidence["keys"]
                    if type(keys) is not list or len(keys) != 1 or not isinstance(keys[0], str):
                        return False
                    actual = ast.literal_eval(keys[0])
                    if type(actual) is not tuple or actual != expected:
                        return False
                if set(native) != {4, 6, 8, 12, 16, 24, 32}:
                    return False
            return bool(records)
        except (ValueError, KeyError, TypeError, SyntaxError, AttributeError):
            return False
    if knob == "VLLM_GLM53_EP_TILED":
        if "[ep-tiled-selftest] FAIL" in log or not _ep_tiled_sf6_release_proof(log):
            return False
        for lane, low, high in (("decode", 1, 32), ("prefill", 33, 16384)):
            prefix = "[ep-tiled] LAUNCHED " + lane + " E72/H4096/I2048/top8 T="
            values = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                      if prefix in line]
            if not values or any(re.fullmatch(r"[0-9]+", value) is None
                                 or not low <= int(value) <= high for value in values):
                return False
        records = _json_marker_receipts(log, "[ep-tiled-selftest] PASS ")
        expected = {"mixed6", "balanced12", "concentrated24", "zeros32", "remote33",
                    "balanced2128", "balanced4096", "concentrated6912", "balanced8192",
                    "mixed4", "balanced8", "concentrated16"}
        return bool(records) and all(
            record.get("verdict") == "PASS" and record.get("phase") == "complete"
            and record.get("caller_preserved") is True
            and record.get("actual_weight_owner") is True
            and record.get("geometry") == dict(E=72, K=4096, I=2048, top8=8)
            and _ep_tiled_sf6_proof(record)
            and not any(key in record for key in ("error", "cleanup_error"))
            and isinstance(record.get("cases"), list)
            and len(record["cases"]) == len(expected)
            and all(isinstance(case, dict) and isinstance(case.get("case"), str)
                    for case in record["cases"])
            and {case.get("case") for case in record["cases"] if isinstance(case, dict)} == expected
            and all(_ep_tiled_case_proof(case) for case in record["cases"])
            for record in records)
    if knob == "VLLM_GLM53_STARTUP_TRIM":
        records = _json_marker_receipts(log, "[glm53-startup-trim] ")
        if records is None or len(records) != 1:
            return False
        record = records[0]
        try:
            if (type(record.get("schema")) is not int or record["schema"] != 1
                    or record.get("verdict") != "COMPLETE"
                    or type(record.get("rank")) is not int or record["rank"] != 0
                    or record.get("measurement_errors") != []
                    or any(key in ("error", "failed_stage") or key.endswith("_error") for key in record)):
                return False
            stages = record["stages"]
            if ([item["stage"] for item in stages] !=
                    ["synchronize", "gc_collect", "empty_cache", "malloc_trim"]
                    or any(item["status"] != "COMPLETE"
                           or any(key == "error" or key.endswith("_error") for key in item)
                           for item in stages)
                    or type(stages[1]["collected"]) is not int or stages[1]["collected"] < 0
                    or type(stages[3]["returned"]) is not int or stages[3]["returned"] not in (0, 1)):
                return False
            for phase in ("before", "after"):
                if set(record[phase]) != {"allocated", "reserved", "mem_available", "vm_rss"}:
                    return False
                if any(type(value) is not int or value < 0 for value in record[phase].values()):
                    return False
            started, completed = record["started_at"], record["completed_at"]
            return (type(started) in (int, float) and type(completed) in (int, float)
                    and 0 <= started <= completed < float("inf"))
        except (KeyError, TypeError, ValueError):
            return False
    if knob == "VLLM_GLM53_TP_SF6_Q0":
        prefix = "[tp-sf6-q0] LAUNCHED E288/H4096/I512/top8 T="
        if "[tp-sf6-q0-selftest] FAIL" in log:
            return False
        launches = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                    if prefix in line]
        if not launches or any(re.fullmatch(r"[0-9]+", value) is None
                               or not 4096 <= int(value) <= 8192 for value in launches):
            return False
        records = _json_marker_receipts(log, "[tp-sf6-q0-selftest] PASS ")
        return bool(records) and all(record.get("verdict") == "PASS"
                                     and record.get("phase") == "complete"
                                     and not any(key in record for key in ("error", "cleanup_error"))
                                     for record in records)
    if knob == "VLLM_B12X_EP_ZERO_WEIGHT_MICRO":
        prefix = "b12x EP zero-weight micro: "
        lines = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                 if prefix in line]
        if not lines:
            return False
        for line in lines:
            match = re.fullmatch(
                r"([0-9]+) tokens -> ([0-9]+) top-k=8 calls "
                r"\(8 tokens / 64 routed pairs each; padded tail=([01])\)", line)
            if match is None:
                return False
            tokens, calls, padded = map(int, match.groups())
            if not (1 <= tokens <= 80 and calls == (tokens + 7) // 8
                    and padded == int(tokens % 8 != 0)):
                return False
        return True
    if knob == "VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE":
        skipped = ("[glm53-graph-profile] skipped unused estimate rank=0; "
                   "model/MM profile and real graph warmup retained")
        position = log.find(skipped)
        return (position >= 0 and "Profiling CUDA graph memory" not in log
                and re.search(r"Graph capturing finished in [0-9]+ secs, took "
                              r"[0-9]+(?:\.[0-9]+)? GiB", log[position + len(skipped):]) is not None)
    if knob == "VLLM_B12X_EP_WARM_COMPACT":
        prefix = "[b12x EP compact warmup]"
        if any(prefix + " " + state in log for state in ("FAILED", "INCOMPLETE")):
            return False
        lines = [line.split(prefix, 1)[1].strip() for line in log.splitlines()
                 if prefix + " COMPLETE" in line]
        if not lines:
            return False
        for line in lines:
            match = re.fullmatch(
                r"COMPLETE device=cuda:[0-9]+ launch_rows=([0-9]+) specializations=([0-9]+) "
                r"static=([0-9]+) dynamic=([0-9]+) required=([0-9]+) ready=([0-9]+) "
                r"representatives=([0-9]+(?:,[0-9]+)*)", line)
            if match is None:
                return False
            rows, total, static, dynamic, required, ready = map(int, match.groups()[:6])
            representatives = list(map(int, match[7].split(",")))
            if not (rows >= total > 0 and total == static + dynamic == required == ready
                    and len(representatives) == len(set(representatives)) == total
                    and min(representatives) > 0):
                return False
        return True
    return None


def markers(path: str = MARKERS) -> dict[str, tuple[str, str]]:
    """{knob: (marker, src)} -- comments and blank lines skipped."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else parts[1])
    return out


def served_knobs(repo: str | None = None) -> dict[str, str]:
    """The container's non-default VLLM_GLM53_* knobs (same rule as onepass)."""
    sys.path.insert(0, HERE)
    try:
        from onepass import _served_build  # type: ignore
    except Exception:
        return {}
    repo = repo or os.path.dirname(HERE)
    return (_served_build(repo) or {}).get("knobs") or {}


def _spec_counters(text):
    """Keep one actual metric series and each acceptance position, not label sums."""
    import math
    pattern = re.compile(r'(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)_total)'
                         r'(?:\{(.*)\})?\s+([^\s]+)\s*')
    label = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:[^"\\]|\\.)*")(?:,|$)')
    scalars, positions, groups = {}, {}, set()
    for line in text.splitlines():
        if not line.startswith('vllm:spec_decode_num_'):
            continue
        if re.match(r'vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)_created(?:\{|\s)', line):
            continue  # Prometheus Counter metadata is not a token counter.
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError('unsupported speculative counter')
        labels, cursor = {}, 0
        raw_labels = match[2] or ''
        while cursor < len(raw_labels):
            item = label.match(raw_labels, cursor)
            if item is None or item[1] in labels:
                raise ValueError('ambiguous speculative labels')
            labels[item[1]] = json.loads(item[2])
            cursor = item.end()
        value = float(match[3])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError('invalid speculative counter')
        name = match[1].removeprefix('vllm:spec_decode_num_').removesuffix('_total')
        if name == 'accepted_tokens_per_pos':
            position = labels.pop('position', None)
            if not isinstance(position, str) or re.fullmatch(r'0|[1-9][0-9]*', position) is None:
                raise ValueError('missing acceptance position')
            target, key = positions, int(position)
        else:
            target, key = scalars, name
        if key in target:
            raise ValueError('multiple speculative metric series')
        target[key] = int(value)
        groups.add(tuple(sorted(labels.items())))
    if set(scalars) != {'drafts', 'draft_tokens', 'accepted_tokens'} or len(groups) != 1:
        raise ValueError('missing or mixed speculative counters')
    return scalars, positions, next(iter(groups))


def spec_k_evidence(context):
    """Prove K from the same boot's actual argv and whole-onepass counter delta.

    This is not fixed-request acceptance or a performance verdict. A log marker
    or an echoed environment setting cannot supply the context.
    """
    import hashlib
    result = dict(verdict='REJECTED', scope='whole onepass metrics window; not fixed-request accepted counts')
    try:
        if not isinstance(context, dict) or context.get('exclusive') is not True:
            raise ValueError('exclusive onepass evidence missing')
        expected = context['expected_k']
        if not isinstance(expected, str) or re.fullmatch(r'[1-7]', expected) is None:
            raise ValueError('invalid declared speculative K')
        k = int(expected)
        before, after = context['launch_before'], context['launch_after']
        if (not isinstance(before, dict) or before != after or before.get('boot_id') != context['boot_id']
                or not isinstance(before.get('boot_id'), str)
                or re.fullmatch(r'[0-9a-f]{64}\|[^|]+', before['boot_id']) is None
                or before.get('method') != 'dflash' or type(before.get('node_rank')) is not int
                or before['node_rank'] != 0
                or re.fullmatch(r'sha256:[0-9a-f]{64}', str(before.get('image'))) is None
                or type(before.get('num_speculative_tokens')) is not int
                or before['num_speculative_tokens'] != k or before.get('environment_spec_k') != expected
                or any(re.fullmatch(r'[0-9a-f]{64}', str(before.get(key))) is None
                       for key in ('command_sha256', 'config_sha256'))):
            raise ValueError('actual speculative launch or boot differs')
        a, pa, ga = _spec_counters(context['metrics_before'])
        b, pb, gb = _spec_counters(context['metrics_after'])
        if ga != gb or set(pa) != set(pb) or set(pa) != set(range(k)):
            raise ValueError('speculative series or position count differs')
        delta = {key:b[key]-a[key] for key in a}
        positions = [pb[i]-pa[i] for i in range(k)]
        if (any(v < 0 for v in (*delta.values(), *positions)) or delta['drafts'] <= 0
                or not 0 < delta['draft_tokens'] <= k*delta['drafts']
                or not 0 < delta['accepted_tokens'] <= delta['draft_tokens']
                or sum(positions) != delta['accepted_tokens']
                or positions[0] > delta['drafts'] or positions[-1] <= 0
                or any(right > left for left, right in zip(positions, positions[1:]))):
            raise ValueError('speculative counters reset, idle, or disagree with K')
        result.update(verdict='PASS', num_speculative_tokens=k, launch=before,
                      counter_delta=delta, accepted_tokens_per_position_delta=positions,
                      highest_position_observed=True,
                      metrics_before_sha256=hashlib.sha256(context['metrics_before'].encode()).hexdigest(),
                      metrics_after_sha256=hashlib.sha256(context['metrics_after'].encode()).hexdigest(),
                      metric_series_sha256=hashlib.sha256(json.dumps(ga).encode()).hexdigest())
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        # Only local fixed messages are recorded, never metric payloads or argv.
        result['reason'] = type(exc).__name__
    return result


def check(knobs: list[str], log_path: str, table: dict[str, tuple[str, str]] | None = None,
          *, speculation=None, preparation=None) -> dict:
    table = table or markers()
    try:
        with open(log_path, "rb") as fh:
            log = fh.read().decode("utf-8", "replace")
    except Exception:
        log = ""
    res = {}
    spec = None
    prep = None
    decode_opt = None
    for k in knobs:
        if k == 'VLLM_GLM53_EP_DECODE_OPT':
            from glm53_prep_proof import evidence
            decode_opt = evidence(preparation, log_path, decode_opt=True)
            res[k] = (_startup_proof(k, log) is True and decode_opt['verdict'] == 'PASS')
            continue
        if k == 'VLLM_GLM53_PREP_FUSED':
            from glm53_prep_proof import evidence
            prep = evidence(preparation, log_path)
            res[k] = prep['verdict'] == 'PASS'
            continue
        if k == 'VLLM_GLM53_SPEC_K':
            spec = spec_k_evidence(speculation)
            res[k] = spec['verdict'] == 'PASS'
            continue
        startup = _startup_proof(k, log)
        if startup is not None:
            res[k] = startup
            continue
        if k not in table:
            res[k] = None                      # no marker known: cannot judge
            continue
        res[k] = table[k][0] in log            # fixed string, never a regex
    judged = [v for v in res.values() if v is not None]
    result = {"proof": res, "proof_ok": f"{sum(judged)}/{len(judged)}",
              "log": log_path, "log_bytes": len(log)}
    if spec is not None:
        result['speculation'] = spec
    if prep is not None:
        result['preparation'] = prep
    if decode_opt is not None:
        result['decode_optimization'] = decode_opt
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=HEAD_LOG)
    ap.add_argument("--knobs", default=None, help="comma list; default: the container's non-default knobs")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.knobs:
        knobs = [k.strip() for k in a.knobs.split(",") if k.strip()]
    else:
        served = served_knobs()
        knobs = [k for k, v in served.items() if v not in ("0", "", "off")]
    r = check(knobs, a.log)
    if a.json:
        print(json.dumps(r))
        return 0
    if not knobs:
        print("proof: no non-default knob to prove (defaults arm)")
        return 0
    for k, v in r["proof"].items():
        tag = "PASS" if v else ("----" if v is False else "no-marker")
        print(f"  {tag:<9} {k}")
    print(f"  -> {r['proof_ok']} lanes proved serving ({r['log_bytes']} bytes of {r['log']})")
    return 0 if all(value is True for value in r["proof"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
