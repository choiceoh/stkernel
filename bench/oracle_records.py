"""One-pass evidence and arrival waves for the CPU oracle; no runtime/metric fitting here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def load_records(path):
    """Accept both the historical JSON artifact and the append-only JSONL ledger."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: expected measurement objects")
    return records


def fingerprint(record):
    # A renamed copy of the same observations is still training data.
    observations = {k: record[k] for k in ("requests", "c1_requests", "prefill", "prefill_c1",
                                           "decode", "decode_c1", "c4") if k in record}
    return hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()


def identity(record):
    return {key: record.get(key) for key in ("engine", "engine_source_sha256", "git", "image", "knobs")}


def views(record):
    """C1 and each C4 wave have different counters/validity; never borrow C1 rates for C4."""
    requests = record.get("requests") or record.get("c1_requests")
    if requests:
        yield dict(record, requests=requests, decode=record.get("decode") or record.get("decode_c1") or {},
                   prefill=record.get("prefill") or record.get("prefill_c1") or [], oracle_arm="requests")
    for index, group in enumerate(record.get("c4") or []):
        width = group.get("concurrency", 4)
        requests = [dict(q, ctx=q.get("ctx", group.get("ctx")), concurrency=width)
                    for q in group.get("requests") or []]
        yield dict(record, requests=requests, decode=group.get("decode") or {}, prefill=[],
                   valid=group.get("valid"), evidence_issues=group.get("issues") or [],
                   oracle_arm=f"c4-{index}", aggregate_output_tok_s=group.get("aggregate_output_tok_s"))


def workload(record):
    """Preserve barrier waves and actual token budgets; close the loop on simulated completion.

    Offsets describe the recorded schedule for inspection. Replaying uses wave IDs,
    so a small model error cannot accidentally overlap separate benchmark waves.
    """
    requests = record.get("requests") or []
    if not requests:
        return None
    prefill = {r.get("ctx"): r for r in record.get("prefill") or []}
    out = dict(arrive_ms=[], gens=[], prompts=[], labels=[], groups=[], cold_keys=[], prompt_sources=[])
    cursor, elapsed, wave = 0, 0.0, 0
    while cursor < len(requests):
        first = requests[cursor]
        width = first.get("concurrency", 1)
        if type(width) is not int or width < 1:
            raise ValueError("recorded concurrency must be a positive integer")
        group = requests[cursor:cursor + width]
        if len(group) != width or any(q.get("concurrency", 1) != width or q.get("ctx") != first.get("ctx")
                                      for q in group):
            raise ValueError("incomplete or mixed concurrent request wave")
        # Client IDs are not launch order when the barrier releases several threads.
        if all(type(q.get("started_monotonic")) in (int, float) and
               math.isfinite(q["started_monotonic"]) for q in group):
            group = sorted(group, key=lambda q: q["started_monotonic"])
        durations = []
        for q in group:
            ctx, gen = q.get("ctx"), q.get("completion_tokens")
            if type(ctx) is not int or ctx < 1 or type(gen) is not int or gen < 1:
                raise ValueError("replay requires context and positive observed completion_tokens")
            fallback = prefill.get(ctx, {}).get("tok") or ctx
            prompt = q.get("prompt_tokens") if positive(q.get("prompt_tokens")) else fallback
            if type(prompt) is not int or prompt < 1:
                raise ValueError("replay requires a positive integer prompt token count")
            duration = q.get("elapsed_s")
            if not positive(duration):
                duration = (q.get("ttft_s") or 0.0) + (q.get("decode_s") or 0.0)
            if not positive(duration):
                raise ValueError("replay requires a positive recorded request duration")
            durations.append(duration)
            for key, value in (("arrive_ms", round(elapsed * 1000, 1)), ("gens", gen),
                               ("prompts", prompt), ("labels", ctx), ("groups", wave),
                               ("cold_keys", fallback),
                               ("prompt_sources", "request" if positive(q.get("prompt_tokens")) else "context summary")):
                out[key].append(value)
        elapsed += max(durations)
        cursor += width
        wave += 1
    return out
