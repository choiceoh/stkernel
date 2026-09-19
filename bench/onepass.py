#!/usr/bin/env python3
"""Canonical Korean consumer test, harness 47: C=1 2K/32K/128K; C=N 2K/32K (N = the door's admission limit, at most 4).

Every invocation prepares full prompts with bounded output, measures with the
profiler off and a unique prefix salt, then runs separate bounded GPU diagnostic
replays. Preparation changes, prefix reuse, missing evidence and external traffic
invalidate steady-state comparisons. First reasoning/content, SSE gaps, per-request
rates, actual batch widths, host/device stages and raw traces are retained under
onepass-runs/<run-id> beside the append-only output ledger, including partial runs.

C=N sends N independent requests simultaneously for each canonical question;
its aggregate output rate includes prefill and remains separate from C=1 decode.
N is the door's `max_concurrent_requests`, capped at the historical 4: GLM-5.3
admits two since #950, and four requests there decode two rows beside two
waiting ones. The record keys (`c4`, `quality_c4`) predate that; every group
and `concurrency_coverage.width` carry the width actually measured.
Run 1 also measures bench-dec's multiplier (`concurrency_fixed`): four different
2K prompts forced to exactly --fixed-concurrency-tokens (1024) output tokens, one
at a time and then N at a time, so neither answer-length spread nor the prefill
ramp enters the ratio. It is an observation and never invalidates the record.
The legacy cold_s/warm_s fields are aliases for first/median prepared fresh-prefix
TTFT, not claims about compiler or cache warmth. Harness 41 is incompatible:
42 uses seeded reasoning dossiers and visible-answer proof certificates, 43 and
44 doubled the completion budgets, and 44 asks the ko-reasoning-v2 questions.
47 clarifies the ledger decision/reservation rules and logic core's U6 scope in ko-reasoning-v3.

    python3 bench/onepass.py --name RUN [--ctx 2000,32000,128000]

Optional --fixed-decode-tokens 2048 --fixed-decode-reps 3 retains the C=1 fixed
response gate. Detailed GPU recording requires the ST latency endpoint.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.request
import uuid

from onepass_recording import CURRENT, Run, group, steady_errors
import measurement_contract as contract
import onepass_quality as quality

_RUN = None

from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_COMBINED_MAX_TOKENS = quality.COMBINED_MAX_TOKENS
DEFAULT_COMBINED_REASONING_BUDGET = quality.COMBINED_REASONING_BUDGET


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics_text(url):
    return urllib.request.urlopen(url, timeout=5).read().decode()


def _counter(text, name):
    m = re.search(r"^vllm:%s\{[^}]*\}\s+([0-9.e+]+)" % re.escape(name), text, re.M)
    return float(m.group(1)) if m else 0.0


def _st_counter(text, name):
    """The engine's own series. `_counter` above reads only vLLM's dialect."""
    m = re.search(r"^st:%s\{[^}]*\}\s+([0-9.e+]+)" % re.escape(name), text, re.M)
    return float(m.group(1)) if m else 0.0


CACHE_HIT_FRACTION = 0.5
"""Past this share of its prompt taken from the cache, a bracket did not prefill.

`first tok/s` is prompt tokens over the first TTFT, and it only means prefill throughput when the
prompt was actually computed. Three rows on 2026-09-12 were not, and nothing in the record said so:
ST-GRAPH-45c's 128K reads 46,554 tok/s at 2.76 s, and an evening run read 32K in 5.13 s and 128K in
5.22 s -- the same number for two prompts four times apart, which is the tell. They then sat in the
ledger's column beside rows that had prefilled (45차 §82). CHARTER D17 says to drop them; this makes
the record say which to drop.

Half is not a knob, it is a gap: a bracket that shares only the system preamble reuses ~1% of a
32,545-token prompt, and one that resumes a boundary reuses nearly all of it. Nothing lands in
between, so the number is recorded either way and this only decides what gets called a hit.
"""


def ask_stream(url, model, content, max_tokens, timing=None, min_tokens=0, seed=None,
               channel_trace=None, reasoning_budget=None, on_first_token=None):
    """(text, ttft_s, prompt_tokens, completion_tokens, finish_reason) of one
    streamed chat completion: ttft = first chunk carrying content."""
    body_obj = {"model": model, "max_tokens": max_tokens, "min_tokens": min_tokens,
                "seed": seed, "temperature": 0.0,
                "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": content}],
                # 39차: thinking ON, explicitly. The stock template ignored this
                # kwarg and always reasoned, so every reference (BASE39-*, DEF40, ...)
                # was measured with reasoning in the stream; the v2 template honours
                # the kwarg and thinking=false gives answers too short for the 2 s
                # decode windows (TPL1: no windows). Keep the condition constant.
                "chat_template_kwargs": {"thinking": True},
                # Harness 45: a measurement request is nobody's conversation. Kept, each one parked its
                # whole slot state and KV (0.25-0.5 GiB a rank) to the tier after it finished: on a bracket
                # boot that write overlapped the next request, and on the live door a D17 probe filled
                # production's tier with conversations no one would continue (engine: PR #858; an engine
                # older than that ignores the field and parks as before).
                "retain": False}
    if reasoning_budget is not None:
        body_obj["reasoning_budget"] = reasoning_budget
    identity = hashlib.sha256(json.dumps(body_obj).encode()).hexdigest()
    run = CURRENT.get()
    headers = {"Content-Type": "application/json"}
    if run is not None:
        body_obj['cache_salt'] = uuid.uuid4().hex
        headers['X-ST-Latency-Token'] = run.token
        timing = timing if timing is not None else {}
    body = json.dumps(body_obj).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    stream_channels = [] if channel_trace is not None or run is not None else None
    t0 = time.monotonic()
    ttft = None
    arrivals = []
    parts = []
    usage = {}
    finish = None
    first_channels = {}
    response_id = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            response_id = obj.get('id', response_id)
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    arrived = time.monotonic()
                    arrivals.append(arrived)
                    if ttft is None:
                        ttft = arrived - t0
                        if on_first_token is not None:
                            on_first_token()
                    parts.append(piece)
                    for name in ('content', 'reasoning_content', 'reasoning'):
                        if d.get(name):
                            first_channels.setdefault(name, arrived - t0)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if stream_channels is not None:
                    # Retain references after the original arrival timestamp.
                    # Classify only after all requests and sampling have ended.
                    stream_channels.append(d)
    if ttft is None:
        ttft = time.monotonic() - t0
    if timing is not None:
        ended = time.monotonic()
        elapsed = ended - t0
        ctok = int(usage.get("completion_tokens", 0) or 0)
        decode_s = elapsed - ttft
        # Standard request TPOT includes the final stream/usage tail. SSE
        # chunks can contain several speculative tokens; gaps are NOT ITL.
        timing.update(completion_tokens=ctok, prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                      min_tokens=min_tokens, max_tokens=max_tokens, seed=seed,
                      request_sha256=hashlib.sha256(body).hexdigest(),
                      output_sha256=hashlib.sha256("".join(parts).encode()).hexdigest(),
                      ttft_s=ttft, elapsed_s=elapsed,
                      decode_s=decode_s, finish_reason=finish,
                      tpot_ms=1000 * decode_s / (ctok - 1) if ctok > 1 else None,
                      decode_tok_s=(ctok - 1) / decode_s if ctok > 1 and decode_s > 0 else None,
                      chunk_gaps_ms=[1000 * (b - a) for a, b in zip(arrivals, arrivals[1:])])
        timing.update(started_monotonic=t0, ended_monotonic=ended, response_id=response_id,
                      workload_sha256=identity, first_channels_s=first_channels,
                      cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                      # Overall stop can follow a forced reasoning-end token.
                      # Keep the server's count; absent usage means unknown, not zero.
                      reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                      ttft_scope='first nonempty reasoning or content chunk',
                      chunk_gap_scope='SSE event gaps, not token ITL',
                      prefix_policy='unique salt' if run else 'server default')
        if reasoning_budget is not None:
            timing["reasoning_budget"] = reasoning_budget
    if channel_trace is not None:
        channel_trace.append(stream_channels)
    if run is not None:
        run.request(timing, ''.join(parts), stream_channels)
    return ("".join(parts), ttft, int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0), finish)


def _channel_diagnostics(events, text, finish, hits, scanner):
    """Attribute existing combined-text hits; never change the Korean gate.

    Transient deltas are not serialized. At most one 96-character combined
    context per gated kind is retained, with the exact offending channel.
    """
    names = ("content", "reasoning_content", "reasoning")
    gated_kinds = ("replacement", "lone_jamo", "cjk_mixed", "control")
    counts = {name: dict(raw_chars=0, raw_pieces=0, selected_chars=0,
                         selected_pieces=0, non_text_fields=0,
                         gated_counts={kind: 0 for kind in gated_kinds}) for name in names}
    spans, selected, offset = [], [], 0
    for delta in events:
        for name in names:
            value = delta.get(name)
            if isinstance(value, str):
                counts[name]["raw_chars"] += len(value)
                counts[name]["raw_pieces"] += bool(value)
            elif value is not None:
                counts[name]["non_text_fields"] += 1
        # Preserve the existing truthy precedence, including simultaneous keys.
        name = next((name for name in names if delta.get(name)), None)
        if name is not None:
            value = delta[name]
            row = counts[name]
            spans.append((offset, offset + len(value), name, row["selected_chars"]))
            selected.append(value)
            offset += len(value)
            row["selected_chars"] += len(value)
            row["selected_pieces"] += 1
    if "".join(selected) != text:
        raise ValueError("channel trace does not match the unchanged onepass text")

    effective = text[:-1] if finish == "length" and text.endswith("\ufffd") else text
    positions = []
    if hits.get("replacement"):
        positions.extend((i, "replacement") for i, char in enumerate(effective) if char == "\ufffd")
    if hits.get("lone_jamo"):
        # Attribute each jamo, not the preceding syllable (possibly another channel).
        positions.extend((match.end() - 1, "lone_jamo")
                         for match in scanner.WELDED_JAMO.finditer(effective))
    if hits.get("cjk_mixed"):
        glosses = [(m.start(), m.end()) for m in scanner.HANJA_GLOSS.finditer(effective)]
        for match in scanner.HAN.finditer(effective):
            if not any(a <= match.start() < b for a, b in glosses):
                positions.append((match.start(), "cjk_mixed"))
    if hits.get("control"):
        positions.extend((i, "control") for i, char in enumerate(effective)
                         if scanner.unicodedata.category(char) == "Cc" and char not in "\t\n\r")
    offenses, seen, span_index = [], set(), 0
    for position, kind in sorted(positions):
        while position >= spans[span_index][1]:
            span_index += 1
        start, end, channel, channel_start = spans[span_index]
        counts[channel]["gated_counts"][kind] += 1
        if kind not in seen:
            seen.add(kind)
            context_start = max(0, position - 40)
            offenses.append(dict(kind=kind, channel=channel, combined_offset=position,
                                 selected_channel_offset=channel_start + position - start,
                                 snippet_start=context_start,
                                 snippet=text[context_start:context_start + 96]))
    combined_counts = {kind: sum(row["gated_counts"][kind] for row in counts.values())
                       for kind in gated_kinds}
    if combined_counts != {kind: hits.get(kind, 0) for kind in gated_kinds}:
        raise ValueError("channel attribution does not match the existing combined Korean scan")
    located = {item["kind"] for item in offenses}
    return dict(schema=1, scope="diagnostic-only; existing combined-text gate",
                precedence=list(names), channels=counts, combined_chars=len(text),
                combined_gated_counts=combined_counts,
                first_offending_channel=offenses[0]["channel"] if offenses else None,
                offenses=offenses, snippet_scope="combined-selected-text",
                unlocated_kinds=[kind for kind, count in hits.items()
                                 if count and kind not in scanner.INFORMATIONAL and kind not in located])


def _st_build(names, name: str = "st-glm53") -> dict:
    """What the ST engine's container is running, read off the container itself.

    `engine_source_sha256` is the runtime manifest's hash of the engine tree the container
    mounts (the same identity measurements/st_production_*/README cites); `release` is what the
    launcher stamped (ST_RELEASE: the release directory's name, a sha12 for a release deploy-watch
    or the bracket cut); `knobs` are the STK_* the boot was given. Every failure degrades to a
    missing field: a bench must never die over its own label.
    """
    import subprocess
    out = {"engine": "st"}
    try:
        boot = subprocess.run(["docker", "inspect", "-f", "{{.Id}}|{{.State.StartedAt}}", name],
                              capture_output=True, text=True, timeout=10)
        if boot.returncode == 0 and "|" in boot.stdout.strip():
            out["boot_id"] = boot.stdout.strip()
        raw = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Env}}", name],
                             capture_output=True, text=True, timeout=10).stdout
        env = dict(e.split("=", 1) for e in json.loads(raw or "[]") if "=" in e)
        out["knobs"] = {k: v for k, v in sorted(env.items()) if k.startswith("STK_")}
        if env.get("ST_RELEASE"):
            out["release"] = env["ST_RELEASE"]
        image = subprocess.run(["docker", "inspect", "-f", "{{.Config.Image}}", name],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        if image:
            out["image"] = image
    except Exception:
        pass
    try:
        manifest = subprocess.run(["docker", "exec", name, "cat", "/opt/st/runtime-manifest.json"],
                                  capture_output=True, text=True, timeout=15).stdout
        source = json.loads(manifest or "{}").get("engine_source_sha256")
        if source:
            out["engine_source_sha256"] = source
    except Exception:
        pass
    return out


def _served_build(repo: str) -> dict:
    """What the ST serving container runs, read off the container itself.

    The engine's identity is the source the container runs (the runtime
    manifest's sha256 of the engine tree), the release the launcher stamped,
    and STK_* knobs. The retired vLLM overlay's stamp and knob parsing went
    with the overlay stack (2026-09-18). ONEPASS_ST_CONTAINER identifies a
    non-GLM serving container; the default remains st-glm53. An explicitly
    selected missing container never borrows another model's identity.
    Every failure degrades to a missing field: a bench must
    never die over its own label.
    """
    import subprocess
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True,
                               timeout=10).stdout.split()
    except Exception:
        names = []
    name = os.environ.get("ONEPASS_ST_CONTAINER", "st-glm53")
    if name in names:
        return _st_build(names, name=name)
    return {}


def build_record(args, revision):
    from measurement_contract import from_args, metadata
    return dict(name=args.name, t=time.strftime("%F %T"), git=revision, evidence_scope='full',
                prefill=[], quality={}, decode={}, korean={}, **metadata(from_args(args)))


def kda_state_storage(metrics_text: str):
    """Actual bound precision, separate from shape so an A/B may vary it."""
    for line in metrics_text.splitlines():
        if line.startswith("st:lane_info{"):
            match = re.search(r'\bkda_state_dtype="(fp32|fp16)"', line)
            if match:
                return match.group(1)
    return None


def engine_shape(completion_url: str) -> dict:
    """The served shape this run measured, stamped on the record.

    A record without it cannot say whether a slower number is a regression or a different
    engine. 2026-09-12 is the case: two onepass boots twenty-seven minutes apart on the same
    commit, and the second captured decode widths 1-8 with a 364,032 context ceiling where the
    first had 1-4 and 1,035,264 -- `MAX_SEQS = 8` is the repo default and production pins 4.
    Capture went 32.3 s to 71.9 s and the boot 163.5 s to 208.7 s, and nothing in the record
    said so, so the two numbers looked comparable and were not.

    Read from the door, best-effort: an engine that predates these fields contributes what it
    has and the rest stays absent rather than guessed.
    """
    base = completion_url.split("/v1/")[0].rstrip("/")
    out = {}
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            card = (json.loads(r.read()).get("data") or [{}])[0]
    except Exception:                                   # noqa: BLE001 -- a shape we could not read is not a failed run
        card = {}
    caps = card.get("capabilities") or {}
    for key, value in (("model", card.get("id")), ("max_model_len", card.get("max_model_len")),
                       ("max_concurrent_requests", caps.get("max_concurrent_requests")),
                       ("speculative_tokens", caps.get("speculative_tokens")),
                       ("prefix_cache", caps.get("prefix_cache")),
                       ("conversation_tier", caps.get("conversation_tier"))):
        if value is not None:
            out[key] = value
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            status = json.loads(r.read())
        for key in ("engine", "parked"):
            if status.get(key) is not None:
                out.setdefault(key, status[key])
    except Exception:                                   # noqa: BLE001
        pass
    return out


def workload_requests(args, cq):
    items = []
    for ctx in map(int, args.ctx.split(',')):
        cases = quality.cases(args.seed + ctx)
        combined = bool(args.combine_min_ctx) and ctx >= args.combine_min_ctx
        if combined:
            items.append(quality.request_item(ctx, args.seed + ctx, cases, cq.filler,
                args.combined_max_tokens, args.combined_reasoning_budget, 'all'))
        else:
            items.extend(quality.request_item(ctx, args.seed + ctx, [case], cq.filler,
                args.max_tokens, args.max_tokens // 2, i) for i, case in enumerate(cases))
    return items


def diagnostic_request(item, num_spec):
    """Keep the original prompt, with enough output for four profiled steps.

    Diagnostic requests are not graded or timed as consumer measurements.
    Eight maximum-width steps leave room for prefill's first output and for
    all four clients to enter decode. A fixed floor prevents EOS from ending
    this evidence replay before the recorder has its four steps.
    """
    tokens = max(64, 8 * (max(0, math.ceil(num_spec)) + 1))
    return dict(item, max_tokens=tokens, min_tokens=tokens, reasoning_budget=tokens // 2)


def preparation_request(item, num_spec):
    """Warm the unchanged input with bounded output, outside quality scoring.

    Model boot captures the declared decode ladder. Preparation need not
    solve the long reasoning task a second time. Any specialization or
    capture that remains during scoring still fails steady_errors; this
    shortened preparation is not itself evidence of steady state.
    """
    return diagnostic_request(item, num_spec)


FIXED_CONCURRENCY_CLIENTS = 4
DEFAULT_FIXED_CONCURRENCY_TOKENS = 1024
MAX_CONCURRENCY = 4


def serving_concurrency(shape):
    """The C>1 arm's width: the door's own admission limit, capped at the historical four.

    #950 made GLM-5.3 admit two resident requests. Four requests then decode two rows beside two waiting
    ones: the arm measures the queue, and steady_errors refuses every such block ("actual decode width 2
    != 4"). A door that reports no limit keeps four, the width every record before it measured."""
    width = (shape or {}).get('max_concurrent_requests')
    if type(width) is not int or width < 2:
        return MAX_CONCURRENCY
    return min(MAX_CONCURRENCY, width)


def fixed_concurrency_items(seed, cq, tokens):
    """bench-dec's C=N question on this harness (operator, 2026-09-14): four DIFFERENT 2K prompts, each
    forced to exactly `tokens` output tokens. The canonical C=N rate carries the answers' length spread
    (the longest answer finishes alone) and the serialized prefill ramp; equal lengths remove both, so the
    four N at a time against the same four one at a time is the engine's concurrency multiplier."""
    items = []
    for client in range(FIXED_CONCURRENCY_CLIENTS):
        case_seed = seed + 4000 + client
        item = quality.request_item(2000, case_seed, quality.cases(case_seed), cq.filler,
                                    tokens, tokens // 2, f'fixed-concurrency-{client}')
        item['min_tokens'] = tokens
        items.append(item)
    return items


def fixed_concurrency_groups(items, width):
    """The four prompts in releases of `width`: one release of four at C=4, two of two at C=2."""
    if width < 2 or len(items) % width:
        raise ValueError(f'{len(items)} fixed prompts do not split into releases of {width}')
    return [items[i:i + width] for i in range(0, len(items), width)]


def fixed_concurrency_summary(tokens, c1, releases, width):
    """`c1`: the four requests sent one at a time; `releases`: `group`'s results for the same four, `width`
    at a time. bench-dec's rate is output tokens over wall time from first start to last completion (a 2K
    prefill is a small share), summed over the releases; the decode rate sums each request's own rate after
    its first token, per release."""
    issues = []
    requests = [r for release in releases for r in release['requests']]
    if (len(c1) != FIXED_CONCURRENCY_CLIENTS or len(requests) != FIXED_CONCURRENCY_CLIENTS
            or any(len(release['requests']) != width for release in releases)):
        issues.append(f'fixed concurrency needs {FIXED_CONCURRENCY_CLIENTS} requests at C=1 and in releases of {width}')
    lengths = sorted({r.get('completion_tokens') for r in list(c1) + requests}, key=str)
    if lengths != [tokens]:
        issues.append(f'fixed concurrency output lengths {lengths} != [{tokens}]')
    seconds = sum(r.get('elapsed_s') or 0. for r in c1)
    c1_rate = sum(r.get('completion_tokens') or 0 for r in c1) / seconds if seconds > 0 else None
    c1_decode = [r['decode_tok_s'] for r in c1 if r.get('decode_tok_s')]
    c1_decode = median(c1_decode) if c1_decode else None
    rates = [release.get('aggregate_output_tok_s') for release in releases]
    walls = [sum(r.get('completion_tokens') or 0 for r in release['requests']) / rate
             for release, rate in zip(releases, rates) if rate]
    many_rate = (sum(r.get('completion_tokens') or 0 for r in requests) / sum(walls)
                 if releases and len(walls) == len(releases) else None)
    many_decode = (sum(sum(r.get('decode_tok_s') or 0. for r in release['requests']) for release in releases)
                   / len(releases) if releases else None)
    return dict(tokens=tokens, clients=FIXED_CONCURRENCY_CLIENTS, concurrency=width,
                definition=f'bench-dec: four different 2K prompts with exactly `tokens` output tokens each; C=1 sends '
                           f'them one at a time, C={width} releases them {width} at a time; rate = output tokens / '
                           'first start to last completion of each release, summed',
                c1_tok_s=c1_rate, many_tok_s=many_rate,
                multiplier=many_rate / c1_rate if c1_rate and many_rate else None,
                c1_decode_tok_s=c1_decode, many_decode_tok_s_sum=many_decode,
                decode_multiplier=many_decode / c1_decode if c1_decode and many_decode else None, issues=issues)


def diagnostic_complete(report):
    def complete(rank):
        traces = rank.get('traces', [])
        def steps(phase):
            return {t.get('step') for t in traces if t.get('phase') == phase
                    and type(t.get('step')) is int and t.get('activities', 0) > 0}
        return rank.get('complete') and len(steps('prefill')) >= 1 and len(steps('decode')) >= 4
    return bool(report.get('ranks')) and all(complete(rank) for rank in report['ranks'])


def main() -> int:
    global _RUN
    try:
        return _main()
    except BaseException as exc:
        if _RUN is not None and not _RUN.complete:
            _RUN.finish(error=exc)
            if _RUN.supported and _RUN.token:
                try:
                    _RUN.control(op='abort', token=_RUN.token)
                except Exception:
                    pass  # The partial manifest retains the token for manual recovery.
        raise
    finally:
        CURRENT.set(None)


def _main() -> int:
    global _RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="onepass")
    # The workload this run IS, by name (bench/measurement_contract.PROFILES). `default` is the cheap
    # one every routine measurement uses -- the D17 probe after a deploy and both arms of a bracket --
    # so base and candidate always measure the same thing. `extended` is the full set, by name. An
    # explicit --ctx or --fixed-concurrency-tokens still wins; the record then says `custom` and the
    # judge keeps it out of the profiles' comparisons.
    ap.add_argument("--profile", default=os.environ.get("ONEPASS_PROFILE", contract.DEFAULT_PROFILE),
                    choices=sorted(contract.PROFILES), help="the named workload this run measures")
    chosen = contract.profile(os.environ.get("ONEPASS_PROFILE", contract.DEFAULT_PROFILE))
    ap.add_argument("--ctx", default=os.environ.get("QUALITY_CTX", ",".join(map(str, chosen["ctx"]))))
    ap.add_argument("--max-tokens", type=int, default=quality.MAX_TOKENS)
    ap.add_argument("--combined-max-tokens", type=int,
                    default=int(os.environ.get("ONEPASS_COMBINED_MAX_TOKENS",
                                               str(DEFAULT_COMBINED_MAX_TOKENS))),
                    help="total completion budget for the three-question combined request")
    ap.add_argument("--combined-reasoning-budget", type=int,
                    default=int(os.environ.get("ONEPASS_COMBINED_REASONING_BUDGET",
                                               str(DEFAULT_COMBINED_REASONING_BUDGET))),
                    help="reasoning token cap inside the combined completion")
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("SPEC_K", "7")))
    ap.add_argument("--combine-min-ctx", type=int, default=int(os.environ.get("ONEPASS_COMBINE_MIN_CTX", "32000")),
                    help="contexts at or above this size ask the three questions in ONE request (one prefill "
                         "instead of three; the fleet has no prefix cache). 0 = never combine")
    ap.add_argument("--out", default=os.environ.get("ONEPASS_JSONL",
                                                   os.path.expanduser("~/glm53-logs/bracket-onepass.jsonl")))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fixed-decode-tokens", type=int, default=int(os.environ.get("ONEPASS_FIXED_DECODE_TOKENS", "0")))
    ap.add_argument("--fixed-decode-reps", type=int, default=int(os.environ.get("ONEPASS_FIXED_DECODE_REPS", "3")))
    ap.add_argument("--fixed-concurrency-tokens", type=int,
                    default=int(os.environ.get("ONEPASS_FIXED_CONCURRENCY_TOKENS",
                                               str(chosen["fixed_concurrency_tokens"]))),
                    help="run 1: four different 2K prompts forced to exactly this many output tokens, one at a time "
                         "and then N at a time -- bench-dec's C=N/C=1 multiplier. 0 = skip")
    ap.add_argument("--require-exclusive", action="store_true", default=os.environ.get("ONEPASS_REQUIRE_EXCLUSIVE") == "1")
    args = ap.parse_args()
    if os.environ.get("FLEET_WORKLOAD"):
        from measurement_contract import workload
        work = workload(json.loads(os.environ["FLEET_WORKLOAD"]))
        for key, value in work.items():
            setattr(args, key, ','.join(map(str, value)) if key == 'ctx' else value)
        if not args.fixed_decode_tokens:
            args.fixed_decode_reps = 3  # CLI validity; normalized identity records zero when disabled
    if args.fixed_decode_tokens < 0 or args.fixed_decode_reps < 1:
        ap.error("fixed decode needs nonnegative tokens and positive repetitions")
    if args.fixed_concurrency_tokens < 0:
        ap.error("fixed concurrency tokens must be nonnegative")
    min_combined = args.max_tokens * 3
    if args.combined_max_tokens < min_combined:
        ap.error(f"combined max tokens must be at least {min_combined}")
    if not 0 <= args.combined_reasoning_budget < args.combined_max_tokens:
        ap.error("combined reasoning budget must be nonnegative and below combined max tokens")

    kq = _load("korean-corruption.py", "onepass_korean")
    cq = _load("check-quality.py", "onepass_quality")
    bd = br = _load("onepass_metrics.py", "onepass_metrics")
    rec = build_record(args, br._git_sha())
    if os.environ.get("FLEET_EXPERIMENT_ID"):
        rec["experiment_id"] = os.environ["FLEET_EXPERIMENT_ID"]
        rec["runtime"] = json.loads(os.environ.get("FLEET_CONTEXT", "{}"))
    rec["endpoint"] = {"completion": bd.URL, "metrics": bd.METRICS}
    rec["engine_shape"] = engine_shape(bd.URL)
    rec["generation_budget"] = {
        "individual_max_tokens": args.max_tokens,
        "individual_reasoning_budget": args.max_tokens // 2,
        "combined_max_tokens": args.combined_max_tokens,
        "combined_reasoning_budget": args.combined_reasoning_budget,
    }
    rec.update(_served_build(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if os.environ.get("ONEPASS_RUN_INDEX"):
        # D17 (45차 §93): two runs on one boot. Run 1 carries the cold column (TTFT, the compile
        # tail); run 2 the warm one. bench/st_judge.py judges warm against warm.
        rec["run_index"] = int(os.environ["ONEPASS_RUN_INDEX"])
    # Operator policy (2026-09-13): retain C=1 repeats, pay for C=N once.
    # The user removed C=N 128K, including preparation and diagnostic replays.
    # Accept both decimal and binary spellings; C=1 retains every requested context.
    # N follows the door's admission limit (`serving_concurrency`); the c4_* key names predate it.
    many = serving_concurrency(rec['engine_shape'])
    contexts = list(map(int, args.ctx.split(',')))
    # A fresh-identity arm cannot tag a concurrent group (one cache_salt per
    # request), so ONEPASS_C1_ONLY drops the C=N arm instead of racing it.
    c4_contexts = ([] if os.environ.get("ONEPASS_C1_ONLY") == "1"
                   else [ctx for ctx in contexts if ctx not in (128000, 131072)])
    first_run = rec.get('run_index', 1) == 1
    concurrencies = (1, many) if first_run and c4_contexts else (1,)
    rec['concurrency_coverage'] = dict(
        policy='c1-twice-c4-once-no-128k-v2' if many == MAX_CONCURRENCY else f'c1-twice-c{many}-once-no-128k-v3',
        width=many, included=list(concurrencies),
        contexts={'1': contexts, str(many): c4_contexts if first_run else []},
        c4_excluded_contexts=[ctx for ctx in contexts if ctx not in c4_contexts],
        c4_status=('scheduled' if many in concurrencies else
                   'omitted_after_run_1' if not first_run else 'omitted_context_scope'))
    if os.environ.get("ST_BRACKET_SHA"):
        rec["arm_sha"] = os.environ["ST_BRACKET_SHA"]              # the commit the bracket named for this arm
    if os.environ.get("ST_BRACKET_TREE"):
        rec["arm_tree"] = os.environ["ST_BRACKET_TREE"]            # the engine/ tree at that commit: the sample's real identity
    if os.environ.get("ST_BRACKET_COLD"):
        rec["cold"] = os.environ["ST_BRACKET_COLD"]                # what run 1 followed: a boot, or none (live: a D17 probe)
    run = _RUN = Run(rec, args.out, bd.URL)
    items = workload_requests(args, cq)
    fixed_item = None
    if args.fixed_decode_tokens:
        fixed_item = quality.request_item(2000, args.seed + 2000, quality.cases(args.seed + 2000),
            cq.filler, args.fixed_decode_tokens, args.fixed_decode_tokens // 2, 'fixed-all')
        fixed_item['min_tokens'] = args.fixed_decode_tokens
    run.workloads(items + ([fixed_item] if fixed_item else []))
    if os.environ.get("FLEET_SESSION"):
        rec["session"] = os.environ["FLEET_SESSION"]          # who held the fleet (fleet.sh run)
    if os.environ.get("MK_COLD_COMPILE") == "1":
        rec["cold_compile"] = True                              # first boot on this build (ab-lever)
    t_all = time.time()
    texts = []          # (tag, text, finish) for the corruption scan
    channel_traces = []  # Transient SSE references; only bounded diagnostics enter the record.
    phases = []         # (ctx, t_first_token, t_end): each answer's decode phase
    fixed_phases = []
    rec["requests"] = []
    pending_quality = []
    gen_tokens = 0

    prep_example = preparation_request(items[0], args.num_spec)
    rec['preparation_budget'] = dict(version=1, max_tokens=prep_example['max_tokens'],
        min_tokens=prep_example['min_tokens'], reasoning_budget=prep_example['reasoning_budget'],
        scope='full original prompts; bounded ungraded output; scored steady-state checks remain mandatory')
    print(f"prepare C=1: full context ladder, {prep_example['max_tokens']} output tokens per request; retained separately", flush=True)
    run.begin('prepare-c1')
    for item in items:
        group(run, ask_stream, bd.URL, cq.MODEL, preparation_request(item, args.num_spec), 1)
    if args.fixed_decode_tokens:
        for rep in range(args.fixed_decode_reps):
            group(run, ask_stream, bd.URL, cq.MODEL,
                  preparation_request(dict(fixed_item, seed=args.seed + rep), args.num_spec), 1)
    run.end()
    run.begin('measure-c1')

    from window_metrics import traffic_state, exclusive_errors, decode_windows
    metrics_before = _metrics_text(bd.METRICS)
    if precision := kda_state_storage(metrics_before):
        rec["kda_state_dtype"] = precision
    before_traffic = traffic_state(metrics_before)
    if args.require_exclusive and (before_traffic["running"] != 0 or before_traffic["waiting"] != 0):
        raise RuntimeError("exclusive onepass requires an idle server before sending requests")
    m0 = bd._parse_spec_metrics(metrics_before)
    print(f"{'ctx':>7} {'tok':>7} {'first tok/s':>11} {'median tok/s':>11} {'first TTFT':>10} {'median TTFT':>10} {'reuse':>6} quality"
          "   (* = the cache carried it: not a prefill, CHARTER D17)", flush=True)
    t_dec0 = time.time()
    # Keep the established 1 s window and 0.5 s edge margins; harder questions
    # change the workload, not the timing definition.
    with br._StepWindows(bd, period=1.0) as sw:
        for ctx in (int(c) for c in args.ctx.split(",")):
            ttfts, tok, first_tok = [], 0, 0
            reused_before = _st_counter(_metrics_text(bd.METRICS), "prefix_reused_tokens_total")
            ctx_items = [item for item in items if item['ctx'] == ctx]
            combined = len(ctx_items) == 1
            for item in ctx_items:
                t_req = time.monotonic()
                timing = {"ctx": ctx, "question": item['question'], "concurrency": 1}
                text, ttft, ptok, ctok, finish = ask_stream(
                    bd.URL, cq.MODEL, item['content'], item['max_tokens'], timing,
                    channel_trace=channel_traces, reasoning_budget=item['reasoning_budget'])
                phases.append((ctx, t_req + ttft, time.monotonic()))
                rec["requests"].append(timing)
                pending_quality.append((item, timing, finish))
                if ptok:
                    tok += ptok
                    if not first_tok:
                        first_tok = ptok
                gen_tokens += ctok
                ttfts.append(ttft)
                texts.append((f"ctx{ctx // 1000}K q{item['question']}", text, finish))
            cold, warm = ttfts[0], median(ttfts)
            # What this bracket took from the prefix cache instead of computing. Without it the
            # first column cannot be read: it is prompt tokens over the first TTFT either way.
            reused = max(0.0, _st_counter(_metrics_text(bd.METRICS), "prefix_reused_tokens_total") - reused_before)
            share = reused / tok if tok else 0.0
            # `tok` is every request of this context (three at 2K); the cold
            # column is the first request's prompt over its own TTFT, and the
            # reuse share is the counter delta over the same total the counter
            # was taken across.
            cold_tok_s = first_tok / cold if first_tok and cold > 0 else 0.0
            rec["prefill"].append({"ctx": ctx, "tok": tok, "cold_s": cold, "warm_s": warm,
                                   "cold_tok_s": cold_tok_s,
                                   "warm_tok_s": tok / warm if warm > 0 else 0.0,
                                   "ttft_samples_s": ttfts, "first_s": cold, "median_s": warm,
                                   "state": "prepared fresh-prefix; see steady_state validity",
                                   "reused_tok": int(reused), "reused_frac": round(share, 4),
                                   "cache_hit": share >= CACHE_HIT_FRACTION,
                                   "combined": combined})
            warm_col = f"{tok / warm:>11.0f}" if not combined else f"{'(1 req)':>11}"
            warm_t = f"{warm:>9.2f}s" if not combined else f"{'-':>10}"
            reuse_col = f"{share * 100:>5.0f}%" + ("*" if share >= CACHE_HIT_FRACTION else " ")
            print(f"{ctx:>7} {tok:>7} {cold_tok_s:>11.0f} {warm_col} {cold:>9.2f}s {warm_t} {reuse_col} quality deferred", flush=True)
        if args.fixed_decode_tokens:
            for rep in range(args.fixed_decode_reps):
                timing = {"ctx": 2000, "question": "fixed-all", "rep": rep, "fixed_decode": True}
                t_req = time.monotonic()
                text, ttft, ptok, ctok, finish = ask_stream(
                    bd.URL, cq.MODEL, fixed_item['content'], args.fixed_decode_tokens, timing,
                    min_tokens=args.fixed_decode_tokens, seed=args.seed + rep,
                    channel_trace=channel_traces, reasoning_budget=fixed_item['reasoning_budget'])
                phase = (2000, t_req + ttft, time.monotonic())
                phases.append(phase)
                fixed_phases.append(phase)
                rec["requests"].append(timing)
                gen_tokens += ctok
                pending_quality.append((fixed_item, timing, finish))
                texts.append((f"fixed2K rep{rep}", text, finish))
                print(f"fixed2K rep={rep} tokens={ctok}/{args.fixed_decode_tokens} "
                      f"decode={timing['decode_tok_s']:.2f} tok/s", flush=True)
    wall = time.time() - t_dec0
    metrics_after = _metrics_text(bd.METRICS)
    c1_report = run.end()
    c1_issues = steady_errors(c1_report, rec['requests'], 1)
    rec['steady_state'] = dict(valid=not c1_issues, issues=c1_issues, profile='off', prefix='fresh',
                              preparation='no observed specialization or capture' if not c1_issues else 'unverified')
    m1 = bd._parse_spec_metrics(metrics_after)
    traffic_issues = exclusive_errors(before_traffic, traffic_state(metrics_after),
                                      sw.traffic_samples, len(rec["requests"]))
    rec["traffic"] = {"before": before_traffic, "after": traffic_state(metrics_after),
                      "samples": sw.traffic_samples, "issues": traffic_issues}
    rows = rec["prefill"]
    if len(rows) >= 2:
        print(f"  준비 후 처리량: {rows[0]['warm_tok_s']:.0f} -> {rows[-1]['warm_tok_s']:.0f} tok/s "
              f"(over {rows[0]['tok']} -> {rows[-1]['tok']} tokens)")
    # Grade only after sampling, traffic counters and the server session end.
    # Raw completions have already been fsynced; these proof checks cannot add
    # gaps to the measured decode windows.
    quality_rows = []
    if len(pending_quality) != len(channel_traces):
        raise RuntimeError('missing final-channel evidence for quality grading')
    for (item, timing, finish), events in zip(pending_quality, channel_traces):
        result = run.grade(item, timing, events, finish, phase='measure-c1')
        quality_rows.extend(result)
        failed = [r['case'] + ':' + ','.join(k for k, ok in r['checks'].items() if not ok)
                  for r in result if not r['passed']]
        if failed:
            print(f"    QUALITY ctx={item['ctx']}: {'; '.join(failed)}", flush=True)
    rec['quality'] = quality.summarize(quality_rows)
    quality_ok, quality_total = rec['quality']['ok'], rec['quality']['total']
    print(f"=> {quality_ok}/{quality_total} reasoning cases passed; "
          f"rubric {rec['quality']['score']}/{rec['quality']['max_score']}", flush=True)

    # ---- decode: only windows that lie INSIDE an answer's decode phase (after
    # its first token, before its end) count, bucketed by the context length --
    # decode at 128K context is heavier than at 2K, and the legs' number is
    # the short-context one. Acceptance from the counters over the whole run.
    legacy, raw = br._spec_delta(m0, m1)
    k_eff = br.spec_k_eff(m0, m1) or args.num_spec   # 37차: the served k, not the flag
    samp = sw.samples
    # 39차: 1 s windows with 0.5 s margins (were 2 s / 1 s) so the 2K answers
    # (~3-4 s of decode) yield windows every boot; medians of step/s are
    # comparable across window sizes.
    by_ctx, fixed_intervals = decode_windows(samp, phases, fixed_phases, margin=0.5)
    rates = [r for v in by_ctx.values() for r in v]
    if args.fixed_decode_tokens:
        rates = [w["steps"] / w["seconds"] for w in fixed_intervals]
    win_med = median(rates) if rates else None
    if rates:
        per = "  ".join(f"{('ko' if c == 0 else str(c // 1000) + 'K')}: n={len(v)} med {median(v):.1f}"
                        for c, v in sorted(by_ctx.items()))
        print(f"decode: windows n={len(rates)} med {win_med:.1f} [{min(rates):.1f}, {max(rates):.1f}] step/s "
              f"(inside the answers only; per context: {per}), raw acc {(raw or 0) * 100:.1f}%, "
              f"tokens/step {1 + k_eff * (raw or 0):.3f} (k={k_eff:.1f}), generated {gen_tokens} tokens over {wall:.0f}s",
              flush=True)
    else:
        print("decode: no windows", flush=True)
    rec["decode"] = {"gen_tokens": gen_tokens, "wall_s": wall, "acc_raw": raw, "acc_legacy": legacy,
                     "tokens_per_step": 1 + k_eff * (raw or 0), "windows": rates,
                     "windows_med": win_med, "windows_by_ctx": {str(k): v for k, v in by_ctx.items()},
                     "num_spec": k_eff}
    if args.fixed_decode_tokens:
        rec["decode"].update(primary="fixed-2K", fixed_intervals=fixed_intervals,
            fixed_pooled_step_s=(sum(w["steps"] for w in fixed_intervals)
                                 / sum(w["seconds"] for w in fixed_intervals)) if fixed_intervals else None)

    # ---- Korean corruption on every answer
    dirty, chars, kinds_tot = [], 0, {}
    for index, (tag, text, finish) in enumerate(texts):
        chars += len(text)
        h = kq.scan(text, truncated=(finish == "length"))
        rec["requests"][index]["channel_diagnostics"] = _channel_diagnostics(
            channel_traces[index], text, finish, h, kq)
        gated = {k: v for k, v in h.items() if k not in kq.INFORMATIONAL}
        for k, v in gated.items():
            kinds_tot[k] = kinds_tot.get(k, 0) + v
        kinds = {k: v for k, v in gated.items() if v}
        if kinds:
            dirty.append((tag, kinds, text))
    n = len(texts)
    print(f"응답 {n}개 · 문자 {chars:,}")
    print(f"  깨진 응답: {len(dirty)}/{n} ({100 * len(dirty) / max(n, 1):.0f}%)")
    for k in ("replacement", "lone_jamo", "cjk_mixed", "control"):
        v = kinds_tot.get(k, 0)
        print(f"  {k:<14}{v:>4}  {1e6 * v / max(chars, 1):6.1f}/백만자")
    for tag, kinds, text in dirty:
        ks = " ".join(f"{k}={v}" for k, v in kinds.items())
        shown = False
        for k in kinds:
            pat = {"cjk_mixed": kq.HAN, "lone_jamo": kq.WELDED_JAMO}.get(k)
            m = pat.search(text) if pat is not None else None
            if m:
                a = max(0, m.start() - 40)
                print(f"\n  [{tag}] {ks}\n    …{text[a:m.end() + 40]!r}…")
                shown = True
                break
        if not shown:
            print(f"\n  [{tag}] {ks}")
    rec["korean"] = {"dirty": len(dirty), "n": n, "kinds": kinds_tot,
                     "hits": [(tag, k) for tag, k, _ in dirty]}
    issues = list(traffic_issues) + c1_issues
    if quality_ok != quality_total or dirty:
        issues.append('C=1 quality or Korean corruption gate failed')
    if args.fixed_decode_tokens:
        if any(sb < sa for (_, sa), (_, sb) in zip(samp, samp[1:])):
            issues.append("engine step counter reset during workload")
        if any(q["completion_tokens"] != args.fixed_decode_tokens for q in rec["requests"] if q.get("fixed_decode")):
            issues.append("fixed decode token count differs from requested length")
        if len(fixed_intervals) < 20:
            issues.append(f"too few fixed decode windows: {len(fixed_intervals)} < 20")
    if issues:
        rec["evidence_issues"] = issues
        rec["decode"]["raw_windows_med"] = rec["decode"]["windows_med"]
        rec["decode"]["windows_med"] = None
        print("INVALID measurement: " + "; ".join(issues), flush=True)
    # C=N and profiler replays have their own counters, requests and artifacts.
    c4_items = [item for item in items if item['ctx'] in c4_contexts] if many in concurrencies else []
    rec['c4'] = []
    for item in c4_items:
        ctx = item['ctx']
        suffix = f"{ctx}-q{item['question']}"
        print(f'prepare C={many} ctx={ctx}', flush=True)
        run.begin(f'prepare-c{many}-{suffix}', many)
        group(run, ask_stream, bd.URL, cq.MODEL, preparation_request(item, args.num_spec), many)
        run.end()
        run.begin(f'measure-c{many}-{suffix}', many)
        before = traffic_state(_metrics_text(bd.METRICS))
        result = group(run, ask_stream, bd.URL, cq.MODEL, item, many, kq, grade=True)
        after = traffic_state(_metrics_text(bd.METRICS))
        report = run.end()
        errors = steady_errors(report, result['requests'], many) + exclusive_errors(before, after, [], many)
        if any(r.get('corruption') or not all(q['passed'] for q in r['quality']) for r in result['requests']):
            errors.append(f'C={many} quality or Korean corruption gate failed')
        result.update(valid=not errors, issues=errors, traffic=dict(before=before, after=after),
                      latency_artifacts=f'measure-c{many}-{suffix}')
        rec['c4'].append(result)
        issues.extend(f'C={many} ctx={ctx}: {e}' for e in errors)
        print(f"C={many} ctx={ctx}: {result['aggregate_output_tok_s']:.2f} total tok/s; valid={not errors}", flush=True)
        run.checkpoint()
    if c4_items:
        rec['concurrency_coverage']['c4_status'] = 'measured'
    rec['quality_c4'] = (quality.summarize([q for result in rec['c4'] for r in result['requests'] for q in r['quality']])
                         if c4_items else None)

    # ---- bench-dec's multiplier: the same four fixed-length requests one at a time, then N at a time. Run 1
    # only (C=N is paid once a boot). An observation beside the canonical C=N rate: its issues stay in its
    # own block and never invalidate the record's decode or quality evidence.
    rec['concurrency_fixed'] = None
    if args.fixed_concurrency_tokens and first_run:
        tokens = args.fixed_concurrency_tokens
        fixed_items = fixed_concurrency_items(args.seed, cq, tokens)
        releases = fixed_concurrency_groups(fixed_items, many)
        print(f'fixed concurrency: four 2K prompts, exactly {tokens} output tokens each; C=1 then C={many}', flush=True)
        # Prepare at BOTH widths. Preparing only at C=N left the C=1 leg to JIT the width-1
        # shapes while it was being measured: 2026-09-16 (onepass-iso-0916, ST-3bff59a76e69)
        # counted specializations 55 -> 59 inside measure-fixed-c1 against 59 -> 61 inside
        # measure-fixed-c{N}. Both legs then failed steady_errors, and the multiplier they
        # printed was biased UP: the arm that paid more compile time is the denominator.
        # Preserve the full output/reasoning budgets and warm every arrival role.
        # Different simultaneous arrivals select different coexistence-prefill
        # tails (e.g. 1087 versus 1101 tokens). Even a full-length replay missed
        # these shapes on 2026-09-17. Ordered preparation waits for each client's
        # first token before admitting the next; rotations give each prompt each
        # role. Measured releases remain simultaneous and fail on any compilation.
        run.begin(f'prepare-fixed-c{many}', many)
        for release in releases:
            for offset in range(many):
                ordered = release[offset:] + release[:offset]
                group(run, ask_stream, bd.URL, cq.MODEL, ordered, many, prepare_ordered=True)
        run.end()
        run.begin('prepare-fixed-c1')
        for item in fixed_items:
            group(run, ask_stream, bd.URL, cq.MODEL, item, 1)
        run.end()
        run.begin('measure-fixed-c1')
        before = traffic_state(_metrics_text(bd.METRICS))
        # One client at a time through the same group path the C=N releases use; these ungraded requests
        # keep their channel traces out of the canonical C=1 quality/Korean scan.
        fixed_c1 = [group(run, ask_stream, bd.URL, cq.MODEL, item, 1)['requests'][0] for item in fixed_items]
        after = traffic_state(_metrics_text(bd.METRICS))
        fixed_errors = [f'C=1: {e}' for e in steady_errors(run.end(), fixed_c1, 1)
                        + exclusive_errors(before, after, [], FIXED_CONCURRENCY_CLIENTS)]
        run.begin(f'measure-fixed-c{many}', many)
        before = traffic_state(_metrics_text(bd.METRICS))
        fixed_many = [group(run, ask_stream, bd.URL, cq.MODEL, release, many) for release in releases]
        after = traffic_state(_metrics_text(bd.METRICS))
        many_requests = [r for release in fixed_many for r in release['requests']]
        fixed_errors += [f'C={many}: {e}' for e in steady_errors(run.end(), many_requests, many)
                         + exclusive_errors(before, after, [], FIXED_CONCURRENCY_CLIENTS)]
        summary = fixed_concurrency_summary(tokens, fixed_c1, fixed_many, many)
        summary['preparation_policy'] = dict(version=2, output_tokens=tokens,
            arrivals='first-token ordered cyclic rotations; all prompts in all arrival roles',
            measured_arrivals='simultaneous')
        summary['issues'] = fixed_errors + summary['issues']
        summary.update(valid=not summary['issues'], c1_requests=fixed_c1, many_requests=many_requests,
                       latency_artifacts=['measure-fixed-c1', f'measure-fixed-c{many}'])
        rec['concurrency_fixed'] = summary
        rate = lambda v: 'n/a' if v is None else f'{v:.1f}'
        ratio = lambda v: 'n/a' if v is None else f'{v:.2f}x'
        print(f"fixed concurrency {tokens} tok: C=1 {rate(summary['c1_tok_s'])} -> C={many} {rate(summary['many_tok_s'])} tok/s "
              f"= {ratio(summary['multiplier'])} (decode {ratio(summary['decode_multiplier'])}); "
              f"valid={summary['valid']}", flush=True)
        run.checkpoint()
    rec['diagnostics'] = []
    diagnostic_items = [diagnostic_request(next(item for item in items if item['ctx'] == ctx), k_eff)
                        for ctx in map(int, args.ctx.split(','))]
    rec['diagnostic_budget'] = dict(version=1, max_tokens=diagnostic_items[0]['max_tokens'],
        min_tokens=diagnostic_items[0]['min_tokens'], reasoning_budget=diagnostic_items[0]['reasoning_budget'],
        scope='diagnostic replay only; consumer generation_budget and quality workloads unchanged')
    for concurrency in concurrencies:
        for item in diagnostic_items:
            if concurrency == many and item['ctx'] not in c4_contexts:
                continue
            phase = f"diagnostic-c{concurrency}-{item['ctx']}"
            print(phase, flush=True)
            run.begin(phase, concurrency, diagnostic=True)
            group(run, ask_stream, bd.URL, cq.MODEL, item, concurrency)
            report = run.end()
            complete = diagnostic_complete(report)
            rec['diagnostics'].append(dict(phase=phase, complete=complete))
            if not complete:
                issues.append(f'{phase}: detailed GPU evidence incomplete')
    rec['evidence_issues'] = issues
    if issues:
        rec['decode'].setdefault('raw_windows_med', rec['decode']['windows_med'])
        rec['decode']['windows_med'] = None
    print(f"== onepass {args.name}: {time.time() - t_all:.0f}s total; artifacts {run.path}", flush=True)
    run.finish()
    return 2 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
