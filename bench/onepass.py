#!/usr/bin/env python3
"""Canonical Korean consumer test, harness 42: C=1 and C=4 at 2K/32K/128K.

Every invocation prepares each workload with a full replay, measures with the
profiler off and a unique prefix salt, then runs separate bounded GPU diagnostic
replays. Preparation changes, prefix reuse, missing evidence and external traffic
invalidate steady-state comparisons. First reasoning/content, SSE gaps, per-request
rates, actual batch widths, host/device stages and raw traces are retained under
onepass-runs/<run-id> beside the append-only output ledger, including partial runs.

C=4 sends four independent requests simultaneously for each canonical question;
its aggregate output rate includes prefill and remains separate from C=1 decode.
The legacy cold_s/warm_s fields are aliases for first/median prepared fresh-prefix
TTFT, not claims about compiler or cache warmth. Harness 41 is incompatible:
42 uses seeded reasoning dossiers and visible-answer proof certificates.

    python3 bench/onepass.py --name RUN [--ctx 2000,32000,128000]

Optional --fixed-decode-tokens 2048 --fixed-decode-reps 3 retains the C=1 fixed
response gate. Detailed GPU recording requires the ST latency endpoint.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
import uuid

from onepass_recording import CURRENT, Run, group, steady_errors
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
               channel_trace=None, reasoning_budget=None):
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
                "chat_template_kwargs": {"thinking": True}}
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


def _served_build(repo: str, profile: str = "glm53") -> dict:
    """What the SERVER is running: the deployed overlay stamp and the knobs
    that differ from the profile's defaults.

    NOT this process's environment -- ab-lever.sh boots the server with the
    arm's env and then runs this bench in a plain shell, so os.environ here
    carries none of it. The serving container's own Config.Env is the only
    honest source, and the overlay stamp identifies the BUILD (a bench with
    no deploy reuses the previous build whatever the git sha says).
    An empty `knobs` IS that build's baseline -- bench/baseline.py reads it
    so the next session can skip re-measuring one. Every failure degrades to
    a missing field: a bench must never die over its own label.
    """
    import subprocess
    out = {}
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True,
                               timeout=10).stdout.split()
    except Exception:
        names = []
    if "st-glm53" in names:
        # The ST engine, not vLLM: its identity is the source the container runs (the runtime
        # manifest's sha256), the release the launcher stamped, and STK_* knobs -- never the
        # vLLM overlay stamp, which would make every ST release look like one build.
        return _st_build(names)
    try:
        stamp = os.environ.get("MK_OVERLAY_STAMP",
                               "/home/choiceoh/glm53-cache/.overlay-sha")
        with open(stamp) as fh:
            out["overlay"] = fh.read().strip()[:12]
    except Exception:
        pass
    try:
        name = next((n for n in names if n.startswith("glm53")), None)
        if not name:
            return out
        boot = subprocess.run(["docker", "inspect", "-f", "{{.Id}}|{{.State.StartedAt}}", name],
                              capture_output=True, text=True, timeout=10)
        if boot.returncode == 0 and "|" in boot.stdout.strip():
            out["boot_id"] = boot.stdout.strip()
        raw = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Env}}", name],
                             capture_output=True, text=True, timeout=10).stdout
        served = dict(e.split("=", 1) for e in json.loads(raw or "[]")
                      if "=" in e and e.startswith("VLLM_"))
        declared = {}
        with open(os.path.join(repo, "profiles", profile + ".env")) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("VLLM_") and "=" in line:
                    k, v = line.split("=", 1)
                    declared[k] = v.strip().strip('"')
                elif line.startswith("SPEC_K="):
                    # The launcher exports this profile setting under a
                    # VLLM alias for the compile-cache key. Matching values
                    # belong to the baseline, not an experimental knob.
                    declared["VLLM_GLM53_SPEC_K"] = line.split("=", 1)[1].strip().strip('"')
        knobs = {k: v for k, v in served.items() if k in declared and v != declared[k]}
        knobs.update({k: v for k, v in served.items()
                      if k.startswith("VLLM_GLM53_") and k not in declared})
        out["knobs"] = dict(sorted(knobs.items()))
    except Exception:
        pass
    return out


def _served_speculation(boot_id, *, preparation=False):
    """Read the identified head's actual command; no serving imports or requests."""
    import subprocess
    try:
        from glm53_launch_metadata import launch_speculation
        if not isinstance(boot_id, str) or re.fullmatch(r'[0-9a-f]{64}\|[^|]+', boot_id) is None:
            raise ValueError('missing identified serving boot')
        container_id, started = boot_id.split('|', 1)
        raw = subprocess.check_output(['docker', 'inspect', container_id], text=True,
                                      stderr=subprocess.DEVNULL, timeout=10)
        containers = json.loads(raw)
        if not isinstance(containers, list) or len(containers) != 1:
            raise ValueError('ambiguous serving container')
        container = containers[0]
        state = container['State']
        if (container['Id'] != container_id or state['StartedAt'] != started
                or state['Running'] is not True or state['Paused'] or state['Restarting']):
            raise ValueError('serving boot changed or stopped')
        env = {}
        for item in container['Config']['Env']:
            key, value = item.split('=', 1)
            if key in env:
                raise ValueError('duplicate serving environment')
            env[key] = value
        result = dict(launch_speculation(container['Config']['Cmd']), boot_id=boot_id,
                      image=container['Image'], environment_spec_k=env.get('VLLM_GLM53_SPEC_K'))
        if preparation:
            result.update(preparation_mode=env.get('VLLM_GLM53_PREP_FUSED'),
                preparation_kernel=env.get('VLLM_GLM53_PREP_FUSED_KERNEL', 'cuda'),
                shadow_every=env.get('VLLM_GLM53_PREP_FUSED_SHADOW_EVERY', '1'),
                selfcheck_every=env.get('VLLM_GLM53_PREP_FUSED_SELFCHECK_EVERY', '64'))
        return result
    except (KeyError, ValueError, TypeError, OSError, subprocess.SubprocessError):
        return None  # Missing evidence never arms SPEC_K proof.


def build_record(args, revision):
    from measurement_contract import from_args, metadata
    return dict(name=args.name, t=time.strftime("%F %T"), git=revision,
                prefill=[], quality={}, decode={}, korean={}, **metadata(from_args(args)))


def _require_preparation(rec):
    # Defaults stay knobs={}; execution must still be proved. Seed a rejection
    # before collection so an exception cannot erase this requirement.
    rec['required_proofs'] = ['VLLM_GLM53_PREP_FUSED']
    rec['proof'] = {'VLLM_GLM53_PREP_FUSED': False}
    rec['proof_ok'] = '0/1'
    rec['preparation'] = dict(verdict='REJECTED', reason='preparation proof not completed')


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
    ap.add_argument("--ctx", default=os.environ.get("QUALITY_CTX", "2000,32000,128000"))
    ap.add_argument("--max-tokens", type=int, default=quality.MAX_TOKENS)
    ap.add_argument("--combined-max-tokens", type=int,
                    default=int(os.environ.get("ONEPASS_COMBINED_MAX_TOKENS",
                                               str(DEFAULT_COMBINED_MAX_TOKENS))),
                    help="total completion budget for the three-question combined request")
    ap.add_argument("--combined-reasoning-budget", type=int,
                    default=int(os.environ.get("ONEPASS_COMBINED_REASONING_BUDGET",
                                               str(DEFAULT_COMBINED_REASONING_BUDGET))),
                    help="reasoning token cap inside the combined completion")
    ap.add_argument("--num-spec", type=int, default=int(os.environ.get("SPEC_K", "6")))
    ap.add_argument("--combine-min-ctx", type=int, default=int(os.environ.get("ONEPASS_COMBINE_MIN_CTX", "32000")),
                    help="contexts at or above this size ask the three questions in ONE request (one prefill "
                         "instead of three; the fleet has no prefix cache). 0 = never combine")
    ap.add_argument("--out", default=os.environ.get("ONEPASS_JSONL",
                                                   os.path.expanduser("~/glm53-logs/bracket-onepass.jsonl")))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fixed-decode-tokens", type=int, default=int(os.environ.get("ONEPASS_FIXED_DECODE_TOKENS", "0")))
    ap.add_argument("--fixed-decode-reps", type=int, default=int(os.environ.get("ONEPASS_FIXED_DECODE_REPS", "3")))
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
    min_combined = args.max_tokens * 3
    if args.combined_max_tokens < min_combined:
        ap.error(f"combined max tokens must be at least {min_combined}")
    if not 0 <= args.combined_reasoning_budget < args.combined_max_tokens:
        ap.error("combined reasoning budget must be nonnegative and below combined max tokens")

    kq = _load("korean-corruption.py", "onepass_korean")
    cq = _load("check-quality.py", "onepass_quality")
    bd = _load("bench-dec.py", "onepass_bench_dec")
    br = _load("bracket.py", "onepass_bracket")
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
    if os.environ.get("ST_BRACKET_SHA"):
        rec["arm_sha"] = os.environ["ST_BRACKET_SHA"]              # the commit the bracket named for this arm
    if os.environ.get("ST_BRACKET_COLD"):
        rec["cold"] = os.environ["ST_BRACKET_COLD"]                # what run 1 followed: a boot, or only a prefix reset
    run = _RUN = Run(rec, args.out, bd.URL)
    items = workload_requests(args, cq)
    fixed_item = None
    if args.fixed_decode_tokens:
        fixed_item = quality.request_item(2000, args.seed + 2000, quality.cases(args.seed + 2000),
            cq.filler, args.fixed_decode_tokens, args.fixed_decode_tokens // 2, 'fixed-all')
        fixed_item['min_tokens'] = args.fixed_decode_tokens
    run.workloads(items + ([fixed_item] if fixed_item else []))
    prove_spec = 'VLLM_GLM53_SPEC_K' in (rec.get('knobs') or {})
    spec_before = _served_speculation(rec.get('boot_id')) if prove_spec else None
    prove_prep = os.environ.get('ONEPASS_REQUIRE_PREP_FUSED') == '1'
    if prove_prep:
        _require_preparation(rec)
    prep_before = _served_speculation(rec.get('boot_id'), preparation=True) if prove_prep else None
    prep_log_path = os.environ.get('MK_HEAD_LOG', '/home/choiceoh/glm53-logs/glm53.log')
    prep_prefix = None
    if prove_prep:
        from glm53_prep_proof import log_prefix
        prep_prefix = log_prefix(prep_log_path)
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

    print('prepare C=1: full context ladder; retained separately', flush=True)
    run.begin('prepare-c1')
    for item in items:
        group(run, ask_stream, bd.URL, cq.MODEL, item, 1)
    if args.fixed_decode_tokens:
        for rep in range(args.fixed_decode_reps):
            group(run, ask_stream, bd.URL, cq.MODEL, dict(fixed_item, seed=args.seed + rep), 1)
    run.end()
    run.begin('measure-c1')

    from window_metrics import traffic_state, exclusive_errors, decode_windows
    metrics_before = _metrics_text(bd.METRICS)
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
            ttfts, tok = [], 0
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
                tok = ptok or tok
                gen_tokens += ctok
                ttfts.append(ttft)
                texts.append((f"ctx{ctx // 1000}K q{item['question']}", text, finish))
            cold, warm = ttfts[0], median(ttfts)
            # What this bracket took from the prefix cache instead of computing. Without it the
            # first column cannot be read: it is prompt tokens over the first TTFT either way.
            reused = max(0.0, _st_counter(_metrics_text(bd.METRICS), "prefix_reused_tokens_total") - reused_before)
            share = reused / tok if tok else 0.0
            rec["prefill"].append({"ctx": ctx, "tok": tok, "cold_s": cold, "warm_s": warm,
                                   "cold_tok_s": tok / cold if cold > 0 else 0.0,
                                   "warm_tok_s": tok / warm if warm > 0 else 0.0,
                                   "ttft_samples_s": ttfts, "first_s": cold, "median_s": warm,
                                   "state": "prepared fresh-prefix; see steady_state validity",
                                   "reused_tok": int(reused), "reused_frac": round(share, 4),
                                   "cache_hit": share >= CACHE_HIT_FRACTION,
                                   "combined": combined})
            warm_col = f"{tok / warm:>11.0f}" if not combined else f"{'(1 req)':>11}"
            warm_t = f"{warm:>9.2f}s" if not combined else f"{'-':>10}"
            reuse_col = f"{share * 100:>5.0f}%" + ("*" if share >= CACHE_HIT_FRACTION else " ")
            print(f"{ctx:>7} {tok:>7} {tok / cold:>11.0f} {warm_col} {cold:>9.2f}s {warm_t} {reuse_col} quality deferred", flush=True)
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
    spec_after = _served_speculation(rec.get('boot_id')) if prove_spec else None
    prep_after = _served_speculation(rec.get('boot_id'), preparation=True) if prove_prep else None
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
    # armed != serving: which of the arm's lanes actually ran, from the head log,
    # checked after the traffic above (serving markers appear only then).
    try:
        from proof import check as _proof_check
        _kn = [kk for kk, vv in (rec.get("knobs") or {}).items() if vv not in ("0", "", "off")]
        if prove_prep and 'VLLM_GLM53_PREP_FUSED' not in _kn:
            _kn.append('VLLM_GLM53_PREP_FUSED')
        if _kn:
            rec.update({kk: vv for kk, vv in _proof_check(
                _kn, os.environ.get("MK_HEAD_LOG", "/home/choiceoh/glm53-logs/glm53.log"),
                speculation=dict(expected_k=rec['knobs'].get('VLLM_GLM53_SPEC_K'),
                    boot_id=rec.get('boot_id'), launch_before=spec_before, launch_after=spec_after,
                    exclusive=args.require_exclusive and not traffic_issues,
                    metrics_before=metrics_before, metrics_after=metrics_after),
                preparation=dict(expected_mode=rec['knobs'].get('VLLM_GLM53_PREP_FUSED', '1'),
                    boot_id=rec.get('boot_id'), launch_before=prep_before, launch_after=prep_after,
                    exclusive=args.require_exclusive and not traffic_issues,
                    log_prefix=prep_prefix) if prove_prep else None).items()
                if kk in ("proof", "proof_ok", "speculation", "preparation")})
    except Exception:
        pass

    # C=4 and profiler replays have their own counters, requests and artifacts.
    c4_items = items
    rec['c4'] = []
    for item in c4_items:
        ctx = item['ctx']
        suffix = f"{ctx}-q{item['question']}"
        print(f'prepare C=4 ctx={ctx}', flush=True)
        run.begin(f'prepare-c4-{suffix}', 4)
        group(run, ask_stream, bd.URL, cq.MODEL, item, 4)
        run.end()
        run.begin(f'measure-c4-{suffix}', 4)
        before = traffic_state(_metrics_text(bd.METRICS))
        result = group(run, ask_stream, bd.URL, cq.MODEL, item, 4, kq, grade=True)
        after = traffic_state(_metrics_text(bd.METRICS))
        report = run.end()
        errors = steady_errors(report, result['requests'], 4) + exclusive_errors(before, after, [], 4)
        if any(r.get('corruption') or not all(q['passed'] for q in r['quality']) for r in result['requests']):
            errors.append('C=4 quality or Korean corruption gate failed')
        result.update(valid=not errors, issues=errors, traffic=dict(before=before, after=after),
                      latency_artifacts=f'measure-c4-{suffix}')
        rec['c4'].append(result)
        issues.extend(f'C=4 ctx={ctx}: {e}' for e in errors)
        print(f"C=4 ctx={ctx}: {result['aggregate_output_tok_s']:.2f} total tok/s; valid={not errors}", flush=True)
        run.checkpoint()
    rec['quality_c4'] = quality.summarize([q for result in rec['c4'] for r in result['requests'] for q in r['quality']])
    rec['diagnostics'] = []
    diagnostic_items = [next(item for item in items if item['ctx'] == ctx) for ctx in map(int, args.ctx.split(','))]
    for concurrency in (1, 4):
        for item in diagnostic_items:
            phase = f"diagnostic-c{concurrency}-{item['ctx']}"
            print(phase, flush=True)
            run.begin(phase, concurrency, diagnostic=True)
            group(run, ask_stream, bd.URL, cq.MODEL, item, concurrency)
            report = run.end()
            complete = bool(report['ranks']) and all(r.get('complete') and r.get('traces') for r in report['ranks'])
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
