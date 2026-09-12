"""Join passive PR760 samples to completed canonical C=1 request intervals.

Both inputs must come from the same head host and immutable boot. Preparation
and measurement stay separate. A completed request is required: an unfinished
stream cannot supply its end boundary. This does not replace onepass grading.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench'))
from step_peek import series


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(samples, requests):
    results = []
    totals = defaultdict(list)
    for request in requests:
        if request.get('concurrency', 1) != 1:
            continue
        start = request['started_monotonic'] + request['ttft_s'] + .5
        end = request['ended_monotonic'] - .5
        windows = []
        for a, b in zip(samples, samples[1:]):
            ta, tb = a['monotonic'], b['monotonic']
            if not start <= ta < tb <= end:
                continue
            sa, sb = a['series'], b['series']
            if any(series(s, 'vllm:num_requests_running') != 1 or
                   series(s, 'vllm:num_requests_waiting') != 0 for s in (sa, sb)):
                continue
            deltas = {}
            for key, name in (
                    ('steps', 'st:steps_decode_total'),
                    ('prefill', 'st:steps_prefill_total'),
                    ('finished', 'vllm:request_success_total'),
                    ('tokens', 'vllm:generation_tokens_total'),
                    ('accepted', 'vllm:spec_decode_num_accepted_tokens_total'),
                    ('drafted', 'vllm:spec_decode_num_draft_tokens_total')):
                va, vb = series(sa, name), series(sb, name)
                deltas[key] = None if va is None or vb is None else vb - va
            if any(v is None or v < 0 for v in deltas.values()):
                continue
            if deltas['prefill'] or deltas['finished']:
                continue
            windows.append(dict(seconds=tb-ta, **deltas))
        label = (request.get('phase'), request.get('ctx'))
        totals[label].extend(windows)
        fields = ('phase', 'ctx', 'question', 'client', 'completion_tokens',
                  'prompt_tokens', 'ttft_s', 'decode_tok_s', 'elapsed_s',
                  'finish_reason', 'output_sha256', 'reasoning_budget', 'max_tokens')
        results.append({**{k: request.get(k) for k in fields}, 'observed': aggregate(windows)})
    return dict(
        scope='same-host, same-boot PR760 samples inside completed canonical C=1 requests',
        edge_margin_s=.5,
        requests=results,
        by_phase_context=[dict(phase=phase, ctx=ctx, **aggregate(windows))
                          for (phase, ctx), windows in totals.items()],
        limitation='consumer quality and exclusivity verdict come from the canonical onepass record; no baseline on this build')


def aggregate(windows):
    seconds = sum(w['seconds'] for w in windows)
    steps = sum(w['steps'] for w in windows)
    accepted = sum(w['accepted'] for w in windows)
    drafted = sum(w['drafted'] for w in windows)
    rates = [w['steps']/w['seconds'] for w in windows if w['steps'] > 0]
    return dict(windows=len(windows), zero_step_windows=sum(w['steps'] == 0 for w in windows),
                seconds=seconds, steps=steps,
                pooled_step_s=steps/seconds if seconds else None,
                positive_window_median_step_s=statistics.median(rates) if rates else None,
                accepted=accepted, drafted=drafted,
                acceptance=accepted/drafted if drafted else None,
                output_counter_tok_s=sum(w['tokens'] for w in windows)/seconds if seconds else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples')
    parser.add_argument('requests')
    args = parser.parse_args()
    print(json.dumps(summarize(read_rows(args.samples), read_rows(args.requests)), indent=2))
