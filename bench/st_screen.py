#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Short ST screening on one boot. Observations are never adoption evidence.

Use the onepass stream, workload and durable latency recorder, with a bounded
2K question at C=1/C=4. Preparation is separate; no profiler replay or long
context quality campaign is scheduled. Full onepass remains the adoption gate.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import onepass as op
from onepass_recording import CURRENT, Run, group, steady_errors
from window_metrics import exclusive_errors, traffic_state

POLICY = dict(version=1, scope='screen', concurrency=[1, 4], ctx=[2000],
              max_tokens=512, min_tokens=128, reasoning_budget=256,
              preparation_tokens=64, prefix='unique salt per request',
              preparation='64 tokens at each concurrency, retained separately',
              measurement='profiler off; preparation changes invalidate timing',
              diagnostic='omitted for screening', adoption_eligible=False)


def health_errors(report, requests, concurrency, *, require_width=True):
    """Runtime/coverage failures remain failures, regardless of the speed observed."""
    errors = []
    ranks = report.get('ranks', [])
    if len(ranks) != 4 or {r.get('rank') for r in ranks} != {0, 1, 2, 3}:
        errors.append('missing TP4 rank coverage')
    for rank in ranks:
        if rank.get('error') or not rank.get('complete') or rank.get('diagnostic'):
            errors.append(f"rank {rank.get('rank')}: failed or incomplete measurement")
        widths = [len(row['rows']) for row in rank.get('rows', [])
                  if row.get('kind') == 'host_step' and row.get('phase') == 'decode']
        if require_width and max(widths, default=0) != concurrency:
            errors.append(f"rank {rank.get('rank')}: C={concurrency} decode was not exercised")
    if len(requests) != concurrency or any(
            r.get('error') or not r.get('completion_tokens')
            or r.get('finish_reason') not in ('stop', 'length') for r in requests):
        errors.append('incomplete generation')
    for request in requests:
        for key in ('ttft_s', 'elapsed_s', 'decode_tok_s'):
            value = request.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f'nonfinite or missing {key}')
    return list(dict.fromkeys(errors))


def collect(run, item, bd, model, scanner, bracket):
    """Persist both concurrency arms; short-budget quality misses are observations."""
    record = run.record
    record.update(measurement_policy=POLICY, screen=[], evidence_scope='screen',
                  adoption_eligible=False, evidence_issues=['screen only; full validation pending'])
    run.workloads([item])
    for concurrency in POLICY['concurrency']:
        before = traffic_state(op._metrics_text(bd.METRICS))
        if before['running'] != 0 or before['waiting'] != 0:
            raise RuntimeError('screen requires an idle server before preparation')
        phase = f'prepare-c{concurrency}'
        print(f'{phase}: {POLICY["preparation_tokens"]} tokens', flush=True)
        run.begin(phase, concurrency)
        preparation = dict(item, min_tokens=64, max_tokens=64, reasoning_budget=32)
        result = group(run, op.ask_stream, bd.URL, model, preparation, concurrency)
        after = traffic_state(op._metrics_text(bd.METRICS))
        report = run.end()
        # Short preparation may finish before all prefills join the decode
        # batch. Require actual C=4 coverage in the measured arm, not here.
        errors = health_errors(report, result['requests'], concurrency, require_width=False)
        errors += exclusive_errors(before, after, [], concurrency)
        if errors:
            raise RuntimeError('preparation failed: ' + '; '.join(errors))

        phase = f'measure-c{concurrency}'
        run.begin(phase, concurrency)
        metrics_before = op._metrics_text(bd.METRICS)
        before = traffic_state(metrics_before)
        if before['running'] != 0 or before['waiting'] != 0:
            raise RuntimeError('screen requires an idle server before measurement')
        result = group(run, op.ask_stream, bd.URL, model, item, concurrency, scanner, grade=True)
        metrics_after = op._metrics_text(bd.METRICS)
        after = traffic_state(metrics_after)
        report = run.end()
        errors = health_errors(report, result['requests'], concurrency)
        errors += exclusive_errors(before, after, [], concurrency)
        timing_issues = steady_errors(report, result['requests'], concurrency)
        # These are the actual deltas for this arm, including acceptance. No
        # step-window floor or speed target is required to finish an experiment.
        m0, m1 = map(bd._parse_spec_metrics, (metrics_before, metrics_after))
        _, acceptance = bracket._spec_delta(m0, m1)
        result.update(health_errors=errors, timing_issues=timing_issues,
                      valid=not errors and not timing_issues,
                      traffic=dict(before=before, after=after), acc_raw=acceptance,
                      latency_artifacts=phase,
                      quality_scope='observation only; short budget and minimum output length')
        record['screen'].append(result)
        record['kda_state_dtype'] = op.kda_state_storage(metrics_before)
        run.checkpoint()
        print(f'C={concurrency}: {result["aggregate_output_tok_s"]:.2f} total tok/s; '
              f'timing valid={result["valid"]}; artifacts={run.path / phase}', flush=True)
        if errors:
            raise RuntimeError('; '.join(errors))
    record['screen_status'] = ('observed' if all(r['valid'] for r in record['screen'])
                               else 'timing_unverified')
    record['concurrency_coverage'] = dict(included=[1, 4], c4_status='measured', policy='screen-v1')
    print('Screen complete; quality observations and latency saved. Full comparison pending.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--name', required=True)
    parser.add_argument('--require-exclusive', action='store_true', help='always enforced for screening')
    parser.add_argument('--out', default=os.environ.get('ONEPASS_JSONL', '~/glm53-logs/bracket-onepass.jsonl'))
    args = parser.parse_args(argv)
    bd = op._load('bench-dec.py', 'screen_bench_dec')
    cq = op._load('check-quality.py', 'screen_quality')
    scanner = op._load('korean-corruption.py', 'screen_korean')
    bracket = op._load('bracket.py', 'screen_bracket')
    record = dict(name=args.name, t=time.strftime('%F %T'), git=bracket._git_sha(),
                  evidence_scope='screen', adoption_eligible=False, harness=44, screening_protocol=1,
                  session=os.environ.get('FLEET_SESSION', ''),
                  run_index=int(os.environ.get('ONEPASS_RUN_INDEX', '1')),
                  arm_sha=os.environ.get('ST_BRACKET_SHA', ''),
                  arm_tree=os.environ.get('ST_BRACKET_TREE', ''),
                  cold=os.environ.get('ST_BRACKET_COLD', 'boot'),
                  engine_shape=op.engine_shape(bd.URL),
                  endpoint=dict(completion=bd.URL, metrics=bd.METRICS))
    record.update(op._served_build(str(Path(__file__).resolve().parents[1])))
    run = Run(record, args.out, bd.URL)
    try:
        if record.get('engine') != 'st' or not run.supported or run.server.get('ranks') != 4:
            raise RuntimeError('screen requires an ST door with TP4 latency recording')
        record['boot_id'] = record.get('boot_id') or run.server.get('boot_id')
        if not record['boot_id']:
            raise RuntimeError('screen requires a served boot identity')
        # One deterministic hard question, with the same original prompt at
        # C=1 and C=4. It retains raw reasoning/content and certificate results.
        item = op.quality.request_item(2000, 2042, op.quality.cases(2042)[:1], cq.filler,
                                      POLICY['max_tokens'], POLICY['reasoning_budget'], 'screen-ledger')
        item.update(seed=42, min_tokens=POLICY['min_tokens'])
        collect(run, item, bd, cq.MODEL, scanner, bracket)
        run.finish()
        return 0
    except BaseException as exc:
        record['screen_status'] = 'failed'
        run.finish(error=exc)
        if run.supported and run.token:
            try:
                run.control(op='abort', token=run.token)
            except Exception:
                pass  # The incomplete manifest retains the token for recovery.
        raise
    finally:
        CURRENT.set(None)


if __name__ == '__main__':
    sys.exit(main())
