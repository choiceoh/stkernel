#!/usr/bin/env python3
"""Keep compact, hash-linked receipts from completed canonical onepass records.

Usage: python summarize_consumer.py RECORD.json [RECORD.json ...] --output FILE
Raw phase traces remain at each record's artifacts path. This does not judge or
relax quality, preparation or timing gates; their original outcomes are kept.
"""
import argparse
import hashlib
import json
from pathlib import Path


def select(obj, keys):
    return {key: obj[key] for key in keys.split() if key in obj}


def request_receipt(request):
    return select(request, "ctx question concurrency client completion_tokens prompt_tokens "
                  "min_tokens max_tokens reasoning_budget request_sha256 output_sha256 "
                  "workload_sha256 ttft_s elapsed_s decode_s finish_reason tpot_ms "
                  "decode_tok_s first_channels_s cached_tokens reasoning_tokens quality corruption")


def summarize(path):
    raw = path.read_bytes()
    record = json.loads(raw)
    if record.get('recording', {}).get('status') != 'complete':
        raise ValueError(f'{path}: incomplete record cannot be summarized as evidence')
    result = select(record, "name t git evidence_scope quality quality_c4 korean harness "
                    "workload workload_profile engine_shape generation_budget engine boot_id "
                    "knobs release image engine_source_sha256 kda_state_dtype run_index concurrency_coverage "
                    "arm_sha arm_tree run_id artifacts measurement_policy session "
                    "steady_state evidence_issues prefill diagnostics diagnostic_budget")
    result['source_record_sha256'] = hashlib.sha256(raw).hexdigest()
    result['decode'] = select(record.get('decode', {}), "gen_tokens wall_s acc_raw tokens_per_step "
                              "windows_med raw_windows_med windows_by_ctx steps")
    result['requests'] = [request_receipt(r) for r in record['requests']]
    result['c2'] = []
    for group in record.get('c4', []):
        entry = select(group, "ctx concurrency elapsed_s aggregate_output_tok_s rate_scope valid issues")
        entry['requests'] = [request_receipt(r) for r in group['requests']]
        result['c2'].append(entry)
    fixed = record.get('concurrency_fixed')
    result['concurrency_fixed'] = None
    if fixed:
        result['concurrency_fixed'] = select(fixed, "tokens clients concurrency definition c1_tok_s "
            "many_tok_s multiplier c1_decode_tok_s many_decode_tok_s_sum decode_multiplier issues valid preparation_policy")
        for key in ('c1_requests', 'many_requests'):
            result['concurrency_fixed'][key] = [request_receipt(r) for r in fixed[key]]
    quality_path = path.with_name('quality.jsonl')
    if quality_path.exists():
        quality_raw = quality_path.read_bytes()
        result['source_quality_sha256'] = hashlib.sha256(quality_raw).hexdigest()
        result['quality_details'] = [json.loads(line) for line in quality_raw.splitlines() if line.strip()]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps([summarize(p) for p in args.records],
                                     ensure_ascii=False, indent=2) + '\n')
