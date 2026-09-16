"""Check and summarize preserved official perf data; never rewrite raw metrics."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = json.loads((root / 'raw/benchy-raw.json').read_text())
command = json.loads((root / 'raw/benchy-command.json').read_text())
assert raw['version'] == command['version'] == '0.4.0'
assert raw['prefix_caching_enabled'] is False
assert '--exact-tg' in command['command'] and '--no-cache' in command['command']
for option in ('temperature=1', 'top_p=0.95', 'seed=42', 'retain=false',
               'chat_template_kwargs={"thinking":true}'):
    assert option in command['command']
rows = raw['benchmarks']
assert len(rows) == 6
assert {(r['context_size'], r['concurrency']) for r in rows} == {
    (d, c) for d in (0, 4096, 8192) for c in (1, 2)}
summary = []
for r in rows:
    assert r['prompt_size'] == 2048 and r['response_size'] == 512
    assert not r['is_context_prefill_phase']
    assert len(r['tg_throughput']['values']) == 3
    assert len(r['e2e_ttft']['values']) == 3 * r['concurrency']
    assert len(r['tg_req_throughput']['values']) == 3 * r['concurrency']
    summary.append({
        'depth': r['context_size'], 'concurrency': r['concurrency'],
        'raw_pp_tokens_per_second': r['pp_throughput']['mean'],
        'decode_aggregate_tokens_per_second': r['tg_throughput']['mean'],
        'decode_per_request_tokens_per_second': r['tg_req_throughput']['mean'],
        'e2e_ttft_ms': r['e2e_ttft']['mean'],
        'first_response_ms': r['ttfr']['mean'],
        'raw_estimated_prompt_processing_ms': r['est_ppt']['mean'],
    })
scaling = []
for depth in (0, 4096, 8192):
    a, b = [next(r for r in summary if r['depth'] == depth and r['concurrency'] == c)
            for c in (1, 2)]
    scaling.append({'depth': depth, 'c2_to_c1_decode_aggregate_ratio':
                    b['decode_aggregate_tokens_per_second'] / a['decode_aggregate_tokens_per_second']})
report = {
    'source': 'Unmodified official llama-benchy JSON captured before temporary-file deletion',
    'engine_sha': json.loads((root / 'candidate.json').read_text())['sha'],
    'benchy_version': raw['version'], 'completed_cells': len(rows),
    'runs_per_cell': 3, 'measured_requests': 27, 'cells': summary, 'scaling': scaling,
    'limits': [
        'C1 raw pp throughput and C2 per-request pp throughput use an empty role chunk and are invalid prefill speeds.',
        'C2 aggregate pp uses first-content-token time; do not divide it by faulty C1 pp.',
        'TTFT includes queueing, prefill and the first generation step.',
        'Decode is official client-observed streaming throughput; this is a different prompt and temperature from D17.',
        'Response length 512 was requested with --exact-tg; aggregate JSON does not retain individual usage records.',
        'Per-concurrency speculative acceptance counters were not captured by this CLI.',
    ],
}
(root / 'performance-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
