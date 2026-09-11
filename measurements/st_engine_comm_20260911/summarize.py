"""Validate the single exploratory TP4 run; do not infer a channel winner."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    reports = []
    for rank in range(4):
        name = f'st-tp4-lat-f4d7-r1-16-rank{rank}'
        r = json.loads((HERE/'pilot'/f'{name}.json').read_text())
        assert r['passed'] and r['rank'] == rank and r['chain'] == 32
        assert r['nccl'] == [2, 29, 7]
        assert r['env']['NCCL_MIN_NCHANNELS'] == r['env']['NCCL_MAX_NCHANNELS'] == '16'
        assert len(r['results']) == 8 and len(r['injected_wait']) == 2
        assert [(c['operation'], c['bytes']) for c in r['results']] == [
            ('max', 8), ('max', 48), ('max', 192), ('sum', 8192),
            ('sum', 49152), ('sum', 196608), ('sum', 2097152), ('sum', 8388608)]
        assert [c['delayed_rank'] for c in r['injected_wait']] == [-1, 3]
        for path, sha in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha, path
        for c in r['results'] + r['injected_wait']:
            assert c['exact'] and c['cpu_delta']['throttled_usec'] == 0
            for t in [c['per_operation'], *c['spans']]:
                assert len(t['samples_us']) == 20 and all(v > 0 for v in t['samples_us'])
                assert t['median_us'] == statistics.median(t['samples_us'])
        log = (HERE/'pilot'/f'{name}.log').read_text()
        assert 'Using network IB' in log and 'GDR 0' in log and 'Destroy COMPLETE' in log
        reports.append(r)
    platform = json.loads((HERE/'platform-and-lifecycle.json').read_text())
    assert len(platform) == 4
    for node in platform:
        assert node['cuInit'] == 0
        assert set(node['attributes']) == {'GPU_DIRECT_RDMA_SUPPORTED', 'DMA_BUF_SUPPORTED'}
        assert all(a == {'value': 0, 'error': 0} for a in node['attributes'].values())
        own = [e for e in node['events'] if e['name'] == 'st-tp4-lat-f4d7-r1-16']
        assert any(e['action'] == 'die' and e['exit'] == '0' for e in own)
        assert any(e['action'] == 'destroy' for e in own)
    timings = []
    for i, c in enumerate(reports[0]['results']):
        values = [r['results'][i]['per_operation']['median_us'] for r in reports]
        timings.append(dict(operation=c['operation'], bytes=c['bytes'],
                            rank_median_range_us=[min(values), max(values)]))
    cfg = json.loads((HERE/'drafter-config.json').read_text())
    assert cfg['vocab_size'] == 154880 and cfg['dflash_config']['selector_top_k'] == 16
    old = 5 * 154880 * 2
    proposed = 5 * 4 * 16 * (4 + 8)
    result = dict(status='exploratory baseline only; channel comparison blocked by another fleet owner',
                  source_baseline='b4cb126b1908eb796ea4091196bcada21d57dbaf',
                  correctness_cases=4*10, timings=timings,
                  injected_wait=[dict(rank=r['rank'], baseline_comm_us=r['injected_wait'][0]['spans'][1]['median_us'],
                                      delayed_comm_us=r['injected_wait'][1]['spans'][1]['median_us'],
                                      local_delay_us=r['injected_wait'][1]['spans'][0]['median_us']) for r in reports],
                  target_sum_calls=1+2*45,
                  drafter_proposal=dict(implemented=False, allgather_result_bytes=old,
                                       fp32_score_int64_id_result_bytes=proposed,
                                       logical_payload_reduction_percent=100*(1-proposed/old)))
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for t in timings:
        print(t['operation'],t['bytes'],[round(v,2) for v in t['rank_median_range_us']])
    print('candidate logical payload reduction', result['drafter_proposal']['logical_payload_reduction_percent'])


if __name__ == '__main__':
    main()
