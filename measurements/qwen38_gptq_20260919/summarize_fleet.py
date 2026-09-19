"""Summarize the 330K fleet projection evidence and matched historical 131K fit."""
import argparse
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    rows, coverage = [], []
    for rank in range(4):
        fit = read(root / f'fit330-audit-rank{rank}.json')
        boot = read(root / f'B330pack-audit-rank{rank}.json')
        score = read(root / f'projection-rank{rank}.json')
        held = read(HERE / f'heldout-audit-rank{rank}.json')
        old = read(HERE / 'compare' / f'projection-rank{rank}.json')
        assert fit['statistics_valid'] and fit['sites'] == 193 and fit['minimum_rows'] >= 330000
        assert boot['serving_gptq_verified'] and boot['w4_sites'] == 192 and boot['fp8_sites'] == 193
        assert len({v['weights_id'] for v in (fit, boot, score, held, old)}) == 1
        held_records = {r['name']: r for r in held['records']}
        fit_records = {r['name']: r for r in fit['records']}
        old_cases = {(r['name'], r['lane']): r for r in old['cases']}
        new_cases = {(r['name'], r['lane']): r for r in score['cases']}
        assert len(new_cases) == len(score['cases']) == 385 and new_cases.keys() == old_cases.keys()
        coverage.append(dict(rank=rank, minimum_rows=fit['minimum_rows'], maximum_rows=fit['maximum_rows']))
        for key, case in new_cases.items():
            name, lane = key
            assert case['heldout_rows'] == held_records[name]['ntok'] == 50512
            assert case['heldout_hessian_sha256'] == held_records[name]['hessian_sha256']
            assert case['fit_hessian_sha256'] == fit_records[name]['hessian_sha256']
            assert case['fit_rows'] == fit_records[name]['ntok'] >= 330000
            prior = old_cases[key]
            for field in ('relative_rmse', 'reference_energy', 'error_energy'):
                assert math.isclose(case['rtn'][field], prior['rtn'][field], rel_tol=1e-8, abs_tol=1e-12), (rank, key, field)
            rows.append(dict(rank=rank, name=name, lane=lane,
                             rtn=case['rtn']['relative_rmse'], fit330=case['gptq']['relative_rmse'],
                             fit131=prior['gptq']['relative_rmse']))
    lanes = {}
    for lane, expected in (('w4', 768), ('fp8', 772)):
        cases = [r for r in rows if r['lane'] == lane]
        assert len(cases) == expected
        result = dict(sites=len(cases), median_fit330_rmse=statistics.median(r['fit330'] for r in cases))
        for control in ('rtn', 'fit131'):
            ratios = [r['fit330'] / r[control] for r in cases]
            result['versus_' + control] = dict(
                median_paired_rmse_ratio=statistics.median(ratios),
                minimum_paired_rmse_ratio=min(ratios), maximum_paired_rmse_ratio=max(ratios),
                improved=sum(r < 1 for r in ratios), worsened=sum(r > 1 for r in ratios),
                equal=sum(r == 1 for r in ratios))
        lanes[lane] = result
    return dict(scope='Weight-packing projection error on fixed held-out real-input statistics; not language quality.',
                historical_comparison='Same checkpoint, held-out Hessian bytes and RTN energies; additional fit inputs change both count and mixture.',
                source_sha=(root / 'fleet330-source.sha').read_text().strip(), coverage=coverage, lanes=lanes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE / 'fleet-20260920b')
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / 'projection-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
