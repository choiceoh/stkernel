"""Compare calibration sizes on exactly the same validation statistics and weights."""
import argparse
import json
from pathlib import Path
import statistics


def ratios(values):
    return dict(sites=len(values), median_ratio=statistics.median(values),
                minimum_ratio=min(values), maximum_ratio=max(values),
                improved=sum(v < 1 for v in values), worsened=sum(v > 1 for v in values))


def summarize(root):
    arms, identities = {}, {}
    for arm, minimum in (("B131pack", 131072), ("B240pack", 240490), ("B330pack", 330000)):
        cases = {}
        for rank in range(4):
            score = json.loads((root / f"{arm}-projection-validation-rank{rank}.json").read_bytes())
            audit = json.loads((root / f"{arm}-audit-rank{rank}.json").read_bytes())
            validation = json.loads((root / f"validation-audit-rank{rank}.json").read_bytes())
            assert score['rank'] == audit['rank'] == validation['rank'] == rank
            assert score['weights_id'] == audit['weights_id'] == validation['weights_id']
            assert identities.setdefault(rank, score['weights_id']) == score['weights_id']
            assert audit['serving_gptq_verified'] and audit['minimum_rows'] >= minimum
            fitted = {r['name']: r for r in audit['records']}
            held = {r['name']: r for r in validation['records']}
            assert len(score['cases']) == 385
            for r in score['cases']:
                key = (rank, r['name'], r['lane'])
                assert key not in cases
                assert r['heldout_rows'] == held[r['name']]['ntok'] == 55441
                assert r['heldout_hessian_sha256'] == held[r['name']]['hessian_sha256']
                assert r['fit_hessian_sha256'] == fitted[r['name']]['hessian_sha256']
                assert r['fit_rows'] == fitted[r['name']]['ntok'] >= minimum
                cases[key] = r
        arms[arm] = cases
    first = arms['B131pack']
    for arm, cases in arms.items():
        assert cases.keys() == first.keys()
        for key, r in cases.items():
            control = first[key]
            assert r['heldout_hessian_sha256'] == control['heldout_hessian_sha256']
            for field in ('error_energy', 'reference_energy', 'relative_rmse'):
                a, b = r['rtn'][field], control['rtn'][field]
                assert abs(a-b) <= 1e-10 * max(abs(a), abs(b), 1e-30), 'RTN control changed across sizes'
    summary = {}
    for arm, cases in arms.items():
        summary[arm] = {}
        for lane, expected in (('w4', 768), ('fp8', 772)):
            keys = [key for key in cases if key[2] == lane]
            assert len(keys) == expected
            summary[arm][lane] = dict(
                versus_rtn=ratios([cases[k]['gptq']['relative_rmse'] / cases[k]['rtn']['relative_rmse'] for k in keys]),
                versus_131k=ratios([cases[k]['gptq']['relative_rmse'] / first[k]['gptq']['relative_rmse'] for k in keys]),
                fit_rows=sorted({cases[k]['fit_rows'] for k in keys}))
    incremental = {}
    for lane in ('w4', 'fp8'):
        keys = [k for k in first if k[2] == lane]
        incremental[lane] = ratios([arms['B330pack'][k]['gptq']['relative_rmse'] /
                                   arms['B240pack'][k]['gptq']['relative_rmse'] for k in keys])
    return dict(scope='Actual GPTQ projection errors on fixed independent validation statistics; not language accuracy.',
                arms=summary, ratio_330k_to_240k=incremental, weights_id_by_rank=identities,
                inputs='Nested original training prefix; 240K-to-330K adds real conversations and changes the source mix.',
                adoption='No automatic promotion; consumer quality, acceptance and speed remain separate gates.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
