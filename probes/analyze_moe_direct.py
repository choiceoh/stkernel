#!/usr/bin/env python3
"""Summarize balanced, same-process MoE A/B; never infer serving speed."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(report):
    assert report['status'] == 'PASS'
    assert report['gates']
    rows = []
    for shape, states in report['samples_us'].items():
        for state, arms in states.items():
            b, a = arms['baseline'], arms['candidate']
            assert len(a) == len(b) and len(a) >= 8 and len(a) % 2 == 0
            assert all(v > 0 for v in (*a, *b))
            rows.append(dict(shape=shape, cache=state, pairs=len(a), baseline_us=median(b),
                candidate_us=median(a), latency_reduction_pct=100*(1-median(a)/median(b)),
                faster_pairs=sum(x<y for x,y in zip(a,b)),
                baseline_first_pct=100*(1-median(a[::2])/median(b[::2])),
                candidate_first_pct=100*(1-median(a[1::2])/median(b[1::2]))))
    return dict(variant=report['candidate'], gates=len(report['gates']), rows=rows,
                scope='Synthetic single-kernel A/B; no measured serving step or output improvement')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    results = [summarize(json.loads((args.directory/(variant+'.json')).read_text()))
               for variant in ('pair','vector')]
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
