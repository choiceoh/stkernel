"""Read retained device iteration evidence; never boots or contacts an engine."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def summary(values):
    return dict(samples=len(values), sum_us=sum(values), mean_us=statistics.mean(values),
                median_us=statistics.median(values), min_us=min(values), max_us=max(values))


def analyze(path):
    stages = defaultdict(list)
    iterations = []
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for raw in source:
            digest.update(raw)
            row = json.loads(raw)
            if row.get('rank') != 0 or row.get('kind') != 'gpu_iteration' or row.get('phase') != 'decode':
                continue
            iterations.append(row['duration_us'])
            for name, value in row.get('stages_us', {}).items():
                stages[name].append(value)
    return dict(source=str(path), sha256=digest.hexdigest(), rank=0,
                scope='recorded device body and TP4 stop agreement; excludes client/tokenizer/HTTP time',
                iteration=summary(iterations), device_step_s=1e6*len(iterations)/sum(iterations),
                stages={key: summary(value) for key,value in stages.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('latency', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps([analyze(path) for path in args.latency], indent=2))
