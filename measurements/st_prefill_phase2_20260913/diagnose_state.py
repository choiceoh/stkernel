"""Compare already committed host state at the same dispatch on four ranks."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile


def compare(rows):
    steps = {}
    ranks, unobserved = set(), 0
    for row in rows:
        if row.get('kind') != 'host_step':
            continue
        rank = row['rank']
        ranks.add(rank)
        if not row.get('host_state') or not row.get('dispatch'):
            unobserved += 1
            continue
        key = (row['request_token'], row['step'])
        state = {k: row[k] for k in ('phase', 'rows', 'tokens', 'dispatch', 'host_state')}
        prior = steps.setdefault(key, {}).setdefault(rank, state)
        if prior != state:
            raise ValueError('conflicting duplicate rank/dispatch record')
    checked = 0
    for (token, step), states in sorted(steps.items()):
        if set(states) != {0, 1, 2, 3}:
            continue
        checked += 1
        if any(state != states[0] for state in states.values()):
            return dict(status='divergent_host_state', token=token, step=step,
                        compared_dispatches=checked, ranks=states)
    return dict(status='no_difference_in_observed_dispatches' if checked else 'insufficient_state_records',
                compared_dispatches=checked, observed_ranks=sorted(ranks),
                dispatches_without_state=unobserved,
                partial_dispatches=sum(set(value) != {0, 1, 2, 3} for value in steps.values()),
                note='Host observations do not prove device equality or completion.')


def read_lines(stream):
    for line in io.TextIOWrapper(stream):
        if line.strip():
            yield json.loads(line)


def records(root):
    archives = sorted(root.glob('rank*-latency.tar.gz'))
    if archives:
        for path in archives:
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith('/latency.jsonl'):
                        with archive.extractfile(member) as stream:
                            yield from read_lines(stream)
        return
    for path in sorted(root.glob('dumps/onepass-latency/*/rank-*/latency.jsonl*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream:
            yield from read_lines(stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(records(args.arm)), indent=2))
