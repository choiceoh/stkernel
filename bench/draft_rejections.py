"""Summarize the recorded greedy decisions without mixing requests or ranks."""
from collections import Counter


def summarize(report):
    ranks = []
    for rank in report.get('ranks', []):
        owners, requests = {}, {}
        for row in rank.get('rows', []):
            if row.get('kind') == 'request' and row.get('operation') == 'admit':
                owners[row['row']] = row['request_id']
            elif row.get('kind') == 'draft_rejection':
                seq = row['seq']
                request = owners.get(seq)
                key = (request, seq)
                item = requests.setdefault(key, dict(request_id=request, seq=seq, reasons=Counter(), positions=Counter()))
                item['reasons'][row['reason']] += 1
                if row['reason'] in ('candidate_miss', 'selector_miss'):
                    item['positions'][row['accepted_prefix'] + 1, row['reason']] += 1
        values = []
        for item in requests.values():
            values.append(dict(request_id=item['request_id'], seq=item['seq'], reasons=dict(item['reasons']),
                first_rejection_positions=[dict(position=p, reason=r, count=count)
                                           for (p, r), count in sorted(item['positions'].items())]))
        ranks.append(dict(rank=rank['rank'], requests=values, complete=rank.get('complete') is True,
                          errors=list(rank.get('errors', []))))
    return dict(schema=1, scope='greedy; policy-modified and output-boundary rows are separate; ranks are not pooled',
                recorded=any(r['requests'] for r in ranks),
                complete=bool(ranks) and all(r['complete'] and not r['errors'] for r in ranks), ranks=ranks)
