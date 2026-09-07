"""Fixed FP8 control/candidate diagnostic; completion never admits serving."""
import hashlib
import json
import math
from pathlib import Path

CASES = ((4096, True), (8192, False), (6144, False))
SEED_OFFSETS = (0, 104729, 209759)
TRIALS = 8
MARKER = 'MOE_M64_FP8_DIAGNOSTIC_COMPLETE'
FIELDS = ('error_l2', 'error_peak', 'noise_l2', 'noise_peak', 'limit_l2', 'limit_peak')


def plan():
    return [(rows, skew, 9211+rows+offset, trial)
            for rows, skew in CASES for offset in SEED_OFFSETS for trial in range(TRIALS)]


def order(trial):
    return ('control', 'candidate') if trial % 2 == 0 else ('candidate', 'control')


def pair_summary(control, candidate, *, rank, row_offset=0):
    """Keep all failing row evidence; count intersections without a verdict."""
    if not control or len(control) != len(candidate):
        raise ValueError('equal nonempty row comparisons required')
    def bad(row):
        return (not row['finite'] or any(not math.isfinite(row[k]) for k in FIELDS)
                or row['error_l2'] > row['limit_l2'] or row['error_peak'] > row['limit_peak'])
    cb = {i for i, row in enumerate(control) if bad(row)}
    ab = {i for i, row in enumerate(candidate) if bad(row)}
    def safe(row):
        return {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in row.items()}
    failures = [dict(row=row_offset+i, control=safe(control[i]), candidate=safe(candidate[i]),
                     control_bad=i in cb, candidate_bad=i in ab) for i in sorted(cb | ab)]
    return dict(rank=rank, rows=len(control), row_offset=row_offset,
                control_bad=len(cb), candidate_bad=len(ab), both_bad=len(cb & ab),
                candidate_only=len(ab-cb), control_only=len(cb-ab), failures=failures,
                finite=all(row['finite'] and all(math.isfinite(row[k]) for k in FIELDS)
                           for row in control+candidate))


def row_metrics(torch, value, baseline, repeat):
    """Exactly the original gate's row norms, floors and repeat multiplier."""
    a, b, r = (v.float() for v in (value, baseline, repeat))
    finite = torch.isfinite(a).all(dim=1) & torch.isfinite(b).all(dim=1) & torch.isfinite(r).all(dim=1)
    norm = b.norm(dim=1).clamp_min(1e-6)
    peak = b.abs().amax(dim=1).clamp_min(1e-6)
    error = (a-b).norm(dim=1)/norm
    worst = (a-b).abs().amax(dim=1)/peak
    noise = (r-b).norm(dim=1)/norm
    npeak = (r-b).abs().amax(dim=1)/peak
    l2_limit = torch.maximum(3*noise, torch.full_like(noise, .02))
    peak_limit = torch.maximum(3*npeak, torch.full_like(npeak, .04))
    values = torch.stack((error, worst, noise, npeak, l2_limit, peak_limit), dim=1).cpu().tolist()
    return [dict(zip(FIELDS, row), finite=bool(ok)) for row, ok in zip(values, finite.cpu().tolist())]


def completion(records, provenance):
    if [(r['rows'], r['skew'], r['seed'], r['trial']) for r in records] != plan():
        raise ValueError('missing, duplicate or reordered diagnostic trials')
    groups = {}
    for record in records:
        if tuple(record['order']) != order(record['trial']):
            raise ValueError('alternating independent control/candidate order required')
        for phase in ('transport', 'local'):
            comparisons = record[phase]
            if [v['rank'] for v in comparisons] != list(range(4)):
                raise ValueError('all four ranks required')
            expected = record['rows']//4 if phase == 'transport' else record['rows']
            if any(v['rows'] != expected for v in comparisons):
                raise ValueError('wrong compared row coverage')
            key = (record['rows'], record['skew'], record['seed'], phase)
            group = groups.setdefault(key, dict(rows=key[0], skew=key[1], seed=key[2], phase=phase,
                control_bad=0, candidate_bad=0, candidate_only=0, control_only=0, both_bad=0, finite=True))
            for value in comparisons:
                for field in ('control_bad', 'candidate_bad', 'candidate_only', 'control_only', 'both_bad'):
                    group[field] += value[field]
                group['finite'] &= value['finite']
    return dict(verdict=MARKER, serving_gate=False, numerical_acceptance=False,
                trials=len(records), thresholds=dict(l2=.02, peak=.04, repeat_multiplier=3),
                grouping='descriptive row-trial counts; rows and reused trials are not independent experiments',
                groups=list(groups.values()), provenance=provenance,
                diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def run(*, torch, rank, provenance, reports, require, case_factory):
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(all(v == digest for v in reports(digest)), 'diagnostic source differs across ranks')
    records = []
    for rows, skew in CASES:
        for offset in SEED_OFFSETS:
            seed = 9211+rows+offset
            call, local_call, unchanged = case_factory(rows, skew, seed)
            for trial in range(TRIALS):
                record = dict(kind='MOE_M64_FP8_DIAGNOSTIC_TRIAL', rows=rows, skew=skew,
                    seed=seed, trial=trial, order=list(order(trial)), m64_admitted=rows >= 6144)
                for phase, invoke in (('transport', call), ('local', local_call)):
                    baseline = invoke(False)
                    repeat = invoke(False)
                    values = {arm: invoke(arm == 'candidate') for arm in order(trial)}
                    torch.cuda.synchronize()
                    pair = pair_summary(row_metrics(torch, values['control'], baseline, repeat),
                        row_metrics(torch, values['candidate'], baseline, repeat), rank=rank,
                        row_offset=rank*(rows//4) if phase == 'transport' else 0)
                    record[phase] = reports(pair)
                require(unchanged(), 'diagnostic input was modified')
                records.append(record)
                if rank == 0:
                    print(json.dumps(record, allow_nan=False), flush=True)
    result = completion(records, provenance)
    if rank == 0:
        print(json.dumps(result, allow_nan=False), flush=True)
    return result
