"""Frozen actual partial/packet replay; collection never approves serving."""
import base64
import hashlib
import io
import json
from pathlib import Path

from glm53_moe_m64_fp8_diagnostic import CASES, SEED_OFFSETS, TRIALS, order, plan, pair_summary, row_metrics

MARKER = 'MOE_M64_FP8_TRACE_COMPLETE'
ARMS = ('baseline', 'repeat', 'control', 'candidate')
PHASES = ('transport', 'partial', 'bf16')
MAX_TRACE_ROWS = 128
RAW_FIELDS = ('raw_error_l2', 'raw_error_peak', 'raw_noise_l2', 'raw_noise_peak', 'reference_norm', 'reference_peak')


def metrics(torch, value, baseline, repeat):
    values = row_metrics(torch, value, baseline, repeat)
    a, b, r = (v.float() for v in (value, baseline, repeat))
    raw = torch.stack(((a-b).norm(dim=1), (a-b).abs().amax(dim=1),
        (r-b).norm(dim=1), (r-b).abs().amax(dim=1), b.norm(dim=1).clamp_min(1e-6),
        b.abs().amax(dim=1).clamp_min(1e-6)), dim=1).cpu().tolist()
    for value, row in zip(values, raw):
        value.update(zip(RAW_FIELDS, row))
    return values


def packet_rows(torch, packed, row_ids, *, rows, payload_bytes):
    """Select actual outgoing value bytes and FP32 scales for global rows."""
    local_rows = rows//4
    indices = torch.tensor(row_ids, device=packed.device, dtype=torch.long)
    destinations = indices//local_rows
    offsets = indices % local_rows
    values = packed.reshape(4, payload_bytes)[destinations[:, None],
        offsets[:, None]*4096+torch.arange(4096, device=packed.device)]
    scales = packed.view(torch.float32).reshape(4, payload_bytes//4)[destinations[:, None],
        local_rows*4096//4+offsets[:, None]*2+torch.arange(2, device=packed.device)]
    return values, scales


def replay(torch, h, partial):
    """Use the unchanged production pack/exchange/unpack on a frozen partial.

    The second unpack retains the same FP32 sum before the ordinary BF16 store.
    A separate call to the unmodified public helper is compared by the caller.
    """
    from vllm.distributed import get_tp_group
    rows = partial.shape[0]
    local_n = rows*4096//4
    local_blocks = local_n//2048
    payload_bytes = h._payload_bytes(local_n)
    packed = torch.empty(payload_bytes*4, device=partial.device, dtype=torch.uint8)
    received = torch.empty_like(packed)
    h._pack_rs_payload[(local_blocks*4,)](partial, packed.view(torch.float8_e4m3fn),
        packed.view(torch.float32), N=partial.numel(), LOCAL_N=local_n,
        PAYLOAD_BYTES=payload_bytes, BLOCK=2048)
    torch.distributed.all_to_all_single(received, packed, group=get_tp_group().device_group)
    output = torch.empty((rows//4, 4096), device=partial.device, dtype=torch.bfloat16)
    sum32 = torch.empty_like(output, dtype=torch.float32)
    for out in (output, sum32):
        h._unpack_sum_payload[(local_blocks,)](received.view(torch.float8_e4m3fn),
            received.view(torch.float32), out, LOCAL_N=local_n,
            PAYLOAD_BYTES=payload_bytes, TP=4, BLOCK=2048)
    return dict(output=output, sum32=sum32, packed=packed, payload_bytes=payload_bytes)


def encode_arrays(arrays):
    import numpy as np
    if any(v.dtype.hasobject for v in arrays.values()):
        raise ValueError('object arrays are not trace evidence')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    data = buffer.getvalue()
    return dict(encoding='npz-base64-no-pickle', bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), data_b64=base64.b64encode(data).decode())


def decode_arrays(payload):
    import numpy as np
    if payload['encoding'] != 'npz-base64-no-pickle':
        raise ValueError('unknown trace encoding')
    data = base64.b64decode(payload['data_b64'], validate=True)
    if len(data) != payload['bytes'] or hashlib.sha256(data).hexdigest() != payload['sha256']:
        raise ValueError('trace length/hash mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def trace_arrays(torch, captures, replays, native, row_ids, *, rank, rows):
    import numpy as np
    indices = torch.tensor(row_ids, device=captures['baseline']['partial'].device, dtype=torch.long)
    own_rows = [r for r in row_ids if rank*(rows//4) <= r < (rank+1)*(rows//4)]
    owned = torch.tensor([r-rank*(rows//4) for r in own_rows], device=indices.device, dtype=torch.long)
    arrays = dict(row_ids=np.asarray(row_ids, dtype=np.int64), owned_row_ids=np.asarray(own_rows, dtype=np.int64))
    for arm in ARMS:
        capture, replayed = captures[arm], replays[arm]
        q, scales = packet_rows(torch, replayed['packed'], row_ids, rows=rows,
                                payload_bytes=replayed['payload_bytes'])
        arrays[arm+'_partial_bits'] = capture['partial'].index_select(0, indices).view(torch.int16).cpu().numpy()
        arrays[arm+'_fp8_bytes'] = q.cpu().numpy()
        arrays[arm+'_scales'] = scales.cpu().numpy()
        arrays[arm+'_sum32'] = replayed['sum32'].index_select(0, owned).cpu().numpy()
        arrays[arm+'_output_bits'] = capture['output'].index_select(0, owned).view(torch.int16).cpu().numpy()
        arrays[arm+'_bf16_bits'] = native[arm].index_select(0, owned).view(torch.int16).cpu().numpy()
    return arrays


def completion(records, provenance):
    if [(r['rows'], r['skew'], r['seed'], r['trial']) for r in records] != plan():
        raise ValueError('exact ordered 72-trial trace coverage required')
    groups = {}
    for record in records:
        rows = record['rows']
        if tuple(record['order']) != order(record['trial']):
            raise ValueError('alternating control/candidate order required')
        for phase in PHASES:
            values = record[phase]
            if [v['rank'] for v in values] != list(range(4)):
                raise ValueError('all four comparison ranks required')
            if any(v['rows'] != (rows if phase == 'partial' else rows//4) for v in values):
                raise ValueError('wrong compared row coverage')
            key = (rows, record['skew'], record['seed'], phase)
            group = groups.setdefault(key, dict(rows=rows, skew=key[1], seed=key[2], phase=phase,
                candidate_bad=0, control_bad=0, finite=True))
            for value in values:
                group['candidate_bad'] += value['candidate_bad']
                group['control_bad'] += value['control_bad']
                group['finite'] &= value['finite']
        if [v['rank'] for v in record['replay']] != list(range(4)):
            raise ValueError('all four replay ranks required')
        if any(set(v['arms']) != set(ARMS) for v in record['replay']):
            raise ValueError('every captured arm must be replayed')
        if [v['rank'] for v in record['payloads']] != list(range(4)):
            raise ValueError('all four row payloads required')
        for payload in record['payloads']:
            if payload['row_ids'] != record['row_ids'] or not 0 < payload['bytes'] or len(payload['sha256']) != 64:
                raise ValueError('trace row coverage/hash missing')
    return dict(verdict=MARKER, trials=len(records), serving_gate=False, numerical_acceptance=False,
        thresholds=dict(l2=.02, peak=.04, repeat_multiplier=3), groups=list(groups.values()),
        replay_all_equal=all(arm['original_equal'] and arm['second_equal'] and arm['sum32_store_equal']
            and arm['partial_unchanged'] and arm['gather_unchanged'] and arm['source_unchanged']
            for r in records for rank in r['replay'] for arm in rank['arms'].values()),
        provenance=provenance, trace_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).with_name('glm53_moe_m64_fp8_diagnostic.py').read_bytes()).hexdigest(),
        grouping='descriptive row-trial counts; no independent-sample or numerical acceptance claim')


def verify_logs(directory):
    """Check every process payload against rank 0's collective metadata."""
    trials, finals, payloads = [], [], {}
    for rank in range(4):
        values = []
        for line in (Path(directory)/f'fp8-v3-rank-{rank}.log').read_text().splitlines():
            if not line.startswith('{'):
                continue
            record = json.loads(line)
            if rank == 0 and record.get('kind') == 'MOE_M64_FP8_TRACE_TRIAL':
                trials.append(record)
            if rank == 0 and record.get('verdict') == MARKER:
                finals.append(record)
            if record.get('kind') == 'MOE_M64_FP8_TRACE_ROWS':
                data = base64.b64decode(record['data_b64'], validate=True)
                if record['rank'] != rank or len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                    raise ValueError('payload rank/length/hash mismatch')
                values.append({k: v for k, v in record.items() if k != 'data_b64'})
        if [(v['rows'], v['skew'], v['seed'], v['trial']) for v in values] != plan():
            raise ValueError('every rank must retain all 72 row payloads')
        payloads[rank] = values
    if len(finals) != 1 or completion(trials, finals[0]['provenance']) != finals[0]:
        raise ValueError('trace completion mismatch')
    if any(trial['payloads'][rank] != payloads[rank][i] for i, trial in enumerate(trials) for rank in range(4)):
        raise ValueError('collective payload metadata mismatch')
    return finals[0]


def run(*, torch, h, rank, provenance, reports, require, case_factory):
    digest = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__), Path(__file__).with_name('glm53_moe_m64_fp8_diagnostic.py')))
    require(all(v == digest for v in reports(digest)), 'trace source differs across ranks')
    bit_equal = lambda a, b: torch.equal(a.view(torch.int16), b.view(torch.int16))
    records = []
    for rows, skew in CASES:
        for offset in SEED_OFFSETS:
            seed = 9211+rows+offset
            call, _, unchanged = case_factory(rows, skew, seed)
            for trial in range(TRIALS):
                captures = dict(baseline=call(False), repeat=call(False))
                captures.update({arm: call(arm == 'candidate') for arm in order(trial)})
                record = dict(kind='MOE_M64_FP8_TRACE_TRIAL', rows=rows, skew=skew,
                    seed=seed, trial=trial, order=list(order(trial)))
                def compare(values, phase):
                    pair = pair_summary(metrics(torch, values['control'], values['baseline'], values['repeat']),
                        metrics(torch, values['candidate'], values['baseline'], values['repeat']), rank=rank,
                        row_offset=0 if phase == 'partial' else rank*(rows//4))
                    record[phase] = reports(pair)
                compare({a: c['output'] for a, c in captures.items()}, 'transport')
                compare({a: c['partial'] for a, c in captures.items()}, 'partial')
                replays, native, checks = {}, {}, {}
                for arm in ARMS:
                    c = captures[arm]
                    frozen = c['partial'].clone()
                    replays[arm] = replay(torch, h, c['partial'])
                    second = h.prefill_reduce_scatter(c['partial'])
                    native[arm] = torch.empty_like(c['output'])
                    h._check(c['partial']).reduce_scatter(native[arm], c['partial'])
                    checks[arm] = dict(original_equal=bit_equal(c['output'], replays[arm]['output']),
                        second_equal=bit_equal(second, replays[arm]['output']),
                        sum32_store_equal=bit_equal(replays[arm]['output'], replays[arm]['sum32'].to(torch.bfloat16)),
                        partial_unchanged=bit_equal(c['partial'], frozen),
                        gather_unchanged=c['gather_unchanged'], source_unchanged=c['source_unchanged'])
                record['replay'] = reports(dict(rank=rank, arms=checks))
                compare(native, 'bf16')
                row_ids = sorted({f['row'] for phase in PHASES for pair in record[phase] for f in pair['failures']})
                record['row_ids'] = row_ids
                if len(row_ids) > MAX_TRACE_ROWS:
                    if rank == 0:print(json.dumps(dict(record, kind='MOE_M64_FP8_TRACE_INCOMPLETE')), flush=True)
                    require(False, 'failed-row trace exceeds fixed 128-row budget; no failures discarded')
                payload = dict(kind='MOE_M64_FP8_TRACE_ROWS', rows=rows, skew=skew, seed=seed, trial=trial,
                    rank=rank, row_ids=row_ids, **encode_arrays(trace_arrays(torch, captures, replays, native,
                        row_ids, rank=rank, rows=rows)))
                print(json.dumps(payload, allow_nan=False), flush=True)
                record['payloads'] = reports({k: v for k, v in payload.items() if k != 'data_b64'})
                require(unchanged(), 'trace input was modified')
                records.append(record)
                if rank == 0:print(json.dumps(record, allow_nan=False), flush=True)
    result = completion(records, provenance)
    if rank == 0:print(json.dumps(result, allow_nan=False), flush=True)
    return result
