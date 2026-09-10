"""Same-boot decode preparation execution proof; no inference or serving imports."""
import hashlib
import json
import os
import re
import stat

LIMIT = 64 * 2**20
PLAN = re.compile(r'\[prep-fused\] plan built: (.*)')
STATS = re.compile(r'\[prep-fused\] (on|shadow): fused_steps=(\d+) stock_steps=(\d+) '
                   r'checks ok=(\d+) drift=(\d+)(?:\s|$)')

# Only fixed diagnostic codes leave this module; exception text may contain paths.
REJECTION_CODES = {
    'exclusive context missing': 'exclusive_context_missing',
    'preparation must be armed or shadow': 'unsupported_mode',
    'actual launch differs': 'launch_mismatch',
    'invalid preparation log': 'invalid_log',
    'preparation log replaced or truncated': 'log_replaced_or_truncated',
    'log prefix changed or did not advance': 'log_prefix_mismatch',
    'preparation drift, failure or fallback': 'preparation_failure',
    'plan missing': 'plan_missing',
    'plan differs': 'plan_mismatch',
    'invalid verification checkpoint': 'invalid_checkpoint',
    'verification counters reset': 'counter_reset',
    'no verified execution during onepass': 'no_fresh_checkpoint',
    'decode optimization launch differs': 'decode_opt_launch_mismatch',
    'decode optimization checkpoint differs': 'decode_opt_checkpoint_mismatch',
    'no verified decode optimization during onepass': 'decode_opt_no_fresh_checkpoint',
}


def _read(path):
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= LIMIT:
            raise ValueError('invalid preparation log')
        raw = stream.read(before.st_size)
        after = os.stat(path)
        if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or len(raw) != before.st_size or after.st_size < before.st_size):
            raise ValueError('preparation log replaced or truncated')
        return raw, before


def log_prefix(path):
    try:
        raw, info = _read(path)
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    device=info.st_dev, inode=info.st_ino)
    except (OSError, ValueError):
        return None


def _decode_opt_checkpoints(text, preparation_checkpoints):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('decode optimization checkpoint differs')
            result[key] = value
        return result

    prefix = '[prep-decode-opt] USED '
    records = []
    for line in text.splitlines():
        if prefix not in line:
            continue
        record = json.loads(line.split(prefix, 1)[1], object_pairs_hook=pairs)
        expected = dict(version=1, num_reqs=1, q=6, live_snapshot_clones_elided=True,
                        first_plan_check_passed=True)
        counters = ('fused_steps', 'live_diff_checks')
        if (type(record) is not dict or set(record) != set(expected) | set(counters)
                or any(type(record.get(key)) is not type(value) or record[key] != value
                       for key, value in expected.items())
                or any(type(record.get(key)) is not int or record[key] <= 0 for key in counters)
                or record['live_diff_checks'] > record['fused_steps']
                or not any(check['fused_steps'] == record['fused_steps']
                           and check['checks_ok'] >= record['live_diff_checks']
                           for check in preparation_checkpoints)):
            raise ValueError('decode optimization checkpoint differs')
        if records and any(record[key] < records[-1][key] for key in counters):
            raise ValueError('decode optimization checkpoint differs')
        records.append(record)
    if not records:
        raise ValueError('no verified decode optimization during onepass')
    return records


def evidence(context, path, *, decode_opt=False):
    result = dict(verdict='REJECTED', scope='same-boot preparation log checkpoints during onepass; '
                  'not fixed-request counters or a performance verdict')
    try:
        if not isinstance(context, dict) or context.get('exclusive') is not True:
            raise ValueError('exclusive context missing')
        expected = context['expected_mode']
        if expected not in ('1', 'shadow'):
            raise ValueError('preparation must be armed or shadow')
        mode = 'on' if expected == '1' else 'shadow'
        before, after = context['launch_before'], context['launch_after']
        if (not isinstance(before, dict) or before != after
                or before.get('boot_id') != context['boot_id']
                or re.fullmatch(r'[0-9a-f]{64}\|[^|]+', str(before.get('boot_id'))) is None
                or re.fullmatch(r'sha256:[0-9a-f]{64}', str(before.get('image'))) is None
                or any(re.fullmatch(r'[0-9a-f]{64}', str(before.get(k))) is None
                       for k in ('command_sha256', 'config_sha256'))
                or before.get('method') != 'dflash'
                or type(before.get('node_rank')) is not int or before['node_rank'] != 0
                or type(before.get('num_speculative_tokens')) is not int
                or not 1 <= before['num_speculative_tokens'] <= 7
                or before.get('environment_spec_k') != str(before['num_speculative_tokens'])
                or before.get('preparation_mode') != expected
                or before.get('preparation_kernel') != 'cuda'
                or before.get('shadow_every') != '1'
                or before.get('selfcheck_every') != '64'):
            raise ValueError('actual launch differs')
        result.update(mode=mode, launch={key: before[key] for key in (
            'method', 'num_speculative_tokens', 'config_sha256', 'command_sha256',
            'node_rank', 'boot_id', 'image', 'environment_spec_k', 'preparation_mode',
            'preparation_kernel', 'shadow_every', 'selfcheck_every')})
        if decode_opt:
            if (expected != '1' or before.get('ep_decode_opt') != '1'
                    or before['num_speculative_tokens'] != 5):
                raise ValueError('decode optimization launch differs')
            result['launch']['ep_decode_opt'] = before['ep_decode_opt']
        raw, info = _read(path)
        prefix = context['log_prefix']
        if (not isinstance(prefix, dict) or type(prefix.get('bytes')) is not int
                or not 0 <= prefix['bytes'] < len(raw)
                or (prefix.get('device'), prefix.get('inode')) != (info.st_dev, info.st_ino)
                or hashlib.sha256(raw[:prefix['bytes']]).hexdigest() != prefix.get('sha256')):
            raise ValueError('log prefix changed or did not advance')
        result.update(log_prefix={key: prefix[key] for key in ('bytes', 'sha256', 'device', 'inode')},
                      log_after_sha256=hashlib.sha256(raw).hexdigest(), log_after_bytes=len(raw))
        text = raw.decode('utf-8', 'strict')
        lines = [line for line in text.splitlines() if '[prep-fused]' in line]
        if any(re.search(r'DISARM|DRIFT|plan build failed|fused prepare failed|'
                         r'CUDA kernel unavailable|extension has no run_prep', line) for line in lines):
            raise ValueError('preparation drift, failure or fallback')
        plans = [m[1] for line in lines if (m := PLAN.search(line))]
        expected_fields = dict(mode=mode, kernel='cuda', q=str(before['num_speculative_tokens']+1),
                               shadow_every='1', selfcheck_every='64')
        if not plans:
            raise ValueError('plan missing')
        for plan in plans:
            pairs = re.findall(r'\b(mode|kernel|q|shadow_every|selfcheck_every)=([a-z0-9]+)', plan)
            if len(pairs) != len(expected_fields) or dict(pairs) != expected_fields:
                raise ValueError('plan differs')
        checkpoints = []
        for line in raw[prefix['bytes']:].decode('utf-8', 'strict').splitlines():
            if (match := STATS.search(line)) is None:
                continue
            current_mode, fused, stock, checks, drift = match.groups()
            fused, stock, checks, drift = map(int, (fused, stock, checks, drift))
            if (current_mode != mode or fused <= 0 or not 0 < checks <= fused or drift != 0
                    or mode == 'shadow' and checks != fused):
                raise ValueError('invalid verification checkpoint')
            current = dict(fused_steps=fused, stock_steps=stock, checks_ok=checks, drift=drift)
            if checkpoints and any(current[k] < checkpoints[-1][k] for k in current):
                raise ValueError('verification counters reset')
            checkpoints.append(current)
        if not checkpoints:
            raise ValueError('no verified execution during onepass')
        if decode_opt:
            optimized = _decode_opt_checkpoints(
                raw[prefix['bytes']:].decode('utf-8', 'strict'), checkpoints)
            result['decode_opt_last_checkpoint'] = optimized[-1]
            result['decode_opt_checkpoint_count'] = len(optimized)
        result.update(verdict='PASS', last_checkpoint=checkpoints[-1],
                      checkpoint_count=len(checkpoints))
    except (KeyError, TypeError, ValueError, OSError, UnicodeError) as error:
        result['reason'] = type(error).__name__
        result['reason_code'] = REJECTION_CODES.get(str(error), 'invalid_evidence')
    return result
