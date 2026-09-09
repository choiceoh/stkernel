#!/usr/bin/env python3
"""Passive baseline-only evidence, preserving failed runtime checks unchanged.

Deploy this file outside the frozen repository. Until the exact reservation
owns its boot hold, only reservation/holder/proc files are read. There are no
requests other than GET /health, GPU imports, launches, or lifecycle actions.
Collection COMPLETE means sealed evidence exists, not that runtime tests pass.
"""
import argparse
import base64
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

SCHEMA = 'sf6-baseline-observation-v1'


def variant(args):
    """Keep the original frozen API calls unchanged unless explicitly selected."""
    unpack = getattr(args, 'sf6_unpack', False)
    if type(unpack) is not bool:
        raise ValueError('sf6_unpack must be a boolean')
    if unpack:
        if args.port != 18000:
            raise ValueError('SF6 unpack baseline requires loopback port 18000')
        return dict(sf6_direct=False, sf6_unpack=True)
    return dict(sf6_direct=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def reservation(args, *, proc=Path('/proc')):
    """File-only identity gate; never import the fleet classifier while queued."""
    path = args.fleet_dir / 'pending' / (sha(args.session.encode()) + '.json')
    try:
        row = json.loads(path.read_text())
    except FileNotFoundError:
        return {'state': 'WAIT'}
    wanted = dict(session=args.session, ticket=args.ticket, pid=args.pid, start=args.start)
    if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in wanted.items()):
        raise ValueError('reservation session/ticket/PID/start changed')
    if row.get('state') in ('finished', 'cancelled', 'finishing'):
        return dict(state='TERMINAL', reservation=row)
    try:
        fields = (proc / str(args.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (OSError, IndexError) as exc:
        raise ValueError('reservation supervisor is unavailable') from exc
    if len(fields) <= 19 or fields[0] == 'Z' or fields[19] != args.start:
        raise ValueError('reservation supervisor PID/start is stale')
    try:
        holder = (args.fleet_dir / 'holder').read_text().strip().split('|')
    except FileNotFoundError:
        holder = []
    owned = (len(holder) == 7 and holder[:2] == [args.session, str(args.pid)]
             and holder[-1] == 'boot')
    if row.get('state') != 'running' or not owned:
        return dict(state='WAIT', reservation=row)
    started = row.get('started_at')
    if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
        raise ValueError('admitted reservation lacks started_at')
    return dict(state='GO', reservation=row, holder=holder)


def load_frozen(repo, expected_revision):
    """Called only after GO. Import the exact frozen observer and its helpers."""
    if not isinstance(expected_revision, str) or not re.fullmatch(r'[0-9a-f]{40}', expected_revision):
        raise ValueError('expected revision must be an exact 40-character commit SHA')
    sys.path.insert(0, str(repo / 'probes'))
    spec = importlib.util.spec_from_file_location('_frozen_baseline_observer',
                                                 repo / 'probes/observe_decode_next_onepass.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    revision, expected = module.frozen_manifest(repo)
    if revision != expected_revision:
        raise ValueError('baseline observer requires frozen source ' + expected_revision)
    return module, revision, expected


def boot_time(head):
    try:
        return datetime.fromisoformat(head['boot_id'].rsplit('|', 1)[1].replace('Z', '+00:00')).timestamp()
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError('container boot timestamp unavailable') from exc


def capture(args, module, reader, expected, go, phase, directory, before=None):
    """Keep validation errors without using them to suppress same-boot reads."""
    proof = module.proof
    selected = variant(args)
    ledger = args.out / 'records.raw.jsonl'
    entry_before = module.read_records(ledger, {args.name}).get(args.name)
    head_before = reader.head()
    owned_before = reservation(args)['state'] == 'GO'
    collection, validation, artifacts = [], [], {}
    if (not head_before or module.mode_from_metadata(head_before, **selected) != 'baseline'
            or head_before.get('image') != proof.IMAGE):
        collection.append('head is not the requested running baseline/image')
    elif boot_time(head_before) < go['reservation']['started_at']:
        collection.append('head predates the admitted clean boot')
    if phase == 'prepared':
        if not owned_before or entry_before is not None:
            collection.append('prepared capture requires owned hold and no completed record')
    elif (before is None or head_before != before['receipt']['head_after']
          or not entry_before or entry_before['record'].get('boot_id') != head_before.get('boot_id')):
        collection.append('after capture is not bound to the prepared head and completed record')
    ranks = reader.ranks('baseline', expected) if not collection else {}
    head_after = reader.head()
    owned_after = reservation(args)['state'] == 'GO'
    entry_after = module.read_records(ledger, {args.name}).get(args.name)
    if head_before != head_after:
        collection.append('head boot/image/configuration changed during capture')
    if phase == 'prepared' and (entry_after is not None or not owned_after):
        collection.append('record appeared or owned hold ended during prepared capture')
    if phase == 'runtime' and entry_before != entry_after:
        collection.append('completed record changed during after capture')
    if set(ranks) != set(module.HOSTS):
        collection.append('exactly four rank reports required')
    identities, boots, reports = set(), set(), {}
    directory.mkdir(parents=True)

    def retain(name, data):
        (directory / name).write_bytes(data)
        artifacts[name] = sha(data)

    for host in module.HOSTS:
        rank = ranks.get(host, {})
        report = rank.get('report', {})
        reports[host] = report
        retain(host + '.json', encode(report))
        try:
            errors = proof.validate_report(report, expected, **selected)
            validation.extend(host + ': ' + error for error in errors)
            if report.get('errors') != errors or rank.get('error', '') != '; '.join(errors):
                collection.append(host + ': collector failed or rank changed during snapshot: '
                                  + str(rank.get('error', report.get('errors'))))
            raw = base64.b64decode(rank['log_b64'], validate=True)
            retain(host + '.log', raw)
            parsed = (proof.parse_markers(raw.decode(errors='replace'), sf6_unpack=True)
                      if selected.get('sf6_unpack') else proof.parse_markers(raw.decode(errors='replace')))
            if report.get('log_sha256') != sha(raw) or report.get('markers') != parsed:
                collection.append(host + ': original log SHA/parser mismatch')
            wanted = proof.expected_knobs('baseline', **selected)
            if (report.get('source_sha256') != expected or report.get('image') != proof.IMAGE
                    or report.get('mode') != 'baseline'
                    or any(report.get(key) is not value for key, value in selected.items())
                    or report.get('running') is not True or report.get('oom_killed') is not False
                    or any(report.get('knobs', {}).get(k) != v for k, v in wanted.items())
                    or any(key not in report for key in proof._IMMUTABLE)):
                collection.append(host + ': incomplete or different source/runtime identity')
            if boot_time(report) < go['reservation']['started_at']:
                collection.append(host + ': rank predates admitted clean boot')
            if not report.get('host') or report['host'] in identities or report.get('boot_id') in boots:
                collection.append(host + ': duplicate or missing rank identity')
            identities.add(report.get('host'))
            boots.add(report.get('boot_id'))
            if host == 'srv2' and any(report.get(k) != (head_before or {}).get(k) for k in ('boot_id', 'image', 'knobs')):
                collection.append('head rank report differs from captured head')
            if host == 'srv2' and (proof.cli_option(report['serving_argv'], '--host') != '127.0.0.1'
                                  or proof.cli_option(report['serving_argv'], '--port') != str(args.port)):
                collection.append('head serving endpoint is not the requested loopback port')
            if before:
                original = before['reports'][host]
                validation.extend(host + ' comparison: ' + error for error in
                                  proof.compare_snapshots(original, report, **selected))
                if any(proof._identity(original, k) != proof._identity(report, k) for k in proof._IMMUTABLE):
                    collection.append(host + ': within-arm runtime identity changed')
        except (KeyError, TypeError, ValueError) as exc:
            collection.append(host + ': incomplete rank evidence: ' + str(exc))
    if phase == 'runtime' and entry_before:
        # Preserve the original complete record bytes, including their spacing.
        matches = [line.rstrip(b'\r\n') for line in ledger.read_bytes().splitlines(keepends=True)
                   if line.endswith(b'\n') and line.strip()
                   and sha(line.rstrip(b'\r\n')) == entry_before['sha256']]
        if len(matches) != 1:
            collection.append('original completed record bytes unavailable or duplicated')
        else:
            retain('record.raw.json', matches[0])
    receipt = dict(schema=SCHEMA, phase=phase, name=args.name, mode='baseline', **selected,
        source_commit=args.revision, collection_status='FAILED' if collection else 'COMPLETE',
        runtime_validation='FAIL' if validation else 'PASS', collection_errors=collection,
        validation_errors=validation, head_before=head_before, head_after=head_after,
        owned_before=owned_before, owned_after=owned_after,
        record_sha256=(entry_before or {}).get('sha256'),
        record_present_before=entry_before is not None, record_present_after=entry_after is not None,
        artifacts_sha256=artifacts, completed_at=time.time())
    (directory / 'receipt.json').write_bytes(encode(receipt))
    return dict(receipt=receipt, reports=reports, path=directory)


def observe(args):
    selected = variant(args)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / 'baseline-observer.json'
    if state_path.exists():
        raise ValueError('fresh baseline observation directory required')
    state = dict(schema=SCHEMA, name=args.name, mode='baseline', **selected,
        source_commit=args.revision, session=args.session, ticket=args.ticket,
        pid=args.pid, start=args.start, collection_status='WAITING', runtime_validation='NOT_COLLECTED',
        errors=[], phases={}, attempts=[], helper_sha256=sha(Path(__file__).read_bytes()))

    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_bytes(encode(state))
        temp.replace(state_path)

    deadline = time.monotonic() + args.timeout
    save()
    try:
        while time.monotonic() < deadline:
            go = reservation(args)
            if go['state'] == 'GO':
                break
            if go['state'] == 'TERMINAL':
                raise ValueError('reservation ended before admitted observation')
            time.sleep(1)
        else:
            raise ValueError('reservation admission deadline expired')
        state.update(collection_status='COLLECTING', admission=go)
        save()
        module, revision, expected = load_frozen(args.repo, args.revision)
        (args.out / 'baseline-source.commit').write_text(revision + '\n')
        (args.out / 'baseline-expected.json').write_bytes(encode(expected))
        reader = module.Reader(args.repo, args.port, args.session, args.fleet_dir, **selected)
        before, attempts = None, 0
        while time.monotonic() < deadline:
            record = module.read_records(args.out / 'records.raw.jsonl', {args.name}).get(args.name)
            if record:
                if before is None:
                    raise ValueError('completed record appeared before any accepted prepared capture')
                result = capture(args, module, reader, expected, go, 'runtime', args.out / 'baseline-runtime', before)
                break
            current = reservation(args)
            if current['state'] == 'TERMINAL':
                raise ValueError('reservation ended without completed onepass record')
            if before is None and current['state'] == 'GO':
                head = reader.head()
                if (head and module.mode_from_metadata(head, **selected) == 'baseline'
                        and boot_time(head) >= go['reservation']['started_at'] and reader.healthy()):
                    attempts += 1
                    attempt = capture(args, module, reader, expected, go, 'prepared',
                                      args.out / 'baseline-attempts' / ('prepared-' + str(attempts)))
                    state['attempts'].append(dict(path=str(attempt['path'].relative_to(args.out)),
                        receipt_sha256=sha((attempt['path'] / 'receipt.json').read_bytes()),
                        collection_status=attempt['receipt']['collection_status']))
                    if attempt['receipt']['collection_status'] == 'COMPLETE':
                        before = attempt
                        state['phases']['prepared'] = state['attempts'][-1]
                    elif attempts >= 3:
                        raise ValueError('three prepared collection attempts failed')
                    save()
            time.sleep(1)
        else:
            raise ValueError('baseline record/capture deadline expired')
        state['phases']['runtime'] = dict(path=str(result['path'].relative_to(args.out)),
            receipt_sha256=sha((result['path'] / 'receipt.json').read_bytes()),
            collection_status=result['receipt']['collection_status'])
        state['collection_status'] = result['receipt']['collection_status']
        state['errors'].extend(result['receipt']['collection_errors'])
        state['runtime_validation'] = ('FAIL' if any(x['receipt']['runtime_validation'] == 'FAIL'
                                                   for x in (before, result)) else 'PASS')
    except (OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as exc:
        state['collection_status'] = 'FAILED'
        state['errors'].append(str(exc) or type(exc).__name__)
    save()
    print(json.dumps(state, sort_keys=True), flush=True)
    return int(state['collection_status'] != 'COMPLETE' or state['runtime_validation'] != 'PASS')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('name', 'session', 'ticket', 'start', 'revision'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--port', type=int, default=18000)
    parser.add_argument('--sf6-unpack', action='store_true',
                        help='observe packed SF6 with scalar unpack=0 using the frozen unpack proof API')
    parser.add_argument('--fleet-dir', type=Path, default=Path('/home/choiceoh/glm53-logs/fleet'))
    parser.add_argument('--timeout', type=float, default=10800)
    args = parser.parse_args(argv)
    if args.sf6_unpack and args.port != 18000:
        parser.error('SF6 unpack baseline requires --port 18000')
    if (not re.fullmatch(r'[0-9a-f]{40}', args.revision)
            or args.pid <= 0 or not args.start.isdigit() or not 1 <= args.port <= 65535
            or not 0 < args.timeout <= 43200 or not args.repo.is_absolute() or not args.out.is_absolute()
            or not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value)
                       for value in (args.name, args.session, args.ticket))):
        parser.error('exact 40-character revision, literal identities, absolute paths, positive PID/start, and bounded timeout required')
    return observe(args)


if __name__ == '__main__':
    raise SystemExit(main())
