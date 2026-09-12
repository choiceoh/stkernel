"""Durable onepass sessions. Preparation and diagnostic traffic never enter a measurement window."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CURRENT = ContextVar('onepass_recording', default=None)
POLICY = dict(version=1, concurrency=[1, 4], prefix='unique salt per request',
              preparation='full workload replay before each concurrency arm',
              measurement='profiler off; no observed specialization or graph capture',
              diagnostic='separate replay; one prefill chunk and four decode steps per context and concurrency')


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def steady_errors(report, requests, concurrency):
    errors = []
    ranks = report.get('ranks', [])
    if not ranks:
        return ['server latency/preparation evidence unavailable']
    for rank in ranks:
        if not rank.get('complete') or rank.get('diagnostic'):
            errors.append(f"rank {rank.get('rank')}: incomplete or profiled measurement")
        if rank.get('preparation_changed') is not False:
            errors.append(f"rank {rank.get('rank')}: preparation changed or unknown")
        if 'triton.runtime.jit._do_compile' not in rank.get('preparation_after', {}).get('observers', []):
            errors.append(f"rank {rank.get('rank')}: Triton specialization observer unavailable")
        widths = [len(r['rows']) for r in rank.get('rows', []) if r.get('kind') == 'host_step' and r.get('phase') == 'decode']
        if not widths or max(widths) != concurrency:
            errors.append(f"rank {rank.get('rank')}: actual decode width {max(widths, default=0)} != {concurrency}")
    if any(r.get('cached_tokens') != 0 for r in requests):
        errors.append('prefix reuse occurred or cached-token evidence is missing')
    if not requests or any(r.get('error') or not r.get('completion_tokens') or r.get('finish_reason') not in ('stop', 'length') for r in requests):
        errors.append('incomplete generation')
    return errors


class Run:
    def __init__(self, record, out, url):
        self.record, self.out = record, Path(out).expanduser()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12]
        self.path = self.out.parent / 'onepass-runs' / self.id
        self.path.mkdir(parents=True)
        self.url = url.split('/v1/')[0] + '/v1/engine/latency'
        self.lock = threading.Lock()
        self.phase, self.token, self.requests = 'initializing', None, []
        self.complete = False
        record.update(run_id=self.id, artifacts=str(self.path), measurement_policy=POLICY,
                      recording={'schema': 1, 'status': 'running'})
        try:
            with urllib.request.urlopen(self.url, timeout=5) as r:
                self.server = json.load(r)
            self.supported = self.server.get('schema') == 1
        except Exception as exc:
            self.server, self.supported = {'error': str(exc)}, False
        record['recording']['server'] = self.server
        self.checkpoint()

    def checkpoint(self):
        write(self.path / 'record.json', self.record)

    def control(self, **body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=75) as r:
            reply = json.load(r)
        if reply.get('boot_id') != self.server.get('boot_id') or len(reply.get('ranks', [])) != self.server.get('ranks'):
            raise RuntimeError('server boot or rank coverage changed during onepass')
        if any(r.get('error') for r in reply['ranks']):
            raise RuntimeError(str(reply))
        return reply

    def begin(self, phase, concurrency=1, diagnostic=False):
        self.phase, self.requests = phase, []
        self.token = self.id + '-' + phase
        if self.supported:
            self.control(op='begin', token=self.token, diagnostic=diagnostic, concurrency=concurrency)
        CURRENT.set(self)
        self.record['recording']['phase'] = phase
        self.checkpoint()

    def end(self):
        CURRENT.set(None)
        report = self.control(op='end', token=self.token) if self.supported else {'ranks': []}
        rows = []
        for rank in report['ranks']:
            rows.extend(rank.get('rows', []))
            for trace in rank.get('traces', []):
                path = self.path / self.phase / f"rank-{rank['rank']}" / trace['file']
                path.parent.mkdir(parents=True, exist_ok=True)
                offset, digest = 0, hashlib.sha256()
                with path.open('xb') as f:
                    while offset < trace['bytes']:
                        reply = self.control(op='artifact', token=self.token, rank=rank['rank'], file=trace['file'], offset=offset)
                        block = next(r for r in reply['ranks'] if r['rank'] == rank['rank'])
                        data = base64.b64decode(block['base64'], validate=True)
                        if not data or block['offset'] != offset or block['bytes'] != trace['bytes']:
                            raise RuntimeError('invalid trace artifact chunk')
                        f.write(data)
                        digest.update(data)
                        offset += len(data)
                if offset != trace['bytes'] or digest.hexdigest() != trace['sha256']:
                    raise RuntimeError('trace artifact checksum mismatch')
                from engine.base.latency_trace import attribute
                for row in attribute(json.loads(gzip.decompress(path.read_bytes()))):
                    rows.append(dict(row, rank=rank['rank'], phase=trace['phase'], step=trace['step']))
        from engine.base.latency_trace import summarize, union_us
        directory = self.path / self.phase
        directory.mkdir(exist_ok=True)
        write(directory / 'server.json', report)
        # Different ranks and profiled steps have different clocks. Never union
        # them together, or add CPU launch and GPU device durations.
        unions = []
        keys = {(r.get('rank'), r.get('step')) for r in rows if r['kind'] == 'gpu_activity'}
        for rank, step in sorted(keys):
            members = [r for r in rows if r['kind'] == 'gpu_activity' and (r['rank'], r['step']) == (rank, step)]
            unions.append(dict(rank=rank, step=step,
                device_busy_us=union_us([(r['start_us'], r['start_us'] + r['duration_us']) for r in members]),
                kernel_sum_us=sum(r['duration_us'] for r in members)))
        write(directory / 'latency-summary.json', dict(operations=summarize(rows), device_intervals=unions))
        with (directory / 'latency.jsonl').open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.token = None
        self.checkpoint()
        return report

    def request(self, timing, text, events):
        value = dict(timing, phase=self.phase, text=text, channels=events)
        with self.lock:
            self.requests.append(timing.copy())
            with (self.path / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())

    def finish(self, error=None):
        if error:
            self.record['recording'].update(status='incomplete', error=str(error))
            # Preserve the server token and its on-disk running manifest for recovery.
            self.record['recording']['unfinished_token'] = self.token
        else:
            self.record['recording']['status'] = 'complete'
        self.checkpoint()
        with self.out.open('a') as f:
            f.write(json.dumps(self.record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.complete = True


def group(run, ask, url, model, item, concurrency, scan=None, expected=None):
    barrier = threading.Barrier(concurrency)
    def request(index):
        CURRENT.set(run)
        timing = dict(ctx=item['ctx'], question=item['question'], concurrency=concurrency, client=index)
        barrier.wait(timeout=30)
        text, _, _, _, finish = ask(url, model, item['content'], item['max_tokens'], timing,
            min_tokens=item.get('min_tokens', 0), seed=item.get('seed'), reasoning_budget=item.get('reasoning_budget'))
        if scan is not None:
            hits = scan.scan(text, truncated=finish == 'length')
            timing['corruption'] = {k: v for k, v in hits.items() if k not in scan.INFORMATIONAL and v}
        if expected is not None:
            questions = range(len(expected)) if item['question'] in ('all', 'fixed-all') else [item['question']]
            timing['quality'] = [all(any(alt in text.lower() for alt in forms) for forms in expected[q]) for q in questions]
        return timing
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        requests = list(pool.map(request, range(concurrency)))
    elapsed = max(r['ended_monotonic'] for r in requests) - min(r['started_monotonic'] for r in requests)
    return dict(ctx=item['ctx'], concurrency=concurrency, requests=requests, elapsed_s=elapsed,
                aggregate_output_tok_s=sum(r['completion_tokens'] for r in requests) / elapsed,
                rate_scope='total completion tokens / first request start to last completion; includes prefill')
