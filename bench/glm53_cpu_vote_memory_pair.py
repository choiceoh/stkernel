#!/usr/bin/env python3
"""Owned private warm-hit GPU/CPU/GPU vote comparison; no model requests."""
import argparse
import base64
import copy
import gzip
import hashlib
from pathlib import Path
import re
import sys
import time

import prefill_observation_run as base
from prefill_observation import PrivateObserverAPI, idle_observers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probes'))
import glm53_cpu_vote_host as host
import glm53_cpu_vote_memory as memory

class PrivateMemoryAPI(PrivateObserverAPI):
    JSON_PATHS = (*PrivateObserverAPI.JSON_PATHS, '/glm53/cpu-vote-memory')

    def post(self, path, payload=None):
        if path == '/glm53/cpu-vote-memory' and payload != {}:
            raise ValueError('memory receipt accepts no options')
        return super().post(path, payload)


ARMS = (('PRIME', '0'), ('BASE0', '0'), ('CPU', '1'), ('BASE1', '0'))


def validate_logs(logs, warm):
    if set(logs) != set(base.lifecycle.NODES):
        raise ValueError('four boot logs required')
    receipts = {}
    for rank, node in enumerate(base.lifecycle.NODES):
        text = logs[node]
        rows = re.findall(r'\[rank-cache\] (hit|saved) rank=(\d+) bytes=(\d+)', text)
        if len(rows) != 1 or int(rows[0][1]) != rank or int(rows[0][2]) <= 0:
            raise ValueError('missing or ambiguous rank artifact receipt: '+node)
        if warm and rows[0][0] != 'hit':
            raise ValueError('warm arm loaded source weights: '+node)
        if re.search(r'\[rank-cache\].*(?:rejecting|unavailable|another rank missed|save skipped)', text):
            raise ValueError('rank cache fallback or error: '+node)
        fp8 = re.findall(r'\[fp8-cache\].*?enabled=True hit=(\d+) miss=(\d+) errors=(\d+)', text)
        if len(fp8) < 2 or any(int(e) or (warm and (not int(h) or int(m))) for h, m, e in fp8):
            raise ValueError('FP8 cache receipt failed: '+node)
        nccl = [line for line in text.splitlines() if 'NCCL INFO' in line and 'Init COMPLETE' in line]
        if not nccl:
            raise ValueError('NCCL initialization diagnostic receipt absent: '+node)
        receipts[node] = dict(rank_cache=rows, fp8_cache=fp8, nccl_init_lines=nccl)
    return receipts


def matched(left, right):
    for node in base.lifecycle.NODES:
        a, b = left[node], right[node]
        for key in ('image', 'args', 'mounts', 'manifest_sha', 'model', 'hardware'):
            if a[key] != b[key]:
                raise ValueError('warm arm runtime differs: '+key+' '+node)
        env = lambda x: {k: v for k, v in x['env'].items() if k != host.KNOB}
        if env(a) != env(b):
            raise ValueError('non-vote environment changed between arms')


class Run(base.Run):
    experiment = 'glm53-cpu-vote-memory'
    complete_marker = 'GLM53_CPU_VOTE_MEMORY_CAPTURE_COMPLETE'

    def __init__(self, source, revision, out):
        super().__init__(source, revision, out)
        self.root = out
        self.policy = '0'
        self.arm = 'PRIME'
        self.node_dir = out/self.arm/'worker'
        self.originals = None
        self.memory_sha = hashlib.sha256((source/'probes/glm53_cpu_vote_memory.py').read_bytes()).hexdigest()
        self.cache_hashes = host.source_hashes(source)

    def host(self, node, action):
        base.lifecycle.check_holder()
        kwargs = dict(action=action, name=self.name, session=self.session, directory=str(self.node_dir),
                      source=str(self.source), original=base.lifecycle.name(node), policy=self.policy,
                      expected_original=self.originals[node] if self.originals is not None else None)
        code = 'import json,sys\nsys.path.insert(0,'+repr(str(self.source/'probes'))+')\n'
        code += 'import glm53_cpu_vote_host as h\nprint(json.dumps(h.dispatch(**'+repr(kwargs)+')))'
        result = base.lifecycle.remote(node, code, timeout=180)
        if action in ('prepare', 'state', 'start') and result['candidate_sources'] != self.cache_hashes:
            raise ValueError('all-rank candidate source mismatch')
        return result

    def snapshot(self, original=False, archive=False):
        result = super().snapshot(original=original, archive=archive)
        if not original:
            states = self.all('state')
            for node in base.lifecycle.NODES:
                # The inherited snapshot inventories the unchanged public
                # overlay prefix. Add the two explicitly rebound source files.
                for target in self.cache_hashes:
                    if target in result[node]['mounts']:
                        raise ValueError('candidate source was not rebound')
                result[node]['mounts'].update(states[node]['candidate_sources'])
        return result

    def attest_clone(self, original, cloned):
        expected = copy.deepcopy(original)
        encoded = lambda v: 'sha256:'+hashlib.sha256(v.encode()).hexdigest()
        for node, row in expected.items():
            if not set(self.cache_hashes) <= set(row['mounts']):
                raise ValueError('original cache modules missing from runtime inventory')
            row['mounts'].update(self.cache_hashes)
            row['env'].update({k: encoded(v) for k, v in host.FIXED_ENV.items()})
            row['env'][host.KNOB] = encoded(self.policy)
        super().attest_clone(expected, cloned)

    def logs(self):
        records = self.all('logs')
        texts = {}
        for node, row in records.items():
            if not row['exists']:
                raise ValueError('boot log missing: '+node)
            raw = gzip.decompress(base64.b64decode(row['gzip_base64'], validate=True))
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('boot log transfer differs')
            texts[node] = raw.decode(errors='replace')
        return texts

    def collect(self, prepared, original):
        self.originals = {node: row['original'] for node, row in prepared.items()}
        baseline = None
        try:
            for index, (self.arm, self.policy) in enumerate(ARMS):
                self.out = self.root/self.arm
                self.out.mkdir(parents=True, exist_ok=True)
                self.node_dir = self.out/'worker'
                try:
                    if index:
                        prepared = self.all('prepare')
                    base.save(self.out/'prepared.json', prepared)
                    base.settled(base.lifecycle.NODES[1:], lambda n: self.host(n, 'start'))
                    self.host('local', 'start')
                    self.ready(prepared)
                    initial = self.snapshot()
                    self.attest_clone(original, initial)
                    base.save(self.out/'runtime.json', initial)
                    if baseline is not None:
                        matched(baseline, initial)
                    if self.arm == 'BASE0':
                        baseline = initial
                    api = PrivateMemoryAPI('http://127.0.0.1:18000')
                    base.lifecycle.idle(18000)
                    idle_observers(api.post('/glm53/prefill-observe', {'op': 'status'}), self.sha)
                    time.sleep(15)
                    for sample in range(3):
                        base.lifecycle.idle(18000)
                        report = api.post('/glm53/cpu-vote-memory', {})
                        base.save(self.out/f'memory-{sample}.json', report)
                        memory.validate(report, self.memory_sha, self.cache_hashes[host.PREFIX+host.MODULES[0]], self.policy)
                        base.lifecycle.idle(18000)
                        if sample != 2:
                            time.sleep(5)
                    base.save(self.out/'boot-receipts.json', validate_logs(self.logs(), warm=bool(index)))
                    if self.snapshot() != initial:
                        raise ValueError('private runtime changed during memory capture')
                    base.save(self.out/'complete.json', dict(complete=True, policy=self.policy,
                              warm=bool(index), request_count=0, performance_acceptance=False))
                finally:
                    self.cleanup()
        finally:
            self.out = self.root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return Run(base.ROOT, args.revision, args.out).run()


if __name__ == '__main__':
    raise SystemExit(main())
