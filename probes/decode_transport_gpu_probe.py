#!/usr/bin/env python3
"""Four-rank compact/inline correctness, reusing the exact AR/MHC graph cases.

No timings are collected. The legacy probe report alone cannot admit this
mode: this wrapper additionally binds the actual compiled flags, QP inline
capabilities and both compact/companion graph capture paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'glm53-decode-transport-gpu-v1'
FLAGS = {'VLLM_GLM53_AR_COMPACT_CTA': '1', 'VLLM_GLM53_AR_PROXY_INLINE': '1',
         'VLLM_GLM53_AR_CONSUMER_PDL': '1', 'VLLM_GLM53_MK_PDL': '1'}


def require_modes(actual, *, initialized):
    if (not isinstance(actual, list) or len(actual) != 5
            or any(type(value) is not int for value in actual)
            or actual[:2] != [1, 1]
            or (initialized and min(actual[2:]) < 8)
            or (not initialized and actual[2:] != [0, 0, 0])):
        raise ValueError(f'wrong compact/inline extension or QP capability: {actual}')


class AttestedTransport:
    """Observe real pybind calls without replacing any CUDA implementation."""
    def __init__(self, extension, torch):
        self.extension, self.torch = extension, torch
        self.modes_before = list(extension.transport_modes())
        require_modes(self.modes_before, initialized=False)
        self.modes_initialized = None
        self.captures = set()
        self.mixed_cases = []

    def __getattr__(self, name):
        return getattr(self.extension, name)

    def init(self, *args):
        result = self.extension.init(*args)
        self.modes_initialized = list(self.extension.transport_modes())
        require_modes(self.modes_initialized, initialized=True)
        return result

    def connect(self, *args):
        result = self.extension.connect(*args)
        self.check_mixed_graphs()
        return result

    def check_mixed_graphs(self):
        """12 -> ordinary48 -> consumer48 -> 12 in one retained graph."""
        import torch.distributed as dist
        torch = self.torch
        rank = dist.get_rank()
        for small_n, large_n in ((24576, 32769), (32768, 65536)):
            small = torch.zeros(small_n, dtype=torch.bfloat16, device='cuda')
            large = torch.zeros(large_n, dtype=torch.bfloat16, device='cuda')

            def chain():
                a = self.oneshot_ar_consumer(small)
                b = self.oneshot_ar(a)
                c = self.oneshot_ar_consumer(large)
                d = self.oneshot_ar_consumer(b)
                return a, b, c, d

            dist.barrier()
            chain()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = chain()
            for seed, factor in ((17, 1), (0, 0), (29, 2)):
                host_small = (((torch.arange(small_n) + rank * 3) % 5 - 2 + rank)
                              .to(torch.bfloat16) * (factor / 32))
                host_large = (((torch.arange(large_n) + rank * 5) % 7 - 3 + rank)
                              .to(torch.bfloat16) * (factor / 32))
                small.copy_(host_small)
                large.copy_(host_large)
                dist.all_reduce(host_small)
                dist.all_reduce(host_large)
                expected = (host_small, host_small * 4, host_large, host_small * 16)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                if not all(torch.equal(actual.cpu(), want) for actual, want in zip(outputs, expected)):
                    raise ValueError(f'mixed 12/48 graph failed: rank={rank} sizes={small_n,large_n} seed={seed}')
                self.mixed_cases.append(dict(small_elements=small_n, large_elements=large_n,
                    seed=seed, ctas=[12,48,48,12], exact_outputs=4, passed=True))

    def call(self, value, consumer):
        count = int(value.numel())
        if self.torch.cuda.is_current_stream_capturing():
            self.captures.add((count, consumer, 12 if consumer and 0 < count <= 32768 else 48))
        function = self.extension.oneshot_ar_consumer if consumer else self.extension.oneshot_ar
        return function(value)

    def oneshot_ar(self, value):
        return self.call(value, False)

    def oneshot_ar_consumer(self, value):
        return self.call(value, True)

    def proof(self):
        require_modes(self.modes_initialized, initialized=True)
        return dict(requested_flags=FLAGS, modes_before=self.modes_before,
                    modes_initialized=self.modes_initialized,
                    mixed_graph_cases=self.mixed_cases,
                    captures=[dict(elements=n, consumer=c, ctas=ctas)
                              for n, c, ctas in sorted(self.captures)])


def validate_transport_proof(report):
    from ar_consumer_probe import AR_OWNERSHIP_SIZES
    if report.get('schema') != SCHEMA or report.get('samples') != []:
        raise ValueError('new transport correctness schema with zero timing samples required')
    proof = report.get('transport', {})
    if proof.get('requested_flags') != FLAGS:
        raise ValueError('compact/inline candidate flags are missing')
    require_modes(proof.get('modes_before'), initialized=False)
    require_modes(proof.get('modes_initialized'), initialized=True)
    captures = proof.get('captures')
    if not isinstance(captures, list):
        raise ValueError('missing transport capture proof')
    observed = set()
    for row in captures:
        if (not isinstance(row, dict) or type(row.get('elements')) is not int
                or type(row.get('consumer')) is not bool or type(row.get('ctas')) is not int):
            raise ValueError('invalid transport capture row')
        n, consumer, ctas = row['elements'], row['consumer'], row['ctas']
        if ctas != (12 if consumer and 0 < n <= 32768 else 48):
            raise ValueError('wrong compact/companion geometry')
        observed.add((n, consumer, ctas))
    if len(observed) != len(captures):
        raise ValueError('duplicate transport capture rows')
    expected = {(n, consumer, 12 if consumer and n <= 32768 else 48)
                for n in AR_OWNERSHIP_SIZES for consumer in (False, True)}
    if not expected <= observed:
        raise ValueError('missing boundary/compact/companion captures')
    mixed = proof.get('mixed_graph_cases')
    expected_mixed = {(small, large, seed) for small, large in ((24576,32769),(32768,65536))
                      for seed in (17,0,29)}
    if (not isinstance(mixed, list) or len(mixed) != len(expected_mixed)
            or any(row.get('passed') is not True or row.get('exact_outputs') != 4
                   or row.get('ctas') != [12,48,48,12] for row in mixed)
            or {(row['small_elements'],row['large_elements'],row['seed']) for row in mixed} != expected_mixed):
        raise ValueError('mixed 12/48 retained graph checks incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('fresh output required')
    os.environ.update(FLAGS, VLLM_DSV4_OSAR_MAXEL='131072')
    import torch
    import ar_consumer_probe as original
    original_module = original.module
    attested = []

    def module(name, path):
        loaded = original_module(name, path)
        if name == 'ar_probe_osar':
            build = loaded._build

            def checked_build():
                result = AttestedTransport(build(), torch)
                attested.append(result)
                return result

            loaded._build = checked_build
        return loaded

    original.module = module
    raw = args.out.with_suffix('.base.json')
    before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    saved_argv = sys.argv
    try:
        sys.argv = [str(Path(original.__file__)), '--distributed', '--check-only', '--out', str(raw)]
        original.main()
    finally:
        sys.argv = saved_argv
        original.module = original_module
    if len(attested) != 1:
        raise ValueError('exactly one actual OSAR extension required')
    report = json.loads(raw.read_text())
    if report.get('status') != 'PASS' or report.get('mode') != 'distributed':
        raise ValueError('underlying four-rank graph/oracle checks incomplete')
    after = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if before != after:
        raise ValueError('transport wrapper changed during checks')
    report.update(schema=SCHEMA, transport=attested[0].proof(), wrapper_sha256=after)
    validate_transport_proof(report)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS compact=1 inline=1: four-rank transport graphs and exact consumer checks; no timings', flush=True)


if __name__ == '__main__':
    main()
