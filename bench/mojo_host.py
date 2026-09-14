#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and measure an optional Mojo host-commit experiment, entirely on the CPU.

The baseline is the current BurstDecode._apply_outcome method, extracted without
importing torch/Triton. The rank collective is a common no-op in this experiment;
all Python/native calls, conversions, token-list writes and counters are timed.
No engine, model, GPU, fleet, serving switch or on-import compiler is involved.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
import sys
import sysconfig
import tempfile
from time import perf_counter_ns
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench/mojo_host/commit.mojo"
ORACLE = ROOT / "engine/profiles/glm53/burst_decode.py"
MODULE = "st_mojo_host"
MOJO_VERSION = "1.0.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    return dict(system=platform.system(), machine=platform.machine(),
                python=platform.python_version(), implementation=platform.python_implementation(),
                soabi=sysconfig.get_config_var("SOABI"))


def load_oracle(path=ORACLE):
    """Use the serving method itself, not a hand-maintained faster/slower mirror."""
    tree = ast.parse(Path(path).read_text())
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BurstDecode"]
    method, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_outcome"]
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_outcome"]


def build(directory, compiler="mojo"):
    directory = Path(directory).resolve()
    executable = shutil.which(str(compiler))
    if executable is None:
        raise RuntimeError(f"Mojo compiler not found: {compiler}; install mojo=={MOJO_VERSION} separately")
    version = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if not version.startswith(f"Mojo {MOJO_VERSION} ("):
        raise RuntimeError(f"expected Mojo {MOJO_VERSION}, got {version!r}")
    directory.mkdir(parents=True, exist_ok=True)
    # A compiler failure must never publish a partial library/manifest pair.
    with tempfile.TemporaryDirectory(prefix=".mojo-build-", dir=directory) as temporary:
        binary = Path(temporary) / f"{MODULE}.so"
        command = [executable, "build", str(SOURCE), "--emit", "shared-lib", "-o", str(binary)]
        subprocess.run(command, check=True)
        manifest = dict(schema=1, runtime=runtime_identity(), compiler=version,
                        flags=["--emit", "shared-lib"], source_sha256=sha256(SOURCE),
                        library_sha256=sha256(binary))
        record = Path(temporary) / "manifest.json"
        record.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(binary, directory / binary.name)
        os.replace(record, directory / record.name)
    return manifest


def validate_artifact(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = dict(schema=1, runtime=runtime_identity(), source_sha256=sha256(SOURCE),
                    library_sha256=sha256(directory / f"{MODULE}.so"))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Mojo artifact {key} mismatch; rebuild with this interpreter/source")
    if not manifest.get("compiler", "").startswith(f"Mojo {MOJO_VERSION} ("):
        raise RuntimeError("Mojo artifact compiler mismatch; rebuild")
    return manifest


def load_native(directory):
    directory = Path(directory).resolve()
    manifest = validate_artifact(directory)
    binary = directory / f"{MODULE}.so"
    if MODULE in sys.modules:
        module = sys.modules[MODULE]
        if Path(module.__file__).resolve() != binary:
            raise RuntimeError("a different Mojo artifact is already loaded; use a fresh process")
        if module._st_manifest != manifest:
            raise RuntimeError("Mojo artifact changed after import; use a fresh process")
        return module, manifest
    spec = importlib.util.spec_from_file_location(MODULE, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._st_manifest = manifest
    sys.modules[MODULE] = module
    return module, manifest


def native_wrapper(module):
    """The proposed serving boundary, measured including its extra Python call."""
    def apply(self, pending, result):
        self._agree_outcome(pending.seqs, result, iteration=len(pending.outcomes))
        try:
            module.apply_outcome(self.e, pending.seqs, result, pending.finished, self.t)
        except Exception as exc:
            # Mojo 1.0's generated binding maps Error to plain Exception. Preserve
            # the serving guards' RuntimeError contract, without hiding other errors.
            if type(exc) is Exception and str(exc) in {
                "Mojo host experiment requires 1..4 rows",
                "bounded decode made no progress",
                "bounded decode readback lost row/context order",
            }:
                raise RuntimeError(str(exc)) from exc
            raise
        pending.outcomes.append(result)
    return apply


def fixture(rows, context=0, k=7, block=64):
    seqs = tuple(range(11, 11 + rows))
    e = NS(tokens={s: [0] * context for s in seqs}, ctx=dict.fromkeys(seqs, context),
           staged={}, accepted_per_step=[0] * (k + 2), accepted_total=0, drafted_total=0,
           drafter=NS(k=k), F=NS(block=block))
    pipeline = NS(e=e, t=k + 1, _agree_outcome=lambda *args, **kwargs: None)
    return pipeline, NS(seqs=seqs, finished=[False] * rows, outcomes=[])


def trace(rows, context, steps, k=7):
    """Deterministic synthetic readbacks, built before the timed region."""
    positions = [context] * rows
    results = []
    for step in range(steps):
        counts = [1 + (step + 3 * row) % (k + 1) for row in range(rows)]
        results.append(dict(count=counts, done=[step == steps - 1] * rows,
                            accepted=[n - 1 for n in counts], before=positions.copy(),
                            tokens=[[100 + row * 10 + j for j in range(k + 1)] for row in range(rows)]))
        positions = [p + n for p, n in zip(positions, counts)]
    return [results[i:i + 4] for i in range(0, steps, 4)]


def snapshot(pipeline, pending):
    e = pipeline.e
    return dict(tokens=e.tokens, ctx=e.ctx, staged=e.staged, accepted=e.accepted_total,
                drafted=e.drafted_total, histogram=e.accepted_per_step,
                finished=pending.finished, outcomes=pending.outcomes)


def run_trace(fn, rows, context, bursts):
    pipeline, pending = fixture(rows, context)
    start = perf_counter_ns()
    for burst in bursts:
        # Real pending.outcomes is bounded to four; keep list growth comparable.
        pending.outcomes.clear()
        for result in burst:
            fn(pipeline, pending, result)
    elapsed = perf_counter_ns() - start
    return elapsed, snapshot(pipeline, pending)


def measure_case(functions, rows, context, steps, samples, warmup):
    bursts = trace(rows, context, steps)
    pairs = []
    state_hash = None
    for sample in range(-warmup, samples):
        order = ("python", "mojo") if sample % 2 == 0 else ("mojo", "python")
        times, states = {}, {}
        for name in order:
            elapsed, states[name] = run_trace(functions[name], rows, context, bursts)
            times[name] = elapsed / steps / 1000
        if states["python"] != states["mojo"]:
            raise AssertionError(f"host state mismatch at C={rows}, context={context}, sample={sample}")
        digest = hashlib.sha256(json.dumps(states["python"], sort_keys=True).encode()).hexdigest()
        if state_hash is not None and digest != state_hash:
            raise AssertionError("host state changed between benchmark samples")
        state_hash = digest
        if sample >= 0:
            pairs.append(dict(order=order, us_per_outcome=times))
    medians = {name: median(p["us_per_outcome"][name] for p in pairs) for name in functions}
    return dict(rows=rows, initial_context=context, steps_per_sample=steps,
                host_state_sha256=state_hash, all_samples_state_equal=True, samples=pairs,
                median_us=medians, delta_us=medians["mojo"] - medians["python"],
                mojo_latency_change_pct=100 * (medians["mojo"] / medians["python"] - 1),
                paired_speedup_median=median(p["us_per_outcome"]["python"] /
                                            p["us_per_outcome"]["mojo"] for p in pairs))


def benchmark(directory, *, steps=4096, samples=12, warmup=2):
    if steps < 1 or samples < 2 or warmup < 1:
        raise ValueError("steps >= 1, samples >= 2 and warmup >= 1 are required")
    module, manifest = load_native(directory)
    functions = dict(python=load_oracle(), mojo=native_wrapper(module))
    cases = [measure_case(functions, rows, context, steps, samples, warmup)
             for context in (32760, 131064) for rows in (1, 4)]
    return dict(schema=1, measured_at=datetime.now(timezone.utc).isoformat(),
                scope="CPU host commit only; synthetic readback; not GB10 serving performance",
                compile_and_fixture_setup_timed=False, collective="common no-op", gpu_used=False,
                import_time_compilation=False, warmup_pairs=warmup,
                platform=platform.platform(), runtime=runtime_identity(), artifact=manifest,
                oracle_file=str(ORACLE.relative_to(ROOT)), oracle_sha256=sha256(ORACLE),
                harness_sha256=sha256(__file__), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="explicitly precompile the optional CPU extension")
    build_parser.add_argument("--directory", type=Path, default=ROOT / "build/mojo-host")
    build_parser.add_argument("--compiler", default="mojo")
    run_parser = sub.add_parser("run", help="require a matching prebuilt artifact; never compile/fallback")
    run_parser.add_argument("--directory", type=Path, default=ROOT / "build/mojo-host")
    run_parser.add_argument("--steps", type=int, default=4096)
    run_parser.add_argument("--samples", type=int, default=12)
    run_parser.add_argument("--warmup", type=int, default=2)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.directory, args.compiler), indent=2))
    else:
        result = benchmark(args.directory, steps=args.steps, samples=args.samples, warmup=args.warmup)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for case in result["cases"]:
            p, m = case["median_us"]["python"], case["median_us"]["mojo"]
            print(f"C={case['rows']} ctx={case['initial_context']}: Python {p:.3f} us, "
                  f"Mojo {m:.3f} us ({case['mojo_latency_change_pct']:+.1f}% latency)")
        print(args.output)


if __name__ == "__main__":
    main()
