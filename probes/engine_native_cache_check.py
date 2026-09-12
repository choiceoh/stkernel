"""CPU-only NVCC reproduction of native cache hits across fresh processes.

The mini fixture checks invalidation using a host-only value function. The
oneshot fixture builds and loads ST's real transport with its production
builder, without initializing the transport or executing GPU kernels. Run in
the ST image without --gpus. This prices the native build path, not full-model
compilation or serving startup/throughput.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ONESHOT_FILES = ("__init__.py", "dsv4_oneshot_ar.cu", "dsv4_oneshot_transport.h")
ONESHOT_API = sorted(("init", "local_infos", "connect", "oneshot_ar", "oneshot_ar_consumer",
                      "oneshot_max_int64", "oneshot_ar_hint", "phase_counters", "transport_modes",
                      "healthy", "shutdown"))
SOURCE = '''#include <pybind11/pybind11.h>
#include "value.h"
__global__ void cache_probe(float* out) { *out = CACHE_VALUE + CACHE_BONUS; }
int answer() { return CACHE_VALUE + CACHE_BONUS; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("answer", &answer); }
'''
FLAGS = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a", "-DCACHE_BONUS=0"]


def compilations(directory):
    path = directory / ".ninja_log"
    if not path.exists():
        return 0
    return sum(len(parts := line.split("\t")) == 5 and parts[3].endswith(".cuda.o")
               for line in path.read_text().splitlines())


def worker(args):
    import torch
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import load
    from unittest.mock import patch
    from engine.kernels.native_cache import prepare_sources
    assert not torch.cuda.is_initialized()
    flags = [*FLAGS[:-1], f"-DCACHE_BONUS={args.bonus}"]
    sources = [args.input / "kernel.cu", args.input / "value.h"]
    ldflags, prefix = [], "st_cache_probe_"
    if args.fixture == "oneshot":
        spec = importlib.util.spec_from_file_location("cache_probe_oneshot", args.input / "__init__.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        flags = [*FLAGS[:-1], f"-DMAXEL={builder.MAX_ELEMENTS}"]
        sources = [args.input / name for name in ONESHOT_FILES[1:]]
        ldflags, prefix = ["-libverbs"], "st_oneshot_"
    build = {}

    def measured_load(**kwargs):
        directory = Path(kwargs["build_directory"])
        build.update(directory=directory, before=compilations(directory))
        return load(**(kwargs | {"verbose": True}))

    start = time.perf_counter()
    if args.fixture == "oneshot" and args.arm == "stable":
        # Exercise the actual production builder; the wrapper only records
        # Ninja's directory/count and turns on compiler logs.
        os.environ["ST_ONESHOT_BUILD_ROOT"] = str(args.cache)
        with patch.object(cpp_extension, "load", measured_load):
            module = builder.build()
    elif args.arm == "baseline":
        # The existing oneshot/native convention: content-addressed directory,
        # but the compiler still reads sources from the current checkout.
        key = hashlib.sha256(b"".join(p.read_bytes() for p in sources)
                             + repr((flags, torch.__version__, torch.version.cuda)).encode()).hexdigest()[:16]
        directory = args.cache / key
        directory.mkdir(parents=True, exist_ok=True)
        module = measured_load(name=prefix + key, sources=[str(sources[0])],
                               extra_cuda_cflags=flags, extra_ldflags=ldflags,
                               build_directory=str(directory))
    else:
        key, directory, staged = prepare_sources(args.cache, sources,
                                                (flags, torch.__version__, torch.version.cuda))
        module = measured_load(name=prefix + key, sources=[staged[0]], extra_cuda_cflags=flags,
                               extra_ldflags=ldflags, build_directory=str(directory))
    duration = time.perf_counter() - start
    directory = build["directory"]
    value = module.answer() if args.fixture == "mini" else sorted(n for n in dir(module) if not n.startswith("_"))
    host_guards = []
    if args.fixture == "oneshot":
        # Exercise the actual Tensor type caster and early host-side guards
        # without reaching CUDA stream lookup or initializing the transport.
        for function, dtype, message in (("oneshot_ar", torch.bfloat16, "input.is_cuda()"),
                                         ("oneshot_max_int64", torch.int64,
                                          "oneshot MAX requires contiguous CUDA int64")):
            try:
                getattr(module, function)(torch.empty(8, dtype=dtype, device="cpu"))
            except RuntimeError as exc:
                if message not in str(exc):
                    raise
                host_guards.append(function)
            else:
                raise RuntimeError(f"{function} accepted a CPU tensor")
    assert not torch.cuda.is_initialized()
    print("RESULT " + json.dumps(dict(arm=args.arm, fixture=args.fixture, key=directory.name,
          value=value, load_seconds=duration,
          nvcc_compilations=compilations(directory) - build["before"], directory=str(directory),
          torch=torch.__version__, cuda=torch.version.cuda, cuda_initialized=False,
          host_guards=host_guards)), flush=True)


def run(args):
    if args.output.exists():
        raise ValueError("choose a fresh output directory; existing evidence must not be overwritten")
    args.output.mkdir(parents=True)
    source = args.output / "checkout-a"
    source.mkdir()
    if args.fixture == "mini":
        (source / "kernel.cu").write_text(SOURCE)
        (source / "value.h").write_text("#define CACHE_VALUE 41\n")
    expected, bonus = 41, 0
    rows = []
    steps = ("cold", "warm", "touch-source", "relocate", "touch-header", "edit-header", "edit-source", "flags")
    tracked = [ROOT / "engine/kernels/native_cache.py", Path(__file__).resolve()]
    if args.fixture == "oneshot":
        tracked.extend(ROOT / "engine/kernels/oneshot" / name for name in ONESHOT_FILES)
        for path in tracked[2:]:
            shutil.copy2(path, source / path.name)
        expected = ONESHOT_API
        steps = ("cold", "warm", "relocate")
    for index, case in enumerate(steps):
        if case == "touch-source":
            os.utime(source / "kernel.cu")
        elif case == "relocate":
            destination = args.output / "checkout-b"
            shutil.copytree(source, destination)
            source = destination
        elif case == "touch-header":
            os.utime(source / "value.h")
        elif case == "edit-header":
            (source / "value.h").write_text("#define CACHE_VALUE 43\n")
            expected = 43
        elif case == "edit-source":
            (source / "kernel.cu").write_text(SOURCE.replace("return CACHE_VALUE + CACHE_BONUS;",
                                                            "return CACHE_VALUE + CACHE_BONUS + 2;"))
            expected = 45
        elif case == "flags":
            bonus, expected = 3, 48
        for arm in (("baseline", "stable") if index % 2 == 0 else ("stable", "baseline")):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm,
                       "--fixture", args.fixture,
                       "--input", str(source), "--cache", str(args.output / arm), "--bonus", str(bonus)]
            started = time.perf_counter()
            child = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", MAX_JOBS="2"),
                                   capture_output=True, text=True, timeout=300)
            process_seconds = time.perf_counter() - started
            (args.output / f"{case}-{arm}.log").write_text(child.stdout + child.stderr)
            if child.returncode:
                raise RuntimeError(f"{case}/{arm} failed; see its saved log")
            row = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")))
            row.update(case=case, process_seconds=process_seconds)
            should_compile = 0 if case == "warm" or (arm == "stable" and
                               case in ("touch-source", "relocate", "touch-header")) else 1
            if row["value"] != expected or row["nvcc_compilations"] != should_compile:
                raise RuntimeError(f"unexpected cache/value result: {row}; expected {expected}/{should_compile}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            report = dict(scope=__doc__, fixture=args.fixture, gpu_used=False, complete=False, rows=rows)
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in tracked})
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", choices=("mini", "oneshot"), default="mini")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=("baseline", "stable"), help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bonus", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.output is None:
            parser.error("--output must name a new evidence directory")
        args.output = args.output.resolve()
        run(args)


if __name__ == "__main__":
    main()
