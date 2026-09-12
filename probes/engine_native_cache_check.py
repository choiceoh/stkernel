"""CPU-only NVCC reproduction of native cache hits across fresh processes.

A small CUDA extension exposes a host-only value function; no GPU kernel is
executed. Run in the ST image without --gpus. This prices the build path,
not compilation of the full model or serving startup/throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    assert not torch.cuda.is_initialized()
    flags = [*FLAGS[:-1], f"-DCACHE_BONUS={args.bonus}"]
    sources = [args.input / "kernel.cu", args.input / "value.h"]
    start = time.perf_counter()
    if args.arm == "baseline":
        # The existing oneshot/native convention: content-addressed directory,
        # but the compiler still reads sources from the current checkout.
        key = hashlib.sha256(b"".join(p.read_bytes() for p in sources)
                             + repr((flags, torch.__version__, torch.version.cuda)).encode()).hexdigest()[:16]
        directory = args.cache / key
        directory.mkdir(parents=True, exist_ok=True)
        compiled_sources = [str(sources[0])]
    else:
        key, directory, staged = prepare_sources(args.cache, sources,
                                                (flags, torch.__version__, torch.version.cuda))
        compiled_sources = [staged[0]]
    before = compilations(directory)
    module = load(name="st_cache_probe_" + key, sources=compiled_sources,
                  extra_cuda_cflags=flags, build_directory=str(directory), verbose=True)
    duration = time.perf_counter() - start
    value = module.answer()
    assert not torch.cuda.is_initialized()
    print("RESULT " + json.dumps(dict(arm=args.arm, key=key, value=value, load_seconds=duration,
          nvcc_compilations=compilations(directory) - before, directory=str(directory),
          torch=torch.__version__, cuda=torch.version.cuda, cuda_initialized=False)), flush=True)


def run(args):
    if args.output.exists():
        raise ValueError("choose a fresh output directory; existing evidence must not be overwritten")
    args.output.mkdir(parents=True)
    source = args.output / "checkout-a"
    source.mkdir()
    (source / "kernel.cu").write_text(SOURCE)
    (source / "value.h").write_text("#define CACHE_VALUE 41\n")
    expected, bonus = 41, 0
    rows = []
    steps = ("cold", "warm", "touch-source", "relocate", "touch-header", "edit-header", "edit-source", "flags")
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
            report = dict(scope=__doc__, gpu_used=False, complete=False, rows=rows)
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path("engine/kernels/native_cache.py"), Path(__file__))})
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
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
