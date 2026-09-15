"""Which window of a boot compiled something -- one walk of the JIT caches, after the fact (base).

Twice now a boot's time has turned on the question "was that compile or was that work", and twice it was
answered by hand, after the boot, by listing `/cache` by mtime:

  * `serving on :8000` plus 43 s, a b12x MoE shape took 1.55 s to build, and the engine kept writing
    artifacts for minutes -- inside requests, on their time to first token (boot-time study 5-g).
  * the far prefill pass ran 29.5-30.6 s on five bracket boots and 47.40 s on production. Nobody could
    say why, but the window held no cache write on any of the four nodes, so compilation was not it
    (measurements/st_prefill_gate_reuse_20260915).

The evidence is the same both times and the cost of collecting it is one walk, so it belongs in the boot
rather than in a person's shell history. The shape here is chosen so it is one walk and not one per phase:

  a phase records its wall-clock window as it runs and pays nothing,
  ONE walk at the end lists the artifacts newer than the first window,
  each artifact's mtime says which window it fell in.

Wall clock, not `perf_counter`: an mtime is wall clock and the two have to be comparable. A boot whose
clock steps under it mis-sorts artifacts between adjacent windows; nothing else here depends on it.

The roots are the ones the launcher declares (`launchers/start-st-glm53.sh`), read from the environment
rather than named here -- a cache this file does not know about is one it cannot be wrong about. They all
sit under one directory in production, so nested roots fold into their parent and the walk happens once.

The walk is bounded in files and in seconds, and says which bound it hit. An unbounded walk of somebody
else's cache is exactly the kind of instrument that gets turned off the first time a boot is slow.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ROOT_ENV = ("FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR", "TILELANG_CACHE_DIR", "DG_JIT_CACHE_DIR",
            "CUDA_CACHE_PATH", "ST_MLA_BUILD_ROOT", "ST_DENSE_BUILD_ROOT", "ST_ONESHOT_BUILD_ROOT",
            "ST_NATIVE_BUILD_ROOT")

FILE_CAP = 200_000
SECONDS_CAP = 2.0


def roots(env=None) -> "list[Path]":
    """The declared cache directories that exist, with nested ones folded into their parent."""
    env = os.environ if env is None else env
    found = set()
    for name in ROOT_ENV:
        value = env.get(name)
        if not value:
            continue
        try:
            path = Path(value).resolve()
            if path.is_dir():
                found.add(path)
        except OSError:                       # a root that cannot be read is a root that cannot be walked
            continue
    kept = []
    for path in sorted(found, key=lambda p: (len(p.parts), str(p))):
        if not any(parent in kept for parent in path.parents):
            kept.append(path)
    return kept


class Windows:
    """Named wall-clock windows over a boot, and the cache artifacts that landed inside them."""

    def __init__(self, paths=None, *, file_cap: int = FILE_CAP, seconds_cap: float = SECONDS_CAP):
        self.roots = list(roots() if paths is None else paths)
        self.file_cap, self.seconds_cap = file_cap, seconds_cap
        self.windows: "list[tuple[str, float, float]]" = []
        self.scanned = self.seen = 0
        self.scan_seconds = 0.0
        self.stopped = ""                     # "" while the walk finished; otherwise the bound it hit

    def mark(self, name: str, start: float, end: float) -> None:
        self.windows.append((name, start, end))

    def since(self) -> "float | None":
        return min((start for _, start, _ in self.windows), default=None)

    def scan(self) -> "dict[str, int]":
        """Walk once; count the artifacts newer than the first window, by the window they fell in.

        A file newer than `since` that no window claims is counted under `between` -- the gaps between
        the named windows are a boot's time too, and a count that quietly dropped them would read as
        "nothing compiled" for exactly the part nobody was watching.
        """
        counts = {name: 0 for name, _, _ in self.windows}
        since = self.since()
        if since is None or not self.roots:
            return counts
        counts["between"] = 0
        started = time.perf_counter()
        for root in self.roots:
            if self.stopped:
                break
            for parent, _dirs, files in os.walk(root, onerror=None):
                if self.scanned >= self.file_cap:
                    self.stopped = f"file cap {self.file_cap}"
                    break
                if time.perf_counter() - started > self.seconds_cap:
                    self.stopped = f"{self.seconds_cap:g} s cap"
                    break
                for name in files:
                    self.scanned += 1
                    try:
                        mtime = os.stat(os.path.join(parent, name)).st_mtime
                    except OSError:           # a JIT cache rewrites itself under us; a file that left is not an artifact
                        continue
                    if mtime < since:
                        continue
                    self.seen += 1
                    for window, start, end in self.windows:
                        if start <= mtime < end:
                            counts[window] += 1
                            break
                    else:
                        counts["between"] += 1
        self.scan_seconds = time.perf_counter() - started
        return counts

    def line(self, rank: int, counts: "dict[str, int]") -> str:
        """One log line. It says "no JIT writes" out loud, because that is the answer that gets used."""
        if not self.roots:
            return f"  jit writes: rank {rank} has no declared cache root to walk"
        wrote = {name: n for name, n in counts.items() if n}
        body = ", ".join(f"{name} {n}" for name, n in wrote.items()) if wrote else "none"
        tail = f", stopped at the {self.stopped}" if self.stopped else ""
        return (f"  jit writes: rank {rank} {body} -- {self.seen} artifacts newer than the first window "
                f"over {self.scanned} files in {self.scan_seconds:.2f} s{tail}")


def _selfcheck() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "outer/inner").mkdir(parents=True)
        env = {"TRITON_CACHE_DIR": str(base / "outer/inner"), "FLASHINFER_WORKSPACE_BASE": str(base / "outer"),
               "DG_JIT_CACHE_DIR": str(base / "missing"), "CUDA_CACHE_PATH": ""}
        assert roots(env) == [(base / "outer").resolve()], roots(env)

        w = Windows([base / "outer"])
        t0 = time.time()
        old = base / "outer/old.o"
        old.write_bytes(b"x")
        os.utime(old, (t0 - 100, t0 - 100))
        w.mark("first", t0, t0 + 1)
        w.mark("second", t0 + 2, t0 + 3)
        for name, when in (("a.o", t0 + 0.5), ("b.o", t0 + 0.6), ("c.o", t0 + 2.5), ("gap.o", t0 + 1.5)):
            path = base / "outer/inner" / name
            path.write_bytes(b"x")
            os.utime(path, (when, when))
        counts = w.scan()
        assert counts == {"first": 2, "second": 1, "between": 1}, counts
        assert w.seen == 4 and w.scanned == 5, (w.seen, w.scanned)
        assert "first 2" in w.line(0, counts) and "old" not in w.line(0, counts)

        empty = Windows([base / "outer"])
        empty.mark("only", t0 + 10, t0 + 11)
        assert empty.scan() == {"only": 0, "between": 0}
        assert "none" in empty.line(3, {"only": 0, "between": 0})

        capped = Windows([base / "outer"], file_cap=0)
        capped.mark("only", t0, t0 + 1)
        assert capped.scan()["only"] == 0 and capped.stopped.startswith("file cap")
        assert Windows([]).scan() == {}
    print("  jit_writes: nested roots folded, artifacts sorted into windows, gaps counted, caps honoured OK")


if __name__ == "__main__":
    _selfcheck()
