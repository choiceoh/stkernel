#!/usr/bin/env python3
"""Which lines you just wrote are not actually pinned by a test?

The house standard for "is this guard real" is to delete it and see whether anything notices. Doing that by hand is
fifteen rounds of sed per change and it is easy to forget one, so this does it: for every line this branch ADDED,
break that line and run the tests. A mutation the tests survive is a line nobody is checking.

    python3 tools/mutate.py --tests tests.test_engine_prefix
    python3 tools/mutate.py --tests tests.test_engine_serve --file engine/base/serve.py
    python3 tools/mutate.py --tests tests.test_engine_prefix --ref HEAD~1

Mutations, chosen because they are unambiguous rather than clever:
    a simple statement       -> pass          (the line stops doing its work)
    `if <test>:` on one line -> if True: / if False:   (the branch stops being a decision)

Lines that are structure rather than behaviour -- def, class, return, raise, import, decorators, docstrings -- are
left alone, as are statements spanning more than one line. The file is restored in a `finally`; if this is
interrupted, `git diff` will show you what is left over.
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIMPLE = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Delete, ast.Assert)


def added_lines(ref: str, path: pathlib.Path) -> "set[int]":
    """Line numbers in the working copy of `path` that `ref` does not have."""
    diff = subprocess.run(["git", "diff", "-U0", ref, "--", str(path)], cwd=ROOT, capture_output=True, text=True).stdout
    lines, line = set(), 0
    for row in diff.splitlines():
        if row.startswith("@@"):
            line = int(row.split("+")[1].split(",")[0].split()[0])
        elif row.startswith("+") and not row.startswith("+++"):
            lines.add(line)
            line += 1
        elif not row.startswith("-"):
            line += 1
    return lines


def mutants(source: str, wanted: "set[int]"):
    """(line number, replacement text for that line, what it does) for each mutation worth trying."""
    tree = ast.parse(source)
    rows = source.splitlines()
    out = []
    for node in ast.walk(tree):
        at = getattr(node, "lineno", None)
        if at is None or at not in wanted:
            continue
        indent = rows[at - 1][: len(rows[at - 1]) - len(rows[at - 1].lstrip())]
        if isinstance(node, SIMPLE) and node.end_lineno == at:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue                                            # a docstring is not behaviour
            out.append((at, indent + "pass", "dropped"))
        elif isinstance(node, ast.If) and node.test.end_lineno == at and rows[at - 1].rstrip().endswith(":"):
            was = ast.get_source_segment(source, node.test)
            if was and "\n" not in was:
                for constant in ("True", "False"):
                    out.append((at, rows[at - 1].replace(was, constant, 1), f"if -> {constant}"))
    return sorted(set(out))


def survives(tests, jobs: int, timeout: int) -> bool:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")])))
    try:
        p = subprocess.run([sys.executable, "-m", "unittest", *tests], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False                                                 # a mutant that hangs was noticed
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", required=True, help="unittest module paths, e.g. tests.test_engine_prefix")
    ap.add_argument("--ref", default="origin/main", help="the lines you added are the ones this ref lacks")
    ap.add_argument("--file", nargs="*", help="restrict to these files (default: every non-test file you changed)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=ROOT, check=False)
    if a.file:
        files = [pathlib.Path(f) for f in a.file]
    else:
        changed = subprocess.run(["git", "diff", "--name-only", a.ref], cwd=ROOT, capture_output=True, text=True).stdout
        files = [pathlib.Path(f) for f in changed.split()
                 if f.endswith(".py") and not pathlib.Path(f).name.startswith("test_")]
    if not files:
        print(f"  nothing changed against {a.ref}")
        return 0

    print(f"  baseline: ", end="", flush=True)
    if not survives(a.tests, a.jobs, a.timeout):
        print("the tests do not pass before any mutation -- fix that first")
        return 2
    print("the tests pass\n")

    unkilled, tried = [], 0
    for rel in files:
        path = ROOT / rel
        if not path.exists():
            continue
        original = path.read_text()
        wanted = added_lines(a.ref, rel)
        todo = mutants(original, wanted) if wanted else []
        if not todo:
            continue
        print(f"  {rel}: {len(todo)} mutations over {len(wanted)} added lines")
        rows = original.splitlines(keepends=True)
        try:
            for at, replacement, what in todo:
                broken = list(rows)
                broken[at - 1] = replacement + ("\n" if rows[at - 1].endswith("\n") else "")
                text = "".join(broken)
                try:
                    ast.parse(text)
                except SyntaxError:
                    continue                                         # the mutation did not make a program
                path.write_text(text)
                tried += 1
                alive = survives(a.tests, a.jobs, a.timeout)
                print(f"    {'SURVIVED' if alive else 'killed  '}  {rel}:{at}  {what}  {original.splitlines()[at - 1].strip()[:70]}")
                if alive:
                    unkilled.append((rel, at, what, original.splitlines()[at - 1].strip()))
        finally:
            path.write_text(original)

    print(f"\n  {tried} mutations, {len(unkilled)} survived")
    for rel, at, what, text in unkilled:
        print(f"  NOT PINNED  {rel}:{at}  ({what})  {text[:80]}")
    return 1 if unkilled else 0


if __name__ == "__main__":
    sys.exit(main())
