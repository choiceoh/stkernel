#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Which check answers for this change: the most local one first, the cheapest lane first.

A slow end-to-end number says that something moved, not what. This repo already has the
checks that say what -- some 350 CPU tests, and the one-GPU ST checks the queue admits
beside production -- but nothing said WHICH of them is about the file in hand, so
"relevant CPU tests" and "a short canonical check of the changed path" (EXPERIMENTS.md,
validation by change risk) were left to memory, and the habit was the whole suite or a
boot. This names them, per file:

    python3 bench/feedback.py engine/kernels/mhc_contract.py
    python3 bench/feedback.py --base origin/main          # what this branch changed
    python3 bench/feedback.py --index > feedback.json     # the whole graph, for a code-graph consumer

Nothing here is declared by hand, so nothing here can go stale on its own. A rung's LANE is
what the queue already admits, read from the module that decides it:

    cpu       tests/test_*.py                    tools/check.py, CUDA hidden; seconds to minutes
    single    bench/fleet_onepass.py ST_PROBES   ONE Spark beside production, no fleet drain (fleet_single.py)
    verdict   the ST bracket (fleet.sh st-pair)  four Sparks; the only lane whose answer is a speed verdict

and its DISTANCE is the import path from the check to the file (0 = the file is the check,
1 = the check imports it, or names its repo path in a string). Everything is read with
ast: no module is imported, so this answers on a laptop with no torch exactly as it does
on a Spark.

A probe that reaches the file but that ST_PROBES does not name is listed as `unadmitted`:
the queue refuses it (standalone GPU scripts are rejected), so it is a check nobody can run
through the queue until it is byte-pinned there. Rungs narrow a hypothesis; none of them is
a speed verdict.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LANES = ('cpu', 'single')                        # cost order; `verdict` is always the last rung
# What the repo's own scripts put on sys.path before importing each other.
IMPORT_ROOTS = ('', 'bench', 'tests', 'probes', 'tools')
# Only what ships into serving has a speed verdict, and the ST bracket gives it. The queue,
# the measurement harness and the probes themselves have none.
ST_PAIR = 'bash bench/fleet.sh st-pair <session> <sha>'
VERDICT = (('engine/', ST_PAIR), ('launchers/', ST_PAIR))


def literals(path, names):
    """Top-level literal assignments, read not imported: the queue's policy module stays unexecuted."""
    found = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass                                  # computed, not written: reported by name below
    missing = set(names) - set(found)
    if missing:
        raise ValueError(f'{path}: {", ".join(sorted(missing))} is no longer a top-level literal; bench/feedback.py reads it as one')
    return found


class Graph:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self._direct, self._reach, self._tree = {}, {}, {}
        policy = literals(self.root / 'bench/fleet_onepass.py', ('ST_PROBES', 'ST_PROBE_BUDGET_GIB'))
        self.st_probes, self.budgets = tuple(policy['ST_PROBES']), dict(policy['ST_PROBE_BUDGET_GIB'])

    def rel(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def tree(self, path):
        if path not in self._tree:
            try:
                self._tree[path] = ast.parse(path.read_text(encoding='utf-8'))
            except (OSError, SyntaxError, UnicodeDecodeError):
                self._tree[path] = None
        return self._tree[path]

    def module(self, name):
        bits = name.split('.')
        for base in IMPORT_ROOTS:
            path = self.root.joinpath(base, *bits)
            if path.with_suffix('.py').is_file():
                return path.with_suffix('.py')
            if (path / '__init__.py').is_file():
                return path / '__init__.py'
        return None

    def named(self, text):
        """A string that IS a repo path: how a test reaches a shell script, a launcher or a source it parses."""
        if not text or len(text) > 200 or '/' not in text or '\n' in text or text.startswith(('/', '.', '-')) or '..' in text:
            return None
        path = self.root / text
        return path if path.is_file() else None

    def direct(self, path):
        """(what this file imports, what it names by path); a named file is a leaf, never followed."""
        if path in self._direct:
            return self._direct[path]
        imports, names = set(), set()
        tree = self.tree(path)
        for node in ast.walk(tree) if tree else ():
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    package = path.parent
                    for _ in range(node.level - 1):
                        package = package.parent
                    prefix = '' if package == self.root else package.relative_to(self.root).as_posix().replace('/', '.')
                    base = '.'.join(part for part in (prefix, base) if part)
                modules = [base, *(base + '.' + alias.name for alias in node.names)]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                target = self.named(node.value)
                if target is not None and target != path:
                    names.add(target)
            for name in modules:
                bits = name.split('.')
                for end in range(1, len(bits) + 1):   # every parent package's __init__ runs too
                    target = self.module('.'.join(bits[:end]))
                    if target is not None and target != path:
                        imports.add(target)
        self._direct[path] = (imports, names)
        return self._direct[path]

    def reach(self, source):
        """{file: distance} from one check, breadth first over imports; names only from the check itself."""
        if source in self._reach:
            return self._reach[source]
        imports, names = self.direct(source)
        distance = {path: 1 for path in imports | names}
        frontier, depth = list(imports), 1
        while frontier:
            depth += 1
            following = []
            for path in frontier:
                for target in self.direct(path)[0]:
                    if target not in distance and target != source:
                        distance[target] = depth
                        following.append(target)
            frontier = following
        self._reach[source] = distance
        return distance

    def answers(self, path):
        tree = self.tree(path)
        text = (ast.get_docstring(tree) or '') if tree else ''
        return text.strip().splitlines()[0].strip() if text.strip() else ''

    def sources(self):
        """Every check the repo carries, with the lane the queue gives it today."""
        out = []
        for path in sorted((self.root / 'tests').glob('test_*.py')):
            # tools/check.py: the four-state verdict, PYTHONPATH and hidden CUDA included
            out.append(dict(path=self.rel(path), lane='cpu', command=f'python3 tools/check.py --pattern {path.stem}'))
        probes = {self.rel(p) for p in (self.root / 'probes').glob('*.py')} | set(self.st_probes)
        for relative in sorted(probes):
            if not (self.root / relative).is_file():
                continue
            if relative in self.st_probes:
                out.append(dict(path=relative, lane='single', budget_gib=self.budgets.get(relative, 8),
                                command=f'bash bench/fleet.sh run --gpu <session> -- bash probes/run_engine_probe.sh {relative}'))
            else:
                out.append(dict(path=relative, lane='unadmitted', command=None))
        return out

    def rungs(self, target, limit=3):
        """The ladder for one file: lanes in cost order, the nearest checks of each, the verdict last."""
        path = (self.root / target).resolve()
        relative = self.rel(path)
        ladder, more, found = [], {}, []
        for source in self.sources():
            # a check that was itself edited is its own nearest answer
            distance = 0 if source['path'] == relative else self.reach(self.root / source['path']).get(path)
            if distance is not None:
                found.append(dict(source, distance=distance, breadth=len(self.reach(self.root / source['path'])),
                                  answers=self.answers(self.root / source['path'])))
        found.sort(key=lambda r: (r['distance'], r['breadth'], r['path']))
        for lane in LANES:
            mine = [r for r in found if r['lane'] == lane]
            ladder += mine if limit is None else mine[:limit]
            if limit is not None and len(mine) > limit:
                more[lane] = len(mine) - limit
        unadmitted = [r['path'] for r in found if r['lane'] == 'unadmitted' and r['distance'] <= 1]
        verdict = next((command for prefix, command in VERDICT if relative.startswith(prefix)), None)
        return dict(file=relative, rungs=ladder, more=more, unadmitted=unadmitted,
                    verdict=verdict, local=any(r['distance'] <= 1 for r in ladder))

    def index(self, depth=2):
        """The whole graph: every check, its lane, and the files within `depth` imports of it."""
        entries = []
        for source in self.sources():
            path = self.root / source['path']
            reaches = {self.rel(p): d for p, d in sorted(self.reach(path).items(), key=lambda kv: (kv[1], str(kv[0]))) if d <= depth}
            entries.append(dict(source, answers=self.answers(path), reaches=reaches))
        return dict(schema=1, lanes=[*LANES, 'verdict'], depth=depth, checks=entries,
                    scope='rungs narrow a hypothesis; only the verdict lane is a speed verdict')


def changed(root, base):
    out = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', '--diff-filter=d', base, 'HEAD'], text=True)
    return [line for line in out.splitlines() if line.strip()]


def render(ladder):
    lines = [ladder['file']]
    for rung in ladder['rungs']:
        note = f"   ({rung['budget_gib']} GiB beside production)" if rung.get('budget_gib') else ''
        lines.append(f"  {rung['lane']:<9}d={rung['distance']}  {rung['command']}{note}")
        if rung.get('answers'):
            lines.append(f"  {'':<9}     {rung['answers'][:110]}")
    if not ladder['rungs']:
        lines.append('  no CPU test or admitted one-GPU check reaches this file')
    elif not ladder['local']:
        lines.append('  note: nothing reaches it at distance 1 -- every rung above tests it through something else')
    for lane, count in ladder['more'].items():
        lines.append(f'  {lane:<9}+{count} farther (--all)')
    if ladder['unadmitted']:
        lines.append(f"  unadmitted: {len(ladder['unadmitted'])} probe(s) reach it at distance 1 but bench/fleet_onepass.py ST_PROBES names none "
                     f"of them, so the queue refuses them: {', '.join(ladder['unadmitted'][:4])}{' ...' if len(ladder['unadmitted']) > 4 else ''}")
    if ladder['verdict']:
        lines.append(f"  verdict       {ladder['verdict']}   (the speed verdict; no rung above is one)")
    else:
        lines.append('  verdict       none: this file does not ship into serving, so its checks above are its whole answer')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help='repo-relative files')
    parser.add_argument('--base', help='instead of files: what HEAD changed since this revision')
    parser.add_argument('--all', action='store_true', help='every check of each lane, not the nearest three')
    parser.add_argument('--index', action='store_true', help='the whole graph as JSON')
    parser.add_argument('--depth', type=int, default=2, help='--index: how many imports away a file still counts (2)')
    parser.add_argument('--json', action='store_true', help='the ladders as JSON')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    graph = Graph(args.root)
    if args.index:
        print(json.dumps(graph.index(args.depth), indent=1, ensure_ascii=False))
        return 0
    files = changed(graph.root, args.base) if args.base else args.files
    if not files and args.base:
        print(f'HEAD changed no file since {args.base}')
        return 0
    if not files:
        parser.error('name files, or --base REV, or --index')
    ladders = []
    for name in files:
        if not (graph.root / name).is_file():
            print(f'{name}: not a file in {graph.root}', file=sys.stderr)
            return 2
        ladders.append(graph.rungs(name, None if args.all else 3))
    print(json.dumps(ladders, indent=1, ensure_ascii=False) if args.json else '\n\n'.join(render(l) for l in ladders))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
