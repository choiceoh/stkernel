#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ST Oracle: compare a commit with actual in-development engine source.

    python3 bench/storacle.py compare --base origin/main
    python3 bench/storacle.py predict --base HEAD --json
    python3 bench/storacle.py compare --base HEAD~1 --candidate HEAD --profile paired.json

The working-tree snapshot includes staged, unstaged and untracked engine source.
Source-derived shape/byte changes feed the same cost model on both sides. Changed
kernel timings need a source-bound paired profile; unpriced work is never a 0% win.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import step_kernels as kernels
import step_acceptance as acceptance

HERE = Path(__file__).resolve().parent
EXTENSIONS = {'.py', '.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.hpp', '.json', '.jinja', '.sh'}
COMPONENTS = dict(moe='MoE 전문가', nonmoe='비MoE(정적·dense·KDA·글루)',
                  attention='어텐션·인덱서(컨텍스트)', communication='집합통신', drafter='드래프터(W4)')
ALL_COSTS = set(COMPONENTS) | {'prefill'}
FACTS_PATH = 'engine/profiles/glm53/facts.py'
BOOT_PATH = 'engine/profiles/glm53/boot.py'
DECLARATIONS = {FACTS_PATH: {'SPEC_K', 'TP', 'BLOCK', 'CHUNK_ALIGN', 'KV_DTYPE', 'KDA_STATE_DTYPE'},
                BOOT_PATH: {'TOKEN_BUDGET', 'MAX_SEQS', 'MAX_WAIT_S'}}


def digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def git(root, *args) -> bytes:
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace').strip())
    return result.stdout


def is_source(path):
    return path.startswith('engine/') and Path(path).suffix in EXTENSIONS


@dataclass
class Source:
    ref: str
    commit: str
    files: dict[str, bytes]
    dirty: bool = False

    @property
    def sha256(self):
        return digest({p: digest(v) for p, v in self.files.items()})

    @property
    def identity(self):
        return dict(ref=self.ref, commit=self.commit, sha256=self.sha256, dirty=self.dirty,
                    files=len(self.files), scope='engine source; external libraries belong to the profile runtime')

    @classmethod
    def read(cls, root: Path, ref: str):
        if ref == 'working-tree':
            head = git(root, 'rev-parse', 'HEAD').decode().strip()
            names = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', 'engine').split(b'\0')
            files = {}
            for raw in set(names) - {b''}:
                name = os.fsdecode(raw)
                path = root / name
                if not is_source(name) or not path.exists():
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f'source symlink is not a self-contained snapshot: {name}')
                files[name] = path.read_bytes()
            dirty = bool(git(root, 'status', '--porcelain', '--untracked-files=all', '--', 'engine'))
            return cls(ref, head, files, dirty)
        commit = git(root, 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}').decode().strip()
        files = {}
        with tarfile.open(fileobj=io.BytesIO(git(root, 'archive', commit, 'engine'))) as archive:
            for member in archive:
                if is_source(member.name):
                    if not member.isfile():
                        raise ValueError(f'non-file engine source: {member.name}')
                    files[member.name] = archive.extractfile(member).read()
        return cls(ref, commit, files)

    def inspect(self, settings=None, config=None):
        with tempfile.TemporaryDirectory(prefix='st-oracle-source-') as directory:
            root = Path(directory)
            for name, data in self.files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            result = subprocess.run([sys.executable, '-I', str(HERE / 'step_source_probe.py'), str(root)],
                                    input=json.dumps(dict(settings=settings or {}, config=config)),
                                    capture_output=True, text=True, timeout=30,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
            try:
                data = json.loads(result.stdout)
            except ValueError as exc:
                raise ValueError(f'{self.ref}: source probe failed: {result.stderr[-1500:]}') from exc
            if result.returncode or 'error' in data:
                raise ValueError(f'{self.ref}: {data.get("error", result.stderr[-1500:])}')
        # Settings/config are inputs too. A clean HEAD and an experimental recipe
        # at that HEAD must not consume each other's measurements.
        data['fingerprint'] = digest(dict(source=self.sha256, settings=settings or {}, config=config, probe=data))
        data['source'] = self.identity
        return data


def syntax(source, mask=()):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in mask for t in node.targets):
            node.value = ast.Constant(value='<source-derived declaration>')
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def components(path):
    if any(f'/profiles/{name}/' in path for name in ('qwen38', 'dsv41')):
        return set()
    if any(x in path for x in ('/moe', '/router', '/route')):
        return {'moe', 'prefill'}
    if any(x in path for x in ('/drafter', '/draft_', '/draft/')):
        return {'drafter', 'prefill'}
    if any(x in path for x in ('/index', '/sparse_', '/mla')):
        return {'attention', 'prefill'}
    if any(x in path for x in ('/comm', '/oneshot', '/transport', '/rank_sum')):
        return {'communication', 'prefill'}
    if any(x in path for x in ('/kda', '/mhc', '/hyper_connection', '/linear_attention', '/caches')):
        return {'nonmoe', 'prefill'}
    return ALL_COSTS.copy()


def changes(base: Source, candidate: Source):
    rows = []
    for path in sorted(base.files.keys() | candidate.files.keys()):
        a, b = base.files.get(path), candidate.files.get(path)
        if a == b:
            continue
        kind, affected = 'unpriced', components(path)
        if not affected:
            kind = 'other_profile'
        if a is not None and b is not None and path.endswith('.py'):
            if syntax(a) == syntax(b):
                kind, affected = 'cosmetic', set()
            elif path in DECLARATIONS and syntax(a, DECLARATIONS[path]) == syntax(b, DECLARATIONS[path]):
                kind, affected = 'source_derived', set()
        rows.append(dict(path=path, change='added' if a is None else 'deleted' if b is None else 'modified',
                         handling=kind, unpriced_components=sorted(affected)))
    return rows


def paired_profile(path, base, candidate):
    if path is None:
        return {}, None
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema') != 1 or data.get('base') != base['fingerprint'] or data.get('candidate') != candidate['fingerprint']:
        raise ValueError('paired profile source/settings fingerprints do not match this comparison')
    if not isinstance(data.get('runtime'), str) or not data['runtime'].strip():
        raise ValueError('paired profile must identify the common hardware/image runtime')
    if not isinstance(data.get('evidence'), str) or not data['evidence'].strip():
        raise ValueError('paired profile must identify its measurement evidence')
    if not isinstance(data.get('rows'), list) or not data['rows']:
        raise ValueError('paired profile needs measured rows')
    rows = {}
    for row in data['rows']:
        if not isinstance(row, dict):
            raise ValueError('paired profile rows must be objects')
        key = (row['phase'], row['ctx'], row['width'])
        if key in rows or key[0] not in ('decode', 'prefill') or any(type(x) is not int or x <= 0 for x in key[1:]):
            raise ValueError('invalid/duplicate paired profile shape')
        values = row['components']
        if not isinstance(values, dict) or not values or values.keys() - (set(COMPONENTS) if key[0] == 'decode' else {'prefill'}):
            raise ValueError('unknown/empty paired profile components')
        for pair in values.values():
            if not isinstance(pair, dict) or type(pair.get('samples')) is not int or pair['samples'] < 1:
                raise ValueError('paired profile needs positive sample counts')
            for arm in ('base_ms', 'candidate_ms'):
                value = pair.get(arm)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('paired profile durations must be finite nonnegative milliseconds')
        rows[key] = values
    return rows, dict(path=str(path), sha256=digest(Path(path).read_bytes()), runtime=data['runtime'], evidence=data['evidence'])


def _budgets(probe, ctx, width, fold):
    b = kernels.EngineBytes.for_model('glm53')
    f, g = probe['facts'], probe['geometry']
    b.spec_k, b.tp = f['spec_k'], f['tp']
    b.experts, b.topk, b.moe_layers = g['experts'], g['topk'], g['moe_layers']
    b.expert_mb *= g['hidden'] * g['inter_local'] / (4096 * 512)
    raw = kernels.decode_step(b, ctx, width).ms
    decode = {key: raw[label] for key, label in COMPONENTS.items()}
    prefill = {'prefill': kernels.prefill_ms(ctx, probe['compute_tile'], fold)} if fold else {}
    return decode, prefill


def compare(base: Source, candidate: Source, *, contexts=(2000, 32000, 128000), widths=(1, 4),
            acc=.45, settings=None, config=None, profile=None, acceptance_profile=None):
    if not math.isfinite(acc) or not 0 <= acc <= 1 or any(type(n) is not int or n <= 0 for n in (*contexts, *widths)):
        raise ValueError('contexts/widths must be positive integers; acceptance must be in [0, 1]')
    left, right = base.inspect(config=config), candidate.inspect(settings=settings, config=config)
    if acceptance_profile is not None:
        acceptance_profile = acceptance.profile(acceptance_profile)
    changed = changes(base, candidate)
    unpriced = {c for row in changed for c in row['unpriced_components']}
    if left['settings'] != right['settings']:
        unpriced |= ALL_COSTS  # changing a kernel recipe is not a measured time multiplier
    if left['facts']['spec_k'] != right['facts']['spec_k']:
        unpriced |= {'nonmoe', 'drafter', 'attention', 'communication'}  # only MoE's token scaling is priced
    if left['facts']['tp'] != right['facts']['tp'] or right['facts']['tp'] != 4:
        unpriced |= ALL_COSTS
    if left['facts']['kv_dtype'] != right['facts']['kv_dtype']:
        unpriced |= {'attention', 'prefill'}
    profile_rows, receipt = paired_profile(profile, left, right)
    artifact = HERE.parent / 'measurements/c4_scaling_20260913/chunk-profile-rank3-sf6.json'
    fold = kernels.fold_prefill_from_profile(artifact) if artifact.exists() else {}
    forecasts = []
    for ctx in contexts:
        for width in widths:
            if any(width > p['contract']['max_running'] or p['settings']['context_ceiling'] and ctx > p['settings']['context_ceiling'] for p in (left, right)):
                forecasts.append(dict(ctx=ctx, width=width, unavailable='outside a source serving contract'))
                continue
            arms = [_budgets(p, ctx, width, fold) for p in (left, right)]
            row = dict(ctx=ctx, width=width)
            for phase, index in (('decode', 0), ('prefill', 1)):
                # Prefill is a single-prompt compute estimate; C=4 request queueing
                # belongs to step_sim, never multiply this by the decode width.
                if phase == 'prefill' and width != 1:
                    continue
                costs = [dict(arm[index]) for arm in arms]
                measured = profile_rows.get((phase, ctx, width), {})
                for name, pair in measured.items():
                    costs[0][name], costs[1][name] = pair['base_ms'], pair['candidate_ms']
                missing = (unpriced & (set(COMPONENTS) if phase == 'decode' else {'prefill'})) - measured.keys()
                totals = [sum(c.values()) if c else None for c in costs]
                if any(ms is not None and ms <= 0 for ms in totals):
                    raise ValueError('paired profile phase totals must be positive')
                move = (totals[1]/totals[0]-1) if totals[0] and totals[1] is not None else None
                row[phase] = dict(base_ms=totals[0], candidate_ms=totals[1], components=dict(base=costs[0], candidate=costs[1]),
                                  modeled_delta=move, delta=move if not missing else None,
                                  unpriced_components=sorted(missing), profiled_components=sorted(measured))
                if phase == 'decode':
                    rates = {}
                    for arm, p, ms in zip(('base', 'candidate'), (left, right), totals):
                        lo, hi = acceptance.yield_bounds(p['facts']['spec_k'], observed=acceptance_profile,
                                                        reference_k=left['facts']['spec_k'], raw_acceptance=acc)
                        point = lo if lo == hi else None
                        rates[arm] = dict(tokens_per_row=point, tokens_per_row_range=[lo, hi],
                                          per_request_tok_s=point*1000/ms if point is not None and ms else None,
                                          aggregate_tok_s=width*point*1000/ms if point is not None and ms else None,
                                          per_request_tok_s_range=[v*1000/ms for v in (lo, hi)] if ms else None,
                                          aggregate_tok_s_range=[width*v*1000/ms for v in (lo, hi)] if ms else None)
                    row[phase]['output_rate_assumption'] = rates
            forecasts.append(row)
    return dict(schema=1, base=left, candidate=right, changed_files=changed, forecasts=forecasts,
                memory_delta={name: right['memory'][name]-left['memory'][name]
                              for name in left['memory'] if isinstance(left['memory'][name], (int, float))},
                profile=receipt, model_sha256=digest(dict(decode=asdict(kernels.EngineBytes()), prefill=fold)),
                acceptance_scenario=acceptance_profile,
                assumptions=[('observed prefix distribution transferred to both sources and all requested shapes'
                              if acceptance_profile is not None else f'baseline raw acceptance assumed {acc:g}; a different K is bounded, not refitted'),
                             'one bonus token per row; terminal censoring retained; source edits do not prove acceptance or quality',
                             'yield bounds identify missing prefixes, not sampling uncertainty; no context/width invariance is measured',
                             'component timing coefficients transfer from #838 unless replaced by a paired profile',
                             'memory bytes execute source layout; allocated bytes are not GPU traffic or a speedup',
                             'prefill is compute only, excludes queueing, tokenizer, JIT and prefix reuse',
                             'changed unpriced components leave the total delta unknown; modeled subtotal still shown'])


def profile_template(data):
    """Freeze the inputs before measurement; blank durations cannot be consumed."""
    rows = []
    for forecast in data['forecasts']:
        for phase in ('decode', 'prefill'):
            if phase not in forecast:
                continue
            names = forecast[phase]['unpriced_components'] or (COMPONENTS if phase == 'decode' else ('prefill',))
            rows.append(dict(phase=phase, ctx=forecast['ctx'], width=forecast['width'],
                             components={name: dict(base_ms=None, candidate_ms=None, samples=0) for name in names}))
    return dict(schema=1, base=data['base']['fingerprint'], candidate=data['candidate']['fingerprint'],
                inputs={arm: data[arm]['source'] for arm in ('base', 'candidate')}, runtime='', evidence='', rows=rows)


def format_comparison(data):
    lines = ['ST-Oracle · 개발 코드 비교']
    for arm in ('base', 'candidate'):
        p = data[arm]
        lines.append(f"  {arm}: {p['source']['ref']} {p['source']['commit'][:8]}"
                     f" · 소스 {p['source']['sha256'][:12]} · K={p['facts']['spec_k']}"
                     f" · 청크 {p['prefill_chunk']} · 상태 슬롯 {p['memory']['slot_bytes']/2**20:.2f} MiB")
    lines.append(f"  소스 변경 {len(data['changed_files'])}개 · 상태 슬롯 증감 {data['memory_delta']['slot_bytes']/2**20:+.2f} MiB")
    if data['profile']:
        lines.append(f"  계측 profile: {data['profile']['path']} · runtime {data['profile']['runtime']}")
    for row in data['changed_files']:
        lines.append(f"  {row['path']}: {row['handling']}" + (f" · 미계측 {','.join(row['unpriced_components'])}" if row['unpriced_components'] else ''))
    for row in data['forecasts']:
        lines.append(f"ctx={row['ctx']} C={row['width']}")
        if row.get('unavailable'):
            lines.append('  ' + row['unavailable'])
            continue
        for phase in ('decode', 'prefill'):
            if phase not in row:
                continue
            p = row[phase]
            if p['base_ms'] is None:
                lines.append(f'  {phase}: 계측 계수 없음')
                continue
            delta = f"{p['delta']:+.1%}" if p['delta'] is not None else '전체 변화 미확정'
            lines.append(f"  {phase}: 모형 {p['base_ms']:.2f} → {p['candidate_ms']:.2f} ms · {delta}")
            if p['profiled_components']:
                lines.append(f"    계측 반영: {', '.join(p['profiled_components'])}")
            if p['unpriced_components']:
                lines.append(f"    미계측: {', '.join(p['unpriced_components'])}; paired profile로 교체")
            if phase == 'decode':
                rates = []
                for arm, rate in p['output_rate_assumption'].items():
                    lo, hi = rate['per_request_tok_s_range']
                    value = f'{lo:.2f}' if rate['per_request_tok_s'] is not None else f'{lo:.2f}–{hi:.2f}'
                    rates.append(f'{arm} {value}')
                lines.append('    수락률 가정·시간 모형의 요청당 tok/s: ' + ' → '.join(rates))
    if data['acceptance_scenario']:
        a = data['acceptance_scenario']
        lines.append(f"수락률: K={a['k']}, {a['rows']}행 관측 분포를 양쪽 코드·컨텍스트·폭에 이관하는 가정.")
    else:
        lines.append('수락률: 기준 K의 평균 가정만 사용; 다른 K는 단일 수치 대신 식별 가능한 범위를 표시.')
    lines.append('범위는 관측되지 않은 prefix의 가능 범위이며 표본 신뢰구간이 아니다. 보너스 1토큰/행을 가정한다.')
    lines.append('과거 시간 계수를 유지한 예측. 상태 메모리 절감·소스 변경만으로 실제 속도/품질 개선을 판정하지 않는다.')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tree', type=Path, default=HERE.parent)
    ap.add_argument('--base', default='origin/main')
    ap.add_argument('--candidate', default='working-tree', help='working-tree (includes uncommitted source) or git ref')
    ap.add_argument('--model', choices=['glm53'], default='glm53', help='source probe currently supports the native GLM53 engine')
    ap.add_argument('--ctx', default='2000,32000,128000')
    ap.add_argument('--width', default='1,4')
    acc_input = ap.add_mutually_exclusive_group()
    acc_input.add_argument('--acc', type=float, help='baseline raw acceptance assumption (default .45)')
    acc_input.add_argument('--acceptance-from', type=Path, help='validated peek JSONL; transfer its prefix distribution as a scenario')
    ap.add_argument('--acceptance-k', type=int, help='observed K for older scrapes lacking lane identity')
    ap.add_argument('--config', type=Path, help='checkpoint config.json for the actual model geometry')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=JSON', help='candidate execution setting; explicit what-if override')
    ap.add_argument('--profile', type=Path, help='paired component timings bound to both source fingerprints')
    ap.add_argument('--write-profile-template', type=Path, help='create an unmeasured template with the current source fingerprints')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    try:
        settings = {}
        for item in args.set:
            name, separator, value = item.partition('=')
            if not separator or name in settings:
                raise ValueError('--set needs unique NAME=JSON entries')
            settings[name] = json.loads(value)
        root = Path(git(args.tree, 'rev-parse', '--show-toplevel').decode().strip())
        observed = acceptance.load_profile(args.acceptance_from, args.acceptance_k) if args.acceptance_from else None
        if args.acceptance_k is not None and observed is None:
            raise ValueError('--acceptance-k requires --acceptance-from')
        if observed is not None:
            observed['receipt'] = dict(path=str(args.acceptance_from), sha256=digest(args.acceptance_from.read_bytes()),
                                       usage='explicit scenario transfer; not measured acceptance of either source snapshot')
        data = compare(Source.read(root, args.base), Source.read(root, args.candidate),
                       contexts=tuple(int(x) for x in args.ctx.split(',')), widths=tuple(int(x) for x in args.width.split(',')),
                       acc=args.acc if args.acc is not None else .45, settings=settings,
                       config=json.loads(args.config.read_text()) if args.config else None,
                       profile=args.profile, acceptance_profile=observed)
        if args.write_profile_template:
            with args.write_profile_template.open('x') as f:
                json.dump(profile_template(data), f, indent=2, ensure_ascii=False)
                f.write('\n')
        print(json.dumps(data, ensure_ascii=False, allow_nan=False) if args.json else format_comparison(data))
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        if args.json:
            print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        else:
            print(f'ST-Oracle source comparison unavailable: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
