"""Versioned, closed-world reasoning tasks and deterministic proof checking.

Only visible content is graded. Oracles and grading stay outside timed streams;
no model-as-judge, device imports or network access. See ONEPASS_QUALITY.md.
"""
import hashlib
import itertools
import json
import random
import re

from measurement_contract import MAX_TOKENS, COMBINED_MAX_TOKENS, COMBINED_REASONING_BUDGET

VERSION = 'ko-reasoning-v2'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _case(name, evidence, question, answer, hints=None):
    return dict(id=name, evidence=evidence, question=question, answer=answer, schema_hints=hints or {})


def _ledger(rng):
    opening, boxes, unit = rng.randrange(170, 220), rng.randrange(13, 20), rng.choice([6, 8, 12])
    shipped, returned, reserve = rng.randrange(40, 60), rng.randrange(7, 16), rng.randrange(25, 45)
    loss = rng.choice([5, 10, 15])
    gross = opening + boxes * unit - shipped + returned
    usable = gross * (100 - loss) // 100 - reserve
    demand = usable + rng.choice([-1, 1]) * rng.randrange(2, 8)
    extra = (usable - demand + rng.randrange(2, 8) if usable >= demand
             else usable - demand - rng.randrange(2, 8))
    evidence = {
        'L1': '마감은 10일 18시다. 효력일과 기록일이 모두 마감 이전인 행만 고려한다. '
              '같은 거래는 남은 행 중 가장 높은 개정 번호로 교체한다. 취소 개정은 수량 0이다. 개정들을 합산하지 않는다.',
        'L2': f'기초 재고 {opening}개. 1상자는 {unit}개다. 입고와 반품은 더하고 출고는 뺀다.',
        'L3': f'순재고 전체에 손실률 {loss}%를 한 번 적용하고 소수 부분을 버린 뒤 예약 {reserve}개를 뺀 값이 가용량이다.',
        'L4': f'주문 {demand}개 이상이면 전량승인, 미만이면 보류다. 부분승인은 없다.',
        'L5': f'가정 변경: 예약만 {abs(extra)}개 {"추가" if extra > 0 else "해제"}한다. 다른 조건과 거래는 그대로다.',
        'L10': f'거래 입고갑 개정1: 효력7일, 기록7일, 입고 {boxes - 3}상자.',
        'L11': f'거래 입고갑 개정2: 효력7일, 기록9일, 입고 {boxes}상자.',
        'L12': f'거래 입고갑 개정3: 효력7일, 기록11일, 입고 {boxes + 5}상자.',
        'L13': f'거래 출고을 개정1: 효력8일, 기록8일, 출고 {shipped}개.',
        'L14': f'거래 출고을 개정2: 효력11일, 기록9일, 출고 {shipped + 11}개.',
        'L15': '거래 입고병 개정1: 효력8일, 기록8일, 입고 9상자.',
        'L16': '거래 입고병 개정2: 효력8일, 기록10일 12시, 취소.',
        'L17': f'거래 반품정 개정1: 효력9일, 기록9일, 반품 {returned}개.',
    }
    answer = dict(
        result=dict(available=usable, decision='전량승인' if usable >= demand else '보류'),
        derivation=dict(selected=['L11', 'L13', 'L16', 'L17'], received=boxes * unit,
                        net=gross, after_loss=gross * (100 - loss) // 100),
        evidence=dict(selection=['L1', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17'],
                      arithmetic=['L2', 'L3'], decision=['L4'], counterfactual=['L5']),
        counterfactual=dict(available=usable - extra, shortfall=max(0, demand - (usable - extra)),
                            decision='전량승인' if usable - extra >= demand else '보류'))
    return _case('ledger', evidence,
        '마감 시점의 전량승인 여부를 판정하라. result와 derivation은 L5를 적용하지 않은 기본 조건이고 counterfactual만 L5를 적용한다. '
        'selected에는 취소 개정을 포함해 최종 채택한 거래 행 ID를 적는다. received는 채택한 입고 상자 수에 상자당 개수를 곱한 값이며 '
        '반품과 출고는 넣지 않는다. net은 기초 재고에 채택한 입고·반품을 더하고 출고를 뺀 순재고, after_loss는 순재고에 손실률을 '
        '적용하고 소수 부분을 버린 값이다. counterfactual에는 L5 적용 후 가용량, 주문 부족량(부족하지 않으면 0), 판정을 적는다. '
        'evidence.selection에는 마감·개정 규칙과 모든 거래 행, arithmetic에는 단위·손실 규칙, decision에는 승인 규칙, '
        'counterfactual에는 가정 변경 규칙의 ID를 넣는다.', answer,
        hints={'result.decision': '전량승인|보류', 'counterfactual.decision': '전량승인|보류',
               'derivation.selected': ['거래 행 ID']})


def _portfolio_rows(projects, budget, staff):
    rows = []
    for ids in itertools.combinations('ABCDE', 3):
        cost, people, upside, downside, risk = (sum(projects[k][i] for k in ids) for i in range(5))
        violations = []
        if cost > budget: violations.append('budget')
        if people > staff: violations.append('staff')
        if 'B' in ids and 'A' not in ids: violations.append('dependency')
        if 'C' in ids and 'E' in ids: violations.append('conflict')
        rows.append(dict(ids=''.join(ids), cost=cost, staff=people,
                         score=None if violations else min(upside, downside) - 2 * risk, violations=violations))
    return rows


def _rank(rows):
    return sorted((r for r in rows if not r['violations']), key=lambda r: (-r['score'], r['ids']))


def _portfolio(rng):
    # Require meaningful feasible alternatives and a changed optimum under the
    # counterfactual. The bounded loop cannot hang on an unlucky seed.
    for _ in range(10000):
        projects = {k: [rng.randrange(3, 10), rng.randrange(1, 5), rng.randrange(12, 41),
                        rng.randrange(12, 41), rng.randrange(0, 5)] for k in 'ABCDE'}
        budget, staff = rng.randrange(16, 24), rng.randrange(7, 12)
        rows = _portfolio_rows(projects, budget, staff)
        changed = _portfolio_rows(projects, budget - 3, staff)
        ranked, alternate = _rank(rows), _rank(changed)
        if len(ranked) >= 2 and alternate and ranked[0]['ids'] != alternate[0]['ids']:
            break
    else:
        raise RuntimeError('could not construct nontrivial portfolio')
    evidence = {
        'O1': f'사업 A~E 중 정확히 3개를 고른다. 예산 상한 {budget}, 인력 상한 {staff}. 등호는 허용한다.',
        'O2': 'B를 선택하려면 A도 선택해야 한다. A만 선택하는 것은 허용된다.',
        'O3': 'C와 E는 동시에 선택할 수 없다.',
        'O4': '점수는 min(선택 사업의 호황 수익 합, 불황 수익 합) - 2×위험 합이다. '
              '사업별 최솟값을 먼저 합산하지 않는다. 가능한 조합 중 점수 최대를 택하고 동점은 조합 ID 사전순이다.',
        'O5': '가정 변경: 예산 상한만 3 줄인다. 다시 최적 조합을 선택한다.',
    }
    for k, values in projects.items():
        evidence['O' + k] = f'사업 {k}: 비용, 인력, 호황 수익, 불황 수익, 위험 = ' + ', '.join(map(str, values))
    best, runner = ranked[:2]
    answer = dict(result=dict(best=best['ids'], score=best['score'], runner_up=runner['ids'],
                              margin=best['score'] - runner['score']),
        derivation=dict(candidates=rows),
        evidence=dict(constraints=['O1', 'O2', 'O3'], objective=['O4'],
                      projects=['OA', 'OB', 'OC', 'OD', 'OE'], counterfactual=['O5']),
        counterfactual=dict(best=alternate[0]['ids'], score=alternate[0]['score'],
                            feasible=[r['ids'] for r in alternate]))
    return _case('portfolio', evidence,
        '기본 조건의 최적 조합, 차선 조합, 점수 차를 구하라. 최적성 증명으로 10개 조합 전체의 비용·인력·위반 목록을 '
        'candidates에 제출하라. 점수는 위반이 없는 조합만 계산하고 위반이 있는 조합의 score는 null이다. '
        '위반 이름은 budget, staff, dependency, conflict이고 해당하는 위반을 모두 기재한다. 없으면 []이다. '
        'O5 변경 후 최적 조합과 점수, 가능한 모든 조합 ID도 제시하라. 조합 ID는 문자를 사전순으로 붙인다. '
        'evidence.constraints에는 선택·예산·인력·의존·배타 규칙, objective에는 점수 규칙, projects에는 사업 기록, '
        'counterfactual에는 가정 변경 규칙의 ID를 넣는다.', answer,
        hints={'derivation.candidates.ids': '조합 ID', 'derivation.candidates.score': 'integer|null',
               'derivation.candidates.violations': ['budget|staff|dependency|conflict'],
               'result.best': '조합 ID', 'result.runner_up': '조합 ID',
               'counterfactual.best': '조합 ID', 'counterfactual.feasible': ['조합 ID']})


def _rules(bits):
    a, b, c, d, e, f = bits
    return [bool(a or b), b == c, not (c and d), a + d + e >= 2, e != f, a == 0]


def _logic(rng):
    labels = rng.sample(list('ABCDEF'), 6)
    a, b, c, d, e, f = labels
    # Variable roles and query order vary by seed. Output bit positions remain
    # alphabetical so that renaming the inputs changes the actual certificate.
    universe = list(itertools.product((0, 1), repeat=6))
    worlds = [w for w in universe if all(_rules(w)[:5])]
    strings = [''.join(str(w[labels.index(k)]) for k in 'ABCDEF') for w in worlds]
    queries = [lambda w: w[0] == 1, lambda w: w[1] == 1, lambda w: w[3] == 1,
               lambda w: w[3] == 0 and w[4] == 0]
    query_texts = [f'{a}=1', f'{b}=1', f'{d}=1', f'({d}=0 그리고 {e}=0)']
    order = rng.sample(range(4), 4)
    statuses = []
    witnesses = []
    for index in order:
        query = queries[index]
        true = [s for w, s in zip(worlds, strings) if query(w)]
        false = [s for w, s in zip(worlds, strings) if not query(w)]
        statuses.append('판단불가' if true and false else '참' if true else '거짓')
        witnesses.append(dict(true=true, false=false))
    cores = []
    for size in range(1, 7):
        for indices in itertools.combinations(range(6), size):
            if any(set(core) <= set(indices) for core in cores):
                continue
            if not any(all(_rules(w)[i] for i in indices) for w in universe):
                cores.append(indices)
    evidence = {
        'U0': f'변수는 {", ".join(labels)}이며 각각 0 또는 1이다. 기록에 없는 조건은 추가하지 않는다. '
              '가능 세계는 변수 이름의 알파벳 순서 ABCDEF로 나열한 6자리 비트 문자열로 표시한다. '
              '예: A=1, B=0, C=0, D=1, E=1, F=0이면 100110이다.',
        'U1': f'{a} 또는 {b} 중 적어도 하나는 1이다.',
        'U2': f'{b}와 {c}의 값은 같다.',
        'U3': f'{c}와 {d}가 동시에 1일 수 없다.',
        'U4': f'{a}, {d}, {e} 중 적어도 둘이 1이다.',
        'U5': f'{e}와 {f} 중 정확히 하나가 1이다.',
        'U6': f'가정 변경: U1~U5를 모두 유지하고 {a}=0을 추가한다.',
    }
    answer = dict(result=dict(statuses=statuses), derivation=dict(worlds=strings),
        evidence=dict(domain=['U0'], constraints=['U1', 'U2', 'U3', 'U4', 'U5'], counterfactual=['U6']),
        counterfactual=dict(consistent=False, minimal_cores=[['U' + str(i + 1) for i in core] for core in cores]))
    # Witnesses are validated semantically; any valid true/false world is fine.
    answer['witnesses'] = [dict(true=x['true'][0] if x['true'] else None,
                              false=x['false'][0] if x['false'] else None) for x in witnesses]
    case = _case('logic', evidence,
        '명제 순서 ' + ', '.join(f'Q{i + 1}={query_texts[q]}' for i, q in enumerate(order)) + '에 대해 '
        'U1~U5를 모두 만족하는 가능 세계 전체에서 참이면 참, 모두 거짓이면 거짓, 양쪽이 존재하면 판단불가로 분류하라. '
        '가능 세계 전체를 worlds에 열거하고 명제별 true/false 증인 세계 하나씩을 제출하라. '
        '해당 증인이 없을 때만 null이다. statuses와 witnesses의 배열 순서는 Q1~Q4다. '
        'U6을 추가했을 때의 일관성 여부(consistent)와, 모순이면 모든 최소 모순 규칙 집합을 minimal_cores에 제시하라. '
        '최소란 어떤 한 규칙을 빼도 모순이 사라진다는 뜻이다. U0의 이진 정의는 고정이며 core에 넣지 않는다. '
        'evidence.domain에는 변수 정의, constraints에는 U1~U5, counterfactual에는 가정 변경 규칙의 ID를 넣는다.', answer,
        hints={'result.statuses': ['참|거짓|판단불가'], 'derivation.worlds': ['6자리 비트열'],
               'witnesses.true': '6자리 비트열|null', 'witnesses.false': '6자리 비트열|null',
               'counterfactual.minimal_cores': [['규칙 ID']]})
    case['witness_options'] = witnesses
    return case


def cases(seed):
    rng = random.Random(seed)
    return [_ledger(rng), _portfolio(rng), _logic(rng)]


def _shape(value, hints=None, path=''):
    """The JSON skeleton shown to the model. Enumerated fields spell out their
    choices, evidence arrays hold record IDs, everything else names its type."""
    hints = hints or {}
    if path in hints: return hints[path]
    if isinstance(value, dict): return {k: _shape(v, hints, f'{path}.{k}' if path else k) for k, v in value.items()}
    if isinstance(value, list): return [_shape(value[0], hints, path)] if value else ['...']
    if path.partition('.')[0] == 'evidence': return '기록 ID'
    return 'boolean' if type(value) is bool else 'integer' if type(value) is int else 'string|null'


def request_item(ctx, seed, selected, filler, max_tokens, reasoning_budget, question):
    evidence = [(key, value) for case in selected for key, value in case['evidence'].items()]
    rng = random.Random(seed)
    rng.shuffle(evidence)
    # Context labels are approximate document sizes, not total input tokens.
    # Disperse dependencies across the entire body; never append a short answer
    # summary next to the questions. Actual prompt tokens are recorded by SSE.
    evidence_chars = sum(len(k) + len(v) + 4 for k, v in evidence)
    per = max(0, ctx - evidence_chars / 1.24) / (len(evidence) + 1)
    chunks = []
    for key, value in evidence:
        chunks.extend([filler(per, rng), f'[{key}] {value}'])
    chunks.append(filler(per, rng))
    schemas = {case['id']: _shape(case['answer'], case.get('schema_hints')) for case in selected}
    content = ('아래 문서의 [L*], [O*], [U*] 기록만 해당 문제의 근거다. 일반 배경 문단은 규칙이 아니다.\n문서:\n'
        + '\n'.join(chunks) + '\n\n문제:\n'
        + '\n'.join(case['id'] + ': ' + case['question'] for case in selected)
        + '\n최종 답변은 다음 구조의 JSON 객체 하나만 제출하라. 코드 펜스는 허용하지만 JSON 밖에는 표·설명·문장을 쓰지 않는다. '
          '표시된 자료형은 실제 값으로 대체하고 "a|b" 꼴은 그중 하나를 그대로 적는다. 배열은 요구한 모든 항목을 채우고 '
          '추가 필드는 넣지 않는다. 기록 ID는 L1, O4, U2처럼 대괄호 없이 적고 설명을 붙이지 않는다. '
          '숫자는 정수, 없는 값은 null이다. witnesses와 statuses 외 배열의 순서는 무관하다.\n'
        + json.dumps(schemas, ensure_ascii=False))
    return dict(ctx=ctx, question=question, content=content, max_tokens=max_tokens,
                reasoning_budget=reasoning_budget, quality_cases=selected, quality_version=VERSION)


def visible(events):
    return ''.join(d['content'] for d in events if isinstance(d.get('content'), str))


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out


def _canonical(value, ordered=False):
    if isinstance(value, dict):
        return {k: _canonical(v, k in ('statuses', 'witnesses')) for k, v in value.items()}
    if isinstance(value, list):
        values = [_canonical(v) for v in value]
        return values if ordered else sorted(values, key=lambda x: json.dumps(x, sort_keys=True))
    # Preserve JSON types: True must never match the integer 1.
    return [type(value).__name__, value]


def assess(item, events, finish, *, fixed=False):
    raw_text = visible(events)
    text = raw_text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if fenced: text = fenced[1].strip()
    error, parsed = None, None
    try:
        parsed = json.loads(text, object_pairs_hook=_object,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON number')))
        if not isinstance(parsed, dict): raise ValueError('expected one JSON object')
        if set(parsed) != {c['id'] for c in item['quality_cases']}:
            raise ValueError('missing or unexpected case IDs')
    except (ValueError, RecursionError) as exc:
        error = str(exc)
    results = []
    for case in item['quality_cases']:
        actual = parsed.get(case['id']) if not error else None
        complete = finish == 'stop' or (fixed and finish == 'length')
        checks = dict(format=not error and isinstance(actual, dict) and set(actual) == set(case['answer']),
                      completion=complete)
        failures = []
        for dimension, expected in case['answer'].items():
            value = actual.get(dimension) if isinstance(actual, dict) else None
            if dimension == 'witnesses':
                checks[dimension] = (isinstance(value, list) and len(value) == 4 and all(
                    isinstance(v, dict) and set(v) == {'true', 'false'} and all(
                        (v[k] in options[k] if options[k] else v[k] is None) for k in ('true', 'false'))
                    for v, options in zip(value, case['witness_options'])))
            else:
                checks[dimension] = _canonical({dimension: value}) == _canonical({dimension: expected})
            if not checks[dimension]:
                failures.append(dict(dimension=dimension, expected=expected, actual=value))
        results.append(dict(case=case['id'], version=VERSION, passed=all(checks.values()), checks=checks,
                            score=sum(checks.values()), max_score=len(checks), failures=failures,
                            parse_error=error, finish_reason=finish,
                            visible_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
                            oracle_sha256=digest(case)))
    return results


def summarize(results):
    dimensions = {}
    for row in results:
        for name, good in row['checks'].items():
            d = dimensions.setdefault(name, dict(ok=0, total=0))
            d['ok'] += int(good)
            d['total'] += 1
    return dict(version=VERSION, ok=sum(r['passed'] for r in results), total=len(results),
                score=sum(r['score'] for r in results), max_score=sum(r['max_score'] for r in results),
                dimensions=dimensions, scope='visible final content; deterministic proof certificates')
