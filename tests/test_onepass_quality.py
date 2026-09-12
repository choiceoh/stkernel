"""Hard reasoning oracles, adversarial final answers and canonical C=1/C=4 wiring."""
import ast
import copy
import io
import itertools
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import onepass
import onepass_quality as q
from onepass_recording import CURRENT, Run, group


class OracleTests(unittest.TestCase):
    def test_ledger_oracle_against_asof_transaction_replay(self):
        decisions = set()
        for seed in range(30):
            case = q.cases(seed)[0]
            evidence, expected = case['evidence'], case['answer']
            versions = {}
            for key, text in evidence.items():
                if not key[1:].isdigit() or int(key[1:]) < 10: continue
                transaction, rev, effective, recorded = re.search(
                    r'거래 (\S+) 개정(\d+): 효력(\d+)일, 기록(\d+)일', text).groups()
                if max(int(effective), int(recorded)) > 10: continue
                if transaction not in versions or int(rev) > versions[transaction][0]:
                    versions[transaction] = (int(rev), key, text)
            opening, unit = map(int, re.findall(r'\d+', evidence['L2'])[::2])
            # L2 numbers are [opening, 1, unit]. Do not use the production oracle.
            self.assertEqual(sorted(v[1] for v in versions.values()), expected['derivation']['selected'])
            delta = 0
            for _, _, text in versions.values():
                match = re.search(r'(입고|출고|반품) (\d+)(상자|개)', text)
                if not match: continue  # Cancellation replaces its earlier receipt.
                kind, amount, units = match.groups()
                delta += int(amount) * (unit if units == '상자' else 1) * (-1 if kind == '출고' else 1)
            loss, reserve = map(int, re.findall(r'\d+', evidence['L3']))
            net = opening + delta
            usable = (net * (100 - loss)) // 100 - reserve
            demand = int(re.search(r'\d+', evidence['L4'])[0])
            extra = int(re.search(r'\d+', evidence['L5'])[0]) * (1 if '추가' in evidence['L5'] else -1)
            self.assertEqual(expected['derivation']['net'], net)
            self.assertEqual(expected['result']['available'], usable)
            self.assertEqual(expected['result']['decision'], '전량승인' if usable >= demand else '보류')
            self.assertEqual(expected['counterfactual']['available'], usable - extra)
            self.assertEqual(expected['counterfactual']['shortfall'], max(0, demand - usable + extra))
            self.assertNotEqual(expected['result']['decision'], expected['counterfactual']['decision'])
            decisions.add(expected['result']['decision'])
        self.assertEqual(decisions, {'전량승인', '보류'})

    def test_portfolio_proof_against_independent_bitmask_search(self):
        for seed in range(30):
            case = q.cases(seed)[1]
            evidence, expected = case['evidence'], case['answer']
            projects = {k: list(map(int, evidence['O' + k].split('= ')[1].split(', '))) for k in 'ABCDE'}
            _, budget, staff = map(int, re.findall(r'\d+', evidence['O1']))
            rows = []
            for mask in range(32):
                keys = ''.join(k for i, k in enumerate('ABCDE') if mask & (1 << i))
                if len(keys) != 3: continue
                costs, people, high, low, risks = zip(*(projects[k] for k in keys))
                violations = []
                if sum(costs) > budget: violations.append('budget')
                if sum(people) > staff: violations.append('staff')
                if 'B' in keys and 'A' not in keys: violations.append('dependency')
                if 'C' in keys and 'E' in keys: violations.append('conflict')
                rows.append(dict(ids=keys, cost=sum(costs), staff=sum(people),
                                 score=min(sum(high), sum(low)) - sum(risks) * 2, violations=violations))
            self.assertEqual(sorted(rows, key=lambda x: x['ids']), expected['derivation']['candidates'])
            ranked = sorted((r for r in rows if not r['violations']), key=lambda r: (-r['score'], r['ids']))
            self.assertEqual(expected['result'], dict(best=ranked[0]['ids'], score=ranked[0]['score'],
                runner_up=ranked[1]['ids'], margin=ranked[0]['score'] - ranked[1]['score']))
            changed = [r for r in ranked if r['cost'] <= budget - 3]
            self.assertEqual(expected['counterfactual']['best'], changed[0]['ids'])
            self.assertNotEqual(changed[0]['ids'], ranked[0]['ids'])
            self.assertEqual(expected['counterfactual']['feasible'], [r['ids'] for r in changed])

    def test_logic_worlds_and_minimal_unsat_core_against_named_variable_solver(self):
        certificates = set()
        for seed in range(30):
            case = q.cases(seed)[2]
            evidence, expected = case['evidence'], case['answer']
            terms = [re.findall('[A-F]', evidence['U' + str(i)]) for i in range(1, 7)]
            universe = [''.join(map(str, w)) for w in itertools.product((0, 1), repeat=6)]
            def holds(s, i):
                w = dict(zip('ABCDEF', map(int, s)))
                values = [w[k] for k in terms[i]]
                return [lambda v: any(v), lambda v: v[0] == v[1], lambda v: not all(v),
                        lambda v: sum(v) >= 2, lambda v: sum(v) == 1, lambda v: v[0] == 0][i](values)
            worlds = [s for s in universe if all(holds(s, i) for i in range(5))]
            self.assertEqual(sorted(expected['derivation']['worlds']), worlds)
            self.assertEqual(len(worlds), 4)
            cores = []
            for mask in range(1, 64):
                indices = {i for i in range(6) if mask & (1 << i)}
                if any(all(holds(w, i) for i in indices) for w in universe): continue
                if all(any(all(holds(w, j) for j in indices - {i}) for w in universe) for i in indices):
                    cores.append(['U' + str(i + 1) for i in sorted(indices)])
            self.assertEqual(expected['counterfactual']['minimal_cores'], cores)
            self.assertEqual(expected['counterfactual']['consistent'], False)
            statuses = []
            for a, b, c in re.findall(r'Q\d=([A-F])=1|Q\d=\(([A-F])=0 그리고 ([A-F])=0\)', case['question']):
                values = []
                for s in worlds:
                    w = dict(zip('ABCDEF', map(int, s)))
                    values.append(w[a] == 1 if a else w[b] == w[c] == 0)
                statuses.append('참' if all(values) else '판단불가' if any(values) else '거짓')
            self.assertEqual(expected['result']['statuses'], statuses)
            certificates.add(q.digest(expected))
        self.assertGreater(len(certificates), 20)


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.cases = q.cases(2007)
        self.item = dict(quality_cases=self.cases, ctx=2000, question='all')
        self.answer = {c['id']: copy.deepcopy(c['answer']) for c in self.cases}

    def grade(self, answer=None, finish='stop', **kw):
        answer = self.answer if answer is None else answer
        text = json.dumps(answer, ensure_ascii=False) if not isinstance(answer, str) else answer
        return q.assess(self.item, [{'content': text}], finish, **kw)

    def test_correct_certificates_order_fences_and_alternative_witnesses(self):
        self.answer['portfolio']['derivation']['candidates'].reverse()
        self.answer['ledger']['derivation']['selected'].reverse()
        self.answer['logic']['derivation']['worlds'].reverse()
        for v, options in zip(self.answer['logic']['witnesses'], self.cases[2]['witness_options']):
            for kind in v:
                if options[kind]: v[kind] = options[kind][-1]
        rows = self.grade('```json\n' + json.dumps(self.answer) + '\n```')
        self.assertTrue(all(r['passed'] for r in rows))
        self.assertEqual(q.summarize(rows)['ok'], 3)

    def test_correct_answer_with_false_derivation_or_false_citation_fails(self):
        self.answer['ledger']['derivation']['selected'][0] = 'L12'  # recorded after cutoff
        self.answer['portfolio']['derivation']['candidates'][0]['score'] += 1
        self.answer['logic']['evidence']['constraints'].append('U99')
        rows = self.grade()
        self.assertTrue(all(r['checks']['result'] for r in rows))
        self.assertFalse(any(r['passed'] for r in rows))
        self.assertEqual(rows[0]['failures'][0]['dimension'], 'derivation')

    def test_missing_candidate_world_counterexample_and_nonminimal_core_fail(self):
        self.answer['portfolio']['derivation']['candidates'].pop()
        self.answer['logic']['derivation']['worlds'].pop()
        self.answer['logic']['counterfactual']['minimal_cores'][0].append('U5')
        self.answer['logic']['witnesses'][0] = dict(true='000000', false='111111')
        rows = self.grade()
        self.assertFalse(rows[1]['checks']['derivation'])
        self.assertFalse(rows[2]['checks']['derivation'])
        self.assertFalse(rows[2]['checks']['counterfactual'])
        self.assertFalse(rows[2]['checks']['witnesses'])

    def test_answer_only_in_reasoning_never_passes(self):
        for content in ('', '8127 할보르센 1997년 3월 14일 k-42 북쪽', '모든 답이 정확하다'):
            events = [{'reasoning_content': json.dumps(self.answer)}, {'content': content}]
            self.assertFalse(any(r['passed'] for r in q.assess(self.item, events, 'stop')))
        events = [{'reasoning_content': '틀린 초안', 'content': json.dumps(self.answer)}]
        self.assertTrue(all(r['passed'] for r in q.assess(self.item, events, 'stop')))

    def test_malformed_duplicate_extra_nonfinite_and_truncated_answers_fail_closed(self):
        valid = json.dumps(self.answer)
        for text in ('', valid[:-3], '[1]', valid + valid, valid.replace('287', 'NaN'),
                     '{"ledger":{},"ledger":{},"portfolio":{},"logic":{}}', valid[:-1] + ',"extra":{}}'):
            with self.subTest(text=text[:60]):
                self.assertFalse(any(r['passed'] for r in self.grade(text)))
        self.assertFalse(any(r['passed'] for r in self.grade(finish='length')))
        self.assertTrue(all(r['passed'] for r in self.grade(finish='length', fixed=True)))
        self.assertFalse(any(r['passed'] for r in self.grade(valid[:-1], finish='length', fixed=True)))
        self.answer['logic']['counterfactual']['consistent'] = 0
        self.assertFalse(self.grade()[2]['passed'])

    def test_query_order_matters_and_duplicate_worlds_do_not_count_as_coverage(self):
        self.answer['logic']['result']['statuses'].reverse()
        worlds = self.answer['logic']['derivation']['worlds']
        worlds[-1] = worlds[0]
        row = self.grade()[2]
        self.assertFalse(row['checks']['result'])
        self.assertFalse(row['checks']['derivation'])


class IntegrationTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(ctx='2000,32000,128000', seed=7, combine_min_ctx=32000,
            max_tokens=q.MAX_TOKENS, combined_max_tokens=q.COMBINED_MAX_TOKENS,
            combined_reasoning_budget=q.COMBINED_REASONING_BUDGET)

    def test_seeded_workload_identity_distributed_evidence_and_no_answer_key_in_prompt(self):
        with patch('bench_common.resolve_model', return_value='fixture'):
            cq = onepass._load('check-quality.py', 'quality_test_filler')
        items = onepass.workload_requests(self.fixture(), cq)
        self.assertEqual(q.digest(items), q.digest(onepass.workload_requests(self.fixture(), cq)))
        self.assertEqual([i['question'] for i in items], [0, 1, 2, 'all', 'all'])
        for item in items:
            self.assertEqual(item['quality_cases'], [c for c in q.cases(7 + item['ctx'])
                if c['id'] in {v['id'] for v in item['quality_cases']}])
            for case in item['quality_cases']:
                self.assertNotIn(json.dumps(case['answer'], ensure_ascii=False), item['content'])
                positions = [item['content'].index('[' + key + ']') for key in case['evidence']]
                if item['ctx'] > 2000:
                    self.assertGreater(max(positions) - min(positions), item['ctx'] * .4)
            self.assertLess(item['reasoning_budget'], item['max_tokens'])
        different = self.fixture(); different.seed += 1
        self.assertNotEqual(q.digest(items), q.digest(onepass.workload_requests(different, cq)))

    def test_real_stream_c4_records_visible_grades_and_immutable_workload(self):
        item = q.request_item(2000, 2007, q.cases(2007), lambda n, r: '', 7200, 2400, 'all')
        answer = json.dumps({c['id']: c['answer'] for c in item['quality_cases']})
        frames = [dict(choices=[dict(delta=dict(reasoning_content='검토 중'))]),
                  dict(choices=[dict(delta=dict(content=answer), finish_reason='stop')],
                       usage=dict(completion_tokens=100, prompt_tokens=2000))]
        raw = ''.join('data: ' + json.dumps(f) + '\n' for f in frames).encode()
        with TemporaryDirectory() as root, patch('urllib.request.urlopen', side_effect=OSError('offline')):
            run = Run({}, Path(root) / 'ledger.jsonl', 'http://fixture/v1/chat/completions')
            run.workloads([item])
            run.begin('measure-c4-test', 4)
            with patch('urllib.request.urlopen', side_effect=lambda *a, **kw: io.BytesIO(raw)):
                result = group(run, onepass.ask_stream, 'http://fixture/v1/chat/completions', 'm', item, 4, grade=True)
            run.end()
            run.finish(RuntimeError('fixture preserves partial results'))
            CURRENT.set(None)
            self.assertEqual(len(result['requests']), 4)
            self.assertTrue(all(g['passed'] for r in result['requests'] for g in r['quality']))
            self.assertEqual(len({r['request_sha256'] for r in result['requests']}), 4)
            self.assertEqual(len({r['workload_sha256'] for r in result['requests']}), 1)
            saved = [json.loads(s) for s in (run.path / 'quality.jsonl').read_text().splitlines()]
            self.assertEqual(len(saved), 4)
            self.assertEqual({r['request_sha256'] for r in saved}, {r['request_sha256'] for r in result['requests']})
            self.assertEqual(json.loads((run.path / 'workloads.json').read_text()), [item])
            self.assertEqual(run.record['recording']['status'], 'incomplete')

    def test_c1_grading_is_after_windows_and_latency_session(self):
        tree = ast.parse(Path(onepass.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_main')
        grade = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and ast.unparse(n.func) == 'run.grade')
        windows = next(n for n in main.body if isinstance(n, ast.With))
        end = next(n for n in main.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'c1_report')
        self.assertGreater(grade.lineno, windows.end_lineno)
        self.assertGreater(grade.lineno, end.end_lineno)

    def test_canonical_main_records_all_nine_c1_and_36_c4_cases(self):
        self.addCleanup(setattr, onepass, '_RUN', None)
        from tests.test_onepass_channel_diagnostics import scanner
        cq = SimpleNamespace(MODEL='fixture', filler=lambda n, r: '')
        items = onepass.workload_requests(self.fixture(), cq)
        answers = {item['content']: json.dumps({c['id']: c['answer'] for c in item['quality_cases']})
                   for item in items}
        completed = 0
        def serve(req, **kw):
            nonlocal completed
            if isinstance(req, str): raise OSError('fixture has no GPU recording')
            payload = json.loads(req.data)
            content = answers[payload['messages'][0]['content']]
            completed += 1
            frames = [dict(choices=[dict(delta=dict(content=content), finish_reason='stop')]),
                      dict(usage=dict(prompt_tokens=2000, completion_tokens=900,
                                      prompt_tokens_details=dict(cached_tokens=0)))]
            return io.BytesIO(''.join('data: ' + json.dumps(f) + '\n' for f in frames).encode())
        def metrics(url):
            return (f'vllm:request_success_total{{}} {completed}\n'
                    'vllm:num_requests_running{} 0\nvllm:num_requests_waiting{} 0\n')
        class Windows:
            def __init__(self, *a, **kw): self.samples, self.traffic_samples = [], []
            def __enter__(self): return self
            def __exit__(self, *a): pass
        modules = {'check-quality.py': cq, 'korean-corruption.py': scanner(),
                   'bench-dec.py': SimpleNamespace(URL='http://fixture/v1/chat/completions', METRICS='metrics',
                                                   _parse_spec_metrics=lambda x: {}),
                   'bracket.py': SimpleNamespace(_git_sha=lambda: 'fixture', _StepWindows=Windows,
                       _spec_delta=lambda a, b: (0, 0), spec_k_eff=lambda a, b: 6)}
        with TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, 'argv', ['onepass.py', '--out', str(Path(root) / 'ledger.jsonl')]), \
             patch.object(onepass, '_load', side_effect=lambda name, module: modules[name]), \
             patch.object(onepass, '_served_build', return_value={}), \
             patch.object(onepass, 'engine_shape', return_value={}), \
             patch.object(onepass, '_metrics_text', side_effect=metrics), \
             patch('urllib.request.urlopen', side_effect=serve), patch('sys.stdout', new_callable=io.StringIO):
            # Lack of real GPU/compile evidence must still invalidate acceptance.
            self.assertEqual(onepass.main(), 2)
            record = json.loads((Path(root) / 'ledger.jsonl').read_text())
            self.assertEqual((record['quality']['ok'], record['quality']['total']), (9, 9))
            self.assertEqual((record['quality_c4']['ok'], record['quality_c4']['total']), (36, 36))
            grades = [json.loads(s) for s in (Path(record['artifacts']) / 'quality.jsonl').read_text().splitlines()]
            self.assertEqual(len(grades), 25)
            self.assertEqual({r['phase'] for r in grades}, {'measure-c1'} | {
                f"measure-c4-{item['ctx']}-q{item['question']}" for item in items})
            self.assertEqual(record['recording']['status'], 'complete')
            self.assertFalse(record['steady_state']['valid'])
        onepass._RUN = None

    def test_changed_quality_protocol_cannot_reuse_a_baseline(self):
        import judge
        self.assertFalse(judge.compatible({'quality_protocol': {'version': 'old'}},
                                         {'quality_protocol': {'version': q.VERSION}}))


if __name__ == '__main__':
    unittest.main()
