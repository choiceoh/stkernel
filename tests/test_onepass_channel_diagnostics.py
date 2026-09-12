"""Canonical SSE diagnostics preserve requests, clocks, precedence and gates."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import types
import unicodedata
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import onepass

# Exact pre-diagnostic ask_stream implementation, preserved as a CPU oracle.
# Original function SHA256: bb00de333fc2270ffe769a7950c3a45b435af8c7b0c6b3d484a6e3c58d7b14ab
LEGACY_STREAM_SOURCE = r'''def ask_stream(url, model, content, max_tokens, timing=None, min_tokens=0, seed=None):
    """(text, ttft_s, prompt_tokens, completion_tokens, finish_reason) of one
    streamed chat completion: ttft = first chunk carrying content."""
    body = json.dumps({"model": model, "max_tokens": max_tokens, "min_tokens": min_tokens,
                       "seed": seed, "temperature": 0.0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": content}],
                       # 39차: thinking ON, explicitly. The stock template ignored this
                       # kwarg and always reasoned, so every reference (BASE39-*, DEF40, ...)
                       # was measured with reasoning in the stream; the v2 template honours
                       # the kwarg and thinking=false gives answers too short for the 2 s
                       # decode windows (TPL1: no windows). Keep the condition constant.
                       "chat_template_kwargs": {"thinking": True}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    arrivals = []
    parts = []
    usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    arrived = time.monotonic()
                    arrivals.append(arrived)
                    if ttft is None:
                        ttft = arrived - t0
                    parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    if ttft is None:
        ttft = time.monotonic() - t0
    if timing is not None:
        elapsed = time.monotonic() - t0
        ctok = int(usage.get("completion_tokens", 0) or 0)
        decode_s = elapsed - ttft
        # Standard request TPOT includes the final stream/usage tail. SSE
        # chunks can contain several speculative tokens; gaps are NOT ITL.
        timing.update(completion_tokens=ctok, prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                      min_tokens=min_tokens, max_tokens=max_tokens, seed=seed,
                      request_sha256=hashlib.sha256(body).hexdigest(),
                      output_sha256=hashlib.sha256("".join(parts).encode()).hexdigest(),
                      ttft_s=ttft, elapsed_s=elapsed,
                      decode_s=decode_s, finish_reason=finish,
                      tpot_ms=1000 * decode_s / (ctok - 1) if ctok > 1 else None,
                      decode_tok_s=(ctok - 1) / decode_s if ctok > 1 and decode_s > 0 else None,
                      chunk_gaps_ms=[1000 * (b - a) for a, b in zip(arrivals, arrivals[1:])])
    return ("".join(parts), ttft, int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0), finish)
'''


def legacy_stream():
    namespace = dict(onepass.__dict__)
    exec(compile(LEGACY_STREAM_SOURCE, '<original-onepass-ask-stream>', 'exec'), namespace)
    return namespace['ask_stream']


def scanner():
    # Execute only the existing scanner definitions; never resolve a model or URL.
    tree = ast.parse((ROOT / 'bench/korean-corruption.py').read_text())
    names = {'SYL', 'JAMO', 'WELDED_JAMO', 'HAN', 'HANJA_GLOSS', 'INFORMATIONAL'}
    selected = [node for node in tree.body if
                isinstance(node, ast.FunctionDef) and node.name == 'scan' or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                     for t in node.targets)]
    namespace = dict(re=re, unicodedata=unicodedata)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<existing-korean-scanner>', 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def wire(deltas, finish='length', extra=()):
    records = [b': keepalive\n', b'data: not-json\n']
    for delta in deltas:
        records.append(('data: ' + json.dumps({'choices': [{'delta': delta}]}) + '\n').encode())
    records.extend(extra)
    records.append(('data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}],
                                         'usage': {'prompt_tokens': 2121, 'completion_tokens': 1024}}) + '\n').encode())
    records.append(b'data: [DONE]\n')
    return b''.join(records)


class ChannelDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.korean = scanner()

    def call(self, function, frames, *, trace=False, timed=True):
        timing = {'ctx': 2000, 'rep': 0} if timed else None
        traces = []
        with patch.object(onepass.urllib.request, 'urlopen', return_value=io.BytesIO(frames)) as opener, \
                patch.object(onepass.time, 'monotonic', side_effect=range(100, 200)) as clock:
            kwargs = dict(min_tokens=1024, seed=7)
            if trace:
                kwargs['channel_trace'] = traces
            result = function('http://fixture.invalid/v1/chat/completions', 'glm-fixture',
                              '문서와 질문 원문', 1024, timing, **kwargs)
        request = opener.call_args.args[0]
        return result, timing, traces, (request.full_url, request.data, request.headers,
                                        opener.call_args.kwargs, opener.call_count, clock.call_count)

    def diagnose(self, deltas, finish='length'):
        result, timing, traces, _ = self.call(onepass.ask_stream, wire(deltas, finish), trace=True)
        self.assertEqual(len(traces), 1)
        hits = self.korean.scan(result[0], truncated=finish == 'length')
        old_hits = dict(hits)
        diagnostic = onepass._channel_diagnostics(traces[0], result[0], finish, hits, self.korean)
        self.assertEqual(hits, old_hits)
        expected = {kind: hits[kind] for kind in ('replacement', 'lone_jamo', 'cjk_mixed', 'control')}
        self.assertEqual(diagnostic['combined_gated_counts'], expected)
        self.assertEqual({kind: sum(row['gated_counts'][kind] for row in diagnostic['channels'].values())
                          for kind in expected}, expected)
        self.assertNotIn('channel_diagnostics', timing)  # Deferred until the existing scan phase.
        self.assertEqual(timing['output_sha256'], hashlib.sha256(result[0].encode()).hexdigest())
        return result, diagnostic, hits

    def test_actual_stream_matches_legacy_request_text_hashes_clocks_and_timing(self):
        cases = [[], [{'content': None}], [{'reasoning_content': '근거'}, {'content': '답'}],
                 [{'content': '우선', 'reasoning_content': '博士', 'reasoning': '숨김'}],
                 [{'content': '', 'reasoning_content': '두번째', 'reasoning': '세번째'}],
                 [{'reasoning': '마지막'}, {'content': '본문'}]]
        extra = [('data: ' + json.dumps({'choices': [{'delta': {'reasoning': 'x'}},
                                                     {'delta': {'content': 'y'}}]}) + '\n').encode()]
        for deltas in cases:
            for timed in (True, False):
                with self.subTest(deltas=deltas, timed=timed):
                    frames = wire(deltas, extra=extra)
                    old = self.call(legacy_stream(), frames, timed=timed)
                    new = self.call(onepass.ask_stream, frames, trace=True, timed=timed)
                    self.assertEqual(new[0], old[0])
                    self.assertEqual({k: new[1][k] for k in old[1]} if timed else new[1], old[1])
                    self.assertEqual(new[3], old[3])
                    self.assertEqual(new[3][-2], 1)  # Exactly one unchanged HTTP request.
        # No content still uses the original fallback TTFT and clock count.
        new = self.call(onepass.ask_stream, wire([]), trace=True)
        old = self.call(legacy_stream(), wire([]))
        self.assertEqual(new[0], old[0])
        self.assertEqual({k: new[1][k] for k in old[1]}, old[1])

    def test_reasoning_content_offense_is_not_misattributed_to_clean_content(self):
        result, diag, hits = self.diagnose([
            {'reasoning_content': 'Francis Crick, Halvorsen博士 signing tundra survey'},
            {'content': '할보르센 박사가 서명했습니다.'}])
        self.assertEqual(hits['cjk_mixed'], 2)
        self.assertEqual(diag['first_offending_channel'], 'reasoning_content')
        self.assertEqual(diag['offenses'][0]['combined_offset'], result[0].index('博'))
        self.assertIn('Halvorsen博士', diag['offenses'][0]['snippet'])

    def test_same_offense_in_content_keeps_the_existing_failure(self):
        _, diag, hits = self.diagnose([{'reasoning': '정상 근거'}, {'content': 'Halvorsen博士'}])
        self.assertEqual(hits['cjk_mixed'], 2)
        self.assertEqual(diag['first_offending_channel'], 'content')
        self.assertEqual(diag['offenses'][0]['selected_channel_offset'], len('Halvorsen'))

    def test_reasoning_first_does_not_hide_later_content_hits_across_pieces(self):
        _, diag, hits = self.diagnose([{'reasoning_content': 'Halvorsen博'},
                                      {'reasoning_content': '士'}, {'content': '답은博'},
                                      {'content': '士입니다.'}])
        self.assertEqual(hits['cjk_mixed'], 4)
        self.assertEqual(diag['first_offending_channel'], 'reasoning_content')
        self.assertEqual(diag['channels']['reasoning_content']['gated_counts']['cjk_mixed'], 2)
        self.assertEqual(diag['channels']['content']['gated_counts']['cjk_mixed'], 2)
        self.assertEqual(len(diag['offenses']), 1)  # Counts are complete; snippets remain bounded.

    def test_simultaneous_ignored_channel_is_counted_but_not_a_gate_cause(self):
        _, diag, hits = self.diagnose([{'content': '정상', 'reasoning_content': '博士', 'reasoning': '숨김'}])
        self.assertFalse(any(v for k, v in hits.items() if k not in self.korean.INFORMATIONAL))
        self.assertIsNone(diag['first_offending_channel'])
        self.assertEqual(diag['offenses'], [])
        self.assertEqual(diag['channels']['reasoning_content']['raw_chars'], 2)
        self.assertEqual(diag['channels']['reasoning_content']['selected_chars'], 0)
        self.assertEqual(diag['channels']['content']['selected_chars'], 2)

    def test_precedence_counts_empty_missing_and_nontext_unselected_fields(self):
        _, diag, _ = self.diagnose([{'content': '', 'reasoning_content': '근거'},
                                  {'reasoning': '생각'}, {'content': '답', 'reasoning': {'ignored': True}}, {}])
        self.assertEqual(diag['combined_chars'], 5)
        self.assertEqual(diag['precedence'], ['content', 'reasoning_content', 'reasoning'])
        self.assertEqual(diag['channels']['reasoning_content']['selected_pieces'], 1)
        self.assertEqual(diag['channels']['reasoning']['non_text_fields'], 1)
        self.assertEqual(sum(row['selected_chars'] for row in diag['channels'].values()), 5)

    def test_jamo_at_channel_boundary_belongs_to_the_jamo_channel(self):
        _, diag, hits = self.diagnose([{'reasoning': '하'}, {'content': 'ㄹ수'}])
        self.assertEqual(hits['lone_jamo'], 1)
        self.assertEqual(diag['first_offending_channel'], 'content')
        self.assertEqual(diag['offenses'][0]['combined_offset'], 1)
        self.assertEqual(diag['offenses'][0]['selected_channel_offset'], 0)
        self.assertEqual(diag['channels']['content']['gated_counts']['lone_jamo'], 1)
        self.assertEqual(diag['channels']['reasoning']['gated_counts']['lone_jamo'], 0)

    def test_hanja_gloss_across_channels_is_excluded_before_locating_real_hit(self):
        result, diag, hits = self.diagnose([{'reasoning': '조력 (潮'}, {'content': '力) Halvorsen博士'}])
        self.assertEqual((hits['hanja_gloss'], hits['cjk_mixed']), (2, 2))
        self.assertEqual(diag['first_offending_channel'], 'content')
        self.assertEqual(diag['offenses'][0]['combined_offset'], result[0].index('博'))
        self.assertEqual(diag['channels']['content']['gated_counts']['cjk_mixed'], 2)
        self.assertEqual(diag['channels']['reasoning']['gated_counts']['cjk_mixed'], 0)
        _, clean, _ = self.diagnose([{'reasoning': '조력 (潮'}, {'content': '力)'}])
        self.assertEqual(clean['offenses'], [])

    def test_truncated_terminal_replacement_keeps_original_exception(self):
        for finish, expected in (('length', 0), ('stop', 1)):
            _, diag, hits = self.diagnose([{'content': '정상�'}], finish)
            self.assertEqual(hits['replacement'], expected)
            self.assertEqual(len(diag['offenses']), expected)
        _, diag, hits = self.diagnose([{'reasoning': '�중간'}, {'content': '�'}], 'length')
        self.assertEqual(hits['replacement'], 1)
        self.assertEqual(diag['first_offending_channel'], 'reasoning')

    def test_all_four_kinds_are_bounded_and_first_is_stream_order(self):
        _, diag, hits = self.diagnose([{'reasoning': '먼저\x00'}, {'reasoning_content': '�하ㄹ'},
                                      {'content': '博士'}])
        self.assertEqual({row['kind'] for row in diag['offenses']},
                         {'replacement', 'lone_jamo', 'cjk_mixed', 'control'})
        self.assertEqual(diag['first_offending_channel'], 'reasoning')
        self.assertEqual(diag['offenses'][0]['kind'], 'control')
        self.assertEqual(diag['unlocated_kinds'], [])
        self.assertEqual(hits['control'], 1)
        self.assertEqual(diag['channels']['reasoning']['gated_counts']['control'], 1)
        self.assertEqual(diag['channels']['reasoning_content']['gated_counts']['replacement'], 1)
        self.assertEqual(diag['channels']['reasoning_content']['gated_counts']['lone_jamo'], 1)

    def test_long_output_serializes_counts_and_four_small_snippets_only(self):
        text = '시작' + ('정상' * 20000) + '\x00�하ㄹ博士' + ('끝' * 20000)
        result, diag, _ = self.diagnose([{'content': text}])
        self.assertEqual(result[0], text)
        self.assertEqual(diag['channels']['content']['raw_chars'], len(text))
        self.assertEqual(len(diag['offenses']), 4)
        self.assertTrue(all(len(row['snippet']) <= 96 for row in diag['offenses']))
        encoded = json.dumps(diag, ensure_ascii=False)
        self.assertLess(len(encoded), 2500)
        self.assertNotIn(text, encoded)
        self.assertNotIn('events', diag)

    def test_trace_mismatch_is_not_silently_misattributed(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            onepass._channel_diagnostics([{'content': 'other'}], 'original', 'stop', {}, self.korean)

    def test_hit_count_mismatch_is_rejected_without_rewriting_the_gate(self):
        hits = self.korean.scan('博士')
        hits['cjk_mixed'] = 1
        with self.assertRaisesRegex(ValueError, 'combined Korean scan'):
            onepass._channel_diagnostics([{'content': '博士'}], '博士', 'stop', hits, self.korean)
        self.assertEqual(hits['cjk_mixed'], 1)

    def test_network_failure_preserves_exception_and_emits_no_complete_trace(self):
        traces = []
        with patch.object(onepass.urllib.request, 'urlopen', side_effect=OSError('offline')) as opener:
            with self.assertRaisesRegex(OSError, 'offline'):
                onepass.ask_stream('http://fixture.invalid', 'glm', 'same', 20, {}, channel_trace=traces)
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(traces, [])

    def test_every_canonical_request_is_traced_and_analysis_is_outside_sampling(self):
        tree = ast.parse((ROOT / 'bench/onepass.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_main')
        asks = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'ask_stream']
        self.assertEqual(len(asks), 3)  # combined, ordinary, fixed requests.
        self.assertTrue(all(any(k.arg == 'channel_trace' and isinstance(k.value, ast.Name)
                                and k.value.id == 'channel_traces' for k in call.keywords) for call in asks))
        with_step = next(n for n in main.body if isinstance(n, ast.With) and '_StepWindows' in ast.unparse(n.items[0].context_expr))
        diagnostic_call = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                               and n.func.id == '_channel_diagnostics')
        self.assertGreater(diagnostic_call.lineno, with_step.end_lineno)
        metrics_after = next(n for n in main.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'metrics_after' for t in n.targets))
        self.assertGreater(diagnostic_call.lineno, metrics_after.end_lineno)
        self.assertEqual(ast.unparse(diagnostic_call.args[3]), 'h')  # Same classifier result.


if __name__ == '__main__':
    unittest.main()
