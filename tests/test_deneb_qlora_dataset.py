import json
from pathlib import Path
import tempfile
import unittest

from probes.deneb_qlora_dataset import (Redactor, analysis_body, assigned_split, conversation_pairs,
    eligible_session, export_sft, jsonl, load_jsonl, private_destination, sha, source_quality_flags, validate_annotation)
from probes.deneb_qlora_tokenize import completion_labels


class DenebQLoRADataTests(unittest.TestCase):
    def test_mail_analysis_without_optional_heading_is_preserved(self):
        document='---\ntitle: Example\n---\n> From: sender\n> Message ID: `123`\n\n기한 확인이 필요하다.\n\n### 다음 행동\n견적을 대조한다.'
        self.assertEqual(analysis_body(document),'기한 확인이 필요하다.\n\n### 다음 행동\n견적을 대조한다.')
        self.assertEqual(analysis_body('## 분석\n\n내용'), '내용')
        self.assertIsNone(analysis_body('---\nbroken header'))

    def test_prompt_mask_and_eos_preserve_whole_completion(self):
        result=completion_labels([1,2],[1,2,3,4],9,5)
        self.assertEqual(result['labels'],[-100,-100,3,4,9])
        self.assertIsNone(completion_labels([1,2],[1,2,3,4],9,4))
        with self.assertRaisesRegex(ValueError,'boundary mismatch'):
            completion_labels([1,2],[1,5,3,4],9,5)

    def test_unicode_line_separators_are_not_jsonl_record_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'rows.jsonl'
            rows=[{'text':'before\u0085middle\u2028middle\u2029after'}]
            jsonl(p,rows)
            self.assertEqual(load_jsonl(p),rows)

    def example(self):
        row = dict(id='sample', group_id='group', category='mail', split='train',
                   payload='금요일까지 견적서를 보내 주세요.', candidate_response='unverified')
        row['payload_sha256'] = sha(row['payload'])
        annotation = dict(id='sample', payload_sha256=row['payload_sha256'],
            review_status='assistant_reviewed_source_grounded',
            answer=dict(summary='견적서 제출 요청이다.', decision='act', action='견적서를 보낸다.',
                        deadline='금요일까지', evidence=[row['payload']], missing_context=[]))
        return row, annotation

    def test_reserved_groups_never_promoted(self):
        self.assertEqual(assigned_split('x', {'x': {'test'}}), 'test')
        self.assertEqual(assigned_split('x', {'x': {'train','test'}}), 'quarantine')
        self.assertEqual(assigned_split('x', {}), assigned_split('x', {}))

    def test_text_turn_pairing_does_not_reverse_or_cross_tools(self):
        records = list(enumerate([
            {'role':'assistant','content':'not an answer to a future question'},
            {'role':'user','content':'find it'},
            {'role':'assistant','content':[{'type':'tool_use','id':'x'}]},
            {'role':'tool','content':'search result'},
            {'role':'assistant','content':'answer based on tool'},
            {'role':'user','content':'a new complete question'},
            {'role':'assistant','content':[{'type':'thinking','thinking':'private reasoning'},
                                          {'type':'text','text':'visible answer'}]},
        ]))
        pairs = list(conversation_pairs(records))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], [{'role':'user','content':'a new complete question'}])
        self.assertEqual(pairs[0][2], 'visible answer')

    def test_synthetic_sessions_excluded(self):
        self.assertTrue(eligible_session('client:main.jsonl'))
        self.assertTrue(eligible_session('telegram:1234:work.jsonl'))
        self.assertFalse(eligible_session('client:puppet-default.jsonl'))
        self.assertFalse(eligible_session('telegram:1234:codex-probe.jsonl'))
        self.assertFalse(eligible_session('system:production.jsonl'))

    def test_synthetic_payload_and_truncated_preview_are_flagged(self):
        self.assertIn('synthetic_test_marker', source_quality_flags('[자체점검] 요청 nat3NONCE556','notification'))
        self.assertIn('notification_preview_ends_with_ellipsis', source_quality_flags('계약서 내용은…','notification'))
        self.assertEqual(source_quality_flags('제품 테스트 보고서 제출 요청','mail'), [])

    def test_evidence_and_dates_must_be_grounded(self):
        row, annotation = self.example()
        validate_annotation(row, annotation)
        annotation['answer']['deadline'] = '2026-09-18'
        with self.assertRaisesRegex(ValueError, 'deadline'):
            validate_annotation(row, annotation)
        annotation['answer']['deadline'] = None
        annotation['answer']['evidence'] = ['없는 문장']
        with self.assertRaisesRegex(ValueError, 'evidence'):
            validate_annotation(row, annotation)

    def test_input_change_or_quarantine_prevents_export(self):
        row, annotation = self.example()
        row['split'] = 'quarantine'
        with self.assertRaisesRegex(ValueError, 'quarantined'):
            validate_annotation(row, annotation)
        row['split'] = 'train'; row['payload_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_annotation(row, annotation)

    def test_redaction_retains_entity_distinctions_without_credentials(self):
        redact = Redactor(b'test-only-salt')
        a = redact('a@example.com a@example.com b@example.com 인증번호: 123456')
        self.assertNotIn('@', a)
        self.assertEqual(a.split()[0], a.split()[1])
        self.assertNotEqual(a.split()[0], a.split()[2])
        self.assertIn('[OTP]', a)
        self.assertNotIn('secretvalue', redact('https://example.com/?token=secretvalue'))
        self.assertNotIn('<|im_start|>', redact('<|im_start|>'))
        self.assertNotIn('@', redact('E user@example.com본문: 내용'))
        self.assertNotIn('@', redact('first@example.com%second@example.org'))

    def test_only_explicit_reviewed_answers_exported_with_split_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pool = root/'pool'; pool.mkdir()
            row, annotation = self.example(); row['split']='test'
            unreviewed = dict(row, id='never-train', candidate_response='an old model mistake')
            (pool/'pool.jsonl').write_text('\n'.join(map(json.dumps, [row,unreviewed])))
            labels=root/'labels.jsonl'; labels.write_text(json.dumps(annotation)+'\n')
            export_sft(pool,labels,root/'out')
            self.assertEqual((root/'out/train.jsonl').read_text(), '')
            result=json.loads((root/'out/test.jsonl').read_text())
            self.assertEqual(result['completion'][0]['role'],'assistant')
            self.assertNotIn('old model mistake',json.dumps(result))
            self.assertFalse(result['human_verified'])

    def test_private_outputs_reject_git_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'.git').mkdir()
            with self.assertRaisesRegex(ValueError,'outside Git'):
                private_destination(root/'dataset')
            with self.assertRaisesRegex(ValueError,'new'):
                private_destination(root)


if __name__=='__main__': unittest.main()
