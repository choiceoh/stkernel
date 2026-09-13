import unittest
from bench.draft_rejections import summarize


class DraftRejectionRecordingTests(unittest.TestCase):
    def test_reused_sequence_ids_belong_to_the_new_request_and_ranks_stay_separate(self):
        rows = [dict(kind='request', operation='admit', row=0, request_id=8),
                dict(kind='draft_rejection', seq=0, accepted_prefix=0, reason='candidate_miss'),
                dict(kind='draft_rejection', seq=0, accepted_prefix=1, reason='selector_miss'),
                dict(kind='request', operation='admit', row=0, request_id=10),
                dict(kind='draft_rejection', seq=0, accepted_prefix=6, reason='all_accepted'),
                dict(kind='draft_rejection', seq=0, accepted_prefix=2, reason='output_boundary'),
                dict(kind='draft_rejection', seq=0, accepted_prefix=-1, reason='policy_modified')]
        result = summarize(dict(ranks=[dict(rank=rank, rows=rows) for rank in (0, 1)]))
        self.assertTrue(result['recorded'])
        self.assertEqual(len(result['ranks']), 2)
        first, second = result['ranks'][0]['requests']
        self.assertEqual((first['request_id'], second['request_id']), (8, 10))
        self.assertEqual(first['first_rejection_positions'], [
            dict(position=1, reason='candidate_miss', count=1),
            dict(position=2, reason='selector_miss', count=1)])
        self.assertEqual(second['first_rejection_positions'], [])
        self.assertEqual(second['reasons']['all_accepted'], 1)

    def test_absent_diagnostics_are_not_reported_as_zero_misses(self):
        result = summarize(dict(ranks=[dict(rank=0, rows=[])]))
        self.assertFalse(result['recorded'])
        self.assertEqual(result['ranks'][0]['requests'], [])


if __name__ == '__main__':
    unittest.main()
