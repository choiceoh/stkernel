"""CPU evidence checks for actual-partial replay; no CUDA execution."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probes'))
import glm53_moe_m64_fp8_trace as m


class TraceTests(unittest.TestCase):
    def records(self):
        records = []
        raw = b'fixture payload; array format checked in actual CPU tensor tests'
        for rows, skew, seed, trial in m.plan():
            r = dict(rows=rows, skew=skew, seed=seed, trial=trial, order=list(m.order(trial)), row_ids=[1])
            for phase in m.PHASES:
                r[phase] = [dict(rank=rank, rows=rows if phase == 'partial' else rows//4,
                    candidate_bad=2, control_bad=1, finite=True) for rank in range(4)]
            r['replay'] = [dict(rank=rank, arms={arm: dict(original_equal=True, second_equal=True,
                sum32_store_equal=True, partial_unchanged=True, gather_unchanged=True, source_unchanged=True)
                for arm in m.ARMS}) for rank in range(4)]
            r['payloads'] = [dict(kind='MOE_M64_FP8_TRACE_ROWS', rows=rows, skew=skew, seed=seed,
                trial=trial, rank=rank, row_ids=[1], encoding='npz-base64-no-pickle', bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest()) for rank in range(4)]
            records.append(r)
        return records, raw

    def test_complete_numerical_or_replay_failure_never_approves_serving(self):
        records, _ = self.records()
        records[0]['replay'][2]['arms']['candidate']['original_equal'] = False
        result = m.completion(records, {})
        self.assertEqual(result['trials'], 72)
        self.assertFalse(result['replay_all_equal'])
        self.assertFalse(result['serving_gate']); self.assertFalse(result['numerical_acceptance'])
        self.assertTrue(all(g['candidate_bad'] > g['control_bad'] for g in result['groups']))
        for change in ('trial', 'phase_rank', 'replay_arm', 'payload_rank', 'row_ids'):
            bad = copy.deepcopy(records)
            if change == 'trial': bad.pop()
            if change == 'phase_rank': bad[0]['partial'].pop()
            if change == 'replay_arm': del bad[0]['replay'][0]['arms']['candidate']
            if change == 'payload_rank': bad[0]['payloads'].pop()
            if change == 'row_ids': bad[0]['payloads'][0]['row_ids'] = []
            with self.subTest(change=change), self.assertRaises(ValueError): m.completion(bad, {})

    def test_all_four_logs_and_each_payload_hash_are_required(self):
        records, raw = self.records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(4):
                lines = [json.dumps(dict(r['payloads'][rank], data_b64=base64.b64encode(raw).decode())) for r in records]
                if rank == 0:
                    lines += [json.dumps(dict(r, kind='MOE_M64_FP8_TRACE_TRIAL')) for r in records]
                    lines += [json.dumps(m.completion(records, {}))]
                (root/f'fp8-v3-rank-{rank}.log').write_text('\n'.join(lines)+'\n')
            self.assertEqual(m.verify_logs(root)['trials'], 72)
            path = root/'fp8-v3-rank-3.log'
            original = path.read_text()
            path.write_text(original.replace(base64.b64encode(raw).decode(), base64.b64encode(b'bad').decode(), 1))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): m.verify_logs(root)
            path.write_text('\n'.join(original.splitlines()[1:])+'\n')
            with self.assertRaisesRegex(ValueError, '72 row payloads'): m.verify_logs(root)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'also run with actual torch in pinned CPU image')
    def test_packet_row_selection_covers_destination_boundaries_and_preserves_bytes(self):
        import torch
        rows = 8; local_n = rows//4*4096; payload_bytes = local_n+local_n//2048*4
        packed = torch.zeros(payload_bytes*4, dtype=torch.uint8)
        for dest in range(4):
            p = packed.reshape(4, payload_bytes)[dest]
            for row in range(2): p[row*4096:(row+1)*4096] = dest*10+row
            packed.view(torch.float32).reshape(4, payload_bytes//4)[dest, local_n//4:] = torch.arange(4)+100*dest
        selected = [0, 1, 2, 5, 7]
        q, scales = m.packet_rows(torch, packed, selected, rows=rows, payload_bytes=payload_bytes)
        self.assertEqual(q[:, 0].tolist(), [0, 1, 10, 21, 31])
        self.assertEqual(scales.tolist(), [[0, 1], [2, 3], [100, 101], [202, 203], [302, 303]])
        q0, s0 = m.packet_rows(torch, packed, [], rows=rows, payload_bytes=payload_bytes)
        self.assertEqual(tuple(q0.shape), (0, 4096)); self.assertEqual(tuple(s0.shape), (0, 2))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'also run with actual torch in pinned CPU image')
    def test_trace_arrays_keep_each_arm_and_global_to_owned_row_mapping(self):
        import numpy as np
        import torch
        rows=8; local_n=8192; payload_bytes=local_n+16
        captures, replays, native = {}, {}, {}
        for i, arm in enumerate(m.ARMS):
            partial=torch.arange(rows, dtype=torch.bfloat16)[:, None].expand(-1,4096).contiguous()+i*10
            output=torch.full((2,4096), i, dtype=torch.bfloat16)
            packed=torch.zeros(payload_bytes*4,dtype=torch.uint8)
            captures[arm]=dict(partial=partial,output=output)
            replays[arm]=dict(packed=packed,payload_bytes=payload_bytes,sum32=output.float()+.5)
            native[arm]=output+1
        arrays=m.trace_arrays(torch,captures,replays,native,[1,4,5,7],rank=2,rows=rows)
        np.testing.assert_array_equal(arrays['row_ids'],[1,4,5,7])
        np.testing.assert_array_equal(arrays['owned_row_ids'],[4,5])
        self.assertEqual(set(arrays), {'row_ids','owned_row_ids'} |
            {a+s for a in m.ARMS for s in ('_partial_bits','_fp8_bytes','_scales','_sum32','_output_bits','_bf16_bits')})
        for i, arm in enumerate(m.ARMS):
            values=torch.from_numpy(arrays[arm+'_partial_bits']).view(torch.bfloat16)
            self.assertEqual(values[:,0].tolist(),[1+i*10,4+i*10,5+i*10,7+i*10])
            self.assertEqual(arrays[arm+'_sum32'].shape,(2,4096))
            self.assertEqual(float(arrays[arm+'_sum32'][0,0]),i+.5)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'also run with actual torch in pinned CPU image')
    def test_raw_numerators_expose_boundary_without_changing_normalized_failure(self):
        import torch
        b = torch.tensor([[30.5, 0.]])
        repeat = torch.tensor([[30.5, 1.]])
        candidate = torch.tensor([[30.5, 3.]])
        values = m.metrics(torch, candidate, b, repeat)
        self.assertEqual(values[0]['raw_error_peak'], 3.)
        self.assertEqual(values[0]['raw_noise_peak'], 1.)
        self.assertEqual(values[0]['reference_peak'], 30.5)
        self.assertGreater(values[0]['error_peak'], values[0]['limit_peak'])
        self.assertEqual(m.pair_summary(m.metrics(torch, b, b, repeat), values, rank=0)['candidate_bad'], 1)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'also run in pinned CPU image')
    def test_compressed_arrays_round_trip_bits_and_reject_corruption(self):
        import numpy as np
        arrays = dict(bits=np.asarray([[0, -32768, 32767]], dtype=np.int16),
                      scales=np.asarray([[.5, 2]], dtype=np.float32))
        encoded = m.encode_arrays(arrays)
        decoded = m.decode_arrays(encoded)
        for key, value in arrays.items(): np.testing.assert_array_equal(value, decoded[key])
        bad = dict(encoded, sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): m.decode_arrays(bad)
        with self.assertRaisesRegex(ValueError, 'object arrays'): m.encode_arrays(dict(bad=np.asarray([object()])))


if __name__ == '__main__': unittest.main()
