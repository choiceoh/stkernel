"""Known CPU packet/sum fixture and corruption checks for the offline reader."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import analyze as a
from glm53_moe_m64_fp8_trace import ARMS, PHASES, encode_arrays, metrics, pair_summary


class AnalyzerTests(unittest.TestCase):
    def fixture(self, root, corrupt=False):
        parts={}; arrays=[dict(row_ids=np.array([1],dtype=np.int64),
            owned_row_ids=np.array([1] if rank==0 else [],dtype=np.int64)) for rank in range(4)]
        sums={}
        for arm in ARMS:
            parts[arm]=[torch.full((1,4096),1+.5*rank,dtype=torch.bfloat16) for rank in range(4)]
            if arm=='repeat':parts[arm][0][0,0]+=.125
            if arm=='candidate':parts[arm][0][0,0]+=.5
            sums[arm]=sum(p.float() for p in parts[arm])
            for rank in range(4):
                p=parts[arm][rank]
                blocks=p.float().reshape(1,2,2048)
                scales=torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(dim=2)/448)))
                q=(blocks/scales[:,:,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(1,4096)
                owned=sums[arm] if rank==0 else sums[arm][:0]
                arrays[rank].update({arm+'_partial_bits':p.view(torch.int16).numpy(),
                    arm+'_fp8_bytes':q.numpy(),arm+'_scales':scales.numpy(),arm+'_sum32':owned.numpy(),
                    arm+'_output_bits':owned.to(torch.bfloat16).view(torch.int16).numpy(),
                    arm+'_bf16_bits':owned.to(torch.bfloat16).view(torch.int16).numpy()})
        if corrupt:arrays[2]['candidate_fp8_bytes'][0,0]^=1
        trial=dict(kind='MOE_M64_FP8_TRACE_TRIAL',rows=4096,skew=True,seed=13307,trial=0,row_ids=[1])
        for phase in PHASES:
            values=[]
            for rank in range(4):
                if phase=='transport' and rank==0:
                    p=pair_summary(metrics(torch,sums['control'],sums['baseline'],sums['repeat']),
                        metrics(torch,sums['candidate'],sums['baseline'],sums['repeat']),rank=rank,row_offset=1)
                else:p=dict(rank=rank,candidate_bad=0,control_bad=0,failures=[])
                values.append(p)
            trial[phase]=values
        for rank in range(4):
            payload=dict(kind='MOE_M64_FP8_TRACE_ROWS',rows=4096,skew=True,seed=13307,trial=0,
                rank=rank,row_ids=[1],**encode_arrays(arrays[rank]))
            lines=[json.dumps(payload)]
            if rank==0:lines.append(json.dumps(trial))
            (root/f'fp8-v3-rank-{rank}.log').write_text('\n'.join(lines)+'\n')

    def test_known_packet_and_sum_reconstruction_and_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            # This fixture isolates the CPU reader. Production analysis always
            # runs the unmocked 72-trial/four-log completion verifier first.
            with patch.object(a,'verify_logs',return_value={'cpu_fixture':True}):result=a.analyze(root)
            self.assertEqual(result['verified']['rank_arm_packet_sets'],16)
            self.assertEqual(result['verified']['destination_arm_sum_sets'],16)
            self.assertEqual(len(result['failures']),1)
            failure=result['failures'][0]
            self.assertTrue(failure['raw_peak_exceeds'])
            self.assertEqual(failure['exact_partial_sum_delta'],.5)
            self.assertEqual(failure['decoded_fp8_sum_delta'],.5)
            self.assertEqual(failure['stored_bf16_delta'],.5)

    def test_changed_packet_byte_fails_even_with_a_recomputed_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,corrupt=True)
            with patch.object(a,'verify_logs',return_value={'cpu_fixture':True}), \
                 self.assertRaisesRegex(ValueError,'CPU pack reconstruction differs'):
                a.analyze(root)


if __name__=='__main__':unittest.main()
