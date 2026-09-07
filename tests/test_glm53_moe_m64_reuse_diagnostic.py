"""A diagnostic preserves failures and cannot silently omit reuse coverage."""
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probes'))
import glm53_moe_m64_reuse_diagnostic as m


class ReuseTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'CPU tensor validation also runs in the pinned image')
    def test_actual_tensor_collection_keeps_failures_and_validates_payloads(self):
        import torch
        def factory(rows,skew):
            x=torch.ones(rows,4096,dtype=torch.bfloat16)
            return x,lambda candidate:x.clone()+(.25 if candidate else 0.)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build=root/'build/glm53';build.mkdir(parents=True)
            (build/'fixture').write_bytes(b'fixture')
            (build/'manifest.tsv').write_text('fixture\t/unused\n')
            provenance={'fixture':hashlib.sha256(b'fixture').hexdigest()}
            captured=io.StringIO()
            with patch.object(m,'CASES',((4,True),)),patch.object(torch.cuda,'synchronize'),redirect_stdout(captured):
                result=m.run(torch=torch,case_factory=factory,provenance=provenance)
                self.assertEqual(m.verify_log(captured.getvalue(),root),result)
                self.assertEqual(sum(g['candidate_bad'] for g in result['groups']),32)
                self.assertFalse(result['numerical_acceptance'])
                damaged=captured.getvalue().replace('"data_b64": "','"data_b64": "broken',1)
                with self.assertRaises(Exception):m.verify_log(damaged,root)
                (build/'fixture').write_bytes(b'changed')
                with self.assertRaises(ValueError):m.verify_log(captured.getvalue(),root)

    def records(self):
        return [dict(rows=rows,skew=skew,phase=phase,trial=trial,order=list(m.order(trial)),
            input_unchanged=True,retained_unchanged=True,trace_rows=[3],
            comparison=dict(rows=rows,control_bad=0,candidate_bad=1,control_only=0,
                candidate_only=1,both_bad=0,finite=True,
                failures=[dict(row=3,control_bad=False,candidate_bad=True)]))
            for rows,skew,phase,trial in m.plan()]

    def test_failure_counts_survive_completed_collection(self):
        result=m.completion(self.records(),{'source':'hash'})
        self.assertEqual(result['trials'],48)
        self.assertEqual(sum(g['candidate_bad'] for g in result['groups']),48)
        self.assertEqual(sum(g['control_bad'] for g in result['groups']),0)
        self.assertFalse(result['serving_gate']);self.assertFalse(result['numerical_acceptance'])
        self.assertEqual(result['thresholds'],dict(l2=.02,peak=.04,repeat_multiplier=3))

    def test_missing_order_source_data_or_lifetime_proof_blocks_completion(self):
        for mutation in ('missing','duplicate','order','rows','input','retained','trace','count'):
            records=copy.deepcopy(self.records())
            if mutation=='missing':records.pop()
            elif mutation=='duplicate':records[1]=records[0]
            elif mutation=='order':records[0]['order'].reverse()
            elif mutation=='rows':records[0]['comparison']['rows']-=1
            elif mutation=='input':records[0]['input_unchanged']=False
            elif mutation=='retained':records[0]['retained_unchanged']=False
            elif mutation=='trace':records[0]['trace_rows']=[]
            elif mutation=='count':records[0]['comparison']['candidate_bad']=0
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.completion(records,{})


if __name__=='__main__':unittest.main()
