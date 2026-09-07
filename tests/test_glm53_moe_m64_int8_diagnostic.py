"""Full-row fidelity coverage and failure-preserving diagnostic completion."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probes'))
import glm53_moe_m64_int8_diagnostic as m


class DiagnosticTests(unittest.TestCase):
    def fixture(self):
        codec=[dict(rows=n,case=c,bad_bytes=0,cpu_reference=True,source_unchanged=True,finite=True)
            for n in (128,129,130,131,4095,4096,4097,8192) for c in ('zero','random','ties','extreme')]
        preflight=dict(codec=[dict(rank=r,cases=copy.deepcopy(codec)) for r in range(4)],
            short=[dict(rank=r,cases=[dict(rows=n,gather_equal=True,reduce_equal=True) for n in (2128,4095)]) for r in range(4)])
        records=[]
        for rows,skew,seed,trial in m.plan():
            record=dict(rows=rows,skew=skew,seed=seed,trial=trial,order=list(m.order(trial)))
            for phase in m.PHASES:
                record[phase]=[dict(rank=r,rows=rows if phase=='partial' else rows//4,
                    candidate_bad=2 if phase=='fp8' else 0,control_bad=1 if phase=='fp8' else 0,finite=True) for r in range(4)]
            record['checks']=[dict(rank=r,arms={a:dict(rows=rows,cpu_reference=trial==0,bad_bytes=0,
                source_unchanged=True,finite=True,output_reference_equal=True,gather_unchanged=True,capture_unchanged=True)
                for a in m.ARMS}) for r in range(4)]
            record['quality']=[dict(rank=r,arms={a:{t:dict(rows=rows//4,finite=True) for t in ('fp8','int8')}
                for a in m.ARMS}) for r in range(4)]
            records.append(record)
        return records,preflight

    def test_int8_success_keeps_fp8_failures_and_cannot_approve_serving(self):
        records,preflight=self.fixture();result=m.completion(records,preflight,{})
        self.assertEqual(result['trials'],72)
        self.assertFalse(result['serving_gate']);self.assertFalse(result['numerical_acceptance'])
        self.assertTrue(all(g['candidate_bad']>0 for g in result['groups'] if g['phase']=='fp8'))
        self.assertTrue(all(g['candidate_bad']==0 for g in result['groups'] if g['phase']=='int8'))

    def test_missing_or_bad_rows_ranks_references_and_short_identity_are_rejected(self):
        for case in ('trial','rank','rows','cpu_reference','packet','unpack','quality','codec','short'):
            records,preflight=self.fixture()
            if case=='trial':records.pop()
            if case=='rank':records[0]['int8'].pop()
            if case=='rows':records[0]['int8'][0]['rows']-=1
            if case=='cpu_reference':records[0]['checks'][0]['arms']['candidate']['cpu_reference']=False
            if case=='packet':records[0]['checks'][0]['arms']['candidate']['bad_bytes']=1
            if case=='unpack':records[0]['checks'][0]['arms']['candidate']['output_reference_equal']=False
            if case=='quality':records[0]['quality'][0]['arms']['candidate']['int8']['rows']-=1
            if case=='codec':preflight['codec'][3]['cases'].pop()
            if case=='short':preflight['short'][1]['cases'][0]['reduce_equal']=False
            with self.subTest(case=case),self.assertRaises(ValueError):m.completion(records,preflight,{})

    def test_codec_setting_is_restored_after_an_exception(self):
        h=SimpleNamespace(_RS_INT8=False)
        with self.assertRaises(RuntimeError):
            with m.rs_mode(h,True):
                self.assertTrue(h._RS_INT8);raise RuntimeError('test failure')
        self.assertFalse(h._RS_INT8)


if __name__=='__main__':unittest.main()
