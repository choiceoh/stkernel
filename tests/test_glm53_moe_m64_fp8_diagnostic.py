"""Fixed diagnostic plan, row evidence and failure-preserving completion."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('fp8_diagnostic',ROOT/'probes/glm53_moe_m64_fp8_diagnostic.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def row(error=0., peak=0., noise=0., npeak=0., finite=True):
    return dict(error_l2=error,error_peak=peak,noise_l2=noise,noise_peak=npeak,
                limit_l2=max(.02,3*noise),limit_peak=max(.04,3*npeak),finite=finite)


class DiagnosticTests(unittest.TestCase):
    def test_rows_retain_candidate_excess_control_failures_and_intersections(self):
        control=[row(),row(.03),row(peak=.06),row()]
        candidate=[row(.03),row(),row(peak=.06),row()]
        result=m.pair_summary(control,candidate,rank=2,row_offset=2048)
        self.assertEqual([result[k] for k in ('candidate_bad','control_bad','candidate_only','control_only','both_bad')],[2,2,1,1,1])
        self.assertEqual([r['row'] for r in result['failures']],[2048,2049,2050])
        self.assertEqual(result['failures'][0]['candidate']['limit_l2'],.02)
        self.assertNotIn('pass',result)

    def test_nonfinite_and_peak_only_errors_cannot_disappear(self):
        result=m.pair_summary([row(),row()],[row(float('nan'),finite=False),row(peak=.05)],rank=0)
        self.assertEqual(result['candidate_bad'],2)
        self.assertFalse(result['finite'])
        self.assertIsNone(result['failures'][0]['candidate']['error_l2'])
        with self.assertRaises(ValueError):m.pair_summary([],[],rank=0)

    def test_original_repeat_multiplier_and_strict_boundary_are_preserved(self):
        a=row(.03,.06,noise=.01,npeak=.02)
        result=m.pair_summary([row()], [a],rank=0)
        self.assertEqual(result['candidate_bad'],0)
        a['error_peak']+=.00001
        self.assertEqual(m.pair_summary([row()],[a],rank=0)['candidate_bad'],1)

    def records(self):
        records=[]
        for rows,skew,seed,trial in m.plan():
            record=dict(rows=rows,skew=skew,seed=seed,trial=trial,order=list(m.order(trial)))
            for phase in ('transport','local'):
                record[phase]=[dict(rank=rank,rows=rows//4 if phase=='transport' else rows,
                    control_bad=1,candidate_bad=2,candidate_only=1,control_only=0,both_bad=1,finite=True) for rank in range(4)]
            records.append(record)
        return records

    def test_complete_diagnostic_preserves_failures_and_never_approves_serving(self):
        records=self.records();result=m.completion(records,{'source':'hash'})
        self.assertEqual(len(records),72)
        self.assertEqual(result['verdict'],'MOE_M64_FP8_DIAGNOSTIC_COMPLETE')
        self.assertFalse(result['serving_gate']);self.assertFalse(result['numerical_acceptance'])
        self.assertTrue(all(g['candidate_bad']>g['control_bad'] for g in result['groups']))
        self.assertEqual(len(result['groups']),18)
        for change in ('missing','duplicate','rank','row_coverage','order'):
            bad=copy.deepcopy(records)
            if change=='missing':bad.pop()
            elif change=='duplicate':bad[1]=bad[0]
            elif change=='rank':bad[0]['local'].pop()
            elif change=='row_coverage':bad[0]['transport'][0]['rows']-=1
            elif change=='order':bad[1]['order']=list(m.order(0))
            with self.subTest(change=change),self.assertRaises(ValueError):m.completion(bad,{})

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires torch; also executed in the pinned CPU image')
    def test_actual_cpu_tensor_math_keeps_floors_and_repeat_noise(self):
        import torch
        b=torch.ones(2,4);repeat=b.clone();repeat[1,0]+=.1
        a=b.clone();a[:,0]+=.05
        values=m.row_metrics(torch,a,b,repeat)
        control=m.row_metrics(torch,b,b,repeat)
        result=m.pair_summary(control,values,rank=0)
        self.assertEqual(result['candidate_bad'],1)
        self.assertEqual(result['failures'][0]['row'],0)
        self.assertAlmostEqual(values[0]['error_l2'],.025,places=6)
        self.assertAlmostEqual(values[0]['limit_peak'],.04,places=6)
        self.assertAlmostEqual(values[1]['limit_l2'],.15,places=6)
        a[1,0]=float('nan')
        self.assertFalse(m.row_metrics(torch,a,b,repeat)[1]['finite'])

if __name__=='__main__':unittest.main()
