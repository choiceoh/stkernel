"""Actual launch plus same-window counters; no Docker, HTTP or GPU execution."""
import base64
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import glm53_launch_metadata as launch
import onepass
import proof

KNOB = 'VLLM_GLM53_SPEC_K'
BOOT = 'a'*64 + '|2026-09-10T00:00:00Z'


def command(k=3, *, config=None, extra=''):
    cfg = json.dumps(dict(method='dflash', model='/private/draft', num_speculative_tokens=k)) if config is None else config
    body = ('vllm serve /private/model --tensor-parallel-size 4 --nnodes 4 --node-rank 0 '
            + ("--speculative-config '"+cfg+"' " if cfg else '') + extra)
    script = launch._GID_PRELUDE + body + launch._REDIRECTION + '\n'
    return ['-c', 'echo '+base64.b64encode(script.encode()).decode()+' | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh']


def metrics(drafts=0, tokens=0, accepted=0, positions=(0, 0, 0)):
    labels = 'engine="0",model_name="/private/model"'
    lines = [f'vllm:spec_decode_num_{name}_total{{{labels}}} {value}'
             for name,value in (('drafts',drafts),('draft_tokens',tokens),('accepted_tokens',accepted))]
    lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{labels},position="{i}"}} {value}'
              for i,value in enumerate(positions)]
    lines += [f'vllm:spec_decode_num_drafts_created{{{labels}}} 1788962484.2']
    return '\n'.join(lines)+'\n'


def context():
    actual = dict(launch.launch_speculation(command()), boot_id=BOOT,
                  image='sha256:'+'b'*64, environment_spec_k='3')
    # Four drafts of lengths 3,3,2,1; accepted lengths 3,2,1,0.
    return dict(expected_k='3', boot_id=BOOT, launch_before=actual,
                launch_after=copy.deepcopy(actual), exclusive=True,
                metrics_before=metrics(), metrics_after=metrics(4,9,6,(3,2,1)))


class SpeculationProofTests(unittest.TestCase):
    def test_actual_literal_configuration_is_parsed_without_exposing_paths(self):
        for k in (3,4,5):
            value = launch.launch_speculation(command(k))
            self.assertEqual(value['num_speculative_tokens'], k)
            self.assertEqual(value['method'], 'dflash')
            self.assertNotIn('private', json.dumps(value))

    def test_disabled_duplicate_and_ambiguous_configuration_are_rejected(self):
        variants = [command(config=''), command(config='null'),
                    command(config='{"method":"mtp","num_speculative_tokens":3}'),
                    command(config='{"method":"dflash","num_speculative_tokens":true}'),
                    command(config='{"method":"dflash","num_speculative_tokens":3,"num_speculative_tokens":4}'),
                    command(extra="--speculative-config '{\"method\":\"dflash\",\"num_speculative_tokens\":3}'"),
                    command(extra='; echo unsafe'), command(extra='--speculative-config')]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                launch.launch_speculation(value)

    def test_container_snapshot_reads_only_the_identified_running_boot(self):
        c = dict(Id='a'*64, Image='sha256:'+'b'*64,
                 State=dict(StartedAt=BOOT.split('|')[1], Running=True, Paused=False, Restarting=False),
                 Config=dict(Cmd=command(), Env=[KNOB+'=3']))
        with patch('subprocess.check_output', return_value=json.dumps([c])) as read:
            actual = onepass._served_speculation(BOOT)
            self.assertEqual(actual, context()['launch_before'])
            self.assertEqual(read.call_args.args[0], ['docker','inspect','a'*64])
        for mutate in (lambda x:x.update(Id='c'*64), lambda x:x['State'].update(Running=False),
                       lambda x:x['State'].update(StartedAt='2026-09-10T01:00:00Z'),
                       lambda x:x['Config']['Env'].append(KNOB+'=4')):
            bad=copy.deepcopy(c);mutate(bad)
            with patch('subprocess.check_output', return_value=json.dumps([bad])):
                self.assertIsNone(onepass._served_speculation(BOOT))
        with patch('subprocess.check_output') as read:
            self.assertIsNone(onepass._served_speculation(None));read.assert_not_called()

    def test_short_final_proposals_keep_exact_positive_highest_position_proof(self):
        result = proof.spec_k_evidence(context())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['counter_delta'], dict(drafts=4,draft_tokens=9,accepted_tokens=6))
        self.assertEqual(result['accepted_tokens_per_position_delta'], [3,2,1])
        self.assertTrue(result['highest_position_observed'])
        self.assertIn('not fixed-request', result['scope'])
        self.assertNotIn('/private', json.dumps(result))

    def test_wrong_actual_k_environment_disabled_or_changed_boot_never_proves(self):
        changes = [lambda c:c.update(expected_k='4'), lambda c:c.update(exclusive=False),
                   lambda c:c['launch_after'].update(boot_id='c'*64+'|old'),
                   lambda c:c.update(boot_id='c'*64+'|old'), lambda c:c.update(launch_before=None),
                   lambda c:c['launch_after'].update(config_sha256='c'*64)]
        for field,value in (('environment_spec_k','4'),('method','mtp'),('node_rank',1),('node_rank',False),
                            ('image',None),('image','sha256:short'),('num_speculative_tokens',5)):
            changes.append(lambda c,f=field,v=value:(c['launch_before'].update({f:v}),c['launch_after'].update({f:v})))
        for change in changes:
            c=context();change(c)
            self.assertEqual(proof.spec_k_evidence(c)['verdict'], 'REJECTED')

    def test_missing_counters_positions_and_multiple_series_are_rejected(self):
        for field in ('metrics_before','metrics_after'):
            for transform in (lambda t:'', lambda t:'\n'.join(x for x in t.splitlines() if 'draft_tokens_total' not in x),
                              lambda t:'\n'.join(x for x in t.splitlines() if 'position="2"' not in x),
                              lambda t:t+t.splitlines()[0]+'\n',
                              lambda t:t.replace('position="2"','position="3"'),
                              lambda t:t.replace('drafts_total{engine="0"','drafts_total{engine="1"')):
                c=context();c[field]=transform(c[field])
                self.assertEqual(proof.spec_k_evidence(c)['verdict'],'REJECTED')

    def test_counter_reset_bad_sums_bounds_and_nonfinite_samples_are_rejected(self):
        variants = [metrics(0,0,0,(0,0,0)), metrics(4,13,6,(3,2,1)),
                    metrics(4,9,7,(3,2,1)), metrics(4,9,6,(2,3,1)),
                    metrics(4,9,9,(5,3,1)), metrics(4,5,6,(3,2,1)),
                    metrics(4,9,6,(3,2,1)).replace(' 4\n',' NaN\n'),
                    metrics(4,9,6,(3,2,1)).replace(' 4\n',' 4.5\n')]
        for after in variants:
            c=context();c['metrics_after']=after
            self.assertEqual(proof.spec_k_evidence(c)['verdict'],'REJECTED')
        c=context();c['metrics_before']=metrics(10,30,9,(5,3,1))
        self.assertEqual(proof.spec_k_evidence(c)['verdict'],'REJECTED')

    def test_configured_but_never_observed_last_position_is_rejected(self):
        c=context();c['metrics_after']=metrics(4,9,5,(3,2,0))
        self.assertEqual(proof.spec_k_evidence(c)['verdict'],'REJECTED')

    def test_earlier_cumulative_highest_position_cannot_prove_this_window(self):
        c=context();c['metrics_before']=metrics(4,9,6,(3,2,1))
        c['metrics_after']=metrics(8,18,11,(6,4,1))
        self.assertEqual(proof.spec_k_evidence(c)['verdict'],'REJECTED')

    def test_stale_or_fabricated_log_cannot_replace_live_proof_context(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'old.log'
            log.write_text('[spec] PASS K=3\nVLLM_GLM53_SPEC_K=3\n')
            for c in (None,{}):
                result=proof.check([KNOB],str(log),speculation=c)
                self.assertFalse(result['proof'][KNOB]);self.assertEqual(result['proof_ok'],'0/1')
            result=proof.check([KNOB],str(log),speculation=context())
            self.assertTrue(result['proof'][KNOB]);self.assertEqual(result['proof_ok'],'1/1')
            self.assertEqual(result['speculation']['counter_delta']['draft_tokens'],9)


if __name__ == '__main__':
    unittest.main()
