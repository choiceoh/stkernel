"""Paused reservations keep identity/age and do not receive GPUs or recovery debt."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
import fleet_handoff as handoff
import fleet_pending as pending
import fleet_pause as pause
import fleet_inspect as inspect


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.pid=os.getpid()
        self.row=f'100|fixture|100|10|candidate|boot|{self.pid}\n'
        (self.directory/'queue').write_text(self.row)
        env=patch.dict(os.environ,REPO=str(ROOT),FLEET_PREPARE_MANIFEST='',FLEET_LAUNCH_ID='')
        env.start();self.addCleanup(env.stop)
        identity=patch.object(handoff,'identity',return_value='start');identity.start();self.addCleanup(identity.stop)
        self.value=pending.register(self.directory,'fixture',[sys.executable,'-c','pass'],str(ROOT/'bench/fleet.sh'),'boot')
        handoff.ready(self.directory,'fixture',self.pid)

    def test_pause_keeps_ticket_age_and_resumes_only_after_validation(self):
        result=pause.pause(self.directory,'fixture','update needed',1)
        self.assertEqual((self.directory/'queue').read_text(),'')
        self.assertEqual(result['parked_row'],self.row.strip().split('|'))
        self.assertEqual(result['enqueued_at'],'100');self.assertEqual(result['ticket'],'100')
        self.assertEqual(result['state'],'paused');self.assertEqual(result['revision'],2)
        shown=inspect.show(self.directory,'fixture')
        self.assertEqual(shown['state'],'paused');self.assertTrue(shown['editable'])
        self.assertIsNone(shown['position']);self.assertEqual(shown['ahead'],[])
        self.assertEqual(inspect.show(self.directory)[0]['session'],'fixture')
        self.assertIn('resume',shown['actions'])
        with patch.object(pending,'validate') as validate:
            resumed=pause.resume(self.directory,'fixture',2)
        validate.assert_called_once()
        self.assertEqual(resumed['state'],'queued');self.assertEqual(resumed['revision'],3)
        self.assertEqual((self.directory/'queue').read_text(),self.row)

    def test_paused_front_cannot_accept_hold_or_block_successor(self):
        pause.pause(self.directory,'fixture')
        (self.directory/'priority-front').write_text('fixture')
        other=dict(self.value,session='other',ticket='101',pid=self.pid+1,state='queued')
        pending.save_record(self.directory,other)
        with (self.directory/'queue').open('a') as out:out.write(f'101|other|101|10|next|boot|{self.pid+1}\n')
        handoff.ready(self.directory,'other',self.pid+1)
        self.assertFalse(handoff.admit(self.directory,'fixture',self.pid,'boot'))
        self.assertFalse((self.directory/'holder').exists())
        self.assertEqual(handoff.successor(self.directory,'donor')['session'],'other')

    def test_failed_old_check_cannot_pause_new_revision(self):
        checked=copy.deepcopy(self.value)
        new=dict(self.value,revision=2);pending.save_record(self.directory,new)
        self.assertFalse(pause.pause_failed(self.directory,'fixture',checked,'old failure'))
        self.assertEqual(pending.read_record(self.directory,'fixture')['state'],'queued')
        self.assertTrue(pause.pause_failed(self.directory,'fixture',new,'current failure'))
        self.assertEqual(pending.read_record(self.directory,'fixture')['state'],'paused')
        self.assertEqual((self.directory/'queue').read_text(),'')

    def test_resume_does_not_publish_when_validation_fails_or_edit_wins(self):
        pause.pause(self.directory,'fixture')
        with patch.object(pending,'validate',side_effect=ValueError('missing input')):
            with self.assertRaisesRegex(ValueError,'missing input'):pause.resume(self.directory,'fixture')
        self.assertTrue(pause.paused(self.directory,'fixture'))
        def concurrent_edit(*args):
            value=pending.read_record(self.directory,'fixture');value['revision']+=1
            pending.save_record(self.directory,value)
        with patch.object(pending,'validate',side_effect=concurrent_edit):
            with self.assertRaisesRegex(ValueError,'changed while checking'):pause.resume(self.directory,'fixture')
        self.assertTrue(pause.paused(self.directory,'fixture'))

    def test_active_or_legacy_reservation_refuses_pause(self):
        old=dict(self.value);old.pop('pause_protocol');pending.save_record(self.directory,old)
        with self.assertRaisesRegex(ValueError,'pinned controller cannot pause'):pause.pause(self.directory,'fixture')
        pending.save_record(self.directory,self.value)
        (self.directory/'holder').write_text(f'fixture|{self.pid}|fixture|100|10|held|boot\n')
        with self.assertRaisesRegex(ValueError,'already admitted'):pause.pause(self.directory,'fixture')

    def test_log_publication_does_not_unpause_reservation(self):
        from fleet_boot import Supervisor
        pause.pause(self.directory,'fixture')
        with patch.dict(os.environ,FLEET_DIR=str(self.directory)):
            supervisor=Supervisor(str(ROOT/'bench/fleet.sh'),'fixture',10,'test',[sys.executable,'-c','pass'])
        supervisor.open_log(self.value)
        self.addCleanup(os.close,supervisor.log_fd)
        value=pending.read_record(self.directory,'fixture')
        self.assertEqual(value['state'],'paused');self.assertEqual(value['phase'],'paused')
        self.assertEqual(value['revision'],2);self.assertTrue(value['log_path'])

    def test_stale_or_unsupported_record_is_not_paused(self):
        value=pause.pause(self.directory,'fixture')
        value['pause_protocol']=0;pending.save_record(self.directory,value)
        self.assertFalse(pause.paused(self.directory,'fixture'))
        value['pause_protocol']=1;value['start']='reused-pid';handoff.write(pending.path(self.directory,'fixture'),value)
        self.assertFalse(pause.paused(self.directory,'fixture'))

    def test_pausing_offer_releases_target_but_preserves_donor_debt(self):
        donor=dict(session='donor',pid=self.pid+2,start='start',host=socket.gethostname(),protocol=handoff.PROTOCOL)
        target=handoff.read(handoff.receipt(self.directory,'fixture'))
        handoff.write(self.directory/'restore-debt.json',dict(owner=donor,target=target))
        pause.pause(self.directory,'fixture')
        debt=handoff.read(self.directory/'restore-debt.json')
        self.assertEqual(debt['owner'],donor);self.assertNotIn('target',debt)

    def test_legacy_priority_and_probe_selection_cannot_see_parked_ticket(self):
        from fleet_priority import rank
        (self.directory/'queue').write_text(self.row + f'101|older-controller|101|60|long|boot|{self.pid+1}\n')
        pause.pause(self.directory,'fixture')
        raw=(self.directory/'queue').read_text().splitlines()
        # The pre-pause controller's rank operates exclusively on physical rows.
        self.assertEqual([r['session'] for r in rank(raw,{},200)],['older-controller'])
        self.assertEqual(len(raw),1)

    def test_edit_parked_metadata_does_not_requeue_and_resume_keeps_ticket_age(self):
        pause.pause(self.directory,'fixture')
        updated=pending.edit(self.directory,'fixture',estimate=7,note='revised',expected=2)
        self.assertEqual(updated['state'],'paused');self.assertIsNone(updated['position'])
        self.assertEqual((self.directory/'queue').read_text(),'')
        self.assertEqual(updated['parked_row'],f'100|fixture|100|7|revised|boot|{self.pid}'.split('|'))
        with patch.object(pending,'validate'):
            pause.resume(self.directory,'fixture',3)
        self.assertEqual((self.directory/'queue').read_text(),f'100|fixture|100|7|revised|boot|{self.pid}\n')
        self.assertEqual(pause.parked(self.directory),[])

    def test_interrupted_park_and_resume_repair_from_canonical_record(self):
        with patch.object(pause,'write_rows',side_effect=OSError('interrupted projection')):
            with self.assertRaisesRegex(OSError,'interrupted projection'):
                pause.pause(self.directory,'fixture')
        self.assertTrue(pause.paused(self.directory,'fixture'))
        self.assertEqual(pause.parked(self.directory)[0]['ticket'],'100')
        self.assertEqual((self.directory/'queue').read_text(),self.row)
        with pending.lock(self.directory):
            self.assertEqual(pause.reconcile(self.directory,'fixture',self.pid),0)
        self.assertEqual((self.directory/'queue').read_text(),'')
        with patch.object(pending,'validate'),patch.object(pause,'write_rows',side_effect=OSError('interrupted resume')):
            with self.assertRaisesRegex(OSError,'interrupted resume'):
                pause.resume(self.directory,'fixture')
        self.assertEqual(pending.read_record(self.directory,'fixture')['state'],'queued')
        self.assertEqual((self.directory/'queue').read_text(),'')
        # cancel must still locate the owner before its waiter repairs the row.
        output=io.StringIO()
        with patch.dict(os.environ,FLEET_DIR=str(self.directory)),contextlib.redirect_stdout(output):
            self.assertEqual(pause.main(['pid','fixture']),0)
        self.assertEqual(output.getvalue().strip(),str(self.pid))
        # A retry or waiter's enqueue repair must use the original row exactly.
        self.assertFalse(pause.resume(self.directory,'fixture')['changed'])
        self.assertEqual((self.directory/'queue').read_text(),self.row)

    def test_another_pid_or_conflicting_ticket_cannot_replace_parked_reservation(self):
        pause.pause(self.directory,'fixture')
        with self.assertRaisesRegex(ValueError,'live parked reservation'):
            pause.reconcile(self.directory,'fixture',self.pid+1)
        (self.directory/'queue').write_text(self.row.replace('100|fixture','999|fixture'))
        before=(self.directory/'queue').read_text()
        with self.assertRaisesRegex(ValueError,'conflicts'):
            pause.reconcile(self.directory,'fixture',self.pid)
        self.assertEqual((self.directory/'queue').read_text(),before)
        with self.assertRaisesRegex(ValueError,'conflicts'):
            pending.edit(self.directory,'fixture',note='must not overwrite')

    def test_finished_or_reused_record_is_not_listed_by_stale_discovery_index(self):
        pause.pause(self.directory,'fixture')
        self.assertEqual(len(pause.parked(self.directory)),1)
        pending.transition(self.directory,'fixture','finished')
        handoff.write(self.directory/'pending'/'parked-index.json',dict(sessions=['fixture']))
        self.assertEqual(pause.parked(self.directory),[])
        self.assertEqual(inspect.show(self.directory),[])

    def test_prior_in_queue_pause_is_upgraded_without_losing_its_row(self):
        value=dict(self.value,state='paused',phase='paused')
        pending.save_record(self.directory,value)
        self.assertTrue(pause.paused(self.directory,'fixture'))
        pause.reconcile(self.directory,'fixture',self.pid)
        self.assertEqual((self.directory/'queue').read_text(),'')
        self.assertEqual(pending.parked_row(pending.read_record(self.directory,'fixture')),self.row.strip().split('|'))

    def test_edit_validates_signed_targets_before_commit_and_resume_only_verifies(self):
        import fleet_prepare
        import fleet_prepared
        old=self.directory/'old-prep.json';old.write_text(json.dumps({'spec_path':None}))
        value=copy.deepcopy(self.value)
        value.update(prepare_manifest=str(old),prepare_receipt_required=False)
        value['validation_env']['FLEET_VALIDATION_REQUIRED']='1'
        pending.save_record(self.directory,value)
        pause.pause(self.directory,'fixture')
        before=pending.read_record(self.directory,'fixture')
        target=self.directory/'new-prep.json'
        with patch.object(pending,'validate'),patch.object(fleet_prepare,'prepare',return_value=target), \
             patch.object(fleet_prepare,'validate_targets',side_effect=ValueError('target gate failed')):
            with self.assertRaisesRegex(ValueError,'target gate failed'):
                pending.edit(self.directory,'fixture',command=[sys.executable,'-c','print(1)'])
        self.assertEqual(pending.read_record(self.directory,'fixture'),before)
        with patch.object(pending,'validate'),patch.object(fleet_prepare,'prepare',return_value=target), \
             patch.object(fleet_prepare,'validate_targets') as targets:
            pending.edit(self.directory,'fixture',command=[sys.executable,'-c','print(1)'])
        self.assertEqual(targets.call_count,1)
        self.assertEqual(targets.call_args.args,(self.directory,target))
        self.assertEqual(targets.call_args.kwargs['controller']['fleet'],value['fleet'])
        self.assertNotIn('FLEET_VALIDATION_LEVEL',targets.call_args.kwargs['controller']['validation_env'])
        self.assertEqual((self.directory/'queue').read_text(),'')
        with patch.object(pending,'validate'),patch.object(fleet_prepared,'read',return_value={}), \
             patch.object(fleet_prepare,'validate'),patch.object(fleet_prepare,'validate_targets') as targets:
            pause.resume(self.directory,'fixture')
        self.assertEqual(targets.call_count,1)
        self.assertEqual(targets.call_args.args,(self.directory,str(target)))
        self.assertTrue(targets.call_args.kwargs['verify_only'])
        self.assertEqual(targets.call_args.kwargs['controller']['fleet'],value['fleet'])
        self.assertEqual((self.directory/'queue').read_text(),self.row)

    def test_edit_and_resume_use_original_payload_environment_not_editor(self):
        import fleet_prepare
        import fleet_prepared
        owner_pid=self.pid+100000
        value=copy.deepcopy(self.value)
        value.update(pid=owner_pid,ticket='original-owner',prepare_receipt_required=False)
        value['validation_env']={'PATH':os.environ['PATH'],'FLEET_VALIDATION_REQUIRED':'1',
                                 'FLEET_VALIDATION_STORE':'/original/validation-store'}
        old=self.directory/'original-prep.json';old.write_text(json.dumps({'spec_path':None}))
        value['prepare_manifest']=str(old)
        (self.directory/'queue').write_text(f'original-owner|fixture|100|10|candidate|boot|{owner_pid}\n')
        pending.save_record(self.directory,value)
        pause.pause(self.directory,'fixture')
        content=b'IMAGE=original-image\0CPU_GATE_CONFIG=original-config\0SSH_AUTH_SOCK=original-socket\0'
        original_open=Path.open
        def open_file(path,*args,**kwargs):
            return io.BytesIO(content) if str(path)==f'/proc/{owner_pid}/environ' else original_open(path,*args,**kwargs)
        seen=[]
        def check_environment(stage):
            self.assertEqual(os.environ['IMAGE'],'original-image')
            self.assertEqual(os.environ['CPU_GATE_CONFIG'],'original-config')
            self.assertEqual(os.environ['FLEET_VALIDATION_STORE'],'/original/validation-store')
            self.assertEqual(os.environ['SSH_AUTH_SOCK'],'original-socket')
            seen.append(stage)
        def preflight(argv,**kwargs):
            check_environment('preflight')
            self.assertEqual(kwargs['env']['IMAGE'],'original-image')
            self.assertEqual(kwargs['env']['CPU_GATE_CONFIG'],'original-config')
            return subprocess.CompletedProcess(argv,0,'')
        target=self.directory/'prepared-for-owner.json'
        def prepare(*args,**kwargs):
            check_environment('prepare')
            return target
        def targets(*args,**kwargs):
            check_environment('verify-targets' if kwargs.get('verify_only') else 'prepare-targets')
        with patch.object(Path,'open',open_file),patch.dict(os.environ,IMAGE='editor-image',
                 CPU_GATE_CONFIG='editor-config',FLEET_VALIDATION_STORE='/editor/store',SSH_AUTH_SOCK='editor-socket'):
            editor=dict(os.environ)
            with patch.object(pending.subprocess,'run',side_effect=preflight), \
                 patch.object(fleet_prepare,'prepare',side_effect=prepare), \
                 patch.object(fleet_prepare,'validate_targets',side_effect=targets):
                pending.edit(self.directory,'fixture',command=[sys.executable,'-c','pass'])
            self.assertEqual(dict(os.environ),editor)
            with patch.object(pending.subprocess,'run',side_effect=preflight), \
                 patch.object(fleet_prepared,'read',return_value={}), \
                 patch.object(fleet_prepare,'validate',side_effect=lambda *a,**k:check_environment('validate-source')), \
                 patch.object(fleet_prepare,'validate_targets',side_effect=targets):
                pause.resume(self.directory,'fixture')
            self.assertEqual(dict(os.environ),editor)
        self.assertEqual(seen,['preflight','prepare','prepare-targets','preflight','validate-source','verify-targets'])
        self.assertNotIn('original-config',pending.path(self.directory,'fixture').read_text())

    def test_owner_environment_pid_reuse_read_failure_and_exception_restore(self):
        value=dict(self.value,pid=self.pid+100000)
        with patch.object(handoff,'live',side_effect=[True,False]), \
             patch.object(Path,'open',return_value=io.BytesIO(b'CPU_GATE_CONFIG=private\0')):
            with self.assertRaisesRegex(ValueError,'changed while reading'):
                pending.supervisor_environment(value,self.directory)
        with patch.object(Path,'open',side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(ValueError,'cannot read the original supervisor'):
                pending.supervisor_environment(value,self.directory)
        before=dict(os.environ)
        with self.assertRaisesRegex(ValueError,'fixture failure'):
            with pending.owner_environment(self.value,self.directory):
                os.environ['TEMPORARY_PRIVATE_CONFIG']='private-value'
                raise ValueError('fixture failure')
        self.assertEqual(dict(os.environ),before)


if __name__=='__main__':unittest.main()
