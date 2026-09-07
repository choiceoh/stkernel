"""The idle-serving probe must reject low memory and stop only its own job."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

PROBES=Path(__file__).resolve().parents[1]/'probes'
sys.path.insert(0,str(PROBES))
spec=importlib.util.spec_from_file_location('moe_direct_guard_test',PROBES/'run_moe_direct_guarded.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
sys.path.pop(0)


@pytest.mark.parametrize('drop_during_run',[False,True])
def test_memory_guard_preserves_service_and_records_failure(tmp_path,monkeypatch,drop_during_run):
    out=tmp_path/'evidence'
    monkeypatch.setenv('FLEET_SESSION','guardtest')
    monkeypatch.setattr(sys,'argv',['probe','--out',str(out)])
    server={'boot_id':'public-head','image':guard.IMAGE,'running':True}
    monkeypatch.setattr(guard,'inspect_server',lambda:server.copy())
    counters={'num_requests_running':0,'num_requests_waiting':0,'request_success_total':19}
    monkeypatch.setattr(guard,'traffic',lambda:counters.copy())
    reads=iter([20*1024**3]+[11*1024**3]*100 if drop_during_run else [15*1024**3])
    monkeypatch.setattr(guard,'memory_available',lambda:next(reads))
    original=Path.read_text
    def read(path,*args,**kwargs):
        if str(path)=='/home/choiceoh/glm53-logs/fleet/holder':
            return 'guardtest|1|test|0|1|guard test|probe'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    calls=[];stopped=threading.Event()
    def check(command,**kwargs):
        assert command[0]=='git'
        return 'test-commit\n' if 'rev-parse' in command else ''
    def run(command,**kwargs):
        calls.append(command)
        if command[:2]==['docker','stop']:
            assert command[-1]=='moedirect-guardtest'
            stopped.set()
            return subprocess.CompletedProcess(command,0)
        assert command[:2]==['docker','run'] and drop_during_run
        assert stopped.wait(3),'memory watchdog did not stop the probe'
        return subprocess.CompletedProcess(command,137)
    with patch.object(guard.subprocess,'check_output',check),patch.object(guard.subprocess,'run',run):
        with pytest.raises(AssertionError):guard.main()
    receipt=json.loads((out/'admission.json').read_text())
    assert receipt['server_before']==receipt['server_after']==server
    assert receipt['before']==receipt['after']==counters
    assert receipt['issues']
    if drop_during_run:
        assert 'host headroom fell below 12 GiB' in receipt['issues']
        assert receipt['returncode']==137
    else:
        assert not any(command[:2]==['docker','run'] for command in calls)
        assert 'need 16 GiB' in receipt['issues'][0]
