"""Sessions cannot use the production restoration entrypoint."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RestoreTests(unittest.TestCase):
    def test_session_restore_is_refused_without_changing_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fleet=root/'fleet'; fleet.mkdir()
            production=root/'production'; production.mkdir()
            source=production/'source'; source.write_text('preserve candidate')
            (fleet/'production-repo').write_text(str(production))
            (fleet/'holder').write_text(f'fixture|{os.getpid()}|host|0|1|test|boot\n')
            env={k:v for k,v in os.environ.items() if not k.startswith('FLEET_')}
            env.update(FLEET_DIR=str(fleet),FLEET_SESSION='fixture',FLEET_RUNNER_REPO=str(ROOT),
                       FLEET_BOOT_INTENT='recovery')
            result=subprocess.run(['bash',str(ROOT/'bench/fleet_restore.sh')],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertIn('session restore is disabled',result.stderr)
            self.assertEqual(source.read_text(),'preserve candidate')

if __name__=='__main__': unittest.main()
