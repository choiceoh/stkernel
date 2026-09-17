import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import fleet_prepare


class PrepareCliTests(unittest.TestCase):
    def test_create_accepts_current_prepare_signature(self):
        with patch.dict(os.environ, FLEET_DIR='/tmp/fleet-test'), \
                patch.object(fleet_prepare, 'prepare', autospec=True, return_value='receipt') as prepare, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fleet_prepare.main(['create', 'test', '--', 'bash', 'bench/st_bracket.sh']), 0)
        self.assertEqual(prepare.call_args.args[2], ['bash', 'bench/st_bracket.sh'])

    def test_removed_overlay_approval_is_refused(self):
        with patch.dict(os.environ, FLEET_DIR='/tmp/fleet-test'), \
                patch.object(fleet_prepare, 'prepare', autospec=True) as prepare, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(fleet_prepare.main(['create', 'test', '--approve-deploy', '--', 'bash', 'bench/st_bracket.sh']), 3)
        prepare.assert_not_called()
