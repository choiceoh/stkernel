"""C1 scheduling must preserve every MHC field and rearm across mixed rows."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'requires a reserved GB10')
class MhcStaticTailsTests(unittest.TestCase):
    def test_packed_and_packet_consumers_preserve_bytes_and_mixed_replay(self):
        from probes.engine_mhc_c1_tails import check
        check(lambda *args, **kwargs: None, timing=False)
