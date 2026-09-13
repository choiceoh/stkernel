"""Compare the terminal writer with the actual served mHC post and Torch mean."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton")
                     and importlib.util.find_spec("tilelang"), "requires CUDA, Triton and TileLang")
class MhcContractTests(unittest.TestCase):
    @staticmethod
    def inputs(rows, seed=1):
        torch.manual_seed(seed)
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(rows, 4, 4096, device="cuda", dtype=torch.bfloat16)
        post = torch.rand(rows, 4, 1, device="cuda", dtype=torch.float32)
        comb = torch.randn(rows, 4, 4, device="cuda", dtype=torch.float32)
        return x, residual, post, comb

    @staticmethod
    def baseline(x, residual, post, comb):
        from engine.kernels.mhc import mhc_post_tilelang
        return mhc_post_tilelang(x, residual, post, comb).float().mean(1).to(x.dtype)

    def exact(self, actual, expected):
        self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype))
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

    def test_decode_and_prefill_match_the_served_consumer(self):
        from engine.kernels.mhc_contract import contract
        for rows in (1, 6, 7, 8, 28, 65, 1728, 6912):
            with self.subTest(rows=rows):
                values = self.inputs(rows)
                actual, expected = contract(*values), self.baseline(*values)
                if not torch.equal(actual, expected):
                    self.diagnose(values, actual, expected)
                self.exact(actual, expected)

    @staticmethod
    def diagnose(values, actual, expected):
        """Retain exact synthetic failures at post and mean boundaries."""
        import ctypes as C
        import ctypes.util
        import json
        from engine.kernels.mhc import mhc_post_tilelang
        fma = C.CDLL(ctypes.util.find_library('m')).fmaf
        fma.argtypes, fma.restype = [C.c_float] * 3, C.c_float
        x, residual, post, comb = (v.cpu() for v in values)
        served = mhc_post_tilelang(*values).float().cpu()
        samples = []
        for row, col in (actual != expected).nonzero()[:3].tolist():
            channels = []
            for channel in range(4):
                value = C.c_float(float(comb[row, 0, channel]) * float(residual[row, 0, col])).value
                value = fma(float(post[row, channel, 0]), float(x[row, col]), value)
                for source in range(1, 4):
                    value = fma(float(comb[row, source, channel]), float(residual[row, source, col]), value)
                channels.append(torch.tensor(value).bfloat16().float().item())
            total = 0.
            for value in channels:
                total = C.c_float(total + value).value
            samples.append(dict(row=row, col=col, actual=actual[row,col].item(), expected=expected[row,col].item(),
                                served_channels=served[row,:,col].tolist(), oracle_channels=channels,
                                oracle_mean=torch.tensor(total*.25).bfloat16().float().item()))
        print(json.dumps(dict(mhc_mismatch_rows=x.shape[0], different=int((actual != expected).sum()), samples=samples)), flush=True)

    def test_rounding_and_cancellation_are_not_folded_through_the_mean(self):
        from engine.kernels.mhc_contract import contract
        from engine.kernels.mhc import mhc_post_tilelang
        values = self.inputs(8)
        x, residual, post, comb = values
        # The served PTX rounds comb[0] * residual[0] before adding post * x
        # with FMA. Rounding post * x first instead produces 8.67843628e-5.
        # Reproduce this rare random failure deterministically at every cell.
        x.fill_(-1.8671875)
        residual.zero_(); residual[:, 0].fill_(1.21875)
        post.fill_(0.8058584332466125)
        comb.zero_(); comb[:, 0, :].fill_(1.2346874475479126)
        served = self.baseline(*values)
        self.assertTrue(torch.all(served == 8.630752563476562e-5).item())
        self.exact(contract(*values), served)
        values = self.inputs(8)
        x, residual, post, comb = values
        # Identity mix with a small, different post contribution per channel:
        # averaging unrounded channels first is a different BF16 operation.
        residual.copy_(torch.randn_like(residual) * 64)
        comb.copy_(torch.eye(4, device="cuda").expand_as(comb))
        self.exact(contract(*values), self.baseline(*values))
        unrounded = residual.float() + post * x[:, None, :].float()
        folded = unrounded.mean(1).to(x.dtype)
        self.assertFalse(torch.equal(folded, self.baseline(*values)))
        # A tree reduction loses the small term in a different place than
        # Torch's serial four-channel sum. The channel BF16 casts are exact.
        x.zero_(); post.zero_()
        for channel, value in enumerate((2**30, 1, -(2**30), 1)):
            residual[:, channel].fill_(value)
        self.exact(contract(*values), self.baseline(*values))
        self.assertTrue(torch.all(self.baseline(*values) == 0.25).item())
        # All zeros must not read stale destination bytes.
        residual.zero_()
        self.exact(contract(*values), self.baseline(*values))

    def test_direct_feature_columns_preserve_neighbours_and_input_storage(self):
        from engine.kernels.mhc_contract import contract
        rows, hidden = 7, 4096
        packed = torch.full((rows, 5 * hidden + 32), 123, device="cuda", dtype=torch.bfloat16)
        features = []
        for feature in range(5):
            values = self.inputs(rows, feature + 2)
            before_inputs = [v.clone() for v in values]
            before = packed.clone()
            start = 16 + feature * hidden
            view = packed[:, start:start+hidden]
            self.assertIs(contract(*values, out=view), view)
            expected = self.baseline(*values)
            self.exact(view, expected)
            self.assertTrue(torch.equal(packed[:, :start], before[:, :start]))
            self.assertTrue(torch.equal(packed[:, start+hidden:], before[:, start+hidden:]))
            for actual, original in zip(values, before_inputs):
                self.assertTrue(torch.equal(actual, original))
            features.append(expected)
        self.exact(packed[:, 16:-16], torch.cat(features, dim=-1))

    def test_graph_replay_updates_each_feature_without_reallocation(self):
        from engine.kernels.mhc_contract import contract
        values = self.inputs(6)
        packed = torch.empty(6, 5 * 4096, device="cuda", dtype=torch.bfloat16)
        views = packed.split(4096, dim=-1)
        for view in views:
            contract(*values, out=view)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for view in views:
                contract(*values, out=view)
        try:
            for seed in (11, 12):
                for current, changed in zip(values, self.inputs(6, seed)):
                    current.copy_(changed)
                expected = torch.cat([self.baseline(*values)] * 5, dim=-1)
                graph.replay()
                self.exact(packed, expected)
        finally:
            graph.reset()

    def test_rejects_output_alias_and_overlapping_rows(self):
        from engine.kernels.mhc_contract import contract
        values = self.inputs(2)
        with self.assertRaisesRegex(ValueError, "overlap an input"):
            contract(*values, out=values[0])
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            contract(*values, out=values[0][:1].expand(2, -1))


if __name__ == "__main__":
    unittest.main()
